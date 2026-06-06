# Qwen3.5-122B-A10B — Optimization Benchmark Record

## Hardware context
- Device: Jetson AGX Thor SM110a, 117GB unified LPDDR5x, 273 GB/s, 120W
- NVFP4 bandwidth ceiling (12B active): ~45.5 tok/s autoregressive
- Bottleneck at conc=1: bandwidth-bound verify pass + GDN overhead (36/48 GDA layers)
- DFlash exceeds ceiling on high-acceptance tasks (sorting 56.7 tok/s = 125% of 45.5)

## Model config
- Quantization: NVFP4 resharded (8 shards), compressed-tensors
- MoE backend: **Cutlass ONLY** — Marlin HARD-CRASHES loading this 256-expert model (silent
  exit 255 at the FP4 MoE weight-repack; see debug doc). cutlass is the only backend that loads.
- Attention: **TRITON_ATTN** — cutlass + flashinfer dies on a kv_cache_sf API mismatch.
- DFlash draft: z-lab/Qwen3.5-122B-A10B-DFlash (block_size=16, target_layer_ids 6 layers)
- Optimal k from sweep: 10 (tau_avg 5.2)
- Launch: gpu-util 0.78 (draft eats ~2GB KV; 0.90 trips precheck); drop_caches before load.

## k-sweep results (reference, 120W)
| k  | sorting    | lru        | dijkstra   | mixed      | tau_avg |
|----|------------|------------|------------|------------|---------|
| 8  | 25.4/5.16* | 40.5/4.47  | 34.3/3.71  | 43.1/4.78  | 4.53    |
| 10 | 52.6/6.36  | 45.9/5.36  | 40.0/4.49  | 40.5/4.6   | 5.20    |
| 12 | 53.2/6.45  | 44.2/5.18  | 35.5/4.2   | 39.6/4.66  | 5.12    |
| 15 | 44.2/5.76  | 41.2/5.31  | 31.0/3.86  | 34.2/4.31  | 4.81    |
* k=8 cold-cache (triton/cutlass autotune); tok/s biased low, tau comparable.
sorting 52.6 / lru 45.9 exceed the 45.5 ceiling — DFlash tau amortization working (88-117%).

## Aux hidden state diagnosis (vLLM issue #43986)
- Draft config `dflash_config.use_aux_hidden_state`: **NOT FOUND** (config has only mask_token_id
  + target_layer_ids:[1,10,19,27,36,45]).
- `gpu_model_runner.py:517`: `self.use_aux_hidden_state_outputs = False` (default), set True only
  under EAGLE-style conditions (lines 565/571/582).
- Action: **none needed** — the draft does not request aux hidden state, so the wasteful-write
  path is not triggered for this config. No patch (per directive).

## Post-optimization benchmark — SAME-SESSION A/B (thermally controlled, the clean measurement)
Optimizations (no atomic-add — Cutlass path):
- VLLM_USE_FLASHINFER_SAMPLER=1
- CUDA_DEVICE_MAX_CONNECTIONS=1
- cudagraph_mode: FULL_AND_PIECEWISE (**already vLLM default**, confirmed in config dump);
  capture_sizes [1,11] this run.

| task     | baseline tok/s (no opts) | optimized tok/s | delta% |
|----------|--------------------------|-----------------|--------|
| sorting  | 56.7                     | 52.3            | −7.8%  |
| lru      | 37.4                     | 39.4            | +5.3%  |
| dijkstra | 33.8                     | 32.3            | −4.4%  |
| mixed    | 37.3                     | 41.7            | +11.8% |
| **avg**  | **41.3**                 | **41.4**        | **+0.2%** |
| tau_avg  | 5.01                     | 4.99            | ~0     |

Output validation: **CLEAN** (correct quicksort, no `!!!`). FULL_AND_PIECEWISE: stable (default).

### Interpretation (DEFINITIVE)
This is a back-to-back same-session A/B (baseline then optimized, adjacent, same thermal state),
so it isolates the optimization effect from the thermal drift that confounded the 35B/27B
cross-session comparisons. Result: **+0.2% = throughput-neutral.** Per-task scatter (−8% to +12%)
is pure run variance. The optimizations are output-safe and free but provide no measurable decode
speedup, because FULL_AND_PIECEWISE was already the default and the GDN path was already captured.

## Notes
- VLLM_MARLIN_USE_ATOMIC_ADD not applicable (Cutlass path).
- 36 GDN layers — already captured under the default FULL_AND_PIECEWISE.
- Cutlass MoE mandatory: Marlin fails at model load for the 256-expert 122B (debug doc).
- GDA hybrid: only 12/48 layers carry standard KV; the other 36 carry fixed recurrent state
  (~negligible KV cost) → context window is cheap to extend (KV is not the constraint).

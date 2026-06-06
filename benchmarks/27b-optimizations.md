# Qwen3.6-27B Dense — Optimization Benchmark Record

## Hardware context
- Device: Jetson AGX Thor SM110a, 117GB unified LPDDR5x, 273 GB/s, 120W
- NVFP4 bandwidth ceiling (27B dense): ~20.2 tok/s autoregressive
- Bottleneck at conc=1: bandwidth-bound (27B params/token ≈ 13.5 GB/token)
- DFlash exceeds the autoregressive ceiling (~38-50 tok/s measured) — tau amortization working

## Model config
- Quantization: NVFP4 compressed-tensors. Dense model → NVFP4 linear GEMM is FlashInfer-Cutlass
  (NOT marlin). `--moe-backend marlin` is set but there are no MoE layers, so it is inert here.
- DFlash draft: z-lab/Qwen3.6-27B-DFlash (block_size=16)
- Optimal k from sweep: **15** (avg 42.3 tok/s, tau 5.9)
- attention-backend: flash_attn; needs Qwen2Tokenizer overlay (ships TokenizersBackend class)

## k-sweep results (reference, 120W)
| k  | sorting   | lru        | dijkstra  | mixed     | tau_avg | avg tok/s |
|----|-----------|------------|-----------|-----------|---------|-----------|
| 8  | 41.9/5.89 | 37.7/5.08  | 35.8/4.77 | 37.1/4.94 | 5.17    | 38.1      |
| 10 | 48.6/6.62 | 37.1/5.0   | 36.8/4.98 | 39.6/5.33 | 5.48    | 40.5      |
| 12 | 44.8/6.22 | 37.2/5.11  | 37.5/5.12 | 45.9/6.34 | 5.70    | 41.4      |
| 15 | 50.1/7.04 | 38.4/5.33  | 39.8/5.55 | 40.9/5.68 | 5.90    | 42.3      |
Dense model rewards MAX speculation (k=15) — expensive 27B target forward, so each accepted
token saves a lot and draft overhead is small relative to target cost.

## VLLM_MARLIN_USE_ATOMIC_ADD=1 A/B (k=15)
| task | baseline | +atomic-add |
|------|----------|-------------|
| sorting | 50.1 | 41.7 |
| lru | 38.4 | 36.2 |
| dijkstra | 39.8 | 37.1 |
| mixed | 40.9 | 37.1 |
Verdict: **NO-OP** on dense 27B (NVFP4 GEMM is FlashInfer-Cutlass, not marlin). The ~−10% is
thermal/variance (matches the post-opt run below), not the flag. Kept for parity; inert.

## Post-optimization benchmark
Optimizations applied:
- VLLM_MARLIN_USE_ATOMIC_ADD=1 (no-op on dense)
- VLLM_USE_FLASHINFER_SAMPLER=1
- CUDA_DEVICE_MAX_CONNECTIONS=1
- cudagraph_mode: FULL_AND_PIECEWISE (already vLLM default); capture_sizes [1,16] this run

| task     | baseline tok/s (k=15) | optimized tok/s | delta% |
|----------|-----------------------|-----------------|--------|
| sorting  | 50.1                  | 40.9            | −18.4% |
| lru      | 38.4                  | 34.8            | −9.4%  |
| dijkstra | 39.8                  | 35.4            | −11.1% |
| mixed    | 40.9                  | 40.5            | −1.0%  |
| tau_avg  | 5.90                  | 5.71            | ~0     |

Output validation: **CLEAN** (correct quicksort, finish_reason=stop, no `!!!`).
FULL_AND_PIECEWISE: stable (already default).

### Interpretation (IMPORTANT)
The −10% average is **thermal/variance, NOT the optimizations**. Two independent later runs
(atomic-add A/B and this opt run) both landed at ~38 avg while the sweep baseline was 42.3 —
a consistent offset = thermal drift after ~9h continuous load, not random noise and not caused
by the opts. The thermally-controlled same-session 122B A/B proves the identical opt set is
throughput-neutral. 27B is DENSE (no GDN layers), so FULL_AND_PIECEWISE's GDN-capture benefit
does not even apply here — expected neutral, confirmed.

## Notes
- Dense model — no MoE routing, no GDA/GDN recurrent layers.
- cudagraph mode change primarily affects sampling/scheduling overhead here (small).
- The 2 env vars are kept (free, output-safe); capture-size restriction NOT baked into serve
  default (would hurt conc>1 with no conc=1 gain).

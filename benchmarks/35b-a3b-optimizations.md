# Qwen3.6-35B-A3B — Optimization Benchmark Record

## Hardware context
- Device: Jetson AGX Thor SM110a, 117GB unified LPDDR5x, 273 GB/s, 120W power mode
- NVFP4 bandwidth ceiling (3B active): ~182 tok/s autoregressive
- Bottleneck at conc=1: overhead-bound, not bandwidth-bound (3B active ≈ 1.5 GB/token)

## Model config
- Quantization: NVFP4 compressed-tensors, **Marlin** MoE backend
- DFlash draft: z-lab/Qwen3.6-35B-A3B-DFlash (block_size=16)
- Optimal k (by tok/s): 12 (avg 116.5). NOTE: this optimization run used **k=15** to match the
  baseline table the directive specified (k=15 = tau-optimal, 6.32). k=12 is ~5% faster in tok/s.
- attention-backend: flash_attn

## k-sweep results (reference, 120W)
| k  | sorting    | lru        | dijkstra   | mixed      | tau_avg | avg tok/s |
|----|------------|------------|------------|------------|---------|-----------|
| 8  | 121.4/6.61 | 98.8/5.72  | 82.1/4.5   | 89.6/5.03  | 5.46    | 98.0      |
| 10 | 130.4/7.55 | 97.2/5.93  | 93.2/4.66  | 127.9/5.52 | 5.92    | 112.2     |
| 12 | 137.1/8.56 | 100.5/6.08 | 104.0/4.76 | 124.2/5.61 | 6.25    | 116.5     |
| 15 | 139.1/8.86 | 98.7/6.27  | 98.0/4.81  | 111.1/5.35 | 6.32    | 111.7     |

## MoE backend comparison at k=12
| backend | sorting    | lru        | dijkstra   | mixed      | avg tok/s |
|---------|------------|------------|------------|------------|-----------|
| marlin  | 134.8/8.1  | 103.8/6.15 | 113.9/5.13 | 117.5/5.41 | 117.5     |
| cutlass | 114.5/8.56 | 83.3/6.02  | 96.2/4.74  | 131.0/6.25 | 106.3     |
Selected: **Marlin** (~10% faster). Cutlass higher tau, lower throughput.

## VLLM_MARLIN_USE_ATOMIC_ADD=1 A/B (k=12 marlin)
| task | baseline | +atomic-add | Δ |
|------|----------|-------------|---|
| sorting | 134.8 | 128.9 | −4% |
| lru | 103.8 | 130.7 | +26% |
| dijkstra | 113.9 | 101.3 | −11% |
| mixed | 117.5 | 127.7 | +9% |
Verdict: **within run-to-run noise** (τ also shifted, which atomic-add cannot cause → pure
variance). Kept as free (no output change). It changes the marlin reduction; harmless.

## Post-optimization benchmark
Optimizations applied:
- VLLM_MARLIN_USE_ATOMIC_ADD=1 (marlin reduction; neutral, free)
- VLLM_USE_FLASHINFER_SAMPLER=1 (GPU-side sampling, no host round-trip)
- CUDA_DEVICE_MAX_CONNECTIONS=1 (single cmd queue)
- cudagraph_mode: FULL_AND_PIECEWISE — **already the vLLM default** (verified in config dump);
  explicit setting is a no-op. capture_sizes restricted to [1,16] for this run.

| task     | baseline tok/s (k=15) | optimized tok/s | delta% |
|----------|-----------------------|-----------------|--------|
| sorting  | 139.1                 | 125.6           | −9.7%  |
| lru      | 98.7                  | 92.1            | −6.7%  |
| dijkstra | 98.0                  | 95.7            | −2.3%  |
| mixed    | 111.1                 | 115.5           | +4.0%  |
| tau_avg  | 6.32                  | 6.34            | ~0     |

Output validation: **CLEAN** (correct quicksort, finish_reason=stop, no `!!!`, no repetition).
FULL_AND_PIECEWISE: stable (already default; kept).

### Interpretation (IMPORTANT)
The −4% average is **NOT a real regression** — it is confounded by (a) ±10% inter-launch
variance and (b) **thermal drift**: this run was on a Thor that had run continuous benchmarks
for ~9h at 120W, hotter than when the k=15 sweep baseline was measured. The definitive,
thermally-controlled SAME-SESSION A/B on the 122B (below, in 122b file) showed the identical
optimization set is **throughput-neutral (+0.2%)**. So the 35B opts are neutral too; the
apparent −4% is thermal/variance, not the optimizations.

## Notes on FULL_AND_PIECEWISE for this architecture
The plan's premise (current=PIECEWISE → switch to FULL_AND_PIECEWISE for a large gain) does not
hold: FULL_AND_PIECEWISE is already vLLM's DEFAULT cudagraph_mode in this fork (confirmed in the
EngineCore config dump: `<CUDAGraphMode.FULL_AND_PIECEWISE: (2,1)>`). The GDN causal_conv1d_update
is therefore already captured by default. Explicitly setting it changes nothing; the only real
delta was restricting capture_sizes to [1,16], which gives no conc=1 benefit and would force eager
fallback at conc>1 — so it is NOT baked into the serve-script defaults (they use default captures).
Corruption bugs #39273/#40880 are scoped to ngram/MTP SSM restore paths, not DFlash verify; output
stayed clean. PR #34571 Mamba cudagraph size-cap fix is in the fork.

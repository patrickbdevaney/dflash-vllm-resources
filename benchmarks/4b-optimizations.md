# Qwen3.5-4B Dense (GDA-hybrid) + DFlash — Benchmark Record

## Hardware context
- Device: Jetson AGX Thor SM110a, 117GB unified LPDDR5x, 273 GB/s, 120W
- NVFP4 bandwidth ceiling (4B): ~136.5 tok/s autoregressive (273/2.0)
- Bottleneck at conc=1: **overhead-bound** (small model — tiny per-layer GEMMs mean kernel-launch
  latency dominates, so autoregressive sits far BELOW the bandwidth ceiling). DFlash amortizes
  this by batching k+1 tokens per verify step → large relative speedup.
- Base: ~/models/Qwen3.5-4B-NVFP4
- Draft: ~/models/Qwen3.5-4B-DFlash (block_size=16, target_layer_ids [1,8,15,22,29], 5 layers)
- Quantization: declared `modelopt` NVFP4 in config; loads fine with `--quantization compressed-tensors`
- gpu-memory-utilization: 0.40 | max-model-len: 32768 | attention-backend: flash_attn
- Env: VLLM_USE_FLASHINFER_SAMPLER=1, CUDA_DEVICE_MAX_CONNECTIONS=1

## Architecture note (correction)
The 4B is NOT pure dense — it is a **dense GDA hybrid** (no MoE, but
`layer_types=[linear_attention ×3, full_attention]`, 32 layers, hidden 2560, 16 heads/4 KV).
It is also a VLM wrapper (Qwen3_5ForConditionalGeneration) → served with `--language-model-only`
to skip the vision encoder. Only the full_attention layers carry standard KV (cheap context).

## Autoregressive baseline (no DFlash)
- Dijkstra tok/s: **47.9** (tau n/a — no spec decode)
- vs 136.5 tok/s ceiling: **35%** (overhead-bound, as expected for a small model at conc=1)

## DFlash k-sweep results (capture_sizes [1,k+1], 120W)
| k  | sorting    | lru        | dijkstra   | mixed      | tau_avg | avg tok/s |
|----|------------|------------|------------|------------|---------|-----------|
| 8  | 120.7/5.08 | 108.1/4.48 | 148.1/4.52 | 142.2/4.34 | 4.61    | 129.8     |
| 10 | 129.3/5.42 | 145.2/4.72 | 152.2/4.68 | 146.7/4.5  | 4.83    | 143.4     |
| 12 | 143.8/6.19 | 139.3/5.12 | 143.6/4.44 | 127.2/3.92 | 4.92    | 138.5     |
| 15 | 130.6/5.75 | 141.0/4.57 | 135.5/4.37 | 155.8/5.01 | 4.92    | 140.7     |

Optimal k: **15** (highest mixed tok/s = 155.8). NOTE: k=10 has the best 4-task AVERAGE (143.4);
k=10/12/15 are all ~138-143 avg and within run-to-run noise (~±10%), so either k=10 or k=15 is a
defensible "optimal" — k=15 by the highest-mixed criterion, k=10 by best-average.
Best tok/s: **155.8** on mixed (k=15). DFlash dijkstra peak 152.2 (k=10) vs 47.9 baseline = **3.2×**.

## Physics comparison (all models, conc=1, 120W)
| model        | active | ceiling  | best DFlash | vs ceiling |
|--------------|--------|----------|-------------|------------|
| 4B (GDA-hyb) | 4B     | 136.5    | 155.8       | 114%       |
| 27B dense    | 27B    | 20.2     | 50.1        | 248%       |
| 35B-A3B MoE  | 3B     | 182.0    | 139.1       | 76%        |
| 122B-A10B    | 12B    | 45.5     | 52.6        | 116%       |

## Notes
- Dense (no MoE routing) but GDA-hybrid; tau saturates at ~4.9 by k=12 (acceptance plateau),
  so beyond k=12 higher k mainly trades per-task variance, not more acceptance.
- Overhead-bound small model: DFlash's 3.2× over autoregressive is the largest relative speedup
  of the four models — because the autoregressive baseline was overhead-starved (35% of ceiling)
  and DFlash's batched verify reclaims that overhead. Exceeds the bandwidth ceiling (114%) via
  tau amortization, same mechanism as 27B (248%) and 122B (116%).
- Draft: z-lab/Qwen3.5-4B-DFlash. aux_hidden_state in draft config: **not found** (uses
  target_layer_ids only; the #43986 wasteful-write path is not requested).

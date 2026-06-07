---
license: mit
library_name: vllm
pipeline_tag: text-generation
tags:
- dflash
- speculative-decoding
- block-diffusion
- jetson-thor
- sm110a
- nvfp4
- edge-inference
- qwen
---

# DFlash on NVIDIA Jetson AGX Thor (SM110a) — Proven Single-Node Deployment Guide

Block-diffusion speculative decoding (**[DFlash](https://github.com/z-lab/dflash)**) for Qwen3.5 /
Qwen3.6 **NVFP4** models, running on a **single NVIDIA Jetson AGX Thor** — aarch64, compute
**sm_110a**, ~117 GB unified LPDDR5X, CUDA 13, L4T r38. This repo is the **field guide for actually
getting these models stable on Thor**: the from-source vLLM image, per-model launch configs, the
122B Marlin-crash discovery and its fix, full k-sweeps, and conc=1 decode benchmarks for four
model sizes.

> This is Thor-specific. The upstream DFlash drafters and the general model cards live at
> [z-lab](https://huggingface.co/z-lab). Numbers here are **measured on Thor at concurrency 1**,
> not the B200 numbers from the paper.

---

## TL;DR — what works

| Model | Active | MoE backend | Attn backend | Optimal k | Peak DFlash tok/s (conc=1) | vs AR |
|---|---|---|---|---|---|---|
| Qwen3.5-4B-NVFP4 (GDA-hybrid) | 4B | n/a (dense) | flash_attn | 15 | **155.8** | **3.2×** |
| Qwen3.6-35B-A3B-NVFP4 (MoE) | 3B | **marlin** | flash_attn | 12 | **139.1** | — |
| Qwen3.6-27B-NVFP4 (dense) | 27B | n/a (dense) | flash_attn | 15 | **50.1** | — |
| Qwen3.5-122B-A10B-NVFP4 (MoE) | 12B | **cutlass** ⚠️ | **TRITON_ATTN** ⚠️ | 10 | **52.6** | **1.7×** |

⚠️ **The 122B is the special case.** Marlin (the backend that's *fastest* for the 35B)
**hard-crashes** loading the 256-expert 122B. It must run **cutlass MoE + TRITON_ATTN**. See
[the 122B section](#the-122b-marlin-crash-and-the-cutlass--triton_attn-fix).

---

## The image: vLLM from source + fastsafetensors

The stock `nvidia-ai-iot/vllm:latest-jetson-thor` is vLLM 0.19.0 with **no DFlash**. DFlash landed
after 0.19.0, so the image is built **from source for sm_110a**:

- **Source:** vLLM **PR #40898** head — `vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/40898/head`
  (DFlash block-diffusion + interleaved SWA support). Version string: `0.20.0.dev0+dflash`.
- **Arch:** `TORCH_CUDA_ARCH_LIST=11.0a` → native **sm_110a SASS** in `_C.abi3.so` and `_moe_C.abi3.so`.
- **Flash-attn:** native **sm_110 FA2** copied from the stock Thor image. **FA3 crashes on Thor**
  (`CUTE_ARCH_TMA_SM90_ENABLED` → `cudaErrorLaunchFailure`; FA3 needs Hopper SM90 TMA). vLLM
  selects FA2, so this is transparent. FA2-vs-FA3 makes **no decode difference** (decode is
  GEMM/bandwidth-bound, not attention-bound).
- **fastsafetensors 0.3.2** layer (`Dockerfile.fastsafe`): image tag `vllm-dflash-thor:fastsafe`
  (identical image ID to `fa-native`, plus the fastsafetensors wheel).

**Why fastsafetensors is mandatory for the 122B.** The default safetensors loader mmaps the weight
file (CPU) *then* copies to GPU. On Thor's single 117 GB unified pool both copies coexist: for the
122B that's 72 GB(GPU) + up to 72 GB(CPU mmap) = **144 GB > 117 GB → the box crashes during load**.
fastsafetensors streams disk→GPU directly (no CPU staging), peaking near 72 GB. `FASTSAFETENSORS_NOGDS=1`
forces the no-GDS fallback (GPU Direct Storage is unsupported on Thor; it still avoids CPU staging).

Prebuilt image tarballs (aarch64 / sm_110a only) are published alongside this work:
```bash
docker load < vllm-dflash-thor-fastsafe.tar.gz   # 19 GB, has fastsafetensors 0.3.2
# or the base (no fastsafetensors): vllm-dflash-thor-fa-native.tar.gz
```

Full build + debug narrative: **[BUILD-AND-DEBUG.md](BUILD-AND-DEBUG.md)**. All numbers + the raw
debug log: **[RESULTS.md](RESULTS.md)**. Per-model detail: **[benchmarks/](benchmarks/)**.

---

## How to run (proven configs)

All four use the same image, `--load-format fastsafetensors`, `--quantization compressed-tensors`
(the 4B is declared `modelopt` but loads fine as compressed-tensors), `--trust-remote-code`,
`--language-model-only` (the 4B and 122B are VLM wrappers), and these env vars:
```
-e VLLM_USE_FLASHINFER_MOE_FP4=0          # mandatory: FlashInfer FP4 MoE has no sm_110a kernel
-e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1   # else import vllm._C dies
-e FASTSAFETENSORS_NOGDS=1 -e HF_HUB_DISABLE_XET=1 -e NCCL_IGNORE_CPU_AFFINITY=1
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
-e VLLM_USE_FLASHINFER_SAMPLER=1 -e CUDA_DEVICE_MAX_CONNECTIONS=1   # free, output-safe (see Optimizations)
```
Pre-flight before every launch: `for c in $(docker ps -q); do docker kill $c; done; sudo sync;
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`. **Never** `docker stop` (hangs, leaks page cache)
or `sudo fuser -k /dev/nvidia*` (kills Xorg/RustDesk, not CUDA).

Exact, copy-pasteable launch commands for all four models are in **[REVERT.md](REVERT.md)**, and the
wrapper scripts are `scripts/serve-{35b,27b,122b}.sh`. The key per-model differences:

```bash
# 35B-A3B (MoE) — marlin is fastest
--moe-backend marlin --attention-backend flash_attn \
--gpu-memory-utilization 0.78 --max-model-len 65536 --max-num-seqs 4 \
--speculative-config '{"method":"dflash","num_speculative_tokens":12,"model":"/draft"}'

# 27B (dense) — needs Qwen2Tokenizer overlay (ships tokenizer_class=TokenizersBackend)
--attention-backend flash_attn --tokenizer /tokenizer \
--gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-seqs 4 \
--speculative-config '{"method":"dflash","num_speculative_tokens":15,"model":"/draft"}'

# 122B-A10B (MoE) — CUTLASS + TRITON_ATTN (NOT marlin/flashinfer), draft KV capped
--moe-backend cutlass --attention-backend TRITON_ATTN \
--gpu-memory-utilization 0.78 --max-model-len 16384 --max-num-seqs 2 \
--speculative-config '{"method":"dflash","num_speculative_tokens":10,"model":"/draft","max_model_len":1024}'

# 4B (dense GDA-hybrid VLM)
--attention-backend flash_attn \
--gpu-memory-utilization 0.40 --max-model-len 32768 --max-num-seqs 1 \
--speculative-config '{"method":"dflash","num_speculative_tokens":15,"model":"/draft"}'
```

---

## The 122B Marlin crash, and the cutlass + TRITON_ATTN fix

This is the central Thor discovery. The 122B-A10B (**256 experts × 48 layers**) crashed on every
launch for weeks. Root cause, established by reproduction + `cuobjdump`:

- After weights load (~60 s), the EngineCore dies with **exit 255, no Python traceback, Docker
  OOMKilled=false, no host OOM line**, at exactly `nvfp4.py:491 Using MoEPrepareAndFinalizeNoDPEPModular`
  — i.e. the **Marlin NVFP4 MoE weight-repack**.
- It is **not** DFlash / SWA / OOM / config — the identical crash reproduces with the **base model
  alone** (no speculative config).
- It is **not** a missing kernel image: `_moe_C.abi3.so` *does* contain `sm_110a`, and the MoE dims
  are 128-aligned. It is a **genuine in-kernel fault in the Marlin FP4 MoE kernel at 256-expert
  scale** on Thor (matches open vLLM issues #35566 / #35519 / #35922). **A rebuild does not fix it.**

**The fix:**
1. `--moe-backend cutlass` — cutlass (VLLM_CUTLASS) processes all 256 experts cleanly (70.46 GiB
   load, no crash). It is the **only** NVFP4 MoE backend that loads the 122B on Thor.
   (`triton` is rejected for NVFP4; `flashinfer_*` have no sm_110a kernels.)
   This is the **opposite** of the 35B, where marlin is ~10% *faster* than cutlass — marlin only
   *crashes* at the large expert count.
2. `--attention-backend TRITON_ATTN` — cutlass + flashinfer dies in attention warmup with
   `BatchDecodeWithPagedKVCacheWrapper.run() got an unexpected keyword 'kv_cache_sf'` (a FlashInfer
   API mismatch in this fork). TRITON_ATTN avoids it.
3. `--gpu-memory-utilization 0.78` + draft `"max_model_len":1024` — the DFlash draft eats ~2 GB KV;
   at 0.72 the 16384 KV request fails ("estimated max length 8960"); 0.90 trips the startup
   precheck (needs 110.5 > ~108 GB free). 0.78 yields ~71k-token KV.

Result: base 122B 10.9 tok/s → DFlash **27–42 tok/s** (τ 4.2–6.5), **1.7×+**, stable, coherent.

### cuobjdump arch facts (this image)
| extension | arches | meaning |
|---|---|---|
| `_moe_C.abi3.so` (Marlin MoE) | sm_110a, sm_80 | has Thor image, but **faults at 256 experts** |
| `_C.abi3.so` | sm_110a, sm_80, sm_90 | core kernels |
| flashinfer `fused_moe_103.so` | sm_100a, sm_103a | **no sm_110a** → `VLLM_USE_FLASHINFER_MOE_FP4=0` |
| `gemm_sm120.so` / `trtllm_low_latency_gemm.so` | sm_120 / sm_100a | no Thor |

---

## Benchmarks — concurrency 1, single Thor, 120W, T=0

4 coding tasks (sorting / lru / dijkstra / mixed), 3-run median, `tok/s (τ)`. Full k-sweeps in
[RESULTS.md](RESULTS.md); per-model files in [benchmarks/](benchmarks/).

**Qwen3.6-35B-A3B (MoE, marlin)** — k-sweep optimal **k=12** (avg 116.5 tok/s):
| k | sorting | lru | dijkstra | mixed | τ_avg |
|---|---|---|---|---|---|
| 12 | 137.1/8.56 | 100.5/6.08 | 104.0/4.76 | 124.2/5.61 | 6.25 |
| 15 | 139.1/8.86 | 98.7/6.27 | 98.0/4.81 | 111.1/5.35 | 6.32 |
MoE profile @k=12: **marlin 117.5 avg > cutlass 106.3** → marlin default.

**Qwen3.6-27B (dense)** — k-sweep optimal **k=15** (avg 42.3 tok/s):
| k | sorting | lru | dijkstra | mixed | τ_avg |
|---|---|---|---|---|---|
| 15 | 50.1/7.04 | 38.4/5.33 | 39.8/5.55 | 40.9/5.68 | 5.90 |

**Qwen3.5-122B-A10B (MoE, cutlass+TRITON_ATTN)** — optimal **k=10**, base 10.9 tok/s:
| k | sorting | lru | dijkstra | mixed | τ_avg |
|---|---|---|---|---|---|
| 10 | 52.6/6.36 | 45.9/5.36 | 40.0/4.49 | 40.5/4.6 | 5.20 |

**Qwen3.5-4B (dense GDA-hybrid)** — optimal **k=15** (mixed 155.8), AR baseline 47.9 (dijkstra):
| k | sorting | lru | dijkstra | mixed | τ_avg |
|---|---|---|---|---|---|
| 15 | 130.6/5.75 | 141.0/4.57 | 135.5/4.37 | 155.8/5.01 | 4.92 |

### Optimal-k pattern across model classes
`122B k=10 · 35B k=12 · 27B k=15 · 4B k=15`. **Cheaper-per-token models prefer higher k.** A dense
27B (or overhead-bound 4B) pays an expensive full-weight forward per verify, so each accepted token
saves a lot and the draft overhead is small relative to it → push k to the block max. A fast MoE
(35B, 3B active) hits the acceptance/overhead cliff sooner → lower optimal k.

### Roofline (NVFP4, 273 GB/s)
| model | active | AR ceiling | best DFlash | vs ceiling |
|---|---|---|---|---|
| 4B (GDA-hybrid) | 4B | 136.5 | 155.8 | **114%** |
| 27B dense | 27B | 20.2 | 50.1 | **248%** |
| 35B-A3B MoE | 3B | 182.0 | 139.1 | 76% |
| 122B-A10B MoE | 12B | 45.5 | 52.6 | **116%** |
DFlash's τ amortization lets three of four models **exceed** their autoregressive bandwidth ceiling
at conc=1 — the verify step processes k+1 tokens per weight load, so effective tok/s ≈ ceiling × τ /
verify-cost. The 35B sits at 76% because it's already the fastest (3B active, overhead-bound).

---

## Why this matters: conc=1 single-node decode, and DFlash vs MTP

The realistic edge / agentic deployment on a single Thor is **concurrency 1** — one user, one stream,
latency-bound. In that regime, *all* of these current-gen NVFP4 models are **overhead- or
bandwidth-starved**: at conc=1 there's no batch to amortize kernel launches, so a 4B model runs at
only **35% of its bandwidth ceiling** autoregressively (47.9 / 136.5 tok/s), and a 122B is a slideshow
at 10.9 tok/s. This is exactly where speculative decoding pays off most, and where the *choice* of
speculator matters.

**DFlash (block diffusion) vs MTP (multi-token prediction) at conc=1:**
- **MTP** appends a few extra prediction heads (typically `num_speculative_tokens=3–4`) and drafts that
  many tokens autoregressively-cheaply. Its draft *depth* and acceptance are inherently shallow.
- **DFlash** denoises a whole **block** (block_size 16 → k up to 15) **in parallel**, with a
  non-causal drafter trained for it, reaching **acceptance length τ ≈ 4.5–8.9** here. Each target
  forward then verifies ~5–9 tokens instead of MTP's ~2–3.
- At conc=1 the per-step target cost is fixed, so throughput ≈ τ / verify-cost. DFlash's **deeper,
  higher-acceptance** drafts amortize that fixed cost across far more accepted tokens than MTP's
  shallow drafts — which is why DFlash delivers the larger conc=1 speedups (we measured **1.7×** on
  the 122B and **3.2×** on the overhead-bound 4B; z-lab reports up to **2.9×** vs AR at conc=1).
  MTP's advantage only narrows at high concurrency, where batching already hides launch overhead —
  the opposite of the single-Thor edge case.

**The practical upshot:** DFlash is what makes a **122B-class model usable interactively on one
117 GB edge box** (10.9 → ~52 tok/s peak), and turns a 4B into a 150+ tok/s coding assistant — on
hardware with no discrete VRAM, drawing 120 W. The optimal-k tuning per model (above) is the single
highest-value lever; the rest of the "optimizations" are noise (next section).

---

## Optimizations that DON'T help (so you don't chase them)

Measured honestly on Thor. See [hypotheses.md](hypotheses.md) and the per-model benchmark files.
- **`cudagraph_mode FULL_AND_PIECEWISE`**: **already the vLLM default** in this fork (confirmed in
  the config dump). Explicitly setting it is a no-op. A thermally-controlled 122B same-session A/B
  of the full opt set measured **+0.2%** (neutral). Apparent 35B/27B "regressions" vs the sweep
  baselines were **thermal drift** from ~9 h continuous load, not the opts.
- **`VLLM_MARLIN_USE_ATOMIC_ADD=1`**: within run-to-run noise on the 35B (marlin); a **no-op** on the
  27B (dense → FlashInfer-Cutlass GEMM, not marlin). Kept because it's free.
- **`VLLM_USE_FLASHINFER_SAMPLER=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`**: output-safe, neutral — kept.
- **MAXN power mode**: that's removing a power/thermal ceiling (overclocking), not a software lever.
  All benchmarks here are 120W; MAXN is a separate, reversible knob.
- **Restricting `cudagraph_capture_sizes` to `[1,k+1]`**: no conc=1 benefit and would force eager
  fallback at conc>1 — **not** baked into the serve defaults.
- **Dead ends:** `--moe-backend triton` (rejected for NVFP4), `VLLM_USE_FLASHINFER_MOE_FP4=1`
  (no sm_110a kernel), FA3 (Hopper TMA), prefix-caching with GDA hybrid on long context (risky).

---

## Operational rules (Thor-specific)
1. `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1` — or `import vllm._C` dies with
   `undefined symbol: cuPointerGetAttribute`.
2. Stop with **`docker kill`** / `pkill -9`, **never `docker stop`** (hangs + page-cache leak).
3. **Never `sudo fuser -k /dev/nvidia*`** — it kills Xorg/gnome-shell/RustDesk (display, not CUDA);
   `docker kill` + `drop_caches` fully releases the GPU context.
4. `gpu_memory_utilization` allocates **unified** memory (= system RAM). Too high starves the OS and
   hard-crashes the box — the 122B is the danger case (0.90 trips the precheck; use 0.78).
5. On Thor, `nvidia-smi` reports `[N/A]` for GPU memory — track `free -h` instead.
6. `drop_caches` before every 122B load (the 70 GB weight file fills the unified page cache).
7. 27B ships `tokenizer_class=TokenizersBackend` → patch overlay to `Qwen2Tokenizer` (`--tokenizer`).
8. `kv-cache-dtype auto` (BF16) for the DFlash draft path; the draft can reject quantized KV.

---

## Files
- `scripts/serve-{35b,27b,122b}.sh`, `scripts/_common.sh` — proven launch wrappers
- `scripts/ksweep-{4b,35b,27b,122b}.sh`, `scripts/bench.py`, `scripts/test-atomic-add.sh` — repro
- `benchmarks/{4b,35b,27b,122b-a10b}-optimizations.md` — per-model detail
- `moe-profiles/{35b-a3b,122b-a10b}-moe-profile.json` — MoE backend selection data
- `REVERT.md` — exact proven launch commands + revert steps · `BUILD-AND-DEBUG.md` — image build +
  122B debug journey · `hypotheses.md` — what was tried · `RESULTS.md` — everything

## Model weights
- Bases (NVFP4): `Qwen3.5-4B-NVFP4`, `Qwen3.6-35B-A3B-NVFP4`, `Qwen3.6-27B-NVFP4`,
  resharded `Qwen3.5-122B-A10B-NVFP4`
- DFlash drafts (`hf download z-lab/<name>`): `z-lab/Qwen3.5-4B-DFlash`, `z-lab/Qwen3.6-35B-A3B-DFlash`,
  `z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.5-122B-A10B-DFlash`

## Credits
DFlash is by [z-lab](https://github.com/z-lab/dflash) (Chen, Liang, Liu — arXiv:2602.06036). This
repo is the Thor SM110a port, debugging, and benchmarking. vLLM PR #40898.

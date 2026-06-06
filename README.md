# DFlash speculative decoding on Jetson AGX Thor (SM110a)

Block-diffusion speculative decoding (**DFlash**) for Qwen3.6 / Qwen3.5 NVFP4 models on a
single **NVIDIA Jetson AGX Thor** (aarch64, compute **sm_110a**, ~117 GB usable unified
LPDDR5X, CUDA 13, torch 2.10). vLLM built from source with DFlash support.

> Benchmark numbers and the full debugging log are in **[RESULTS.md](RESULTS.md)**.

## Hardware / platform
- Jetson AGX Thor, SM110a (`compute_110a` / `sm_110`), aarch64, L4T r38 / JetPack, CUDA 13.0
- ~117 GB unified memory shared between CPU and GPU (no discrete VRAM)
- torch 2.10.0 (cu13), vLLM `0.20.0.dev0+dflash`

## What DFlash is
DFlash is speculative decoding where the **drafter is a lightweight block-diffusion model**
that proposes a whole block of tokens in parallel, which the target then verifies. Block
size 16 → `num_speculative_tokens = 15` (`k=15`). Because the drafter denoises a masked
block, its attention is **non-causal (bidirectional)** — which is why the target+drafter
must run on the `flash_attn` backend, not flashinfer.

**Acceptance length τ** = mean tokens accepted per target forward pass (higher = faster).
Measured here: **τ ≈ 4.8–8.5** depending on how predictable the output is.

## Build (vLLM from source for sm_110a)
The stock `nvidia-ai-iot/vllm:latest-jetson-thor` is vLLM 0.19.0 with **no DFlash**. DFlash
landed on `main` after 0.19.0, so a full from-source build was required:
- Base image: `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` (toolchain, torch 2.10, CUDA 13)
- Source: vLLM **PR #40898** head (`vllm @ git+…@refs/pull/40898/head`) — DFlash + SWA
- `TORCH_CUDA_ARCH_LIST=11.0a`, ~92 min compile → native **sm_110 SASS** in `_C.abi3.so`
- Image tag: `vllm-dflash-thor:fa-native`

See [`reproduce-build.sh`](reproduce-build.sh) and `Dockerfile.dflash-thor` /
`Dockerfile.fa-native`.

### FA2 vs FA3 on Thor (important)
The from-source build produced flash-attn with **no native Thor kernels** (FA2 sm_80, FA3
sm_75). The stock image has **native sm_110** FA. We confirmed the FA C++ op symbols are
byte-identical between the two (arch-only difference) and built `:fa-native` =
`:latest` + the stock sm_110 FA `.so`. Findings:
- **FA3 native sm_110 CRASHES on Thor** — `CUTE_ARCH_TMA_SM90_ENABLED` →
  `cudaErrorLaunchFailure`. FA3's kernel needs Hopper **SM90 TMA**; Thor doesn't have it.
- **FA2 native sm_110 works.** vLLM selects FA2 (not FA3), so the server runs clean.
- **FA2 vs FA3 = identical decode tok/s** — decode is bound by the **MoE GEMM /
  NVFP4 weight bandwidth**, not attention. Native FA mainly helps long-context prefill.

## MoE backend: Marlin wins
NVFP4 MoE backend comparison on the 35B (Dijkstra task):

| backend | result |
|---|---|
| **marlin** | ✅ **69.3 tok/s** — fastest, **default** |
| cutlass | ✅ 56.0 tok/s |
| flashinfer_cutlass | ❌ `does not support current device cuda` (Thor FP4 breakage) |
| triton | ❌ `not supported for NvFP4 MoE` (triton can't do NVFP4 at all) |

Valid NVFP4 MoE backends: `cutlass, flashinfer_trtllm, flashinfer_cutlass,
flashinfer_cutedsl, marlin, emulation`. τ is identical across backends (MoE doesn't affect
acceptance). **Use `--moe-backend marlin`.**

## Per-model memory utilization (and why)
| model | params (active) | gpu_mem_util | max_len | why |
|---|---|---|---|---|
| Qwen3.6-35B-A3B-NVFP4 | 35B (3B active, MoE) | **0.78** | 65536 | ~21 GB weights + draft + KV; DFlash drafter needs its own KV (0.32 capped ctx at ~77k) |
| Qwen3.6-27B-NVFP4 | 27B (dense) | **0.85** | 65536 | dense 27B → more weight resident; headroom for KV |
| Qwen3.5-122B-A10B-NVFP4 | 122B (10B active, MoE) | see RESULTS | 16–32k | 72 GB weights on a 117 GB unified box — very tight; high util risks **system** OOM |

> ⚠️ On Thor, `gpu_memory_utilization` allocates **unified** memory (= system RAM). Setting
> it too high starves the OS and can hard-crash the box (the 122B is the danger case).

## Benchmarks (single request, T=0, marlin, DFlash k=15)
| Task | 35B-A3B (MoE) | 27B (dense) |
|---|---|---|
| sorting  | 111.2 tok/s (τ 8.54) | 31.3 (τ 6.83) |
| lru      |  83.9 tok/s (τ 6.36) | 35.5 (τ 5.22) |
| dijkstra |  94.7 tok/s (τ 4.76) | 32.8 (τ 4.83) |
| mixed    | 117.5 tok/s (τ 5.98) | 37.7 (τ 5.59) |

The "smaller" 27B is **~3× slower** than 35B-A3B: it's **dense** (27B active params) vs the
35B MoE's **3B active** — decode is bandwidth-bound on active params. (122B numbers: see
RESULTS.md.)

## Key gotchas
1. **`LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`** is required, or
   `import vllm._C` dies with `undefined symbol: cuPointerGetAttribute`.
2. **Stop with `docker kill` / `pkill -9`, NEVER `docker stop`** (Thor page-cache leak / hang).
3. **`gpu_memory_utilization` is unified memory** — too high crashes the OS (esp. 122B).
4. **PIECEWISE CUDA graphs need capture sizes that are multiples of `k+1=16`** (else
   `cudaErrorIllegalAddress` on partial accepts).
5. **No `kv-cache-dtype fp8`** — DFlash rejects quantized KV (non-causal draft attn). Use `auto` (BF16).
6. **No `--enable-prefix-caching`** with DFlash + hybrid attn (known bug).
7. **`--attention-backend flash_attn`** (drafter is non-causal); flashinfer for the drafter is unsupported.
8. **FA3 crashes on Thor** (Hopper TMA); FA2 is the correct kernel.
9. **27B tokenizer** ships `tokenizer_class=TokenizersBackend` → patch overlay to `Qwen2Tokenizer`.

## Run
```bash
# requires: docker, the vllm-dflash-thor:fa-native image, models under ~/ and ~/models
scripts/serve-35b.sh        # Qwen3.6-35B-A3B  on :8001  (marlin, DFlash k=15)
scripts/serve-27b.sh        # Qwen3.6-27B      on :8001
scripts/serve-122b.sh       # Qwen3.5-122B-A10B on :8001 (memory-tight; see RESULTS)
python3 scripts/bench.py --url http://localhost:8001 --tasks all --runs 3
```
MoE backend / FP4 toggle (Step 3 A/B): `MOE_BACKEND=cutlass scripts/serve-35b.sh`.

## Reproduce the image
```bash
./reproduce-build.sh        # builds vllm-dflash-thor:{latest,fa-native} from source (~90 min)
```
Or load the prebuilt image (aarch64 / SM110a only):
`docker load < vllm-dflash-thor-fa-native.tar.gz`
(see the HF image repo: `patrickbdevaney/qwen-3.6-35b-a3b-dflash-jetson-agx-thor`).

## Model weights
- Bases (NVFP4, compressed-tensors): `unsloth/Qwen3.6-35B-A3B-NVFP4`, `unsloth/Qwen3.6-27B-NVFP4`,
  resharded `Qwen3.5-122B-A10B-NVFP4`
- DFlash drafts (`hf download z-lab/<name>`): `z-lab/Qwen3.6-35B-A3B-DFlash`,
  `z-lab/Qwen3.6-27B-DFlash`, `z-lab/Qwen3.5-122B-A10B-DFlash`

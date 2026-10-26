# DFlash Speculative Decoding on Jetson AGX Thor with vLLM

**A from-source vLLM image for `sm_110a`, per-model launch configurations, a root-caused fix for the Qwen3.5-122B
load crash, and full k-sweeps for four Qwen NVFP4 models at concurrency 1**

| | |
|---|---|
| **Technique** | [DFlash](https://github.com/z-lab/dflash) block-diffusion speculative decoding (Chen, Liang, Liu; arXiv:2602.06036): a non-causal drafter denoises a 16-token block in parallel |
| **Targets** | Qwen3.5-4B, Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B, all NVFP4 (compressed-tensors), with [z-lab](https://huggingface.co/z-lab) DFlash drafters |
| **Runtime** | vLLM `0.20.0.dev0+dflash`, built from source at PR [#40898](https://github.com/vllm-project/vllm/pull/40898) head for `TORCH_CUDA_ARCH_LIST=11.0a`; FlashAttention-2 (`sm_110`); fastsafetensors 0.3.2 |
| **Hardware** | NVIDIA Jetson AGX Thor: `sm_110a`, 128 GB unified LPDDR5X (~117 GB usable), 273 GB/s, aarch64, CUDA 13.0, L4T r38, 120 W mode |
| **Image** | `vllm-dflash-thor:fastsafe`, 18.3 GiB tarball on [Hugging Face](https://huggingface.co/patrickbdevaney/qwen-3.6-35b-a3b-dflash-jetson-agx-thor) |
| **Paper** | [`dflash_vllm_thor_paper.pdf`](dflash_vllm_thor_paper.pdf), a technical report of these measurements |
| **Validated** | 2026-06-06 |

---

## Summary

The stock Jetson Thor vLLM image (0.19.0) has no DFlash. DFlash landed on vLLM `main` after 0.19.0. PR #40898 is
1,138 commits ahead and touches 165 C++/CUDA files, so a Python overlay on the stock binaries is not possible. This
repository builds vLLM whole for `sm_110a` and records what it took to make four model sizes stable on one Thor at
concurrency 1, the realistic single-user edge regime.

| Model | Type | MoE backend | Attention | Best k (4-task mean) | Mean tok/s at best k | Peak tok/s | Speedup over autoregressive |
|---|---|---|---|--:|--:|--:|---|
| Qwen3.5-4B | dense, gated-delta hybrid | — | `flash_attn` | 10 (15 for peak) | 143.4 | **155.8** | **3.2×** (Dijkstra: 47.9 → 152.2) |
| Qwen3.6-35B-A3B | MoE, 3B active | **marlin** | `flash_attn` | 12 | 116.5 | **139.1** | not measured |
| Qwen3.6-27B | dense | — | `flash_attn` | 15 | 42.3 | **50.1** | not measured |
| Qwen3.5-122B-A10B | MoE, 256 experts | **cutlass** | **`TRITON_ATTN`** | 10 | 44.8 | **53.2** | see note |

Measured at concurrency 1 and temperature 0 in 120 W mode. Each cell is the median of 3 runs over 4 coding tasks
(sorting, LRU cache, Dijkstra, mixed).

**122B speedup note:** the only non-speculative 122B measurement is one cold run at 10.9 tok/s on Dijkstra. Against
it, DFlash Dijkstra at k=10 is 40.0 tok/s (3.7×). A cold run understates warm decode, so 3.7× is an upper bound. An
earlier first-light run recorded 1.7×. A warm autoregressive baseline is the open measurement.

---

## The 122B load crash and its fix

Qwen3.5-122B-A10B (256 experts × 48 layers) crashed on every launch for weeks. The failure always looked the same:
- after ~60 s of weight loading, the engine core exits with code 255;
- there is no Python traceback, Docker reports `OOMKilled=false`, and the host logs no OOM;
- the last log line is `nvfp4.py:491 Using MoEPrepareAndFinalizeNoDPEPModular`, the Marlin NVFP4 MoE weight repack.

**Ruled out by reproduction:**

| Hypothesis | Test | Result |
|---|---|---|
| DFlash, sliding-window draft, or config | Base model alone, no speculative config | Same crash at the same line |
| Out of memory | Docker state, `dmesg`, journal | No OOM anywhere |
| Missing kernel image | `cuobjdump` of `_moe_C.abi3.so` | Contains `sm_110a` |
| Alignment | MoE dims 1024 and 3072 | Both 128-aligned |

The cause is a runtime fault inside the Marlin FP4 MoE kernel at 256-expert scale on Thor. It is consistent with
open vLLM issues #35519, #35566 and #35922, and a rebuild does not fix it. The 35B (fewer experts) loads fine on
Marlin.

**The fix: three settings, each forced by a measured failure**

| Setting | Why |
|---|---|
| `--moe-backend cutlass` | The only NVFP4 MoE backend that loads all 256 experts on Thor (70.46 GiB, no fault). `triton` is rejected for NVFP4, and the `flashinfer_*` MoE paths have no `sm_110a` kernels |
| `--attention-backend TRITON_ATTN` | With cutlass, FlashInfer attention fails in warm-up: `BatchDecodeWithPagedKVCacheWrapper.run() got an unexpected keyword 'kv_cache_sf'`, an API mismatch in this build |
| `--gpu-memory-utilization 0.78` with draft `max_model_len: 1024` | The draft's KV cache takes ~2 GB. At 0.72 the 16,384-token target KV does not fit ("estimated max length 8960"); at 0.90 the start-up precheck fails (needs 110.5 GB, ~108 GB free). 0.78 gives a ~71K-token KV cache |

**Kernel availability in this image** (`cuobjdump`):

| Library | Architectures | Consequence |
|---|---|---|
| `_moe_C.abi3.so` (Marlin MoE) | `sm_110a`, `sm_80` | Present, but faults at 256 experts |
| `_C.abi3.so` | `sm_110a`, `sm_80`, `sm_90` | Core kernels |
| FlashInfer `fused_moe_103.so` | `sm_100a`, `sm_103a` | No Thor build; set `VLLM_USE_FLASHINFER_MOE_FP4=0` |
| `gemm_sm120.so`, `trtllm_low_latency_gemm.so` | `sm_120`, `sm_100a` | No Thor build |

---

## The image

| Layer | Detail |
|---|---|
| Base | `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor` (CUDA 13.0, cmake 3.31, gcc 13.3, Python 3.12, torch 2.10.0) |
| vLLM | PR #40898 head built whole for `11.0a` (native `sm_110a` SASS in `_C` and `_moe_C`); build time ≈ 92 min. The PR pins torch 2.11; the image keeps 2.10 via `use_existing_torch.py` |
| Attention | FlashAttention-2 for `sm_110`, copied from the stock image. FA3 crashes on Thor because it requires Hopper (`sm_90`) TMA; vLLM selects FA2 automatically. FA2 versus FA3 makes no decode difference here |
| Weight loading | fastsafetensors 0.3.2 (`Dockerfile.fastsafe`), required for the 122B |

**Why fastsafetensors is required on unified memory.** The default loader memory-maps the safetensors file on the
CPU side and then copies it to the GPU. On Thor both copies live in the same 117 GB pool. For the 122B that is
72 GB + up to 72 GB = 144 GB, and the system crashes during load. fastsafetensors streams from disk to GPU directly,
peaking near 72 GB. `FASTSAFETENSORS_NOGDS=1` selects its non-GDS path, since GPU Direct Storage is unsupported on
Thor.

### Get the image

```bash
hf download patrickbdevaney/qwen-3.6-35b-a3b-dflash-jetson-agx-thor vllm-dflash-thor-fastsafe.tar.gz --local-dir .
docker load < vllm-dflash-thor-fastsafe.tar.gz              # -> vllm-dflash-thor:fastsafe (aarch64, sm_110a only)
```

To build it yourself, use [`Dockerfile.dflash-thor`](Dockerfile.dflash-thor), then
[`Dockerfile.fa-native`](Dockerfile.fa-native) (FA2 from the stock image), then
[`Dockerfile.fastsafe`](Dockerfile.fastsafe). [`reproduce-build.sh`](reproduce-build.sh) drives the build, and
[`BUILD-AND-DEBUG.md`](BUILD-AND-DEBUG.md) records the build and the 122B debugging path.

---

## Deployment

### Environment (all models)

```bash
-e VLLM_USE_FLASHINFER_MOE_FP4=0                                # mandatory: no sm_110a FlashInfer FP4 MoE kernel
-e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1    # else `import vllm._C` fails: undefined cuPointerGetAttribute
-e FASTSAFETENSORS_NOGDS=1 -e HF_HUB_DISABLE_XET=1 -e NCCL_IGNORE_CPU_AFFINITY=1
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
-e VLLM_USE_FLASHINFER_SAMPLER=1 -e CUDA_DEVICE_MAX_CONNECTIONS=1   # output-safe, measured neutral
```

All models also use these flags:
- `--load-format fastsafetensors`;
- `--quantization compressed-tensors`. The 4B checkpoint declares `modelopt` but loads correctly this way;
- `--trust-remote-code`;
- `--language-model-only`, because the 4B and 122B are vision-language wrappers.

### Per-model settings

| Model | MoE | Attention | `gpu_memory_utilization` | `max_model_len` | `max_num_seqs` | Speculative config |
|---|---|---|--:|--:|--:|---|
| 4B | — | `flash_attn` | 0.40 | 32,768 | 1 | `dflash`, k=15 |
| 35B-A3B | `marlin` | `flash_attn` | 0.78 | 65,536 | 4 | `dflash`, k=12 |
| 27B | — | `flash_attn` | 0.85 | 65,536 | 4 | `dflash`, k=15; `--tokenizer` overlay to `Qwen2Tokenizer` |
| 122B-A10B | `cutlass` | `TRITON_ATTN` | 0.78 | 16,384 | 2 | `dflash`, k=10, draft `max_model_len: 1024` |

```bash
# example: 122B-A10B
--moe-backend cutlass --attention-backend TRITON_ATTN \
--gpu-memory-utilization 0.78 --max-model-len 16384 --max-num-seqs 2 \
--speculative-config '{"method":"dflash","num_speculative_tokens":10,"model":"/draft","max_model_len":1024}'
```

Wrappers: [`scripts/serve-35b.sh`](scripts/serve-35b.sh), [`serve-27b.sh`](scripts/serve-27b.sh),
[`serve-122b.sh`](scripts/serve-122b.sh). Complete `docker run` commands for all four models, pinned to the
known-good commit `e49459e`, are in [`REVERT.md`](REVERT.md).

### Pre-flight and verification

```bash
for c in $(docker ps -q); do docker kill "$c"; done           # never `docker stop` on Thor (hangs, leaks page cache)
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'   # mandatory before the 122B
free -h | grep Mem                                            # expect > 100 GB free; nvidia-smi shows [N/A] on Thor
./verify-dflash.sh http://localhost:<port>                    # coherence check + spec_decode acceptance metrics
```

### Weights

| Target (NVFP4) | Drafter (`hf download z-lab/<name>`) |
|---|---|
| Qwen3.5-4B-NVFP4 | `z-lab/Qwen3.5-4B-DFlash` |
| Qwen3.6-35B-A3B-NVFP4 | `z-lab/Qwen3.6-35B-A3B-DFlash` |
| Qwen3.6-27B-NVFP4 | `z-lab/Qwen3.6-27B-DFlash` |
| Qwen3.5-122B-A10B-NVFP4, resharded ([`patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded`](https://huggingface.co/patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded)) | `z-lab/Qwen3.5-122B-A10B-DFlash` |

---

## Benchmarks

Concurrency 1, temperature 0, thinking off, 120 W. Each cell is `tok/s / τ` (τ = mean accepted length per verify),
the median of 3 runs. Raw tables are in [`ksweep-data/`](ksweep-data/); per-model write-ups are in
[`benchmarks/`](benchmarks/); everything is collected in [`RESULTS.md`](RESULTS.md).

**Qwen3.6-35B-A3B** (marlin, `flash_attn`)

| k | sorting | lru | dijkstra | mixed | mean tok/s | τ mean |
|--:|--|--|--|--|--:|--:|
| 8 | 121.4 / 6.61 | 98.8 / 5.72 | 82.1 / 4.50 | 89.6 / 5.03 | 98.0 | 5.46 |
| 10 | 130.4 / 7.55 | 97.2 / 5.93 | 93.2 / 4.66 | 127.9 / 5.52 | 112.2 | 5.92 |
| **12** | 137.1 / 8.56 | 100.5 / 6.08 | 104.0 / 4.76 | 124.2 / 5.61 | **116.5** | 6.25 |
| 15 | 139.1 / 8.86 | 98.7 / 6.27 | 98.0 / 4.81 | 111.1 / 5.35 | 111.7 | 6.32 |

MoE backend at k=12: marlin 117.5 tok/s mean against cutlass 106.3, so marlin is the default for this model
([`moe-profiles/`](moe-profiles/)).

**Qwen3.6-27B** (dense, `flash_attn`)

| k | sorting | lru | dijkstra | mixed | mean tok/s | τ mean |
|--:|--|--|--|--|--:|--:|
| 8 | 41.9 / 5.89 | 37.7 / 5.08 | 35.8 / 4.77 | 37.1 / 4.94 | 38.1 | 5.17 |
| 10 | 48.6 / 6.62 | 37.1 / 5.00 | 36.8 / 4.98 | 39.6 / 5.33 | 40.5 | 5.48 |
| 12 | 44.8 / 6.22 | 37.2 / 5.11 | 37.5 / 5.12 | 45.9 / 6.34 | 41.4 | 5.70 |
| **15** | 50.1 / 7.04 | 38.4 / 5.33 | 39.8 / 5.55 | 40.9 / 5.68 | **42.3** | 5.90 |

**Qwen3.5-122B-A10B** (cutlass, `TRITON_ATTN`)

| k | sorting | lru | dijkstra | mixed | mean tok/s | τ mean |
|--:|--|--|--|--|--:|--:|
| 8 | 25.4 / 5.16 | 40.5 / 4.47 | 34.3 / 3.71 | 43.1 / 4.78 | 35.8 | 4.53 |
| **10** | 52.6 / 6.36 | 45.9 / 5.36 | 40.0 / 4.49 | 40.5 / 4.60 | **44.8** | 5.20 |
| 12 | 53.2 / 6.45 | 44.2 / 5.18 | 35.5 / 4.20 | 39.6 / 4.66 | 43.1 | 5.12 |

The sorting task varies widely between sessions: an earlier k=12 run measured 27.2 tok/s on the same task. Treat
single-task 122B figures as ±50%; the 4-task means are more stable.

**Qwen3.5-4B** (dense gated-delta hybrid, `flash_attn`; autoregressive Dijkstra baseline 47.9 tok/s)

| k | sorting | lru | dijkstra | mixed | mean tok/s | τ mean |
|--:|--|--|--|--|--:|--:|
| 8 | 120.7 / 5.08 | 108.1 / 4.48 | 148.1 / 4.52 | 142.2 / 4.34 | 129.8 | 4.61 |
| 10 | 129.3 / 5.42 | 145.2 / 4.72 | 152.2 / 4.68 | 146.7 / 4.50 | **143.4** | 4.83 |
| 12 | 143.8 / 6.19 | 139.3 / 5.12 | 143.6 / 4.44 | 127.2 / 3.92 | 138.5 | 4.92 |
| 15 | 130.6 / 5.75 | 141.0 / 4.57 | 135.5 / 4.37 | **155.8** / 5.01 | 140.7 | 4.92 |

k=10 and k=15 are within noise of each other on the 4B.

---

## Findings

**1. Optimal k falls as the target gets cheaper per verify.** Best k by 4-task mean was:

| Model | Best k |
|---|--:|
| 122B-A10B | 10 |
| 35B-A3B | 12 |
| 27B | 15 |
| 4B | 10–15 (within noise) |

A dense 27B pays a full-weight forward on every verify, so each accepted token saves a lot and a deep draft is
worth it. A fast MoE (35B, 3B active) reaches the point where draft overhead exceeds the acceptance gain sooner. At
k=15 the 35B has its highest τ (6.32) but lower throughput. **Per-model k is the highest-value tuning lever
measured here.**

**2. DFlash exceeds the autoregressive bandwidth ceiling on three of four models.** The ceiling here is 273 GB/s
divided by the NVFP4 weight bytes read per token. Verification processes k+1 tokens per weight pass, so speculative
throughput is not bounded by it.

| Model | Bytes-per-token basis | Autoregressive ceiling | Best DFlash | Ratio |
|---|---|--:|--:|--:|
| 4B | 2.0 GB | 136.5 | 155.8 | 1.14× |
| 27B | 13.5 GB | 20.2 | 50.1 | 2.48× |
| 35B-A3B | ~3B active | 182.0 | 139.1 | 0.76× |
| 122B-A10B | ~12B read per token (routed, shared, attention) | 45.5 | 53.2 | 1.17× |

The 35B is already the fastest target per token, which leaves the least for drafting to hide.

**3. At concurrency 1 the base models are overhead-bound.** The 4B decodes at 47.9 tok/s autoregressively, 35% of
its bandwidth ceiling, because there is no batch to amortize launch and synchronization cost. That is the regime
where a deep, high-acceptance drafter pays most: τ here is 3.7–8.9.

Context, not measured here: MTP heads typically draft 3–4 tokens with shallower acceptance, and z-lab reports up to
2.9× over autoregressive at concurrency 1.

**4. Several optimizations did not help.** These are recorded so others don't chase them
([`hypotheses.md`](hypotheses.md)):

| Change | Result |
|---|---|
| `cudagraph_mode FULL_AND_PIECEWISE` | Already the default in this build. A same-session 122B A/B of the full option set measured +0.2% |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | Within noise on the 35B; no-op on the 27B (no Marlin path) |
| FlashInfer sampler, `CUDA_DEVICE_MAX_CONNECTIONS=1` | Neutral; kept because they are output-safe |
| `cudagraph_capture_sizes: [1, k+1]` | No benefit at concurrency 1, and forces eager mode at concurrency > 1; not adopted |
| `--moe-backend triton`, `VLLM_USE_FLASHINFER_MOE_FP4=1`, FA3 | Not available on Thor (rejected, no kernel, needs `sm_90` TMA) |
| MAXN power mode | A power and thermal ceiling change, not a software lever. All numbers here are 120 W |

Apparent 35B and 27B regressions in later runs were traced to thermal drift after ~9 h of continuous load at 120 W,
not to configuration.

---

## Operational rules for Thor

1. **Set `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`.** Without it, `import vllm._C` fails with
   `undefined symbol: cuPointerGetAttribute`.
2. **Stop containers with `docker kill`, never `docker stop`.** `docker stop` hangs and leaks page cache.
3. **Never run `sudo fuser -k /dev/nvidia*`.** It kills the display stack (Xorg, GNOME, remote desktop), not CUDA.
4. **`gpu_memory_utilization` allocates unified memory, which is system RAM.** Too high starves the OS and can hard-crash
   the device.
5. **Watch `free -h`, not `nvidia-smi`.** `nvidia-smi` reports `[N/A]` for memory on Thor.
6. **Drop caches before loading the 122B.** Its 70 GB weight file otherwise fills the unified page cache.
7. **The 27B needs a tokenizer overlay.** It ships `tokenizer_class=TokenizersBackend`; point `--tokenizer` at a
   `Qwen2Tokenizer` overlay.
8. **Use `kv_cache_dtype auto` (BF16)** on the DFlash path; the draft can reject a quantized KV cache.

---

## Limitations

- **Concurrency 1 only.** Throughput under batching was not measured.
- **Coding tasks only, 4 prompts.** Acceptance, and therefore speed, depends on content.
- **Autoregressive baselines are incomplete.** The 4B has a measured baseline. The 122B has one cold run. The 35B and
  27B have none.
- **No accuracy evaluation.** Outputs were checked for coherence (`verify-dflash.sh`), not scored. Speculative
  decoding is designed to preserve the target's greedy output, but that was not verified token by token here.
- **One device, one build.** Results are tied to PR #40898 head and this image. Later vLLM versions may change both
  the kernel choices and the 122B crash behavior.

---

## Repository layout

| Path | Contents |
|---|---|
| [`Dockerfile.dflash-thor`](Dockerfile.dflash-thor), [`Dockerfile.fa-native`](Dockerfile.fa-native), [`Dockerfile.fastsafe`](Dockerfile.fastsafe), [`reproduce-build.sh`](reproduce-build.sh) | Image build: vLLM from source for `sm_110a`, FA2 overlay, fastsafetensors |
| [`scripts/serve-*.sh`](scripts/), [`scripts/_common.sh`](scripts/_common.sh) | Validated launch wrappers |
| [`scripts/ksweep-*.sh`](scripts/), [`scripts/bench.py`](scripts/bench.py), [`scripts/test-atomic-add.sh`](scripts/test-atomic-add.sh) | Benchmark and A/B reproduction |
| [`ksweep-data/`](ksweep-data/), [`moe-profiles/`](moe-profiles/) | Raw sweep tables and MoE backend profiles |
| [`benchmarks/`](benchmarks/) | Per-model optimization write-ups |
| [`RESULTS.md`](RESULTS.md), [`BUILD-AND-DEBUG.md`](BUILD-AND-DEBUG.md), [`REVERT.md`](REVERT.md), [`hypotheses.md`](hypotheses.md) | All results; build and debug narrative; known-good commands; what was tried |
| [`serve-dflash.sh`](serve-dflash.sh), [`qwen36-dflash-config.yaml`](qwen36-dflash-config.yaml), [`verify-dflash.sh`](verify-dflash.sh), [`bench-decode.sh`](bench-decode.sh) | First-light 35B-A3B setup (262K context, `gpu_memory_utilization` 0.50) and the verification suite |
| [`dflash_vllm_thor_paper.pdf`](dflash_vllm_thor_paper.pdf) | Technical report |

## License

- **This repository** (Dockerfiles, scripts, configurations, documentation): MIT ([`LICENSE`](LICENSE)).
- **vLLM:** Apache-2.0. **DFlash drafters:** see the z-lab model cards. **Qwen models:** see their model cards.
- **NVIDIA Jetson container images** are governed by their NVIDIA license terms.

This is an independent community project, not affiliated with NVIDIA, the vLLM project, z-lab or the Qwen team.

## Citation

```bibtex
@misc{devaney2026dflashthor,
  title  = {DFlash Block-Diffusion Speculative Decoding on NVIDIA Jetson AGX Thor (SM110a) with vLLM},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/dflash-vllm-resources}
}
```

DFlash is the work of z-lab (Chen, Liang, Liu; arXiv:2602.06036). This repository contributes the Thor `sm_110a`
port, the debugging and the measurements.

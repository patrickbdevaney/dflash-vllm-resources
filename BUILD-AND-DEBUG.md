# Build & Debug Journey — DFlash NVFP4 on Jetson AGX Thor SM110a

This documents how the working image was built and the debugging path that took the 122B from
"crashes on every launch for weeks" to "stable DFlash at 1.7×+ uplift". Two companion archives
hold the runnable image: `vllm-dflash-thor-fa-native.tar.gz` and (with fastsafetensors)
`vllm-dflash-thor-fastsafe.tar.gz`.

## 1. Building the image with fastsafetensors

Base build: `vllm-dflash-thor:fa-native` = vLLM `0.20.0.dev0+dflash` (PR #40898 head), compiled
from source for `sm_110a` (Thor Blackwell), with native sm_110 flash-attention copied from the
stock Jetson-Thor image (FA2 works; FA3 needs Hopper SM90 TMA → unusable on Thor). See
`Dockerfile.dflash-thor` / `Dockerfile.fa-native`.

fastsafetensors layer (`Dockerfile.fastsafe`):
```dockerfile
FROM vllm-dflash-thor:fa-native
RUN /opt/venv/bin/pip install --no-cache-dir fastsafetensors \
    && python -c "import fastsafetensors; print('OK', fastsafetensors.__version__)"
```
Result: `vllm-dflash-thor:fastsafe` (== `fa-native`, same image ID, fastsafetensors **0.3.2**).

WHY fastsafetensors matters on Thor unified memory: the default safetensors loader mmaps the
weight file (CPU side) and then copies to GPU. On Thor's single 117GB unified pool both copies
coexist: for the 122B that is 72GB(GPU) + up to 72GB(CPU mmap) = 144GB > 117GB → the system
crashes during load. fastsafetensors streams disk→GPU directly (no CPU staging), so the 72GB
load peaks near 72GB instead of 144GB. `FASTSAFETENSORS_NOGDS=1` is set because GDS (GPU Direct
Storage) is unsupported on this platform — it falls back cleanly and still avoids CPU staging.

## 2. 35B-A3B in Marlin (the straightforward case)

The 35B-A3B (and 27B) load and serve cleanly with `--moe-backend marlin --attention-backend
flash_attn`. Marlin is the NVFP4 weight-only MoE path; on Thor it has a real `sm_110a` kernel
image (`_moe_C.abi3.so` contains sm_110a). For the 35B's expert count it processes weights and
runs without issue. MoE-backend profile (35B @ k=12): marlin 117.5 avg tok/s vs cutlass 106.3 →
**marlin is the 35B default** (~10% faster). VLLM_USE_FLASHINFER_MOE_FP4=0 is mandatory (the
FlashInfer FP4 MoE kernels are sm_100a/103a/120 only — no sm_110a).

## 3. The 122B debug journey → why Cutlass

The 122B-A10B (256 experts × 48 layers) crashed on every launch for weeks. The path to root cause:

1. **Load-time OOM (early)** — fixed with fastsafetensors + `vm.vfs_cache_pressure=1000` + drop
   caches before load. Weights then loaded reliably (~55-60s).
2. **Post-load silent death** — after "Loading weights took 60s", the EngineCore subprocess died
   with **exit 255, no Python traceback, Docker OOMKilled=false, and NO host kernel OOM line**,
   at exactly: `marlin_utils_fp4.py:300` (weight-only FP4) → `nvfp4.py:491 Using
   MoEPrepareAndFinalizeNoDPEPModular`. i.e. the **Marlin NVFP4 MoE weight-repack**.
3. **Ruled out the red herrings** (the long chase): NOT DFlash, NOT the SWA draft, NOT
   kv-cache-memory-bytes, NOT compilation-config, NOT gpu-util/precheck, NOT OOM — proven by
   reproducing the identical crash with the BASE model alone (no speculative config).
4. **Ruled out "missing kernel image"** via `cuobjdump`: `_moe_C.abi3.so` DOES contain `sm_110a`,
   and the MoE dims are 128-aligned (1024=8×128, 3072=24×128) → not the #38022 alignment bug.
   Conclusion: a genuine **in-kernel fault in the Marlin FP4 MoE kernel at 256-expert scale** on
   Thor, matching open vLLM bugs #35566 / #35519 / #35922. A rebuild does NOT fix it.
5. **The fix — escape Marlin**:
   - `--moe-backend triton` → REJECTED ("not supported for NvFP4 MoE").
   - `--moe-backend cutlass` (VLLM_CUTLASS) → **processes all 256 experts cleanly** (70.46 GiB
     model load, no crash). This is the breakthrough: cutlass is the only NVFP4 MoE backend that
     loads the 256-expert 122B on Thor. (Note: the OPPOSITE of the 35B, where marlin is faster
     and cutlass slower — marlin only *crashes* at the large expert count.)
   - cutlass + `--attention-backend flashinfer` then died in attention warmup:
     `TypeError: BatchDecodeWithPagedKVCacheWrapper.run() got an unexpected keyword argument
     'kv_cache_sf'` (FlashInfer API mismatch in this fork). Fix: `--attention-backend TRITON_ATTN`.
   - DFlash draft then needed `gpu-util 0.78` (it eats ~2GB KV; at 0.72 the 16384 KV request
     fails "estimated max length 8960"; 0.90 trips the startup precheck) and a draft KV cap
     `"max_model_len":1024`.

Final working 122B config: **cutlass MoE + TRITON_ATTN + gpu-util 0.78 + draft max_model_len
1024**, DFlash k=10 → 27-42 tok/s, acceptance τ 4.2-6.5, 1.7×+ over the 10.9 tok/s base.

## 4. cuobjdump arch facts (this image)
| extension | arches | note |
|---|---|---|
| `_moe_C.abi3.so` | sm_110a, sm_80 | Marlin MoE — HAS Thor image (but faults at 256 experts) |
| `_C.abi3.so` | sm_110a, sm_80, sm_90 | core kernels |
| flashinfer `fused_moe_103.so` | sm_100a, sm_103a | NO sm_110a → VLLM_USE_FLASHINFER_MOE_FP4=0 |
| `gemm_sm120.so` | sm_120 | no Thor |
| `trtllm_low_latency_gemm.so` | sm_100a | no Thor |

## 5. Operational rules learned
- Stop containers with `docker kill` (never `docker stop` — hangs + leaks page cache on Thor).
- Never `sudo fuser -k /dev/nvidia*` — it kills Xorg/gnome-shell/RustDesk (display, not CUDA
  compute) and drops the remote session. `docker kill` + `drop_caches` fully releases the GPU.
- On Thor unified memory, `nvidia-smi` reports `[N/A]` for GPU mem — track `free -h` instead.

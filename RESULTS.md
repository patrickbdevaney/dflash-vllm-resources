# DFlash on Jetson AGX Thor (SM110) — Results  (2026-06-06 03:13)

## Path taken: B (build required)
- Stock latest-jetson-thor = vLLM 0.19.0, NO DFlash. DFlash landed on main post-0.19.0.
- Built image: vllm-dflash-thor:latest  (vLLM 0.20.0.dev0+dflash, from PR #40898 head)
- Full from-source sm_110a compile (~92 min). sm_110 SASS native in _C.abi3.so.
- DFlash is pure-Python; sm_110 flash-attn already in base image (but rebuilt as FA2-PTX).

## Correctness (eager): PASS
- method=dflash active, DFlashDraftModel resolved, num_spec_tokens=15, global flash_attn.
- Coherent code output (thinking off), finish_reason=stop.

## Graph mode (PIECEWISE, capture sizes multiples of 16): PASS
- 6/6 CUDA graphs captured, NO cudaErrorIllegalAddress.
- Available KV: 23.11 GiB @ gpu_memory_utilization=0.50 (262k context OK, 1.75x concurrency).

## Decode speed (single request, T=0, thinking off, steady-state)
| Task                    | tok/s (e2e) | acceptance/draft | tau (tokens/step) |
|-------------------------|------------:|-----------------:|------------------:|
| Sorting (quick+merge)   |        81.6 |             6.94 |              7.94 |
| LRU cache               |        65.9 |             5.55 |              6.55 |
| Dijkstra                |        52.4 |             3.81 |              4.81 |
| bench avg (mixed)       |     ~54-66  |            ~4.4  |             ~5.4  |

Cumulative spec-decode counters at end: drafts=1312, draft_tokens=19680, accepted=6046
=> overall accepted/draft = 4.61, mean acceptance length tau ~ 5.61 tokens/step.

## Key fixes vs original brief
1. Build: checkout PR #40898 head + full source build (NOT clone-v0.19.0 + git-merge).
2. requirements/build.txt does not exist -> only setuptools-scm was missing.
3. Verify must run outside /build/vllm AND with --runtime nvidia (driver symbols).
4. serve: REQUIRES LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1
5. gpu_memory_utilization 0.32 -> 0.50 (DFlash drafter needs its own KV).
6. attention backend: global flash_attn (per official README), not target-flashinfer split.

## Flash-attention investigation (native sm_110 swap)
Our from-source build produced FA with NO native Thor kernels:
  _vllm_fa2_C = sm_80 only, _vllm_fa3_C = sm_75 only.
The stock latest-jetson-thor image has NATIVE sm_110 for both. The FA C++ op symbols
are byte-identical between the two (verified via nm) -> arch-only difference, ABI-safe.
Built vllm-dflash-thor:fa-native = our image + stock sm_110 FA .so (COPY --from).

Direct GPU kernel test (flash_attn_varlen_func):
  - FA2 native sm_110: WORKS (correct, finite output).
  - FA3 native sm_110: CRASHES -> "TMA Descriptor Prefetch without CUTE_ARCH_TMA_SM90_ENABLED"
    -> cudaErrorLaunchFailure. FA3's kernel needs Hopper SM90 TMA; UNUSABLE on Thor.
  vLLM selects FA2 (not FA3), so the server runs fine on both images.

Head-to-head decode (same prompts): fa-native vs original = +1% / -0% / +0% (IDENTICAL).
=> Decode is MoE-GEMM / NVFP4-bandwidth bound at these context lengths; FA arch does not
   move decode tok/s. Native FA2 mainly helps long-context PREFILL/TTFT (not measured H2H).
RECOMMENDATION: run vllm-dflash-thor:fa-native (strictly more correct, no downside, no
   FA3 crash risk). It is the current default in serve-dflash.sh and the running server.

## Open caveats / not done
- VLLM_FLASH_ATTN_VERSION env is unknown to this vLLM (harmless warning); can drop from Dockerfile.
- Native-FA2 long-context PREFILL/TTFT speedup vs sm_80 build NOT measured head-to-head.
- MTP-2 head-to-head on identical prompts NOT measured (needs prod MTP server).

================================================================================
## MULTI-MODEL BENCHMARKS (corrected serve scripts, scripts/serve-*.sh)
================================================================================
Date: 2026-06-06. Image: vllm-dflash-thor:fa-native. Single request, T=0, thinking off,
4 coding tasks x 3 runs (median reported). Backend flash_attn, kv auto(BF16), DFlash k=15.
Corrected from the original multi-model plan (settings that would break DFlash):
  fp8 KV -> auto ; flashinfer attn -> flash_attn ; prefix-caching off ;
  spec key 'model' (not draft_model) ; quantization compressed-tensors (not modelopt).

## MoE backend comparison (35B, Dijkstra task)  — Step 3
  marlin             : 69.3 tok/s  tau 5.15   WORKS  <-- fastest, chosen default
  cutlass            : 56.0 tok/s  tau 5.11   works
  flashinfer_cutlass : FAILS  "NvFp4 MoE backend FLASHINFER_CUTLASS does not support current device cuda"
  triton             : FAILS  "moe_backend='triton' is not supported for NvFP4 MoE"
                       (valid NVFP4 MoE backends: cutlass, flashinfer_trtllm, flashinfer_cutlass,
                        flashinfer_cutedsl, marlin, emulation)
  NOTE: the original plan's "marlin vs triton" is impossible (triton can't do NVFP4).
        tau is identical across backends (MoE choice doesn't affect acceptance).
  => DEFAULT MoE backend = marlin (for all NVFP4 models on Thor).

## Per-model decode benchmark (marlin, tok/s e2e | tau)
| Task     | 35B-A3B (MoE, 3B act) | 27B (dense, 27B act) | 122B-A10B (MoE, 10B act) |
|----------|----------------------|----------------------|--------------------------|
| sorting  | 111.2 | tau 8.54     | 31.3 | tau 6.83      | see note                 |
| lru      |  83.9 | tau 6.36     | 35.5 | tau 5.22      | see note                 |
| dijkstra |  94.7 | tau 4.76     | 32.8 | tau 4.83      | see note                 |
| mixed    | 117.5 | tau 5.98     | 37.7 | tau 5.59      | see note                 |
gpu_memory_utilization: 35B=0.78, 27B=0.85, 122B=0.72->see note. max_len 65536 (35B/27B).

Why 27B (smaller) is ~3x SLOWER than 35B: 27B is DENSE (27B active params/token);
35B-A3B is MoE with only 3B active. Decode is memory-bandwidth bound on active params,
so the dense 27B moves ~9x more weight per token. DFlash still helps (tau ~5-7).

## 27B tokenizer gotcha
27B ships tokenizer_config.json with tokenizer_class="TokenizersBackend" which
transformers/vLLM cannot resolve (AutoTokenizer raises). The tokenizer.json is a valid
fast tokenizer (vocab 248044). Fix: serve-27b.sh builds a patched overlay
(~/dflash-setup/tokenizer-fix-27b) with tokenizer_class="Qwen2Tokenizer" and passes
--tokenizer to it. 35B/122B use standard Qwen2Tokenizer, no fix needed.

## docker stop vs docker kill (correction to the plan)
The plan's Step 1 used `docker stop` and Step 5 said to document "docker stop not docker
kill". This is BACKWARDS. On Thor, `docker stop` hangs and leaks page cache. Correct rule:
stop with `docker kill` (instant SIGKILL) or `pkill -9 -f "vllm serve"`, then drop caches.

================================================================================
## 122B-A10B-NVFP4 + DFlash — SOLVED (2026-06-06)  [the "see note" above]
================================================================================
The 122B (256 experts x 48 layers) crashed for weeks. Root cause was NOT DFlash, NOT SWA,
NOT OOM, NOT the draft model. Proven by reproducing the crash with the BASE model alone
(no speculative config): silent exit 255, no traceback, no OOM (Docker OOMKilled=false, no
host kernel OOM line), dying at exactly `nvfp4.py:491 Using MoEPrepareAndFinalizeNoDPEPModular`
i.e. the **Marlin NVFP4 MoE weight-repack** step.

ROOT CAUSE: the Marlin FP4 MoE kernel FAULTS at 256-expert scale on Thor SM110a. cuobjdump
confirms _moe_C.abi3.so DOES contain sm_110a (not a missing-kernel-image problem) and the MoE
dims are 128-aligned (not the #38022 alignment bug) -> a genuine in-kernel fault at scale,
matching open vLLM bugs #35566/#35519/#35922. A rebuild does not fix it.

THE FIX: move the MoE off marlin onto **cutlass**, and the attention off flashinfer onto
**TRITON_ATTN**:
  --moe-backend cutlass        (marlin crashes at 256 experts; cutlass processes them cleanly,
                                70.46 GiB, no crash. NOTE: opposite of the 35B finding above
                                where marlin was fine — marlin only faults at large expert count.)
  --attention-backend TRITON_ATTN  (cutlass + flashinfer dies in attention warmup:
                                "BatchDecodeWithPagedKVCacheWrapper.run() got an unexpected
                                keyword argument 'kv_cache_sf'" — a flashinfer API mismatch.)
  --moe-backend triton -> REJECTED ("not supported for NvFP4 MoE"); never an option.
  gpu-memory-utilization 0.78  (DFlash draft eats ~2 GiB KV; 0.72 fails the 16384 KV request
                                "estimated max length 8960"; 0.90 trips the startup precheck
                                needs 110.5 > ~108 GiB free).
  speculative-config: {"method":"dflash","num_speculative_tokens":12,"model":"/draft",
                       "max_model_len":1024}   (cap draft KV so it doesn't starve target KV)

## 122B decode benchmark (cutlass MoE + TRITON_ATTN, DFlash k=12, gpu_util 0.78, max_len 16384)
Base (no DFlash): 10.9 tok/s (cold, Dijkstra).  GPU KV cache 70,924 tokens, concurrency 4.33x.
| Task     | 122B-A10B +DFlash tok/s | tau  |
|----------|------------------------:|-----:|
| sorting  |                    27.2 | 6.45 |
| lru      |                    42.3 | 5.18 |
| dijkstra |                    34.7 | 4.20 |
| mixed    |                    38.0 | 4.66 |
DFlash uplift on Dijkstra: 34.7 warm vs 10.9 cold base. Acceptance tau 4.2-6.5 (healthy).
Image vllm-dflash-thor:fa-native. Stable, health 200, coherent output, finish_reason ok.

## 122B k-sweep (num_speculative_tokens; cutlass MoE + TRITON_ATTN, gpu_util 0.78, 120W)
| k  | sorting tok/s/tau | lru        | dijkstra   | mixed      | tau_avg |
|----|-------------------|------------|------------|------------|---------|
| 8  | 25.4/5.16 *       | 40.5/4.47  | 34.3/3.71  | 43.1/4.78  | 4.53    |
| 10 | 52.6/6.36         | 45.9/5.36  | 40.0/4.49  | 40.5/4.6   | 5.20 <- OPTIMAL |
| 12 | 53.2/6.45         | 44.2/5.18  | 35.5/4.2   | 39.6/4.66  | 5.12    |
| 15 | 44.2/5.76         | 41.2/5.31  | 31.0/3.86  | 34.2/4.31  | 4.81    |
* k=8 ran first with COLD triton/cutlass autotune caches -> its tok/s is biased low (caches
  warm for k=10/12/15 via mounted ~/.triton). tau is cache-independent and comparable.
OPTIMAL k=10: highest avg tau (5.20) + best on dijkstra/lru; k=12 ~tied; k=15 over-drafts
(acceptance cliff -> wasted target compute -> lower tok/s). serve-122b.sh default set to k=10.

# DFlash on Jetson AGX Thor — Decode-Optimization Hypotheses

Proposed optimizations to push decode tok/s / throughput **without breaking launch**, ranked
by expected payoff × safety. Each has: expected effect, OOM/launch risk, how to test, how to
revert. Baselines are the proven configs in `scripts/serve-*.sh` (see `REVERT.md`).

Context: Jetson AGX Thor, sm_110a, 117 GB **unified** LPDDR5X (CPU+GPU share the pool — an OOM
can starve the OS), vLLM 0.20.0.dev0+dflash, NVFP4 models. Single-stream interactive decode
(1–2 seqs) is how we benchmark, so latency knobs matter more than aggregate-throughput knobs.

---

## TIER 1 — Safe, high payoff (apply + measure)

### H1. MAXN power mode (`nvpmodel -m 0` + `jetson_clocks`)  ★ headline
- **Current state:** Thor is at power mode **ID=1 (120W)**, NOT MAXN (ID=0). All sweeps so far
  ran at 120W.
- **Expected effect:** Largest free lever on Thor. NVIDIA documents up to 3.3–3.5× model-level
  gains from max power+clocks; a comparable MoE went 34→81 tok/s single-stream from the right
  power/kernel setup. Realistic decode uplift here: meaningful double-digit %.
- **OOM/launch risk:** NONE — power/clock only, no memory change.
- **Test:** `sudo nvpmodel -m 0 && sudo jetson_clocks`; re-run the same benchmark at fixed config
  (e.g. 122B k=best) and compare tok/s vs the 120W number.
- **Revert:** `sudo nvpmodel -m 1` (back to 120W). Thermals: MAXN throttles if module power
  exceeds TDP, so it's "max allowed," monitor `tegrastats`.

### H2. `VLLM_MARLIN_USE_ATOMIC_ADD=1` (35B/27B marlin path only)
- **Expected effect:** atomicAdd reduction in the marlin kernels can speed the marlin MoE/dense
  GEMM on the 35B/27B (NOT the 122B, which is cutlass). Small single-digit % plausible.
- **OOM/launch risk:** None (no footprint change). Verify output correctness (atomic ordering).
- **Test:** add `-e VLLM_MARLIN_USE_ATOMIC_ADD=1` to 35B/27B launch; A/B the bench.
- **Revert:** drop the env var.

### H3. Trim `cudagraph_capture_sizes` to real concurrency
- **Expected effect:** No per-replay speedup, but cuts graph-capture memory + startup, freeing
  unified RAM for KV — lets you raise gpu-util or max-model-len elsewhere. At seqs=1–2 with
  k=12, real decode widths are ~[1, 13]; capture e.g. `[1,2,4,8,16]`.
- **OOM/launch risk:** Reduces OOM risk.
- **Test:** `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}'`; confirm health +
  no regression; check freed KV (GPU KV cache size line).
- **Revert:** remove the override (default list returns).

---

## TIER 2 — Likely-safe, may be no-op for DFlash (gate behind smoke test)

### H4. `disable_padded_drafter_batch: true` in speculative-config
- **Expected effect:** removes draft-input padding → less wasted draft compute → higher decode
  tok/s. CAVEAT: docs scope this to EAGLE/MTP drafters; may no-op or error on DFlash
  block-diffusion, and requires the attention backend (TRITON_ATTN/flash_attn) to accept ragged
  spec batches. Free to try.
- **OOM/launch risk:** None (less padding). Risk is launch rejection → just drop it.
- **Test:** add `"disable_padded_drafter_batch":true` to `--speculative-config`; if it launches
  and benches faster, keep; if it errors, revert.
- **Revert:** remove the key.

### H5. `--enable-prefix-caching`
- **Expected effect:** no steady-decode gain, but large TTFT / effective-throughput win on
  repeated/long shared prefixes (agentic, multi-turn). NVIDIA's Thor MoE serve command includes
  it. Keys on block hashes, composes with everything.
- **OOM/launch risk:** Negligible (reuses KV blocks). DFlash caveat: was historically gated;
  smoke-test launch + a 2-turn correctness check on the custom fork.
- **Test:** add `--enable-prefix-caching`; verify health + correct multi-turn output.
- **Revert:** remove the flag.

### H6. `--kv-cache-dtype fp8` as a MEMORY lever (not a short-prompt speed lever)
- **Expected effect:** halves KV memory → lets 122B run larger context / more concurrency. Speed
  win only appears above ~4–7k context; below break-even BF16 KV is faster, so do NOT expect a
  short-prompt tok/s gain. DFlash has rejected fp8 KV on the draft cache before.
- **OOM/launch risk:** REDUCES KV memory ~2×. Launch risk: DFlash may reject it → revert.
- **Test:** only if we want longer context; `--kv-cache-dtype fp8`; confirm DFlash accepts +
  correctness holds.
- **Revert:** `--kv-cache-dtype auto`.

---

## TIER 3 — Investigated, NOT recommended now (documented so we don't re-chase)

- **`--async-scheduling` for interactive:** KEEP OFF for single-stream — it trades TTFT for
  aggregate throughput. Only enable for high-concurrency throughput serving. (122B config already
  omits it; 35B/27B `_common.sh` enables it — consider removing for latency benchmarks.)
- **`parallel_drafting`, `draft_tensor_parallel_size>1`:** EAGLE/multi-GPU only — irrelevant on a
  single Thor / non-EAGLE DFlash. Do not set.
- **`VLLM_USE_FLASHINFER_MOE_FP4=1`, `--moe-backend flashinfer_*`/triton:** dead on sm_110a /
  rejected for NVFP4 (proven). Never use.
- **`VLLM_NVFP4_GEMM_BACKEND`, `VLLM_USE_FLASHINFER_MOE_FP4`:** deprecated → `--linear-backend`/
  `--moe-backend`. Don't tune the env.
- **vLLM-Tune / fused-MoE JSON autotuning:** +9.5% decode on DGX Spark, BUT Triton-MoE + FP8 only
  — does NOT cover our NVFP4 cutlass/marlin paths. Revisit only if we move a model to FP8.
- **Jetson-Thor-specific vLLM container (ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor):** took a
  comparable MoE 34→81 tok/s via Thor-tuned fused-MoE configs. Our DFlash fork can't be swapped
  wholesale, but worth checking whether our build embeds Thor-tuned fused-MoE JSONs. Larger effort.

---

## Recommended apply order (after k-sweeps complete, at fixed best-k)
1. H1 MAXN (re-measure all three models at MAXN vs 120W) →
2. H3 trim cudagraph sizes →
3. H2 marlin atomic-add (35B/27B) →
4. H4 disable_padded_drafter_batch (smoke test) →
5. H5 prefix caching (smoke test) →
Keep async-scheduling OFF for the interactive benchmarks. H6 reserved for long-context needs.

Sources: NVIDIA Jetson Thor blog (7× GenAI), forum thread 364663 (34→81 MoE), forum 368039
(vLLM-Tune), vLLM CUDA-graph + speculative-decoding + FP8-KV docs, Hackster 122B-on-Thor.

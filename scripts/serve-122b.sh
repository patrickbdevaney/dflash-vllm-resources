#!/bin/bash
# serve-122b.sh — Qwen3.5-122B-A10B-NVFP4 (resharded, 72GB) + DFlash draft on Jetson Thor SM110
#
# WHY THIS IS DIFFERENT FROM 35B/27B — two 122B-specific crash modes:
#
#   1. LOAD-TIME OOM: Default safetensors loader mmap's the file (CPU side) then copies
#      to GPU. On Thor unified memory, both exist in the same 117GB pool simultaneously:
#      72GB (GPU) + up to 72GB (CPU mmap file cache) = 144GB > 117GB → system crash.
#      Fix A: --load-format fastsafetensors  (direct disk→GPU, no CPU staging)
#      Fix B: vm.vfs_cache_pressure=1000     (kernel aggressively evicts file pages)
#
#   2. PROFILING-TIME OOM: vLLM does a forward pass with (MAX_SEQS × MAX_BATCHED) tokens
#      to measure peak GPU memory. At the _common.sh default of 8192 × 2 = 16384 tokens,
#      activation memory for 122B MoE peaks at ~15-20GB on top of 72GB weights → OOM.
#      Fix: MAX_BATCHED=4096, MAX_SEQS=1 → profiling peak drops to ~5GB.
#
# MEMORY MATH (stable config):
#   weights:     72GB (NVFP4 quantized, direct GPU)
#   draft head:   1GB (DFlash drafter, BF16)
#   CUDA context: 2GB
#   KV cache:    tiny — only 12/48 layers are full-attention, 2 KV heads, head_dim=256
#                → ~24KB/token; 16K ctx × 24KB = 393MB. Not the constraint.
#   Total at 0.72 util: ~84GB budget → ~9GB overhead headroom. Comfortable.
#
# CUDA GRAPHS: enabled by default. Graph capture runs within the profiled memory budget
#   (same or smaller batch sizes than the profiling pass — no extra memory needed).
#   If graphs OOM after weights load successfully, use: ENFORCE_EAGER=1 scripts/serve-122b.sh
#
# PREREQUISITES:
#   Build the fastsafe image (one-time, ~2-3 min):
#     docker build -f ~/dflash-setup/Dockerfile.fastsafe -t vllm-dflash-thor:fastsafe ~/dflash-setup/
#   If build unavailable, fall back (higher OOM risk):
#     IMAGE=vllm-dflash-thor:fa-native LOAD_FORMAT='' scripts/serve-122b.sh
#
# ESCALATING FALLBACK (if this config still OOMs):
#   GPU_UTIL=0.68 scripts/serve-122b.sh          — less KV cache, more headroom
#   GPU_UTIL=0.65 MAX_LEN=8192 scripts/serve-122b.sh  — minimal footprint
#   DRAFT_DIR='' scripts/serve-122b.sh            — base model only (no DFlash, diagnose)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
export MODEL_DIR="$HOME/Qwen3.5-122B-A10B-NVFP4/resharded"
export DRAFT_DIR="$HOME/models/Qwen3.5-122B-A10B-DFlash"
export MODEL_NAME="Qwen3.5-122B-A10B-NVFP4"

# ── 122B-specific tuning ──────────────────────────────────────────────────────
# WORKING BACKENDS (proven 2026-06-06 on Thor SM110a):
#   MoE backend MUST be cutlass — marlin FAULTS at 256-expert scale (silent exit 255 at the
#   Marlin FP4 MoE weight-repack). cutlass (VLLM_CUTLASS) processes 256 experts cleanly.
#   Attention MUST be TRITON_ATTN — flashinfer paged-decode hits a kv_cache_sf API mismatch
#   (TypeError in BatchDecodeWithPagedKVCacheWrapper.run) in this dflash fork.
#   triton MoE is REJECTED for NVFP4 ("not supported"); do not use it.
export GPU_UTIL="${GPU_UTIL:-0.78}"          # 0.78 needed: DFlash draft eats ~2GB KV; 0.90 trips precheck
export MAX_LEN="${MAX_LEN:-16384}"           # 70,924-token KV at 0.78; 4.33x concurrency
export MAX_SEQS="${MAX_SEQS:-2}"
export MAX_BATCHED="${MAX_BATCHED:-4096}"
export MOE_BACKEND="${MOE_BACKEND:-cutlass}"        # NOT marlin (crashes at 256 experts)
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"  # NOT flashinfer (kv_cache_sf bug)
export NUM_SPEC="${NUM_SPEC:-10}"            # k-sweep optimal: k=10 (tau 5.2); k=12 ~tied; k=15 regresses
export DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-1024}"       # cap draft KV so it doesn't starve target KV

# Load format: fastsafetensors bypasses CPU staging buffer (core OOM fix for 122B)
export LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"

# Default CUDA-graph capture works with cutlass+TRITON_ATTN (the [1,13] Marlin/GDA workaround
# is no longer needed once off the marlin MoE path). Override COMPILATION_CONFIG only if needed.
export COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"

# CUDA graphs: ENABLED by default (ENFORCE_EAGER=0).
# The real OOM causes (load-time double-copy + profiling activation peak) are fixed by the
# flags above; graph capture runs within the profiled memory budget. Disabling graphs
# costs ~20-40% decode throughput. Use ENFORCE_EAGER=1 only if graphs crash after weights load.
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

export HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"   # 20 min — 72GB load takes 10-15 min
export IMAGE="${IMAGE:-vllm-dflash-thor:fastsafe}"

# ── Validate image ────────────────────────────────────────────────────────────
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image '$IMAGE' not found."
  if [ "$IMAGE" = "vllm-dflash-thor:fastsafe" ]; then
    echo ""
    echo "  Build it (one-time, ~2-3 min):"
    echo "    docker build -f ~/dflash-setup/Dockerfile.fastsafe -t vllm-dflash-thor:fastsafe ~/dflash-setup/"
    echo ""
    echo "  Or fall back (higher OOM risk — no fastsafetensors):"
    echo "    IMAGE=vllm-dflash-thor:fa-native LOAD_FORMAT='' scripts/serve-122b.sh"
  fi
  exit 1
fi

# Confirm fastsafetensors is importable in the selected image (catches pip install failures)
if [ -n "$LOAD_FORMAT" ] && [ "$LOAD_FORMAT" = "fastsafetensors" ]; then
  if ! docker run --rm "$IMAGE" python -c "import fastsafetensors" 2>/dev/null; then
    echo "WARNING: fastsafetensors not importable in $IMAGE — falling back to default loader."
    echo "         The 122B is at higher OOM risk without it. Kernel tuning below will help."
    export LOAD_FORMAT=""
  fi
fi

# ── Validate model paths ──────────────────────────────────────────────────────
[ -d "$MODEL_DIR" ]          || { echo "ERROR: MODEL_DIR not found: $MODEL_DIR"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: config.json missing in $MODEL_DIR"; exit 1; }
[ -f "$DRAFT_DIR/config.json" ] || { echo "ERROR: DRAFT_DIR not found: $DRAFT_DIR"; exit 1; }

echo "================================================================"
echo " 122B pre-flight check"
echo " Base  : $MODEL_DIR  ($(du -sh $MODEL_DIR 2>/dev/null | cut -f1))"
echo " Draft : $DRAFT_DIR  ($(du -sh $DRAFT_DIR 2>/dev/null | cut -f1))"
echo "================================================================"

# ── Kernel memory tuning ──────────────────────────────────────────────────────
# These prevent file page cache from accumulating during the 72GB weight load,
# which would otherwise compete with GPU allocations in unified memory.
# Restored to defaults after launch via the EXIT trap below.
_RESTORE_SYSCTL() {
  sudo sysctl -w vm.vfs_cache_pressure=100  >/dev/null 2>&1 || true
  sudo sysctl -w vm.watermark_scale_factor=10 >/dev/null 2>&1 || true
  sudo sysctl -w vm.min_free_kbytes=65536   >/dev/null 2>&1 || true
}
trap _RESTORE_SYSCTL EXIT INT TERM

sudo sysctl -w vm.vfs_cache_pressure=1000   >/dev/null 2>&1 && echo "vm.vfs_cache_pressure=1000 (aggressive file cache eviction during load)" || true
sudo sysctl -w vm.watermark_scale_factor=200 >/dev/null 2>&1 && echo "vm.watermark_scale_factor=200 (earlier reclaim)" || true
sudo sysctl -w vm.min_free_kbytes=2097152   >/dev/null 2>&1 && echo "vm.min_free_kbytes=2GB (kernel headroom)" || true

# ── Drop page cache (mandatory before 122B load) ──────────────────────────────
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' && echo "page cache dropped" || true
sleep 1

echo "Memory state:"
free -h | grep -E 'Mem:|Swap:'
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
if [ "${AVAIL_GB}" -lt 100 ]; then
  echo "WARNING: only ${AVAIL_GB}GB available — expected >100GB for 122B."
  echo "  Check: docker ps, nvidia-smi, kill any process holding GPU/RAM."
fi

# ── Ensure cache directories exist ───────────────────────────────────────────
mkdir -p "$HOME/thor-vllm-cache/vllm-dflash" "$HOME/thor-vllm-cache/flashinfer"

echo ""
echo ">>> Launching 122B (cold load ~10-15 min) — $(date)"
echo "    Monitor: docker logs -f \$(cat ~/thor-vllm-cache/current-dflash-container)"
echo "    Memory:  watch -n5 'free -h | grep Mem'"
echo ""

source "$(dirname "$0")/_common.sh"

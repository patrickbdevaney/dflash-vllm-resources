#!/bin/bash
# serve-dflash.sh  —  CORRECTED (global flash_attn backend per official drafter README)
# DFlash block-diffusion speculative decoding on Jetson Thor SM110.
# Target: Qwen3.6-35B-A3B-NVFP4  |  Drafter: Qwen3.6-35B-A3B-DFlash  | block 16 (k=15)
#
# CORRECTION vs brief: the drafter's official README launches with ONE global
#   `--attention-backend flash_attn` (the drafter's non-causal attn requires it).
#   The brief's "target=FLASHINFER + drafter=flash_attn" split is not how the
#   official path works, so we set flash_attn globally here.
#
# USAGE: ./serve-dflash.sh [eager]
#   (no args) = PIECEWISE CUDA graphs, production mode
#   eager     = --enforce-eager; run this FIRST to prove correctness
set -euo pipefail
MODE="${1:-normal}"

TARGET_MODEL="$HOME/Qwen3.6-35B-A3B-NVFP4"
DRAFTER_MODEL="$HOME/Qwen3.6-35B-A3B-DFlash"
CONFIG_FILE="$HOME/dflash-setup/qwen36-dflash-config.yaml"
CACHE_ROOT="$HOME/thor-vllm-cache"
VLLM_CACHE="$CACHE_ROOT/vllm-dflash"
FLASHINFER_CACHE="$CACHE_ROOT/flashinfer"
TORCH_CACHE="$CACHE_ROOT/torch-dflash"
PORT=8001

# Default to fa-native (our build + stock NATIVE sm_110 FA2 .so). Strictly more correct
# than :latest (which had sm_80-only FA2); identical decode perf, no FA3 crash (vLLM uses FA2).
IMAGE_DFLASH="${DFLASH_IMAGE:-vllm-dflash-thor:fa-native}"
IMAGE_STOCK="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor"
if docker image inspect "$IMAGE_DFLASH" &>/dev/null; then
    IMAGE="$IMAGE_DFLASH"; echo "Using DFlash-built image: $IMAGE_DFLASH"
else
    echo "ERROR: $IMAGE_DFLASH not found. The stock 0.19.0 image has NO DFlash —"
    echo "       build it first (Dockerfile.dflash-thor). Refusing to start a"
    echo "       server that would fall back to non-DFlash decoding."
    exit 1
fi

[ ! -d "$TARGET_MODEL" ]  && { echo "ERROR: target not found: $TARGET_MODEL"; exit 1; }
[ ! -d "$DRAFTER_MODEL" ] && { echo "ERROR: drafter not found: $DRAFTER_MODEL"; exit 1; }
mkdir -p "$VLLM_CACHE" "$FLASHINFER_CACHE" "$TORCH_CACHE"

# ── Thor perf + memory mitigations ────────────────────────────────────────────
sudo nvpmodel -m 1 2>/dev/null && echo "nvpmodel: 120W sustained" || echo "WARN: nvpmodel failed"
sudo jetson_clocks 2>/dev/null && echo "jetson_clocks: locked" || echo "WARN: jetson_clocks failed"
sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 && echo "page cache: dropped" || true
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
echo "host MemAvailable: ${AVAIL_GB} GiB"
[ "${AVAIL_GB:-0}" -lt 40 ] && echo "WARNING: only ${AVAIL_GB} GiB free — consider reboot"

# ── Kill any prior vllm (never docker stop on Thor — page-cache leak) ──────────
if pgrep -f "vllm serve" >/dev/null 2>&1; then
    sudo pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 2
    sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true
fi
sudo fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 1

EXTRA_ARGS=""
if [ "$MODE" = "eager" ]; then
    EXTRA_ARGS="--enforce-eager"
    echo ""; echo "=== EAGER MODE: skipping CUDA graph capture — proves correctness first ==="
fi

echo ""; echo "=== Config ==="; cat "$CONFIG_FILE"; echo "=============="
echo "=== Starting DFlash server (global flash_attn) ==="
echo "  Target: Qwen3.6-35B-A3B-NVFP4 | Drafter: Qwen3.6-35B-A3B-DFlash | k=16 (15 spec tokens)"
echo "  When READY: ~/dflash-setup/verify-dflash.sh  then  ~/dflash-setup/bench-decode.sh"

CONTAINER_NAME="vllm-dflash-$(date +%s)"
echo "$CONTAINER_NAME" > "$CACHE_ROOT/current-dflash-container"

# Use a TTY only when attached to one (so this script also works headless/background).
TTY_FLAG=""; [ -t 0 ] && TTY_FLAG="-it"

docker run --rm $TTY_FLAG \
    --name "$CONTAINER_NAME" \
    --runtime nvidia --gpus all \
    --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -e VLLM_FLASHINFER_MOE_BACKEND=latency \
    -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
    -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
    -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=524288000 \
    -e VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=163840 \
    -e VLLM_CPU_OMP_THREADS_BIND=auto \
    -e USE_FASTSAFETENSOR=true \
    -e VLLM_ENABLE_CUDAGRAPH_GC=0 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_XET=1 \
    -e VLLM_TUNED_CONFIG_FOLDER=/moe-configs \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    -v "${TARGET_MODEL}:/model:ro" \
    -v "${DRAFTER_MODEL}:/drafter:ro" \
    -v "${VLLM_CACHE}:/root/.cache/vllm" \
    -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
    -v "${TORCH_CACHE}:/root/.cache/torch" \
    -v "$HOME/moe-configs:/moe-configs:ro" \
    -v "${CONFIG_FILE}:/root/dflash-config.yaml:ro" \
    "$IMAGE" \
    vllm serve /model \
        --config /root/dflash-config.yaml \
        --attention-backend flash_attn \
        --language-model-only \
        $EXTRA_ARGS

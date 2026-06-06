#!/bin/bash
# _common.sh — shared DFlash launch core for Jetson Thor SM110 (sourced by serve-*.sh)
#
# CORRECTED vs the original multi-model plan (settings that would break DFlash):
#   - kv-cache-dtype: auto (BF16)        NOT fp8   (DFlash rejects quantized KV, issue #41559)
#   - attention-backend: flash_attn      NOT flashinfer (drafter is non-causal)
#   - NO --enable-prefix-caching                  (DFlash + hybrid-attn bug #40624)
#   - speculative-config key: "model"    NOT "draft_model" (real SpeculativeConfig field)
#   - quantization: compressed-tensors   NOT modelopt (models are nvfp4-pack-quantized)
#   - stop with `docker kill` / pkill     NEVER `docker stop` (Thor page-cache leak)
#
# Required env (set by each wrapper): MODEL_DIR DRAFT_DIR MODEL_NAME GPU_UTIL MAX_LEN MAX_SEQS
# Optional env: PORT(8001) MAX_BATCHED(8192) MOE_BACKEND("") VLLM_USE_FLASHINFER_MOE_FP4(0)
#               NUM_SPEC(15) IMAGE(vllm-dflash-thor:fa-native)
set -euo pipefail

PORT="${PORT:-8001}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
NUM_SPEC="${NUM_SPEC:-15}"
IMAGE="${IMAGE:-vllm-dflash-thor:fa-native}"
FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
MOE_BACKEND="${MOE_BACKEND:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"          # e.g. "fastsafetensors" for 122B unified-memory safety
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"     # set to 1 to skip CUDA graph capture (first 122B launch)
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash_attn}"  # 122B needs TRITON_ATTN (flashinfer kv_cache_sf bug)
DRAFT_MAX_LEN="${DRAFT_MAX_LEN:-}"      # cap draft model KV (122B: 1024) so it doesn't starve target KV
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"   # seconds to wait for /health

[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: base model config not found: $MODEL_DIR"; exit 1; }
[ -f "$DRAFT_DIR/config.json" ] || { echo "ERROR: draft model config not found: $DRAFT_DIR"; exit 1; }

echo "================================================================"
echo " Model : $MODEL_NAME"
echo " Base  : $MODEL_DIR"
echo " Draft : $DRAFT_DIR   (DFlash, k=$((NUM_SPEC+1)) -> num_speculative_tokens=$NUM_SPEC)"
echo " Port  : $PORT   gpu_util=$GPU_UTIL  max_len=$MAX_LEN  max_seqs=$MAX_SEQS"
echo " MoE   : backend='${MOE_BACKEND:-auto}'  VLLM_USE_FLASHINFER_MOE_FP4=$FP4"
echo " Load  : format='${LOAD_FORMAT:-auto}'  enforce_eager=${ENFORCE_EAGER}"
echo " Image : $IMAGE"
echo "================================================================"

# ── Stop any prior server (docker kill, NEVER docker stop) + clean stale ────────
# NOTE: containers are launched WITHOUT --rm so logs survive a crash for debugging.
for c in $(docker ps -q --filter "publish=${PORT}" 2>/dev/null); do
  echo "killing prior container $c on :$PORT"; docker kill "$c" >/dev/null 2>&1 || true
done
for c in $(docker ps -aq --filter "name=vllm-dflash" 2>/dev/null); do
  docker rm -f "$c" >/dev/null 2>&1 || true
done
sleep 2
sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 && echo "page cache dropped" || true

# ── Thor perf governor ─────────────────────────────────────────────────────────
sudo nvpmodel -m 1 >/dev/null 2>&1 && echo "nvpmodel: max" || true
sudo jetson_clocks >/dev/null 2>&1 && echo "jetson_clocks: locked" || true

CONTAINER="vllm-dflash-$(date +%s)"
MOE_ARG="";         [ -n "$MOE_BACKEND" ]       && MOE_ARG="--moe-backend $MOE_BACKEND"
LOAD_FORMAT_ARG=""; [ -n "$LOAD_FORMAT" ]        && LOAD_FORMAT_ARG="--load-format $LOAD_FORMAT"
EAGER_ARG="";       [ "${ENFORCE_EAGER}" = "1" ] && EAGER_ARG="--enforce-eager"
# COMPILATION_CONFIG: JSON string, e.g. '{"cudagraph_capture_sizes":[1,13]}' for GDA/Mamba models
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"
COMPILATION_CONFIG_ARGS=()
[ -n "$COMPILATION_CONFIG" ] && COMPILATION_CONFIG_ARGS=(--compilation-config "$COMPILATION_CONFIG")
# Speculative config: optionally cap the draft model's max_model_len (keeps draft KV tiny)
if [ -n "$DRAFT_MAX_LEN" ]; then
  SPEC_CONFIG="{\"method\":\"dflash\",\"num_speculative_tokens\":${NUM_SPEC},\"model\":\"/drafter\",\"max_model_len\":${DRAFT_MAX_LEN}}"
else
  SPEC_CONFIG="{\"method\":\"dflash\",\"num_speculative_tokens\":${NUM_SPEC},\"model\":\"/drafter\"}"
fi
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: pass through if set
MEM_PROFILER_ARG="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-}"

# Optional tokenizer override (e.g. 27B ships tokenizer_class=TokenizersBackend which
# AutoTokenizer cannot resolve; a patched overlay is mounted at /tokenizer).
TOK_MOUNT=(); TOK_ARG=()
if [ -n "${TOKENIZER_DIR:-}" ]; then
  TOK_MOUNT=(-v "${TOKENIZER_DIR}:/tokenizer:ro")
  TOK_ARG=(--tokenizer /tokenizer)
  echo " Tokenizer override: $TOKENIZER_DIR"
fi

set -x
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4="$FP4" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FASTSAFETENSORS_NOGDS=1 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  ${VLLM_MARLIN_USE_ATOMIC_ADD:+-e VLLM_MARLIN_USE_ATOMIC_ADD="$VLLM_MARLIN_USE_ATOMIC_ADD"} \
  ${VLLM_USE_FLASHINFER_SAMPLER:+-e VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER"} \
  ${CUDA_DEVICE_MAX_CONNECTIONS:+-e CUDA_DEVICE_MAX_CONNECTIONS="$CUDA_DEVICE_MAX_CONNECTIONS"} \
  ${MEM_PROFILER_ARG:+-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="$MEM_PROFILER_ARG"} \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DRAFT_DIR}:/drafter:ro" \
  -v "$HOME/thor-vllm-cache/vllm-dflash:/root/.cache/vllm" \
  -v "$HOME/thor-vllm-cache/flashinfer:/root/.cache/flashinfer" \
  "${TOK_MOUNT[@]}" \
  "$IMAGE" \
  vllm serve /model \
    --served-model-name "$MODEL_NAME" /model \
    "${TOK_ARG[@]}" \
    --speculative-config "$SPEC_CONFIG" \
    --quantization compressed-tensors \
    --kv-cache-dtype auto \
    --attention-backend "$ATTENTION_BACKEND" \
    $MOE_ARG \
    $LOAD_FORMAT_ARG \
    $EAGER_ARG \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --enable-chunked-prefill \
    --async-scheduling \
    "${COMPILATION_CONFIG_ARGS[@]}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --language-model-only \
    --port "$PORT"
set +x
echo "$CONTAINER" > "$HOME/thor-vllm-cache/current-dflash-container"

# ── Health check loop ──────────────────────────────────────────────────────────
echo "waiting for http://localhost:${PORT}/health (timeout ${HEALTH_TIMEOUT}s)..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
while true; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo ""
    echo ">>> READY: $MODEL_NAME  on  http://localhost:${PORT}  (container $CONTAINER)"
    exit 0
  fi
  if ! docker ps -q --filter "name=$CONTAINER" | grep -q .; then
    echo ""; echo "!!! container exited before becoming healthy. Last 40 log lines:"
    docker logs "$CONTAINER" 2>&1 | tail -40 || true
    exit 1
  fi
  if [ $SECONDS -ge $deadline ]; then
    echo ""; echo "!!! TIMEOUT after ${HEALTH_TIMEOUT}s. Last 30 log lines:"
    docker logs "$CONTAINER" 2>&1 | tail -30 || true
    exit 1
  fi
  sleep 4
done

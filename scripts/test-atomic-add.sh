#!/bin/bash
# test-atomic-add.sh — A/B test VLLM_MARLIN_USE_ATOMIC_ADD=1 vs baseline on 35B (marlin MoE)
# and 27B (dense). Compares to the k-sweep baselines. Writes a small report.
set -uo pipefail
OUT=~/dflash-setup/moe-configs; REPORT="$OUT/atomic-add-ab.txt"
: > "$REPORT"
echo "VLLM_MARLIN_USE_ATOMIC_ADD A/B — $(date)" | tee -a "$REPORT"

bench() { python3 ~/dflash-setup/scripts/bench.py --url "http://localhost:$1" --tasks all --runs 3 --label "$2" 2>&1 | grep -E "task|SUMMARY|^[a-z]"; }
kill_all() { for c in $(docker ps -q); do docker kill "$c" >/dev/null 2>&1; done; sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 2; }
wait_health() { for i in $(seq 1 96); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$1/health 2>/dev/null)" = "200" ] && return 0; docker ps -q --filter "name=$2" | grep -q . || return 1; sleep 5; done; return 1; }

# ---- 35B + atomic-add (k=12 marlin) ----
echo "" | tee -a "$REPORT"; echo "=== 35B marlin k=12 WITH VLLM_MARLIN_USE_ATOMIC_ADD=1 ===" | tee -a "$REPORT"
echo "(baseline no-flag from sweep: sorting 134.8/8.1 | lru 103.8/6.15 | dijkstra 113.9/5.13 | mixed 117.5/5.41)" | tee -a "$REPORT"
kill_all
docker run -d --name vllm-aa-35b --runtime=nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_DISABLE_XET=1 -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -v ~/Qwen3.6-35B-A3B-NVFP4:/model -v ~/Qwen3.6-35B-A3B-DFlash:/draft \
  -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
  vllm-dflash-thor:fa-native \
  vllm serve /model --speculative-config '{"method":"dflash","num_speculative_tokens":12,"model":"/draft"}' \
    --quantization compressed-tensors --load-format fastsafetensors --moe-backend marlin --attention-backend flash_attn \
    --gpu-memory-utilization 0.78 --max-model-len 65536 --max-num-seqs 4 --trust-remote-code --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --port 8007 >/dev/null 2>&1
if wait_health 8007 vllm-aa-35b; then bench 8007 "35b-atomicadd" | tee -a "$REPORT"; else echo "35B atomic-add FAILED TO START" | tee -a "$REPORT"; docker logs vllm-aa-35b 2>&1|tail -8|tee -a "$REPORT"; fi
docker kill vllm-aa-35b >/dev/null 2>&1; docker rm -f vllm-aa-35b >/dev/null 2>&1

# ---- 27B + atomic-add (k=15) ----
echo "" | tee -a "$REPORT"; echo "=== 27B dense k=15 WITH VLLM_MARLIN_USE_ATOMIC_ADD=1 ===" | tee -a "$REPORT"
echo "(baseline no-flag from sweep: sorting 50.1/7.04 | lru 38.4/5.33 | dijkstra 39.8/5.55 | mixed 40.9/5.68)" | tee -a "$REPORT"
echo "(note: 27B dense uses FlashInferCutlass NVFP4 GEMM, not marlin -> flag expected to be a no-op)" | tee -a "$REPORT"
kill_all
docker run -d --name vllm-aa-27b --runtime=nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_DISABLE_XET=1 -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -v ~/models/Qwen3.6-27B-NVFP4:/model -v ~/models/Qwen3.6-27B-DFlash:/draft -v ~/dflash-setup/tokenizer-fix-27b:/tokenizer:ro \
  -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
  vllm-dflash-thor:fa-native \
  vllm serve /model --tokenizer /tokenizer --speculative-config '{"method":"dflash","num_speculative_tokens":15,"model":"/draft"}' \
    --quantization compressed-tensors --load-format fastsafetensors --attention-backend flash_attn \
    --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-seqs 4 --trust-remote-code --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --port 8008 >/dev/null 2>&1
if wait_health 8008 vllm-aa-27b; then bench 8008 "27b-atomicadd" | tee -a "$REPORT"; else echo "27B atomic-add FAILED TO START" | tee -a "$REPORT"; docker logs vllm-aa-27b 2>&1|tail -8|tee -a "$REPORT"; fi
docker kill vllm-aa-27b >/dev/null 2>&1; docker rm -f vllm-aa-27b >/dev/null 2>&1

echo "" | tee -a "$REPORT"; echo "=== atomic-add A/B DONE $(date) ===" | tee -a "$REPORT"

#!/bin/bash
# ksweep-4b.sh — DFlash k-sweep for Qwen3.5-4B-NVFP4 (dense GDA-hybrid VLM) + 4B-DFlash draft.
# block_size=16 so k<=15. compressed-tensors quant, flash_attn, gpu-util 0.40, --language-model-only.
set -uo pipefail
OUT=~/dflash-setup/moe-configs; mkdir -p "$OUT"
TABLE="$OUT/ksweep-4b-table.txt"; LOG="$OUT/ksweep-4b.log"
PORT=8001
K_VALUES=(${K_VALUES:-8 10 12 15})
MODEL=~/models/Qwen3.5-4B-NVFP4
DRAFT=~/models/Qwen3.5-4B-DFlash

: > "$LOG"
printf "%-4s | %-12s | %-12s | %-12s | %-12s | %-6s\n" k sorting lru dijkstra mixed tau_avg > "$TABLE"
printf -- "-----+--------------+--------------+--------------+--------------+-------\n" >> "$TABLE"

for k in "${K_VALUES[@]}"; do
  kp1=$((k+1))
  echo "==================== k=$k (capture [1,$kp1]) $(date) ====================" | tee -a "$LOG"
  docker rm -f vllm-4b-ksweep >/dev/null 2>&1
  for c in $(docker ps -q); do docker kill "$c" >/dev/null 2>&1; done
  sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 2

  docker run -d --name vllm-4b-ksweep --runtime=nvidia --gpus all --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 -e HF_HUB_DISABLE_XET=1 \
    -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$MODEL":/model -v "$DRAFT":/draft \
    -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
    vllm-dflash-thor:fa-native \
    vllm serve /model \
      --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":$k,\"model\":\"/draft\"}" \
      --quantization compressed-tensors --load-format fastsafetensors --attention-backend flash_attn \
      --gpu-memory-utilization 0.40 --max-model-len 32768 --max-num-seqs 1 \
      --compilation-config "{\"cudagraph_capture_sizes\":[1,$kp1]}" \
      --trust-remote-code --language-model-only \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --port $PORT >/dev/null 2>&1
  ok=0
  for i in $(seq 1 72); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    docker ps -q --filter name=vllm-4b-ksweep | grep -q . || { echo "k=$k DIED" | tee -a "$LOG"; docker logs vllm-4b-ksweep 2>&1|tail -8|tee -a "$LOG"; break; }
    sleep 5
  done
  if [ "$ok" != "1" ]; then printf "%-4s | %s\n" "$k" "FAILED" >> "$TABLE"; continue; fi
  B=$(python3 ~/dflash-setup/scripts/bench.py --url http://localhost:$PORT --tasks all --runs 3 --label "4b-k$k" 2>&1)
  echo "$B" | tee -a "$LOG"
  row=$(echo "$B" | grep SUMMARY_JSON | sed 's/SUMMARY_JSON //' | python3 -c "
import json,sys
d=json.load(sys.stdin)
g=lambda k:(d.get(k,{}).get('tok_s','?'), d.get(k,{}).get('tau','?'))
taus=[v['tau'] for v in d.values() if isinstance(v,dict) and isinstance(v.get('tau'),(int,float))]
ta=round(sum(taus)/len(taus),2) if taus else '?'
print(f\"{g('sorting')[0]}/{g('sorting')[1]} | {g('lru')[0]}/{g('lru')[1]} | {g('dijkstra')[0]}/{g('dijkstra')[1]} | {g('mixed')[0]}/{g('mixed')[1]} | {ta}\")
" 2>/dev/null || echo "parse-fail")
  printf "%-4s | %s\n" "$k" "$row" >> "$TABLE"
  docker rm -f vllm-4b-ksweep >/dev/null 2>&1
done
echo "=== 4B sweep DONE $(date) ===" | tee -a "$LOG"
cat "$TABLE" | tee -a "$LOG"

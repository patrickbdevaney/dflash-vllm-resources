#!/bin/bash
# ksweep-35b.sh — DFlash k-sweep for Qwen3.6-35B-A3B-NVFP4 (marlin MoE + flash_attn).
# block_size=16 so k<=15. Writes table + log. Also does a quick MoE-backend profile at k=12.
set -uo pipefail
OUT=~/dflash-setup/moe-configs; mkdir -p "$OUT"
TABLE="$OUT/ksweep-35b-table.txt"; LOG="$OUT/ksweep-35b.log"
PORT=8007
K_VALUES=(${K_VALUES:-8 10 12 15})
MODEL=~/Qwen3.6-35B-A3B-NVFP4
DRAFT=~/Qwen3.6-35B-A3B-DFlash

: > "$LOG"
printf "%-4s | %-12s | %-12s | %-12s | %-12s | %-6s\n" k sorting lru dijkstra mixed tau_avg > "$TABLE"
printf -- "-----+--------------+--------------+--------------+--------------+-------\n" >> "$TABLE"

launch() {  # $1=k  $2=moe_backend
  docker run -d --name vllm-35b-ksweep --runtime=nvidia --gpus all --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    -e HF_HUB_DISABLE_XET=1 -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -v "$MODEL":/model -v "$DRAFT":/draft \
    -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
    vllm-dflash-thor:fa-native \
    vllm serve /model \
      --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":$1,\"model\":\"/draft\"}" \
      --quantization compressed-tensors --load-format fastsafetensors \
      --moe-backend "$2" --attention-backend flash_attn \
      --gpu-memory-utilization 0.78 --max-model-len 65536 --max-num-seqs 4 \
      --trust-remote-code --language-model-only \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --port $PORT >/dev/null 2>&1
}

run_one() {  # $1=k $2=moe  -> echoes "sorting/tau | lru/tau | dij/tau | mixed/tau | tau_avg" or FAIL
  docker rm -f vllm-35b-ksweep >/dev/null 2>&1
  for c in $(docker ps -q); do docker kill "$c" >/dev/null 2>&1; done
  sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 2
  launch "$1" "$2"
  local ok=0
  for i in $(seq 1 96); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    docker ps -q --filter "name=vllm-35b-ksweep" | grep -q . || break
    sleep 5
  done
  if [ "$ok" != "1" ]; then docker logs vllm-35b-ksweep 2>&1 | tail -6 >> "$LOG"; echo "FAIL"; return; fi
  local B=$(python3 ~/dflash-setup/scripts/bench.py --url http://localhost:$PORT --tasks all --runs 3 --label "35b-k$1-$2" 2>&1)
  echo "$B" >> "$LOG"
  echo "$B" | grep SUMMARY_JSON | sed 's/SUMMARY_JSON //' | python3 -c "
import json,sys
d=json.load(sys.stdin)
g=lambda k:(d.get(k,{}).get('tok_s','?'), d.get(k,{}).get('tau','?'))
taus=[v['tau'] for v in d.values() if isinstance(v,dict) and isinstance(v.get('tau'),(int,float))]
ta=round(sum(taus)/len(taus),2) if taus else '?'
print(f\"{g('sorting')[0]}/{g('sorting')[1]} | {g('lru')[0]}/{g('lru')[1]} | {g('dijkstra')[0]}/{g('dijkstra')[1]} | {g('mixed')[0]}/{g('mixed')[1]} | {ta}\")
" 2>/dev/null || echo "parse-fail"
}

# --- k-sweep (marlin) ---
for k in "${K_VALUES[@]}"; do
  echo "==================== k=$k marlin $(date) ====================" | tee -a "$LOG"
  row=$(run_one "$k" marlin)
  printf "%-4s | %s\n" "$k" "$row" >> "$TABLE"
done

# --- MoE profile at k=12: marlin vs cutlass vs emulation ---
echo "" >> "$TABLE"; echo "# MoE profile @k=12 (tok/s avg across 4 tasks)" >> "$TABLE"
for moe in marlin cutlass emulation; do
  echo "==================== MoE=$moe k=12 $(date) ====================" | tee -a "$LOG"
  row=$(run_one 12 "$moe")
  printf "moe=%-9s | %s\n" "$moe" "$row" >> "$TABLE"
done

docker rm -f vllm-35b-ksweep >/dev/null 2>&1
echo "=== 35B sweep DONE $(date) ===" | tee -a "$LOG"
cat "$TABLE" | tee -a "$LOG"

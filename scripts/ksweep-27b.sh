#!/bin/bash
# ksweep-27b.sh — DFlash k-sweep for Qwen3.6-27B-NVFP4 (dense, marlin MoE n/a -> dense, flash_attn).
# 27B ships tokenizer_class=TokenizersBackend -> build a patched Qwen2Tokenizer overlay.
# block_size=16 so k<=15.
set -uo pipefail
OUT=~/dflash-setup/moe-configs; mkdir -p "$OUT"
TABLE="$OUT/ksweep-27b-table.txt"; LOG="$OUT/ksweep-27b.log"
PORT=8008
K_VALUES=(${K_VALUES:-8 10 12 15})
MODEL=~/models/Qwen3.6-27B-NVFP4
DRAFT=~/models/Qwen3.6-27B-DFlash
TOK_FIX=~/dflash-setup/tokenizer-fix-27b

# build tokenizer overlay
mkdir -p "$TOK_FIX"
for f in tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json added_tokens.json vocab.json merges.txt generation_config.json; do
  [ -f "$MODEL/$f" ] && cp -f "$MODEL/$f" "$TOK_FIX/$f"
done
python3 - "$TOK_FIX/tokenizer_config.json" <<'PY'
import json, sys
p=sys.argv[1]; c=json.load(open(p))
if c.get("tokenizer_class") in (None,"TokenizersBackend"):
    c["tokenizer_class"]="Qwen2Tokenizer"; json.dump(c,open(p,"w"),ensure_ascii=False,indent=2)
PY

: > "$LOG"
printf "%-4s | %-12s | %-12s | %-12s | %-12s | %-6s\n" k sorting lru dijkstra mixed tau_avg > "$TABLE"
printf -- "-----+--------------+--------------+--------------+--------------+-------\n" >> "$TABLE"

launch() {  # $1=k
  docker run -d --name vllm-27b-ksweep --runtime=nvidia --gpus all --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
    -e HF_HUB_DISABLE_XET=1 -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -v "$MODEL":/model -v "$DRAFT":/draft -v "$TOK_FIX":/tokenizer:ro \
    -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
    vllm-dflash-thor:fa-native \
    vllm serve /model --tokenizer /tokenizer \
      --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":$1,\"model\":\"/draft\"}" \
      --quantization compressed-tensors --load-format fastsafetensors \
      --attention-backend flash_attn \
      --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-seqs 4 \
      --trust-remote-code --language-model-only \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --port $PORT >/dev/null 2>&1
}

for k in "${K_VALUES[@]}"; do
  echo "==================== k=$k $(date) ====================" | tee -a "$LOG"
  docker rm -f vllm-27b-ksweep >/dev/null 2>&1
  for c in $(docker ps -q); do docker kill "$c" >/dev/null 2>&1; done
  sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'; sleep 2
  launch "$k"
  ok=0
  for i in $(seq 1 96); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    docker ps -q --filter "name=vllm-27b-ksweep" | grep -q . || { docker logs vllm-27b-ksweep 2>&1|tail -6|tee -a "$LOG"; break; }
    sleep 5
  done
  if [ "$ok" != "1" ]; then printf "%-4s | %s\n" "$k" "FAILED to start" >> "$TABLE"; continue; fi
  B=$(python3 ~/dflash-setup/scripts/bench.py --url http://localhost:$PORT --tasks all --runs 3 --label "27b-k$k" 2>&1)
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
  docker rm -f vllm-27b-ksweep >/dev/null 2>&1
done
echo "=== 27B sweep DONE $(date) ===" | tee -a "$LOG"
cat "$TABLE" | tee -a "$LOG"

# PROVEN-WORKING CONFIGS + REVERT GUIDE (Jetson AGX Thor, sm_110a)

These are the exact launch commands verified to load and serve stably with DFlash on
2026-06-06 (image `vllm-dflash-thor:fa-native`, vLLM 0.20.0.dev0+dflash). If any optimization
breaks launch, revert to the matching command here. Git tag for this known-good state:
commit `e49459e` (and the configs in `scripts/serve-*.sh` at that commit).

Pre-flight before any launch (NEVER use `docker stop` or `sudo fuser -k /dev/nvidia*` on Thor —
the latter kills Xorg/gnome/RustDesk):
```
for c in $(docker ps -q); do docker kill "$c"; done
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
free -h | grep Mem      # expect >100 GB free
```

---

## 122B-A10B-NVFP4 + DFlash  — PROVEN  (base 10.9 → DFlash 27–42 tok/s, τ 4.2–6.5)
MoE=cutlass (marlin CRASHES at 256 experts), attention=TRITON_ATTN (flashinfer kv_cache_sf bug),
gpu-util 0.78 (0.72 starves KV, 0.90 trips precheck), draft KV capped at 1024.
```
docker run -d --name vllm-122b --runtime=nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_DISABLE_XET=1 -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -v ~/Qwen3.5-122B-A10B-NVFP4/resharded:/model -v ~/models/Qwen3.5-122B-A10B-DFlash:/draft \
  -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
  vllm-dflash-thor:fa-native \
  vllm serve /model \
    --speculative-config '{"method":"dflash","num_speculative_tokens":12,"model":"/draft","max_model_len":1024}' \
    --quantization compressed-tensors --load-format fastsafetensors \
    --moe-backend cutlass --attention-backend TRITON_ATTN \
    --gpu-memory-utilization 0.78 --max-model-len 16384 --max-num-seqs 2 \
    --trust-remote-code --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --port 8001
```

## 35B-A3B-NVFP4 + DFlash  — PROVEN  (marlin MoE + flash_attn)
```
docker run -d --name vllm-35b --runtime=nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_DISABLE_XET=1 -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -v ~/Qwen3.6-35B-A3B-NVFP4:/model -v ~/Qwen3.6-35B-A3B-DFlash:/draft \
  -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
  vllm-dflash-thor:fa-native \
  vllm serve /model \
    --speculative-config '{"method":"dflash","num_speculative_tokens":12,"model":"/draft"}' \
    --quantization compressed-tensors --load-format fastsafetensors \
    --moe-backend marlin --attention-backend flash_attn \
    --gpu-memory-utilization 0.78 --max-model-len 65536 --max-num-seqs 4 \
    --trust-remote-code --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --port 8001
```

## 27B-NVFP4 (dense) + DFlash  — PROVEN  (flash_attn; needs Qwen2Tokenizer overlay)
Build overlay first (27B ships tokenizer_class=TokenizersBackend which vLLM can't resolve):
```
TOK=~/dflash-setup/tokenizer-fix-27b; mkdir -p "$TOK"
cp ~/models/Qwen3.6-27B-NVFP4/tokenizer*.json ~/models/Qwen3.6-27B-NVFP4/*.txt "$TOK"/ 2>/dev/null
python3 -c "import json;p='$TOK/tokenizer_config.json';c=json.load(open(p));c['tokenizer_class']='Qwen2Tokenizer';json.dump(c,open(p,'w'))"
```
Then:
```
docker run -d --name vllm-27b --runtime=nvidia --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_DISABLE_XET=1 -e FASTSAFETENSORS_NOGDS=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -v ~/models/Qwen3.6-27B-NVFP4:/model -v ~/models/Qwen3.6-27B-DFlash:/draft \
  -v ~/dflash-setup/tokenizer-fix-27b:/tokenizer:ro \
  -v ~/thor-vllm-cache:/root/.cache/vllm -v ~/thor-triton-cache:/root/.triton \
  vllm-dflash-thor:fa-native \
  vllm serve /model --tokenizer /tokenizer \
    --speculative-config '{"method":"dflash","num_speculative_tokens":12,"model":"/draft"}' \
    --quantization compressed-tensors --load-format fastsafetensors \
    --attention-backend flash_attn \
    --gpu-memory-utilization 0.85 --max-model-len 65536 --max-num-seqs 4 \
    --trust-remote-code --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --port 8001
```

---

## Full revert to known-good
```
cd ~/dflash-vllm-resources && git checkout e49459e -- scripts/ RESULTS.md
sudo nvpmodel -m 1              # back to 120W if MAXN caused thermal/stability issues
```
System defaults (kernel tuning applied during 122B loads, restored on serve-script EXIT trap):
`vm.vfs_cache_pressure=100  vm.watermark_scale_factor=10  vm.min_free_kbytes=65536`.

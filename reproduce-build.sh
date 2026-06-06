#!/bin/bash
# reproduce-build.sh — build the DFlash vLLM image for Jetson Thor (sm_110a) from source.
# Produces:  vllm-dflash-thor:latest      (vLLM PR #40898 from source, ~92 min compile)
#            vllm-dflash-thor:fa-native   (latest + stock NATIVE sm_110 flash-attn .so)
#
# Prereqs:
#   - Jetson AGX Thor (aarch64, sm_110a), CUDA 13, Docker with nvidia runtime
#   - the stock base image: ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor  (vLLM 0.19.0)
#   - >= 24 GB swap recommended (aarch64 CUDA compile is memory-hungry)
set -euo pipefail
cd "$(dirname "$0")"

# ── ensure swap (compile OOMs without it) ──────────────────────────────────────
if [ "$(free -g | awk '/^Swap:/{print $2}')" -lt 24 ]; then
  echo "Expanding swap to 32G..."
  sudo swapoff /swapfile 2>/dev/null || true
  sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
fi
sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true

# ── 1. source build (DFlash, sm_110a) ──────────────────────────────────────────
echo "=== Building vllm-dflash-thor:latest (source build, ~90 min) ==="
printf '*\n' > .dockerignore   # Dockerfile self-clones; send empty context
docker build --network host -f Dockerfile.dflash-thor -t vllm-dflash-thor:latest \
  --progress=plain . 2>&1 | tee build.log

# ── 2. native sm_110 flash-attn swap ───────────────────────────────────────────
# The source build's flash-attn lacks native Thor kernels (FA2 sm_80 / FA3 sm_75).
# The stock image ships native sm_110 FA with byte-identical op symbols -> drop-in.
echo "=== Building vllm-dflash-thor:fa-native (native sm_110 FA2) ==="
docker build -f Dockerfile.fa-native -t vllm-dflash-thor:fa-native . 2>&1 | tee build-fa.log

# ── 3. verify (needs GPU + LD_PRELOAD) ─────────────────────────────────────────
echo "=== Verify ==="
docker run --rm --runtime nvidia --gpus all \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  vllm-dflash-thor:fa-native python -c "
import vllm, importlib.util, torch
print('vLLM:', vllm.__version__)
print('dflash:', importlib.util.find_spec('vllm.v1.spec_decode.dflash') is not None)
print('device:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
"
echo "=== Done. Images: vllm-dflash-thor:latest, vllm-dflash-thor:fa-native ==="

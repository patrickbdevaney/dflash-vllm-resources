#!/bin/bash
# serve-35b.sh — Qwen3.6-35B-A3B-NVFP4 + DFlash draft, Jetson Thor SM110
# MoE backend default 'marlin' (Step 3 compares vs triton); override via env:
#   MOE_BACKEND=triton VLLM_USE_FLASHINFER_MOE_FP4=1 ./serve-35b.sh
export MODEL_DIR="$HOME/Qwen3.6-35B-A3B-NVFP4"
export DRAFT_DIR="$HOME/Qwen3.6-35B-A3B-DFlash"
export MODEL_NAME="Qwen3.6-35B-A3B-NVFP4"
export GPU_UTIL="${GPU_UTIL:-0.78}"
export MAX_LEN="${MAX_LEN:-65536}"
export MAX_SEQS="${MAX_SEQS:-4}"
export MOE_BACKEND="${MOE_BACKEND-marlin}"
export NUM_SPEC="${NUM_SPEC:-12}"   # k-sweep optimal: k=12 (avg 116.5 tok/s, tau 6.25); k=15 tau higher but slower
# Decode optimizations (output-safe; measured A/B in benchmarks/35b-a3b-optimizations.md):
export VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"   # marlin reduction; ~neutral but free
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}" # GPU-side sampling, no host round-trip
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" # single cmd queue, less jitter at conc=1
source "$(dirname "$0")/_common.sh"

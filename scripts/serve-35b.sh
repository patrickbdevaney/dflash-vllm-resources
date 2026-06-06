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
source "$(dirname "$0")/_common.sh"

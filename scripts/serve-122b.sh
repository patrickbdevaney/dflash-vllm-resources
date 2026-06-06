#!/bin/bash
# serve-122b.sh — Qwen3.5-122B-A10B-NVFP4 (resharded) + DFlash draft, Jetson Thor SM110
# Tighter KV budget at 122B: lower util, max_len 32768, max_seqs 2.
export MODEL_DIR="$HOME/Qwen3.5-122B-A10B-NVFP4/resharded"
export DRAFT_DIR="$HOME/models/Qwen3.5-122B-A10B-DFlash"
export MODEL_NAME="Qwen3.5-122B-A10B-NVFP4"
export GPU_UTIL="${GPU_UTIL:-0.72}"
export MAX_LEN="${MAX_LEN:-32768}"
export MAX_SEQS="${MAX_SEQS:-2}"
export MOE_BACKEND="${MOE_BACKEND-marlin}"
source "$(dirname "$0")/_common.sh"

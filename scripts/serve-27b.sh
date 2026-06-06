#!/bin/bash
# serve-27b.sh — Qwen3.6-27B-NVFP4 + DFlash draft, Jetson Thor SM110
# NOTE: this 27B ships tokenizer_class="TokenizersBackend" which transformers/vLLM
# cannot resolve. We build a small patched tokenizer overlay (class -> Qwen2Tokenizer;
# tokenizer.json loads fine as a fast tokenizer) and point vLLM at it via --tokenizer.
set -euo pipefail
export MODEL_DIR="$HOME/models/Qwen3.6-27B-NVFP4"
export DRAFT_DIR="$HOME/models/Qwen3.6-27B-DFlash"
export MODEL_NAME="Qwen3.6-27B-NVFP4"
export GPU_UTIL="${GPU_UTIL:-0.85}"
export MAX_LEN="${MAX_LEN:-65536}"
export MAX_SEQS="${MAX_SEQS:-4}"
export MOE_BACKEND="${MOE_BACKEND-marlin}"

# ── build patched tokenizer overlay ────────────────────────────────────────────
TOK_FIX="$HOME/dflash-setup/tokenizer-fix-27b"
mkdir -p "$TOK_FIX"
for f in tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json added_tokens.json vocab.json merges.txt generation_config.json; do
  [ -f "$MODEL_DIR/$f" ] && cp -f "$MODEL_DIR/$f" "$TOK_FIX/$f"
done
python3 - "$TOK_FIX/tokenizer_config.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
if c.get("tokenizer_class") in (None, "TokenizersBackend"):
    c["tokenizer_class"] = "Qwen2Tokenizer"   # tokenizer.json is the actual fast tokenizer
    json.dump(c, open(p, "w"), ensure_ascii=False, indent=2)
    print("patched tokenizer_class -> Qwen2Tokenizer")
else:
    print("tokenizer_class already:", c.get("tokenizer_class"))
PY
export TOKENIZER_DIR="$TOK_FIX"

source "$(dirname "$0")/_common.sh"

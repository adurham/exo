#!/bin/bash
# Q1D Task B -- throttled sampled-shard fetch of the ORIGINAL pre-EXL3 uncensored
# checkpoint (dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8, FP8, public/not gated).
# Layers 0, 20, 39 live one per shard: model-00003/00023/00042 (of 48).
# Throttled to protect owner traffic; resumable. Run under nohup on studio1.
set -u
T=$(cat "$HOME/.cache/huggingface/token")
R="dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8"
D=/tmp/q1d_mem/orig
mkdir -p "$D"; cd "$D" || exit 1
for F in model-00003-of-00048.safetensors model-00023-of-00048.safetensors model-00042-of-00048.safetensors; do
  echo "$(date '+%F %T') START $F"
  curl -L -C - --limit-rate 20M --retry 5 --retry-delay 5 -m 7200 \
    -H "Authorization: Bearer $T" \
    -o "$F" "https://huggingface.co/$R/resolve/main/$F"
  echo "$(date '+%F %T') rc=$? $F bytes=$(wc -c < "$F" 2>/dev/null)"
done
echo "$(date '+%F %T') ALL DONE"

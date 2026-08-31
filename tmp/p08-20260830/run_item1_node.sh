#!/bin/zsh
# P08 Item 1 capture runner on macstudio-m4-1 (p07 run_node.sh recipe).
# Production-parity env BEFORE python starts.
set -e
cd ~/repos/exo
export METAL_CAPTURE_ENABLED=1
export MLX_GPU_TIME=1
export MLX_DISPATCH_COUNT=1
export MLX_GEMV_BATCH_INVARIANT=1
export MLX_STEEL_BATCH_INVARIANT=1
export MLX_MAX_OPS_PER_BUFFER=200
export MLX_MAX_MB_PER_BUFFER=200
export EXO_DSV4_INDEX_TOPK=512
.venv/bin/python tmp/p08-20260830/p08_item1_capture.py \
  2>&1 | tee tmp/p08-20260830/item1_stdout.log
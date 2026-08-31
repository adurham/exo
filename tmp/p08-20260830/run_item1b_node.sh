#!/bin/zsh
# P08 Item 1b REACHABILITY harness on macstudio-m4-1 (same recipe as run_item1_node.sh).
# Standalone beside the live runner (PID 59909, DO NOT TOUCH).
set -e
cd ~/repos/exo
export METAL_CAPTURE_ENABLED=1
export MLX_GPU_TIME=1
export MLX_DISPATCH_COUNT=1
export MLX_GEMV_BATCH_INVARIANT=1
export MLX_STEEL_BATCH_INVARIANT=1
export MLX_MAX_OPS_PER_BUFFER=200
export MLX_MAX_MB_PER_BUFFER=200
.venv/bin/python tmp/p08-20260830/p08_item1b_reachability.py \
  2>&1 | tee tmp/p08-20260830/item1b_stdout.log
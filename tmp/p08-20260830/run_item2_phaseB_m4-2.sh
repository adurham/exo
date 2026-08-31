#!/bin/zsh
# P08 Item2 Phase B capture on macstudio-m4-2 (beside live runner PID 60392).
# Production-kernel-parity env BEFORE python starts (harness asserts them).
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
.venv/bin/python tmp/p08-20260830/item2_phaseB.py \
  2>&1 | tee tmp/p08-20260830/item2_phaseB_capture.log
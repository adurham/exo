#!/bin/zsh
# P07 capture runner on macstudio-m4-1. Mirrors P03 recipe:
# production-kernel-parity env BEFORE python starts (mlx import asserts them).
set -e
cd ~/repos/exo
export METAL_CAPTURE_ENABLED=1
export MLX_GPU_TIME=1
export MLX_DISPATCH_COUNT=1
export MLX_GEMV_BATCH_INVARIANT=1
export MLX_STEEL_BATCH_INVARIANT=1
export MLX_MAX_OPS_PER_BUFFER=200
export MLX_MAX_MB_PER_BUFFER=200
export EXO_DSV4_HC_EXPAND_KERNEL=1
export EXO_DSV4_HC_COLLAPSE_KERNEL=1
export EXO_DSV4_INDEX_TOPK=512
export EXO_DSV4_EXACT_TOPK=1   # explicit: default-on, asserted on
.venv/bin/python tmp/p07-20260830/p07_prefill_remainder_capture.py \
  2>&1 | tee tmp/p07-20260830/capture_stdout.log

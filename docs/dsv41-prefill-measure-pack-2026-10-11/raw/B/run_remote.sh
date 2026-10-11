#!/bin/bash
# Remote runner for Task B (executed ON studio2 under the phase20 guard).
# args: <ctx> <layers> <reps> <out_json>
cd /tmp/p20_pmp_B || exit 3
export MLX_GPU_TIME=1
export MTL_DISABLE_TIMEOUT=1
export MTL_COMMAND_BUFFER_TIMEOUT=0
export P20_PMP_TRACE=/tmp/p20_pmp_B/trace.json
export P20_PMP_CTX="$1"
export P20_PMP_LAYERS="$2"
export P20_PMP_REPS="$3"
export P20_PMP_OUT="$4"
exec env -u PYTHONPATH /Users/adam.durham/repos/exo/.venv/bin/python p20_pmp_taskB_opclass.py

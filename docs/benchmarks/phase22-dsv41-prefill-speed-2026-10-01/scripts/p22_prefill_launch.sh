#!/bin/sh
# usage: p22_prefill_launch.sh <rank>   -- launch one rank of a phase-22 job.
#
# Same RDMA/coordinator layout as production's launcher. Deliberately does NOT
# set MLX_ENABLE_TELEMETRY=1 (a prior campaign found every slow arm shared that
# env var -- two-node-mac-runner skill: "Diff the LAUNCHER and its full env").
# MLX_MAX_OPS_PER_BUFFER is an env override so the lever sweep needs no file
# edit. PYTHONPATH puts the DEPLOYED mlx_lm tree first; the harness logs what it
# actually imported. The log name carries $P22_LBL so each arm's evidence is
# archived under its own label instead of clobbering the previous arm's.
RANK=$1
LBL=${P22_LBL:-nolabel}
cd "$HOME" || exit 1
echo '[[null, "rdma_en3"], ["rdma_en4", null]]' > "$HOME/p47_ibv.json"
if [ "$RANK" = 0 ]; then COORD=0.0.0.0:49231; else COORD=192.168.201.2:49231; fi
export MLX_IBV_DEVICES="$HOME/p47_ibv.json" MLX_RANK=$RANK MLX_JACCL_COORDINATOR=$COORD
export MLX_JACCL_RELIABLE_DATA=1 MLX_JACCL_RELIABLE_MAX_SZ=2 MLX_JACCL_RELIABLE_INFLIGHT=8 \
  MLX_JACCL_RELIABLE_OPTIMISTIC=1 MLX_JACCL_RECONNECT_FRESH=1 MLX_JACCL_ACK_SYNC_PRE=1 \
  MLX_JACCL_ACK_RETRANSMIT_US=500000 IBV_FORK_SAFE=1 MLX_EVENT_WAIT_TIMEOUT_MS=20000 \
  EXL3_MM_MAX_ROWS=100000 PYTHONUNBUFFERED=1 \
  MTL_DISABLE_TIMEOUT=1 MTL_COMMAND_BUFFER_TIMEOUT=0 EXO_DISABLE_METAL_TIMEOUT=1 \
  AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 \
  MLX_MAX_OPS_PER_BUFFER=${MLX_MAX_OPS_PER_BUFFER:-200} \
  MLX_MAX_MB_PER_BUFFER=${MLX_MAX_MB_PER_BUFFER:-200} \
  DSV41_SYNC_COLLECTIVES=${DSV41_SYNC_COLLECTIVES:-0} \
  DSV41_SYNC_WARM_CALLS=${DSV41_SYNC_WARM_CALLS:-2}
export PYTHONPATH="$HOME/dsv41-test"
echo "LAUNCH rank=$RANK lbl=$LBL ops_per_buffer=$MLX_MAX_OPS_PER_BUFFER mb_per_buffer=$MLX_MAX_MB_PER_BUFFER"
nohup "$HOME/repos/exo/.venv/bin/python" -u "$HOME/p22_prefill.py" > "$HOME/p22_prefill-${LBL}-r$RANK.log" 2>&1 &
echo "pid=$!"

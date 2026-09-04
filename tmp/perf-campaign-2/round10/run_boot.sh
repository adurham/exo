#!/bin/bash
# Round-10 paired-boot driver. Usage:
#   run_boot.sh <label> <rv_value|DEFAULT> [extra "K=V K=V"]
# Does: teardown -> launch -> wait READY -> 300s idle -> ps eww verify on BOTH nodes.
# Writes everything to /tmp/r10_boot<LABEL>_*.log and .../round10/results/<LABEL>_env.txt
#
# If RV is the literal string DEFAULT, launch with ONLY DSV4_KV_CACHE_BITS=0 and
# NO EXO_BATCHED_PREFILL_RENDEZVOUS_MS env var at all, so start_cluster.sh's own
# default takes effect. Still ps-eww-verify afterward to prove it.
set -u
N="$1"
RV="$2"
EXTRA="${3:-}"
R10=/Users/adam.durham/repos/exo/tmp/perf-campaign-2/round10
mkdir -p "$R10/results"
LAUNCH_LOG=/tmp/r10_boot${N}_launch.log
STATUS=/tmp/r10_boot${N}_status.txt

echo "PHASE=teardown $(date -u +%FT%TZ)" > "$STATUS"
for h in macstudio-m4-1 macstudio-m4-2; do
  ssh -o ConnectTimeout=15 "$h" 'screen -S exorun -X quit >/dev/null 2>&1; pkill -f "exo -v" >/dev/null 2>&1; sleep 2; pgrep -af "exo -v" | wc -l' \
    >> "$STATUS" 2>&1
done
sleep 5

echo "PHASE=launch RV=$RV EXTRA=$EXTRA $(date -u +%FT%TZ)" >> "$STATUS"
cd /Users/adam.durham/repos/exo || exit 1
# push-gate prompt is informational (nodes rsync from THIS working tree, not github)
printf 'y\ny\ny\ny\ny\n' > /dev/null  # (stdin piping does NOT work: ssh calls inside
# start_cluster.sh at lines 944/984/1000/1033 have no -n and drain the pipe before
# the push-gate `read -p` at :1132 ever runs, leaving it at EOF -> "Aborted.")
# Launch in its own tmux session (real pty) and answer the gate via send-keys,
# exactly as R9 did.
tmux kill-session -t "r10launch${N}" 2>/dev/null
if [ "$RV" = "DEFAULT" ]; then
  # No EXO_BATCHED_PREFILL_RENDEZVOUS_MS at all: prove the script's own default applies.
  tmux new-session -d -s "r10launch${N}" \
    "cd /Users/adam.durham/repos/exo && DSV4_KV_CACHE_BITS=0 $EXTRA ./start_cluster.sh 2>&1 | tee $LAUNCH_LOG; echo LAUNCH_SCRIPT_EXITED; sleep 999999"
else
  tmux new-session -d -s "r10launch${N}" \
    "cd /Users/adam.durham/repos/exo && DSV4_KV_CACHE_BITS=0 EXO_BATCHED_PREFILL_RENDEZVOUS_MS=$RV $EXTRA ./start_cluster.sh 2>&1 | tee $LAUNCH_LOG; echo LAUNCH_SCRIPT_EXITED; sleep 999999"
fi
echo "PHASE=launch_tmux r10launch${N} $(date -u +%FT%TZ)" >> "$STATUS"

# answer the push gate when/if it appears
for i in $(seq 1 120); do
  if grep -q "Continue anyway" "$LAUNCH_LOG" 2>/dev/null; then
    sleep 1
    tmux send-keys -t "r10launch${N}" "y"
    echo "PHASE=push_gate_answered $(date -u +%FT%TZ)" >> "$STATUS"
    break
  fi
  grep -q "READY (2/2)\|Skipping push check\|is on origin/" "$LAUNCH_LOG" 2>/dev/null && break
  sleep 2
done

# wait for READY (2/2)
for i in $(seq 1 240); do
  if grep -q "READY (2/2)" "$LAUNCH_LOG" 2>/dev/null; then
    echo "PHASE=ready $(date -u +%FT%TZ)" >> "$STATUS"
    break
  fi
  sleep 5
done
if ! grep -q "READY (2/2)" "$LAUNCH_LOG" 2>/dev/null; then
  echo "PHASE=READY_TIMEOUT $(date -u +%FT%TZ)" >> "$STATUS"
  exit 2
fi

READY_TS=$(date -u +%FT%TZ)
echo "READY_TS=$READY_TS" >> "$STATUS"
echo "PHASE=idle_300s_start $(date -u +%FT%TZ)" >> "$STATUS"
sleep 300
echo "PHASE=idle_300s_done $(date -u +%FT%TZ)" >> "$STATUS"

# ---- ps eww verification on the REAL runner pids, BOTH nodes ----
ENVOUT="$R10/results/${N}_env.txt"
: > "$ENVOUT"
for h in macstudio-m4-1 macstudio-m4-2; do
  echo "=== $h ===" >> "$ENVOUT"
  ssh -o ConnectTimeout=15 "$h" 'for p in $(pgrep -f "exo -v"); do
      echo "--- PID $p : $(ps -o command= -p $p | cut -c1-90)";
      ps eww $p | tr " " "\n" | grep -E "^(EXO_BATCHED_PREFILL_RENDEZVOUS_MS|EXO_DSV4_BATCHED_PREFILL|EXO_DECODE_PROBE|MLX_GPU_TIME|EXO_SPECULATIVE_GAMMA|EXO_DSV4_MTP|MLX_STEEL_BATCH_INVARIANT|DSV4_KV_CACHE_BITS)=";
    done' >> "$ENVOUT" 2>&1
done
echo "PHASE=verify_done $(date -u +%FT%TZ)" >> "$STATUS"
echo "BOOT${N}_COMPLETE" >> "$STATUS"

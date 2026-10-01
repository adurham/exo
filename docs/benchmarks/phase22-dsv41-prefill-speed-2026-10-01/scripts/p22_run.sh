#!/bin/bash
# p22_run.sh <script.py> <label> [ENV=VAL ...]
#
# Two-node phase-22 runner. Holds ~/dsv41-gpu.lock on BOTH Macs through the
# parent's own parent_lock.sh (token-based want file, bounded self-release),
# refuses to start if the hold cannot be taken, launches rank 0 on
# macstudio-m4-2 and rank 1 on macstudio-m4-1 in SEPARATE calls (the ranks fail
# to rendezvous if launched in one command), waits for the script's own
# P22_DONE marker by polling the rank-0 log, archives both logs into
# $RUNROOT, and ALWAYS drops the lock via a trap.
#
# The launch template used is the script's own <stem>_launch.sh next to this
# file -- deliberately NOT regenerated here, so the env each arm runs under is
# visible and reviewable (a prior campaign lost a day to a launcher diff).
#
# Env for the CALLER: P22_TAG (results dir name), P22_LENS, P22_REPS, P22_ARMS,
# P22_CHUNKS, P22_ATTR, P22_ORACLE, P22_MAXWAIT, RUNROOT, and any MACHINE env
# (e.g. MLX_MAX_OPS_PER_BUFFER=<n>) which is forwarded to both ranks verbatim.
set -u
S="$(cd "$(dirname "$0")" && pwd)"
PY="$1"; LBL="$2"; shift 2
# Machine env (MLX_MAX_OPS_PER_BUFFER=<n> ...) passed as ENV=VAL args, PLUS the
# P22_* control vars taken from the CALLER's environment. Without this the
# control vars never reach the nodes (they are not in "$*"), and two arms
# silently write to the same results dir with the same defaults -- observed:
# a second run overwrote the first's results.json.
CTRL=""
for v in P22_TAG P22_LENS P22_REPS P22_ARMS P22_CHUNKS P22_CHUNK P22_ATTR \
         P22_ATTR_LEN P22_ORACLE P22_ORACLE_TOKENS P22_MAX_SEQ_PAD \
         P22_PROMPTS P22_SESSION_PY P22_PKG; do
  eval "val=\${$v:-}"
  [ -n "$val" ] && CTRL="$CTRL $v=$val"
done
ENVP="$CTRL $*"
STEM="$(basename "$PY" .py)"
RUNROOT="${RUNROOT:-$HOME/p22_runs}"
mkdir -p "$RUNROOT"
MAXWAIT="${P22_MAXWAIT:-7200}"
PROMPTS_SRC="${P22_PROMPTS_SRC:-$HOME/.hermes/cache/scratch/exl3patch/p22_prompts}"

if [ ! -f "$S/${STEM}_launch.sh" ]; then
  echo "P22_ABORT missing $S/${STEM}_launch.sh"; exit 2
fi

"$S/parent_lock.sh" take || { echo "P22_ABORT lock not taken"; exit 3; }
trap '"$S/parent_lock.sh" drop >/dev/null 2>&1' EXIT INT TERM HUP

for n in macstudio-m4-1 macstudio-m4-2; do
  scp -q "$S/$PY" "$S/${STEM}_launch.sh" "$n:" || { echo "P22_ABORT scp $n"; exit 4; }
  # the fixed prompts must be present on BOTH nodes (rank 0 runs on m4-2, rank 1
  # on m4-1; a missing prompt file is a run that dies after paying the load)
  ssh -o BatchMode=yes "$n" "rm -f ~/${STEM}-*.log ~/${STEM}.log; mkdir -p ~/p22_out ~/p22_prompts" || true
  scp -q "$PROMPTS_SRC"/*.json "$PROMPTS_SRC"/*.txt "$n:p22_prompts/" || { echo "P22_ABORT prompts -> $n"; exit 4; }
done

# rank 0 (coordinator) on m4-2, rank 1 on m4-1 -- production's layout.
# P22_LBL is exported so both ranks' logs carry the same arm label.
ssh -o BatchMode=yes macstudio-m4-2 "cd ~ && P22_LBL='$LBL' $ENVP sh ~/${STEM}_launch.sh 0"
sleep 3
ssh -o BatchMode=yes macstudio-m4-1 "cd ~ && P22_LBL='$LBL' $ENVP sh ~/${STEM}_launch.sh 1"

t0=$(date +%s)
while :; do
  sleep 15
  now=$(date +%s); el=$(( now - t0 ))
  st=$(ssh -o BatchMode=yes macstudio-m4-2 "grep -c 'P22_DONE' ~/${STEM}-${LBL}-r0.log 2>/dev/null" 2>/dev/null)
  st=${st:-0}
  to=$(ssh -o BatchMode=yes macstudio-m4-2 "grep -c 'GPU Timeout' ~/${STEM}-${LBL}-r0.log 2>/dev/null" 2>/dev/null)
  to=${to:-0}
  up0=$(ssh -o BatchMode=yes macstudio-m4-2 "pgrep -f '[${STEM:0:1}]${STEM:1}.py' >/dev/null && echo up || echo down" 2>/dev/null)
  up1=$(ssh -o BatchMode=yes macstudio-m4-1 "pgrep -f '[${STEM:0:1}]${STEM:1}.py' >/dev/null && echo up || echo down" 2>/dev/null)
  echo "P22_WAIT t=${el}s done=${st} to=${to} r0=${up0} r1=${up1}"
  [ "${st}" -gt 0 ] && { echo "P22_MARKER t=${el}s"; break; }
  [ "${to}" -gt 50 ] && { echo "P22_ABORT gpu_timeouts=${to} t=${el}s"; break; }
  if [ "${up0:-up}" = down ] && [ "${up1:-up}" = down ]; then
    echo "P22_EXIT_NO_MARKER t=${el}s"; break
  fi
  [ "$el" -ge "$MAXWAIT" ] && { echo "P22_ABORT wait cap ${MAXWAIT}s"; break; }
done

# archive on the NODES first (a scp of a node-side path then cannot fail on a
# missing local dir), then pull
for n in macstudio-m4-1 macstudio-m4-2; do
  ssh -o BatchMode=yes "$n" "pkill -KILL -f '[${STEM:0:1}]${STEM:1}.py' 2>/dev/null; for r in 0 1; do [ -f ~/${STEM}-${LBL}-r\$r.log ] && cp ~/${STEM}-${LBL}-r\$r.log ~/p22_out/; done; true" || true
done
mkdir -p "$RUNROOT"
scp -q macstudio-m4-2:~/p22_out/${STEM}-${LBL}-r0.log "$RUNROOT/" 2>/dev/null || true
scp -q macstudio-m4-1:~/p22_out/${STEM}-${LBL}-r1.log "$RUNROOT/" 2>/dev/null || true

for r in 0 1; do
  H=macstudio-m4-2; [ "$r" = 1 ] && H=macstudio-m4-1
  echo "===== rank $r on $H"
  ssh -o BatchMode=yes "$H" "grep -E '^\[p22|^P22|GPU Timeout' ~/${STEM}-${LBL}-r$r.log 2>/dev/null | cut -c1-600; echo '--- errors:'; grep -m3 -E 'Traceback|NameError|FileNotFound|RuntimeError|TypeError|Error:' ~/${STEM}-${LBL}-r$r.log 2>/dev/null | cut -c1-300"
done
echo "P22_LOGS $RUNROOT/${STEM}-${LBL}-r{0,1}.log"
echo "P22_RUN_END $LBL"

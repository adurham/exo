#!/usr/bin/env bash
# 4-arm matched-depth entropy A/B driver.
# Supersedes run_entropy.sh (2-mode/3-iter). Adds: random arm, repetitive_recheck
# drift-control arm, correct probe rc capture, runner PID/lstart capture,
# error-window harvest (G3), and pre-window anchor harvest for de-aggregation.
#
# READ-ONLY w.r.t. cluster config: only fires API requests + reads logs.
# NO relaunch, NO config change, NO kill. The PM runs this; do not run it here.
set -uo pipefail   # NOT -e: one failed arm must not abort the rest.

OUT=/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901
ERAW=$OUT/entropy/raw
N1=macstudio-m4-1
N2=macstudio-m4-2
PROBE=$OUT/entropy_probe.py
PY=/Users/adam.durham/repos/exo/.venv/bin/python
PYTHONPATH=/Users/adam.durham/repos/exo/tools/src
mkdir -p "$ERAW"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$ERAW/entropy_ab_driver.log"; }
logbytes(){ ssh -o ConnectTimeout=10 "$1" 'wc -c < ~/exo.log' 2>/dev/null | tr -d ' '; }

# Isolate the GENUINE runner (not the SCREEN/login/zsh wrappers that also match '-m exo').
runner_pids(){
  ssh -o ConnectTimeout=15 "$1" "ps -eo pid,lstart,command | grep '[.]venv/bin/python -m exo' | grep -v grep" 2>/dev/null
}

idle_wait(){
  # Block until both nodes report gpu_usage_ratio < 0.10 (max ~10 min).
  for i in $(seq 1 60); do
    r1=$(curl -s --max-time 10 "http://192.168.86.201:52415/metrics" | awk '/^exo_gpu_usage_ratio/{print $2}')
    r2=$(curl -s --max-time 10 "http://192.168.86.202:52415/metrics" | awk '/^exo_gpu_usage_ratio/{print $2}')
    ok=$(python3 -c "
try:
    print('1' if (float('${r1:-1}')<0.10 and float('${r2:-1}')<0.10) else '0')
except Exception:
    print('0')")
    if [ "$ok" = "1" ]; then log "idle OK (n1=$r1 n2=$r2)"; return 0; fi
    log "waiting for idle (n1=$r1 n2=$r2)"; sleep 10
  done
  log "WARN idle timeout; proceeding"; return 0
}

run_arm(){
  local mode="$1" words="$2" tag="$3" iterations="$4"
  log "=== ARM $tag (mode=$mode words=$words iterations=$iterations) ==="
  idle_wait

  # Runner identity BEFORE the arm (G3 restart check).
  runner_pids $N1 > "$ERAW/pids_${tag}_before_n1.txt"
  runner_pids $N2 > "$ERAW/pids_${tag}_before_n2.txt"
  log "pids before n1=$(wc -l < "$ERAW/pids_${tag}_before_n1.txt") n2=$(wc -l < "$ERAW/pids_${tag}_before_n2.txt")"

  local b1 b2 a1 a2 rc
  b1=$(logbytes $N1); b2=$(logbytes $N2)
  log "log offsets before n1=$b1 n2=$b2"

  # Neutral cwd (avoids a direnv/nix trap in the repo dir).
  cd /tmp || exit 1
  PYTHONPATH=$PYTHONPATH \
  "$PY" \
    "$PROBE" \
    --mode "$mode" --fixed-words "$words" \
    --iterations "$iterations" --warmup 1 --max-tokens 256 --timeout 3600 --seed 1234 \
    --json-out "$ERAW/entropy_${tag}.json" \
    > "$ERAW/entropy_${tag}.log" 2>&1
  rc=$?   # MUST be the line immediately after the probe; run_entropy.sh's `log "rc=$?"` captured the wrong command.
  log "probe rc=$rc for $tag"

  a1=$(logbytes $N1); a2=$(logbytes $N2)
  log "log offsets after n1=$a1 n2=$a2"

  # Runner identity AFTER the arm (G3 restart check).
  runner_pids $N1 > "$ERAW/pids_${tag}_after_n1.txt"
  runner_pids $N2 > "$ERAW/pids_${tag}_after_n2.txt"
  log "pids after n1=$(wc -l < "$ERAW/pids_${tag}_after_n1.txt") n2=$(wc -l < "$ERAW/pids_${tag}_after_n2.txt")"

  # Harvest ONLY this arm's byte-window: MTP-PROF lines (profiler), error lines (G3).
  ssh -o ConnectTimeout=15 $N1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -a 'MTP-PROF'" > "$ERAW/prof_ent_${tag}_n1.txt" 2>/dev/null
  ssh -o ConnectTimeout=15 $N2 "tail -c +$((b2+1)) ~/exo.log | head -c $((a2-b2)) | grep -a 'MTP-PROF'" > "$ERAW/prof_ent_${tag}_n2.txt" 2>/dev/null
  log "prof lines n1=$(wc -l < "$ERAW/prof_ent_${tag}_n1.txt") n2=$(wc -l < "$ERAW/prof_ent_${tag}_n2.txt")"

  # Integrity check G3: errors / degeneration / restarts in the same window.
  ssh -o ConnectTimeout=15 $N1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -aiE 'error|degenerat|traceback|exception|restart|CRASH'" > "$ERAW/errs_${tag}_n1.txt" 2>/dev/null
  ssh -o ConnectTimeout=15 $N2 "tail -c +$((b2+1)) ~/exo.log | head -c $((a2-b2)) | grep -aiE 'error|degenerat|traceback|exception|restart|CRASH'" > "$ERAW/errs_${tag}_n2.txt" 2>/dev/null
  log "err lines n1=$(wc -l < "$ERAW/errs_${tag}_n1.txt") n2=$(wc -l < "$ERAW/errs_${tag}_n2.txt")"

  # Anchor: [MTP-PROF] lines from the last ~200KB STRICTLY BEFORE the window.
  # analyze_prof.anchors_from() needs the last pre-window dump per phase so
  # cumulative means can be de-aggregated without importing pre-window cycles.
  ssh -o ConnectTimeout=15 $N1 "head -c $b1 ~/exo.log | tail -c 200000 | grep -a 'MTP-PROF'" > "$ERAW/prof_anchor_${tag}_n1.txt" 2>/dev/null
  ssh -o ConnectTimeout=15 $N2 "head -c $b2 ~/exo.log | tail -c 200000 | grep -a 'MTP-PROF'" > "$ERAW/prof_anchor_${tag}_n2.txt" 2>/dev/null
  log "anchor lines n1=$(wc -l < "$ERAW/prof_anchor_${tag}_n1.txt") n2=$(wc -l < "$ERAW/prof_anchor_${tag}_n2.txt")"

  log "########## ARM $tag COMPLETE ##########"
}

log "########## ENTROPY A/B START ##########"
run_arm repetitive 75000  repetitive          5
run_arm natural    65646  natural            5
run_arm random     23525  random             5
run_arm repetitive 75000  repetitive_recheck 2   # drift control; mode repetitive, distinct tag
log "########## ENTROPY A/B COMPLETE ##########"

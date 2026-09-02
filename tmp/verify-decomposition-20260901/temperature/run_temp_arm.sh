#!/usr/bin/env bash
# ONE-arm temperature driver: temp=1.0 vs the established temp=0.8 baseline.
# Near-copy of entropy/run_entropy_ab.sh run_arm() with a single arm and a
# --temperature parameter on the probe.
#
# READ-ONLY w.r.t. cluster config: only fires API requests + reads logs.
# NO relaunch, NO config change, NO kill, NO restart.
set -uo pipefail

TDIR=/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901/temperature
RAW=$TDIR/raw
OUT=/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901
N1=macstudio-m4-1
N2=macstudio-m4-2
PROBE=$TDIR/temp_probe.py
PY=/Users/adam.durham/repos/exo/.venv/bin/python
PYTHONPATH=/Users/adam.durham/repos/exo/tools/src
mkdir -p "$RAW"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RAW/temp_arm_driver.log"; }
logbytes(){ ssh -o ConnectTimeout=10 "$1" 'wc -c < ~/exo.log' 2>/dev/null | tr -d ' '; }

# Isolate the GENUINE runner (not the SCREEN/login/zsh wrappers that also match '-m exo').
runner_pids(){
  ssh -o ConnectTimeout=15 "$1" "ps -eo pid,lstart,command | grep '[.]venv/bin/python -m exo' | grep -v grep" 2>/dev/null
}

gpu_sample(){
  # One sample of gpu_usage_ratio on both nodes, appended with a timestamp.
  r1=$(curl -s --max-time 10 "http://192.168.86.201:52415/metrics" | awk '/^exo_gpu_usage_ratio/{print $2}')
  r2=$(curl -s --max-time 10 "http://192.168.86.202:52415/metrics" | awk '/^exo_gpu_usage_ratio/{print $2}')
  echo "$(date '+%H:%M:%S') n1=${r1:-NA} n2=${r2:-NA}" >> "$RAW/gpu_samples_temp10.log"
}

gpu_sampler(){
  # Background sampler: every 30s until killed by the driver at arm end.
  while true; do
    gpu_sample
    sleep 30
  done
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

MODE=repetitive
WORDS=75000
TAG=temp10
ITERATIONS=5
TEMPERATURE=1.0

log "########## TEMP ARM START (mode=$MODE words=$WORDS tag=$TAG iters=$ITERATIONS temperature=$TEMPERATURE) ##########"
idle_wait

# Runner identity BEFORE the arm (G1 restart check).
runner_pids $N1 > "$RAW/pids_${TAG}_before_n1.txt"
runner_pids $N2 > "$RAW/pids_${TAG}_before_n2.txt"
log "pids before n1=$(wc -l < "$RAW/pids_${TAG}_before_n1.txt") n2=$(wc -l < "$RAW/pids_${TAG}_before_n2.txt")"
log "pids before n1: $(cat "$RAW/pids_${TAG}_before_n1.txt")"
log "pids before n2: $(cat "$RAW/pids_${TAG}_before_n2.txt")"

b1=$(logbytes $N1); b2=$(logbytes $N2)
log "log offsets before n1=$b1 n2=$b2"

# Start the GPU contention sampler (killed after the arm).
gpu_sampler_pid=""
gpu_sampler() { while true; do gpu_sample; sleep 30; done; }
gpu_sampler &
gpu_sampler_pid=$!
log "gpu sampler started pid=$gpu_sampler_pid (every 30s -> gpu_samples_${TAG}.log)"

# Neutral cwd (avoids a direnv/nix trap in the repo dir).
cd /tmp || exit 1
PYTHONPATH=$PYTHONPATH \
"$PY" \
  "$PROBE" \
  --mode "$MODE" --fixed-words "$WORDS" \
  --iterations "$ITERATIONS" --warmup 1 --max-tokens 256 --timeout 3600 --seed 1234 \
  --temperature "$TEMPERATURE" \
  --json-out "$RAW/temp_${TAG}.json" \
  > "$RAW/temp_${TAG}.log" 2>&1
rc=$?   # MUST be the line immediately after the probe invocation.
log "probe rc=$rc for $TAG"

a1=$(logbytes $N1); a2=$(logbytes $N2)
log "log offsets after n1=$a1 n2=$a2"

kill "$gpu_sampler_pid" 2>/dev/null
log "gpu sampler stopped"

# Runner identity AFTER the arm (G1 restart check).
runner_pids $N1 > "$RAW/pids_${TAG}_after_n1.txt"
runner_pids $N2 > "$RAW/pids_${TAG}_after_n2.txt"
log "pids after n1=$(wc -l < "$RAW/pids_${TAG}_after_n1.txt") n2=$(wc -l < "$RAW/pids_${TAG}_after_n2.txt")"
log "pids after n1: $(cat "$RAW/pids_${TAG}_after_n1.txt")"
log "pids after n2: $(cat "$RAW/pids_${TAG}_after_n2.txt")"

# Harvest ONLY this arm's byte-window: MTP-PROF lines (profiler), error lines (G3).
ssh -o ConnectTimeout=15 $N1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -a 'MTP-PROF'" > "$RAW/prof_${TAG}_n1.txt" 2>/dev/null
ssh -o ConnectTimeout=15 $N2 "tail -c +$((b2+1)) ~/exo.log | head -c $((a2-b2)) | grep -a 'MTP-PROF'" > "$RAW/prof_${TAG}_n2.txt" 2>/dev/null
log "prof lines n1=$(wc -l < "$RAW/prof_${TAG}_n1.txt") n2=$(wc -l < "$RAW/prof_${TAG}_n2.txt")"

# Integrity check G3: errors / degeneration / restarts / SIGKILL in the same window.
ssh -o ConnectTimeout=15 $N1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -aiE 'error|degenerat|traceback|exception|restart|CRASH|SIGKILL|PPSpec'" > "$RAW/errs_${TAG}_n1.txt" 2>/dev/null
ssh -o ConnectTimeout=15 $N2 "tail -c +$((b2+1)) ~/exo.log | head -c $((a2-b2)) | grep -aiE 'error|degenerat|traceback|exception|restart|CRASH|SIGKILL|PPSpec'" > "$RAW/errs_${TAG}_n2.txt" 2>/dev/null
log "err lines n1=$(wc -l < "$RAW/errs_${TAG}_n1.txt") n2=$(wc -l < "$RAW/errs_${TAG}_n2.txt")"

# Anchor: [MTP-PROF] lines from the last ~200KB STRICTLY BEFORE the window.
# analyze_prof.anchors_from() needs the last pre-window dump per phase so
# cumulative means can be de-aggregated without importing pre-window cycles.
ssh -o ConnectTimeout=15 $N1 "head -c $b1 ~/exo.log | tail -c 200000 | grep -a 'MTP-PROF'" > "$RAW/prof_anchor_${TAG}_n1.txt" 2>/dev/null
ssh -o ConnectTimeout=15 $N2 "head -c $b2 ~/exo.log | tail -c 200000 | grep -a 'MTP-PROF'" > "$RAW/prof_anchor_${TAG}_n2.txt" 2>/dev/null
log "anchor lines n1=$(wc -l < "$RAW/prof_anchor_${TAG}_n1.txt") n2=$(wc -l < "$RAW/prof_anchor_${TAG}_n2.txt")"

# G4 evidence: server-reported prompt_tokens per iteration (raw from probe JSON).
if [ -f "$RAW/temp_${TAG}.json" ]; then
  python3 -c "
import json
d = json.load(open('$RAW/temp_${TAG}.json'))
print('prompt_tokens per scored iteration:', [r['prompt_tokens'] for r in d['iterations']])
print('achieved_prompt_tokens:', d['achieved_prompt_tokens'])
print('temperature in config:', d['config'].get('temperature'))
" >> "$RAW/temp_arm_driver.log" 2>&1
fi

log "########## TEMP ARM COMPLETE ##########"
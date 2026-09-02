#!/usr/bin/env bash
# Matched-depth entropy A/B: same ~89.4K context, ONLY prompt entropy differs.
# Word counts pre-calibrated OFFLINE against the real tokenizer (free) so no
# server probe requests are burned:
#   repetitive: 75,000 words -> 89,404 tokens
#   natural   : 65,646 words -> 89,294 tokens   (-0.13% vs repetitive)
# READ-ONLY w.r.t. cluster config. No relaunch, no config change.
set -uo pipefail
OUT=/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901
RAW=$OUT/raw
N1=macstudio-m4-1
N2=macstudio-m4-2
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RAW/entropy_driver.log"; }
logbytes(){ ssh -o ConnectTimeout=10 "$1" 'wc -c < ~/exo.log' 2>/dev/null | tr -d ' '; }

idle_wait(){
  for i in $(seq 1 60); do
    r1=$(curl -s --max-time 10 "http://192.168.86.201:52415/metrics" | awk '/^exo_gpu_usage_ratio/{print $2}')
    r2=$(curl -s --max-time 10 "http://192.168.86.202:52415/metrics" | awk '/^exo_gpu_usage_ratio/{print $2}')
    ok=$(python3 -c "
try:
    print('1' if (float('${r1:-1}')<0.10 and float('${r2:-1}')<0.10) else '0')
except Exception:
    print('0')")
    [ "$ok" = "1" ] && { log "idle OK (n1=$r1 n2=$r2)"; return 0; }
    log "waiting for idle (n1=$r1 n2=$r2)"; sleep 10
  done
  log "WARN idle timeout; proceeding"; return 0
}

run_mode(){
  local mode="$1" words="$2"
  log "=== MODE $mode (words=$words) ==="
  idle_wait
  local b1 b2 a1 a2
  b1=$(logbytes $N1); b2=$(logbytes $N2)
  log "offsets before n1=$b1 n2=$b2"
  cd /tmp || exit 1
  PYTHONPATH=/Users/adam.durham/repos/exo/tools/src \
  /Users/adam.durham/repos/exo/.venv/bin/python \
    "$OUT/entropy_probe.py" \
    --mode "$mode" --fixed-words "$words" \
    --iterations 3 --warmup 1 --max-tokens 256 --timeout 3600 \
    --json-out "$RAW/entropy_${mode}.json" \
    > "$RAW/entropy_${mode}.log" 2>&1
  log "rc=$? for $mode"
  a1=$(logbytes $N1); a2=$(logbytes $N2)
  log "offsets after n1=$a1 n2=$a2"
  ssh -o ConnectTimeout=15 $N1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -a 'MTP-PROF'" > "$RAW/prof_ent_${mode}_n1.txt" 2>/dev/null
  ssh -o ConnectTimeout=15 $N2 "tail -c +$((b2+1)) ~/exo.log | head -c $((a2-b2)) | grep -a 'MTP-PROF'" > "$RAW/prof_ent_${mode}_n2.txt" 2>/dev/null
  log "prof lines n1=$(wc -l < "$RAW/prof_ent_${mode}_n1.txt") n2=$(wc -l < "$RAW/prof_ent_${mode}_n2.txt")"
  log "=== MODE $mode COMPLETE ==="
}

log "########## ENTROPY A/B START ##########"
run_mode repetitive 75000
run_mode natural 65646
log "########## ENTROPY A/B COMPLETE ##########"

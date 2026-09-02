#!/usr/bin/env bash
# Context-depth scan driver — verify-decomposition-20260901
# READ-ONLY w.r.t. cluster config: only fires API requests + reads logs.
# NO relaunch, NO config change, NO kill.
set -uo pipefail

OUT=/Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901
RAW=$OUT/raw
MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
HOST=192.168.86.201
PORT=52415
N1=macstudio-m4-1
N2=macstudio-m4-2

mkdir -p "$RAW"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RAW/driver.log"; }

logbytes(){ ssh -o ConnectTimeout=10 "$1" 'wc -c < ~/exo.log' 2>/dev/null | tr -d ' '; }

idle_wait(){
  # Block until both nodes report gpu_usage_ratio < 0.10 (max ~10 min)
  for i in $(seq 1 60); do
    r1=$(curl -s --max-time 10 "http://192.168.86.201:$PORT/metrics" | awk -F' ' '/^exo_gpu_usage_ratio/{print $2}')
    r2=$(curl -s --max-time 10 "http://192.168.86.202:$PORT/metrics" | awk -F' ' '/^exo_gpu_usage_ratio/{print $2}')
    ok=$(python3 -c "
try:
    a=float('${r1:-1}'); b=float('${r2:-1}')
    print('1' if (a<0.10 and b<0.10) else '0')
except Exception:
    print('0')")
    if [ "$ok" = "1" ]; then log "idle OK (n1=$r1 n2=$r2)"; return 0; fi
    log "waiting for idle (n1=$r1 n2=$r2)"; sleep 10
  done
  log "WARN: idle wait timed out; proceeding anyway"; return 0
}

run_depth(){
  local tag="$1" words="$2"
  log "=== DEPTH $tag : --prompt-words $words ==="
  idle_wait

  local b1 b2 a1 a2
  b1=$(logbytes $N1); b2=$(logbytes $N2)
  echo "{\"tag\":\"$tag\",\"words\":$words,\"n1_before\":$b1,\"n2_before\":$b2,\"t_before\":\"$(date -u +%FT%TZ)\"}" > "$RAW/marker_${tag}_before.json"
  log "log offsets before: n1=$b1 n2=$b2"

  cd /tmp || exit 1
  PYTHONPATH=/Users/adam.durham/repos/exo/tools/src \
  /Users/adam.durham/repos/exo/.venv/bin/python \
    /Users/adam.durham/repos/exo/bench/concurrent_bench.py \
    --model "$MODEL" --host "$HOST" --port "$PORT" \
    --concurrency 1 --iterations 5 --warmup 1 \
    --max-tokens 256 --prompt-words "$words" \
    --timeout 3600 \
    --label "depth_$tag" \
    --json-out "$RAW/bench_${tag}.json" \
    > "$RAW/bench_${tag}.log" 2>&1
  local rc=$?
  log "bench rc=$rc for $tag"

  a1=$(logbytes $N1); a2=$(logbytes $N2)
  echo "{\"tag\":\"$tag\",\"words\":$words,\"n1_after\":$a1,\"n2_after\":$a2,\"t_after\":\"$(date -u +%FT%TZ)\",\"rc\":$rc}" > "$RAW/marker_${tag}_after.json"
  log "log offsets after: n1=$a1 n2=$a2"

  # Harvest ONLY the byte-window written during this depth, MTP-PROF lines only.
  ssh -o ConnectTimeout=15 $N1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -a 'MTP-PROF'" > "$RAW/prof_${tag}_n1.txt" 2>/dev/null
  ssh -o ConnectTimeout=15 $N2 "tail -c +$((b2+1)) ~/exo.log | head -c $((a2-b2)) | grep -a 'MTP-PROF'" > "$RAW/prof_${tag}_n2.txt" 2>/dev/null
  log "harvested prof lines: n1=$(wc -l < "$RAW/prof_${tag}_n1.txt") n2=$(wc -l < "$RAW/prof_${tag}_n2.txt")"

  # Acceptance / spec diagnostics in the same window
  ssh -o ConnectTimeout=15 $N1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -aE 'accept|ACCEPT|gamma|spec' | head -400" > "$RAW/accept_${tag}_n1.txt" 2>/dev/null
  log "=== DEPTH $tag COMPLETE ==="
}

log "########## SCAN START ##########"
run_depth "089k" 75000
run_depth "150k" 125828
run_depth "250k" 209713
log "########## SCAN COMPLETE ##########"

#!/bin/bash
# Soak 3 — r1M re-proof on the ladder build. TRUE deltas now: each rung appends
# to the SAME conversation, so (with the ladder) rung N refeeds only
# (target_N - target_{N-1}) rows plus the margin-rung tail. Pre-ladder rungs
# were FULL re-prefills (r500 500K rows, r750 750K rows, r1m 1.04M rows).
#
# Expectations on the ladder build (measured rates: ~110 tok/s shallow, ~75 mid,
# ~56 deep):
#   r160: cold ~160K rows -> ~25 min
#   r500: delta ~340K rows -> ~55 min
#   r750: delta ~250K rows -> ~60 min
#   r1m:  delta ~290K rows -> ~65 min
#   over: refusal, seconds
# Client max-time 10800 per rung; on cap, wait for server idle then re-capture
# via the exact-repeat turn (fast).
set -u
SCRATCH=/Users/adam.durham/.hermes/cache/scratch
API=http://macstudio-m4-1.tail19c543.ts.net:52415
MODEL="dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
REC="$SCRATCH/soak3_record.txt"
: > "$REC"

sentence="The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
python3 - "$SCRATCH/soak3_prompts.json" "$MODEL" <<'PY'
import json, sys
s = "The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
base = s * 8888  # ~160K tok
def pad(tok): return s * int(tok * 5.111 / 92)
rungs = {
  "r500": base + " done" + pad(500000-160000) + " Now reply with exactly: done",
  "r750": base + " done" + pad(750000-160000) + " Now reply with exactly: done",
  "r1m":  base + " done" + pad(1040000-160000) + " Now reply with exactly: done",
  "over": base + " done" + pad(1100000-160000) + " Now reply with exactly: done",
}
json.dump(rungs, open(sys.argv[1], "w"))
for k, v in rungs.items(): print(k, "chars:", len(v), "~tok:", int(len(v)/5.111))
PY

snapshot() {
  local label="$1" ts; ts=$(date '+%T')
  local ram; ram=$(curl -s --max-time 10 "$API/state" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(' '.join('%s=%.1fG' % (nid[:8], (nv.get('ramAvailable') or {}).get('inBytes',0)/2**30) for nid,nv in (d.get('nodeMemory') or {}).items()))
except Exception: print('n/a')
" 2>/dev/null)
  local fp1 fp2
  fp1=$(ssh -o ConnectTimeout=6 -o BatchMode=yes macstudio-m4-1 'P=$(ps axo pid,command | grep "repos/exo/.venv/bin/python" | grep spawn_main | grep -v grep | awk "{print \$1}" | head -1); [ -n "$P" ] && footprint "$P" 2>/dev/null | grep -oE "phys_footprint: [0-9]+ GB" | head -1' 2>/dev/null)
  fp2=$(ssh -o ConnectTimeout=6 -o BatchMode=yes macstudio-m4-2 'P=$(ps axo pid,command | grep "repos/exo/.venv/bin/python" | grep spawn_main | grep -v grep | awk "{print \$1}" | head -1); [ -n "$P" ] && footprint "$P" 2>/dev/null | grep -oE "phys_footprint: [0-9]+ GB" | head -1' 2>/dev/null)
  echo "SNAP|$label|$ts|ram $ram|m4-1 [$fp1] m4-2 [$fp2]" | tee -a "$REC"
}

server_running() {
  curl -s --max-time 12 "$API/state" 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('unknown'); raise SystemExit
print('running' if any(v.get('taskStatus')=='Running' for tv in (d.get('tasks') or {}).values() for k,v in (tv or {}).items() if k=='TextGeneration') else 'idle')
" 2>/dev/null
}

run_rung() { # $1=name
  local name="$1"
  python3 - "$SCRATCH/soak3_prompts.json" "$name" "$SCRATCH/soak3_$name.json" "$MODEL" <<'PY'
import json, sys
rungs = json.load(open(sys.argv[1]))
json.dump({"model": sys.argv[4], "messages": [{"role":"user","content": rungs[sys.argv[2]]}],
           "max_tokens": 64, "reasoning_effort": "low", "stream": False}, open(sys.argv[3], "w"))
PY
  echo "=== $(date '+%T') RUNG $name launch ===" | tee -a "$REC"
  ( curl -sS --max-time 10800 -X POST "$API/v1/chat/completions" -H "Content-Type: application/json" \
      --data-binary @"$SCRATCH/soak3_$name.json" -o "$SCRATCH/soak3_$name.resp.json" \
      -w "HTTP %{http_code} total %{time_total}s\n" > "$SCRATCH/soak3_$name.meta" 2>&1
    echo "curl exit: $?" >> "$SCRATCH/soak3_$name.meta" ) &
  local pid=$! i=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 150; i=$((i+1))
    [ $((i % 2)) -eq 0 ] && snapshot "$name/s$i"
  done
  wait "$pid"
  echo "=== $(date '+%T') RUNG $name curl returned ===" | tee -a "$REC"
  cat "$SCRATCH/soak3_$name.meta" | tee -a "$REC"
  local s; s=$(server_running)
  if grep -q "curl exit: 28" "$SCRATCH/soak3_$name.meta" 2>/dev/null && [ "$s" = "running" ]; then
    echo "--- client capped; waiting for server prefill ($(date '+%T'))" | tee -a "$REC"
    for j in $(seq 1 40); do
      sleep 120; s=$(server_running)
      [ $((j % 6)) -eq 0 ] && snapshot "$name/wait$j"
      [ "$s" = "idle" ] && break
    done
    echo "--- server $s; firing exact-repeat capture ($(date '+%T'))" | tee -a "$REC"
    ( curl -sS --max-time 1800 -X POST "$API/v1/chat/completions" -H "Content-Type: application/json" \
        --data-binary @"$SCRATCH/soak3_$name.json" -o "$SCRATCH/soak3_$name.capture.json" \
        -w "HTTP %{http_code} total %{time_total}s\n" > "$SCRATCH/soak3_$name.capture.meta" 2>&1
      echo "curl exit: $?" >> "$SCRATCH/soak3_$name.capture.meta" ) &
    local cpid=$! k=0
    while kill -0 "$cpid" 2>/dev/null; do sleep 60; k=$((k+1)); [ $((k % 4)) -eq 0 ] && snapshot "$name/cap$k"; done
    wait "$cpid"
    cat "$SCRATCH/soak3_$name.capture.meta" | tee -a "$REC"
    parse() {
      python3 - "$1" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("PARSE FAIL:", e); raise SystemExit
if "error" in d: print("ERROR:", json.dumps(d["error"])[:300]); raise SystemExit
ch = (d.get("choices") or [{}])[0]; m = ch.get("message") or {}; u = d.get("usage") or {}
print("RESULT finish:", ch.get("finish_reason"), "| prompt_tokens:", u.get("prompt_tokens"),
      "| completion:", u.get("completion_tokens"), "| content:", repr((m.get("content") or "")[:60]))
PY
    }
    parse "$SCRATCH/soak3_$name.capture.json" | tee -a "$REC"
  else
    python3 - "$SCRATCH/soak3_$name.resp.json" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("PARSE FAIL:", e); raise SystemExit
if "error" in d: print("ERROR:", json.dumps(d["error"])[:300]); raise SystemExit
ch = (d.get("choices") or [{}])[0]; m = ch.get("message") or {}; u = d.get("usage") or {}
print("RESULT finish:", ch.get("finish_reason"), "| prompt_tokens:", u.get("prompt_tokens"),
      "| completion:", u.get("completion_tokens"), "| content:", repr((m.get("content") or "")[:60]))
PY
    python3 -c "
import json
try:
    d=json.load(open('$SCRATCH/soak3_$name.resp.json'))
    ch=(d.get('choices') or [{}])[0]; m=ch.get('message') or {}; u=d.get('usage') or {}
    print('RESULT finish:', ch.get('finish_reason'), '| prompt_tokens:', u.get('prompt_tokens'), '| completion:', u.get('completion_tokens'), '| content:', repr((m.get('content') or '')[:60]))
except Exception as e: print('PARSE FAIL:', e)
" | tee -a "$REC"
  fi
  snapshot "$name/done"
}

# r160 build first (cold), then the delta rungs
echo "=== $(date '+%T') SOAK3 START (ladder build) ===" | tee -a "$REC"
python3 - "$SCRATCH/soak3_prompts.json" "$MODEL" <<'PY'
import json, sys
s = "The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
base = s * 8888
rungs = json.load(open(sys.argv[1]))
rungs["r160"] = base + " Now reply with exactly: done"
json.dump(rungs, open(sys.argv[1], "w"))
print("r160 chars:", len(rungs["r160"]))
PY
run_rung r160
run_rung r500
run_rung r750
run_rung r1m
run_rung over
echo "=== SOAK3 COMPLETE $(date '+%T') ===" | tee -a "$REC"

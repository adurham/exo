#!/bin/bash
# r500 deep-validation soak on the FINAL next12 build (bf16 default-ON).
# r160 cold (ladder check) -> r500 delta. ~42min rung expected.
set -u
SCRATCH=/Users/adam.durham/.hermes/cache/scratch
API=http://macstudio-m4-1.tail19c543.ts.net:52415
MODEL="dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
REC="$SCRATCH/soak_next12_record.txt"
: > "$REC"
s="The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
python3 - "$SCRATCH/soak12_prompts.json" <<'PY'
import json, sys
s = "The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
base = s * 8888
def pad(tok): return s * int(tok * 5.111 / len(s))
rungs = {
  "r160": base + " Now reply with exactly: done",
  "r500": base + " done" + pad(500000-160000) + " Now reply with exactly: done",
}
json.dump(rungs, open(sys.argv[1], "w"))
for k, v in rungs.items(): print(k, "~tok:", int(len(v)/5.111))
PY
echo "=== SOAK12 START $(date '+%F %T') ===" | tee -a "$REC"
for RUNG in r160 r500; do
  echo "=== $(date '+%T') RUNG $RUNG launch ===" | tee -a "$REC"
  # Payload -> file, then curl --data @file. NEVER put a 500K-token prompt on the
  # command line: the shell substitution dies with E2BIG ("Argument list too
  # long", curl exit 126) -- measured 2026-10-07 on the r500 rung.
  python3 - "$SCRATCH/soak12_prompts.json" "$RUNG" "$MODEL" "$SCRATCH" <<'PY'
import json, sys
prompts_path, rung, model, scratch = sys.argv[1:5]
rungs = json.load(open(prompts_path))
payload = {"model": model,
           "messages": [{"role": "user", "content": rungs[rung]}],
           "max_tokens": 32, "reasoning_effort": "low", "stream": False}
out = f"{scratch}/soak12_{rung}_payload.json"
json.dump(payload, open(out, "w"))
print(f"payload written: {out} (~{len(rungs[rung])//5.111} tok)")
PY
  t0=$(date +%s)
  curl -s --max-time 10800 -X POST "$API/v1/chat/completions" -H "Content-Type: application/json" \
    --data @"$SCRATCH/soak12_$RUNG_payload.json" -o "$SCRATCH/soak12_$RUNG.resp.json"
  rc=$?
  t1=$(date +%s)
  echo "HTTP curl exit $rc total $((t1-t0))s" | tee -a "$REC"
  python3 - "$SCRATCH/soak12_$RUNG.resp.json" <<'PY' | tee -a "$REC"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    u = d.get("usage") or {}
    print("RESULT prompt:", u.get("prompt_tokens"), "completion:", u.get("completion_tokens"),
          "finish:", (d.get("choices") or [{}])[0].get("finish_reason"))
except Exception as e:
    print("parse fail:", e)
PY
done
echo "=== SOAK12 COMPLETE $(date '+%F %T') ===" | tee -a "$REC"

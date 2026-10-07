#!/bin/bash
# Deep soak on next13 (consumer-skip ON): full ladder r160 -> r500 -> r750 -> r1m.
# One conversation with a fresh salt; each rung appends so the ladder makes it a DELTA
# (verify via the turn reuse line). Payload-to-file pattern (E2BIG-safe).
set -u
SCRATCH=/Users/adam.durham/.hermes/cache/scratch
API=http://macstudio-m4-1.tail19c543.ts.net:52415
MODEL="dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
REC="$SCRATCH/soak_next13_record.txt"
: > "$REC"
SALT=$(python3 -c "import secrets; print(secrets.token_hex(8))")
echo "salt: $SALT"

python3 - "$SCRATCH/soak13_prompts.json" "$SALT" <<'PY'
import json, sys
path, salt = sys.argv[1], sys.argv[2]
s = "The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
base = s * 8888  # ~160K tok
def pad(tok): return s * int(tok * 5.111 / len(s))
# All rungs share one prefix (r160 + " TURN<salt> done") -> LCP matching + ladder = deltas.
prefix = base + f" TURN{salt} done"
rungs = {
  "r160": base + f" TURN{salt} Now reply with exactly: done",
  "r500": prefix + pad(500000-160000) + " Now reply with exactly: done",
  "r750": prefix + pad(750000-160000) + " Now reply with exactly: done",
  "r1m":  prefix + pad(1040000-160000) + " Now reply with exactly: done",
}
json.dump(rungs, open(path, "w"))
for k, v in rungs.items(): print(k, "~tok:", int(len(v)/5.111))
PY

echo "=== SOAK13 START $(date '+%F %T') ===" | tee -a "$REC"
for RUNG in r160 r500 r750 r1m; do
  echo "=== $(date '+%T') RUNG $RUNG launch ===" | tee -a "$REC"
  python3 - "$SCRATCH/soak13_prompts.json" "$RUNG" "$MODEL" "$SCRATCH" <<'PY'
import json, sys
prompts_path, rung, model, scratch = sys.argv[1:5]
rungs = json.load(open(prompts_path))
payload = {"model": model,
           "messages": [{"role": "user", "content": rungs[rung]}],
           "max_tokens": 32, "reasoning_effort": "low", "stream": False}
out = f"{scratch}/soak13_{rung}_payload.json"
json.dump(payload, open(out, "w"))
print(f"payload written: {out} (~{len(rungs[rung])//5.111} tok)")
PY
  t0=$(date +%s)
  curl -s --max-time 14400 -X POST "$API/v1/chat/completions" -H "Content-Type: application/json" \
    --data @"$SCRATCH/soak13_$RUNG_payload.json" -o "$SCRATCH/soak13_$RUNG.resp.json"
  rc=$?
  t1=$(date +%s)
  echo "HTTP curl exit $rc total $((t1-t0))s" | tee -a "$REC"
  python3 - "$SCRATCH/soak13_$RUNG.resp.json" <<'PY' | tee -a "$REC"
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
echo "=== SOAK13 COMPLETE $(date '+%F %T') ===" | tee -a "$REC"

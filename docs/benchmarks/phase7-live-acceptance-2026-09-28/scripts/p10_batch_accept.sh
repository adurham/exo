#!/bin/bash
# p10: bigger live-acceptance sample from the PRODUCTION cluster (no relaunch).
# Drives 3 sizable completions and measures the master's MTP counters delta.
set -u
API=http://100.91.246.26:52415
S=/home/hermes/.hermes/cache/scratch/exl3patch

scrape() {
  curl -s -m 25 "$API/metrics" | grep -E '^exo_mtp_(cycles|accepted_drafts)_total'
}

echo "=== SCRAPE #0 (before batch) $(date '+%T') ==="
scrape | tee "$S/p10_batch_before.txt"
echo

for i in 1 2 3; do
  echo "--- request $i $(date '+%T') ---"
  cat > "$S/p10_req_$i.json" <<JSON
{"model":"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
 "messages":[{"role":"user","content":"Write a long, detailed technical essay (about eight paragraphs) about the history and engineering of distributed inference systems for large language models. Cover pipeline parallelism, tensor parallelism, expert routing, and speculative decoding. Be specific and thorough."}],
 "max_tokens":600,
 "temperature":0.7}
JSON
  curl -s -m 300 -o "$S/p10_batch_gen$i.json" \
    -w 'http=%{http_code} time=%{time_total}s\n' \
    "$API/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"$S/p10_req_$i.json"
done

echo
echo "=== SCRAPE #1 (after batch) $(date '+%T') ==="
scrape | tee "$S/p10_batch_after.txt"

echo
echo "=== ANALYSIS ==="
python3 - <<'PY'
import re, pathlib, json
S = pathlib.Path("/home/hermes/.hermes/cache/scratch/exl3patch")

def parse(p):
    d = {}
    for line in p.read_text().splitlines():
        m = re.match(r'^(exo_mtp_\w+)\{[^}]*\}\s+([\d.e+]+)$', line.strip())
        if m:
            d[m.group(1)] = float(m.group(2))
    return d

a = parse(S / "p10_batch_before.txt")
b = parse(S / "p10_batch_after.txt")
c0 = a.get("exo_mtp_cycles_total"); c1 = b.get("exo_mtp_cycles_total")
d0 = a.get("exo_mtp_accepted_drafts_total"); d1 = b.get("exo_mtp_accepted_drafts_total")
dc = c1 - c0; da = d1 - d0
print(f"cumulative at start : cycles={c0} accepted={d0}  (all-time mean {d0/c0:.4f}/cycle)")
print(f"cumulative at end   : cycles={c1} accepted={d1}")
print(f"BATCH sample        : cycles={dc} accepted={da}")
if dc > 0:
    m = da/dc
    print(f"mean accepted/cycle : {m:.4f} of gamma=3  ({100*m/3:.1f}% of drafts)")
    print(f"tokens/cycle        : {1+m:.4f}  (1 anchor + accepted)")
    # completions
tot = 0; ttime = 0.0
for i in (1,2,3):
    p = S / f"p10_batch_gen{i}.json"
    try:
        d = json.loads(p.read_text())
        u = d.get("usage", {})
        ct = u.get("completion_tokens", 0)
        tot += ct
    except Exception as e:
        print(f"req{i} parse err: {e}")
print(f"completion tokens total: {tot}")
if tot and dc > 0:
    print(f"observed tokens/cycle from actual: {tot/dc:.4f}")
PY
echo "P10-BATCH-DONE"

#!/bin/bash
# p10: greedy sample (temp=0) of live MTP acceptance on production cluster.
# temp=0 matches the port's spec-loop measurement conditions (greedy).
set -u
API=http://100.91.246.26:52415
S=/home/hermes/.hermes/cache/scratch/exl3patch

scrape() {
  curl -s -m 25 "$API/metrics" | grep -E '^exo_mtp_(cycles|accepted_drafts)_total'
}

echo "=== SCRAPE #0 (before greedy batch) $(date '+%T') ==="
scrape | tee "$S/p10_greedy_before.txt"
echo

for i in 1 2; do
  echo "--- greedy request $i $(date '+%T') ---"
  cat > "$S/p10_greq_$i.json" <<JSON
{"model":"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
 "messages":[{"role":"user","content":"Describe the complete process by which a modern compiler turns source code into an optimized executable, covering lexing, parsing, IR construction, optimization passes, register allocation, and code generation. Be detailed and technical."}],
 "max_tokens":300,
 "temperature":0}
JSON
  curl -s -m 300 -o "$S/p10_greedy_gen$i.json" \
    -w 'http=%{http_code} time=%{time_total}s\n' \
    "$API/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"$S/p10_greq_$i.json"
done

echo
echo "=== SCRAPE #1 (after greedy batch) $(date '+%T') ==="
scrape | tee "$S/p10_greedy_after.txt"

echo
echo "=== ANALYSIS (greedy, temp=0) ==="
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

a = parse(S / "p10_greedy_before.txt")
b = parse(S / "p10_greedy_after.txt")
c0 = a.get("exo_mtp_cycles_total"); c1 = b.get("exo_mtp_cycles_total")
d0 = a.get("exo_mtp_accepted_drafts_total"); d1 = b.get("exo_mtp_accepted_drafts_total")
dc = c1 - c0; da = d1 - d0
print(f"cycles delta = {dc}, accepted delta = {da}")
if dc > 0:
    print(f"mean accepted/cycle = {da/dc:.4f} (gamma=3)  per-draft = {100*da/dc/3:.1f}%")
    print(f"tokens/cycle = {1 + da/dc:.4f}")
tot = 0
for i in (1,2):
    try:
        d = json.loads((S / f"p10_greedy_gen{i}.json").read_text())
        tot += d.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        print(f"req{i}: {e}")
print(f"completion tokens = {tot}; identity check: cycles+accepted = {dc+da} (should equal tokens)")
print(f"all-time mean accepted/cycle = {d1/c1:.4f}")
PY
echo "P10-GREEDY-DONE"

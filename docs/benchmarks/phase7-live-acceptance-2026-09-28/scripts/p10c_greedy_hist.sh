#!/bin/bash
# p10c: greedy (temp=0) acceptance sample + full histogram dump.
set -u
API=http://100.91.246.26:52415
S=/home/hermes/.hermes/cache/scratch/exl3patch

echo "=== FULL MTP FAMILY + BUCKETS (before) $(date '+%T') ==="
curl -s -m 25 "$API/metrics" | grep -E '^exo_mtp' | tee "$S/p10_full_before.txt"
echo

for i in 1 2; do
  echo "--- greedy req $i $(date '+%T') ---"
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
echo "=== FULL MTP FAMILY + BUCKETS (after) $(date '+%T') ==="
curl -s -m 25 "$API/metrics" | grep -E '^exo_mtp' | tee "$S/p10_full_after.txt"

echo
echo "=== ANALYSIS ==="
python3 - <<'PY'
import re, pathlib, json
S = pathlib.Path("/home/hermes/.hermes/cache/scratch/exl3patch")

def parse(p):
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r'^(exo_mtp_\w+)\{([^}]*)\}\s+([\d.eE+]+)$', line.strip())
        if m:
            labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
            key = (m.group(1), tuple(sorted(labels.items())))
            out[key] = float(m.group(3))
    return out

a = parse(S / "p10_full_before.txt")
b = parse(S / "p10_full_after.txt")

def tot(d, name):
    return sum(v for (n, _), v in d.items() if n == name)

c0, c1 = tot(a, "exo_mtp_cycles_total"), tot(b, "exo_mtp_cycles_total")
d0, d1 = tot(a, "exo_mtp_accepted_drafts_total"), tot(b, "exo_mtp_accepted_drafts_total")
dc, da = c1 - c0, d1 - d0
print(f"GREEDY SAMPLE: cycles={dc} accepted={da} -> {da/dc:.4f}/cycle ({100*da/dc/3:.1f}% of gamma=3), tokens/cycle={1+da/dc:.4f}" if dc else "no movement")
tot_tok = 0
for i in (1, 2):
    try:
        d = json.loads((S / f"p10_greedy_gen{i}.json").read_text())
        tot_tok += d.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        print(f"req{i}: {e}")
print(f"completion tokens={tot_tok}; cycles+accepted={dc+da} (identity check)")
print(f"ALL-TIME since worker start: cycles={c1} accepted={d1} -> {d1/c1:.4f}/cycle")
print()
print("BUCKET DISTRIBUTION (after, master):")
rows = []
for (n, lbl), v in b.items():
    if n == "exo_mtp_acceptance_bucket_total":
        rows.append((dict(lbl).get("accepted"), v))
rows.sort(key=lambda r: int(r[0]))
for acc, v in rows:
    print(f"  accepted={acc}: {v:,.0f}")
PY
echo "P10C-DONE"

#!/bin/bash
# p10 live-acceptance sample: scrape -> drive real decode -> scrape -> delta.
# Reads the ALREADY-LIVE Prometheus MTP counters on the elected master.
set -u
API=http://100.91.246.26:52415
S=/home/hermes/.hermes/cache/scratch/exl3patch

scrape() {
  curl -s -m 25 "$API/metrics" | grep -E '^exo_mtp_(cycles|accepted_drafts)_total'
}

echo "=== SCRAPE #0 (before traffic) $(date '+%T') ==="
scrape | tee "$S/p10_metrics_before.txt"

cat > "$S/p10_req.json" <<'JSON'
{"model":"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
 "messages":[{"role":"user","content":"Explain in about five paragraphs why the sky is blue, covering Rayleigh scattering, the solar spectrum, and human color perception. Be thorough and technical."}],
 "max_tokens":320,
 "temperature":0}
JSON

echo
echo "=== DRIVING ONE DECODE $(date '+%T') ==="
curl -s -m 300 -o "$S/p10_gen1.json" \
  -w 'http=%{http_code} time=%{time_total}s\n' \
  "$API/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @"$S/p10_req.json"

echo
echo "=== SCRAPE #1 (after traffic) $(date '+%T') ==="
scrape | tee "$S/p10_metrics_after.txt"

echo
echo "=== DELTA ==="
python3 - <<'PY'
import re, pathlib
S = pathlib.Path("/home/hermes/.hermes/cache/scratch/exl3patch")

def parse(p):
    d = {}
    for line in p.read_text().splitlines():
        m = re.match(r'^(exo_mtp_\w+)\{[^}]*\}\s+([\d.e+]+)$', line.strip())
        if m:
            d[m.group(1)] = float(m.group(2))
    return d

a = parse(S / "p10_metrics_before.txt")
b = parse(S / "p10_metrics_after.txt")
c0 = a.get("exo_mtp_cycles_total"); c1 = b.get("exo_mtp_cycles_total")
d0 = a.get("exo_mtp_accepted_drafts_total"); d1 = b.get("exo_mtp_accepted_drafts_total")
print(f"cycles      : {c0} -> {c1}  (delta {c1-c0})")
print(f"accepted    : {d0} -> {d1}  (delta {d1-d0})")
if c1 > c0:
    print(f"MEAN ACCEPTANCE per cycle this sample: {(d1-d0)/(c1-c0):.4f}")
    print(f"tokens/cycle (1 anchor + accepted)  : {1 + (d1-d0)/(c1-c0):.4f}")
else:
    print("COUNTERS DID NOT MOVE — need more/other traffic, or a relaunch.")
PY

echo
echo "=== completion sanity ==="
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/home/hermes/.hermes/cache/scratch/exl3patch/p10_gen1.json")
try:
    d = json.loads(p.read_text())
    print("usage:", d.get("usage"))
    msg = d["choices"][0]["message"]
    c = msg.get("content") or ""
    print("content chars:", len(c))
    print("head:", c[:200].replace("\n", " "))
except Exception as e:
    print("parse failed:", e)
    print(p.read_text()[:500])
PY
echo "P10-LIVE-ACCEPT-DONE"

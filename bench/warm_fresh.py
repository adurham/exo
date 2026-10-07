#!/usr/bin/env python3
"""Warm-fresh probe: three DISTINCT ~100K fresh feeds back-to-back on one process.

Purpose: the process already served fresh feeds at 22:00/22:06, so its 2048-row
chunk shapes may be warm. Measures whether "fresh 100K" cost drops on repeat
(all-warm) = one-time JIT/first-touch share, or stays flat = steady-state cost.

Each feed uses distinct repeated filler (never used before in this session) so
the session cache matches nothing -> true fresh prefill (reuse=0).
"""
import json, sys, time, urllib.request

API = "http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions"
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

FILLERS = {
    "A": "Alpha waves ripple through the silicon lattice as the morning sun climbs slowly over the eastern ridge. ",
    "B": "Pack my box with five dozen liquor jugs while the quiet river carries autumn leaves downstream. ",
    "C": "Seventeen crows settled along the fence wire, counting the slow freight cars rolling westward. ",
}
TARGET_CHARS = 511100  # ~100K tokens at 5.111 chars/tok (established probe scaling)

def feed(label, filler):
    n = int(TARGET_CHARS / len(filler))
    content = filler * n + " Now reply with exactly: done"
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                          "max_tokens": 24, "reasoning_effort": "low", "stream": False}).encode()
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read())
        el = time.time() - t0
        u = d.get("usage") or {}
        pt = u.get("prompt_tokens", 0)
        fr = (d.get("choices") or [{}])[0].get("finish_reason")
        print(f"[{label}] wall {el:.1f}s, prompt {pt} tok -> {pt/max(el,1):.1f} tok/s, finish={fr}", flush=True)
        return el, pt
    except Exception as e:
        print(f"[{label}] FAILED after {time.time()-t0:.0f}s: {str(e)[:200]}", flush=True)
        return None, 0

print(f"=== WARM-FRESH probe {time.strftime('%F %T')} ===", flush=True)
res = {}
for label in ("A", "B", "C"):
    res[label] = feed(label, FILLERS[label])
    time.sleep(5)
print(flush=True)
for label, (el, pt) in res.items():
    if el:
        print(f"[{label}] {pt/el:.1f} tok/s ({el:.1f}s)")
if res["A"][0] and res["B"][0]:
    a = res["A"][1] / res["A"][0]
    b = res["B"][1] / res["B"][0]
    print(f"COLD->WARM: A {a:.1f} -> B {b:.1f} tok/s = {b/a-1:+.1%}")
    if res["C"][0]:
        c = res["C"][1] / res["C"][0]
        print(f"B->C: {c:.1f} tok/s = {c/b-1:+.1%}")
print(f"=== done {time.strftime('%F %T')} ===")

#!/usr/bin/env python3
"""Consumer-skip A/B arm: fresh 100K + r160 cold -> r500 delta.

The consumer-skip win scales with offset (sizing: 3.9% @100K -> 14.8% @750K),
so the r500 delta is the measurement that matters. Comparators (next12, skip OFF):
  fresh 100K = 274.0 tok/s
  r500 delta = 340,237 rows / 2,378 s = 143.1 rows/s
Protocol: unique-salted fillers; the r500 rung extends the r160 conversation so the
ladder makes it a true delta (verify via the turn reuse line, refed < prompt).
"""
import json, secrets, sys, time, urllib.request

API = "http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions"
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "arm"
SP = "The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "


def chat(content, max_tokens=32, timeout=10800, tag="?"):
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                          "max_tokens": max_tokens, "reasoning_effort": "low", "stream": False}).encode()
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        el = time.time() - t0
        pt = (d.get("usage") or {}).get("prompt_tokens", 0)
        fr = (d.get("choices") or [{}])[0].get("finish_reason")
        print(f"[{tag}] wall {el:.1f}s, prompt {pt}, wire {pt/max(el,1):.1f} tok/s, finish={fr}", flush=True)
        return el, pt
    except Exception as e:
        print(f"[{tag}] FAILED after {time.time()-t0:.0f}s: {str(e)[:200]}", flush=True)
        return None, 0


print(f"=== CONSUMER-SKIP ARM [{LABEL}] {time.strftime('%F %T')} ===", flush=True)

# 1) fresh 100K (unique salt; like-for-like with the 274.0 comparator)
salt = secrets.token_hex(8)
fresh = f"SALT{salt} " + SP * int(100000 * 5.111 / len(SP)) + " Now reply with exactly: done"
fe, fp = chat(fresh, tag="fresh100K")
time.sleep(5)

# 2) r160 cold build
base = SP * 8888
r160 = base + f" TURN{salt} Now reply with exactly: done"
e1, p1 = chat(r160, tag="r160")
time.sleep(5)

# 3) r500 delta: extend the SAME conversation (prompt = r160 + pad + ask)
def pad(tok): return SP * int(tok * 5.111 / len(SP))
r500 = base + f" TURN{salt} done" + pad(500000 - 160000) + " Now reply with exactly: done"
e2, p2 = chat(r500, tag="r500")

print(flush=True)
print(f"[{LABEL}] fresh100K: {fp/max(fe,1):.1f} tok/s ({(fe or 0):.0f}s)   [cmp: 274.0 next12]")
print(f"[{LABEL}] r160: {p1/max(e1,1):.1f} tok/s ({(e1 or 0):.0f}s)   [cmp: 247.7 next12]")
if e2:
    rows = p2 - p1 if p2 > p1 else p2 - 160000
    print(f"[{LABEL}] r500 wire: {p2/max(e2,1):.1f} tok/s ({e2:.0f}s) over {p2} tokens")
    print(f"[{LABEL}] r500 refed-rows approx: {rows} -> {rows/max(e2,1):.1f} rows/s  [cmp: 143.1 next12]")
    print(f"        ^^ CONFIRM via the turn reuse line (prefill= vs prompt=)")
print(f"=== done {time.strftime('%F %T')} ===")

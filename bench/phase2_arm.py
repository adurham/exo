#!/usr/bin/env python3
"""Phase 2 arm v3: like-for-like fresh feeds (100K tokens, SP filler + salt prefix).

v1 bug: reused fillers matched parked sessions (88,064 rows reuse -> fake 1914 tok/s).
v2 bug: salt-mixed lorem filler tokenizes at ~3.97 chars/tok -> 128.8K-token feeds,
not comparable to the 263.3 @100K comparator.

v3: ONE salt token-prefix (kills LCP matching -> true fresh) + the ESTABLISHED
SP filler at its measured 5.111 chars/tok -> ~100K-token prompts, directly
comparable to the 263.3 / 257.8-266.8 baseline band.
"""
import json, secrets, sys, time, urllib.request

API = "http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions"
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "arm"
SP = "The quick brown fox jumps over the lazy dog while the cluster serves tokens at steady pace. "
CHARS = int(100000 * 5.111)

def feed(tag, repeat=None):
    salt = secrets.token_hex(8)
    if repeat is None:
        content = f"SALT{salt} " + SP * (CHARS // len(SP)) + f" Now reply with exactly: done"
    else:
        content = repeat + " done Now reply with exactly: done"
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                          "max_tokens": 24, "reasoning_effort": "low", "stream": False}).encode()
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.loads(r.read())
        el = time.time() - t0
        pt = (d.get("usage") or {}).get("prompt_tokens", 0)
        fr = (d.get("choices") or [{}])[0].get("finish_reason")
        print(f"[{tag}] wall {el:.1f}s, prompt {pt}, {pt/max(el,1):.1f} tok/s, finish={fr}", flush=True)
        return el, pt, content
    except Exception as e:
        print(f"[{tag}] FAILED after {time.time()-t0:.0f}s: {str(e)[:200]}", flush=True)
        return None, 0, content

print(f"=== {LABEL} ARM v3 {time.strftime('%F %T')} ===", flush=True)
feed("cold")                 # discard
time.sleep(3)
r1 = feed("f1")
time.sleep(3)
r2 = feed("f2")
time.sleep(3)
rd = feed("delta", repeat=r2[2]) if r2[2] else (None, 0, None)
print(flush=True)
fresh = [r for r in (r1, r2) if r[0]]
if fresh:
    rates = sorted(pt / el for el, pt, _ in fresh)
    med = rates[len(rates)//2] if len(rates) % 2 else sum(rates)/len(rates)
    for el, pt, _ in fresh:
        print(f"fresh: {pt/el:.1f} tok/s ({el:.0f}s, {pt} tok)")
    print(f"FRESH MEDIAN: {med:.1f} tok/s  (comparator: next9 263.3 @100K; band +-3%)")
if rd[0]:
    print(f"DELTA: {rd[1]/rd[0]:.1f} rows/s wire-rate")
print(f"=== done {time.strftime('%F %T')} ===")

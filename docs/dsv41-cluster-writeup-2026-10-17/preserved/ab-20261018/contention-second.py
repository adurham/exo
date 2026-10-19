#!/usr/bin/env python3
"""Second client for the 300K contention test.

Wait for the 300K row's FIRST content token in its line-stamped SSE, then fire a
long streaming generation so it is decoding while the big row decodes. Records
per-frame timestamps for the second stream so its own rate can be reported.

Usage: contention-second.py <sse_path> <out_json>
"""
import json
import os
import sys
import time
import urllib.request

URL = "http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions"
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

SSE = sys.argv[1]
OUT = sys.argv[2]

PROMPT = ("Write a detailed technical explanation of how Thunderbolt 5 RDMA works between "
          "two Apple Silicon Macs: the PCIe tunnelling, the UC completion model, queue pairs, "
          "and what fails when a completion is lost. Be concrete and go long.")


def first_token_seen(path):
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if '"delta"' not in line:
                    continue
                if '"content":"' in line or '"reasoning_content":"' in line:
                    return True
    except FileNotFoundError:
        return False
    return False


t_wait0 = time.time()
while not first_token_seen(SSE):
    if time.time() - t_wait0 > 5400:
        print(json.dumps({"error": "timed out waiting for the 300K row's first token"}))
        sys.exit(1)
    time.sleep(0.05)

t_fire = time.time()
body = {"model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4000, "temperature": 0.7, "stream": True, "enable_thinking": False}
req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})

stamps, chars, errs = [], 0, []
first = None
with urllib.request.urlopen(req, timeout=1800) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            d = json.loads(payload)
        except Exception:
            continue
        if "error" in d:
            errs.append(str(d["error"])[:200])
            continue
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            txt = delta.get("content") or delta.get("reasoning_content") or ""
            if txt:
                now = time.time()
                if first is None:
                    first = now
                stamps.append(now)
                chars += len(txt)
            if ch.get("finish_reason"):
                break

res = {"fired_at": t_fire, "first_token": first,
       "ttft_s": (first - t_fire) if first else None,
       "frames": len(stamps), "chars": chars,
       "decode_s": (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
       "tok_s": (len(stamps) / (stamps[-1] - stamps[0])) if len(stamps) > 1 else None,
       "errors": errs}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res))

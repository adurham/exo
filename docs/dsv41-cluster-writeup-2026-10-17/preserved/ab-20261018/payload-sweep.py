#!/usr/bin/env python3
"""Cross-node command-size sweep.

The 1M row (~3.33 MB body) is published by an API process but the task is only
created where the elected master runs. On today's boots:
  * master on the SAME node as the API -> 1.2 MB and 3.4 MB bodies were accepted in 0.5 s;
  * master on the OTHER node (API on node 1, master on node 2) -> the 1M command never
    appears in the master's log at all, while a 1.03 MB 300K command does.

So this sweep sends a tiny prompt with padding of known sizes to the NON-MASTER node's
API and records which ones get through. Body size is what is varied; prompt tokens stay
at a handful, so prefill cost is nil.

Usage: payload-sweep.py <api_base_url> <label>
"""
import json
import sys
import time
import urllib.error
import urllib.request

API = sys.argv[1].rstrip("/")
LABEL = sys.argv[2]
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

# JSON-body sizes to test, in bytes (300K row body = 1,031,711; 1M row body = 3,334,717)
TARGETS = [900_000, 1_031_711, 1_200_000, 1_600_000, 2_200_000, 3_334_717]

results = []
for target in TARGETS:
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 8, "temperature": 0.0, "stream": False, "enable_thinking": False,
            "stop": ["Z"]}
    raw = json.dumps(body).encode()
    pad = target - len(raw) - 4
    if pad > 0:
        body["stop"] = ["Z" * pad]
        raw = json.dumps(body).encode()
    rec = {"target_bytes": target, "actual_bytes": len(raw), "api": API}
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f"{API}/v1/chat/completions", data=raw,
            headers={"Content-Type": "application/json"}), timeout=120))
        ch = r["choices"][0]
        rec.update(outcome="accepted", secs=round(time.time() - t0, 2),
                   content=(ch["message"].get("content") or "")[:20])
    except urllib.error.HTTPError as e:
        rec.update(outcome=f"http_{e.code}", secs=round(time.time() - t0, 2),
                   detail=e.read()[:160].decode("utf-8", "replace"))
    except Exception as e:
        rec.update(outcome="error", secs=round(time.time() - t0, 2),
                   detail=f"{type(e).__name__}: {e}"[:160])
    results.append(rec)
    print(json.dumps(rec), flush=True)

out = {"label": LABEL, "api": API, "results": results}
json.dump(out, open(f"/Users/adam.durham/.hermes/cache/scratch/ab300k-20261018/out/payload-sweep-{LABEL}.json", "w"), indent=1)
print("WROTE out/payload-sweep-%s.json" % LABEL)

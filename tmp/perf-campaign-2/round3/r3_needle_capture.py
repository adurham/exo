#!/usr/bin/env python3
"""Round-3 settling experiment: byte-identity capture for the B=2 200K needle.

Fires the SAME deterministic prompt as bench/quality_probe_dsv4.py (build_prompt
seeded 7749) as TWO CONCURRENT streams at temperature 0, capturing the full
per-stream text (content + reasoning) for byte-level diffing between config A
(flag off) and config B (EXO_DSV4_MOE_EARLY_ALLSUM=1).

Usage: r3_needle_capture.py <out.json> [base_url] [model] [target_tokens]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import importlib.util

spec = importlib.util.spec_from_file_location(
    "qp", "/Users/adam.durham/repos/exo/bench/quality_probe_dsv4.py"
)
qp = importlib.util.module_from_spec(spec)
sys.modules["qp"] = qp
spec.loader.exec_module(qp)

import urllib.request

OUT = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://macstudio-m4-1:52415"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
MAX_TOKENS = 96

prompt, expected = qp.build_prompt(TARGET)
print(f"prompt {len(prompt):,} chars; needle expected: {expected}", flush=True)


def post_stream(idx: int):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "seed": 42,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    content, reasoning, ttft = [], [], None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                # exo omits [DONE] on dropped queues; presence is a bonus
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c = delta.get("content") or ""
            r = delta.get("reasoning_content") or ""
            if (c or r) and ttft is None:
                ttft = time.time() - t0
            if c:
                content.append(c)
            if r:
                reasoning.append(r)
    return {
        "idx": idx,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "ttft_s": ttft,
        "needle_found": expected in "".join(content) + "".join(reasoning),
        "wall_s": time.time() - t0,
    }


t0 = time.time()
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(post_stream, [0, 1]))

rec = {
    "model": MODEL,
    "target_tokens": TARGET,
    "max_tokens": MAX_TOKENS,
    "temperature": 0.0,
    "seed": 42,
    "expected_needle": expected,
    "wall_s": time.time() - t0,
    "streams": results,
}
with open(OUT, "w") as f:
    json.dump(rec, f, indent=1)
for s in results:
    print(
        f"stream {s['idx']}: needle={s['needle_found']} ttft={s['ttft_s'] and round(s['ttft_s'],1)}s "
        f"len(content)={len(s['content'])} len(reasoning)={len(s['reasoning'])}",
        flush=True,
    )
print(f"wrote {OUT}", flush=True)
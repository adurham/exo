#!/usr/bin/env python3
"""Decode-focused A/B probe: steady-state decode t/s over a long generation.

Moderate prompt, long forced generation (temp 0). Reports decode t/s
computed from completion_tokens / (t_end - t_first_token), which at
1000+ tokens amortizes the first-token edge. Quality eyeball: prints
head/tail + BOS-spam check.

Usage: python3 decode_probe.py [--max-tokens N] [--tag LABEL] [--iters N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

API = "http://192.168.86.201:52415"
MODEL = "mlx-community/DeepSeek-V4-Flash"

PROMPT = (
    "Write a detailed technical essay about the history of distributed "
    "computing, covering: mainframes and time-sharing, the client-server "
    "era, grid computing, MapReduce and the big-data era, and modern "
    "cloud-native architectures. Be thorough and specific."
)


def one_iter(client: httpx.Client, max_tokens: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    text_parts: list[str] = []
    usage = None
    with client.stream("POST", f"{API}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {}).get("content") or ""
                reasoning = ch.get("delta", {}).get("reasoning_content") or ""
                if delta or reasoning:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_chunks += 1
                    text_parts.append(delta or reasoning)
    t_end = time.perf_counter()
    text = "".join(text_parts)
    completion = usage.get("completion_tokens") if usage else None
    decode_s = (t_end - t_first) if t_first else float("nan")
    return {
        "completion_tokens": completion,
        "decode_s": round(decode_s, 2),
        "decode_tps": round(completion / decode_s, 2) if completion else None,
        "bos_spam": "begin▁of▁sentence" in text,
        "text_head": text[:150],
        "text_tail": text[-150:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    iters = []
    with httpx.Client(timeout=httpx.Timeout(3600.0, connect=30.0)) as client:
        for i in range(args.iters):
            r = one_iter(client, args.max_tokens)
            print(f"iter {i}: {r['decode_tps']} t/s "
                  f"({r['completion_tokens']} tok, bos_spam={r['bos_spam']})",
                  file=sys.stderr)
            iters.append(r)

    tpss = [r["decode_tps"] for r in iters if r["decode_tps"]]
    result = {
        "tag": args.tag,
        "iters": iters,
        "decode_tps_mean": round(statistics.mean(tpss), 2) if tpss else None,
        "decode_tps_stdev": round(statistics.stdev(tpss), 2) if len(tpss) > 1 else 0.0,
    }
    print(json.dumps(result, indent=2))
    out = f"/tmp/decode_probe_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

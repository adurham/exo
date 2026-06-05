# type: ignore
#!/usr/bin/env python3
"""Minimal decode-throughput A/B probe for MTP on/off.

Hits the running exo instance with a fixed, reasoning-suppressed prompt and
measures streaming decode tokens/sec over N iterations. Prints per-iter t/s,
mean/median/min/max/stddev, and a sample of generated text so throughput is
never quoted without showing the actual output (BOS-spam guard).

Usage (run ON a node, against localhost):
    .venv/bin/python bench/mtp_ab_probe.py --iters 10 --max-tokens 200 --label MTP_ON
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

PROMPT = (
    "Write a detailed 200-word description of how a four-stroke internal "
    "combustion engine works. Be technical and continuous. /no_think"
)


def run_once(base_url: str, model: str, max_tokens: int) -> tuple[float, int, str]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    first_tok_t: float | None = None
    last_tok_t = 0.0
    n_tokens = 0
    pieces: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                now = time.perf_counter()
                if first_tok_t is None:
                    first_tok_t = now
                last_tok_t = now
                n_tokens += 1
                pieces.append(piece)
    # decode-only rate: exclude the time-to-first-token (prefill) interval
    if first_tok_t is None or n_tokens < 2:
        return 0.0, n_tokens, "".join(pieces)[:300]
    decode_elapsed = last_tok_t - first_tok_t
    tps = (n_tokens - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
    return tps, n_tokens, "".join(pieces)[:300]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:52415")
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-8bit")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    for _ in range(args.warmup):
        run_once(args.base_url, args.model, args.max_tokens)

    rates: list[float] = []
    sample = ""
    for i in range(args.iters):
        tps, ntok, text = run_once(args.base_url, args.model, args.max_tokens)
        rates.append(tps)
        if i == 0:
            sample = text
        print(f"[{args.label}] iter {i + 1:2d}: {tps:6.2f} tok/s  ({ntok} tokens)")

    print(f"\n=== {args.label} SUMMARY ({args.iters} iters) ===")
    print(f"  mean   {statistics.mean(rates):.2f} tok/s")
    print(f"  median {statistics.median(rates):.2f} tok/s")
    print(f"  min    {min(rates):.2f}")
    print(f"  max    {max(rates):.2f}")
    if len(rates) > 1:
        print(f"  stddev {statistics.pstdev(rates):.3f}")
    print(f"\n  sample output (first 300 chars):\n  {sample!r}")


if __name__ == "__main__":
    main()

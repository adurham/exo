#!/usr/bin/env python3
"""Long-generation decode probe at a target context depth.

WHY THIS EXISTS: bench/ab_probe_tier1.py asks a needle question that the
model answers in ~80-100 tokens. A 0.2-0.3 s decode window is dominated by
first-token/startup effects, so its `decode_tps` is noise -- the standing
cluster rule is to never quote t/s from generations under ~400 tokens.
Comparing a knob arm against a baseline when BOTH are ~90-token samples
compares noise to noise.

This probe keeps the same long-context prefill (same filler + needle, same
cache-busting) but asks a question whose answer is necessarily long, so the
decode window is hundreds of tokens and the resulting tok/s is meaningful.
It still verifies the needle was retrieved (quality signal) AND saves the
full generated text so a human can read what the model actually wrote.

Usage:
  python3 bench/long_decode_probe.py 100000 --max-tokens 1200 --tag A1 --out X.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid

import httpx

API = "http://192.168.86.201:52415"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

FILLER = (
    "The observer pattern is a software design pattern in which an object, "
    "named the subject, maintains a list of its dependents, called observers, "
    "and notifies them automatically of any state changes. "
    "B-trees are self-balancing tree data structures that maintain sorted "
    "data and allow searches, sequential access, insertions, and deletions "
    "in logarithmic time. "
)

NEEDLE = "The secret authorization code for project Nightingale is: FALCON-MERCURY-7749."


def build_prompt(target_tokens: int) -> str:
    """~4 chars/token heuristic; needle at 40% depth, cache-busted."""
    run_id = uuid.uuid4().hex
    rng = random.Random(run_id)
    total_chars = target_tokens * 4
    n_fill = max(1, total_chars // len(FILLER))
    fillers = [
        f"[run {run_id} seq {i} salt {rng.randint(0, 10**9)}] " + FILLER
        for i in range(n_fill)
    ]
    fillers.insert(int(n_fill * 0.4), " " + NEEDLE + " ")
    doc = "".join(fillers)
    # The ASK is the difference from ab_probe_tier1: it forces a long answer
    # so the decode window is large enough to measure honestly.
    return (
        f"Session {run_id}. Below is a long document. Read it carefully.\n\n"
        + doc
        + "\n\nTwo tasks. FIRST, state the secret authorization code for "
        "project Nightingale exactly as it appears. SECOND, write a detailed "
        "essay of at least 900 words explaining the observer pattern and "
        "B-trees: how each works, their time and space complexity, when you "
        "would choose one over alternatives, and concrete examples of each "
        "in real systems. Be thorough and specific."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target_tokens", type=int, nargs="?", default=100_000)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--out", type=str)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    prompt = build_prompt(args.target_tokens)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.time()
    t_first: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, object] = {}
    finish_reason = None
    # Server-side stats (perf_counter-timed INSIDE the generator at
    # batch_generate.py:1255-1257/4559-4576), emitted on the streaming
    # path as an SSE comment line (": generation_stats {...}") rather
    # than inside a "data: " chunk -- see chat_completions.py
    # generate_chat_stream(). This is the ONLY trusted throughput
    # number; client-side decode_tps below remains a cross-check only.
    server_stats: dict[str, object] | None = None

    with httpx.Client(timeout=1800.0) as client:
        with client.stream("POST", f"{API}/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line and line.startswith(": generation_stats "):
                    try:
                        server_stats = json.loads(line[len(": generation_stats "):])
                    except json.JSONDecodeError:
                        pass
                    continue
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []) or []:
                    delta = ch.get("delta") or {}
                    # thinking models split output; capture BOTH streams
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or ""
                    if (piece or think) and t_first is None:
                        t_first = time.time()
                    if piece:
                        content_parts.append(piece)
                    if think:
                        reasoning_parts.append(think)
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]

    t_end = time.time()
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    full = reasoning + content

    prefill_s = round((t_first - t_start), 2) if t_first else None
    decode_s = round((t_end - t_first), 2) if t_first else None
    ctok = usage.get("completion_tokens")
    ptok = usage.get("prompt_tokens")

    result = {
        "tag": args.tag,
        "model": args.model,
        "target_tokens": args.target_tokens,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "prefill_s": prefill_s,
        "prefill_tps": (
            round(ptok / prefill_s, 2) if ptok and prefill_s else None
        ),
        "decode_s": decode_s,
        "decode_tps": (
            round(ctok / decode_s, 2) if ctok and decode_s else None
        ),
        # honesty flag: the standing rule is that t/s from a short
        # generation is startup noise, not a throughput measurement.
        "decode_sample_trustworthy": bool(ctok and ctok >= 400),
        # Server-side, perf_counter-timed decode throughput -- the
        # trusted number. decode_tps above is a client-side cross-check
        # only, never the decision input (round-6 phase-0 rule).
        "server_stats": server_stats,
        "server_generation_tps": (
            server_stats.get("generation_tps") if server_stats else None
        ),
        "finish_reason": finish_reason,
        "needle_hit": "FALCON-MERCURY-7749" in full,
        "gen_chars": len(full),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "reasoning_content": reasoning,
        "content": content,
    }

    print(
        json.dumps(
            {k: v for k, v in result.items()
             if k not in ("reasoning_content", "content")},
            indent=2,
        )
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Tier-1 A/B probe: prefill + decode throughput at a target context size.

Sends ONE completing request of ~TARGET_TOKENS prompt to the DSv4 instance
and reports: prefill_s, prefill_tps, decode_tps, first 200 chars of output
(quality eyeball), and BOS-spam check. Modeled on /tmp/probe.py from the
prefill-cliff investigation but self-contained and local.

Usage: python3 ab_probe.py [TARGET_TOKENS] [--max-tokens N] [--tag LABEL]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

API = "http://192.168.86.201:52415"
MODEL = "mlx-community/DeepSeek-V4-Flash"

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
    # ~4 chars/token heuristic; needle at 40% depth.
    # Cache-busting: a unique random header + per-run shuffled filler order
    # so the KV prefix cache can NEVER serve a prior run's prefill (byte-
    # identical prompts short-circuit prefill entirely — measured 98813
    # "t/s" on a warm cluster, i.e. no prefill at all).
    import random
    import uuid
    run_id = uuid.uuid4().hex
    rng = random.Random(run_id)
    total_chars = target_tokens * 4
    n_fill = max(1, total_chars // len(FILLER))
    fillers = [f"[run {run_id} seq {i} salt {rng.randint(0, 10**9)}] " + FILLER
               for i in range(n_fill)]
    fillers.insert(int(n_fill * 0.4), " " + NEEDLE + " ")
    doc = "".join(fillers)
    return (
        f"Session {run_id}. Below is a long document. Read it carefully.\n\n" + doc +
        "\n\nWhat is the secret authorization code for project Nightingale? "
        "Reply with the code only."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target_tokens", type=int, nargs="?", default=100_000)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--tag", default="untagged")
    args = ap.parse_args()

    prompt = build_prompt(args.target_tokens)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    t_first = None
    text_parts: list[str] = []
    usage = None
    with httpx.Client(timeout=httpx.Timeout(7200.0, connect=30.0)) as client:
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
                    if delta:
                        if t_first is None:
                            t_first = time.perf_counter()
                        text_parts.append(delta)
    t_end = time.perf_counter()

    text = "".join(text_parts)
    prefill_s = (t_first - t0) if t_first else float("nan")
    decode_s = (t_end - t_first) if t_first else float("nan")

    prompt_toks = usage.get("prompt_tokens") if usage else None
    completion_toks = usage.get("completion_tokens") if usage else None
    prefill_tps = prompt_toks / prefill_s if (prompt_toks and prefill_s) else None
    decode_tps = completion_toks / decode_s if (completion_toks and decode_s) else None

    bos_spam = "begin▁of▁sentence" in text or "begin_of_sentence" in text
    needle_hit = "FALCON-MERCURY-7749" in text

    result = {
        "tag": args.tag,
        "target_tokens": args.target_tokens,
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "prefill_s": round(prefill_s, 2),
        "prefill_tps": round(prefill_tps, 2) if prefill_tps else None,
        "decode_s": round(decode_s, 2),
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "needle_hit": needle_hit,
        "bos_spam": bos_spam,
        "output_head": text[:200],
        "output_tail": text[-200:],
    }
    print(json.dumps(result, indent=2))
    out = f"/tmp/ab_probe_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

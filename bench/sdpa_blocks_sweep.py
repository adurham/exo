#!/usr/bin/env python3
"""SDPA blocks sweep at ~20K context for DSv4-Flash-8bit.

For each MLX_SDPA_BLOCKS value, the cluster must be restarted with
that env var set. This script ONLY runs the workload + measurement;
the caller restarts the cluster between sweeps.

Outputs decode tok/s for a fixed 20K-context prompt with 256 generated
tokens. Repeats N times to average out noise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

import httpx


FILLER = (
    "The observer pattern is a software design pattern in which an object, "
    "named the subject, maintains a list of its dependents, called observers, "
    "and notifies them automatically of any state changes. "
    "A binary search tree is a rooted binary tree data structure with the "
    "key of each internal node being greater than all the keys in the "
    "respective node's left subtree and less than the ones in its right "
    "subtree. Garbage collection is a form of automatic memory management. "
    "MapReduce is a programming model for processing big data sets with a "
    "parallel, distributed algorithm on a cluster. The CAP theorem states "
    "that any distributed data store can provide only two of the following "
    "three guarantees: consistency, availability, and partition tolerance. "
    "Functional programming is a programming paradigm where programs are "
    "constructed by applying and composing functions. A hash table is a "
    "data structure that implements an associative array or dictionary. "
    "Consensus algorithms are fundamental to distributed computing. "
    "The actor model is a mathematical model of concurrent computation. "
    "B-trees are self-balancing tree data structures that maintain sorted "
    "data and allow searches, sequential access, insertions, and deletions "
    "in logarithmic time. Type theory is the academic study of type systems. "
    "Event sourcing is a software architecture pattern in which changes to "
    "application state are stored as a sequence of events instead of just "
    "the current state.\n\n"
)


def build_prompt(target_tokens: int) -> str:
    rng = random.Random(0xC0FFEE)
    target_chars = target_tokens * 4
    parts = []
    char_count = 0
    while char_count < target_chars:
        block = FILLER + f"Sequence number: {rng.randrange(10**9)}.\n\n"
        parts.append(block)
        char_count += len(block)
    body = "".join(parts)[:target_chars]
    return (
        "Read the following technical document carefully. "
        "After reading, write a 200-word summary of the major themes.\n\n"
        f"{body}\n\nNow write the summary."
    )


async def one_run(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "enable_thinking": False,
        "max_tokens": max_tokens,
    }
    t_submit = time.perf_counter()
    t_first = None
    decode_count = 0
    usage = {}
    async with httpx.AsyncClient(timeout=900.0) as client:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=body
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                if delta.get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    decode_count += 1
    t_end = time.perf_counter()
    if t_first is None:
        return {"error": "no_decode_tokens"}
    prefill_s = t_first - t_submit
    decode_s = t_end - t_first
    decode_tps = decode_count / decode_s if decode_s > 0 else 0.0
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", decode_count),
        "decode_chunks": decode_count,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    parser.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    parser.add_argument("--target-tokens", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    prompt = build_prompt(args.target_tokens)
    print(f"[sweep] label={args.label} ctx_target={args.target_tokens} runs={args.runs} warmup={args.warmup}")

    for i in range(args.warmup):
        print(f"[sweep] warmup {i+1}/{args.warmup}", flush=True)
        r = await one_run(args.base_url, args.model, prompt, args.max_tokens)
        if "error" in r:
            print(f"[sweep] warmup error: {r}", file=sys.stderr)
            sys.exit(1)
        print(f"  prompt_tokens={r['prompt_tokens']} prefill={r['prefill_s']:.2f}s decode={r['decode_tps']:.2f}tok/s")

    results = []
    for i in range(args.runs):
        print(f"[sweep] run {i+1}/{args.runs}", flush=True)
        r = await one_run(args.base_url, args.model, prompt, args.max_tokens)
        if "error" in r:
            print(f"[sweep] run error: {r}", file=sys.stderr)
            sys.exit(1)
        print(f"  prompt_tokens={r['prompt_tokens']} prefill={r['prefill_s']:.2f}s decode={r['decode_tps']:.2f}tok/s")
        results.append(r)

    decode_rates = [r["decode_tps"] for r in results]
    prefill_times = [r["prefill_s"] for r in results]
    avg_decode = sum(decode_rates) / len(decode_rates)
    avg_prefill = sum(prefill_times) / len(prefill_times)
    print(f"\n[sweep RESULT] label={args.label}")
    print(f"  decode_tps: avg={avg_decode:.2f} min={min(decode_rates):.2f} max={max(decode_rates):.2f}")
    print(f"  prefill_s:  avg={avg_prefill:.2f} min={min(prefill_times):.2f} max={max(prefill_times):.2f}")

    summary = {
        "label": args.label,
        "ctx_target": args.target_tokens,
        "max_tokens": args.max_tokens,
        "runs": results,
        "avg_decode_tps": avg_decode,
        "avg_prefill_s": avg_prefill,
    }
    with open(f"/tmp/sdpa_sweep_{args.label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sweep] wrote /tmp/sdpa_sweep_{args.label}.json")


if __name__ == "__main__":
    asyncio.run(main())

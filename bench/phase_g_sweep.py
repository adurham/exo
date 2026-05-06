#!/usr/bin/env python3
"""Post-Phase-G prefill+decode sweep for DSv4-Flash 8bit on the 2-node TP cluster.

Two sweeps, both via OpenAI streaming so we can separate prefill (TTFT)
from decode (steady-state inter-token) without instrumenting the runner.

* Prefill sweep: prompt lengths 1K, 8K, 32K, 64K, 100K. Each request
  asks for 16 generated tokens; report prefill tok/s = ``prompt_tokens
  / TTFT_seconds``.
* Decode sweep: at each context length 1K, 10K, 50K, 100K, 150K, 200K
  fire concurrent requests at c=1 and c=2. Each generates 64 tokens;
  report per-stream and aggregate decode tok/s from the post-TTFT
  inter-token interval median.

Output: JSON to stdout + ``/tmp/phase_g_sweep_results.json``.

The cluster must already be running with the model placed. Auto-place
via start_cluster.sh first if not.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any

import httpx

API = "http://192.168.86.201:52415/v1/chat/completions"
MODEL = "mlx-community/DeepSeek-V4-Flash-8bit"

# Roughly token-counted filler. Picked so we hit prompt_tokens within
# ~5% of the target without needing a tokenizer locally; the API
# returns the exact prompt_tokens in usage.
_FILLER = (
    "The observer pattern is a software design pattern in which an object "
    "named the subject maintains a list of its dependents called observers "
    "and notifies them of state changes. A binary search tree is a rooted "
    "binary data structure with each internal node greater than the keys "
    "in its left subtree and less than those in its right. Quicksort uses "
    "a divide and conquer strategy partitioning around a pivot element. "
    "TCP three-way handshake establishes a reliable connection via SYN "
    "SYN-ACK ACK before any payload bytes are exchanged. Cache coherence "
    "in a multi-core CPU is enforced through MESI MOESI or directory based "
    "protocols. The CAP theorem states that a distributed datastore can "
    "guarantee at most two of consistency availability and partition "
    "tolerance simultaneously. Garbage collection in modern runtimes uses "
    "a generational hypothesis treating young allocations differently from "
    "tenured ones. A fenwick tree supports prefix sum queries and point "
    "updates in O log n. Bloom filters give probabilistic set membership "
    "with one sided false positive errors and zero false negatives. "
)
_FILLER_TOK_EST = 220  # measured: ~218-222 prompt_tokens per filler pass


def build_prompt(target_tokens: int) -> str:
    """Build a user-message prompt sized to ~target_tokens prompt_tokens."""
    repeats = max(1, target_tokens // _FILLER_TOK_EST)
    return (
        "Read the following technical context and answer in 1 sentence.\n\n"
        + (_FILLER * repeats)
        + "\nQuestion: name one of the topics above."
    )


async def _stream_one(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int,
    label: str,
) -> dict[str, Any]:
    """Fire one streaming completion. Return timing + token counts."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.time()
    t_first: float | None = None
    inter_token_times: list[float] = []
    last_chunk: float | None = None
    n_chunks = 0
    usage: dict[str, Any] | None = None
    err: str | None = None
    try:
        async with client.stream(
            "POST", API, json=payload, timeout=httpx.Timeout(900.0)
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                now = time.time()
                # Some chunks carry only usage (final), no choices.
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("choices"):
                    delta = obj["choices"][0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        if t_first is None:
                            t_first = now
                        else:
                            inter_token_times.append(now - last_chunk)
                        last_chunk = now
                        n_chunks += 1
    except Exception as exc:
        err = repr(exc)
    t_end = time.time()
    return {
        "label": label,
        "ttft_s": (t_first - t_send) if t_first else None,
        "wall_s": t_end - t_send,
        "n_content_chunks": n_chunks,
        "decode_inter_p50_ms": (
            1000.0 * statistics.median(inter_token_times)
            if inter_token_times
            else None
        ),
        "decode_inter_p10_ms": (
            1000.0 * statistics.quantiles(inter_token_times, n=10)[0]
            if len(inter_token_times) >= 10
            else None
        ),
        "usage": usage,
        "error": err,
    }


async def prefill_sweep(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    targets = [1_000, 8_000, 32_000, 64_000, 100_000]
    out: list[dict[str, Any]] = []
    for n in targets:
        prompt = build_prompt(n)
        print(f"\n[prefill] target≈{n} tokens", flush=True)
        r = await _stream_one(client, prompt, max_tokens=16, label=f"prefill_{n}")
        if r.get("error"):
            print(f"  ERROR: {r['error']}", flush=True)
            out.append({"target_tokens": n, **r})
            continue
        usage = r.get("usage") or {}
        prompt_toks = int(usage.get("prompt_tokens") or 0)
        ttft = r.get("ttft_s") or 0
        prefill_tps = prompt_toks / ttft if ttft else None
        rec = {
            "target_tokens": n,
            "actual_prompt_tokens": prompt_toks,
            "ttft_s": ttft,
            "prefill_tok_per_s": prefill_tps,
            "wall_s": r["wall_s"],
            "completion_tokens": int((usage.get("completion_tokens") or 0)),
        }
        print(
            f"  actual_prompt={prompt_toks} ttft={ttft:.2f}s "
            f"prefill={prefill_tps:.1f} tok/s",
            flush=True,
        )
        out.append(rec)
    return out


async def decode_at(
    client: httpx.AsyncClient,
    ctx_target: int,
    concurrency: int,
    max_tokens: int,
) -> dict[str, Any]:
    prompts = [build_prompt(ctx_target) for _ in range(concurrency)]
    print(
        f"\n[decode] ctx≈{ctx_target} c={concurrency} max_tokens={max_tokens}",
        flush=True,
    )
    t0 = time.time()
    results = await asyncio.gather(
        *[
            _stream_one(client, p, max_tokens, label=f"c{concurrency}_{i}")
            for i, p in enumerate(prompts)
        ]
    )
    t_total = time.time() - t0
    if any(r.get("error") for r in results):
        return {
            "ctx_target": ctx_target,
            "concurrency": concurrency,
            "errors": [r.get("error") for r in results if r.get("error")],
            "wall_s": t_total,
        }
    # Per-stream decode tok/s: completion_tokens / (wall - ttft).
    per_stream: list[dict[str, Any]] = []
    for r in results:
        usage = r.get("usage") or {}
        ct = int(usage.get("completion_tokens") or 0)
        wall = r.get("wall_s") or 0
        ttft = r.get("ttft_s") or 0
        denom = wall - ttft
        per_stream.append(
            {
                "label": r["label"],
                "completion_tokens": ct,
                "ttft_s": ttft,
                "decode_wall_s": denom,
                "decode_tok_per_s": ct / denom if denom > 0 else None,
                "decode_inter_p50_ms": r.get("decode_inter_p50_ms"),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            }
        )
    rates = [s["decode_tok_per_s"] for s in per_stream if s["decode_tok_per_s"]]
    per_stream_median = statistics.median(rates) if rates else None
    aggregate = sum(rates) if rates else None
    rec = {
        "ctx_target": ctx_target,
        "concurrency": concurrency,
        "wall_s_total": t_total,
        "per_stream": per_stream,
        "per_stream_median_tok_per_s": per_stream_median,
        "aggregate_tok_per_s": aggregate,
    }
    if per_stream_median is not None:
        print(
            f"  per_stream≈{per_stream_median:.2f} tok/s  "
            f"agg≈{aggregate:.2f} tok/s",
            flush=True,
        )
    return rec


async def decode_sweep(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    ctx_targets = [1_000, 10_000, 50_000, 100_000, 150_000, 200_000]
    concurrencies = [1, 2]
    out: list[dict[str, Any]] = []
    for ctx in ctx_targets:
        for c in concurrencies:
            r = await decode_at(client, ctx, c, max_tokens=64)
            out.append(r)
    return out


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--out", default="/tmp/phase_g_sweep_results.json")
    args = p.parse_args()

    print(f"API={API} model={MODEL}")
    async with httpx.AsyncClient() as client:
        prefill = [] if args.skip_prefill else await prefill_sweep(client)
        decode = [] if args.skip_decode else await decode_sweep(client)
    out = {
        "api": API,
        "model": MODEL,
        "ts": time.time(),
        "prefill_sweep": prefill,
        "decode_sweep": decode,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""DSv4 100K-context quality probe with needle-in-a-haystack.

Builds a ~100K-token prompt with a hidden secret code and sends it to
whatever DSv4 instance is currently placed on the cluster. Verifies the
model can recall the needle AND that the response doesn't contain
``<｜begin▁of▁sentence｜>`` token spam (the "throughput-clean +
quality-dead" failure mode documented in skill pitfall #41).

Supports concurrent firing (``--concurrency N``) so the **c=2 batched
draft path is exercised** — the prior single-request probes went through
the BS=1 / ``mtp_module.draft_tokens`` path which has a per-step
``mx.eval(tok_arr)`` fence, and could never detect the c=2-specific
``_draft_tokens_batched`` BOS-spam regression that hits at γ≥2 100K.

Schema bumped to v2 (2026-05-23). Output now always carries ``iters[]``
+ ``streams[]`` arrays plus a ``summary`` object. Old top-level fields
are no longer populated — callers should read from the per-iter /
per-stream structure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

FILLER_TOPICS = [
    "The observer pattern is a software design pattern in which an object, named the subject, maintains a list of its dependents, called observers, and notifies them automatically of any state changes, usually by calling one of their methods.",
    "A binary search tree is a rooted binary tree data structure with the key of each internal node being greater than all the keys in the respective node's left subtree and less than the ones in its right subtree.",
    "Garbage collection is a form of automatic memory management. The garbage collector attempts to reclaim memory which was allocated by the program but is no longer referenced.",
    "MapReduce is a programming model and an associated implementation for processing and generating big data sets with a parallel, distributed algorithm on a cluster.",
    "The CAP theorem states that any distributed data store can provide only two of the following three guarantees: consistency, availability, and partition tolerance.",
    "Functional programming is a programming paradigm where programs are constructed by applying and composing functions. It is a declarative programming paradigm.",
    "A hash table is a data structure that implements an associative array or dictionary. It is an abstract data type that maps keys to values.",
    "Consensus algorithms are fundamental to distributed computing. They allow multiple processes to agree on a single value even in the presence of failures.",
    "The actor model is a mathematical model of concurrent computation that treats actor as the universal primitive of concurrent computation.",
    "B-trees are self-balancing tree data structures that maintain sorted data and allow searches, sequential access, insertions, and deletions in logarithmic time.",
    "Type theory is the academic study of type systems. A type system is a syntactic method for enforcing levels of abstraction in programs.",
    "Event sourcing is a software architecture pattern in which changes to application state are stored as a sequence of events. Instead of storing just the current state, the full history of actions is stored.",
    "Raft is a consensus algorithm designed as an alternative to the Paxos family of algorithms. Raft is more understandable than Paxos by separation of logic and decomposition.",
    "The Byzantine generals problem is a description of a situation where actors must agree on a strategy to avoid catastrophic system failure, but some of the actors are unreliable.",
    "Vector clocks are a data structure used for determining the partial ordering of events in a distributed system and detecting causality violations.",
    "Conflict-free replicated data types are data structures that can be replicated across multiple computers in a network, with the replicas being updated independently and concurrently without coordination.",
]

NEEDLE = "The secret authorization code for project Nightingale is: FALCON-MERCURY-7749."
QUESTION = (
    "Below is a long document. Read it carefully. After the document, "
    "I will ask you to recall a specific piece of information that is "
    "embedded somewhere in the middle.\n\n"
)
INSTRUCTION = (
    "\n\n"
    "Question: What is the secret authorization code for project Nightingale? "
    "Respond with just the code in the format CODE-NAME-NUMBER on a single line."
)

# Special-token leakage patterns. DSv4's tokenizer emits these for sentence
# delimiters at training time; they should NEVER appear in chat-completion
# decoded output because the chat template strips them. Their presence in a
# response means the model has fallen into BOS-spam mode (a real failure
# mode — see skill pitfall #41, references/2026-05-22-throughput-vs-quality-trap.md).
# These are the exact Unicode glyphs (｜ = U+FF5C, ▁ = U+2581) that DSv4
# uses in its delimiter tokens; substring-checking the rendered bytes is
# the cheapest "is the model emitting raw special-tokens" detector.
SPECIAL_TOKEN_PATTERNS = [
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
]


def build_prompt(target_tokens: int, seed: int = 7749):
    random.seed(seed)
    target_chars = target_tokens * 4
    body_chars = target_chars - len(QUESTION) - len(INSTRUCTION)
    needle_position = random.randint(int(body_chars * 0.33), int(body_chars * 0.66))
    paragraphs = []
    char_count = 0
    needle_placed = False
    while char_count < body_chars:
        if not needle_placed and char_count >= needle_position:
            paragraphs.append(NEEDLE)
            char_count += len(NEEDLE)
            needle_placed = True
        else:
            topic = random.choice(FILLER_TOPICS)
            paragraphs.append(topic)
            char_count += len(topic) + 2
    if not needle_placed:
        mid = len(paragraphs) // 2
        paragraphs.insert(mid, NEEDLE)
    body = "\n\n".join(paragraphs)
    prompt = QUESTION + body + INSTRUCTION
    return prompt, "FALCON-MERCURY-7749"


def detect_quality_issues(response: str) -> dict[str, Any]:
    """Inspect a decoded response for the documented quality failure modes.

    Returns a dict with:
      ``special_token_counts``: per-pattern leak counts (BOS, EOS, role tags)
      ``total_special_tokens``: sum across all patterns
      ``issues``: list of human-readable tags (e.g. ``"BOS_SPAM:34"``)
      ``has_quality_issue``: bool — True if ANY special-token pattern leaked

    Detection rationale: any positive count of these special-token strings
    in the response is a regression. DSv4's chat template should strip
    them; their presence in decoded text means the model is emitting raw
    tokenizer special-IDs as content. Even a single occurrence is a fail.
    """
    counts: dict[str, int] = {}
    issues: list[str] = []
    for pat in SPECIAL_TOKEN_PATTERNS:
        c = response.count(pat)
        counts[pat] = c
        if c > 0:
            tag = pat.replace("<｜", "").replace("｜>", "").replace("▁", "_").upper()
            issues.append(f"{tag}_LEAK:{c}")
    total = sum(counts.values())
    return {
        "special_token_counts": counts,
        "total_special_tokens": total,
        "issues": issues,
        "has_quality_issue": total > 0,
    }


async def run_one_stream(
    args: argparse.Namespace,
    prompt: str,
    expected_needle: str,
    stream_idx: int,
) -> dict[str, Any]:
    """Fire one streamed chat-completion request and collect the result.

    Returns a single-stream record with timing, decoded text, needle
    detection, and quality-issue scan.
    """
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": args.max_tokens,
    }
    response_chunks: list[str] = []
    usage: dict[str, Any] = {}
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    start = time.perf_counter()
    first_token_time: float | None = None
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "POST", f"{args.base_url}/v1/chat/completions", json=body
        ) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_str = line[6:].strip()
            if payload_str == "[DONE]":
                break
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            if "usage" in payload and payload["usage"]:
                usage = payload["usage"]
            choices = payload.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    response_chunks.append(content)
    end = time.perf_counter()
    response = "".join(response_chunks)
    ttft = (first_token_time - start) if first_token_time else 0.0
    total_s = end - start
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    generation_tokens = int(
        usage.get("completion_tokens", 0) or len(response.split())
    )
    found_needle = expected_needle.lower() in response.lower()
    decode_tps = (
        generation_tokens / (total_s - ttft) if (total_s - ttft) > 0 else 0.0
    )
    prefill_tps = prompt_tokens / ttft if ttft > 0 else 0.0
    quality = detect_quality_issues(response)
    stream_ok = found_needle and not quality["has_quality_issue"]
    return {
        "stream_idx": stream_idx,
        "ttft_s": ttft,
        "total_s": total_s,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prefill_tps_apparent": prefill_tps,
        "decode_tps_apparent": decode_tps,
        "needle_found": found_needle,
        "quality": quality,
        "stream_ok": stream_ok,
        # Full text is sometimes huge — keep it for the JSON record (this
        # is a diagnostic tool, not a hot-path metric), but also expose a
        # short repr-quoted prefix so BOS bytes are immediately visible
        # to a human reader scanning the log.
        "response_text": response,
        "response_text_repr_prefix": repr(response[:200]),
    }


async def run_one_iter(
    args: argparse.Namespace,
    prompt: str,
    expected_needle: str,
    iter_idx: int,
) -> dict[str, Any]:
    """Fire ``args.concurrency`` streams in parallel against the same prompt.

    At ``--concurrency 1`` this matches the pre-2026-05-23 behavior. At
    ``--concurrency N>=2`` the cluster's c=N batched draft path is
    exercised — the path that holds the BOS-spam quality bug.
    """
    iter_start = time.perf_counter()
    tasks = [
        run_one_stream(args, prompt, expected_needle, idx)
        for idx in range(args.concurrency)
    ]
    streams = await asyncio.gather(*tasks)
    iter_wall = time.perf_counter() - iter_start
    # Sort by stream_idx so output is deterministic across runs.
    streams.sort(key=lambda s: s["stream_idx"])
    decode_tpss = [s["decode_tps_apparent"] for s in streams]
    if decode_tpss:
        agg = sum(decode_tpss)
        sym = (
            min(decode_tpss) / max(decode_tpss) if max(decode_tpss) > 0 else 0.0
        )
    else:
        agg = 0.0
        sym = 0.0
    return {
        "iter_idx": iter_idx,
        "iter_wall_s": iter_wall,
        "aggregate_decode_tps": agg,
        "symmetry": sym,
        # 0.85 mirrors the threshold the c=2 instrumentation tracer uses
        # (per skill recipe). Anything under indicates a one-stream stall.
        "bistability_flag": sym < 0.85,
        "all_streams_ok": all(s["stream_ok"] for s in streams),
        "any_special_tokens_leaked": any(
            s["quality"]["has_quality_issue"] for s in streams
        ),
        "all_needles_found": all(s["needle_found"] for s in streams),
        "streams": streams,
    }


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Build the prompt once, run ``args.iters`` iterations, summarize."""
    prompt, expected = build_prompt(args.target_tokens)
    print(
        f"Prompt built: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens est)"
    )
    print(f"Expected needle: {expected}")
    print(
        f"Sending to {args.base_url} model {args.model} | "
        f"concurrency={args.concurrency} iters={args.iters}"
    )
    iters_out: list[dict[str, Any]] = []
    for i in range(args.iters):
        rec = await run_one_iter(args, prompt, expected, i)
        iters_out.append(rec)
        # One-line per-iter summary for human polling.
        per_stream_tps = [
            f"{s['decode_tps_apparent']:.2f}" for s in rec["streams"]
        ]
        any_leak = rec["any_special_tokens_leaked"]
        any_needle = rec["all_needles_found"]
        print(
            f"iter {i}: wall={rec['iter_wall_s']:.1f}s "
            f"per-stream-tps=[{', '.join(per_stream_tps)}] "
            f"agg={rec['aggregate_decode_tps']:.2f} "
            f"sym={rec['symmetry']:.3f} "
            f"bistab={rec['bistability_flag']} "
            f"all_needles={any_needle} "
            f"special_tokens_leaked={any_leak}"
        )
        # If quality broke on this iter, dump the per-stream repr prefix so
        # the human can immediately see what the model emitted (BOS spam,
        # repeated EOS, role-tag leak, etc.) — pitfall #41 says throughput
        # numbers alone lie, so we surface the text right here.
        if any_leak or not any_needle:
            for s in rec["streams"]:
                print(
                    f"  stream {s['stream_idx']}: "
                    f"needle_found={s['needle_found']} "
                    f"quality_issues={s['quality']['issues']}"
                )
                print(
                    f"    text_repr_prefix={s['response_text_repr_prefix']}"
                )
    # Summary
    iter_aggs = [it["aggregate_decode_tps"] for it in iters_out]
    iter_syms = [it["symmetry"] for it in iters_out]
    summary = {
        "all_iters_ok": all(it["all_streams_ok"] for it in iters_out),
        "iters_with_special_token_leak": sum(
            1 for it in iters_out if it["any_special_tokens_leaked"]
        ),
        "iters_with_bistability": sum(
            1 for it in iters_out if it["bistability_flag"]
        ),
        "iters_with_all_needles": sum(
            1 for it in iters_out if it["all_needles_found"]
        ),
        "agg_tps_mean": statistics.mean(iter_aggs) if iter_aggs else 0.0,
        "agg_tps_min": min(iter_aggs) if iter_aggs else 0.0,
        "agg_tps_max": max(iter_aggs) if iter_aggs else 0.0,
        "agg_tps_stdev": (
            statistics.stdev(iter_aggs) if len(iter_aggs) > 1 else 0.0
        ),
        "symmetry_mean": statistics.mean(iter_syms) if iter_syms else 0.0,
        "symmetry_min": min(iter_syms) if iter_syms else 0.0,
    }
    out_rec = {
        "schema_version": 2,
        "label": args.label,
        "model": args.model,
        "target_tokens": args.target_tokens,
        "concurrency": args.concurrency,
        "iters_n": args.iters,
        "expected_needle": expected,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "iters": iters_out,
        "summary": summary,
    }
    print()
    print("=== SUMMARY ===")
    print(
        f"all_iters_ok={summary['all_iters_ok']} "
        f"all_needles={summary['iters_with_all_needles']}/{args.iters} "
        f"special_token_leaks={summary['iters_with_special_token_leak']}/{args.iters} "
        f"bistability_iters={summary['iters_with_bistability']}/{args.iters}"
    )
    print(
        f"agg_tps: mean={summary['agg_tps_mean']:.2f} σ={summary['agg_tps_stdev']:.3f} "
        f"min={summary['agg_tps_min']:.2f} max={summary['agg_tps_max']:.2f}"
    )
    print(
        f"symmetry: mean={summary['symmetry_mean']:.3f} "
        f"min={summary['symmetry_min']:.3f}"
    )
    if args.out:
        out_path = Path(os.path.expanduser(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]] = []
        if out_path.exists():
            try:
                loaded = json.loads(out_path.read_text())
                if isinstance(loaded, list):
                    existing = loaded
                elif isinstance(loaded, dict):
                    existing = [loaded]
            except Exception:
                existing = []
        existing.append(out_rec)
        out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"appended result to {out_path}")
    return out_rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    ap.add_argument("--target-tokens", type=int, default=100000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Number of parallel streamed requests fired per iteration. "
            "1 = legacy single-request behavior (BS=1 mtp_module.draft_tokens "
            "path). 2+ = exercise the c=N batched _draft_tokens_batched path "
            "where the BOS-spam quality bug at γ≥2 100K lives."
        ),
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help=(
            "Repeat the (concurrency-wide) request iters times. The known "
            "BOS-spam regression fires at iter 1+, so use --iters 3 for "
            "validation runs that need to catch the iter-N+1 transition."
        ),
    )
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="unlabeled")
    args = ap.parse_args()
    if args.concurrency < 1:
        ap.error("--concurrency must be >= 1")
    if args.iters < 1:
        ap.error("--iters must be >= 1")
    asyncio.run(run_probe(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

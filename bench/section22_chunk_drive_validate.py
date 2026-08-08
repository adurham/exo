#!/usr/bin/env python3
"""Section 22 real-hardware validation: ~72K-token chunk-drive prefill
against the deployed DeepSeek-V4-Flash pipeline-parallel cluster.
Exercises the bounded-blocking-ack chunk-boundary fix (ee7fae663).

REQUIRES both EXO_PP_METAFRAME=1 and EXO_PP_BATCHED_DECODE=1 to be set
at cluster launch time -- BOTH flags default to 0 in start_cluster.sh,
and EXO_PP_BATCHED_DECODE=1 alone does NOT install the batched pipeline
layers (install_batched_decode_pipeline_layers only runs inside the
EXO_PP_METAFRAME=1 branch in utils_mlx.py). Without both flags, requests
silently fall back to the unmodified synchronous prefill path and this
script's traffic never touches Section 22's code at all (confirmed the
hard way, see docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md
Section 23). Verify chunk-drive is actually engaging via:
  ssh <node> "tail -f ~/.exo/exo_log/runner_log/stderr.log | grep -i chunk_index"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import Any

import httpx

FILLER_TOPICS = [
    "The observer pattern is a software design pattern in which an object, named the subject, maintains a list of its dependents, called observers, and notifies them automatically of any state changes, usually by calling one of their methods. It is mainly used for implementing distributed event handling systems.",
    "A binary search tree is a rooted binary tree data structure with the key of each internal node being greater than all the keys in the respective node's left subtree and less than the ones in its right subtree.",
    "Garbage collection is a form of automatic memory management. The garbage collector attempts to reclaim memory which was allocated by the program but is no longer referenced.",
    "MapReduce is a programming model and an associated implementation for processing and generating big data sets with a parallel, distributed algorithm on a cluster.",
    "The CAP theorem states that any distributed data store can provide only two of the following three guarantees: consistency, availability, and partition tolerance.",
    "Functional programming is a programming paradigm where programs are constructed by applying and composing functions.",
    "A hash table is a data structure that implements an associative array or dictionary using a hash function to compute an index into an array of buckets.",
    "Consensus algorithms are fundamental to distributed computing. They allow multiple processes to agree on a single value even in the presence of failures.",
    "The actor model is a mathematical model of concurrent computation that treats actor as the universal primitive of concurrent computation.",
    "B-trees are self-balancing tree data structures that maintain sorted data and allow searches, sequential access, insertions, and deletions in logarithmic time.",
    "Type theory is the academic study of type systems. A type system is a syntactic method for enforcing levels of abstraction in programs.",
    "Event sourcing is a software architecture pattern in which changes to application state are stored as a sequence of events.",
]

NEEDLE = "The secret code for project Nightingale is: FALCON-MERCURY-7749."


def build_prompt(target_tokens: int) -> str:
    target_chars = target_tokens * 4
    paragraphs: list[str] = []
    char_count = 0
    needle_placed = False
    needle_position = random.randint(target_chars // 3, 2 * target_chars // 3)
    while char_count < target_chars:
        if not needle_placed and char_count >= needle_position:
            paragraphs.append(NEEDLE)
            char_count += len(NEEDLE)
            needle_placed = True
        else:
            topic = random.choice(FILLER_TOPICS)
            paragraphs.append(topic)
            char_count += len(topic)
    paragraphs.append(
        "\n\nWhat is the secret code for project Nightingale? Answer with just the code."
    )
    return "\n\n".join(paragraphs)


async def run(base_url: str, model: str, target_tokens: int, label: str) -> dict[str, Any]:
    prompt = build_prompt(target_tokens)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": 200,
    }
    start = time.perf_counter()
    first_token_time: float | None = None
    last_chunk_progress_time = start
    content_chunks: list[str] = []
    usage: dict[str, Any] = {}
    chunk_count = 0

    print(f"[{label}] Prompt ~{len(prompt) // 4:,} tokens, sending...")
    async with httpx.AsyncClient() as client, client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body, timeout=900.0
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                if "prefill_progress" in line or "PrefillProgressChunk" in line:
                    try:
                        payload = line.split(" ", 1)[1] if " " in line else line
                        prog: dict[str, Any] = json.loads(payload)
                        chunk_progress: dict[str, Any] = prog.get(
                            "PrefillProgressChunk", prog
                        )
                        processed = chunk_progress.get("processed_tokens", 0)
                        total = chunk_progress.get("total_tokens", 0)
                        chunk_count += 1
                        now = time.perf_counter()
                        gap = now - last_chunk_progress_time
                        last_chunk_progress_time = now
                        if total:
                            print(
                                f"[{label}]  chunk#{chunk_count} {processed:,}/{total:,} "
                                f"(+{gap:.2f}s since last)",
                                flush=True,
                            )
                    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                        pass
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                sse_chunk: dict[str, Any] = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if sse_chunk.get("usage"):
                usage = sse_chunk["usage"]
            choices: list[dict[str, Any]] = sse_chunk.get("choices", [])
            if choices:
                delta: dict[str, Any] = choices[0].get("delta", {})
                text: str = delta.get("content") or ""
                if text and first_token_time is None:
                    first_token_time = time.perf_counter()
                if text:
                    content_chunks.append(text)
                finish_reason = choices[0].get("finish_reason")
                if finish_reason:
                    print(f"[{label}] finish_reason={finish_reason}")

    total_time = time.perf_counter() - start
    content = "".join(content_chunks)
    needle_found = "FALCON-MERCURY-7749" in content
    ttft = (first_token_time - start) if first_token_time is not None else -1.0
    print(f"[{label}] DONE in {total_time:.1f}s, ttft={ttft:.1f}s")
    print(f"[{label}] content: {content!r}")
    print(f"[{label}] needle_found={needle_found} usage={usage}")
    return {
        "label": label,
        "total_time": total_time,
        "needle_found": needle_found,
        "content": content,
        "usage": usage,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash")
    ap.add_argument("--tokens", type=int, default=72000)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    results: list[dict[str, Any]] = []
    for i in range(args.runs):
        r = await run(args.base_url, args.model, args.tokens, label=f"run{i + 1}")
        results.append(r)
        await asyncio.sleep(2)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['label']}: time={r['total_time']:.1f}s needle={r['needle_found']}")


if __name__ == "__main__":
    asyncio.run(main())

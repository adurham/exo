#!/usr/bin/env python3
"""Section 17 memory-headroom check: fire N concurrent deep-context
requests against the live PP+batched-decode cluster and report whether
real memory pressure (active+wired, NOT the misleading "used%" that
includes reclaimable inactive/compressed pages -- see warm memory fact
650) stays under the iogpu.wired_limit_mb ceiling (115GB/node) on both
studios throughout.

Requires the cluster already running with EXO_PP_METAFRAME=1
EXO_PP_BATCHED_DECODE=1 (raises EXO_MAX_CONCURRENT_REQUESTS to 2 for
Pipeline mode -- see start_cluster.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import subprocess
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


def build_prompt(target_tokens: int, needle: str) -> str:
    target_chars = target_tokens * 4
    paragraphs: list[str] = []
    char_count = 0
    needle_placed = False
    needle_position = random.randint(target_chars // 3, 2 * target_chars // 3)
    while char_count < target_chars:
        if not needle_placed and char_count >= needle_position:
            paragraphs.append(needle)
            char_count += len(needle)
            needle_placed = True
        else:
            topic = random.choice(FILLER_TOPICS)
            paragraphs.append(topic)
            char_count += len(topic)
    paragraphs.append("\n\nWhat is the code embedded above? Answer with just the code.")
    return "\n\n".join(paragraphs)


def get_node_memory(host: str) -> dict[str, Any]:
    """Real resident footprint: active + wired pages via vm_stat, NOT the
    misleading dashboard "used%" that counts reclaimable inactive/compressed
    pages as used (warm memory fact 650)."""
    try:
        result = subprocess.run(
            ["ssh", host, "vm_stat"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        stats: dict[str, int] = {}
        page_size = 16384  # Apple Silicon default
        for line in lines:
            if "page size of" in line:
                # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                page_size = int(line.split("page size of")[1].split()[0])
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            val = val.strip().rstrip(".")
            if val.isdigit():
                stats[key.strip()] = int(val)
        active_gb = stats.get("Pages active", 0) * page_size / 1e9
        wired_gb = stats.get("Pages wired down", 0) * page_size / 1e9
        compressed_gb = stats.get("Pages occupied by compressor", 0) * page_size / 1e9
        return {
            "host": host,
            "active_gb": round(active_gb, 2),
            "wired_gb": round(wired_gb, 2),
            "compressed_gb": round(compressed_gb, 2),
            "resident_gb": round(active_gb + wired_gb, 2),
        }
    except Exception as e:
        return {"host": host, "error": str(e)}


async def memory_poller(
    hosts: list[str], interval: float, stop_event: asyncio.Event, samples: list[dict[str, Any]]
) -> None:
    while not stop_event.is_set():
        t = time.perf_counter()
        for host in hosts:
            mem = get_node_memory(host)
            mem["t"] = t
            samples.append(mem)
        await asyncio.sleep(interval)


async def run_request(
    base_url: str, model: str, target_tokens: int, needle: str, label: str
) -> dict[str, Any]:
    prompt = build_prompt(target_tokens, needle)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": 100,
    }
    start = time.perf_counter()
    content_chunks: list[str] = []
    finish_reason = None
    error_message = None
    print(f"[{label}] Prompt ~{len(prompt) // 4:,} tokens, sending...")
    try:
        async with httpx.AsyncClient() as client, client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=body, timeout=1200.0
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                import json as _json

                try:
                    chunk = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    error_message = chunk["error"].get("message", str(chunk["error"]))
                    break
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content") or ""
                    if text:
                        content_chunks.append(text)
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
    except Exception as e:
        error_message = str(e)

    elapsed = time.perf_counter() - start
    content = "".join(content_chunks)
    needle_found = needle.split(":")[-1].strip().rstrip(".") in content
    print(
        f"[{label}] DONE in {elapsed:.1f}s finish_reason={finish_reason} "
        f"error={error_message!r} needle_found={needle_found}"
    )
    return {
        "label": label,
        "elapsed": elapsed,
        "finish_reason": finish_reason,
        "error": error_message,
        "needle_found": needle_found,
        "content": content,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash")
    ap.add_argument("--tokens", type=int, default=150000)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument(
        "--hosts",
        default="macstudio-m4-1,macstudio-m4-2",
        help="Comma-separated SSH hosts to poll memory on",
    )
    ap.add_argument("--poll-interval", type=float, default=5.0)
    args = ap.parse_args()

    hosts = args.hosts.split(",")
    stop_event = asyncio.Event()
    samples: list[dict[str, Any]] = []
    poller_task = asyncio.create_task(
        memory_poller(hosts, args.poll_interval, stop_event, samples)
    )

    print(f"=== Section 17: {args.concurrency}x concurrent {args.tokens:,}-token requests ===")
    baseline = [get_node_memory(h) for h in hosts]
    print(f"Baseline memory: {baseline}")

    needles = [
        f"The secret code for stream {i} is: CODE-{i}-{random.randint(1000,9999)}."
        for i in range(args.concurrency)
    ]

    tasks = [
        run_request(args.base_url, args.model, args.tokens, needles[i], label=f"stream{i}")
        for i in range(args.concurrency)
    ]
    results = await asyncio.gather(*tasks)

    stop_event.set()
    await poller_task

    print("\n=== MEMORY SAMPLES OVER TIME ===")
    for s in samples:
        if "error" in s:
            print(f"  t={s['t']:.0f} {s['host']}: ERROR {s['error']}")
        else:
            print(
                f"  t={s['t']:.0f} {s['host']}: active={s['active_gb']:.1f}GB "
                f"wired={s['wired_gb']:.1f}GB resident={s['resident_gb']:.1f}GB "
                f"compressed={s['compressed_gb']:.1f}GB"
            )

    print("\n=== PEAK MEMORY PER NODE ===")
    for host in hosts:
        host_samples = [s for s in samples if s.get("host") == host and "error" not in s]
        if host_samples:
            peak = max(host_samples, key=lambda s: s["resident_gb"])
            print(f"  {host}: peak resident={peak['resident_gb']:.1f}GB (at t={peak['t']:.0f})")

    print("\n=== REQUEST RESULTS ===")
    for r in results:
        print(
            f"  {r['label']}: elapsed={r['elapsed']:.1f}s finish_reason={r['finish_reason']} "
            f"error={r['error']!r} needle_found={r['needle_found']}"
        )

    all_ok = all(r["finish_reason"] == "stop" and r["needle_found"] for r in results)
    print(f"\n=== OVERALL: {'PASS' if all_ok else 'FAIL/PARTIAL'} ===")


if __name__ == "__main__":
    asyncio.run(main())

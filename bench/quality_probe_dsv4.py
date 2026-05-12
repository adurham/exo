#!/usr/bin/env python3
"""DSv4 100K-context quality probe with needle-in-a-haystack.

Builds a ~100K-token prompt with a hidden secret code, sends it to whatever
DSv4 instance is currently placed on the cluster, and verifies the model can
recall the needle. Captures prefill time, decode rate, and full response text
to a JSON file so quality and perf can be compared across configs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

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


async def run_probe(args):
    prompt, expected = build_prompt(args.target_tokens)
    print(f"Prompt built: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens est)")
    print(f"Expected needle: {expected}")
    print(f"Sending to {args.base_url} model {args.model}")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": args.max_tokens,
    }
    response_chunks = []
    usage = {}
    start = time.perf_counter()
    first_token_time = None
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{args.base_url}/v1/chat/completions", json=body) as resp:
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
    generation_tokens = int(usage.get("completion_tokens", 0) or len(response.split()))
    found_needle = expected.lower() in response.lower()
    decode_tps = generation_tokens / (total_s - ttft) if (total_s - ttft) > 0 else 0.0
    prefill_tps = prompt_tokens / ttft if ttft > 0 else 0.0
    result = {
        "target_tokens": args.target_tokens,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "ttft_s": ttft,
        "total_s": total_s,
        "prefill_tps_apparent": prefill_tps,
        "decode_tps_apparent": decode_tps,
        "needle_found": found_needle,
        "expected_needle": expected,
        "response_text": response,
        "model": args.model,
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    print("\n=== RESULT ===")
    print(f"prompt_tokens: {prompt_tokens:,}")
    print(f"generation_tokens: {generation_tokens}")
    print(f"ttft (prefill): {ttft:.1f}s -> apparent prefill {prefill_tps:.1f} tok/s")
    print(f"total wall: {total_s:.1f}s")
    print(f"decode tps (apparent): {decode_tps:.2f}")
    print(f"needle_found: {found_needle}")
    print(f"response[:400]: {response[:400]!r}")
    if args.out:
        out_path = Path(os.path.expanduser(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
        existing.append(result)
        out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"appended result to {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    ap.add_argument("--target-tokens", type=int, default=100000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="unlabeled")
    args = ap.parse_args()
    asyncio.run(run_probe(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

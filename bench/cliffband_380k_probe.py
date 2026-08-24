#!/usr/bin/env python3
# 2026-08-24 cliff-band confirmation probe: copy of the phase3 precheck
# depth/throughput script with the stream timeout raised to 5400s so
# a single 381,619-token cold prefill completes without client-side cutoff.
# Used to produce the "381K @ 328.6 tok/s, needle PASS" live datapoint
# cited in docs/prefill-cliff-mechanism-2026-08-24.md and PERFORMANCE_HISTORY.md §3.2.
"""Pre-Phase-3 measurement checkpoint (per Fable's review, 2026-08-08):
measure decode tok/s and prefill tok/s at real ~500K context depth on
TODAY's Phase 2 code (EXO_PP_METAFRAME=1 EXO_PP_BATCHED_DECODE=1,
concurrency=1/single-session), against the real 671B DeepSeek-V4-Flash
checkpoint on the live 2-node cluster.

Why this exists: the design doc's requirement 3 ("30 tok/s decode @
500K context") was confirmed this session to mean PER-SESSION (as
defined by hermes-agent), not aggregate across concurrent requests.
Component 4 (micro-batch interleaving, the planned Phase 3 work) fills
the pipeline bubble across MULTIPLE concurrent streams -- it raises
AGGREGATE throughput, not a single session's decode rate. Before
spending effort building it, we need to know today's real per-session
number at depth: if it's already near/at 30, Phase 3 isn't the lever
that matters for requirement 3. If it's far below (e.g. Fable's
example: 12 tok/s), no amount of micro-batching fixes it -- 30 tok/s
per-session is a single-stream compute/bandwidth problem, not a
pipeline-utilization problem, and needs a different fix.

Needle-in-haystack verification (adapted from context_stress.py) so a
throughput number is never reported without confirming the output was
actually coherent, not garbage/degenerate.
"""
import argparse
import asyncio
import json
import os
import random
import time
from functools import lru_cache


@lru_cache(maxsize=4)
def _load_tokenizer(model: str):
    """Load the real tokenizer for ground-truth prompt token counts.

    Section 55 (2026-08-15): prefill tok/s must never be computed from a
    server-reported token count. Returns None if the tokenizer cannot be
    loaded, in which case the caller falls back to the old estimate and
    LOUDLY labels the number as an estimate rather than silently
    reporting it as measured.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    candidates = [model]
    # The HF repo id and the local MLX conversion share a tokenizer; fall
    # back to whichever DeepSeek-V4-Flash snapshot is actually on disk so
    # this works offline and regardless of which checkpoint is served.
    local_glob = os.path.expanduser(
        "~/.cache/huggingface/hub/models--*DeepSeek-V4-Flash*/snapshots/*"
    )
    import glob

    candidates.extend(sorted(glob.glob(local_glob)))

    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
        except Exception:  # noqa: BLE001 - any load failure means "try the next"
            continue
    return None


def count_prompt_tokens(model: str, prompt: str) -> int | None:
    """Ground-truth token count for ``prompt``, or None if unavailable."""
    tokenizer = _load_tokenizer(model)
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(prompt))
    except Exception:  # noqa: BLE001 - never let instrumentation kill a run
        return None

import httpx

FILLER_TOPICS = [
    "The observer pattern is a software design pattern in which an object, named the subject, maintains a list of its dependents, called observers, and notifies them automatically of any state changes, usually by calling one of their methods.",
    "A binary search tree is a rooted binary tree data structure with the key of each internal node being greater than all the keys in the respective node's left subtree and less than the ones in its right subtree.",
    "Garbage collection is a form of automatic memory management. The garbage collector attempts to reclaim memory which was allocated by the program but is no longer referenced.",
    "MapReduce is a programming model and an associated implementation for processing and generating big data sets with a parallel, distributed algorithm on a cluster.",
    "The CAP theorem states that any distributed data store can provide only two of the following three guarantees: consistency, availability, and partition tolerance.",
    "Functional programming is a programming paradigm where programs are constructed by applying and composing functions.",
    "A hash table is a data structure that implements an associative array or dictionary, mapping keys to values via a hash function.",
    "Consensus algorithms are fundamental to distributed computing. Paxos and Raft are two well-known consensus algorithms used in practice.",
    "The actor model is a mathematical model of concurrent computation that treats actor as the universal primitive of concurrent computation.",
    "B-trees are self-balancing tree data structures that maintain sorted data and allow searches, insertions, and deletions in logarithmic time.",
    "Type theory is the academic study of type systems, created to avoid paradoxes in a variety of formal logics and rewrite systems.",
    "Event sourcing is a software architecture pattern in which changes to application state are stored as a sequence of events.",
]

NEEDLE = "The secret code for project Nightingale is: FALCON-MERCURY-7749."


def build_prompt(target_tokens: int) -> tuple[str, str]:
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
            char_count += len(topic) + 2

    if not needle_placed:
        paragraphs.insert(len(paragraphs) // 2, NEEDLE)

    filler = "\n\n".join(paragraphs)
    prompt = (
        "I'm going to give you a very long document. Read it carefully. "
        "At the end, I'll ask you a question about a specific detail buried in the text.\n\n"
        "--- BEGIN DOCUMENT ---\n\n"
        f"{filler}\n\n"
        "--- END DOCUMENT ---\n\n"
        "Question: What is the secret code for project Nightingale? "
        "Answer with just the code, nothing else."
    )
    return prompt, "FALCON-MERCURY-7749"


async def measure(base_url: str, model: str, target_tokens: int, max_tokens: int) -> dict:
    print(f"\n{'='*70}")
    print(f"Measuring {target_tokens:,}-token context, model={model}")
    print(f"{'='*70}")

    prompt, expected = build_prompt(target_tokens)
    prompt_chars = len(prompt)
    est_tokens = prompt_chars // 4
    ground_truth_prompt_tokens = count_prompt_tokens(model, prompt)
    if ground_truth_prompt_tokens is not None:
        print(
            f"Prompt: {prompt_chars:,} chars, {ground_truth_prompt_tokens:,} tokens "
            f"(tokenizer ground truth; chars//4 estimate would say "
            f"{est_tokens:,}, off by "
            f"{est_tokens / ground_truth_prompt_tokens:.2f}x)"
        )
    else:
        print(
            f"Prompt: {prompt_chars:,} chars (~{est_tokens:,} tokens) "
            f"-- WARNING: tokenizer unavailable, prefill tok/s will be an "
            f"ESTIMATE and is NOT comparable to tokenizer-based runs"
        )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    first_token_time: float | None = None
    response_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: dict = {}
    token_timestamps: list[float] = []

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST", f"{base_url}/v1/chat/completions", json=body, timeout=5400.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        if "prefill_progress" in line:
                            try:
                                prog = json.loads(line.split(" ", 1)[1])
                                chunk = prog.get("PrefillProgressChunk", {})
                                processed = chunk.get("processed_tokens", 0)
                                total = chunk.get("total_tokens", 0)
                                if total > 0:
                                    elapsed = time.perf_counter() - start
                                    print(f"  Prefill: {processed:,}/{total:,} tokens ({elapsed:.1f}s)", end="\r", flush=True)
                            except (json.JSONDecodeError, IndexError):
                                pass
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if "usage" in chunk and chunk["usage"]:
                        usage = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    now = time.perf_counter()
                    if content:
                        if first_token_time is None:
                            first_token_time = now
                        response_chunks.append(content)
                        token_timestamps.append(now)
                    elif reasoning:
                        if first_token_time is None:
                            first_token_time = now
                        reasoning_chunks.append(reasoning)
                        token_timestamps.append(now)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            return {"target_tokens": target_tokens, "error": str(e)}

    end = time.perf_counter()
    response = "".join(response_chunks)
    reasoning = "".join(reasoning_chunks)
    ttft_s = (first_token_time - start) if first_token_time else 0
    total_s = end - start
    decode_s = total_s - ttft_s

    # Section 55 (2026-08-15): NEVER derive throughput from a
    # server-reported token count. Three separate bugs in this campaign
    # all reduce to the same root -- a tok/s number whose numerator was
    # an API field that silently changed definition underneath it:
    #
    #   * usage.prompt_tokens reported the prompt TAIL (fixed 7d14daea7)
    #   * generation_tps reported machine uptime as generation time
    #   * this harness's own `usage.get("prompt_tokens", est_tokens)`,
    #     which fell back to `chars // 4` whenever the field was
    #     unusable -- and `chars // 4` overcounts this prompt by 1.42x
    #     (real ratio ~5.68 chars/token). That single fallback produced
    #     the phantom "30% prefill regression" between Sections 50-51
    #     and 52; renormalized, the two run sets agreed within 0.4%.
    #
    # `ground_truth_prompt_tokens` is computed offline from the real
    # tokenizer against the exact prompt string we sent, so no
    # server-side accounting change can move a throughput number again.
    # The API's own counts are still RECORDED (for cross-checking and to
    # catch future accounting drift) but are never load-bearing.
    api_prompt_tokens = usage.get("prompt_tokens")
    prompt_tokens = (
        ground_truth_prompt_tokens
        if ground_truth_prompt_tokens is not None
        else usage.get("prompt_tokens", est_tokens)
    )
    # Completion tokens counted locally from the stream: one increment
    # per streamed content/reasoning delta, independent of usage.*.
    completion_tokens = len(token_timestamps)
    api_completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

    found_needle = expected.lower() in response.lower() or expected.lower() in reasoning.lower()
    prefill_tps = prompt_tokens / ttft_s if ttft_s > 0 else 0
    decode_tps = completion_tokens / decode_s if decode_s > 0 else 0

    token_source = "tokenizer" if ground_truth_prompt_tokens is not None else "ESTIMATE"
    print(f"\n  Prompt tokens: {prompt_tokens:,} (source: {token_source})")
    if api_prompt_tokens is not None and api_prompt_tokens != prompt_tokens:
        print(
            f"  WARNING: API reported prompt_tokens={api_prompt_tokens:,}, "
            f"ground truth is {prompt_tokens:,} -- accounting drift, "
            f"throughput below uses ground truth"
        )
    print(f"  TTFT (prefill time): {ttft_s:.1f}s  -> prefill throughput: {prefill_tps:.1f} tok/s")
    print(f"  Completion tokens: {completion_tokens} (reasoning: {reasoning_tokens})")
    if api_completion_tokens is not None and api_completion_tokens != completion_tokens:
        print(
            f"  WARNING: API reported completion_tokens={api_completion_tokens}, "
            f"locally counted {completion_tokens}"
        )
    print(f"  Decode time: {decode_s:.1f}s -> decode throughput: {decode_tps:.2f} tok/s")
    print(f"  Response: {response[:200]!r}")
    print(f"  Needle found: {'YES' if found_needle else 'NO'}")
    print(f"  Total: {total_s:.1f}s")

    return {
        "target_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_token_source": token_source,
        "api_prompt_tokens": api_prompt_tokens,
        "completion_tokens": completion_tokens,
        "api_completion_tokens": api_completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "ttft_s": ttft_s,
        "prefill_tok_s": prefill_tps,
        "decode_s": decode_s,
        "decode_tok_s": decode_tps,
        "total_s": total_s,
        "response": response,
        "needle_found": found_needle,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash")
    ap.add_argument("--targets", default="100000,300000,500000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--json-out", default="/tmp/phase3_precheck_results.json")
    args = ap.parse_args()

    targets = [int(x) for x in args.targets.split(",")]
    results = []
    for t in targets:
        r = await measure(args.base_url, args.model, t, args.max_tokens)
        results.append(r)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        if "error" in r:
            print(f"  {r['target_tokens']:>7,} tokens: ERROR - {r['error']}")
        else:
            needle = "OK" if r["needle_found"] else "FAIL"
            print(
                f"  {r['target_tokens']:>7,} tokens: prefill={r['prefill_tok_s']:.1f} tok/s "
                f"decode={r['decode_tok_s']:.2f} tok/s  needle={needle}"
            )
    print(f"\nWritten to {args.json_out}")


if __name__ == "__main__":
    asyncio.run(main())

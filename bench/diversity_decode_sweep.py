#!/usr/bin/env python3
"""Does per-token decode cost depend on prompt CONTENT DIVERSITY?

Section 57 retracted the claim that `EXO_PREFILL_STEP_SIZE=2048` caused
this campaign's 50x decode collapse. The control that killed it: at the
same config and nearly the same token count, two prompts behaved
oppositely --

    official harness prompt : 14,133 tok, 398 unique blocks -> 0.49 tok/s
    degenerate probe prompt : 12,177 tok,  30 unique blocks -> 23.74 tok/s

Length was controlled. Config was controlled. Only CONTENT differed.

This sweep isolates that one variable properly. It holds the token
count fixed (within a tight tolerance) and varies ONLY the number of
distinct filler paragraphs, from pathologically repetitive to fully
varied. If decode tok/s tracks diversity, the leading hypothesis is
supported: DeepSeek-V4's sparse indexer attention
(`EXO_DSV4_INDEX_TOPK=512`, `sliding_window=128`) selects top-K over
distinct key blocks, so a context with few distinct blocks is cheap and
a varied one is not.

Methodology notes, learned the hard way in this campaign:
  * Reports the per-token gap DISTRIBUTION, never a bare mean. An 18.01
    tok/s figure from a 52-token sample already misled this campaign
    once (Section 52).
  * Counts tokens with the real tokenizer offline, never from
    `usage.*` -- that field family has had two separate real bugs
    (Sections 50, and the generation_tps uptime bug).
  * Prints the generated answer. A throughput number without the output
    it describes is how a "win" shipped that emitted zero tokens
    (Section 51).
  * Padding to the target length uses the SAME pool of distinct
    paragraphs already chosen for that arm, so raising length never
    smuggles in extra diversity.

Usage:
  python bench/diversity_decode_sweep.py                 # default sweep
  python bench/diversity_decode_sweep.py --target-tokens 14000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
NEEDLE = "The access code for project Wintergreen is 55831."
NEEDLE_ANSWER = "55831"

# Distinct, non-repeating paragraph templates. Rendered with varying
# subjects/numbers so a large pool is genuinely diverse rather than
# near-duplicate.
_TEMPLATES = [
    "Subsystem {n} completed its {a} calibration cycle with all telemetry channels nominal and no operator intervention required.",
    "The {a} survey of sector {n} recorded ambient drift within tolerance, though channel {n} showed brief variance during the transition window.",
    "Maintenance crew {n} logged a {a} inspection of the auxiliary coolant loop and replaced two seals ahead of the scheduled interval.",
    "Archive record {n} describes a {a} anomaly in the relay network that resolved without escalation after the backup path engaged.",
    "During the {a} review, analysts noted that buffer {n} sustained higher throughput than modeled and recommended revising the baseline.",
    "Observation post {n} transmitted a {a} summary indicating stable atmospheric readings across the monitored corridor.",
    "The engineering board accepted proposal {n}, a {a} revision to the thermal margin policy affecting downstream scheduling.",
    "Incident {n} was reclassified as a {a} event after the root-cause review found no correlation with the preceding maintenance window.",
    "Depot {n} reported {a} inventory reconciliation, with three line items pending verification against the manifest.",
    "Simulation run {n} produced a {a} divergence from the reference trajectory, attributed to updated drag coefficients.",
]
_ADJECTIVES = [
    "routine", "quarterly", "unscheduled", "preliminary", "comprehensive",
    "abbreviated", "independent", "joint", "provisional", "final",
]


def make_paragraph_pool(n_unique: int, seed: int = 1234) -> list[str]:
    """Build exactly ``n_unique`` DISTINCT paragraphs."""
    rng = random.Random(seed)
    pool: list[str] = []
    i = 0
    while len(pool) < n_unique:
        template = _TEMPLATES[i % len(_TEMPLATES)]
        para = template.format(n=1000 + i, a=_ADJECTIVES[(i // len(_TEMPLATES)) % len(_ADJECTIVES)])
        if para not in pool:
            pool.append(para)
        i += 1
        if i > n_unique * 50:  # pathological guard
            break
    rng.shuffle(pool)
    return pool


def build_prompt(n_unique: int, target_tokens: int, tokenizer) -> tuple[str, int]:
    """Prompt with exactly ``n_unique`` distinct paragraphs, cycled to hit
    ``target_tokens``. Returns (prompt, real_token_count).

    Padding cycles the SAME pool -- never introduces new distinct
    content -- so diversity is genuinely the only variable across arms.
    """
    pool = make_paragraph_pool(n_unique)

    def render(count: int) -> str:
        body = "\n\n".join(pool[i % len(pool)] for i in range(count))
        mid = body.find("\n\n", len(body) // 2)
        if mid == -1:
            mid = len(body) // 2
        return (
            "Read the following log carefully. A specific detail is buried in it.\n\n"
            "--- BEGIN LOG ---\n\n"
            + body[:mid] + "\n\n" + NEEDLE + "\n\n" + body[mid:]
            + "\n\n--- END LOG ---\n\n"
            "Question: What is the access code for project Wintergreen? "
            "Answer with just the code, nothing else."
        )

    # Binary-search the paragraph count that lands on target_tokens.
    lo, hi = 1, 200_000
    best: tuple[str, int] | None = None
    while lo <= hi:
        mid_count = (lo + hi) // 2
        prompt = render(mid_count)
        n_tok = len(tokenizer.encode(prompt))
        if best is None or abs(n_tok - target_tokens) < abs(best[1] - target_tokens):
            best = (prompt, n_tok)
        if n_tok < target_tokens:
            lo = mid_count + 1
        elif n_tok > target_tokens:
            hi = mid_count - 1
        else:
            return prompt, n_tok
    assert best is not None
    return best


def unique_blocks(text: str, size: int = 200) -> int:
    return len({text[i : i + size] for i in range(0, len(text), size)})


async def measure(base_url: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": max_tokens,
    }
    start = time.perf_counter()
    first: float | None = None
    stamps: list[float] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish: str | None = None

    async with httpx.AsyncClient() as client, client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body, timeout=3600.0
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if not content and not reasoning:
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            stamps.append(now)
            (text_parts if content else reasoning_parts).append(content or reasoning)

    answer = "".join(text_parts)
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    return {
        "ttft_s": (first - start) if first else 0.0,
        "gaps": gaps,
        "answer": answer,
        "reasoning_head": "".join(reasoning_parts)[:200],
        "finish_reason": finish,
        "needle_found": NEEDLE_ANSWER in answer or NEEDLE_ANSWER in "".join(reasoning_parts),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    ap.add_argument("--target-tokens", type=int, default=14000)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument(
        "--unique",
        default="1,4,16,64,256,1024",
        help="comma-separated distinct-paragraph counts",
    )
    ap.add_argument("--json-out", default="/tmp/diversity_sweep.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = None
    import glob
    import os

    for cand in [MODEL] + sorted(
        glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*DeepSeek-V4-Flash*/snapshots/*"))
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
            break
        except Exception:  # noqa: BLE001
            continue
    if tokenizer is None:
        raise SystemExit("no tokenizer available -- cannot control token length")

    print(f"Diversity sweep @ ~{args.target_tokens:,} tokens (length held FIXED)")
    print(f"{'uniq_para':>10} {'tokens':>8} {'blocks':>7} {'ttft_s':>8} "
          f"{'p50_ms':>9} {'tok/s':>7} {'slow%':>6}  needle")
    print("-" * 78)

    results = []
    for n_unique in [int(x) for x in args.unique.split(",")]:
        prompt, n_tok = build_prompt(n_unique, args.target_tokens, tokenizer)
        blocks = unique_blocks(prompt)
        try:
            r = await measure(args.base_url, prompt, args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"{n_unique:>10} {n_tok:>8,}  ERROR: {exc}")
            continue
        gaps = r["gaps"]
        if not gaps:
            print(f"{n_unique:>10} {n_tok:>8,} {blocks:>7}  no gaps (finish={r['finish_reason']})")
            continue
        p50 = statistics.median(gaps)
        slow_pct = 100.0 * len([g for g in gaps if g > 0.5]) / len(gaps)
        print(f"{n_unique:>10} {n_tok:>8,} {blocks:>7} {r['ttft_s']:>8.1f} "
              f"{p50 * 1000:>9.1f} {1 / p50:>7.2f} {slow_pct:>5.0f}%  {r['needle_found']}")
        results.append({
            "unique_paragraphs": n_unique,
            "real_tokens": n_tok,
            "unique_blocks": blocks,
            "ttft_s": r["ttft_s"],
            "p50_gap_ms": p50 * 1000,
            "decode_tok_s": 1 / p50,
            "slow_gap_pct": slow_pct,
            "n_gaps": len(gaps),
            "finish_reason": r["finish_reason"],
            "needle_found": r["needle_found"],
            "answer": r["answer"][:200],
        })

    with open(args.json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.json_out}")

    if len(results) >= 2:
        fastest = max(results, key=lambda r: r["decode_tok_s"])
        slowest = min(results, key=lambda r: r["decode_tok_s"])
        spread = fastest["decode_tok_s"] / slowest["decode_tok_s"]
        tok_spread = max(r["real_tokens"] for r in results) / min(r["real_tokens"] for r in results)
        print(f"\nspread: {spread:.1f}x  "
              f"(fastest {fastest['unique_paragraphs']} uniq -> {fastest['decode_tok_s']:.2f} tok/s, "
              f"slowest {slowest['unique_paragraphs']} uniq -> {slowest['decode_tok_s']:.2f} tok/s)")
        print(f"token-count spread across arms: {tok_spread:.3f}x (want ~1.00 -- length is controlled)")
        if spread > 3 and tok_spread < 1.15:
            print("VERDICT: decode cost tracks CONTENT DIVERSITY at fixed length.")
        elif spread < 1.5:
            print("VERDICT: diversity does NOT explain the collapse -- hypothesis refuted.")
        else:
            print("VERDICT: inconclusive -- inspect the curve.")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Per-token decode latency probe -- is the batched-decode cost FIXED or depth-scaled?

Section 52/53 measured ~1.86 s/tok at 100K and ~2.17 s/tok at 300K context on the
PP batched-decode path. That is essentially FLAT across a 3x context increase, which
is the signature of a FIXED per-token overhead rather than attention/KV scaling.

If the cost is genuinely fixed, it must also be present at SHORT context -- which is
testable in seconds instead of the ~7.5 minutes a 100K prefill costs.

This probe streams a request and records the wall-clock delta between every
consecutive streamed token, then reports the full distribution (not a mean, and not
a sub-100-token sample presented as a throughput number -- both of those produced
wrong conclusions earlier in this campaign).

Deliberately does NOT use usage.prompt_tokens for anything load-bearing: that field
had a real bug (reported the prompt TAIL, fixed at exo@7d14daea7) and every
throughput number in this campaign that depended on it is now suspect. Token counts
here are counted locally from the stream.

Usage:
  python bench/pertoken_latency_probe.py --host adams-mac-studio-m4-1.local
  python bench/pertoken_latency_probe.py --ctx 5000 --max-tokens 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"


def build_prompt(target_ctx_tokens: int) -> tuple[str, str]:
    """Return (prompt, expected_needle). ~4 chars/token is this fork's usual estimate."""
    if target_ctx_tokens <= 0:
        return ("What is the capital of France? Answer in one word.", "paris")

    needle = "The launch code for the Kestrel satellite is 84213."
    filler_unit = (
        "The maintenance log records routine telemetry checks and nominal "
        "system status across all monitored subsystems. "
    )
    approx_chars = target_ctx_tokens * 4
    body = filler_unit * (approx_chars // len(filler_unit) + 1)
    mid = len(body) // 2
    prompt = (
        body[:mid]
        + "\n\n"
        + needle
        + "\n\n"
        + body[mid:approx_chars]
        + "\n\nWhat is the launch code for the Kestrel satellite? Answer with the number only."
    )
    return (prompt, "84213")


async def probe(
    base_url: str, model: str, prompt: str, expected: str, max_tokens: int
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": True,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    first_token_time: float | None = None
    stamps: list[float] = []
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage: dict = {}
    finish_reason: str | None = None

    async with httpx.AsyncClient() as client, client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body, timeout=1800.0
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
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if not content and not reasoning:
                continue
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            stamps.append(now)
            if content:
                content_chunks.append(content)
            else:
                reasoning_chunks.append(reasoning)

    end = time.perf_counter()
    text = "".join(content_chunks)
    reasoning_text = "".join(reasoning_chunks)

    ttft = (first_token_time - start) if first_token_time else 0.0
    # Inter-token gaps: exclude TTFT entirely, it is prefill cost not decode cost.
    gaps = [(b - a) for a, b in zip(stamps, stamps[1:], strict=False)]

    return {
        "ttft_s": ttft,
        "total_s": end - start,
        "streamed_events": len(stamps),
        "gaps": gaps,
        "finish_reason": finish_reason,
        "needle_found": (
            expected.lower() in text.lower() or expected.lower() in reasoning_text.lower()
        ),
        "text": text,
        "reasoning_head": reasoning_text[:400],
        "usage": usage,
    }


def report(label: str, r: dict) -> None:
    gaps = r["gaps"]
    print(f"\n=== {label} ===")
    print(f"  TTFT (prefill):    {r['ttft_s']:.2f}s")
    print(f"  total wall clock:  {r['total_s']:.2f}s")
    print(f"  streamed events:   {r['streamed_events']}")
    print(f"  finish_reason:     {r['finish_reason']}")
    print(f"  needle found:      {r['needle_found']}")
    print(f"  usage (untrusted): {r['usage']}")
    # Always show the actual generated text. A throughput number without
    # the output it describes is exactly how this campaign shipped a
    # "win" whose generation emitted zero tokens (Section 51).
    print(f"  answer text:       {r['text'][:300]!r}")
    print(f"  reasoning head:    {r['reasoning_head'][:300]!r}")

    if not gaps:
        print("  NO inter-token gaps -- fewer than 2 streamed events.")
        return

    ordered = sorted(gaps)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(len(ordered) * p))
        return ordered[idx]

    print(f"  inter-token gaps (n={len(gaps)}):")
    print(f"    min    {min(gaps) * 1000:9.1f} ms")
    print(f"    p50    {statistics.median(gaps) * 1000:9.1f} ms")
    print(f"    p90    {pct(0.90) * 1000:9.1f} ms")
    print(f"    p99    {pct(0.99) * 1000:9.1f} ms")
    print(f"    max    {max(gaps) * 1000:9.1f} ms")
    print(f"    mean   {statistics.fmean(gaps) * 1000:9.1f} ms")
    print(f"  implied steady-state decode: {1.0 / statistics.median(gaps):.2f} tok/s (from p50 gap)")

    # The diagnostic that matters: is the cost uniform (fixed overhead on every
    # token) or bimodal (most tokens fast, a few catastrophically slow)?
    slow = [g for g in gaps if g > 0.5]
    print(f"  gaps >500ms: {len(slow)} / {len(gaps)} ({100.0 * len(slow) / len(gaps):.1f}%)")
    if statistics.median(gaps) > 0.5:
        print("  SHAPE: uniformly slow -> fixed per-token overhead on EVERY token.")
    elif slow:
        print("  SHAPE: bimodal -> fast baseline punctuated by stalls.")
    else:
        print("  SHAPE: uniformly fast -> no fixed overhead at this context depth.")

    print(f"  first 30 gaps (ms): {[round(g * 1000, 1) for g in gaps[:30]]}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="adams-mac-studio-m4-1.local")
    ap.add_argument("--port", type=int, default=52415)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--ctx",
        type=int,
        action="append",
        help="target context depth in tokens; repeatable; 0 = short prompt",
    )
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    depths = args.ctx if args.ctx else [0]
    base_url = f"http://{args.host}:{args.port}"
    print(f"probing {base_url} model={args.model} max_tokens={args.max_tokens}")

    for depth in depths:
        prompt, expected = build_prompt(depth)
        label = "short prompt" if depth <= 0 else f"~{depth:,} token context"
        print(f"\nrunning {label} ({len(prompt):,} chars)...")
        try:
            r = await probe(base_url, args.model, prompt, expected, args.max_tokens)
        except Exception as exc:  # noqa: BLE001 - probe should report, not crash
            print(f"  ERROR: {exc}")
            continue
        report(label, r)


if __name__ == "__main__":
    asyncio.run(main())

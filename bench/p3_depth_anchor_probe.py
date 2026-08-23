#!/usr/bin/env python3
"""P3 worker B1: live decode-throughput depth anchors with a REAL EOS ban.

WHY A NEW SCRIPT (2026-08-23):

Neither existing probe can produce the anchor this phase needs:

  * bench/decode_probe.py posts ``{"bench": true}`` to /v1/chat/completions.
    ChatCompletionRequest has NO ``bench`` field, so pydantic DROPS it and the
    request runs as a normal completion -- EOS is NOT banned. Verified live
    2026-08-23: /v1 + bench:true returned finish_reason="stop" with
    completion_tokens=56 < max_tokens=60, while /bench/chat/completions
    returned finish_reason="length" with completion_tokens=60 == max_tokens.
    This is exactly the failure that invalidated the prior T5 capture down to a
    ~9s decode window. decode_probe.py also records no usage and no gaps.
  * bench/pertoken_latency_probe.py records usage + the full inter-token gap
    distribution, but posts to /v1 as well, so a thinking model can EOS out
    early and collapse the decode window.

This script = pertoken_latency_probe.py's measurement methodology pointed at
the endpoint that actually bans EOS (/bench/chat/completions, which sets
task_params.bench=True -> ban_token_ids(eos_ids) in batch_generate.py:2658).

METHODOLOGY (inherited from decode_depth_sweep.py / pertoken_latency_probe.py):
  * Prompt depth is TARGETED with the real model tokenizer locally, then the
    authoritative depth is read back from usage.prompt_tokens (the field bug
    that reported the prompt TAIL was fixed at exo@7d14daea7, which is an
    ancestor of the live HEAD).
  * Unique nonce at the FRONT of every prompt + use_prefix_cache=False, so a
    KV prefix-cache hit cannot silently turn a deep measurement into a shallow
    one.
  * Non-degenerate filler (decode_depth_sweep._filler), not one repeated
    sentence: a degenerate prompt exercises different routing behaviour.
  * TTFT (prefill) is excluded from the decode window by construction; the
    window is last_streamed_event - first_streamed_event.
  * Full inter-token gap distribution reported, never a lone mean.
  * READ-ONLY with respect to the cluster: this script issues exactly one HTTP
    POST per depth point and nothing else. No restarts, no instance mutation,
    no admin endpoints.

Usage:
  python bench/p3_depth_anchor_probe.py --target-tokens 512 --max-tokens 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid

import httpx

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_HOST = "adams-mac-studio-m4-1.local"
DEFAULT_PORT = 52415
TOKENIZER_DIR = "/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash"

# Measured live 2026-08-23: text "Say hello." = 3 tokenizer tokens,
# usage.prompt_tokens = 7 -> the chat template adds 4 tokens.
CHAT_TEMPLATE_OVERHEAD = 4

_TOPICS = [
    "distributed inference schedulers allocate pipeline stages across nodes",
    "thunderbolt fabrics expose queue pairs with distinct completion semantics",
    "mixture of experts routing selects a sparse subset of feedforward blocks",
    "key value caches grow linearly with sequence length during autoregressive decode",
    "quantised weight formats trade numerical precision for memory bandwidth",
    "speculative decoding drafts several tokens before a single verification pass",
    "unified memory architectures remove explicit host to device staging copies",
    "attention indexers rank historical positions before gathering a top subset",
]


def _paragraph(i: int) -> str:
    t = _TOPICS[i % len(_TOPICS)]
    return (
        f"Section {i}. In practice {t}; the resulting behaviour depends on "
        f"configuration {i * 7 % 97} and on the observed interaction between "
        f"stage {i % 11} and stage {(i * 3) % 13} of the surrounding system."
    )


def build_prompt(target_tokens: int, tokenizer) -> tuple[str, int]:
    """Binary-search paragraph count so usage.prompt_tokens lands on target.

    Returns (prompt, locally_predicted_prompt_tokens). The prediction is a
    sanity check only -- the reported depth is always usage.prompt_tokens.
    """
    nonce = uuid.uuid4().hex
    header = f"Reference identifier {nonce}. Corpus follows.\n\n"
    tail = "\n\nBriefly summarise the corpus above."

    def total_for(n_paragraphs: int) -> tuple[str, int]:
        body = " ".join(_paragraph(i) for i in range(n_paragraphs))
        text = header + body + tail
        n = len(tokenizer.encode(text, add_special_tokens=False))
        return text, n + CHAT_TEMPLATE_OVERHEAD

    # Cheap linear calibration, then binary search.
    _, per_100 = total_for(100)
    _, per_0 = total_for(0)
    per_para = max(1.0, (per_100 - per_0) / 100.0)
    lo, hi = 0, max(1, int(target_tokens / per_para * 1.4) + 8)
    best_text, best_n = total_for(hi)
    while lo < hi:
        mid = (lo + hi) // 2
        text, n = total_for(mid)
        if n < target_tokens:
            lo = mid + 1
        else:
            best_text, best_n, hi = text, n, mid
    return best_text, best_n


async def probe(
    base_url: str, model: str, prompt: str, max_tokens: int
) -> dict:
    """One streamed request against the EOS-BANNING /bench endpoint."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # BenchChatCompletionRequest-only field. Its presence is what forces
        # task_params.bench = True server-side, which bans EOS.
        "use_prefix_cache": False,
    }

    start = time.perf_counter()
    stamps: list[float] = []
    text_parts: list[str] = []
    usage: dict = {}
    finish_reason: str | None = None

    async with httpx.AsyncClient() as client, client.stream(
        "POST", f"{base_url}/bench/chat/completions", json=body, timeout=7200.0
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
            piece = (delta.get("content") or "") + (
                delta.get("reasoning_content") or ""
            )
            if not piece:
                continue
            stamps.append(time.perf_counter())
            text_parts.append(piece)

    end = time.perf_counter()
    ttft = (stamps[0] - start) if stamps else 0.0
    decode_window = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]

    n_events = len(stamps)
    completion_tokens = usage.get("completion_tokens")
    # Decode-only throughput. The window starts at the FIRST streamed event, so
    # the token that produced that event is not inside the window -> n-1.
    tok_s_events = (n_events - 1) / decode_window if decode_window > 0 else 0.0
    tok_s_usage = (
        (completion_tokens - 1) / decode_window
        if decode_window > 0 and completion_tokens
        else 0.0
    )

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "usage": usage,
        "finish_reason": finish_reason,
        "ttft_s": ttft,
        "decode_window_s": decode_window,
        "total_s": end - start,
        "streamed_events": n_events,
        "decode_tok_s_events": tok_s_events,
        "decode_tok_s_usage": tok_s_usage,
        "ms_per_token_events": (1000.0 / tok_s_events) if tok_s_events else 0.0,
        "ms_per_token_usage": (1000.0 / tok_s_usage) if tok_s_usage else 0.0,
        "gaps": gaps,
        "text_head": "".join(text_parts)[:300],
        "text_tail": "".join(text_parts)[-200:],
    }


def report(label: str, r: dict, target: int, predicted: int) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(f"  target_tokens:        {target:,} (locally predicted {predicted:,})")
    print(f"  REAL prompt_tokens:   {r['prompt_tokens']}")
    print(f"  completion_tokens:    {r['completion_tokens']}")
    print(f"  finish_reason:        {r['finish_reason']}  "
          f"(must be 'length' -- EOS banned)")
    print(f"  streamed events:      {r['streamed_events']}")
    print(f"  TTFT (prefill):       {r['ttft_s']:.2f}s")
    print(f"  DECODE WINDOW:        {r['decode_window_s']:.2f}s")
    print(f"  total wall clock:     {r['total_s']:.2f}s")
    print(f"  decode tok/s (usage): {r['decode_tok_s_usage']:.2f}  "
          f"-> {r['ms_per_token_usage']:.2f} ms/tok")
    print(f"  decode tok/s (events):{r['decode_tok_s_events']:.2f}  "
          f"-> {r['ms_per_token_events']:.2f} ms/tok")
    print(f"  full usage:           {r['usage']}")
    print(f"  text head:            {r['text_head'][:200]!r}")
    print(f"  text tail:            {r['text_tail'][:120]!r}")

    gaps = r["gaps"]
    if not gaps:
        print("  NO inter-token gaps -- fewer than 2 streamed events.")
        return
    ordered = sorted(gaps)

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    print(f"  inter-token gap distribution (n={len(gaps)}):")
    for name, v in (
        ("min", min(gaps)), ("p10", pct(0.10)), ("p50", statistics.median(gaps)),
        ("p90", pct(0.90)), ("p99", pct(0.99)), ("max", max(gaps)),
        ("mean", statistics.fmean(gaps)),
    ):
        print(f"    {name:<5} {v * 1000:9.2f} ms")
    print(f"    stdev {statistics.pstdev(gaps) * 1000:9.2f} ms")
    print(f"  implied steady-state from p50 gap: "
          f"{1.0 / statistics.median(gaps):.2f} tok/s")
    slow = [g for g in gaps if g > 3 * statistics.median(gaps)]
    print(f"  gaps > 3x median: {len(slow)} / {len(gaps)} "
          f"({100.0 * len(slow) / len(gaps):.2f}%)")
    if slow:
        print(f"  slow-gap values (ms, first 20): "
              f"{[round(g * 1000, 1) for g in sorted(slow, reverse=True)[:20]]}")
    print(f"  first 25 gaps (ms): {[round(g * 1000, 1) for g in gaps[:25]]}")
    print(f"  last 25 gaps (ms):  {[round(g * 1000, 1) for g in gaps[-25:]]}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--target-tokens", type=int, action="append", required=True,
                    help="real prompt-token depth; repeatable")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--out", default="/tmp/p3_depth_anchor.json")
    ap.add_argument("--depth-cap", type=int, default=355_000,
                    help="hard safety cap on real prompt depth")
    args = ap.parse_args()

    for t in args.target_tokens:
        if t > args.depth_cap:
            raise SystemExit(
                f"REFUSING target {t:,} > depth cap {args.depth_cap:,}: "
                "deeper than the max previously-validated KV footprint on the "
                "live cluster."
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    base_url = f"http://{args.host}:{args.port}"
    print(f"probing {base_url}/bench/chat/completions model={args.model} "
          f"max_tokens={args.max_tokens}", flush=True)

    out: list[dict] = []
    for target in args.target_tokens:
        prompt, predicted = build_prompt(target, tokenizer)
        label = f"depth target ~{target:,} tokens"
        print(f"\nbuilding {label}: {len(prompt):,} chars, "
              f"locally predicted {predicted:,} tokens", flush=True)
        t0 = time.time()
        try:
            r = await probe(base_url, args.model, prompt, args.max_tokens)
        except Exception as exc:  # noqa: BLE001 - probe reports, never crashes
            print(f"  ERROR after {time.time() - t0:.0f}s: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            out.append({"target": target, "error": f"{type(exc).__name__}: {exc}"})
            continue
        report(label, r, target, predicted)
        r["target"] = target
        r["predicted_prompt_tokens"] = predicted
        out.append(r)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  [wrote {args.out}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

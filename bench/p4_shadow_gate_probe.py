#!/usr/bin/env python3
"""P4v2 M1 harness: byte-identity gate + decode-window capture for DSpark shadow mode.

Two jobs, both against the live cluster's OpenAI-compatible API:

  1. ``--mode identity`` — emit a temp=0 generation for a FIXED prompt at a
     given context depth and write the raw text to a file. Run once per build
     (production vs shadow) and ``diff`` the two files. Byte-identical output
     is the M1 correctness precondition: shadow mode must emit exactly the
     sequential-decode token stream.

  2. ``--mode decode`` — the p3_depth_anchor_probe measurement (EOS-banned
     /bench endpoint, decode window = last_event - first_event) so the shadow
     build's own tok/s can be compared against the non-spec anchor.

Determinism notes:
  * ``/bench/chat/completions`` bans EOS, so the completion always runs to
    exactly ``max_tokens`` — a fixed-length window makes the byte diff a real
    comparison rather than an early-stop coincidence.
  * The prompt is built from a FIXED seed (no uuid nonce) so the same depth is
    reproduced across builds; ``use_prefix_cache=False`` keeps a cache hit from
    silently shortening a deep run.
  * temperature=0.0.

READ-ONLY w.r.t. the cluster: one HTTP POST per invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_HOST = "adams-mac-studio-m4-1.local"
DEFAULT_PORT = 52415
TOKENIZER_DIR = "/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash"
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
    """Deterministic corpus prompt sized to ``target_tokens``.

    Unlike p3_depth_anchor_probe there is NO uuid nonce: the same depth must
    reproduce byte-for-byte across two different cluster builds so the emitted
    text can be diffed. ``use_prefix_cache=False`` on the request side is what
    keeps a stale prefix cache from turning a deep probe shallow.
    """
    header = "Reference identifier p4v2-shadow-gate-fixed. Corpus follows.\n\n"
    tail = "\n\nBriefly summarise the corpus above."

    def total_for(n_paragraphs: int) -> tuple[str, int]:
        body = " ".join(_paragraph(i) for i in range(n_paragraphs))
        text = header + body + tail
        n = len(tokenizer.encode(text, add_special_tokens=False))
        return text, n + CHAT_TEMPLATE_OVERHEAD

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


async def probe(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
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
        "gap_mean_ms": (statistics.mean(gaps) * 1000.0) if gaps else 0.0,
        "gap_median_ms": (statistics.median(gaps) * 1000.0) if gaps else 0.0,
        "gap_p95_ms": (
            (sorted(gaps)[int(0.95 * (len(gaps) - 1))] * 1000.0) if gaps else 0.0
        ),
        "text": "".join(text_parts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("identity", "decode"), default="identity")
    ap.add_argument("--target-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True, help="path for the emitted text")
    ap.add_argument("--meta", default=None, help="path for the JSON metrics")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    prompt, predicted = build_prompt(args.target_tokens, tok)

    base_url = f"http://{args.host}:{args.port}"
    result = asyncio.run(probe(base_url, args.model, prompt, args.max_tokens))

    with open(args.out, "w") as fh:
        fh.write(result["text"])

    meta = {k: v for k, v in result.items() if k != "text"}
    meta["predicted_prompt_tokens"] = predicted
    meta["target_tokens"] = args.target_tokens
    meta["mode"] = args.mode
    if args.meta:
        with open(args.meta, "w") as fh:
            json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

# type: ignore
#!/usr/bin/env python3
"""Long-context decode-throughput + quality probe for Qwen3.6 MTP A/B.

Builds a ~target-token prompt with a hidden needle, fires it, measures
streaming decode tok/s, checks needle recall, and scans for special-token
(BOS) spam. Parses BOTH content and reasoning_content so thinking-mode
models are measured correctly. Disables prefix cache so every iter does
real work.

Usage (on a node):
    .venv/bin/python bench/mtp_longctx_probe.py --target-tokens 100000 \
        --iters 3 --max-tokens 200 --label MTP_ON
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request

FILLER = [
    "The transformer architecture relies on self-attention to weigh the "
    "relevance of each token against every other token in the sequence.",
    "Distributed systems must contend with partial failure, network "
    "partitions, and the impossibility results captured by the CAP theorem.",
    "Memory bandwidth, not raw compute, is the dominant constraint for "
    "autoregressive decoding on modern accelerators.",
    "Speculative decoding drafts several tokens with a cheap predictor and "
    "verifies them in a single forward pass of the full model.",
    "A lighthouse keeper trims the wick at dusk and logs the passing ships "
    "in a leather-bound ledger that smells of salt and oil.",
]

NEEDLE_CODE = "FALCON-MERCURY-7749"
SPECIAL_TOKEN_MARKERS = [
    "<|begin_of_sentence|>", "<｜begin▁of▁sentence｜>", "<|endoftext|>",
    "<|im_start|>", "<|im_end|>", "begin_of_sentence",
]


def build_prompt(
    target_tokens: int, needle_frac: float | None = None, chars_per_token: float = 2.9
) -> tuple[str, str]:
    # Qwen3.6 tokenizer ~1.39 tokens/word-ish; empirically ~2.9 chars/token.
    # DSv4-Flash's tokenizer is denser on this FILLER text: measured
    # 262652 real tokens from 1450222 chars (target_tokens=500000) on
    # 2026-07-16 == ~5.52 chars/token. Pass --chars-per-token 5.52 for DSv4.
    target_chars = int(target_tokens * chars_per_token)
    needle = (
        f"\n\nIMPORTANT SECRET: The access code is {NEEDLE_CODE}. "
        f"Remember it.\n\n"
    )
    chunks: list[str] = []
    char_count = 0
    if needle_frac is not None:
        # Deterministic needle placement at a fixed fraction of the doc.
        # Used to test LOCAL-window (frac~0.99, last ~100 tok) vs deep-POOL
        # (frac~0.1) retrieval separately.
        needle_at = int(target_chars * needle_frac)
    else:
        needle_at = random.randint(int(target_chars * 0.35), int(target_chars * 0.6))
    placed = False
    i = 0
    while char_count < target_chars:
        if not placed and char_count >= needle_at:
            chunks.append(needle)
            char_count += len(needle)
            placed = True
        p = FILLER[i % len(FILLER)]
        chunks.append(p)
        char_count += len(p) + 1
        i += 1
    if not placed:
        chunks.append(needle)
    body = " ".join(chunks)
    prompt = (
        "Below is a long document. Read it carefully. After the document, "
        "answer the question.\n\n" + body +
        "\n\nQUESTION: What is the access code mentioned in the document? "
        "Reply with just the code."
    )
    return prompt, NEEDLE_CODE


def run_once(base_url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "use_prefix_cache": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    first_t = None
    last_t = 0.0
    n = 0
    content_pieces: list[str] = []
    all_pieces: list[str] = []
    prefill_tps = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            # Server-measured generation_stats arrives as an SSE COMMENT
            # (": generation_stats {...}", not "data: ...") right when the
            # turn finishes -- see chat_completions.py's
            # `chunk.stats.model_dump_json()` emission. This is the
            # authoritative server-side prompt_tps (GenerationStats.prompt_tps
            # in api/types/api.py), not a client-side estimate -- prefer it
            # over anything derivable from streamed chunk timing alone,
            # since the client never sees the prefill phase's own chunks
            # (no tokens stream out until decode starts).
            if line.startswith(": generation_stats"):
                stats_json = line[len(": generation_stats"):].strip()
                try:
                    stats = json.loads(stats_json)
                    prefill_tps = stats.get("prompt_tps")
                except json.JSONDecodeError:
                    pass
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            c = delta.get("content") or ""
            rc = delta.get("reasoning_content") or ""
            piece = c or rc
            if piece:
                now = time.perf_counter()
                if first_t is None:
                    first_t = now
                last_t = now
                n += 1
                if c:
                    content_pieces.append(c)
                all_pieces.append(piece)
    ttft = (first_t - t_start) if first_t else None
    decode_tps = (n - 1) / (last_t - first_t) if (first_t and last_t > first_t and n > 1) else 0.0
    full = "".join(all_pieces)
    content = "".join(content_pieces)
    return {
        "ttft": ttft, "decode_tps": decode_tps, "n_tokens": n,
        "prefill_tps": prefill_tps,
        "content": content, "full": full,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:52415")
    ap.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-8bit")
    ap.add_argument("--target-tokens", type=int, default=100000)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--label", default="run")
    ap.add_argument(
        "--seed",
        type=int,
        default=7749,
        help="Prompt seed. Vary it to defeat the server-side prefix cache "
        "and force a genuine COLD prefill each run.",
    )
    ap.add_argument(
        "--needle-frac",
        type=float,
        default=None,
        help="Fixed needle position as fraction of doc (e.g. 0.99 = local "
        "window, 0.1 = deep pool). Omit for random 0.35-0.6.",
    )
    ap.add_argument(
        "--chars-per-token",
        type=float,
        default=2.9,
        help="Calibration for this FILLER text under the target tokenizer. "
        "Default 2.9 is Qwen3.6-calibrated. Use 5.52 for DSv4-Flash "
        "(measured 2026-07-16: 262652 real tok from 1450222 chars at "
        "target=500000, i.e. the default under-shoots DSv4 by ~1.9x).",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    prompt, needle = build_prompt(args.target_tokens, args.needle_frac, args.chars_per_token)
    print(f"[{args.label}] prompt chars={len(prompt)} (~{args.target_tokens} tok target), needle={needle}")

    rates, ttfts, prefill_rates = [], [], []
    needle_hits = 0
    spam_hits = 0
    sample = ""
    for i in range(args.iters):
        r = run_once(args.base_url, args.model, prompt, args.max_tokens)
        rates.append(r["decode_tps"])
        if r["ttft"]:
            ttfts.append(r["ttft"])
        if r["prefill_tps"]:
            prefill_rates.append(r["prefill_tps"])
        found = needle in r["full"]
        spam = any(m in r["full"] for m in SPECIAL_TOKEN_MARKERS)
        needle_hits += int(found)
        spam_hits += int(spam)
        if i == 0:
            sample = r["content"][:200] or r["full"][:200]
        print(f"[{args.label}] iter {i+1}: prefill={r['prefill_tps'] or 0:8.2f} tok/s  "
              f"decode={r['decode_tps']:6.2f} tok/s  "
              f"ttft={r['ttft'] or 0:.1f}s  ntok={r['n_tokens']}  "
              f"needle={'OK' if found else 'MISS'}  spam={'YES' if spam else 'no'}")

    print(f"\n=== {args.label} SUMMARY ({args.iters} iters @ ~{args.target_tokens} tok) ===")
    if prefill_rates:
        print(f"  prefill mean   {statistics.mean(prefill_rates):.2f} tok/s")
        print(f"  prefill median {statistics.median(prefill_rates):.2f} tok/s")
        print(f"  prefill min/max {min(prefill_rates):.2f} / {max(prefill_rates):.2f}")
        if len(prefill_rates) > 1:
            print(f"  prefill stddev {statistics.pstdev(prefill_rates):.3f}")
    print(f"  decode mean   {statistics.mean(rates):.2f} tok/s")
    print(f"  decode median {statistics.median(rates):.2f} tok/s")
    print(f"  decode min/max {min(rates):.2f} / {max(rates):.2f}")
    if len(rates) > 1:
        print(f"  decode stddev {statistics.pstdev(rates):.3f}")
    if ttfts:
        print(f"  ttft mean     {statistics.mean(ttfts):.1f}s (prefill)")
    print(f"  needle recall {needle_hits}/{args.iters}")
    print(f"  BOS-spam      {spam_hits}/{args.iters}")
    print(f"\n  sample: {sample!r}")


if __name__ == "__main__":
    main()

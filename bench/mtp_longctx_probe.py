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


def build_prompt(target_tokens: int) -> tuple[str, str]:
    # Qwen3.6 tokenizer ~1.39 tokens/word-ish; empirically ~2.9 chars/token
    target_chars = int(target_tokens * 2.9)
    needle = (
        f"\n\nIMPORTANT SECRET: The access code is {NEEDLE_CODE}. "
        f"Remember it.\n\n"
    )
    chunks: list[str] = []
    char_count = 0
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
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode().strip()
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
    args = ap.parse_args()

    random.seed(7749)
    prompt, needle = build_prompt(args.target_tokens)
    print(f"[{args.label}] prompt chars={len(prompt)} (~{args.target_tokens} tok target), needle={needle}")

    rates, ttfts = [], []
    needle_hits = 0
    spam_hits = 0
    sample = ""
    for i in range(args.iters):
        r = run_once(args.base_url, args.model, prompt, args.max_tokens)
        rates.append(r["decode_tps"])
        if r["ttft"]:
            ttfts.append(r["ttft"])
        found = needle in r["full"]
        spam = any(m in r["full"] for m in SPECIAL_TOKEN_MARKERS)
        needle_hits += int(found)
        spam_hits += int(spam)
        if i == 0:
            sample = r["content"][:200] or r["full"][:200]
        print(f"[{args.label}] iter {i+1}: decode={r['decode_tps']:6.2f} tok/s  "
              f"ttft={r['ttft'] or 0:.1f}s  ntok={r['n_tokens']}  "
              f"needle={'OK' if found else 'MISS'}  spam={'YES' if spam else 'no'}")

    print(f"\n=== {args.label} SUMMARY ({args.iters} iters @ ~{args.target_tokens} tok) ===")
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

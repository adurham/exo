#!/usr/bin/env python3
"""Standalone decode-focused throughput probe: small prompt, long generation,
bench=True (bans EOS so length is the only stop signal -- gives a clean,
long decode sample instead of natural-EOS short completions).
"""
import argparse
import asyncio
import json
import time

import httpx


async def measure(base_url: str, model: str, prompt_tokens_hint: int, max_tokens: int) -> dict:
    # Small filler prompt sized roughly to prompt_tokens_hint via repetition.
    filler = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens_hint // 10 + 1)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": filler}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "bench": True,
    }
    start = time.perf_counter()
    first_token_time = None
    n_tokens = 0
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=body, timeout=1800.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    n_tokens += 1
    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else 0
    decode_s = end - first_token_time if first_token_time else 0
    decode_tok_s = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else 0
    return {
        "n_tokens": n_tokens,
        "ttft_s": ttft,
        "decode_s": decode_s,
        "decode_tok_s": decode_tok_s,
        "total_s": end - start,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://adams-mac-studio-m4-1.local:52415")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    for i in range(args.repeat):
        r = await measure(args.base_url, args.model, args.prompt_tokens, args.max_tokens)
        print(f"run {i+1}: n_tokens={r['n_tokens']} ttft={r['ttft_s']:.2f}s "
              f"decode={r['decode_s']:.2f}s decode_tok_s={r['decode_tok_s']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

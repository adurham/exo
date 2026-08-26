#!/usr/bin/env python3
"""Golden (unbanned) greedy probe at depth — /v1/chat/completions, NO bench:true.

Captures the TRUE greedy token stream (model's natural end, EOS allowed)
that the /bench endpoint can never produce (it bans EOS).
Reuses p3_depth_anchor_probe's deep prompt builder for identical context.
"""
import asyncio
import json
import os
import sys
import time

import httpx

sys.path.insert(0, "/Users/adam.durham/repos/exo/bench")
from p3_depth_anchor_probe import build_prompt  # noqa: E402


async def golden(base_url: str, model: str, target_tokens: int, max_tokens: int, out_path: str) -> dict:
    from transformers import AutoTokenizer
    # Same tokenizer dir + trust_remote_code as p3_depth_anchor_probe.py
    # (the -0731 dir on the laptop has no tokenizer files; the preview
    # checkpoint's tokenizer is the canonical one for this model).
    tok = AutoTokenizer.from_pretrained(
        "/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash",
        trust_remote_code=True,
    )
    prompt, predicted = build_prompt(target_tokens, tok)
    print(f"prompt chars={len(prompt)} predicted_tokens={predicted}", flush=True)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # NOTE: no bench:true — EOS NOT banned. Natural end.
    }
    tokens: list[str] = []
    token_ts: list[float] = []  # per-token arrival timestamp (perf_counter, seconds)
    start = time.perf_counter()
    first_token_time = None
    finish_reason = None
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body, timeout=3600.0) as resp:
            print(f"HTTP {resp.status_code}", flush=True)
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ds = line[6:].strip()
                if ds == "[DONE]":
                    break
                try:
                    chunk = json.loads(ds)
                except json.JSONDecodeError:
                    continue
                ch = chunk.get("choices", [{}])[0]
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                delta = ch.get("delta", {})
                c = delta.get("content") or delta.get("reasoning_content")
                if c:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    tokens.append(c)
                    token_ts.append(now)
    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else 0
    decode_s = end - first_token_time if first_token_time else 0
    n = len(tokens)
    # Fixed-window decode tok/s over the first FIXED_WINDOW tokens (events-based,
    # per the corrected spec-decode protocol). tok/s = (N-1)/(t[N-1]-t[0]).
    fixed_window_size = int(os.environ.get("GOLDEN_FIXED_WINDOW", "256"))
    fixed_window_toks = n
    fixed_window_tok_s = 0.0
    fixed_window_decode_s = 0.0
    if n >= 2 and first_token_time is not None and len(token_ts) == n:
        fixed_window_toks = min(fixed_window_size, n)
        if fixed_window_toks >= 2:
            # window: tokens[0 .. fixed_window_toks-1], time span ts[0]..ts[fw-1]
            fixed_window_decode_s = token_ts[fixed_window_toks - 1] - token_ts[0]
            if fixed_window_decode_s > 0:
                fixed_window_tok_s = (fixed_window_toks - 1) / fixed_window_decode_s
    result = {
        "base_url": base_url, "model": model, "target_tokens": target_tokens,
        "n_tokens": n, "ttft_s": ttft, "decode_s": decode_s,
        "decode_tok_s": (n - 1) / decode_s if decode_s > 0 and n > 1 else 0,
        "fixed_window": fixed_window_toks,
        "fixed_window_decode_s": fixed_window_decode_s,
        "fixed_window_tok_s": fixed_window_tok_s,
        "finish_reason": finish_reason, "tokens": tokens, "token_ts": token_ts,
    }
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(json.dumps({k: v for k, v in result.items() if k != "tokens"}, indent=1), flush=True)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.86.201")
    ap.add_argument("--port", default="52415")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--target-tokens", type=int, default=100000)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(golden(f"http://{args.host}:{args.port}", args.model, args.target_tokens, args.max_tokens, args.out))

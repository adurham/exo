#!/usr/bin/env python3
"""Capture ground-truth greedy completions for the MTP system-prompt
degeneration hunt (2026-05-29).

The degeneration (champion-2026-05-25 Eagle K=8 MTP) collapses output to
BOS-spam / short loops, but ONLY for system+user two-message prompts —
single-user prompts stay clean. See warm memory fact 251.

Run this against the cluster running **MTP-OFF** (the known-good greedy
path) to record the ground-truth token sequence each trigger prompt
SHOULD produce. Then run the same prompts against an MTP-ON cluster with
EXO_DSV4_SPEC_TRACE=1 and feed both into spec_degen_diff.py to find the
first cycle where the spec path's committed tokens diverge from greedy.

Output JSON (one entry per prompt):
  { "label", "messages", "content", "reasoning_content", "finish_reason",
    "token_ids" (if the API returns them; else null) }

Usage (run on a Studio so the loopback API + tokenizer are reachable):
  uv run python3 bench/spec_degen_capture.py \\
    --base-url http://localhost:52415 \\
    --model mlx-community/DeepSeek-V4-Flash-8bit \\
    --max-tokens 200 --out ~/spec_degen_groundtruth.json
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
from urllib.parse import urlparse

# The trigger set: each is a benign system+user pair that degenerates on
# the MTP champion but is correct under greedy. Keep these STABLE — the
# diff depends on identical prompts across the two runs. A single-user
# control is included as a negative case (should be clean on both).
PROMPTS: list[tuple[str, list[dict[str, str]]]] = [
    (
        "sys_primary_colors",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Name three primary colors."},
        ],
    ),
    (
        "sys_capital_france",
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ],
    ),
    (
        "sys_count_to_five",
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Count from one to five."},
        ],
    ),
    (
        "control_user_only",  # negative control: single-user, should be clean
        [
            {"role": "user", "content": "Name three primary colors."},
        ],
    ),
]


def _post(base_url: str, body: dict, timeout: float) -> dict:
    u = urlparse(base_url)
    conn_cls = (
        http.client.HTTPSConnection
        if u.scheme == "https"
        else http.client.HTTPConnection
    )
    host = u.hostname or "localhost"
    conn = conn_cls(host, u.port or (443 if u.scheme == "https" else 80),
                    timeout=timeout)
    payload = json.dumps(body)
    conn.request("POST", "/v1/chat/completions", payload,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {raw[:500]}")
    return json.loads(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:52415")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    for label, messages in PROMPTS:
        body = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            # ask for token-level logprobs; if the server honors it we get
            # token ids/strings for exact alignment, else we fall back to text.
            "logprobs": True,
            "top_logprobs": 0,
        }
        try:
            d = _post(args.base_url, body, args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] ERROR: {e}", file=sys.stderr)
            results.append({"label": label, "messages": messages,
                            "error": str(e)})
            continue
        choice = d["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        # Best-effort token-id extraction from logprobs, schema varies.
        token_ids = None
        lp = choice.get("logprobs")
        if isinstance(lp, dict) and isinstance(lp.get("content"), list):
            token_ids = [
                tok.get("token") for tok in lp["content"]
                if isinstance(tok, dict)
            ]
        leaked = "<|begin" in content or "<|begin" in reasoning
        print(f"[{label}] finish={choice.get('finish_reason')} "
              f"leak={leaked} content={content[:80]!r}")
        results.append({
            "label": label,
            "messages": messages,
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": choice.get("finish_reason"),
            "special_token_leak": leaked,
            "token_ids": token_ids,
            "usage": d.get("usage"),
        })

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "max_tokens": args.max_tokens,
                   "results": results}, f, indent=2)
    print(f"\nwrote {len(results)} results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

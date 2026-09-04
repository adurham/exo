#!/usr/bin/env python3
"""I9 helper: send ONE ~89K-token request, record precise wall timestamps
for request-start / first-token / request-end so the powermetrics log
(captured separately on the node) can be split into idle / prefill /
decode windows. Reuses build_prompt() from bench/long_decode_probe.py
per round8 instructions (do not invent a new prompt-building approach).
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bench"))
from long_decode_probe import build_prompt  # noqa: E402

import httpx  # noqa: E402

API = "http://192.168.86.201:52415"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"


def main() -> int:
    target_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 89000
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    out_path = sys.argv[3] if len(sys.argv) > 3 else "i9_client_result.json"

    run_id = uuid.uuid4().hex
    prompt = build_prompt(target_tokens, run_id=run_id)

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.time()
    t_first = None
    content_parts = []
    usage = {}
    server_stats = None
    finish_reason = None

    with httpx.Client(timeout=1800.0) as client:
        with client.stream("POST", f"{API}/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line and line.startswith(": generation_stats "):
                    try:
                        server_stats = json.loads(line[len(": generation_stats "):])
                    except json.JSONDecodeError:
                        pass
                    continue
                if not line or not line.startswith("data: "):
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
                for ch in chunk.get("choices", []) or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    think = delta.get("reasoning_content") or ""
                    if (piece or think) and t_first is None:
                        t_first = time.time()
                    if piece:
                        content_parts.append(piece)
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]

    t_end = time.time()

    result = {
        "run_id": run_id,
        "target_tokens": target_tokens,
        "max_tokens": max_tokens,
        "t_start": t_start,
        "t_first_token": t_first,
        "t_end": t_end,
        "prefill_s": (t_first - t_start) if t_first else None,
        "decode_s": (t_end - t_first) if t_first else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "server_stats": server_stats,
        "content_len": len("".join(content_parts)),
    }
    print(json.dumps(result, indent=2))
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

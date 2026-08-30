#!/usr/bin/env python3
"""P05 G5 same-prompt divergence probe (2026-08-30).

Non-streaming chat completion of ONE fixed prompt at temperature 0 —
run once per arm (A = mxfp8 lm_head knob ON, B = production BF16) and
diff the outputs byte-for-byte. Nonzero divergence is EXPECTED (~16%
near-tie flip rate from the offline numerics); this quantifies the
visible quality cost. Modeled on the PM's throwaway /tmp/p05_same_prompt.py
(arm-B sample: /tmp/p05_same_prompt_B.json) but --model-aware and saved
under the tmp/ P05 workspace so it survives /tmp cleanup.

Usage:
  python3 same_prompt_probe.py --model deepseek-ai/DeepSeek-V4-Flash-0731 \
      --arm A --out tmp/p05-lmhead-mxfp8-20260830/live_ab/same_prompt_A_0731.json

Output json: {arm, model, finish, text, reasoning, compl} — the PM diffs
the "text" fields across arms (the throwaway used the same key set).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

API = "http://192.168.86.201:52415"

PROMPT = (
    "Write exactly four sentences about why the sky is blue. "
    "Be precise and scientific. Do not add anything else."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="model id, e.g. deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--arm", required=True, choices=["A", "B"],
                    help="arm label: A = mxfp8 knob ON, B = BF16 baseline")
    ap.add_argument("--out", required=True,
                    help="output json path (parent dir must exist)")
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="completion budget (default 200; NOTE 0731 is a "
                         "reasoning model and can spend ALL 200 on "
                         "reasoning_content before any content — bump if "
                         "the arms' text fields come back empty)")
    ap.add_argument("--api", default=API)
    args = ap.parse_args()

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    # Generous timeout: cold JIT-placement of the 0731 fp8 checkpoint
    # (~77GB) can take minutes on first request after a relaunch.
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=30.0)) as client:
        r = client.post(f"{args.api}/v1/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()

    ch = d["choices"][0]
    text = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    out = {
        "arm": args.arm,
        "model": args.model,
        "finish": ch["finish_reason"],
        "text": text,
        "reasoning": reasoning,
        "compl": d["usage"]["completion_tokens"],
    }
    print(json.dumps(out))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
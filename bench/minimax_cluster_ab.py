"""Minimal A/B bench for the live MiniMax cluster instance.

Bypasses exo_bench.py's placement-management. Assumes the MiniMax model
is already placed by start_cluster.sh and just hits the
``/v1/chat/completions`` endpoint at 50K input tokens, 200 generated, then
reads ``generation_tps`` from the response. Usage:

    EXO_HOST=192.168.86.201 uv run bench/minimax_cluster_ab.py --label baseline
    EXO_HOST=192.168.86.201 uv run bench/minimax_cluster_ab.py --label fused

Prints decode-tps for each run; the caller can compute the delta.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from statistics import mean

from transformers import AutoTokenizer

MODEL_ID = "mlx-community/MiniMax-M2.7-5bit"


def build_prompt_at_tokens(target: int, tokenizer) -> str:
    atom = "a "
    base_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(base_ids, "input_ids"):
        base_ids = base_ids.input_ids
    base = len(base_ids)

    def count(s: str) -> int:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return len(ids)

    sample_atoms = 100
    per_atom = (count(atom * sample_atoms) - base) / sample_atoms
    n = int((target - base) / per_atom)
    low, high = 0, n * 2 + 100
    while low < high:
        mid = (low + high) // 2
        if count(atom * mid) < target:
            low = mid + 1
        else:
            high = mid
    body = atom * low
    actual = count(body)
    if actual != target:
        # Try a couple of nudges before giving up.
        for delta in (-1, 1, -2, 2):
            cand = atom * (low + delta)
            if count(cand) == target:
                return cand
        raise RuntimeError(f"Could not hit exact target {target}; got {actual}")
    return body


def post_chat(host: str, port: int, content: str, max_tokens: int, use_prefix_cache: bool):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "max_tokens": max_tokens,
        "logprobs": False,
        "use_prefix_cache": use_prefix_cache,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/bench/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return out, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("EXO_HOST", "192.168.86.201"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("EXO_PORT", "52415")))
    ap.add_argument("--pp", type=int, default=50000, help="prompt tokens")
    ap.add_argument("--tg", type=int, default=200, help="generation tokens")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--label", default="run", help="label printed with results")
    ap.add_argument("--out", default=None, help="optional path to dump raw stats as JSON")
    args = ap.parse_args()

    print(f"[{args.label}] tokenizer load...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"[{args.label}] building {args.pp}-token prompt...", file=sys.stderr)
    content = build_prompt_at_tokens(args.pp, tokenizer)

    rows = []

    for i in range(args.warmup):
        out, elapsed = post_chat(args.host, args.port, content, args.tg, use_prefix_cache=True)
        stats = out.get("generation_stats") or {}
        gen_tps = stats.get("generation_tps", 0.0)
        prompt_tps = stats.get("prompt_tps", 0.0)
        print(f"[{args.label}] warmup {i+1}: prompt_tps={prompt_tps:.1f} gen_tps={gen_tps:.2f} elapsed={elapsed:.1f}s", file=sys.stderr)

    for i in range(args.repeat):
        out, elapsed = post_chat(args.host, args.port, content, args.tg, use_prefix_cache=True)
        stats = out.get("generation_stats") or {}
        gen_tps = stats.get("generation_tps", 0.0)
        prompt_tps = stats.get("prompt_tps", 0.0)
        rows.append({
            "label": args.label,
            "run": i + 1,
            "elapsed_s": elapsed,
            "prompt_tps": prompt_tps,
            "generation_tps": gen_tps,
            "stats": stats,
        })
        print(f"[{args.label}] run {i+1}: prompt_tps={prompt_tps:.1f} gen_tps={gen_tps:.2f} elapsed={elapsed:.1f}s")

    gens = [r["generation_tps"] for r in rows if r["generation_tps"] > 0]
    if gens:
        print(f"\n[{args.label}] decode_tps: mean={mean(gens):.2f}  min={min(gens):.2f}  max={max(gens):.2f}  n={len(gens)}")
    else:
        print(f"[{args.label}] no valid generation_tps samples", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[{args.label}] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generation-quality gate for the EXO_DSV4_LMHEAD_MXFP8 knob.

Throughput alone can never justify a quantization ship decision: an mxfp8
lm_head is a THROUGHPUT-for-QUALITY trade, and the only honest gate is
reading what the model actually writes. This harness runs a fixed battery
of temp-0 prompts and saves the FULL generated text (reasoning + content)
so a knob-ON run can be diffed character-by-character against a knob-OFF
run of the identical prompts.

The battery deliberately mixes:
  - exact-recall items (a wrong token is unambiguously wrong, not a style
    difference),
  - arithmetic/multi-step reasoning (top-1 flips compound over a chain),
  - a long-form instruction-following item (divergence shows up as drift).

Usage:
    python3 bench/lmhead_quality_gate.py --tag ON  --out DIR
    python3 bench/lmhead_quality_gate.py --tag OFF --out DIR
Then:
    python3 bench/lmhead_quality_gate.py --compare DIR/ON.json DIR/OFF.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

API = "http://192.168.86.201:52415"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

# Fixed, deterministic battery. Kept small enough to run sequentially on a
# c=1 cluster (concurrent probes get 500 admission errors -- see the
# 2026-08-30 collection gotcha) but broad enough that a ~11.5% top-1 flip
# rate has somewhere to show itself.
PROMPTS: list[dict[str, object]] = [
    {
        "id": "factual_recall",
        "max_tokens": 300,
        "prompt": (
            "List the first 10 elements of the periodic table in order, "
            "with their atomic numbers and chemical symbols. "
            "Format: number. Name (Symbol)"
        ),
    },
    {
        "id": "arithmetic_chain",
        "max_tokens": 500,
        "prompt": (
            "Compute 47 * 83, then subtract 1,229 from the result, then "
            "divide by 7. Show each step and give the final exact value."
        ),
    },
    {
        "id": "code_exact",
        "max_tokens": 400,
        "prompt": (
            "Write a Python function `binary_search(arr, target)` that "
            "returns the index of target in a sorted list arr, or -1 if "
            "absent. Return only the code, no explanation."
        ),
    },
    {
        "id": "logic_multistep",
        "max_tokens": 500,
        "prompt": (
            "Alice is twice as old as Bob. In 5 years, the sum of their "
            "ages will be 40. How old is each now? Show your reasoning."
        ),
    },
    {
        "id": "longform_instruction",
        "max_tokens": 600,
        "prompt": (
            "Explain in exactly three numbered paragraphs why quantizing "
            "a language model's output projection layer can change which "
            "token it picks, even when average error is small."
        ),
    },
]


def run_one(client: httpx.Client, item: dict[str, object]) -> dict[str, object]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": item["max_tokens"],
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.time()
    resp = client.post(f"{API}/v1/chat/completions", json=body, timeout=900.0)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]
    # Thinking models split output: a harness reading only `content`
    # mis-scores them as blank. Capture both, always.
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = data.get("usage", {})
    return {
        "id": item["id"],
        "prompt": item["prompt"],
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "elapsed_s": round(elapsed, 2),
        "decode_tps": (
            round(usage["completion_tokens"] / elapsed, 2)
            if usage.get("completion_tokens") and elapsed > 0
            else None
        ),
    }


def cmd_run(args: argparse.Namespace) -> int:
    results = []
    with httpx.Client() as client:
        for item in PROMPTS:
            print(f"[{args.tag}] running {item['id']} ...", flush=True)
            try:
                r = run_one(client, item)
            except Exception as exc:  # record, never silently drop a probe
                r = {"id": item["id"], "error": repr(exc)}
                print(f"  ERROR {exc!r}", flush=True)
            else:
                n = len(r["content"]) + len(r["reasoning_content"])
                print(
                    f"  ok tokens={r['completion_tokens']} chars={n} "
                    f"tps={r['decode_tps']} finish={r['finish_reason']}",
                    flush=True,
                )
            results.append(r)
    out = {
        "tag": args.tag,
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a = json.load(open(args.compare[0]))
    b = json.load(open(args.compare[1]))
    by_a = {r["id"]: r for r in a["results"]}
    by_b = {r["id"]: r for r in b["results"]}
    print(f"COMPARE  A={a['tag']}  B={b['tag']}\n")
    n_ident = 0
    for pid in by_a:
        ra, rb = by_a[pid], by_b.get(pid, {})
        ta = (ra.get("reasoning_content") or "") + (ra.get("content") or "")
        tb = (rb.get("reasoning_content") or "") + (rb.get("content") or "")
        if ta == tb:
            n_ident += 1
            print(f"  {pid:24s} IDENTICAL ({len(ta)} chars)")
            continue
        # first divergent character position
        idx = next(
            (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
            min(len(ta), len(tb)),
        )
        print(
            f"  {pid:24s} DIVERGES at char {idx} "
            f"(lenA={len(ta)} lenB={len(tb)})"
        )
        print(f"      A: ...{ta[max(0, idx - 60):idx + 120]!r}")
        print(f"      B: ...{tb[max(0, idx - 60):idx + 120]!r}")
    print(f"\n{n_ident}/{len(by_a)} prompts byte-identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--out", default="/tmp/lmhead_quality.json")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    args = ap.parse_args()
    if args.compare:
        return cmd_compare(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())

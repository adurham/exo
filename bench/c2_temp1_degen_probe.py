#!/usr/bin/env python3
"""c=2 temp>0 degeneration probe.

Fires two concurrent DSv4 chat requests (ragged prompt lengths) at the
model-card default sampling (temp=1.0, top_p=0.95, min_p=0.05, count-aware
rep_pen=1.1 from the instance) and scans the outputs for the degeneration
signatures observed in prod 2026-07-11 19:39-19:42:

  * kill-switch error finishes (finish_reason == "error" / HTTP 500 mid-stream)
  * period-1/2 token loops in the tail
  * word-stutter ("the the the", "They They're") anywhere in the text

Usage: python3 c2_temp1_degen_probe.py [--long-tokens N] [--short-tokens N]
                                       [--gen N] [--rounds N] [--solo]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.request

API = "http://localhost:52415/v1/chat/completions"
MODEL = "mlx-community/DeepSeek-V4-Flash"

# Filler corpus: repeats of a varied paragraph so the tokenizer doesn't
# collapse it, with a counter so no two sentences are identical.
PARA = (
    "Section {i}: The maintenance log for unit {i} records a replacement of "
    "the coolant manifold, a firmware update to revision {i}.4.2, and an "
    "inspection of the torque calibration rig. Ambient humidity measured "
    "{h} percent; the vibration spectrum showed a minor peak at {f} Hz "
    "attributed to bearing wear on the secondary spindle. "
)


def make_filler(n_tokens_approx: int) -> str:
    # ~55 tokens per paragraph instance
    n = max(1, n_tokens_approx // 55)
    return "".join(
        PARA.format(i=i, h=30 + (i * 7) % 40, f=40 + (i * 13) % 200)
        for i in range(n)
    )


def stutter_score(text: str) -> list[str]:
    """Return doubled-word occurrences like 'the the' / 'They They're'."""
    hits = re.findall(r"\b(\w{2,})\s+\1\b", text, flags=re.IGNORECASE)
    # also catch tripled+
    return hits


def tail_loop(text: str) -> str | None:
    """Detect a short repeating unit at the end of the text."""
    tail = text[-400:]
    for unit_len in range(1, 40):
        unit = tail[-unit_len:]
        if unit and tail.endswith(unit * min(6, len(tail) // max(1, unit_len))):
            reps = 0
            t = tail
            while t.endswith(unit) and unit:
                reps += 1
                t = t[: -len(unit)]
            if reps >= 6 and unit.strip():
                return f"unit={unit!r} reps={reps}"
    return None


def fire(tag: str, prompt: str, gen: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "stream": False,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            payload = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"tag": tag, "error": str(e)[:300], "elapsed": time.time() - t0}
    choice = payload["choices"][0]
    msg = choice["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    usage = payload.get("usage", {})
    return {
        "tag": tag,
        "elapsed": round(time.time() - t0, 1),
        "finish": choice.get("finish_reason"),
        "gen_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "stutters": stutter_score(text)[:12],
        "n_stutters": len(stutter_score(text)),
        "tail_loop": tail_loop(text),
        "tail": text[-160:].replace("\n", " "),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-tokens", type=int, default=8000)
    ap.add_argument("--short-tokens", type=int, default=200)
    ap.add_argument("--gen", type=int, default=1600)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--solo", action="store_true", help="run streams one at a time (control)")
    ap.add_argument("--stagger", type=float, default=20.0,
                    help="seconds to delay the short stream so the long one is mid-generation")
    args = ap.parse_args()

    long_prompt = (
        make_filler(args.long_tokens)
        + "\n\nSummarize the recurring maintenance themes above in detail, then "
        "propose a 12-month preventative schedule with rationale for each item."
    )
    short_prompt = (
        "Write a detailed technical explanation of how speculative decoding "
        "with a draft model works, including acceptance sampling math."
    )

    for round_i in range(args.rounds):
        print(f"=== round {round_i} (solo={args.solo}) ===", flush=True)
        if args.solo:
            for tag, p in [("long", long_prompt), ("short", short_prompt)]:
                print(json.dumps(fire(tag, p, args.gen)), flush=True)
        else:
            with cf.ThreadPoolExecutor(max_workers=2) as ex:
                fa = ex.submit(fire, "long", long_prompt, args.gen)
                time.sleep(args.stagger)
                fb = ex.submit(fire, "short", short_prompt, args.gen)
                for f in (fa, fb):
                    print(json.dumps(f.result()), flush=True)


if __name__ == "__main__":
    main()

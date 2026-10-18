#!/usr/bin/env python3
"""Summarise §11 run artifacts (run-s11.py + s11-correlate.py output) against §9.2/§9.4.

Reads out/*.json, prints a markdown table plus a JSON block. Claimed values are the
writeup's own, keyed by row label; the delta columns are what the reproduction shows.

Usage: python3 s11-summarize.py --out ./out [--json summary.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# §9.2 "Decode and prefill vs context depth" (writeup at b88d65dd) + §9.4.
CLAIMED = {
    "d1k":    dict(prompt=1003,   cached=0,     completion=82,  finish="stop",   prefill_s=3.93,    prefill_tps=255.2, client=34.5, server=29.2, wall=6.95),
    "d10k":   dict(prompt=10004,  cached=0,     completion=152, finish="stop",   prefill_s=36.37,   prefill_tps=275.1, client=31.4, server=31.3, wall=41.34),
    "d50k":   dict(prompt=50004,  cached=8192,  completion=163, finish="stop",   prefill_s=162.27,  prefill_tps=257.7, client=29.4, server=29.3, wall=167.94),
    "d100k":  dict(prompt=100003, cached=49152, completion=200, finish="length", prefill_s=207.50,  prefill_tps=245.1, client=30.7, server=30.6, wall=214.27),
    "d160k":  dict(prompt=160004, cached=0,     completion=200, finish="length", prefill_s=661.18,  prefill_tps=242.0, client=None, server=30.4, wall=668.15),
    "d300k":  dict(prompt=300003, cached=0,     completion=189, finish="stop",   prefill_s=1535.25, prefill_tps=195.4, client=5.9,  server=6.0,  wall=1600.82),
    "d1m":    dict(prompt=1000003, cached=299008, completion=100, finish="length", prefill_s=6725.0, prefill_tps=104.2, client=2.92, server=2.92, wall=6760.5),
}


def pct(new, old):
    if new is None or old in (None, 0):
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.out, "*.json"))):
        d = json.load(open(path))
        s = d.get("server") or {}
        c = CLAIMED.get(d["label"])
        rows.append((d, s, c))

    order = ["d1k", "d10k", "d50k", "d100k", "d160k", "d300k", "d1m", "guard_banana", "guard_loop", "limiter_96", "limiter_default"]
    rows.sort(key=lambda r: order.index(r[0]["label"]) if r[0]["label"] in order else 99)

    print("| row | prompt | cached | completion | finish | prefill s | prefill tok/s (uncached) | client decode tok/s | server decode tok/s | wall s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d, s, c in rows:
        print(
            "| {label} | {p} | {ca} | {co} | {f} | {ps} | {pt} | {cd} | {sd} | {w} |".format(
                label=d["label"],
                p=d.get("prompt_tokens"),
                ca=d.get("cached_tokens"),
                co=d.get("completion_tokens"),
                f=d.get("finish_reason"),
                ps=s.get("prefill_s", "n/a"),
                pt=s.get("prefill_tok_s_uncached", "n/a"),
                cd=d.get("client_decode_tok_s", "n/a"),
                sd=s.get("server_decode_tok_s", "n/a"),
                w=d.get("wall_s"),
            )
        )

    print("\nDeltas vs the writeup's §9.2/§9.4 claims (client/server decode and uncached prefill rate):\n")
    print("| row | claim prefill tok/s | measured | delta | claim server decode | measured | delta |")
    print("|---|---|---|---|---|---|---|")
    for d, s, c in rows:
        if not c:
            continue
        print(
            f"| {d['label']} | {c['prefill_tps']} | {s.get('prefill_tok_s_uncached','n/a')} | "
            f"{pct(s.get('prefill_tok_s_uncached'), c['prefill_tps'])} | {c['server']} | "
            f"{s.get('server_decode_tok_s','n/a')} | {pct(s.get('server_decode_tok_s'), c['server'])} |"
        )

    print("\n## generated text (first 300 chars per row)\n")
    for d, s, c in rows:
        text = (d.get("content") or "").replace("\n", " ")
        print(f"- **{d['label']}** ({d.get('finish_reason')}, {d.get('completion_tokens')} tokens): {text[:300]}")

    if args.json:
        blob = []
        for d, s, c in rows:
            blob.append(
                {
                    "label": d["label"],
                    "prompt_tokens": d.get("prompt_tokens"),
                    "cached_tokens": d.get("cached_tokens"),
                    "completion_tokens": d.get("completion_tokens"),
                    "finish_reason": d.get("finish_reason"),
                    "prefill_s": s.get("prefill_s"),
                    "prefill_tok_s_uncached": s.get("prefill_tok_s_uncached"),
                    "client_decode_tok_s": d.get("client_decode_tok_s"),
                    "server_decode_tok_s": s.get("server_decode_tok_s"),
                    "wall_s": d.get("wall_s"),
                    "claimed": c,
                    "content_head": (d.get("content") or "")[:300],
                    "reasoning_tokens_seen": len(d.get("reasoning") or ""),
                    "error_events": d.get("error_events"),
                }
            )
        json.dump(blob, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

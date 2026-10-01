#!/usr/bin/env python3
"""p22_analyze -- turn the harness's results.json + logits .npy into the report.

Reads every ~/p22_out/<tag>/results.json, prints:
  * a per-arm table of tok/s, per-chunk wall timeline, GPU-busy share, peak;
  * the oracle comparison: sha256 equality, first-token max-abs-logit-diff,
    top-1 agreement and first divergence position against the reference arm;
  * the attribution bucket shares for any arm whose log carried `attr:` lines.

Usage: p22_analyze.py <tag> [<tag> ...]      (tags are ~/p22_out/<tag>)
Reads only; writes nothing.
"""
import glob
import json
import os
import sys

import numpy as np

OUT = os.environ.get("P22_OUT", os.path.expanduser("~/p22_out"))


def load(tag):
    with open(os.path.join(OUT, tag, "results.json")) as f:
        return json.load(f)


def chunks_of(res):
    return res.get("per_chunk") or []


def main(tags):
    arms = {}
    for t in tags:
        try:
            payload = load(t)
        except FileNotFoundError:
            print(f"!! {t}: no results.json under {OUT}/{t}")
            continue
        for r in payload["results"]:
            arms[(t, r["label"])] = r
        print(f"== {t}: {len(payload['results'])} arms, ops_per_buffer="
              f"{payload['env'].get('MLX_MAX_OPS_PER_BUFFER')}, "
              f"session_md5={payload['env'].get('session_md5')}")

    print("\n=== ARMS")
    hdr = f"{'tag':10s} {'label':34s} {'L':>6s} {'chunk':>6s} {'tok/s':>7s} " \
          f"{'gpu%':>6s} {'peak_gb':>8s} {'dec ms':>7s} {'dec tok/s':>9s}"
    print(hdr)
    for (t, lbl), r in arms.items():
        gpu = r.get("gpu_ms_total")
        tot = (r.get("total_s") or 0) * 1000
        gpu_pct = f"{100 * gpu / tot:.1f}" if (gpu and tot) else "-"
        print(f"{t:10s} {lbl:34s} {r.get('L', 0):6d} {r.get('chunk', 0):6d} "
              f"{r.get('tok_s', 0):7.1f} {gpu_pct:>6s} "
              f"{r.get('peak_gb', 0):8.2f} {str(r.get('decode_ms_median', '-')):>7s} "
              f"{str(r.get('decode_tok_s', '-')):>9s}")

    # per-chunk timeline: wall deltas come from the driver's own cumulative
    # `elapsed` in progress(); GPU is cumulative too, so diff both here.
    print("\n=== PER-CHUNK (wall ms delta / gpu ms delta / peak_gb / tokens-per-expert)")
    for (t, lbl), r in arms.items():
        ch = chunks_of(r)
        if not ch:
            continue
        prev_w = prev_g = 0.0
        parts = []
        for c in ch:
            w = c.get("wall_ms", 0) - prev_w
            g = c.get("gpu_ms_cum", 0) - prev_g
            prev_w, prev_g = c.get("wall_ms", 0), c.get("gpu_ms_cum", 0)
            parts.append(f"c{c.get('idx')}(p{c.get('rows_done')}) "
                         f"{w:.0f}/{g:.0f}ms {c.get('peak_gb')}GB {c.get('cache_mb')}MB")
        print(f"{t} {lbl}: " + " | ".join(parts))

    print("\n=== ORACLE")
    ref_sha = None
    ref_logits = {}
    for (t, lbl) in sorted(arms):
        r = arms[(t, lbl)]
        if not r.get("oracle_sha256"):
            continue
        tag_ok = r["oracle_sha256"]
        lp_path = os.path.join(OUT, t, f"logits_{lbl}.npy")
        if os.path.exists(lp_path):
            ref_logits[(t, lbl)] = np.load(lp_path)
        print(f"{t} {lbl}: sha={tag_ok[:16]} dec_ms={r.get('decode_ms_median')} "
              f"tok_s={r.get('decode_tok_s')}")
    # first-token logit comparison, WITHIN one prompt length only: 2K/8K/16K are
    # different prompts (different final token), so their next-token logits are
    # legitimately different and comparing across them is meaningless.
    groups = {}
    for (t, lbl), v in ref_logits.items():
        L = arms[(t, lbl)].get("L")
        groups.setdefault(L, []).append(((t, lbl), v))
    for L in sorted(groups):
        keys = sorted(groups[L])
        base_k, b = keys[0]
        print(f"\n-- L={L}: reference logits {base_k} (argmax={int(np.argmax(b))})")
        for k, v in keys[1:]:
            if v.shape != b.shape:
                print(f"   {k}: SHAPE {v.shape} vs {b.shape}")
                continue
            d = np.abs(v - b)
            print(f"   {k}: max|dlogit|={d.max():.6g} mean={d.mean():.6g} "
                  f"top1_agree={int(np.argmax(v)) == int(np.argmax(b))} "
                  f"argmax={int(np.argmax(v))}")

    # attribution shares come from the progress.jsonl (rank 0 only)
    print("\n=== ATTR BUCKETS (eval-fenced: shares only)")
    for t in tags:
        p = os.path.join(OUT, t, "progress.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            if '"attr:' in line:
                print(f"{t}: {line.strip()[:600]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["run1"])

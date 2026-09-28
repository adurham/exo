#!/usr/bin/env python3
"""Does the chunk-shape perturbation matter as the distribution sharpens?

p6e showed, at a 16-token context, that the one-shot-vs-chunked logit
perturbation (2.28) is 4.7x the top1-top2 margin (0.48) -- so the argmax flips.

That margin is a property of HOW MUCH CONTEXT the model has seen, and the 4-layer
reduced build has seen almost nothing, so its distribution is nearly flat. The
deployment-relevant question is whether the perturbation stays comparable to the
margin as the model gets more context (a real 40-layer body at 2K+ tokens).

This measures, at several prompt lengths:
  * the top1-top2 margin
  * the one-shot-vs-chunked logit perturbation
  * whether the argmax actually flips

Interpretation: if the margin grows much faster than the perturbation, the
divergence is an artifact of an under-trained depth and would not appear on the
real model. If they track each other, the port is genuinely chunk-sensitive and
that has to be dealt with before MTP-chunk can be deployed.

Usage: python3 p6g_margin.py <MODEL_DIR>
"""
from __future__ import annotations

import os
import sys

import mlx.core as mx

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx")
sys.path.insert(0, REPO)
from deepseek_v41_mlx.load import load  # noqa: E402

PATH = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/v41-mtp-converted")
try:
    mx.set_wired_limit(int(float(os.environ.get("V41_WIRED_GB", "100")) * 1e9))
except Exception as e:  # noqa: BLE001
    print("[warn] wired:", e)

model, args = load(PATH)
print(f"=== {args.n_layers} layers, window={args.window_size} ===")
print()

# a real-ish passage so the context is not degenerate
TEXT_IDS = [
    43281, 108435, 109597, 105148, 56193, 56712, 100489, 87240, 25057, 74673,
    55421, 55421, 43281, 108435, 109597, 105148, 25057, 74673, 24002, 129273,
    93752, 93752, 64390, 44115, 44115, 25057, 74673, 69386, 56712, 100489,
    87240, 56193, 105148, 109597, 108435, 43281, 25057, 74673, 24002, 129273,
    93752, 64390, 44115, 55421, 69386, 56712, 100489, 87240, 56193, 105148,
]
while len(TEXT_IDS) < 200:
    TEXT_IDS = TEXT_IDS + TEXT_IDS
print(f"context pool: {len(TEXT_IDS)} ids")
print()


def prefill(ids, chunks):
    cache = model.make_cache(bsz=1, max_seq_len=512)
    i = 0
    lg = None
    for c in chunks:
        lg = model(mx.array([ids[i:i + c]]), cache, last_logit_only=True)
        mx.eval(lg)
        i += c
    if i < len(ids):
        lg = model(mx.array([ids[i:]]), cache, last_logit_only=True)
        mx.eval(lg)
    return lg[:, -1]


print(f"{'ctx':>5} {'margin':>10} {'perturb':>10} {'ratio':>8} {'argmax flips':>13}")
print("-" * 52)
for NP in (8, 16, 32, 64, 128):
    ids = TEXT_IDS[:NP]
    a = prefill(ids, [NP])
    b = prefill(ids, [2] * (NP // 2))
    mx.eval(a, b)
    srt = mx.sort(a, axis=-1)
    margin = float(srt[0, -1] - srt[0, -2])
    pert = float(mx.max(mx.abs(a - b)))
    ratio = pert / max(margin, 1e-9)
    flip = int(mx.argmax(a)) != int(mx.argmax(b))
    print(f"{NP:>5} {margin:>10.4f} {pert:>10.4f} {ratio:>7.2f}x "
          f"{'YES' if flip else 'no':>13}")

print()
print("=== READ-OUT ===")
print("  margin   = top1 - top2 logit at the final position (bigger = peakier)")
print("  perturb  = max|one-shot - chunked| over the logit vector")
print("  ratio    = perturb / margin; >1 means the argmax can flip")
print()
print("  If margin grows and ratio falls as context grows, the divergence is a")
print("  property of the near-flat short-context regime (amplified by the");
print("  reduced depth) and would be far rarer on the full 40-layer model.")
print()
print("MARGIN-DONE")

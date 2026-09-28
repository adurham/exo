#!/usr/bin/env python3
"""Close out the chunk-shape question, and sanity-check the draft head.

THREE MEASUREMENTS

1. DETERMINISM. Run the same chunked prefill twice. If the two runs are
   bit-identical, the chunk effect is a DETERMINISTIC function of the split (a
   kernel-shape/accumulation-order difference), not nondeterminism.

2. FLATNESS. Measure the top1-top2 logit margin at the final position. A 4-layer
   body has seen almost nothing, so its next-token distribution is nearly flat:
   a 2-ulp perturbation then flips the argmax easily. If the margins here are
   tiny in absolute terms, the "different argmax" result is explained by
   amplification, not by a logic difference -- and it predicts that a real
   40-layer body (much peakier) would rarely flip.

3. DRAFT HEAD ALIVENESS. Agreement between the draft head's top-1 and the
   body's greedy argmax, plus the confidence-head output spread. A mis-wired
   head (garbage weight mapping) gives chance agreement and ~flat confidence;
   a correctly-wired head on a 4-layer body gives above-chance agreement and
   structured confidence.

Usage: python3 p6e_closeout.py <MODEL_DIR>
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
V = args.vocab_size
print(f"=== {args.n_layers} layers, vocab={V}, block={args.dspark_block_size} ===")
print()

PROMPT = [43281, 108435, 109597, 105148, 56193, 56712, 100489, 87240,
          25057, 74673, 55421, 55421, 43281, 108435, 109597, 105148]


def prefill(chunks):
    cache = model.make_cache(bsz=1, max_seq_len=256)
    i = 0
    lg = None
    for c in chunks:
        lg = model(mx.array([PROMPT[i:i + c]]), cache, last_logit_only=True)
        mx.eval(lg)
        i += c
    return lg


print("=== 1. determinism ===")
splits = {
    "one-shot": [len(PROMPT)],
    "of-2": [2] * (len(PROMPT) // 2),
    "1-then-3s": [1, 3, 3, 3, 3, 3],
}
runs = {}
for name, ch in splits.items():
    a = prefill(ch)[:, -1]
    b = prefill(ch)[:, -1]
    mx.eval(a, b)
    same = bool(mx.all(a == b))
    runs[name] = a
    print(f"  {name:<12} repeat-identical: {same}   argmax={int(mx.argmax(a))}")
print()

print("=== 2. flatness (top1-top2 margin at the final position) ===")
for name, a in runs.items():
    srt = mx.sort(a, axis=-1)
    m = srt[:, -1] - srt[:, -2]
    mx.eval(m)
    print(f"  {name:<12} margin = {float(m[0]):8.4f}")
os_ = runs["one-shot"]
of2 = runs["of-2"]
d = mx.abs(os_ - of2)
mx.eval(d)
print(f"  logit max|d| (one-shot vs of-2) = {float(mx.max(d)):.4f}")
print(f"  logit mean|d|                   = {float(mx.mean(d)):.6f}")
srt = mx.sort(os_, axis=-1)
marg = float(srt[0, -1] - srt[0, -2])
mx.eval()
print()
print(f"  --> perturbation is {(float(mx.max(d))/max(marg,1e-9)):.1f}x the top1-top2 margin"
      if marg > 0 else "")
print()

print("=== 3. draft head agreement (200 positions) ===")
cache = model.make_cache(bsz=1, max_seq_len=512)
lg = model(mx.array([PROMPT]), cache, last_logit_only=True)
mx.eval(lg)
head = model.mtp
dsc = head.make_cache(bsz=1)

# context: feed the prompt's last hidden states as tap context
cache2 = model.make_cache(bsz=1, max_seq_len=512)
lg2, taps = model(mx.array([PROMPT]), cache2, return_taps=True)
mx.eval(lg2, *taps.values())
TAPS = list(args.dspark_target_layer_ids)
ctx = mx.concatenate([taps[L] for L in TAPS], axis=-1)
head.append_ctx(ctx, dsc)

tok = mx.argmax(lg[:, -1], axis=-1)
mx.eval(tok)
agree = 0
N = 200
margins = []
confs = []
for step in range(N):
    dt, conf = head.draft(tok, model.embed, model.head, dsc, width=1)
    mx.eval(dt, conf)
    # body's own greedy next token for the same prefix
    lg = model(tok[:, None], cache, last_logit_only=True)
    mx.eval(lg)
    body_next = int(mx.argmax(lg[:, -1], axis=-1)[0])
    draft_top = int(dt[0, 0])
    if draft_top == body_next:
        agree += 1
    confs.append(float(conf[0, 0]))
    tok = mx.array([body_next])

print(f"  positions tested          : {N}")
print(f"  draft top-1 == body greedy: {agree} ({agree/N:.1%})")
print(f"  chance level (1/vocab)    : {1.0/V:.2e}")
print(f"  confidence head: min={min(confs):.4f} max={max(confs):.4f} "
      f"mean={sum(confs)/len(confs):.4f}")
print()
print("=== READ-OUT ===")
print("  agreement >> chance and a confidence spread that is not all-0.5")
print("  together mean the draft weights landed where they belong.")
print()
print("CLOSEOUT-DONE")

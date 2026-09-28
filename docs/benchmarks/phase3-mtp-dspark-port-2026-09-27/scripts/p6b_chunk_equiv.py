#!/usr/bin/env python3
"""Is the port's CHUNKED prefill equivalent to a single-shot prefill?

Why this matters right now: the speculative verify forward feeds a chunk of
(block+1) tokens through the body in ONE call. That is exactly what chunked
prefill does. If the port's chunked path is not equivalent to one-shot prefill,
then the verify forward is computing something a plain decode would never
compute -- and speculative decode can never match greedy, no matter how correct
my accept/rollback bookkeeping is.

This is a pre-existing-port question, independent of MTP: it needs no draft head
at all. Split the same prompt into chunks of 1, 2, 3, 6 and compare the argmax
of the last position's logits, plus a few greedy steps after the boundary.

Usage: python3 p6b_chunk_equiv.py <MODEL_DIR>
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
print(f"=== {args.n_layers} layers, window={args.window_size}, "
      f"ratios={[args.compress_ratio(i) for i in range(args.n_layers)]} ===")
print()

PROMPT = [43281, 108435, 109597, 105148, 56193, 56712, 100489, 87240,
          25057, 74673, 55421, 55421, 43281, 108435, 109597, 105148]
NP = len(PROMPT)
print(f"prompt: {NP} ids")
print()


def run_chunked(chunks):
    """Prefill PROMPT in the given chunk sizes, then 4 greedy steps.

    Returns (last-prefill argmax, [4 next tokens]).
    """
    cache = model.make_cache(bsz=1, max_seq_len=256)
    i = 0
    lg = None
    for c in chunks:
        part = PROMPT[i:i + c]
        lg = model(mx.array([part]), cache, last_logit_only=True)
        mx.eval(lg)
        i += c
    if i < NP:                      # tail
        lg = model(mx.array([PROMPT[i:]]), cache, last_logit_only=True)
        mx.eval(lg)
    first = int(mx.argmax(lg[:, -1], axis=-1)[0])
    out = [first]
    t = mx.array([first])
    for _ in range(4):
        lg = model(t[:, None], cache, last_logit_only=True)
        mx.eval(lg)
        t = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(t)
        out.append(int(t[0]))
    return first, out


def split(n_total, size):
    """Chunk sizes summing to n_total, last one may be shorter."""
    cs = []
    rem = n_total
    while rem > 0:
        cs.append(min(size, rem))
        rem -= size
    return cs


print("=== one-shot vs chunked prefill ===")
ref_first, ref_next = run_chunked([NP])
print(f"  one-shot ({NP:2d})       -> first={ref_first:7d}  next4={ref_next}")
print()
ok = True
for size in (1, 2, 3, 4, 6, 8):
    for offset in (0, 1):
        n = NP - offset
        if n <= 0:
            continue
        chunks = [offset] + split(n, size) if offset else split(n, size)
        chunks = [c for c in chunks if c]
        f, nxt = run_chunked(chunks)
        same = (f == ref_first and nxt == ref_next)
        ok &= same
        tag = "OK " if same else "DIFF"
        print(f"  {tag} chunks={str(chunks):<32} first={f:7d}  next4={nxt}")
print()
print("=== VERDICT ===")
if ok:
    print("  chunked prefill is EQUIVALENT to one-shot  --> the verify chunk")
    print("  shape is safe; the divergence is in MY rollback bookkeeping.")
else:
    print("  chunked prefill DIVERGES from one-shot  --> this is a PRE-EXISTING")
    print("  port bug that the speculative verify forward is the first to hit.")
    print("  Fix this before any acceptance-rate work; speculative decode cannot")
    print("  match greedy until a multi-token chunk computes the same thing.")
print()
print("CHUNK-EQUIV-DONE")

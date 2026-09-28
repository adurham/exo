#!/usr/bin/env python3
"""Localize the chunk-shape divergence: which layer, and which cache, differs?

p6b established that the port's prefill result depends on how the prompt is
split into chunks (first chunk == 1 token matches one-shot; first chunk >= 2
does not). That is a PRE-EXISTING body bug, and it blocks MTP: the speculative
verify forward is a multi-token chunk.

This probe runs the same prefix two ways and compares, for the FINAL position:

  1. the output logits            -> how far apart the answers are
  2. per-layer tap hiddens (0..3) -> the first layer whose hidden state differs
  3. layer 2's comp_kv groups     -> did the compressor pool the same latents?
  4. layer 2's index_k rows       -> did the indexer derive the same keys?

(2) localizes the bug to a layer; (3) and (4) say whether it is the compressor,
the indexer, or the attention/selection path that reads them.

Usage: python3 p6c_localize.py <MODEL_DIR>
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
      f"ratios={[args.compress_ratio(i) for i in range(args.n_layers)]}, "
      f"kv_src={args.kv_source_layers}, index_src={args.index_source_layers} ===")
print()

PROMPT = [43281, 108435, 109597, 105148, 56193, 56712, 100489, 87240,
          25057, 74673, 55421, 55421, 43281, 108435, 109597, 105148]

# capture ALL layers this run, so the first differing layer is visible
args.dspark_target_layer_ids = list(range(args.n_layers))


def run(chunks):
    cache = model.make_cache(bsz=1, max_seq_len=256)
    i = 0
    logits = taps = None
    for c in chunks:
        part = PROMPT[i:i + c]
        logits, taps = model(mx.array([part]), cache, return_taps=True,
                             last_logit_only=True)
        mx.eval(logits, *taps.values())
        i += c
    if i < len(PROMPT):
        logits, taps = model(mx.array([PROMPT[i:]]), cache, return_taps=True,
                             last_logit_only=True)
        mx.eval(logits, *taps.values())
    return cache, logits, taps


print("=== run A: one-shot ===")
cA, lgA, tpA = run([len(PROMPT)])
print(f"  argmax = {int(mx.argmax(lgA[:, -1], axis=-1)[0])}")
print()
print("=== run B: chunks of 2 ===")
cB, lgB, tpB = run([2] * (len(PROMPT) // 2))
print(f"  argmax = {int(mx.argmax(lgB[:, -1], axis=-1)[0])}")
print()

print("=== 1. logits (final position) ===")
d = mx.abs(lgA[:, -1] - lgB[:, -1])
mx.eval(d)
print(f"  max abs diff = {float(mx.max(d)):.4f}   mean = {float(mx.mean(d)):.4f}")
print()

print("=== 2. per-layer tap hidden (final position) ===")
print(f"  {'layer':>5} {'norm A':>10} {'norm B':>10} {'max|d|':>10} {'rel':>10}")
first_bad = None
for L in range(args.n_layers):
    a = tpA[L][:, -1]
    b = tpB[L][:, -1]
    mx.eval(a, b)
    md = float(mx.max(mx.abs(a - b)))
    rel = md / (float(mx.max(mx.abs(a))) + 1e-12)
    flag = ""
    if md > 1e-3 and first_bad is None:
        first_bad = L
        flag = "   <-- first divergence"
    print(f"  {L:>5} {float(mx.max(mx.abs(a))):>10.4f} "
          f"{float(mx.max(mx.abs(b))):>10.4f} {md:>10.5f} {rel:>10.3e}{flag}")
print()

print("=== 3. layer-2 comp_kv (pooled groups) ===")
for L in (2, 3):
    lc = cA.layers[L] if cA.layers[L].comp_state is not None else None
    lcb = cB.layers[L]
    if lc is None or lc.comp_kv is None:
        print(f"  layer {L}: no comp_kv")
        continue
    n = min(lc.comp_kv.shape[1], ((len(PROMPT)) // max(lc.ratio, 1)))
    a = lc.comp_kv[:, :n]
    b = lcb.comp_kv[:, :n]
    mx.eval(a, b)
    md = float(mx.max(mx.abs(a - b)))
    print(f"  layer {L}: ratio={lc.ratio} groups_compared={n} max|d|={md:.6f}")
    cs_a, cs_b = lc.comp_state, lcb.comp_state
    if cs_a is not None:
        ka, kb = cs_a.kv_state, cs_b.kv_state
        mx.eval(ka, kb)
        print(f"           carry kv_state max|d| = {float(mx.max(mx.abs(ka - kb))):.6f}")
    if lc.index_k is not None and lcb.index_k is not None:
        ia = lc.index_k[:, :n]
        ib = lcb.index_k[:, :n]
        mx.eval(ia, ib)
        print(f"           index_k max|d| = {float(mx.max(mx.abs(ia - ib))):.6f}")
print()

print("=== 4. window ring (layer 2) ===")
wa = cA.layers[2].win_kv
wb = cB.layers[2].win_kv
mx.eval(wa, wb)
print(f"  win_kv max|d| = {float(mx.max(mx.abs(wa - wb))):.6f}")
print()

print("=== VERDICT ===")
if first_bad is None:
    print("  no layer diverged at the final position (within 1e-3)")
else:
    print(f"  first divergent layer: {first_bad}")
    print(f"  ratios: {[args.compress_ratio(i) for i in range(args.n_layers)]}")
    print(f"  -> ratio 0 means a pure-window layer (no compressor, no indexer)")
print()
print("LOCALIZE-DONE")

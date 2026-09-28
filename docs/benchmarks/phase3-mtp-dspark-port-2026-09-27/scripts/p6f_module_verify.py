#!/usr/bin/env python3
"""Install the MTP decode module and re-run the exactness proof through it.

This is the verification step for mtp_decode.py: the same 48-token comparison
that p6d_spec_rowseq.py used, but driven through the packaging (module import,
``speculative_generate``) instead of the ad-hoc script. If rowseq is still 48/48
identical to greedy THROUGH THE MODULE, the shipped code is the validated code.
"""
from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx")
sys.path.insert(0, REPO)
from deepseek_v41_mlx.load import load  # noqa: E402
from deepseek_v41_mlx.mtp_decode import speculative_generate  # noqa: E402

MODEL = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                           else "~/v41-mtp-converted")
try:
    mx.set_wired_limit(int(float(os.environ.get("V41_WIRED_GB", "100")) * 1e9))
except Exception as e:  # noqa: BLE001
    print("[warn] wired:", e)

model, args = load(MODEL)
print(f"=== {args.n_layers} layers, {args.n_mtp_layers} draft stages, "
      f"block={args.dspark_block_size}, taps={list(args.dspark_target_layer_ids)} ===")
print()

PROMPT = [43281, 108435, 109597, 105148, 56193, 56712, 100489, 87240]

print("=== baseline: plain greedy (module path) ===")
import mlx.core as mx2  # noqa: E402
from deepseek_v41_mlx.generate import greedy_generate  # noqa: E402
base = greedy_generate(model, PROMPT, max_new_tokens=48)
print(f"  {len(base)} tokens")
print(f"  first 12: {base[:12]}")
print()

print("=== speculative via mtp_decode.speculative_generate (rowseq) ===")
# fresh model state: greedy_generate used its own cache, so this is independent
spec, stats = speculative_generate(model, PROMPT, max_new_tokens=48,
                                   verify="rowseq")
print(f"  {len(spec)} tokens in {stats['seconds']:.3f}s -> "
      f"{stats['tok_per_s']:.2f} tok/s")
print(f"  first 12: {spec[:12]}")
print(f"  rounds={stats['rounds']}  mean_accepted={stats['mean_accepted']:.2f}"
      f" / {stats['gamma']}")
print()

print("=== VERDICT (this validates the PACKAGED module) ===")
print(f"  baseline[:48] == module[:48] : {base[:48] == spec[:48]}")
if base[:48] != spec[:48]:
    for i, (x, y) in enumerate(zip(base, spec)):
        if x != y:
            print(f"    divergence at {i}: module={x} greedy={y}")
            break
print()
print("MODULE-VERIFY-DONE")

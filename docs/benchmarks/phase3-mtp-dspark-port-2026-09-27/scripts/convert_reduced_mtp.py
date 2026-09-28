#!/usr/bin/env python3
"""Run the CONVERTER over the reduced+MTP source tree.

Why this is required rather than optional: build_reduced_mtp.py produced a RAW
copy — the release stores routed experts as `ffn.experts.{N}.w{1,2,3}.weight`
(one tensor per expert, ~128 of them per projection), while the runtime's
SwitchGLU wants a single stacked `[n_experts, out, in]` tensor. Feeding raw
tensors to load_weights raises `KeyError: 0` deep inside mlx's update()
because tree_unflatten builds a list where the module has an array.

convert.convert() does exactly this transformation (plus fp8/fp4 dequant and the
hc_* / gate.bias dtype normalisation), and it now KEEPS mtp.* thanks to the
earlier edits. So: point it at the reduced tree, let it write a converted build.

Usage: python3 convert_reduced_mtp.py <SRC> <DST>
"""
from __future__ import annotations

import os
import sys

REPO = os.path.expanduser("~/repos/ref/deepseek-v41-mlx")
sys.path.insert(0, REPO)

SRC = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/v41-build-mtp")
DST = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/v41-mtp-converted")

from deepseek_v41_mlx.convert import convert  # noqa: E402

print(f"=== convert ===")
print(f"  src {SRC}")
print(f"  dst {DST}")
print(f"  bits=4 expert_bits=4 (affine; matches the other reduced builds)")
print()

acc = convert(SRC, DST, bits=4, expert_bits=4, engram_bits=None,
              group_size=64)

print()
print("=== accounting ===")
for k, v in acc.items():
    print(f"  {k}: {v}")
print()
print(f"  mtp kept: {acc.get('mtp_kept', 'ABSENT — edits did not take')}")

import glob  # noqa: E402
files = sorted(glob.glob(os.path.join(DST, "*.safetensors")))
print()
print(f"=== output: {len(files)} shards ===")
tot = 0
for f in files:
    sz = os.path.getsize(f)
    tot += sz
    print(f"  {os.path.basename(f):<42} {sz/1e9:6.2f} GB")
print(f"  total: {tot/1e9:.2f} GB")
print()

# confirm mtp landed in the output
import json  # noqa: E402
ip = os.path.join(DST, "model.safetensors.index.json")
if os.path.exists(ip):
    wm = json.load(open(ip))["weight_map"]
    mtp = [k for k in wm if k.startswith("mtp")]
    print(f"  index tensors: {len(wm)}  mtp: {len(mtp)}")
    for k in sorted(mtp)[:8]:
        print(f"    {k} -> {wm[k]}")
    # any raw (unstacked) expert names left?
    raw = [k for k in wm if ".experts." in k and not k.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight"))]
    print(f"  unstacked expert names remaining: {len(raw)}")
    if raw:
        print(f"    e.g. {raw[:3]}")
print()
print("CONVERT-MTP-DONE")

#!/usr/bin/env python3
"""Two questions:

(A) DEPLOYMENT BLOCKER CHECK — can production exo load the EXL3 checkpoint's
    MTP tensors? EXL3 stores them in its own trellis format
    (.trellis/.suh/.svh/.mul1), while production's sanitizer + model expect the
    affine pair (.weight/.scale). Count how many EXL3 MTP tensors are in a form
    production can consume at all.

(B) HOW MANY BODY LAYERS are available locally? The acceptance-rate measurement
    needs the deepest model I can build; this enumerates which layer indices the
    present shards actually cover.

Usage: python3 p7c_exl3_loadable_and_layers.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

EXL3 = os.path.expanduser("~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
NATIVE = os.path.expanduser("~/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram")

print("=" * 74)
print("(A) EXL3 MTP: which tensor forms does it use, and how many?")
print("=" * 74)
idx = json.load(open(os.path.join(EXL3, "model.safetensors.index.json")))
wm = idx["weight_map"]

mtp = {k: v for k, v in wm.items() if k.startswith("mtp")}
forms = Counter()
affine_ok = 0
trellis_only = 0
other = 0
for k in mtp:
    leaf = k.rsplit(".", 1)[-1]
    forms[leaf] += 1
    if leaf in ("weight", "scale", "suh", "svh", "trellis", "mul1"):
        pass

# group by the tensor's BASE name (strip the format leaf) to see which tensors
# are quantized (have a trellis set) vs which are plain scalars
bases = defaultdict(set)
for k in mtp:
    base, _, leaf = k.rpartition(".")
    bases[base].add(leaf)

quant_bases = [b for b, lv in bases.items() if "trellis" in lv]
plain_bases = [b for b, lv in bases.items() if "trellis" not in lv]
print(f"  total mtp tensors        : {len(mtp)}")
print(f"  distinct base names      : {len(bases)}")
print(f"  quantized (trellis set)  : {len(quant_bases)}")
print(f"  plain (no trellis)       : {len(plain_bases)}")
print()
print("  format leaf histogram:")
for l, c in forms.most_common():
    print(f"    {l:<20} {c:>5}")
print()
print("  example quantized tensor's full leaf set:")
if quant_bases:
    b = sorted(quant_bases)[0]
    print(f"    {b}  ->  {sorted(bases[b])}")
print("  example plain tensor's leaves:")
if plain_bases:
    b = sorted(plain_bases)[0]
    print(f"    {b}  ->  {sorted(bases[b])}")

print()
print("  Can production consume these? Production's sanitizer/loader expects")
print("  `.weight` + `.scale` (affine). EXL3 quantized tensors have NO `.scale`:")
n_no_scale = sum(1 for b in quant_bases if "scale" not in bases[b])
print(f"    quantized bases lacking a `.scale` leaf: {n_no_scale} / {len(quant_bases)}")
n_weight = sum(1 for b in quant_bases if "weight" in bases[b])
print(f"    quantized bases that DO carry `.weight`: {n_weight}")

print()
print("=" * 74)
print("(A2) Does production have ANY exl3-aware loading path?")
print("=" * 74)
r = os.popen("grep -rln 'trellis\\|suh\\|svh\\|mul1\\|exl3' /home/hermes/repos/exo/mlx-lm/ "
             "/home/hermes/repos/exo/src/exo/worker/engines/mlx/ 2>/dev/null | head -20")
hits = [l.strip() for l in r if l.strip()]
print(f"  files mentioning exl3/trellis in mlx-lm + mlx engine: {len(hits)}")
for h in hits:
    print(f"    {h}")

print()
print("=" * 74)
print("(B) Which body layers do the locally-present shards cover?")
print("=" * 74)
nat_idx = json.load(open(os.path.join(NATIVE, "model.safetensors.index.json")))
nwm = nat_idx["weight_map"]
present = set()
for f in os.listdir(NATIVE):
    if f.endswith(".safetensors"):
        present.add(f)

# map layer -> shards, then filter to shards we have
layer_shards = defaultdict(set)
for k, sh in nwm.items():
    if k.startswith("layers."):
        layer = int(k.split(".")[1])
        layer_shards[layer].add(sh)

have_layers = sorted(L for L, shs in layer_shards.items() if shs <= present)
partial = sorted(L for L, shs in layer_shards.items() if shs & present and not shs <= present)
missing = sorted(L for L, shs in layer_shards.items() if not (shs & present))
print(f"  shards present            : {len(present & set(nwm.values()))} / {len(set(nwm.values()))}")
print(f"  layers FULLY present      : {len(have_layers)}")
print(f"    {have_layers}")
if partial:
    print(f"  layers PARTIALLY present  : {len(partial)}")
    for L in partial[:10]:
        print(f"    L{L}: shards {sorted(layer_shards[L])} present={sorted(layer_shards[L] & present)}")
if missing:
    print(f"  layers absent             : {len(missing)} (lowest: {missing[:8]})")

# also: the kv/index/engram config for the reduced build
cfg = json.load(open(os.path.join(NATIVE, "config.json")))
tc = cfg.get("text_config", cfg)
print()
print(f"  config: {tc['num_hidden_layers']} layers, "
      f"kv_src={tc.get('kv_source_layer_ids')}, "
      f"compress_ratios={tc.get('compress_ratios')}")
print(f"  engram_layer_ids={tc.get('engram_layer_ids')}")
print()
print("DONE")

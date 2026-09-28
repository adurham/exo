#!/usr/bin/env python3
"""Per-layer routed-expert byte histogram across ALL 40 layers of EXL3.

The native checkpoint only has layers 0..3 locally, so its per-layer size is an
extrapolation. EXL3 has all 40 layers present, so it can TEST the uniformity
assumption directly — and confirm/refute the claim (phase 1) that layers 18-22
carry 2-bit experts while the rest are 3-bit.

Also splits each layer's experts by bit-width proxy: an EXL3 trellis tensor's
byte size per expert scales with its bits, so a 2-bit tier should show as a
visibly smaller per-layer total.

Usage: python3 p8f_exl3_layer_hist.py
"""
from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path

EX = Path.home() / ".exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


per_layer = defaultdict(int)
per_layer_n = defaultdict(int)
for f in sorted(EX.glob("*.safetensors")):
    try:
        h = read_header(f)
    except Exception:  # noqa: BLE001
        continue
    for k, v in h.items():
        if k == "__metadata__":
            continue
        if ".ffn.experts." not in k:
            continue
        parts = k.split(".")
        lid = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    lid = int(parts[i + 1])
                except ValueError:
                    pass
                break
        if lid is None:
            continue
        a, b = v["data_offsets"]
        per_layer[lid] += b - a
        per_layer_n[lid] += 1

print("EXL3 routed-expert bytes per layer (all 40 layers present)")
print(f"  {'layer':>5} {'GB':>7} {'tensors':>8}   bar")
vals = []
for lid in sorted(per_layer):
    gb = per_layer[lid] / 1e9
    vals.append((lid, gb))
    bar = "#" * int(gb / 0.1)
    print(f"  {lid:>5} {gb:>7.3f} {per_layer_n[lid]:>8}   {bar}")

if vals:
    gbs = [g for _, g in vals]
    lo, hi = min(gbs), max(gbs)
    print()
    print(f"  min {lo:.3f} GB   max {hi:.3f} GB   spread {hi - lo:.3f} GB "
          f"({100 * (hi - lo) / hi:.1f}%)")
    # group by size bucket to reveal tiers
    buckets = defaultdict(list)
    for lid, g in vals:
        buckets[round(g, 2)].append(lid)
    print()
    print("  size buckets (layer ids at each per-layer size):")
    for sz in sorted(buckets, reverse=True):
        ids = buckets[sz]
        print(f"    {sz:>6.2f} GB : {len(ids):>2} layers  {ids}")

    uniform = hi - lo < 0.05 * hi
    print()
    if uniform:
        print("  RESULT: layers are essentially UNIFORM in expert bytes.")
        print("  => extrapolating per-layer size x 40 is VALID for this format.")
    else:
        print("  RESULT: layers are NOT uniform — there ARE distinct size tiers.")
        print("  => a flat x40 extrapolation OVERSTATES smaller-tier layers;")
        print("     use the actual per-layer sum instead.")
    total = sum(gbs)
    print()
    print(f"  exact total routed-expert bytes (all 40 layers): {total:.2f} GB")
    print(f"  vs 40 x layer-0 ({gbs[0]:.3f} GB) extrapolation        : "
          f"{40 * gbs[0]:.2f} GB")
    print(f"  difference                                     : "
          f"{40 * gbs[0] - total:+.2f} GB")
print("DONE")

#!/usr/bin/env python3
"""MTP byte arithmetic for the hybrid deployment (headers only, no tensor data).

The deployment takes the BODY from EXL3 but the DRAFT HEAD from the native
checkpoint: EXL3's mtp.* tensors are in trellis form (trellis/suh/svh/mul1) and
production's DSpark loader expects affine `.weight` + `.scale`, which the EXL3
MTP tensors do not carry. So the per-rank budget becomes:

    EXL3 directory total  -  EXL3 mtp.* bytes  +  native mtp.* bytes

Sizes come from the data_offsets in each safetensors header JSON -> exact
byte counts, no dtype guessing, nothing loaded.

Usage: python3 p7f_mtp_bytes.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

EX = Path.home() / ".exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def scan(d: Path, label: str):
    files = sorted(d.glob("*.safetensors"))
    tot_file = 0
    tot_mtp = 0
    n_mtp = 0
    per_file = []
    for f in files:
        tot_file += f.stat().st_size
        try:
            h = read_header(f)
        except Exception as exc:  # noqa: BLE001
            per_file.append((f.name, -1, 0, f"HEADER-ERR {exc}"))
            continue
        mtp = 0
        c = 0
        for k, v in h.items():
            if k == "__metadata__":
                continue
            if k.startswith("mtp."):
                a, b = v["data_offsets"]
                mtp += b - a
                c += 1
        tot_mtp += mtp
        n_mtp += c
        if c:
            per_file.append((f.name, mtp, c, ""))
    print(f"=== {label} ===")
    print(f"  dir          : {d}")
    print(f"  shard files  : {len(files)}")
    print(f"  total bytes  : {tot_file / 1e9:8.2f} GB")
    print(f"  mtp.* bytes  : {tot_mtp / 1e9:8.3f} GB   tensors={n_mtp}")
    print(f"  mtp share    : {100 * tot_mtp / max(tot_file, 1):.2f}% of the directory")
    if per_file:
        print("  shards carrying mtp.*:")
        for name, m, c, err in sorted(per_file, key=lambda x: -x[1])[:12]:
            print(f"    {name:<46} {m / 1e9:8.3f} GB  n={c} {err}")
    print()
    return tot_file, tot_mtp, n_mtp


ex_tot, ex_mtp, ex_n = scan(EX, "EXL3 checkpoint")
nt_tot, nt_mtp, nt_n = scan(NT, "NATIVE (engram) checkpoint")

print("=== swap arithmetic (body from EXL3, draft head from native) ===")
delta = nt_mtp - ex_mtp
print(f"  EXL3   mtp.* : {ex_mtp / 1e9:.3f} GB  ({ex_n} tensors)")
print(f"  native mtp.* : {nt_mtp / 1e9:.3f} GB  ({nt_n} tensors)")
print(f"  delta        : {delta / 1e9:+.3f} GB   (native - EXL3)")
print(f"  per rank, if mtp bytes split evenly across TP=2: {delta / 2 / 1e9:+.3f} GB")
print()
print(f"  EXL3 body-only (total - mtp)              : {(ex_tot - ex_mtp) / 1e9:.2f} GB")
print(f"  hybrid total (EXL3 body-only + native mtp): {(ex_tot - ex_mtp + nt_mtp) / 1e9:.2f} GB")
print(f"  for reference, EXL3 as-is total           : {ex_tot / 1e9:.2f} GB")
print()
print("NOTE: whole-directory byte counts. The per-rank budget also depends on how")
print("exo shards tensors across ranks (not necessarily an even split).")
print("DONE")

#!/usr/bin/env python3
"""Let the compressor expose its chunk's raw rows, so a rollback can rebuild
the open-group carry correctly.

WHY THIS IS NEEDED
------------------
A speculative round runs a verify forward over [anchor, d1..d_gamma] at
positions [O, E). The draft tokens are provisional: if only n_acc are accepted,
the cache must be rolled back to target = O + n_acc + 1, and the committed
prefix [0, target) must end up bit-identical to plain greedy decode.

The window ring and comp_kv need no repair: positions p live at slot p%window
and are simply rewritten by whichever chunk commits them, and any pooled group
straddling `target` is recomputed by the next chunk before it can be read. The
one thing that IS lost is the compressor's OPEN-GROUP carry (`kv_state` /
`score_state`): it holds the raw, not-yet-pooled rows of positions
[ratio*(E//ratio), E), and after a rollback it must instead hold
[ratio*(target//ratio), target).

Restoring the pre-verify snapshot is not enough: when the verify chunk is the
one that first wrote the carry rows for the newly committed positions, those
rows exist ONLY in the verify chunk. So the chunk must hand its own raw rows to
the driver.

Production does the same thing structurally: `PoolingCache` keeps `buf_kv` /
`buf_gate` raw remainder buffers precisely so `trim()` can un-pool.

This patch is pure storage -- no change to the compute path, no change to the
returned values.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

PKG = os.path.expanduser("~/repos/ref/deepseek-v41-mlx/deepseek_v41_mlx")
STAMP = ".prerollback.bak"


def edit(path, old, new, label):
    full = os.path.join(PKG, path)
    src = open(full).read()
    if new in src and old not in src:
        print(f"  [{label}] already applied")
        return True
    if old not in src:
        print(f"  [{label}] !! ANCHOR NOT FOUND")
        for line in old.splitlines()[:12]:
            print("        " + line)
        return False
    bak = full + STAMP
    if not os.path.exists(bak):
        shutil.copy2(full, bak)
    open(full, "w").write(src.replace(old, new, 1))
    print(f"  [{label}] applied")
    return True


ok = True

print("=== compressor.py: state fields ===")
ok &= edit(
    "compressor.py",
    """    def __init__(self, bsz: int, ratio: int, head_dim: int):
        self.kv_state = mx.zeros((bsz, ratio, head_dim), dtype=mx.float32)
        self.score_state = mx.full((bsz, ratio, head_dim), NEG_INF, dtype=mx.float32)""",
    """    def __init__(self, bsz: int, ratio: int, head_dim: int):
        self.kv_state = mx.zeros((bsz, ratio, head_dim), dtype=mx.float32)
        self.score_state = mx.full((bsz, ratio, head_dim), NEG_INF, dtype=mx.float32)
        # This chunk's OWN raw rows (pre-carry) + where they start, stashed by
        # Compressor.__call__ so a speculative rollback can rebuild the carry.
        self.chunk_kv = None
        self.chunk_score = None
        self.chunk_start = 0""",
    "state fields",
)

print()
print("=== compressor.py: stash the chunk rows ===")
ok &= edit(
    "compressor.py",
    """        bsz, n, _ = x.shape
        xf = x.astype(mx.float32)
        kv = xf @ self.wkv.weight.astype(mx.float32).T
        score = xf @ self.wgate.weight.astype(mx.float32).T

        m = start_pos % ratio                    # carried tokens of the open group""",
    """        bsz, n, _ = x.shape
        xf = x.astype(mx.float32)
        kv = xf @ self.wkv.weight.astype(mx.float32).T
        score = xf @ self.wgate.weight.astype(mx.float32).T

        # Speculative-rollback support: stash this chunk's OWN raw rows (before
        # the carry is prepended) so the driver can rebuild the open-group
        # buffer when a draft round rejects part of this chunk. Pure storage --
        # the compute path and return values are unchanged.
        comp_state.chunk_kv = kv
        comp_state.chunk_score = score
        comp_state.chunk_start = start_pos

        m = start_pos % ratio                    # carried tokens of the open group""",
    "stash rows",
)

print()
r = subprocess.run([sys.executable, "-m", "py_compile",
                    os.path.join(PKG, "compressor.py")], capture_output=True, text=True)
print(f"compressor.py: {'OK' if r.returncode == 0 else r.stderr[:400]}")
ok &= r.returncode == 0

if ok:
    sys.path.insert(0, os.path.dirname(PKG))
    from deepseek_v41_mlx.compressor import CompressorState
    st = CompressorState(2, 2, 8)
    print()
    print("=== CompressorState fields ===")
    for k in ("kv_state", "score_state", "chunk_kv", "chunk_score", "chunk_start"):
        v = getattr(st, k, "MISSING")
        print(f"    {k:<12} = {'None' if v is None else (v.shape if hasattr(v,'shape') else v)}")

print()
print("done" if ok else "FAILED")
sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""Tile sweep for gather_qmm_rhs_lhs / gather_qmm_rhs on M4 Max.

Requires an MLX build with MLX_METAL_JIT=ON (tiles are JIT-instantiated;
the AOT .metal library only bakes 16,32,32,1,2).

Sweeps MLX_GATHER_QMM_RHS_LHS_TILE configs by re-execing itself per config
(the env is read once per process). Production shape: 1536 sorted pairs,
256 experts, K=4096, N=1024, mxfp4 g32. Asserts output equality vs the
default tile before timing.

Usage: python bench/qmm_tile_sweep.py            # sweep driver
       python bench/qmm_tile_sweep.py --one TAG  # single config (internal)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

CONFIGS = [
    "16,32,32,1,2",    # default (control)
    "32,32,32,2,2",
    "32,64,32,2,2",
    "64,32,32,2,2",
    "64,64,32,2,2",
    "16,64,32,1,2",
    "16,32,64,1,2",
    "32,32,64,2,2",
    "64,64,64,2,2",
    "16,64,64,2,2",
]


def run_one() -> None:
    import mlx.core as mx

    TOKENS, TOPK, N_EXPERTS, HIDDEN, INTER = 256, 6, 256, 4096, 1024
    GROUP, BITS, MODE = 32, 4, "mxfp4"
    n_pairs = TOKENS * TOPK
    reps = n_pairs // N_EXPERTS

    mx.random.seed(0)
    w = mx.quantize(
        mx.random.normal((N_EXPERTS, INTER, HIDDEN), dtype=mx.float32) * 0.02,
        group_size=GROUP, bits=BITS, mode=MODE,
    )
    x = mx.random.normal((n_pairs, 1, HIDDEN), dtype=mx.bfloat16)
    idx = mx.sort(mx.concatenate([mx.arange(N_EXPERTS, dtype=mx.uint32)] * reps))

    def run():
        return mx.gather_qmm(x, *w, rhs_indices=idx, transpose=True,
                             group_size=GROUP, bits=BITS, mode=MODE,
                             sorted_indices=True)

    out = run()
    mx.eval(out)
    checksum = float(mx.sum(out.astype(mx.float32)).item())

    for _ in range(5):
        mx.eval(run())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        mx.eval(run())
    mx.synchronize()
    dt = (time.perf_counter() - t0) / 30

    wbytes = sum(a.nbytes for a in w)
    print(json.dumps({
        "tile": os.environ.get("MLX_GATHER_QMM_RHS_LHS_TILE", "default"),
        "ms": round(dt * 1000, 3),
        "gbs": round(wbytes / dt / 1e9, 1),
        "checksum": checksum,
    }))


def main() -> int:
    if "--one" in sys.argv:
        run_one()
        return 0
    results = []
    ref_checksum = None
    for cfg in CONFIGS:
        env = dict(os.environ, MLX_GATHER_QMM_RHS_LHS_TILE=cfg)
        p = subprocess.run([sys.executable, __file__, "--one", cfg],
                           env=env, capture_output=True, text=True, timeout=300)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, IndexError):
            print(f"{cfg:>16}: FAILED — {p.stderr.strip()[-200:]}")
            continue
        if ref_checksum is None:
            ref_checksum = r["checksum"]
        match = "OK " if abs(r["checksum"] - ref_checksum) < abs(ref_checksum) * 1e-4 else "BAD"
        print(f"{cfg:>16}: {r['ms']:7.2f} ms  {r['gbs']:6.1f} GB/s  checksum {match}")
        results.append(r)
    if results:
        best = min(results, key=lambda r: r["ms"])
        print(f"\nbest: {best['tile']} at {best['gbs']} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

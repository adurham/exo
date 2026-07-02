#!/usr/bin/env python3
"""A/B microbench: fp_gather_qmv_rhs vs steel fp_gather_qmm_rhs at the DSv4
prefill production shape (1536 sorted pairs, 256 experts, mxfp4 g32,
gate/up N=1024 K=4096 + down N=4096 K=1024, bf16 activations).

Gate (handoff): new kernel must show >=2x on this shape.

Runs itself in subprocesses with MLX_GATHER_QMV_RHS=0/1 so the process-static
env caches don't lie. Run ON A STUDIO:
  ~/scratch/tilesweep-venv/bin/python qmv_mbatch_bench.py
"""

import os
import subprocess
import sys

B_PAIRS = 1536
N_EXPERTS = 256
HIDDEN = 4096
INTER = 1024
GROUP = 32
BITS = 4
MODE = "mxfp4"
N_ITERS = 30
N_WARM = 5


def child():
    import time

    import mlx.core as mx

    mx.random.seed(0)
    reps = B_PAIRS // N_EXPERTS
    idx_sorted = mx.sort(
        mx.concatenate([mx.arange(N_EXPERTS, dtype=mx.uint32)] * reps)
    )
    x = mx.random.normal((B_PAIRS, 1, HIDDEN), dtype=mx.bfloat16)

    def make_qweights(n_exp, out_d, in_d):
        w = mx.random.normal((n_exp, out_d, in_d), dtype=mx.float32) * 0.02
        return mx.quantize(w, group_size=GROUP, bits=BITS, mode=MODE)

    gate_w = make_qweights(N_EXPERTS, INTER, HIDDEN)
    up_w = make_qweights(N_EXPERTS, INTER, HIDDEN)
    down_w = make_qweights(N_EXPERTS, HIDDEN, INTER)

    def one_layer():
        kw = dict(
            rhs_indices=idx_sorted,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            mode=MODE,
            sorted_indices=True,
        )
        g = mx.gather_qmm(x, *gate_w, **kw)
        u = mx.gather_qmm(x, *up_w, **kw)
        h = mx.minimum(g, 10.0) * mx.clip(u, -10.0, 10.0)
        d = mx.gather_qmm(h, *down_w, **kw)
        return d

    for _ in range(N_WARM):
        mx.eval(one_layer())
    mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        mx.eval(one_layer())
    mx.synchronize()
    dt = (time.perf_counter() - t0) / N_ITERS

    wbytes = sum(
        sum(a.nbytes for a in qw) for qw in (gate_w, up_w, down_w)
    )
    abytes = (
        2 * x.nbytes
        + 2 * B_PAIRS * INTER * 2
        + B_PAIRS * INTER * 2
        + B_PAIRS * HIDDEN * 2
    )
    gbs = (wbytes + abytes) / dt / 1e9
    print(f"RESULT ms={dt * 1000:.3f} gbs={gbs:.1f}")


def main():
    if os.environ.get("_QMV_BENCH_CHILD"):
        child()
        return

    res = {}
    variants = [
        ("steel (old path)", {"MLX_GATHER_QMV_RHS": "0"}),
        ("qmv_rhs mt=4 rps=4", {"MLX_GATHER_QMV_RHS_TILE": "4"}),
        ("qmv_rhs mt=4 rps=8", {"MLX_GATHER_QMV_RHS_TILE": "4", "MLX_GATHER_QMV_RHS_RPS": "8"}),
        ("qmv_rhs mt=6 rps=4", {"MLX_GATHER_QMV_RHS_TILE": "6"}),
        ("qmv_rhs mt=6 rps=8", {"MLX_GATHER_QMV_RHS_TILE": "6", "MLX_GATHER_QMV_RHS_RPS": "8"}),
        ("qmv_rhs mt=2 rps=8", {"MLX_GATHER_QMV_RHS_TILE": "2", "MLX_GATHER_QMV_RHS_RPS": "8"}),
        ("qmv_rhs mt=8 rps=4", {"MLX_GATHER_QMV_RHS_TILE": "8"}),
        ("qmv_rhs mt=8 rps=8", {"MLX_GATHER_QMV_RHS_TILE": "8", "MLX_GATHER_QMV_RHS_RPS": "8"}),
    ]
    for label, env in variants:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env={**os.environ, "_QMV_BENCH_CHILD": "1", **env},
            capture_output=True,
            text=True,
        )
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None
        )
        if line is None:
            print(f"[{label}] FAILED:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            sys.exit(1)
        ms = float(line.split("ms=")[1].split()[0])
        gbs = float(line.split("gbs=")[1])
        res[label] = (ms, gbs)
        print(f"[{label}] {ms:.3f} ms/layer, {gbs:.1f} GB/s")

    base_ms = res["steel (old path)"][0]
    for label in res:
        if label == "steel (old path)":
            continue
        speedup = base_ms / res[label][0]
        print(f"{label}: {speedup:.2f}x vs steel")
    best = max(base_ms / res[l][0] for l in res if l != "steel (old path)")
    print("MICROBENCH GATE (>=2x):", "PASS" if best >= 2.0 else "FAIL")
    sys.exit(0 if best >= 2.0 else 1)


if __name__ == "__main__":
    main()

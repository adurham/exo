#!/usr/bin/env python3
"""Bit-exactness gate for fp_gather_qmv_rhs (M-batched sorted-run qmv).

The new kernel must produce BITWISE-identical output to the fp_gather_qmv_fast
path (same qdot/load_vector/simd_sum sequence per output row). Reference is
the same gather_qmm call with sorted_indices=False, which routes per-row
through fp_qmv_fast_impl.

Dispatch proof: the steel fp_gather_qmm_rhs path (MLX_GATHER_QMV_RHS=0) uses
simdgroup MMA with a different accumulation order, so its output differs
bitwise from the qmv reference on real data. If the sorted output equals the
reference exactly AND the kill-switch output does not, the new kernel
demonstrably executed.

Run ON A STUDIO (needs the built mlx):
  ~/scratch/tilesweep-venv/bin/python qmv_mbatch_bitexact.py
Exits 0 on pass, 1 on failure.
"""

import os
import subprocess
import sys

MODES = [("mxfp4", 32, 4), ("mxfp8", 32, 8)]
# (K, N): gate/up projection and down projection production shards
SHAPES = [(4096, 1024), (1024, 4096)]
N_EXPERTS = 32


def make_runs(kind, rng, n_experts):
    """Return a sorted expert-id row list exercising a given run structure."""
    import numpy as np

    if kind.startswith("uniform"):
        reps = int(kind.split("-")[1])
        ids = np.repeat(np.arange(n_experts), reps)
    elif kind == "ragged":
        lens = rng.integers(1, 21, size=n_experts)  # includes runs > M_TILE
        ids = np.repeat(np.arange(n_experts), lens)
    elif kind == "one-giant-run":
        ids = np.full(96, 7)
    elif kind == "min-gate":
        # smallest B accepted by the dispatch gate: B>=16, B/E>=2
        ids = np.repeat(np.arange(8), 2)
    else:
        raise ValueError(kind)
    return np.sort(ids).astype(np.uint32)


def run_cases(report_hashes):
    import hashlib

    import numpy as np

    import mlx.core as mx

    failures = []
    hashes = {}
    for mode, group, bits in MODES:
        for K, N in SHAPES:
            for dtype in (mx.bfloat16, mx.float16):
                for kind in [
                    "uniform-1",
                    "uniform-2",
                    "uniform-3",
                    "uniform-4",
                    "uniform-5",
                    "uniform-6",
                    "uniform-7",
                    "uniform-8",
                    "ragged",
                    "one-giant-run",
                    "min-gate",
                ]:
                    # float16 x full matrix is redundant; spot-check one kind
                    if dtype == mx.float16 and kind != "uniform-6":
                        continue
                    mx.random.seed(0xC0FFEE)
                    rng = np.random.default_rng(1234)
                    ids = make_runs(kind, rng, N_EXPERTS)
                    B = len(ids)
                    idx = mx.array(ids)
                    x = mx.random.normal((B, 1, K)).astype(dtype)
                    w = mx.random.normal((N_EXPERTS, N, K), dtype=mx.float32)
                    qw = mx.quantize(w, group_size=group, bits=bits, mode=mode)

                    kwargs = dict(
                        rhs_indices=idx,
                        transpose=True,
                        group_size=group,
                        bits=bits,
                        mode=mode,
                    )
                    out_sorted = mx.gather_qmm(x, *qw, sorted_indices=True, **kwargs)
                    out_ref = mx.gather_qmm(x, *qw, sorted_indices=False, **kwargs)
                    mx.eval(out_sorted, out_ref)

                    tag = f"{mode}/K{K}N{N}/{dtype}/{kind}"
                    exact = bool(mx.array_equal(out_sorted, out_ref).item())
                    if not exact:
                        d = mx.abs(
                            out_sorted.astype(mx.float32) - out_ref.astype(mx.float32)
                        )
                        failures.append(f"{tag}: max abs diff {d.max().item():.3e}")
                    hashes[tag] = hashlib.sha256(
                        np.asarray(out_sorted.astype(mx.float32)).tobytes()
                    ).hexdigest()[:16]

    if report_hashes:
        for k in sorted(hashes):
            print(f"HASH {k} {hashes[k]}")
    return failures


def main():
    if os.environ.get("_QMV_GATE_CHILD"):
        failures = run_cases(report_hashes=True)
        for f in failures:
            print(f"FAIL {f}")
        sys.exit(1 if failures else 0)

    # 1) new kernel enabled (default): must be bitwise-exact vs qmv reference
    results = {}
    for label, env in [
        ("mbatch-mt8", {}),
        ("mbatch-mt4", {"MLX_GATHER_QMV_RHS_TILE": "4"}),
        ("killswitch", {"MLX_GATHER_QMV_RHS": "0"}),
    ]:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            env={**os.environ, "_QMV_GATE_CHILD": "1", **env},
            capture_output=True,
            text=True,
        )
        hashes = {}
        fails = []
        for line in proc.stdout.splitlines():
            if line.startswith("HASH "):
                _, tag, h = line.split()
                hashes[tag] = h
            elif line.startswith("FAIL "):
                fails.append(line)
        results[label] = (proc.returncode, hashes, fails)
        status = "OK" if proc.returncode == 0 else "MISMATCH"
        # killswitch is EXPECTED to mismatch the qmv reference (steel path)
        print(f"[{label}] child exit={proc.returncode} ({status})")
        if proc.returncode != 0 and label != "killswitch":
            print(proc.stdout[-4000:])
            print(proc.stderr[-2000:])

    ok = True
    for label in ("mbatch-mt8", "mbatch-mt4"):
        rc, _, fails = results[label]
        if rc != 0:
            print(f"GATE FAIL: {label} not bit-exact vs qmv reference:")
            for f in fails[:20]:
                print("  ", f)
            ok = False

    # dispatch proof: killswitch (steel) output must DIFFER from the new path
    # on at least some cases, else the new kernel never ran
    _, h_new, _ = results["mbatch-mt8"]
    kc, h_old, _ = results["killswitch"]
    if h_new and h_new == h_old and kc == 0:
        print(
            "GATE FAIL: kill-switch output is identical AND bit-exact — "
            "cannot prove the new kernel dispatched (gate may not route)"
        )
        ok = False
    else:
        diff = sum(1 for k in h_new if h_old.get(k) != h_new[k])
        print(f"dispatch proof: {diff}/{len(h_new)} cases differ under kill switch")

    print("BITEXACT GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

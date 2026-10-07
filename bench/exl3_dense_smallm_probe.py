#!/usr/bin/env python3
"""exl3_dense_smallm_probe -- SYNTHETIC-WEIGHTS microbench of the EXL3 dense
small-batch (M = 1..16) trellis GEMM at the real DeepSeek-V4.1-Flash per-rank
shapes (TP=2).  Phase-19 / phase-3 kernel go/no-go.

WHY SYNTHETIC: the real EXL3 checkpoint
(~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw) lives on
the two Mac Studios, which are DOWN.  The kernel's *cost* depends only on
(k, in_features, out_features) and the packed layout -- all reproduced exactly
here (k=3, packed tile = 256*k/16 = 48 uint16).  Trellis *values* are random;
the mul1 decode is an ALU/table path that runs identically whatever the
codeword, so random content does not change timing.  A correctness check
(cosine of the synthetic EXL3Linear output vs the fp16 reconstruction) is run
and printed, so the synthetic layer is shown to be a faithful stand-in.

MEASUREMENT METHOD (important): this machine's ``mx.eval`` has a large fixed
per-eval cost (~150-250 us, measured below and dominated by host/command-buffer
submission), which *swamps* a single small-kernel call.  Timing one call per
eval gave nonsense (e.g. 5.5 ms for a 7.9 MB kernel).  We therefore AMORTIZE:
each timed eval enqueues K=64 independent calls and we divide the median eval
time by K.  That divides the fixed per-eval floor by K (down to <5 us/call) and
leaves the true per-call kernel cost.  The floor is measured and reported.

ARMS
  A  the vendored EXL3Linear kernel (the thing under test) -- real per-rank shapes
  B  fp16 ceiling: reconstruct each layer to fp16 W once, time `x @ W`
     (the "if compute were free of decode" ceiling)
  C  analytic memory roofline = trellis_bytes / 450e9  (flat across M)

Run:
  PYTHONPATH=/Users/adam.durham/repos/exo/mlx-lm \
  /Users/adam.durham/repos/exo/.venv/bin/python bench/exl3_dense_smallm_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# -- make the populated mlx-lm fork importable (worktree's mlx-lm/ is empty) --
for _p in ("/Users/adam.durham/repos/exo/mlx-lm",):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
from mlx_lm.models.exl3 import EXL3Linear  # noqa: E402
from mlx_lm.models.exl3.ref.layer import EXL3Layer  # noqa: E402
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx  # noqa: E402

# ---------------------------------------------------------------------------
# Real DeepSeek-V4.1-Flash per-rank (TP=2) dense groups.  ~141 M params/layer.
# wo_a is a GROUP of 8 linears sharing one key prefix (in = 4096, out = 1024).
# ---------------------------------------------------------------------------
K = 3                      # 2.9 bpw -> packed_to_k(48) = 3
PACKED = 256 * K // 16     # 48 uint16 per tile
BW = 450e9                 # campaign's measured M4 Max streaming bandwidth
K_AMORT = 64               # calls per timed eval (divides out the eval floor)
ITERS = 9                  # timed evals per (M, linear)

# (group-name, [(in, out), ...])
GROUPS: list[tuple[str, list[tuple[int, int]]]] = [
    ("wq_a",      [(5120, 1280)]),
    ("wq_b",      [(1280, 16384)]),
    ("wkv",       [(5120, 512)]),
    ("wo_b",      [(8192, 5120)]),
    ("wo_a_x8",   [(4096, 1024)] * 8),
    ("shared_w1", [(5120, 2304)]),
    ("shared_w3", [(5120, 2304)]),
    ("shared_w2", [(2304, 5120)]),
]

MS = [1, 2, 3, 4, 5, 6, 8, 16]

OUT_JSON = ("/private/tmp/levers-wt/docs/benchmarks/phase19-latency/raw/"
            "phase3-kernel-microbench.json")


def synth_layer(key: str, in_f: int, out_f: int, rng: np.random.Generator) -> EXL3Layer:
    assert in_f % 16 == 0 and out_f % 16 == 0
    trellis = rng.integers(0, 1 << 16, size=(in_f // 16, out_f // 16, PACKED),
                           dtype=np.uint16)
    suh = rng.choice(np.array([-1.0, 1.0], dtype=np.float16), size=in_f).astype(np.float16)
    svh = rng.choice(np.array([-1.0, 1.0], dtype=np.float16), size=out_f).astype(np.float16)
    lay = EXL3Layer(key=key, in_features=in_f, out_features=out_f, k=K,
                    trellis=trellis, suh=suh, svh=svh, mcg=False, mul1=True)
    lay.validate()
    return lay


def trellis_bytes(in_f: int, out_f: int) -> int:
    return (in_f // 16) * (out_f // 16) * PACKED * 2


def amort_us(make, K=K_AMORT, iters=ITERS) -> float:
    """Median per-call us for `make()` enqueuing K independent calls per eval."""
    for _ in range(3):
        mx.eval(make(K))
    ts = []
    for _ in range(iters):
        o = make(K)
        s = time.perf_counter()
        mx.eval(o)
        ts.append(time.perf_counter() - s)
    return float(np.median(ts) / K * 1e6)


def main() -> None:
    rng = np.random.default_rng(0)
    mx.random.seed(0)

    # ---- eval floor diagnostics -------------------------------------------
    xn = mx.array([1.0])
    mx.eval(xn)
    floor = amort_us(lambda k: [xn + 1.0 for _ in range(k)], K=K_AMORT)
    print(f"mx.eval floor (amortized K={K_AMORT}): {floor:.2f} us/call  "
          f"(machine load-dependent; load avg shown in report)", flush=True)

    # ---- build linears (arm A) + fp16 reconstructions (arm B) --------------
    entries: list[dict] = []
    grand_params = 0
    grand_trellis = 0
    for gname, members in GROUPS:
        for i, (in_f, out_f) in enumerate(members):
            key = gname if len(members) == 1 else f"{gname}.{i}"
            lay = synth_layer(key, in_f, out_f, rng)
            lin = EXL3Linear(lay)
            Wt = mx.contiguous(reconstruct_public_mlx(lay))   # [in, out] fp16
            mx.eval(Wt)
            lin.release_source()                               # drop host numpy
            entries.append(dict(group=gname, key=key, in_f=in_f, out_f=out_f,
                                lin=lin, Wt=Wt,
                                tb=trellis_bytes(in_f, out_f),
                                params=in_f * out_f))
            grand_params += in_f * out_f
            grand_trellis += trellis_bytes(in_f, out_f)
    print(f"built {len(entries)} dense linears: {grand_params/1e6:.1f} M params, "
          f"trellis {grand_trellis/1e6:.1f} MB/layer  (k={K}, packed={PACKED})",
          flush=True)

    # ---- correctness check: synthetic EXL3Linear vs fp16 reconstruction ----
    e0 = entries[3]  # wo_b, largest
    xc = mx.random.normal((4, e0["in_f"])).astype(mx.float16)
    mx.eval(xc)
    ya = e0["lin"](xc).astype(mx.float32)
    yb = (xc @ e0["Wt"]).astype(mx.float32)
    mx.eval(ya, yb)
    cos = ((ya * yb).sum() /
           (mx.linalg.norm(ya) * mx.linalg.norm(yb))).item()
    print(f"correctness cos(A, B_fp16) on {e0['key']} @M=4 = {cos:.7f}", flush=True)

    # ---- dispatch_count availability --------------------------------------
    mx.metal.reset_dispatch_count()
    mx.eval(e0["lin"](xc))
    disp_probe = int(mx.metal.dispatch_count())
    print(f"mx.metal.dispatch_count() after 1 EXL3 call = {disp_probe} "
          f"({'exposed' if disp_probe else 'reports 0 / not wired -> skipped'})",
          flush=True)

    # ---- per-M sweep -------------------------------------------------------
    results: dict[int, dict] = {}
    for M in MS:
        xs = {}
        for e in entries:
            x = mx.random.normal((M, e["in_f"])).astype(mx.float16)
            mx.eval(x)
            xs[e["key"]] = x

        us_A = us_B = 0.0
        per_lin = []
        for e in entries:
            x = xs[e["key"]]
            ua = amort_us(lambda k, l=e["lin"], xx=x: [l(xx) for _ in range(k)])
            ub = amort_us(lambda k, w=e["Wt"], xx=x: [xx @ w for _ in range(k)])
            us_A += ua
            us_B += ub
            per_lin.append(dict(key=e["key"], group=e["group"],
                                in_f=e["in_f"], out_f=e["out_f"],
                                us_A=ua, us_B=ub, tbytes=e["tb"],
                                gbps_A=e["tb"] / (ua * 1e-6) / 1e9))
        ms_C = grand_trellis / BW * 1e3
        results[M] = dict(us_A=us_A, us_B=us_B, ms_C=ms_C, per_lin=per_lin)
        print(f"M={M:2d}  A {us_A:8.1f} us/layer  B {us_B:8.1f} us/layer  "
              f"C {ms_C:6.3f} ms/layer  "
              f"A_eff {grand_trellis/(us_A*1e-6)/1e9:5.1f} GB/s", flush=True)

    # ---- table + projections ----------------------------------------------
    ms_C = grand_trellis / BW * 1e3
    print("\n" + "=" * 96)
    print("SYNTHETIC-WEIGHTS  M-sweep of the EXL3 dense small-batch GEMM "
          "(DSv4.1-Flash, per-rank TP=2, 40-layer model)")
    print("=" * 96)
    hdr = f"{'M':>4} {'A us/layer':>11} {'A x40 ms':>9} {'B us/layer':>11} " \
          f"{'B x40 ms':>9} {'C x40 ms':>9} {'A GB/s':>8}"
    print(hdr)
    print("-" * len(hdr))
    for M in MS:
        r = results[M]
        print(f"{M:>4} {r['us_A']:>11.1f} {r['us_A']*40/1000:>9.3f} "
              f"{r['us_B']:>11.1f} {r['us_B']*40/1000:>9.3f} "
              f"{ms_C*40:>9.3f} {grand_trellis/(r['us_A']*1e-6)/1e9:>8.1f}")

    A = {M: results[M]["us_A"] for M in MS}
    C_us = ms_C * 1e3
    d54 = A[5] - A[4]
    d65 = A[6] - A[5]
    d84 = A[8] - A[4]
    d21 = A[2] - A[1]
    print(f"\nCRITICAL DELTAS (us/layer):  A(4)={A[4]:.1f}  A(5)={A[5]:.1f}  "
          f"A(6)={A[6]:.1f}  A(8)={A[8]:.1f}")
    print(f"  A(5)-A(4) = {d54:+.1f} us   A(6)-A(5) = {d65:+.1f} us   "
          f"A(8)-A(4) = {d84:+.1f} us   A(2)-A(1) = {d21:+.1f} us")

    A4_40 = A[4] * 40 / 1000
    A1_40 = A[1] * 40 / 1000
    C_40 = ms_C * 40
    headroom_roofline = A4_40 - C_40
    smallm_penalty = A4_40 - 1.25 * A1_40
    ratio_41 = A[4] / A[1]
    print(f"\nPROJECTIONS at M=4 (all dense projections, x40 layers):")
    print(f"  A(M=4) x40 = {A4_40:.3f} ms   A(M=1) x40 = {A1_40:.3f} ms   "
          f"C x40 = {C_40:.3f} ms")
    print(f"  roofline headroom   (A4_40 - C_40)        = {headroom_roofline:+.3f} ms/round")
    print(f"  small-M penalty     (A4_40 - 1.25*A1_40)  = {smallm_penalty:+.3f} ms/round")
    print(f"  A(M=4)/A(M=1) = {ratio_41:.3f}   "
          f"A(M=4)/max(A(M=1),C_us) = {A[4]/max(A[1], C_us):.3f}")

    # ---- verdict rubric ----------------------------------------------------
    exhausted = (A[4] <= 1.25 * max(A[1], C_us)) and (abs(d54 - d65) <= 0.10 * max(1.0, abs(d54)))
    go = (ratio_41 >= 1.6) and (max(headroom_roofline, 0.0) >= 2.0)
    verdict = "EXHAUSTED" if exhausted else ("GO" if go else "INCONCLUSIVE")
    print(f"\nVERDICT: {verdict}")

    # ---- machine-readable summary -----------------------------------------
    summary = dict(
        synthetic_weights=True,
        eval_floor_us_per_call=floor,
        correctness_cos_vs_fp16=cos,
        dispatch_count_exposed=bool(disp_probe),
        k=K, packed=PACKED, bandwidth_GBs=BW / 1e9,
        amort_K=K_AMORT, iters=ITERS,
        n_linears=len(entries), params_per_layer=grand_params,
        trellis_bytes_per_layer=grand_trellis,
        table={str(M): dict(us_A=results[M]["us_A"], us_B=results[M]["us_B"],
                            ms_C=ms_C,
                            A_x40_ms=results[M]["us_A"] * 40 / 1000,
                            B_x40_ms=results[M]["us_B"] * 40 / 1000,
                            C_x40_ms=ms_C * 40,
                            gbps_A=grand_trellis / (results[M]["us_A"] * 1e-6) / 1e9)
               for M in MS},
        deltas_us=dict(A2_minus_A1=d21, A5_minus_A4=d54,
                       A6_minus_A5=d65, A8_minus_A4=d84),
        projections_ms_per_round=dict(
            A4_x40=A4_40, A1_x40=A1_40, C_x40=C_40,
            roofline_headroom=headroom_roofline,
            smallm_penalty=smallm_penalty,
            ratio_A4_over_A1=ratio_41),
        verdict=verdict,
        per_linear={str(M): results[M]["per_lin"] for M in MS},
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON -> {OUT_JSON}")
    print("JSON_BEGIN")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_linear"}, indent=2))
    print("JSON_END")


if __name__ == "__main__":
    main()

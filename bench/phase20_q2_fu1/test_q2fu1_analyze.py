#!/usr/bin/env python3
# Copyright © 2026 Adam Durham (hermes-gw).
"""OFFLINE self-test for ``q2fu1_analyze.py`` -- SYNTHETIC merged npz fixtures.

No cluster, no GPU, no checkpoint: pure numpy builds tiny synthetic dumps in the
PINNED format (``<arm>_M<M>_<tag>.merged.npz``; D=8, V=16, 4 body layers,
topk=2, 6 rows) that encode each hypothesis signature, then asserts the
analyzer reads out the right thing:

  * first-differing router layer (row0 vs the M=1 chain),
  * the routing-IDENTICAL vs routing-DIFFERED delta-logit buckets,
  * per-layer residual-delta curves + step detection,
  * router margin at the flip,
  * H1 / H2 / H3 classification,
  * the rollback A/B H3 signature,
  * the noise floor + allsum/fence control arms,
  * NaN rank-slice masking and the single-rank fallback,
  * missing-file handling,
  * the CLI writes q2fu1_analysis.json + q2fu1_analysis.md.

Run directly (pytest-compatible too):

  PYTHONPATH=bench/phase20_q2_fu1 /usr/bin/python3 \
    bench/phase20_q2_fu1/test_q2fu1_analyze.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import q2fu1_analyze as A  # noqa: E402

D = 8
V = 16
NL = 4
TOPK = 2
NROWS_REF = 6
NAN_FROM = V // 2        # columns >= 8 are NaN (rank-slice stand-in)

_N = [0]


def check(cond: bool, msg: str) -> None:
    _N[0] += 1
    if not cond:
        raise AssertionError(f"SELF-TEST FAILED: {msg}")


# ---------------------------------------------------------------------------
# fixture builders (pinned format)
# ---------------------------------------------------------------------------
def _base_hid(L: int) -> np.ndarray:
    return np.full(D, 1.0 + 0.1 * L, np.float32)


def row(*, hid_delta=None, idx=None, margin=None, logit_delta=0.0,
        nan_cols=None, finite_to=None):
    """One row of a forward: residual/idx/margin per layer + a logit vector.

    ``hid_delta``: {L: delta}.  ``idx``: {L: [ids]} (router top-k SET).
    ``margin``: {L: value}.  ``logit_delta`` fills the finite slice uniformly.
    """
    hid = {L: _base_hid(L) + float((hid_delta or {}).get(L, 0.0))
           for L in range(NL)}
    ids = {L: np.array((idx or {}).get(L, [2 * L, 2 * L + 1]), np.int32)
           for L in range(NL)}
    marg = {L: np.float32((margin or {}).get(L, 1.0)) for L in range(NL)}
    lg = np.zeros(V, np.float32)
    hi = NAN_FROM if finite_to is None else finite_to
    lg[:hi] = logit_delta
    lg[hi:] = np.nan
    for c in (nan_cols or []):
        lg[c] = np.nan
    return {"hid": hid, "idx": ids, "marg": marg, "logits": lg}


def build_dump(path: str, M: int, rows: list, *, tag: str = "X", arm: str = "qa"):
    arrays = {
        "M": np.int64(M), "start_pos": np.int64(40), "gamma": np.int64(0),
        "round_idx": np.int64(-1), "arm": np.array(arm), "rank": np.int64(0),
        "vocab_lo": np.int64(0), "vocab_hi": np.int64(V),
        "n_captured_layers": np.int64(NL),
        "rows_ids": np.array(list(range(100, 100 + M)), np.int32),
        "out_ids": np.zeros(M, np.int32),
        "rows_txt": np.array("rows"), "flags": np.array(""),
        "note": np.array("synthetic"), "dtype_logits": np.array("float32"),
        "hidden_in": np.stack([r["hid"][0] for r in rows]).astype(np.float32),
        "hidden_out": np.stack([r["hid"][NL - 1] for r in rows]).astype(np.float32),
    }
    for L in range(NL):
        arrays[f"idx_{L}"] = np.stack([r["idx"][L] for r in rows]).astype(np.int32)
        arrays[f"w_{L}"] = np.full((M, TOPK), 0.5, np.float32)
        arrays[f"hid_{L}"] = np.stack([r["hid"][L] for r in rows]).astype(np.float32)
        arrays[f"margin_{L}"] = np.array([r["marg"][L] for r in rows], np.float32)
    arrays["logits"] = np.stack([r["logits"] for r in rows]).astype(np.float32)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _w(capdir: str, name: str, M: int, rows: list, *, arm: str = "qa",
       merged: bool = True):
    tag = name
    fname = f"{arm}_M{M}_{tag}" + (".merged.npz" if merged else ".npz")
    build_dump(os.path.join(capdir, fname), M, rows, tag=tag, arm=arm)


def write_refs(capdir: str, *, merged: bool = True):
    for r in range(NROWS_REF):
        _w(capdir, f"s{r}", 1, [row()], merged=merged)


def write_common_arms(capdir: str, base_M4_rows: list, *, merged: bool = True,
                      allsum_fp32_delta: float = 2e-3,
                      fence_delta: float = 1e-4):
    # noise floor: M4 repeat is byte-identical to M4
    _w(capdir, "M4rep2", 4, list(base_M4_rows), merged=merged)
    # allsum arm: fp32 moe-allreduce (perturb the finite slice)
    m1 = row()
    _w(capdir, "M1", 1, [m1], merged=merged)
    m1f = row()
    m1f["logits"][:NAN_FROM] = allsum_fp32_delta
    m1f["logits"][NAN_FROM:] = np.nan
    _w(capdir, "M1fp32", 1, [m1f], merged=merged)
    m4f = [dict(r) for r in base_M4_rows]
    for r in m4f:
        lg = np.array(r["logits"], copy=True)
        lg[:NAN_FROM] = lg[:NAN_FROM] + allsum_fp32_delta
        r["logits"] = lg
    _w(capdir, "M4fp32", 4, m4f, merged=merged)
    # fence arm
    m1fe = row()
    m1fe["logits"][:NAN_FROM] = fence_delta
    m1fe["logits"][NAN_FROM:] = np.nan
    _w(capdir, "M1fence0", 1, [m1fe], merged=merged)
    m4fe = [dict(r) for r in base_M4_rows]
    for r in m4fe:
        lg = np.array(r["logits"], copy=True)
        lg[:NAN_FROM] = lg[:NAN_FROM] + fence_delta
        r["logits"] = lg
    _w(capdir, "M4fence0", 4, m4fe, merged=merged)


def write_rollback(capdir: str, collapse_gammas=(), *, merged: bool = True,
                   a_large: float = 0.9, others: float = 0.0):
    for gm in A.GAMMAS:
        if gm in collapse_gammas:
            ra = row(logit_delta=a_large)
            rb = row(logit_delta=0.0)         # == ref
        else:
            ra = row(logit_delta=others)
            rb = row(logit_delta=others)
        _w(capdir, f"g{gm}_A_m1", 1, [ra], merged=merged)
        _w(capdir, f"g{gm}_B_m1", 1, [rb], merged=merged)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_h1_capdir(root: str, *, merged: bool = True) -> str:
    cap = os.path.join(root, "h1")
    os.makedirs(cap, exist_ok=True)
    write_refs(cap, merged=merged)
    base_m4 = []
    for M in (3, 4, 5, 6):
        rows = [row(hid_delta={0: 1e-4, 1: 1e-4, 2: 5e-3, 3: 5e-3},
                    idx={2: [5, 6]}, margin={2: 0.001}, logit_delta=0.3)]
        for _ in range(M - 1):
            rows.append(row(hid_delta={L: 1e-5 for L in range(NL)},
                            logit_delta=1e-4))
        _w(cap, f"M{M}", M, rows, merged=merged)
        if M == 4:
            base_m4 = rows
    write_common_arms(cap, base_m4, merged=merged)
    write_rollback(cap, collapse_gammas=(), merged=merged)
    return cap


def make_h2_capdir(root: str, *, merged: bool = True) -> str:
    cap = os.path.join(root, "h2")
    os.makedirs(cap, exist_ok=True)
    write_refs(cap, merged=merged)
    base_m4 = []
    for M in (3, 4, 5, 6):
        rows = [row(hid_delta={0: 1e-4, 1: 1e-4, 2: 0.5, 3: 0.5},
                    logit_delta=0.5) for _ in range(M)]
        _w(cap, f"M{M}", M, rows, merged=merged)
        if M == 4:
            base_m4 = rows
    write_common_arms(cap, base_m4, merged=merged)
    write_rollback(cap, collapse_gammas=(), merged=merged)
    return cap


def make_h3_capdir(root: str, *, merged: bool = True) -> str:
    cap = os.path.join(root, "h3")
    os.makedirs(cap, exist_ok=True)
    write_refs(cap, merged=merged)
    base_m4 = []
    for M in (3, 4, 5, 6):
        rows = [row(hid_delta={L: 1e-4 for L in range(NL)}, logit_delta=0.7)
                for _ in range(M)]
        _w(cap, f"M{M}", M, rows, merged=merged)
        if M == 4:
            base_m4 = rows
    write_common_arms(cap, base_m4, merged=merged)
    write_rollback(cap, collapse_gammas=(3,), merged=merged)  # B collapses
    return cap


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_first_differing_layer():
    with tempfile.TemporaryDirectory() as root:
        a = A.analyze(make_h1_capdir(root))
        m3 = a["per_M"]["M3"]
        check(m3["first_routing_flip_layer_row0"] == 2,
              f"(a) H1: first router flip layer == 2 (got "
              f"{m3['first_routing_flip_layer_row0']})")
        check(m3["first_resid_exceed_layer_row0"] is None,
              "(a) H1: residual stays <= SMALL_RESID (ulp-scale) -> no exceed")
        check(m3["resid_threshold"] == A.SMALL_RESID,
              "(a) residual threshold reported")
        # H2: routing never flips, but the residual steps over the threshold
        a2 = A.analyze(make_h2_capdir(root))
        m4 = a2["per_M"]["M4"]
        check(m4["first_routing_flip_layer_row0"] is None,
              "(a) H2: no router flip")
        check(m4["first_resid_exceed_layer_row0"] == 2,
              f"(a) H2: first residual exceed layer == 2 (got "
              f"{m4['first_resid_exceed_layer_row0']})")


def test_buckets():
    with tempfile.TemporaryDirectory() as root:
        a = A.analyze(make_h1_capdir(root))
        b = a["per_M"]["M4"]["logit_buckets"]
        check(b["row0"]["bucket"] == "differed",
              "(b) H1: row0 routing flipped -> 'differed' bucket")
        check(b["routing_differed"]["rows"] == [0],
              f"(b) H1: differed bucket == row0 only (got "
              f"{b['routing_differed']['rows']})")
        check(len(b["routing_identical"]["rows"]) == 3,
              "(b) H1: rows 1..3 in the identical bucket")
        check(b["routing_identical"]["max_abs"] < A.LARGE_LOGIT,
              f"(b) H1: identical-bucket max {b['routing_identical']['max_abs']} "
              f"< LARGE ({A.LARGE_LOGIT}) -> H1 signal")
        check(abs(b["routing_differed"]["max_abs"] - 0.3) < 1e-6,
              f"(b) H1: differed-bucket max ~0.3 (got "
              f"{b['routing_differed']['max_abs']})")
        check(b["routing_identical"]["mean_abs"] is not None
              and b["routing_differed"]["mean_abs"] is not None,
              "(b) both buckets report a mean")


def test_residual_curve_and_step():
    with tempfile.TemporaryDirectory() as root:
        a = A.analyze(make_h2_capdir(root))
        v = a["per_M"]["M4"]
        r0 = [c["abs_delta"] for c in v["residual_curve_row0"]]
        check(len(r0) == NL, "(c) residual curve has one value per layer")
        check(abs(r0[0] - 1e-4) < 1e-7 and abs(r0[2] - 0.5) < 1e-6,
              f"(c) H2 curve steps at L2: {r0}")
        check(v["resid_step_layer"] == 2,
              f"(c) H2 step-change layer == 2 (got {v['resid_step_layer']})")
        check(v["resid_step_routing_identical"] is True,
              "(c) H2 step occurs with routing identical")
        # H1: no step (ulp-scale)
        a1 = A.analyze(make_h1_capdir(root))
        check(a1["per_M"]["M4"]["resid_step_layer"] is None,
              "(c) H1: no step change detected")
        mx = a1["per_M"]["M4"]["residual_curve_max_over_rows"]
        check(len(mx) == NL and all(c["max_abs_delta"] is not None for c in mx),
              "(c) max-over-rows curve computed for every layer")


def test_margin_and_classification():
    with tempfile.TemporaryDirectory() as root:
        h1 = A.analyze(make_h1_capdir(root))
        v = h1["per_M"]["M4"]
        check(abs(v["router_margin_at_first_flip"] - 0.001) < 1e-7,
              f"(d) margin at flip == 0.001 (got "
              f"{v['router_margin_at_first_flip']})")
        check(v["classification"]["hypothesis"] == "H1",
              "(e) H1 fixture classified as H1")
        check(h1["classification"]["overall"] == "H1",
              "(e) H1 overall == H1")

        h2 = A.analyze(make_h2_capdir(root))
        check(h2["per_M"]["M4"]["classification"]["hypothesis"] == "H2",
              "(e) H2 fixture classified as H2")
        check(h2["classification"]["overall"] == "H2",
              "(e) H2 overall == H2")

        h3 = A.analyze(make_h3_capdir(root))
        check(h3["per_M"]["M4"]["classification"]["hypothesis"] == "H3",
              "(e) H3 fixture (large identical-routing delta) -> H3")
        rb = h3["rollback_ab"]["g3"]
        check(rb["present"] and rb["B_collapse"] is True,
              "(e) H3: rollback g3 B collapses to the M=1 ref while A stays large")
        check(rb["A_vs_ref_max_abs"] > A.A_LARGE
              and rb["B_vs_ref_max_abs"] < A.COLLAPSE_TOL,
              f"(e) H3 numbers A={rb['A_vs_ref_max_abs']} B={rb['B_vs_ref_max_abs']}")
        check(h3["rollback_ab"]["g2"]["B_collapse"] is False,
              "(e) H3: a non-collapsing gamma (g2) is not flagged")
        check(h3["classification"]["overall"] == "H3",
              "(e) H3 overall == H3")


def test_noise_floor_and_arms():
    with tempfile.TemporaryDirectory() as root:
        a = A.analyze(make_h1_capdir(root))
        nf = a["noise_floor"]
        check(nf is not None and nf["max_abs_delta"] == 0.0,
              f"(f) noise floor (M4 vs M4rep2) == 0.0 (got {nf})")
        check(nf["bitwise_equal"] is True, "(f) noise floor bitwise equal")
        check(abs(a["arms"]["allsum_M1"]["max_abs_delta"] - 2e-3) < 1e-7,
              "(f) allsum M1 arm delta ~2e-3")
        check(abs(a["arms"]["allsum_M4"]["max_abs_delta"] - 2e-3) < 1e-7,
              "(f) allsum M4 arm delta ~2e-3")
        check(abs(a["arms"]["fence_M1"]["max_abs_delta"] - 1e-4) < 1e-9,
              "(f) fence M1 arm delta ~1e-4")
        check(abs(a["arms"]["fence_M4"]["max_abs_delta"] - 1e-4) < 1e-9,
              "(f) fence M4 arm delta ~1e-4")


def test_nan_rank_slice_masking():
    with tempfile.TemporaryDirectory() as root:
        cap = os.path.join(root, "nan")
        os.makedirs(cap, exist_ok=True)
        write_refs(cap)
        # ref: finite [0,8)==0, NaN [8,16). target: same finite region, but a
        # HUGE value placed at column 10 (NaN in ref) must be excluded, and a
        # small real delta at column 3 must be counted.
        r = row()
        tgt = row()
        lg = np.array(tgt["logits"], copy=True)
        lg[3] = 0.25                 # finite&finite -> counted
        lg[10] = 99.0                # finite in target, NaN in ref -> excluded
        tgt["logits"] = lg
        _w(cap, "M3", 3, [tgt, row(), row()])
        a = A.analyze(cap)
        got = a["per_M"]["M3"]["logit_buckets"]["row0"]["max_abs"]
        check(abs(got - 0.25) < 1e-6,
              f"(mask) NaN-slice excluded, finite delta 0.25 read (got {got})")


def test_single_rank_fallback():
    with tempfile.TemporaryDirectory() as root:
        cap = make_h1_capdir(root, merged=False)   # plain npz, no .merged
        a = A.analyze(cap)
        files = a["files_used"]
        check(all(f.endswith(".npz") and ".merged.npz" not in f for f in files),
              "(fallback) analyzer reads plain single-rank npz")
        check(a["per_M"]["M3"]["first_routing_flip_layer_row0"] == 2
              and a["per_M"]["M4"]["classification"]["hypothesis"] == "H1",
              "(fallback) plain-dump read-out matches the merged read-out")


def test_missing_files():
    with tempfile.TemporaryDirectory() as root:
        cap = make_h1_capdir(root)
        os.remove(os.path.join(cap, "qa_M5_M5.merged.npz"))
        a = A.analyze(cap)
        check(a["per_M"]["M5"]["present"] is False,
              "(missing) absent M5 marked not present")
        check(a["per_M"]["M4"]["present"] is True,
              "(missing) present shapes still analysed")
        check(any("M5" in n for n in a["notes"]),
              f"(missing) note recorded (notes={a['notes']})")


def test_cli_outputs():
    with tempfile.TemporaryDirectory() as root:
        cap = make_h1_capdir(root)
        out = os.path.join(root, "out")
        rc = A.main(["--capdir", cap, "--out", out])
        check(rc == 0, "(cli) exit code 0")
        jp = os.path.join(out, "q2fu1_analysis.json")
        mp = os.path.join(out, "q2fu1_analysis.md")
        check(os.path.exists(jp) and os.path.exists(mp),
              "(cli) q2fu1_analysis.json + .md written")
        with open(jp, encoding="utf-8") as fh:
            j = json.load(fh)
        check(j["classification"]["overall"] == "H1", "(cli) json verdict H1")
        md = open(mp, encoding="utf-8").read()
        check("H1" in md and "(a)" in md, "(cli) markdown has verdict + sections")


def test_driver_not_broken():
    """Smoke: the driver's own analyze_captures still runs on these fixtures."""
    try:
        import q2fu1_driver as drv
    except Exception:  # pragma: no cover
        check(True, "(driver) driver not importable offline; skipped")
        return
    with tempfile.TemporaryDirectory() as root:
        cap = make_h1_capdir(root)
        an = drv.analyze_captures(cap)
        check(isinstance(an, dict) and "shape_readout" in an,
              "(driver) driver analyze_captures unaffected (returns read-out)")


_ALL_TESTS = [test_first_differing_layer, test_buckets,
              test_residual_curve_and_step, test_margin_and_classification,
              test_noise_floor_and_arms, test_nan_rank_slice_masking,
              test_single_rank_fallback, test_missing_files,
              test_cli_outputs, test_driver_not_broken]


def main() -> int:
    import time
    import traceback
    t0 = time.time()
    passed = 0
    for fn in _ALL_TESTS:
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
            continue
        passed += 1
        print(f"  ok   {fn.__name__}")
    print(f"\nSELF-TEST {'OK' if passed == len(_ALL_TESTS) else 'FAILED'} "
          f"({passed}/{len(_ALL_TESTS)} test fns, {_N[0]} assertions, "
          f"{time.time() - t0:.2f}s)")
    return 0 if passed == len(_ALL_TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())

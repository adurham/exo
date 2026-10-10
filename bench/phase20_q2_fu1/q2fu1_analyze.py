#!/usr/bin/env python3
# Copyright © 2026 Adam Durham (hermes-gw).
"""Round Q2-FU1 offline DEEP-ANALYSIS CLI -- the reference-spec read-out.

Consumes the pinned per-forward ``.npz`` dumps written by ``q2fu1_capture.py``
(``<arm>_M<M>_<tag>.npz`` per rank, ``<arm>_M<M>_<tag>.merged.npz`` after the
driver splices both ranks) and emits the full reference-spec read-out as
``q2fu1_analysis.json`` + a human-readable ``q2fu1_analysis.md``.

Read-out implemented (see bench/phase20_q2_fu1/README.md, 'Pinned dump format'
+ the round brief):

  (a) For each M in {3,4,5,6} vs the M=1 chain reference (``s{r}``): the FIRST
      body layer index where row0's MoE router top-k SET (``idx_L``) differs,
      AND the first layer where row0's residual delta |hid_L - ref_hid_L|
      exceeds ``SMALL_RESID`` (the threshold is reported).
  (b) max |delta-logit| (row0, and per-row) SPLIT into bucket (i) routing
      IDENTICAL across ALL layers vs the reference, and bucket (ii) routing
      differs -- the H1-vs-H2/H3 discriminator (small identical-bucket => H1;
      large => H2/H3).  Both buckets' max and mean are reported.
  (c) A per-layer residual-delta curve (|delta| per body layer, row0 and
      max-over-rows) so a STEP CHANGE at a specific layer (H2) is visible vs
      ulp-scale noise with isolated jumps at top-k-flip layers (H1).
  (d) The router margin (``margin_L``) at the first flipped layer (near zero
      => near-tie => benign H1 amplification).
  (e) An explicit H1/H2/H3 classification + confidence + the exact numbers,
      plus the rollback A/B H3 test: production A (``g{gm}_A_m1``) vs
      ring-restored B (``g{gm}_B_m1``) row0 logits vs the M=1 reference -- if
      B's |delta| collapses toward ~0 while A stays large => direct H3
      evidence for that gamma.  A vs B is cross-checked directly too.
  (f) Noise floor (M4 vs M4rep2), allsum arm (M1 vs M1fp32, M4 vs M4fp32),
      fence arm (M1 vs M1fence0, M4 vs M4fence0).

OFFLINE ONLY: numpy + the local dump dir.  No cluster, no GPU, no mlx import.
The driver's ``_load`` / ``merge_all`` helpers are reused (not duplicated); if
the capdir holds only the per-rank subdirs, the analyzer runs the driver's
merge first.  Rank-merged logits carry NaN outside each rank's vocab slice ->
only finite&finite positions are compared; single-rank (plain) dumps work too.

Usage:
  python3 bench/phase20_q2_fu1/q2fu1_analyze.py --capdir <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:  # reuse the driver's load/merge helpers where sensible
    import q2fu1_driver as drv  # type: ignore
except Exception:  # pragma: no cover - driver optional (analyzer is standalone)
    drv = None


# ---------------------------------------------------------------------------
# thresholds (sane defaults; all overridable on the CLI and echoed in output)
# ---------------------------------------------------------------------------
SMALL_RESID = 1e-2     # residual delta considered "ulp-scale" (=> H1 region)
STEP_ABS = 5e-2        # absolute residual delta needed for a step change
STEP_FACTOR = 5.0      # ratio jump (vs running max) needed for a step change
LARGE_LOGIT = 1e-1     # |delta-logit| (nats) considered "large" under ident.
COLLAPSE_TOL = 1e-2    # B-vs-ref "collapse toward zero" tolerance (H3)
A_LARGE = 1e-1         # A-vs-ref "stays large" floor (H3)
NEAR_TIE_MARGIN = 5e-2  # |margin| below this => near-tie => H1 amplification
MS = (3, 4, 5, 6)
GAMMAS = (2, 3, 4, 5)
MAX_ROWS = 6           # s0..s5 are the row references

_NPZ_RE = re.compile(
    r"(?P<arm>.+)_M(?P<M>\d+)_(?P<tag>.+?)(?P<merged>\.merged)?\.npz$")


# ---------------------------------------------------------------------------
# loading / index
# ---------------------------------------------------------------------------
def _load(path: str) -> dict:
    """Reuse the driver's loader when available (identical semantics)."""
    if drv is not None and hasattr(drv, "_load"):
        return drv._load(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _maybe_merge(capdir: str, notes: list) -> None:
    """If only per-rank subdirs are present, run the driver's merge in place."""
    s1 = os.path.join(capdir, "rank_studio1")
    s2 = os.path.join(capdir, "rank_studio2")
    have_merged = any(f.endswith(".merged.npz") for f in os.listdir(capdir)
                      if not f.startswith("._"))
    if have_merged or not (os.path.isdir(s1) and os.path.isdir(s2)):
        return
    if drv is None or not hasattr(drv, "merge_all"):
        notes.append("rank_* subdirs present but the driver merge is "
                     "unavailable; analyzer reads plain npz only")
        return
    try:
        res = drv.merge_all(capdir)
        notes.append(f"driver merge_all: {len(res['merged'])} pairs merged, "
                     f"{len(res['problems'])} with problems")
        if res["problems"]:
            notes.append(f"merge problems: {sorted(res['problems'])}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"driver merge_all raised: {exc!r}")


def _tag_index(capdir: str) -> tuple:
    """Return (by_tag, files_used, notes).  Prefers ``*.merged.npz``."""
    notes: list = []
    _maybe_merge(capdir, notes)
    all_npz = sorted(f for f in os.listdir(capdir)
                     if f.endswith(".npz") and not f.startswith("._"))
    merged = [f for f in all_npz if f.endswith(".merged.npz")]
    files = merged or all_npz
    if merged and len(merged) < len(all_npz):
        notes.append("using *.merged.npz (rank-spliced); plain dumps ignored")
    if not files:
        notes.append("no .npz dumps found in capdir")
    by_tag: dict = {}
    for f in files:
        m = _NPZ_RE.match(f)
        if not m:
            notes.append(f"unparsed npz ignored: {f}")
            continue
        by_tag.setdefault(m.group("tag"), os.path.join(capdir, f))
    return by_tag, files, notes


def _get(by_tag: dict, tag: str):
    """Exact tag, else the shortest tag that starts with ``tag + '_'`` (so a
    round-suffixed rollback tag ``g3_A_m1_r1`` satisfies ``g3_A_m1``)."""
    if tag in by_tag:
        return _load(by_tag[tag])
    pref = sorted(t for t in by_tag if t.startswith(tag + "_"))
    if pref:
        return _load(by_tag[pref[0]])
    return None


# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------
def _scalar(d: dict, key: str, default=None):
    if key not in d:
        return default
    try:
        return int(np.asarray(d[key]).item())
    except Exception:  # noqa: BLE001
        return default


def _routing_layers(d: dict) -> list:
    Ls = []
    for k in d:
        if k.startswith("idx_"):
            try:
                Ls.append(int(k[4:]))
            except ValueError:
                pass
    return sorted(Ls)


def _resid_layers(d: dict) -> list:
    Ls = []
    for k in d:
        if k.startswith("hid_"):
            try:
                Ls.append(int(k[4:]))
            except ValueError:
                pass
    return sorted(Ls)


def _max_abs_finite(a, b):
    """max |a-b| over positions finite in BOTH (rank slice NaN handling)."""
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if aa.shape != bb.shape:
        return None
    m = np.isfinite(aa) & np.isfinite(bb)
    if not m.any():
        return None
    return float(np.max(np.abs(aa[m] - bb[m])))


def _route_set(d: dict, key: str, r: int):
    if key not in d:
        return None
    arr = np.asarray(d[key])
    if r >= arr.shape[0]:
        return None
    return frozenset(int(x) for x in arr[r].ravel().tolist())


def _agg(items: list) -> dict:
    """items: list of (row, value).  Returns n/rows/max_abs/mean_abs."""
    vals = [v for _, v in items]
    if not vals:
        return {"n": 0, "rows": [], "max_abs": None, "mean_abs": None}
    return {"n": len(vals), "rows": [r for r, _ in items],
            "max_abs": float(max(vals)),
            "mean_abs": float(sum(vals) / len(vals))}


def _detect_step(curve: list, step_abs: float, step_factor: float):
    """First index k>=1 where curve[k] jumps above the running max by both an
    absolute (step_abs) and a ratio (step_factor) margin -- the H2 step change.
    Returns the index into ``curve`` or None."""
    if not curve or len(curve) < 2:
        return None
    prev_max = curve[0]
    for k in range(1, len(curve)):
        c = curve[k]
        if c is not None and c > step_abs and (prev_max <= 0.0 or c >= step_factor * prev_max):
            return k
        if c is not None and c > prev_max:
            prev_max = c
    return None


# ---------------------------------------------------------------------------
# the read-out
# ---------------------------------------------------------------------------
def _ref_row(by_tag: dict, r: int):
    """The M=1 chain reference dump for row r (``s{r}``; row0 falls back to
    ``M1``).  Returns (tag, dump) or (None, None)."""
    cands = [f"s{r}"]
    if r == 0:
        cands.append("M1")
    for t in cands:
        if t in by_tag:
            return t, _load(by_tag[t])
    return None, None


def _residual_curve(d: dict, by_tag: dict, row: int, rows: int):
    """[(layer, abs_delta), ...] for ``row`` vs its M=1 reference."""
    out = []
    _, ref = _ref_row(by_tag, row)
    if ref is None:
        return out
    for L in _resid_layers(d):
        k = f"hid_{L}"
        if k not in ref:
            continue
        v = _max_abs_finite(np.asarray(d[k])[row], np.asarray(ref[k])[0])
        out.append((L, v))
    return out


def _residual_curve_max_rows(d: dict, by_tag: dict, rows: int):
    """[(layer, max_over_rows abs_delta), ...]."""
    out = []
    for L in _resid_layers(d):
        k = f"hid_{L}"
        mx = None
        for r in range(min(rows, MAX_ROWS)):
            _, ref = _ref_row(by_tag, r)
            if ref is None or k not in ref:
                continue
            v = _max_abs_finite(np.asarray(d[k])[r], np.asarray(ref[k])[0])
            if v is not None:
                mx = v if mx is None else max(mx, v)
        out.append((L, mx))
    return out


def _classify_M(rec: dict, small_resid: float, large_logit: float,
                near_tie: float) -> tuple:
    """Return (hypothesis, confidence, reasons) for one M (row0 read-out)."""
    reasons: list = []
    step_L = rec["resid_step_layer"]
    ident_max = rec["logit_buckets"]["routing_identical"]["max_abs"]
    pre = rec["pre_flip_resid_max_row0"]
    first_flip = rec["first_routing_flip_layer_row0"]

    if step_L is not None and rec["resid_step_routing_identical"]:
        reasons.append(
            f"residual-delta STEP CHANGE at layer {step_L} "
            f"(delta {rec['resid_step_delta']:.3g}) with MoE routing IDENTICAL "
            f"through that layer -> kernel-path/precision discontinuity")
        return "H2", 0.85, reasons

    if ident_max is not None and ident_max > large_logit:
        reasons.append(
            f"large |delta-logit| {ident_max:.3g} nats under IDENTICAL routing "
            f"(> {large_logit}) not explained by router flips -> semantic/kernel "
            f"divergence")
        return "H3", 0.6, reasons

    if pre is not None and pre <= small_resid:
        conf = 0.8
        reasons.append(
            f"routing-identical residual max {pre:.3g} <= {small_resid} "
            f"(ulp-scale; no step change)")
        m = rec["router_margin_at_first_flip"]
        if first_flip is not None and m is not None and abs(m) <= near_tie:
            conf = 0.9
            reasons.append(
                f"first flip at layer {first_flip} has near-zero router margin "
                f"{m:.3g} (<= {near_tie}: near-tie -> benign amplification)")
        if first_flip is None:
            reasons.append("no router top-k flip detected: fully benign shape")
        return "H1", conf, reasons

    reasons.append(
        "residual deltas are NOT ulp-scale in the identical-routing region and "
        "no clean step change was detected -> inconclusive (read as H3-ish)")
    return "INCONCLUSIVE", 0.3, reasons


def _analyze_M(by_tag: dict, M: int, *, small_resid: float, step_abs: float,
               step_factor: float, large_logit: float, near_tie: float) -> dict:
    d = _get(by_tag, f"M{M}")
    if d is None:
        return {"M": M, "present": False}
    rows = _scalar(d, "M", M) or M
    R = _routing_layers(d)
    ref0_tag, ref0 = _ref_row(by_tag, 0)

    # (a) first differing router layer (row0) + first residual exceedance
    flips = []
    first_flip = None
    first_resid_exceed = None
    for L in R:
        if ref0 is not None:
            a = _route_set(d, f"idx_{L}", 0)
            b = _route_set(ref0, f"idx_{L}", 0)
            if a is not None and b is not None and a != b:
                flips.append(L)
                if first_flip is None:
                    first_flip = L
    resid_row0 = _residual_curve(d, by_tag, 0, rows)
    for L, v in resid_row0:
        if v is not None and v > small_resid:
            first_resid_exceed = L
            break

    # (b) per-row routing identity + logit deltas, bucketed
    per_row = []
    for r in range(rows):
        rtag, refr = _ref_row(by_tag, r)
        rec = {"row": r, "reference_tag": rtag, "routing_identical": None,
               "max_abs_delta_logit": None}
        if refr is None:
            rec["note"] = "no M=1 reference dump for this row"
            per_row.append(rec)
            continue
        ident = True
        for L in R:
            a = _route_set(d, f"idx_{L}", r)
            b = _route_set(refr, f"idx_{L}", 0)
            if a is None or b is None:
                continue
            if a != b:
                ident = False
                break
        rec["routing_identical"] = ident
        rec["max_abs_delta_logit"] = _max_abs_finite(
            np.asarray(d["logits"])[r], np.asarray(refr["logits"])[0])
        per_row.append(rec)

    ident_items = [(x["row"], x["max_abs_delta_logit"]) for x in per_row
                   if x["routing_identical"] is True
                   and x["max_abs_delta_logit"] is not None]
    diff_items = [(x["row"], x["max_abs_delta_logit"]) for x in per_row
                  if x["routing_identical"] is False
                  and x["max_abs_delta_logit"] is not None]
    row0 = per_row[0] if per_row else {}
    bucket_row0 = ("identical" if row0.get("routing_identical") is True else
                   "differed" if row0.get("routing_identical") is False else
                   "unknown")

    # (c) residual curves
    resid_maxrows = _residual_curve_max_rows(d, by_tag, rows)

    # (d) margin at the first flip
    margin_at_flip = None
    if first_flip is not None and f"margin_{first_flip}" in d:
        try:
            margin_at_flip = float(np.asarray(d[f"margin_{first_flip}"])[0])
        except Exception:  # noqa: BLE001
            margin_at_flip = None

    # step-change detection on the row0 residual curve
    curve_vals = [v for _, v in resid_row0 if v is not None]
    curve_Ls = [L for L, v in resid_row0 if v is not None]
    step_idx = _detect_step(curve_vals, step_abs, step_factor)
    step_L = curve_Ls[step_idx] if step_idx is not None else None
    step_delta = (curve_vals[step_idx] if step_idx is not None else None)
    routing_ident_through_step = (step_L is None) or all(
        (f > step_L) for f in flips) or not flips

    pre_flip_vals = [v for L, v in resid_row0
                     if v is not None and (first_flip is None or L < first_flip)]
    pre_flip_max = float(max(pre_flip_vals)) if pre_flip_vals else (
        float(max(curve_vals)) if curve_vals else None)

    rec = {
        "M": M, "present": True, "rows": rows,
        "layers": R,
        "reference_tag_row0": ref0_tag,
        "first_routing_flip_layer_row0": first_flip,
        "routing_flip_layers_row0": flips,
        "first_resid_exceed_layer_row0": first_resid_exceed,
        "resid_threshold": small_resid,
        "logit_buckets": {
            "row0": {"row": 0, "bucket": bucket_row0,
                     "routing_identical": row0.get("routing_identical"),
                     "max_abs": row0.get("max_abs_delta_logit"),
                     "mean_abs": row0.get("max_abs_delta_logit")},
            "routing_identical": _agg(ident_items),
            "routing_differed": _agg(diff_items),
            "per_row": per_row,
        },
        "residual_curve_row0": [{"layer": L, "abs_delta": v}
                                for L, v in resid_row0],
        "residual_curve_max_over_rows": [{"layer": L, "max_abs_delta": v}
                                         for L, v in resid_maxrows],
        "resid_step_layer": step_L,
        "resid_step_delta": step_delta,
        "resid_step_routing_identical": bool(routing_ident_through_step),
        "pre_flip_resid_max_row0": pre_flip_max,
        "router_margin_first_flip_layer": first_flip,
        "router_margin_at_first_flip": margin_at_flip,
    }
    hyp, conf, reasons = _classify_M(rec, small_resid, large_logit, near_tie)
    rec["classification"] = {"hypothesis": hyp, "confidence": conf,
                             "reasons": reasons}
    return rec


def _rollback_readout(by_tag: dict, ref0) -> dict:
    out: dict = {}
    for gm in GAMMAS:
        A = _get(by_tag, f"g{gm}_A_m1")
        B = _get(by_tag, f"g{gm}_B_m1")
        if A is None or B is None:
            out[f"g{gm}"] = {"present": False}
            continue
        la = np.asarray(A["logits"])[0]
        lb = np.asarray(B["logits"])[0]
        a_vs_b = _max_abs_finite(la, lb)
        a_vs_ref = b_vs_ref = None
        routing_ident_A = routing_ident_B = None
        if ref0 is not None:
            lr = np.asarray(ref0["logits"])[0]
            a_vs_ref = _max_abs_finite(la, lr)
            b_vs_ref = _max_abs_finite(lb, lr)
            RA = _routing_layers(A)
            for L in RA:
                if f"idx_{L}" not in ref0:
                    continue
                routing_ident_A = routing_ident_A is not False and (
                    _route_set(A, f"idx_{L}", 0) == _route_set(ref0, f"idx_{L}", 0))
            if not RA:
                routing_ident_A = None
            RB = _routing_layers(B)
            for L in RB:
                if f"idx_{L}" not in ref0:
                    continue
                routing_ident_B = routing_ident_B is not False and (
                    _route_set(B, f"idx_{L}", 0) == _route_set(ref0, f"idx_{L}", 0))
            if not RB:
                routing_ident_B = None
        collapse = (b_vs_ref is not None and b_vs_ref <= COLLAPSE_TOL
                    and a_vs_ref is not None and a_vs_ref > A_LARGE
                    and a_vs_b is not None and a_vs_b > A_LARGE)
        out[f"g{gm}"] = {
            "present": True,
            "A_vs_ref_max_abs": a_vs_ref,
            "B_vs_ref_max_abs": b_vs_ref,
            "A_vs_B_max_abs": a_vs_b,
            "A_stays_large": (a_vs_ref is not None and a_vs_ref > A_LARGE),
            "B_collapse": bool(collapse),
            "routing_identical_A": routing_ident_A,
            "routing_identical_B": routing_ident_B,
        }
    return out


def _arm_delta(a, b) -> dict:
    return {"max_abs_delta": _max_abs_finite(np.asarray(a["logits"]),
                                             np.asarray(b["logits"]))}


def analyze(capdir: str, *, small_resid: float = SMALL_RESID,
            step_abs: float = STEP_ABS, step_factor: float = STEP_FACTOR,
            large_logit: float = LARGE_LOGIT, near_tie: float = NEAR_TIE_MARGIN
            ) -> dict:
    by_tag, files, notes = _tag_index(capdir)
    out: dict = {
        "capdir": capdir,
        "files_used": files,
        "n_files": len(files),
        "tags": sorted(by_tag),
        "thresholds": {
            "SMALL_RESID": small_resid, "STEP_ABS": step_abs,
            "STEP_FACTOR": step_factor, "LARGE_LOGIT": large_logit,
            "NEAR_TIE_MARGIN": near_tie, "COLLAPSE_TOL": COLLAPSE_TOL,
            "A_LARGE": A_LARGE,
        },
        "notes": notes,
        "per_M": {},
        "rollback_ab": {},
        "noise_floor": None,
        "arms": {},
        "classification": {},
    }

    ref0_tag, ref0 = _ref_row(by_tag, 0)
    out["reference_tag_row0"] = ref0_tag
    for r in range(MAX_ROWS):
        t, _ = _ref_row(by_tag, r)
        if t is not None:
            out.setdefault("reference_tags", {})[f"row{r}"] = t

    for M in MS:
        rec = _analyze_M(by_tag, M, small_resid=small_resid, step_abs=step_abs,
                         step_factor=step_factor, large_logit=large_logit,
                         near_tie=near_tie)
        out["per_M"][f"M{M}"] = rec
        if not rec.get("present"):
            notes.append(f"M{M}: dump missing, skipped")

    # (e) rollback A/B -- direct H3 evidence
    out["rollback_ab"] = _rollback_readout(by_tag, ref0)

    # (f) noise floor: M4 vs M4rep2
    m4 = _get(by_tag, "M4")
    rep = _get(by_tag, "M4rep2")
    if m4 is not None and rep is not None:
        a = np.asarray(m4["logits"])
        b = np.asarray(rep["logits"])
        per_row = []
        if a.shape == b.shape:
            for r in range(a.shape[0]):
                per_row.append({"row": r, "max_abs_delta":
                                _max_abs_finite(a[r], b[r])})
            bitwise = bool(np.array_equal(a, b, equal_nan=True))
        else:
            bitwise = False
        out["noise_floor"] = {
            "M": 4, "ref_tag": "M4", "repeat_tag": "M4rep2",
            "max_abs_delta": _max_abs_finite(a, b),
            "bitwise_equal": bitwise, "per_row": per_row,
        }
    else:
        notes.append("noise floor: M4 or M4rep2 missing")

    # (f) allsum + fence control arms
    for name, t0, t1 in (("allsum_M1", "M1", "M1fp32"),
                         ("allsum_M4", "M4", "M4fp32"),
                         ("fence_M1", "M1", "M1fence0"),
                         ("fence_M4", "M4", "M4fence0")):
        a, b = _get(by_tag, t0), _get(by_tag, t1)
        if a is None or b is None:
            out["arms"][name] = {"present": False, "a": t0, "b": t1}
            notes.append(f"arm {name}: {t0}/{t1} missing")
            continue
        d = _arm_delta(a, b)
        d.update({"present": True, "a": t0, "b": t1})
        out["arms"][name] = d

    # (e) overall classification
    per_M = {k: v["classification"]["hypothesis"] for k, v in out["per_M"].items()
             if v.get("present")}
    collapse_gms = [gm for gm, v in out["rollback_ab"].items()
                    if v.get("B_collapse")]
    labels = list(per_M.values())
    sev = {"H3": 3, "H2": 2, "H1": 1, "INCONCLUSIVE": 0}
    reasons: list = []
    if collapse_gms:
        overall = "H3"
        conf = 0.9
        reasons.append(
            f"rollback H3 signature for {collapse_gms}: ring-restored B "
            f"collapses to the M=1 reference (B_vs_ref <= {COLLAPSE_TOL}) while "
            f"production A stays large -- the rejected verify rows clobber "
            f"live sliding-window slots")
    elif "H3" in labels:
        overall = "H3"
        conf = 0.6
        reasons.append("large identical-routing delta at >=1 shape (H3)")
    elif "H2" in labels:
        overall = "H2"
        conf = 0.75
        reasons.append("residual step change under identical routing (H2)")
    elif labels and all(l == "H1" for l in labels):
        overall = "H1"
        conf = 0.8
        reasons.append("all shapes: ulp-scale residual, isolated router "
                       "near-tie flips only -> benign amplification")
    else:
        overall = "INCONCLUSIVE"
        conf = 0.3
        reasons.append("mixed/insufficient evidence across shapes")
    out["classification"] = {
        "overall": overall, "confidence": conf, "reasons": reasons,
        "per_M": per_M,
        "per_M_confidence": {k: v["classification"]["confidence"]
                             for k, v in out["per_M"].items() if v.get("present")},
        "rollback_collapse_gammas": collapse_gms,
    }
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


def to_markdown(a: dict) -> str:
    t = a["thresholds"]
    md = ["# Q2-FU1 deep analysis (reference-spec read-out)", ""]
    md.append(f"- capdir: `{a['capdir']}`  ({a['n_files']} npz, "
              f"{len(a['tags'])} tags)")
    md.append(f"- row-0 M=1 reference tag: `{a.get('reference_tag_row0')}`")
    md.append(f"- thresholds: residual ulp-scale `SMALL_RESID={t['SMALL_RESID']}`, "
              f"step `abs>{t['STEP_ABS']} & ratio>={t['STEP_FACTOR']}`, "
              f"large logit `>{t['LARGE_LOGIT']}` nats, near-tie margin "
              f"`<={t['NEAR_TIE_MARGIN']}`, B-collapse `<={t['COLLAPSE_TOL']}`")
    md += ["", "## Verdict",
           f"**{a['classification'].get('overall')}** "
           f"(confidence {a['classification'].get('confidence')})"]
    for r in a["classification"].get("reasons", []):
        md.append(f"- {r}")

    md += ["", "## (a) first differing layer (row0 vs M=1 chain)"]
    md.append("| M | first router top-k flip | first |resid|>thr | thr |")
    md.append("|---|---|---|---|")
    for k, v in a["per_M"].items():
        if not v.get("present"):
            md.append(f"| {k} | (missing) | | |")
            continue
        md.append(f"| {k} | {v['first_routing_flip_layer_row0']} | "
                  f"{v['first_resid_exceed_layer_row0']} | "
                  f"{_fmt(v['resid_threshold'])} |")

    md += ["", "## (b) |delta-logit| buckets (H1-vs-H2/H3 discriminator)"]
    md.append("| M | row0 max | bucket | ident n/max/mean | differed n/max/mean |")
    md.append("|---|---|---|---|---|")
    for k, v in a["per_M"].items():
        if not v.get("present"):
            continue
        b = v["logit_buckets"]
        md.append(
            f"| {k} | {_fmt(b['row0']['max_abs'])} | {b['row0']['bucket']} | "
            f"{b['routing_identical']['n']}/"
            f"{_fmt(b['routing_identical']['max_abs'])}/"
            f"{_fmt(b['routing_identical']['mean_abs'])} | "
            f"{b['routing_differed']['n']}/"
            f"{_fmt(b['routing_differed']['max_abs'])}/"
            f"{_fmt(b['routing_differed']['mean_abs'])} |")

    md += ["", "## (c) residual-delta curve (row0 | max-over-rows)"]
    for k, v in a["per_M"].items():
        if not v.get("present"):
            continue
        r0 = {c["layer"]: c["abs_delta"] for c in v["residual_curve_row0"]}
        rx = {c["layer"]: c["max_abs_delta"] for c in v["residual_curve_max_over_rows"]}
        cells = " ".join(
            f"L{L}:{_fmt(r0.get(L))}|{_fmt(rx.get(L))}" for L in v["layers"])
        md.append(f"- {k}: {cells}")

    md += ["", "## (d) router margin at first flip"]
    md.append("| M | flip layer | margin | near-tie |")
    md.append("|---|---|---|---|")
    for k, v in a["per_M"].items():
        if not v.get("present"):
            continue
        m = v["router_margin_at_first_flip"]
        tie = (m is not None and abs(m) <= t["NEAR_TIE_MARGIN"])
        md.append(f"| {k} | {v['router_margin_first_flip_layer']} | {_fmt(m)} | {tie} |")

    md += ["", "## (e) per-shape classification",
           "| M | hypothesis | confidence | first flip | step layer | reasons |",
           "|---|---|---|---|---|---|"]
    for k, v in a["per_M"].items():
        if not v.get("present"):
            continue
        c = v["classification"]
        md.append(f"| {k} | {c['hypothesis']} | {c['confidence']} | "
                  f"{v['first_routing_flip_layer_row0']} | {v['resid_step_layer']} | "
                  f"{'; '.join(c['reasons'])} |")

    md += ["", "## (e) rollback A/B (production vs +ring-restore) vs M=1 ref",
           "| gamma | A vs ref | B vs ref | A vs B | B collapse | ident A/B |",
           "|---|---|---|---|---|---|"]
    for k, v in a["rollback_ab"].items():
        if not v.get("present"):
            md.append(f"| {k} | (missing) | | | | |")
            continue
        md.append(f"| {k} | {_fmt(v['A_vs_ref_max_abs'])} | "
                  f"{_fmt(v['B_vs_ref_max_abs'])} | {_fmt(v['A_vs_B_max_abs'])} | "
                  f"{v['B_collapse']} | {v['routing_identical_A']}/"
                  f"{v['routing_identical_B']} |")

    md += ["", "## (f) noise floor + control arms"]
    nf = a["noise_floor"]
    if nf:
        md.append(f"- noise floor (M4 vs M4rep2): max|delta| = "
                  f"{_fmt(nf['max_abs_delta'])} (bitwise_equal={nf['bitwise_equal']})")
    else:
        md.append("- noise floor: n/a")
    for name, v in a["arms"].items():
        if v.get("present"):
            md.append(f"- {name} ({v['a']} vs {v['b']}): max|delta| = "
                      f"{_fmt(v['max_abs_delta'])}")
        else:
            md.append(f"- {name}: missing")

    if a.get("notes"):
        md += ["", "## notes"] + [f"- {n}" for n in a["notes"]]
    return "\n".join(md) + "\n"


def write_outputs(a: dict, outdir: str) -> tuple:
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "q2fu1_analysis.json")
    mpath = os.path.join(outdir, "q2fu1_analysis.md")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(a, fh, indent=1, default=str)
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(a))
    return jpath, mpath


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="q2fu1_analyze")
    ap.add_argument("--capdir", required=True,
                    help="dir of pinned npz dump(s) (merged preferred)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--small-resid", type=float, default=SMALL_RESID)
    ap.add_argument("--step-abs", type=float, default=STEP_ABS)
    ap.add_argument("--step-factor", type=float, default=STEP_FACTOR)
    ap.add_argument("--large-logit", type=float, default=LARGE_LOGIT)
    ap.add_argument("--near-tie", type=float, default=NEAR_TIE_MARGIN)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.capdir):
        print(f"capdir not found: {args.capdir}", file=sys.stderr)
        return 2
    analysis = analyze(args.capdir, small_resid=args.small_resid,
                       step_abs=args.step_abs, step_factor=args.step_factor,
                       large_logit=args.large_logit, near_tie=args.near_tie)
    jpath, mpath = write_outputs(analysis, args.out)
    print(f"[q2fu1_analyze] {analysis['n_files']} dumps, "
          f"{len(analysis['tags'])} tags")
    print(f"[q2fu1_analyze] VERDICT: {analysis['classification']['overall']} "
          f"(confidence {analysis['classification']['confidence']})")
    for r in analysis["classification"]["reasons"]:
        print(f"  - {r}")
    for k, v in analysis["per_M"].items():
        if v.get("present"):
            c = v["classification"]
            print(f"  {k}: {c['hypothesis']} (conf {c['confidence']}); "
                  f"first_flip={v['first_routing_flip_layer_row0']} "
                  f"step={v['resid_step_layer']} "
                  f"row0_dlogit={_fmt(v['logit_buckets']['row0']['max_abs'])} "
                  f"ident_max={_fmt(v['logit_buckets']['routing_identical']['max_abs'])}")
    print(f"[q2fu1_analyze] wrote {jpath}")
    print(f"[q2fu1_analyze] wrote {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

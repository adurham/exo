#!/usr/bin/env python3
"""Q2-Gamma evaluators — the pre-registered gates, pure stdlib, NO cluster.

Round Q2-Gamma re-prices the dsv41 speculative draft depth ``gamma`` on the
post-dense production build.  This module implements the four pre-registered
gates that judge a finished arm matrix; it NEVER touches the cluster, the kit,
or the network.  Every function takes plain records so the whole module is
unit-testable on synthetic inputs (see ``selftest.py`` / the pytest wrapper).

Arm-record schema consumed here (one dict per arm)::

    {
      "gamma": int,
      "logprobs_available": bool,           # top-2 logps captured for the identity signal
      "benign":  [rep, ...],                # benign 20K fixed replays
      "agentic": [rep, ...],                # agentic 91K fixed replays
      "peak_alloc_bytes":   {node: bytes},  # VM exo_peak_memory_bytes / 1.073741824 (CORRECTED)
      "peak_resident_bytes":{node: bytes},  # footprint -f bytes phys_footprint_peak
    }

Each rep (produced by ``q2_gamma_driver.replay_once`` + ``RM.derive``) carries
at least: ``completion_tokens``, ``decode_s``, ``ttft_s``, ``tokens`` (the
first ~256 emitted tokens), ``top2_logprobs`` (list of ``(lp_top1, lp_top2)``
aligned to ``tokens``), ``hist_delta`` (per-position accepted-draft histogram
DELTA for the request) and ``mean_accepted_hist``.

The four gates
--------------
* ``identity_gate``   — determinism control: gamma3a vs each arm, first-divergence
                        index + top-1/top-2 logit margin; PASS/ABORT.
* ``drift_control``   — gamma3a vs gamma3b decode tok/s drift > 1.5% => RE-RUN.
* ``bars``            — agentic winner must beat the control median by >= 3% OR
                        have disjoint IQRs; benign must not be worse than 1.5%
                        and its IQR must not fall wholly below gamma3's; a split
                        => "per-workload gamma is a future item" (do NOT ship).
* ``memory_gate``     — an arm whose corrected peak is within 0.5 GB of W is
                        ineligible to win even if faster.
"""
from __future__ import annotations

import math
import statistics

# ---------------------------------------------------------------------------
# Pre-registered constants (fixed BEFORE the run; changes are amendments)
# ---------------------------------------------------------------------------
EPSILON = 0.05  # identity: confident-position threshold on the top1-top2 logprob margin
DRIFT_PCT = 0.015  # gamma3a-vs-gamma3b decode tok/s relative drift that flags RE-RUN
AGENTIC_WIN_PCT = 0.03  # agentic must beat the control median by >= 3%
BENIGN_TOL_PCT = 0.015  # benign must not be worse than 1.5% on median
MEM_MARGIN_GB = 0.5  # within 0.5 GB of W => memory-ineligible
ALLOC_CORRECTION = 1.073741824  # VM gauge over-reports true bytes by x1.073741824
W_MB = 120000  # wired limit MiB (iogpu.wired_limit_mb)
W_BYTES = W_MB * 1024 * 1024  # = 125_829_120_000 B = 125.829 GB
GB = 1_000_000_000  # decimal GB (footprint / the campaign's unit convention)
SUPPORTED_GAMMA = (2, 3, 4, 5)

# The literal "<= ~2x the gamma3a-vs-gamma3b rate" rule collapses to 0 when the
# determinism control is byte-identical (the expected case).  We pre-register an
# absolute allowance floor so an ISOLATED near-tie flip (the D4-documented,
# expected behaviour of a gamma-changing lever) does not abort the whole round.
# Divergence RATE is measured at the *captured-position* level (first ~256).
RATE_FLOOR = 0.05


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def median(xs) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def quantile(xs, q: float) -> float | None:
    """Linear-interpolation quantile (inclusive).  Returns None on empty input."""
    vals = sorted(float(x) for x in xs if x is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def iqr(xs) -> tuple[float | None, float | None]:
    """Interquartile range (q1, q3).  None-padded when the sample is empty."""
    return quantile(xs, 0.25), quantile(xs, 0.75)


def iqr_disjoint(a, b) -> bool:
    """True iff the (q1,q3) boxes of ``a`` and ``b`` do NOT overlap."""
    a1, a3 = iqr(a)
    b1, b3 = iqr(b)
    if a1 is None or a3 is None or b1 is None or b3 is None:
        return False
    return a3 < b1 or b3 < a1


def iqr_wholly_below(a, b) -> bool:
    """True iff ``a``'s whole IQR box sits strictly below ``b``'s (q3_a < q1_b)."""
    a1, a3 = iqr(a)
    b1, b3 = iqr(b)
    if a1 is None or a3 is None or b1 is None or b3 is None:
        return False
    return a3 < b1


# ---------------------------------------------------------------------------
# Per-rep metrics
# ---------------------------------------------------------------------------
def decode_s_per_output_token(rec) -> float | None:
    """THE metric of this round: decode seconds per output token, prefill excluded.

    ``decode_s`` spans first-token -> last-token (inter-token time only), so
    dividing by the output-token count excludes the prefill entirely.
    """
    ds, ct = rec.get("decode_s"), rec.get("completion_tokens")
    if not ds or not ct or ct < 1 or ds <= 0:
        return None
    return ds / ct


def decode_tok_s(rec) -> float | None:
    """r1kit-comparable decode tok/s: (completion_tokens - 1) / decode_s."""
    ds, ct = rec.get("decode_s"), rec.get("completion_tokens")
    if not ds or not ct or ct < 2 or ds <= 0:
        return None
    return (ct - 1) / ds


def hist_delta(cur, prev) -> list[int] | None:
    """Per-request accepted-draft histogram delta (index k = rounds accepting k drafts)."""
    if cur is None:
        return None
    if prev is None:
        return list(cur)
    n = max(len(cur), len(prev))
    return [(cur[i] if i < len(cur) else 0) - (prev[i] if i < len(prev) else 0) for i in range(n)]


def mean_accepted_from_hist(h) -> float | None:
    """Mean accepted drafts per verify step from an accepted-length histogram."""
    if not h:
        return None
    tot = sum(h)
    if tot <= 0:
        return None
    return sum(k * v for k, v in enumerate(h)) / tot


# ---------------------------------------------------------------------------
# Identity signal (logprobs path)
# ---------------------------------------------------------------------------
def first_divergence(seq_a, seq_b) -> int | None:
    """Index of the first differing element of two sequences, else None.

    Equal up to the common prefix but different lengths => the common length.
    """
    n = min(len(seq_a), len(seq_b))
    for i in range(n):
        if seq_a[i] != seq_b[i]:
            return i
    if len(seq_a) != len(seq_b):
        return n
    return None


def margin_at(top2, i: int) -> float | None:
    """top1 - top2 logprob at position ``i`` from a captured ``[(lp1, lp2), ...]``.

    ``lp2 is None`` (only one logprob captured) => +inf (a maximally confident
    position: no near-tie observed).
    """
    if i is None or i >= len(top2):
        return None
    pair = top2[i]
    if pair is None:
        return None
    lp1, lp2 = pair[0], pair[1] if len(pair) > 1 else None
    if lp1 is None:
        return None
    if lp2 is None:
        return float("inf")
    return lp1 - lp2


def _rep_tokens(rec) -> list:
    return list(rec.get("tokens") or [])


def _rep_top2(rec) -> list:
    return list(rec.get("top2_logprobs") or [])


def _identity_logprobs(arms, ref, ctrl, epsilon) -> dict:
    ref_reps = arms[ref].get("agentic") or []
    out = {"reference": ref, "control": ctrl, "signal": "logprobs",
           "control_rate": None, "allowed_rate": None, "arms": {}, "reasons": []}

    def compare(arm_reps):
        """Pooled first-divergence + margin, and a POSITION-level divergence rate."""
        div_pos = 0
        cmp_pos = 0
        firsts = []
        for r, ref_rec in enumerate(ref_reps):
            if r >= len(arm_reps):
                break
            a = _rep_tokens(ref_rec)
            b = _rep_tokens(arm_reps[r])
            top2 = _rep_top2(ref_rec)
            cmp_pos += max(len(a), len(b)) or 0
            n = min(len(a), len(b))
            div_pos += sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
            idx = first_divergence(a, b)
            if idx is not None:
                firsts.append({
                    "rep": r, "index": idx,
                    "tok_ref": a[idx] if idx < len(a) else None,
                    "tok_arm": b[idx] if idx < len(b) else None,
                    "margin": margin_at(top2, idx),
                })
        rate = (div_pos / cmp_pos) if cmp_pos else 0.0
        return firsts, rate

    ctrl_firsts, ctrl_rate = compare(arms[ctrl].get("agentic") or [])
    out["control_rate"] = ctrl_rate
    allowed = max(2.0 * ctrl_rate, RATE_FLOOR)
    out["allowed_rate"] = allowed

    # The determinism control itself must not diverge at a confident position.
    ctrl_confident = [f for f in ctrl_firsts if f["margin"] is not None and f["margin"] >= epsilon]
    if ctrl_confident:
        out["reasons"].append(
            f"determinism control {ctrl} vs {ref} diverged at a CONFIDENT position "
            f"(rep {ctrl_confident[0]['rep']} idx {ctrl_confident[0]['index']} "
            f"margin {ctrl_confident[0]['margin']:.4f} >= {epsilon}) -> cannot trust the control")

    for arm, rec in arms.items():
        if arm == ref:
            continue
        firsts, rate = compare(rec.get("agentic") or [])
        confident = [f for f in firsts if f["margin"] is not None and f["margin"] >= epsilon]
        identical = not firsts
        reasons = []
        if confident:
            verdict = "ABORT"
            reasons.append(
                f"confident-position divergence: rep {confident[0]['rep']} "
                f"idx {confident[0]['index']} margin {confident[0]['margin']:.4f} >= {epsilon}")
        elif identical:
            verdict = "PASS"
        elif rate <= allowed:
            verdict = "PASS"
        else:
            verdict = "ABORT"
            reasons.append(f"divergence rate {rate:.4f} > allowance {allowed:.4f} "
                           f"(2x control {ctrl_rate:.4f} floored at {RATE_FLOOR})")
        out["arms"][arm] = {"first_divergence": firsts, "rate": rate, "allowed_rate": allowed,
                            "identical": identical, "confident": bool(confident),
                            "verdict": verdict, "reasons": reasons}

    aborts = [a for a, r in out["arms"].items() if r["verdict"] == "ABORT"]
    if ctrl_confident:
        aborts.append(ctrl)
    out["verdict"] = "ABORT" if aborts else "PASS"
    if aborts:
        out["reasons"].append("ABORT arms: " + ", ".join(sorted(set(aborts))))
    return out


def _identity_histogram(arms, ref, ctrl, epsilon) -> dict:
    """Fallback identity signal when the endpoint cannot return logprobs.

    Uses per-rep mean-accepted-from-histogram and the histogram L1 distance
    instead of token+margin.  Weaker by construction (no positional index), so
    it is flagged loudly.
    """
    out = {"reference": ref, "control": ctrl, "signal": "histogram_fallback",
           "control_spread": None, "allowed": None, "arms": {}, "reasons": [
               "endpoint returned no logprobs; identity judged on the acceptance-"
               "histogram signal (distributional, NOT positional) — weaker"]}
    ref_reps = arms[ref].get("agentic") or []
    ctrl_reps = arms[ctrl].get("agentic") or []

    def stat(rec):
        return (rec.get("mean_accepted_hist"), rec.get("hist_delta"))

    def l1(a, b):
        if a is None or b is None:
            return None
        n = max(len(a), len(b))
        return sum(abs((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) for i in range(n))

    ctrl_spread = 0.0
    for r, ref_rec in enumerate(ref_reps):
        if r >= len(ctrl_reps):
            break
        mr, _ = stat(ref_rec)
        mc, _ = stat(ctrl_reps[r])
        if mr is not None and mc is not None:
            ctrl_spread = max(ctrl_spread, abs(mr - mc))
    allowed = max(2.0 * ctrl_spread, epsilon)
    out["control_spread"] = ctrl_spread
    out["allowed"] = allowed

    for arm, rec in arms.items():
        if arm == ref:
            continue
        divs = []
        for r, ref_rec in enumerate(ref_reps):
            reps = rec.get("agentic") or []
            if r >= len(reps):
                break
            mr, hr = stat(ref_rec)
            ma, ha = stat(reps[r])
            dm = abs(mr - ma) if (mr is not None and ma is not None) else None
            dl = l1(hr, ha)
            if (dm is not None and dm > allowed) or (dl is not None and dl > 0):
                divs.append({"rep": r, "d_mean_accepted": dm, "hist_l1": dl})
        verdict = "PASS" if not divs else "ABORT"
        out["arms"][arm] = {"divergences": divs, "allowed": allowed,
                            "verdict": verdict,
                            "reasons": [] if verdict == "PASS" else
                            [f"histogram-signal divergence in {len(divs)} rep(s)"]}
    aborts = [a for a, r in out["arms"].items() if r["verdict"] == "ABORT"]
    out["verdict"] = "ABORT" if aborts else "PASS"
    if aborts:
        out["reasons"].append("ABORT arms: " + ", ".join(sorted(aborts)))
    return out


def identity_gate(arms, ref="gamma3a", ctrl="gamma3b", epsilon=EPSILON) -> dict:
    """Pre-registered determinism control over the arm matrix.

    Per arm: first-divergence index vs ``ref`` (per rep index) and the top-1/top-2
    logit margin at that position.  PASS if identical OR every divergence is at
    margin < epsilon AND the divergence rate <= ~2x the ref-vs-ctrl rate.  ABORT
    on any confident-position (margin >= epsilon) divergence.
    """
    if ref not in arms or ctrl not in arms:
        return {"verdict": "ABORT", "signal": None, "reasons": [f"missing {ref} or {ctrl}"],
                "arms": {}}
    lp_ok = bool(arms[ref].get("logprobs_available")) and any(
        (arms[a].get("logprobs_available") is not False) for a in arms)
    if lp_ok:
        return _identity_logprobs(arms, ref, ctrl, epsilon)
    return _identity_histogram(arms, ref, ctrl, epsilon)


# ---------------------------------------------------------------------------
# Drift control
# ---------------------------------------------------------------------------
def drift_control(a_recs, b_recs, threshold=DRIFT_PCT) -> dict:
    """gamma3a vs gamma3b: relative decode tok/s median drift > ``threshold`` => RE-RUN."""
    a = [decode_tok_s(r) for r in (a_recs or [])]
    b = [decode_tok_s(r) for r in (b_recs or [])]
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    ma, mb = median(a), median(b)
    rel = abs(ma - mb) / ma if (ma and mb is not None) else None
    flagged = rel is not None and rel > threshold
    return {"median_a": ma, "median_b": mb, "rel_diff": rel, "threshold": threshold,
            "flagged": flagged,
            "verdict": "RE-RUN (drift-contaminated)" if flagged else "ok"}


# ---------------------------------------------------------------------------
# Memory gate
# ---------------------------------------------------------------------------
def memory_gate(arms, W_bytes: int = W_BYTES, margin_gb: float = MEM_MARGIN_GB) -> dict:
    """Flag any arm whose corrected peak is within ``margin_gb`` of W as ineligible.

    The "corrected peak" is the maximum over both nodes of every reading we hold:
    the CORRECTED allocator gauge (VM exo_peak_memory_bytes / 1.073741824) and
    the resident footprint.  Either reading crossing W - margin disqualifies the
    arm, because the wired limit is a hard per-node ceiling.
    """
    margin = int(margin_gb * GB)
    out = {}
    for arm, rec in arms.items():
        reads = []
        for node, b in (rec.get("peak_alloc_bytes") or {}).items():
            reads.append({"node": node, "kind": "alloc_corrected", "bytes": b})
        for node, b in (rec.get("peak_resident_bytes") or {}).items():
            reads.append({"node": node, "kind": "resident", "bytes": b})
        worst = max(reads, key=lambda t: t["bytes"]) if reads else None
        # No reading => not flagged (the gate can only disqualify on evidence;
        # the driver always collects both readings, so a missing one is reported).
        eligible = (worst is None) or (worst["bytes"] < (W_bytes - margin))
        out[arm] = {
            "eligible": eligible,
            "no_reading": worst is None,
            "worst": worst,
            "reads": reads,
            "headroom_gb": (round((W_bytes - worst["bytes"]) / GB, 3) if worst else None),
        }
    return out


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
def bars(arms, candidates=None, mem=None, iqr_enabled=True) -> dict:
    """Pre-registered performance bars vs the gamma3a+gamma3b control pools.

    * agentic win  = candidate agentic median >= (1 + 3%) * control agentic median
                     OR disjoint IQRs (only when ``iqr_enabled``).
    * benign ok    = candidate benign median >= (1 - 1.5%) * control benign median
                     AND candidate benign IQR is not wholly below the control's.
    * winner       = agentic win AND benign ok AND memory-eligible.
    * split        = agentic win AND benign lost beyond noise (or memory-ineligible)
                     => "gamma3 confirmed; per-workload gamma is a future item" (no ship).
    * otherwise    => "gamma3 confirmed on the post-dense build".
    """
    ctrl_arms = [a for a in ("gamma3a", "gamma3b") if a in arms]
    ctrl_ag = [decode_tok_s(r) for a in ctrl_arms for r in (arms[a].get("agentic") or [])]
    ctrl_be = [decode_tok_s(r) for a in ctrl_arms for r in (arms[a].get("benign") or [])]
    ctrl_ag = [x for x in ctrl_ag if x is not None]
    ctrl_be = [x for x in ctrl_be if x is not None]
    ctrl_ag_med, ctrl_be_med = median(ctrl_ag), median(ctrl_be)

    if candidates is None:
        candidates = [a for a in arms if a not in ("gamma3a", "gamma3b")]
    mem = mem or {}

    results = {}
    split = None
    winner = None
    for arm in candidates:
        ag = [x for x in (decode_tok_s(r) for r in (arms[arm].get("agentic") or [])) if x is not None]
        be = [x for x in (decode_tok_s(r) for r in (arms[arm].get("benign") or [])) if x is not None]
        ag_med, be_med = median(ag), median(be)
        ag_vs_ctrl = (ag_med / ctrl_ag_med - 1.0) if (ag_med and ctrl_ag_med) else None
        be_vs_ctrl = (be_med / ctrl_be_med - 1.0) if (be_med and ctrl_be_med) else None

        ag_win_median = ag_vs_ctrl is not None and ag_vs_ctrl >= AGENTIC_WIN_PCT
        ag_win_iqr = iqr_enabled and iqr_disjoint(ag, ctrl_ag)
        ag_win = ag_win_median or ag_win_iqr
        be_ok_median = be_vs_ctrl is not None and be_vs_ctrl >= -BENIGN_TOL_PCT
        be_iqr_ok = (not iqr_enabled) or (not iqr_wholly_below(be, ctrl_be))
        ben_ok = be_ok_median and be_iqr_ok
        mem_ok = mem.get(arm, {}).get("eligible", True)

        if ag_win and ben_ok and mem_ok:
            klass = "WINNER"
            if winner is None:
                winner = arm
        elif ag_win:
            klass = "SPLIT"
            if split is None:
                split = arm
        else:
            klass = "no-win"
        results[arm] = {
            "classification": klass,
            "agentic_median_tok_s": ag_med, "benign_median_tok_s": be_med,
            "agentic_vs_control_pct": (round(ag_vs_ctrl * 100, 3) if ag_vs_ctrl is not None else None),
            "benign_vs_control_pct": (round(be_vs_ctrl * 100, 3) if be_vs_ctrl is not None else None),
            "agentic_win_median": ag_win_median, "agentic_win_iqr": ag_win_iqr,
            "benign_ok_median": be_ok_median, "benign_iqr_ok": be_iqr_ok,
            "memory_eligible": mem_ok,
        }

    if winner is not None:
        verdict = (f"{winner} WINS — candidate ships (agentic "
                   f"{results[winner]['agentic_vs_control_pct']:+.2f}% vs control median; "
                   f"benign {results[winner]['benign_vs_control_pct']:+.2f}%)")
    elif split is not None:
        verdict = "gamma3 confirmed; per-workload gamma is a future item"
    else:
        verdict = "gamma3 confirmed on the post-dense build"

    return {"control_agentic_median_tok_s": ctrl_ag_med,
            "control_benign_median_tok_s": ctrl_be_med,
            "iqr_enabled": iqr_enabled, "results": results,
            "winner": winner, "split": split, "verdict": verdict}

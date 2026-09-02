#!/usr/bin/env python3
"""
ROUND 3 — SDPA two-length per-call analyzer (PREP-only deliverable).

Parses the `[SDPA-CALL]` lines emitted by the EXO_DSV4_SDPA_CALL_PROFILE gate
(see artifacts/sdpa_2length_timing.patch), separates warmup from steady-state,
computes per-call ABSOLUTE milliseconds and the per-call ratio, runs the
mandatory reductio check, and applies the PM's pre-registered decision rule.

LINE FORMAT (emitted to runner stderr, captured in ~/exo.log):
    [SDPA-CALL] local L=<layer_idx> ms=<float>
    [SDPA-CALL] compressed L=<layer_idx> ms=<float>
    [SDPA-CALL] sparse L=<layer_idx> ms=<float>

UNITS: everything here is PER-CALL (absolute ms). ms/token appears ONLY as an
explicitly-labeled derived secondary (the "ms/token (derived)" column), never
as the primary metric. The round-2 closure was invalidated by conflating these
two units; we do not repeat that.

RATIO DEFINITION (fixed by design — no sweep):
    A single 2048-token prefill chunk fires, inside ONE `attn.sdpa` span
    (deepseek_v4.py:4865):
        * 21 SparseCompressedAttention -> banded per-rank rows L_q = 1024
        *  2 LocalAttention            -> full rows            L_q = 2048
    (both at identical depth, identical KV/pool per rank).
    R(L) = mean(local_ms) / mean(sparse_ms)  at context length L.
    This is the same 1024->2048 per-rank row doubling the round-2 4.06x came
    from, measured WITHOUT varying EXO_PREFILL_STEP_SIZE.

REDUCTIO (mandatory, fails loudly): total measured SDPA time
    sum over tags of (call_count * mean_ms)
must be <= the arm's wall clock (--prefill-wall-sec), else the per-call
numbers are mathematically impossible and the whole measurement is discarded.

USAGE:
    python3 sdpa_2length_analyze.py \\
        --log /tmp/sdpa_2len.log \\
        --prefill-wall-sec 30 \\
        [--prefill-wall-sec-64k 65]
    (per-arm wall clock in SECONDS; use the two-arg form when the two arms ran
    for different wall times; if only --prefill-wall-sec is given it applies to
    both arms.)
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

LINE_RE = re.compile(
    r"\[SDPA-CALL\]\s+(?P<tag>\S+)\s+L=(?P<layer>\d+)\s+ms=(?P<ms>\d+\.?\d*)"
)

# Pre-registered decision rule (fixed by the PM — do not alter).
R_LO, R_HI = 3.0, 5.0          # band that a "real multiplicative constant" must sit in
RATIO_CONSISTENCY = 0.25       # |R(64K)-R(12K)| / R(12K) <= 25%
OVERHEAD_HARD = 2.2            # R(64K) <= 2.2  => fixed-overhead artifact
OVERHEAD_FRACTION = 0.6        # R(64K) < 0.6*R(12K) => fixed-overhead artifact


def parse_log(path: str) -> "list[tuple[str, float]]":
    """Return ordered [(tag, ms)] across the whole two-arm run (global order)."""
    records: "list[tuple[str, float]]" = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LINE_RE.search(line)
            if not m:
                continue
            records.append((m.group("tag"), float(m.group("ms"))))
    if not records:
        sys.exit(
            "FATAL: no [SDPA-CALL] lines parsed. The probe did not run (did the "
            "env gate reach the runner? was the patch applied&reinstalled?)."
        )
    return records


def split_arms(records: "list[tuple[str, float]]", split_idx: int) -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]]
]:
    """Split the GLOBAL ordered call stream at line ``split_idx`` (12K arm end).

    The two arms are emitted sequentially into one log (12K first, then 64K).
    ``split_idx`` is the count of [SDPA-CALL] lines belonging to the 12K arm;
    every record before it is arm-12K, every record at/after it is arm-64K.
    Each arm is then regrouped by tag. Splitting on the global stream (not
    per-tag) is essential — sparse and local calls interleave within every
    chunk, so a per-tag split index would mix the two arms' data.
    """
    arm_a: Dict[str, List[float]] = defaultdict(list)
    arm_b: Dict[str, List[float]] = defaultdict(list)
    for i, (tag, ms) in enumerate(records):
        (arm_a if i < split_idx else arm_b)[tag].append(ms)
    return dict(arm_a), dict(arm_b)


def warmup_split(calls: Dict[str, List[float]]) -> Tuple[
    Dict[str, List[float]],        # warmup: {tag: [first-call ms]}
    Dict[str, List[float]],        # steady-state: {tag: [rest ms]}
]:
    """Separate the FIRST call per tag (warmup) from steady-state.

    The first SDPA at each (length, tag) pays the Metal kernel-compile /
    allocator warmup cost and would contaminate n>=5. We drop it and report it
    separately so nothing is silently misattributed.
    """
    warm: Dict[str, List[float]] = {}
    steady: Dict[str, List[float]] = {}
    for tag, ms_list in calls.items():
        if not ms_list:
            warm[tag] = []
            steady[tag] = []
            continue
        warm[tag] = [ms_list[0]]
        steady[tag] = ms_list[1:]
    return warm, steady


def per_call_stats(tag_ms: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tag, ms in tag_ms.items():
        if not ms:
            continue
        out[tag] = {
            "n": len(ms),
            "mean_ms": statistics.mean(ms),
            "median_ms": statistics.median(ms),
            "min_ms": min(ms),
            "max_ms": max(ms),
        }
    return out


def reductio_check(
    steady: Dict[str, List[float]],
    wall_sec: float,
    arm_label: str,
) -> bool:
    """calls x per-call MUST be <= measured wall clock. Fail loudly otherwise.

    Uses per-tag sums of call_count * mean_ms (an upper bound on SDPA's share of
    wall, since SDPA can't exceed the whole prefill). If it exceeds wall even at
    this loose bound, the per-call numbers are impossible -> measurement invalid.
    """
    total_ms = 0.0
    n_calls = 0
    for tag, ms in steady.items():
        if not ms:
            continue
        n_calls += len(ms)
        total_ms += len(ms) * statistics.mean(ms)
    wall_ms = wall_sec * 1000.0
    print(f"\n[REDUCTIO {arm_label}] calls={n_calls} "
          f"sum(n*mean)={total_ms:.1f} ms | wall={wall_ms:.1f} ms")
    ok = total_ms <= wall_ms * 1.02  # 2% slack for clock rounding
    if not ok:
        print(
            f"FATAL [REDUCTIO {arm_label}]: measured SDPA total ({total_ms:.1f} ms) "
            f"exceeds arm wall clock ({wall_ms:.1f} ms). Per-call numbers are "
            f"mathematically impossible. Discarding this arm."
        )
    return ok


def ratio_and_decision(r12: "float | None", r64: "float | None") -> str:
    """Apply the PM's pre-registered decision rule verbatim."""
    # Indeterminate guard: if either ratio is not computable (missing tag), bail.
    if r12 is None or r64 is None:
        return "INDETERMINATE (one arm lacked a computable ratio)"

    r_both_in_band = (R_LO <= r12 <= R_HI) and (R_LO <= r64 <= R_HI)
    consistency = abs(r64 - r12) / r12 if r12 != 0 else float("inf")
    consistent = consistency <= RATIO_CONSISTENCY

    if r_both_in_band and consistent:
        return (
            f"REAL MULTIPLICATIVE CONSTANT (R12={r12:.2f}, R64={r64:.2f}) — both in "
            f"[{R_LO},{R_HI}] and |ΔR|/R12={consistency:.3f}<=25%. SDPA per-call "
            f"superlinearity is a real constant: it matters at 250K."
        )
    if r64 <= OVERHEAD_HARD or r64 < OVERHEAD_FRACTION * r12:
        return (
            f"FIXED-OVERHEAD ARTIFACT (R64={r64:.2f}<=2.2 or "
            f"<0.6*R12={0.6*r12:.2f}) — ratio collapses at depth. The ~4.06x is "
            f"a fixed per-call overhead that amortizes away; it does NOT matter "
            f"at 250K."
        )
    return (
        f"INDETERMINATE (R12={r12:.2f}, R64={r64:.2f}, ΔR/R12={consistency:.3f}) — "
        f"not cleanly in either band. Re-run or inspect raw per-call distributions."
    )


def fmt_ms_per_token(mean_ms: float, rows: int) -> str:
    """Explicitly-labeled DERIVED secondary: ms/token. Never the primary metric."""
    return f"{mean_ms / rows * 1000.0:.3f} µs/token (derived)" if rows else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to concatenated [SDPA-CALL] log")
    ap.add_argument("--prefill-wall-sec", type=float, required=True,
                    help="wall clock (seconds) of the 12K arm")
    ap.add_argument("--prefill-wall-sec-64k", type=float, default=None,
                    help="wall clock (seconds) of the 64K arm; default = "
                         "--prefill-wall-sec")
    ap.add_argument("--sparse-rows", type=int, default=1024,
                    help="per-rank sparse query rows (1024 at STEP_SIZE=2048 "
                         "seq-split; only used for the derived ms/token column)")
    ap.add_argument("--local-rows", type=int, default=2048,
                    help="LocalAttention query rows (2048 at STEP_SIZE=2048; "
                         "derived-only)")
    ap.add_argument("--split-calls", type=int, default=None,
                    help="number of [SDPA-CALL] lines belonging to the 12K arm "
                         "(first N lines). If omitted, the analyzer reports the "
                         "single-stream ratio and skips the cross-length verdict.")
    args = ap.parse_args()

    by_tag = parse_log(args.log)

    # --- split the two arms ------------------------------------------------
    # Both arms emit MANY calls (each chunk fires 21 sparse + 2 local spans).
    # The runbook runs 12K first, 64K second in one log. --split-calls is the
    # number of [SDPA-CALL] lines that belong to the 12K arm (everything after
    # it is the 64K arm). If omitted, treat all calls as one stream and warn.
    wall_64 = args.prefill_wall_sec_64k or args.prefill_wall_sec

    if args.split_calls:
        arm12, arm64 = split_arms(by_tag, args.split_calls)
        arms: "list[tuple[str, Dict[str, List[float]]]]" = [("12K", arm12), ("64K", arm64)]
    else:
        # No split: treat everything as one contiguous stream. Report the ratio
        # but mark the cross-length decision as not runnable.
        arm12, arm64 = {}, {"all": [ms for _, ms in by_tag]}
        print("[WARN] --split-calls not provided; cannot separate 12K from 64K. "
              "Reporting single-stream ratio only.")
        arms = [("12K", arm12), ("64K", arm64)]

    # --- bucketing -----------------------------------------------------------
    print("=== STEADY-STATE PER-CALL ABSOLUTE TIMES (ms) ===")
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    walls = {"12K": args.prefill_wall_sec, "64K": wall_64}
    for label, arm in arms:
        if not arm:
            print(f"[{label}] no calls parsed.")
            continue
        _, steady = warmup_split(arm)
        stats = per_call_stats(steady)
        if not reductio_check(steady, walls[label], label):
            continue  # measurement invalid for this arm; skip decision
        results[label] = stats
        rows = {"sparse": args.sparse_rows, "local": args.local_rows}
        for tag in ("sparse", "local", "compressed"):
            if tag in stats:
                s = stats[tag]
                print(
                    f"[{label}] {tag:<10} n={s['n']:<4} mean={s['mean_ms']:8.3f} ms "
                    f"med={s['median_ms']:8.3f} min={s['min_ms']:7.3f} "
                    f"max={s['max_ms']:8.3f}  | "
                    f"{fmt_ms_per_token(s['mean_ms'], rows.get(tag, args.sparse_rows))}"
                )

    # --- warmup report (transparency, not used in the ratio) -------------------
    print("\n=== WARMUP (first call per tag — EXCLUDED from steady-state) ===")
    for label, arm in arms:
        if not arm:
            continue
        warm, _ = warmup_split(arm)
        for tag, ms in warm.items():
            for m in ms:
                print(f"[{label}] {tag} first-call ms={m:.3f} (warmup)")

    # --- ratio ---------------------------------------------------------------
    def _ratio(arm_stats: Dict[str, Dict[str, float]]):
        if "local" in arm_stats and "sparse" in arm_stats:
            l = arm_stats["local"]["mean_ms"]
            s = arm_stats["sparse"]["mean_ms"]
            if s > 0:
                return l / s
        return None

    print("\n=== PER-CALL RATIO R(L) = mean(local) / mean(sparse) ===")
    r12 = _ratio(results.get("12K", {}))
    r64 = _ratio(results.get("64K", {}))
    for lbl, r in (("12K", r12), ("64K", r64)):
        print(f"R({lbl}) = {r:.3f}" if r is not None else f"R({lbl}) = n/a")

    print("\n=== VERDICT (PM pre-registered rule) ===")
    if not args.split_calls:
        print("--split-calls not set: cross-length verdict NOT computable. "
              "Provide it for the final decision.")
    else:
        r12_v = r12 if r12 is not None else float("nan")
        r64_v = r64 if r64 is not None else float("nan")
        print(ratio_and_decision(r12_v, r64_v))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

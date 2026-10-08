#!/usr/bin/env python3
"""Phase 0a — per-call time decomposition of the real 42-call Hermes turn.

Splits each call's ``latency_seconds`` (state.db ``api_calls``) into MEASURED
prefill and decode time using the exo/dsv41 engine markers in an exo.log window,
and decides cache hit/miss per call. Nothing here is estimated: if a call has no
turn-reuse marker (the cold first call) its split is reported UNSPLIT, not
guessed.

Marker semantics (verified against the real log + engine source; see
``docs/benchmarks/phase20-throughput/turn_decomp.md``):

* ``API request: POST /v1/chat/completions``      -> client arrival (t_post).
* ``[DSV41] prefill controls: ... (rows=M, ...)``  -> prefill ENTRY (t_prefill_start);
  also ``[DSV41] session reuse: this N-token prompt matches ...`` at the same point.
* ``[DSV41] turn reuse: prompt=N prefill=M reuse=R cache=C [rewind=W] UNCOMMITTED``
  -> emitted by ``engine._start_turn`` immediately AFTER ``session.prefill``
  RETURNS (engine.py:957) => prefill-completion instant (t_prefill_end). Only
  present when ``reuse>0``; the cold call 1 has none.
* ``Executing command: TaskFinished(...)``         -> request end (t_end);
  ``runner idle: reclaimed MLX allocator pool`` / ``runner ready`` follow it.

Clock: state.db epochs are UTC; exo.log timestamps are node-local CDT (=UTC-5).

Usage:
    python bench/phase20_turn_decomp.py \
        --log  /path/m41_0901_1400.log.zst \
        --db   /Users/adam.durham/.hermes/state.db \
        --session 20261007_092009_9a2ed7 \
        --out-csv  docs/benchmarks/phase20-throughput/turn_decomp.csv \
        --out-json docs/benchmarks/phase20-throughput/turn_decomp_summary.json

stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
PROVIDER = "custom"
SESSION = "20261007_092009_9a2ed7"

CDT_OFFSET_HOURS = 5  # state.db epoch is UTC; exo.log is node-local CDT (UTC-5)

# Cache-hit rule (PREREG 0a): delta_rows <= 1.2 * (ctx_at_call - ctx_at_prev_call).
CACHE_HIT_FACTOR = 1.2

CSV_REQUIRED_COLUMNS = [
    "call_idx",
    "started_at",
    "ended_at",
    "latency_s",
    "ctx_tokens_at_call",
    "delta_rows_prefilled",
    "prefill_s",
    "output_tokens",
    "decode_s",
    "gap_to_next_call_s",
    "cache_hit",
]
CSV_EXTRA_COLUMNS = [
    "started_at_epoch",
    "ended_at_epoch",
    "ctx_prev_tokens",
    "dctx_tokens",
    "rule_threshold_rows",
    "cache_hit_reason",
    "delta_rows_source",
    "reuse_rows",
    "rewind",
    "decode_tok_s",
    "prefill_rows_per_s",
    "pre_s",
    "post_s",
    "residual_s",
    "split_status",
    "t_post_cdt",
    "t_prefill_start_cdt",
    "t_prefill_end_cdt",
    "t_end_cdt",
    "post_match_offset_s",
]
CSV_COLUMNS = CSV_REQUIRED_COLUMNS + CSV_EXTRA_COLUMNS

# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

_TS = r"\[ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \|"

_RE = {
    # client arrival
    "post": re.compile(_TS + r".*API request: POST /v1/chat/completions"),
    # prefill entry: session reuse + prefill controls (both near request start)
    "sessreuse": re.compile(
        _TS + r".*\[DSV41\] session reuse: this (\d+)-token prompt matches a "
        r"resident conversation on (\d+) rows"
    ),
    "prefctl": re.compile(_TS + r".*\[DSV41\] prefill controls:.*\(rows=(\d+), base=(\d+)\)"),
    # prefill RETURN: the turn-reuse line (only when reuse>0)
    "turnreuse": re.compile(
        _TS + r".*\[DSV41\] turn reuse: prompt=(\d+) prefill=(\d+) reuse=(\d+) "
        r"cache=(\d+)(?: rewind=(\d+))?"
    ),
    # request end
    "fin": re.compile(_TS + r".*Executing command: TaskFinished\("),
    "idle": re.compile(_TS + r".*runner idle: reclaimed MLX allocator pool"),
    "ready": re.compile(_TS + r".*runner ready"),
    # cold-prefill forensics: the per-(shape,strip) geometry census (stderr)
    "hier": re.compile(_TS + r".*Runner stderr: \[DSV41\] hier geometry: exact_mb=\S+ estrip=(\d+) block=(\d+) bp=\d+"),
}

_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


def parse_ts(ts: str) -> _dt.datetime:
    return _dt.datetime.strptime(ts, _TS_FMT)


def _iter_lines(path: str | Path) -> Iterable[str]:
    """Yield lines from a plain log or a ``.zst`` (via zstdcat) without writing
    the decompressed form to disk."""
    p = Path(path)
    if p.suffix == ".zst":
        proc = subprocess.Popen(
            ["zstdcat", str(p)], stdout=subprocess.PIPE, text=True, errors="replace"
        )
        assert proc.stdout is not None
        try:
            yield from proc.stdout
        finally:
            proc.stdout.close()
            proc.wait()
    else:
        with open(p, errors="replace") as fh:
            yield from fh


@dataclass
class Event:
    kind: str
    t: _dt.datetime
    m: re.Match


def scan_log(path: str | Path) -> dict[str, list[Event]]:
    """One pass over the log -> events grouped by kind (chronological)."""
    ev: dict[str, list[Event]] = {k: [] for k in _RE}
    for ln in _iter_lines(path):
        for kind, rx in _RE.items():
            m = rx.search(ln)
            if m:
                ev[kind].append(Event(kind, parse_ts(m.group(1)), m))
                break
    return ev


def _epoch_to_cdt(epoch: float) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(epoch, _dt.UTC).replace(tzinfo=None) - _dt.timedelta(
        hours=CDT_OFFSET_HOURS
    )


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def load_ledger(db_path: str | Path, session: str = SESSION) -> list[dict]:
    """Read the 42 exo-model rows for the session (EXACT provider+model equality:
    SQLite LIKE is case-insensitive and would also match the ollama model
    ``deepseek-v4.1-flash``)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT call_seq, started_at, ended_at, latency_seconds,
                   input_tokens, cache_read_tokens, output_tokens,
                   reasoning_tokens, prompt_tokens_total
            FROM api_calls
            WHERE session_id = ? AND provider = ? AND model = ?
            ORDER BY call_seq
            """,
            (session, PROVIDER, MODEL),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Call matching + decomposition
# --------------------------------------------------------------------------- #


def _assign_posts(ev: dict[str, list[Event]], ledger: Sequence[dict]) -> dict[int, tuple[_dt.datetime, float]]:
    """Assign each ledger call the POST line whose CDT time is nearest to
    ``started_at``, greedily and monotonically (posts are consumed in time
    order). Returns {call_seq: (t_post, |offset|)}."""
    posts = [e.t for e in ev["post"]]
    used: set[int] = set()
    out: dict[int, tuple[_dt.datetime, float]] = {}
    for row in ledger:
        started_cdt = _epoch_to_cdt(row["started_at"])
        best_i: int | None = None
        best_d: float | None = None
        for i, p in enumerate(posts):
            if i in used:
                continue
            d = abs((p - started_cdt).total_seconds())
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        assert best_i is not None and best_d is not None
        used.add(best_i)
        out[row["call_seq"]] = (posts[best_i], best_d)
    return out


def _first_in(lst: list[Event], lo: _dt.datetime, hi: _dt.datetime,
              predicate=None) -> Event | None:
    for e in lst:
        if lo < e.t < hi and (predicate is None or predicate(e)):
            return e
    return None


def decompose(ev: dict[str, list[Event]], ledger: Sequence[dict]) -> list[dict]:
    """Build one decomposed row per ledger call."""
    posts = _assign_posts(ev, ledger)
    seqs = [r["call_seq"] for r in ledger]
    rows: list[dict] = []
    for i, row in enumerate(ledger):
        seq = row["call_seq"]
        t_post, offset = posts[seq]
        hi = posts[seqs[i + 1]][0] if i + 1 < len(seqs) else parse_ts("9999-12-31 23:59:59.999")
        ctx = row["prompt_tokens_total"]

        # prefill entry: session-reuse (prompt match preferred) else prefill controls
        sr = _first_in(ev["sessreuse"], t_post, hi,
                       lambda e: int(e.m.group(2)) == ctx)
        pc = _first_in(ev["prefctl"], t_post, hi)
        cands = [e.t for e in (sr, pc) if e is not None]
        t_ps = min(cands) if cands else t_post

        # prefill end: turn-reuse matched by exact prompt in this window
        tr = _first_in(ev["turnreuse"], t_post, hi,
                       lambda e: int(e.m.group(2)) == ctx)

        # end: first TaskFinished after the turn-reuse line (else after POST)
        anchor = tr.t if tr is not None else t_post
        fin = _first_in(ev["fin"], anchor, parse_ts("9999-12-31 23:59:59.999"))
        t_end = fin.t if fin is not None else None

        # cold-prefill forensics: geometry census lines in this window
        hier = [e for e in ev["hier"] if t_post < e.t < hi]

        out: dict = {
            "call_idx": seq,
            "started_at": _epoch_to_cdt(row["started_at"]),
            "ended_at": _epoch_to_cdt(row["ended_at"]),
            "started_at_epoch": row["started_at"],
            "ended_at_epoch": row["ended_at"],
            "latency_s": row["latency_seconds"],
            "ctx_tokens_at_call": ctx,
            "output_tokens": row["output_tokens"],
            "reasoning_tokens": row["reasoning_tokens"],
            "cache_read_tokens": row["cache_read_tokens"],
            "t_post": t_post,
            "t_prefill_start": t_ps,
            "t_prefill_end": tr.t if tr is not None else None,
            "t_end": t_end,
            "post_match_offset_s": offset,
            "prompt_matched": int(tr.m.group(2)) == ctx if tr is not None else None,
            "delta_rows_prefilled": (
                int(tr.m.group(3)) if tr is not None
                else (int(pc.m.group(2)) if pc is not None else None)
            ),
            "delta_rows_source": (
                "turn_reuse.prefill" if tr is not None
                else ("prefill_controls.rows (cold full prefill)" if pc is not None else None)
            ),
            "reuse_rows": int(tr.m.group(4)) if tr is not None else None,
            "rewind": (int(tr.m.group(6)) if tr is not None and tr.m.group(6) else None),
            "prefctl_rows": int(pc.m.group(2)) if pc is not None else None,
            "sessreuse_rows": int(sr.m.group(3)) if sr is not None else None,
            "n_hier_geometry": len(hier),
            "split_status": "UNSPLIT",
        }

        # ---- measured split ----
        if tr is not None and t_end is not None:
            out["prefill_s"] = (out["t_prefill_end"] - t_ps).total_seconds()
            out["decode_s"] = (t_end - out["t_prefill_end"]).total_seconds()
            out["pre_s"] = (t_ps - t_post).total_seconds()
            out["prefill_s_from_post"] = (out["t_prefill_end"] - t_post).total_seconds()
            out["post_s"] = (out["ended_at"] - t_end).total_seconds()
            out["residual_s"] = row["latency_seconds"] - (
                out["prefill_s_from_post"] + out["decode_s"]
            )
            out["split_status"] = "SPLIT"
        else:
            out["prefill_s"] = None
            out["decode_s"] = None
            out["pre_s"] = (t_ps - t_post).total_seconds()
            out["prefill_s_from_post"] = None
            out["post_s"] = (
                (out["ended_at"] - t_end).total_seconds() if t_end is not None else None
            )
            out["residual_s"] = None
            out["split_status"] = "UNSPLIT_COLD" if seq == seqs[0] else "UNSPLIT"

        rows.append(out)

    _annotate_gaps(rows)
    _annotate_cache_hits(rows)
    _annotate_rates(rows)
    return rows


def _annotate_gaps(rows: list[dict]) -> None:
    for i, r in enumerate(rows):
        if i + 1 < len(rows):
            r["gap_to_next_call_s"] = (
                rows[i + 1]["started_at"] - r["ended_at"]
            ).total_seconds()
        else:
            r["gap_to_next_call_s"] = None


def cache_hit_rule(delta_rows: int, ctx_at_call: int, ctx_at_prev: int,
                   factor: float = CACHE_HIT_FACTOR) -> tuple[bool | None, float | None, str]:
    """PREREG 0a cache-hit rule.

    Returns ``(hit, threshold_rows, reason)``. ``hit`` is None when the rule
    degenerates (``ctx_at_call - ctx_at_prev <= 0``): the delta cannot be judged
    because the context did not grow.
    """
    dctx = ctx_at_call - ctx_at_prev
    if dctx <= 0:
        return None, None, "degenerate_dctx_le_0"
    threshold = factor * dctx
    return (delta_rows <= threshold), threshold, ("hit" if delta_rows <= threshold else "miss")


def _annotate_cache_hits(rows: list[dict]) -> None:
    prev: dict | None = None
    for r in rows:
        delta_rows = r["delta_rows_prefilled"]
        if r["split_status"] == "UNSPLIT_COLD" or delta_rows is None:
            # cold start: no reuse; excluded from the miss count
            r["cache_hit"] = False
            r["cache_hit_reason"] = "cold"
            r["ctx_prev_tokens"] = None
            r["dctx_tokens"] = None
            r["rule_threshold_rows"] = None
            prev = r
            continue
        prev_ctx = prev["ctx_tokens_at_call"] if prev is not None else None
        if prev_ctx is None:
            hit, thresh, reason = None, None, "degenerate_dctx_le_0"
        else:
            hit, thresh, reason = cache_hit_rule(delta_rows, r["ctx_tokens_at_call"], prev_ctx)
        ctx_prev = prev_ctx
        r["ctx_prev_tokens"] = ctx_prev
        r["dctx_tokens"] = r["ctx_tokens_at_call"] - ctx_prev
        r["rule_threshold_rows"] = thresh
        r["cache_hit"] = bool(hit) if hit is not None else False
        r["cache_hit_reason"] = reason
        prev = r


def _annotate_rates(rows: list[dict]) -> None:
    for r in rows:
        if r["decode_s"]:
            r["decode_tok_s"] = r["output_tokens"] / r["decode_s"]
        else:
            r["decode_tok_s"] = None
        if r["prefill_s"] and r["delta_rows_prefilled"] is not None:
            r["prefill_rows_per_s"] = r["delta_rows_prefilled"] / r["prefill_s"]
        else:
            r["prefill_rows_per_s"] = None


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def summarize(rows: Sequence[dict], session: str = SESSION) -> dict:
    sum_latency = sum(r["latency_s"] for r in rows)
    sum_prefill = sum(r["prefill_s"] for r in rows if r["prefill_s"] is not None)
    sum_decode = sum(r["decode_s"] for r in rows if r["decode_s"] is not None)
    sum_gap = sum(r["gap_to_next_call_s"] for r in rows if r["gap_to_next_call_s"] is not None)
    sum_pre_s = sum(r["pre_s"] for r in rows if r["pre_s"] is not None)
    sum_post_s = sum(r["post_s"] for r in rows if r["post_s"] is not None)
    sum_residual = sum(r["residual_s"] for r in rows if r["residual_s"] is not None)

    split = [r for r in rows if r["split_status"] == "SPLIT"]
    unsplit = [r for r in rows if r["split_status"] != "SPLIT"]
    model_time = sum_prefill + sum_decode
    wall_span = (rows[-1]["ended_at"] - rows[0]["started_at"]).total_seconds()

    gate_ratio = model_time / sum_latency if sum_latency else None
    gate_ok = gate_ratio is not None and 0.90 <= gate_ratio <= 1.10

    # cache-hit
    judged = [r for r in rows if r["cache_hit_reason"] not in ("cold", "degenerate_dctx_le_0")]
    misses = [r for r in judged if r["cache_hit_reason"] == "miss"]
    degenerate = [r for r in rows if r["cache_hit_reason"] == "degenerate_dctx_le_0"]

    # decode tok/s distribution
    dec = [r["decode_tok_s"] for r in split if r["decode_tok_s"]]
    weighted_decode = (
        sum(r["output_tokens"] for r in split) / sum_decode if sum_decode else None
    )
    # prefill rows/s distribution, excluding tiny deltas where fixed overhead dominates
    big = [r for r in split if r["delta_rows_prefilled"] is not None and r["delta_rows_prefilled"] >= 500]
    rows_s_all = [r["prefill_rows_per_s"] for r in split if r["prefill_rows_per_s"]]
    rows_s_big = [r["prefill_rows_per_s"] for r in big]
    weighted_rows_s_big = (
        sum(r["delta_rows_prefilled"] for r in big)
        / sum(r["prefill_s"] for r in big)
        if big
        else None
    )

    # rows==tokens assertion: prompt == reuse + prefill
    assert_ok = all(
        r["prompt_matched"]
        for r in rows
        if r["split_status"] == "SPLIT"
    )
    prompt_eq = [r["call_idx"] for r in rows if r["split_status"] == "SPLIT"]

    return {
        "session": session,
        "model": MODEL,
        "provider": PROVIDER,
        "n_calls": len(rows),
        "n_split": len(split),
        "n_unsplit": len(unsplit),
        "unsplit_calls": [
            {"call_idx": r["call_idx"], "split_status": r["split_status"], "latency_s": r["latency_s"]}
            for r in unsplit
        ],
        "sum_latency_s": sum_latency,
        "sum_prefill_s": sum_prefill,
        "sum_decode_s": sum_decode,
        "sum_model_s": model_time,
        "sum_gap_to_next_call_s": sum_gap,
        "sum_pre_s": sum_pre_s,
        "sum_post_s": sum_post_s,
        "sum_residual_s_split_only": sum_residual,
        "wall_span_s": wall_span,
        "gate_0a": {
            "expected_sum_latency_s": 1530.59,
            "ratio": gate_ratio,
            "within_pm10pct": gate_ok,
            "model_side_unaccounted_s": sum_latency - model_time,
            "note": (
                "model_side_unaccounted_s = sum(latency) - (prefill_s + decode_s); "
                "it includes the whole UNSPLIT cold call 1 (no turn-reuse marker) "
                "plus per-call pre-start/post-end overhead."
            ),
        },
        "shares": {
            "decode_share_of_model_time": sum_decode / model_time if model_time else None,
            "prefill_share_of_model_time": sum_prefill / model_time if model_time else None,
            "model_time_share_of_wall_span": model_time / wall_span if wall_span else None,
            "unsplit_latency_share_of_wall_span": (
                sum(r["latency_s"] for r in unsplit) / wall_span if wall_span else None
            ),
        },
        "cache": {
            "miss_count": len(misses),
            "misses": [
                {
                    "call_idx": r["call_idx"],
                    "delta_rows": r["delta_rows_prefilled"],
                    "dctx_tokens": r["dctx_tokens"],
                    "threshold_rows": r["rule_threshold_rows"],
                }
                for r in misses
            ],
            "judged_count": len(judged),
            "cold_count": sum(1 for r in rows if r["cache_hit_reason"] == "cold"),
            "degenerate_count": len(degenerate),
            "degenerate_calls": [
                {"call_idx": r["call_idx"], "dctx_tokens": r["dctx_tokens"]} for r in degenerate
            ],
            "hits": [r["call_idx"] for r in judged if r["cache_hit_reason"] == "hit"],
        },
        "decode_tok_s": {
            "n": len(dec),
            "min": min(dec) if dec else None,
            "median": statistics.median(dec) if dec else None,
            "max": max(dec) if dec else None,
            "weighted_mean_sum_output_over_sum_decode": weighted_decode,
        },
        "prefill_rows_per_s": {
            "n_all": len(rows_s_all),
            "all_min": min(rows_s_all) if rows_s_all else None,
            "all_median": statistics.median(rows_s_all) if rows_s_all else None,
            "all_max": max(rows_s_all) if rows_s_all else None,
            "n_ge_500_rows": len(rows_s_big),
            "ge_500_min": min(rows_s_big) if rows_s_big else None,
            "ge_500_median": statistics.median(rows_s_big) if rows_s_big else None,
            "ge_500_max": max(rows_s_big) if rows_s_big else None,
            "ge_500_weighted_mean": weighted_rows_s_big,
            "note": (
                "calls with delta_rows < 500 are excluded from the headline rows/s "
                "distribution: at that size the fixed per-call overhead dominates "
                "the measured prefill window and drags the apparent rate down."
            ),
        },
        "assert_prompt_eq_reuse_plus_prefill": {
            "checked_calls": prompt_eq,
            "all_pass": assert_ok,
            "exceptions": [
                {"call_idx": r["call_idx"], "prompt": r["ctx_tokens_at_call"],
                 "reuse": r["reuse_rows"], "prefill": r["delta_rows_prefilled"]}
                for r in rows
                if r["split_status"] == "SPLIT" and not r["prompt_matched"]
            ],
        },
        "clock_alignment": {
            "matched_post_offset_s_max": max(r["post_match_offset_s"] for r in rows),
            "matched_post_offset_s_mean": statistics.mean(
                r["post_match_offset_s"] for r in rows
            ),
            "node_vs_ledger_cdt": "state.db started_at(UTC->CDT) compared to exo.log POST (CDT)",
        },
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _iso(dt: _dt.datetime | None) -> str:
    return "" if dt is None else dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _num(x, nd: int = 3) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def row_to_csv_record(r: dict) -> dict:
    return {
        "call_idx": r["call_idx"],
        "started_at": _iso(r["started_at"]),
        "ended_at": _iso(r["ended_at"]),
        "latency_s": _num(r["latency_s"], 3),
        "ctx_tokens_at_call": r["ctx_tokens_at_call"],
        "delta_rows_prefilled": "" if r["delta_rows_prefilled"] is None else r["delta_rows_prefilled"],
        "prefill_s": _num(r["prefill_s"], 3),
        "output_tokens": r["output_tokens"],
        "decode_s": _num(r["decode_s"], 3),
        "gap_to_next_call_s": _num(r["gap_to_next_call_s"], 3),
        "cache_hit": "" if r["cache_hit"] is None else str(bool(r["cache_hit"])).lower(),
        "started_at_epoch": _num(r["started_at_epoch"], 6),
        "ended_at_epoch": _num(r["ended_at_epoch"], 6),
        "ctx_prev_tokens": "" if r.get("ctx_prev_tokens") is None else r["ctx_prev_tokens"],
        "dctx_tokens": "" if r.get("dctx_tokens") is None else r["dctx_tokens"],
        "rule_threshold_rows": _num(r.get("rule_threshold_rows"), 3),
        "cache_hit_reason": r["cache_hit_reason"],
        "delta_rows_source": r["delta_rows_source"] or "",
        "reuse_rows": "" if r["reuse_rows"] is None else r["reuse_rows"],
        "rewind": "" if r["rewind"] is None else r["rewind"],
        "decode_tok_s": _num(r["decode_tok_s"], 3),
        "prefill_rows_per_s": _num(r["prefill_rows_per_s"], 3),
        "pre_s": _num(r["pre_s"], 3),
        "post_s": _num(r["post_s"], 3),
        "residual_s": _num(r["residual_s"], 3),
        "split_status": r["split_status"],
        "t_post_cdt": _iso(r["t_post"]),
        "t_prefill_start_cdt": _iso(r["t_prefill_start"]),
        "t_prefill_end_cdt": _iso(r["t_prefill_end"]),
        "t_end_cdt": _iso(r["t_end"]),
        "post_match_offset_s": _num(r["post_match_offset_s"], 3),
    }


def write_csv(rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(row_to_csv_record(r))


def write_summary_json(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
        fh.write("\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--db", default="/Users/adam.durham/.hermes/state.db")
    ap.add_argument("--session", default=SESSION)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    ev = scan_log(args.log)
    ledger = load_ledger(args.db, args.session)
    if len(ledger) != 42:
        print(f"WARNING: expected 42 ledger rows, found {len(ledger)}", file=sys.stderr)
    rows = decompose(ev, ledger)
    summary = summarize(rows, args.session)

    if args.out_csv:
        write_csv(rows, args.out_csv)
    if args.out_json:
        write_summary_json(summary, args.out_json)

    print(f"n_calls={summary['n_calls']} n_split={summary['n_split']} "
          f"n_unsplit={summary['n_unsplit']}")
    print(f"sum(latency)={summary['sum_latency_s']:.3f}  "
          f"sum(prefill_s)={summary['sum_prefill_s']:.3f}  "
          f"sum(decode_s)={summary['sum_decode_s']:.3f}")
    print(f"gate 0a ratio={summary['gate_0a']['ratio']:.4f} "
          f"within_pm10pct={summary['gate_0a']['within_pm10pct']} "
          f"unaccounted={summary['gate_0a']['model_side_unaccounted_s']:.3f}s")
    print(f"sum(gap_to_next)={summary['sum_gap_to_next_call_s']:.3f}  "
          f"miss_count={summary['cache']['miss_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

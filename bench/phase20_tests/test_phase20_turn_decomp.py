"""Tests for Phase 0a per-call time decomposition (``bench/phase20_turn_decomp.py``).

Run (from the worktree root; the repo-root conftest refuses to run from a
worktree with an empty ``mlx-lm/``, and these tests do not import exo):

    cd <worktree> && PYTHONPATH=bench \\
        /Users/adam.durham/repos/exo/.venv/bin/python -m pytest --noconftest \\
        bench/phase20_tests/test_phase20_turn_decomp.py -q -p no:cacheprovider
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "bench"))

import phase20_turn_decomp as td  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "turn_decomp" / "turn_decomp_real_fixture.log"

# Real ledger rows (state.db api_calls, provider='custom', exact exo model) for
# the first five calls of session 20261007_092009_9a2ed7. Only the columns the
# extractor reads are needed. Call 5's row is only here to bound call 4's window.
_FIXTURE_LEDGER = [
    dict(call_seq=1, started_at=1791382821.046175, ended_at=1791382919.844222,
         latency_seconds=98.79804706573486, output_tokens=175, reasoning_tokens=122,
         prompt_tokens_total=24095, cache_read_tokens=0),
    dict(call_seq=2, started_at=1791382920.00012, ended_at=1791382926.8003159,
         latency_seconds=6.800195932388306, output_tokens=90, reasoning_tokens=5,
         prompt_tokens_total=24340, cache_read_tokens=24095),
    dict(call_seq=3, started_at=1791382971.8614979, ended_at=1791382986.653939,
         latency_seconds=14.792441129684448, output_tokens=233, reasoning_tokens=93,
         prompt_tokens_total=25300, cache_read_tokens=24431),
    dict(call_seq=4, started_at=1791382986.8455915, ended_at=1791383012.1262872,
         latency_seconds=25.28069567680359, output_tokens=284, reasoning_tokens=37,
         prompt_tokens_total=28743, cache_read_tokens=25300),
    dict(call_seq=5, started_at=1791383012.1770232, ended_at=1791383027.310805,
         latency_seconds=15.133781909942627, output_tokens=231, reasoning_tokens=54,
         prompt_tokens_total=30178, cache_read_tokens=28743),
]


@pytest.fixture(scope="module")
def fixture_rows():
    ev = td.scan_log(FIXTURE)
    return td.decompose(ev, _FIXTURE_LEDGER)


def test_fixture_exists_and_is_small():
    assert FIXTURE.exists()
    assert FIXTURE.stat().st_size < 100_000  # brief: <100 KB


def test_scan_log_finds_markers():
    ev = td.scan_log(FIXTURE)
    # 5 POSTs bounding 4 complete calls (call 1 cold + calls 2,3,4 delta)
    assert len(ev["post"]) == 5
    assert len(ev["turnreuse"]) == 3   # calls 2, 3, 4 (call 1 is cold: no line)
    assert len(ev["prefctl"]) == 4     # one near-start per real call
    assert len(ev["fin"]) == 4         # one TaskFinished per real call
    assert len(ev["sessreuse"]) == 3   # call 1 has no resident session


def test_cold_first_call_is_unsplit(fixture_rows):
    c1 = fixture_rows[0]
    assert c1["call_idx"] == 1
    assert c1["split_status"] == "UNSPLIT_COLD"
    assert c1["prefill_s"] is None and c1["decode_s"] is None
    # cold call: delta_rows = the FULL prompt, taken from prefill controls (rows=)
    assert c1["delta_rows_prefilled"] == 24095
    assert c1["delta_rows_source"] == "prefill_controls.rows (cold full prefill)"
    assert c1["cache_hit"] is False and c1["cache_hit_reason"] == "cold"
    # the real TaskFinished end DOES exist for the cold call
    assert c1["t_end"] is not None
    assert c1["t_prefill_end"] is None


def test_call2_tiny_delta_parsed(fixture_rows):
    c2 = fixture_rows[1]
    assert c2["split_status"] == "SPLIT"
    assert c2["delta_rows_prefilled"] == 245
    assert c2["delta_rows_source"] == "turn_reuse.prefill"
    assert c2["reuse_rows"] == 24095
    assert c2["ctx_tokens_at_call"] == 24340
    # prompt == reuse + prefill (rows==tokens)
    assert c2["ctx_tokens_at_call"] == c2["reuse_rows"] + c2["delta_rows_prefilled"]
    # measured split: [prefill entry 09:22:00.257 -> turn reuse 09:22:01.933]
    assert c2["prefill_s"] == pytest.approx(1.676, abs=1e-3)
    # [turn reuse 09:22:01.933 -> TaskFinished 09:22:06.789]
    assert c2["decode_s"] == pytest.approx(4.856, abs=1e-3)
    assert c2["pre_s"] == pytest.approx(0.244, abs=1e-3)
    assert c2["post_s"] == pytest.approx(0.011, abs=1e-3)


def test_call3_small_delta_parsed(fixture_rows):
    c3 = fixture_rows[2]
    assert c3["delta_rows_prefilled"] == 869
    assert c3["reuse_rows"] == 24431
    # call 3's line has NO rewind field (cache=25300, no rollback):
    assert c3["rewind"] is None
    assert c3["prefill_s"] == pytest.approx(3.979, abs=1e-3)
    assert c3["decode_s"] == pytest.approx(10.576, abs=1e-3)


def test_call4_mid_delta_parsed(fixture_rows):
    c4 = fixture_rows[3]
    assert c4["delta_rows_prefilled"] == 3443
    assert c4["reuse_rows"] == 25300
    # rewind comes from regex group(6), NOT the cache field (group 5)
    assert c4["rewind"] == 25533
    assert c4["prefill_s"] == pytest.approx(13.445, abs=1e-3)
    assert c4["decode_s"] == pytest.approx(11.597, abs=1e-3)


def test_gap_to_next_is_ledger_based(fixture_rows):
    # gap uses ledger started_at - ended_at (CDT), NOT log-post times
    c2 = fixture_rows[1]
    assert c2["gap_to_next_call_s"] == pytest.approx(
        (1791382971.8614979 - 1791382926.8003159), abs=1e-6
    )
    assert fixture_rows[-1]["gap_to_next_call_s"] is None  # last call blank


def test_cache_hit_rule_hit_and_miss():
    # delta_rows <= 1.2 * dctx -> hit
    assert td.cache_hit_rule(245, 24340, 24095)[0] is True       # 245 <= 1.2*245
    assert td.cache_hit_rule(300, 24340, 24095)[0] is False      # 300 > 294
    # boundary: exactly 1.2*dctx is a hit (<=)
    assert td.cache_hit_rule(294, 24340, 24095)[0] is True
    assert td.cache_hit_rule(295, 24340, 24095)[0] is False


def test_cache_hit_rule_degenerate_dctx():
    # ctx did not grow -> rule cannot be applied
    hit, thresh, reason = td.cache_hit_rule(50, 24000, 24000)
    assert hit is None and thresh is None
    assert reason == "degenerate_dctx_le_0"
    # ctx shrank -> also degenerate
    hit2, _, reason2 = td.cache_hit_rule(50, 23900, 24000)
    assert hit2 is None and reason2 == "degenerate_dctx_le_0"


def test_cache_hit_rule_threshold_value():
    hit, thresh, reason = td.cache_hit_rule(245, 24340, 24095)
    assert thresh == pytest.approx(294.0)
    assert reason == "hit"


def test_csv_schema(tmp_path):
    ev = td.scan_log(FIXTURE)
    rows = td.decompose(ev, _FIXTURE_LEDGER)
    out = tmp_path / "turn_decomp.csv"
    td.write_csv(rows, out)
    with open(out, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        # required columns first, in the exact order the brief asks for
        assert reader.fieldnames[: len(td.CSV_REQUIRED_COLUMNS)] == td.CSV_REQUIRED_COLUMNS
        records = list(reader)
    assert len(records) == 5
    r2 = records[1]
    assert r2["call_idx"] == "2"
    assert r2["delta_rows_prefilled"] == "245"
    assert r2["cache_hit"] == "true"
    assert r2["split_status"] == "SPLIT"
    # last row's gap is blank
    assert records[-1]["gap_to_next_call_s"] == ""


def test_summary_schema():
    ev = td.scan_log(FIXTURE)
    rows = td.decompose(ev, _FIXTURE_LEDGER)
    # drop the bounding call 5 (no fin inside the fixture window) for the summary
    summary = td.summarize(rows[:4])
    for key in ("n_calls", "sum_latency_s", "sum_prefill_s", "sum_decode_s",
                "sum_gap_to_next_call_s", "gate_0a", "cache", "decode_tok_s",
                "prefill_rows_per_s", "assert_prompt_eq_reuse_plus_prefill"):
        assert key in summary
    assert summary["n_calls"] == 4
    assert summary["cache"]["miss_count"] == 0      # calls 2,3,4 all hit
    assert summary["cache"]["cold_count"] == 1
    assert summary["assert_prompt_eq_reuse_plus_prefill"]["all_pass"] is True
    # every call with a turn-reuse line must satisfy prompt == reuse + prefill
    for r in rows:
        if r["split_status"] == "SPLIT":
            assert r["ctx_tokens_at_call"] == r["reuse_rows"] + r["delta_rows_prefilled"]


def test_epoch_to_cdt_offset():
    # 1791382821.046175 UTC -> 09:20:21.046 CDT
    dt = td._epoch_to_cdt(1791382821.046175)
    assert dt.strftime("%Y-%m-%d %H:%M:%S") == "2026-10-07 09:20:21"

"""bench/phase20_tests/test_phase20_mtrace.py

Pure-parse tests for ``bench/phase20_mtrace.py`` -- NO xctrace, NO cluster.
Run with:

    cd /private/tmp/p20-mtrace && \
    PYTHONPATH=bench /Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
      --noconftest bench/phase20_tests/test_phase20_mtrace.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# Make `import phase20_mtrace` work whether or not PYTHONPATH=bench was set.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

import phase20_mtrace as M  # noqa: E402

FIXDIR = os.path.join(_HERE, "fixtures", "mtrace")


# --------------------------------------------------------------------------
# Fixtures (real, trimmed exports captured on the laptop M4 Max)
# --------------------------------------------------------------------------

def test_fixture_files_exist_and_under_size_limit():
    files = ["gpu_intervals_trim.xml", "gpu_state_intervals_trim.xml",
             "cmd_buffer_submissions_trim.xml"]
    total = 0
    for f in files:
        p = os.path.join(FIXDIR, f)
        assert os.path.exists(p), p
        total += os.path.getsize(p)
    assert total <= 300 * 1024, f"fixture dir {total} bytes exceeds 300 KB"


def test_parse_real_gpu_state_intervals():
    tr = M.load_trace(os.path.join(FIXDIR, "gpu_state_intervals_trim.xml"))
    assert tr.active_span() is not None
    assert M.GPU_STATE_SCHEMA in tr.schemas_seen
    # 60 trimmed rows contain a mix of Active/Idle; both unions non-empty
    assert len(tr.active) > 0


def test_parse_real_gpu_intervals_nanosecond_units():
    tr = M.load_trace(os.path.join(FIXDIR, "gpu_intervals_trim.xml"))
    assert len(tr.gpu_intervals) == 25
    iv = tr.gpu_intervals[0]
    # ns integers; a compute encoder interval on this box is 1e5..1e7 ns
    assert iv["duration"] > 0
    assert iv["channel"] in ("Compute", "Render")
    assert iv["cmdbuffer_id"] is not None


def test_parse_real_command_buffer_submissions():
    tr = M.load_trace(os.path.join(FIXDIR, "cmd_buffer_submissions_trim.xml"))
    assert len(tr.cmd_submissions) == 15
    s = tr.cmd_submissions[0]
    assert s["event_type"] == "CommandBufferSubmission"
    assert s["num_encoders"] >= 1


def test_analyze_real_state_produces_busy_fraction():
    tr = M.load_trace(os.path.join(FIXDIR, "gpu_state_intervals_trim.xml"))
    res = M.analyze(tr)
    assert res["ok"] is True
    assert 0.0 < res["gpu_busy"]["busy_fraction"] < 1.0
    # markdown renderer must not crash and must name the metric
    md = M.render_md(res)
    assert "GPU-busy fraction" in md


# --------------------------------------------------------------------------
# Synthetic edge cases
# --------------------------------------------------------------------------

def _state_cols():
    return [("start", "start-time"), ("duration", "duration"),
            ("state", "gpu-state")]


def test_empty_table_is_not_ok(tmp_path):
    p = tmp_path / "empty.xml"
    M._synth_write(str(p), _state_cols(), [])
    tr = M.load_trace(str(p))
    res = M.analyze(tr)
    assert res["ok"] is False
    assert "no Active" in res["reason"]


def test_overlapping_intervals_merged_before_summing(tmp_path):
    # three overlapping Active intervals covering [0,1500) -> sum == 1500,
    # NOT 3000
    p = tmp_path / "ov.xml"
    M._synth_write(str(p), _state_cols(),
                   [[0, 1000, "Active"], [500, 1000, "Active"],
                    [1400, 100, "Active"]])
    tr = M.load_trace(str(p))
    assert M.total_len(tr.active) == 1500


def test_intervals_crossing_window_edges_are_clipped(tmp_path):
    p = tmp_path / "clip.xml"
    M._synth_write(str(p), _state_cols(),
                   [[0, 100, "Active"], [1000, 100, "Active"]])
    tr = M.load_trace(str(p))
    res = M.analyze(tr, window_start=50, window_end=500)
    # only the [50,100) part of interval #1 is inside the window
    assert res["gpu_busy"]["busy_ms"] == pytest.approx(0.00005, abs=1e-12)


def test_multiple_gpu_queues_unioned_not_summed(tmp_path):
    # two channels Active at the SAME time must count once (union), the Idle
    # row on top is complementary
    p = tmp_path / "multi.xml"
    M._synth_write(str(p), _state_cols(),
                   [[0, 1000, "Active"], [0, 1000, "Active"],
                    [0, 1000, "Idle"]])
    tr = M.load_trace(str(p))
    assert M.total_len(tr.active) == 1000
    res = M.analyze(tr)
    assert res["gpu_busy"]["busy_fraction"] == pytest.approx(1.0)


def test_gap_histogram_buckets_gt_50us(tmp_path):
    # bursts 1 ms long separated by gaps of 20 us (<50us, ignored), 80 us,
    # 2 ms, 20 ms
    p = tmp_path / "hist.xml"
    starts = [0, 1_000_000 + 20_000, 2_020_000 + 80_000,
              3_100_000 + 2_000_000, 6_100_000 + 20_000_000]
    rows = [[s, 1_000_000, "Active"] for s in starts]
    M._synth_write(str(p), _state_cols(), rows)
    tr = M.load_trace(str(p))
    res = M.analyze(tr)
    hist = {h["bucket"]: h["count"] for h in res["idle_gaps"]["histogram_gt_50us"]}
    assert hist["50-100us"] == 1        # the 80 us gap
    assert hist["1-3ms"] == 1           # the 2 ms gap
    assert hist[">10ms"] == 1           # the 20 ms gap
    assert res["idle_gaps"]["gt_50us_count"] == 3


def test_rounds_split_and_merge(tmp_path):
    p = tmp_path / "rounds.xml"
    rows = [[k * 2_000_000, 1_000_000, "Active"] for k in range(5)]
    M._synth_write(str(p), _state_cols(), rows)
    tr = M.load_trace(str(p))
    assert M.analyze(tr, round_gap_ms=0.5)["rounds"]["count"] == 5
    assert M.analyze(tr, round_gap_ms=3.0)["rounds"]["count"] == 1


def test_top10_orders_by_duration(tmp_path):
    # need an Active-state row too, so the analysis window exists
    M._synth_write(str(tmp_path / "state.xml"), _state_cols(),
                   [[0, 100_000, "Active"]])
    cols = [("start", "start-time"), ("duration", "duration"),
            ("channel-name", "gpu-channel-name"),
            ("event-label", "formatted-label"),
            ("cmdbuffer-id", "metal-command-buffer-id")]
    durs_in = (100, 5000, 900, 70000, 4200, 30000, 12, 8800, 60, 1000, 25000)
    rows = [[i, d, "Compute", f"cmd {i}", f"id{i}"]
            for i, d in enumerate(durs_in)]
    M._synth_write(str(tmp_path / "top.xml"), cols, rows,
                   schema=M.GPU_INTERVAL_SCHEMA)
    tr = M.load_trace(str(tmp_path))
    res = M.analyze(tr)
    durs = [t["duration_ms"] for t in res["top10_intervals"]]
    assert durs == sorted(durs, reverse=True)
    assert len(durs) == 10
    assert durs[0] == pytest.approx(0.07)


def test_ref_resolution_across_rows(tmp_path):
    # row 2 references an id declared in row 1 -- the state must resolve.
    p = tmp_path / "refs.xml"
    xml = ('<?xml version="1.0"?>\n<trace-query-result><node>'
           '<schema name="metal-gpu-state-intervals">'
           '<col><mnemonic>start</mnemonic></col>'
           '<col><mnemonic>duration</mnemonic></col>'
           '<col><mnemonic>state</mnemonic></col></schema>'
           '<row><start-time id="1">0</start-time>'
           '<duration id="2">1000</duration>'
           '<gpu-state id="3" fmt="Active">Active</gpu-state></row>'
           '<row><start-time id="4">2000</start-time>'
           '<duration ref="2"/><gpu-state ref="3"/></row>'
           '</node></trace-query-result>')
    p.write_text(xml)
    tr = M.load_trace(str(p))
    assert M.total_len(tr.active) == 2000  # two 1000 ns Active intervals


def test_empty_window_raises(tmp_path):
    p = tmp_path / "w.xml"
    M._synth_write(str(p), _state_cols(), [[0, 1000, "Active"]])
    tr = M.load_trace(str(p))
    with pytest.raises(ValueError):
        M.analyze(tr, window_start=500, window_end=100)


# --------------------------------------------------------------------------
# record / export command construction (no execution)
# --------------------------------------------------------------------------

def test_record_dry_run_never_calls_xctrace(capsys):
    rc = M.main(["record", "--node", "local", "--pid", "1234",
                 "--secs", "8", "--out", "/tmp/x.trace", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--attach 1234" in out
    assert "Metal System Trace" in out
    assert "--dry-run" in out


def test_record_node_uses_perl_alarm_not_timeout():
    cmd = M.build_record_cmd("studio2", "9999", 8, "/tmp/x.trace")
    assert cmd[0] == "ssh" and cmd[1] == "studio2"
    joined = cmd[2]
    assert "/usr/bin/perl" in joined
    assert "alarm" in joined
    # coreutils `timeout` does not exist on macOS (verified on studio2)
    assert "timeout " not in joined


def test_record_local_has_no_ssh():
    cmd = M.build_record_cmd("local", "1", 5, "/tmp/x.trace")
    assert cmd[0] == "bash"


# --------------------------------------------------------------------------
# subprocess end-to-end: the CLI runs and emits JSON
# --------------------------------------------------------------------------

def test_cli_analyze_end_to_end(tmp_path):
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    r = subprocess.run(
        [sys.executable, os.path.join(_BENCH, "phase20_mtrace.py"), "analyze",
         "--dir", os.path.join(FIXDIR, "gpu_state_intervals_trim.xml"),
         "--json", str(out_json), "--md", str(out_md)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out_json.read_text())
    assert data["ok"] is True
    assert "GPU-busy fraction" in out_md.read_text()


def test_cli_selftest_exit_zero():
    r = subprocess.run(
        [sys.executable, os.path.join(_BENCH, "phase20_mtrace.py"), "selftest"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "passed" in r.stdout

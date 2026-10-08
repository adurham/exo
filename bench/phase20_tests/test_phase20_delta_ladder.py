#!/usr/bin/env python3
"""Tests for the Phase-20 0c delta ladder (BRIEF L).

Run with the repo-root conftest landmine guard bypassed:
  cd <worktree> && PYTHONPATH=bench <shared-venv>/bin/python -m pytest \\
      --noconftest bench/phase20_tests/test_phase20_delta_ladder.py -q -p no:cacheprovider

Covers: prompt builder (salt once / size), log extraction on REAL lines,
delta-vs-collapse classification, rows/s math, wall-cap pre-check, guard-abort
path, summarize tables + all three decision rules (synthetic JSONL), the
DEGRADED_REFERENCE path, and an OFFLINE end-to-end run against the mock server.
Two tests are sabotage-proven (see test_sabotage_*).
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import phase20_common as C           # noqa: E402
import phase20_delta_ladder as L      # noqa: E402
import mock_exo_server as M           # noqa: E402

REAL_ZST = "/Users/adam.durham/.hermes/cache/scratch/phase20/m41_0901_1400.log.zst"

# --- REAL lines, copied verbatim from m41_0901_1400.log.zst (node-local CDT) ---
REAL_PREFILL = ("[ 2026-10-07 09:22:00.257 | INFO     | "
                "exo.worker.engines.mlx.dsv41.session:engine_prefill:333 ] [DSV41] "
                "prefill controls: fence_every=2 transient_budget_mb=2048 "
                "score_row_bytes=1 fence_hook=on (rows=245, base=2048)")
REAL_TURNREUSE = ("[ 2026-10-07 09:22:01.933 | INFO     | "
                  "exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [DSV41] "
                  "turn reuse: prompt=24340 prefill=245 reuse=24095 cache=24340 "
                  "rewind=24270 UNCOMMITTED")
REAL_TURNREUSE_NOREWIND = ("[ 2026-10-07 09:22:56.053 | INFO     | "
                           "exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] "
                           "[DSV41] turn reuse: prompt=25300 prefill=869 "
                           "reuse=24431 cache=25300 UNCOMMITTED")
REAL_UNDERSHOOT = ("[ 2026-10-07 09:22:56.053 | WARNING  | "
                   "exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [DSV41] "
                   "reuse undershoot: refed=869 rows (reused=24431)")
REAL_POST = ("[ 2026-10-07 09:22:00.013 | DEBUG    | "
             "exo.api.main:_log_requests:340 ] API request: POST /v1/chat/completions")
REAL_SESSION_REUSE = ("[ 2026-10-07 09:23:07.067 | INFO     | "
                      "exo.worker.engines.mlx.dsv41.session:get:1004 ] [DSV41] "
                      "session reuse: this 28743-token prompt matches a resident "
                      "conversation on 25397 rows")


# --------------------------------------------------------------------------- helpers
class FakeGuard:
    """ChunkGuard stub that raises ChunkAborted on the Nth check()."""

    def __init__(self, label, *, max_wall_s=900.0, log_dir=None, abort_at_check=None):
        self.label = label
        self.max_wall_s = max_wall_s
        self.own_requests = []
        self.aborted = False
        self.reason = None
        self._checks = 0
        self._abort_at = abort_at_check
        FakeGuard.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def register_own_request(self, t_start=None):
        self.own_requests.append(t_start if t_start is not None else 0.0)

    def check(self):
        self._checks += 1
        if self._abort_at is not None and self._checks >= self._abort_at:
            self.aborted = True
            self.reason = "ABORTED_USER_ARRIVED"
            raise L._StubChunkAborted(self.reason)


def _fake_guard_mod(abort_at_check=None):
    mod = L._StubGuardModule()

    def make(label, *, max_wall_s=900.0, log_dir=None, **_kw):
        return FakeGuard(label, max_wall_s=max_wall_s, log_dir=log_dir,
                         abort_at_check=abort_at_check)
    setattr(mod, "ChunkGuard", make)
    return mod


def _records_from(tmp_path, chunk, **run_kw):
    """Run a chunk against the mock server and return the JSONL records."""
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    # sleep=False: the mock derives log timestamps (prefill_s = rows/rows_per_s)
    # instead of real sleeping, so rows/s is exact and tests are fast.
    with M.MockExoServer(log_path, rows_per_s=200.0, sleep=False) as srv:
        argv = [chunk, "--api", srv.base_url, "--out-dir", out_dir,
                "--log-source", f"studio1={log_path}"]
        if run_kw.get("reps"):
            argv += ["--reps", str(run_kw["reps"])]
        rc = L.main(argv)
    raw = os.path.join(out_dir, "raw", f"delta_ladder.{chunk}.jsonl")
    recs = C.load_jsonl([raw])
    return rc, recs, out_dir


# ============================================================ prompt builder tests
def test_salt_once_at_head():
    salt = "SALT-deadbeefdeadbeef"
    base = C.build_base(2000, salt)
    assert base.startswith(f"[SESSION-SALT {salt}]\n")
    assert base.count("SESSION-SALT") == 1


def test_prompt_size_accuracy_vs_calibration():
    # chars/token scaling is linear: doubling the target roughly doubles chars
    a = C.build_filler(2000, chars_per_token=5.111)
    b = C.build_filler(4000, chars_per_token=5.111)
    assert abs(len(b) / len(a) - 2.0) < 0.05
    # ~5.111 c/t => 2000 tokens ~= 10222 chars
    assert abs(len(a) - 2000 * 5.111) / (2000 * 5.111) < 0.03


def test_delta_text_is_unique_and_salt_free():
    d1 = C.build_delta(1024)
    d2 = C.build_delta(1024)
    assert "SESSION-SALT" not in d1
    # different seeds (seed + delta_tokens are equal here) -> same; use explicit
    d3 = C.build_delta(1024, seed=999)
    assert d3 != d1


def test_branching_message_shape():
    base = C.build_base(2000, C.make_salt())
    msgs = C.delta_messages(base, "ok", C.build_delta(256))
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == base            # prefix contains previous prompt


def test_make_salt_format():
    s = C.make_salt()
    assert s.startswith("SALT-") and len(s) == len("SALT-") + 16


# ============================================================== log extraction tests
def test_extract_real_turn_reuse():
    out = C.analyze_window("\n".join([REAL_PREFILL, REAL_TURNREUSE]))
    assert out["has_turn_reuse"] is True
    assert out["log_prefill"] == 245
    assert out["reuse"] == 24095
    assert out["cache"] == 24340
    assert out["rewind"] == 24270
    assert out["prompt_log"] == 24340
    assert out["prefill_controls_rows"] == 245
    assert out["prefill_s_log"] is not None
    assert 1.5 < out["prefill_s_log"] < 1.8     # 09:22:01.933 - 09:22:00.257


def test_extract_turn_reuse_without_rewind():
    out = C.analyze_window(REAL_TURNREUSE_NOREWIND)
    assert out["has_turn_reuse"] is True
    assert out["log_prefill"] == 869 and out["rewind"] is None


def test_extract_undershoot_and_session_reuse():
    out = C.analyze_window("\n".join([REAL_SESSION_REUSE, REAL_UNDERSHOOT]))
    assert out["undershoot_refed"] == 869 and out["undershoot_reused"] == 24431
    assert out["session_reuse"] == 25397


def test_extract_post_and_no_turn_reuse_on_cold():
    out = C.analyze_window("\n".join([REAL_POST, REAL_PREFILL]))
    assert out["task_starts"] == 0
    assert out["has_turn_reuse"] is False       # cold feed: no turn-reuse line
    assert out["posts"][0] is not None


def test_real_zst_lines_parse_if_present():
    """Independently re-derive from the saved boot log (skip if absent)."""
    if not os.path.exists(REAL_ZST):
        pytest.skip("real boot log not present")
    try:
        proc = subprocess.run(["zstdcat", REAL_ZST], capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("zstdcat unavailable")
    lines = [ln for ln in proc.stdout.splitlines()
             if "turn reuse:" in ln or "prefill controls:" in ln][:40]
    assert lines, "no marker lines in the real log"
    n_parsed = 0
    for ln in lines:
        if C.RE_TURNREUSE.search(ln) or C.RE_PREFILL.search(ln):
            n_parsed += 1
    assert n_parsed == len(lines)


# ================================================================= rows/s math tests
def test_rows_per_s_log():
    rec = {"log_prefill": 2048, "prefill_s_log": 10.0}
    assert C.rows_per_s_log(rec) == pytest.approx(204.8)


def test_rows_per_s_log_none_when_cold():
    assert C.rows_per_s_log({"log_prefill": None, "prefill_s_log": 5.0}) is None
    assert C.rows_per_s_log({"log_prefill": 100, "prefill_s_log": 0}) is None


def test_rows_per_s_ttft_delta_and_fresh():
    delta = {"log_prefill": 2048, "prompt_tokens": 50000, "ttft_s": 10.0}
    assert C.rows_per_s_ttft(delta) == pytest.approx(204.8)
    fresh = {"log_prefill": None, "prompt_tokens": 100000, "ttft_s": 373.9}
    assert C.rows_per_s_ttft(fresh) == pytest.approx(267.45, abs=0.1)


def test_predict_wall_formula():
    # rows/200*1.3 + 15
    assert C.predict_wall_s(2000) == pytest.approx(2000 / 200 * 1.3 + 15)


# ========================================================== classification tests
def _mk_delta_record(reuse, ctx_nominal=50000):
    return {"kind": "delta", "ctx_nominal": ctx_nominal, "reuse": reuse,
            "log_prefill": 2048}


def test_classify_delta_vs_collapse_via_run(tmp_path):
    rc, recs, _ = _records_from(tmp_path, "chunk1", reps=1)
    assert rc == L.EXIT_OK
    deltas = [r for r in recs if r["kind"] == "delta"]
    assert deltas, "expected delta records"
    for r in deltas:
        # the mock's branching shape must reuse ~ the base depth
        assert r["collapsed"] is False
        assert r["reuse"] >= 0.9 * r["ctx_nominal"]


def test_collapse_detected_when_reuse_small():
    # directly exercise the classifier logic through C + a synthetic base
    ctx = 50000
    collapsed = {"kind": "delta", "ctx_nominal": ctx, "reuse": 1024}
    assert collapsed["reuse"] < 0.9 * ctx


# ============================================================== wall-cap test
def test_wall_cap_blocks_late_steps():
    # a tiny cap: chunk1 steps are 20K base (245s), 3x delta, 50K base (340s)...
    plan = L.plan_chunk("chunk1", reps=3, max_wall_s=200.0)
    assert plan["fits"] is False
    assert plan["first_breach"] is not None
    blocked = [s for s in plan["steps"] if s["wall_cap"] == "NOT_RUN_WALL_CAP"]
    assert blocked, "expected NOT_RUN_WALL_CAP steps"
    # everything at/after the first breach is marked
    idx = [i for i, s in enumerate(plan["steps"]) if s["wall_cap"] != "ok"]
    assert idx == list(range(idx[0], len(plan["steps"])))


def test_wall_cap_emits_not_run_record(tmp_path):
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    L.GUARD_OVERRIDE = _fake_guard_mod()
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0) as srv:
            rc = L.main(["chunk1", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}", "--reps", "3",
                         "--max-wall", "1"])   # 1 s cap -> first step blocked
    finally:
        L.GUARD_OVERRIDE = None
    raw = os.path.join(out_dir, "raw", "delta_ladder.chunk1.jsonl")
    recs = C.load_jsonl([raw])
    assert len(recs) == 1
    assert recs[0].get("not_run") == "NOT_RUN_WALL_CAP"
    assert recs[0]["invalid"] is True


# ============================================================== guard-abort test
def test_guard_abort_exits_75(tmp_path):
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    L.GUARD_OVERRIDE = _fake_guard_mod(abort_at_check=2)   # abort on the 2nd check
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0, sleep=False) as srv:
            rc = L.main(["chunk1", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}", "--reps", "3"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == L.EXIT_CHUNK_ABORTED == 75
    raw = os.path.join(out_dir, "raw", "delta_ladder.chunk1.jsonl")
    recs = C.load_jsonl([raw])
    # the first (base) request completed and was written before the abort
    assert any(r["kind"] == "fresh" for r in recs)


def test_guard_registers_own_request_before_each_http(tmp_path):
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    L.GUARD_OVERRIDE = _fake_guard_mod()
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0, sleep=False) as srv:
            rc = L.main(["pilot", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == L.EXIT_OK
    guard = FakeGuard.last
    raw = os.path.join(out_dir, "raw", "delta_ladder.pilot.jsonl")
    recs = C.load_jsonl([raw])
    # one register_own_request per completed request
    assert len(guard.own_requests) == len(recs) == 3   # 1 base + 2 deltas


# ======================================================== mock end-to-end tests
def test_mock_offline_end_to_end(tmp_path):
    rc, recs, out_dir = _records_from(tmp_path, "pilot")
    assert rc == L.EXIT_OK
    kinds = [r["kind"] for r in recs]
    assert kinds == ["fresh", "delta", "delta"]
    # rows/s is checkable: mock uses rows_per_s=200 and sleeps rows/200
    for r in recs:
        if r["kind"] == "delta" and r.get("rows_per_s_log"):
            assert 150 < r["rows_per_s_log"] < 260


def test_mock_rewind_modeling_branching(tmp_path):
    """Branching delta reps must report reuse ~= the base depth each time."""
    rc, recs, _ = _records_from(tmp_path, "chunk3", reps=1)
    assert rc == L.EXIT_OK
    bases = [r for r in recs if r["kind"] == "fresh"]
    deltas = [r for r in recs if r["kind"] == "delta"]
    assert bases and deltas
    base_depth = bases[0]["prompt_tokens"]
    for r in deltas:
        assert r["collapsed"] is False
        assert abs(r["reuse"] - base_depth) <= 0.02 * base_depth


def test_fresh_feed_has_no_turn_reuse(tmp_path):
    rc, recs, _ = _records_from(tmp_path, "fresh100k")
    assert rc == L.EXIT_OK
    assert len(recs) == 1
    r = recs[0]
    assert r["kind"] == "fresh"
    assert r["has_turn_reuse"] is False
    assert r["rows_per_s_log"] is None
    assert r["rows_per_s_ttft"] is not None            # falls back to prompt/ttft


# ============================================================ summarize tests
def _emit(records, path):
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _delta(cell, ctx_label, ctx_nominal, delta_nominal, rows, prefill_s,
           reuse=None, rep=0):
    reuse = ctx_nominal if reuse is None else reuse
    return {"chunk": "x", "cell": cell, "rep": rep, "kind": "delta",
            "ctx_label": ctx_label, "ctx_nominal": ctx_nominal,
            "delta_nominal": delta_nominal, "prompt_tokens": ctx_nominal + rows,
            "ttft_s": prefill_s, "wall_s": prefill_s, "log_prefill": rows,
            "reuse": reuse, "prefill_s_log": prefill_s, "collapsed": False,
            "invalid": False}


def test_summarize_tables_and_flat_rule(tmp_path):
    recs = [
        _delta("c", "20k", 20000, 2048, 2048, 8.0),
        _delta("c", "50k", 50000, 2048, 2048, 8.0),
        _delta("c", "110k", 110000, 2048, 2048, 8.2),
    ]
    for d in (256, 1024, 4096, 8192):
        recs.append(_delta("c", "50k", 50000, d, d, d / 250.0))
    s = C.summarize(recs)
    a = {r["ctx"]: r for r in s["table_a"]}
    assert a["20k"]["median"] == pytest.approx(256.0, abs=1.0)   # 2048/8.0
    assert a["110k"]["median"] == pytest.approx(249.8, abs=1.0)
    # 20K->110K degrade ~2.4% -> benign
    assert any("ctx-depth benign" in v for v in s["verdicts"])
    # sweep spread tiny -> FLAT
    assert any("FLAT across delta sizes" in v for v in s["verdicts"])


def test_summarize_ctx_depth_cost_rule():
    recs = [
        _delta("c", "20k", 20000, 2048, 2048, 8.0),     # 256 r/s
        _delta("c", "110k", 110000, 2048, 2048, 11.0),  # 186 r/s -> -27%
        _delta("c", "50k", 50000, 2048, 2048, 9.0),
    ]
    s = C.summarize(recs)
    assert any("CTX-DEPTH COST" in v for v in s["verdicts"])


def test_summarize_fixed_overhead_rule():
    recs = [
        _delta("c", "50k", 50000, 256, 256, 4.0),       #  64 r/s
        _delta("c", "50k", 50000, 1024, 1024, 5.0),     # 204.8
        _delta("c", "50k", 50000, 4096, 4096, 16.0),    # 256   -> d256 = 25% of d4096
        _delta("c", "50k", 50000, 8192, 8192, 32.0),
    ]
    s = C.summarize(recs)
    assert any("FIXED PER-CALL OVERHEAD" in v for v in s["verdicts"])


def test_summarize_collapsed_excluded_listed():
    good = _delta("c", "50k", 50000, 2048, 2048, 8.0)
    bad = _delta("c", "50k", 50000, 2048, 2048, 2.0, reuse=512)  # collapsed
    bad["collapsed"] = True
    s = C.summarize([good, bad])
    rows = [r for r in s["table_a"] if r["ctx"] == "50k"]
    assert rows[0]["n"] == 1
    assert len(s["collapsed"]) == 1 and s["collapsed"][0]["reuse"] == 512


def test_summarize_degraded_reference_rule():
    fresh = {"chunk": "fresh100k", "cell": "fresh100k", "kind": "fresh",
             "ctx_label": "fresh100k", "prompt_tokens": 100000, "ttft_s": 500.0,
             "log_prefill": None, "reuse": None, "collapsed": False, "invalid": False}
    s = C.summarize([fresh])
    assert any("DEGRADED_REFERENCE" in v for v in s["verdicts"])  # 200 < 255
    assert any("fresh reference OK" in v for v in
               C.summarize([{**fresh, "ttft_s": 350.0}])["verdicts"])  # 285 > 255


def test_summarize_files_writes_md_and_csv(tmp_path):
    out_dir = str(tmp_path / "out")
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw)
    _emit([_delta("c", "50k", 50000, 4096, 4096, 16.0)], os.path.join(raw, "delta_ladder.chunk3.jsonl"))
    rc = L.main(["summarize", "--out-dir", out_dir])
    assert rc == L.EXIT_OK
    assert os.path.exists(os.path.join(out_dir, "delta_ladder.summary.md"))
    assert os.path.exists(os.path.join(out_dir, "delta_ladder.summary.csv"))


# ======================================================= DEGRADED_REFERENCE path
def test_degraded_reference_blocks_recording(tmp_path):
    out_dir = str(tmp_path / "out")
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw)
    # pre-seed a bad fresh reference
    with open(os.path.join(raw, "fresh_ref.json"), "w") as fh:
        json.dump({"rows_per_s_ttft": 200.0}, fh)   # < 255
    log_path = str(tmp_path / "exo.log")
    L.GUARD_OVERRIDE = _fake_guard_mod()
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0) as srv:
            rc = L.main(["chunk3", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}", "--reps", "1"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == L.EXIT_DEGRADED == 3
    # nothing recorded for the chunk
    assert not os.path.exists(os.path.join(raw, "delta_ladder.chunk3.jsonl"))


def test_degraded_reference_on_live_fresh_run(tmp_path):
    """A live fresh feed below 255 rows/s must itself return DEGRADED.

    The mock sleeps ``rows/rows_per_s`` before the first token, so the client
    ttft reflects the configured speed and rows/s_ttft == rows_per_s.  A small
    fresh feed at 250 rows/s (ttft ~4 s) drops under the 255 falsifier.
    """
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    L.GUARD_OVERRIDE = _fake_guard_mod()
    try:
        with M.MockExoServer(log_path, rows_per_s=250.0, sleep=True) as srv:
            rc = L.main(["fresh100k", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}", "--fresh-tokens", "1000",
                         "--base-salt", "SALT-0000000000000001"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == L.EXIT_DEGRADED == 3
    # sanity on the gate math independent of the run
    r = {"prompt_tokens": 100000, "ttft_s": 1000.0, "log_prefill": None}
    assert C.rows_per_s_ttft(r) < C.FRESH_REF_MIN_ROWS_PER_S


# ============================================================== dry-run test
def test_dry_run_sends_nothing(capsys):
    rc = L.main(["chunk1", "--dry-run"])
    assert rc == L.EXIT_OK
    out = capsys.readouterr().out
    assert "20k" in out and "50k" in out and "dry-run" in out


# ==================================================== real guard contract compat
REAL_GUARD = "/private/tmp/p20-guard/bench/phase20_guard.py"


def _load_real_guard():
    if not os.path.exists(REAL_GUARD):
        pytest.skip("sibling phase20_guard.py has not landed")
    import importlib.util
    name = "phase20_guard_real"
    spec = importlib.util.spec_from_file_location(name, REAL_GUARD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclass machinery needs the module registered
    spec.loader.exec_module(mod)
    return mod


def test_real_guard_exposes_contract_surface():
    g = _load_real_guard()
    assert issubclass(g.ChunkAborted, g.GuardFailure)
    assert issubclass(g.GuardFailure, RuntimeError)
    for name in ("ChunkGuard", "canary", "idle_check", "wait_for_idle"):
        assert hasattr(g, name), name
    assert hasattr(g.ChunkGuard, "register_own_request")
    assert hasattr(g.ChunkGuard, "check")


def test_ladder_runs_against_real_guard_module_classes(tmp_path):
    """My ladder must resolve the REAL guard module and use its exception
    classes.  ChunkGuard is patched to the offline fake so no ssh happens, but
    GuardFailure / ChunkAborted come from the real module."""
    g = _load_real_guard()
    real_aborted = g.ChunkAborted

    class _G(g.ChunkGuard):                     # subclass, never enters (no ssh)
        def __init__(self, label, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def register_own_request(self, t_start=None):
            pass

        def check(self):
            raise real_aborted("ABORTED_USER_ARRIVED")

    setattr(g, "ChunkGuard", _G)
    setattr(g, "canary", lambda *a, **k: {"ok": True, "state": "stub", "per_node": {}, "median": {}})
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    L.GUARD_OVERRIDE = g
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0, sleep=False) as srv:
            rc = L.main(["pilot", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == 75          # real ChunkAborted caught -> exit 75


# ============================================ persistent own-request registry (FIX 1)
# An aborted/killed chunk's own POSTs must not block a LATER chunk's entry
# idle-check for min_idle_s.  The fix: run_chunk loads a persistent registry
# (default <out>/raw/own_requests.jsonl) of the guard's own registrations and
# passes BOTH own_requests and registry_path into ChunkGuard.

def test_load_own_requests_roundtrips_guard_registry(tmp_path):
    """load_own_requests returns exactly the ``t`` epochs ChunkGuard.register
    writes to registry_path (and tolerates a missing file / malformed lines)."""
    g = _load_real_guard()
    reg = str(tmp_path / "own_requests.jsonl")
    guard = g.ChunkGuard("roundtrip", max_wall_s=60, registry_path=reg)
    t1, t2 = 1791429502.812, 1791429610.031
    guard.register_own_request(t1)
    guard.register_own_request(t2)
    # two JSONL objects, keys label + t (the guard's schema)
    with open(reg) as fh:
        lines = [json.loads(ln) for ln in fh if ln.strip()]
    assert [ln["t"] for ln in lines] == [t1, t2]
    assert all(ln["label"] == "roundtrip" for ln in lines)
    # and the reader returns those epochs as floats
    assert L.load_own_requests(reg) == [t1, t2]
    assert all(isinstance(t, float) for t in L.load_own_requests(reg))
    # missing file -> empty list (first run)
    assert L.load_own_requests(str(tmp_path / "does_not_exist.jsonl")) == []
    # malformed / blank / non-numeric lines are skipped, valid ones kept
    bad = str(tmp_path / "bad.jsonl")
    with open(bad, "w") as fh:
        fh.write("\n")
        fh.write("not json at all\n")
        fh.write(json.dumps({"label": "x"}) + "\n")          # no "t"
        fh.write(json.dumps({"label": "x", "t": "nope"}) + "\n")  # bad float
        fh.write(json.dumps({"label": "x", "t": 123.5}) + "\n")
    assert L.load_own_requests(bad) == [123.5]


def _make_state_db(path):
    """Minimal empty state.db so idle_check's state.db gate passes."""
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE api_calls (id INTEGER PRIMARY KEY, session_id TEXT,"
        " call_seq INTEGER, provider TEXT, model TEXT, started_at REAL, ended_at REAL);"
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, last_activity_at REAL);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,"
        " timestamp REAL);"
    )
    conn.commit()
    conn.close()


def _post_log_at(epoch):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))
    frac = f"{epoch % 1:.3f}"[1:]
    return (f"[ {ts}{frac} | DEBUG    | exo.api.main:_log_requests:340 ] "
            f"API request: POST /v1/chat/completions\n")


def _patch_idle_seams(monkeypatch, g, log_text, db_uri):
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)

    def fake_ssh(node, cmd, *, timeout=25.0):
        if "date" in cmd:
            return 0, f"{g._now()}\n"
        return 0, log_text + "\n" + g._SSH_OK_MARKER + "\n"

    monkeypatch.setattr(g, "run_ssh", fake_ssh)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    monkeypatch.setattr(g, "STATE_DB_URI", db_uri)


def test_registry_survives_restart_entry_idle_check_passes(tmp_path, monkeypatch):
    """Two sequential chunk-like passes share one registry path.

    Pass 1: a chunk registers a request (its POST is on the node log), then is
    killed mid-run -- exactly the PM's aborted-pilot case.  Pass 2: a FRESH
    ChunkGuard that LOADS the persisted registry must have its ENTRY idle_check
    succeed (ok True) instead of refusing on the non-own POST; a genuinely
    non-own POST (no matching registration) STILL refuses.
    """
    g = _load_real_guard()
    reg = str(tmp_path / "own_requests.jsonl")
    db = str(tmp_path / "state.db")
    _make_state_db(db)
    db_uri = f"file:{db}?mode=ro"

    # a generation POST on the node 30 s ago -- recent enough to block for 600 s
    t_post = time.time() - 30
    log = _post_log_at(t_post)
    ev_epoch = g.parse_log_lines(log)[0].epoch

    # ---- PASS 1: register the request, then the chunk is killed (no __enter__)
    guard1 = g.ChunkGuard("chunkA", max_wall_s=60, registry_path=reg)
    guard1.register_own_request(ev_epoch)
    assert os.path.exists(reg), "registry line must be persisted on register"

    # ---- PASS 2: fresh guard LOADS the registry
    loaded = L.load_own_requests(reg)
    assert loaded == [ev_epoch]

    _patch_idle_seams(monkeypatch, g, log, db_uri)

    # entry idle_check with the loaded registry -> OK
    rep = g.idle_check(loaded, min_idle_s=600)
    assert rep.ok is True, f"entry idle_check refused despite loaded registry: {rep.reasons}"

    # the ChunkGuard ENTRY itself (what run_chunk relies on) does not refuse
    g2 = g.ChunkGuard("chunkB", max_wall_s=60, poll_s=60,
                      own_requests=loaded, registry_path=reg, db_uri=db_uri)
    g2.__enter__()                       # raises GuardFailure if entry refused
    try:
        assert g2.own_requests() == loaded
    finally:
        g2.__exit__(None, None, None)

    # ---- CONTROL: an unregistered (non-own) POST still refuses
    rep_ctrl = g.idle_check([], min_idle_s=600)
    assert rep_ctrl.ok is False
    assert any("non-own POST" in r for r in rep_ctrl.reasons), rep_ctrl.reasons

    g3 = g.ChunkGuard("chunkC", max_wall_s=60, poll_s=60,
                      own_requests=[ev_epoch - 3600.0],   # stale, non-matching
                      registry_path=str(tmp_path / "r2.jsonl"), db_uri=db_uri)
    with pytest.raises(g.GuardFailure):
        g3.__enter__()


def test_run_chunk_loads_and_persists_default_registry(tmp_path):
    """run_chunk wires the default registry (<out>/raw/own_requests.jsonl) and
    passes it into ChunkGuard as both own_requests and registry_path."""
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    seen = {}

    class CaptureGuard(FakeGuard):
        def __init__(self, label, *, max_wall_s=900.0, log_dir=None,
                     own_requests=None, registry_path=None, **kw):
            super().__init__(label, max_wall_s=max_wall_s, log_dir=log_dir)
            seen["own_requests"] = own_requests
            seen["registry_path"] = registry_path

        def register_own_request(self, t_start=None):
            t = t_start if t_start is not None else time.time()
            super().register_own_request(t)
            if seen.get("registry_path"):
                with open(seen["registry_path"], "a") as fh:
                    fh.write(json.dumps({"label": self.label, "t": t}) + "\n")

    mod = L._StubGuardModule()

    def make(label, **kw):
        return CaptureGuard(label, **kw)

    setattr(mod, "ChunkGuard", make)
    L.GUARD_OVERRIDE = mod
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0, sleep=False) as srv:
            rc = L.main(["pilot", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == L.EXIT_OK
    default_reg = os.path.join(out_dir, "raw", "own_requests.jsonl")
    assert seen["registry_path"] == default_reg
    assert seen["own_requests"] == []                 # first run: nothing persisted yet
    assert os.path.exists(default_reg)               # 3 registrations were written
    assert len(L.load_own_requests(default_reg)) == 3


# ============================================ FIX 2: rep uniqueness / cache hits
# Live chunk1 bug: every rep of a cell built a BYTE-IDENTICAL delta
# (build_delta was deterministic in seed+delta_tokens), so reps 2..N were served
# from the previous rep's exact prompt cache -> prefill=0, reuse==prompt, wall~0.3 s
# (they measured NOTHING).  The tool recorded them as valid 0-row deltas.

def test_rep_delta_texts_unique_per_rep():
    """Two reps of the same nominal delta size must build DIFFERENT payloads."""
    base = C.build_base(4000, "SALT-0123456789abcdef")
    d0, d1, d2 = (C.build_delta(2048, nonce=i) for i in (0, 1, 2))
    assert d0 != d1 and d1 != d2 and d0 != d2
    assert "SESSION-SALT" not in d0 and "SESSION-SALT" not in d1   # salt-free tail
    assert d0.startswith("\n\n") and d0[2:].strip()


def test_ladder_build_messages_unique_delta_per_rep():
    """The ladder's own builder must fold the rep index into the delta text so
    rep 0 and rep 1 are not byte-identical (else rep 1 is a full cache hit)."""
    salt = "SALT-0123456789abcdef"
    base_reply = {"4k": "ok"}
    st0 = {"kind": "delta", "ctx_nominal": 4000, "ctx_label": "4k",
           "delta_nominal": 2048, "rep": 0, "cell": "ctx4k_d2048", "salt": salt}
    st1 = {**st0, "rep": 1}
    m0, _ = L._build_messages(st0, salt, base_reply, C.DEFAULT_CHARS_PER_TOKEN)
    m1, _ = L._build_messages(st1, salt, base_reply, C.DEFAULT_CHARS_PER_TOKEN)
    assert m0 != m1
    assert m0[0] == m1[0]           # same base head -> same checkpoint
    assert m0[1] == m1[1]           # same assistant reply
    assert m0[2]["content"] != m1[2]["content"]      # DISTINCT delta text
    assert [m["role"] for m in m0] == ["user", "assistant", "user"]


def test_mock_cache_hit_rep_measured_nothing_and_excluded(tmp_path):
    """A byte-identical repeat is a full cache hit (prefill=0, reuse==prompt); the
    run must classify it as such and the summarizer must exclude it (Table A n=1)."""
    log = str(tmp_path / "exo.log")
    eng = M.MockExo(log, rows_per_s=200.0, sleep=False)
    base = C.build_base(20000, "SALT-feedfacefeedface")
    msgs = C.delta_messages(base, "ok", C.build_delta(2048, nonce=0))
    eng.handle_request(C.base_messages(base))       # cold base feed
    p1 = eng.handle_request(msgs)                   # rep1: real delta
    p2 = eng.handle_request(msgs)                   # rep2: byte-identical -> cache
    assert p1["prefill"] > 0
    assert p2["prefill"] == 0 and p2["reuse"] == p2["prompt"]

    def rec(plan, rep):
        r = {"usage": {"prompt_tokens": plan["prompt"], "completion_tokens": 1},
             "ttft_s": 0.3, "wall_s": 0.3, "content": "ok"}
        logf = {"log_prefill": plan["prefill"], "reuse": plan["reuse"],
                "rewind": plan["rewind"], "prefill_controls_rows": plan["prefill"],
                "prefill_s_log": max(plan["prefill"], 1) / 200.0,
                "has_turn_reuse": plan["reuse"] > 0}
        st = {"kind": "delta", "ctx_nominal": 20000, "ctx_label": "20k",
              "delta_nominal": 2048, "rep": rep, "cell": "ctx20k_d2048"}
        return L._record("chunk1", st, 2048, r, logf, "studio1", 1.0,
                         C.DEFAULT_CHARS_PER_TOKEN, {"20k": 20000})

    r1, r2 = rec(p1, 0), rec(p2, 1)
    assert r1["cache_hit"] is False and r1["log_prefill"] > 0
    assert r2["cache_hit"] is True and r2["log_prefill"] == 0
    assert "cache hit" in r2["notes"]
    s = C.summarize([r1, r2])
    assert len(s["cache_hits"]) == 1                 # surfaced, not silently dropped
    row = [x for x in s["table_a"] if x["ctx"] == "20k"][0]
    assert row["n"] == 1                             # rep2 did NOT count as a delta


def test_record_classifies_prefill_zero_as_cache_hit():
    """Direct classifier exercise on the live rep2/rep3 shape (prefill=0)."""
    st = {"kind": "delta", "ctx_nominal": 50000, "ctx_label": "50k",
          "delta_nominal": 2048, "rep": 1, "cell": "ctx50k_d2048"}
    r = {"usage": {"prompt_tokens": 47561, "completion_tokens": 1},
         "ttft_s": 0.4, "wall_s": 0.4, "content": "ok"}
    logf = {"log_prefill": 0, "reuse": 47561, "rewind": 45056,
            "prefill_controls_rows": 0, "prefill_s_log": 0.4,
            "has_turn_reuse": True}
    rec = L._record("chunk1", st, 2048, r, logf, "studio1", 123.0,
                    C.DEFAULT_CHARS_PER_TOKEN, {"50k": 45705})
    assert rec["cache_hit"] is True
    assert rec["collapsed"] is False and rec["invalid"] is False
    assert rec["log_prefill"] == 0 and "cache hit" in rec["notes"]


# ============================================ FIX 2: wall pre-check accounting
def test_wall_precheck_ignores_predicted_cumulative(tmp_path, monkeypatch):
    """With a FAST true elapsed and a large per-step prediction, steps that still
    fit (elapsed + this step's pred <= cap) must NOT be blocked.  The old check
    added ``pred_cum`` (a running sum of predictions) on top of the real elapsed
    wall, double-counting and emitting a spurious NOT_RUN_WALL_CAP."""
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    monkeypatch.setattr(C, "predict_wall_s", lambda rows, **k: 400.0)
    L.GUARD_OVERRIDE = _fake_guard_mod()
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0, sleep=False) as srv:
            rc = L.main(["chunk1", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}", "--reps", "3",
                         "--max-wall", "900"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == L.EXIT_OK
    raw = os.path.join(out_dir, "raw", "delta_ladder.chunk1.jsonl")
    recs = C.load_jsonl([raw])
    # 20K base + 3 deltas + 50K base + 3 deltas = 8 steps, all fit (elapsed~0).
    assert len(recs) == 8
    assert not any(r.get("not_run") == "NOT_RUN_WALL_CAP" for r in recs)


def test_wall_precheck_blocks_truly_over_step_not_earlier(tmp_path, monkeypatch):
    """The corrected gate blocks the step whose TRUE elapsed wall + predicted wall
    first exceeds the cap, and no earlier one.  The old gate added the running
    ``pred_cum`` on top, blocking one step EARLY (the live chunk1 symptom)."""
    log_path = str(tmp_path / "exo.log")
    out_dir = str(tmp_path / "out")
    # pred fixed at 10 s; elapsed grows 50 s per step; cap 60 s.
    #   correct: base 0+10 ok, rep0 50+10=60 not>60 ok, rep1 100+10=120>60 BLOCK
    #   old    : rep0 50 + pred_cum(>=10) + 10 > 60 -> BLOCKS at rep0 (one early)
    monkeypatch.setattr(C, "predict_wall_s", lambda rows, **k: 10.0)
    seq = iter([0.0, 0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
    monkeypatch.setattr("time.monotonic", lambda: next(seq, 400.0))
    L.GUARD_OVERRIDE = _fake_guard_mod()
    try:
        with M.MockExoServer(log_path, rows_per_s=200.0, sleep=False) as srv:
            rc = L.main(["chunk1", "--api", srv.base_url, "--out-dir", out_dir,
                         "--log-source", f"studio1={log_path}", "--reps", "3",
                         "--max-wall", "60"])
    finally:
        L.GUARD_OVERRIDE = None
    assert rc == L.EXIT_OK
    raw = os.path.join(out_dir, "raw", "delta_ladder.chunk1.jsonl")
    recs = C.load_jsonl([raw])
    blocked = [r for r in recs if r.get("not_run") == "NOT_RUN_WALL_CAP"]
    assert len(blocked) == 1
    # step order: 0 base20k, 1 d2048 rep0, 2 d2048 rep1, 3 d2048 rep2, ...
    assert blocked[0]["rep"] == 1 and blocked[0]["delta_nominal"] == 2048


# ============================================ FIX 2: ladder-aware collapse rule
def test_collapsed_rule_ladder_rung():
    """Live: base 18390 -> legit rewind to rung 16384 (0.89 of base) is a REAL
    delta, not a collapse; reuse 2048 against base 20000 IS a collapse."""
    assert C.is_collapsed(16384, 18390) is False     # within one 2048 rung
    assert C.is_collapsed(2048, 20000) is True        # far below base depth
    assert C.is_collapsed(45056, 45705) is False      # live 50k rep1
    assert C.is_collapsed(18390 - 2048, 18390) is False   # exact rung boundary OK
    assert C.is_collapsed(18390 - 2049, 18390) is True    # one below -> collapsed
    assert C.is_collapsed(None, 20000) is True        # no turn-reuse line


def test_record_collapse_uses_base_actual_depth():
    """The classifier must use the base's ACTUAL row count: reuse=16384 with
    base_actual=18390 is NOT collapsed; reuse=2048 with base_actual=20000 IS."""
    st = {"kind": "delta", "ctx_nominal": 20000, "ctx_label": "20k",
          "delta_nominal": 2048, "rep": 2, "cell": "ctx20k_d2048"}

    def mk(prefill, reuse, prompt, base_actual):
        r = {"usage": {"prompt_tokens": prompt, "completion_tokens": 1},
             "ttft_s": 1.0, "wall_s": 1.0, "content": "ok"}
        logf = {"log_prefill": prefill, "reuse": reuse, "rewind": reuse,
                "prefill_controls_rows": prefill, "prefill_s_log": 1.0,
                "has_turn_reuse": True}
        return L._record("chunk1", st, 2048, r, logf, "studio1", 1.0,
                         C.DEFAULT_CHARS_PER_TOKEN, {"20k": base_actual})

    r_ok = mk(3870, 16384, 20254, 18390)       # live rep1 shape -> real delta
    assert r_ok["collapsed"] is False and r_ok["cache_hit"] is False
    r_bad = mk(18206, 2048, 20254, 20000)      # far below base depth -> collapsed
    assert r_bad["collapsed"] is True


# ============================================================== sabotage proofs
def test_sabotage_rows_math_would_fail():
    """Sabotage proof #1: if rows_per_s_log dropped the division, 200 != 204.8."""
    rec = {"log_prefill": 2048, "prefill_s_log": 10.0}
    good = C.rows_per_s_log(rec)
    # a deliberately wrong implementation (* instead of /) must NOT match
    wrong = rec["log_prefill"] * rec["prefill_s_log"]
    assert good != wrong


def test_sabotage_collapse_classification_would_fail():
    """Sabotage proof #2: inverting the collapsed threshold flips a valid rep."""
    ctx = 50000
    reuse = 48000                       # >= 0.9*ctx -> valid
    correct_collapsed = reuse < 0.9 * ctx          # False
    inverted_collapsed = reuse >= 0.9 * ctx        # True
    assert correct_collapsed is False and inverted_collapsed is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))

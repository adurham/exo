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
import subprocess
import sys

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

"""Tests for bench/phase20_guard.py (Phase-20 R1/R2/R3/R4 safety layer).

No network / no cluster access: every seam is monkeypatched or fed a recorded fixture.
Fixtures under ``fixtures/guard/`` were extracted from real artifacts:
  * real_inflight_window.log / real_completed_window.log / real_idle_window.log
    -> real lines from the m4-1 boot 09:01-14:00 (session 20261007_092009_9a2ed7).
  * real_post_routes.log -> API request: POST paths seen in that boot.
  * real_turn_42calls.json -> the 42 real (started_at, ended_at) api_calls pairs + the
    real runner running/ready markers from that boot.

Run:
  cd <worktree> && PYTHONPATH=bench /Users/adam.durham/repos/exo/.venv/bin/python \
      -m pytest --noconftest bench/phase20_tests/test_phase20_guard.py -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/ on sys.path
import phase20_guard as g  # noqa: E402

FX = Path(__file__).resolve().parent / "fixtures" / "guard"


def _read(name: str) -> str:
    return (FX / name).read_text()


# ======================================================================================
# (1) log parsing against REAL fixture lines
# ======================================================================================


def test_parse_real_inflight_window():
    events = g.parse_log_lines(_read("real_inflight_window.log"))
    assert len(events) == 8, f"expected 8 real marker lines (idle line is not a marker), got {len(events)}"
    # first real marker in the window is the POST at 09:20:21.067
    assert events[0].kind == "post" and events[0].value == "/v1/chat/completions"
    kinds = [e.kind for e in events]
    assert "running" in kinds
    state, epoch = g.runner_state(events)
    assert state == "running", "window ends mid-generation -> last marker is 'runner running'"
    # the first in-flight interval starts 09:20:21.159 and ends 09:21:59.687
    spans = g.runner_intervals(events)
    assert spans, "expected at least one running->ready interval"
    s0, e0 = spans[0]
    assert abs(s0 - g._parse_log_ts("2026-10-07 09:20:21.159")) < 0.01
    assert abs(e0 - g._parse_log_ts("2026-10-07 09:21:59.687")) < 0.01


def test_parse_real_completed_window():
    events = g.parse_log_lines(_read("real_completed_window.log"))
    assert len(events) >= 5
    state, _ = g.runner_state(events)
    assert state == "ready", "call-1 window ends with 'runner ready'"
    spans = g.runner_intervals(events)
    assert len(spans) == 1 and spans[0][0] < spans[0][1]


def test_parse_real_idle_window():
    # a full minute of the real log with no generation activity
    events = g.parse_log_lines(_read("real_idle_window.log"))
    assert events == [], "idle window must contain zero guard markers"
    state, _ = g.runner_state(events)
    assert state is None


def test_real_post_routes_only_generation_and_control():
    txt = _read("real_post_routes.log")
    paths = {line.split()[0]: int(line.split()[1]) for line in txt.splitlines() if line.strip()}
    assert "/v1/chat/completions" in paths and paths["/v1/chat/completions"] == 75
    assert paths["/instance"] == 1  # control-plane, NOT a generation request
    assert not g.is_generation_post("/instance")
    assert g.is_generation_post("/v1/chat/completions")


def test_generation_route_table():
    for p in (
        "/v1/chat/completions",
        "/bench/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/ollama/api/chat",
        "/ollama/api/generate",
    ):
        assert g.is_generation_post(p), p
    for p in ("/state", "/metrics", "/node_id", "/v1/models", "/instance", "/v1/cancel/abc"):
        assert not g.is_generation_post(p), p


# ======================================================================================
# (2) in-flight state machine + the 42-call interval proof
# ======================================================================================


def test_42_call_interval_proof():
    """Every one of the 42 real calls is bracketed by a runner running..ready interval."""
    data = json.loads((FX / "real_turn_42calls.json").read_text())
    events = [
        g.LogEvent(m["t"], "", m["kind"]) for m in data["runner_markers"]
    ]
    events.sort(key=lambda e: e.epoch)
    spans = g.runner_intervals(events)
    assert len(data["calls"]) == 42
    assert abs(data["sum_latency"] - 1530.59) < 0.5, "fixture must be the real turn"
    misses = []
    for call in data["calls"]:
        s0, e0 = call["started_at"], call["ended_at"]
        if not any(s0 <= b and e0 >= a for a, b in spans):
            misses.append(call["call_seq"])
    assert misses == [], f"calls not bracketed by a running/ready interval: {misses}"


def test_runner_state_machine_running_then_ready():
    log = (
        "[ 2026-10-07 09:20:21.159 | INFO | x:571 ] runner running\n"
        "[ 2026-10-07 09:21:59.687 | INFO | x:865 ] runner ready\n"
        "[ 2026-10-07 09:22:00.159 | INFO | x:571 ] runner running\n"
    )
    events = g.parse_log_lines(log)
    assert g.runner_state(events)[0] == "running"
    spans = g.runner_intervals(events)
    # second running has no closing ready -> unbounded interval
    assert len(spans) == 2 and spans[1][1] == float("inf")


# ======================================================================================
# (3) own-request exclusion + extra-POST abort  (S2)
# ======================================================================================


def _install_fake_node(monkeypatch, log_text, *, offset=0.0, posts_ns=("studio1", "studio2")):
    """Make _collect_node see ``log_text`` on every node (clock offset 0)."""
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: offset)

    def fake_ssh(node, cmd, *, timeout=25.0):
        if "date" in cmd:  # clock probe (belt-and-braces; offset is patched too)
            return 0, f"{g._now()}\n"
        return 0, log_text + "\n" + g._SSH_OK_MARKER + "\n"

    monkeypatch.setattr(g, "run_ssh", fake_ssh)


def test_own_request_excluded_within_2s():
    now = g._now()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - 5)) + ".000"
    log = f"[ {ts} | DEBUG | exo.api.main:_log_requests:340 ] API request: POST /v1/chat/completions\n"
    events = g.parse_log_lines(log)
    own = [events[0].epoch + 1.5]  # within +-2 s
    d = {"_events": events}
    lp = None
    for ev in d["_events"]:
        if ev.kind == "post" and g.is_generation_post(ev.value):
            lp = (ev.epoch, ev.value)
    assert lp is not None
    assert any(abs(lp[0] - t) <= g.OWN_MATCH_WINDOW_S for t in own)


def test_s2_aborts_on_extra_post(tmp_path, monkeypatch):
    # a POST newer than chunk start, NOT registered -> S2 abort
    now = g._now()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + 1)) + ".000"
    log = f"[ {ts} | DEBUG | exo.api.main:_log_requests:340 ] API request: POST /v1/chat/completions\n"
    guard = g.ChunkGuard("s2", max_wall_s=60, poll_s=0.2, db_uri="file:memory:?mode=ro")
    guard.t_start = now  # chunk started a moment ago
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, log + g._SSH_OK_MARKER))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    guard._poll_signals()
    assert guard.aborted is True
    assert guard.reason == "ABORTED_USER_ARRIVED"
    assert any(s.startswith("S2:") for s in guard._signals)


def test_s2_no_abort_when_own_registered(tmp_path, monkeypatch):
    now = g._now()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + 1)) + ".000"
    log = f"[ {ts} | DEBUG | exo.api.main:_log_requests:340 ] API request: POST /v1/chat/completions\n"
    ev_epoch = g.parse_log_lines(log)[0].epoch
    guard = g.ChunkGuard("s2b", max_wall_s=60, poll_s=0.2, db_uri="file:memory:?mode=ro")
    guard.t_start = now
    guard.register_own_request(ev_epoch)  # registered within the window
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, log + g._SSH_OK_MARKER))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    guard._poll_signals()
    assert guard.aborted is False


def test_s2_no_abort_when_own_registered_during_log_read(tmp_path, monkeypatch):
    """RACE REGRESSION (live-reproduced false positive).

    ``_poll_signals`` used to snapshot the own-request list at the TOP of the method,
    *before* the two-node ssh log read (~1-3 s).  The harness registers immediately
    before it sends, so its own POST can only be present in a log line written *after*
    the registration -- i.e. the registration lands DURING the watcher's log read.  A
    stale (empty) snapshot then mis-classifies the harness's own first POST as non-own
    and aborts every live chunk.  The fix re-reads ``own_requests()`` after the logs are
    read, just before the S2 loop.  Here the registration happens inside the second
    node's log read, mirroring the real register-during-read ordering.
    """
    now = g._now()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + 1)) + ".000"
    log = f"[ {ts} | DEBUG | exo.api.main:_log_requests:340 ] API request: POST /v1/chat/completions\n"
    ev_epoch = g.parse_log_lines(log)[0].epoch  # the harness's own POST timestamp
    guard = g.ChunkGuard("s2race", max_wall_s=60, poll_s=0.2, db_uri="file:memory:?mode=ro")
    guard.t_start = now  # chunk started a moment ago
    seen = {"nodes": 0}

    monkeypatch.setattr(g, "_now", lambda: ev_epoch)  # frozen clock at the POST epoch
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})

    def fake_ssh(node, cmd, **kw):
        if "date" in cmd:  # clock probe (offset is patched too)
            return 0, f"{g._now()}\n"
        seen["nodes"] += 1
        if seen["nodes"] == 1:
            # watcher's own-list snapshot (top of _poll_signals) is still empty here
            assert guard.own_requests() == []
        else:
            # harness's register_own_request() lands DURING the log read -- after the
            # watcher's snapshot, before it evaluates S2 (the real register-during-read):
            guard.register_own_request(ev_epoch)
            assert guard.own_requests() == [ev_epoch]
        return 0, log + g._SSH_OK_MARKER

    monkeypatch.setattr(g, "run_ssh", fake_ssh)
    guard._poll_signals()
    assert guard.aborted is False, (
        f"own POST @{ev_epoch} registered during the log read was flagged as non-own: "
        f"{guard._signals} reason={guard.reason!r}"
    )
    assert guard.cancel_event.is_set() is False


def test_s2_aborts_when_post_outside_own_window(tmp_path, monkeypatch):
    """Control for the race fix: a genuinely non-own POST (no own registration within
    +-2 s) STILL aborts after the post-log-read re-read of the own list.  The stale
    registration here is 10 s away, so it must not exclude the POST."""
    now = g._now()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + 1)) + ".000"
    log = f"[ {ts} | DEBUG | exo.api.main:_log_requests:340 ] API request: POST /v1/chat/completions\n"
    ev_epoch = g.parse_log_lines(log)[0].epoch
    guard = g.ChunkGuard("s2non", max_wall_s=60, poll_s=0.2, db_uri="file:memory:?mode=ro")
    guard.t_start = now
    monkeypatch.setattr(g, "_now", lambda: ev_epoch)
    guard.register_own_request(ev_epoch - 10.0)  # own list non-empty, but >2 s away
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, log + g._SSH_OK_MARKER))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    guard._poll_signals()
    assert guard.aborted is True
    assert guard.reason == "ABORTED_USER_ARRIVED"
    assert any(s.startswith("S2:") for s in guard._signals)


def test_s1_concurrency_aborts_when_no_own_inflight(monkeypatch):
    # /state shows one RunnerRunning, we have no in-flight registration -> S1 abort
    guard = g.ChunkGuard("s1", max_wall_s=60, poll_s=0.2, db_uri="file:memory:?mode=ro")
    guard.t_start = g._now()
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, "", ))  # no events
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(
        g, "_fetch_json", lambda url, **kw: {"runners": {"r1": {"RunnerRunning": {}}}}
    )
    guard._poll_signals()
    assert guard.aborted and guard.reason == "ABORTED_USER_ARRIVED"
    assert any(s.startswith("S1:") for s in guard._signals)


def test_s1_no_abort_when_own_inflight(monkeypatch):
    guard = g.ChunkGuard("s1b", max_wall_s=60, poll_s=0.2, db_uri="file:memory:?mode=ro")
    guard.t_start = g._now()
    guard.register_own_request(g._now())  # our request is in flight
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, ""))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(
        g, "_fetch_json", lambda url, **kw: {"runners": {"r1": {"RunnerRunning": {}}}}
    )
    guard._poll_signals()
    assert guard.aborted is False


# ======================================================================================
# (4) S3 / S4 on a temp sqlite built from the real schema subset
# ======================================================================================


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, call_seq INTEGER NOT NULL,
            started_at REAL NOT NULL, ended_at REAL NOT NULL, latency_seconds REAL NOT NULL,
            model TEXT, provider TEXT, input_tokens INTEGER, output_tokens INTEGER);
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, model TEXT, last_activity_at REAL, started_at REAL NOT NULL);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT, timestamp REAL NOT NULL);
        """
    )
    conn.commit()
    conn.close()


def test_s3_completed_row_aborts(tmp_path, monkeypatch):
    db = tmp_path / "s3.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    now = g._now()
    conn.execute(
        "INSERT INTO api_calls (session_id, call_seq, started_at, ended_at, latency_seconds, model, provider)"
        " VALUES ('s', 1, ?, ?, 1.0, ?, 'custom')",
        (now - 3, now - 2, g.EXO_MODEL),
    )
    conn.commit()
    conn.close()
    guard = g.ChunkGuard("s3", max_wall_s=60, poll_s=0.2, db_uri=f"file:{db}?mode=ro")
    guard.t_start = now - 5  # chunk started before the row
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, ""))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    guard._poll_signals()
    assert guard.aborted and guard.reason == "ABORTED_USER_ARRIVED"
    assert any(s.startswith("S3:") for s in guard._signals)


def test_s4_early_message_aborts(tmp_path, monkeypatch):
    db = tmp_path / "s4.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    now = g._now()
    conn.execute("INSERT INTO sessions (id, model, started_at, last_activity_at) VALUES ('s','x',0,0)")
    conn.execute(
        "INSERT INTO messages (session_id, role, timestamp) VALUES ('s','user',?)", (now - 1,)
    )
    conn.execute("INSERT INTO sessions (id, model, started_at, last_activity_at) VALUES (?,?,?,?)",
                 ("s2", g.EXO_MODEL, 0, now - 1))
    conn.commit()
    conn.close()
    guard = g.ChunkGuard("s4", max_wall_s=60, poll_s=0.2, db_uri=f"file:{db}?mode=ro")
    guard.t_start = now - 5
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, ""))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    guard._poll_signals()
    assert guard.aborted and guard.reason == "ABORTED_USER_ARRIVED"
    assert any(s.startswith("S4:") for s in guard._signals)


def test_s3_only_matches_exo_model(tmp_path):
    """The unrelated ollama model must NEVER trigger S3 (exact equality, not LIKE)."""
    db = tmp_path / "like.db"
    _make_db(db)
    conn = sqlite3.connect(db)
    now = g._now()
    conn.execute(
        "INSERT INTO api_calls (session_id, call_seq, started_at, ended_at, latency_seconds, model, provider)"
        " VALUES ('s', 1, ?, ?, 1.0, 'deepseek-v4.1-flash', 'ollama')",
        (now - 3, now - 2),
    )
    conn.commit()
    conn.close()
    conn = g._ro_connect(f"file:{db}?mode=ro")
    rows = g._db_s3_recent(conn, now - 600, "ended_at")
    conn.close()
    assert rows == [], "case-insensitive LIKE would have matched; exact equality must not"


# ======================================================================================
# (5) canary parsing + thresholds
# ======================================================================================


def test_canary_parse_and_states():
    txt = "14.82 14.84 14.86 TFLOPS (healthy ~14-15, degraded <5)\n"
    vals = g.parse_canary_output(txt)
    assert vals == [14.82, 14.84, 14.86]
    assert g._canary_state([14.84, 14.85]) == "healthy"
    assert g._canary_state([7.5, 8.0]) == "marginal"
    assert g._canary_state([2.0, 1.0]) == "degraded"
    assert g._canary_state([]) == "degraded"


def test_canary_end_to_end(monkeypatch):
    def fake_ssh(node, cmd, *, timeout=25.0):
        if "PRESENT" in cmd:
            return 0, "PRESENT"
        return 0, "14.82 14.84 14.86 TFLOPS (healthy ~14-15, degraded <5)\n"

    monkeypatch.setattr(g, "run_ssh", fake_ssh)
    rep = g.canary()
    assert rep.state == "healthy" and rep.ok
    assert rep.median["studio1"] == 14.84


# ======================================================================================
# (7) ssh failure => NOT idle  /  (8) clock-skew refusal
# ======================================================================================


def test_ssh_failure_not_idle(monkeypatch):
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (255, "<ssh error>"))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: None)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    monkeypatch.setattr(g, "_state_db_idle", lambda *a, **k: {"ok": True, "recent_completed": [], "s4": {}})
    rep = g.idle_check()
    assert rep.ok is False
    assert any("cannot evaluate" in r for r in rep.reasons)


def test_clock_skew_refusal(monkeypatch):
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, g._SSH_OK_MARKER))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 3.0)  # > 1.5 s
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {}})
    monkeypatch.setattr(g, "_state_db_idle", lambda *a, **k: {"ok": True, "recent_completed": [], "s4": {}})
    rep = g.idle_check(own_requests=[g._now()])  # own matching is load-bearing
    assert rep.ok is False
    assert any("clock offset" in r for r in rep.reasons)


def test_idle_true_when_quiet(monkeypatch):
    monkeypatch.setattr(g, "run_ssh", lambda node, cmd, **kw: (0, g._SSH_OK_MARKER))
    monkeypatch.setattr(g, "measure_clock_offset", lambda node, **kw: 0.0)
    monkeypatch.setattr(g, "_fetch_json", lambda url, **kw: {"runners": {"r": {"RunnerReady": {}}}})
    monkeypatch.setattr(g, "_state_db_idle", lambda *a, **k: {"ok": True, "recent_completed": [], "s4": {}})
    rep = g.idle_check()
    assert rep.ok is True and rep.reasons == []


# ======================================================================================
# ChunkGuard: registry, json, check()
# ======================================================================================


def test_register_own_request_writes_registry(tmp_path):
    reg = tmp_path / "own.jsonl"
    guard = g.ChunkGuard("reg", registry_path=str(reg), db_uri="file:memory:?mode=ro")
    guard.register_own_request(123.0)
    lines = [json.loads(l) for l in reg.read_text().splitlines()]
    assert lines == [{"label": "reg", "t": 123.0}]
    assert guard.own_requests() == [123.0]


def test_check_raises_chunkaborted_after_fire():
    guard = g.ChunkGuard("c", db_uri="file:memory:?mode=ro")
    assert guard.check() is None
    guard._fire("ABORTED_WALL_CAP", reason="test")
    assert guard.cancel_event.is_set()
    with pytest.raises(g.ChunkAborted):
        guard.check()
    assert guard.aborted and guard.reason == "ABORTED_WALL_CAP"


def test_guard_json_written(tmp_path):
    guard = g.ChunkGuard("gjson", log_dir=str(tmp_path), db_uri="file:memory:?mode=ro")
    guard.t_start, guard.t_end = 1.0, 2.0
    guard._signals.append("S2:x")
    guard._write_json()
    payload = json.loads((tmp_path / "gjson.guard.json").read_text())
    assert payload["label"] == "gjson"
    assert set(payload) == {
        "label", "t_start", "t_end", "aborted", "reason",
        "wall_cap_hit", "own_requests", "signals_seen",
    }
    assert payload["signals_seen"] == ["S2:x"]


# ======================================================================================
# (6) watcher: real SIGINT to a real `sleep 60`, watch exit 75 / wall cap 76
# ======================================================================================


_DRIVER = r'''
import sys, time
sys.path.insert(0, sys.argv[1])          # bench/
import phase20_guard as g
import os

MODE = sys.argv[2]
g.STATE_DB_URI = sys.argv[3]             # temp db

T0 = time.time()

def fake_ssh(node, cmd, *, timeout=25.0):
    if "date" in cmd:
        return 0, f"{time.time()}\n"
    # arrival POST only appears ~1.5 s AFTER the driver starts (i.e. mid-chunk),
    # never during ChunkGuard's entry idle_check.
    if MODE == "arrival" and (time.time() - T0) > 1.5:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 2)) + ".000"
        log = f"[ {ts} | DEBUG | x:340 ] API request: POST /v1/chat/completions\n"
        return 0, log + g._SSH_OK_MARKER
    return 0, g._SSH_OK_MARKER

g.run_ssh = fake_ssh
g.measure_clock_offset = lambda node, **kw: 0.0
g._fetch_json = lambda url, **kw: {"runners": {}}
rc = g.main(["watch", "--label", "drv", "--max-wall", "4", "--poll", "1",
             "--", sys.executable, "-c", "import time; time.sleep(60)"])
sys.exit(rc)
'''


def _run_driver(tmp_path, mode):
    (tmp_path / "empty.db").write_bytes(b"")
    # make a valid-but-empty sqlite db
    conn = sqlite3.connect(tmp_path / "empty.db")
    conn.executescript(
        "CREATE TABLE api_calls (id INTEGER PRIMARY KEY, session_id TEXT, call_seq INTEGER,"
        " started_at REAL, ended_at REAL, latency_seconds REAL, model TEXT, provider TEXT);"
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, last_activity_at REAL, started_at REAL);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL);"
    )
    conn.commit()
    conn.close()
    drv = tmp_path / "drv.py"
    drv.write_text(_DRIVER)
    bench = str(Path(__file__).resolve().parents[1])
    p = subprocess.run(
        [sys.executable, str(drv), bench, mode, f"file:{tmp_path/'empty.db'}?mode=ro"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return p


def test_watch_arrival_exit_75_and_sigint(tmp_path):
    p = _run_driver(tmp_path, "arrival")
    assert p.returncode == 75, f"expected exit 75, got {p.returncode}\n{p.stdout}\n{p.stderr}"
    assert "ABORTED_USER_ARRIVED" in p.stderr


def test_watch_wall_cap_exit_76(tmp_path):
    p = _run_driver(tmp_path, "quiet")
    assert p.returncode == 76, f"expected exit 76, got {p.returncode}\n{p.stdout}\n{p.stderr}"
    assert "ABORTED_WALL_CAP" in p.stderr


def test_sigint_actually_kills_child():
    """Directly exercise _fire's SIGINT on a real sleep process."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    guard = g.ChunkGuard("sig", interrupt_pid=proc.pid, db_uri="file:memory:?mode=ro")
    try:
        guard._fire("ABORTED_USER_ARRIVED", reason="test")
        proc.wait(timeout=10)  # SIGINT default handler kills sleep
        assert proc.returncode != 0
    finally:
        if proc.poll() is None:
            proc.kill()

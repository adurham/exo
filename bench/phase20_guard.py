"""bench/phase20_guard.py -- Phase-20 safety layer (R1 idle guard, R4 canary, R2/R3 chunk guard).

Implements the frozen API in ``docs/benchmarks/phase20-throughput/GUARD-CONTRACT.md``
verbatim (names, signatures, exit codes).  Pure stdlib + ``ssh``/``scp`` subprocesses +
read-only sqlite.  This module NEVER sends a generation request, never starts/stops a
process on a node, and never writes to state.db.

Design notes / rationale live in ``docs/benchmarks/phase20-throughput/GUARD-NOTES.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Constants (frozen by GUARD-CONTRACT.md)
# --------------------------------------------------------------------------------------

NODES: dict[str, str] = {"studio1": "m4-1", "studio2": "m4-2"}  # ssh aliases -> short tags
API_BASE = "http://192.168.86.48:52415"  # cluster API (studio1 is master/API node)
STATE_DB_URI = "file:/Users/adam.durham/.hermes/state.db?mode=ro"

EXO_MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
# The exo rows are provider='custom' AND model=EXO_MODEL (exact equality; SQLite LIKE is
# case-insensitive and would also match the unrelated ollama model deepseek-v4.1-flash).

CANARY_SRC = "/Users/adam.durham/.hermes/cache/scratch/gpu_canary2.py"
CANARY_DST = "/tmp/gpu_canary2.py"
NODE_VENV_PYTHON = "~/repos/exo/.venv/bin/python"

REMOTE_LOG = '"$HOME/.exo/exo_log/exo.log"'
# One constant for every log pattern the guard greps for.  Matched on MESSAGE TEXT only
# (loguru line numbers :571/:860/:865 vary across builds).
GREP_PATTERN = (
    r"API request: POST|runner running|runner ready|runner idle"
    r"|Starting task TextGeneration|Executing command: TaskFinished"
)
_LOG_SCAN_CMD = (
    "{ grep -a -E '" + GREP_PATTERN + "' " + REMOTE_LOG + r" 2>/dev/null || true; }"
    r" | tail -n 400; echo __EXO_LOG_OK__"
)
_SSH_OK_MARKER = "__EXO_LOG_OK__"

# Generation-triggering POST routes, enumerated from the SHARED checkout
# src/exo/api/main.py::_setup_routes (deploy/next13 @ f4bb14746).  GET /state, /metrics,
# /node_id, /v1/models, /models and dashboard polls are NOT requests.  The remaining
# POST routes (download/*, /instance, /place_instance, /v1/instance-links, /models/add,
# /v1/traces/delete, /onboarding, /v1/cancel/{id}, /ollama/api/show) are control-plane,
# not generation.
GENERATION_POST_ROUTES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/bench/chat/completions",
    "/v1/images/generations",
    "/bench/images/generations",
    "/v1/images/edits",
    "/bench/images/edits",
    "/v1/messages",
    "/v1/responses",
    "/v1/cancel/",  # not generation, but kept explicit; matched as prefix below
    "/ollama/v1/chat/completions",
    "/ollama/api/chat",
    "/ollama/api/api/chat",
    "/ollama/api/v1/chat",
    "/ollama/api/generate",
)
# /v1/cancel/ is a cancellation, NOT a generation request: handled separately.
_CANCEL_PREFIX = "/v1/cancel/"
_GENERATION_POST_SET = frozenset(
    p for p in GENERATION_POST_ROUTES if not p.startswith(_CANCEL_PREFIX)
)

OWN_MATCH_WINDOW_S = 2.0  # +-2 s after clock-offset correction
DEFAULT_MIN_IDLE_S = 600
CLOCK_SKEW_ABORT_S = 1.5  # |offset| > this => fail loudly
_EXIT_IDLE_OK, _EXIT_BUSY = 0, 1
_EXIT_CANARY = {"healthy": 0, "marginal": 1, "degraded": 2}
_EXIT_ABORTED_USER = 75
_EXIT_ABORTED_WALL = 76

# --------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------


class GuardFailure(RuntimeError):
    """Raised when the guard cannot certify a precondition (or times out)."""


class ChunkAborted(GuardFailure):
    """Raised by ChunkGuard.check() after an abort (user arrival or wall cap)."""


# --------------------------------------------------------------------------------------
# Injectable seams (tests monkeypatch these)
# --------------------------------------------------------------------------------------


def run_ssh(node: str, cmd: str, *, timeout: float = 25.0) -> tuple[int, str]:
    """Run a read-only command on ``node``; return ``(returncode, stdout+stderr)``.

    Seam for tests: monkeypatch ``phase20_guard.run_ssh``.
    """
    try:
        p = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:  # ssh itself failed
        return 255, f"<ssh error: {exc}>"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _run_scp(src: str, dst: str, node: str, *, timeout: float = 30.0) -> tuple[int, str]:
    """Copy a local file TO a node (only ever used for the canary script)."""
    try:
        p = subprocess.run(
            ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", src, f"{node}:{dst}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 255, f"<scp error: {exc}>"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _fetch_json(url: str, *, timeout: float = 10.0) -> dict:
    """GET a JSON endpoint (read-only).  stdlib urllib only."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https? no, http LAN)
        return json.loads(resp.read().decode("utf-8"))


def _now() -> float:
    """Laptop wall clock (UTC epoch seconds).  Seam for tests."""
    return time.time()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


# --------------------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------------------

_LOG_TS_RE = re.compile(r"^\[\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\|")
_POST_RE = re.compile(r"API request: POST (\S+)")
_RUNNING = "runner running"
_READY = "runner ready"


def _parse_log_ts(ts: str) -> float:
    """Node-local wall time (no tz marker) -> UTC epoch, assuming node tz == laptop tz.

    Both Studios run CDT, the same tz as the laptop, so a naive datetime parsed as
    laptop-local yields the node-clock epoch.  Clock skew is applied separately.
    """
    import datetime

    dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


@dataclass
class LogEvent:
    epoch: float  # UTC epoch, node clock, offset NOT yet applied
    ts: str  # raw node-local timestamp text
    kind: str  # "post" | "running" | "ready" | "start_task" | "task_finished"
    value: str = ""  # POST path, else ""


def parse_log_lines(text: str) -> list[LogEvent]:
    """Parse already-collected exo.log lines into ordered :class:`LogEvent`s."""
    out: list[LogEvent] = []
    for line in text.splitlines():
        m = _LOG_TS_RE.match(line)
        if not m:
            continue
        try:
            epoch = _parse_log_ts(m.group(1))
        except ValueError:
            continue
        body = line[m.end() :]
        pm = _POST_RE.search(body)
        if pm:
            out.append(LogEvent(epoch, m.group(1), "post", pm.group(1)))
            continue
        if _RUNNING in body:
            out.append(LogEvent(epoch, m.group(1), "running"))
        elif _READY in body:
            out.append(LogEvent(epoch, m.group(1), "ready"))
        elif "Starting task TextGeneration" in body:
            out.append(LogEvent(epoch, m.group(1), "start_task"))
        elif "Executing command: TaskFinished" in body:
            out.append(LogEvent(epoch, m.group(1), "task_finished"))
    out.sort(key=lambda e: e.epoch)
    return out


def runner_state(events: Sequence[LogEvent]) -> tuple[str | None, float | None]:
    """Return (state, epoch) where state is the last of ``running``/``ready``.

    state is None when neither was seen (no evidence of any generation on this node).
    """
    last_kind: str | None = None
    last_epoch: float | None = None
    for ev in events:
        if ev.kind in ("running", "ready"):
            last_kind, last_epoch = ev.kind, ev.epoch
    return last_kind, last_epoch


def runner_intervals(events: Sequence[LogEvent]) -> list[tuple[float, float]]:
    """Pair every ``runner running`` with the next ``runner ready`` -> in-flight spans."""
    spans: list[tuple[float, float]] = []
    open_start: float | None = None
    for ev in events:
        if ev.kind == "running":
            open_start = ev.epoch
        elif ev.kind == "ready" and open_start is not None:
            spans.append((open_start, ev.epoch))
            open_start = None
    if open_start is not None:
        spans.append((open_start, float("inf")))  # still running (unbounded)
    return spans


def is_generation_post(path: str) -> bool:
    if path.startswith(_CANCEL_PREFIX):
        return False
    return path in _GENERATION_POST_SET


# --------------------------------------------------------------------------------------
# Clock offset
# --------------------------------------------------------------------------------------


def measure_clock_offset(node: str, *, samples: int = 3) -> float | None:
    """Node clock minus laptop clock, RTT-midpoint corrected, min-RTT of ``samples``.

    Returns None when ssh fails (caller treats it as "cannot evaluate").
    """
    best_rtt = None
    best_off = None
    for _ in range(samples):
        t0 = _now()
        rc, out = run_ssh(node, "date +%s.%N", timeout=20)
        t1 = _now()
        if rc != 0:
            return None
        try:
            node_epoch = float(out.strip().splitlines()[0])
        except (ValueError, IndexError):
            return None
        rtt = t1 - t0
        mid = (t0 + t1) / 2.0
        if best_rtt is None or rtt < best_rtt:
            best_rtt, best_off = rtt, node_epoch - mid
    return best_off


def _apply_offset(node_epoch: float, offset: float | None) -> float:
    """Node clock epoch -> laptop epoch.  offset = node - laptop (positive => node ahead)."""
    return node_epoch - (offset or 0.0)


# --------------------------------------------------------------------------------------
# state.db (READ-ONLY)
# --------------------------------------------------------------------------------------


def _ro_connect(db_uri: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _db_s3_recent(conn: sqlite3.Connection, threshold: float, column: str = "ended_at") -> list[dict]:
    """api_calls rows (provider='custom', exact model) whose ``column`` > threshold."""
    assert column in ("ended_at", "started_at")
    rows = conn.execute(
        f"SELECT id, session_id, call_seq, started_at, ended_at FROM api_calls "
        f"WHERE provider='custom' AND model=? AND {column} > ? ORDER BY {column} DESC",
        (EXO_MODEL, threshold),
    ).fetchall()
    return [dict(r) for r in rows]


def _db_s4_recent(conn: sqlite3.Connection, threshold: float) -> dict:
    """Early session activity (S4): messages/sessions rows for exo-model sessions."""
    msgs = conn.execute(
        "SELECT m.id, m.session_id, m.role, m.timestamp FROM messages m "
        "JOIN sessions s ON m.session_id = s.id "
        "WHERE s.model=? AND m.timestamp > ? ORDER BY m.timestamp DESC LIMIT 5",
        (EXO_MODEL, threshold),
    ).fetchall()
    sess = conn.execute(
        "SELECT id, last_activity_at FROM sessions WHERE model=? AND last_activity_at > ? "
        "ORDER BY last_activity_at DESC LIMIT 5",
        (EXO_MODEL, threshold),
    ).fetchall()
    return {"messages": [dict(r) for r in msgs], "sessions": [dict(r) for r in sess]}


# --------------------------------------------------------------------------------------
# R1 -- idle check (PREREG D1)
# --------------------------------------------------------------------------------------


@dataclass
class IdleReport:
    ok: bool
    reasons: list[str]
    detail: dict


def _collect_node(node: str, timeout: float) -> dict:
    """Read one node's log + clock offset.  Returns a detail dict with ``ok`` gate."""
    d: dict = {"log_ok": False, "events": [], "state": None, "running": False,
               "post_routes": {}, "last_post": None, "clock_offset_s": None, "error": None}
    offset = measure_clock_offset(node)
    d["clock_offset_s"] = offset
    rc, out = run_ssh(node, _LOG_SCAN_CMD, timeout=timeout)
    if rc != 0 or _SSH_OK_MARKER not in out:
        d["error"] = f"ssh {node} failed (rc={rc})"
        return d
    text = out.replace(_SSH_OK_MARKER, "")
    events = parse_log_lines(text)
    d["log_ok"] = True
    d["_events"] = events
    d["n_events"] = len(events)
    d["events_tail"] = [
        {"ts": e.ts, "kind": e.kind, "value": e.value} for e in events[-40:]
    ]
    st, st_epoch = runner_state(events)
    d["state"] = st
    d["state_epoch"] = _apply_offset(st_epoch, offset) if st_epoch is not None else None
    d["running"] = st == "running"
    routes: dict[str, int] = {}
    last_post: tuple[float, str] | None = None
    for ev in events:
        if ev.kind == "post":
            routes[ev.value] = routes.get(ev.value, 0) + 1
            if is_generation_post(ev.value):
                le = _apply_offset(ev.epoch, offset)
                if last_post is None or le > last_post[0]:
                    last_post = (le, ev.value)
    d["post_routes"] = routes
    d["last_post"] = last_post
    return d


def _state_active_tasks(state: dict) -> int:
    """Count active cluster-wide ``TextGeneration`` TASKS in a ``GET /state`` payload.

    A generation request is a cluster-wide task: on a tensor-parallel deployment ONE
    request is broadcast to every rank and carries a single shared ``task_id``, so the
    number of active ``state['tasks']`` entries is the true in-flight request count.
    ``state['runners']`` is deliberately NOT consulted: on TP2 one request puts BOTH
    runners in ``RunnerRunning``, so counting runners double-counts a single request
    (live-reproduced S1 false positive).  Returns 0 when the payload is unknown or
    malformed.
    """
    tasks = state.get("tasks")
    if not isinstance(tasks, dict):
        return 0
    n = 0
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        tg = task.get("TextGeneration")
        if isinstance(tg, dict) and tg.get("taskStatus") in ("Pending", "Running"):
            n += 1
    return n


def idle_check(
    own_requests: Sequence[float] | None = None, *, min_idle_s: int = DEFAULT_MIN_IDLE_S
) -> IdleReport:
    """R1 idle guard (PREREG D1): ALL of (i) no active TextGeneration, (ii) no non-own
    generation POST in the window, (iii) no recent exo-model state.db activity.

    Conservative: any signal that cannot be evaluated (ssh failure, unparsable source)
    => NOT idle.
    """
    now = _now()
    own = list(own_requests or [])
    reasons: list[str] = []
    detail: dict = {"now": now, "min_idle_s": min_idle_s, "clock_offset_s": {},
                    "nodes": {}, "state_db": {}}

    node_ok: dict[str, bool] = {}
    for node in NODES:
        d = _collect_node(node, timeout=25.0)
        detail["clock_offset_s"][node] = d.get("clock_offset_s")
        detail["nodes"][node] = d
        node_ok[node] = d["log_ok"]
        if d["error"]:
            reasons.append(f"{node}: {d['error']} -> cannot evaluate, not idle")
            continue
        if d["running"]:
            reasons.append(
                f"{node}: last runner marker is 'runner running' (active TextGeneration "
                f"task) at {d.get('state_epoch')}"
            )
        lp = d.get("last_post")
        if lp is not None:
            age = now - lp[0]
            own_hit = any(abs(lp[0] - t) <= OWN_MATCH_WINDOW_S for t in own)
            if age <= min_idle_s and not own_hit:
                reasons.append(
                    f"{node}: non-own POST {lp[1]} age {age:.0f}s <= {min_idle_s}s "
                    f"(epoch {lp[0]:.3f})"
                )

    # Secondary: GET /state
    try:
        state = _fetch_json(f"{API_BASE}/state")
        active = _state_active_tasks(state)
        detail["state_active_tasks"] = active
        if active > 0:
            reasons.append(f"/state: {active} active TextGeneration task(s)")
    except Exception as exc:  # noqa: BLE001
        detail["state_error"] = repr(exc)
        reasons.append(f"/state unreachable ({exc!r}) -> cannot evaluate, not idle")
        state = None
    detail["state_db"] = _state_db_idle(now, min_idle_s, own, reasons)

    # clock skew gate: only fail loudly if a node offset is large AND own-request
    # matching is load-bearing (we have registered own requests).
    for node, off in detail["clock_offset_s"].items():
        if off is not None and abs(off) > CLOCK_SKEW_ABORT_S and own:
            reasons.append(
                f"{node}: clock offset {off:+.2f}s > {CLOCK_SKEW_ABORT_S}s -> "
                f"own-request matching unsafe, not idle"
            )

    ok = not reasons
    return IdleReport(ok=ok, reasons=reasons, detail=detail)


def _state_db_idle(now: float, min_idle_s: int, own: Sequence[float], reasons: list[str]) -> dict:
    out: dict = {"ok": False, "error": None, "recent_completed": [], "s4": {}}
    try:
        conn = _ro_connect(STATE_DB_URI)
    except sqlite3.Error as exc:
        out["error"] = repr(exc)
        reasons.append(f"state.db unreachable ({exc!r}) -> cannot evaluate, not idle")
        return out
    try:
        thr = now - min_idle_s
        rows = _db_s3_recent(conn, thr, "ended_at")
        # exclude rows that are our own (started_at near a registered own request)
        rows = [r for r in rows if not any(abs(r["started_at"] - t) <= OWN_MATCH_WINDOW_S for t in own)]
        out["recent_completed"] = rows
        out["ok"] = True
        if rows:
            r = rows[0]
            reasons.append(
                f"state.db: provider='custom' row id={r['id']} (call {r['call_seq']}) ended "
                f"{now - r['ended_at']:.0f}s ago"
            )
        s4 = _db_s4_recent(conn, thr)
        out["s4"] = s4
        if s4["messages"]:
            m = s4["messages"][0]
            reasons.append(
                f"state.db S4: exo-model session message id={m['id']} role={m['role']} "
                f"{now - m['timestamp']:.0f}s ago"
            )
        if s4["sessions"]:
            s = s4["sessions"][0]
            reasons.append(
                f"state.db S4: exo-model session {s['id']} last_activity "
                f"{now - s['last_activity_at']:.0f}s ago"
            )
    except sqlite3.Error as exc:
        out["error"] = repr(exc)
        reasons.append(f"state.db query failed ({exc!r}) -> cannot evaluate, not idle")
    finally:
        conn.close()
    return out


def wait_for_idle(
    *, poll_s: float = 30, max_wait_s: float = 3600, own_requests=None
) -> IdleReport:
    """Block until :func:`idle_check` reports idle (or timeout -> GuardFailure)."""
    deadline = _now() + max_wait_s
    last: IdleReport | None = None
    while True:
        last = idle_check(own_requests)
        if last.ok:
            return last
        if _now() >= deadline:
            raise GuardFailure(
                f"wait_for_idle timed out after {max_wait_s:.0f}s; reasons: "
                + "; ".join(last.reasons)
            )
        _sleep(poll_s)


# --------------------------------------------------------------------------------------
# R4 -- raw-GPU canary
# --------------------------------------------------------------------------------------


@dataclass
class CanaryReport:
    ok: bool
    state: str  # "healthy" | "marginal" | "degraded"
    per_node: dict[str, list[float]]
    median: dict[str, float]


def parse_canary_output(text: str) -> list[float]:
    """Extract the TFLOPS floats a node's canary printed (ignores the trailer text)."""
    out: list[float] = []
    for line in text.splitlines():
        if "TFLOPS" not in line:
            continue
        for tok in line.replace("TFLOPS", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def _canary_state(medians: Sequence[float]) -> str:
    if not medians:
        return "degraded"
    worst = min(medians)
    if worst >= 10:
        return "healthy"
    if worst >= 5:
        return "marginal"
    return "degraded"


def canary(nodes: Sequence[str] = ("studio1", "studio2"), *, timeout_s: float = 90) -> CanaryReport:
    """R4: run /tmp/gpu_canary2.py on each node SEQUENTIALLY (one GPU at a time)."""
    per_node: dict[str, list[float]] = {}
    medians: dict[str, float] = {}
    for node in nodes:
        # ensure the script exists (scp if missing) -- the only allowed node write
        rc, out = run_ssh(node, f"test -f {CANARY_DST} && echo PRESENT", timeout=20)
        if "PRESENT" not in out:
            src = CANARY_SRC
            if os.path.exists(src):
                _run_scp(src, CANARY_DST, node)
        rc, out = run_ssh(
            node,
            f"{NODE_VENV_PYTHON} {CANARY_DST}",
            timeout=timeout_s,
        )
        vals = parse_canary_output(out)
        per_node[node] = vals
        if vals:
            s = sorted(vals)
            medians[node] = s[len(s) // 2]
    state = _canary_state(list(medians.values()))
    return CanaryReport(ok=state == "healthy", state=state, per_node=per_node, median=medians)


# --------------------------------------------------------------------------------------
# R2 + R3 -- chunk guard / abort watcher
# --------------------------------------------------------------------------------------


class ChunkGuard:
    """Context manager: R1 idle gate on entry, background arrival/wall-cap watcher.

    On the first arrival signal the watcher sets ``cancel_event``, prints the literal
    token to stderr and (if ``interrupt_pid``) sends SIGINT.  On the wall cap it prints
    ``ABORTED_WALL_CAP``.  On exit it writes ``<log_dir>/<label>.guard.json``.
    """

    def __init__(
        self,
        label: str,
        *,
        max_wall_s: float = 900,
        poll_s: float = 15,
        registry_path: str | None = None,
        interrupt_pid: int | None = None,
        log_dir: str | None = None,
        own_requests: Sequence[float] | None = None,
        db_uri: str = STATE_DB_URI,
    ):
        self.label = label
        self.max_wall_s = max_wall_s
        self.poll_s = poll_s
        self.registry_path = registry_path
        self.interrupt_pid = interrupt_pid
        self.log_dir = log_dir
        self.db_uri = db_uri
        self.cancel_event = threading.Event()
        self._stop = threading.Event()
        self._own: list[float] = list(own_requests or [])
        self._own_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.t_start: float | None = None
        self.t_end: float | None = None
        self._aborted = False
        self._reason: str | None = None
        self._wall_cap_hit = False
        self._signals: list[str] = []
        self._detail: dict = {}

    # -- properties ---------------------------------------------------------------
    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def reason(self) -> str | None:
        return self._reason

    # -- own-request registry -----------------------------------------------------
    def register_own_request(self, t_start: float | None = None) -> None:
        """Call IMMEDIATELY BEFORE each HTTP request (time.time())."""
        t = _now() if t_start is None else t_start
        with self._own_lock:
            self._own.append(t)
        if self.registry_path:
            try:
                with open(self.registry_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"label": self.label, "t": t}) + "\n")
            except OSError:
                pass

    def own_requests(self) -> list[float]:
        with self._own_lock:
            return list(self._own)

    # -- context manager ----------------------------------------------------------
    def __enter__(self) -> "ChunkGuard":
        rep = idle_check(self.own_requests())
        if not rep.ok:
            raise GuardFailure(
                "ChunkGuard entry refused: cluster not idle: " + "; ".join(rep.reasons)
            )
        self.t_start = _now()
        self._thread = threading.Thread(
            target=self._watch, name=f"phase20-guard-{self.label}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll_s + 2))
        self.t_end = _now()
        self._write_json()

    # -- watcher ------------------------------------------------------------------
    def _watch(self) -> None:
        while not self._stop.is_set():
            if self.t_start is not None and (_now() - self.t_start) >= self.max_wall_s:
                self._wall_cap_hit = True
                self._fire("ABORTED_WALL_CAP", reason="wall cap")
                return
            try:
                self._poll_signals()
            except Exception as exc:  # noqa: BLE001 - never let the watcher die silently
                self._signals.append(f"poll_error:{exc!r}")
            if self.cancel_event.is_set():
                return
            self._stop.wait(self.poll_s)

    def _poll_signals(self) -> None:
        assert self.t_start is not None
        chunk_start = self.t_start
        # NOTE: the own-request list is deliberately NOT snapshotted here.  The two-node
        # ssh log read below takes ~1-3 s, and the harness calls register_own_request()
        # *immediately before* it sends -- i.e. AFTER this point but before the logs are
        # read.  A snapshot taken here would miss that registration and mis-classify the
        # harness's own first POST as non-own (live-reproduced false positive).  A POST
        # line can only be in a node log if its registration already happened, so the
        # own list is (re-)read AFTER the logs are read, just before each signal is
        # evaluated (see below).

        # ---- read both nodes' logs + offsets
        node_data: dict[str, dict] = {}
        for node in NODES:
            node_data[node] = _collect_node(node, timeout=20.0)
        events: list[LogEvent] = []
        offsets: dict[str, float | None] = {}
        for node, d in node_data.items():
            offsets[node] = d.get("clock_offset_s")
            for e in d.get("_events", []):
                events.append(
                    LogEvent(_apply_offset(e.epoch, offsets[node]), e.ts, e.kind, e.value)
                )
        events.sort(key=lambda e: e.epoch)

        # Re-read the own-request list now that the logs have been read: any registration
        # the harness made during the read is now visible, so its own POST lines match.
        own = self.own_requests()

        # ---- S2: unregistered generation POST newer than chunk start
        for ev in events:
            if ev.kind != "post" or not is_generation_post(ev.value):
                continue
            if ev.epoch <= chunk_start:
                continue
            if any(abs(ev.epoch - t) <= OWN_MATCH_WINDOW_S for t in own):
                continue
            self._signals.append(f"S2:non-own POST {ev.value} @{ev.ts}")
            self._fire("ABORTED_USER_ARRIVED", reason=f"S2 unregistered POST {ev.value} at {ev.ts}")
            return

        # ---- S1: concurrency / in-flight
        state_active = 0
        try:
            st = _fetch_json(f"{API_BASE}/state", timeout=8)
            state_active = _state_active_tasks(st)
        except Exception as exc:  # noqa: BLE001
            self._signals.append(f"S1:state_error:{exc!r}")
        last_kind, _ = runner_state(events)
        log_running = 1 if last_kind == "running" else 0
        active = max(state_active, log_running)

        # Re-read the own list for the S1 in-flight computation: the /state fetch above
        # takes time, during which the harness may have registered its request.
        own = self.own_requests()

        # own_inflight: registrations newer than the last observed "runner ready"
        last_ready = max((e.epoch for e in events if e.kind == "ready"), default=None)
        if last_ready is None:
            own_inflight = 1 if own else 0
        else:
            own_inflight = sum(1 for t in own if t > last_ready)
        own_inflight = min(own_inflight, 1)  # harnesses are sequential

        if active >= 2 or (active >= 1 and own_inflight == 0):
            self._signals.append(
                f"S1:active={active} own_inflight={own_inflight} state={state_active} log={log_running}"
            )
            self._fire(
                "ABORTED_USER_ARRIVED",
                reason=f"S1 concurrency: {active} active TextGeneration, own_inflight={own_inflight}",
            )
            return

        # ---- S3 / S4: state.db
        try:
            conn = _ro_connect(self.db_uri)
        except sqlite3.Error:
            return
        try:
            rows = _db_s3_recent(conn, chunk_start, "started_at")
            rows = [r for r in rows if not any(abs(r["started_at"] - t) <= OWN_MATCH_WINDOW_S for t in own)]
            if rows:
                r = rows[0]
                self._signals.append(f"S3:row id={r['id']} @{r['started_at']:.3f}")
                self._fire("ABORTED_USER_ARRIVED", reason=f"S3 completed-row id={r['id']}")
                return
            s4 = _db_s4_recent(conn, chunk_start)
            if s4["messages"]:
                m = s4["messages"][0]
                if any(abs(m["timestamp"] - t) <= OWN_MATCH_WINDOW_S for t in own):
                    return
                self._signals.append(f"S4:message id={m['id']}")
                self._fire("ABORTED_USER_ARRIVED", reason=f"S4 early session message id={m['id']}")
                return
            if s4["sessions"]:
                s = s4["sessions"][0]
                self._signals.append(f"S4:session {s['id']}")
                self._fire("ABORTED_USER_ARRIVED", reason=f"S4 session {s['id']} activity")
                return
        except sqlite3.Error:
            return
        finally:
            conn.close()

    def _fire(self, token: str, *, reason: str) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._reason = token
        print(token, file=sys.stderr, flush=True)
        self.cancel_event.set()
        if self.interrupt_pid:
            try:
                os.kill(self.interrupt_pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        self._stop.set()

    # -- harness hook -------------------------------------------------------------
    def check(self) -> None:
        """Raise :class:`ChunkAborted` if the watcher has fired."""
        if self.cancel_event.is_set():
            raise ChunkAborted(self._reason or "aborted")

    # -- persistence --------------------------------------------------------------
    def _write_json(self) -> None:
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            payload = {
                "label": self.label,
                "t_start": self.t_start,
                "t_end": self.t_end,
                "aborted": self._aborted,
                "reason": self._reason,
                "wall_cap_hit": self._wall_cap_hit,
                "own_requests": self.own_requests(),
                "signals_seen": self._signals,
            }
            with open(os.path.join(self.log_dir, f"{self.label}.guard.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
        except OSError:
            pass


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _print_json(obj) -> None:
    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, (list, tuple)):
            return [scrub(v) for v in o]
        return o

    print(json.dumps(scrub(dataclasses.asdict(obj)), indent=1, default=str))


def _cmd_idle(_args) -> int:
    rep = idle_check()
    _print_json(rep)
    return _EXIT_IDLE_OK if rep.ok else _EXIT_BUSY


def _cmd_wait_idle(args) -> int:
    try:
        rep = wait_for_idle(poll_s=args.poll, max_wait_s=args.max_wait)
    except GuardFailure as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=1))
        return _EXIT_BUSY
    _print_json(rep)
    return _EXIT_IDLE_OK


def _cmd_canary(_args) -> int:
    rep = canary()
    _print_json(rep)
    return _EXIT_CANARY[rep.state]


def _cmd_watch(args) -> int:
    if not args.command:
        print("watch: no command given after --", file=sys.stderr)
        return 2
    if args.canary:
        rep = canary()
        _print_json(rep)
        if not rep.ok:
            print(f"canary {rep.state}: refusing to start chunk", file=sys.stderr)
            return _EXIT_CANARY[rep.state]

    # R1: idle gate BEFORE starting the command.
    rep = idle_check()
    if not rep.ok:
        _print_json(rep)
        return _EXIT_BUSY

    proc = subprocess.Popen(args.command)
    guard = ChunkGuard(
        args.label,
        max_wall_s=args.max_wall,
        poll_s=args.poll,
        registry_path=args.registry,
        interrupt_pid=proc.pid,  # set BEFORE the watcher starts (no SIGINT race)
        log_dir=args.log_dir,
    )
    try:
        guard.__enter__()  # re-checks idle (tiny race window) and starts the watcher
    except GuardFailure as exc:
        proc.kill()
        proc.wait()
        print(json.dumps({"ok": False, "error": str(exc)}, indent=1))
        return _EXIT_BUSY

    rc: int
    try:
        while True:
            try:
                rc = proc.wait(timeout=guard.poll_s)
                break
            except subprocess.TimeoutExpired:
                if guard.cancel_event.is_set():
                    # SIGINT already sent by the watcher; escalate to SIGTERM after 20 s
                    deadline = _now() + 20
                    while proc.poll() is None and _now() < deadline:
                        _sleep(1)
                    if proc.poll() is None:
                        proc.terminate()
                    try:
                        rc = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        rc = proc.wait()
                    break
    finally:
        guard.__exit__()

    if guard.reason == "ABORTED_USER_ARRIVED":
        return _EXIT_ABORTED_USER
    if guard.reason == "ABORTED_WALL_CAP":
        return _EXIT_ABORTED_WALL
    return rc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase20_guard", description="Phase-20 safety guard")
    sub = p.add_subparsers(dest="sub", required=True)

    sp = sub.add_parser("idle", help="R1 idle check; exit 0 idle / 1 busy")
    sp.set_defaults(func=_cmd_idle)

    sp = sub.add_parser("wait-idle", help="block until idle")
    sp.add_argument("--max-wait", type=float, default=3600)
    sp.add_argument("--poll", type=float, default=30)
    sp.set_defaults(func=_cmd_wait_idle)

    sp = sub.add_parser("canary", help="R4 GPU canary; exit 0 healthy / 1 marginal / 2 degraded")
    sp.set_defaults(func=_cmd_canary)

    sp = sub.add_parser("watch", help="guard a chunk: run a command under the watcher")
    sp.add_argument("--label", required=True)
    sp.add_argument("--max-wall", type=float, default=900)
    sp.add_argument("--poll", type=float, default=15)
    sp.add_argument("--canary", action="store_true")
    sp.add_argument("--registry", default=None)
    sp.add_argument("--log-dir", default=None)
    sp.set_defaults(func=_cmd_watch)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    # `watch -- <cmd>`: split the command tail ourselves so argparse never sees it.
    command: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        command = argv[i + 1 :]
        argv = argv[:i]
    args = parser.parse_args(argv)
    args.command = command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

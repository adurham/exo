"""CPU-only tests for the §G3 park-on-shutdown fast path (plan §G3, audit A8).

No GPU, no cluster. Layers:

* the PURE §G3 bookkeeping: the env gate, the budget math (``DRAIN + PARK +
  MARGIN <= GRACE`` and the wall-clock park cap), and the idempotent signal
  state machine (a duplicate signal is a no-op; a signal before the runner loop
  exists unwinds cleanly via ``SystemExit``);
* the runner hooks built with ``object.__new__(Runner)``: the off-state no-op,
  the bounded idle wait while armed, and the drain-timeout predicate;
* the engine-thread sequence over the REAL fork ``ModelCache`` (stub body)
  through ``Dsv41Sessions.abort_inflight`` + ``force_checkpoint_all`` ->
  ``park.checkpoint_conversation`` -> ``coord.run_checkpoint``:
  - a mid-turn conversation (during PREFILL and during DECODE) is NEVER parked
    (nothing with ``n_ctx != offset`` is written); its last §G1 generation stays
    the on-disk restore point and restores byte-identically;
  - the aborted in-flight session is dropped from residency while a DIFFERENT
    between-turns dirty session still gets its force checkpoint;
  - the ``PARK_S`` deadline stops the loop;
  - the force checkpoint stays rank-invariant (G4 lockstep) and COLD on any
    candidate disagreement;
* the DEFAULT-OFF gate: with ``EXO_DSV41_SHUTDOWN_PARK`` unset every hook is inert.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import park as pk
from exo.worker.engines.mlx.dsv41 import session as s_
from exo.worker.engines.mlx.dsv41 import shutdown as sh_

from ._stub_harness import StubBody, run_turn
from .test_dsv41_park_g4 import _both, _fp, _Lockstep, _on

CONV_KW: dict[str, Any] = dict(chunk=4, long_chunk=4, long_threshold=10**9)
IDS = np.arange(10, 82, dtype=np.int64)  # 72 prompt rows
MIN_TOKENS = 8


@pytest.fixture(autouse=True)
def _ckpt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # The force checkpoint rides the §G1 path, so it needs §G1 enabled.
    monkeypatch.setenv(s_.CKPT_ENV, "1")
    monkeypatch.setenv(s_.CKPT_IDLE_S_ENV, "0")
    monkeypatch.setenv(s_.CKPT_MIN_DELTA_ENV, "16")
    monkeypatch.delenv("EXO_DSV41_PARK", raising=False)
    # §G3 gate is opt-in; clear it unless a test sets it.
    monkeypatch.delenv(sh_.SHUTDOWN_PARK_ENV, raising=False)


# ---------------------------------------------------------------------------
# harness (mirrors the §G1 file)
# ---------------------------------------------------------------------------

def _body() -> StubBody:
    return StubBody(initial_capacity=4, engram_layer_ids=(0,))


def _store(root: Path, body: Any, **kw: Any) -> pk.ParkedStore:
    return pk.ParkedStore(body, None, root=root, conv_kw=CONV_KW,
                          min_tokens=MIN_TOKENS, fingerprint=_fp(), **kw)


def _sessions(body: Any, store: Any, **kw: Any) -> s_.Dsv41Sessions:
    return s_.Dsv41Sessions(body, None, max_seq_len=256, park_store=store,
                            **CONV_KW, **kw)


def _gens(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for d in root.iterdir():
        parsed = pk._gen_from_name(d.name)
        if parsed is not None and (d / pk.MANIFEST).is_file():
            out[parsed[1]] = d
    return out


def _gen_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [d for d in root.iterdir()
            if pk._gen_from_name(d.name) is not None
            and (d / pk.MANIFEST).is_file()]


def _tmp_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [d.name for d in root.iterdir() if pk._is_tmp_name(d.name)]


def _cache_bytes(conv: Any) -> dict[str, bytes]:
    """Every live-reachable buffer of a conversation, as host bytes."""
    import mlx.core as mx

    mc = conv.cache.cache
    off = int(mc.offset)
    out: dict[str, bytes] = {"ids": np.asarray(conv.tokens, np.int64).tobytes()}
    for i, lc in enumerate(mc.layers):
        out[f"win{i}"] = np.asarray(lc.win_kv.astype(mx.float32)).tobytes()
        ratio = max(int(lc.ratio or 0), 1)
        used = -(-off // ratio)
        if lc.comp_kv is not None:
            out[f"comp{i}"] = np.asarray(lc.comp_kv[:, :used].astype(mx.float32)).tobytes()
        if lc.index_k is not None:
            out[f"idx{i}"] = np.asarray(lc.index_k[:, :used].astype(mx.float32)).tobytes()
        if lc.comp_state is not None:
            out[f"carry{i}"] = (np.asarray(lc.comp_state.kv_state).tobytes()
                                + np.asarray(lc.comp_state.score_state).tobytes())
    if mc.engram_ids is not None:
        out["engram"] = np.asarray(mc.engram_ids[:, :off]).tobytes()
    return out


def _turn(sess: s_.Dsv41Sessions, ids: np.ndarray, key: str = "A") -> Any:
    conv, _fed, _gen = run_turn(sess, ids, key=key)
    return conv


def _budget(**over: float) -> sh_.ShutdownBudget:
    base = dict(drain_s=10.0, park_s=4.0, exit_margin_s=1.0, grace_s=15.0)
    base.update(over)
    return sh_.ShutdownBudget(**base)  # type: ignore[arg-type]


def _hash(conv: Any) -> str:
    return pk.prefix_hash(np.asarray(conv.tokens, dtype=np.int64))


# ---------------------------------------------------------------------------
# env gate + budget math
# ---------------------------------------------------------------------------

def test_shutdown_park_gate_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not sh_.shutdown_park_enabled()
    for on in ("1", "true", "YES", "on"):
        monkeypatch.setenv(sh_.SHUTDOWN_PARK_ENV, on)
        assert sh_.shutdown_park_enabled()
    for off in ("0", "false", "no", "off", ""):
        monkeypatch.setenv(sh_.SHUTDOWN_PARK_ENV, off)
        assert not sh_.shutdown_park_enabled()


def test_shutdown_budget_defaults_and_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (sh_.SHUTDOWN_DRAIN_S_ENV, sh_.SHUTDOWN_PARK_S_ENV,
                sh_.SHUTDOWN_EXIT_MARGIN_S_ENV, sh_.GRACE_ENV):
        monkeypatch.delenv(var, raising=False)
    b = sh_.shutdown_budget()
    assert (b.drain_s, b.park_s, b.exit_margin_s, b.grace_s) == (10.0, 4.0, 1.0, 15.0)
    assert b.fits  # 10 + 4 + 1 == 15
    # Overrides are honoured and the fit test reports a real overrun.
    monkeypatch.setenv(sh_.SHUTDOWN_DRAIN_S_ENV, "9")
    monkeypatch.setenv(sh_.SHUTDOWN_PARK_S_ENV, "4")
    monkeypatch.setenv(sh_.SHUTDOWN_EXIT_MARGIN_S_ENV, "1")
    monkeypatch.setenv(sh_.GRACE_ENV, "10")
    b2 = sh_.shutdown_budget()
    assert (b2.drain_s, b2.park_s, b2.exit_margin_s, b2.grace_s) == (9.0, 4.0, 1.0, 10.0)
    assert not b2.fits  # 9 + 4 + 1 > 10


def test_shutdown_budget_park_deadline_hard_cap() -> None:
    b = _budget(drain_s=10.0, park_s=4.0, exit_margin_s=1.0, grace_s=15.0)
    # Drain finished early (park starts at t=2): the nominal phase budget wins.
    assert b.park_deadline(signal_time=0.0, park_start=2.0) == 6.0
    # Drain overran (park starts at t=14, past the nominal 4 s window): the hard
    # GRACE cap (0 + 15 - 1) is what protects the SIGKILL boundary.
    assert b.park_deadline(signal_time=0.0, park_start=14.0) == 14.0


def test_run_shutdown_park_past_deadline_only_aborts(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    _turn(sess, IDS)
    # signal 100 s ago with a 15 s GRACE: the park deadline is already behind us
    # -> abort only, no checkpoint attempted.
    out = sh_.run_shutdown_park(
        sess, signal_time=time.monotonic() - 100.0, budget=_budget())
    assert out.deadline_hit is True and out.parked == [] and out.aborted == []
    assert not _gen_dirs(tmp_path)


# ---------------------------------------------------------------------------
# signal state machine: idempotency + clean-exit guarantee
# ---------------------------------------------------------------------------

def test_shutdown_signal_idempotent_two_signals_one_sequence() -> None:
    st = sh_.ShutdownSignal(park=True)
    st.runner_active = True  # the runner loop is live
    st.on_signal(15, None)
    first_time = st.signal_time
    assert st.requested and st.count == 1 and first_time is not None
    # A duplicate signal is a no-op: no re-arm, no raise, timestamp frozen.
    st.on_signal(15, None)
    st.on_signal(2, None)
    assert st.count == 3 and st.requested and st.signal_time == first_time


def test_shutdown_signal_before_runner_unwinds_cleanly() -> None:
    # Park armed but the runner loop does not exist yet (during load): the signal
    # must unwind with SystemExit -> entrypoint finally -> RDMA teardown.
    st = sh_.ShutdownSignal(park=True)
    assert not st.runner_active
    with pytest.raises(SystemExit):
        st.on_signal(15, None)
    assert st.requested and st.count == 1


def test_shutdown_signal_default_off_keeps_systemexit() -> None:
    # With the fast path off the handler is the pre-G3 one: always SystemExit,
    # even for a duplicate (it is never installed in production; bootstrap keeps
    # ``_graceful_sigterm`` untouched when the gate is unset).
    st = sh_.ShutdownSignal(park=False)
    st.runner_active = True
    with pytest.raises(SystemExit):
        st.on_signal(15, None)
    with pytest.raises(SystemExit):
        st.on_signal(15, None)
    assert not st.requested  # never arms the park path


# ---------------------------------------------------------------------------
# runner hooks (object.__new__(Runner))
# ---------------------------------------------------------------------------

def _fake_runner(**attrs: Any) -> Any:
    import queue as _queue

    from exo.worker.runner.runner import Runner

    r = object.__new__(Runner)
    r._work_queue = _queue.Queue()
    r._shutdown_state = None
    r._shutdown_budget = None
    r.active_tasks = {}
    r.seen = set()
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


def test_runner_shutdown_hooks_off_are_noops() -> None:
    r = _fake_runner()
    assert r._shutdown_pending() is False
    assert r._shutdown_drain_expired() is False
    # No engine attached: _run_shutdown_park returns without touching anything.
    r._run_shutdown_park()
    r.generator = object()
    r._run_shutdown_park()  # still a no-op (not an Engine)


def test_runner_drain_timeout_predicate() -> None:
    st = sh_.ShutdownSignal(park=True)
    st.runner_active = True
    st.on_signal(15, None)
    r = _fake_runner(_shutdown_state=st,
                     _shutdown_budget=_budget(drain_s=10.0))
    assert r._shutdown_pending() is True
    # Just signalled -> the drain window is still open (keep draining).
    assert r._shutdown_drain_expired() is False
    # Signal arrived 11 s ago -> the drain window is over (abort).
    st.signal_time = time.monotonic() - 11.0
    assert r._shutdown_drain_expired() is True


def test_runner_idle_wait_bounded_only_when_armed() -> None:
    """Mirror of the §G1 runner test: OFF -> blocking get; §G3 armed -> bounded."""
    from exo.shared.types.worker.runners import RunnerReady
    from exo.worker.engines.base import Engine

    class _Eng:
        def idle_tick_seconds(self) -> float | None:
            return None  # §G1 idle ticks off

        def idle_tick(self, request_pending: Any = None) -> None:  # pragma: no cover
            raise AssertionError("no idle tick expected with the engine offering none")

    Engine.register(_Eng)
    r = _fake_runner(generator=_Eng(),
                     current_status=RunnerReady(prefill_server_port=None))
    r._work_queue.put("item")
    assert r._next_work_item() == "item"  # OFF: plain blocking get path
    st = sh_.ShutdownSignal(park=True)
    st.poll_s = 0.01
    r._shutdown_state = st
    # Armed: the wait is bounded by poll_s, so it returns None quickly (no tick).
    t0 = time.monotonic()
    assert r._next_work_item() is None
    assert time.monotonic() - t0 < 5.0


def test_runner_drain_abort_fails_tasks() -> None:
    class _Sent:
        def __init__(self) -> None:
            self.failed: list[Any] = []

    r = _fake_runner()
    r.active_tasks = {"t1": object(), "t2": object()}
    r._sent = _Sent()
    r.send_task_status = lambda tid, status: r._sent.failed.append(tid)
    r._abort_inflight_for_shutdown()
    assert r._sent.failed == ["t1", "t2"]
    assert r.active_tasks == {}


# ---------------------------------------------------------------------------
# engine wiring: shutdown_park runs abort then force checkpoint
# ---------------------------------------------------------------------------

class _RecordingSessions:
    def __init__(self) -> None:
        self.aborted = 0
        self.force_calls: list[Any] = []
        self.pc_result: bool | None = None

    def abort_inflight(self) -> list[str]:
        self.aborted += 1
        return ["id:MID"]

    def force_checkpoint_all(self, *, deadline: float, now: Any, exclude: Any = None,
                             precheck: Any = None, request_pending: Any = None) -> Any:
        pc = precheck() if precheck is not None else None
        self.pc_result = pc
        self.force_calls.append((deadline, pc))
        return ([("id:DIRTY", "outcome")], False)


def _bare_engine() -> Any:
    from exo.worker.engines.mlx.dsv41 import agreement as ag
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine

    eng = object.__new__(Dsv41Engine)
    eng._active = None
    eng._agreement = ag.RankAgreement(None)
    eng._draft_windows = {}
    eng._shutting_down = False
    return eng


def test_engine_shutdown_park_wires_abort_then_force() -> None:
    eng = _bare_engine()
    rec = _RecordingSessions()
    eng._sessions = rec
    eng._active = object()
    out = eng.shutdown_park(signal_time=time.monotonic(), budget=_budget())
    assert eng._shutting_down is True          # step 1: stop admitting
    assert eng._active is None                 # active request released
    assert rec.aborted == 1                    # step 3: abort + drop in-flight
    assert len(rec.force_calls) == 1           # step 4: force checkpoint ran
    # The precheck pairs agree_on_tasks (rank 0 local -> no peer task -> proceed).
    assert rec.pc_result is True
    assert out.aborted == ["id:MID"] and out.parked_count == 1


def test_engine_begin_shutdown_stops_admitting() -> None:
    eng = _bare_engine()
    eng._agreement.queue.append(object())
    eng.begin_shutdown()
    assert eng._shutting_down is True
    # _activate_next must not start the queued request while shutting down.
    eng._activate_next([])
    assert eng._active is None and len(eng._agreement.queue) == 1


# ---------------------------------------------------------------------------
# sequence over the real cache: mid-turn is never parked
# ---------------------------------------------------------------------------

def test_midturn_prefill_never_parked_last_generation_restored(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    conv = _turn(sess, IDS)                       # committed turn
    assert sess.maybe_checkpoint().all_committed  # §G1 gen 1 on disk
    disk = _cache_bytes(conv)
    gen1 = _gens(tmp_path)[1]
    assert _cache_bytes(pk.restore_conversation(gen1, body, None, **CONV_KW)) == disk
    disk_tokens = np.asarray(conv.tokens).copy()

    # Start a NEW turn and stop mid-PREFILL: offset advanced, _inflight True.
    more = np.concatenate([IDS, np.arange(500, 540)])
    conv.prefill(more)
    assert conv._inflight and int(conv.cache.cache.offset) > len(disk_tokens)
    live = _cache_bytes(conv)
    assert live != disk  # the live cache is a partial turn

    out = sh_.run_shutdown_park(sess, signal_time=time.monotonic(), budget=_budget())
    # The in-flight conversation was aborted + dropped, never parked.
    assert out.aborted == ["id:A"]
    assert all(e.session is not conv for e in sess._entries.values())
    # The last §G1 generation is untouched and still restores the pre-turn state.
    assert list(_gens(tmp_path)) == [1] and not _tmp_dirs(tmp_path)
    assert _cache_bytes(pk.restore_conversation(_gens(tmp_path)[1], body, None,
                                                **CONV_KW)) == disk
    assert store.has_committed(pk.prefix_hash(disk_tokens), 1)


def test_midturn_decode_never_parked_last_generation_restored(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    conv = _turn(sess, IDS)
    assert sess.maybe_checkpoint().all_committed
    disk = _cache_bytes(conv)

    # Start a new turn, decode some rounds, and stop mid-DECODE.
    nxt = np.concatenate([IDS, np.arange(600, 660)])
    run_turn(sess, nxt, checkpoint=False, key="A")
    assert conv._inflight and int(conv.cache.cache.offset) > len(IDS)

    out = sh_.run_shutdown_park(sess, signal_time=time.monotonic(), budget=_budget())
    assert out.aborted == ["id:A"]
    assert list(_gens(tmp_path)) == [1] and not _tmp_dirs(tmp_path)
    # Every committed generation is a between-turns state: the on-disk rows
    # equal its committed offset (nothing parked with n_ctx != offset).
    for d in _gen_dirs(tmp_path):
        assert _cache_bytes(pk.restore_conversation(d, body, None, **CONV_KW)) == disk


def test_aborted_inflight_dropped_while_dirty_betweenturn_parked(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    mid = _turn(sess, IDS, key="MID")                       # committed, dirty
    dirty = _turn(sess, np.arange(200, 280), key="DIRTY")  # committed, dirty
    assert not mid._inflight and not dirty._inflight

    # MID goes mid-turn (decode); DIRTY stays between turns and dirty.
    run_turn(sess, np.concatenate([IDS, np.arange(700, 730)]),
             checkpoint=False, key="MID")
    assert mid._inflight
    dirty_live = _cache_bytes(dirty)

    out = sh_.run_shutdown_park(sess, signal_time=time.monotonic(), budget=_budget())
    keys = {k for k, _ in out.parked}
    # MID aborted + dropped; DIRTY got its force checkpoint and stays resident.
    assert out.aborted == ["id:MID"]
    assert keys == {"id:DIRTY"}
    assert "id:MID" not in sess._entries and "id:DIRTY" in sess._entries
    assert store.has_committed(_hash(dirty), 1)
    # Nothing was written for the aborted conversation.
    assert not store.has_committed(_hash(mid), 1)
    # DIRTY's committed generation restores byte-identically (between-turns).
    gens = _gens(tmp_path)
    assert _cache_bytes(pk.restore_conversation(gens[1], body, None, **CONV_KW)) == dirty_live
    assert not _tmp_dirs(tmp_path)


# ---------------------------------------------------------------------------
# budget: the PARK_S deadline stops the loop
# ---------------------------------------------------------------------------

def test_force_checkpoint_deadline_in_the_past_stops_immediately(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    _turn(sess, IDS, key="A")
    _turn(sess, np.arange(200, 280), key="B")
    results, hit = sess.force_checkpoint_all(deadline=time.monotonic() - 1.0)
    assert results == [] and hit is True and not _gen_dirs(tmp_path)


def test_force_checkpoint_deadline_allows_exactly_one(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    _turn(sess, IDS, key="A")
    time.sleep(0.01)
    _turn(sess, np.arange(200, 280), key="B")  # most recent -> attempted first

    # An advancing clock: the deadline is 3.0, each now() steps +2.0, so the loop
    # gets through exactly one checkpoint before the next now() passes the cap.
    clock = {"t": 0.0}

    def now() -> float:
        clock["t"] += 2.0
        return clock["t"]

    results, hit = sess.force_checkpoint_all(deadline=3.0, now=now)
    assert [k for k, _ in results] == ["id:B"] and hit is True
    assert len(_gen_dirs(tmp_path)) == 1


def test_force_checkpoint_most_recent_first_and_one_attempt_each(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    _turn(sess, IDS, key="A")
    time.sleep(0.01)
    _turn(sess, np.arange(200, 280), key="B")  # most recent

    results, hit = sess.force_checkpoint_all(deadline=time.monotonic() + 30.0)
    assert hit is False
    assert [k for k, _ in results] == ["id:B", "id:A"]  # most-recent first
    assert all(not e.session.dirty_tokens for e in sess._entries.values())
    assert len(_gen_dirs(tmp_path)) == 2


def test_force_checkpoint_skips_inflight(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    a = _turn(sess, IDS, key="A")
    run_turn(sess, np.concatenate([IDS, np.arange(800, 820)]),
             checkpoint=False, key="B")  # B in flight
    results, _hit = sess.force_checkpoint_all(deadline=time.monotonic() + 30.0)
    assert [k for k, _ in results] == ["id:A"]
    assert store.has_committed(_hash(a), 1)
    assert len(_gen_dirs(tmp_path)) == 1


# ---------------------------------------------------------------------------
# flag-off: nothing runs
# ---------------------------------------------------------------------------

def test_flag_off_runner_and_engine_do_nothing() -> None:
    assert not sh_.shutdown_park_enabled()
    r = _fake_runner()
    assert r._shutdown_pending() is False and r._shutdown_drain_expired() is False

    class _Eng:
        def __init__(self) -> None:
            self.calls = 0

        def shutdown_park(self, **kw: Any) -> Any:  # pragma: no cover
            self.calls += 1
            return None

    from exo.worker.engines.base import Engine

    Engine.register(_Eng)
    eng = _Eng()
    r.generator = eng
    r._run_shutdown_park()
    assert eng.calls == 0


# ---------------------------------------------------------------------------
# two simulated ranks: the force checkpoint is rank-invariant / COLD-safe
# ---------------------------------------------------------------------------

class _R3:
    """One simulated rank: store + sessions sharing the G4 lockstep coordinator."""

    def __init__(self, root: Path, rank: int, ls: _Lockstep, ids: np.ndarray = IDS) -> None:
        self.root = root
        self.rank = rank
        self.ids = ids
        self.coord = ls.coordinator(rank)
        _on(rank, self._build)

    def _build(self) -> None:
        self.body = _body()
        self.store = pk.ParkedStore(
            self.body, None, root=self.root, conv_kw=CONV_KW, min_tokens=MIN_TOKENS,
            fingerprint=_fp(layout=f"rank{self.rank}"), coordinator=None, rank=self.rank)
        self.sess = s_.Dsv41Sessions(
            self.body, None, max_seq_len=256, park_store=self.store,
            coordinator=None, rank=self.rank, **CONV_KW)
        self.conv = _turn(self.sess, self.ids)   # committed, dirty, between turns
        self.attach()

    def attach(self) -> None:
        self.sess._coordinator = self.coord
        self.store.coordinator = self.coord

    def detach(self) -> None:
        self.sess._coordinator = None
        self.store.coordinator = None

    def gens(self) -> set[int]:
        return set(_gens(self.root))


def test_tp2_force_checkpoint_agrees_same_generation(tmp_path: Path) -> None:
    ls = _Lockstep(2, timeout=20.0)
    a = _R3(tmp_path / "r0", 0, ls)
    b = _R3(tmp_path / "r1", 1, ls)

    def run(r: _R3) -> Any:
        return sh_.run_shutdown_park(r.sess, signal_time=time.monotonic(), budget=_budget())

    res = _both([lambda: run(a), lambda: run(b)])
    assert [len(x.parked) for x in res] == [1, 1]
    assert all(x.parked[0][1].all_committed for x in res)
    assert a.gens() == b.gens() == {1}          # same agreed generation (G4)
    assert ls.calls[0] == ls.calls[1]           # rank-invariant exchange count


def test_tp2_candidate_disagreement_is_cold_safe(tmp_path: Path) -> None:
    """Two ranks with DIFFERENT between-turns candidates (same count): the IDLE
    exchange is a no-go on BOTH ranks and the force loop stops without writing a
    generation (COLD-safe)."""
    ls = _Lockstep(2, timeout=20.0)
    a = _R3(tmp_path / "r0", 0, ls)
    b = _R3(tmp_path / "r1", 1, ls, ids=np.arange(300, 380))  # a different stream

    def run(r: _R3) -> Any:
        return sh_.run_shutdown_park(r.sess, signal_time=time.monotonic(), budget=_budget())

    res = _both([lambda: run(a), lambda: run(b)])
    assert all(
        len(x.parked) == 1 and not x.parked[0][1].all_committed for x in res
    )
    assert a.gens() == set() and b.gens() == set()
    assert ls.calls[0] == ls.calls[1] == 1  # exactly one IDLE exchange each

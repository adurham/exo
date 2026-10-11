"""CPU-only tests for the §G1 after-turn idle checkpoint (keep-resident park).

No GPU, no cluster. Layers:

* the PURE cross-rank decisions (``coord.decide_idle`` / ``decide_poll``);
* the keep-resident checkpoint over the REAL fork ``ModelCache`` (stub body)
  through ``Dsv41Sessions.maybe_checkpoint`` -> ``park.checkpoint_conversation``
  -> ``coord.run_checkpoint`` -> ``ParkedStore.open_generation`` (single rank,
  and two simulated ranks with the G4 lockstep gather);
* **T7** -- a request arriving mid-checkpoint aborts the write within one unit
  (one layer), leaves NO partial generation, keeps the previous generation, and
  the request's turn is byte-identical to a never-checkpointed control;
* **T9** -- the MLX buffer-donation trap: after a checkpoint (committed or
  aborted) the decode-step in-place writes into the cache buffers still DONATE
  (data pointers unchanged step to step), proven against a negative control
  that deliberately leaks one reference; plus a decode-step timing comparison
  and the "pause <= one layer write" bound;
* the engine idle tick + runner hook (default OFF; gating; the agree_on_tasks
  pairing with a peer that is inside ``step()``).
"""

from __future__ import annotations

import ctypes
import errno
import os
import queue
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import agreement as ag
from exo.worker.engines.mlx.dsv41 import coord as co
from exo.worker.engines.mlx.dsv41 import park as pk
from exo.worker.engines.mlx.dsv41 import session as s_

from ._stub_harness import StubBody, run_turn
from .test_dsv41_park_g4 import _both, _fp, _Lockstep, _on

CONV_KW: dict[str, Any] = dict(chunk=4, long_chunk=4, long_threshold=10**9)
IDS = np.arange(10, 82, dtype=np.int64)  # 72 prompt rows
MIN_TOKENS = 8


@pytest.fixture(autouse=True)
def _ckpt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(s_.CKPT_ENV, "1")
    monkeypatch.setenv(s_.CKPT_IDLE_S_ENV, "0")
    monkeypatch.setenv(s_.CKPT_MIN_DELTA_ENV, "16")
    monkeypatch.delenv("EXO_DSV41_PARK", raising=False)


def _body() -> StubBody:
    return StubBody(initial_capacity=4, engram_layer_ids=(0,))


def _store(root: Path, body: Any, **kw: Any) -> pk.ParkedStore:
    return pk.ParkedStore(body, None, root=root, conv_kw=CONV_KW,
                          min_tokens=MIN_TOKENS, fingerprint=_fp(), **kw)


def _sessions(body: Any, store: Any, **kw: Any) -> s_.Dsv41Sessions:
    return s_.Dsv41Sessions(body, None, max_seq_len=256, park_store=store, **CONV_KW, **kw)


def _gens(root: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for d in root.iterdir():
        parsed = pk._gen_from_name(d.name)
        if parsed is not None and (d / pk.MANIFEST).is_file():
            out[parsed[1]] = d
    return out


def _tmp_dirs(root: Path) -> list[str]:
    return [d.name for d in root.iterdir() if pk._is_tmp_name(d.name)] if root.is_dir() else []


def _cache_bytes(conv: Any) -> dict[str, bytes]:
    """Every live-reachable buffer of a conversation, as host bytes."""
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


def _turn(sess: s_.Dsv41Sessions, ids: np.ndarray) -> tuple[Any, list[int]]:
    conv, _fed, gen = run_turn(sess, ids, key="A")
    return conv, gen


# ---------------------------------------------------------------------------
# pure decisions
# ---------------------------------------------------------------------------

def test_decide_idle_pure() -> None:
    row = co._idle_row(True, 0xABC, 100)
    assert co.decide_idle([row, row]) == co.IdleDecision(True, 0xABC, 100)
    assert co.decide_idle([row]).go
    assert co.decide_idle([row, co._idle_row(False, 0xABC, 100)]).reason == "rank-not-ready"
    assert co.decide_idle([row, co._idle_row(True, 0xABD, 100)]).reason == "candidate-mismatch"
    assert co.decide_idle([row, co._idle_row(True, 0xABC, 101)]).reason == "candidate-mismatch"
    assert co.decide_idle([row, []]).reason == "malformed"
    assert co.decide_idle([row, [co.MAGIC_PARK_GEN, 0, 0, 0, 0]]).reason == "malformed"
    assert not co.decide_idle([]).go


def test_decide_poll_pure() -> None:
    p = co._poll_row
    assert co.decide_poll([p(False, True, True), p(False, True, False)]) == "continue"
    assert co.decide_poll([p(False, True, False)] * 2) == "done"
    assert co.decide_poll([p(False, True, True), p(True, True, True)]) == "abort-pending"
    # A failure beats a pending request (both abort; reported as a failure).
    assert co.decide_poll([p(True, True, True), p(False, False, False)]) == "abort-failed"
    assert co.decide_poll([p(False, True, True), []]) == "abort-failed"
    assert co.decide_poll([]) == "abort-failed"


# ---------------------------------------------------------------------------
# single rank: keep-resident checkpoint, gate, dirty tracking
# ---------------------------------------------------------------------------

def test_checkpoint_keeps_resident_and_is_restorable(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    conv, _ = _turn(sess, IDS)
    assert conv.last_turn_end_ts > 0 and not conv.ever_checkpointed
    assert conv.dirty_tokens == conv.offset
    before = _cache_bytes(conv)

    out = sess.maybe_checkpoint()
    assert out is not None and out.go and out.all_committed and out.written
    # Resident: same object, same cache, NOT closed, nothing changed.
    assert list(sess._entries.values())[0].session is conv
    assert conv.cache.cache is not None and _cache_bytes(conv) == before
    assert conv.ckpt_offset == conv.offset and conv.dirty_tokens == 0
    gens = _gens(tmp_path)
    assert list(gens) == [1] and not _tmp_dirs(tmp_path)
    # Same on-disk format as an eviction park: restores byte-identically.
    restored = pk.restore_conversation(gens[1], body, None, **CONV_KW)
    assert restored is not None and _cache_bytes(restored) == before
    # The manifest is the ordinary one (§G2 fingerprint, §G4 generation).
    assert store.has_committed(pk.prefix_hash(np.asarray(conv.tokens)), 1)

    # Clean -> the next idle tick does nothing (no exchange, nothing written).
    assert sess.maybe_checkpoint() is None
    # The conversation still serves the next turn from residency.
    conv2, _ = _turn(sess, np.concatenate([IDS, np.arange(200, 230)]))
    assert conv2 is conv and conv.dirty_tokens > 0


def test_park_conversation_format_unchanged_by_split(tmp_path: Path) -> None:
    """The eviction park (one-shot) and the checkpoint (unit by unit) write the
    same file set and the same manifest content."""
    body = _body()
    conv, _ = _turn(_sessions(body, None), IDS)
    a, b = tmp_path / "a" / "x.gen1.aaaaaaaa", tmp_path / "b" / "x.gen1.bbbbbbbb"
    a.parent.mkdir()
    b.parent.mkdir()
    assert pk.park_conversation(conv, a, key="A", fingerprint=_fp()) is True
    w = pk._GenerationWriter(conv, b, key="A", fingerprint=_fp(), rank=None)
    assert w.begin()
    n = 0
    while w.has_more():
        w.write_next()
        n += 1
    assert n == w.total_units >= len(conv.cache.cache.layers)
    assert w.commit()
    assert sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir())
    import json
    ma, mb = (json.loads((d / pk.MANIFEST).read_text()) for d in (a, b))
    ma.pop("created")
    mb.pop("created")
    assert ma == mb
    for f in a.iterdir():
        if f.name == pk.MANIFEST:
            continue  # compared above minus the ``created`` timestamp
        assert f.read_bytes() == (b / f.name).read_bytes(), f.name


def test_gate_min_tokens_delta_idle_and_most_recent_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _body()
    sess = _sessions(body, _store(tmp_path, body), max_sessions=3)
    sess.ckpt_idle_s = 20.0
    a, _f, _ = run_turn(sess, IDS, key="A")
    b, _f, _ = run_turn(sess, np.arange(300, 380), key="B")
    t = time.monotonic()
    # Idle not elapsed -> nothing.
    assert sess.checkpoint_candidate(now=t) is None
    # Idle elapsed -> the MOST RECENT turn first (B).
    cand = sess.checkpoint_candidate(now=t + 25)
    assert cand is not None
    k, c = cand
    assert c is b and k == "id:B"
    # Too short -> never.
    store_short = _store(tmp_path / "s", body)
    store_short.min_tokens = 10_000
    sess._park = store_short
    assert sess.checkpoint_candidate(now=t + 25) is None
    sess._park = _store(tmp_path, body)
    # Dirty below CKPT_MIN_DELTA after a checkpoint -> not again.
    b.ckpt_offset = b.offset - 3
    cand = sess.checkpoint_candidate(now=t + 25)
    assert cand is not None and cand[1] is a
    a.ckpt_offset = a.offset
    assert sess.checkpoint_candidate(now=t + 25) is None
    # Mid-turn -> never.
    b.ckpt_offset = -1
    b._inflight = True
    assert sess.checkpoint_candidate(now=t + 25) is None
    b._inflight = False
    # Feature flag off -> never (default OFF).
    monkeypatch.delenv(s_.CKPT_ENV)
    assert sess.checkpoint_candidate(now=t + 25) is None
    assert s_.ckpt_tick_seconds() is None


def test_one_session_per_idle_tick(tmp_path: Path) -> None:
    body = _body()
    sess = _sessions(body, _store(tmp_path, body), max_sessions=3)
    a, _f, _ = run_turn(sess, IDS, key="A")
    b, _f, _ = run_turn(sess, np.arange(300, 380), key="B")
    sess.maybe_checkpoint()
    assert [a.ever_checkpointed, b.ever_checkpointed] == [False, True]
    sess.maybe_checkpoint()
    assert [a.ever_checkpointed, b.ever_checkpointed] == [True, True]
    assert sess.maybe_checkpoint() is None
    assert sess.stats["ckpt_committed"] == 2


def test_generations_and_retention_across_checkpoints(tmp_path: Path) -> None:
    body = _body()
    sess = _sessions(body, _store(tmp_path, body))
    ids = IDS
    conv = None
    for i in range(4):
        conv, _ = _turn(sess, ids)
        assert sess.maybe_checkpoint().all_committed
        ids = np.concatenate([np.asarray(conv.tokens), np.arange(500 + 40 * i, 520 + 40 * i)])
    # Each checkpoint is a NEW generation of a NEW prefix (the token stream
    # grew), so every turn's dir is distinct; nothing is left half-written.
    assert not _tmp_dirs(tmp_path)
    assert conv is not None and conv.dirty_tokens == 0


def test_write_error_enospc_keeps_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    conv, _ = _turn(sess, IDS)
    assert sess.maybe_checkpoint().all_committed
    prev = _gens(tmp_path)
    prev_bytes = {p.name: p.read_bytes() for p in prev[1].iterdir()}
    # Same conversation, re-dirty it in place (same key -> gen 2 would follow).
    conv.ckpt_offset = 0
    real = pk._write_blob
    calls = {"n": 0}

    def enospc(d: Path, name: str, arr: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(d, name, arr)

    monkeypatch.setattr(pk, "_write_blob", enospc)
    out = sess.maybe_checkpoint()
    assert out is not None and out.go and not out.all_committed and out.aborted
    assert out.reason == "abort-failed"
    assert _gens(tmp_path) == prev and not _tmp_dirs(tmp_path)
    assert {p.name: p.read_bytes() for p in prev[1].iterdir()} == prev_bytes
    assert sess.stats["ckpt_failed"] == 1
    # Back-off: not retried immediately; conversation still resident + usable.
    assert sess.checkpoint_candidate() is None
    assert conv.cache.cache is not None and not conv._inflight
    conv2, _ = _turn(sess, np.concatenate([np.asarray(conv.tokens), np.arange(900, 905)]))
    assert conv2 is conv


def test_disabled_store_or_failed_store_is_nogo_not_raise(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body, enabled=False)
    sess = _sessions(body, store)
    _turn(sess, IDS)
    out = sess.maybe_checkpoint()
    assert out is not None and not out.go and out.reason == "rank-not-ready"
    assert not _gens(tmp_path)


def test_clean_eviction_skips_rewrite(tmp_path: Path) -> None:
    """A conversation already on disk at its offset is not rewritten on LRU
    eviction (counted as parked; the generation on disk is the newest)."""
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store, max_sessions=1)
    conv, _ = _turn(sess, IDS)
    assert sess.maybe_checkpoint().all_committed
    off = conv.offset
    before = dict(_gens(tmp_path))
    run_turn(sess, np.arange(400, 480), key="B")  # evicts A (closes it)
    assert sess.stats["parked"] == 1 and _gens(tmp_path) == before
    # ...and it restores from that generation.
    again = sess.get(IDS.tolist(), "A")
    assert sess.stats["restored"] == 1 and again.offset == off


# ---------------------------------------------------------------------------
# T7 -- a request arrives during a checkpoint
# ---------------------------------------------------------------------------

def test_t7_request_during_checkpoint_aborts_within_one_unit(tmp_path: Path) -> None:
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    conv, _ = _turn(sess, IDS)
    assert sess.maybe_checkpoint().all_committed  # gen 1 = the previous gen
    prev = _gens(tmp_path)
    prev_files = {p.name: p.read_bytes() for p in prev[1].iterdir()}

    # Re-dirty: next turn on the same conversation.
    ids2 = np.concatenate([np.asarray(conv.tokens), np.arange(600, 640)])
    _turn(sess, ids2)
    # Control: an identical conversation that is never checkpointed.
    ctrl_sess = _sessions(_body(), None)
    ctrl, _ = _turn(ctrl_sess, IDS)
    _turn(ctrl_sess, ids2)
    assert _cache_bytes(ctrl) == _cache_bytes(conv)

    flag = {"pending": False, "flip_after": 1, "after_flip": 0, "seen": 0}

    def on_unit(_w: Any) -> None:
        flag["seen"] += 1
        if flag["pending"]:
            flag["after_flip"] += 1
        if flag["seen"] == flag["flip_after"]:
            flag["pending"] = True  # the request lands while unit 1 is written

    out = sess.maybe_checkpoint(request_pending=lambda: flag["pending"], on_unit=on_unit)
    assert out is not None and out.go and out.aborted and out.pending_abort
    # Abort within ONE unit: the unit in flight finished, no further unit ran.
    assert out.units == flag["flip_after"] and flag["after_flip"] == 0
    # No partial generation: no new committed dir, temp dir removed.
    assert _gens(tmp_path).keys() == prev.keys() and not _tmp_dirs(tmp_path)
    assert {p.name: p.read_bytes() for p in prev[1].iterdir()} == prev_files
    # The conversation is resident, still dirty, no back-off for a pending abort.
    assert conv.dirty_tokens > 0 and sess.stats["ckpt_aborted"] == 1
    assert sess.checkpoint_candidate() is not None

    # Serve "the request": byte-identical to the never-checkpointed control.
    ids3 = np.concatenate([np.asarray(conv.tokens), np.arange(700, 720)])
    c3, g3 = _turn(sess, ids3)
    k3, gk = _turn(ctrl_sess, ids3)
    assert c3 is conv and g3 == gk and _cache_bytes(c3) == _cache_bytes(k3)
    # The next idle period checkpoints it fully.
    assert sess.maybe_checkpoint().all_committed


@pytest.mark.parametrize("flip_after", [0, 1, 2, 3])
def test_t7_abort_point_sweep(tmp_path: Path, flip_after: int) -> None:
    """A request seen before any unit / after unit k: exactly k units written,
    never a committed partial generation."""
    body = _body()
    sess = _sessions(body, _store(tmp_path, body))
    _turn(sess, IDS)
    state = {"n": 0}

    def on_unit(_w: Any) -> None:
        state["n"] += 1

    out = sess.maybe_checkpoint(
        request_pending=lambda: state["n"] >= flip_after, on_unit=on_unit)
    if flip_after == 0:
        # Pending before anything started: the IDLE exchange itself says no-go.
        assert not out.go and out.reason == "rank-not-ready" and out.units == 0
    else:
        assert out.pending_abort and out.units == flip_after
    assert not _gens(tmp_path) and not _tmp_dirs(tmp_path)


def test_t7_pause_bounded_by_one_unit_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wall-clock form of T7/T9's "pause <= one layer write": with every layer
    unit slowed to ~50 ms, the time from the request landing to the checkpoint
    returning is at most one unit (+ bookkeeping), not the whole write."""
    body = _body()
    sess = _sessions(body, _store(tmp_path, body))
    _turn(sess, IDS)
    real = pk._GenerationWriter._layer_unit
    unit_s = 0.05

    def slow(self: Any, i: int) -> None:
        time.sleep(unit_s)
        real(self, i)

    monkeypatch.setattr(pk._GenerationWriter, "_layer_unit", slow)
    landed: dict[str, float] = {}

    def pending() -> bool:
        if "t" not in landed and threading.current_thread() is threading.main_thread():
            landed.setdefault("t0", time.perf_counter())
        return "t" in landed

    def on_unit(_w: Any) -> None:
        if "t" not in landed:
            landed["t"] = time.perf_counter()  # request arrives right after unit 1

    t0 = time.perf_counter()
    out = sess.maybe_checkpoint(request_pending=pending, on_unit=on_unit)
    t_end = time.perf_counter()
    assert out.pending_abort and out.units == 1
    pause = t_end - landed["t"]
    assert pause < unit_s, f"request waited {pause * 1e3:.1f} ms (> one unit)"
    assert t_end - t0 < 2 * unit_s + 0.5


# ---------------------------------------------------------------------------
# two simulated ranks (G4 lockstep gather; one MLX thread per rank)
# ---------------------------------------------------------------------------

class _R:
    def __init__(self, root: Path, rank: int, ls: _Lockstep) -> None:
        self.root = root
        self.coord = ls.coordinator(rank)
        self.rank = rank
        _on(rank, self._build)

    def _build(self) -> None:
        self.body = _body()
        self.store = pk.ParkedStore(
            self.body, None, root=self.root, conv_kw=CONV_KW, min_tokens=MIN_TOKENS,
            fingerprint=_fp(layout=f"rank{self.rank}"), coordinator=None, rank=self.rank)
        # The first turn runs before the peer exists (ranks are built one at a
        # time), so it opens without the coordinator; the coordinator is then
        # attached for every checkpoint / restore exchange under test.
        self.sess = s_.Dsv41Sessions(
            self.body, None, max_seq_len=256, park_store=self.store,
            coordinator=None, rank=self.rank, **CONV_KW)
        self.conv, _ = _turn(self.sess, IDS)
        self.attach()

    def attach(self) -> None:
        self.sess._coordinator = self.coord
        self.store.coordinator = self.coord

    def detach(self) -> None:
        self.sess._coordinator = None
        self.store.coordinator = None


def _two(tmp_path: Path) -> tuple[_Lockstep, _R, _R]:
    ls = _Lockstep(2, timeout=20.0)
    return ls, _R(tmp_path / "r0", 0, ls), _R(tmp_path / "r1", 1, ls)


def test_two_ranks_checkpoint_same_generation(tmp_path: Path) -> None:
    ls, a, b = _two(tmp_path)
    outs = _both([lambda: a.sess.maybe_checkpoint(), lambda: b.sess.maybe_checkpoint()])
    assert all(o.go and o.all_committed for o in outs)
    assert outs[0].generation == outs[1].generation == 1
    assert list(_gens(a.root)) == list(_gens(b.root)) == [1]
    assert ls.calls[0] == ls.calls[1]  # rank-invariant exchange count
    assert a.conv.dirty_tokens == b.conv.dirty_tokens == 0


def test_t7_two_ranks_request_on_one_rank_aborts_both(tmp_path: Path) -> None:
    """The request lands on rank 1 only; rank 0 never sees it locally. Both
    abort at the SAME poll, no generation on either rank, no hang."""
    ls, a, b = _two(tmp_path)
    n1 = {"n": 0}

    def on_unit_b(_w: Any) -> None:
        n1["n"] += 1

    outs = _both([
        lambda: a.sess.maybe_checkpoint(request_pending=lambda: False),
        lambda: b.sess.maybe_checkpoint(request_pending=lambda: n1["n"] >= 2,
                                        on_unit=on_unit_b),
    ])
    assert all(o.go and o.pending_abort for o in outs)
    assert outs[0].units == outs[1].units == 2 and outs[0].polls == outs[1].polls
    assert not _gens(a.root) and not _gens(b.root)
    assert not _tmp_dirs(a.root) and not _tmp_dirs(b.root)
    assert ls.calls[0] == ls.calls[1]


def test_two_ranks_write_failure_on_one_rank_is_all_or_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ls, a, b = _two(tmp_path)
    assert all(o.all_committed for o in _both([
        lambda: a.sess.maybe_checkpoint(), lambda: b.sess.maybe_checkpoint()]))
    a.conv.ckpt_offset = 0
    b.conv.ckpt_offset = 0
    real = pk._GenerationWriter._layer_unit
    rank1_thread: dict[str, Any] = {}

    def flaky(self: Any, i: int) -> None:
        if self.rank == 1 and i == 1:
            rank1_thread["hit"] = True
            raise OSError(errno.EIO, "I/O error")
        real(self, i)

    monkeypatch.setattr(pk._GenerationWriter, "_layer_unit", flaky)
    outs = _both([lambda: a.sess.maybe_checkpoint(), lambda: b.sess.maybe_checkpoint()])
    assert rank1_thread.get("hit")
    assert all(o.go and o.aborted and not o.all_committed for o in outs)
    # Gen 1 survives on both; no gen 2 anywhere (never a mixed state).
    assert list(_gens(a.root)) == list(_gens(b.root)) == [1]
    assert ls.calls[0] == ls.calls[1]
    # Symmetric bookkeeping: both back off, both still dirty.
    assert a.sess.stats["ckpt_failed"] == b.sess.stats["ckpt_failed"] == 1
    assert a.conv.dirty_tokens > 0 and b.conv.dirty_tokens > 0


def test_two_ranks_candidate_mismatch_is_nogo(tmp_path: Path) -> None:
    """Resident-set divergence (should not happen; relied on, not enforced):
    the IDLE exchange sees different (key, offset) rows -> agreed no-go on BOTH
    ranks, nothing written, no further exchange."""
    ls, a, b = _two(tmp_path)

    def b_other() -> Any:
        b.detach()  # resident hit; no peer exchange needed
        conv, _ = _turn(b.sess, np.concatenate([IDS, np.arange(800, 820)]))
        b.attach()
        return conv

    _on(1, b_other)
    outs = _both([lambda: a.sess.maybe_checkpoint(), lambda: b.sess.maybe_checkpoint()])
    assert [o.go for o in outs] == [False, False]
    assert {o.reason for o in outs} == {"candidate-mismatch"}
    assert ls.calls == [1, 1]
    assert not _gens(a.root) and not _gens(b.root)


def test_two_ranks_one_rank_store_unusable_is_nogo(tmp_path: Path) -> None:
    ls, a, b = _two(tmp_path)
    b.store.enabled = False
    outs = _both([lambda: a.sess.maybe_checkpoint(), lambda: b.sess.maybe_checkpoint()])
    assert [o.go for o in outs] == [False, False] and ls.calls == [1, 1]


def test_two_ranks_restore_after_checkpoint(tmp_path: Path) -> None:
    """End to end: checkpoint on both ranks, then a fresh process (new
    sessions) on both restores the checkpoint generation (agreed, G4)."""
    ls, a, b = _two(tmp_path)
    _both([lambda: a.sess.maybe_checkpoint(), lambda: b.sess.maybe_checkpoint()])
    fresh = [
        s_.Dsv41Sessions(r.body, None, max_seq_len=256, park_store=r.store,
                         coordinator=r.coord, rank=i, **CONV_KW)
        for i, r in enumerate((a, b))
    ]
    convs = _both([lambda: fresh[0].get(IDS.tolist(), "A"),
                   lambda: fresh[1].get(IDS.tolist(), "A")])
    assert [s.stats["restored"] for s in fresh] == [1, 1]
    assert all(c.offset == a.conv.offset and c.dirty_tokens == 0 for c in convs)


# ---------------------------------------------------------------------------
# T9 -- the MLX buffer-donation trap
# ---------------------------------------------------------------------------

def _ptr(a: mx.array) -> int:
    """Host address of an evaluated array's data buffer (the memoryview and the
    ctypes view are released before returning, so this pins nothing)."""
    mv = memoryview(a)
    c = ctypes.c_char.from_buffer(mv)
    p = ctypes.addressof(c)
    del c
    mv.release()
    return p


def _big_cache(rows: int = 131072) -> tuple[StubBody, Any]:
    """A real fork ``ModelCache`` with MB-scale latent buffers (so a
    non-donated step is a measurable whole-buffer copy)."""
    body = StubBody(initial_capacity=rows, engram_layer_ids=(0,),
                    window_size=128, compress_ratios=(4, 4),
                    kv_source_layers=(0, 1), index_source_layers=(0, 1))
    conv = s_.Conversation(body, None, max_seq_len=rows, **CONV_KW)
    mc = conv.cache.cache
    mc.ensure_capacity(rows)
    mx.eval([x for lc in mc.layers for x in (lc.win_kv, lc.comp_kv, lc.index_k) if x is not None])
    return body, conv


def _decode_step(mc: Any, pos: int) -> None:
    """One decode row written exactly the way the real attention/indexer write
    it: in-place slice updates into the preallocated buffers, then an eval."""
    row = mx.full((1, 1, 512), float(pos % 251), dtype=mx.float32)
    for lc in mc.layers:
        lc.write_window(pos, row)
        if lc.comp_kv is not None and pos % lc.ratio == 0:
            g0 = pos // lc.ratio
            lc.comp_kv[:1, g0:g0 + 1] = row.astype(lc.dtype)
            if lc.index_k is not None:
                lc.index_k[:1, g0:g0 + 1] = row[:, :, : lc.index_k.shape[2]].astype(lc.dtype)
    mc.offset = pos + 1
    mx.eval([x for lc in mc.layers for x in (lc.win_kv, lc.comp_kv, lc.index_k) if x is not None])


def _ptrs(mc: Any) -> list[int]:
    return [_ptr(x) for lc in mc.layers for x in (lc.win_kv, lc.comp_kv, lc.index_k)
            if x is not None]


def _prime(conv: Any, n: int) -> None:
    mc = conv.cache.cache
    for p in range(n):
        _decode_step(mc, p)
    conv.cache._ids = np.arange(n, dtype=np.int64)
    conv.last_turn_end_ts = time.monotonic()


def test_t9_donation_survives_checkpoint_commit_and_abort(tmp_path: Path) -> None:
    """After a COMMITTED checkpoint and after an ABORTED one, every decode step
    still writes in place (buffer addresses unchanged step to step) -- the
    checkpoint holds no reference to any cache array across the next forward.
    Negative control: holding ONE reference (what a background writer would
    do) makes the very next step copy the whole buffer (address changes)."""
    body, conv = _big_cache()
    _prime(conv, 64)
    mc = conv.cache.cache
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    sess._entries["id:A"] = s_._Entry(conv)

    base = _ptrs(mc)
    _decode_step(mc, 64)
    assert _ptrs(mc) == base, "control: decode steps must donate"

    out = sess.maybe_checkpoint()
    assert out is not None and out.all_committed
    p0 = _ptrs(mc)
    for pos in range(65, 75):
        _decode_step(mc, pos)
        assert _ptrs(mc) == p0, f"donation broken after commit at step {pos}"

    conv.ckpt_offset = 0
    n_units = {"n": 0}

    def count(_w: Any) -> None:
        n_units["n"] += 1

    out = sess.maybe_checkpoint(request_pending=lambda: n_units["n"] >= 1, on_unit=count)
    assert out is not None and out.pending_abort and out.units == 1
    p1 = _ptrs(mc)
    for pos in range(75, 85):
        _decode_step(mc, pos)
        assert _ptrs(mc) == p1, f"donation broken after abort at step {pos}"

    # Negative control: a leaked slice reference (what holding a cache view
    # across the next forward pass looks like) forces a whole-buffer copy.
    leaked = mc.layers[0].comp_kv[:, :8]
    mx.eval(leaked)
    before = _ptr(mc.layers[0].comp_kv)
    _decode_step(mc, 88)  # 88 % 4 == 0 -> writes comp_kv
    assert _ptr(mc.layers[0].comp_kv) != before, "detector must see a non-donated copy"
    del leaked


def _median_step_us(mc: Any, start: int, n: int) -> float:
    ts = []
    for pos in range(start, start + n):
        t = time.perf_counter()
        _decode_step(mc, pos)
        ts.append((time.perf_counter() - t) * 1e6)
    return statistics.median(ts)


def test_t9_decode_step_time_unchanged_after_checkpoint(tmp_path: Path) -> None:
    """Timing form of T9 on the CPU harness: median decode-step time after a
    checkpoint vs before, alternating blocks; and the leaked-reference control
    to show what the regression would look like on the same harness.

    The 2% bound is asserted on the min-of-blocks medians (robust to scheduler
    noise); set ``EXO_DSV41_T9_STRICT=0`` to report-only on a noisy box. The
    authoritative check is the address-identity test above.
    """
    body, conv = _big_cache()
    _prime(conv, 64)
    mc = conv.cache.cache
    sess = _sessions(body, _store(tmp_path, body))
    sess._entries["id:A"] = s_._Entry(conv)
    pos = 64
    for _ in range(50):  # warm up
        _decode_step(mc, pos)
        pos += 1
    before, after = [], []
    for _blk in range(5):
        before.append(_median_step_us(mc, pos, 80))
        pos += 80
        conv.ckpt_offset = -1
        conv.cache._ids = np.arange(pos, dtype=np.int64)
        assert sess.maybe_checkpoint().all_committed
        after.append(_median_step_us(mc, pos, 80))
        pos += 80
    def leak() -> list[Any]:
        # What a background writer holding views across the next forward pass
        # looks like: one live slice of every cache buffer.
        held = [x[:, :8] for lc in mc.layers
                for x in (lc.win_kv, lc.comp_kv, lc.index_k) if x is not None]
        mx.eval(held)
        return held

    leaked = leak()
    leak_ts = []
    for _ in range(20):
        t = time.perf_counter()
        _decode_step(mc, pos - pos % 4)  # comp_kv/index_k write each step
        leak_ts.append((time.perf_counter() - t) * 1e6)
        leaked = leak()  # keep re-leaking (a held view)
        pos += 4
    del leaked
    b, a, lk = min(before), min(after), statistics.median(leak_ts)
    print(f"\nT9 decode-step median: before={b:.1f}us after={a:.1f}us "
          f"ratio={a / b:.4f}; leaked-ref control={lk:.1f}us ({lk / b:.1f}x)")
    assert lk > 2 * b, "the harness must be able to see a donation regression"
    if os.environ.get("EXO_DSV41_T9_STRICT", "1") != "0":
        assert a <= b * 1.02 + 2.0, f"decode step slowed after checkpoint: {a:.1f} vs {b:.1f} us"


# ---------------------------------------------------------------------------
# engine idle tick + runner hook
# ---------------------------------------------------------------------------

_CAND: Any = ("id:A", object())


class _StubSessions:
    def __init__(self, cand: Any = _CAND) -> None:
        self.cand = cand
        self.calls: list[Any] = []
        self.ckpt_idle_s = 0.0

    def checkpoint_candidate(self) -> Any:
        return self.cand

    def maybe_checkpoint(self, *, request_pending: Any = None, candidate: Any = None) -> Any:
        self.calls.append((request_pending(), candidate))
        return "ran"


def _bare_engine() -> Any:
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine

    eng = object.__new__(Dsv41Engine)
    eng._active = None
    eng._agreement = ag.RankAgreement(None)
    eng._sessions = _StubSessions()
    eng._last_step_monotonic = 0.0
    return eng


def test_engine_idle_tick_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(s_.CKPT_ENV)
    eng = _bare_engine()
    assert eng.idle_tick_seconds() is None
    monkeypatch.setenv(s_.CKPT_ENV, "1")
    monkeypatch.setenv(s_.CKPT_TICK_S_ENV, "0.5")
    assert eng.idle_tick_seconds() == 0.5


def test_engine_idle_tick_gating() -> None:
    eng = _bare_engine()
    assert eng.idle_tick(lambda: False) == "ran"
    assert eng._sessions.calls[-1][0] is False
    # Active request / queued / submitted-not-agreed -> no work, no collective.
    for setup in (lambda e: setattr(e, "_active", object()),
                  lambda e: e._agreement.queue.append(object()),
                  lambda e: e._agreement.maybe_queue.append(object())):
        e = _bare_engine()
        setup(e)
        assert e.idle_tick(lambda: False) is None and not e._sessions.calls
    # No candidate -> nothing.
    e = _bare_engine()
    e._sessions.cand = None
    assert e.idle_tick(lambda: False) is None
    # Served a request too recently -> nothing.
    e = _bare_engine()
    e._sessions.ckpt_idle_s = 60.0
    e._last_step_monotonic = time.monotonic()
    assert e.idle_tick(lambda: False) is None and not e._sessions.calls
    # request_pending is forwarded (polled between units).
    e = _bare_engine()
    e.idle_tick(lambda: True)
    assert e._sessions.calls[-1][0] is True


def test_idle_tick_pairs_with_peer_in_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """The central cross-rank risk. Rank 1 already has a task (it is in
    ``step()``), rank 0 has not received it yet and runs an idle tick. Rank 0's
    ``agree_on_tasks`` pairs with rank 1's (same collectives, same order), the
    task is NOT agreed (rank 0 lacks it), rank 0 stands down WITHOUT issuing
    any checkpoint exchange, and once rank 0 receives the task both agree on it
    in the next ``step()``-side ``agree_on_tasks``."""
    barrier = threading.Barrier(2, timeout=10.0)
    slots: list[Any] = [None, None]
    log: list[list[str]] = [[], []]

    def gather(rank: int, value: Any, tag: str) -> list[Any]:
        log[rank].append(tag)
        slots[rank] = value
        barrier.wait()
        out = list(slots)
        barrier.wait()
        return out

    tls = threading.local()

    def fake_any(b: bool, group: Any) -> bool:
        return any(gather(tls.rank, bool(b), "any"))

    def fake_gather_tasks(tasks: list[Any], group: Any) -> tuple[list[Any], list[Any]]:
        rows = gather(tls.rank, [t.task_id for t in tasks], "tasks")
        common = set(rows[0]) & set(rows[1])
        agreed = sorted((t for t in tasks if t.task_id in common), key=lambda t: t.task_id)
        return agreed, [t for t in tasks if t.task_id not in common]

    monkeypatch.setattr(ag, "mx_any", fake_any)
    monkeypatch.setattr(ag, "mx_all_gather_tasks", fake_gather_tasks)

    class _T:
        def __init__(self, tid: str) -> None:
            self.task_id = tid

    e0, ag1 = _bare_engine(), ag.RankAgreement(object())
    e0._agreement = ag.RankAgreement(object())
    ag1.submit(_T("t1"))  # type: ignore[arg-type]
    res: list[Any] = [None, None]

    def r0() -> None:
        tls.rank = 0
        res[0] = e0.idle_tick(lambda: False)

    def r1() -> None:
        tls.rank = 1
        res[1] = ag1.agree_on_tasks()  # what step() does with no active request

    ts = [threading.Thread(target=r0), threading.Thread(target=r1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert not any(t.is_alive() for t in ts), "a rank hung"
    assert res == [None, True]
    assert not e0._sessions.calls, "rank 0 must not start a checkpoint"
    assert log == [["any", "tasks"], ["any", "tasks"]]
    assert not ag1.queue and len(ag1.maybe_queue) == 1  # not agreed yet
    # Rank 0 now receives the task; both step() -> agreed on both.
    e0._agreement.submit(_T("t1"))  # type: ignore[arg-type]

    def s0() -> None:
        tls.rank = 0
        e0._agreement.agree_on_tasks()

    def s1() -> None:
        tls.rank = 1
        ag1.agree_on_tasks()

    ts = [threading.Thread(target=s0), threading.Thread(target=s1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert [t.task_id for t in e0._agreement.queue] == [t.task_id for t in ag1.queue] == ["t1"]


def test_runner_idle_wait_unchanged_when_off_and_ticks_when_on() -> None:
    from exo.shared.types.worker.runners import RunnerReady
    from exo.worker.runner.runner import Runner

    class _Eng:
        def __init__(self, tick: float | None) -> None:
            self.tick = tick
            self.ticks: list[bool] = []

        def idle_tick_seconds(self) -> float | None:
            return self.tick

        def idle_tick(self, request_pending: Any = None) -> None:
            self.ticks.append(request_pending())

    from exo.worker.engines.base import Engine

    Engine.register(_Eng)  # virtual subclass: isinstance(gen, Engine)
    r = object.__new__(Runner)
    r._work_queue = queue.Queue()
    r.current_status = RunnerReady(prefill_server_port=None)
    # OFF: plain blocking get (an item already queued returns immediately).
    r.generator = _Eng(None)
    r._work_queue.put("item")
    assert r._next_work_item() == "item" and r.generator.ticks == []
    # ON: timeout -> one idle tick on THIS thread, then None.
    r.generator = _Eng(0.01)
    assert r._next_work_item() is None and r.generator.ticks == [False]
    r._work_queue.put("x")
    assert r._next_work_item() == "x"


# ---------------------------------------------------------------------------
# hardening: independent adversarial audit (HIGH-1 / MED-1 / MED-2)
# ---------------------------------------------------------------------------

def _gen_dir_count(root: Path) -> int:
    return len([d for d in root.iterdir()
                if pk._gen_from_name(d.name) is not None]) if root.is_dir() else 0


def _fresh_store(root: Path) -> pk.ParkedStore:
    return pk.ParkedStore(_body(), None, root=root, conv_kw=CONV_KW,
                          min_tokens=MIN_TOKENS, fingerprint=_fp())


def test_audit_h1_budget_abort_keeps_previous_generation(tmp_path: Path) -> None:
    """HIGH-1: an aborted checkpoint under byte-budget pressure must never delete
    the conversation's OWN earlier generation.

    Mutation caught: pre-fix ``_reserve_headroom`` protected only dirs whose name
    prefix equals the CURRENT stream's ``prefix_hash``. A grown conversation's
    previous generation lives under an EARLIER (shorter) stream's hash, so the
    headroom pass deleted it before the new generation existed; the abort then
    left ZERO generations on disk (data loss). With
    ``protect=self._protect_own(conv, key)`` the previous generation survives the
    abort, byte-unchanged, and still restores.
    """
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    conv, _ = _turn(sess, IDS)
    assert sess.maybe_checkpoint().all_committed       # gen 1 = the only generation
    off1 = conv.offset
    prev = _gens(tmp_path)
    assert set(prev) == {1}
    prev_files = {p.name: p.read_bytes() for p in prev[1].iterdir()}
    # Room for the previous generation plus ONE new write, but NOT the full 2x
    # headroom: headroom eviction MUST fire.
    store.max_bytes = int(pk.park_dir_size(prev[1]) * 2.5)
    # Grow the conversation (a new turn, a new prefix hash) and abort mid-write.
    ids2 = np.concatenate([np.asarray(conv.tokens), np.arange(500, 520)])
    _turn(sess, ids2)
    n = {"n": 0}

    def on_unit(_w: Any) -> None:
        n["n"] += 1

    out = sess.maybe_checkpoint(request_pending=lambda: n["n"] >= 1, on_unit=on_unit)
    assert out.go and out.pending_abort and not out.all_committed
    assert not _tmp_dirs(tmp_path)
    # The conversation's own previous generation SURVIVED the abort, unchanged...
    assert set(_gens(tmp_path)) == {1}
    assert {p.name: p.read_bytes() for p in prev[1].iterdir()} == prev_files
    # ...and still restores.
    restored = _fresh_store(tmp_path).try_restore(IDS.tolist(), key="A")
    assert restored is not None and restored.offset == off1


def test_audit_med1_same_offset_branch_change_is_dirty_and_rewritten(
    tmp_path: Path,
) -> None:
    """MED-1: a rewind+refeed back to the SAME offset on a different branch must
    be seen as dirty and rewritten, not believed clean.

    Mutation caught: pre-fix ``dirty_tokens = offset - ckpt_offset`` was ``0`` for
    a same-offset branch change, so ``checkpoint_candidate`` returned ``None`` and
    eviction's clean-skip never rewrote the generation: ``ids.bin`` kept the
    pre-rewind branch. ``stream_changed`` (the checkpointed stream's hash) now
    flags it, the candidate is offered, the rewrite commits, and a restart
    restores the NEW branch.
    """
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    conv, _ = _turn(sess, IDS)
    assert sess.maybe_checkpoint().all_committed
    off = conv.offset
    before_tokens = list(conv.tokens)
    # Rewind into the prompt, then a DIFFERENT tail of equal total length -> the
    # SAME final offset, a different branch.
    ids2 = np.concatenate([IDS[:60], np.arange(1000, 1000 + (len(before_tokens) - 60) - 5)])
    conv2, _ = _turn(sess, ids2)
    assert conv2 is conv and conv.offset == off
    assert list(conv.tokens) != before_tokens
    # The offset-only test said "clean"; the stream identity says otherwise.
    assert conv.dirty_tokens == 0 and conv.stream_changed
    cand = sess.checkpoint_candidate()
    assert cand is not None and cand[1] is conv
    # The rewrite commits the NEW branch...
    assert sess.maybe_checkpoint().all_committed
    assert not conv.stream_changed
    # ...and a restart restores the new branch, not the stale one.
    restored = _fresh_store(tmp_path).try_restore(ids2.tolist(), key="A")
    assert restored is not None
    assert list(restored.tokens) == list(conv.tokens)


def test_audit_med2_keyed_restore_newest_and_bounded(tmp_path: Path) -> None:
    """MED-2: a keyed restore returns the NEWEST generation (not arbitrary), and
    a growing keyed conversation's dirs stay bounded by ``_KEEP_GENERATIONS``.

    Mutation caught: pre-fix key-mode ``find`` broke ties by generation alone
    (every turn produced a fresh ``prefix_hash`` with generation 1), so the
    restore picked the FIRST dir by name -- offset 152 when the newest was 177 --
    and ``prune_acked`` / ``_KEEP_GENERATIONS`` (keyed on ``prefix_hash``) never
    pruned, so one dir accumulated per turn. Now key-mode ordering is by stored
    stream LENGTH and ``retain_keyed`` keeps the newest ``_KEEP_GENERATIONS``.
    """
    body = _body()
    store = _store(tmp_path, body)
    sess = _sessions(body, store)
    ids = IDS
    offs: list[int] = []
    conv = None
    for i in range(5):
        conv, _ = _turn(sess, ids)
        assert sess.maybe_checkpoint().all_committed
        offs.append(conv.offset)
        ids = np.concatenate([np.asarray(conv.tokens), np.arange(500 + 40 * i, 520 + 40 * i)])
    assert offs == sorted(offs) and offs[-1] > offs[0]
    # Accumulation is bounded to _KEEP_GENERATIONS dirs for one conversation.
    assert _gen_dir_count(tmp_path) <= pk._KEEP_GENERATIONS
    # A fresh single-rank restore returns the NEWEST generation.
    restored = _fresh_store(tmp_path).try_restore(ids.tolist(), key="A")
    assert restored is not None and restored.offset == offs[-1]


def test_audit_med2_two_ranks_keyed_restore_newest(tmp_path: Path) -> None:
    """MED-2 (cross-rank): after several keyed checkpoints the G4 agreed restore
    returns the NEWEST generation on BOTH ranks, and retention is bounded."""
    ls, a, b = _two(tmp_path)
    offs: list[int] = []
    ids = IDS
    for i in range(4):
        outs = _both([lambda: a.sess.maybe_checkpoint(),
                      lambda: b.sess.maybe_checkpoint()])
        assert all(o.go and o.all_committed for o in outs)
        assert outs[0].generation == outs[1].generation
        offs.append(a.conv.offset)
        ids = np.concatenate([np.asarray(a.conv.tokens), np.arange(500 + 40 * i, 520 + 40 * i)])
        _both([lambda ids=ids: _turn(a.sess, ids),
               lambda ids=ids: _turn(b.sess, ids)])
    assert offs == sorted(offs) and offs[-1] > offs[0]
    assert ls.calls[0] == ls.calls[1]
    for r in (a, b):
        assert _gen_dir_count(r.root) <= pk._KEEP_GENERATIONS
    fresh = [
        s_.Dsv41Sessions(r.body, None, max_seq_len=256, park_store=r.store,
                         coordinator=r.coord, rank=i, **CONV_KW)
        for i, r in enumerate((a, b))
    ]
    convs = _both([lambda: fresh[0].get(ids.tolist(), "A"),
                   lambda: fresh[1].get(ids.tolist(), "A")])
    assert [s.stats["restored"] for s in fresh] == [1, 1]
    assert [c.offset for c in convs] == [offs[-1], offs[-1]]

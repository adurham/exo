# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnusedFunction=false, reportMissingModuleSource=false, reportOptionalMemberAccess=false
"""CPU-only tests for the DSv4.1 reuse checkpoint LADDER.

The defect under test (confirmed live): ``SessionCache.plan`` rewinds a reused
prompt to the NEWEST checkpoint at or below its longest common prefix, and
checkpoints used to exist only at offset 0 and turn ends. A follow-up prompt
that undershot the newest turn end by 1-2 rows (a BPE seam) therefore rewound
all the way to 0 and re-fed the whole context (a 30K-token conversation re-fed
30007 rows, 112.5 s instead of seconds).

This module pins the fix on the exo side, over the shared stub harness (no GPU,
no checkpoint): a periodic checkpoint ladder plus an end-anchored margin rung,
fed from the per-chunk tap path so every rung is a CONSISTENT (body, draft)
pair; the decode-side cadence; retention; and the loud undershoot warning.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import engine as eng_mod
from exo.worker.engines.mlx.dsv41 import session as s_

from ._stub_harness import STEPS, StubBody, StubHead

#: The three ladder env knobs; the autouse fixture clears them so a developer's
#: shell cannot change a test's behaviour (explicit kwargs win anyway).
_ENV_KEYS = (
    s_.CHECKPOINT_SPACING_ENV,
    s_.CHECKPOINT_MARGIN_ENV,
    s_.CHECKPOINT_KEEP_ENV,
)


@pytest.fixture(autouse=True, name="clean_ladder_env")
def _clean_ladder_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


class _FakeLogger:
    """Records (level, message) pairs instead of touching the loguru singleton."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def _record(self, level: str, message: str) -> None:
        self.messages.append((level, message))

    def info(self, message: str, *_a: object, **_k: object) -> None:
        self._record("INFO", message)

    def warning(self, message: str, *_a: object, **_k: object) -> None:
        self._record("WARNING", message)

    def critical(self, message: str, *_a: object, **_k: object) -> None:
        self._record("CRITICAL", message)

    def of(self, level: str) -> list[str]:
        return [m for lvl, m in self.messages if lvl == level]


@contextmanager
def _capture(module: Any) -> Generator[_FakeLogger]:
    cap = _FakeLogger()
    original = module.logger
    module.logger = cap
    try:
        yield cap
    finally:
        module.logger = original


def _draft_body() -> StubBody:
    """Stub body with a draft target so the per-chunk tap path really feeds."""
    return StubBody(initial_capacity=4, dspark_target_layer_ids=(0,))


def _store(body: Any, head: Any | None = None, **kw: Any) -> s_.Dsv41Sessions:
    return s_.Dsv41Sessions(
        body, head, max_seq_len=1024, chunk=4, long_chunk=4, long_threshold=10**9,
        **kw,
    )


def _prefill(store: s_.Dsv41Sessions, total: int, key: str = "conv") -> s_.Conversation:
    conv = store.get(list(range(1, total + 1)), key)
    conv.prefill(np.arange(1, total + 1, dtype=np.int64))
    return conv


# ---------------------------------------------------------------------------
# (1) grid cadence
# ---------------------------------------------------------------------------


def test_cadence_fires_at_spacing() -> None:
    """With spacing 4 and 4-row chunks, a rung lands at every chunk boundary."""
    store = _store(StubBody(), checkpoint_spacing=4, checkpoint_margin=0)
    conv = _prefill(store, 40)

    assert conv.cache.boundaries == list(range(0, 41, 4)), conv.cache.boundaries
    # snapshots really grew (not just the boundary list)
    assert len(conv.cache._snaps) == len(conv.cache.boundaries)
    assert conv.offset == 40


def test_cadence_off_when_spacing_is_zero() -> None:
    """spacing=0 disables the grid ladder: only the prompt-end checkpoint."""
    store = _store(StubBody(), checkpoint_spacing=0, checkpoint_margin=0)
    conv = _prefill(store, 40)
    assert conv.cache.boundaries == [0, 40], conv.cache.boundaries


def test_cadence_does_not_fire_before_spacing() -> None:
    """Under ``spacing`` rows the newest boundary is the only one there is."""
    store = _store(StubBody(), checkpoint_spacing=64, checkpoint_margin=0)
    conv = _prefill(store, 40)
    assert conv.cache.boundaries == [0, 40], conv.cache.boundaries


# ---------------------------------------------------------------------------
# (2) end-anchored margin rung
# ---------------------------------------------------------------------------


def test_margin_rung_fires_within_margin_of_end() -> None:
    """A rung is placed within ``margin`` rows of the delta end."""
    store = _store(StubBody(), checkpoint_spacing=10**9, checkpoint_margin=8)
    conv = _prefill(store, 61)  # 4-row chunks -> boundaries at multiples of 4

    total = conv.offset
    below = [b for b in conv.cache.boundaries if b < total]
    assert below, conv.cache.boundaries
    # the newest rung before the prompt-end checkpoint sits in the margin window
    assert max(below) >= total - 8, (max(below), total, conv.cache.boundaries)
    assert conv.cache.boundaries[-1] == total  # the prompt-end checkpoint


def test_margin_rung_skipped_for_tiny_calls() -> None:
    """A call no longer than ``margin`` keeps just the offset-0/prompt-end pair."""
    store = _store(StubBody(), checkpoint_spacing=10**9, checkpoint_margin=64)
    conv = _prefill(store, 40)  # total 40 <= margin 64
    assert conv.cache.boundaries == [0, 40], conv.cache.boundaries


def test_margin_rung_skipped_for_exact_hit() -> None:
    """An exact repeat feeds zero rows: no new rung, anchor from the checkpoint."""
    store = _store(StubBody(), checkpoint_spacing=10**9, checkpoint_margin=8)
    _prefill(store, 61)
    conv = store.get(list(range(1, 62)), "conv")
    before = list(conv.cache.boundaries)
    r2 = conv.prefill(np.arange(1, 62, dtype=np.int64))
    assert r2.prefill_tokens == 0 and r2.reused_tokens == 61
    assert conv.cache.boundaries == before, "an exact hit must add no rung"


# ---------------------------------------------------------------------------
# (3) per-chunk tap feeding keeps the draft/body lockstep at every rung
# ---------------------------------------------------------------------------


def test_per_chunk_feeding_keeps_draft_ctx_equal_offset_at_every_checkpoint() -> None:
    """ORDER IS LOAD-BEARING: at each checkpoint draft_ctx == offset."""
    body = _draft_body()
    store = _store(body, StubHead(), checkpoint_spacing=4, checkpoint_margin=0)
    conv = store.get(list(range(1, 41)), "conv")

    seen: list[tuple[int, int]] = []
    original = conv._checkpoint

    def spy() -> None:
        seen.append((conv.offset, conv.draft_ctx()))
        original()

    conv._checkpoint = spy
    conv.prefill(np.arange(1, 41, dtype=np.int64))

    assert seen, "no checkpoint ran"
    assert all(off == ctx for off, ctx in seen), seen
    assert conv.draft_ctx() == conv.offset == 40


def test_draft_window_ends_even_with_the_cache_across_multiple_chunks() -> None:
    """The per-chunk feed leaves the draft window at ``offset`` (no double-apply)."""
    body = _draft_body()
    store = _store(body, StubHead(), checkpoint_spacing=64, checkpoint_margin=0)
    conv = _prefill(store, 40)
    assert conv.draft_ctx() == conv.offset == 40


def test_no_draft_head_feeds_nothing_and_stays_consistent() -> None:
    """Without a head the callback is a no-op; the body-only path still ladders."""
    store = _store(StubBody(), None, checkpoint_spacing=8, checkpoint_margin=0)
    conv = _prefill(store, 40)
    assert conv.draft_ctx() == -1
    assert conv.cache.boundaries == list(range(0, 41, 8)), conv.cache.boundaries


# ---------------------------------------------------------------------------
# (4) dedupe + lockstep guard
# ---------------------------------------------------------------------------


def test_no_snapshot_when_offset_is_already_a_boundary() -> None:
    """Dedupe: a boundary already at ``offset`` must not be re-snapshotted."""
    store = _store(StubBody(), checkpoint_spacing=4, checkpoint_margin=0)
    conv = _prefill(store, 40)
    n_before = len(conv.cache._snaps)

    # 40 is a boundary; a forced cadence at the same row must be a no-op.
    assert conv.maybe_checkpoint(force=True) is False
    assert conv.maybe_checkpoint(force=True) is False
    assert len(conv.cache._snaps) == n_before
    assert conv.cache.boundaries.count(40) == 1


def test_refeed_re_crossing_a_rung_dedupes() -> None:
    """A refeed that re-crosses an existing rung adds no duplicate snapshot."""
    store = _store(StubBody(), checkpoint_spacing=4, checkpoint_margin=0)
    conv = _prefill(store, 40)
    # A divergent turn whose LCP lands between rungs rewinds to the newest rung
    # and re-feeds the tail; every rung it re-crosses already exists.
    divergent = np.concatenate([
        np.arange(1, 33, dtype=np.int64), np.arange(500, 532, dtype=np.int64),
    ])
    conv.prefill(divergent)
    # boundaries are unique and sorted
    assert conv.cache.boundaries == sorted(set(conv.cache.boundaries))


def test_lockstep_violation_skips_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drift between draft and body SKIPS the checkpoint and warns ONCE."""
    body = _draft_body()
    store = _store(body, StubHead(), checkpoint_spacing=4, checkpoint_margin=0)
    conv = _prefill(store, 40)
    # Move the offset OFF a boundary (a prompt end always is one) so the guard
    # under test -- the lockstep check, not the dedupe -- is the one that trips.
    conv.model(mx.array([[99]], dtype=mx.int32), conv.cache.cache,
               last_logit_only=True)
    conv.mark_rows([99])
    before = list(conv.cache.boundaries)
    n_before = len(conv.cache._snaps)
    assert conv.offset not in conv.cache.boundaries

    # Corrupt the draft window's row count by one (simulates a drifted feed).
    assert conv.draft_state is not None
    conv.draft_state[0].n_ctx = conv.offset - 1

    with _capture(s_) as cap:
        assert conv.maybe_checkpoint(force=True) is False
        assert conv.maybe_checkpoint(force=True) is False

    warns = [w for w in cap.of("WARNING") if "checkpoint cadence skipped" in w]
    assert len(warns) == 1, cap.messages
    assert list(conv.cache.boundaries) == before
    assert len(conv.cache._snaps) == n_before


# ---------------------------------------------------------------------------
# (5) decode-side ladder
# ---------------------------------------------------------------------------


def _engine_with_store(store: s_.Dsv41Sessions) -> eng_mod.Dsv41Engine:
    from exo.shared.types.common import ModelId
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded

    body = store.model
    loaded = Dsv41Loaded(
        model=body, tokenizer=None, args=body.args, model_path=None,  # type: ignore[arg-type]
        built_layers=[0], full_stack=False, rank=0, world=1, load_seconds=0.0,
        head=None,
    )

    class _Sender:
        def send(self, item: object) -> None:
            pass

    class _Receiver:
        def collect(self) -> list[object]:
            return []

    engine = eng_mod.Dsv41Engine(
        loaded=loaded, model_id=ModelId("x/y"), group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=_Sender(),  # type: ignore[arg-type]
        device_rank=0, speculative=False, prefill_chunk_size=4, max_kv_tokens=1024,
    )
    # Drive the rounds against our own conversation store.
    engine._sessions = store
    return engine


def test_decode_side_cadence_fires_at_round_boundaries() -> None:
    """The decode loop checkpoints at round boundaries (spacing reached)."""
    store = _store(StubBody(), checkpoint_spacing=2, checkpoint_margin=0)
    conv = _prefill(store, 8)
    assert conv.cache.boundaries == [0, 4, 8], conv.cache.boundaries

    engine = _engine_with_store(store)
    anchor = int(np.asarray(conv.cache.last_output).reshape(-1)[-1])
    for _batch, _lps in engine._rounds(conv, anchor, 6):
        pass
    # Rounds each feed one row; with spacing 2 a rung appears at 10 and 12.
    assert conv.offset == 8 + STEPS
    after = conv.cache.boundaries
    assert max(after) > 8, after
    assert all(b == 0 or b >= 4 for b in after), after


def test_decode_side_cadence_respects_dedupe() -> None:
    """A decode rung is not re-added when offset is already a boundary."""
    store = _store(StubBody(), checkpoint_spacing=1, checkpoint_margin=0)
    conv = _prefill(store, 4)  # offset 4, boundary at 4
    engine = _engine_with_store(store)
    anchor = int(np.asarray(conv.cache.last_output).reshape(-1)[-1])
    boundaries_before = list(conv.cache.boundaries)
    for _batch, _lps in engine._rounds(conv, anchor, 2):
        pass
    assert len(conv.cache.boundaries) == len(set(conv.cache.boundaries))
    assert conv.cache.boundaries[: len(boundaries_before)] == boundaries_before


# ---------------------------------------------------------------------------
# (6) loud undershoot warning
# ---------------------------------------------------------------------------


def test_undershoot_warning_fires_on_a_large_refeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reuse that still re-fed > 256 rows logs a WARNING under ``_start_turn``."""
    # Rungs every 64 rows so a near-seam undershoot rewinds to a recent rung
    # rather than 0 -- and still re-feeds more than 256 rows here.
    monkeypatch.setenv(s_.CHECKPOINT_SPACING_ENV, "64")
    monkeypatch.setenv(s_.CHECKPOINT_MARGIN_ENV, "64")
    store = _store(StubBody())
    engine = _engine_with_store(store)

    p1 = list(range(1, 513))
    conv = store.get(p1, "conv")
    engine._start_turn(conv, p1, embeddings=None, image_span_end=0)

    # LCP N-100 -> newest rung at/below that, then a 400-row tail: prefill > 256.
    p2 = p1[: len(p1) - 100] + list(range(2000, 2400))
    with _capture(eng_mod) as cap:
        turn, _anchor = engine._start_turn(conv, p2, embeddings=None, image_span_end=0)

    assert turn.reused_tokens > 0, turn
    assert turn.prefill_tokens > 256, turn
    warns = [w for w in cap.of("WARNING") if "reuse undershoot" in w]
    assert warns, cap.messages
    assert f"refed={turn.prefill_tokens}" in warns[-1]


def test_undershoot_warning_silent_for_a_small_refeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine small delta does not warn (the warning is for collapse only)."""
    monkeypatch.setenv(s_.CHECKPOINT_SPACING_ENV, "64")
    monkeypatch.setenv(s_.CHECKPOINT_MARGIN_ENV, "64")
    store = _store(StubBody())
    engine = _engine_with_store(store)

    p1 = list(range(1, 513))
    conv = store.get(p1, "conv")
    engine._start_turn(conv, p1, embeddings=None, image_span_end=0)

    p2 = p1[: len(p1) - 100] + list(range(2000, 2010))  # 10-row tail
    with _capture(eng_mod) as cap:
        turn, _anchor = engine._start_turn(conv, p2, embeddings=None, image_span_end=0)

    assert turn.reused_tokens > 0
    assert turn.prefill_tokens <= 256, turn
    assert not [w for w in cap.of("WARNING") if "reuse undershoot" in w], cap.messages


# ---------------------------------------------------------------------------
# (7) retention
# ---------------------------------------------------------------------------


def test_keep_is_forwarded_and_caps_retention() -> None:
    """``max_snapshots`` reaches the SessionCache and caps the retained set."""
    store = _store(StubBody(), checkpoint_spacing=4, checkpoint_margin=0,
                   max_snapshots=3)
    conv = _prefill(store, 40)
    assert conv.cache.max_snapshots == 3
    assert len(conv.cache._snaps) <= 3, conv.cache.boundaries


def test_keep_env_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(s_.CHECKPOINT_KEEP_ENV, "5")
    store = _store(StubBody(), checkpoint_spacing=4, checkpoint_margin=0)
    conv = _prefill(store, 40)
    assert conv.cache.max_snapshots == 5
    assert len(conv.cache._snaps) <= 5


def test_cadence_params_reach_the_conversation() -> None:
    store = _store(StubBody(), checkpoint_spacing=32, checkpoint_margin=16,
                   max_snapshots=7)
    conv = _prefill(store, 8)
    assert conv._spacing == 32
    assert conv._margin == 16
    assert conv.cache.max_snapshots == 7


# ---------------------------------------------------------------------------
# (8) exactness: rewind-to-a-rung + refeed == a straight full feed
# ---------------------------------------------------------------------------


def _bits(a: Any) -> "np.ndarray":
    """Exact numpy view of a buffer (bf16 via uint16, so bits are compared)."""
    return np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a)


def test_rewind_to_rung_refeed_matches_full_feed() -> None:
    """A rung refeed reproduces the full feed's state and next-token output.

    Laptop-level part of the bitwise gate: the stub body writes position-derived
    bytes, so if the rewind landed on the wrong row, or the refeed skipped a
    row, the final buffers would differ. Compares offset, token view, every live
    per-layer buffer, the engram ids, and the model's answer for the next row.
    """
    p1 = np.arange(10, 82, dtype=np.int64)  # 72 rows, rungs every 8
    tail = np.arange(500, 560, dtype=np.int64)
    final = np.concatenate([p1[:40], tail])  # diverges at row 40 -> rewind

    def run_reuse() -> s_.Conversation:
        store = _store(StubBody(engram_layer_ids=(0,)), checkpoint_spacing=8,
                       checkpoint_margin=8)
        conv = _prefill(store, 72)
        assert any(b <= 40 for b in conv.cache.boundaries), conv.cache.boundaries
        conv.prefill(final)
        return conv

    def run_cold() -> s_.Conversation:
        store = _store(StubBody(engram_layer_ids=(0,)), checkpoint_spacing=8,
                       checkpoint_margin=8)
        conv = store.get(list(final), "cold")
        conv.prefill(final)
        return conv

    a = run_reuse()
    b = run_cold()

    assert a.offset == b.offset == len(final)
    assert [int(t) for t in a.tokens] == [int(t) for t in b.tokens]
    # Live per-layer buffers must be bit-equal (position-addressed rings).
    for la, lb in zip(a.cache.cache.layers, b.cache.cache.layers, strict=True):
        assert np.array_equal(_bits(la.win_kv), _bits(lb.win_kv))
        if la.comp_kv is not None:
            u = -(-a.offset // max(int(la.ratio or 0), 1))
            assert np.array_equal(_bits(la.comp_kv[:, :u]), _bits(lb.comp_kv[:, :u]))
            assert np.array_equal(_bits(la.index_k[:, :u]), _bits(lb.index_k[:, :u]))
    assert np.array_equal(_bits(a.cache.cache.engram_ids),
                          _bits(b.cache.cache.engram_ids))
    # Same answer for the next row (the decode loop's first step).
    nxt = mx.array([[900]], dtype=mx.int32)
    oa = np.asarray(a.model(nxt, a.cache.cache, last_logit_only=True))
    ob = np.asarray(b.model(nxt, b.cache.cache, last_logit_only=True))
    assert np.array_equal(oa, ob)


def test_refeed_after_rewind_reproduces_tokens_and_offset() -> None:
    """cancel() -> refeed still reproduces (the ladder did not break rollback)."""
    store = _store(StubBody(), checkpoint_spacing=8, checkpoint_margin=8)
    conv = _prefill(store, 72)
    tail = np.concatenate([np.arange(1, 41, dtype=np.int64),
                           np.arange(700, 740, dtype=np.int64)])
    conv.prefill(tail)
    base = conv._base
    assert conv.cancel() > 0
    # The ladder makes the turn START a rung, so cancel rolls back there and a
    # rerun re-feeds exactly the discarded tail.
    assert conv.offset == base
    # No orphan draft/anchor snapshots above the rewind target (hazard 5).
    live = set(conv.cache.boundaries)
    assert set(conv._draft_snaps) <= live, (sorted(conv._draft_snaps), sorted(live))
    assert set(conv._anchor_at) <= live, (sorted(conv._anchor_at), sorted(live))
    conv.prefill(tail)
    assert conv.offset == len(tail)

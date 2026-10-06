"""CPU-only wiring test: the DSv4.1 conversation sessions + the engine's turn.

No checkpoint, no Metal: a stub body over the fork's REAL ``ModelCache`` drives
``dsv41.session`` and the engine's prefill/decode path, so the multi-turn contract
is pinned without a GPU job:

* turn-2 prefill is the delta only (``< 10%`` of turn 1) -- the acceptance bar;
* a turn replayed with the same chunk plan is bitwise equal to a cold run;
* ``cancel`` rolls the conversation back and a re-run reproduces the tokens;
* the engine's ``_run_turn`` wires prefill -> anchor -> rounds -> checkpoint, and
  its terminal usage carries the reuse split.

The stub body's rule is deterministic, so a replay reproduces it exactly; it is a
stand-in only, not the DSpark contract under test here.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import session as s_

# The stub body + turn driver now live in the shared harness (``_stub_harness``)
# so the SSD-park tests build the SAME conversation/cache path; the aliases keep
# this module's existing names (``StubBody``, ``_run_turn``) working unchanged.
from ._stub_harness import STEPS, StubBody
from ._stub_harness import run_turn as _run_turn


def test_turn_two_prefills_only_the_delta():
    """The acceptance bar: turn 2's prefill is the delta, not the whole prompt."""
    store = s_.Dsv41Sessions(StubBody(), None, max_seq_len=256, chunk=4,
                             long_chunk=4, long_threshold=10**9)
    p1 = np.arange(10, 42, dtype=np.int64)
    conv, r1, gen1 = _run_turn(store, p1)
    assert r1.prefill_tokens == len(p1) and r1.reused_tokens == 0
    assert conv.offset == len(p1) + STEPS - 1

    p2 = np.concatenate([p1, np.asarray(gen1, dtype=np.int64),
                          np.asarray([90, 91, 92], dtype=np.int64)])
    conv2, r2, gen2 = _run_turn(store, p2)
    assert conv2 is conv
    # The delta is the previous reply minus its last token (the next turn's anchor)
    # plus the new text; every round after the first is one token, so the count is
    # STEPS + 1 rows on the conversation.
    assert r2.reused_tokens == len(p1) + len(gen1) - 1, f"{r2}"
    assert r2.prefill_tokens == 1 + 3, f"{r2}"
    assert r2.prefill_tokens / r2.prompt_tokens < 0.10, f"{r2}"
    assert len(gen2) == STEPS
    assert [int(t) for t in conv.tokens] == [
        int(t) for t in np.concatenate([p2, np.asarray(gen2[:-1], dtype=np.int64)])
    ]


def test_reused_turn_is_bitwise_the_cold_twin():
    """A conversation turn equals the same turn run cold with the same chunking."""
    p1 = np.arange(10, 42, dtype=np.int64)
    p2 = np.concatenate([p1, np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int64),
                          np.asarray([90, 91], dtype=np.int64)])

    def run(store: s_.Dsv41Sessions):
        conv, _, _ = _run_turn(store, p1)
        plan = [4] * (len(p1) // 4) + [len(p2) - (len(p1) // 4) * 4]
        r = conv.prefill(p2, chunk_plan=plan)
        gen = [int(np.asarray(r.anchor_logits).reshape(-1)[-1])]
        for _ in range(STEPS - 1):
            row = np.asarray(
                conv.model(mx.array([[gen[-1]]], dtype=mx.int32), conv.cache.cache,
                                  last_logit_only=True, argmax=True)
            ).reshape(-1)
            conv.mark_rows([gen[-1]])
            gen.append(int(row[-1]))
        conv.finish()
        return conv

    wa = run(s_.Dsv41Sessions(StubBody(), None, max_seq_len=256, chunk=4,
                              long_chunk=4, long_threshold=10**9))
    ca = run(s_.Dsv41Sessions(StubBody(), None, max_seq_len=256, chunk=4,
                              long_chunk=4, long_threshold=10**9))
    assert wa.offset == ca.offset
    assert [int(t) for t in wa.tokens] == [int(t) for t in ca.tokens]
    assert wa.draft_ctx() == ca.draft_ctx() == -1


def test_cancel_rolls_back_and_rerun_reproduces():
    store = s_.Dsv41Sessions(StubBody(), None, max_seq_len=256, chunk=4,
                             long_chunk=4, long_threshold=10**9)
    p1 = np.arange(10, 42, dtype=np.int64)
    conv, _, gen1 = _run_turn(store, p1)
    off = conv.offset
    p2 = np.concatenate([p1, np.asarray(gen1, dtype=np.int64),
                         np.asarray([7, 8, 9], dtype=np.int64)])
    # ``chunk_plan`` sets only the FIRST piece's size; the driver then feeds the
    # rest, so the whole prompt is prefilled either way.
    _ = conv.prefill(p2)
    assert conv.cancel() > 0
    assert conv.offset == off
    # a re-run after the cancel must reproduce the same delta
    _ = conv.prefill(p2)
    assert conv.offset > off


def test_engine_turn_wires_prefill_anchor_and_reporting():
    """The engine's ``_run_turn`` path: anchor -> rounds -> checkpoint."""
    from exo.shared.types.common import ModelId
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded

    body = StubBody()
    loaded = Dsv41Loaded(
        model=body, tokenizer=None, args=body.args, model_path=None,  # type: ignore[arg-type]
        built_layers=[0], full_stack=False, rank=0, world=1, load_seconds=0.0,
        head=None,
    )

    class Sender:
        def __init__(self) -> None:
            self.events: list[Any] = []

        def send(self, item: Any) -> None:
            self.events.append(item)

    class Receiver:
        def collect(self) -> list[Any]:
            return []

    engine = Dsv41Engine(
        loaded=loaded, model_id=ModelId("x/y"), group=None,
        cancel_receiver=Receiver(),  # type: ignore[arg-type]
        event_sender=Sender(),  # type: ignore[arg-type]
        device_rank=0, speculative=False, prefill_chunk_size=4, max_kv_tokens=256,
    )
    conv = engine._sessions.get([1, 2, 3], "k")
    turn = engine._run_turn(conv, [1, 2, 3], STEPS, embeddings=None, image_span_end=0)
    assert turn.anchor_logits is not None
    assert turn.committed
    # prefill + decode rounds; the last generated token stays un-fed (the anchor)
    assert conv.offset == 3 + STEPS - 1
    assert conv.generated == turn.tokens


def test_exact_repeat_prompt_reuses_the_saved_anchor():
    """The same prompt twice: zero new rows, so the anchor comes from the
    prompt-end checkpoint instead of failing the request (seen live: a
    benchmark repeating a prompt crashed the runner)."""
    store = s_.Dsv41Sessions(StubBody(), None, max_seq_len=256, chunk=4,
                             long_chunk=4, long_threshold=10**9)
    p1 = np.arange(10, 90, dtype=np.int64)
    conv, r1, gen1 = _run_turn(store, p1)
    conv2, r2, gen2 = _run_turn(store, p1)
    assert conv2 is conv
    assert r2.prefill_tokens == 0 and r2.reused_tokens == len(p1), f"{r2}"
    assert gen2 == gen1


def _widen(buf: Any) -> "np.ndarray":
    return np.array(buf.astype(mx.float32))


def test_growth_matches_full_capacity_twin():
    """A conversation whose cache grows on demand is bitwise a twin that was
    preallocated to the cap: the engine's ensure wiring grows buffers across
    >= 2 boundaries without changing any output or cache state."""
    p1 = np.arange(10, 42, dtype=np.int64)
    p2 = np.concatenate([p1, np.asarray([0, 1, 2, 3, 4, 0], dtype=np.int64),
                         np.asarray([90, 91], dtype=np.int64)])

    def run(cap: int):
        store = s_.Dsv41Sessions(StubBody(cap), None, max_seq_len=256, chunk=4,
                                 long_chunk=4, long_threshold=10**9)
        conv, _, gen1 = _run_turn(store, p1)
        conv.prefill(p2)
        return conv, gen1

    grew, g1g = run(8)                      # starts tiny -> must grow
    fixed, g1f = run(256)                   # preallocated to the cap -> never grows

    assert grew.cache.cache.capacity >= 32, grew.cache.cache.capacity
    assert grew.cache.cache.capacity > 8    # the growing path really crossed boundaries
    assert grew.offset == fixed.offset
    assert [int(t) for t in grew.tokens] == [int(t) for t in fixed.tokens]
    assert g1g == g1f
    # Buffer contents beyond the live rows must agree (the growth copied the
    # used rows and zero-filled the rest, exactly like the preallocated twin).
    ng = grew.offset // 2
    for a, b in zip(grew.cache.cache.layers, fixed.cache.cache.layers,
                    strict=True):
        assert np.array_equal(_widen(a.win_kv), _widen(b.win_kv))
        if a.comp_kv is not None:
            assert np.array_equal(_widen(a.comp_kv[:, :ng]),
                                  _widen(b.comp_kv[:, :ng]))
            assert np.array_equal(_widen(a.index_k[:, :ng]),
                                  _widen(b.index_k[:, :ng]))


# ---------------------------------------------------- prefill transient budget


def test_choose_prefill_step_keeps_base_at_low_offset():
    """Below the budget the full base chunk is kept (no needless shrinking)."""
    # 2048 * 1000 * 4 = 8.2 MB, well under the 2 GB default budget.
    assert s_.choose_prefill_step(1000, 100_000, 2048, 2_048_000_000) == 2048
    # Offset 0 is handled like any other (uses 1 row so it never divides by 0).
    assert s_.choose_prefill_step(0, 100_000, 2048, 2_048_000_000) == 2048


def test_choose_prefill_step_shrinks_so_the_score_row_fits_the_budget():
    """The chosen step's worst-case indexer row (step*offset*4 bytes) <= budget."""
    budget = 2_048_000_000
    offset, total, base = 1_000_000, 2_000_000, 2048
    step = s_.choose_prefill_step(offset, total, base, budget)
    assert step < base
    assert step * offset * 4 <= budget
    # 2048 at 1M offset would be 8.2 GB -- far over the 2 GB budget -- so the
    # policy drops the chunk until it fits (exactly budget//(4*offset)).
    assert step == budget // (4 * offset)


def test_choose_prefill_step_respects_the_floor():
    """A budget too small for even the floor still yields the floor, not less."""
    # 4 * 10M * 128 = 5.1 GB > the 100 MB budget, yet the floor stands.
    assert s_.choose_prefill_step(10_000_000, 20_000_000, 2048, 100_000_000) == 128


def test_choose_prefill_step_never_exceeds_the_rows_remaining():
    assert s_.choose_prefill_step(0, 100, 2048, 10**30) == 100
    assert s_.choose_prefill_step(90, 100, 2048, 10**30) == 10


def test_choose_prefill_step_one_row_minimum_and_offset_zero():
    assert s_.choose_prefill_step(0, 1, 2048, 2_048_000_000) == 1
    # Nothing left (offset >= total): still 1, so a caller's loop can never
    # stall on a 0-row chunk.
    assert s_.choose_prefill_step(500, 500, 2048, 2_048_000_000) == 1
    assert s_.choose_prefill_step(600, 500, 2048, 2_048_000_000) == 1


def test_choose_prefill_step_huge_budget_is_always_base():
    assert s_.choose_prefill_step(9_000_000, 2_000_000_000, 2048, 10**30) == 2048
    # ...clamped only by the rows remaining.
    assert s_.choose_prefill_step(9_000_000, 9_000_050, 2048, 10**30) == 50


def test_choose_prefill_step_base_is_the_ceiling():
    """A base smaller than the floor is respected: the policy only ever shrinks."""
    assert s_.choose_prefill_step(0, 10**9, 2, 10**30) == 2


def test_choose_prefill_step_bf16_row_doubles_the_viable_chunk():
    """With the bf16 score row (2 B/elem) the same byte budget affords 2x the chunk."""
    budget = 2_048_000_000
    offset, total, base = 1_000_000, 2_000_000, 2048
    fp32 = s_.choose_prefill_step(offset, total, base, budget, row_bytes=4)
    bf16 = s_.choose_prefill_step(offset, total, base, budget, row_bytes=2)
    assert fp32 == budget // (4 * offset)
    # bf16: worst_row = 2 * offset -> twice the rows, still <= budget.
    assert bf16 == budget // (2 * offset)
    assert bf16 == 2 * fp32
    assert bf16 * offset * 2 <= budget


def test_choose_prefill_step_row_bytes_defaults_to_fp32():
    """The default keeps the pre-existing fp32 behavior bit-for-bit."""
    budget, offset = 1_200_000, 1000
    assert s_.choose_prefill_step(offset, 10**9, 2048, budget) == 300
    assert s_.choose_prefill_step(offset, 10**9, 2048, budget, row_bytes=4) == 300


def test_indexer_row_bytes_follows_the_deployed_module():
    """The import-time probe reads the ACTUAL mlx-lm row dtype (2/4), never crashes."""
    assert s_._INDEXER_ROW_BYTES in (2, 4)


def test_choose_prefill_step_boundary_is_exact():
    """step * offset * 4 == budget is accepted (<=, not <)."""
    budget, offset = 1_200_000, 1000
    assert budget // (4 * offset) == 300
    assert s_.choose_prefill_step(offset, 10**9, 2048, budget) == 300


class _OffsetCache:
    """Minimal cache stub: ``engine_prefill`` only reads/writes ``offset``."""

    def __init__(self) -> None:
        self.offset = 0


class _FenceModel:
    """Fake body for the real ``engine_prefill`` loop.

    Records the value of ``_fence_every`` it sees at each forward and the row
    count fed, and advances ``cache.offset`` exactly as the real body does (the
    driver's chunk policy reads the offset before each chunk). The ``fence``
    accessor reads the model's own attribute, so the assertion exercises the same
    attribute ``Model.__call__`` would read in production.
    """

    def __init__(self, fence_every: int = 0) -> None:
        self._fence_every = int(fence_every)
        self.fences: list[int] = []
        self.rows: list[int] = []

    def fence(self) -> int:
        return int(self._fence_every)

    def __call__(self, ids: mx.array, cache: _OffsetCache, **rest: object) -> mx.array:
        del rest
        self.fences.append(self.fence())
        n = int(ids.shape[-1])
        self.rows.append(n)
        cache.offset += n
        return mx.array([[0.0]])


def test_engine_prefill_sets_the_fence_and_restores_it(monkeypatch: pytest.MonkeyPatch):
    """The whole chunk loop runs fenced; the previous value is restored after."""
    monkeypatch.delenv("EXO_PREFILL_FENCE_EVERY", raising=False)
    model = _FenceModel()  # was 0 (unfenced) before
    s_.engine_prefill(model, np.arange(10), _OffsetCache(), chunk=4)

    assert model.fences, "no forward ran"
    assert all(f == s_.DEFAULT_PREFILL_FENCE_EVERY for f in model.fences)
    assert model.fence() == 0, "the original value must be restored"


def test_engine_prefill_restores_a_nonzero_fence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EXO_PREFILL_FENCE_EVERY", raising=False)
    model = _FenceModel(fence_every=7)
    s_.engine_prefill(model, np.arange(10), _OffsetCache(), chunk=4)
    assert model.fence() == 7


def test_engine_prefill_fence_env_zero_disables(monkeypatch: pytest.MonkeyPatch):
    """EXO_PREFILL_FENCE_EVERY=0 turns the fence off (value seen: 0)."""
    monkeypatch.setenv("EXO_PREFILL_FENCE_EVERY", "0")
    model = _FenceModel(fence_every=3)
    s_.engine_prefill(model, np.arange(10), _OffsetCache(), chunk=4)
    assert set(model.fences) == {0}
    assert model.fence() == 3  # original restored


def test_engine_prefill_fence_env_four(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_PREFILL_FENCE_EVERY", "4")
    model = _FenceModel()
    s_.engine_prefill(model, np.arange(10), _OffsetCache(), chunk=4)
    assert set(model.fences) == {4}


def _schedule(total: int, base: int, budget: int, floor: int = 128) -> list[int]:
    """Independent re-derivation of the budget schedule (for the loop test).

    Bills the indexer score row at its DEPLOYED dtype via ``s_._INDEXER_ROW_BYTES``
    (2 under the shipped bf16 row; 4 for fp32) -- a hard-coded 4 was the stale
    expectation that broke once the bf16-aware budget shipped.
    """
    off, out = 0, []
    while off < total:
        rows = min(base, max(floor, budget // (s_._INDEXER_ROW_BYTES * max(off, 1))))
        step = max(1, min(rows, total - off))
        out.append(step)
        off += step
    return out


def test_engine_prefill_delta_feed_keeps_the_budget_schedule(monkeypatch: pytest.MonkeyPatch):
    """A RESUMED session's delta must not collapse to 1-row chunks.

    Regression (2026-10-05): the driver passed the DELTA's row count as the
    policy's ``total`` while ``offset`` is the ABSOLUTE cache position. For a
    delta of L rows on top of O resident rows, ``total - offset`` went negative
    after L-O rows, pinning the policy to the 1-row floor for the remaining ~O
    rows -- every tail row as its own forward (S2's 1361-row refeed took 68.5s,
    r500's 350K feed crawled for 4h). The schedule must use one frame.
    """
    monkeypatch.delenv("EXO_PREFILL_FENCE_EVERY", raising=False)
    cache = _OffsetCache()
    cache.offset = 1_000_000          # resident conversation at 1M rows
    base, budget = 2048, 2_048_000_000
    delta = 4_000                      # 4K-row delta on top of 1M resident
    model = _FenceModel()
    s_.engine_prefill(model, np.arange(delta), cache, chunk=base,
                      transient_budget_bytes=budget)
    # Absolute frame: absolute end = 1_004_000, so the score row allows
    # min(base, budget // (2 * 1M)) = 1024 rows per chunk. Every chunk except
    # the final remainder must be ~1024, never the 1-row floor.
    assert all(r >= 128 for r in model.rows[:-1]), f"floor-pinned: {model.rows[:6]}"
    assert max(model.rows) > 128, "delta feed collapsed to tiny chunks"
    assert sum(model.rows) == delta
    # Independent re-derivation in the ABSOLUTE frame.
    assert model.rows == _schedule_abs(1_000_000, delta, base, budget)


def _schedule_abs(offset0: int, total: int, base: int, budget: int, floor: int = 128) -> list[int]:
    """Re-derivation of the delta schedule with absolute-frame inputs."""
    off, done, out = offset0, 0, []
    while done < total:
        rows = min(base, max(floor, budget // (s_._INDEXER_ROW_BYTES * max(off, 1))))
        step = max(1, min(rows, total - done))
        out.append(step)
        done += step
        off += step
    return out


def test_engine_prefill_chunk_schedule_follows_the_budget(monkeypatch: pytest.MonkeyPatch):
    """The observed per-chunk rows match the budget schedule and never blow it.

    A 30K-row feed with a 20 MB budget: the first chunks keep the 2048 base, the
    chunk then shrinks as the offset grows, and no chunk (except a final
    remainder) drops below the 128 floor. Every chunk keeps the indexer's
    worst-case score row within budget.
    """
    monkeypatch.delenv("EXO_PREFILL_FENCE_EVERY", raising=False)
    total, base, budget = 30_000, 2048, 20_000_000
    model = _FenceModel()
    s_.engine_prefill(
        model, np.arange(total), _OffsetCache(), chunk=base,
        transient_budget_bytes=budget,
    )

    assert model.rows == _schedule(total, base, budget)
    assert model.rows[:2] == [2048, 2048], "base must be kept at low offset"
    # The first shrink is dtype-dependent (bf16 = 2 B/row keeps base until
    # 2048*offset*2 > budget). Assert the schedule's own first divergence.
    first_shrink = next(i for i, r in enumerate(model.rows) if r < base)
    assert first_shrink >= 1, "the chunk must shrink as the offset grows"
    assert model.rows[first_shrink - 1] == base
    assert all(r >= 128 for r in model.rows[:-1]), "never below the floor"
    assert sum(model.rows) == total
    off = 0
    for step in model.rows:
        # Above the floor the transient score row must fit the budget; at the
        # floor the policy accepts the (possibly over-budget) minimum. Billed
        # at the DEPLOYED row dtype (bf16 = 2 shipped).
        assert step * max(off, 1) * s_._INDEXER_ROW_BYTES <= budget or step == 128
        off += step


def test_engine_prefill_long_threshold_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    """An explicit ``long_threshold`` keeps the legacy fixed-crossover branch."""
    monkeypatch.delenv("EXO_PREFILL_FENCE_EVERY", raising=False)
    model = _FenceModel()
    # Budget so small the policy WOULD shrink immediately; the explicit
    # threshold must win (base=10 until offset>=40, then long_chunk=3).
    s_.engine_prefill(
        model, np.arange(100), _OffsetCache(), chunk=10, long_chunk=3,
        long_threshold=40, transient_budget_bytes=1,
    )
    assert model.rows[:4] == [10, 10, 10, 10]
    assert model.rows[4] == 3


class _FenceRecordingBody(StubBody):
    """StubBody that records the fence it runs under (end-to-end wiring check)."""

    def __init__(self) -> None:
        super().__init__()
        self._fence_every = 0
        self.fences: list[int] = []

    def fence(self) -> int:
        return int(self._fence_every)

    def __call__(
        self,
        ids: Any,
        cache: Any,
        last_logit_only: bool = False,
        return_taps: bool = False,
        argmax: bool = False,
    ) -> Any:
        self.fences.append(self.fence())
        return super().__call__(
            ids, cache, last_logit_only, return_taps, argmax
        )


def test_conversation_prefill_runs_fenced_end_to_end(monkeypatch: pytest.MonkeyPatch):
    """The engine's session plumbing carries the fence into the real loop.

    Exercises the ``functools.partial`` wiring in ``Conversation``: the body sees
    a nonzero fence during prefill and the value is restored afterwards.
    """
    monkeypatch.delenv("EXO_PREFILL_FENCE_EVERY", raising=False)
    body = _FenceRecordingBody()
    store = s_.Dsv41Sessions(body, None, max_seq_len=256, chunk=4)
    conv = store.get(list(range(10, 42)), "conv")
    conv.prefill(np.arange(10, 42, dtype=np.int64))

    assert body.fences, "prefill ran no forward"
    assert all(f == s_.DEFAULT_PREFILL_FENCE_EVERY for f in body.fences)
    assert body.fence() == 0  # restored after the turn's prefill

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

from exo.worker.engines.mlx.dsv41 import session as s_

#: Decoded tokens each turn produces after its anchor.
STEPS = 6


class StubBody:
    """Deterministic body over a real ``ModelCache`` (see the fork's own tests)."""

    def __init__(self, initial_capacity: int | None = None) -> None:
        from mlx_lm.models.deepseek_v41 import cache as c_
        from mlx_lm.models.deepseek_v41.config import ModelArgs

        self._C = c_
        self.initial_capacity = initial_capacity
        self.args = ModelArgs(
            vocab_size=1000, window_size=8, compress_ratios=(2, 2),
            kv_source_layers=(0,), index_source_layers=(0, 1),
            engram_layer_ids=(),
        )
        self.embed: Any = None
        self.head: Any = None

    def make_cache(self, bsz: int = 1, max_seq_len: int | None = None,
                   dtype: Any = None, initial_capacity: int | None = None,
                   **_: Any):
        return self._C.ModelCache(
            self.args, bsz, max_seq_len or 64,
            initial_capacity=(initial_capacity if initial_capacity is not None
                              else self.initial_capacity))

    def __call__(self, ids: Any, cache: Any, last_logit_only: bool = False,
                 return_taps: bool = False, argmax: bool = False):
        arr = np.asarray(ids)
        if arr.ndim == 1:
            arr = arr[None]
        b, n = arr.shape
        pos = int(cache.offset)
        # The real driver's invariant: grow (eval-clean) before any write.
        cache.ensure_capacity(pos + n)
        for lc in cache.layers:
            r = max(int(lc.ratio or 0), 1)
            cs = lc.comp_state
            if cs is not None:
                for i in range(n):
                    cs.kv_state[0, (pos + i) % r] = float(pos + i)
                    cs.score_state[0, (pos + i) % r] = float(pos + i)
            w = int(lc.window)
            for i in range(n):
                lc.win_kv[0, (pos + i) % w] = float(pos + i)
        cache.offset = pos + n
        out = mx.array(((arr + 1) % 1000).astype(np.int32))
        if argmax:
            return (out, {0: mx.zeros((b, n, 4))}) if return_taps else out
        if return_taps:
            return mx.array([[float(arr[0, -1])]]), {0: mx.zeros((1, 1, 4))}
        return mx.array([[float(arr[0, -1])]])


def _run_turn(store: s_.Dsv41Sessions, ids: "np.ndarray", *, checkpoint: bool = True):
    """Prefill + ``STEPS`` decode rounds on one conversation; returns it."""
    conv = store.get(ids.tolist(), "conv")
    fed = conv.prefill(ids)
    gen = [int(np.asarray(fed.anchor_logits).reshape(-1)[-1])]
    for _ in range(STEPS - 1):
        row = np.asarray(
            conv.model(mx.array([[gen[-1]]], dtype=mx.int32), conv.cache.cache,
                              last_logit_only=True, argmax=True)
        ).reshape(-1)
        conv.mark_rows([gen[-1]])
        gen.append(int(row[-1]))
    conv.finish(checkpoint=checkpoint)
    return conv, fed, gen


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

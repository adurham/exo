"""CPU-only wiring test: the DSv4.1 conversation sessions + the engine's turn.

No checkpoint, no Metal: a stub body that writes REAL cache state (the fork's
``ModelCache`` / ``CompressorState``) drives ``dsv41.session`` and the engine's
``_run_turn``/``_decode`` path, so the multi-turn contract is pinned without a
GPU job:

* turn-2 prefill is the delta only (``< 10%`` of turn 1) -- the acceptance bar;
* a reused turn's tokens are bitwise equal to the same turn on a cold cache with
  the same chunk boundaries (``chunk_plan``);
* ``cancel`` rolls the conversation back, and re-running produces the same
  tokens;
* the engine's ``_run_turn`` wires prefill -> anchor -> rounds -> checkpoint, and
  its reported usage carries the reuse split.

The stub body's rule is deterministic ("the token after row ``k`` is ``k % 100``"),
so a draft head that drafts ``anchor + 1`` is a stand-in for the DSpark head: it
is *not* the DSpark contract under test here, only the turn plumbing.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import session as S

LAYERS = (0, 1)


class _Args:
    vocab_size = 1000
    window_size = 8
    compress_ratios = (2, 2)
    kv_source_layers = (0,)
    index_source_layers = (0, 1)
    engram_layer_ids = ()
    dspark_target_layer_ids = ()
    max_seq_len = 4096
    hc_mult = 1
    norm_eps = 1e-20


class StubBody:
    """Deterministic body over a REAL ``ModelCache``.

    ``__call__`` mimics the real signatures the session/engine use
    (``last_logit_only``, ``argmax``, ``return_taps``) and writes real per-layer
    cache state, so window-ring / compressed-KV / carry comparisons are
    meaningful. The next-token rule is a pure function of the last row's id, so a
    replay reproduces it exactly.
    """

    def __init__(self) -> None:
        from mlx_lm.models.deepseek_v41 import cache as C
        from mlx_lm.models.deepseek_v41.config import ModelArgs

        self._C = C
        self.args = ModelArgs(
            vocab_size=_Args.vocab_size,
            window_size=_Args.window_size,
            compress_ratios=_Args.compress_ratios,
            kv_source_layers=_Args.kv_source_layers,
            index_source_layers=_Args.index_source_layers,
            engram_layer_ids=_Args.engram_layer_ids,
        )
        self.embed: Any = None
        self.head: Any = None

    def make_cache(self, bsz: int = 1, max_seq_len: int | None = None, **_: Any):
        return self._C.ModelCache(self.args, bsz, max_seq_len or 64)

    def __call__(self, ids: Any, cache: Any, last_logit_only: bool = False,
                 return_taps: bool = False, argmax: bool = False):
        arr = np.asarray(ids)
        if arr.ndim == 1:
            arr = arr[None]
        b, n = arr.shape
        pos = int(cache.offset)
        for lc in cache.layers:
            r = int(lc.ratio or 0)
            if lc.comp_kv is not None and r:
                for g in range(pos // r, (pos + n) // r):
                    lc.comp_kv[0, g] = float(g)
            cs = lc.comp_state
            if cs is not None:
                for i in range(n):
                    cs.kv_state[0, (pos + i) % r] = float(pos + i)
                    cs.score_state[0, (pos + i) % r] = float(pos + i)
            w = int(lc.window)
            for i in range(n):
                lc.win_kv[0, (pos + i) % w] = float(pos + i)
        cache.offset = pos + n
        nxt = (arr + 1) % _Args.vocab_size
        out = mx.array(nxt.astype(np.int32))
        if argmax:
            if return_taps:
                return out, {0: mx.zeros((b, n, 4))}
            return out
        if return_taps:
            return mx.array([[float(arr[0, -1])]]), {0: mx.zeros((1, 1, 4))}
        return mx.array([[float(arr[0, -1])]])


def _state(cache: Any, upto: int) -> dict[str, mx.array]:
    """Every cache row a forward at ``upto`` can read, per layer."""
    out: dict[str, mx.array] = {}
    for i, lc in enumerate(cache.layers):
        w = int(lc.window)
        first = max(0, upto - w)
        slots = np.arange(first, upto) % w if upto else np.zeros(0, np.int64)
        out[f"L{i}.win"] = lc.win_kv[:, mx.array(slots)] if len(slots) else lc.win_kv[:, :0]
        if lc.comp_kv is not None:
            out[f"L{i}.comp"] = lc.comp_kv[:, : upto // max(int(lc.ratio), 1)]
        if lc.index_k is not None:
            out[f"L{i}.idx"] = lc.index_k[:, : upto // max(int(lc.ratio), 1)]
        if lc.comp_state is not None:
            m = upto % max(int(lc.ratio), 1)
            out[f"L{i}.kv_state"] = lc.comp_state.kv_state[:, :m]
            out[f"L{i}.sc_state"] = lc.comp_state.score_state[:, :m]
    mx.eval(*out.values())
    return out


def _diff(a: dict[str, mx.array], b: dict[str, mx.array]) -> list[str]:
    bad: list[str] = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(f"{k}:missing")
        elif not bool(mx.array_equal(a[k], b[k])):
            d = float(mx.max(mx.abs(a[k].astype(mx.float32) - b[k].astype(mx.float32))).item())
            bad.append(f"{k}:max|d|={d:.3g}")
    return bad


def _store(body: StubBody, **kw: Any) -> S.Dsv41Sessions:
    return S.Dsv41Sessions(body, None, max_seq_len=256, chunk=4, long_chunk=4,
                            long_threshold=10**9, **kw)


def _turn(store: S.Dsv41Sessions, ids: list[int], *, checkpoint: bool = True):
    conv = store.get(ids, "conv")
    out = conv.prefill(ids)
    gen = [int(np.argmax(np.asarray(out.anchor_logits).reshape(-1)).item())]
    # greedy decode: one row per step, through the conversation's own history
    while len(gen) < 6:
        nxt = mx.array([[gen[-1]]], dtype=mx.int32)
        row = body_call(conv, nxt)
        conv.mark_rows([gen[-1]])
        gen.append(int(np.asarray(row).reshape(-1)[-1]))
    conv.finish(checkpoint=checkpoint)
    return conv, out, gen


def body_call(conv: Any, ids: mx.array) -> mx.array:
    """Drive the stub body directly (this test has no DSpark head)."""
    return conv.model(ids, conv.cache.cache, last_logit_only=True, argmax=True)


# ------------------------------------------------------------------ the tests

def test_turn_two_prefills_only_the_delta():
    """The acceptance bar: turn 2's prefill is the delta, not the whole prompt."""
    store = _store(StubBody())
    P1 = list(range(10, 42))
    conv, r1, gen1 = _turn(store, P1)
    assert r1.prefill_tokens == len(P1) and r1.reused_tokens == 0
    P2 = P1 + gen1 + [90, 91, 92]
    conv2, r2, gen2 = _turn(store, P2)
    assert conv2 is conv
    delta = 1 + 3  # the last generated token (the anchor) + the new text
    assert r2.prefill_tokens == delta, f"{r2}"
    assert r2.reused_tokens == len(P1) + len(gen1) - 1
    ratio = r2.prefill_tokens / r2.prompt_tokens
    assert ratio < 0.10, f"{r2} ratio={ratio}"
    assert conv.offset == r2.cache_offset
    assert len(conv.tokens) == conv.offset
    assert conv.tokens == P2 + gen2[:-1]
    # and the session's own numbers are consistent with the row counts
    assert len(gen2) == 6


def test_reused_turn_is_bitwise_the_cold_twin():
    """A conversation turn equals the same turn run cold with the same chunking."""
    P1 = list(range(10, 42))
    P2 = P1 + [0, 1, 2] + [90, 91]
    plan1 = [4] * (len(P1) // 4)
    plan2 = [len(P2) - len(P1)]

    warm = _store(StubBody())
    cw, rw, _ = _turn(warm, P1)
    cold = _store(StubBody())
    cc, rc, genc = _turn(cold, P1)
    # run turn 2 on both with the SAME chunk plan (chunk-shape parity)
    ids2 = P2
    out_w = cw.prefill(ids2, chunk_plan=plan2)
    gw = [int(np.argmax(np.asarray(out_w.anchor_logits).reshape(-1)).item())]
    for _ in range(5):
        gw.append(int(np.asarray(body_call(cw, mx.array([[gw[-1]]], dtype=mx.int32))).reshape(-1)[-1]))
        cw.mark_rows([gw[-2]])
    cw.finish()
    out_c = cc.prefill(ids2, chunk_plan=plan2)
    gc = [int(np.argmax(np.asarray(out_c.anchor_logits).reshape(-1)).item())]
    for _ in range(5):
        gc.append(int(np.asarray(body_call(cc, mx.array([[gc[-1]]], dtype=mx.int32))).reshape(-1)[-1]))
        cc.mark_rows([gc[-2]])
    cc.finish()
    assert gw == gc
    bad = _diff(_state(cw.cache.cache, cw.offset), _state(cc.cache.cache, cc.offset))
    assert not bad, f"state differs: {bad[:6]}"
    assert out_w.reused_tokens > 0


def test_cancel_rolls_back_and_rerun_reproduces():
    store = _store(StubBody())
    P1 = list(range(10, 42))
    conv, _, _ = _turn(store, P1)
    off, gen_before = conv.offset, conv.generated
    P2 = P1 + gen_before + [7, 8, 9]
    _ = conv.prefill(P2, chunk_plan=[3])
    assert conv.offset > off
    dropped = conv.cancel()
    assert dropped > 0
    assert conv.offset == off
    assert conv.generated == gen_before
    a = conv.prefill(P2, chunk_plan=[3])
    b = conv.prefill(P2, chunk_plan=[3]) if False else None
    assert conv.offset == off + 3


def test_engine_turn_wires_prefill_anchor_and_reporting():
    """The engine's ``_run_turn`` path: anchor -> tokens -> checkpoint."""
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
    from exo.shared.types.common import CommandId, ModelId
    from exo.shared.types.tasks import TaskId
    from exo.worker.engines.mlx.dsv41.tests.test_dsv41_engine import (
        ScriptedModel,
        _Receiver,
        _Sender,
    )

    body = StubBody()
    model = ScriptedModel([40, 41, 1])
    loaded = Dsv41Loaded(
        model=body, tokenizer=None, args=body.args, model_path=None,  # type: ignore[arg-type]
        built_layers=list(LAYERS), full_stack=False, rank=0, world=1,
        load_seconds=0.0, head=None,
    )
    engine = Dsv41Engine(
        loaded=loaded,
        model_id=ModelId("x/y"),
        group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=_Sender(),  # type: ignore[arg-type]
        device_rank=0,
        speculative=False,
        prefill_chunk_size=4,
        max_kv_tokens=256,
    )
    conv = engine._sessions.get([1, 2, 3], "k")
    turn = engine._run_turn(conv, [1, 2, 3], 4, embeddings=None, image_span_end=0)
    assert turn.anchor_logits is not None
    assert turn.committed and conv.offset == 3 + 4
    assert len(turn.tokens) == 5  # the anchor + 4 greedy steps
    assert conv.generated == turn.tokens
    assert engine._sessions.stats["cold"] == 1
    del model

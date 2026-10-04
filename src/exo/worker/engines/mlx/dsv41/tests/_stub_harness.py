"""Shared CPU-only stub harness for the DSv4.1 session + park tests.

No GPU, no checkpoint: a deterministic stub body over the fork's REAL
``ModelCache`` drives ``dsv41.session`` (and, in ``test_dsv41_park``, the SSD
serializer). Refactored out of ``test_dsv41_session.py`` so the park tests use
the exact same construction path the session suite pins -- one source of truth
for the stub's rule (``StubBody.__call__`` writes ``pos`` into the window ring
and the compressor carry, then advances ``cache.offset``), so a replay of the
same turns reproduces the same cache bytes.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

#: Decoded tokens each turn produces after its anchor.
STEPS = 6


class StubBody:
    """Deterministic body over a real ``ModelCache`` (see the fork's own tests).

    ``initial_capacity`` forces the grow-on-demand path; ``**arg_overrides``
    tweak the tiny ``ModelArgs`` (e.g. ``engram_layer_ids=(0,)`` to exercise the
    engram-id history, or ``dspark_target_layer_ids=(0,)`` for the draft feed).
    """

    def __init__(self, initial_capacity: int | None = None, **arg_overrides: Any) -> None:
        from mlx_lm.models.deepseek_v41 import cache as c_
        from mlx_lm.models.deepseek_v41.config import ModelArgs

        self._C = c_
        self.initial_capacity = initial_capacity
        base: dict[str, Any] = dict(
            vocab_size=1000, window_size=8, compress_ratios=(2, 2),
            kv_source_layers=(0,), index_source_layers=(0, 1),
            engram_layer_ids=(),
        )
        base.update(arg_overrides)
        self.args = ModelArgs(**base)
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
            # n tap rows (one per fed row): the draft window advances with the
            # cache, exactly as the real head's per-row context feed does.
            return mx.array([[float(arr[0, -1])]]), {0: mx.zeros((b, n, 4))}
        return mx.array([[float(arr[0, -1])]])


class StubHead:
    """Minimal DSpark head for the draft-window park round-trip.

    ``make_cache`` returns the fork's real ``DraftWindow`` objects (fp32, no
    weights); ``append_ctx`` appends a deterministic patterned row per fed
    context row so the parked window is non-trivial. Only the surface
    ``Conversation`` touches is implemented.
    """

    def __init__(self, window: int = 8, head_dim: int = 512, n_stages: int = 3) -> None:
        self.window = window
        self.head_dim = head_dim
        self.n_stages = n_stages

    def make_cache(self, bsz: int = 1):
        from mlx_lm.models.deepseek_v41.mtp import DraftWindow

        return [DraftWindow(bsz, self.window, self.head_dim)
                for _ in range(self.n_stages)]

    def append_ctx(self, main_hidden_cat: mx.array, caches: Any) -> None:
        rows = int(main_hidden_cat.shape[1])
        for w in caches:
            base = int(w.n_ctx)
            pat = mx.array(
                (np.arange(base, base + rows)[None, :, None]
                 + np.arange(w.head_dim)[None, None, :]).astype(np.float32))
            w.append(pat)


def run_turn(store: Any, ids: "np.ndarray", *, checkpoint: bool = True,
             key: str | None = "conv"):
    """Prefill + ``STEPS`` decode rounds on one conversation; returns it."""
    conv = store.get(ids.tolist(), key)
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


def _bf16(rng: "np.random.Generator", shape: tuple[int, ...]) -> mx.array:
    """Normal-range fp32 values cast to bf16: every bit round-trips exactly."""
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)


def fill_buffers(conv: Any, *, seed: int = 0) -> None:
    """Deterministic patterned contents for the LIVE rows of every buffer.

    Only rows a read can reach are filled (``comp_kv``/``index_k`` up to
    ``ceil(offset/ratio)``, engram ids up to ``offset``) so the source is a
    valid cache; ``win_kv`` and the snapshot rings are filled whole. Filling the
    live rows is what makes the round-trip test non-trivial: the tail stays zero
    in the source, so a restore that forgot to zero-fill would still pass unless
    the poison invariant is checked separately.
    """
    model = conv.cache.cache
    offset = int(model.offset)
    capacity = int(model.capacity)
    rng = np.random.default_rng(seed)
    for lc in model.layers:
        shape = tuple(int(s) for s in lc.win_kv.shape)
        lc.win_kv = _bf16(rng, shape)
        ratio = max(int(lc.ratio or 0), 1)
        used = -(-min(offset, capacity) // ratio)
        if lc.comp_kv is not None:
            lc.comp_kv[:, :used] = _bf16(rng, (int(lc.comp_kv.shape[0]), used,
                                               int(lc.comp_kv.shape[2])))
        if lc.index_k is not None:
            lc.index_k[:, :used] = _bf16(rng, (int(lc.index_k.shape[0]), used,
                                               int(lc.index_k.shape[2])))
        cs = lc.comp_state
        if cs is not None:
            m = offset % ratio
            if m:
                cs.kv_state[:, :m] = mx.array(
                    rng.standard_normal((1, m, int(cs.kv_state.shape[2]))).astype(np.float32))
                cs.score_state[:, :m] = mx.array(
                    rng.standard_normal((1, m, int(cs.score_state.shape[2]))).astype(np.float32))
    if model.engram_ids is not None:
        model.engram_ids[:, :offset] = rng.integers(
            0, 1_000_000, size=(int(model.engram_ids.shape[0]), offset), dtype=np.int64)

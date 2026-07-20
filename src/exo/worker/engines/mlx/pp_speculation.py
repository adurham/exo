"""PP idle-time speculation for pipeline parallel decode.

During normal PP decode, rank 0 is idle for ~15ms while rank 1 computes.
This module uses that idle time to speculatively compute layers 0-29
for a draft token. If the draft matches rank 1's actual token, rank 0
sends the pre-computed hidden state immediately — saving ~15ms.

Gated by EXO_PP_DRAFT_MODEL. When unset, upstream's stream_generate
runs unchanged.

Architecture:
- Subclasses PipelineFirstLayer/PipelineLastLayer with a speculative mode
- Custom decode loop with explicit PP phase separation
- Draft model runs on rank 0 ONLY during idle time
- Zero modifications to upstream code

Overlap strategy:
- Hidden exchange uses send/recv (not all_gather) so rank 0 can proceed
  immediately after sending, drafting DURING rank 1's compute time.
- Token exchange uses all_gather (both ranks need the sampled token).
"""

import os
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Generator

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import CacheList
from mlx_lm.sample_utils import make_sampler

from .auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
)
from .pp_speculation_spec_tag import (
    HIT_MISS_HIT,
    HIT_MISS_MISS,
    HIT_MISS_NA,
    SPEC_TAG_WIRE_LEN,
    SpecHiddenBuffer,
    SpecId,
    SpecTagValidator,
    coerce_hit_miss,
    deep_draft_ext_len,
    pack_deep_draft_ext,
    pack_spec_tag,
    unpack_deep_draft_ext,
    unpack_spec_tag,
)

import loguru

logger: "loguru.Logger" = loguru.logger

_TRACE = os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1")

# Real-compute-vs-wait profiling for pp_dspark_decode_loop's r1_verify_fwd
# phase (2026-07-18): the naive wall-clock timing around model(...) on
# rank1 conflates "blocked waiting for rank0's hidden send" with "actual
# rank1 compute after receiving it" -- a second opinion flagged this as
# likely inflating the apparent verify cost. This module-level accumulator
# lets SpecPipelineFirstLayer.__call__ (where the actual blocking recv
# happens) report just the wait portion back up to the decode loop, which
# subtracts it to get real compute time.
_PROF_DSPARK_RECV_WAIT = (
    _TRACE and os.environ.get("EXO_PP_DSPARK_PROFILE_WAIT", "1") == "1"
)
_DSPARK_RECV_WAIT_ACCUM = [0.0]


def _log(msg: str) -> None:
    if _TRACE:
        sys.stderr.write(f"[pp-spec] {msg}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Pipeline info extraction
# ---------------------------------------------------------------------------


def get_pipeline_info(model: nn.Module) -> tuple[int, int, mx.distributed.Group] | None:
    """Extract (rank, world_size, group) from pipeline layer wrappers.
    Returns None if model is not pipeline-parallel.
    """
    for layer in model.layers:  # type: ignore
        if isinstance(layer, PipelineLastLayer):
            return (layer.r, layer.s, layer.group)
    return None


# ---------------------------------------------------------------------------
# Cache snapshot / restore (for speculative rollback)
# ---------------------------------------------------------------------------


def _snapshot_one(c: Any) -> Any:  # pyright: ignore[reportAny]
    """Snapshot a single cache entry, recursing through CacheList.

    ROOT CAUSE (2026-07-19, found live-debugging step 3a's degeneration +
    jaccl transport fault): the ORIGINAL version of this function only
    recognized bare ``ArraysCache``/``KVCache`` via isinstance and fell
    through to ``None`` (= "no-op, do nothing on restore") for every
    other cache type -- including ``RotatingKVCache``, ``CacheList``, and
    ``PoolingCache``. DeepseekV4Model.make_cache() (mlx-lm) returns
    EXACTLY those three types for every layer, NEVER bare KVCache or
    ArraysCache, so the "snapshot before speculative forward, restore on
    both HIT and MISS" invariant this file's docstrings and commit
    messages claimed was a COMPLETE NO-OP on this model: the speculative
    forward's real KV writes were never rolled back on any layer,
    corrupting subsequent decode (observed live: a "Paris.</think>
    Paris.</think>..." repeat loop, plus a jaccl transport fault from
    ranks diverging after the corruption).

    FIX: use each cache type's OWN state-capture primitive rather than a
    hand-rolled isinstance allowlist, so the speculative-rollback
    behavior tracks whatever cache types the model actually uses
    instead of drifting out of sync with them:
    1. ``RotatingKVCache`` has a PURPOSE-BUILT ``save_spec_state()`` for
       exactly this (materializes ``mx.array()`` copies of keys/values,
       not aliased views -- required because ring-buffer writes mutate
       in place, so a bare reference would "restore" to the
       already-corrupted post-write state). Prefer this via ``hasattr``.
    2. ``CacheList`` (used for DSv4's sparse/compressed attention
       layers, wrapping a RotatingKVCache + one or two PoolingCache
       instances) has no state-capture of its own -- recurse into
       ``.caches`` (bare-attribute access, matching the existing
       ``hasattr(cache, "caches")`` convention in auto_parallel.py's
       PipelineLastLayer -- the .pyi stub doesn't declare this attribute
       even though the real mlx-lm source does, hence the ignore).
    3. Everything else (``PoolingCache``, bare ``KVCache``,
       ``ArraysCache``, ``QuantizedKVCache``, ...) falls back to the
       generic ``_BaseCache.state``/``.meta_state`` protocol every cache
       type implements, with an explicit ``tree_map(mx.array, ...)``
       deep-copy -- MLX array assignment/``__setitem__`` mutates in
       place, so capturing ``.state`` without copying would alias the
       live buffer exactly like the original bug, just one level later.
    """
    if hasattr(c, "save_spec_state"):  # pyright: ignore[reportAny]
        save_fn: Callable[[], Any] = c.save_spec_state  # pyright: ignore[reportAny]
        return ("spec", save_fn())  # pyright: ignore[reportAny]
    if isinstance(c, CacheList):
        sub_caches: tuple[Any, ...] = c.caches  # type: ignore[attr-defined]
        return ("list", [_snapshot_one(sub) for sub in sub_caches])  # pyright: ignore[reportAny]
    state: Any = getattr(c, "state", None)  # pyright: ignore[reportAny]
    meta: Any = getattr(c, "meta_state", None)  # pyright: ignore[reportAny]
    return ("generic", tree_map(mx.array, state), meta)  # pyright: ignore[reportAny]


def _restore_one(c: Any, snap: Any) -> None:  # pyright: ignore[reportAny]
    """Restore a single cache entry from `_snapshot_one`'s output."""
    if snap is None:
        return
    kind: str = snap[0]  # pyright: ignore[reportAny]
    payload: Any = snap[1]  # pyright: ignore[reportAny]
    if kind == "spec":
        restore_fn: Callable[[Any], None] = c.restore_spec_state  # pyright: ignore[reportAny]
        restore_fn(payload)
    elif kind == "list":
        sub_caches: tuple[Any, ...] = c.caches  # type: ignore[attr-defined]
        for sub_c, sub_snap in zip(  # pyright: ignore[reportAny]
            sub_caches, payload, strict=True
        ):
            _restore_one(sub_c, sub_snap)
    elif kind == "generic":
        state_snap: Any = snap[1]  # pyright: ignore[reportAny]
        meta_snap: Any = snap[2]  # pyright: ignore[reportAny]
        if state_snap is not None:
            c.state = state_snap
        if meta_snap is not None:
            c.meta_state = meta_snap


def _snapshot_cache(cache: list[Any]) -> list[Any]:
    """Snapshot every cache entry for speculative rollback. See _snapshot_one."""
    return [_snapshot_one(c) for c in cache]


def _restore_cache(cache: list[Any], snap: list[Any]) -> None:
    """Restore cache from a `_snapshot_cache` snapshot. See _restore_one."""
    for c, s in zip(cache, snap):
        _restore_one(c, s)


# ---------------------------------------------------------------------------
# Speculative pipeline layer wrappers (subclass, not modify)
# ---------------------------------------------------------------------------


class SpecPipelineFirstLayer(PipelineFirstLayer):
    """PipelineFirstLayer with PP recv mode for overlapped hidden exchange."""

    def __init__(self, base: PipelineFirstLayer):
        super().__init__(base.original_layer, base.r, base.group)
        self.is_prefill = base.is_prefill
        self._pp_recv: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if _TRACE:
            _log(
                f"SpecPipelineFirstLayer.__call__ r={self.r} _pp_recv={self._pp_recv} "
                f"is_prefill={self.is_prefill}"
            )
        if self._pp_recv and self.r != 0:
            # Recv hidden from previous rank (blocks until rank 0 sends)
            # JACCL/RDMA requires bf16 for transport
            if _TRACE:
                _log(f"SpecPipelineFirstLayer TAKING SPEC RECV BRANCH r={self.r}")
            x_dtype = x.dtype
            x_bf16 = x.astype(mx.bfloat16) if x_dtype != mx.bfloat16 else x
            mx.eval(x_bf16)
            _t_recv_wait0 = time.perf_counter() if _PROF_DSPARK_RECV_WAIT else 0.0
            x = mx.distributed.recv_like(x_bf16, (self.r - 1), group=self.group)
            mx.eval(x)
            if _PROF_DSPARK_RECV_WAIT:
                _DSPARK_RECV_WAIT_ACCUM[0] += time.perf_counter() - _t_recv_wait0
            if x_dtype != mx.bfloat16:
                x = x.astype(x_dtype)
            return self.original_layer(x, *args, **kwargs)
        # Normal path (prefill or rank 0)
        if _TRACE:
            _log(f"SpecPipelineFirstLayer FALLTHROUGH TO BASE r={self.r}")
        return super().__call__(x, *args, **kwargs)


class SpecPipelineLastLayer(PipelineLastLayer):
    """PipelineLastLayer with PP send + speculative modes."""

    def __init__(self, base: PipelineLastLayer):
        super().__init__(base.original_layer, base.r, base.s, base.group)
        self.is_prefill = base.is_prefill
        self.queue_sends = base.queue_sends
        self._pp_send: bool = False
        self._pp_decode: bool = False
        self._speculative: bool = False
        self._state_list: list[mx.array] | None = None
        self._hidden_idx: int = -1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if _TRACE:
            _log(
                f"SpecPipelineLastLayer.__call__ r={self.r} s={self.s} "
                f"_speculative={self._speculative} _pp_send={self._pp_send} "
                f"_pp_decode={self._pp_decode} is_prefill={self.is_prefill}"
            )
        if self._speculative:
            # Speculative mode: compute, store, NO send (don't leak speculation to rank 1)
            output = self.original_layer(x, *args, **kwargs)
            mx.eval(output)
            if self._state_list is not None:
                self._state_list[self._hidden_idx] = output
            return output

        if self._pp_send:
            # Send mode (rank 0): compute, send to rank 1, store locally
            # JACCL/RDMA requires bf16 for transport
            output = self.original_layer(x, *args, **kwargs)
            mx.eval(output)
            if self.r != self.s - 1:
                out_bf16 = (
                    output.astype(mx.bfloat16)
                    if output.dtype != mx.bfloat16
                    else output
                )
                sent = mx.distributed.send(
                    out_bf16, (self.r + 1) % self.s, group=self.group
                )
                mx.eval(sent)
            if self._state_list is not None:
                self._state_list[self._hidden_idx] = output
            return output

        if self._pp_decode:
            # Decode mode (rank 1): compute, store, no comms
            output = self.original_layer(x, *args, **kwargs)
            mx.eval(output)
            if self._state_list is not None:
                self._state_list[self._hidden_idx] = output
            return output

        # Normal path (prefill)
        if _TRACE:
            _log(f"SpecPipelineLastLayer FALLTHROUGH TO BASE r={self.r} s={self.s}")
        return super().__call__(x, *args, **kwargs)


# ---------------------------------------------------------------------------
# Layer replacement (additive — wraps existing layers)
# ---------------------------------------------------------------------------


def _install_spec_layers(
    model: nn.Module,
) -> tuple[SpecPipelineFirstLayer | None, SpecPipelineLastLayer | None]:
    """Replace PipelineFirst/LastLayer with speculative versions. Returns refs."""
    layers = model.layers  # type: ignore
    spec_first: SpecPipelineFirstLayer | None = None
    spec_last: SpecPipelineLastLayer | None = None

    for i, layer in enumerate(layers):
        if isinstance(layer, PipelineFirstLayer) and not isinstance(
            layer, SpecPipelineFirstLayer
        ):
            spec_first = SpecPipelineFirstLayer(layer)
            layers[i] = spec_first
        elif isinstance(layer, PipelineLastLayer) and not isinstance(
            layer, SpecPipelineLastLayer
        ):
            spec_last = SpecPipelineLastLayer(layer)
            layers[i] = spec_last

    return spec_first, spec_last


def _configure_layers(
    spec_first: SpecPipelineFirstLayer | None,
    spec_last: SpecPipelineLastLayer | None,
    *,
    pp_send: bool = False,
    pp_recv: bool = False,
    pp_decode: bool = False,
    speculative: bool = False,
    state_list: list[mx.array] | None = None,
    hidden_idx: int = -1,
) -> None:
    """Configure spec layer modes."""
    if spec_first is not None:
        spec_first._pp_recv = pp_recv
    if spec_last is not None:
        spec_last._pp_send = pp_send
        spec_last._pp_decode = pp_decode
        spec_last._speculative = speculative
        spec_last._state_list = state_list
        spec_last._hidden_idx = hidden_idx


# ---------------------------------------------------------------------------
# Hidden state capture (for MTP integration)
# ---------------------------------------------------------------------------


class _CapturingNorm(nn.Module):
    """Wraps a final RMSNorm to record its input each forward pass.

    Defined at module scope (not nested inside _install_hidden_capture)
    so the isinstance check in _install_hidden_capture is stable across
    chat-completions requests — the previous nested-class form created
    a NEW _CapturingNorm class object per call, defeating the isinstance
    guard and stacking another wrapper on every request. By iter N,
    inner_model.norm became N nested CapturingNorms (perf leak that
    correlates with γ=N decode bistability — captured 2026-05-16).
    """

    def __init__(self, orig: nn.RMSNorm, captured: dict[str, mx.array]):
        super().__init__()
        self._orig = orig
        self.weight = orig.weight
        # Shared dict reference, written to on each forward.
        self._captured = captured

    def __call__(self, x: mx.array) -> mx.array:
        self._captured["pre_norm"] = x
        return self._orig(x)


def _install_hidden_capture(model: nn.Module) -> dict[str, mx.array]:
    """Wrap model's final norm to capture pre-norm hidden state (rank 1 only).

    Returns a dict that will contain {'pre_norm': tensor} after each forward pass.
    The captured tensor is the input to the final RMSNorm — exactly what MTP needs.

    Idempotent: if the model's norm is already a _CapturingNorm (from a
    prior chat-completions request in this process), reuses the same
    capture dict and reset its contents — does NOT stack another wrapper.
    """
    inner = getattr(model, "language_model", model)
    inner_model = getattr(inner, "model", inner)
    current_norm = inner_model.norm

    if isinstance(current_norm, _CapturingNorm):
        # Already installed — reuse the same dict so the spec loop reads
        # from a fresh state but the model graph stays single-wrapper.
        current_norm._captured.clear()
        return current_norm._captured

    captured: dict[str, mx.array] = {}
    inner_model.norm = _CapturingNorm(current_norm, captured)
    return captured


# ---------------------------------------------------------------------------
# Core decode loop with PP idle-time speculation (overlapped)
# ---------------------------------------------------------------------------


def pp_speculative_decode_loop(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: list[Any],
    draft_cache: list[Any],
    sampler: Callable,
    logits_processors: list[Any],
    first_y: mx.array,
    first_logprobs: mx.array,
    max_tokens: int,
    pp_rank: int,
    pp_world_size: int,
    pp_group: mx.distributed.Group,
    mtp_predictor: Any | None = None,
) -> Generator[tuple[int, mx.array], None, None]:
    """PP decode loop with idle-time speculation. Yields (token_id, logprobs).

    Overlapped flow per step:
    1. Rank 0: compute layers 0-29, SEND hidden to rank 1
    2. PARALLEL:
       - Rank 0: draft + speculative forward (during rank 1's compute)
       - Rank 1: RECV hidden, compute layers 30-59, sample token
    3. all_gather: exchange sampled token (both ranks get it)
    4. Hidden exchange: rank 1 sends pre-norm hidden to rank 0 (for MTP)
    5. Verify: if draft matches, skip rank 0's compute next step
    """
    is_rank0 = pp_rank == 0
    is_last_rank = pp_rank == pp_world_size - 1

    # Get model's inner structure for hidden size
    inner = getattr(model, "language_model", model)
    inner_model = getattr(inner, "model", inner)
    embed_tokens = inner_model.embed_tokens
    hidden_size = getattr(embed_tokens, "dims", embed_tokens.weight.shape[1])

    # Find speculative layer wrappers (already installed by caller)
    # FIX (2026-07-17): must scan/install on inner_model.layers (the real,
    # persistent layer list), not inner.layers. For DeepSeek-V4 (no
    # language_model attr, inner falls back to `model` itself), `model`'s
    # own `.layers` is a PROPERTY returning `self.model.pipeline_layers`
    # -- itself a property computing a FRESH slice of the real list on
    # every access. Scanning/mutating that disposable slice has zero
    # effect on what the model's actual forward pass iterates (confirmed
    # via canary prints in SpecPipelineFirstLayer/SpecPipelineLastLayer's
    # __call__ that never fired despite this scan reporting valid-looking
    # non-None spec_first/spec_last). inner_model (resolved just above)
    # is the object whose `.layers` is the real, persistent list.
    spec_first, spec_last = None, None
    for layer in inner_model.layers:  # type: ignore
        if isinstance(layer, SpecPipelineFirstLayer):
            spec_first = layer
        elif isinstance(layer, SpecPipelineLastLayer):
            spec_last = layer
    if spec_first is None and spec_last is None:
        spec_first, spec_last = _install_spec_layers(inner_model)
    _log(
        f"spec layers found: spec_first={'r=' + str(spec_first.r) if spec_first else None} "
        f"spec_last={'r=' + str(spec_last.r) + ' s=' + str(spec_last.s) if spec_last else None} "
        f"pp_rank={pp_rank} pp_world_size={pp_world_size}"
    )

    # State list for hidden exchange
    _cache_state = [c.state if hasattr(c, "state") else c for c in prompt_cache]
    _hidden_idx = len(_cache_state)
    _cache_state.append(mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16))

    # Skip lm_head on rank 0 (saves ~500MB weight reads per step)
    # FIX (2026-07-17): `_skip_lm_head` was DEAD -- the model's actual
    # __call__ (mlx-lm's outer `Model` class) never reads this attribute
    # at all; it only checks a separate, file-based `_get_nop_targets()`
    # debug toggle (/tmp/dsv4_nop_targets) unrelated to this flag. Rank0's
    # _rank0_compute/_rank0_speculative_fwd both call model(...) and
    # DISCARD the return value entirely (only rank1's _rank1_compute
    # output is ever used for sampling) -- confirmed rank0's forward
    # here only needs the LAYER-STACK's hidden-state side effect (sent
    # to rank1 / cached for a later accepted-draft send), never the
    # logits. Empirically confirmed via the nop-file mechanism: skipping
    # lm_head cut rank0's per-step draft-forward cost by ~25% (26.8ms ->
    # 19.9ms) on a live test. Since the model has no real per-instance
    # gate for this, swap `lm_head` itself for a cheap zero-cost stand-in
    # for the duration of this decode loop (restored in the `finally`
    # block below) rather than relying on the dead attribute or the
    # global, file-polling debug toggle.
    _lm_head_owner = getattr(model, "language_model", model)
    _real_lm_head = None
    if is_rank0:
        _real_lm_head = getattr(_lm_head_owner, "lm_head", None)
        if _real_lm_head is not None:

            def _nop_lm_head(h: mx.array) -> mx.array:
                # Same leading dims as a real lm_head output would have,
                # zero-cost: no weight read, no matmul. Callers here
                # (_rank0_compute / _rank0_speculative_fwd) discard the
                # return value entirely, so correctness only requires
                # this not raise -- shape/dtype fidelity is irrelevant.
                return h

            _lm_head_owner.lm_head = _nop_lm_head  # type: ignore

    # Install hidden state capture on rank 1 (for MTP feedback)
    _captured: dict[str, mx.array] = {}
    if is_last_rank and mtp_predictor is not None:
        _captured = _install_hidden_capture(model)

    # MTP state (rank 0 only)
    _mtp_hidden: mx.array | None = None  # pre-norm hidden from rank 1
    if is_rank0 and mtp_predictor is not None:
        mtp_predictor.reset_cache()

    # Speculation state
    _draft_token: int | None = None
    _spec_snap: list[Any] | None = None
    _accepted = 0
    _rejected = 0
    _mtp_accepted = 0
    _mtp_rejected = 0

    y = first_y
    logprobs = first_logprobs

    def _rank0_compute(token: mx.array) -> None:
        """Rank 0: forward layers 0-29 in pp_send mode (sends hidden to rank 1)."""
        _configure_layers(
            spec_first,
            spec_last,
            pp_send=True,
            state_list=_cache_state,
            hidden_idx=_hidden_idx,
        )
        with mx.stream(generation_stream):
            model(token[None], cache=prompt_cache)

    def _rank0_speculative_fwd(token_id: int) -> None:
        """Rank 0: speculatively forward draft token (no send)."""
        if spec_last is not None:
            spec_last._speculative = True
            spec_last._pp_send = False
        with mx.stream(generation_stream):
            model(mx.array([[token_id]]), cache=prompt_cache)
            mx.eval(_cache_state[_hidden_idx])
        if spec_last is not None:
            spec_last._speculative = False

    def _rank1_compute(token: mx.array) -> tuple[mx.array, mx.array]:
        """Rank 1: recv hidden, forward layers 30-59, sample."""
        _configure_layers(
            spec_first,
            spec_last,
            pp_recv=True,
            pp_decode=True,
            state_list=_cache_state,
            hidden_idx=_hidden_idx,
        )
        with mx.stream(generation_stream):
            out = model(token[None], cache=prompt_cache)
            out = out[:, -1, :]
            lp = out - mx.logsumexp(out, keepdims=True)
            sampled = sampler(lp)
            return sampled, lp.squeeze(0)

    _log(
        f"decode loop start: max_tokens={max_tokens}, mtp={'yes' if mtp_predictor else 'no'}"
    )

    # ── profiling accumulators ──
    _prof_r0_compute = 0.0
    _prof_r0_draft = 0.0
    _prof_r1_compute = 0.0
    _prof_token_exchange = 0.0
    _prof_hidden_exchange = 0.0
    _prof_verify = 0.0
    _prof_clear_cache = 0.0
    _prof_total = 0.0

    try:
        n = 0
        while n < max_tokens:
            _loop_tic = time.perf_counter()

            # ==== RANK 0: compute + send hidden ====
            _t0 = time.perf_counter()
            if is_rank0:
                if _draft_token is None:
                    _log(f"n={n} r0_compute ENTER (fresh forward, no draft pending)")
                    _rank0_compute(y)
                    _log(f"n={n} r0_compute EXIT")
                else:
                    mx.eval(_cache_state[_hidden_idx])
                    _to_send = _cache_state[_hidden_idx]
                    if _to_send.dtype != mx.bfloat16:
                        _to_send = _to_send.astype(mx.bfloat16)
                    _log(f"n={n} r0_hidden_send PRE (from cached speculative fwd)")
                    sent = mx.distributed.send(
                        _to_send, (pp_rank + 1) % pp_world_size, group=pp_group
                    )
                    mx.eval(sent)
                    _log(f"n={n} r0_hidden_send POST")
            _dt_r0_compute = time.perf_counter() - _t0
            _prof_r0_compute += _dt_r0_compute

            # ==== RANK 1: recv hidden + compute + sample ====
            _t0 = time.perf_counter()
            if is_last_rank:
                sampled, lp = _rank1_compute(y)
            _dt_r1_compute = time.perf_counter() - _t0
            _prof_r1_compute += _dt_r1_compute

            # ==== TOKEN EXCHANGE (both ranks sync here) ====
            # FIX (2026-07-17): this was mx.distributed.all_gather(group=pp_group)
            # -- a genuine collective op on the same jaccl mesh group that
            # PipelineFirstLayer/PipelineLastLayer use for their own automatic
            # p2p send/recv handoff during decode. Mixing a collective with
            # raw p2p sends/recvs on one transport starved the jaccl
            # reliability layer's ack bookkeeping: reproduced consistently as
            # "[jaccl] drain_acks STALLED ... UC completion lost" a few steps
            # into PP+MTP decode, forcing a runner crash + re-place every
            # time. For world_size=2 (the only topology this module
            # supports -- see is_rank0/is_last_rank throughout), all_gather's
            # result is just fan-out of the last rank's own contribution:
            # gathered_token[-1] is rank (world_size-1)'s value, i.e.
            # is_last_rank's own `sampled`, unchanged. Replaced with a plain
            # send (last rank -> rank 0) + recv (rank 0 only), matching the
            # send/recv discipline already used by every other cross-rank
            # exchange in this file (hidden-state exchange right below,
            # PipelineFirstLayer/PipelineLastLayer's own handoff) instead of
            # introducing a second, collective-based protocol on the same
            # transport.
            _t0 = time.perf_counter()
            if is_last_rank:
                final_token = sampled.reshape(1)
                mx.eval(final_token)
                _log(f"n={n} tok_send PRE (rank{pp_rank}->0)")
                _sent_tok = mx.distributed.send(final_token, 0, group=pp_group)
                mx.eval(_sent_tok)
                _log(f"n={n} tok_send POST")
            elif is_rank0:
                _log(f"n={n} tok_recv PRE (rank0<-{pp_world_size - 1})")
                final_token = mx.distributed.recv_like(
                    mx.zeros(1, dtype=mx.int32),
                    pp_world_size - 1,
                    group=pp_group,
                )
                mx.eval(final_token)
                _log(f"n={n} tok_recv POST")
            else:
                # Middle ranks (pp_world_size > 2): not a supported topology
                # elsewhere in this module (only is_rank0/is_last_rank are
                # ever branched on), so there is nothing meaningful to do
                # here today. Keep a defined value rather than an
                # UnboundLocalError if this module is ever extended to >2
                # ranks without updating this exchange.
                final_token = mx.zeros(1, dtype=mx.int32)
            _dt_tok_xchg = time.perf_counter() - _t0
            _prof_token_exchange += _dt_tok_xchg

            # ==== HIDDEN STATE EXCHANGE (rank 1 → rank 0, for MTP drafting) ====
            # ORDERING FIX (2026-07-17): both this exchange and the token
            # exchange above used to run AFTER rank 0's draft block (MTP
            # predict + speculative forward). That block's cost is variable
            # and, on its first real invocation (n=1 -- n=0 always skips it,
            # no prior MTP hidden yet), can run long enough that rank 0
            # doesn't get around to posting/evaluating its recv for many
            # seconds -- meanwhile rank 1 is already blocked inside
            # mx.distributed.send(), waiting for rank 0 to consume it, and
            # gives up with "[jaccl] send() deadline in drain" once its own
            # retry budget is exhausted. Reproduced live, confirmed via
            # per-rank tracing: rank 1 logged "tok_send PRE" with no
            # matching POST for 15s before crashing, while rank 0 was still
            # inside the draft try/except the whole time, never having
            # reached "tok_recv PRE".
            #
            # Fix: move both exchanges to run immediately after rank 1's
            # compute (i.e. as soon as rank 0's own local work for this
            # step is done), BEFORE the draft block, so rank 0 posts and
            # evaluates its recvs promptly regardless of how long drafting
            # takes. This does NOT change which hidden state the draft
            # uses -- _mtp_draft_input, saved below before this exchange
            # overwrites _mtp_hidden, is deliberately the value captured by
            # the PREVIOUS iteration's exchange (drafting the token that
            # follows `y` must condition on the hidden state that produced
            # `y`, not on the hidden state this iteration's exchange is
            # about to deliver, which corresponds to the token about to be
            # verified). Only the WHEN of posting the recv changed, not
            # which value each iteration's draft consumes.
            _mtp_draft_input = _mtp_hidden
            _t0 = time.perf_counter()
            if mtp_predictor is not None:
                if is_last_rank and "pre_norm" in _captured:
                    _pn = _captured["pre_norm"][:, -1:, :].astype(mx.bfloat16)
                    mx.eval(_pn)
                    _log(f"n={n} hidden_send PRE (rank{pp_rank}->0, MTP feedback)")
                    _sent = mx.distributed.send(_pn, 0, group=pp_group)
                    mx.eval(_sent)
                    _log(f"n={n} hidden_send POST")
                elif is_rank0:
                    _log(
                        f"n={n} hidden_recv PRE (rank0<-{pp_world_size - 1}, MTP feedback)"
                    )
                    _mtp_hidden = mx.distributed.recv_like(
                        mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16),
                        pp_world_size - 1,
                        group=pp_group,
                    )
                    mx.eval(_mtp_hidden)
                    _log(f"n={n} hidden_recv POST")
                    # Cast back to model's compute dtype after bf16 transport
                    if _mtp_hidden.dtype != mx.float16:
                        from exo.worker.engines.mlx.patches.qwen3_5_moe.common import (
                            COMPUTE_DTYPE,
                        )

                        _mtp_hidden = _mtp_hidden.astype(COMPUTE_DTYPE)
            _dt_hidden_xchg = time.perf_counter() - _t0
            _prof_hidden_exchange += _dt_hidden_xchg

            # -- Draft AFTER this step's exchanges complete (see ordering-fix
            # comment above hidden exchange for why this moved here from
            # before rank1_compute/the exchanges). Uses _mtp_draft_input
            # (captured above, BEFORE this iteration's hidden exchange
            # overwrote _mtp_hidden with the value for NEXT iteration's
            # draft) so the draft still targets "the token that follows y",
            # exactly as before the reorder -- only the wall-clock position
            # of this block changed, not which MTP hidden state it reads. --
            _t0 = time.perf_counter()
            if is_rank0:
                _used_mtp = False
                try:
                    if mtp_predictor is not None and _mtp_draft_input is not None:
                        logits = mtp_predictor.predict(
                            _mtp_draft_input, mx.array([[y.item()]])
                        )
                        draft_tok = logits.argmax(axis=-1)
                        mx.eval(draft_tok)
                        _draft_token = int(draft_tok.item())
                        _used_mtp = True
                    elif mtp_predictor is None and draft_model is not None:
                        draft_logits = draft_model(
                            mx.array([[y.item()]]), cache=draft_cache
                        )
                        draft_tok = draft_logits[0, -1].argmax()
                        mx.eval(draft_tok)
                        _draft_token = int(draft_tok.item())

                    if _draft_token is not None:
                        _spec_snap = _snapshot_cache(prompt_cache)
                        _rank0_speculative_fwd(_draft_token)
                        _log(
                            f"n={n} drafted={_draft_token} ({'mtp' if _used_mtp else 'draft'})"
                        )
                except Exception as _draft_err:
                    _log(f"n={n} draft FAILED: {_draft_err}")
                    _draft_token = None
                    _spec_snap = None
                    if spec_last is not None:
                        spec_last._speculative = False
            _dt_r0_draft = time.perf_counter() - _t0
            _prof_r0_draft += _dt_r0_draft

            # ==== VERIFY draft ====
            _t0 = time.perf_counter()
            if is_rank0 and _draft_token is not None:
                real_token = int(final_token.item())
                if real_token == _draft_token:
                    _accepted += 1
                    if _used_mtp:
                        _mtp_accepted += 1
                    _log(f"n={n} ACCEPT draft={_draft_token}")
                    if not _used_mtp and draft_model is not None:
                        draft_model(mx.array([[real_token]]), cache=draft_cache)
                else:
                    _rejected += 1
                    if _used_mtp:
                        _mtp_rejected += 1
                    _log(f"n={n} REJECT draft={_draft_token} real={real_token}")
                    if _spec_snap is not None:
                        _restore_cache(prompt_cache, _spec_snap)
                    if mtp_predictor is not None and mtp_predictor.kv_cache is not None:
                        mtp_predictor.kv_cache.trim(1)
                    _spec_snap = None
                    _draft_token = None
                    if not _used_mtp and draft_model is not None:
                        draft_model(mx.array([[real_token]]), cache=draft_cache)
            elif is_rank0:
                if mtp_predictor is None and draft_model is not None:
                    draft_model(
                        mx.array([[int(final_token.item())]]), cache=draft_cache
                    )
            _dt_verify = time.perf_counter() - _t0
            _prof_verify += _dt_verify

            yield int(final_token.item()), lp if is_last_rank else mx.zeros(1)

            y = final_token
            n += 1

            _dt_clear_cache = 0.0
            if n % 2048 == 0:
                _t0 = time.perf_counter()
                mx.clear_cache()
                _dt_clear_cache = time.perf_counter() - _t0
                _prof_clear_cache += _dt_clear_cache

            _loop_dt = time.perf_counter() - _loop_tic
            _prof_total += _loop_dt

            # ── per-step outlier log (>25ms) ──
            if _TRACE and _loop_dt > 0.025:
                logger.info(
                    f"[PROF pp-spec OUTLIER R{pp_rank} n={n}] "
                    f"loop={_loop_dt * 1000:.1f}ms "
                    f"r0_compute={_dt_r0_compute * 1000:.1f}ms "
                    f"r0_draft={_dt_r0_draft * 1000:.1f}ms "
                    f"r1_compute={_dt_r1_compute * 1000:.1f}ms "
                    f"tok_xchg={_dt_tok_xchg * 1000:.1f}ms "
                    f"hidden_xchg={_dt_hidden_xchg * 1000:.1f}ms "
                    f"verify={_dt_verify * 1000:.1f}ms "
                    f"clear_cache={_dt_clear_cache * 1000:.1f}ms "
                    f"draft_accepted={'yes' if is_rank0 and _draft_token is not None else 'no'}"
                )

            # ── periodic profiling log ──
            if _TRACE and n % 64 == 0:
                _n = 64
                logger.info(
                    f"[PROF pp-spec R{pp_rank} x{_n}] "
                    f"r0_compute={_prof_r0_compute / _n * 1000:.2f}ms "
                    f"r0_draft={_prof_r0_draft / _n * 1000:.2f}ms "
                    f"r1_compute={_prof_r1_compute / _n * 1000:.2f}ms "
                    f"tok_xchg={_prof_token_exchange / _n * 1000:.2f}ms "
                    f"hidden_xchg={_prof_hidden_exchange / _n * 1000:.2f}ms "
                    f"verify={_prof_verify / _n * 1000:.2f}ms "
                    f"clear_cache={_prof_clear_cache / _n * 1000:.2f}ms "
                    f"loop={_prof_total / _n * 1000:.2f}ms"
                )
                _prof_r0_compute = 0.0
                _prof_r0_draft = 0.0
                _prof_r1_compute = 0.0
                _prof_token_exchange = 0.0
                _prof_hidden_exchange = 0.0
                _prof_verify = 0.0
                _prof_clear_cache = 0.0
                _prof_total = 0.0

    finally:
        # Restore model state
        _configure_layers(spec_first, spec_last)  # all modes off
        if is_rank0 and _real_lm_head is not None:
            _lm_head_owner.lm_head = _real_lm_head  # type: ignore
            # Guard against MLX's Module.__setattr__ shadowing: a plain
            # (non-Module) value assigned over a submodule attribute is
            # stored in the instance __dict__ and popped from the
            # parameter/module tree; restoring the real nn.Linear should
            # both rebind the attribute AND re-register it in the tree.
            # If some MLX version doesn't clean the __dict__ shadow entry
            # on restore, `model.lm_head` would keep resolving to the
            # nop function forever (parameters()/mx.eval would see the
            # real Linear, but calls would silently keep skipping it) --
            # assert here so that regression fails loud in this decode
            # loop's own logs instead of silently corrupting every
            # request after the first PP+MTP one on this process.
            assert _lm_head_owner.lm_head is _real_lm_head, (  # type: ignore
                "lm_head restore failed: instance __dict__ still "
                "shadows the real nn.Linear with the nop stand-in"
            )
            assert "lm_head" in _lm_head_owner, (
                "lm_head restore failed: not re-registered in the "
                "module/parameter tree after restore"
            )

        total = _accepted + _rejected
        if total > 0:
            logger.debug(
                f"PP speculation: {_accepted}/{total} accepted ({_accepted / total * 100:.0f}%)"
            )
        mtp_total = _mtp_accepted + _mtp_rejected
        if mtp_total > 0:
            logger.debug(
                f"MTP: {_mtp_accepted}/{mtp_total} accepted ({_mtp_accepted / mtp_total * 100:.0f}%), "
                f"draft-fallback: {total - mtp_total} steps"
            )


# ---------------------------------------------------------------------------
# Chained multi-token MTP draft + batched verify (opt-in, EXO_PP_MTP_CHAIN_K>1)
# ---------------------------------------------------------------------------
#
# Single-token PP+MTP (pp_speculative_decode_loop above) is structurally
# capped near ~1/rank1_step_time tok/s regardless of MTP draft-acceptance
# rate: rank1 (the "verify" rank, owning the model's last N layers) always
# does exactly one full forward per loop iteration, whether the draft is
# accepted or not. At long context (500K+), that forward is ~55-60ms,
# dominated by KV-cache read bandwidth -- capping throughput around 17-20
# tok/s even at high (~80-90%) acceptance. Confirmed via direct profiling
# on a live cluster, 2026-07-17/18.
#
# Mechanism: rank0 chains k MTP-head-only forwards (mtp_predictor.predict
# with return_hidden=True feeding each step's predicted token + returned
# hidden into the next) to cheaply draft k candidate tokens WITHOUT
# touching the main (22/21-layer) model at all for drafting -- the MTP
# head is a single lightweight transformer block, not a full model
# re-forward (unlike the single-token path's _rank0_speculative_fwd,
# which does re-forward the WHOLE main model per draft). Rank0 then does
# ONE main-model forward with [y, d_1, ..., d_{k-1}] as a batched
# sequence (k tokens total: the current committed token plus k-1 of the
# k drafts -- d_k is deliberately excluded, see trim-arithmetic note
# below) to get k hidden states, sent to rank1 in one message. Rank1
# does ONE forward over those k positions, samples/verifies each
# position's logits against the corresponding draft token in order
# (standard "accept prefix until first mismatch"), and returns a fixed-
# shape 2-int32 message: accepted count m (0<=m<=k-1) and a "bonus
# token" resampled from rank1's own logits at position m (legitimate:
# rank1 already computed real logits there, whether or not the draft at
# that position matched).
#
# Wire protocol is fixed-shape regardless of the ACTUAL value of m, to
# avoid the exact class of jaccl protocol-mismatch bug fixed earlier
# this session (send/recv count or shape divergence between ranks
# causes a transport-level stall, not a clean application-level error).
#
# KV rollback: both ranks appended k-1 speculative positions (d_1
# through d_{k-1}) to their respective caches (main model AND, on
# rank0, the MTP predictor's OWN cache) during this iteration's forward.
# If m < k-1, both ranks trim (k-1-m) positions from their MAIN model
# cache (deterministic given k and m, no extra data needed -- rank0
# already knows k, and m arrives from rank1 in the fixed 2-int32
# message). The MTP predictor's cache trim amount is a SEPARATE
# quantity (it advanced by the chain depth actually drafted, which may
# differ from k-1 if EOS or an error truncated the chain early) and
# must be computed independently, not reused from the main-model trim.
_PP_MTP_CHAIN_K = int(os.environ.get("EXO_PP_MTP_CHAIN_K", "1"))


def _mtp_chain_draft(
    mtp_predictor: Any,
    seed_hidden: mx.array,
    seed_token: int,
    depth: int,
) -> tuple[list[int], mx.array]:
    """Chain ``depth`` MTP-head-only predict() calls to draft candidate
    tokens cheaply, without touching the main model.

    Args:
        seed_hidden: (1, 1, hidden) pre-norm hidden from the PREVIOUS
            real step (the hidden that produced ``seed_token``).
        seed_token: the token actually committed last step (``y``).
        depth: number of chained draft tokens to produce (k-1 in the
            caller's k-token-batch terminology -- see module docstring
            above for why the seed token itself isn't counted here).

    Returns:
        (draft_token_ids, last_hidden) -- draft_token_ids has length
        <= depth (shorter if a draft step raises, e.g. transient cache
        state issue; caller must handle a short list, NOT assume
        exactly `depth` tokens came back). last_hidden is the final
        chained hidden state (unused today, returned for future
        deeper-chain / diagnostic use).
    """
    draft_ids: list[int] = []
    h = seed_hidden
    tok = seed_token
    for _ in range(depth):
        try:
            logits, h = mtp_predictor.predict(
                h,
                mx.array([[tok]]),
                return_hidden=True,
            )
            draft_tok = logits.argmax(axis=-1)
            mx.eval(draft_tok)
            tok = int(draft_tok.item())
            draft_ids.append(tok)
        except Exception as _chain_err:
            _log(f"mtp chain draft step {len(draft_ids)} FAILED: {_chain_err}")
            break
    return draft_ids, h


def pp_chained_decode_loop(
    model: nn.Module,
    prompt_cache: list[Any],
    sampler: Callable,
    first_y: mx.array,
    first_logprobs: mx.array,
    max_tokens: int,
    pp_rank: int,
    pp_world_size: int,
    pp_group: mx.distributed.Group,
    mtp_predictor: Any,
    chain_k: int,
) -> Generator[tuple[int, mx.array], None, None]:
    """PP decode loop with k-token chained MTP draft + batched verify.

    Opt-in replacement for pp_speculative_decode_loop's single-token
    path when EXO_PP_MTP_CHAIN_K>1. See the module-level comment block
    above _PP_MTP_CHAIN_K for the full design rationale and wire
    protocol. Requires mtp_predictor (native DSv4 MTP head) -- the
    generic draft-model fallback path is not supported here (chaining
    relies on predict()'s return_hidden=True, which the Qwen3.5-style
    MTPPredictor class does not implement identically -- scope this to
    DSv4MTPPredictor only for now).
    """
    is_rank0 = pp_rank == 0
    is_last_rank = pp_rank == pp_world_size - 1
    k = max(2, chain_k)  # k=1 has no batching benefit; caller should
    # have dispatched to pp_speculative_decode_loop instead, but clamp
    # defensively rather than silently no-op the whole mechanism.

    inner = getattr(model, "language_model", model)
    inner_model = getattr(inner, "model", inner)
    embed_tokens = inner_model.embed_tokens
    hidden_size = getattr(embed_tokens, "dims", embed_tokens.weight.shape[1])

    spec_first, spec_last = None, None
    for layer in inner_model.layers:  # type: ignore
        if isinstance(layer, SpecPipelineFirstLayer):
            spec_first = layer
        elif isinstance(layer, SpecPipelineLastLayer):
            spec_last = layer
    if spec_first is None and spec_last is None:
        spec_first, spec_last = _install_spec_layers(inner_model)

    _captured: dict[str, mx.array] = {}
    if is_last_rank:
        _captured = _install_hidden_capture(model)

    if is_rank0:
        mtp_predictor.reset_cache()

    _lm_head_owner = getattr(model, "language_model", model)
    _real_lm_head = None
    if is_rank0:
        _real_lm_head = getattr(_lm_head_owner, "lm_head", None)
        if _real_lm_head is not None:

            def _nop_lm_head(h: mx.array) -> mx.array:
                return h

            _lm_head_owner.lm_head = _nop_lm_head  # type: ignore
            _log(
                f"lm_head nop check: mtp_predictor.lm_head is real_lm_head = "
                f"{getattr(mtp_predictor, 'lm_head', None) is _real_lm_head}, "
                f"mtp_predictor.lm_head is nop = "
                f"{getattr(mtp_predictor, 'lm_head', None) is _nop_lm_head}"
            )

    y = first_y
    # first_logprobs unused here -- kept in the function signature only
    # for parity with the PP priming call site's shared setup pattern
    # (see pp_speculative_decode_loop's identical parameter).
    del first_logprobs
    # Seed hidden for the MTP chain: the pre-norm hidden that PRODUCED
    # `y`. On the very first iteration this comes from the ordinary PP
    # priming step (stream_generate, before this loop starts) the same
    # way pp_speculative_decode_loop's _mtp_hidden starts as None and
    # gets populated by the first real hidden-exchange -- see that
    # function's own n=0 handling for the precedent this mirrors.
    _mtp_seed_hidden: mx.array | None = None

    _accepted_total = 0
    _drafted_total = 0

    _log(f"chained decode loop start: max_tokens={max_tokens}, k={k}")

    try:
        n = 0
        while n < max_tokens:
            # ==== RANK 0: draft k-1 tokens via cheap MTP-head-only chain ====
            draft_ids: list[int] = []
            if is_rank0 and _mtp_seed_hidden is not None:
                draft_ids, _ = _mtp_chain_draft(
                    mtp_predictor,
                    _mtp_seed_hidden,
                    int(y.item()),
                    k - 1,
                )
                _drafted_total += len(draft_ids)
            # Pad with a sentinel that can never match a real sampled
            # token id (real ids are >= 0) if the chain came back short
            # (error, or fewer than k-1 requested -- shouldn't happen
            # today since _mtp_chain_draft always requests exactly k-1,
            # but a short list from an exception must still produce a
            # fixed-length batch for the main-model forward and wire
            # message). This keeps the wire protocol fixed-shape
            # regardless of how many real drafts came back.
            while len(draft_ids) < k - 1:
                draft_ids.append(-1)

            # ==== RANK 0: ONE main-model forward over [y, d_1..d_{k-1}] ====
            if is_rank0:
                _configure_layers(
                    spec_first, spec_last, pp_send=True, state_list=None, hidden_idx=-1
                )
                _batch_ids = [int(y.item())] + [
                    max(0, d)
                    for d in draft_ids  # sentinel -1 -> 0 for the
                    # forward itself (never sampled/committed if rejected;
                    # using 0 keeps embed_tokens lookup in-bounds without
                    # affecting correctness, since verify will reject any
                    # position whose draft was the -1 sentinel anyway).
                ]
                _batch_tok = mx.array([_batch_ids])
                with mx.stream(generation_stream):
                    model(_batch_tok, cache=prompt_cache)

            # ==== RANK 1: recv k-token hidden batch (via layer wrapper's
            # existing recv, now naturally k-wide -- SpecPipelineFirstLayer's
            # recv branch is shape-agnostic, no changes needed there) +
            # ONE forward + per-position verify ====
            accepted_ids: list[int] = []
            bonus_token: int | None = None
            m = 0
            _next_seed_hidden: mx.array | None = None
            if is_last_rank:
                _configure_layers(
                    spec_first,
                    spec_last,
                    pp_recv=True,
                    pp_decode=True,
                    state_list=None,
                    hidden_idx=-1,
                )
                with mx.stream(generation_stream):
                    _batch_tok_r1 = mx.array(
                        [[int(y.item())] + [max(0, d) for d in draft_ids]]
                    )
                    out = model(_batch_tok_r1, cache=prompt_cache)
                    # out: (1, k, vocab). Position i's logits predict the
                    # token AFTER input position i -- i.e. out[:, i, :]
                    # is what should equal draft_ids[i] (the token fed at
                    # input position i+1), for i in 0..k-2. Standard
                    # speculative-decoding verify: accept the longest
                    # matching PREFIX, starting from i=0.
                    lp_all = out - mx.logsumexp(out, axis=-1, keepdims=True)
                    if draft_ids[0] < 0:
                        # No real draft available at all (cold start --
                        # _mtp_seed_hidden was None going into this
                        # iteration, e.g. n=0 before rank0's first chain
                        # has anything to seed from). Fall back to a
                        # plain single-token forward-and-sample at
                        # position 0 (predicting what follows `y`) so
                        # this iteration still makes real progress
                        # instead of silently producing zero accepted +
                        # zero bonus tokens forever. BUG (2026-07-18,
                        # found via live test): the original code just
                        # `break`-ed here with bonus_token left at its
                        # None default, m=0, accepted_ids=[] -- meaning
                        # NOTHING was ever yielded and `n` never
                        # advanced, an infinite no-progress loop on the
                        # very first iteration every single time.
                        _bonus_sampled = sampler(lp_all[:, 0, :])
                        mx.eval(_bonus_sampled)
                        bonus_token = int(_bonus_sampled.item())
                    else:
                        for i in range(k - 1):
                            if draft_ids[i] < 0:
                                # Chain ran short mid-way (rare: an
                                # exception inside _mtp_chain_draft
                                # truncated it below k-1). Position i's
                                # logits are still real (the forward ran
                                # over the full padded batch) -- sample
                                # here rather than silently stopping with
                                # no bonus token, same reasoning as the
                                # n=0 cold-start case above.
                                _bonus_sampled = sampler(lp_all[:, i, :])
                                mx.eval(_bonus_sampled)
                                bonus_token = int(_bonus_sampled.item())
                                break
                            pos_sampled = sampler(lp_all[:, i, :])
                            mx.eval(pos_sampled)
                            real_tok = int(pos_sampled.item())
                            if real_tok == draft_ids[i]:
                                accepted_ids.append(real_tok)
                                m += 1
                            else:
                                # Reject: this position's REAL sampled token
                                # becomes the bonus token (legitimate --
                                # rank1 already computed real logits here).
                                bonus_token = real_tok
                                break
                        else:
                            # All k-1 drafts accepted -- bonus token comes
                            # from position k-1 (the last real forward
                            # position, predicting what follows d_{k-1}).
                            _bonus_sampled = sampler(lp_all[:, k - 1, :])
                            mx.eval(_bonus_sampled)
                            bonus_token = int(_bonus_sampled.item())
                    _pn = _captured.get("pre_norm")
                    # Seed hidden for rank0's NEXT chain: the pre-norm
                    # hidden at whichever position corresponds to the
                    # last COMMITTED token this iteration (index m, since
                    # accepted_ids has length m and the bonus token sits
                    # logically at position m too).
                    _seed_idx = min(m, k - 1)
                    _next_seed_hidden = (
                        _pn[:, _seed_idx : _seed_idx + 1, :].astype(mx.bfloat16)
                        if _pn is not None
                        else None
                    )
            # ==== FIXED-SHAPE WIRE: rank1 -> rank0, (m, bonus_token) ====
            if is_last_rank:
                _wire = mx.array(
                    [m, bonus_token if bonus_token is not None else 0], dtype=mx.int32
                )
                mx.eval(_wire)
                _log(
                    f"n={n} chain_verify_send PRE (rank1->0) m={m} bonus={bonus_token}"
                )
                _sent_wire = mx.distributed.send(_wire, 0, group=pp_group)
                mx.eval(_sent_wire)
                _log(f"n={n} chain_verify_send POST")
            elif is_rank0:
                _log(f"n={n} chain_verify_recv PRE (rank0<-{pp_world_size - 1})")
                _wire = mx.distributed.recv_like(
                    mx.zeros(2, dtype=mx.int32),
                    pp_world_size - 1,
                    group=pp_group,
                )
                mx.eval(_wire)
                _log(f"n={n} chain_verify_recv POST")
                m = int(_wire[0].item())
                bonus_token = int(_wire[1].item())
            else:
                m = 0
                bonus_token = 0

            # ==== FIXED-SHAPE WIRE: rank1 -> rank0, next seed hidden ====
            if mtp_predictor is not None:
                if is_last_rank:
                    _hs = (
                        _next_seed_hidden
                        if _next_seed_hidden is not None
                        else mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16)
                    )
                    mx.eval(_hs)
                    _log(f"n={n} chain_hidden_send PRE (rank1->0)")
                    _sent_h = mx.distributed.send(_hs, 0, group=pp_group)
                    mx.eval(_sent_h)
                    _log(f"n={n} chain_hidden_send POST")
                elif is_rank0:
                    _log(f"n={n} chain_hidden_recv PRE (rank0<-{pp_world_size - 1})")
                    _mtp_seed_hidden = mx.distributed.recv_like(
                        mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16),
                        pp_world_size - 1,
                        group=pp_group,
                    )
                    mx.eval(_mtp_seed_hidden)
                    # BUG FIX (2026-07-18, found via live test showing 0%
                    # MTP draft acceptance despite the working single-
                    # token path getting 70-90% with the same predictor):
                    # this cast-back was MISSING here. The working path's
                    # equivalent hidden-exchange recv (pp_speculative_
                    # decode_loop, `_mtp_hidden`) explicitly casts back
                    # from the bf16 wire dtype to the model's real compute
                    # dtype before ever feeding it to mtp_predictor.predict().
                    # Without this, the MTP head was silently computing on
                    # raw bf16 input (a lossy, wrong-precision reinterpret
                    # relative to what its weights expect), producing
                    # numerically wrong -- but not exception-raising --
                    # predictions on every single call.
                    if _mtp_seed_hidden.dtype != mx.float16:
                        from exo.worker.engines.mlx.patches.qwen3_5_moe.common import (
                            COMPUTE_DTYPE,
                        )

                        _mtp_seed_hidden = _mtp_seed_hidden.astype(COMPUTE_DTYPE)
                    _log(f"n={n} chain_hidden_recv POST")

            # ==== KV ROLLBACK (both ranks, deterministic given k and m) ====
            # Main model cache: this iteration's forward appended k-1
            # speculative positions (d_1..d_{k-1}). Trim back whatever
            # wasn't actually committed. Committed count this iteration
            # is m (accepted drafts) -- the bonus token is NOT yet in
            # any cache (it's freshly sampled, fed as `y` next iter's
            # forward input, same as the k=1 path's committed token).
            _trim_n = (k - 1) - m
            if _trim_n > 0:
                for c in prompt_cache:
                    if hasattr(c, "trim"):
                        c.trim(_trim_n)
            # MTP predictor cache: rank0's chain advanced its OWN cache
            # by len(draft_ids) positions this iteration (computed
            # independently from the main-model trim -- see module
            # docstring). Trim back to m accepted positions.
            if (
                is_rank0
                and mtp_predictor is not None
                and mtp_predictor.kv_cache is not None
            ):
                _mtp_drafted_this_iter = sum(1 for d in draft_ids if d >= 0)
                _mtp_trim_n = _mtp_drafted_this_iter - m
                if _mtp_trim_n > 0:
                    mtp_predictor.kv_cache.trim(_mtp_trim_n)

            # ==== Yield accepted tokens + bonus token ====
            for tok_id in accepted_ids:
                yield tok_id, mx.zeros(1)
                n += 1
                if n >= max_tokens:
                    break
            if n < max_tokens and bonus_token is not None:
                yield bonus_token, mx.zeros(1)
                y = mx.array([bonus_token])
                n += 1
            elif accepted_ids:
                y = mx.array([accepted_ids[-1]])

            _accepted_total += m

    finally:
        _configure_layers(spec_first, spec_last)
        if is_rank0 and _real_lm_head is not None:
            _lm_head_owner.lm_head = _real_lm_head  # type: ignore
        if _drafted_total > 0:
            logger.debug(
                f"PP chained MTP: {_accepted_total}/{_drafted_total} "
                f"drafted tokens accepted ({_accepted_total / _drafted_total * 100:.0f}%), "
                f"k={k}"
            )


# ---------------------------------------------------------------------------
# PP + DSpark (opt-in, EXO_PP_DSPARK=1): rank1-owned draft+verify cycle
# ---------------------------------------------------------------------------
#
# DSpark (arXiv:2607.05147) is a dedicated 3-stage semi-autoregressive draft
# head -- a completely separate, more capable mechanism than the plain
# single-layer MTP head (mtp[0]) the pp_chained_decode_loop/pp_speculative_
# decode_loop functions above target. ONE parallel forward over
# [anchor, noise x (block_size-1)] produces draft logits for the WHOLE
# block at once (not a sequential feed-back-in chain), followed by a
# lightweight Markov correction pass and confidence scoring.
#
# CRITICAL PP-specific fact (found 2026-07-18): DSpark's context-conditioning
# mechanism (append_ctx, fed by hc-mean hidden states captured at specific
# GLOBAL target layers -- e.g. layers 40-42) and the real lm_head both live
# on RANK1 (the rank owning the model's LAST layers) in our 2-rank PP split,
# not rank0. This is the OPPOSITE of pp_speculative_decode_loop's design,
# which puts the drafter on rank0 to use its otherwise-idle time. That
# overlap doesn't even apply to DSpark: drafting can't start until rank1's
# own forward has produced the tap-layer hiddens for THIS cycle, so
# "rank0 drafts during rank1's compute" has no analogue here regardless.
#
# Architecture: RANK1 owns 100% of the DSpark draft+verify+accept logic,
# entirely self-contained (own model, own cache, own lm_head, own DSpark
# module/cache) -- rank0 contributes NOTHING to the drafting itself. Rank0's
# only role is a "dumb" forward of whatever fixed-width token batch rank1
# tells it to run (this cycle's committed token + rank1's just-drafted
# candidates), to keep rank0's own KV cache in sync and produce the hidden
# state rank1 needs to continue its own forward through the pipeline
# boundary (the existing SpecPipelineFirstLayer/LastLayer send/recv,
# unmodified, now naturally (block_size+1)-wide instead of single-token).
#
# Per-cycle protocol (rank1 drives, rank0 follows):
#   1. rank1 (from the END of the previous cycle): already knows this
#      cycle's committed token `y` and has already drafted d_0..d_{bs-1}
#      via _dspark.draft() (self-contained, rank1-only, no rank0 needed).
#   2. rank1 -> rank0: ONE fixed-shape message, the (bs+1)-token batch
#      [y, d_0, ..., d_{bs-1}] as int32 (bs = dspark block_size, config-
#      fixed per checkpoint -- NOT sample-dependent, so the message shape
#      never varies run-to-run).
#   3. rank0: trims its own KV by however many of the PREVIOUS cycle's
#      speculative positions weren't actually committed (computed from
#      the accept-count rank1 sends back at the end of ITS OWN previous
#      cycle -- see step 6), then forwards the (bs+1)-token batch through
#      its own layers, sending the resulting hidden batch to rank1 via
#      the existing pipeline boundary send (unmodified).
#   4. rank1: receives the hidden batch (existing pipeline boundary recv,
#      unmodified), continues its own forward through its own layers
#      (this is what naturally captures the tap-layer hiddens DSpark
#      needs for cycle N+1's append_ctx, as a side effect of computing
#      real logits for verification).
#   5. rank1: verifies (accept-longest-matching-prefix against d_0..
#      d_{bs-1}, temp=0 argmax-equality per the established TP pattern),
#      computes n_accepted and the bonus token, commits DSpark ctx for
#      the accepted prefix via append_ctx, trims its OWN dspark draft
#      cache back to empty (draft-only KV never persists), drafts the
#      NEXT cycle's d_0..d_{bs-1} using the bonus token as the new anchor
#      (self-contained on rank1 again, no rank0 needed for this step).
#   6. rank1 -> rank0: a SECOND small fixed-shape message -- the trim
#      amount for rank0's OWN main-model KV cache (bs - n_accepted) plus
#      the bonus token id (rank0 needs the bonus token as part of NEXT
#      cycle's batch -- it's the committed `y` for cycle N+1).
#
# Only temp=0 (greedy) is implemented for the first version, matching how
# this was validated tonight -- temp>0 needs the TP path's categorical-
# sampling + rejection-sampling machinery (accept_ratios/uniforms), not
# ported here yet.
_PP_DSPARK_ENABLED = os.environ.get("EXO_PP_DSPARK", "0") == "1"


def pp_dspark_decode_loop(
    model: nn.Module,
    prompt_cache: list[Any],
    first_y: mx.array,
    max_tokens: int,
    pp_rank: int,
    pp_world_size: int,
    pp_group: mx.distributed.Group,
) -> Generator[tuple[int, mx.array], None, None]:
    """PP decode loop with DSpark draft+verify entirely owned by rank1.

    Opt-in via EXO_PP_DSPARK=1. Requires model.model.dspark to already be
    attached (via _overlay_dsv4_dspark, gated on EXO_DSV4_DSPARK=1 at model
    load time) AND the pipeline_start_idx PP-tap-capture fix (auto_parallel.py
    + deepseek_v4.py, 2026-07-18) to be present, or DSpark's context
    conditioning silently never populates under PP (get_dspark_ctx returns
    None every cycle -- draft() still runs, just without ctx conditioning,
    degrading quality/acceptance rather than crashing).
    """
    from mlx_lm.models.deepseek_v4 import get_dspark_ctx

    is_rank0 = pp_rank == 0
    is_last_rank = pp_rank == pp_world_size - 1

    inner = getattr(model, "language_model", model)
    inner_model = getattr(inner, "model", inner)
    embed_tokens = inner_model.embed_tokens

    spec_first, spec_last = None, None
    for layer in inner_model.layers:  # type: ignore
        if isinstance(layer, SpecPipelineFirstLayer):
            spec_first = layer
        elif isinstance(layer, SpecPipelineLastLayer):
            spec_last = layer
    if spec_first is None and spec_last is None:
        spec_first, spec_last = _install_spec_layers(inner_model)

    _captured: dict[str, mx.array] = {}
    if is_last_rank:
        _captured = _install_hidden_capture(model)

    _dspark = getattr(inner_model, "dspark", None)
    if _dspark is None:
        raise RuntimeError(
            "pp_dspark_decode_loop: model.model.dspark not found on rank "
            f"{pp_rank} -- requires EXO_DSV4_DSPARK=1 at model load time "
            "(DSpark must be attached on EVERY rank, even though only "
            "rank1 ever calls .draft()/.append_ctx() on it -- rank0 just "
            "needs its own copy's block_size to size the wire messages)"
        )
    bs = _dspark.block_size
    # Verify-width truncation (2026-07-18, real lever found via the width-
    # scaling sweep above): DSpark's draft() ALWAYS computes the full
    # block_size internally regardless (fixed ~10.6ms cost either way,
    # can't be reduced) -- but the SEPARATE main-model batch send/forward/
    # verify only needs to cover however many of those draft positions we
    # actually choose to verify. The width sweep proved forward cost
    # scales ~linearly with width, NOT flat as earlier profiling wrongly
    # inferred (rank0 ~7.1ms/token + ~14.5ms fixed, rank1 ~7.8ms/token +
    # ~17.7ms fixed) -- so verifying fewer, lower-value tail positions
    # (per-position accept histogram showed positions 3-4 contribute very
    # little expected value: 43%/46% accept-if-reached, on top of already
    # low reach probability) trades a small amount of expected-tokens for
    # a real reduction in wall time per cycle. Projected from the fitted
    # cost model + measured accept histogram: verify_width=2 (commit +
    # only the single highest-value draft position 0) projects ~21.9
    # tok/s vs the current width=6/full-block ~17.3 tok/s -- a ~26%
    # improvement, NOT the ~9% tree-verification would have bought for
    # far more implementation complexity. Default matches today's
    # behavior (verify the whole block) unless explicitly overridden.
    # DEFAULT CHANGED 2026-07-18 after live empirical validation: width=3
    # measured 24.2-26.7 tok/s across 3 repeated runs (vs width=6/full-
    # block's measured 15.8-17.3 tok/s baseline) -- a real, repeatable
    # 53-70% improvement, not just the linear-cost-model projection.
    # width=2 measured slightly slower (23.1 tok/s) than width=3, matching
    # the model's prediction that 3 is close to the sweet spot. Still
    # overridable via EXO_PP_DSPARK_VERIFY_WIDTH for further tuning.
    vw: int = min(bs, int(os.environ.get("EXO_PP_DSPARK_VERIFY_WIDTH", "3")))
    # DEFAULT OFF (2026-07-18): draft()-width truncation (vs main-model
    # verify-width truncation above, which IS safe and stays default-on)
    # found a real correctness bug during validation -- occasional silent
    # BOS-token-spam degeneration, no crash, high reported accept rate
    # (RotatingKVCache.trim() is documented as only valid pre-wraparound;
    # suspected root cause, not yet confirmed). Opt-in only via
    # EXO_PP_DSPARK_DRAFT_WIDTH_TRUNCATE=1 until root-caused and fixed --
    # do NOT flip this default without first confirming the wrap
    # diagnostic below never fires across a real validation pass.
    _draft_width_truncate = (
        os.environ.get("EXO_PP_DSPARK_DRAFT_WIDTH_TRUNCATE", "0") == "1"
    )
    _draft_width = (vw - 1) if _draft_width_truncate else None
    _dsc = _dspark.make_cache() if is_last_rank else None

    def _dspark_sample_greedy(step_logits: mx.array, _k: int) -> mx.array:
        return mx.argmax(step_logits, axis=-1)

    y = first_y
    # rank1's own running state across cycles (self-contained, no rank0
    # visibility needed for any of this except the two small wire messages).
    _draft_ids: list[int] = []

    _accepted_total = 0
    _drafted_total = 0
    # Per-position acceptance histogram (2026-07-18, decisive diagnostic
    # per Fable consult): discriminates "healthy speculative-decode
    # degradation curve" (position 0 high, decaying positions 1-4) from
    # "conditioning/append_ctx bug" (position 0 ITSELF low -- can't be
    # explained by compounding error since position 0 sees the REAL
    # anchor token, not a guessed one). _reached[i] counts cycles that
    # got far enough to check position i; _accepted[i] counts how many
    # of those actually matched.
    _pos_reached = [0] * bs
    _pos_accepted = [0] * bs
    # Tree-verification feasibility: at each REJECTED position, did the
    # draft head's own rank-2 (second-highest-logit) candidate match what
    # the real model actually picked? High hit rate -> tree verify worth
    # building. Low -> not worth it, the draft head's ranking doesn't
    # correlate with ground truth when it's wrong.
    _pos_rejected_reached = [0] * bs
    _pos_rank2_would_hit = [0] * bs
    _corrected: mx.array | None = None
    # Draft-ahead hit-rate counters (2026-07-19, EXO_PP_DSPARK_DRAFT_AHEAD_LOG
    # -- see the usage site below for full rationale).
    _da_cycles = 0
    _da_full_accept = 0
    _da_would_hit = 0

    # STEP 1 spec-tag diagnostic (2026-07-19, EXO_PP_DSPARK_DRAFT_AHEAD=1,
    # default OFF, orthogonal to the pure-measurement _DRAFT_AHEAD_LOG
    # switch above). Adds a fixed 5-slot int32 tag exchange after msg2
    # so that spec_id packing/unpacking is exercised under real
    # cross-rank timing BEFORE the speculative forward branch (STEP 3)
    # is wired in. Rank1 constructs a SpecId capturing its view of
    # this cycle's assumed prefix and sends it; rank0 unpacks and
    # validates against the SpecId it would independently construct
    # from _wire_batch (which encodes the exact same prefix). Any
    # mismatch means the tag mechanism itself is not sound -- log
    # loudly, but never branch on the result (diagnostic mode always
    # falls through to the existing non-speculative path).
    _draft_ahead_enabled = os.environ.get("EXO_PP_DSPARK_DRAFT_AHEAD", "0") == "1"
    _spec_tag_validator = SpecTagValidator() if _draft_ahead_enabled else None
    _spec_tag_matches = 0
    _spec_tag_mismatches = 0

    # STEP 2b msg1 deep-draft extension (2026-07-19, EXO_PP_DSPARK_DRAFT_AHEAD=1,
    # default OFF -- same gate as the tag-exchange diagnostic above). Extends
    # msg1 by piggybacking a companion int32 array of `vw-1` "extension"
    # draft tokens -- the positions DSpark's draft() already computed for
    # free (see the block_size-vs-verify_width comment in the draft() call
    # site below and in deepseek_v4.py::draft) but that today's protocol
    # never sends across the wire. Once received on rank0, these are
    # purely BYTES ON THE WIRE for now: they're counted / round-trip
    # equality-checked against what rank1 sent (via a fresh SpecTagValidator
    # instance so counters are separate from the msg2 tag exchange), then
    # discarded. NO rank0-side speculative forward is wired to consume
    # them yet -- STEP 3 will. This mirrors the msg2-tag-exchange
    # diagnostic's discipline: prove the wire mechanism itself is sound
    # under real cross-rank timing before adding any decode-path branching.
    #
    # Only enabled when EXO_PP_DSPARK_DRAFT_WIDTH_TRUNCATE=0 (the default):
    # if the drafter is opt-in-truncating to width vw-1 (the OTHER env var,
    # currently off pending a wrap-related correctness bug), then DSpark's
    # draft() returns only vw-1 tokens and there ARE no extra positions to
    # send -- gracefully no-op in that case rather than build a second
    # code path.
    _da_msg1_ext_active = _draft_ahead_enabled and not _draft_width_truncate and vw >= 2
    _da_msg1_ext_len = deep_draft_ext_len(int(vw)) if _da_msg1_ext_active else 0
    _da_msg1_ext_matches = 0
    _da_msg1_ext_mismatches = 0

    # STEP 3 draft-ahead execution gate (2026-07-19,
    # EXO_PP_DSPARK_DRAFT_AHEAD_EXECUTE=1, default OFF). Orthogonal to the
    # STEP 1/2b diagnostic gate above (EXO_PP_DSPARK_DRAFT_AHEAD): when
    # EXECUTE is unset/0, decode is byte-identical to the diagnostic-only
    # behaviour (which itself is byte-identical to the pre-STEP-1 loop
    # apart from the harmless diagnostic tag/ext round-trips). Only when
    # EXECUTE=1 does rank 0 actually perform the speculative forward and
    # do rank 1 skip work on HIT. Requires the diagnostic gate to also be
    # on (the msg1b extension must ride the wire so rank 0 has a batch to
    # speculatively forward, and the msg2 hit/miss slot must be present).
    _draft_ahead_execute = (
        _draft_ahead_enabled
        and _da_msg1_ext_active
        and _da_msg1_ext_len > 0
        and os.environ.get("EXO_PP_DSPARK_DRAFT_AHEAD_EXECUTE", "0") == "1"
    )
    # STEP 3b YIELD substitution gate (2026-07-19,
    # EXO_PP_DSPARK_DRAFT_AHEAD_YIELD=1, default OFF). Orthogonal *third*
    # env var layered on top of STEP 3a's EXECUTE gate. When unset/0 the
    # code path is byte-identical to STEP 3a's shipped behaviour (rank 0
    # restores the KV snapshot and discards the buffered hidden on both
    # HIT and MISS, wall-time-neutral). When set to 1, on a HIT rank 0
    # KEEPS its speculative KV writes AND parks the buffered hidden for
    # the NEXT cycle, which becomes a "consume cycle": no msg1/msg1b
    # exchange, no fresh rank 0 forward, rank 0 just sends the buffered
    # hidden and rank 1 verifies the (vw-1)-position extension batch
    # rank 0 already forwarded. Per the design doc's failure-mode
    # discipline, we deliberately do NOT chain: a consume cycle NEVER
    # attempts a fresh speculative forward of its own, so at most one
    # consecutive HIT is consumed before we fall back to the normal
    # cycle structure.
    _draft_ahead_yield = _draft_ahead_execute and (
        os.environ.get("EXO_PP_DSPARK_DRAFT_AHEAD_YIELD", "0") == "1"
    )
    # Cross-cycle state driven by the previous cycle's hit/miss outcome:
    #   * ``_pending_hit_extension`` -- on BOTH ranks: the extension token
    #     list (length ``vw-1``) that rank 0 speculatively forwarded and
    #     rank 1 must now consume as the CURRENT cycle's batch. ``None``
    #     means the previous cycle was a MISS (or draft-ahead-execute was
    #     off), so the normal msg1/msg1b/rank0-fwd flow runs.
    #   * ``_pending_spec_hidden`` -- on rank 0 only: the buffered hidden
    #     produced by the previous cycle's speculative forward, waiting
    #     to be sent to rank 1 in place of the skipped fwd. ``None``
    #     otherwise. Held on the module-scope buffer via SpecHiddenBuffer
    #     as belt-and-braces (design doc: buffer-until-confirmed) even
    #     though at ``pp_world_size == 2`` a single-slot handle is
    #     equivalent -- keeps SpecId matching in the loop so the same
    #     desync protection the tag exchange gives us is also present
    #     for the hidden itself.
    _spec_hidden_buffer: SpecHiddenBuffer | None = (
        SpecHiddenBuffer() if _draft_ahead_execute else None
    )
    _pending_hit_extension: list[int] | None = None
    _pending_hit_spec_id: SpecId | None = None
    # STEP 3b cross-cycle carry for the YIELD path (both ranks). Written
    # at the tail of a HIT cycle when _draft_ahead_yield is on, read at
    # the top of the very next cycle to activate consume mode. Always
    # cleared after being read so an isolated HIT never leaks state into
    # a later cycle by accident.
    #   * ``_consume_active_next``  -- BOTH ranks. True iff the NEXT cycle
    #     is a consume cycle (skip msg1/msg1b/rank0-fwd, rank 0 sends
    #     buffered hidden, rank 1 verifies the extension batch).
    #   * ``_consume_ext_ids_next`` -- RANK 1 only. The (vw-1)-token
    #     extension batch [bonus_N, ext_N[1:vw-1]] rank 1 will verify.
    #     Snapshot BEFORE the redraft overwrites _draft_ids.
    #   * ``_consume_spec_id_next`` -- RANK 0 only. The SpecId whose
    #     buffered hidden must be released (not discarded) at the top of
    #     the next cycle. Belt-and-braces beyond world_size==2.
    _consume_active_next: bool = False
    _consume_ext_ids_next: list[int] | None = None
    _consume_spec_id_next: SpecId | None = None
    # STEP 3b consume-cycle counters (default off; only meaningful when
    # _draft_ahead_yield is on). Cheap counters used to log the real
    # consume-cycle rate and per-consume-cycle token yield, so live
    # cluster runs can measure whether the wall-time win predicted by
    # the design doc actually materialises.
    _consume_cycles = 0
    _consume_tokens_yielded = 0

    # ── profiling (gated by _TRACE, same pattern as pp_speculative_decode_loop
    # above) -- added 2026-07-18 to get real per-cycle timing before deciding
    # how to restructure the wire protocol (collapse messages / overlap
    # drafting). Do NOT guess bottleneck location from log timestamps alone.
    _prof_cycle_n = 0
    _prof_batch_xchg = 0.0  # rank1->rank0 batch send/recv (message 1)
    _prof_r0_fwd = 0.0  # rank0's forward (includes its internal hidden send)
    _prof_r1_verify_wait = 0.0  # rank1 blocked waiting for rank0's hidden send
    _prof_r1_verify_fwd = 0.0  # rank1's ACTUAL compute (wait subtracted out)
    _prof_draft = 0.0  # rank1's DSpark draft() call for next cycle
    _prof_trim_xchg = 0.0  # rank1->rank0 trim/accept send/recv (message 2)
    _prof_total = 0.0

    _log(
        f"dspark decode loop start: max_tokens={max_tokens}, "
        f"block_size={bs if is_last_rank else '?'}, verify_width={vw}"
    )

    # ── ONE-SHOT width-scaling diagnostic (2026-07-18) ──────────────────
    # Prior profiling assumed forward cost is FLAT across batch widths
    # 1-6 (inferred from "cycle time didn't change much whether 1 or 4
    # tokens got accepted") -- but that inference was never actually
    # tested in isolation, and the user correctly pushed back on treating
    # PP's ~20 tok/s as a hard ceiling without looking harder. If width
    # scaling is NOT flat (real width-1 forwards are meaningfully cheaper
    # than width-6), that reopens sequence-chunked pipelining as a lever:
    # rank0 could send a SMALLER chunk first (cheap, fast), letting rank1
    # start on it while rank0 keeps computing the rest -- genuine
    # cross-rank overlap within a single cycle, unlike anything tried so
    # far. Both ranks run MATCHING widths in lockstep (required -- a
    # width mismatch between ranks would hang the pipeline-boundary
    # send/recv), each width tested via a real forward through this
    # rank's own layers, then mx.eval'd for accurate timing, then
    # trim()'d back off the cache so the real decode cycle below is
    # completely unaffected by this test.
    if os.environ.get("EXO_PP_DSPARK_WIDTH_SWEEP", "0") == "1":
        _sweep_widths = [1, 2, 3, 4, 5, 6]
        _sweep_results: dict[int, float] = {}
        _sweep_lazy_results: dict[int, float] = {}
        for _w in _sweep_widths:
            _sweep_tok = mx.array([[int(y.item())] * _w], dtype=mx.int32)
            _t0 = time.perf_counter()
            if is_rank0:
                _configure_layers(
                    spec_first, spec_last, pp_send=True, state_list=None, hidden_idx=-1
                )
                with mx.stream(generation_stream):
                    _sweep_logits = model(_sweep_tok, cache=prompt_cache)
                    # LAZY-ONLY checkpoint (2026-07-18): time up to just
                    # before mx.eval() isolates the pure Python/graph-
                    # construction cost (tracing ~21 layers' worth of ops
                    # into the lazy graph) from actual GPU dispatch+compute
                    # -- per a second-opinion review of the fitted fixed-
                    # overhead term (~14.5ms/rank0, ~17.7ms/rank1), this is
                    # the 5-minute experiment that settles whether that
                    # intercept is real bandwidth-bound compute (expected,
                    # not shave-able) or avoidable Python-side overhead
                    # (worth investigating further if it's not).
                    _t_lazy = time.perf_counter() - _t0
                    mx.eval(_sweep_logits)
            elif is_last_rank:
                _configure_layers(
                    spec_first,
                    spec_last,
                    pp_recv=True,
                    pp_decode=True,
                    state_list=None,
                    hidden_idx=-1,
                )
                with mx.stream(generation_stream):
                    _sweep_out = model(_sweep_tok, cache=prompt_cache)
                    _t_lazy = time.perf_counter() - _t0
                    mx.eval(_sweep_out)
            else:
                _t_lazy = 0.0
            _dt = time.perf_counter() - _t0
            _sweep_results[_w] = _dt
            _sweep_lazy_results[_w] = _t_lazy
            for _c in prompt_cache:
                if hasattr(_c, "trim"):
                    _c.trim(_w)
            _log(
                f"width-sweep w={_w} dt={_dt * 1000:.2f}ms "
                f"lazy={_t_lazy * 1000:.2f}ms rank={pp_rank}"
            )
        logger.info(
            f"[WIDTH SWEEP R{pp_rank}] "
            + ", ".join(f"w={w}:{dt * 1000:.2f}ms" for w, dt in _sweep_results.items())
        )
        logger.info(
            f"[WIDTH SWEEP LAZY-ONLY R{pp_rank}] "
            + ", ".join(
                f"w={w}:{dt * 1000:.2f}ms" for w, dt in _sweep_lazy_results.items()
            )
        )

    try:
        n = 0
        # rank1 drafts ONCE up front (cold start -- no previous cycle's
        # verify to draft from yet, matches pp_chained_decode_loop's
        # cold-start precedent: the very first cycle's "draft" is
        # whatever DSpark produces from the PP-priming call's anchor
        # token, conditioned on whatever ctx state exists -- likely
        # empty/uninitialized on the true first cycle, same as the TP
        # path's own first-cycle behavior).
        if is_last_rank:
            _toks, _corrected, _conf = _dspark.draft(
                y.reshape(1),
                embed_tokens,
                model.lm_head,
                _dsc,
                temperature=0.0,
                sample_fn=_dspark_sample_greedy,
                width=_draft_width,
            )
            mx.eval(_toks)
            _draft_ids = [int(v) for v in _toks[0].tolist()]
            # Trim by the ACTUAL width drafted this call: bs (full block)
            # normally, or vw-1 when EXO_PP_DSPARK_DRAFT_WIDTH_TRUNCATE=1
            # is opted in (currently default-off pending a correctness
            # bug fix -- see _draft_width_truncate comment above).
            _trim_this_draft = (vw - 1) if _draft_width_truncate else bs
            for _c in _dsc:
                _c.trim(_trim_this_draft)
            _drafted_total += min(len(_draft_ids), vw - 1)

        while n < max_tokens:
            _cycle_t0 = time.perf_counter()
            _t_after_draft = _cycle_t0  # default for rank0 (never set on that branch)
            _t_after_verify = _cycle_t0  # default for rank0 (outlier-log safety)
            _recv_wait_this_cycle = 0.0  # default for rank0 (outlier-log safety)
            # ==== STEP 3b: consume-cycle activation (top of cycle) ====
            # If the previous cycle was a HIT and _draft_ahead_yield is
            # on, activate consume mode for THIS cycle: skip the normal
            # msg1/msg1b/rank0-fwd/spec-fwd phases; rank 0 sends the
            # already-buffered hidden, rank 1 verifies the extension
            # batch rank 0 already forwarded last cycle. Always clear
            # the carry state after reading it -- consume mode never
            # chains, one HIT yields at most one consume cycle, then
            # the very next iteration is a normal cycle with a fresh
            # spec-forward attempt of its own (design-doc-compatible;
            # explicitly NOT multi-cycle lookahead, which the task
            # guidance calls out as out-of-scope).
            _consume_this_cycle = _consume_active_next
            _consume_ext_ids_this_cycle: list[int] | None = _consume_ext_ids_next
            _consume_spec_id_this_cycle: SpecId | None = _consume_spec_id_next
            _consume_active_next = False
            _consume_ext_ids_next = None
            _consume_spec_id_next = None
            if _consume_this_cycle:
                # Belt-and-braces: on rank 0, the buffered hidden the
                # previous HIT cycle deliberately DID NOT discard must
                # still be present under its SpecId. Fail loudly if
                # it's gone -- that would mean the carry state and the
                # buffer disagree, which is the exact desync class the
                # SpecId matching mechanism exists to detect.
                if (
                    is_rank0
                    and _spec_hidden_buffer is not None
                    and _consume_spec_id_this_cycle is not None
                ):
                    _pending_keys = _spec_hidden_buffer.peek_keys()
                    _wanted_key = (
                        _consume_spec_id_this_cycle.cycle_n,
                        _consume_spec_id_this_cycle.prefix_hash,
                        _consume_spec_id_this_cycle.prefix_len,
                    )
                    if _wanted_key not in _pending_keys:
                        raise RuntimeError(
                            f"STEP 3b consume: buffered hidden missing on rank 0 "
                            f"for spec_id={_consume_spec_id_this_cycle!r}; "
                            f"pending={_pending_keys}"
                        )
                _consume_cycles += 1
                if _TRACE:
                    _log(
                        f"n={n} STEP3b consume-cycle ENTER "
                        f"ext_len={len(_consume_ext_ids_this_cycle) if _consume_ext_ids_this_cycle else 0}"
                    )
            # ==== rank1 -> rank0: fixed-shape (bs+1)-token batch ====
            # world_size is always 2 for this module (is_rank0/is_last_rank
            # exhaustive, no middle-rank case -- see module docstring),
            # so _wire_batch/_trim_amount are always set on both branches
            # below despite the type-checker's inability to prove that
            # statically across an if/elif with no else.
            # STEP 3b: on a consume cycle both sides skip the msg1 wire
            # exchange (see explicit skip inside each branch below) --
            # rank 1 has no new drafts to send (its extension batch is
            # the one rank 0 already forwarded speculatively last cycle)
            # and rank 0 has no new batch to receive. _wire_batch stays
            # at zeros; downstream spec-tag / spec-fwd paths that would
            # consume it are ALSO guarded on ``not _consume_this_cycle``,
            # so the zero value is never read as if it were a real
            # batch.
            _wire_batch: mx.array = mx.zeros(vw, dtype=mx.int32)
            if is_last_rank and not _consume_this_cycle:
                # Truncate to the verify width -- only the first vw-1 of
                # DSpark's bs drafted positions get sent/forwarded/
                # verified this cycle (see verify-width comment above).
                _batch_ids = [int(y.item())] + _draft_ids[: vw - 1]
                _wire_batch = mx.array(_batch_ids, dtype=mx.int32)
                mx.eval(_wire_batch)
                _log(f"n={n} dspark_batch_send PRE (rank1->0)")
                _sent = mx.distributed.send(_wire_batch, 0, group=pp_group)
                mx.eval(_sent)
                _log(f"n={n} dspark_batch_send POST")
            elif is_rank0 and not _consume_this_cycle:
                _log(f"n={n} dspark_batch_recv PRE (rank0<-{pp_world_size - 1})")
                _wire_batch = mx.distributed.recv_like(
                    mx.zeros(vw, dtype=mx.int32),
                    pp_world_size - 1,
                    group=pp_group,
                )
                # CRITICAL (2026-07-18, found via intermittent-deadlock
                # investigation): mx.distributed ops are LAZY graph nodes
                # with no tags/sequence numbers -- point-to-point messages
                # on a group are matched purely by the ORDER both ranks
                # actually EXECUTE them, not the order they're issued in
                # Python. Without forcing this recv to materialize here,
                # MLX's scheduler is free to defer it relative to the
                # model(...) call below -- which ALSO does an internal
                # send/recv on this SAME pp_group (SpecPipelineLastLayer's
                # automatic hidden-state send). If rank0's recv here and
                # rank1's corresponding operations end up interleaved
                # differently than rank1 expects, both ranks can end up
                # blocked waiting on messages the other believes it
                # already sent/received -- a genuine, intermittent (not
                # code-path-dependent, so not always-reproducing) circular
                # wait. Confirmed empirically: ~1-in-3 runs of the
                # IDENTICAL request stalled for 70+s with zero progress
                # then cascaded into a peer-EOF crash; other runs of the
                # same code completed cleanly. Forcing eval here pins this
                # recv to complete before the model(...) call's own
                # internal traffic on the same group can begin.
                mx.eval(_wire_batch)
                _log(f"n={n} dspark_batch_recv POST")

            # ==== STEP 2b msg1b: deep-draft extension (diagnostic, bytes-only) ====
            # Sends the `vw-1` draft positions that DSpark's draft() already
            # computed internally but that msg1 does NOT carry today (msg1 is
            # truncated to `vw` = anchor + first vw-1 drafted). These are the
            # tokens rank0 will (in STEP 3, not this commit) use to
            # speculatively pre-forward the NEXT cycle's block during its
            # otherwise-idle window. For now the extension ONLY rides the
            # wire; rank0 discards it after a round-trip equality check.
            # Same eval-pinning discipline as msg1 above (mx.distributed ops
            # are lazy, must be materialized before the model(...) call's own
            # internal pp_group traffic).
            # STEP 3b: also snapshot the extension token list on rank 1
            # HERE (while _draft_ids still reflects this cycle's draft)
            # so the post-msg2 HIT arming can use it -- by the time
            # cleanup runs below, rank 1's redraft has already
            # overwritten _draft_ids. Only relevant on YIELD, but cheap
            # to always capture on the diagnostic path.
            _ext_ids_this_cycle_r1: list[int] = []
            if _da_msg1_ext_active and not _consume_this_cycle and is_last_rank:
                _ext_ids_this_cycle_r1 = list(_draft_ids[vw - 1 : (2 * vw) - 2])
            _da_msg1_ext_wire: mx.array = mx.zeros(
                max(_da_msg1_ext_len, 1), dtype=mx.int32
            )
            if _da_msg1_ext_active and not _consume_this_cycle:
                if is_last_rank:
                    # DSpark's draft() computed `bs` positions this cycle
                    # (guarded by `not _draft_width_truncate` above); the
                    # first vw-1 rode msg1, positions [vw-1 : 2*vw-2] are
                    # the extension. pack_deep_draft_ext handles pad/reject.
                    _ext_ids = _draft_ids[vw - 1 : (2 * vw) - 2]
                    _da_msg1_ext_wire = pack_deep_draft_ext(_ext_ids, int(vw))
                    mx.eval(_da_msg1_ext_wire)
                    _ext_sent = mx.distributed.send(
                        _da_msg1_ext_wire, 0, group=pp_group
                    )
                    mx.eval(_ext_sent)
                elif is_rank0:
                    _da_msg1_ext_wire = mx.distributed.recv_like(
                        mx.zeros(_da_msg1_ext_len, dtype=mx.int32),
                        pp_world_size - 1,
                        group=pp_group,
                    )
                    mx.eval(_da_msg1_ext_wire)
                    # Correctness of the round-trip is our only assertion
                    # here: rank0 has NO way to independently reconstruct
                    # what rank1 drafted for the extension positions (they
                    # come from a rank1-only DSpark forward), so the check
                    # is a self-consistency shape/length one -- confirm
                    # exactly _da_msg1_ext_len int32s were received and
                    # they materialized without exception. Anything more
                    # requires the rank0-side draft(), which is ruled out
                    # (see design doc's "Why rank0 can't just draft()"
                    # section).
                    try:
                        _ = unpack_deep_draft_ext(_da_msg1_ext_wire, int(vw))
                    except ValueError as _ext_err:
                        _da_msg1_ext_mismatches += 1
                        logger.warning(
                            f"[PP DSpark STEP2b DIAG] msg1b ext wire "
                            f"invalid at n={n}: {_ext_err}"
                        )
                    else:
                        _da_msg1_ext_matches += 1
                    if (_da_msg1_ext_matches + _da_msg1_ext_mismatches) % 32 == 0:
                        logger.info(
                            f"[PP DSpark STEP2b DIAG] msg1b ext round-trip "
                            f"matches={_da_msg1_ext_matches} "
                            f"mismatches={_da_msg1_ext_mismatches}"
                        )

            _t_after_batch_xchg = time.perf_counter()
            _prof_batch_xchg += _t_after_batch_xchg - _cycle_t0

            # ==== rank0: forward the batch through its own layers ====
            # STEP 3b consume cycle: skip the fresh rank 0 forward
            # entirely. Instead, rank 0 sends the ALREADY-BUFFERED
            # speculative hidden from last cycle straight to rank 1
            # over the same pp_group boundary the normal
            # SpecPipelineLastLayer.pp_send path uses -- rank 1's
            # SpecPipelineFirstLayer recv (configured below via
            # pp_recv=True) sees an ordinary bf16 hidden with no
            # visible difference from a fresh forward's output.
            if is_rank0 and not _consume_this_cycle:
                _configure_layers(
                    spec_first, spec_last, pp_send=True, state_list=None, hidden_idx=-1
                )
                _batch_tok = _wire_batch.reshape(1, -1)
                with mx.stream(generation_stream):
                    model(_batch_tok, cache=prompt_cache)
            elif is_rank0 and _consume_this_cycle:
                # Pull the buffered hidden back out (release, not
                # discard -- this is the whole point of the yield
                # path). The buffer entry was stashed last cycle with
                # eval_sentinel set so its KV writes are already
                # materialised; we just need to forward the tensor.
                if _spec_hidden_buffer is None or _consume_spec_id_this_cycle is None:
                    raise RuntimeError(
                        "STEP 3b consume: rank 0 entered consume mode with "
                        "no buffer / no spec_id -- invariant violated"
                    )
                _buffered = _spec_hidden_buffer.release(_consume_spec_id_this_cycle)
                _hidden_to_send = _buffered.hidden
                # JACCL/RDMA transport is bf16; mirror
                # SpecPipelineLastLayer._pp_send exactly for wire
                # compatibility with rank 1's existing recv path.
                if _hidden_to_send.dtype != mx.bfloat16:
                    _hidden_to_send = _hidden_to_send.astype(mx.bfloat16)
                mx.eval(_hidden_to_send)
                _log(
                    f"n={n} STEP3b consume: rank0 hidden_send PRE "
                    f"(shape={_hidden_to_send.shape})"
                )
                _consume_sent = mx.distributed.send(
                    _hidden_to_send,
                    (pp_rank + 1) % pp_world_size,
                    group=pp_group,
                )
                mx.eval(_consume_sent)
                _log(f"n={n} STEP3b consume: rank0 hidden_send POST")
            _t_after_r0_fwd = time.perf_counter()
            _prof_r0_fwd += _t_after_r0_fwd - _t_after_batch_xchg

            # ==== STEP 3 rank0: speculative forward on the msg1b extension
            # tokens during rank0's ~50ms idle window (2026-07-19,
            # EXO_PP_DSPARK_DRAFT_AHEAD_EXECUTE=1, default OFF). This is
            # the "safe increment" first-cut per the design doc's failure-
            # mode discipline: rank0 actually runs the forward, buffers
            # the hidden, tags it with a SpecId, and captures a KV
            # snapshot -- but the HIT-path *consumer* (rank1 skipping its
            # own next-cycle fwd and rank0 sending the buffered hidden
            # instead) is deliberately NOT wired yet. Instead, on EVERY
            # cycle end (regardless of hit/miss), rank0 restores the KV
            # snapshot and discards the buffered hidden -- so rank0's
            # observable KV state after each cycle is bit-identical to
            # the non-STEP-3 path. That deliberately foregoes the
            # ~50ms/cycle wall-time win in exchange for isolating the
            # correctness-critical primitive (spec fwd + snapshot/restore
            # + tagging) from the wall-time-critical HIT-path state
            # machine, so a live cluster run can validate this primitive
            # (does spec-fwd corrupt KV / does snapshot/restore reproduce
            # today's state) independently of any yield-path change.
            #
            # Rank 1 still computes and sends the REAL hit_miss code in
            # msg2 (below), which lets us log the actual hit rate under
            # spec-fwd-enabled execution -- the empirical answer to
            # failure mode #2 (stale conditioning on the extension block:
            # HIT-cycle acceptance may be measurably lower than the
            # would_hit estimate the pure-diagnostic mode reported).
            _spec_fwd_ran_this_cycle = False
            _spec_snapshot: list[Any] | None = None
            _spec_id_this_cycle: SpecId | None = None
            if _draft_ahead_execute and is_rank0 and not _consume_this_cycle:
                # msg1b's extension tokens are what rank1 would verify
                # NEXT cycle if this cycle fully accepts. Rank0 has just
                # them from _da_msg1_ext_wire above -- pull, sanity-check
                # length, forward through its own layers with the
                # SpecPipelineLastLayer in speculative mode (no auto-send
                # to rank1). Any pack/unpack failure was already logged
                # above; here we re-parse locally rather than plumb the
                # list through, keeping the diagnostic and executive
                # paths textually independent.
                try:
                    _ext_ids_r0 = unpack_deep_draft_ext(_da_msg1_ext_wire, int(vw))
                except ValueError as _spec_ext_err:
                    logger.warning(
                        f"[PP DSpark STEP3] msg1b ext unpack failed on rank0 "
                        f"at n={n}: {_spec_ext_err} -- skipping speculative fwd"
                    )
                    _ext_ids_r0 = []
                if len(_ext_ids_r0) == _da_msg1_ext_len and _da_msg1_ext_len > 0:
                    # Failure mode #3 defence: snapshot BEFORE the
                    # speculative forward runs so the restore path can
                    # deterministically reproduce today's KV state on
                    # both HIT and MISS (in this "safe increment" build,
                    # we restore in both cases -- see block-level comment
                    # above). Snapshot must materialize before the
                    # forward's own KV writes get scheduled or the
                    # KVCache offsets we captured could theoretically be
                    # torn by MLX's lazy graph (defensive; today's
                    # snapshot code reads plain-Python offset ints so
                    # this is a belt on top of braces).
                    _spec_snapshot = _snapshot_cache(prompt_cache)
                    # Build the assumed-prefix SpecId for this
                    # speculative branch: anchor = _wire_batch[0], then
                    # the vw-1 drafts already verified this cycle, then
                    # the assumed bonus token (= extension token 0).
                    # This mirrors the STEP 1 diagnostic tag exchange
                    # exactly, so the same failure-mode-#1 protection
                    # (explicit SpecId matching, no implicit ordering
                    # assumptions) also covers the buffered hidden.
                    _wire_batch_raw_r0 = _wire_batch.tolist()
                    if not isinstance(_wire_batch_raw_r0, list):
                        raise RuntimeError(
                            f"_wire_batch.tolist() must be a list, "
                            f"got {type(_wire_batch_raw_r0)!r}"
                        )
                    _wb_ints: list[int] = []
                    for _wv in _wire_batch_raw_r0:
                        if isinstance(_wv, list):
                            raise RuntimeError(
                                f"_wire_batch has nested list element: {_wv!r}"
                            )
                        _wb_ints.append(int(_wv))
                    _anchor_r0 = _wb_ints[0]
                    _drafts_r0 = tuple(_wb_ints[1:vw])
                    _assumed_bonus_r0 = int(_ext_ids_r0[0])
                    _spec_id_this_cycle = SpecId.build(
                        spec_kind="draft_ahead",
                        cycle_n=n,
                        prefix=(_anchor_r0, *_drafts_r0, _assumed_bonus_r0),
                    )
                    # Extension batch shape: rank1's would-be next-cycle
                    # verify batch is (assumed_bonus, ext[1:vw-1]) --
                    # length vw-1. When vw >= 2 and _da_msg1_ext_len ==
                    # vw-1 this is a well-formed batch. (For vw == 2 the
                    # batch degenerates to length 1, which is a valid
                    # single-token forward.)
                    _spec_batch_ids = [_assumed_bonus_r0] + list(_ext_ids_r0[1:])
                    _spec_batch_tok = mx.array([_spec_batch_ids], dtype=mx.int32)
                    # Configure spec_last in *speculative* mode: computes
                    # KV writes into prompt_cache but does NOT send the
                    # hidden across the pp boundary automatically. That
                    # would otherwise be the exact "unsolicited spec
                    # hidden arriving untagged" desync vector called out
                    # in the design doc's wire-protocol section.
                    _spec_state_list: list[mx.array] = [mx.zeros(1)]
                    _configure_layers(
                        spec_first,
                        spec_last,
                        speculative=True,
                        state_list=_spec_state_list,
                        hidden_idx=0,
                    )
                    with mx.stream(generation_stream):
                        model(_spec_batch_tok, cache=prompt_cache)
                    # Failure mode #3: force materialisation NOW so the
                    # forward's KV writes and the buffered hidden are
                    # committed on the timeline before we enter the
                    # msg2 recv branch. Without this, MLX is free to
                    # defer both to whichever downstream op forces
                    # eval, which could interleave with msg2 traffic on
                    # this same pp_group -- same class of intermittent
                    # deadlock the existing mx.eval(_wire_batch) above
                    # was added to prevent.
                    _spec_hidden = _spec_state_list[0]
                    mx.eval(_spec_hidden)
                    if _spec_hidden_buffer is not None:
                        _spec_hidden_buffer.stash(
                            _spec_id_this_cycle,
                            _spec_hidden,
                            eval_sentinel=_spec_hidden,
                        )
                    _spec_fwd_ran_this_cycle = True
                    # Restore the layer config to today's default (no
                    # pp_send / pp_decode / speculative flags set) so
                    # any code path that touches the layers between
                    # here and the next cycle sees the unspeculative
                    # state.
                    _configure_layers(spec_first, spec_last)
            _t_after_spec_fwd = time.perf_counter()

            # ==== rank1: recv hidden batch (existing pipeline boundary,
            # unmodified -- shape-agnostic) + continue its own forward ====
            accepted_ids: list[int] = []
            bonus_token: int | None = None
            n_accepted = 0
            _assumed_bonus_this_cycle: int | None = None
            _consume_drafts: list[int] = []
            if is_last_rank:
                _configure_layers(
                    spec_first,
                    spec_last,
                    pp_recv=True,
                    pp_decode=True,
                    state_list=None,
                    hidden_idx=-1,
                )
                _recv_wait_before = _DSPARK_RECV_WAIT_ACCUM[0]
                with mx.stream(generation_stream):
                    # STEP 3b consume cycle: verify batch is the (vw-1)
                    # extension tokens saved from last HIT cycle's msg1b,
                    # NOT the current-cycle msg1 payload. Batch layout:
                    #   [assumed_bonus, ext[1], ..., ext[vw-2]]
                    # Position 0 (assumed_bonus) predicts what should
                    # appear at position 1 (i.e. verifies ext[1]); the
                    # last position's logit becomes the new bonus. So
                    # this cycle can accept at most vw-2 draft tokens
                    # plus 1 bonus = vw-1 tokens (design doc / Fable
                    # consult, 2026-07-19). `y` is already assumed_bonus
                    # from last cycle's yield loop; consistent with the
                    # normal path's use of y as the batch anchor.
                    if _consume_this_cycle:
                        if _consume_ext_ids_this_cycle is None:
                            raise RuntimeError(
                                "STEP 3b consume: rank 1 in consume mode "
                                "with no _consume_ext_ids_this_cycle"
                            )
                        # ext[0] is assumed_bonus == int(y.item()); use
                        # it explicitly rather than trusting y so a
                        # mismatch fails loudly.
                        _consume_anchor = int(_consume_ext_ids_this_cycle[0])
                        _consume_drafts = _consume_ext_ids_this_cycle[1:]
                        _verify_input = mx.array([[_consume_anchor] + _consume_drafts])
                        _consume_verify_positions = len(_consume_drafts)
                    else:
                        _verify_input = mx.array(
                            [[int(y.item())] + _draft_ids[: vw - 1]]
                        )
                        _consume_verify_positions = vw - 1
                    out = model(_verify_input, cache=prompt_cache)
                    all_next = mx.argmax(out[0], axis=-1)
                    mx.eval(all_next)
                    _all_next_list = [int(v) for v in all_next.tolist()]
                    # Tree-verification feasibility diagnostic (2026-07-18):
                    # cheap to check since DSpark's draft() already computes
                    # per-position logits (`_corrected`, previously discarded
                    # after only using its argmax via `_toks`) -- no extra
                    # forward pass needed. At each REJECTED position, check
                    # whether the draft head's RANK-2 candidate (its own
                    # second-highest-logit token) would have matched the
                    # real model's actual pick. If this hit rate is high,
                    # a 2-candidate tree verify (same flat verify cost,
                    # confirmed via profiling) could recover many of these
                    # rejections "for free." If low, tree verification isn't
                    # worth building -- the draft head's own ranking doesn't
                    # correlate with what actually gets picked when it's
                    # wrong, so widening the tree wouldn't help.
                    if _corrected is not None and not _consume_this_cycle:
                        _top2 = mx.argsort(_corrected[0], axis=-1)[:, -2]
                        mx.eval(_top2)
                        _top2_list = [int(v) for v in _top2.tolist()]
                    else:
                        # Consume cycle: `_corrected` was produced by the
                        # PREVIOUS cycle's draft (conditioned on
                        # bonus_token_{N-1}) and does NOT correspond to
                        # the extension positions we're verifying now.
                        # Skip the tree-verify diagnostic rather than
                        # index into unrelated logits.
                        _top2_list = []
                    # STEP 3b: on a consume cycle the "drafts" being
                    # verified are the (vw-2) extension tokens saved
                    # from last cycle's msg1b (ext[1..vw-2]), not
                    # `_draft_ids`. Same accept/reject shape; different
                    # source array. `_pos_reached/_pos_accepted` share
                    # the histogram since positions have the same
                    # semantic meaning ("position i in the verify batch
                    # after the anchor").
                    if _consume_this_cycle:
                        _verify_drafts = _consume_drafts
                    else:
                        _verify_drafts = _draft_ids[: vw - 1]
                    for i in range(len(_verify_drafts)):
                        _pos_reached[i] += 1
                        if _all_next_list[i] == _verify_drafts[i]:
                            accepted_ids.append(_verify_drafts[i])
                            _pos_accepted[i] += 1
                            n_accepted += 1
                        else:
                            _pos_rejected_reached[i] += 1
                            if _top2_list and _top2_list[i] == _all_next_list[i]:
                                _pos_rank2_would_hit[i] += 1
                            break
                    bonus_token = _all_next_list[n_accepted]
                    _accepted_total += n_accepted
                    # Snapshot the pre-redraft view of the assumed-bonus
                    # position (extension token 0 == _draft_ids[vw-1]) so
                    # the msg2 hit/miss decision below reads the SAME
                    # value rank 0 used when building its speculative
                    # forward -- if we read it after the next-cycle
                    # redraft further down, _draft_ids has been
                    # overwritten and the hit code silently lies about
                    # what rank 0 actually did.
                    # STEP 3b: on a consume cycle no NEW spec-fwd ran
                    # this cycle (we're consuming last cycle's spec-fwd
                    # instead), so there IS no assumed_bonus for THIS
                    # cycle's msg2 hit codec -- leaving this at None
                    # keeps the msg2 hit_miss code at HIT_MISS_NA, which
                    # is the correct signal on the wire ("no speculative
                    # forward this cycle to hit or miss against"). This
                    # is completely independent of last cycle's hit
                    # code, which already drove the consume-mode entry.
                    _assumed_bonus_this_cycle = (
                        None
                        if _consume_this_cycle
                        else (
                            int(_draft_ids[vw - 1])
                            if len(_draft_ids) > vw - 1
                            else None
                        )
                    )

                    # DRAFT-AHEAD HIT-RATE INSTRUMENTATION (2026-07-19, zero
                    # wire/KV cost -- gated by EXO_PP_DSPARK_DRAFT_AHEAD_LOG).
                    # Scoping the "optimistic overlap" architecture (rank0
                    # speculatively forwards the NEXT block during its
                    # otherwise-idle ~61.6ms/cycle window, assuming full
                    # acceptance) per a Fable consult, 2026-07-19: DSpark's
                    # draft() ALWAYS computes the full block_size internally
                    # regardless of verify_width truncation (fixed ~10.6ms
                    # cost either way -- see the verify-width comment above),
                    # so _draft_ids[vw-1] -- the token that WOULD become the
                    # next cycle's anchor if this cycle fully accepts -- is
                    # already sitting here, unused, on every single cycle.
                    # Checking whether it matches the REAL bonus_token this
                    # cycle actually produced is the exact "would a
                    # speculative one-block-ahead forward have hit" question,
                    # with NO new forward pass, NO wire changes, NO KV
                    # mutation -- pure post-hoc comparison of values already
                    # computed. This is deliberately a measurement-only step
                    # BEFORE committing to building the real draft-ahead
                    # protocol (extra wire message, spec_id-tagged hidden
                    # buffering on rank0, hit/miss bit in msg2, rollback path)
                    # -- Fable's own numbers: ~14% E2E speedup at a 30% hit
                    # rate, ~22% at 50%; below ~15% the protocol complexity
                    # isn't worth it. Guarded by len(_draft_ids) > vw-1 since
                    # EXO_PP_DSPARK_DRAFT_WIDTH_TRUNCATE=1 (opt-in, default
                    # off) would make draft() produce only vw-1 tokens,
                    # putting index vw-1 out of range.
                    if (
                        os.environ.get("EXO_PP_DSPARK_DRAFT_AHEAD_LOG", "0") == "1"
                        and not _consume_this_cycle
                    ):
                        _full_accept = n_accepted == vw - 1
                        _da_cycles += 1
                        if _full_accept and len(_draft_ids) > vw - 1:
                            _da_full_accept += 1
                            if _draft_ids[vw - 1] == bonus_token:
                                _da_would_hit += 1
                        if _da_cycles % 20 == 0:
                            _of_fa_pct = (
                                f"{_da_would_hit / _da_full_accept * 100:.1f}%"
                                if _da_full_accept > 0
                                else "n/a"
                            )
                            logger.info(
                                f"[DRAFT_AHEAD_LOG] cycles={_da_cycles} "
                                f"full_accept={_da_full_accept} "
                                f"({_da_full_accept / _da_cycles * 100:.1f}%) "
                                f"would_hit={_da_would_hit} "
                                f"(of_full_accept={_of_fa_pct})"
                            )

                    _t_after_verify = time.perf_counter()
                    _recv_wait_this_cycle = (
                        _DSPARK_RECV_WAIT_ACCUM[0] - _recv_wait_before
                    )
                    _prof_r1_verify_wait += _recv_wait_this_cycle
                    _prof_r1_verify_fwd += (
                        _t_after_verify - _t_after_r0_fwd - _recv_wait_this_cycle
                    )

                    # Commit DSpark ctx for the accepted prefix (matches TP
                    # pattern exactly: get_dspark_ctx pulls the hc-mean
                    # hiddens captured as a side effect of THIS verify
                    # forward, at the tap layers -- now correctly keyed by
                    # GLOBAL layer index per the pipeline_start_idx fix).
                    _ctx_cat = get_dspark_ctx(_dspark.target_layer_ids)
                    if _ctx_cat is not None:
                        _dspark.append_ctx(_ctx_cat[:, : n_accepted + 1], _dsc)

                    # Draft the NEXT cycle now (self-contained, rank1-only)
                    # using the bonus token as the new anchor.
                    _next_toks, _next_corrected, _next_conf = _dspark.draft(
                        mx.array([bonus_token]),
                        embed_tokens,
                        model.lm_head,
                        _dsc,
                        temperature=0.0,
                        sample_fn=_dspark_sample_greedy,
                        width=_draft_width,
                    )
                    mx.eval(_next_toks)
                    _draft_ids = [int(v) for v in _next_toks[0].tolist()]
                    _corrected = _next_corrected
                    _t_after_draft = time.perf_counter()
                    _prof_draft += _t_after_draft - _t_after_verify
                    # Trim by whatever width was actually drafted this
                    # call -- vw-1 when opted in, bs (full block)
                    # otherwise. See _draft_width_truncate comment above.
                    _trim_this_draft = (vw - 1) if _draft_width_truncate else bs
                    for _c in _dsc:
                        _c.trim(_trim_this_draft)
                    # DIAGNOSTIC (2026-07-18): investigating a genuine
                    # silent-degeneration bug found after this width
                    # change (BOS-spam output, no crash, high reported
                    # accept rate -- exactly the "corrupted shared cache,
                    # verifier sees the same broken context so happily
                    # accepts garbage" signature per a second-opinion
                    # review). RotatingKVCache.trim()/is_trimmable() are
                    # documented as only valid PRE-WRAP (offset <
                    # max_size); this logs each cache's offset relative
                    # to its max_size every cycle to determine whether
                    # wraparound is actually being hit despite the
                    # request being short (30 max_tokens).
                    if _TRACE:
                        for _ci, _c in enumerate(_dsc):
                            _off = getattr(_c, "offset", None)
                            _mx = getattr(_c, "max_size", None)
                            if _off is not None and _mx is not None and _off >= _mx:
                                logger.warning(
                                    f"[PP DSpark CACHE WRAP] stage={_ci} "
                                    f"offset={_off} max_size={_mx} -- "
                                    f"RotatingKVCache.trim() is invalid "
                                    f"post-wrap, corruption likely"
                                )
                    # BUG FIX (2026-07-18): this used to count DSpark's
                    # full internal block_size (5) as "drafted" every
                    # cycle, even though verify_width truncation (see
                    # above) means only vw-1 of those positions are EVER
                    # eligible to be checked/accepted. That inflated the
                    # denominator of the accept-rate summary ~2.5x,
                    # making a genuinely healthy 59-69% acceptance
                    # (confirmed via the per-position histogram, which
                    # doesn't have this bug) look like a concerning 23%.
                    _drafted_total += min(len(_draft_ids), vw - 1)

            # ==== rank1 -> rank0: (n_accepted, accepted_ids padded to bs,
            # bonus_token) -- FIXED SHAPE regardless of n_accepted's value.
            # BUG FIX (2026-07-18): the original version sent only
            # trim_amount (a derived count) + bonus_token. rank0 has no
            # other source for the actual accepted draft token IDs -- its
            # own `accepted_ids` list stays permanently empty (only ever
            # appended to inside the `if is_last_rank:` block above), so
            # its yield loop below silently emitted ZERO of the accepted
            # tokens every cycle, only the single bonus token. Since
            # send_chunk() in runner.py ONLY emits ChunkGenerated on
            # rank0 (device_rank==0; existing PP dedup mechanism, not
            # specific to this code), the HTTP client received roughly
            # 1/6th of the real generated tokens (bonus tokens only) --
            # producing what looked like an empty/stalled response even
            # though the server-side decode loop completed perfectly
            # cleanly (confirmed via logs: correct accept rates, no
            # errors). This ALSO desynced the two ranks' own `n` counters
            # (rank0's incremented ~6x slower than rank1's), which could
            # independently cause the `while n < max_tokens:` loop to
            # terminate at different points on each rank -- a second,
            # latent bug this same fix resolves.
            _accepted_ids: list[int] = []
            _n_accepted_wire = 0
            _bonus_wire = 0
            # msg2 layout (STEP 1 of draft-ahead, 2026-07-19): existing
            # [n_accepted, bonus_token, padded accepted_ids...] extended by
            # ONE int32 trailing slot for the hit/miss/na code. Adding the
            # slot now (default HIT_MISS_NA) forces both ranks to speak the
            # same wire protocol at commit time, so STEP 3 (actual
            # speculative branch) can be enabled with a pure logic change
            # and no further wire-shape churn. See
            # pp_speculation_spec_tag.HitMissCode for the encoding.
            _msg2_len = int(2 + (vw - 1) + 1)
            _hit_miss_code = int(HIT_MISS_NA)
            # When EXO_PP_DSPARK_DRAFT_AHEAD_EXECUTE=1, rank 1 computes and
            # sends the REAL hit/miss verdict for rank 0's speculative
            # forward: HIT iff full acceptance (n_accepted == vw-1) AND
            # the assumed bonus token (extension token 0 == _draft_ids[vw-1])
            # matched what verify actually produced. In the safe-increment
            # first cut the code is diagnostic (rank 0 restores the KV
            # snapshot regardless), but shipping the REAL codec now lets
            # live-cluster runs measure the hit rate under actual
            # spec-fwd-enabled execution -- the answer to failure mode #2
            # (stale conditioning on the extension block possibly lowering
            # HIT-cycle acceptance below the pure-diagnostic estimate).
            if (
                _draft_ahead_execute
                and is_last_rank
                and bonus_token is not None
                and _assumed_bonus_this_cycle is not None
            ):
                _full_accept_exec = n_accepted == vw - 1
                if _full_accept_exec and _assumed_bonus_this_cycle == bonus_token:
                    _hit_miss_code = int(HIT_MISS_HIT)
                else:
                    _hit_miss_code = int(HIT_MISS_MISS)
            if is_last_rank:
                _accepted_ids = accepted_ids
                _n_accepted_wire = n_accepted
                _bonus_wire = bonus_token if bonus_token is not None else 0
                _padded = _accepted_ids + [-1] * (vw - 1 - len(_accepted_ids))
                _wire2 = mx.array(
                    [_n_accepted_wire, _bonus_wire] + _padded + [_hit_miss_code],
                    dtype=mx.int32,
                )
                mx.eval(_wire2)
                _log(
                    f"n={n} dspark_trim_send PRE (rank1->0) "
                    f"n_accepted={_n_accepted_wire} bonus={_bonus_wire} "
                    f"hit_miss={_hit_miss_code}"
                )
                _sent2 = mx.distributed.send(_wire2, 0, group=pp_group)
                mx.eval(_sent2)
                _log(f"n={n} dspark_trim_send POST")
            elif is_rank0:
                _log(f"n={n} dspark_trim_recv PRE (rank0<-{pp_world_size - 1})")
                _wire2 = mx.distributed.recv_like(
                    mx.zeros(_msg2_len, dtype=mx.int32),
                    pp_world_size - 1,
                    group=pp_group,
                )
                mx.eval(_wire2)
                _log(f"n={n} dspark_trim_recv POST")
                _wire2_list = [int(v) for v in _wire2.tolist()]
                _n_accepted_wire = _wire2_list[0]
                bonus_token = _wire2_list[1]
                _accepted_ids = _wire2_list[2 : 2 + _n_accepted_wire]
                # coerce_hit_miss narrows into HitMissCode; unknown values
                # raise, so any garbage on the wire fails loudly rather
                # than silently corrupting the KV state.
                _hit_miss_code = int(coerce_hit_miss(_wire2_list[-1]))
                if _hit_miss_code != HIT_MISS_NA and not _draft_ahead_execute:
                    # When execute is OFF the diagnostic protocol reserves
                    # this slot to always-NA; any other code is the exact
                    # desync we're guarding against. Log loudly, do NOT
                    # act on it.
                    logger.warning(
                        f"[PP DSpark STEP1 DIAG] unexpected hit_miss code "
                        f"{_hit_miss_code} in diagnostic mode; ignoring "
                        f"(should be {int(HIT_MISS_NA)})"
                    )

            # ==== STEP 3 rank0 post-msg2 cleanup ====
            # STEP 3b (YIELD gate on): on a HIT, KEEP rank 0's spec KV
            # writes (do NOT restore snapshot) and DEFER discard of the
            # buffered hidden -- it will be released next cycle by the
            # consume-mode branch above. On a MISS (or with YIELD off),
            # behaviour is exactly STEP 3a's: restore snapshot, discard
            # buffered hidden.
            #
            # Cross-cycle carry (both ranks): a HIT arms the next cycle
            # as a consume cycle by populating _consume_active_next.
            # Rank 1 additionally saves the extension token list under
            # _consume_ext_ids_next BEFORE its redraft below overwrites
            # _draft_ids -- msg1b's extension positions live at
            # _draft_ids[vw-1 : 2*vw-2] and disappear once _dspark.draft
            # runs for the next cycle. This is the exact defence
            # against failure mode #2 (double-counting / dropping
            # tokens): the extension IDs are captured here, once, from
            # a stable source, and consumed exactly once next cycle.
            _hm_narrowed = coerce_hit_miss(_hit_miss_code)
            _yield_hit_this_cycle = (
                _draft_ahead_yield
                and _hm_narrowed == HIT_MISS_HIT
                and not _consume_this_cycle
            )
            if is_rank0 and _spec_fwd_ran_this_cycle:
                if _yield_hit_this_cycle:
                    # HIT + YIELD on: do NOT restore snapshot, do NOT
                    # discard buffered hidden. Both stay live for the
                    # next cycle's consume phase.
                    _consume_active_next = True
                    _consume_spec_id_next = _spec_id_this_cycle
                    if _TRACE:
                        _log(
                            f"n={n} STEP3b HIT-yield: keeping spec KV writes "
                            f"and buffered hidden for next-cycle consume "
                            f"(spec_id={_spec_id_this_cycle!r})"
                        )
                else:
                    if _spec_snapshot is not None:
                        _restore_cache(prompt_cache, _spec_snapshot)
                    if (
                        _spec_hidden_buffer is not None
                        and _spec_id_this_cycle is not None
                    ):
                        _spec_hidden_buffer.discard(_spec_id_this_cycle)
                    if _TRACE:
                        _log(
                            f"n={n} STEP3 spec-fwd cleanup: hit_miss="
                            f"{_hm_narrowed} restored_snapshot="
                            f"{_spec_snapshot is not None}"
                        )
            # Rank-1 side of the carry: capture extension IDs for next
            # cycle's verify BEFORE _draft_ids gets overwritten by the
            # redraft below. Do this unconditionally on the YIELD path
            # + HIT + last_rank (rank 1); mirrors the rank-0 side.
            if _yield_hit_this_cycle and is_last_rank:
                # The extension tokens rank 1 sent in msg1b THIS cycle
                # were snapshotted into _ext_ids_this_cycle_r1 BEFORE
                # rank 1's next-cycle redraft overwrote _draft_ids. On
                # a HIT these are exactly what rank 0 speculatively
                # forwarded, and what we must verify next cycle.
                _ext_slice = _ext_ids_this_cycle_r1
                # Sanity: on a HIT we must have vw-1 extension tokens.
                # If not, the msg1b path was somehow silently truncated;
                # fail loudly rather than proceed into consume mode
                # with a short batch (which would corrupt token
                # accounting per failure mode #2).
                if len(_ext_slice) != vw - 1:
                    raise RuntimeError(
                        f"STEP 3b HIT-yield: expected {vw - 1} extension "
                        f"tokens, got {len(_ext_slice)} at n={n} -- refusing "
                        f"to enter consume mode with an incomplete batch"
                    )
                _consume_active_next = True
                _consume_ext_ids_next = list(_ext_slice)
                if _TRACE:
                    _log(
                        f"n={n} STEP3b HIT-yield: rank1 armed consume cycle "
                        f"with ext_ids={_consume_ext_ids_next}"
                    )

            _t_after_trim_xchg = time.perf_counter()
            # rank0's baseline for this phase is _t_after_r0_fwd (it has no
            # verify/draft phases); rank1's is _t_after_draft. Both are
            # correct "end of previous phase" markers on their own rank.
            _prof_trim_xchg += _t_after_trim_xchg - (
                _t_after_draft if is_last_rank else _t_after_r0_fwd
            )

            # ==== KV rollback (both ranks, deterministic given bs and
            # n_accepted) ====
            # STEP 3b: on a consume cycle both ranks appended vw-1 KV
            # positions this cycle (rank 0 via last cycle's spec-fwd
            # that was KEPT; rank 1 via this cycle's verify forward),
            # so the trim baseline is vw-1 rather than the normal
            # cycle's vw. Both ranks compute the SAME _trim_amount from
            # the same wire-recovered _n_accepted_wire, so no
            # cross-rank consistency issue.
            _cycle_forward_len = (vw - 1) if _consume_this_cycle else vw
            _trim_amount = (_cycle_forward_len - 1) - _n_accepted_wire
            if _trim_amount > 0:
                for c in prompt_cache:
                    if hasattr(c, "trim"):
                        c.trim(_trim_amount)

            # ==== STEP 1 diagnostic spec-tag exchange (draft-ahead)  ====
            # Gated by EXO_PP_DSPARK_DRAFT_AHEAD=1 (default OFF). Fixed
            # SPEC_TAG_WIRE_LEN int32 slots. Both ranks now know:
            #   anchor        = _wire_batch[0]  (also y at cycle start)
            #   drafted_ids   = _wire_batch[1..vw]
            #   n_accepted    = _n_accepted_wire  (0..vw-1)
            #   accepted_ids  = _accepted_ids (populated on BOTH ranks)
            #   bonus_token   = bonus_token
            # so both can independently construct the SAME SpecId for
            # this cycle. If the packed<->unpacked round-trip disagrees,
            # that's the tag-mechanism-itself failing under real
            # cross-rank timing. Diagnostic-only: never branches decode.
            if (
                _draft_ahead_enabled
                and _spec_tag_validator is not None
                and not _consume_this_cycle
            ):
                _wire_batch_raw = _wire_batch.tolist()
                if not isinstance(_wire_batch_raw, list):
                    raise RuntimeError(
                        f"_wire_batch.tolist() must be a list, got {type(_wire_batch_raw)!r}"
                    )
                _wire_batch_list: list[int] = []
                for _v in _wire_batch_raw:
                    if isinstance(_v, list):
                        raise RuntimeError(
                            f"_wire_batch has nested list element: {_v!r}"
                        )
                    _wire_batch_list.append(int(_v))
                _anchor_this_cycle = _wire_batch_list[0]
                _drafted_this_cycle = tuple(_wire_batch_list[1:vw])
                _bonus_for_tag = bonus_token if bonus_token is not None else 0
                # Assumed-prefix for a hypothetical draft-ahead this cycle:
                # (anchor, drafted..., assumed_bonus). Both ranks have all
                # three components after msg2 completes.
                _assumed_prefix = (
                    _anchor_this_cycle,
                    *_drafted_this_cycle,
                    _bonus_for_tag,
                )
                _local_spec_id = SpecId.build(
                    spec_kind="draft_ahead",
                    cycle_n=n,
                    prefix=_assumed_prefix,
                )
                if is_last_rank:
                    _tag_wire = pack_spec_tag(_local_spec_id)
                    mx.eval(_tag_wire)
                    _tag_sent = mx.distributed.send(_tag_wire, 0, group=pp_group)
                    mx.eval(_tag_sent)
                elif is_rank0:
                    _tag_wire = mx.distributed.recv_like(
                        mx.zeros(SPEC_TAG_WIRE_LEN, dtype=mx.int32),
                        pp_world_size - 1,
                        group=pp_group,
                    )
                    mx.eval(_tag_wire)
                    try:
                        _incoming_spec_id = unpack_spec_tag(_tag_wire)
                    except ValueError as e:
                        logger.warning(
                            f"[PP DSpark STEP1 DIAG] spec-tag unpack failed "
                            f"at n={n}: {e}"
                        )
                    else:
                        _result = _spec_tag_validator.validate(
                            incoming=_incoming_spec_id,
                            expected=_local_spec_id,
                        )
                        if _result.ok:
                            _spec_tag_matches += 1
                        else:
                            _spec_tag_mismatches += 1
                            logger.warning(
                                f"[PP DSpark STEP1 DIAG] spec-tag mismatch "
                                f"at n={n}: {_result.reason}"
                            )
                        if (_spec_tag_matches + _spec_tag_mismatches) % 32 == 0:
                            logger.info(
                                f"[PP DSpark STEP1 DIAG] spec-tag round-trip "
                                f"matches={_spec_tag_matches} "
                                f"mismatches={_spec_tag_mismatches}"
                            )

            _prof_cycle_n += 1
            _this_cycle_dt = time.perf_counter() - _cycle_t0
            _prof_total += _this_cycle_dt

            # ── immediate outlier log (2026-07-18) ──────────────────────
            # User observed the exo dashboard showing one node near-idle
            # while the other was fully busy for what looked like ~10s
            # during a live run. Checked all cycle timing across the full
            # 50K-500K suite afterward and found nothing above ~130ms --
            # but that check only had the PERIODIC average (every 16
            # cycles) to go on, which would silently absorb and hide a
            # single genuinely slow outlier cycle into its mean (e.g. one
            # 10s cycle among 15 normal ~100ms ones averages to well under
            # 1s/cycle, invisible in the periodic log). This closes that
            # blind spot: log immediately, UNCONDITIONALLY (not gated by
            # _TRACE -- a real production stall needs to be visible even
            # when tracing isn't explicitly enabled), the instant any
            # single cycle exceeds a generous threshold, with the full
            # phase breakdown so the SPECIFIC stalled phase (batch
            # exchange, rank0 forward, rank1 verify wait/forward, draft,
            # trim exchange) is immediately identifiable from the log
            # alone next time, rather than needing to reproduce it live.
            if _this_cycle_dt > 1.0:
                _outlier_msg = (
                    f"[PP DSpark OUTLIER R{pp_rank} n={n}] cycle took "
                    f"{_this_cycle_dt * 1000:.1f}ms (>1000ms threshold) -- "
                    f"batch_xchg={(_t_after_batch_xchg - _cycle_t0) * 1000:.1f}ms "
                    f"r0_fwd={(_t_after_r0_fwd - _t_after_batch_xchg) * 1000:.1f}ms"
                )
                if is_last_rank:
                    _outlier_msg += (
                        f" r1_verify_wait={_recv_wait_this_cycle * 1000:.1f}ms "
                        f"r1_verify_fwd={(_t_after_verify - _t_after_r0_fwd - _recv_wait_this_cycle) * 1000:.1f}ms "
                        f"draft={(_t_after_draft - _t_after_verify) * 1000:.1f}ms"
                    )
                _outlier_msg += f" trim_xchg={(_t_after_trim_xchg - (_t_after_draft if is_last_rank else _t_after_r0_fwd)) * 1000:.1f}ms"
                logger.warning(_outlier_msg)

            if _TRACE and _prof_cycle_n % 16 == 0:
                _pn = _prof_cycle_n
                logger.info(
                    f"[PROF dspark R{pp_rank} x{_pn}] "
                    f"batch_xchg={_prof_batch_xchg / _pn * 1000:.2f}ms "
                    f"r0_fwd={_prof_r0_fwd / _pn * 1000:.2f}ms "
                    f"r1_verify_wait={_prof_r1_verify_wait / _pn * 1000:.2f}ms "
                    f"r1_verify_fwd={_prof_r1_verify_fwd / _pn * 1000:.2f}ms "
                    f"draft={_prof_draft / _pn * 1000:.2f}ms "
                    f"trim_xchg={_prof_trim_xchg / _pn * 1000:.2f}ms "
                    f"cycle_total={_prof_total / _pn * 1000:.2f}ms"
                )
                _prof_batch_xchg = 0.0
                _prof_r0_fwd = 0.0
                _prof_r1_verify_wait = 0.0
                _prof_r1_verify_fwd = 0.0
                _prof_draft = 0.0
                _prof_trim_xchg = 0.0
                _prof_total = 0.0
                _prof_cycle_n = 0

            # ==== yield accepted tokens + bonus token ====
            # Uses _accepted_ids (correctly populated on BOTH ranks now via
            # the wire message fix above), NOT the stale `accepted_ids`
            # local (which stayed empty on rank0 -- see bug-fix comment
            # above the wire2 message construction).
            _tokens_yielded_this_cycle = 0
            for tok_id in _accepted_ids:
                yield tok_id, mx.zeros(1)
                n += 1
                _tokens_yielded_this_cycle += 1
                if n >= max_tokens:
                    break
            if n < max_tokens and bonus_token is not None:
                yield bonus_token, mx.zeros(1)
                y = mx.array([bonus_token])
                n += 1
                _tokens_yielded_this_cycle += 1
            elif _accepted_ids:
                y = mx.array([_accepted_ids[-1]])
            if _consume_this_cycle:
                _consume_tokens_yielded += _tokens_yielded_this_cycle
                if (
                    _draft_ahead_yield
                    and _consume_cycles % 20 == 0
                    and _consume_cycles > 0
                ):
                    _avg_tok = _consume_tokens_yielded / _consume_cycles
                    logger.info(
                        f"[PP DSpark STEP3b YIELD] consume_cycles={_consume_cycles} "
                        f"tokens_yielded={_consume_tokens_yielded} "
                        f"avg_tok_per_consume={_avg_tok:.2f} "
                        f"(theoretical max = vw-1 = {vw - 1})"
                    )

    finally:
        _configure_layers(spec_first, spec_last)
        if _drafted_total > 0:
            logger.debug(
                f"PP DSpark: {_accepted_total}/{_drafted_total} "
                f"drafted tokens accepted ({_accepted_total / _drafted_total * 100:.0f}%), "
                f"block_size={bs}"
            )
        if is_last_rank and sum(_pos_reached) > 0:
            _hist = ", ".join(
                f"pos{i}={_pos_accepted[i]}/{_pos_reached[i]} "
                f"({_pos_accepted[i] / _pos_reached[i] * 100:.0f}%)"
                if _pos_reached[i] > 0
                else f"pos{i}=n/a"
                for i in range(bs)
            )
            logger.debug(f"PP DSpark per-position accept histogram: {_hist}")
        if is_last_rank and sum(_pos_rejected_reached) > 0:
            _tree_hist = ", ".join(
                f"pos{i}={_pos_rank2_would_hit[i]}/{_pos_rejected_reached[i]} "
                f"({_pos_rank2_would_hit[i] / _pos_rejected_reached[i] * 100:.0f}%)"
                if _pos_rejected_reached[i] > 0
                else f"pos{i}=n/a"
                for i in range(bs)
            )
            logger.debug(
                f"PP DSpark tree-verify feasibility (rank-2 hit rate at "
                f"rejected positions): {_tree_hist}"
            )


# ---------------------------------------------------------------------------
# Draft model loading
# ---------------------------------------------------------------------------

_DRAFT_KV_WINDOW = int(os.environ.get("EXO_DRAFT_KV_WINDOW", "4096"))


def load_draft_model(model_path: str) -> tuple[nn.Module, list[Any]] | None:
    """Load a small draft model for speculation. Returns (model, cache) or None."""
    try:
        from mlx_lm.utils import load
        from mlx_lm.models.cache import make_prompt_cache

        _log(f"Loading draft model: {model_path}")
        model, _ = load(model_path)
        mx.eval(model.parameters())
        cache = make_prompt_cache(model, max_kv_size=_DRAFT_KV_WINDOW)
        _log(f"Draft model loaded (KV window={_DRAFT_KV_WINDOW})")
        return model, cache
    except Exception as e:
        _log(f"Failed to load draft model: {e}")
        return None

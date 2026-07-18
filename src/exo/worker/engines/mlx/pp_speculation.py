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
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.sample_utils import make_sampler

from .auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
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
_PROF_DSPARK_RECV_WAIT = _TRACE and os.environ.get("EXO_PP_DSPARK_PROFILE_WAIT", "1") == "1"
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

def _snapshot_cache(cache: list[Any]) -> list[Any]:
    """Lightweight snapshot: save offsets for KVCache, shallow-copy for ArraysCache."""
    snap: list[Any] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            snap.append(list(c.cache))
        elif isinstance(c, KVCache):
            snap.append(c.offset)
        else:
            snap.append(None)
    return snap


def _restore_cache(cache: list[Any], snap: list[Any]) -> None:
    """Restore cache from snapshot. KVCache trims keys/values; ArraysCache restores list."""
    for c, s in zip(cache, snap):
        if s is None:
            continue
        if isinstance(c, ArraysCache):
            c.cache = s
        elif isinstance(c, KVCache) and c.offset > s:
            c.keys = c.keys[:, :, :s, :]
            c.values = c.values[:, :, :s, :]
            c.offset = s


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
            _log(f"SpecPipelineFirstLayer.__call__ r={self.r} _pp_recv={self._pp_recv} "
                 f"is_prefill={self.is_prefill}")
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
            _log(f"SpecPipelineLastLayer.__call__ r={self.r} s={self.s} "
                 f"_speculative={self._speculative} _pp_send={self._pp_send} "
                 f"_pp_decode={self._pp_decode} is_prefill={self.is_prefill}")
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
                out_bf16 = output.astype(mx.bfloat16) if output.dtype != mx.bfloat16 else output
                sent = mx.distributed.send(out_bf16, (self.r + 1) % self.s, group=self.group)
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

def _install_spec_layers(model: nn.Module) -> tuple[SpecPipelineFirstLayer | None, SpecPipelineLastLayer | None]:
    """Replace PipelineFirst/LastLayer with speculative versions. Returns refs."""
    layers = model.layers  # type: ignore
    spec_first: SpecPipelineFirstLayer | None = None
    spec_last: SpecPipelineLastLayer | None = None

    for i, layer in enumerate(layers):
        if isinstance(layer, PipelineFirstLayer) and not isinstance(layer, SpecPipelineFirstLayer):
            spec_first = SpecPipelineFirstLayer(layer)
            layers[i] = spec_first
        elif isinstance(layer, PipelineLastLayer) and not isinstance(layer, SpecPipelineLastLayer):
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
        self._captured['pre_norm'] = x
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
    _log(f"spec layers found: spec_first={'r='+str(spec_first.r) if spec_first else None} "
         f"spec_last={'r='+str(spec_last.r)+' s='+str(spec_last.s) if spec_last else None} "
         f"pp_rank={pp_rank} pp_world_size={pp_world_size}")

    # State list for hidden exchange
    _cache_state = [c.state if hasattr(c, 'state') else c for c in prompt_cache]
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
        _configure_layers(spec_first, spec_last,
                          pp_send=True, state_list=_cache_state, hidden_idx=_hidden_idx)
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
        _configure_layers(spec_first, spec_last,
                          pp_recv=True, pp_decode=True,
                          state_list=_cache_state, hidden_idx=_hidden_idx)
        with mx.stream(generation_stream):
            out = model(token[None], cache=prompt_cache)
            out = out[:, -1, :]
            lp = out - mx.logsumexp(out, keepdims=True)
            sampled = sampler(lp)
            return sampled, lp.squeeze(0)

    _log(f"decode loop start: max_tokens={max_tokens}, mtp={'yes' if mtp_predictor else 'no'}")

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
                        _to_send,
                        (pp_rank + 1) % pp_world_size, group=pp_group
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
                    mx.zeros(1, dtype=mx.int32), pp_world_size - 1, group=pp_group,
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
                if is_last_rank and 'pre_norm' in _captured:
                    _pn = _captured['pre_norm'][:, -1:, :].astype(mx.bfloat16)
                    mx.eval(_pn)
                    _log(f"n={n} hidden_send PRE (rank{pp_rank}->0, MTP feedback)")
                    _sent = mx.distributed.send(_pn, 0, group=pp_group)
                    mx.eval(_sent)
                    _log(f"n={n} hidden_send POST")
                elif is_rank0:
                    _log(f"n={n} hidden_recv PRE (rank0<-{pp_world_size - 1}, MTP feedback)")
                    _mtp_hidden = mx.distributed.recv_like(
                        mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16),
                        pp_world_size - 1, group=pp_group,
                    )
                    mx.eval(_mtp_hidden)
                    _log(f"n={n} hidden_recv POST")
                    # Cast back to model's compute dtype after bf16 transport
                    if _mtp_hidden.dtype != mx.float16:
                        from exo.worker.engines.mlx.patches.qwen3_5_moe.common import COMPUTE_DTYPE
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
                        logits = mtp_predictor.predict(_mtp_draft_input, mx.array([[y.item()]]))
                        draft_tok = logits.argmax(axis=-1)
                        mx.eval(draft_tok)
                        _draft_token = int(draft_tok.item())
                        _used_mtp = True
                    elif mtp_predictor is None and draft_model is not None:
                        draft_logits = draft_model(mx.array([[y.item()]]), cache=draft_cache)
                        draft_tok = draft_logits[0, -1].argmax()
                        mx.eval(draft_tok)
                        _draft_token = int(draft_tok.item())

                    if _draft_token is not None:
                        _spec_snap = _snapshot_cache(prompt_cache)
                        _rank0_speculative_fwd(_draft_token)
                        _log(f"n={n} drafted={_draft_token} ({'mtp' if _used_mtp else 'draft'})")
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
                    draft_model(mx.array([[int(final_token.item())]]), cache=draft_cache)
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
                    f"loop={_loop_dt*1000:.1f}ms "
                    f"r0_compute={_dt_r0_compute*1000:.1f}ms "
                    f"r0_draft={_dt_r0_draft*1000:.1f}ms "
                    f"r1_compute={_dt_r1_compute*1000:.1f}ms "
                    f"tok_xchg={_dt_tok_xchg*1000:.1f}ms "
                    f"hidden_xchg={_dt_hidden_xchg*1000:.1f}ms "
                    f"verify={_dt_verify*1000:.1f}ms "
                    f"clear_cache={_dt_clear_cache*1000:.1f}ms "
                    f"draft_accepted={'yes' if is_rank0 and _draft_token is not None else 'no'}"
                )

            # ── periodic profiling log ──
            if _TRACE and n % 64 == 0:
                _n = 64
                logger.info(
                    f"[PROF pp-spec R{pp_rank} x{_n}] "
                    f"r0_compute={_prof_r0_compute/_n*1000:.2f}ms "
                    f"r0_draft={_prof_r0_draft/_n*1000:.2f}ms "
                    f"r1_compute={_prof_r1_compute/_n*1000:.2f}ms "
                    f"tok_xchg={_prof_token_exchange/_n*1000:.2f}ms "
                    f"hidden_xchg={_prof_hidden_exchange/_n*1000:.2f}ms "
                    f"verify={_prof_verify/_n*1000:.2f}ms "
                    f"clear_cache={_prof_clear_cache/_n*1000:.2f}ms "
                    f"loop={_prof_total/_n*1000:.2f}ms"
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
            logger.debug(f"PP speculation: {_accepted}/{total} accepted ({_accepted/total*100:.0f}%)")
        mtp_total = _mtp_accepted + _mtp_rejected
        if mtp_total > 0:
            logger.debug(f"MTP: {_mtp_accepted}/{mtp_total} accepted ({_mtp_accepted/mtp_total*100:.0f}%), "
                         f"draft-fallback: {total - mtp_total} steps")


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
                h, mx.array([[tok]]), return_hidden=True,
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
            _log(f"lm_head nop check: mtp_predictor.lm_head is real_lm_head = "
                 f"{getattr(mtp_predictor, 'lm_head', None) is _real_lm_head}, "
                 f"mtp_predictor.lm_head is nop = "
                 f"{getattr(mtp_predictor, 'lm_head', None) is _nop_lm_head}")

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
                    mtp_predictor, _mtp_seed_hidden, int(y.item()), k - 1,
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
                _configure_layers(spec_first, spec_last,
                                  pp_send=True, state_list=None, hidden_idx=-1)
                _batch_ids = [int(y.item())] + [
                    max(0, d) for d in draft_ids  # sentinel -1 -> 0 for the
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
                _configure_layers(spec_first, spec_last,
                                  pp_recv=True, pp_decode=True,
                                  state_list=None, hidden_idx=-1)
                with mx.stream(generation_stream):
                    _batch_tok_r1 = mx.array([[int(y.item())] + [
                        max(0, d) for d in draft_ids
                    ]])
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
                    _pn = _captured.get('pre_norm')
                    # Seed hidden for rank0's NEXT chain: the pre-norm
                    # hidden at whichever position corresponds to the
                    # last COMMITTED token this iteration (index m, since
                    # accepted_ids has length m and the bonus token sits
                    # logically at position m too).
                    _seed_idx = min(m, k - 1)
                    _next_seed_hidden = (
                        _pn[:, _seed_idx:_seed_idx + 1, :].astype(mx.bfloat16)
                        if _pn is not None else None
                    )
            # ==== FIXED-SHAPE WIRE: rank1 -> rank0, (m, bonus_token) ====
            if is_last_rank:
                _wire = mx.array([m, bonus_token if bonus_token is not None else 0], dtype=mx.int32)
                mx.eval(_wire)
                _log(f"n={n} chain_verify_send PRE (rank1->0) m={m} bonus={bonus_token}")
                _sent_wire = mx.distributed.send(_wire, 0, group=pp_group)
                mx.eval(_sent_wire)
                _log(f"n={n} chain_verify_send POST")
            elif is_rank0:
                _log(f"n={n} chain_verify_recv PRE (rank0<-{pp_world_size - 1})")
                _wire = mx.distributed.recv_like(
                    mx.zeros(2, dtype=mx.int32), pp_world_size - 1, group=pp_group,
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
                    _hs = _next_seed_hidden if _next_seed_hidden is not None else mx.zeros(
                        (1, 1, hidden_size), dtype=mx.bfloat16
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
                        pp_world_size - 1, group=pp_group,
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
                        from exo.worker.engines.mlx.patches.qwen3_5_moe.common import COMPUTE_DTYPE
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
            if is_rank0 and mtp_predictor is not None and mtp_predictor.kv_cache is not None:
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
                f"drafted tokens accepted ({_accepted_total/_drafted_total*100:.0f}%), "
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

    # ── profiling (gated by _TRACE, same pattern as pp_speculative_decode_loop
    # above) -- added 2026-07-18 to get real per-cycle timing before deciding
    # how to restructure the wire protocol (collapse messages / overlap
    # drafting). Do NOT guess bottleneck location from log timestamps alone.
    _prof_cycle_n = 0
    _prof_batch_xchg = 0.0     # rank1->rank0 batch send/recv (message 1)
    _prof_r0_fwd = 0.0         # rank0's forward (includes its internal hidden send)
    _prof_r1_verify_wait = 0.0  # rank1 blocked waiting for rank0's hidden send
    _prof_r1_verify_fwd = 0.0  # rank1's ACTUAL compute (wait subtracted out)
    _prof_draft = 0.0          # rank1's DSpark draft() call for next cycle
    _prof_trim_xchg = 0.0      # rank1->rank0 trim/accept send/recv (message 2)
    _prof_total = 0.0

    _log(f"dspark decode loop start: max_tokens={max_tokens}, "
         f"block_size={bs if is_last_rank else '?'}")

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
        for _w in _sweep_widths:
            _sweep_tok = mx.array([[int(y.item())] * _w], dtype=mx.int32)
            _t0 = time.perf_counter()
            if is_rank0:
                _configure_layers(spec_first, spec_last,
                                  pp_send=True, state_list=None, hidden_idx=-1)
                with mx.stream(generation_stream):
                    _sweep_logits = model(_sweep_tok, cache=prompt_cache)
                    mx.eval(_sweep_logits)
            elif is_last_rank:
                _configure_layers(spec_first, spec_last,
                                  pp_recv=True, pp_decode=True,
                                  state_list=None, hidden_idx=-1)
                with mx.stream(generation_stream):
                    _sweep_out = model(_sweep_tok, cache=prompt_cache)
                    mx.eval(_sweep_out)
            _dt = time.perf_counter() - _t0
            _sweep_results[_w] = _dt
            for _c in prompt_cache:
                if hasattr(_c, "trim"):
                    _c.trim(_w)
            _log(f"width-sweep w={_w} dt={_dt*1000:.2f}ms rank={pp_rank}")
        logger.info(
            f"[WIDTH SWEEP R{pp_rank}] " +
            ", ".join(f"w={w}:{dt*1000:.2f}ms" for w, dt in _sweep_results.items())
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
                y.reshape(1), embed_tokens, model.lm_head, _dsc,
                temperature=0.0, sample_fn=_dspark_sample_greedy,
            )
            mx.eval(_toks)
            _draft_ids = [int(v) for v in _toks[0].tolist()]
            for _c in _dsc:
                _c.trim(bs)
            _drafted_total += len(_draft_ids)

        while n < max_tokens:
            _cycle_t0 = time.perf_counter()
            _t_after_draft = _cycle_t0  # default for rank0 (never set on that branch)
            # ==== rank1 -> rank0: fixed-shape (bs+1)-token batch ====
            # world_size is always 2 for this module (is_rank0/is_last_rank
            # exhaustive, no middle-rank case -- see module docstring),
            # so _wire_batch/_trim_amount are always set on both branches
            # below despite the type-checker's inability to prove that
            # statically across an if/elif with no else.
            _wire_batch: mx.array = mx.zeros(bs + 1, dtype=mx.int32)
            if is_last_rank:
                _batch_ids = [int(y.item())] + _draft_ids
                _wire_batch = mx.array(_batch_ids, dtype=mx.int32)
                mx.eval(_wire_batch)
                _log(f"n={n} dspark_batch_send PRE (rank1->0)")
                _sent = mx.distributed.send(_wire_batch, 0, group=pp_group)
                mx.eval(_sent)
                _log(f"n={n} dspark_batch_send POST")
            elif is_rank0:
                _log(f"n={n} dspark_batch_recv PRE (rank0<-{pp_world_size - 1})")
                _wire_batch = mx.distributed.recv_like(
                    mx.zeros(bs + 1, dtype=mx.int32),
                    pp_world_size - 1, group=pp_group,
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

            _t_after_batch_xchg = time.perf_counter()
            _prof_batch_xchg += _t_after_batch_xchg - _cycle_t0

            # ==== rank0: forward the batch through its own layers ====
            if is_rank0:
                _configure_layers(spec_first, spec_last,
                                  pp_send=True, state_list=None, hidden_idx=-1)
                _batch_tok = _wire_batch.reshape(1, -1)
                with mx.stream(generation_stream):
                    model(_batch_tok, cache=prompt_cache)
            _t_after_r0_fwd = time.perf_counter()
            _prof_r0_fwd += _t_after_r0_fwd - _t_after_batch_xchg

            # ==== rank1: recv hidden batch (existing pipeline boundary,
            # unmodified -- shape-agnostic) + continue its own forward ====
            accepted_ids: list[int] = []
            bonus_token: int | None = None
            n_accepted = 0
            if is_last_rank:
                _configure_layers(spec_first, spec_last,
                                  pp_recv=True, pp_decode=True,
                                  state_list=None, hidden_idx=-1)
                _recv_wait_before = _DSPARK_RECV_WAIT_ACCUM[0]
                with mx.stream(generation_stream):
                    _verify_input = mx.array([[int(y.item())] + _draft_ids])
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
                    if _corrected is not None:
                        _top2 = mx.argsort(_corrected[0], axis=-1)[:, -2]
                        mx.eval(_top2)
                        _top2_list = [int(v) for v in _top2.tolist()]
                    else:
                        _top2_list = []
                    for i in range(bs):
                        _pos_reached[i] += 1
                        if _all_next_list[i] == _draft_ids[i]:
                            accepted_ids.append(_draft_ids[i])
                            _pos_accepted[i] += 1
                            n_accepted += 1
                        else:
                            _pos_rejected_reached[i] += 1
                            if _top2_list and _top2_list[i] == _all_next_list[i]:
                                _pos_rank2_would_hit[i] += 1
                            break
                    bonus_token = _all_next_list[n_accepted]
                    _accepted_total += n_accepted

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
                        _dspark.append_ctx(
                            _ctx_cat[:, : n_accepted + 1], _dsc
                        )

                    # Draft the NEXT cycle now (self-contained, rank1-only)
                    # using the bonus token as the new anchor.
                    _next_toks, _next_corrected, _next_conf = _dspark.draft(
                        mx.array([bonus_token]), embed_tokens, model.lm_head,
                        _dsc, temperature=0.0, sample_fn=_dspark_sample_greedy,
                    )
                    mx.eval(_next_toks)
                    _draft_ids = [int(v) for v in _next_toks[0].tolist()]
                    _corrected = _next_corrected
                    _t_after_draft = time.perf_counter()
                    _prof_draft += _t_after_draft - _t_after_verify
                    for _c in _dsc:
                        _c.trim(bs)
                    _drafted_total += len(_draft_ids)

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
            if is_last_rank:
                _accepted_ids = accepted_ids
                _n_accepted_wire = n_accepted
                _bonus_wire = bonus_token if bonus_token is not None else 0
                _padded = _accepted_ids + [-1] * (bs - len(_accepted_ids))
                _wire2 = mx.array(
                    [_n_accepted_wire, _bonus_wire] + _padded, dtype=mx.int32
                )
                mx.eval(_wire2)
                _log(f"n={n} dspark_trim_send PRE (rank1->0) "
                     f"n_accepted={_n_accepted_wire} bonus={_bonus_wire}")
                _sent2 = mx.distributed.send(_wire2, 0, group=pp_group)
                mx.eval(_sent2)
                _log(f"n={n} dspark_trim_send POST")
            elif is_rank0:
                _log(f"n={n} dspark_trim_recv PRE (rank0<-{pp_world_size - 1})")
                _wire2 = mx.distributed.recv_like(
                    mx.zeros(2 + bs, dtype=mx.int32),
                    pp_world_size - 1, group=pp_group,
                )
                mx.eval(_wire2)
                _log(f"n={n} dspark_trim_recv POST")
                _wire2_list = [int(v) for v in _wire2.tolist()]
                _n_accepted_wire = _wire2_list[0]
                bonus_token = _wire2_list[1]
                _accepted_ids = _wire2_list[2 : 2 + _n_accepted_wire]

            _t_after_trim_xchg = time.perf_counter()
            # rank0's baseline for this phase is _t_after_r0_fwd (it has no
            # verify/draft phases); rank1's is _t_after_draft. Both are
            # correct "end of previous phase" markers on their own rank.
            _prof_trim_xchg += _t_after_trim_xchg - (
                _t_after_draft if is_last_rank else _t_after_r0_fwd
            )

            # ==== KV rollback (both ranks, deterministic given bs and
            # n_accepted) ====
            _trim_amount = bs - _n_accepted_wire
            if _trim_amount > 0:
                for c in prompt_cache:
                    if hasattr(c, "trim"):
                        c.trim(_trim_amount)

            _prof_cycle_n += 1
            _prof_total += time.perf_counter() - _cycle_t0
            if _TRACE and _prof_cycle_n % 16 == 0:
                _pn = _prof_cycle_n
                logger.info(
                    f"[PROF dspark R{pp_rank} x{_pn}] "
                    f"batch_xchg={_prof_batch_xchg/_pn*1000:.2f}ms "
                    f"r0_fwd={_prof_r0_fwd/_pn*1000:.2f}ms "
                    f"r1_verify_wait={_prof_r1_verify_wait/_pn*1000:.2f}ms "
                    f"r1_verify_fwd={_prof_r1_verify_fwd/_pn*1000:.2f}ms "
                    f"draft={_prof_draft/_pn*1000:.2f}ms "
                    f"trim_xchg={_prof_trim_xchg/_pn*1000:.2f}ms "
                    f"cycle_total={_prof_total/_pn*1000:.2f}ms"
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
            for tok_id in _accepted_ids:
                yield tok_id, mx.zeros(1)
                n += 1
                if n >= max_tokens:
                    break
            if n < max_tokens and bonus_token is not None:
                yield bonus_token, mx.zeros(1)
                y = mx.array([bonus_token])
                n += 1
            elif _accepted_ids:
                y = mx.array([_accepted_ids[-1]])

    finally:
        _configure_layers(spec_first, spec_last)
        if _drafted_total > 0:
            logger.debug(
                f"PP DSpark: {_accepted_total}/{_drafted_total} "
                f"drafted tokens accepted ({_accepted_total/_drafted_total*100:.0f}%), "
                f"block_size={bs}"
            )
        if is_last_rank and sum(_pos_reached) > 0:
            _hist = ", ".join(
                f"pos{i}={_pos_accepted[i]}/{_pos_reached[i]} "
                f"({_pos_accepted[i]/_pos_reached[i]*100:.0f}%)"
                if _pos_reached[i] > 0 else f"pos{i}=n/a"
                for i in range(bs)
            )
            logger.debug(f"PP DSpark per-position accept histogram: {_hist}")
        if is_last_rank and sum(_pos_rejected_reached) > 0:
            _tree_hist = ", ".join(
                f"pos{i}={_pos_rank2_would_hit[i]}/{_pos_rejected_reached[i]} "
                f"({_pos_rank2_would_hit[i]/_pos_rejected_reached[i]*100:.0f}%)"
                if _pos_rejected_reached[i] > 0 else f"pos{i}=n/a"
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

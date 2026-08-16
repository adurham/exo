# pyright: reportPrivateUsage=false
"""Batched-decode pipeline layers for Phase 1 (N=2 concurrent
decode-only requests) -- see
``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 6.2
and the design's Phase 1 scope.

Kept as NEW, separate classes in a NEW file (not modifications to
``pp_metaframe.py``'s ``MetaFramedPipelineFirstLayer``/
``MetaFramedPipelineLastLayer``) -- matching this fork's own
established pattern (``pp_metaframe.py``'s module docstring point 5:
"new classes, not in-place flag-gating") of never touching an
already-cluster-verified transport path when adding new capability.
Phase 0.5's metaframe layers (concurrency=1, fixed ``request_uid`` at
construction) are completely untouched by this module.

Design question this module answers: how does ONE layer instance
serve DIFFERENT sets of request_uids on different calls (the batching
case), without falling back to the ambient-mutable-instance-flag
anti-pattern (``is_prefill``/``queue_sends``) that ``pp_metaframe.py``
was explicitly built to eliminate?

Reviewed via two `consult` calls (2026-08-05) before writing code:

1. **Per-call context is the right mechanism** -- instance flags fail
   precisely in the case pipeline parallelism creates (multiple
   microbatches/steps potentially in flight through the same layer
   instance), so this isn't just aesthetics.

2. **``contextvars.ContextVar`` scoped by a context manager around the
   single per-step ``model(...)`` call**, not embedding context in a
   cache-slot object (the alternative the first consult call
   suggested for stock mlx-lm calling conventions). Chosen because:
   - exo's forward loop is fully owned by this fork (it already
     replaced pipeline layer wiring via ``pipeline_auto_parallel``),
     so "explicit context, cleanest when you own the loop" (the first
     consult's option 2) applies directly -- no need for the
     cache-slot workaround meant for callers who DON'T own the loop.
   - The lifetime match is exact: per-step batch composition/ordering
     is per-CALL data; a cache slot is a per-REQUEST persistent
     structure that can go stale across steps as batch membership
     changes (a request finishes, a new one is admitted) -- the second
     consult call flagged this as the cache-slot approach's real
     weakness for this specific shape.
   - This module's actual execution shape (confirmed safe against the
     second consult's stated caveats): synchronous, single-process per
     rank, exactly one ``model(...)`` call per step, decode-only (no
     threads inside the forward pass, no cross-process pipeline stages
     within one rank's own layer set, no deferred/overlapped step
     execution). ContextVars do NOT propagate across
     ``threading.Thread`` boundaries or process boundaries -- neither
     applies here since rank-to-rank communication already goes
     through the real wire (metaframe), not through the Python
     context.

3. **No default value on the ContextVar** -- reading it outside an
   active step's scope raises ``LookupError`` immediately (a real
   scoping bug fails loudly) rather than silently defaulting to an
   empty/None context, which the second consult call flagged as "the
   worst failure class in a batched KV-cache system" (silently wrong
   output routed to the wrong request, not a crash).

4. **``token = var.set(...)`` / ``var.reset(token)`` in ``finally``**,
   not ``var.set(None)`` on exit, so exceptions during the step don't
   leave the ContextVar in an inconsistent state for whatever runs
   next on the same thread.

5. **Explicit ordering assertion in the last layer**: nothing
   structurally ties the context's ``request_uids`` ordering to the
   actual row order of the batched tensor except convention at the two
   sites that set it up -- the second consult call flagged this as the
   main defense point against a silent row/uid misalignment (which
   would swap tokens between requests, a real correctness bug, not a
   crash). ``BatchedMetaFramedPipelineLastLayer`` asserts
   ``len(ctx.request_uids) == x.shape[0]`` on every call rather than
   trusting the caller silently.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from exo.worker.engines.mlx.auto_parallel import _LayerCallable
from exo.worker.engines.mlx.pp_metaframe import (
    MetaFramedPipelineFirstLayer,
    MetaFramedPipelineLastLayer,
    encode_batched_decode_metaframe,
    recv_metaframe,
    send_metaframe,
)

# Per-layer decode phase attribution, opt-in via
# EXO_DECODE_PHASE_TRACE=1 (Section 63). Read once at import so the hot
# path costs a single resolved bool test when disabled. Mirrors the flag
# in pp_batched_decode_glue.py / pp_batched_decode_runtime.py.
#
# Section 62 narrowed the ~2.08 s/token to the model(...) call itself on
# BOTH ranks simultaneously, with both GPUs at ~5% and the follower NOT
# starved -- a symmetric mutual block inside the forward. These probes
# split that call into its real blocking points (first-layer recv,
# layer body, last-layer send, gather send/recv) to answer the one
# question Section 62 could not: is the time spread across many small
# synchronous stalls, or concentrated in a single blocking transfer?
_DECODE_PHASE_TRACE: bool = os.environ.get("EXO_DECODE_PHASE_TRACE", "0") == "1"


@dataclass(frozen=True, slots=True)
class BatchStepContext:
    """Per-step batch composition, read by the batched-decode layers
    via a ``ContextVar`` scoped to exactly one ``model(...)`` call.

    ``request_uids``: ORDERED, matching the batched activation tensor's
    batch-axis (axis 0) row order exactly -- row ``i`` of the tensor
    belongs to ``request_uids[i]``. This ordering convention is
    maintained by whoever constructs the batch (the caller assembling
    the step, e.g. from ``BatchedCacheRouter.occupied_slots()``'s
    ascending-order convention) and this dataclass does not itself
    enforce it beyond the last layer's length check -- see module
    docstring point 5.
    """

    request_uids: tuple[int, ...]


_batch_step_context: ContextVar[BatchStepContext] = ContextVar("batch_step_context")


@contextmanager
def batch_step_scope(ctx: BatchStepContext) -> Iterator[None]:
    """Scope ``ctx`` as the active ``BatchStepContext`` for the
    duration of exactly one ``model(...)`` call. Must wrap the ENTIRE
    forward pass (not just part of it) -- both
    ``BatchedMetaFramedPipelineFirstLayer`` (reads it on receive) and
    ``BatchedMetaFramedPipelineLastLayer`` (reads it on send) need it
    live for their respective calls within the same forward pass.
    """
    token = _batch_step_context.set(ctx)
    try:
        yield
    finally:
        _batch_step_context.reset(token)


def _require_batch_step_context() -> BatchStepContext:
    """Read the active ``BatchStepContext``, raising immediately
    (rather than silently defaulting) if called outside a
    ``batch_step_scope`` block -- per module docstring point 3, a
    scoping bug must fail loudly, never silently route to an empty or
    wrong request set."""
    try:
        return _batch_step_context.get()
    except LookupError as e:
        raise RuntimeError(
            "BatchedMetaFramedPipelineFirstLayer/LastLayer called outside "
            "an active batch_step_scope(...) block -- this is a caller "
            "bug (forgot to wrap the model(...) call), not a data-driven "
            "condition; refusing to guess at a request set"
        ) from e


class BatchedMetaFramedPipelineFirstLayer(MetaFramedPipelineFirstLayer):
    """Batched-decode counterpart to
    ``pp_metaframe.MetaFramedPipelineFirstLayer``. Reads the current
    ``BatchStepContext`` (via ``ContextVar``, never an instance
    attribute) to know how many requests to expect on receive, instead
    of Phase 0.5's implicit single-request assumption.

    SUBCLASSES ``MetaFramedPipelineFirstLayer`` (rather than wrapping
    an owned instance of it) specifically so that any forward pass
    OUTSIDE an active ``batch_step_scope`` -- prefill, warmup, or any
    other single-request call site this fork's existing code already
    makes without ever being updated to know about batching --
    transparently falls back to Phase 0.5's already-cluster-verified
    single-request behavior via ``super().__call__()``, with ZERO
    state duplication (same ``r``/``group``, same base class, no
    setter-forwarding to keep in sync). Found and fixed 2026-08-05
    after the first real cluster run: installing these layers at
    model-load time means EVERY forward pass through this model goes
    through them, not just batched-decode's own tick()-driven steps --
    warmup's ``prefill()`` call (unmodified, no ``batch_step_scope``
    wrapper) crashed immediately on `_require_batch_step_context`'s
    fail-loud guard, which is exactly correct behavior for a genuine
    caller bug but was actually THIS class failing to also handle
    the legitimate no-batch-context case. Per a `consult` review:
    subclassing (not composition-with-setter-forwarding) is the
    correct fix here specifically because both layer kinds construct
    with the exact same (r/group) or (r/s/group) arguments as their
    base class -- there is no divergent state to reconcile.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if _batch_step_context.get(None) is None:
            return super().__call__(x, *args, **kwargs)
        if self.r != 0:
            ctx = _require_batch_step_context()
            frame = recv_metaframe(self.r - 1, group=self.group)
            if frame.batch_axis != 1:
                raise RuntimeError(
                    f"BatchedMetaFramedPipelineFirstLayer expected a "
                    f"batch_axis=1 frame (decode batching), got "
                    f"batch_axis={frame.batch_axis} -- a non-batched "
                    f"metaframe layer's frame reached the batched path, "
                    f"or the sender used the wrong encode function"
                )
            if frame.num_requests != len(ctx.request_uids):
                raise RuntimeError(
                    f"BatchedMetaFramedPipelineFirstLayer received a "
                    f"frame describing {frame.num_requests} requests but "
                    f"this step's BatchStepContext expects "
                    f"{len(ctx.request_uids)} "
                    f"({ctx.request_uids}) -- sender/receiver batch "
                    f"composition mismatch, refusing to proceed rather "
                    f"than silently misroute rows between requests"
                )
            if tuple(frame.request_uids) != ctx.request_uids:
                raise RuntimeError(
                    f"BatchedMetaFramedPipelineFirstLayer received a "
                    f"frame with request_uids={frame.request_uids} but "
                    f"this step's BatchStepContext expects "
                    f"{ctx.request_uids} -- ORDER or IDENTITY mismatch "
                    f"between sender and receiver; refusing to proceed "
                    f"rather than silently swap tokens between requests "
                    f"(the exact failure mode module docstring point 5 "
                    f"exists to catch)"
                )
            template_shape = frame.activation_template_shape()
            x_bf16_template = mx.zeros(template_shape, dtype=mx.bfloat16)
            _t_recv = time.perf_counter() if _DECODE_PHASE_TRACE else 0.0
            x_recv = mx.distributed.recv_like(
                x_bf16_template, self.r - 1, group=self.group
            )
            mx.eval(x_recv)
            if _DECODE_PHASE_TRACE:
                logger.info(
                    f"[LAYER_PHASE] first_layer_recv="
                    f"{(time.perf_counter() - _t_recv) * 1000.0:.1f}ms"
                )
            x_dtype = x.dtype
            x = x_recv.astype(x_dtype) if x_dtype != mx.bfloat16 else x_recv
        _t_body = time.perf_counter() if _DECODE_PHASE_TRACE else 0.0
        _out = self.original_layer(x, *args, **kwargs)
        if _DECODE_PHASE_TRACE:
            logger.info(
                f"[LAYER_PHASE] first_layer_body="
                f"{(time.perf_counter() - _t_body) * 1000.0:.1f}ms"
            )
        return _out


class BatchedMetaFramedPipelineLastLayer(MetaFramedPipelineLastLayer):
    """Batched-decode counterpart to
    ``pp_metaframe.MetaFramedPipelineLastLayer``. Decode-only (Phase 1
    scope, per the design doc: "2 concurrent plain (no DSpark)
    decode-only requests") -- deliberately does NOT implement the
    prefill/``queue_sends`` paths that make the Phase 0.5 class more
    general FOR THE BATCHED PATH; those aren't in scope for the
    batched case yet, and omitting them keeps that branch's control
    flow simple enough to reason about directly rather than adding
    untested branches.

    SUBCLASSES ``MetaFramedPipelineLastLayer`` for the SAME reason as
    ``BatchedMetaFramedPipelineFirstLayer`` above (see that class's
    docstring for the full rationale) -- any forward pass outside an
    active ``batch_step_scope`` (prefill, warmup) falls back to
    ``super().__call__()``, reusing Phase 0.5's already-cluster-
    verified single-request prefill/queue_sends/decode-gather logic
    UNCHANGED, rather than this batched class needing to reimplement
    (and get right a second time) prefill-phase handling it was never
    actually designed to cover. ``request_uid`` defaults to 1 (a
    single stable placeholder, never read as meaningfully unique --
    matches ``install_metaframed_pipeline_layers(model, group,
    request_uid=1)``'s own established Phase 0.5 convention for
    concurrency=1 call sites) since the fallback path only ever
    serves ONE request at a time by construction; the batched path
    below never reads ``self.request_uid`` at all, it reads
    ``BatchStepContext.request_uids`` instead.
    """

    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group,
    ) -> None:
        super().__init__(original_layer, r=r, s=s, group=group, request_uid=1)

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if _batch_step_context.get(None) is None:
            return super().__call__(x, *args, **kwargs)
        ctx = _require_batch_step_context()
        if x.shape[0] != len(ctx.request_uids):
            raise RuntimeError(
                f"BatchedMetaFramedPipelineLastLayer: activation tensor's "
                f"batch dim is {x.shape[0]} but this step's "
                f"BatchStepContext describes {len(ctx.request_uids)} "
                f"requests ({ctx.request_uids}) -- the tensor's batch "
                f"row count must always match the context's ordered "
                f"request list (module docstring point 5's ordering "
                f"invariant), refusing to proceed on a mismatch rather "
                f"than silently misattributing rows to the wrong "
                f"requests"
            )
        _t_body = time.perf_counter() if _DECODE_PHASE_TRACE else 0.0
        output: mx.array = self.original_layer(x, *args, **kwargs)
        mx.eval(output)
        if _DECODE_PHASE_TRACE:
            logger.info(
                f"[LAYER_PHASE] last_layer_body_and_eval="
                f"{(time.perf_counter() - _t_body) * 1000.0:.1f}ms"
            )

        if self.r != self.s - 1:
            out_dtype = output.dtype
            output_to_send = (
                output.astype(mx.bfloat16) if output.dtype != mx.bfloat16 else output
            )
            seq_len = int(output_to_send.shape[1])
            hidden_dim = int(output_to_send.shape[-1])
            extra_dim = int(output_to_send.shape[2]) if output_to_send.ndim == 4 else 0
            if seq_len != 1:
                raise RuntimeError(
                    f"BatchedMetaFramedPipelineLastLayer: decode-only "
                    f"scope requires seq_len=1 per request, got "
                    f"seq_len={seq_len} -- a prefill-shaped activation "
                    f"reached the batched-decode-only path"
                )
            header, table = encode_batched_decode_metaframe(
                hidden_dim=hidden_dim,
                request_uids=list(ctx.request_uids),
                seq_len=seq_len,
                extra_dim=extra_dim,
            )
            dst = (self.r + 1) % self.s
            _t_send = time.perf_counter() if _DECODE_PHASE_TRACE else 0.0
            send_metaframe(header, table, dst, group=self.group)
            sent_forward = mx.distributed.send(output_to_send, dst, group=self.group)
            # CRITICAL: force this send to execute NOW, exactly per the
            # incident documented in pp_metaframe.py's
            # MetaFramedPipelineLastLayer (found on the first real
            # 2-node cluster run, 2026-08-05) -- MLX distributed ops
            # are lazy; the decode-gather block below reassigns
            # `output`, which would silently drop the only reference to
            # this send's lazy graph node before anything forces it to
            # run.
            mx.eval(sent_forward)
            if _DECODE_PHASE_TRACE:
                logger.info(
                    f"[LAYER_PHASE] last_layer_send="
                    f"{(time.perf_counter() - _t_send) * 1000.0:.1f}ms"
                )
            output = sent_forward
            if out_dtype != mx.bfloat16:
                output = output.astype(out_dtype)

        # Decode-only final-hidden-state handoff (last rank -> rank 0),
        # matching pp_metaframe.py's MetaFramedPipelineLastLayer
        # exactly -- required for decode to function at all, not
        # optional (rank 0 must receive the final hidden state to
        # sample the next token for EVERY request in the batch).
        gather_dtype = output.dtype
        output_for_gather = (
            output.astype(mx.bfloat16) if output.dtype != mx.bfloat16 else output
        )
        if self.r == self.s - 1:
            seq_len = int(output_for_gather.shape[1])
            hidden_dim = int(output_for_gather.shape[-1])
            extra_dim = (
                int(output_for_gather.shape[2]) if output_for_gather.ndim == 4 else 0
            )
            header, table = encode_batched_decode_metaframe(
                hidden_dim=hidden_dim,
                request_uids=list(ctx.request_uids),
                seq_len=seq_len,
                extra_dim=extra_dim,
            )
            send_metaframe(header, table, 0, group=self.group)
            _t_g = time.perf_counter() if _DECODE_PHASE_TRACE else 0.0
            sent = mx.distributed.send(output_for_gather, 0, group=self.group)
            mx.eval(sent)
            if _DECODE_PHASE_TRACE:
                logger.info(
                    f"[LAYER_PHASE] gather_send="
                    f"{(time.perf_counter() - _t_g) * 1000.0:.1f}ms"
                )
        elif self.r == 0:
            frame = recv_metaframe(self.s - 1, group=self.group)
            if tuple(frame.request_uids) != ctx.request_uids:
                raise RuntimeError(
                    f"BatchedMetaFramedPipelineLastLayer (rank 0 gather): "
                    f"received frame with request_uids="
                    f"{frame.request_uids} but this step's "
                    f"BatchStepContext expects {ctx.request_uids} -- "
                    f"gather-path order/identity mismatch"
                )
            template = mx.zeros(frame.activation_template_shape(), dtype=mx.bfloat16)
            _t_gr = time.perf_counter() if _DECODE_PHASE_TRACE else 0.0
            output_for_gather = mx.distributed.recv_like(
                template, self.s - 1, group=self.group
            )
            mx.eval(output_for_gather)
            if _DECODE_PHASE_TRACE:
                logger.info(
                    f"[LAYER_PHASE] gather_recv="
                    f"{(time.perf_counter() - _t_gr) * 1000.0:.1f}ms"
                )
        # Middle ranks (s > 2): no-op passthrough.

        if gather_dtype != mx.bfloat16:
            output_for_gather = output_for_gather.astype(gather_dtype)
        return output_for_gather


def get_batched_pipeline_info(
    model: nn.Module,
) -> tuple[int, int, mx.distributed.Group] | None:
    """Extract (rank, world_size, group) from the BATCHED pipeline
    layer wrappers -- mirrors ``pp_speculation.get_pipeline_info``'s
    exact contract (rank/world_size/group tuple, or ``None`` if the
    model isn't batched-pipeline-parallel), one level up: THIS
    function detects ``BatchedMetaFramedPipelineLastLayer``
    specifically, not the legacy ``PipelineLastLayer``
    ``get_pipeline_info`` looks for. A caller needing to know whether
    ``EXO_PP_BATCHED_DECODE=1``'s layers are actually installed (as
    opposed to Phase 0.5's single-request metaframe layers, or no
    metaframe layers at all) should use this function, not
    ``get_pipeline_info``, which would silently return ``None`` for a
    batched-installed model (a real, easy-to-hit false negative if
    conflated with this function).
    """
    for layer in model.layers:  # type: ignore
        if isinstance(layer, BatchedMetaFramedPipelineLastLayer):
            return (layer.r, layer.s, layer.group)
    return None


def install_batched_decode_pipeline_layers(
    model: nn.Module,
    group: mx.distributed.Group,
) -> None:
    """Replace an already-``pipeline_auto_parallel``-sharded model's
    first/last ``PipelineFirstLayer``/``PipelineLastLayer`` instances
    with ``BatchedMetaFramedPipelineFirstLayer``/
    ``BatchedMetaFramedPipelineLastLayer``, in place, at MODEL-LOAD
    TIME -- mirrors ``pp_metaframe.install_metaframed_pipeline_layers``
    exactly (same idempotency check, same ``_set_layers`` call, same
    "raise if nothing replaced" failure mode), one level up: THIS
    function installs the BATCHED classes instead of the Phase 0.5
    single-request ones.

    Deliberately a one-time, load-time swap (never a per-request
    mid-flight swap) -- both batched layer classes take no per-request
    state at construction (``r``/``s``/``group`` only; per-step batch
    composition flows entirely through ``batch_step_scope``'s
    ``ContextVar``, never through instance attributes), so installing
    them once before any request runs is safe and sufficient. This
    avoids inventing risky new "swap a request's layers while another
    request's decode may already be in flight in the same batch"
    logic, which was flagged as a genuinely new, unverified piece of
    engineering during this integration's own design review and
    explicitly avoided in favor of this simpler, load-time-only
    installation shape (matching the already-cluster-verified
    ``EXO_PP_METAFRAME`` pattern this function's caller uses
    identically).
    """
    from exo.worker.engines.mlx.auto_parallel import (
        PipelineFirstLayer,
        PipelineLastLayer,
        _set_layers,
        get_inner_model,
        get_layers,
    )

    inner = get_inner_model(model)
    layers = list(get_layers(inner))
    replaced = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, PipelineFirstLayer):
            layers[i] = BatchedMetaFramedPipelineFirstLayer(
                layer.original_layer, r=layer.r, group=layer.group
            )
            replaced += 1
        elif isinstance(layer, PipelineLastLayer):
            layers[i] = BatchedMetaFramedPipelineLastLayer(
                layer.original_layer,
                r=layer.r,
                s=layer.s,
                group=layer.group,
            )
            replaced += 1
    if replaced == 0:
        raise RuntimeError(
            "install_batched_decode_pipeline_layers: found no "
            "PipelineFirstLayer/PipelineLastLayer instances to replace "
            "-- model is not pipeline-sharded, or was already swapped "
            "to a metaframed variant (this function must run directly "
            "on pipeline_auto_parallel's raw output, matching "
            "install_metaframed_pipeline_layers's own precondition)"
        )
    _set_layers(model, layers)

# pyright: reportPrivateUsage=false
"""Metadata-framed PP transport (Phase 0.5 of the batched-PP sharding
design — see ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md``
Section 9).

Scope, per the doc's own Phase 0.5 isolation rationale: replace ONLY the
ambient-mutable-per-layer-state mechanism (``is_prefill``/``queue_sends``
instance flags toggled externally via ``set_pipeline_prefill``/
``set_pipeline_queue_sends``) with an explicit, wire-carried metadata
frame sent immediately before each activation tensor — NO scheduler, NO
batching, concurrency stays =1. That isolation is deliberate: a
transport bug found here, at concurrency=1, is vastly easier to localize
than the same bug discovered later under concurrency=2 batched load
(Risk #11 in the design doc).

Reviewed via `consult` before writing code (2026-08-05). Design points
directly informed by that review:

1. **Frame is header + per-request table, not a single flat tuple.**
   At concurrency=1 the table has exactly one row, but the SHAPE of the
   protocol (fixed header + a `(num_requests, ROW_WIDTH)` table) is the
   one that has to hold at Phase 1+ once multiple requests share one
   step. Building the flat concurrency=1-only version now would force a
   SECOND protocol change at Phase 1 — reintroducing the exact
   "transport bug or batching bug?" ambiguity Phase 0.5 exists to
   eliminate. The header/table split lets Phase 1's scheduler add rows
   without touching the frame's on-wire shape/dtype contract.
2. **A `version` field in the header.** Near-free now, saves a painful
   coordination problem later if the frame layout ever needs to change
   after real code depends on it.
3. **A startup handshake, not just an env var.** If rank 0 runs with
   ``EXO_PP_METAFRAME=1`` and rank 1 doesn't (or vice versa — a
   plausible operator mistake given ``start_cluster.sh``'s per-node
   launch), rank 1 does a plain-activation-shaped ``recv_like`` against
   a 4-int32 header frame (or the reverse) — a silent hang or garbage
   read, NOT a clean error. ``handshake_metaframe_protocol`` exchanges
   and asserts agreement on ``(enabled, version)`` between ranks before
   any real traffic flows, so a config mismatch fails loudly and fast
   at startup instead of wedging mid-request.
4. **Token-parity alone doesn't validate ``queue_sends`` timing.**
   ``queue_sends`` controls WHEN the underlying ``mx.distributed.send``
   actually fires (deferred to a batched flush during chunked prefill,
   vs immediate) — not just shape/metadata. A byte-identical-output test
   would NOT catch a regression here (the model is deterministic
   regardless of send timing, only wall-clock/deadlock risk changes).
   ``MetaFramedPipelineLastLayer`` preserves the exact same
   queue/flush semantics as today's ``PipelineLastLayer`` (reuses
   ``_pending_prefill_sends``/``flush_prefill_sends`` from
   ``auto_parallel.py`` rather than reimplementing queuing) so this
   class of regression is structurally impossible, not just untested.
5. **New classes, not in-place flag-gating.** ``MetaFramedPipelineFirstLayer``/
   ``MetaFramedPipelineLastLayer`` are new, separate classes (thin
   wrappers reusing ``CustomMlxLayer`` and the same send/recv
   primitives) rather than a flag inside the existing
   ``PipelineFirstLayer``/``PipelineLastLayer``. Today's shipped
   transport is completely untouched — the new path can be A/B'd for
   exact parity against it before ever becoming default, with zero risk
   of an in-place change regressing the trusted baseline.

Explicitly NOT addressed here (deferred to Phase 1+, per the doc's own
scoping): a rank-0 scheduler, batching multiple requests into one step,
or cancellation-by-eviction. The frame's ``num_requests``/table
structure is FORWARD-COMPATIBLE with that future work, not an
implementation of it.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    _LayerCallable,
    _pending_prefill_sends,
)

# Bump if the on-wire frame layout ever changes shape/field meaning.
# The startup handshake asserts both ranks agree on this before any
# real traffic flows — see ``handshake_metaframe_protocol``.
#
# v2 (2026-08-05): added the ``extra_dim`` header field after the FIRST
# real cluster run surfaced a genuine correctness bug in v1 — DSv4-Flash
# broadcasts its residual stream to 4D ``(B, L, hc_mult, D)`` right
# after embedding (hyper-connections, see ``HyperHead``/
# ``mlx_lm.models.hyper_connection``) and keeps it that way through
# EVERY layer, only collapsing back to 3D at the final ``hc_head``/
# ``norm``. v1's ``activation_template_shape()`` hardcoded a 3D
# ``(batch, total_tokens, hidden_dim)`` template — silently wrong for
# any model using hyper-connections, since ``mx.distributed.recv_like``
# needs a template matching the ACTUAL sent tensor rank, not just its
# last-dim size. Never caught locally because the Phase 0.5 test suite
# (test_pp_metaframe.py) uses mlx-lm's plain Llama, which has no
# hyper-connections and stays 3D throughout — the gap was invisible
# until the real cluster A/B ran DSv4 for the first time.
METAFRAME_PROTOCOL_VERSION = 2

# Fixed-shape header, sent as one int32 array immediately before the
# per-request table (also int32) and, in turn, immediately before the
# activation tensor itself. Field order:
#   [0] version           — METAFRAME_PROTOCOL_VERSION, sanity-checked
#                            by the receiver on every frame (not just at
#                            handshake) — cheap, catches a hot-reload/
#                            mixed-binary skew the startup handshake
#                            can't see.
#   [1] phase_flag         — 0 = prefill chunk, 1 = decode step.
#   [2] num_requests       — table row count. Always 1 at concurrency=1
#                            (Phase 0.5's scope); Phase 1+ scheduling
#                            increases this without changing the header
#                            shape.
#   [3] hidden_dim         — activation tensor's LAST dim, so the
#                            receiver can derive the full recv_like
#                            template shape from the table's per-request
#                            seq_len without any side-channel knowledge.
#   [4] extra_dim          — v2 addition. 0 means the activation tensor
#                            is 3D: (batch, seq_len, hidden_dim) — the
#                            common case (plain transformer models). A
#                            positive value N means the activation
#                            tensor is 4D: (batch, seq_len, N,
#                            hidden_dim) — DSv4-Flash's hyper-connection
#                            residual stream, where N == config.hc_mult.
#                            Generalizes to "however many extra middle
#                            dims a given model's residual stream needs"
#                            without a further protocol-version bump, as
#                            long as it's exactly one extra dim between
#                            seq_len and hidden_dim (true for every
#                            architecture in this fork today).
_HEADER_FIELDS = 5


# Per-request table row width (int32). Field order per row:
#   [0] request_uid_low32  — low 32 bits of the request UID. exo's
#                            request UIDs are already representable in
#                            32 bits in practice (monotonic counter,
#                            never near 2^32 in a live cluster's
#                            lifetime) — flagged here explicitly rather
#                            than silently truncating a wider ID if that
#                            assumption ever changes.
#   [1] seq_len             — this request's real (unpadded) token count
#                            in the current activation tensor's batch
#                            slot. At concurrency=1 there's no padding
#                            to describe; this is simply the chunk's
#                            token count.
#   [2] is_last_chunk       — 1 if this is the final chunk of THIS
#                            request's current phase (prefill or
#                            decode), 0 otherwise. Replaces
#                            ``is_prefill``'s implicit "are we still in
#                            this phase" signal with an explicit
#                            per-request flag.
#   [3] reserved            — always 0 today; keeps the row width a
#                            round number and gives Phase 1+ one field
#                            of headroom (e.g. a per-request phase
#                            override) without a shape change.
_ROW_WIDTH = 4


class MetaFrame:
    """Decoded metadata frame — the Python-side view rank 1 (or any
    receiver) constructs after reading the raw header+table int32
    arrays off the wire."""

    __slots__ = (
        "version",
        "phase_flag",
        "hidden_dim",
        "extra_dim",
        "request_uids",
        "seq_lens",
        "is_last_chunk",
    )

    def __init__(
        self,
        version: int,
        phase_flag: int,
        hidden_dim: int,
        extra_dim: int,
        request_uids: list[int],
        seq_lens: list[int],
        is_last_chunk: list[bool],
    ) -> None:
        self.version = version
        self.phase_flag = phase_flag
        self.hidden_dim = hidden_dim
        self.extra_dim = extra_dim
        self.request_uids = request_uids
        self.seq_lens = seq_lens
        self.is_last_chunk = is_last_chunk

    @property
    def num_requests(self) -> int:
        return len(self.request_uids)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    def activation_template_shape(self, batch_size: int = 1) -> tuple[int, ...]:
        """Shape of the activation tensor that immediately follows this
        frame on the wire, for use as a ``recv_like`` template. At
        concurrency=1 (Phase 0.5's scope) this is always a single
        request's shape — the ``batch_size`` parameter exists so
        Phase 1+ batching can reuse this method without a signature
        change, not because it does anything yet.

        Returns a 3-tuple ``(batch, total_tokens, hidden_dim)`` when
        ``extra_dim == 0`` (the common case), or a 4-tuple
        ``(batch, total_tokens, extra_dim, hidden_dim)`` when
        ``extra_dim > 0`` — DSv4-Flash's hyper-connection residual
        stream shape. See ``METAFRAME_PROTOCOL_VERSION``'s v2 comment
        for why this field exists.
        """
        if self.extra_dim > 0:
            return (batch_size, self.total_tokens, self.extra_dim, self.hidden_dim)
        return (batch_size, self.total_tokens, self.hidden_dim)


def encode_metaframe(
    *,
    phase_flag: int,
    hidden_dim: int,
    request_uid: int,
    seq_len: int,
    is_last_chunk: bool,
    extra_dim: int = 0,
) -> tuple[mx.array, mx.array]:
    """Build the (header, table) int32 array pair for a SINGLE request
    (concurrency=1 — Phase 0.5's scope). Returns two separate arrays
    (not one concatenated array) so the receiver can ``recv_like`` the
    header with a truly fixed, protocol-version-independent shape
    before it knows ``num_requests`` — see ``recv_metaframe``.

    ``extra_dim``: 0 for a plain 3D ``(batch, seq_len, hidden_dim)``
    activation tensor (most models); pass the size of the extra middle
    dimension (e.g. DSv4-Flash's ``config.hc_mult``) for a 4D
    ``(batch, seq_len, extra_dim, hidden_dim)`` tensor. Caller derives
    this from the ACTUAL tensor being sent (``tensor.ndim == 4``), not
    from a static model-type check — see the layer classes below.
    """
    header = mx.array(
        [METAFRAME_PROTOCOL_VERSION, phase_flag, 1, hidden_dim, extra_dim],
        dtype=mx.int32,
    )
    table = mx.array(
        [[request_uid & 0xFFFFFFFF, seq_len, int(is_last_chunk), 0]],
        dtype=mx.int32,
    )
    return header, table


def send_metaframe(
    header: mx.array,
    table: mx.array,
    dst: int,
    *,
    group: mx.distributed.Group,
) -> None:
    """Send the header then the table, each fully evaluated before the
    next send — matches the existing ``PipelineFirstLayer``/
    ``PipelineLastLayer`` convention of an explicit ``mx.eval`` before
    every ``mx.distributed.send`` to materialize the tensor and isolate
    the lazy graph (see those classes' own comments in
    ``auto_parallel.py``)."""
    sent_header = mx.distributed.send(header, dst, group=group)
    mx.eval(sent_header)
    sent_table = mx.distributed.send(table, dst, group=group)
    mx.eval(sent_table)


def recv_metaframe(src: int, *, group: mx.distributed.Group) -> MetaFrame:
    """Receive and decode one metadata frame. The header's shape is
    protocol-fixed (``_HEADER_FIELDS`` int32 values) so it can be
    ``recv_like``'d with a constant template with no prior knowledge;
    the table's shape is then derived from the header's decoded
    ``num_requests`` before the table itself is received."""
    header_template = mx.zeros((_HEADER_FIELDS,), dtype=mx.int32)
    header = mx.distributed.recv_like(header_template, src, group=group)
    mx.eval(header)
    header_values = cast(list[int], header.tolist())
    version: int = int(header_values[0])
    phase_flag: int = int(header_values[1])
    num_requests: int = int(header_values[2])
    hidden_dim: int = int(header_values[3])
    extra_dim: int = int(header_values[4])
    if version != METAFRAME_PROTOCOL_VERSION:
        raise RuntimeError(
            f"MetaFrame protocol version mismatch: received {version}, "
            f"this build expects {METAFRAME_PROTOCOL_VERSION} — a rank "
            f"is running mismatched code; do not proceed, this would "
            f"otherwise silently misparse the table/activation that "
            f"follows"
        )
    if num_requests < 1:
        raise RuntimeError(
            f"MetaFrame decoded num_requests={num_requests} (must be "
            f">=1) — malformed frame, refusing to proceed"
        )

    table_template = mx.zeros((num_requests, _ROW_WIDTH), dtype=mx.int32)
    table = mx.distributed.recv_like(table_template, src, group=group)
    mx.eval(table)
    rows = cast(list[list[int]], table.tolist())

    return MetaFrame(
        version=version,
        phase_flag=phase_flag,
        hidden_dim=hidden_dim,
        extra_dim=extra_dim,
        request_uids=[int(row[0]) for row in rows],
        seq_lens=[int(row[1]) for row in rows],
        is_last_chunk=[bool(row[2]) for row in rows],
    )


def handshake_metaframe_protocol(
    enabled: bool,
    group: mx.distributed.Group | None,
) -> None:
    """Exchange and assert agreement on ``(enabled, version)`` across
    ALL ranks in ``group`` before any real request traffic flows.
    Call once at model-load/warmup time, not per-request.

    Rationale (consult review, 2026-08-05): a per-rank env var
    (``EXO_PP_METAFRAME``) mismatch between the two nodes — plausible
    given ``start_cluster.sh``'s per-node launch — would otherwise
    manifest as a silent hang (one rank sends a 4-int32 header frame,
    the other ``recv_like``s an activation-shaped template) or garbage
    data, not a clean error. This makes that failure mode loud and
    immediate instead.

    Uses ``mx.distributed.all_sum`` (available even when coordination
    collectives are otherwise disabled under Pipeline sharding, since
    this runs once at warmup before any PP-specific
    ``EXO_PP_NO_COORD_COLLECTIVE`` gating applies to per-request
    traffic) to agree on an encoded ``(enabled, version)`` value and
    compares it against every rank's own locally-computed value.
    """
    if group is None or group.size() <= 1:
        return
    local_code = mx.array(
        [(1 if enabled else 0) * 1_000_000 + METAFRAME_PROTOCOL_VERSION],
        dtype=mx.int64,
    )
    summed = mx.distributed.all_sum(
        local_code, group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(summed)
    expected_if_unanimous = int(local_code.item()) * group.size()
    if int(summed.item()) != expected_if_unanimous:
        raise RuntimeError(
            f"MetaFrame protocol handshake FAILED: not all {group.size()} "
            f"ranks agree on (EXO_PP_METAFRAME enabled, protocol version) "
            f"— this rank has enabled={enabled}, "
            f"version={METAFRAME_PROTOCOL_VERSION}. Refusing to serve: "
            f"a mismatched pair would otherwise hang or corrupt data on "
            f"the first real request instead of failing here at "
            f"startup. Check EXO_PP_METAFRAME is set IDENTICALLY on "
            f"every node in start_cluster.sh's launch."
        )


class MetaFramedPipelineFirstLayer(CustomMlxLayer):
    """Metadata-framed counterpart to ``PipelineFirstLayer``
    (auto_parallel.py). Structurally a thin wrapper reusing the SAME
    base class and the SAME underlying ``mx.distributed.recv_like``
    primitive — the only behavioral difference is HOW this layer learns
    what to expect (an explicit wire frame, not the ambient
    ``is_prefill`` instance flag no metaframe layer ever reads).

    Deliberately does NOT read or set ``is_prefill`` — a metaframe rank
    never needs it, and NOT reading it makes it structurally impossible
    for a metaframe layer to depend on external ambient state, closing
    off exactly the class of "did the caller forget to call
    set_pipeline_prefill() first" bug that motivated this refactor.
    """

    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        group: mx.distributed.Group,
    ) -> None:
        super().__init__(original_layer)
        self.r: int = r
        self.group = group

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            frame = recv_metaframe(self.r - 1, group=self.group)
            template_shape = frame.activation_template_shape()
            x_bf16_template = mx.zeros(template_shape, dtype=mx.bfloat16)
            x_recv = mx.distributed.recv_like(
                x_bf16_template, self.r - 1, group=self.group
            )
            mx.eval(x_recv)
            x_dtype = x.dtype
            x = x_recv.astype(x_dtype) if x_dtype != mx.bfloat16 else x_recv
        return self.original_layer(x, *args, **kwargs)


class MetaFramedPipelineLastLayer(CustomMlxLayer):
    """Metadata-framed counterpart to ``PipelineLastLayer``
    (auto_parallel.py). Preserves ``queue_sends``' exact queuing
    semantics — the same ``_pending_prefill_sends``/
    ``flush_prefill_sends`` module-level queue from ``auto_parallel.py``
    is reused directly, not reimplemented, so the metaframe path cannot
    diverge from the existing path's send-timing behavior (see module
    docstring point 4).

    Includes the decode-only final-hidden-state handoff (last rank
    sends its output back to rank 0 for sampling, exactly matching
    ``PipelineLastLayer``'s own handoff) — this is REQUIRED for decode
    to function at all (rank 0 must receive the final hidden state to
    sample the next token), not optional scope. An earlier version of
    this docstring incorrectly described this handoff as "out of
    scope"; that was wrong — caught by Phase 0.5's own parity tests
    (decode diverged/deadlocked without it), which is exactly the kind
    of bug this isolated transport-only test phase exists to catch
    before it could compound with scheduler/batching code.
    """

    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group,
        *,
        request_uid: int,
    ) -> None:
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group
        self.request_uid = request_uid
        self.is_prefill: bool = False
        self.queue_sends: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        output: mx.array = self.original_layer(x, *args, **kwargs)
        mx.eval(output)

        if self.r != self.s - 1:
            out_dtype = output.dtype
            output_to_send = (
                output.astype(mx.bfloat16) if output.dtype != mx.bfloat16 else output
            )
            # DSv4-Flash's hyper-connection residual stream is 4D
            # (batch, seq_len, hc_mult, hidden_dim); everything else in
            # this fork is 3D (batch, seq_len, hidden_dim). seq_len is
            # always axis 1 and hidden_dim always the LAST axis
            # regardless of rank, so only the extra middle axis (if
            # present) needs deriving from the actual tensor shape —
            # never a static per-model-type assumption. See
            # METAFRAME_PROTOCOL_VERSION's v2 comment for the incident
            # this fixes.
            seq_len = int(output_to_send.shape[1])
            hidden_dim = int(output_to_send.shape[-1])
            extra_dim = int(output_to_send.shape[2]) if output_to_send.ndim == 4 else 0
            phase_flag = 0 if self.is_prefill else 1
            header, table = encode_metaframe(
                phase_flag=phase_flag,
                hidden_dim=hidden_dim,
                request_uid=self.request_uid,
                seq_len=seq_len,
                is_last_chunk=not self.is_prefill,
                extra_dim=extra_dim,
            )
            dst = (self.r + 1) % self.s
            if self.queue_sends:
                # Match PipelineLastLayer's queued-send semantics exactly:
                # defer the ACTIVATION send (reusing the existing shared
                # queue), but the metadata frame is small and must arrive
                # BEFORE the activation on the wire regardless of queuing
                # — send it immediately; only the (much larger) activation
                # tensor benefits from batched-flush deferral.
                send_metaframe(header, table, dst, group=self.group)
                _pending_prefill_sends.append((output_to_send, dst, self.group))
            else:
                send_metaframe(header, table, dst, group=self.group)
                output = mx.distributed.send(output_to_send, dst, group=self.group)
            if out_dtype != mx.bfloat16:
                # Keep behavior symmetric with PipelineLastLayer, which
                # re-casts `output` (the return value, not the sent copy)
                # back to the caller's original dtype after the send call
                # returns a same-shape "sent" marker array.
                output = output.astype(out_dtype) if self.queue_sends else output

        # DECODE-ONLY final-hidden-state handoff: the last pipeline
        # stage (r == s-1) sends its output back to rank 0 so rank 0 can
        # sample the next token — mirrors PipelineLastLayer's own
        # handoff exactly (see that class's comment in auto_parallel.py
        # for why this is decode-only: during prefill the output is
        # discarded, and firing this handoff mid-prefill-pipeline would
        # deadlock it). Framed the same way as the forward hop for
        # protocol consistency, even though at concurrency=1 the
        # metadata carries no information rank 0 doesn't already know
        # from having sent the original chunk — Phase 1+ batching is
        # exactly the point where this return-path framing starts
        # carrying real information (which of several in-flight
        # requests this handoff belongs to).
        if not self.is_prefill:
            gather_dtype = output.dtype
            output_for_gather = (
                output.astype(mx.bfloat16) if output.dtype != mx.bfloat16 else output
            )
            if self.r == self.s - 1:
                seq_len = int(output_for_gather.shape[1])
                hidden_dim = int(output_for_gather.shape[-1])
                extra_dim = (
                    int(output_for_gather.shape[2])
                    if output_for_gather.ndim == 4
                    else 0
                )
                header, table = encode_metaframe(
                    phase_flag=1,
                    hidden_dim=hidden_dim,
                    request_uid=self.request_uid,
                    seq_len=seq_len,
                    is_last_chunk=True,
                    extra_dim=extra_dim,
                )
                send_metaframe(header, table, 0, group=self.group)
                sent = mx.distributed.send(output_for_gather, 0, group=self.group)
                mx.eval(sent)
            elif self.r == 0:
                frame = recv_metaframe(self.s - 1, group=self.group)
                template = mx.zeros(
                    frame.activation_template_shape(), dtype=mx.bfloat16
                )
                output_for_gather = mx.distributed.recv_like(
                    template, self.s - 1, group=self.group
                )
                mx.eval(output_for_gather)
            # Middle ranks (s > 2): no-op passthrough, matching
            # PipelineLastLayer's own middle-rank behavior.

            if gather_dtype != mx.bfloat16:
                output_for_gather = output_for_gather.astype(gather_dtype)
            output = output_for_gather

        return output


def install_metaframed_pipeline_layers(
    model: nn.Module,
    group: mx.distributed.Group,
    *,
    request_uid: int,
) -> None:
    """Replace an already-``pipeline_auto_parallel``-sharded model's
    first/last ``PipelineFirstLayer``/``PipelineLastLayer`` instances
    with their metaframed counterparts, in place. Intended for Phase
    0.5's isolated A/B test harness (single request, concurrency=1) —
    NOT wired into the production model-loading path yet, since this is
    explicitly a validation-only artifact per the design doc's Phase
    0.5 scoping.
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
            layers[i] = MetaFramedPipelineFirstLayer(
                layer.original_layer, r=layer.r, group=layer.group
            )
            replaced += 1
        elif isinstance(layer, PipelineLastLayer):
            layers[i] = MetaFramedPipelineLastLayer(
                layer.original_layer,
                r=layer.r,
                s=layer.s,
                group=layer.group,
                request_uid=request_uid,
            )
            replaced += 1
    if replaced == 0:
        raise ValueError(
            "install_metaframed_pipeline_layers: found no "
            "PipelineFirstLayer/PipelineLastLayer instances to replace "
            "— model was not sharded via pipeline_auto_parallel, or "
            "install_metaframed_pipeline_layers was called twice"
        )
    _set_layers(model, cast(list[Any], layers))


def set_metaframed_pipeline_prefill(model: nn.Module, is_prefill: bool) -> None:
    """Metaframe-layer counterpart to ``set_pipeline_prefill``. Only
    ``MetaFramedPipelineLastLayer`` needs this (it's used to compute
    ``phase_flag``/``is_last_chunk`` for the OUTGOING frame it builds
    itself) — ``MetaFramedPipelineFirstLayer`` deliberately never reads
    ambient phase state at all (see its class docstring), so it has
    nothing to set here."""
    for layer in model.layers:  # type: ignore[attr-defined]
        if isinstance(layer, MetaFramedPipelineLastLayer):
            layer.is_prefill = is_prefill


def set_metaframed_pipeline_queue_sends(model: nn.Module, queue_sends: bool) -> None:
    """Metaframe-layer counterpart to ``set_pipeline_queue_sends``."""
    for layer in model.layers:  # type: ignore[attr-defined]
        if isinstance(layer, MetaFramedPipelineLastLayer):
            layer.queue_sends = queue_sends

# pyright: reportPrivateUsage=false
"""Real wire encoding for the Phase 1 scheduler-protocol control
messages (``StepMessage``/``EvictMessage``/``EvictAckMessage``,
``pp_scheduler_protocol.py``).

This is the piece a `consult` review (2026-08-05) flagged as missing
before the real decode-loop runtime session
(``pp_batched_decode_runtime.py``) could be trusted across a REAL
2-process transport: rank 1 must receive the ACTUAL ``StepMessage``
rank 0 decided (request ids, cache slots, expected cache lengths, in
order) over the wire -- not re-derive it independently, and not
receive it via in-process Python object sharing the way this
session's simulated-2-thread test harness does today (which only
works because both "ranks" share one process's memory).

Per the consult's warning: **the cache_slot mapping is the dangerous
part, not the uid list** -- a single request_uid alone (what
``pp_metaframe.py``'s ``encode_batched_decode_metaframe`` already
carries in its per-row table) is not sufficient for rank 1 to
validate a step against its OWN independently-tracked cache-router
state; it needs ``cache_slot``, ``phase``, ``expected_cache_len``, and
``n_tokens`` per entry too -- exactly ``BatchEntry``'s fields. This
module is a SEPARATE control-plane wire encoding from
``pp_metaframe.py``'s activation-tensor transport (deliberately kept
distinct: control messages are small, fixed-shape-ish int32 arrays
sent once per real step BEFORE the activation tensors, matching how
``BatchedDecodeSession.prepare_step()`` is designed to be called
before either rank's forward pass starts -- see that module's
docstring).

Wire format, matching ``pp_metaframe.py``'s established convention of
a UNIFORM FIXED-SIZE header regardless of message content (so a
receiver never needs to already know a message's kind before it can
safely receive its header -- avoiding the exact bug class this
module's own first implementation attempt hit: different header
shapes per kind meant a receiver expecting the wrong kind got a raw
transport-level shape-mismatch crash instead of this module's own
clean ``SchedulerWireProtocolError``):

  ALL messages: one fixed 5-int32 header, ALWAYS received first:
    [version, msg_kind, step_id, request_id_or_num_entries, cache_slot_or_zero]

  For MSG_KIND_STEP, ``header[3]`` is ``num_entries`` and
  ``header[4]`` is unused (0) -- the real per-request data is a
  SEPARATE follow-up table, sent/received only after the header
  reveals ``msg_kind == MSG_KIND_STEP``:
    table: num_entries rows of
           [request_id, cache_slot, phase_ordinal, expected_cache_len,
            n_tokens]  (5 int32 per row)

  For MSG_KIND_EVICT / MSG_KIND_EVICT_ACK, ``header[3]`` is
  ``request_id`` and ``header[4]`` is ``cache_slot`` -- the ENTIRE
  message fits in the fixed header, no follow-up table.

  For MSG_KIND_PREFILL, ``header[3]`` is ``cache_slot`` and
  ``header[4]`` is ``n_prompt_tokens`` -- a ``PrefillMessage``'s
  remaining two fields (``request_id``, ``flags``) do not fit
  alongside ``step_id``/``cache_slot``/``n_prompt_tokens`` in the
  5-field header, so they follow in a small FIXED-shape body, sent/
  received only after the header reveals
  ``msg_kind == MSG_KIND_PREFILL``:
    body: [request_id, flags]  (2 int32, always exactly one row --
          unlike MSG_KIND_STEP's variable-length table, a
          ``PrefillMessage`` always carries exactly one request)

MSG_KIND_PREFILL is what folds the "admit request B now, start its
prefill" decision into THIS single-writer control channel instead of
leaving it as an independent per-rank local decision -- see
``PrefillMessage``'s own docstring in ``pp_scheduler_protocol.py`` for
the hardware-confirmed N=2 jaccl deadlock
(``docs/batched-decode-n2-admission-handoff-2026-08-05.md``) this
closes. The header-first-then-branch pattern above extends to it
naturally: a receiver still reads the same fixed 5-int32 header first
and only then learns a 2-int32 body follows.

``phase_ordinal`` encodes ``Phase.PREFILL``/``Phase.DECODE`` as a
fixed integer (NOT ``Enum.value``, which is not part of this module's
stable wire contract) -- see ``_PHASE_TO_ORDINAL``/
``_ORDINAL_TO_PHASE`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from exo.worker.engines.mlx.pp_scheduler_protocol import (
    PREFILL_FLAGS_KNOWN_MASK,
    BatchEntry,
    EvictAckMessage,
    EvictMessage,
    Phase,
    PrefillMessage,
    StepMessage,
)

# Bump if the on-wire message shape/field meaning ever changes. Kept
# independent of pp_metaframe.py's own METAFRAME_PROTOCOL_VERSION --
# this is a different wire protocol (scheduler control messages, not
# activation-tensor transport) with its own compatibility contract.
SCHEDULER_WIRE_PROTOCOL_VERSION = 1

MSG_KIND_STEP = 1
MSG_KIND_EVICT = 2
MSG_KIND_EVICT_ACK = 3
MSG_KIND_PREFILL = 4

_HEADER_FIELDS = 5  # [version, msg_kind, step_id, field_d, field_e]
_STEP_ROW_FIELDS = (
    5  # [request_id, cache_slot, phase_ordinal, expected_cache_len, n_tokens]
)
_PREFILL_BODY_FIELDS = 2  # [request_id, flags]

_PHASE_TO_ORDINAL: dict[Phase, int] = {Phase.PREFILL: 0, Phase.DECODE: 1}
_ORDINAL_TO_PHASE: dict[int, Phase] = {v: k for k, v in _PHASE_TO_ORDINAL.items()}


class SchedulerWireProtocolError(RuntimeError):
    """Raised on any malformed/version-mismatched/unexpected-kind
    control message received off the wire. Fail-stop, matching
    ``pp_scheduler_protocol.py``'s own ``ProtocolViolationError``
    discipline (module docstring point 3 there) -- this module never
    attempts to guess or repair a malformed message, only rejects it
    loudly."""


@dataclass(frozen=True)
class WireHeader:
    """The one, ALWAYS-fixed-shape header every control message
    starts with -- decode this FIRST, unconditionally, before
    deciding (via ``msg_kind``) whether a follow-up table is
    expected. This is what makes kind-mismatch a catchable
    ``SchedulerWireProtocolError`` instead of a raw transport shape
    crash: the receiver never has to guess the shape before it has
    already read enough bytes to know the real kind."""

    version: int
    msg_kind: int
    step_id: int
    field_d: int
    field_e: int


def _encode_header(
    *, msg_kind: int, step_id: int, field_d: int, field_e: int
) -> mx.array:
    return mx.array(
        [SCHEDULER_WIRE_PROTOCOL_VERSION, msg_kind, step_id, field_d, field_e],
        dtype=mx.int32,
    )


def send_header(header: mx.array, dst: int, *, group: mx.distributed.Group) -> None:
    """Send a pre-built header array. Matches
    ``pp_metaframe.send_metaframe``'s explicit-eval-before-next-send
    discipline exactly -- the SAME lazy-eval-across-a-real-RDMA-link
    hazard this session already root-caused once (protocol v3 fix,
    Phase 0.5) applies identically here: an un-evaluated
    ``mx.distributed.send`` result must never be discarded/reassigned
    before ``mx.eval`` runs, or the bytes never actually leave the
    NIC."""
    sent = mx.distributed.send(header, dst, group=group)
    mx.eval(sent)


def recv_header(src: int, *, group: mx.distributed.Group) -> WireHeader:
    """Receive and decode the fixed-shape header ANY control message
    starts with -- callers branch on ``.msg_kind`` to decide what (if
    anything) to receive next. Raises ``SchedulerWireProtocolError``
    on a version mismatch (never silently substitutes a default)."""
    header_template = mx.zeros((_HEADER_FIELDS,), dtype=mx.int32)
    header = mx.distributed.recv_like(header_template, src, group=group)
    mx.eval(header)
    values = cast(list[int], header.tolist())
    version, msg_kind, step_id, field_d, field_e = (int(v) for v in values)
    if version != SCHEDULER_WIRE_PROTOCOL_VERSION:
        raise SchedulerWireProtocolError(
            f"recv_header: version mismatch -- received {version}, "
            f"this rank expects {SCHEDULER_WIRE_PROTOCOL_VERSION}. "
            f"Both ranks must run identical exo builds; refusing to "
            f"guess at a compatible decoding."
        )
    return WireHeader(
        version=version,
        msg_kind=msg_kind,
        step_id=step_id,
        field_d=field_d,
        field_e=field_e,
    )


def _require_kind(header: WireHeader, expected: int, *, fn_name: str) -> None:
    if header.msg_kind != expected:
        _names = {
            MSG_KIND_STEP: "MSG_KIND_STEP",
            MSG_KIND_EVICT: "MSG_KIND_EVICT",
            MSG_KIND_EVICT_ACK: "MSG_KIND_EVICT_ACK",
            MSG_KIND_PREFILL: "MSG_KIND_PREFILL",
        }
        raise SchedulerWireProtocolError(
            f"{fn_name}: expected {_names.get(expected, expected)} "
            f"({expected}), received msg_kind={header.msg_kind} -- the "
            f"two ranks' control-message streams have desynced."
        )


def encode_step_message(message: StepMessage) -> tuple[mx.array, mx.array]:
    """Build the (header, table) int32 array pair for a real
    ``StepMessage`` -- the single source of truth for a step's batch
    composition (see ``pp_batched_decode_driver.py``'s own module
    docstring on why ``StepMessage.entries`` must never be re-derived
    independently). Sent BEFORE the corresponding activation-tensor
    metaframe for the same step, so rank 1 can validate composition
    (via ``RankOneMirror``) and build its own cache/context BEFORE the
    real forward pass begins."""
    header = _encode_header(
        msg_kind=MSG_KIND_STEP,
        step_id=message.step_id,
        field_d=len(message.entries),
        field_e=0,
    )
    if not message.entries:
        # A step with zero entries should never legitimately be
        # encoded (SchedulerCore.handle never produces an empty
        # SendStepCommand) -- but the table's shape must still be
        # well-defined (0 rows) so recv_step_table's recv_like
        # template construction doesn't need a special case.
        table = mx.zeros((0, _STEP_ROW_FIELDS), dtype=mx.int32)
    else:
        table = mx.array(
            [
                [
                    entry.request_id,
                    entry.cache_slot,
                    _PHASE_TO_ORDINAL[entry.phase],
                    entry.expected_cache_len,
                    entry.n_tokens,
                ]
                for entry in message.entries
            ],
            dtype=mx.int32,
        )
    return header, table


def send_step_message(
    message: StepMessage, dst: int, *, group: mx.distributed.Group
) -> None:
    """Send a ``StepMessage`` to ``dst``: header, then table."""
    header, table = encode_step_message(message)
    send_header(header, dst, group=group)
    sent_table = mx.distributed.send(table, dst, group=group)
    mx.eval(sent_table)


def recv_step_table(
    header: WireHeader, src: int, *, group: mx.distributed.Group
) -> StepMessage:
    """Given an already-received ``header`` with
    ``msg_kind == MSG_KIND_STEP`` (see ``recv_header``), receive the
    follow-up table and assemble the full ``StepMessage``. Raises
    ``SchedulerWireProtocolError`` if ``header`` is not actually a
    step header -- callers that already branched on ``header.msg_kind``
    get this as a redundant, cheap safety net, not their only check."""
    _require_kind(header, MSG_KIND_STEP, fn_name="recv_step_table")
    num_entries = header.field_d
    table_template = mx.zeros((num_entries, _STEP_ROW_FIELDS), dtype=mx.int32)
    table = mx.distributed.recv_like(table_template, src, group=group)
    mx.eval(table)
    table_values = cast(list[list[int]], table.tolist())
    entries = tuple(
        BatchEntry(
            request_id=row[0],
            cache_slot=row[1],
            phase=_decode_phase_ordinal(row[2]),
            expected_cache_len=row[3],
            n_tokens=row[4],
        )
        for row in table_values
    )
    return StepMessage(step_id=header.step_id, entries=entries)


def recv_step_message(src: int, *, group: mx.distributed.Group) -> StepMessage:
    """Convenience one-call wrapper: ``recv_header`` + ``recv_step_table``
    for a caller that already knows to expect a ``StepMessage`` next
    (e.g. a test, or a caller with its own out-of-band kind
    expectation). Real production dispatch code that must handle
    ANY of the three kinds arriving next should call ``recv_header``
    directly and branch on ``.msg_kind`` instead."""
    header = recv_header(src, group=group)
    return recv_step_table(header, src, group=group)


def _decode_phase_ordinal(ordinal: int) -> Phase:
    phase = _ORDINAL_TO_PHASE.get(ordinal)
    if phase is None:
        raise SchedulerWireProtocolError(
            f"recv_step_table: unknown phase_ordinal={ordinal} on the "
            f"wire -- not one of {sorted(_ORDINAL_TO_PHASE)}. Refusing "
            f"to guess a phase for a malformed/version-mismatched "
            f"entry."
        )
    return phase


def send_evict_message(
    message: EvictMessage, dst: int, *, group: mx.distributed.Group
) -> None:
    """Send an ``EvictMessage`` to ``dst`` -- fits entirely in the
    fixed header, no follow-up table."""
    header = _encode_header(
        msg_kind=MSG_KIND_EVICT,
        step_id=message.step_id,
        field_d=message.request_id,
        field_e=message.cache_slot,
    )
    send_header(header, dst, group=group)


def decode_evict_message(header: WireHeader) -> EvictMessage:
    """Given an already-received ``header`` with
    ``msg_kind == MSG_KIND_EVICT``, assemble the ``EvictMessage`` --
    no additional receive needed (module docstring: evict messages
    fit entirely in the fixed header)."""
    _require_kind(header, MSG_KIND_EVICT, fn_name="decode_evict_message")
    return EvictMessage(
        step_id=header.step_id, request_id=header.field_d, cache_slot=header.field_e
    )


def recv_evict_message(src: int, *, group: mx.distributed.Group) -> EvictMessage:
    """Convenience one-call wrapper (see ``recv_step_message``'s own
    docstring for the same caveat about production dispatch code)."""
    header = recv_header(src, group=group)
    return decode_evict_message(header)


def encode_prefill_message(message: PrefillMessage) -> tuple[mx.array, mx.array]:
    """Build the (header, body) int32 array pair for a
    ``PrefillMessage`` -- rank 0's in-band "admit this request now and
    prefill it" instruction (see ``PrefillMessage``'s docstring in
    ``pp_scheduler_protocol.py`` for the N=2 jaccl deadlock this
    closes).

    Shape mirrors ``encode_step_message`` deliberately -- header first,
    follow-up array second -- so the receive side is the SAME
    header-then-branch pattern with no new control flow to get wrong.
    The only difference: the body is FIXED-shape ``(2,)``, not a
    variable row count, because a ``PrefillMessage`` always describes
    exactly one request.

    ``flags`` is validated against ``PREFILL_FLAGS_KNOWN_MASK`` HERE,
    at encode time, as well as on receive -- an unknown/reserved bit
    means this build is being asked to transmit semantics it does not
    itself implement, which is a caller bug worth catching on the
    sending rank rather than only surfacing as a peer-side rejection.
    """
    if message.flags & ~PREFILL_FLAGS_KNOWN_MASK:
        raise SchedulerWireProtocolError(
            f"encode_prefill_message: flags={message.flags:#x} sets "
            f"reserved bit(s) outside "
            f"PREFILL_FLAGS_KNOWN_MASK={PREFILL_FLAGS_KNOWN_MASK:#x} -- "
            f"refusing to encode a message whose semantics this build "
            f"does not implement (fail-stop, never mask off)."
        )
    header = _encode_header(
        msg_kind=MSG_KIND_PREFILL,
        step_id=message.step_id,
        field_d=message.cache_slot,
        field_e=message.n_prompt_tokens,
    )
    body = mx.array([message.request_id, message.flags], dtype=mx.int32)
    return header, body


def send_prefill_message(
    message: PrefillMessage, dst: int, *, group: mx.distributed.Group
) -> None:
    """Send a ``PrefillMessage`` to ``dst``: header, then body.

    Follows ``send_header``'s documented eval discipline exactly: the
    ``mx.distributed.send`` result for the body is bound to its own
    name and ``mx.eval``'d immediately, never discarded or reassigned
    before evaluation -- over a real RDMA link an unevaluated send is a
    send that never happened, the exact bug this codebase already
    root-caused once (protocol v3 fix, Phase 0.5).
    """
    header, body = encode_prefill_message(message)
    send_header(header, dst, group=group)
    sent_body = mx.distributed.send(body, dst, group=group)
    mx.eval(sent_body)


def recv_prefill_body(
    header: WireHeader, src: int, *, group: mx.distributed.Group
) -> PrefillMessage:
    """Given an already-received ``header`` with
    ``msg_kind == MSG_KIND_PREFILL`` (see ``recv_header``), receive the
    fixed 2-int32 follow-up body and assemble the full
    ``PrefillMessage``. Raises ``SchedulerWireProtocolError`` if
    ``header`` is not actually a prefill header -- callers that already
    branched on ``header.msg_kind`` get this as a redundant, cheap
    safety net, not their only check (identical rationale to
    ``recv_step_table``).

    Also rejects reserved ``flags`` bits: a peer setting a bit this
    build does not know about is running different admission/routing
    semantics, and silently masking it off would mean routing the
    request's prefill through the WRONG layer stack (batched metaframe
    vs single-request ``MetaFramedPipelineFirstLayer``/``LastLayer``)
    -- i.e. structurally mismatched collectives on the two ranks, which
    is the very deadlock class this message kind exists to eliminate.
    A loud crash here is strictly better.
    """
    _require_kind(header, MSG_KIND_PREFILL, fn_name="recv_prefill_body")
    body_template = mx.zeros((_PREFILL_BODY_FIELDS,), dtype=mx.int32)
    body = mx.distributed.recv_like(body_template, src, group=group)
    mx.eval(body)
    body_values = cast(list[int], body.tolist())
    request_id, flags = (int(v) for v in body_values)
    if flags & ~PREFILL_FLAGS_KNOWN_MASK:
        raise SchedulerWireProtocolError(
            f"recv_prefill_body: received flags={flags:#x} with reserved "
            f"bit(s) set outside "
            f"PREFILL_FLAGS_KNOWN_MASK={PREFILL_FLAGS_KNOWN_MASK:#x} "
            f"(step_id={header.step_id}, request_id={request_id}) -- the "
            f"peer rank is running a build with admission semantics this "
            f"rank does not implement. Both ranks must run identical exo "
            f"builds; refusing to guess at a routing decision."
        )
    return PrefillMessage(
        step_id=header.step_id,
        request_id=request_id,
        cache_slot=header.field_d,
        n_prompt_tokens=header.field_e,
        flags=flags,
    )


def recv_prefill_message(src: int, *, group: mx.distributed.Group) -> PrefillMessage:
    """Convenience one-call wrapper: ``recv_header`` +
    ``recv_prefill_body`` for a caller that already knows to expect a
    ``PrefillMessage`` next (e.g. a test, or a caller with its own
    out-of-band kind expectation). Real production dispatch code that
    must handle ANY of the four kinds arriving next should call
    ``recv_header`` directly and branch on ``.msg_kind`` instead --
    that is precisely the point of the uniform header (see
    ``recv_step_message``'s own docstring)."""
    header = recv_header(src, group=group)
    return recv_prefill_body(header, src, group=group)


def send_evict_ack_message(
    message: EvictAckMessage, dst: int, *, group: mx.distributed.Group
) -> None:
    """Send an ``EvictAckMessage`` to ``dst`` -- the reply rank 1
    sends after receiving an ``EvictMessage`` + genuinely freeing its
    own per-slot cache state, matching ``pp_scheduler_protocol.py``'s
    DRAINING-until-ack invariant (module docstring point 4 there)."""
    header = _encode_header(
        msg_kind=MSG_KIND_EVICT_ACK,
        step_id=message.step_id,
        field_d=message.request_id,
        field_e=message.cache_slot,
    )
    send_header(header, dst, group=group)


def decode_evict_ack_message(header: WireHeader) -> EvictAckMessage:
    """Given an already-received ``header`` with
    ``msg_kind == MSG_KIND_EVICT_ACK``, assemble the
    ``EvictAckMessage``."""
    _require_kind(header, MSG_KIND_EVICT_ACK, fn_name="decode_evict_ack_message")
    return EvictAckMessage(
        step_id=header.step_id, request_id=header.field_d, cache_slot=header.field_e
    )


def recv_evict_ack_message(src: int, *, group: mx.distributed.Group) -> EvictAckMessage:
    """Convenience one-call wrapper (see ``recv_step_message``'s own
    docstring for the same caveat about production dispatch code)."""
    header = recv_header(src, group=group)
    return decode_evict_ack_message(header)

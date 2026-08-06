# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false, reportIndexIssue=false
# pyright: reportUnknownLambdaType=false
"""Tests for pp_scheduler_wire.py -- real wire encoding for the
scheduler control messages (StepMessage/EvictMessage/EvictAckMessage).

Uses the same real-2-thread SimPipelineTransport pattern already
established this session (pp_batched_correctness.py) -- these ARE the
real mx.distributed.send/recv_like calls, mocked only at the transport
layer (queue-based instead of RDMA), matching how pp_metaframe.py's
own tests are structured.
"""

from __future__ import annotations

import threading
from typing import Any, cast
from unittest.mock import patch

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.pp_batched_correctness import (
    _MLX_CALL_LOCK,
    SimPipelineTransport,
    _RankGroup,
)
from exo.worker.engines.mlx.pp_scheduler_protocol import (
    PREFILL_FLAG_SINGLE_REQUEST_FALLBACK,
    PREFILL_FLAGS_KNOWN_MASK,
    BatchEntry,
    EvictAckMessage,
    EvictMessage,
    Phase,
    PrefillMessage,
    StepMessage,
)
from exo.worker.engines.mlx.pp_scheduler_wire import (
    MSG_KIND_EVICT,
    MSG_KIND_EVICT_ACK,
    MSG_KIND_PREFILL,
    MSG_KIND_STEP,
    SCHEDULER_WIRE_PROTOCOL_VERSION,
    SchedulerWireProtocolError,
    decode_evict_ack_message,
    decode_evict_message,
    encode_prefill_message,
    encode_step_message,
    recv_evict_ack_message,
    recv_evict_message,
    recv_header,
    recv_prefill_body,
    recv_prefill_message,
    recv_step_message,
    recv_step_table,
    send_evict_ack_message,
    send_evict_message,
    send_header,
    send_prefill_message,
    send_step_message,
)

_RECV_TIMEOUT_SECONDS = 15.0


def _send_and_recv(send_fn: Any, recv_fn: Any) -> Any:
    """Run ``send_fn(dst=1, group=group0)`` on one real thread and
    ``recv_fn(src=0, group=group1)`` on another, over a real
    SimPipelineTransport, returning whatever ``recv_fn`` returns."""
    transport = SimPipelineTransport()
    group0 = cast(mx.distributed.Group, cast(Any, _RankGroup(0, 2)))
    group1 = cast(mx.distributed.Group, cast(Any, _RankGroup(1, 2)))
    result: dict[str, Any] = {}

    def _sender() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            send_fn(dst=1, group=group0)
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error_send"] = e
        finally:
            _MLX_CALL_LOCK.release()

    def _receiver() -> None:
        _MLX_CALL_LOCK.acquire()
        try:
            result["value"] = recv_fn(src=0, group=group1)
        except BaseException as e:  # noqa: BLE001 - surface on join
            result["error_recv"] = e
        finally:
            _MLX_CALL_LOCK.release()

    with (
        patch("mlx.core.distributed.send", transport.send),
        patch("mlx.core.distributed.recv_like", transport.recv_like),
    ):
        t0 = threading.Thread(target=_sender)
        t1 = threading.Thread(target=_receiver)
        t0.start()
        t1.start()
        t0.join(timeout=_RECV_TIMEOUT_SECONDS)
        t1.join(timeout=_RECV_TIMEOUT_SECONDS)
        if t0.is_alive() or t1.is_alive():
            raise RuntimeError("_send_and_recv: simulated rank thread deadlocked")
    if "error_send" in result:
        raise result["error_send"]
    if "error_recv" in result:
        raise result["error_recv"]
    return result["value"]


def test_encode_step_message_shapes() -> None:
    message = StepMessage(
        step_id=7,
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=42,
                n_tokens=1,
            ),
            BatchEntry(
                request_id=2,
                cache_slot=1,
                phase=Phase.DECODE,
                expected_cache_len=17,
                n_tokens=1,
            ),
        ),
    )
    header, table = encode_step_message(message)
    assert header.shape == (5,)
    assert table.shape == (2, 5)
    header_values = header.tolist()
    assert header_values[0] == SCHEDULER_WIRE_PROTOCOL_VERSION
    assert header_values[2] == 7
    assert header_values[3] == 2
    table_values = table.tolist()
    assert table_values[0] == [1, 0, 1, 42, 1]
    assert table_values[1] == [2, 1, 1, 17, 1]


def test_step_message_roundtrip_over_real_transport() -> None:
    message = StepMessage(
        step_id=3,
        entries=(
            BatchEntry(
                request_id=10,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=5,
                n_tokens=1,
            ),
            BatchEntry(
                request_id=20,
                cache_slot=1,
                phase=Phase.DECODE,
                expected_cache_len=9,
                n_tokens=1,
            ),
        ),
    )
    received = _send_and_recv(
        lambda dst, group: send_step_message(message, dst, group=group),
        recv_step_message,
    )
    assert received == message


def test_step_message_with_prefill_phase_roundtrips() -> None:
    """PREFILL phase entries round-trip correctly too, even though
    Phase 1's own runtime never constructs one yet (module docstring's
    Scope note in pp_scheduler_protocol.py) -- the wire encoding
    itself must not assume DECODE-only, so Phase 2's chunked-prefill
    scheduling doesn't need a wire-format change later."""
    message = StepMessage(
        step_id=1,
        entries=(
            BatchEntry(
                request_id=5,
                cache_slot=0,
                phase=Phase.PREFILL,
                expected_cache_len=100,
                n_tokens=64,
            ),
        ),
    )
    received = _send_and_recv(
        lambda dst, group: send_step_message(message, dst, group=group),
        recv_step_message,
    )
    assert received == message


def test_step_message_roundtrip_single_entry() -> None:
    message = StepMessage(
        step_id=99,
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=1,
                n_tokens=1,
            ),
        ),
    )
    received = _send_and_recv(
        lambda dst, group: send_step_message(message, dst, group=group),
        recv_step_message,
    )
    assert received == message


def test_evict_message_roundtrip_over_real_transport() -> None:
    message = EvictMessage(step_id=4, request_id=2, cache_slot=1)
    received = _send_and_recv(
        lambda dst, group: send_evict_message(message, dst, group=group),
        recv_evict_message,
    )
    assert received == message


def test_evict_ack_message_roundtrip_over_real_transport() -> None:
    message = EvictAckMessage(step_id=5, request_id=2, cache_slot=1)
    received = _send_and_recv(
        lambda dst, group: send_evict_ack_message(message, dst, group=group),
        recv_evict_ack_message,
    )
    assert received == message


def test_recv_step_message_rejects_evict_kind() -> None:
    """A receiver expecting a StepMessage that gets an EvictMessage
    instead (control-message stream desync) must fail loudly, not
    silently misinterpret the bytes -- module docstring's fail-stop
    discipline."""
    evict = EvictMessage(step_id=1, request_id=1, cache_slot=0)
    with pytest.raises(SchedulerWireProtocolError, match="expected MSG_KIND_STEP"):
        _send_and_recv(
            lambda dst, group: send_evict_message(evict, dst, group=group),
            recv_step_message,
        )


def test_recv_evict_message_rejects_step_kind() -> None:
    step = StepMessage(
        step_id=1,
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=1,
                n_tokens=1,
            ),
        ),
    )
    with pytest.raises(SchedulerWireProtocolError, match="expected MSG_KIND_EVICT"):
        _send_and_recv(
            lambda dst, group: send_step_message(step, dst, group=group),
            recv_evict_message,
        )


def test_recv_evict_ack_message_rejects_evict_kind() -> None:
    """EvictMessage and EvictAckMessage share an identical wire shape
    (module docstring) -- MSG_KIND is the ONLY thing distinguishing
    them, so this is the one test that would catch a bug where that
    discrimination silently stopped working."""
    evict = EvictMessage(step_id=1, request_id=1, cache_slot=0)
    with pytest.raises(SchedulerWireProtocolError, match="expected MSG_KIND_EVICT_ACK"):
        _send_and_recv(
            lambda dst, group: send_evict_message(evict, dst, group=group),
            recv_evict_ack_message,
        )


def _real_dispatch_receive(src: int, *, group: mx.distributed.Group) -> Any:
    """The REAL production dispatch pattern this module's docstring
    describes: receive the fixed header FIRST (unconditionally, no
    prior knowledge of what kind is coming), THEN branch on
    ``.msg_kind`` to decide what (if anything) to receive next. This
    is what a real rank-1 control-message loop would do -- unlike
    every other test in this file, which uses the convenience
    one-call wrappers that already assume the caller knows the kind
    in advance."""
    header = recv_header(src, group=group)
    if header.msg_kind == MSG_KIND_STEP:
        return recv_step_table(header, src, group=group)
    if header.msg_kind == MSG_KIND_EVICT:
        return decode_evict_message(header)
    if header.msg_kind == MSG_KIND_EVICT_ACK:
        return decode_evict_ack_message(header)
    if header.msg_kind == MSG_KIND_PREFILL:
        return recv_prefill_body(header, src, group=group)
    raise AssertionError(f"unexpected msg_kind in test: {header.msg_kind}")


def test_real_dispatch_pattern_handles_all_three_kinds_over_one_transport() -> None:
    """The header-first dispatch pattern (recv_header, branch on
    msg_kind) correctly handles ALL THREE message kinds arriving on
    the SAME transport in sequence -- proving send_header's uniform
    5-field shape genuinely lets a receiver with NO prior kind
    knowledge safely receive whatever comes next, which is the whole
    point of unifying the header shape (this module's docstring
    explains the bug the non-uniform first design hit: a receiver
    expecting the wrong kind got a raw transport shape-mismatch crash
    instead of a clean SchedulerWireProtocolError)."""
    step = StepMessage(
        step_id=1,
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=3,
                n_tokens=1,
            ),
        ),
    )
    evict = EvictMessage(step_id=2, request_id=1, cache_slot=0)
    evict_ack = EvictAckMessage(step_id=2, request_id=1, cache_slot=0)

    received_step = _send_and_recv(
        lambda dst, group: send_step_message(step, dst, group=group),
        _real_dispatch_receive,
    )
    assert received_step == step

    received_evict = _send_and_recv(
        lambda dst, group: send_evict_message(evict, dst, group=group),
        _real_dispatch_receive,
    )
    assert received_evict == evict

    received_ack = _send_and_recv(
        lambda dst, group: send_evict_ack_message(evict_ack, dst, group=group),
        _real_dispatch_receive,
    )
    assert received_ack == evict_ack


def test_send_header_and_recv_header_roundtrip_directly() -> None:
    """The lowest-level primitive (send_header/recv_header) round-trips
    correctly in isolation, independent of any message-kind-specific
    encode/decode helper."""
    from exo.worker.engines.mlx.pp_scheduler_wire import _encode_header

    header = _encode_header(msg_kind=MSG_KIND_STEP, step_id=42, field_d=7, field_e=0)
    received = _send_and_recv(
        lambda dst, group: send_header(header, dst, group=group),
        recv_header,
    )
    assert received.version == SCHEDULER_WIRE_PROTOCOL_VERSION
    assert received.msg_kind == MSG_KIND_STEP
    assert received.step_id == 42
    assert received.field_d == 7
    assert received.field_e == 0


def test_encode_prefill_message_shapes() -> None:
    """Pure encode test, no I/O -- mirrors
    test_encode_step_message_shapes. Pins the exact header field
    mapping documented in pp_scheduler_wire.py's module docstring
    (field_d=cache_slot, field_e=n_prompt_tokens) plus the fixed
    2-int32 [request_id, flags] body, so a future field-order change
    can't silently pass round-trip tests (which would still pass if
    encode and decode were changed symmetrically but wrongly)."""
    message = PrefillMessage(
        step_id=11,
        request_id=77,
        cache_slot=1,
        n_prompt_tokens=512,
        flags=PREFILL_FLAG_SINGLE_REQUEST_FALLBACK,
    )
    header, body = encode_prefill_message(message)
    assert header.shape == (5,)
    assert body.shape == (2,)
    header_values = header.tolist()
    assert header_values[0] == SCHEDULER_WIRE_PROTOCOL_VERSION
    assert header_values[1] == MSG_KIND_PREFILL
    assert header_values[2] == 11
    assert header_values[3] == 1  # cache_slot
    assert header_values[4] == 512  # n_prompt_tokens
    assert body.tolist() == [77, PREFILL_FLAG_SINGLE_REQUEST_FALLBACK]


def test_prefill_message_roundtrip_over_real_transport() -> None:
    message = PrefillMessage(
        step_id=6,
        request_id=42,
        cache_slot=0,
        n_prompt_tokens=1024,
        flags=0,
    )
    received = _send_and_recv(
        lambda dst, group: send_prefill_message(message, dst, group=group),
        recv_prefill_message,
    )
    assert received == message


def test_prefill_message_roundtrip_preserves_fallback_flag() -> None:
    """The single-request-fallback bit is the field rank 1 uses to
    decide between the batched metaframe layers and the OLD
    MetaFramedPipelineFirstLayer/LastLayer path -- if it were dropped
    on the wire the two ranks would install structurally different
    layer stacks, which is the same mismatched-collectives deadlock
    this message kind exists to eliminate. So it gets its own explicit
    round-trip test rather than only riding along in the flags=0
    case."""
    message = PrefillMessage(
        step_id=2,
        request_id=9,
        cache_slot=1,
        n_prompt_tokens=7,
        flags=PREFILL_FLAG_SINGLE_REQUEST_FALLBACK,
    )
    received = _send_and_recv(
        lambda dst, group: send_prefill_message(message, dst, group=group),
        recv_prefill_message,
    )
    assert received == message
    assert bool(received.flags & PREFILL_FLAG_SINGLE_REQUEST_FALLBACK)


def test_encode_prefill_message_rejects_reserved_flag_bits() -> None:
    """Fail-stop on the SENDING rank too: a reserved flag bit means
    the caller is asking this build to transmit semantics it does not
    implement."""
    bad = PrefillMessage(
        step_id=1,
        request_id=1,
        cache_slot=0,
        n_prompt_tokens=1,
        flags=~PREFILL_FLAGS_KNOWN_MASK & 0xFF,
    )
    with pytest.raises(SchedulerWireProtocolError, match="reserved bit"):
        encode_prefill_message(bad)


def test_recv_prefill_message_rejects_evict_kind() -> None:
    """A receiver expecting a PrefillMessage that gets an EvictMessage
    instead (control-message stream desync) must fail loudly via
    _require_kind rather than blocking forever waiting for a 2-int32
    body that will never be sent."""
    evict = EvictMessage(step_id=1, request_id=1, cache_slot=0)
    with pytest.raises(SchedulerWireProtocolError, match="expected MSG_KIND_PREFILL"):
        _send_and_recv(
            lambda dst, group: send_evict_message(evict, dst, group=group),
            recv_prefill_message,
        )


def test_recv_step_message_rejects_prefill_kind() -> None:
    """And the converse: the pre-existing STEP receiver rejects the
    NEW kind cleanly instead of misreading the prefill header's
    field_d (cache_slot) as a num_entries row count -- the exact
    confusion the per-kind header field reuse could otherwise cause."""
    prefill = PrefillMessage(
        step_id=1, request_id=1, cache_slot=3, n_prompt_tokens=8, flags=0
    )
    with pytest.raises(SchedulerWireProtocolError, match="expected MSG_KIND_STEP"):
        _send_and_recv(
            lambda dst, group: send_prefill_message(prefill, dst, group=group),
            recv_step_message,
        )


def test_real_dispatch_pattern_handles_prefill_kind() -> None:
    """The established header-first-then-branch receive pattern
    extends to the new third traffic shape with no special-casing --
    the whole reason MSG_KIND_PREFILL was added as a header kind
    rather than as an out-of-band channel."""
    prefill = PrefillMessage(
        step_id=1,
        request_id=5,
        cache_slot=1,
        n_prompt_tokens=256,
        flags=PREFILL_FLAG_SINGLE_REQUEST_FALLBACK,
    )
    received = _send_and_recv(
        lambda dst, group: send_prefill_message(prefill, dst, group=group),
        _real_dispatch_receive,
    )
    assert received == prefill

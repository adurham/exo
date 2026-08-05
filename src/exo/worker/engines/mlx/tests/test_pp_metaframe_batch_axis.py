# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false, reportArgumentType=false
# pyright: reportIndexIssue=false
"""Phase 1 tests for the metaframe protocol's batch-axis extension
(v3): ``encode_batched_decode_metaframe``, ``MetaFrame.batch_axis``,
and ``activation_template_shape``'s two structurally different modes.

See ``pp_metaframe.py``'s ``METAFRAME_PROTOCOL_VERSION`` v3 comment for
the full rationale: Phase 0.5's single-request protocol only ever
needed sequence-axis concatenation (multiple prefill CHUNKS of one
growing sequence); Phase 1's decode batching needs BATCH-axis stacking
(N different requests, one token each, on separate batch rows) --
these produce structurally different tensor shapes for the same
table/header data if not explicitly distinguished, which is exactly
what ``batch_axis`` exists to prevent.

This file does NOT touch or re-test anything from Phase 0.5's already
cluster-verified single-request path (test_pp_metaframe.py) -- purely
additive coverage for the new batch-axis functionality.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from exo.worker.engines.mlx.pp_batched_correctness import (
    _MLX_CALL_LOCK,
    _RankGroup,
)
from exo.worker.engines.mlx.pp_metaframe import (
    METAFRAME_PROTOCOL_VERSION,
    MetaFrame,
    encode_batched_decode_metaframe,
    encode_metaframe,
    recv_metaframe,
    send_metaframe,
)


def test_encode_batched_decode_metaframe_sets_batch_axis_1() -> None:
    header, table = encode_batched_decode_metaframe(
        hidden_dim=4096,
        request_uids=[10, 20],
        seq_len=1,
    )
    header_values = header.tolist()
    assert header_values[0] == METAFRAME_PROTOCOL_VERSION
    assert header_values[1] == 1  # phase_flag: decode
    assert header_values[2] == 2  # num_requests
    assert header_values[3] == 4096  # hidden_dim
    assert header_values[4] == 0  # extra_dim
    assert header_values[5] == 1  # batch_axis

    rows = table.tolist()
    assert len(rows) == 2
    assert rows[0][0] == 10  # request_uid
    assert rows[0][1] == 1  # seq_len
    assert rows[0][2] == 1  # is_last_chunk always True for decode
    assert rows[1][0] == 20


def test_encode_batched_decode_metaframe_empty_uids_raises() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        encode_batched_decode_metaframe(hidden_dim=64, request_uids=[], seq_len=1)


def test_encode_metaframe_single_request_still_batch_axis_0() -> None:
    """Backward-compat guard: Phase 0.5's existing single-request
    encode_metaframe must still produce batch_axis=0 after the v3
    header-field addition -- no behavior change for the
    already-cluster-verified path."""
    header, _table = encode_metaframe(
        phase_flag=1, hidden_dim=64, request_uid=1, seq_len=1, is_last_chunk=True
    )
    assert header.tolist()[5] == 0  # batch_axis


def test_metaframe_batch_axis_1_encode_decode_roundtrip() -> None:
    """Real send/recv roundtrip (via the SimPipelineTransport-adjacent
    fake groups already proven safe by test_pp_metaframe.py) confirming
    a batched-decode frame's batch_axis survives the wire."""
    from exo.worker.engines.mlx.pp_batched_correctness import SimPipelineTransport

    transport = SimPipelineTransport()
    group0 = _RankGroup(0, 2)
    group1 = _RankGroup(1, 2)

    header, table = encode_batched_decode_metaframe(
        hidden_dim=4096,
        request_uids=[1, 2],
        seq_len=1,
    )

    _MLX_CALL_LOCK.acquire()
    try:
        with (
            patch("mlx.core.distributed.send", transport.send),
            patch("mlx.core.distributed.recv_like", transport.recv_like),
        ):
            send_metaframe(header, table, 1, group=group0)
            frame = recv_metaframe(0, group=group1)
    finally:
        _MLX_CALL_LOCK.release()

    assert frame.batch_axis == 1
    assert frame.num_requests == 2
    assert frame.request_uids == [1, 2]
    assert frame.seq_lens == [1, 1]


def test_activation_template_shape_batch_axis_1_3d() -> None:
    frame = MetaFrame(
        version=METAFRAME_PROTOCOL_VERSION,
        phase_flag=1,
        hidden_dim=4096,
        extra_dim=0,
        batch_axis=1,
        request_uids=[1, 2],
        seq_lens=[1, 1],
        is_last_chunk=[True, True],
    )
    # batch_axis=1: num_requests becomes axis 0, NOT the batch_size param.
    assert frame.activation_template_shape(batch_size=99) == (2, 1, 4096)


def test_activation_template_shape_batch_axis_1_4d_hyper_connection() -> None:
    """batch_axis=1 combined with DSv4's hyper-connection extra_dim --
    both v2 and v3 header fields interacting correctly together."""
    frame = MetaFrame(
        version=METAFRAME_PROTOCOL_VERSION,
        phase_flag=1,
        hidden_dim=4096,
        extra_dim=4,  # DSv4's real hc_mult
        batch_axis=1,
        request_uids=[1, 2],
        seq_lens=[1, 1],
        is_last_chunk=[True, True],
    )
    assert frame.activation_template_shape() == (2, 1, 4, 4096)


def test_activation_template_shape_batch_axis_1_requires_uniform_seq_lens() -> None:
    """Phase 1 scope invariant: decode steps are naturally uniform-
    length (design doc Section 6.2 item 1) -- a batch-axis frame with
    mismatched per-request seq_lens indicates a real bug upstream
    (e.g. accidentally mixing a prefill chunk into a decode batch) and
    must fail loudly, not silently pick one length and truncate/pad."""
    frame = MetaFrame(
        version=METAFRAME_PROTOCOL_VERSION,
        phase_flag=1,
        hidden_dim=64,
        extra_dim=0,
        batch_axis=1,
        request_uids=[1, 2],
        seq_lens=[1, 5],  # mismatched -- not valid for decode batching
        is_last_chunk=[True, True],
    )
    with pytest.raises(RuntimeError, match="uniform"):
        frame.activation_template_shape()


def test_activation_template_shape_batch_axis_1_empty_requests_raises() -> None:
    frame = MetaFrame(
        version=METAFRAME_PROTOCOL_VERSION,
        phase_flag=1,
        hidden_dim=64,
        extra_dim=0,
        batch_axis=1,
        request_uids=[],
        seq_lens=[],
        is_last_chunk=[],
    )
    with pytest.raises(RuntimeError, match="at least one request"):
        frame.activation_template_shape()


def test_activation_template_shape_batch_axis_0_unchanged() -> None:
    """batch_axis=0 (Phase 0.5's original convention) must produce the
    exact same shape as before the v3 change -- direct regression
    guard against this refactor silently altering the already
    cluster-verified sequence-axis-concat behavior."""
    frame = MetaFrame(
        version=METAFRAME_PROTOCOL_VERSION,
        phase_flag=0,
        hidden_dim=256,
        extra_dim=0,
        batch_axis=0,
        request_uids=[1],
        seq_lens=[128],
        is_last_chunk=[False],
    )
    assert frame.activation_template_shape() == (1, 128, 256)
    assert frame.activation_template_shape(batch_size=1) == (1, 128, 256)

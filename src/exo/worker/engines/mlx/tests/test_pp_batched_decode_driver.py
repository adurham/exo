"""Tests for pp_batched_decode_driver.py -- Phase 1's rank-0/rank-1
scheduler-glue driver.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section
9 for context. This is pure logic (no MLX, no I/O) -- fast unit tests
covering the driver's own bookkeeping-consistency invariants (the
single-source-of-truth guarantee: BatchedDecodeDriver's cache_router
and SchedulerCore always advance together; RankOneMirrorDriver derives
BatchStepContext from the SAME StepMessage.entries it validated).

Real end-to-end MLX-level verification (2 concurrent decode-only
requests through the ACTUAL batched metaframe layers, driven by these
driver classes rather than a hand-built BatchStepContext) belongs in a
separate test file alongside test_pp_batched_decode_correctness.py,
not here -- this file is scoped to the driver's own protocol-glue
correctness, matching pp_scheduler_protocol.py's own "pure core, fast
unit tests" precedent.
"""

from __future__ import annotations

import pytest

from exo.worker.engines.mlx.pp_batched_decode_driver import (
    BatchedDecodeDriver,
    RankOneMirrorDriver,
    batch_step_context_from_step_message,
)
from exo.worker.engines.mlx.pp_scheduler_protocol import (
    EvictMessage,
    ProtocolViolationError,
)


def test_admit_single_request_produces_step_message_and_advances_cache_router() -> None:
    driver = BatchedDecodeDriver.new(max_concurrency=2)
    message = driver.admit_request(request_id=1, cache_slot=0)
    assert message.entries[0].request_id == 1
    assert message.entries[0].expected_cache_len == 0
    assert driver.cache_router.is_occupied(0)
    assert driver.cache_router.length(0) == 0


def test_admit_two_requests_batch_step_context_matches_step_message_order() -> None:
    driver = BatchedDecodeDriver.new(max_concurrency=2)
    driver.admit_request(request_id=1, cache_slot=0)
    message = driver.admit_request(request_id=2, cache_slot=1)
    ctx = batch_step_context_from_step_message(message)
    assert ctx.request_uids == tuple(e.request_id for e in message.entries)
    assert set(ctx.request_uids) == {1, 2}


def test_on_tokens_generated_advances_both_scheduler_and_cache_router() -> None:
    driver = BatchedDecodeDriver.new(max_concurrency=2)
    driver.admit_request(request_id=1, cache_slot=0)
    driver.admit_request(request_id=2, cache_slot=1)
    message = driver.on_tokens_generated((1, 2))

    entries_by_id = {e.request_id: e for e in message.entries}
    assert entries_by_id[1].expected_cache_len == 1
    assert entries_by_id[2].expected_cache_len == 1
    # Cache router's OWN length bookkeeping must match the scheduler's
    # claimed expected_cache_len exactly -- this is the driver's core
    # single-source-of-truth guarantee.
    assert driver.cache_router.length(0) == 1
    assert driver.cache_router.length(1) == 1


def test_evict_then_ack_releases_slot_in_both_scheduler_and_cache_router() -> None:
    driver = BatchedDecodeDriver.new(max_concurrency=2)
    driver.admit_request(request_id=1, cache_slot=0)
    evict_info = driver.evict_request(1)
    assert evict_info.cache_slot == 0
    assert evict_info.request_id == 1
    # Cache router slot must NOT be released yet -- only after the ack.
    assert driver.cache_router.is_occupied(0)

    driver.on_evict_ack(request_id=1, cache_slot=0)
    assert not driver.cache_router.is_occupied(0)


def test_slot_reuse_after_full_evict_ack_cycle_succeeds() -> None:
    """The real end-to-end reuse case: evict + ack a slot, then admit
    a DIFFERENT request into the same slot -- must succeed cleanly
    with a fresh cache_len=0, not inherit any stale state."""
    driver = BatchedDecodeDriver.new(max_concurrency=2)
    driver.admit_request(request_id=1, cache_slot=0)
    driver.on_tokens_generated((1,))
    driver.on_tokens_generated((1,))
    assert driver.cache_router.length(0) == 2

    driver.evict_request(1)
    driver.on_evict_ack(request_id=1, cache_slot=0)

    message = driver.admit_request(request_id=99, cache_slot=0)
    assert message.entries[0].expected_cache_len == 0
    assert driver.cache_router.length(0) == 0


def test_reusing_slot_before_ack_raises_via_underlying_scheduler() -> None:
    """The driver doesn't re-implement the DRAINING invariant -- it
    must propagate SchedulerCore's own rejection of a premature slot
    reuse."""
    driver = BatchedDecodeDriver.new(max_concurrency=2)
    driver.admit_request(request_id=1, cache_slot=0)
    driver.evict_request(1)
    with pytest.raises(ProtocolViolationError, match="DRAINING"):
        driver.admit_request(request_id=2, cache_slot=0)


def test_on_tokens_generated_for_unknown_request_raises() -> None:
    driver = BatchedDecodeDriver.new(max_concurrency=2)
    with pytest.raises(ProtocolViolationError):
        driver.on_tokens_generated((999,))


# ---------------------------------------------------------------------
# RankOneMirrorDriver: validates rank 0's claims, never decides
# anything itself.
# ---------------------------------------------------------------------


def test_mirror_driver_accepts_well_formed_step_and_advances_cache_router() -> None:
    rank0 = BatchedDecodeDriver.new(max_concurrency=2)
    rank1 = RankOneMirrorDriver(max_concurrency=2)

    message = rank0.admit_request(request_id=1, cache_slot=0)
    ctx = rank1.on_step_message(message)
    assert ctx.request_uids == (1,)
    assert rank1.cache_router is not None
    assert rank1.cache_router.is_occupied(0)
    assert rank1.cache_router.length(0) == 0


def test_mirror_driver_advances_length_on_subsequent_steps() -> None:
    rank0 = BatchedDecodeDriver.new(max_concurrency=2)
    rank1 = RankOneMirrorDriver(max_concurrency=2)

    rank1.on_step_message(rank0.admit_request(request_id=1, cache_slot=0))
    message = rank0.on_tokens_generated((1,))
    rank1.on_step_message(message)

    assert rank1.cache_router is not None
    assert rank1.cache_router.length(0) == 1


def test_mirror_driver_two_requests_batch_step_context_matches_rank0() -> None:
    """The real integration guarantee: rank 0's StepMessage and rank
    1's derived BatchStepContext, via the SAME derivation function,
    must produce identical results -- this is what makes the two
    ranks' batch composition agree without any independent
    re-derivation."""
    rank0 = BatchedDecodeDriver.new(max_concurrency=2)
    rank1 = RankOneMirrorDriver(max_concurrency=2)

    rank1.on_step_message(rank0.admit_request(request_id=1, cache_slot=0))
    rank1.on_step_message(rank0.admit_request(request_id=2, cache_slot=1))

    rank0_message = rank0.on_tokens_generated((1, 2))
    rank0_ctx = batch_step_context_from_step_message(rank0_message)
    rank1_ctx = rank1.on_step_message(rank0_message)

    assert rank0_ctx == rank1_ctx


def test_mirror_driver_rejects_step_disagreeing_with_its_own_tracked_state() -> None:
    """A StepMessage claiming a cache length the mirror's own tracked
    state disagrees with must raise -- RankOneMirrorDriver propagates
    RankOneMirror's own fail-stop validation, it doesn't add a second,
    looser check."""
    rank1 = RankOneMirrorDriver(max_concurrency=2)
    from exo.worker.engines.mlx.pp_scheduler_protocol import BatchEntry, Phase

    bad_message = _step_message_with_entries(
        step_id=1,
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=5,  # nonzero on a NEVER-before-seen slot
                n_tokens=0,
            ),
        ),
    )
    with pytest.raises(ProtocolViolationError):
        rank1.on_step_message(bad_message)


def test_mirror_driver_evict_validates_without_freeing_until_caller_does() -> None:
    rank0 = BatchedDecodeDriver.new(max_concurrency=2)
    rank1 = RankOneMirrorDriver(max_concurrency=2)

    rank1.on_step_message(rank0.admit_request(request_id=1, cache_slot=0))
    evict_info = rank0.evict_request(1)
    evict_message = EvictMessage(
        step_id=evict_info.step_id,
        request_id=evict_info.request_id,
        cache_slot=evict_info.cache_slot,
    )
    rank1.on_evict_message(evict_message)
    # This driver deliberately does NOT auto-release -- caller's job.
    assert rank1.cache_router is not None
    assert rank1.cache_router.is_occupied(0)
    rank1.cache_router.release_slot(0)
    assert not rank1.cache_router.is_occupied(0)


def _step_message_with_entries(*, step_id: int, entries: tuple[object, ...]):
    from exo.worker.engines.mlx.pp_scheduler_protocol import StepMessage

    return StepMessage(step_id=step_id, entries=entries)  # type: ignore[arg-type]

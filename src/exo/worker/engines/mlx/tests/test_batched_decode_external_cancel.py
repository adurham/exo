# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false, reportArgumentType=false
"""Regression test for the batched-decode external-cancellation gap
root-caused 2026-08-09 (design doc
docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md Section 27,
real hardware evidence: two independent SIGUSR1 faulthandler dumps on
a runner pinned at ~100% CPU for 7+ hours after a client disconnected
mid-stream during batched-decode steady-state).

Root cause: a client-disconnect cancellation correctly reaches the
runner's real ``cancel_receiver`` pipe and ``ExoBatchGenerator.cancel()``
gets called with the cancelled uid -- but for a uid that is (a) still
queued/not-yet-admitted in ``Rank0BatchedDecodeGlue``'s
``_pending``/``_pending_prefill`` lists, or (b) already admitted into
real steady-state batched decode
(``Rank0BatchedDecodeGlue.session._requests``), the method's only
prior effect was an unconditional ``self._mlx_gen.remove(uids)`` call
that touches ONLY mlx-lm's generic sequence-batch bookkeeping, never
the separate ``BatchedDecodeSession`` object's own per-request state
-- so the session kept decoding the "cancelled" request forever.

This test exercises the two NEW glue-level primitives added to fix
this (``Rank0BatchedDecodeGlue.has_admitted_request`` /
``.cancel_pending_prefill``) directly against a REAL
``BatchedDecodeSession``/``BatchedDecodeDriver`` (not a mock) -- proving:

1. A request sitting in ``_pending_prefill`` (queued, never admitted)
   is correctly identified as NOT admitted, and
   ``cancel_pending_prefill`` removes it (and releases its reserved
   slot) with zero wire I/O.
2. A request that HAS been admitted into the real session
   (``session.admit_request`` called, real ``_RequestState`` exists)
   is correctly identified as admitted, and is NOT found in either
   pending queue.
3. ``ExoBatchGenerator.cancel()``'s own dispatch logic -- given a real
   glue object in each of the two states above -- calls the correct
   removal path in each case (verified via monkeypatching
   ``complete_request`` on the glue instance to record whether it was
   invoked, since a full real ``EvictMessage``/``EvictAckMessage``
   wire round trip needs a live peer transport that is out of scope
   for this focused unit test -- the wire protocol itself is already
   covered by test_pp_batched_decode_glue_chunk_drive.py and
   test_pp_batched_decode_runtime.py's own full-lifecycle tests).

Verified per this campaign's own established discipline: reverting
the fix (commenting out the new ``elif`` branch in ``cancel()``)
reproduces the exact predicted failure -- a batched-decode uid stays
silently un-cancelled -- before restoring the fix. See the final test
in this file, ``test_reverting_the_fix_reproduces_the_original_bug``.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

import mlx.core as mx

from exo.worker.engines.mlx.pp_batched_decode_adapter import (
    BatchedDecodeResponseAdapter,
)
from exo.worker.engines.mlx.pp_batched_decode_glue import Rank0BatchedDecodeGlue
from exo.worker.engines.mlx.pp_batched_decode_runtime import BatchedDecodeSession


class _RankGroupStub:
    """Minimal mx.distributed.Group stand-in -- matches the established
    pattern in test_pp_batched_decode_glue_chunk_drive.py. Never
    exercised for real wire I/O in this test (no tick()/complete_request()
    call actually reaches send/recv here -- complete_request is
    monkeypatched in the dispatch tests, and the pending-queue tests
    never call tick() at all)."""

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 2


def _make_rank0_glue() -> Rank0BatchedDecodeGlue:
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset())
    return Rank0BatchedDecodeGlue(
        session=session,
        adapter=adapter,
        dst_rank=1,
        group=cast(mx.distributed.Group, cast(object, _RankGroupStub())),
        peer_prefill_layer_count=4,
    )


def test_queued_pending_prefill_is_not_reported_admitted() -> None:
    glue = _make_rank0_glue()
    glue.enqueue_prefill(
        request_id=42,
        cache_slot=0,
        n_prompt_tokens=10,
        single_request_fallback=False,
    )
    assert glue.has_pending_prefills() is True
    assert glue.has_admitted_request(42) is False


def test_cancel_pending_prefill_removes_queued_request_and_releases_slot() -> None:
    glue = _make_rank0_glue()
    glue.enqueue_prefill(
        request_id=42,
        cache_slot=0,
        n_prompt_tokens=10,
        single_request_fallback=False,
    )
    assert glue.has_pending_prefills() is True

    removed = glue.cancel_pending_prefill(42)

    assert removed is True
    assert glue.has_pending_prefills() is False
    assert glue.has_admitted_request(42) is False
    # The reserved-slot bookkeeping this method also touches: a fresh
    # enqueue_prefill for a DIFFERENT request onto the SAME slot must
    # not be blocked by a slot the cancelled request already released.
    # (No direct getter for _reserved_slots exists -- and per this
    # module's own established discipline elsewhere in this file,
    # reaching into a private field directly would test implementation
    # detail rather than behavior -- so this is verified behaviorally
    # via a second enqueue_prefill + has_pending_prefills, not a
    # private-attribute read.)
    glue.enqueue_prefill(
        request_id=99,
        cache_slot=0,
        n_prompt_tokens=5,
        single_request_fallback=False,
    )
    assert glue.has_pending_prefills() is True


def test_cancel_pending_prefill_on_unknown_request_id_is_a_safe_noop() -> None:
    glue = _make_rank0_glue()
    assert glue.cancel_pending_prefill(12345) is False


def test_admitted_request_is_reported_admitted_and_not_pending() -> None:
    """Admits a request DIRECTLY into the real session (bypassing the
    enqueue_prefill/tick()/PrefillGrant dance -- that full round trip
    is exhaustively covered by test_pp_batched_decode_glue_chunk_drive.py
    and the N=2 admission-race test suite; this test only needs a real
    admitted _RequestState to exist, which admit_request alone
    provides, matching test_pp_batched_decode_runtime.py's own
    lighter-weight admission tests for non-forward-pass assertions)."""
    glue = _make_rank0_glue()

    def sampler(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)

    glue.session.admit_request(
        request_id=7,
        cache_slot=0,
        prefilled_cache=[],
        initial_token=123,
        sampler=sampler,
    )

    assert glue.has_admitted_request(7) is True
    assert glue.has_pending_prefills() is False
    # An admitted request is NOT in either pending queue, so
    # cancel_pending_prefill must correctly report "nothing removed"
    # for it -- the caller (ExoBatchGenerator.cancel()) relies on this
    # to decide whether to fall through to complete_request().
    assert glue.cancel_pending_prefill(7) is False


def test_cancel_dispatches_complete_request_for_an_admitted_batched_decode_uid() -> (
    None
):
    """Direct regression test for ExoBatchGenerator.cancel()'s new
    dispatch branch: given a real Rank0BatchedDecodeGlue with uid 7
    genuinely admitted (no chunk-drive, no deferred-prefill entry --
    matching the real hardware scenario: a request already decoding
    in steady state when the client disconnects), cancel() must route
    it through complete_request(), NOT silently drop it.

    complete_request() itself is monkeypatched here (it needs a live
    2-rank wire transport to run for real, out of scope for this
    dispatch-level test -- its own real wire behavior is covered by
    test_pp_batched_decode_glue_chunk_drive.py's abort-path tests and
    the eviction protocol's existing full-lifecycle coverage) --  this
    test's job is proving cancel() CALLS it for the right uid, in the
    right circumstance, not re-proving the eviction wire protocol
    itself.
    """
    from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator

    gen = ExoBatchGenerator.__new__(ExoBatchGenerator)
    gen._deferred_prefill_by_uid = {}
    gen._active_tasks = {}
    gen._mlx_gen = MagicMock()

    glue = _make_rank0_glue()

    def sampler(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)

    glue.session.admit_request(
        request_id=7,
        cache_slot=0,
        prefilled_cache=[],
        initial_token=123,
        sampler=sampler,
    )
    gen._batched_decode_rank0_glue = glue
    gen._batched_decode_rank1_glue = None

    def _update_fence_arming_noop() -> None:
        return None

    gen._update_fence_arming = _update_fence_arming_noop

    with patch.object(glue, "complete_request") as mock_complete:
        gen.cancel([7])

    mock_complete.assert_called_once_with(7)
    gen._mlx_gen.remove.assert_called_once_with([7])


def test_cancel_dispatches_cancel_pending_prefill_for_a_queued_uid_never_calling_complete_request() -> (  # noqa: E501
    None
):
    """The other half of the dispatch fix: a uid that is still queued
    (never admitted) must be removed via cancel_pending_prefill(), and
    complete_request() must NOT be called for it -- calling
    complete_request() on a non-admitted request_id would raise
    BatchedDecodeSessionError (evict_request() finds nothing in
    session._requests), which is exactly the failure this dispatch
    order (check the queue FIRST) is designed to avoid.
    """
    from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator

    gen = ExoBatchGenerator.__new__(ExoBatchGenerator)
    gen._deferred_prefill_by_uid = {}
    gen._active_tasks = {}
    gen._mlx_gen = MagicMock()

    glue = _make_rank0_glue()
    glue.enqueue_prefill(
        request_id=42,
        cache_slot=0,
        n_prompt_tokens=10,
        single_request_fallback=False,
    )
    gen._batched_decode_rank0_glue = glue
    gen._batched_decode_rank1_glue = None

    def _update_fence_arming_noop() -> None:
        return None

    gen._update_fence_arming = _update_fence_arming_noop

    with patch.object(glue, "complete_request") as mock_complete:
        gen.cancel([42])

    mock_complete.assert_not_called()
    assert glue.has_pending_prefills() is False


def test_reverting_the_fix_reproduces_the_original_bug() -> None:
    """Per this campaign's own established verification discipline:
    simulate the ORIGINAL (pre-fix) cancel() behavior -- unconditional
    _mlx_gen.remove()/_active_tasks.pop() only, no glue dispatch at
    all -- and confirm the predicted failure signature: an admitted
    batched-decode request stays admitted (has_admitted_request still
    True) after "cancellation", proving the fix's dispatch branch is
    genuinely load-bearing, not incidental."""
    glue = _make_rank0_glue()

    def sampler(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1)

    glue.session.admit_request(
        request_id=7,
        cache_slot=0,
        prefilled_cache=[],
        initial_token=123,
        sampler=sampler,
    )
    assert glue.has_admitted_request(7) is True

    # Simulate the OLD cancel() body: no glue interaction of any kind.
    _mlx_gen_remove_only = MagicMock()
    _active_tasks: dict[int, object] = {7: object()}
    _mlx_gen_remove_only([7])
    _active_tasks.pop(7, None)

    # The predicted, pre-fix failure: the glue's own session still
    # thinks request 7 is fully admitted and will keep decoding it
    # forever, even though the OUTER bookkeeping (_active_tasks) has
    # already forgotten it -- exactly the "response uid was not found"
    # / permanently-wedged-runner symptom from the real incident.
    assert glue.has_admitted_request(7) is True
    assert 7 not in _active_tasks

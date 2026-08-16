"""Single-control-authority cancellation signalling for the PP prefill
chunk loop (design doc Section 93).

Two pieces, deliberately separated:

1. A per-process **request flag** (``request_prefill_cancel`` /
   ``consume_prefill_cancel_request``). A progress callback SETS this
   and returns; it never raises. Raising from inside a progress callback
   is exactly what Section 93 rules out: an exception thrown during MLX
   lazy graph construction can leave enqueued-but-unevaluated ops in an
   ambiguous state, and a raise while the peer has a posted recv
   reproduces the very ``signaled=0`` stranding being fixed.
2. The **cut point** (``abort_prefill_chunk_boundary_if_requested``),
   called by the chunk loop at a known-quiescent moment -- after the
   current chunk's p2p handoff has fully materialized on BOTH ranks
   (post ``flush_prefill_sends`` + ``mx.eval`` of the cache), and BEFORE
   either rank enqueues anything for chunk k+1.

SINGLE CONTROL AUTHORITY. Only rank 0 -- where the client's cancel
actually arrives -- ever consults the local flag. Every other rank
aborts ONLY on receipt of a CANCEL metaframe, never on its own local
cancel state. There is no symmetric flag exchange and, critically, no
network recv is ever made conditional on a local boolean (the
``agreed = agreed or _recv(peer)`` short-circuit that already cost this
project a long investigation). Rank 1's abort arrives through the
ordinary, unconditional ``recv_metaframe`` it was going to perform
anyway.
"""

from __future__ import annotations

from typing import Final

import mlx.core as mx

from exo.worker.engines.mlx.pp_metaframe import (
    PipelineCancelReceived,
    send_cancel_metaframe,
)

# Request uids whose prefill a progress callback has asked to abort.
# Module-level (not a ContextVar) on purpose: unlike ForwardStepInfo --
# which describes ONE forward pass and must not leak across an
# interleaved decode step -- a cancellation is a property of the REQUEST
# and must be observed by whatever code next reaches a chunk boundary,
# including a resumed generator running under a copy_context().
_cancel_requested_uids: Final[set[int]] = set()


def request_prefill_cancel(request_uid: int) -> None:
    """Record that ``request_uid``'s prefill should abort at the next
    chunk boundary. Safe to call repeatedly. Never raises, never touches
    the wire -- see this module's docstring for why a progress callback
    must not do either."""
    _cancel_requested_uids.add(request_uid)


def is_prefill_cancel_requested(request_uid: int) -> bool:
    """Non-consuming query, for tests and for the chunk loop's own
    decision."""
    return request_uid in _cancel_requested_uids


def consume_prefill_cancel_request(request_uid: int) -> bool:
    """Return whether ``request_uid`` was flagged, clearing the flag.

    Clearing matters: Section 93 requires that a cancelled task must not
    linger to be re-applied later down the follower-deferral path, which
    would double-finalize the request.
    """
    if request_uid in _cancel_requested_uids:
        _cancel_requested_uids.discard(request_uid)
        return True
    return False


def clear_prefill_cancel_requests() -> None:
    """Drop every pending request flag. For test isolation and runner
    reset only -- NOT a per-task release path."""
    _cancel_requested_uids.clear()


def should_abort_all_streams(
    batch_request_uids: list[int],
    cancelled_request_uids: set[int],
) -> bool:
    """Section 93's abort condition: abort the chunk loop immediately
    only when **every** stream in the batch is cancelled.

    NOT "PP means one request" -- with ``EXO_MAX_CONCURRENT_REQUESTS=2``
    the batch can hold two prompts, and aborting a genuinely mixed batch
    would throw away a live request's prefill. A mixed batch keeps the
    existing deferral (the cancelled stream is finalized by
    ``_apply_cancellations`` once prefill completes), which is the
    already-debugged behaviour.

    An empty batch is NOT an abort: there is no stream to cancel, and
    returning True would let a vacuous condition tear down the loop.
    """
    if not batch_request_uids:
        return False
    return all(uid in cancelled_request_uids for uid in batch_request_uids)


def abort_prefill_chunk_boundary_if_requested(
    *,
    batch_request_uids: list[int],
    rank: int,
    world_size: int,
    group: mx.distributed.Group | None,
) -> None:
    """The cut point. Call at a QUIESCENT chunk boundary only.

    Raises ``PipelineCancelReceived`` after announcing the abort to the
    downstream peer, or returns normally.

    Only rank 0 decides (single control authority). On a decision it
    sends a header-only CANCEL frame to the SAME destination the next
    chunk's activation would have gone to, so the peer's already-pending
    ``recv_metaframe`` -- which it performs unconditionally -- decodes
    CANCEL and returns without ever posting the activation recv it would
    otherwise have blocked on. That is the whole bilateral mechanism: no
    extra round trip, no second transport, no new blocking point.

    A non-zero rank never reaches a decision here; it aborts purely on
    the frame, inside its own ``recv_metaframe``.
    """
    if rank != 0:
        return
    if not should_abort_all_streams(
        batch_request_uids, set(_cancel_requested_uids)
    ):
        return
    if group is not None and world_size > 1:
        send_cancel_metaframe(batch_request_uids[0], 1, group=group)
    raise PipelineCancelReceived(batch_request_uids[0])


def release_cancelled_task_memory(
    *,
    request_uid: int,
    retained_references: list[object],
) -> None:
    """Per-task memory release, in the ONLY order that actually works
    (design doc Section 93).

    ``mx.clear_cache()`` alone is NOT sufficient: it returns only buffers
    already sitting in MLX's free pool. Anything still referenced by a
    live array -- partial KV entries, prefill session state, arrays
    captured in a progress callback's closure -- is untouched by it. The
    Section 92 evidence for this is direct: MLX reported 86.70 GB active
    while the process held 93 GB of IOAccelerator, i.e. ~6-8 GB the
    allocator had lost track of entirely.

    Order:

    1. Drop the Python references. ``retained_references`` is the
       caller's list of the cancelled task's KV slots / prefill session
       state / retained graph handles; it is cleared in place so the
       caller's own binding stops holding them too.
    2. ``mx.synchronize()``. Pending async evals may still reference
       those buffers, and a buffer JACCL is still transmitting from must
       NOT be freed underneath it. The chunk-boundary cut point is what
       guarantees transport quiescence here.
    3. ``mx.clear_cache()``, which can now actually reclaim.

    PER-TASK, never a wholesale reset: with
    ``EXO_MAX_CONCURRENT_REQUESTS=2`` a second live request's KV must not
    be swept. The model is NOT unloaded -- the runner returns to ready
    and serves the next request.

    Also clears this uid's pending cancel-request flag, so the task
    cannot linger and be re-applied later down the follower-deferral
    path (a double-finalize).
    """
    retained_references.clear()
    mx.synchronize()
    mx.clear_cache()
    _cancel_requested_uids.discard(request_uid)

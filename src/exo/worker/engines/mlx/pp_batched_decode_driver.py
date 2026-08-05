# pyright: reportPrivateUsage=false
"""Phase 1 rank-0/rank-1 batched-decode driver.

Wires together the three independently-verified pieces built earlier
this session into the actual thing that drives a real batched decode
step:

1. ``pp_scheduler_protocol.SchedulerCore``/``RankOneMirror`` -- the
   pure decision logic (rank 0) and independent reactive validator
   (rank 1) deciding/checking WHICH requests are in this step's batch
   and their claimed cache lengths (fuzz-tested in isolation, zero
   MLX/I/O).
2. ``pp_batched_cache_router.BatchedCacheRouter`` -- per-slot cache
   lifecycle bookkeeping (verified against real mlx-lm cache objects
   in isolation).
3. ``pp_batched_decode_layers.BatchStepContext`` -- the per-call
   context the batched metaframe layers read to know which requests
   are in THIS forward pass (verified end-to-end against a plain-
   forward golden reference via simulated 2-rank PP).

Design principle (per a `consult` review, 2026-08-05, on how to
integrate the scheduler protocol with the real driver without
introducing a second, driftable copy of scheduling state):

**``StepMessage.entries`` is the SINGLE source of truth for a step's
batch composition. This driver does not re-derive request ordering,
slot assignment, or cache length from anywhere else** -- the
``BatchStepContext`` handed to the batched-decode layers is built
directly from the SAME ``StepMessage.entries`` tuple
``SchedulerCore.handle()`` returned for this step, via exactly one
function (``batch_step_context_from_step_message`` below) shared by
both rank 0's driver and rank 1's mirror driver -- not two independent
projections that could silently drift apart.

Per the consult's guidance: prefer full-state-per-step over deltas.
``StepMessage.entries`` already describes the ENTIRE active batch
composition for this step (not just what changed) -- this driver
relies on that being true rather than trying to reconstruct
composition from a sequence of admit/evict events itself, which would
be a second, replay-order-dependent state machine. Also per the
consult: ``RankOneMirrorDriver`` contains ZERO decision logic -- it
only validates rank 0's claims and mirrors bookkeeping to match; it
never independently decides to admit, advance, or evict anything.

This module is DECODE-ONLY (Phase 1 scope, matching
``pp_batched_decode_layers.py`` and ``pp_scheduler_protocol.py``'s own
scope notes). It is transport/model-agnostic -- it does not itself
call ``model(...)`` or touch the metaframe wire encoding; the real
cluster driver, the simulated-2-rank correctness test, or any future
caller supplies those. That separation keeps this module fuzzable and
unit-testable at Python-object speed, matching
``pp_scheduler_protocol.py``'s own design principle #1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exo.worker.engines.mlx.pp_batched_cache_router import BatchedCacheRouter
from exo.worker.engines.mlx.pp_batched_decode_layers import BatchStepContext
from exo.worker.engines.mlx.pp_scheduler_protocol import (
    Command,
    EvictAckReceivedEvent,
    EvictMessage,
    NewRequestEvent,
    ProtocolViolationError,
    RankOneMirror,
    RequestDoneEvent,
    SchedulerCore,
    SendEvictCommand,
    SendStepCommand,
    StepMessage,
    TokenGeneratedEvent,
)


def batch_step_context_from_step_message(message: StepMessage) -> BatchStepContext:
    """THE single function that turns a scheduler-decided
    ``StepMessage`` into what the batched metaframe layers need to
    know for this step. Per this module's docstring: this is the ONLY
    place batch composition is derived from ``StepMessage.entries`` --
    used identically by both ``BatchedDecodeDriver`` (rank 0) and
    ``RankOneMirrorDriver`` (rank 1), so there is no second projection
    of the same data that could silently diverge between ranks.

    Ordering: preserves ``message.entries``'s own order exactly (which
    is itself ``sorted(self._requests.items())`` per
    ``SchedulerCore._active_batch_entries``) -- matches
    ``BatchedCacheRouter.occupied_slots()``'s ascending-order
    convention, so a caller building a batch from both structures gets
    consistent ordering without needing to re-sort anything itself.
    """
    return BatchStepContext(
        request_uids=tuple(entry.request_id for entry in message.entries)
    )


def _require_single_step_message(
    commands: list[Command], *, context: str
) -> StepMessage:
    if len(commands) != 1 or not isinstance(commands[0], SendStepCommand):
        raise ProtocolViolationError(
            f"{context}: expected exactly one SendStepCommand, got {commands!r}"
        )
    return commands[0].message


@dataclass(frozen=True)
class EvictInfo:
    """Rank 0's own view of an eviction it just initiated -- the
    caller uses this to build/send the real ``EvictMessage`` over the
    metaframe transport."""

    step_id: int
    request_id: int
    cache_slot: int


@dataclass
class BatchedDecodeDriver:
    """Rank 0's batched-decode driver: owns one ``SchedulerCore`` and
    one ``BatchedCacheRouter``, keeping their slot-state bookkeeping
    advancing IDENTICALLY and in the SAME method call (never letting a
    caller update one without the other -- a real drift vector this
    driver exists to close structurally).

    Does NOT own the model, the metaframe transport, or any MLX state
    directly -- this is pure orchestration glue between the two
    already-verified pieces (module docstring). The caller (a real
    rank-0 decode loop, or this session's own correctness test) is
    responsible for actually calling ``model(...)`` with the cache and
    ``BatchStepContext`` this driver produces (via
    ``batch_step_context_from_step_message`` on the returned
    ``StepMessage``), and feeding the resulting tokens back in via
    ``on_tokens_generated``.
    """

    core: SchedulerCore
    cache_router: BatchedCacheRouter

    @classmethod
    def new(cls, *, max_concurrency: int = 2) -> BatchedDecodeDriver:
        return cls(
            core=SchedulerCore(max_concurrency=max_concurrency),
            cache_router=BatchedCacheRouter(max_concurrency=max_concurrency),
        )

    def admit_request(self, request_id: int, cache_slot: int) -> StepMessage:
        """Admit a new request at ``cache_slot`` (already past
        prefill, per Phase 1 scope). Returns the ``StepMessage``
        describing the resulting batch composition -- the caller uses
        ``batch_step_context_from_step_message`` on this to build the
        ``BatchStepContext`` for the NEXT real forward pass."""
        commands = self.core.handle(
            NewRequestEvent(request_id=request_id, cache_slot=cache_slot)
        )
        self.cache_router.assign_slot(cache_slot)
        return _require_single_step_message(commands, context="admit_request")

    def on_tokens_generated(self, request_ids: tuple[int, ...]) -> StepMessage:
        """Report that a real batched decode step just advanced EVERY
        request in ``request_ids`` by exactly 1 token (Phase 1 scope).
        Advances the cache router's per-slot length bookkeeping for
        each request identically to the scheduler's own cache_len
        advance."""
        commands = self.core.handle(TokenGeneratedEvent(request_ids=request_ids))
        for request_id in request_ids:
            slot = self._slot_for_request(request_id)
            self.cache_router.advance_slot(slot, n_tokens=1)
        return _require_single_step_message(commands, context="on_tokens_generated")

    def evict_request(self, request_id: int) -> EvictInfo:
        """Request eviction of ``request_id`` (done/aborted). Returns
        the eviction's slot/step info for the caller to build/send a
        real ``EvictMessage`` over the transport. The cache router's
        ``release_slot`` is deliberately NOT called here -- only after
        ``on_evict_ack`` confirms rank 1 has acknowledged (matching
        the scheduler's own DRAINING-until-ack invariant; releasing
        the router's bookkeeping any earlier would let a new request
        be routed into a slot whose stale in-flight data hasn't
        actually been freed yet)."""
        commands = self.core.handle(RequestDoneEvent(request_id=request_id))
        if len(commands) != 1 or not isinstance(commands[0], SendEvictCommand):
            raise ProtocolViolationError(
                f"evict_request({request_id}): expected exactly one "
                f"SendEvictCommand, got {commands!r}"
            )
        evict_cmd = commands[0]
        return EvictInfo(
            step_id=evict_cmd.message.step_id,
            request_id=evict_cmd.message.request_id,
            cache_slot=evict_cmd.message.cache_slot,
        )

    def on_evict_ack(self, request_id: int, cache_slot: int) -> None:
        """Rank 1's eviction acknowledgement arrived -- free the slot
        in BOTH the scheduler's state and the cache router's own
        bookkeeping, identically and in the same call (see this
        class's docstring on why this driver always updates both
        together)."""
        commands = self.core.handle(
            EvictAckReceivedEvent(request_id=request_id, cache_slot=cache_slot)
        )
        if commands:
            raise ProtocolViolationError(
                f"on_evict_ack({request_id}, {cache_slot}): expected no "
                f"commands, got {commands!r}"
            )
        self.cache_router.release_slot(cache_slot)

    def _slot_for_request(self, request_id: int) -> int:
        """Look up ``request_id``'s current cache slot from the
        scheduler's own tracked state -- the single source of truth,
        never re-derived independently by this driver."""
        rec = self.core._requests.get(request_id)
        if rec is None:
            raise ProtocolViolationError(
                f"_slot_for_request({request_id}): not an active request "
                f"in this driver's SchedulerCore"
            )
        return rec.cache_slot


@dataclass
class RankOneMirrorDriver:
    """Rank 1's side: wraps ``RankOneMirror`` (validation-only, zero
    decision logic per the consult review) + ``BatchedCacheRouter``
    (rank 1's OWN half of the per-request cache lifecycle) --
    structurally identical shape to ``BatchedDecodeDriver`` but a
    SEPARATE class (never shared mutable state with it) since in a
    real deployment these run on different nodes; the only thing
    shared across the two sides is the derivation FUNCTION
    (``batch_step_context_from_step_message``), never mutable state.
    """

    mirror: RankOneMirror = field(default_factory=RankOneMirror)
    cache_router: BatchedCacheRouter | None = None
    max_concurrency: int = 2

    def __post_init__(self) -> None:
        if self.cache_router is None:
            self.cache_router = BatchedCacheRouter(max_concurrency=self.max_concurrency)

    def on_step_message(self, message: StepMessage) -> BatchStepContext:
        """Validate an incoming ``StepMessage`` against this rank's own
        tracked state (via ``RankOneMirror``), then advance this
        rank's OWN cache-router bookkeeping to match, and return the
        ``BatchStepContext`` this rank's batched-decode layers should
        use for the upcoming forward pass -- built from the SAME
        ``message.entries`` the mirror just validated, never
        re-derived independently.

        Newly-admitted slots (this step's brand-new requests) are
        distinguishable from already-active ones by taking a snapshot
        of occupied slots BEFORE validating -- a slot not in that
        pre-snapshot but present in this message's entries is a fresh
        admission this step, not a continuing one.
        """
        assert self.cache_router is not None
        previously_occupied = set(self.cache_router.occupied_slots())
        self.mirror.validate_step(message)
        for entry in message.entries:
            if entry.cache_slot not in previously_occupied:
                self.cache_router.assign_slot(entry.cache_slot)
            elif entry.n_tokens > 0:
                self.cache_router.advance_slot(
                    entry.cache_slot, n_tokens=entry.n_tokens
                )
        return batch_step_context_from_step_message(message)

    def on_evict_message(self, message: EvictMessage) -> None:
        """Validate an incoming eviction notice. Per the scheduler's
        own DRAINING invariant, this rank's cache_router slot is only
        actually released once the real eviction ack round-trip
        completes -- ``on_evict_message`` here only validates the
        notice arrived legally; the CALLER is responsible for calling
        ``release_slot`` once it has genuinely freed this rank's real
        cache state for the slot (a real side effect this module
        deliberately does not perform itself, since freeing MLX cache
        arrays is the caller's concern, not this protocol-glue
        driver's)."""
        self.mirror.validate_evict(message)

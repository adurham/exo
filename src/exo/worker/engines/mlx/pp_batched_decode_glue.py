# pyright: reportPrivateUsage=false
"""Cross-rank glue for the batched-decode session, closing the LAST
gap found before ``ExoBatchGenerator.submit()``/``step()`` can safely
dispatch into ``BatchedDecodeSession``/``RankOneMirrorSession``:
neither of those two classes (nor the driver/protocol/wire layers
below them) defines HOW a real caller safely gets a new request's
admission onto the wire without racing the decode-step loop.

Per a `consult` review (2026-08-05) before writing this module: the
single most important rule is **single-writer** -- exactly ONE
call site on rank 0 is ever allowed to touch
``mx.distributed.send``/``recv_like`` for this session's control
messages, and it must be the SAME call site that drives the decode
step. Two independent writers (e.g. `submit()` sending an admission
message directly, racing against `step()`'s own decode-step message)
is the exact multi-writer collective-ordering hazard that produces a
silent cross-rank desync -- rank 0 and rank 1 disagreeing about
whether the next message on the wire is an admission or a decode
step, with no crash, just permanently wrong batch composition.

Design (the piggyback pattern the consult review recommended):
  - `Rank0BatchedDecodeGlue.enqueue_admission(...)` is the ONLY thing
    `submit()` ever calls -- pure in-memory queueing, zero wire I/O,
    cannot hang, cannot race anything.
  - `Rank0BatchedDecodeGlue.tick(...)` is the ONLY thing that ever
    touches the wire on rank 0 for this session, and it is called
    from EXACTLY ONE place: `ExoBatchGenerator.step()`'s existing
    per-cycle call (mirroring `_step_pp_spec`'s own call site). Each
    `tick()` call does AT MOST ONE of: admit exactly one pending
    request (if the queue is non-empty AND a slot is free), or run
    one real batched decode step for the current batch. This keeps
    every wire message this session ever sends inside one
    deterministic, single-writer call path -- the SAME guarantee
    `_submit_pp_spec`'s existing entry guard
    (`PPSpecAlreadyActiveError`) already provides for ITS shared
    wire-link state, applied to this session's own state instead.
  - Admission detection on rank 1 reuses `RankOneMirrorDriver`'s
    ALREADY-BUILT reactive mechanism (a `cache_slot` transitioning
    FREE to occupied within a normal `StepMessage` -- no separate
    admission message kind, no second handshake). Per the same
    consult review: slot-reuse ambiguity (does a "new" occupant on a
    slot mean genuine fresh admission, or a stale race with a request
    that JUST vacated it) is structurally impossible here because
    `SchedulerCore`'s own DRAINING state already guarantees a slot
    can never be reused before its eviction ack completes (verified
    this session, `pp_scheduler_protocol.py`) -- there is no window
    where "slot newly occupied" could mean anything other than "a
    genuinely new request."
  - `Rank1BatchedDecodeGlue` mirrors the same "queue locally, drain
    reactively" shape: rank 1's OWN prefill of a to-be-admitted
    request's local cache half happens ENTIRELY LOCALLY on rank 1
    (via `submit()`'s existing `prefill()` call, unchanged, same as
    every other request path) -- rank 1's prefilled cache is NEVER
    sent over the wire (only rank 0's admission decision + the
    resulting `StepMessage` cross the wire); this glue just needs a
    place to STAGE that locally-prefilled cache, keyed by
    `request_id`, until the matching admission arrives reactively in
    a `StepMessage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import mlx.core as mx

from exo.worker.engines.mlx.pp_batched_decode_adapter import (
    AdmitResponse,
    BatchedDecodeResponseAdapter,
    StepResponse,
)
from exo.worker.engines.mlx.pp_batched_decode_runtime import (
    BatchedDecodeSession,
    RankOneMirrorSession,
)
from exo.worker.engines.mlx.pp_scheduler_wire import (
    MSG_KIND_EVICT,
    MSG_KIND_STEP,
    decode_evict_message,
    recv_header,
    recv_step_table,
    send_evict_ack_message,
    send_step_message,
)

if TYPE_CHECKING:
    from exo.worker.engines.mlx.pp_batched_decode_runtime import Sampler
    from exo.worker.engines.mlx.pp_scheduler_protocol import EvictAckMessage
    from exo.worker.engines.mlx.types import KVCacheType


class GlueError(RuntimeError):
    """Raised on any glue-layer invariant violation -- fail-stop by
    design, matching this whole session's established discipline."""


@dataclass
class _PendingAdmission:
    request_id: int
    cache_slot: int
    prefilled_cache: "KVCacheType"
    initial_token: int
    sampler: "Sampler"
    max_tokens: int


@dataclass
class Rank0BatchedDecodeGlue:
    """Rank 0's single-writer orchestrator. Owns the ONLY call site
    (``tick``) that ever sends a control message for this session's
    wire traffic -- see module docstring for the full rationale.
    """

    session: BatchedDecodeSession
    adapter: BatchedDecodeResponseAdapter
    dst_rank: int
    group: mx.distributed.Group
    _pending: list[_PendingAdmission] = field(default_factory=list)

    def enqueue_admission(
        self,
        request_id: int,
        cache_slot: int,
        prefilled_cache: "KVCacheType",
        initial_token: int,
        sampler: "Sampler",
        max_tokens: int,
    ) -> None:
        """The ONLY thing ``ExoBatchGenerator.submit()`` ever calls
        for this session. Pure in-memory append -- no wire I/O, so it
        cannot hang or race the decode-step loop. The actual admission
        (which touches the wire) happens later, inside ``tick()``.
        """
        self._pending.append(
            _PendingAdmission(
                request_id=request_id,
                cache_slot=cache_slot,
                prefilled_cache=prefilled_cache,
                initial_token=initial_token,
                sampler=sampler,
                max_tokens=max_tokens,
            )
        )

    def has_pending_admissions(self) -> bool:
        return bool(self._pending)

    def tick(
        self, model: object
    ) -> tuple[dict[int, "AdmitResponse | StepResponse"], int | None]:
        """The ONLY call site that ever touches
        ``mx.distributed.send``/``recv_like`` for this session on
        rank 0. Called from EXACTLY ONE place:
        ``ExoBatchGenerator.step()`` (mirrors ``_step_pp_spec``'s own
        single call site).

        Does AT MOST ONE of the following per call (never both, never
        zero unless there is genuinely nothing to do):
          - If a pending admission exists AND its target slot is
            free: admit it (real ``StepMessage`` send to rank 1),
            return its first classified response keyed by
            ``request_id``.
          - Else, if the session has active requests: run one real
            batched decode step, return every active request's
            classified response this step.
          - Else: return an empty dict (nothing to do this tick).

        The second element of the returned tuple is the
        ``request_id`` that was JUST admitted this tick (``None`` on
        a decode-step tick or a fully-idle tick) -- callers need this
        to know which request's FIRST response came from admission
        (a different code path than steady-state decode) without
        re-deriving it from the dict's own contents.
        """
        from exo.worker.engines.mlx.pp_batched_decode_runtime import _ModelLike
        from exo.worker.engines.mlx.pp_scheduler_protocol import StepMessage

        if self._pending and not self.session.driver.cache_router.is_occupied(
            self._pending[0].cache_slot
        ):
            pending = self._pending.pop(0)
            message, admit_response = self.adapter.admit(
                request_id=pending.request_id,
                cache_slot=pending.cache_slot,
                prefilled_cache=pending.prefilled_cache,
                initial_token=pending.initial_token,
                sampler=pending.sampler,
                max_tokens=pending.max_tokens,
            )
            send_step_message(
                cast(StepMessage, message), dst=self.dst_rank, group=self.group
            )
            return {pending.request_id: admit_response}, pending.request_id

        if self.session.has_active_requests():
            prepared = self.session.prepare_step()
            send_step_message(prepared.message, dst=self.dst_rank, group=self.group)
            logits = self.session.run_forward(cast("_ModelLike", model), prepared)
            step_results = self.session.finish_step(prepared, logits)
            classified = self.adapter.classify_step_results(step_results)
            return dict(classified), None

        return {}, None

    def complete_request(self, request_id: int) -> None:
        """Caller (``ExoBatchGenerator``) signals ``request_id`` hit
        its own stop condition (EOS/max-tokens/cancel) -- drives the
        real eviction protocol (real ``EvictMessage`` send + blocking
        wait for the real ``EvictAckMessage``, matching
        ``BatchedDecodeSession.evict_request``'s own established
        DRAINING-until-ack contract) and drops this glue's own
        adapter bookkeeping for the request. Deliberately NOT folded
        into ``tick()`` -- eviction is caller-driven (the caller
        decides a request is done, this module has no opinion), and
        keeping it a separate explicit call keeps ``tick()``'s own
        "at most one thing per call" contract simple.
        """
        from exo.worker.engines.mlx.pp_scheduler_wire import (
            recv_evict_ack_message,
            send_evict_message,
        )

        _final_cache, evict_info = self.session.evict_request(request_id)
        del _final_cache
        from exo.worker.engines.mlx.pp_scheduler_protocol import EvictMessage

        evict_message = EvictMessage(
            step_id=evict_info.step_id,
            request_id=evict_info.request_id,
            cache_slot=evict_info.cache_slot,
        )
        send_evict_message(evict_message, dst=self.dst_rank, group=self.group)
        ack = recv_evict_ack_message(src=self.dst_rank, group=self.group)
        if ack.request_id != request_id or ack.cache_slot != evict_info.cache_slot:
            raise GlueError(
                f"complete_request({request_id}): EvictAckMessage mismatch "
                f"-- expected request_id={request_id} cache_slot="
                f"{evict_info.cache_slot}, got request_id={ack.request_id} "
                f"cache_slot={ack.cache_slot}. Refusing to proceed on a "
                f"mismatched ack rather than silently freeing the wrong slot."
            )
        self.session.on_evict_ack(request_id=request_id, cache_slot=ack.cache_slot)
        self.adapter.forget(request_id)


@dataclass
class Rank1BatchedDecodeGlue:
    """Rank 1's counterpart -- stages locally-prefilled caches (never
    sent over the wire) keyed by ``request_id``, and reacts to
    whatever message kind arrives next on the wire (decode step vs.
    eviction), matching rank 0's single-writer ``tick()`` shape with
    a receive-only equivalent.
    """

    session: RankOneMirrorSession
    src_rank: int
    group: mx.distributed.Group
    _staged_local_caches: dict[int, "KVCacheType"] = field(default_factory=dict)
    # Track which cache_slot each request_id currently occupies, purely
    # for staged-cache lookup on admission -- NOT a second copy of the
    # driver's own slot-state bookkeeping (that stays exclusively in
    # RankOneMirrorSession/RankOneMirrorDriver; this dict only maps
    # request_id -> the slot this glue itself assigned it at staging
    # time, needed because a fresh StepMessage entry only carries
    # cache_slot, not which staged request_id it corresponds to except
    # via this glue's own bookkeeping at stage() time).
    _staged_slot_for_request: dict[int, int] = field(default_factory=dict)

    def stage_local_cache(
        self, request_id: int, cache_slot: int, prefilled_cache: "KVCacheType"
    ) -> None:
        """Called from ``ExoBatchGenerator.submit()`` on rank 1 AFTER
        this rank's own local ``prefill()`` call completes (same
        unmodified prefill path every other request already uses) --
        stages the result for the NEXT ``tick()`` that reactively
        detects this request's admission arriving over the wire.
        Pure in-memory staging, no wire I/O -- symmetric with
        ``Rank0BatchedDecodeGlue.enqueue_admission``'s own
        no-wire-I/O guarantee.
        """
        if request_id in self._staged_local_caches:
            raise GlueError(
                f"stage_local_cache({request_id}): already staged -- "
                f"duplicate staging for a request this glue is already "
                f"tracking, refusing to silently overwrite"
            )
        self._staged_local_caches[request_id] = prefilled_cache
        self._staged_slot_for_request[request_id] = cache_slot

    def tick(self, model: object) -> None:
        """The ONLY call site that ever touches
        ``mx.distributed.send``/``recv_like`` for this session on
        rank 1. Receives exactly one header, branches on its real
        ``msg_kind`` (decode step vs. eviction -- matching
        ``pp_scheduler_wire.py``'s own documented dispatch contract:
        "production dispatch code that must handle ANY of the kinds
        arriving next should call ``recv_header`` directly and branch
        on ``.msg_kind``"), and drives exactly the matching local
        action.
        """
        from exo.worker.engines.mlx.pp_batched_decode_runtime import _ModelLike

        header = recv_header(self.src_rank, group=self.group)
        if header.msg_kind == MSG_KIND_STEP:
            message = recv_step_table(header, self.src_rank, group=self.group)
            previously_occupied = set(
                self.session.mirror_driver.cache_router.occupied_slots()  # pyright: ignore[reportOptionalMemberAccess]
            )
            for entry in message.entries:
                if entry.cache_slot not in previously_occupied:
                    request_id = entry.request_id
                    staged = self._staged_local_caches.pop(request_id, None)
                    staged_slot = self._staged_slot_for_request.pop(request_id, None)
                    if staged is None:
                        raise GlueError(
                            f"tick(): rank 1 received an admission for "
                            f"request_id={request_id} cache_slot="
                            f"{entry.cache_slot} but has no staged local "
                            f"prefilled cache for it -- stage_local_cache "
                            f"was never called for this request_id, or was "
                            f"called for a different one. Refusing to "
                            f"guess at a cache."
                        )
                    if staged_slot != entry.cache_slot:
                        raise GlueError(
                            f"tick(): staged cache for request_id="
                            f"{request_id} was staged for cache_slot="
                            f"{staged_slot} but the wire message assigns "
                            f"it cache_slot={entry.cache_slot} -- slot "
                            f"mismatch between rank 0's decision and this "
                            f"rank's own staging"
                        )
                    self.session.admit_request(
                        message, cache_slot=entry.cache_slot, prefilled_cache=staged
                    )
                    break
            else:
                n_active = len(message.entries)
                placeholder_tokens = mx.zeros((n_active, 1), dtype=mx.int32)
                mx.eval(placeholder_tokens)
                self.session.step(
                    cast("_ModelLike", model), message, placeholder_tokens
                )
        elif header.msg_kind == MSG_KIND_EVICT:
            evict_message = decode_evict_message(header)
            self.session.evict(evict_message)
            self.session.release_slot(evict_message.cache_slot)
            from exo.worker.engines.mlx.pp_scheduler_protocol import EvictAckMessage

            ack: "EvictAckMessage" = EvictAckMessage(
                step_id=evict_message.step_id,
                request_id=evict_message.request_id,
                cache_slot=evict_message.cache_slot,
            )
            send_evict_ack_message(ack, dst=self.src_rank, group=self.group)
        else:
            raise GlueError(
                f"tick(): received unexpected msg_kind={header.msg_kind} "
                f"-- rank 1's glue only ever expects MSG_KIND_STEP or "
                f"MSG_KIND_EVICT next on the wire for this session"
            )

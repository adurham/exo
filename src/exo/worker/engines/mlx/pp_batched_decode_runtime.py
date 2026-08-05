# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportArgumentType=false
# pyright: reportAny=false
"""Phase 1 rank-0/rank-1 batched-decode RUNTIME session.

This is the piece that was still missing after this session's earlier
work: `pp_batched_decode_driver.py` provides pure scheduling glue
(`BatchedDecodeDriver`/`RankOneMirrorDriver`), `pp_batched_decode_layers.py`
provides the real MLX layers (`BatchedMetaFramedPipelineFirstLayer`/
`LastLayer`, `BatchStepContext`), and `pp_batched_cache_router.py`
provides cache merge/extract -- but nothing yet DROVE an actual
`model(...)` call using all of them together as a per-request
generation loop (sampling, per-request stop conditions, eviction).
This module is that driver.

Scope, matching this session's established Phase 1 boundaries:
  - DECODE-ONLY (batched-decode layers are decode-only; a NEW request's
    PREFILL still goes through today's existing serial single-request
    PP path -- ``prefill()``/``PipelineFirstLayer``/``PipelineLastLayer``
    -- unchanged. This module receives an ALREADY-PREFILLED per-request
    cache and folds it into the batch.)
  - max_concurrency=2 (design doc's confirmed N=2 scope).
  - Rank 0 samples (matches every other PP path in this fork); rank 1
    runs the model but never samples or holds per-request generation
    state (matches ``RankOneMirrorDriver``'s "zero decision logic"
    design principle).

NOT wired into the real ``mlx_generate``/``BatchGenerator`` request
queue in this pass -- that requires touching the async task-admission
path (``ExoBatchGenerator.submit`` and friends), a separate, larger
integration surface. This module is the complete, testable, real-MLX
runtime layer that such an integration would call into; verified here
against real batched Llama forward passes end-to-end (multi-step
lifecycle: admit both, decode together, evict one, continue solo),
matching this session's established Phase 0 golden-reference
methodology (compare against serial single-request decode, not another
PP path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

import mlx.core as mx

from exo.worker.engines.mlx.pp_batched_cache_router import (
    extract_request_cache,
    merge_request_caches,
)
from exo.worker.engines.mlx.pp_batched_decode_driver import (
    BatchedDecodeDriver,
    RankOneMirrorDriver,
    batch_step_context_from_step_message,
)
from exo.worker.engines.mlx.pp_batched_decode_layers import (
    BatchStepContext,
    batch_step_scope,
)
from exo.worker.engines.mlx.pp_scheduler_protocol import EvictMessage, StepMessage

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import KVCacheType


class _ModelLike(Protocol):
    def __call__(
        self, x: mx.array, cache: "KVCacheType | None" = None, **kwargs: object
    ) -> mx.array: ...


Sampler = Callable[[mx.array], mx.array]


class BatchedDecodeSessionError(RuntimeError):
    """Raised on any runtime-session invariant violation. Fail-stop by
    design, matching this session's other Phase 1 modules -- no
    auto-correction anywhere in this module either."""


@dataclass(frozen=True)
class PreparedStep:
    """Output of ``BatchedDecodeSession.prepare_step()`` -- everything
    a caller needs to (a) hand rank 1 this step's composition via
    ``message`` and (b) drive rank 0's own forward pass via ``ctx``/
    ``tokens``, WITHOUT rank 0 having called the model yet."""

    message: StepMessage
    ctx: BatchStepContext
    active_ids: tuple[int, ...]
    tokens: mx.array


@dataclass
class _RequestState:
    """Rank 0's per-request generation state -- everything needed to
    continue sampling for one request across steps, kept OUTSIDE
    ``BatchedDecodeDriver`` (which only owns protocol/cache-slot
    bookkeeping, not per-request generation policy)."""

    cache_slot: int
    next_token: int
    sampler: Sampler
    done: bool = False


@dataclass
class BatchedDecodeSession:
    """Rank 0's real decode-loop session: owns a ``BatchedDecodeDriver``
    (protocol + cache-slot bookkeeping), the actual batched MLX cache,
    and per-request generation state (next token, sampler, done flag).

    ``step()`` is the thing a real per-token decode loop calls once per
    real batched forward pass -- it derives the batch's composition and
    ordering ENTIRELY from ``BatchedDecodeDriver`` (never re-derives it
    independently, matching this session's established single-source-
    of-truth principle for this stack), stacks the active requests'
    ``next_token``s in that order, drives one real ``model(...)`` call
    under a ``batch_step_scope``, samples each row's next token with
    THAT request's own sampler, and reports the advance back to the
    driver.
    """

    driver: BatchedDecodeDriver
    batched_cache: "KVCacheType" = field(default_factory=list)
    _requests: dict[int, _RequestState] = field(default_factory=dict)

    @classmethod
    def new(cls, *, max_concurrency: int = 2) -> BatchedDecodeSession:
        return cls(driver=BatchedDecodeDriver.new(max_concurrency=max_concurrency))

    def admit_request(
        self,
        request_id: int,
        cache_slot: int,
        prefilled_cache: "KVCacheType",
        initial_token: int,
        sampler: Sampler,
    ) -> StepMessage:
        """Admit an ALREADY-PREFILLED request (via today's existing
        serial PP prefill path -- out of scope for this module) at
        ``cache_slot``, with the token its prefill pass already
        produced (``initial_token``) and the sampler to use for every
        subsequent decode step.

        Folds ``prefilled_cache`` into ``self.batched_cache`` via
        ``merge_request_caches`` -- correctly handles both the
        upfront-admission case (batched_cache is empty/all-fresh) and
        the mid-stream case (other slots already advanced, nonzero
        offset) identically, since ``merge_request_caches`` (backed by
        mlx-lm's own ``BatchKVCache.merge``) handles heterogeneous
        offsets directly (verified this session:
        ``test_merge_supports_mid_stream_admission_advanced_plus_fresh``,
        ``test_mid_stream_admission_matches_serial_plain_forwards``).
        """
        if request_id in self._requests:
            raise BatchedDecodeSessionError(
                f"admit_request({request_id}): already active in this "
                f"session -- duplicate admission, refusing to silently "
                f"overwrite generation state"
            )
        message = self.driver.admit_request(request_id, cache_slot)

        existing_slot_caches = self._extract_all_current_slot_caches()
        existing_slot_caches[cache_slot] = prefilled_cache
        ordered = [existing_slot_caches[slot] for slot in sorted(existing_slot_caches)]
        self.batched_cache = merge_request_caches(ordered)

        self._requests[request_id] = _RequestState(
            cache_slot=cache_slot,
            next_token=initial_token,
            sampler=sampler,
        )
        return message

    def _extract_all_current_slot_caches(self) -> dict[int, "KVCacheType"]:
        """Snapshot every currently-occupied slot's own single-request
        cache out of the current batched cache, keyed by slot index --
        the building block ``admit_request`` uses to re-merge a NEW
        slot alongside every EXISTING (possibly already-advanced)
        slot's real state, rather than assuming the batch is always
        empty at admission time (the realistic mid-stream case)."""
        result: dict[int, "KVCacheType"] = {}
        for slot in self.driver.cache_router.occupied_slots():
            result[slot] = extract_request_cache(self.batched_cache, slot)
        return result

    def prepare_step(self) -> "PreparedStep":
        """Decide THIS step's batch composition (via the real driver --
        never re-derived independently) and build the batched token
        tensor to feed the model, WITHOUT calling the model yet.

        Split out from ``step()`` (formerly one method) specifically so
        a real 2-rank caller can hand rank 1 the ``StepMessage`` BEFORE
        either rank's forward pass starts -- matching the real
        transport's actual ordering constraint: rank 1 must know this
        step's composition before it can build its own
        ``batch_step_scope``, and rank 0's own forward pass must not
        start first (or a caller driving both ranks from one process,
        e.g. a correctness-test harness, has no safe point to hand
        rank 1 the message without a race). Production and test
        callers alike: call ``prepare_step()`` once, send/mirror
        ``.message`` to rank 1, THEN dispatch both ranks' real forward
        passes (each under ``batch_step_scope(prepared.ctx)``), then
        call ``finish_step(prepared, logits)`` to sample and advance.
        """
        active_ids = tuple(
            rid
            for rid, rec in sorted(
                self._requests.items(), key=lambda kv: kv[1].cache_slot
            )
            if not rec.done
        )
        if not active_ids:
            raise BatchedDecodeSessionError(
                "prepare_step() called with no active (non-done) requests"
            )

        batched_tokens = mx.array(
            [[self._requests[rid].next_token] for rid in active_ids]
        )
        mx.eval(batched_tokens)

        message = self.driver.on_tokens_generated(active_ids)
        ctx = batch_step_context_from_step_message(message)
        if ctx.request_uids != active_ids:
            raise BatchedDecodeSessionError(
                f"prepare_step(): BatchStepContext order {ctx.request_uids} "
                f"does not match this session's own active_ids ordering "
                f"{active_ids} -- ordering mismatch between the driver's "
                f"StepMessage and this session's own request iteration, "
                f"refusing to proceed rather than risk misattributing "
                f"rows to the wrong requests"
            )
        return PreparedStep(
            message=message, ctx=ctx, active_ids=active_ids, tokens=batched_tokens
        )

    def run_forward(self, model: _ModelLike, prepared: "PreparedStep") -> mx.array:
        """Drive rank 0's real ``model(...)`` call for ``prepared``,
        under its ``batch_step_scope``. Returns the raw logits --
        callers that want ``step()``'s convenience one-call behavior
        should call ``finish_step`` immediately after; callers driving
        two ranks in lockstep (real or simulated transport) call this
        AFTER handing rank 1 ``prepared.message`` (see
        ``prepare_step``'s docstring)."""
        with batch_step_scope(prepared.ctx):
            logits = model(prepared.tokens, cache=self.batched_cache)
            mx.eval(logits)
            mx.eval([layer.state for layer in self.batched_cache])
        return logits

    def finish_step(
        self, prepared: "PreparedStep", logits: mx.array
    ) -> dict[int, tuple[int, bool]]:
        """Sample each active request's next token from ``logits``
        (row order matches ``prepared.active_ids``) with THAT
        request's own sampler, update this session's per-request
        state, and return ``{request_id: (new_token, is_done)}`` --
        ``is_done`` reflects the CALLER's own stop-condition check,
        which this module does not itself implement (EOS/max-tokens
        policy is a caller concern, matching ``RankOneMirrorDriver``'s
        own "this module only does mechanics, not policy" boundary) --
        callers set a request's ``done`` externally via ``mark_done``
        before the NEXT ``prepare_step()`` call to stop scheduling it.
        """
        results: dict[int, tuple[int, bool]] = {}
        for row, rid in enumerate(prepared.active_ids):
            rec = self._requests[rid]
            next_token = int(rec.sampler(logits[row, -1:]).item())
            rec.next_token = next_token
            results[rid] = (next_token, False)
        return results

    def step(self, model: _ModelLike) -> dict[int, tuple[int, bool]]:
        """Convenience one-call wrapper: ``prepare_step`` +
        ``run_forward`` + ``finish_step`` for a SINGLE-rank caller (no
        second rank to hand the message to first -- e.g. a
        single-process smoke test). Real 2-rank callers should use the
        three-call split directly (see ``prepare_step``'s docstring)."""
        prepared = self.prepare_step()
        logits = self.run_forward(model, prepared)
        return self.finish_step(prepared, logits)

    def mark_done(self, request_id: int) -> None:
        """Caller signals ``request_id`` has hit its own stop condition
        (EOS/max-tokens/explicit cancel) -- excludes it from future
        ``step()`` calls, but does NOT evict its slot yet (see
        ``evict_request``)."""
        rec = self._requests.get(request_id)
        if rec is None:
            raise BatchedDecodeSessionError(
                f"mark_done({request_id}): not an active request in this session"
            )
        rec.done = True

    def evict_request(self, request_id: int) -> tuple["KVCacheType", "_EvictionInfo"]:
        """Evict ``request_id``: extracts its final single-request
        cache (for a caller that wants to save it, e.g. prefix-cache
        write-back -- out of this module's own scope to decide), then
        drives the real ``BatchedDecodeDriver`` eviction protocol
        (DRAINING state, ``EvictMessage`` info for the caller to send
        over the real transport). The slot is NOT released from this
        session's own bookkeeping until ``on_evict_ack`` is called
        (matching the scheduler's own DRAINING-until-ack invariant)."""
        rec = self._requests.get(request_id)
        if rec is None:
            raise BatchedDecodeSessionError(
                f"evict_request({request_id}): not an active request in this session"
            )
        final_cache = extract_request_cache(self.batched_cache, rec.cache_slot)
        evict_info = self.driver.evict_request(request_id)
        return final_cache, _EvictionInfo(
            step_id=evict_info.step_id,
            request_id=evict_info.request_id,
            cache_slot=evict_info.cache_slot,
        )

    def on_evict_ack(self, request_id: int, cache_slot: int) -> None:
        """Rank 1's eviction ack arrived -- free the slot in the
        driver's bookkeeping AND remove this session's own per-request
        generation state (sampler, next_token). Rebuilds
        ``self.batched_cache`` from the remaining occupied slots only
        -- the evicted slot's stale bytes are never referenced again
        (module docstring's reset-on-assign convention, inherited from
        ``pp_batched_cache_router.py``)."""
        self.driver.on_evict_ack(request_id, cache_slot)
        del self._requests[request_id]
        remaining = self._extract_all_current_slot_caches()
        if remaining:
            ordered = [remaining[slot] for slot in sorted(remaining)]
            self.batched_cache = merge_request_caches(ordered)
        else:
            self.batched_cache = []

    def has_active_requests(self) -> bool:
        return any(not rec.done for rec in self._requests.values())


@dataclass(frozen=True)
class _EvictionInfo:
    step_id: int
    request_id: int
    cache_slot: int


@dataclass
class RankOneMirrorSession:
    """Rank 1's real decode-loop session: wraps ``RankOneMirrorDriver``
    (validation + its own cache-slot bookkeeping) and drives the SAME
    real ``model(...)`` call as rank 0, under the SAME
    ``BatchStepContext`` derived identically from the ``StepMessage``
    rank 0 sent -- never samples, never holds per-request generation
    state (matches ``RankOneMirrorDriver``'s zero-decision-logic
    design).

    Deliberately holds NO per-slot cache dict as separate mutable
    state (an earlier version of this class did -- ``_slot_caches``,
    set once at ``admit_request`` and never refreshed as ``step()``
    advanced the real ``batched_cache``. A real-wire correctness test
    caught this: after eviction, ``release_slot`` rebuilt
    ``batched_cache`` from that STALE dict, silently reverting the
    surviving request's cache to its state at admission time instead
    of its actual current advanced state -- a real, previously-hidden
    bug the in-process object-sharing test suite never exercised
    because it happened to never evict a slot mid-lifecycle in a way
    that surfaced the staleness. Fixed by always extracting current
    per-slot state from the live ``batched_cache`` on demand (mirrors
    ``BatchedDecodeSession._extract_all_current_slot_caches``'s own
    pattern exactly) -- there is only ONE source of truth for a
    slot's cache state: the live batched cache itself.
    """

    mirror_driver: RankOneMirrorDriver
    batched_cache: "KVCacheType" = field(default_factory=list)

    @classmethod
    def new(cls, *, max_concurrency: int = 2) -> RankOneMirrorSession:
        return cls(mirror_driver=RankOneMirrorDriver(max_concurrency=max_concurrency))

    def _extract_all_current_slot_caches(self) -> dict[int, "KVCacheType"]:
        """Snapshot every currently-occupied slot's own single-request
        cache out of the current batched cache, keyed by slot index --
        mirrors ``BatchedDecodeSession``'s identically-named method
        exactly (see this class's docstring for why this must always
        read from the LIVE ``batched_cache``, never a separately
        tracked dict)."""
        assert self.mirror_driver.cache_router is not None
        result: dict[int, "KVCacheType"] = {}
        for slot in self.mirror_driver.cache_router.occupied_slots():
            result[slot] = extract_request_cache(self.batched_cache, slot)
        return result

    def admit_request(
        self, message: StepMessage, cache_slot: int, prefilled_cache: "KVCacheType"
    ) -> None:
        """Validate rank 0's admission ``StepMessage`` and fold this
        rank's own half of the newly-admitted request's prefilled
        cache into the batch, at the SAME slot rank 0 used (the
        message itself carries the slot; this call's ``cache_slot``
        argument is the caller's own already-known assignment, cross-
        checked against the message's own entries for consistency).
        Correctly handles a request joining alongside already-
        ADVANCED slots (mid-stream admission) since it re-extracts
        every existing slot's CURRENT state before re-merging, exactly
        like ``BatchedDecodeSession.admit_request``."""
        entry_slots = {entry.cache_slot for entry in message.entries}
        if cache_slot not in entry_slots:
            raise BatchedDecodeSessionError(
                f"admit_request: cache_slot={cache_slot} not present in "
                f"the StepMessage's entries {entry_slots} -- rank "
                f"0/rank 1 admission mismatch"
            )
        self.mirror_driver.on_step_message(message)
        existing = self._extract_all_current_slot_caches()
        existing[cache_slot] = prefilled_cache
        ordered = [existing[slot] for slot in sorted(existing)]
        self.batched_cache = merge_request_caches(ordered)

    def step(self, model: _ModelLike, message: StepMessage, tokens: mx.array) -> None:
        """Validate rank 0's decode-step ``StepMessage`` and drive this
        rank's half of the SAME real forward pass, using the
        IDENTICAL ``BatchStepContext`` derivation rank 0 used."""
        ctx = self.mirror_driver.on_step_message(message)
        with batch_step_scope(ctx):
            out = model(tokens, cache=self.batched_cache)
            mx.eval(out)
            mx.eval([layer.state for layer in self.batched_cache])

    def evict(self, message: EvictMessage) -> "KVCacheType":
        """Validate the eviction notice and extract this rank's own
        final single-request cache for the evicted slot -- the
        CALLER is responsible for the real ack round-trip and for
        calling ``release_slot`` once it has genuinely freed this
        rank's cache state (matching
        ``RankOneMirrorDriver.on_evict_message``'s own established
        boundary)."""
        self.mirror_driver.on_evict_message(message)
        return extract_request_cache(self.batched_cache, message.cache_slot)

    def release_slot(self, cache_slot: int) -> None:
        """Free ``cache_slot`` in the router's bookkeeping and rebuild
        ``batched_cache`` from every REMAINING occupied slot's CURRENT
        state (see this class's docstring on why this must never use
        a separately tracked, staleness-prone dict)."""
        assert self.mirror_driver.cache_router is not None
        remaining = self._extract_all_current_slot_caches()
        remaining.pop(cache_slot, None)
        self.mirror_driver.cache_router.release_slot(cache_slot)
        if remaining:
            ordered = [remaining[slot] for slot in sorted(remaining)]
            self.batched_cache = merge_request_caches(ordered)
        else:
            self.batched_cache = []

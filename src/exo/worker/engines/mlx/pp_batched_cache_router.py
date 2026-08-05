# pyright: reportPrivateUsage=false
"""Per-request KV-cache routing for the batched-PP decode scheduler.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 6.2
item 3: "Each rank maintains its OWN half of each in-flight request's KV
cache ... just needs to become a dict keyed by request UID instead of a
single active cache, with BatchRotatingKVCache/BatchPoolingCache's
existing per-stream tracking reused for the within-step batch dimension."

Design, per a `consult` review (2026-08-05):

1. **Slot-indexed, matching SchedulerCore's existing slot model exactly
   — not a second independent request_id-keyed structure.**
   ``SchedulerCore``/``RankOneMirror`` (pp_scheduler_protocol.py) already
   track state per ``cache_slot``; adding a parallel request_id-keyed
   cache dict would be redundant indirection two structures could drift
   out of sync on. This router is indexed by the SAME slot numbers.

2. **The batched cache IS the canonical storage — this router manages
   metadata (per-slot length/offset), not physical merge/extract on
   every step.** Physically concatenating per-slot cache lists into a
   batched cache before each forward and slicing back after would be
   O(total cache bytes) per decode token — a real perf mistake the
   consult flagged explicitly. Slot assignment resets that slot's
   length to 0 in the batched cache; slot release is a no-op (the old
   occupant's KV bytes are left in place, physically). ``merge()``/
   ``extract()`` (mlx-lm's existing machinery, already proven by
   ``prefill_batched``) are used only at real BOUNDARIES: constructing
   the initial batched cache from N single-request caches (once, at
   startup of a batch), and extracting a single-request cache back out
   when a request needs to leave the batch (eviction, or in a future
   phase, migrating between different concurrent batches).

3. **No generation counter needed for the ABA slot-reuse race.**
   ``SchedulerCore``'s ``DRAINING`` slot state (pp_scheduler_protocol.py)
   ALREADY prevents a new request from being assigned to a slot before
   the prior occupant's eviction is acknowledged — this router can
   safely assume "slot is FREE" means "genuinely safe to reset," because
   the wire-protocol state machine upstream of this module already
   enforces that invariant structurally. Re-deriving that guarantee here
   (e.g. with a second generation-counter mechanism) would duplicate a
   correctness invariant across two modules that could then silently
   diverge — the DRAINING/evict-ack protocol is the single source of
   truth for "is this slot safe to reuse," and this router trusts it.

4. **Reset-on-assign, not trim-on-release; never trims stale bytes
   from a released slot.** Per the consult: a released slot's old KV
   bytes physically remain in the buffer until overwritten by the next
   occupant's own writes. The invariant that makes this safe is that
   EVERY consumer (the attention mask construction, in particular)
   derives visibility strictly from the tracked PER-SLOT LENGTH, never
   from the buffer's physical extent — a slot with length=0 must never
   contribute attention regardless of what stale bytes sit in its
   buffer. This module's ``reset_slot`` explicitly zeroes the tracked
   length; it deliberately does NOT touch the underlying array data.
   Trade-off (explicitly accepted per the consult, not silently
   incurred): a slot's buffer capacity ratchets up to the longest
   request that ever occupied it and never shrinks — bounded by
   (slots × max_seq × layers-per-rank), a real but acceptable Phase 1
   cost; trim-on-release is a later optimization, not a Phase 1
   requirement.

Scope: this module handles the batched-cache LIFECYCLE (assign a slot,
reset a slot, extract a slot's final single-request cache on eviction)
for a FIXED slot count matching ``SchedulerCore.max_concurrency`` — it
does not itself run any forward pass or touch the metaframe transport.
Wiring this into the real decode loop (Phase 1's remaining scheduler
runtime work) is a separate step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import KVCacheType


class CacheRouterError(RuntimeError):
    """Raised on any invariant violation in this module. Fail-stop by
    design, matching ``pp_scheduler_protocol.py``'s convention — no
    auto-correction/repair anywhere in this module either."""


@dataclass
class _SlotInfo:
    occupied: bool
    length: int


class BatchedCacheRouter:
    """Manages the lifecycle of a fixed-size batched KV cache shared
    across ``max_concurrency`` request slots, matching
    ``pp_scheduler_protocol.SchedulerCore``'s slot numbering exactly.

    This class does NOT construct the batched cache itself (that
    requires real per-layer ``KVCacheType`` instances and belongs to
    the caller, which has model/layer knowledge this module
    deliberately doesn't need) — it wraps an already-merged batched
    cache and tracks per-slot occupancy/length metadata around it,
    exposing the assign/reset/release/extract operations Phase 1's
    scheduler runtime needs.
    """

    def __init__(self, *, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >=1, got {max_concurrency}")
        self.max_concurrency = max_concurrency
        self._slots: dict[int, _SlotInfo] = {
            slot: _SlotInfo(occupied=False, length=0) for slot in range(max_concurrency)
        }

    def _require_slot_index(self, slot: int) -> None:
        if slot not in self._slots:
            raise CacheRouterError(
                f"cache_slot={slot} is out of range for max_concurrency="
                f"{self.max_concurrency} (valid slots: 0..{self.max_concurrency - 1})"
            )

    def is_occupied(self, slot: int) -> bool:
        self._require_slot_index(slot)
        return self._slots[slot].occupied

    def length(self, slot: int) -> int:
        self._require_slot_index(slot)
        return self._slots[slot].length

    def assign_slot(self, slot: int) -> None:
        """Mark ``slot`` as occupied by a brand-new request, with cache
        length reset to 0. Callers of this router are expected to have
        already confirmed via ``SchedulerCore``/``RankOneMirror`` that
        this slot is genuinely FREE (not DRAINING) before calling this
        — this router does NOT re-derive that guarantee itself (see
        module docstring point 3); it only asserts its OWN local
        bookkeeping is consistent (not already marked occupied)."""
        self._require_slot_index(slot)
        info = self._slots[slot]
        if info.occupied:
            raise CacheRouterError(
                f"assign_slot({slot}) called but this router's own "
                f"bookkeeping already shows slot={slot} occupied "
                f"(length={info.length}) -- this router's local state is "
                f"out of sync with the caller's scheduler decision; "
                f"release_slot must be called before reassigning"
            )
        info.occupied = True
        info.length = 0

    def advance_slot(self, slot: int, *, n_tokens: int = 1) -> None:
        """Record that ``slot``'s occupant just advanced by ``n_tokens``
        (decode-only, Phase 1 scope: always exactly 1). Mirrors
        ``pp_scheduler_protocol.known_len_advance``'s semantics but
        deliberately kept as a SEPARATE function in a separate module
        rather than importing that one directly — this router's
        "length" tracks the REAL cache buffer occupancy (a physical MLX
        array property), while the protocol module's "cache_len" tracks
        the WIRE PROTOCOL's claimed value; they must independently
        agree (checked by whatever wires this router to the protocol
        layer) rather than share one implementation that could hide a
        real divergence between "what the protocol says" and "what the
        cache actually holds\"."""
        self._require_slot_index(slot)
        info = self._slots[slot]
        if not info.occupied:
            raise CacheRouterError(
                f"advance_slot({slot}) called but slot={slot} is not "
                f"occupied -- refusing to advance a cache length for a "
                f"slot with no active request"
            )
        if n_tokens < 1:
            raise CacheRouterError(
                f"advance_slot({slot}, n_tokens={n_tokens}) -- n_tokens must be >=1"
            )
        info.length += n_tokens

    def release_slot(self, slot: int) -> None:
        """Mark ``slot`` as no longer occupied. Per module docstring
        point 4, this does NOT touch the underlying cache array data —
        it only resets this router's own occupancy/length bookkeeping.
        The caller is responsible for having already extracted any
        final single-request cache state it needs (via a real
        ``extract()`` call on the underlying batched cache, outside
        this module's scope) BEFORE calling this, if the request's
        cache needs to survive past this point (it normally doesn't —
        eviction means the request is done)."""
        self._require_slot_index(slot)
        info = self._slots[slot]
        if not info.occupied:
            raise CacheRouterError(
                f"release_slot({slot}) called but slot={slot} is already "
                f"not occupied -- duplicate/stale release, refusing"
            )
        info.occupied = False
        info.length = 0

    def occupied_slots(self) -> tuple[int, ...]:
        """Slots currently marked occupied, in ascending order --
        matches ``SchedulerCore._active_batch_entries``'s
        ``sorted(self._requests.items())`` ordering convention so a
        caller building a batch snapshot from both structures gets
        consistent slot ordering."""
        return tuple(sorted(s for s, info in self._slots.items() if info.occupied))


def merge_request_caches(caches: list["KVCacheType"]) -> "KVCacheType":
    """Merge N single-request ``KVCacheType`` instances (e.g. from N
    completed serial prefills) into ONE batched cache, using mlx-lm's
    existing per-layer ``merge()`` classmethods (the same machinery
    ``prefill_batched`` already uses and has proven correct) -- this
    module does not reimplement merge logic, it only orchestrates the
    per-layer call across however many layers this rank owns.

    Requires all N caches to have the SAME per-layer cache-type
    structure (e.g. all ``CacheList(RotatingKVCache, PoolingCache)``)
    -- a real, expected precondition since every request runs the same
    model architecture; a structural mismatch here indicates a caller
    bug (mixing caches from different model configs), not a data-driven
    condition to silently handle.

    Force-evaluates the merged cache's array state before returning.
    MLX arrays are lazy; ``merge()``'s internal ``mx.zeros``/indexing
    ops build a graph that is NOT materialized until something calls
    ``mx.eval`` on it. If the caller hands this merged cache to a
    DIFFERENT thread than the one that called ``merge_request_caches``
    (a real, expected usage pattern -- e.g. driving 2 simulated PP
    ranks on separate OS threads, as this fork's own correctness test
    harnesses do), that other thread would be the one to first force
    evaluation of a graph node it never built -- MLX's per-thread
    stream/command-queue context is thread-local, so evaluating a
    lazy node from a different thread than the one it was built on
    raises ``RuntimeError: There is no Stream(gpu, N) in current
    thread.`` (confirmed empirically the first time this function was
    exercised across a real thread boundary in
    test_pp_batched_decode_correctness.py, 2026-08-05 -- not a
    hypothetical). Evaluating HERE, before this function returns
    control to any caller, closes that hazard structurally rather than
    requiring every future caller to remember to ``mx.eval`` the
    result themselves.
    """
    if not caches:
        raise CacheRouterError("merge_request_caches called with an empty list")
    n_layers = len(caches[0])
    for i, c in enumerate(caches):
        if len(c) != n_layers:
            raise CacheRouterError(
                f"merge_request_caches: cache at index {i} has "
                f"{len(c)} layers, expected {n_layers} (from index 0) "
                f"-- all caches being merged must share the same "
                f"per-layer structure"
            )
    merged: list[Any] = [
        caches[0][layer_idx].merge([c[layer_idx] for c in caches])  # type: ignore[attr-defined]
        for layer_idx in range(n_layers)
    ]
    mx.eval([layer.state for layer in merged])  # pyright: ignore[reportAny]
    return cast("KVCacheType", merged)


def extract_request_cache(batched_cache: "KVCacheType", slot: int) -> "KVCacheType":
    """Extract slot ``slot``'s single-request cache back out of a
    batched cache, using mlx-lm's existing per-layer ``extract()``
    methods. Used at real boundaries (module docstring point 2) --
    e.g. a request finishing and its final KV state being handed off
    for a prefix-cache save, NOT called on every decode step."""
    return [
        layer.extract(slot)  # type: ignore[attr-defined]
        for layer in batched_cache
    ]

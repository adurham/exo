# pyright: reportPrivateUsage=false
"""Phase 1 wire-protocol state machine for the batched-PP scheduler.

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 8
Risk #11: "No wire-protocol state machine or deadlock analysis exists
yet ... needs to be written out as an explicit state machine ... BEFORE
Phase 1 starts, not discovered empirically while debugging a hang."
This module IS that state machine, per a `consult` review's guidance
(2026-08-05): a PURE, zero-I/O, zero-MLX core, importable and runnable
identically on both ranks, so both the fuzz tests AND the real rank-0
scheduler AND the real rank-1 mirror validator share exactly one
implementation of "what's legal" — not three independent
reimplementations that could silently drift out of sync.

Design principles (all from the consult review, kept explicit here so
future edits don't accidentally violate them):

1. **Pure core, no I/O.** No MLX, no sockets, no threads, no time.time().
   ``SchedulerCore.handle(event) -> list[Command]`` is a deterministic
   pure function of (current state, event). This is what makes
   deterministic fuzzing possible at all — a fuzzer needs to replay the
   exact same event sequence and get the exact same result every time.
2. **Rank-1 runs a MIRROR of the same state machine, not a different
   one.** ``RankOneMirror`` validates every incoming ``StepMessage``/
   ``EvictMessage`` against ITS OWN independently-tracked per-slot
   state and raises loudly (``ProtocolViolationError``) on any
   mismatch. This is the concrete fix for Risk #11's "no deadlock
   analysis" complaint: illegal messages get rejected immediately and
   loudly on rank 1, they never silently corrupt cache state or cause
   rank 1 to block forever waiting for something rank 0 will never
   send.
3. **Fail-stop, never repair.** Per the consult's explicit warning:
   "auto-correction code is where silent corruption hides." Every
   invariant violation in this module raises; nothing here attempts to
   recover, retry, or guess what was "probably meant."
4. **Slot exclusivity + explicit DRAINING state, not omission-based
   cancellation.** Directly fixes design doc Risk #10 ("cancellation-
   by-omission ... is AMBIGUOUS to rank 1: it cannot distinguish 'not
   scheduled this step' from 'cancelled, free the state'"). A request
   being cancelled transitions through DRAINING (an explicit
   ``EvictMessage`` is sent, and the slot is NOT reusable until rank 1
   acknowledges the eviction) rather than simply stopping to appear in
   step batches. This is also THE #1 concrete corruption vector the
   consult flagged: abort a request while a microbatch referencing its
   slot is still in flight, reassign that slot immediately, and rank 1
   runs the stale in-flight entry's activations into the NEW request's
   cache — silent cross-request data corruption, exactly the failure
   mode Risk #5/#11 warn about. The DRAINING state + eviction-ack
   requirement structurally prevents that reuse-before-ack race.
5. **Lockstep monotonic step_id, even though the transport is
   ordered.** A duplicate/skipped/out-of-order ``step_id`` on rank 1 is
   a FATAL protocol violation, not a warning — cheap to check (one
   int), and catches SCHEDULER bugs (a logic error in what rank 0
   decides to send), which is a different bug class than a TRANSPORT
   bug (which Phase 0.5 already covers via
   ``handshake_metaframe_protocol`` and the metaframe's own
   ``version``/shape fields).
6. **Explicit per-entry `expected_cache_len`.** Rank 1 recomputes each
   scheduled request's actual cache length from its own cache-slot
   state and compares against what rank 0's message claims. A mismatch
   is the cheap tripwire that turns "plausibly wrong tokens, silently"
   into "loud crash, immediately" — directly targeting the "off-by-one
   KV lengths from mixed prefill/decode batches" failure the consult
   flagged as the classic silent-corruption case for this kind of
   system.

Scope: this module implements the SIMPLEST case Phase 1 targets per the
design doc — 2 concurrent DECODE-ONLY requests (both already prefilled
via today's existing serial PP prefill), NO speculative decode. PREFILL
batching/chunking (Phase 2) and DSpark gating (Phase 4) are explicitly
NOT modeled here yet; ``Phase.PREFILL`` exists in the state enum as a
forward-compatible placeholder (a request must pass through it before
DECODE, matching real usage) but this module's Phase-1 scope only ever
constructs requests that begin already in DECODE, mirroring "both
already prefilled via today's existing serial PP prefill" from the
design doc's own Phase 1 description.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ProtocolViolationError(RuntimeError):
    """Raised by ``RankOneMirror`` (or ``SchedulerCore`` on rank 0) the
    instant any message or event violates this module's invariants.
    Fail-stop by design (see module docstring point 3) -- callers must
    NOT attempt to catch this and continue serving the offending
    request; the whole pipeline step (or the specific request slot,
    depending on which invariant fired) must be torn down.
    """


class Phase(Enum):
    """A request's current phase. See module docstring's Scope note:
    Phase-1's actual scope only ever starts requests already in DECODE,
    but PREFILL is modeled now so Phase 2 doesn't need a state-enum
    change to add chunked-prefill scheduling."""

    PREFILL = auto()
    DECODE = auto()


class SlotState(Enum):
    """A cache slot's lifecycle, independently tracked identically by
    ``SchedulerCore`` (rank 0's view) and ``RankOneMirror`` (rank 1's
    view) -- see module docstring point 2. The two views MUST agree at
    every step; any divergence is exactly the class of bug this module
    exists to catch loudly instead of letting it corrupt cache state.
    """

    FREE = auto()
    ACTIVE = auto()
    # DRAINING: an eviction has been requested (request cancelled/done)
    # but rank 1 has not yet acknowledged freeing its cache state for
    # this slot. The slot is NOT reusable while DRAINING -- this is the
    # single invariant that prevents the "reassign slot while a stale
    # in-flight microbatch entry still targets the old request" race
    # (module docstring point 4).
    DRAINING = auto()


@dataclass(frozen=True)
class BatchEntry:
    """One request's contribution to a single pipeline step's batch.
    Wire-equivalent to what an eventual real ``encode_step_metaframe``
    would serialize -- kept as a plain frozen dataclass here (not an
    ``mx.array``-backed structure) so this module has zero MLX
    dependency and can be fuzzed at Python-object speed with no GPU/CPU
    tensor allocation at all.
    """

    request_id: int
    cache_slot: int
    phase: Phase
    expected_cache_len: int
    n_tokens: int


@dataclass(frozen=True)
class StepMessage:
    """One pipeline step's full batch composition, as rank 0 would send
    it (preceded by, or embedding, an eventual real metaframe header --
    this module models the SCHEDULING decision, not the wire encoding,
    which is Phase 0.5's ``pp_metaframe.py`` concern)."""

    step_id: int
    entries: tuple[BatchEntry, ...]


@dataclass(frozen=True)
class PrefillMessage:
    """Rank 0's explicit "admit request ``request_id`` NOW, on
    ``cache_slot``, and run its prefill" instruction to rank 1.

    WHY THIS EXISTS (this is not a convenience message -- it closes a
    real, hardware-confirmed deadlock):
    ``docs/batched-decode-n2-admission-handoff-2026-08-05.md`` and
    ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md``
    Section 15 document that N=2 genuinely concurrent requests
    deadlock the real 2-node cluster with
    ``[jaccl] reliable_all_reduce_v2 deadline``. The root cause is NOT
    that prefill and decode traffic overlap within a rank (each
    rank's ``runner.py`` loop is synchronous and cannot do both at
    once) -- it is that admission decisions are UNSYNCHRONIZED ACROSS
    RANKS. Each rank's runner independently pulls work off its OWN
    local work queue and independently decides, per loop iteration,
    whether to prefill a newly-arrived request (the old single-request
    metaframe path, structurally different wire traffic) or to run a
    batched decode step (``Rank0BatchedDecodeGlue``/
    ``Rank1BatchedDecodeGlue.tick()``). Nothing today makes both ranks
    reach the SAME decision at the SAME logical moment, so rank 0 can
    begin request B's prefill collectives in the very window rank 1 is
    still mid-decode-step -- two ranks issuing mismatched jaccl
    collectives, which is a real deadlock.

    The fix this message implements: the "admit request B now, start
    its prefill" decision stops being a per-rank local decision and
    becomes an explicit control message on the SAME single-writer wire
    channel the decode-step traffic already uses. Rank 1 no longer
    decides when to prefill; it is TOLD, in-band, in the same ordered
    stream as ``StepMessage``/``EvictMessage``, so the tick boundary at
    which admission happens is by construction identical on both
    ranks.

    Fields:
      ``step_id``: the usual lockstep monotonic step counter shared
        with ``StepMessage``/``EvictMessage`` (module docstring point
        5) -- an admission consumes a step id exactly like any other
        control message, so a skipped/duplicated admission is caught by
        the same cheap tripwire.
      ``request_id``/``cache_slot``: which request is being admitted,
        and the slot rank 1 must bind its cache state to. The slot MUST
        be FREE on rank 1's own independently-tracked view (module
        docstring point 4) -- admitting onto a DRAINING slot is the
        slot-reuse-before-ack corruption vector.
      ``n_prompt_tokens``: the prompt length rank 1 should expect to
        prefill, so rank 1 can validate the activation shapes it
        subsequently receives instead of inferring them (same
        "explicit expected length as a tripwire" rationale as
        ``BatchEntry.expected_cache_len``, module docstring point 6).
      ``flags``: bitfield, see ``PREFILL_FLAG_*`` below. Reserved bits
        MUST be zero; this module never silently ignores an unknown
        flag bit (module docstring point 3: fail-stop, never repair) --
        the wire layer rejects them.

    ``flags`` bit 0 (``PREFILL_FLAG_SINGLE_REQUEST_FALLBACK``) means
    "this request is INELIGIBLE for the batched-decode path; prefill it
    through the OLD single-request ``MetaFramedPipelineFirstLayer``/
    ``MetaFramedPipelineLastLayer`` path, not the batched metaframe
    layers." Rank 1 cannot derive this itself -- eligibility is
    evaluated on rank 0 (see the batched-decode eligibility logic) --
    and getting it wrong means the two ranks install structurally
    DIFFERENT metaframe layer stacks for the same request, which is the
    same mismatched-collectives deadlock in a different disguise. So it
    is carried explicitly on the wire rather than recomputed.
    """

    step_id: int
    request_id: int
    cache_slot: int
    n_prompt_tokens: int
    flags: int


# Bit 0 of ``PrefillMessage.flags`` -- see that dataclass's docstring.
# Set => route this request's prefill through the legacy
# single-request MetaFramedPipelineFirstLayer/LastLayer path (it is
# ineligible for batched decode). Clear => batched metaframe path.
PREFILL_FLAG_SINGLE_REQUEST_FALLBACK = 1 << 0

# Every currently-defined flag bit OR'd together. Anything outside this
# mask is a reserved bit and MUST be zero on the wire -- the wire layer
# (``pp_scheduler_wire.py``) rejects violations loudly rather than
# masking them off, because a set reserved bit means the peer is
# running a build with semantics this build does not implement, and
# silently dropping it is precisely how a version skew turns into
# wrong-path routing instead of a clean crash.
PREFILL_FLAGS_KNOWN_MASK = PREFILL_FLAG_SINGLE_REQUEST_FALLBACK


@dataclass(frozen=True)
class EvictMessage:
    """Explicit eviction notice for one cache slot -- the fix for Risk
    #10's cancellation-by-omission ambiguity (module docstring point
    4). Rank 1 must free its cache state for ``cache_slot`` upon
    receipt and is expected to reply with an ``EvictAckMessage``."""

    step_id: int
    request_id: int
    cache_slot: int


@dataclass(frozen=True)
class EvictAckMessage:
    """Rank 1's acknowledgement that it has freed cache state for
    ``cache_slot``. Only after this is received may
    ``SchedulerCore``/``RankOneMirror`` transition the slot from
    DRAINING back to FREE and permit it to be reused by a new
    request."""

    step_id: int
    request_id: int
    cache_slot: int


@dataclass(frozen=True)
class PrefillReadyMessage:
    """Rank 1's reply to a ``PrefillMessage`` -- closes the SECOND
    real, hardware-confirmed race the PrefillMessage/PrefillGrant
    mechanism left open (see ``docs/batched-decode-n2-admission-
    handoff-2026-08-05.md``'s "2026-08-06 finding: prefill
    forward-pass race" section for the full incident writeup this
    message was built to fix -- a genuine
    ``SchedulerWireProtocolError: recv_header: version mismatch``
    crash on real N=2 hardware).

    THE RACE THIS CLOSES: rank 0's own real prefill forward pass (the
    thing ``PrefillMessage`` announces is about to happen) uses a
    SEPARATE wire transport (``pp_metaframe.py``'s
    ``send_metaframe``/``recv_metaframe``) from the scheduler-wire
    control channel ``PrefillMessage`` itself travels on. Both ranks
    MUST enter their respective ``model(...)`` forward calls at
    roughly the SAME real time for the per-layer p2p hidden-state
    exchange inside those calls to succeed (this is the SAME
    already-proven invariant single-request PP prefill has always
    relied on -- nothing new here). But `agree_on_tasks()` under
    `EXO_PP_NO_COORD_COLLECTIVE=1` is PURELY LOCAL (confirmed by
    reading ``get_coord_group``/``mx_all_gather_tasks``: `group=None`
    is a genuine local-only no-op, not a degraded collective) -- so
    rank 1's own local task-dispatch pipeline (its `_work_queue` /
    `submit()` call for the SAME logical request) can genuinely lag
    rank 0's, for real reasons (rank 1 busy mid-decode-step for a
    DIFFERENT active request, or its own runner event loop simply not
    having reached that point yet), not just microsecond jitter.
    Without this ack, rank 0 had no way to know rank 1 wasn't ready
    yet and would start sending real metaframe bytes onto the wire
    regardless -- exactly the byte-stream corruption the real crash
    exhibited (rank 1's own SCHEDULER-WIRE `recv_header()` call, made
    from its own next unrelated `tick()` iteration, read rank 0's
    METAFRAME bytes and decoded `version=3` where it expected `1`).

    Rank 0 now BLOCKS on receiving this message (bounded, fails loud
    on timeout -- never an unbounded wait) before running its own
    prefill forward pass. ``ready=True`` means rank 1 has its matching
    ``_DeferredPrefill`` entry registered and is about to run the SAME
    real forward pass itself, immediately, synchronously, in the same
    call that sends this ack -- so by the time rank 0 receives
    ``ready=True`` and starts its own forward pass, rank 1 is either
    already inside its matching ``model(...)`` call or about to be,
    restoring the "both ranks enter together" timing invariant.
    ``ready=False`` (a genuine, expected NACK, not an error) means
    rank 1's own local dispatch hasn't registered this request yet --
    rank 0 must NOT run its forward pass this tick; it returns to its
    own loop and re-grants the SAME request on a LATER tick (bounded
    retry count, fails loud rather than looping forever) rather than
    blocking rank 1's own event loop with a synchronous wait for a
    dict entry only that SAME loop's own later iteration can ever
    populate (a real, confirmed deadlock hazard: rank 1's task-reader
    thread only feeds a queue -- the SAME main-thread loop that would
    run this wait is also the loop responsible for draining that
    queue and calling `submit()`, so a blocking poll here cannot ever
    resolve; consult-reviewed, see the handoff doc's own writeup)."""

    step_id: int
    request_id: int
    ready: bool


# ---------------------------------------------------------------------
# Events fed into SchedulerCore.handle() (rank 0's pure decision logic).
# These are NOT wire messages -- they are the local inputs that drive
# rank 0's own scheduling decisions (a new request arriving, a token
# being generated, a request finishing/being cancelled). What rank 0
# actually SENDS to rank 1 in response is a Command (below).
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NewRequestEvent:
    """A new request has arrived and been assigned a cache slot,
    already past prefill (Phase-1 scope -- see module docstring)."""

    request_id: int
    cache_slot: int


@dataclass(frozen=True)
class TokenGeneratedEvent:
    """One real decode step completed for ``request_ids`` -- EVERY
    request in this tuple advanced by exactly 1 token, as part of the
    SAME real batched forward pass.

    ``request_ids`` is a tuple, not a single ``request_id`` (v1 of this
    event, changed 2026-08-05 when batched-decode wiring work began) --
    per a `consult` review: a real batched decode step where N
    requests advance SIMULTANEOUSLY in one model forward call is a
    single atomic occurrence, and must map to a single atomic protocol
    event. Splitting it into N single-request events would make
    ``RankOneMirror`` pass through N-1 intermediate states that never
    existed on rank 0 (e.g. after event 1 of 2, the mirror would see
    request A advanced but B not) -- exactly the kind of core/mirror
    divergence surface this module's fail-stop design exists to avoid
    creating in the first place. A tuple (not a ``set``) preserves a
    defined iteration order matching ``_active_batch_entries``'s own
    CACHE_SLOT sort convention (2026-08-06: changed from request_id --
    see ``_active_batch_entries``'s own docstring for the real bug
    that fix closes) -- both ranks must see identical ordering for
    anything order-dependent.

    Non-empty by construction (see ``__post_init__``) -- an empty
    tuple would silently mean \"a decode step advanced zero requests\",
    which is not a real occurrence this protocol should ever need to
    represent.
    """

    request_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ProtocolViolationError(
                "TokenGeneratedEvent.request_ids must be non-empty -- a "
                "decode step with zero advancing requests is not a real "
                "occurrence this event should represent"
            )


@dataclass(frozen=True)
class RequestDoneEvent:
    """``request_id`` has reached EOS/max-tokens naturally and should be
    evicted."""

    request_id: int


@dataclass(frozen=True)
class RequestAbortedEvent:
    """``request_id`` has been cancelled externally (client
    disconnect/explicit cancel) and should be evicted -- distinct event
    type from ``RequestDoneEvent`` only for observability; both drive
    identical eviction logic in ``SchedulerCore``."""

    request_id: int


@dataclass(frozen=True)
class EvictAckReceivedEvent:
    """Rank 1's ``EvictAckMessage`` has arrived back at rank 0 -- the
    slot may now transition DRAINING -> FREE."""

    request_id: int
    cache_slot: int


Event = (
    NewRequestEvent
    | TokenGeneratedEvent
    | RequestDoneEvent
    | RequestAbortedEvent
    | EvictAckReceivedEvent
)


# ---------------------------------------------------------------------
# Commands emitted by SchedulerCore.handle() -- what rank 0's real
# scheduler (Phase 1's actual runtime code, not this module) must DO in
# response to an event: send a StepMessage, send an EvictMessage, or
# nothing this event (e.g. an EvictAckReceivedEvent just updates local
# state, no wire traffic).
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SendStepCommand:
    message: StepMessage


@dataclass(frozen=True)
class SendEvictCommand:
    message: EvictMessage


Command = SendStepCommand | SendEvictCommand


@dataclass
class _RequestRecord:
    cache_slot: int
    phase: Phase
    cache_len: int


class SchedulerCore:
    """Rank 0's PURE scheduling decision logic. See module docstring
    for why this must stay zero-I/O: ``handle()`` is deterministically
    replayable, which is what makes the fuzz test in
    ``test_pp_scheduler_protocol.py`` meaningful -- the exact same
    event sequence always produces the exact same command sequence and
    the exact same internal state, with no hidden nondeterminism from
    real timing/threading to obscure a reproduction.

    Phase-1 scope (per module docstring): decode-only, 2 concurrent
    requests max (``max_concurrency``, defaults to 2 per the design
    doc's explicitly-confirmed N=2 target -- Section 10 "Supporting
    more than N=2 concurrent requests (CONFIRMED explicitly out of
    scope...)"). A batch is emitted (``SendStepCommand``) once per
    ``handle()`` call that changes the active set's composition or
    advances any active request's decode step -- callers drive this by
    feeding one event per real occurrence, not by polling.
    """

    def __init__(self, *, max_concurrency: int = 2) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >=1, got {max_concurrency}")
        self.max_concurrency = max_concurrency
        self._requests: dict[int, _RequestRecord] = {}
        self._slot_state: dict[int, SlotState] = {}
        self._step_id: int = 0

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    def _active_batch_entries(
        self, *, advancing_request_ids: frozenset[int] = frozenset()
    ) -> tuple[BatchEntry, ...]:
        """Snapshot the full current active set for rank 1's routing.

        ``advancing_request_ids``: the set of requests whose state
        actually changed as a result of THIS ``handle()`` call -- each
        gets ``n_tokens=1`` (Phase-1 scope: decode-only, always exactly
        1 token per real advance). Every OTHER active request
        co-listed in the same snapshot gets ``n_tokens=0`` -- being
        included in a step message for rank 1's bookkeeping does NOT
        mean every co-listed request generated a token in lockstep with
        this specific event; only the ones actually named here did.
        Empty (the default) means no request advanced this call (e.g.
        a brand new request just joined at its baseline cache_len=0,
        or the batch composition changed for bookkeeping reasons
        only).

        A ``frozenset`` (not a plain ``set``) -- immutable, matching
        this dataclass-heavy module's preference for frozen/hashable
        types throughout (see ``Event``/``Command`` types, all
        ``@dataclass(frozen=True)``) even though membership testing
        (not hashing this specific object) is all that's needed here.

        Earlier versions of this method hardcoded ``n_tokens=1`` for
        EVERY entry regardless of which request's event fired -- this
        silently double-counted the cache-length advance for every
        OTHER co-listed request on every single step (e.g. a
        ``TokenGeneratedEvent`` for request A would claim request B, C,
        etc. also advanced by 1 token, when they didn't). Caught
        immediately by this module's own fuzz test
        (``test_fuzz_random_event_sequences_never_desync``, seed=0) the
        first time a multi-request sequence actually exercised it --
        the earlier single-request-only directed unit tests never had
        more than one active request at a time and so never surfaced
        this.

        2026-08-06 bug fix (found via a real 2-process N=2 test after
        the admission-race fix -- see docs/batched-decode-n2-
        admission-handoff-2026-08-05.md): sort key changed from
        REQUEST_ID (``sorted(self._requests.items())``, sorts by dict
        key = request_id) to CACHE_SLOT. The physical
        ``batched_cache`` row order (built by
        ``merge_request_caches`` in pp_batched_decode_runtime.py) and
        ``BatchedDecodeSession.prepare_step()``'s own token-tensor row
        order are BOTH cache_slot-ordered -- that ordering is a real,
        physical constraint (merge_request_caches concatenates rows in
        cache_slot order; the batched input tensor must be
        row-aligned with the batched cache), not a convention that can
        be changed on that side. This function's own OLD
        request_id-based order was purely a convention with no
        physical constraint behind it (confirmed: every consumer of
        ``StepMessage.entries``'s order -- ``RankOneMirror
        .validate_step``'s duplicate-slot check, ``RankOneMirrorDriver
        .on_step_message``'s assign/advance loop -- keys off each
        entry's own ``cache_slot``/``request_id`` fields, never
        positional index, so realigning this side's sort key is safe
        for every existing consumer). Under real N=2 slot reuse (a
        NEW, higher request_id admitted into a LOWER, just-freed
        cache_slot), request_id order and cache_slot order diverge --
        ``BatchedDecodeSession.prepare_step()``'s own defensive check
        (``ctx.request_uids != active_ids``) caught this divergence
        immediately the first time a real test exercised it: "order
        (2, 3) does not match ... (3, 2)".
        """
        return tuple(
            BatchEntry(
                request_id=rid,
                cache_slot=rec.cache_slot,
                phase=rec.phase,
                expected_cache_len=rec.cache_len,
                n_tokens=1 if rid in advancing_request_ids else 0,
            )
            for rid, rec in sorted(
                self._requests.items(), key=lambda kv: kv[1].cache_slot
            )
        )

    def handle(self, event: Event) -> list[Command]:
        match event:
            case NewRequestEvent(request_id=rid, cache_slot=slot):
                return self._handle_new_request(rid, slot)
            case TokenGeneratedEvent(request_ids=rids):
                return self._handle_tokens_generated(rids)
            case RequestDoneEvent(request_id=rid) | RequestAbortedEvent(request_id=rid):
                return self._handle_evict_request(rid)
            case EvictAckReceivedEvent(request_id=rid, cache_slot=slot):
                return self._handle_evict_ack(rid, slot)
        # Exhaustive match over the Event union above; unreachable in
        # practice, kept as a fail-stop guard rather than falling
        # through silently if the union is ever extended without
        # updating this match.
        raise ProtocolViolationError(  # pragma: no cover
            f"SchedulerCore.handle: unrecognized event type {type(event)!r}"
        )

    def _handle_new_request(self, request_id: int, cache_slot: int) -> list[Command]:
        if request_id in self._requests:
            raise ProtocolViolationError(
                f"NewRequestEvent for request_id={request_id} which is "
                f"ALREADY active (slot={self._requests[request_id].cache_slot}) "
                f"-- duplicate request_id, refusing to silently overwrite"
            )
        current_state = self._slot_state.get(cache_slot, SlotState.FREE)
        if current_state != SlotState.FREE:
            raise ProtocolViolationError(
                f"NewRequestEvent targets cache_slot={cache_slot} which is "
                f"{current_state.name}, not FREE -- this is EXACTLY the "
                f"slot-reuse-before-eviction-ack race this module exists "
                f"to prevent (see module docstring point 4); refusing to "
                f"assign a request to a slot that hasn't been evicted yet"
            )
        if len(self._requests) >= self.max_concurrency:
            raise ProtocolViolationError(
                f"NewRequestEvent would exceed max_concurrency="
                f"{self.max_concurrency} (currently {len(self._requests)} "
                f"active) -- N>{self.max_concurrency} concurrency is "
                f"explicitly out of scope for this design "
                f"(design doc Section 10)"
            )
        self._requests[request_id] = _RequestRecord(
            cache_slot=cache_slot,
            # Phase-1 scope: every request modeled by this module begins
            # already-prefilled, in DECODE -- see module docstring's
            # Scope note.
            phase=Phase.DECODE,
            cache_len=0,
        )
        self._slot_state[cache_slot] = SlotState.ACTIVE
        return [
            SendStepCommand(
                StepMessage(
                    step_id=self._next_step_id(),
                    entries=self._active_batch_entries(),
                )
            )
        ]

    def _handle_tokens_generated(self, request_ids: tuple[int, ...]) -> list[Command]:
        """Handle a single real (possibly batched) decode step -- EVERY
        request in ``request_ids`` advanced by exactly 1 token as part
        of the SAME real forward pass (see ``TokenGeneratedEvent``'s
        docstring). Validates ALL requests before mutating ANY state,
        so a bad request_id in the middle of a batch can't leave some
        requests advanced and others not (partial-application would be
        its own silent-corruption surface -- exactly the class of bug
        this module's fail-stop design exists to avoid)."""
        missing = [rid for rid in request_ids if rid not in self._requests]
        if missing:
            raise ProtocolViolationError(
                f"TokenGeneratedEvent for request_ids={missing} which "
                f"{'are' if len(missing) > 1 else 'is'} not active -- "
                f"stale/duplicate event, refusing to process (this would "
                f"otherwise silently create a phantom cache-length "
                f"increment for a slot no request owns)"
            )
        for rid in request_ids:
            rec = self._requests[rid]
            rec.cache_len = known_len_advance(rec.cache_len, n_tokens=1)
        return [
            SendStepCommand(
                StepMessage(
                    step_id=self._next_step_id(),
                    entries=self._active_batch_entries(
                        advancing_request_ids=frozenset(request_ids)
                    ),
                )
            )
        ]

    def _handle_evict_request(self, request_id: int) -> list[Command]:
        rec = self._requests.pop(request_id, None)
        if rec is None:
            raise ProtocolViolationError(
                f"RequestDoneEvent/RequestAbortedEvent for "
                f"request_id={request_id} which is not active -- "
                f"stale/duplicate eviction event, refusing to process"
            )
        self._slot_state[rec.cache_slot] = SlotState.DRAINING
        return [
            SendEvictCommand(
                EvictMessage(
                    step_id=self._next_step_id(),
                    request_id=request_id,
                    cache_slot=rec.cache_slot,
                )
            )
        ]

    def _handle_evict_ack(self, request_id: int, cache_slot: int) -> list[Command]:
        current_state = self._slot_state.get(cache_slot)
        if current_state != SlotState.DRAINING:
            raise ProtocolViolationError(
                f"EvictAckReceivedEvent for cache_slot={cache_slot} "
                f"(request_id={request_id}) but slot is "
                f"{current_state.name if current_state else 'UNKNOWN'}, "
                f"not DRAINING -- unexpected/duplicate ack, refusing to "
                f"free a slot that was never (or already) evicted"
            )
        self._slot_state[cache_slot] = SlotState.FREE
        return []


class RankOneMirror:
    """Rank 1's independent validator, running the SAME invariants as
    ``SchedulerCore`` but purely REACTIVELY against incoming wire
    messages (module docstring point 2) -- it never initiates a
    decision, only validates rank 0's claims against its own
    independently-tracked slot state and raises immediately on any
    mismatch (module docstring point 3: fail-stop, never repair).

    This is deliberately a SEPARATE class from ``SchedulerCore``, not
    the same class reused, even though the state each tracks overlaps
    heavily -- rank 0 DECIDES batch composition, rank 1 only ever
    VALIDATES what it's told. Collapsing them into one class would
    risk rank 1 silently trusting rank 0's claims rather than
    independently re-deriving and cross-checking them, which is the
    exact "auto-correction hides corruption" failure mode the consult
    review warned against.
    """

    def __init__(self) -> None:
        self._slot_state: dict[int, SlotState] = {}
        self._slot_cache_len: dict[int, int] = {}
        self._last_step_id: int = 0

    def _check_step_id(self, step_id: int) -> None:
        if step_id != self._last_step_id + 1:
            raise ProtocolViolationError(
                f"step_id={step_id} is not the expected next step "
                f"{self._last_step_id + 1} -- duplicate, skipped, or "
                f"out-of-order step message. Even though the underlying "
                f"transport is ordered, this is checked independently "
                f"because it catches SCHEDULER logic bugs, a different "
                f"class of bug than a transport bug (see module "
                f"docstring point 5)"
            )
        self._last_step_id = step_id

    def validate_step(self, message: StepMessage) -> None:
        self._check_step_id(message.step_id)
        seen_slots: set[int] = set()
        for entry in message.entries:
            if entry.cache_slot in seen_slots:
                raise ProtocolViolationError(
                    f"StepMessage step_id={message.step_id} references "
                    f"cache_slot={entry.cache_slot} MORE THAN ONCE in the "
                    f"same step -- two requests claiming the same slot "
                    f"in one batch is exactly the cross-request-corruption "
                    f"shape this module exists to catch"
                )
            seen_slots.add(entry.cache_slot)
            slot_state = self._slot_state.get(entry.cache_slot, SlotState.FREE)
            if slot_state == SlotState.DRAINING:
                raise ProtocolViolationError(
                    f"StepMessage step_id={message.step_id} schedules "
                    f"request_id={entry.request_id} on cache_slot="
                    f"{entry.cache_slot} which is DRAINING (eviction "
                    f"in flight, not yet acked) -- THIS IS THE #1 "
                    f"corruption vector this module was built to prevent: "
                    f"a stale in-flight microbatch entry running into a "
                    f"slot that's being freed. Refusing to process"
                )
            known_len = self._slot_cache_len.get(entry.cache_slot)
            if slot_state == SlotState.FREE:
                # First time this rank has seen this slot scheduled --
                # rank 0's claimed expected_cache_len becomes the
                # baseline (Phase-1 scope: always 0, since every
                # request modeled here starts already-prefilled with an
                # empty per-rank decode cache -- see module docstring's
                # Scope note. A nonzero baseline here would indicate a
                # scheduling bug in Phase-1 scope, not a real prefilled
                # cache length, since this mirror has no prefill state
                # to compare against yet).
                if entry.expected_cache_len != 0:
                    raise ProtocolViolationError(
                        f"StepMessage step_id={message.step_id} claims "
                        f"expected_cache_len={entry.expected_cache_len} for "
                        f"NEWLY-scheduled cache_slot={entry.cache_slot} "
                        f"(request_id={entry.request_id}), but this mirror "
                        f"has no prior cache state for this slot -- Phase-1 "
                        f"scope requires new requests to start with an "
                        f"empty per-rank decode cache (already prefilled "
                        f"elsewhere); a nonzero claim here indicates a "
                        f"scheduler bug, not a legitimate prefilled length"
                    )
                self._slot_state[entry.cache_slot] = SlotState.ACTIVE
                self._slot_cache_len[entry.cache_slot] = 0
            else:
                if known_len is None:
                    # Invariant violation, not a user-triggerable
                    # condition: any slot NOT in FREE state (i.e.
                    # ACTIVE or already excluded via the DRAINING check
                    # above) must always have a tracked cache length --
                    # this branch existing is a bug in this module
                    # itself, not a malformed message from rank 0.
                    raise ProtocolViolationError(
                        f"internal invariant violation: cache_slot="
                        f"{entry.cache_slot} has slot_state={slot_state.name} "
                        f"but no tracked cache length -- this is a bug in "
                        f"RankOneMirror itself, not a rejected message"
                    )
                if (
                    known_len_advance(known_len, entry.n_tokens)
                    != entry.expected_cache_len
                ):
                    raise ProtocolViolationError(
                        f"StepMessage step_id={message.step_id} claims "
                        f"expected_cache_len={entry.expected_cache_len} "
                        f"(n_tokens={entry.n_tokens}) for cache_slot="
                        f"{entry.cache_slot} (request_id={entry.request_id}), "
                        f"but this mirror's own tracked cache length is "
                        f"{known_len} (+{entry.n_tokens} would be "
                        f"{known_len_advance(known_len, entry.n_tokens)}) -- exactly the 'off-by-"
                        f"one KV length from mixed prefill/decode batches' "
                        f"silent-corruption case flagged by the consult "
                        f"review; refusing to process a step whose claimed "
                        f"state disagrees with this rank's own ground truth"
                    )
            # ``entry.expected_cache_len`` IS the ground-truth length
            # AFTER this step's token(s) are applied (validated above
            # against this mirror's own prior tracked length + the
            # claimed advance). Setting it here directly (not re-adding
            # ``entry.n_tokens`` again) is what keeps this mirror's
            # ground truth in sync with rank 0's, since the addition
            # already happened conceptually in the comparison above.
            self._slot_cache_len[entry.cache_slot] = entry.expected_cache_len

    def validate_evict(self, message: EvictMessage) -> None:
        self._check_step_id(message.step_id)
        slot_state = self._slot_state.get(message.cache_slot)
        if slot_state != SlotState.ACTIVE:
            raise ProtocolViolationError(
                f"EvictMessage step_id={message.step_id} targets "
                f"cache_slot={message.cache_slot} which is "
                f"{slot_state.name if slot_state else 'UNKNOWN'}, not "
                f"ACTIVE -- cannot evict a slot that isn't currently "
                f"occupied by an active request"
            )
        self._slot_state[message.cache_slot] = SlotState.DRAINING

    def build_evict_ack(self, message: EvictMessage) -> EvictAckMessage:
        """Call AFTER ``validate_evict`` has accepted the eviction and
        this rank has actually freed its cache state for the slot --
        transitions the mirror's own view to FREE and returns the ack
        message the real transport layer would send back to rank 0."""
        slot_state = self._slot_state.get(message.cache_slot)
        if slot_state != SlotState.DRAINING:
            raise ProtocolViolationError(
                f"build_evict_ack called for cache_slot={message.cache_slot} "
                f"which is {slot_state.name if slot_state else 'UNKNOWN'}, "
                f"not DRAINING -- call validate_evict first"
            )
        self._slot_state[message.cache_slot] = SlotState.FREE
        del self._slot_cache_len[message.cache_slot]
        return EvictAckMessage(
            step_id=message.step_id,
            request_id=message.request_id,
            cache_slot=message.cache_slot,
        )


def known_len_advance(current: int, n_tokens: int) -> int:
    """Trivial helper kept as a standalone function (not inlined) so a
    future Phase-2+ change to how cache length advances under chunked
    prefill (n_tokens > 1) has exactly one place to change, rather than
    needing to hunt for every inline `+= n_tokens` across this module.
    """
    return current + n_tokens

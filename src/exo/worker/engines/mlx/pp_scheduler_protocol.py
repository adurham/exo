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

Scope: this module originally implemented ONLY Phase 1's simplest case
— 2 concurrent DECODE-ONLY requests (both already prefilled via today's
existing serial PP prefill), NO speculative decode. Phase 1's own scope
note (kept below for historical record) said ``Phase.PREFILL`` existed
only as a forward-compatible placeholder. **2026-08-06 (Phase 2
scoping session): chunked-prefill support was added to this pure
protocol layer** -- ``NewChunkedPrefillRequestEvent``/
``PrefillChunkAdvancedEvent`` let a request begin in PREFILL and
advance by >1 token per chunk, with ``RankOneMirror`` INDEPENDENTLY
DERIVING when a request's prefill is complete (comparing its own
tracked ``cache_len`` against a ``total_prompt_tokens`` value recorded
via ``record_prefill_admission`` -- mirroring how a real
``PrefillMessage.n_prompt_tokens`` would be consumed) rather than
trusting a caller-claimed "this is the final chunk" flag -- this is
the SAME "rank 1 never trusts, always independently validates"
discipline every other invariant in this module already follows (see
point 2 above). **Deliberately still OUT of scope, same session:** the
actual MLX model-forward layer-segmentation surgery (driving a real
model's per-layer loop from outside to yield between segments and
interleave a real decode step) -- that is real, model-specific code
against `generate.py`'s forward-pass internals, not a protocol-layer
change, and is a separate, larger, not-yet-started piece of work (see
the design doc's own Phase 2 entries for the real hardware
measurements/estimates that inform its later sizing). This module's
job is only to make the WIRE PROTOCOL correctly representable and
independently verifiable for chunked admission -- it says nothing about
how or when a real forward pass gets interrupted; that's a purely LOCAL
scheduling decision on rank 0's side, invisible to this protocol by
design (an interleaved decode step is already its own ordinary
``StepMessage``; where rank 0 chose to yield within its own layer loop
to produce it is not this protocol's concern). Old Phase-1-era note,
still accurate for the ORIGINAL decode-only case: this module's
original scope only ever constructed requests that begin already in
DECODE, mirroring "both
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
class PrefillAdvanceMessage:
    """Rank 0's per-tick "advance your OWN local chunked-prefill
    session forward, do NOT run a decode step this tick" instruction
    to rank 1 -- 2026-08-06, Phase 2 Stage 3 (chunked-prefill layer-
    segmentation). A distinct, NEW message kind, not folded into
    ``StepMessage`` or piggybacked onto ``PrefillChunkAdvancedEvent``
    -- per a `consult` review of this exact design question.

    WHY A NEW MESSAGE KIND (not reused/repurposed existing ones):
    ``tick()``'s own established invariant is "one action per tick,
    and the MESSAGE KIND received IS the instruction for what rank 1's
    tick does" -- ``StepMessage`` means "this tick is a decode step",
    ``PrefillMessage`` means "this tick is an admission handshake".
    Advancing a chunked-prefill session by a real layer-segment is a
    THIRD variant of "what happens this tick", so it needs a THIRD
    message kind for the same reason the first two are distinct: a
    degenerate empty-batch ``StepMessage`` sent on an advance-only tick
    would drag ``BatchedDecodeSession``'s decode-step code path into a
    tick where it must never run -- exactly the "mixed prefill/decode
    wire traffic within one rank's synchronous loop" conflation the
    original N=2 admission-race campaign closed for chunk-vs-decode
    timing; this closes the identical class of bug one level deeper
    (LAYER-segment-vs-decode timing, not chunk-vs-decode).

    WHY ``PrefillChunkAdvancedEvent`` (``pp_scheduler_protocol.py``,
    already built/tested) is NOT reused for this: that event fires
    once per COMPLETED CHUNK (e.g. every ~2048 tokens); a single chunk
    is advanced across MANY real layer-segment ``advance()`` calls
    (tens of them, at 2-3 layers per call against a ~20-layer per-rank
    stack) -- a strictly FINER granularity than chunk completion. Using
    the coarse event for the fine-grained lockstep signal would mean
    rank 1 only learns to advance once per chunk, not once per real
    ``tick()``-driven layer-segment -- rank 0 and rank 1 would
    desynchronize their per-segment collective timing immediately.
    ``PrefillChunkAdvancedEvent`` keeps its existing job (protocol-
    level bookkeeping: cache_len tracking, PREFILL->DECODE derivation)
    completely unchanged; this message is a per-tick WIRE COMMAND, a
    different layer with a different job.

    Fields:
      ``step_id``: the usual lockstep monotonic step counter shared
        with every other control message kind (module docstring point
        5) -- an advance consumes a step id exactly like a decode step
        or admission does, so a skipped/duplicated advance is caught
        by the same cheap tripwire.
      ``request_id``: which request's session to advance -- at the
        confirmed "at most ONE request mid-prefill at a time" scope
        (design doc's Phase 2 entry), this is redundant with "the one
        active session," but carried explicitly anyway so a future
        relaxation of that scope constraint doesn't need a wire-shape
        change, and so a mismatch is a loud, cheap tripwire rather
        than an implicit assumption.
      ``advance_seq``: monotonic PER-REQUEST advance counter (distinct
        from ``step_id``, which is shared/global across ALL message
        kinds) -- per a `consult` review's explicit recommendation: a
        cheap sequence-number check on rank 1 turns a real desync from
        a silent multi-tick-later jaccl/RDMA hang into an IMMEDIATE,
        loud assertion at the exact point divergence began. Starts at
        1 for a request's first advance (mirrors ``step_id``'s own
        1-based convention).
      ``max_layers``: the layer-segment size for THIS advance --
        deliberately RANK-AGNOSTIC (a count, not a specific layer
        index), since rank 0 and rank 1 hold potentially DIFFERENT
        per-rank layer counts (an uneven PP split -- CONFIRMED real
        on the real 43-layer DSv4-Flash topology: exo's own
        placement layer allocates layers per-node by MEMORY WEIGHT,
        ``src/exo/master/placement_utils.py``'s
        ``_allocate_and_validate_layers``, not an even split formula)
        and each rank maps this count to its own local
        ``ResumablePrefillSession.advance(max_layers=...)`` call
        independently -- this message tells rank 1 HOW MANY of its
        own local layers to advance by, not WHICH global layer index
        to reach.
      ``chunk_index``: 2026-08-06 (Phase 2 chunk-completion-barrier
        fix -- see ``PrefillAdvanceAckMessage``'s own docstring for
        the full incident this closes). Which REAL CHUNK this advance
        belongs to, 0-based, monotonic per request. Because ranks can
        hold DIFFERENT layer counts (confirmed above), one rank can
        reach its own ``ResumablePrefillSession`` completion for a
        chunk in FEWER real advances than its peer -- this field is
        what lets a rank tell "an advance for a chunk I've already
        finished locally, from BEFORE the two-sided completion
        barrier cleared" apart from "a genuinely new chunk's first
        advance," which ``advance_seq`` alone (monotonic but chunk-
        agnostic) cannot distinguish.
    """

    step_id: int
    request_id: int
    advance_seq: int
    max_layers: int
    chunk_index: int

    def __post_init__(self) -> None:
        if self.advance_seq < 1:
            raise ProtocolViolationError(
                f"PrefillAdvanceMessage.advance_seq={self.advance_seq} must "
                f"be >=1 -- a zero/negative advance sequence number is not "
                f"a real occurrence this message should represent"
            )
        if self.max_layers < 1:
            raise ProtocolViolationError(
                f"PrefillAdvanceMessage.max_layers={self.max_layers} must "
                f"be >=1 -- a zero/negative-layer advance is not a real "
                f"occurrence this message should represent"
            )
        if self.chunk_index < 0:
            raise ProtocolViolationError(
                f"PrefillAdvanceMessage.chunk_index={self.chunk_index} must "
                f"be >=0 -- a negative chunk index is not a real "
                f"occurrence this message should represent"
            )


@dataclass(frozen=True)
class PrefillAbortMessage:
    """Rank 0's instruction to rank 1: "abandon your own local
    chunked-prefill session for ``request_id`` right now, do not
    expect any further ``PrefillAdvanceMessage`` for it" --
    2026-08-07, real cancel/abort mechanism (closes the fail-stop-only
    gap ``ExoBatchGenerator.cancel()`` previously guarded against
    rather than handled).

    WHY A NEW MESSAGE KIND (not reused/repurposed existing ones):
    mirrors ``PrefillAdvanceMessage``'s own "one action per tick, the
    MESSAGE KIND received IS the instruction" reasoning exactly --
    an abort is a FOURTH distinct thing that can happen on a
    chunk-drive tick (alongside decode/admission/advance), so it gets
    its own kind rather than overloading ``PrefillAdvanceMessage``
    with an abort flag (a `consult` review explicitly flagged
    overloading advance semantics as the wrong move: an abort-flagged
    "advance" would still need ``max_layers``/``chunk_index`` fields
    that mean nothing for an abort, and every existing
    ``PrefillAdvanceMessage`` reader would need new branching to
    ignore them).

    WHY THIS NEEDS A BLOCKING ACK (``PrefillAbortAckMessage``, not
    fire-and-forget): mirrors ``EvictMessage``/``EvictAckMessage``'s
    own DRAINING-until-ack contract exactly, for the identical reason
    -- rank 0 must not reuse its own ``_active_prefill_session`` slot
    (register a NEW session, or grant a new request) until it has
    PROOF rank 1 has genuinely freed its own mirrored slot, or the
    exact wedge class the priority-order guard was built to prevent
    for the grant path recurs here: rank 0 races ahead locally while
    rank 1's glue is still occupied by state nothing will ever advance
    again.

    Only valid to send once rank 1 is CONFIRMED to hold live session
    state for ``request_id`` -- i.e. at least one real
    ``PrefillAdvanceMessage`` has already been sent for it (see
    ``ResumablePrefillSession.abort()``'s own docstring and
    ``Rank0BatchedDecodeGlue``'s cancel-handling call site for the
    full "local-only vs needs-wire-abort" decision this message exists
    downstream of -- an abort BEFORE rank 1 has ever registered a
    session for this request is a pure local no-op on rank 0's side,
    no message needed, since rank 1 holds zero state to free).
    """

    step_id: int
    request_id: int


@dataclass(frozen=True)
class PrefillAbortAckMessage:
    """Rank 1's acknowledgement that it has genuinely closed its own
    local ``ResumablePrefillSession`` for ``request_id`` (via
    ``ResumablePrefillSession.abort()``, routed through the session's
    own captured ``contextvars.Context`` -- never a raw
    ``._gen.close()`` call, see that method's own docstring) and
    cleared its glue's ``_active_prefill_session``/
    ``_last_prefill_advance_seq`` bookkeeping. Only after THIS is
    received may rank 0 register a new session or grant a new
    request -- mirrors ``EvictAckMessage``'s own "transition back to
    FREE only after ack" contract exactly.
    """

    step_id: int
    request_id: int


@dataclass(frozen=True)
class PrefillChunkDoneMessage:
    """Rank 0's notice to rank 1: "I have SENT all of chunk N's
    advances -- reply once you have genuinely FINISHED PROCESSING
    them (your own local ``ResumablePrefillSession.advance()`` calls
    for this chunk's tail layers have actually completed), so I can
    safely register chunk N+1" -- 2026-08-08, real production
    incident fix (see design doc Section 21 for the full incident).

    WHY THIS EXISTS (a genuine gap tonight's stale-message-seq fix did
    NOT close): rank 0's own "chunk complete" decision
    (``_prefill_rank1_advances_remaining == 0``) is a pure LOCAL
    send-count decrement -- it means "I have sent 11 advances", not
    "rank 1 has finished computing on them". Rank 1's own local
    compute for a chunk's tail layers (the real Metal forward-pass
    work inside ``ResumablePrefillSession.advance()``) can genuinely
    still be running when rank 0's last advance SEND completes --
    confirmed on real hardware via a 14-second Metal ``Event::wait``
    stall on rank 1 immediately after receiving chunk N's final
    advance, during which rank 0 raced ahead: registered chunk N+1
    and began sending ITS advances before rank 1 had even registered
    chunk N+1's session, let alone processed anything for it. Fully
    valid, in-order, correctly-sequenced messages (tonight's earlier
    fix's own seq-tag validation confirmed no transport desync) --
    the actual bug is a missing cross-rank barrier at the chunk
    boundary, a different failure class entirely from a stale/
    duplicate wire message.

    WHY THIS NEEDS A BLOCKING ACK (not fire-and-forget): mirrors
    ``PrefillAbortMessage``/``PrefillAbortAckMessage``'s own
    DRAINING-until-ack contract exactly, for the identical structural
    reason -- rank 0 must not register a NEW session (reuse its own
    ``_active_prefill_session`` slot) until it has PROOF rank 1 has
    genuinely finished with the current one, or the exact class of
    race this message exists to close recurs at the NEXT chunk
    boundary too.

    Deliberately a SEPARATE message kind from ``PrefillAdvanceMessage``
    (not an "advance with a done flag") for the same reason
    ``PrefillAbortMessage`` is separate from it: a fire-and-forget
    advance and a genuinely-must-block completion notice are different
    enough contracts that overloading one message shape to carry both
    makes every reader branch on a flag that means nothing for the
    other case.
    """

    step_id: int
    request_id: int
    chunk_index: int


@dataclass(frozen=True)
class PrefillChunkDoneAckMessage:
    """Rank 1's acknowledgement that it has genuinely finished
    processing chunk ``chunk_index`` (its own local
    ``ResumablePrefillSession.advance()`` calls for this chunk's tail
    layers have actually completed -- not merely that it received all
    the wire messages). Only after THIS is received may rank 0
    register chunk ``chunk_index + 1``'s session -- mirrors
    ``PrefillAbortAckMessage``'s own "transition back to registerable
    only after ack" contract exactly. See ``PrefillChunkDoneMessage``'s
    own docstring for the full incident this closes.
    """

    step_id: int
    request_id: int
    chunk_index: int


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
class NewChunkedPrefillRequestEvent:
    """A new request has arrived and been assigned a cache slot, but is
    NOT yet prefilled -- it begins in ``Phase.PREFILL`` with
    ``cache_len=0`` (2026-08-06, Phase 2 scoping session -- see module
    docstring's Phase 2 addendum). ``total_prompt_tokens`` is recorded
    so ``RankOneMirror`` can independently derive when this request's
    prefill is COMPLETE (``cache_len == total_prompt_tokens``) rather
    than trusting a caller-supplied "this is the final chunk" claim --
    the same "rank 1 never trusts, always independently re-derives"
    discipline every other invariant in this module already follows.
    Mirrors what a real ``PrefillMessage.n_prompt_tokens`` field would
    carry over the wire at admission time.
    """

    request_id: int
    cache_slot: int
    total_prompt_tokens: int

    def __post_init__(self) -> None:
        if self.total_prompt_tokens < 1:
            raise ProtocolViolationError(
                f"NewChunkedPrefillRequestEvent.total_prompt_tokens="
                f"{self.total_prompt_tokens} must be >=1 -- a prefill "
                f"request with zero or negative prompt length is not a "
                f"real occurrence this event should represent"
            )


@dataclass(frozen=True)
class PrefillChunkAdvancedEvent:
    """One real prefill chunk completed for ``request_id`` -- its cache
    advanced by ``n_tokens_this_chunk`` (NOT always 1, unlike
    ``TokenGeneratedEvent`` -- a chunk may cover many tokens, per
    ``EXO_PREFILL_STEP_SIZE``). Deliberately a SINGLE-request event,
    not a tuple like ``TokenGeneratedEvent`` -- per this session's
    scope decision (design doc's Phase 2 entry, "at most ONE request
    mid-prefill at a time"), only one request is ever mid-chunked-
    prefill at once, so there is no multi-request atomicity concern to
    preserve here the way there is for real batched decode steps.

    Does NOT itself claim whether this is the final chunk -- see
    ``NewChunkedPrefillRequestEvent``'s docstring: finality is DERIVED
    by ``RankOneMirror`` from ``cache_len == total_prompt_tokens``,
    never asserted by the event. ``SchedulerCore`` derives it
    identically from its own tracked state for the same reason (so
    both ranks compute the SAME derived fact from the SAME kind of
    ground truth, not one side trusting the other's claim).
    """

    request_id: int
    n_tokens_this_chunk: int

    def __post_init__(self) -> None:
        if self.n_tokens_this_chunk < 1:
            raise ProtocolViolationError(
                f"PrefillChunkAdvancedEvent.n_tokens_this_chunk="
                f"{self.n_tokens_this_chunk} must be >=1 -- a chunk "
                f"advancing zero tokens is not a real occurrence this "
                f"event should represent"
            )


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
    | NewChunkedPrefillRequestEvent
    | TokenGeneratedEvent
    | PrefillChunkAdvancedEvent
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
    # None for a request admitted via NewRequestEvent (Phase-1's
    # original decode-only path, no prefill tracked here at all).
    # Set for a request admitted via NewChunkedPrefillRequestEvent --
    # the ground truth SchedulerCore checks its OWN prefill-advance
    # bookkeeping against (module docstring's Phase 2 addendum;
    # RankOneMirror tracks the identical fact independently via
    # ``record_prefill_admission``, never trusting this side's claim).
    total_prompt_tokens: int | None = None


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
        self, *, advancing: dict[int, int] | None = None
    ) -> tuple[BatchEntry, ...]:
        """Snapshot the full current active set for rank 1's routing.

        ``advancing``: maps request_id -> n_tokens for requests whose
        state actually changed as a result of THIS ``handle()`` call.
        Every OTHER active request co-listed in the same snapshot gets
        ``n_tokens=0`` -- being included in a step message for rank 1's
        bookkeeping does NOT mean every co-listed request generated a
        token/chunk-advance in lockstep with this specific event; only
        the ones actually named here did. Empty/``None`` (the default)
        means no request advanced this call (e.g. a brand new request
        just joined at its baseline cache_len=0, or the batch
        composition changed for bookkeeping reasons only).

        2026-08-06 (Phase 2 scoping session): generalized from
        ``advancing_request_ids: frozenset[int]`` (every advance always
        exactly 1 token, Phase-1's decode-only assumption) to this
        ``dict[int, int]`` shape so a single prefill CHUNK advance
        (``n_tokens_this_chunk`` -- may be many tokens, per
        ``EXO_PREFILL_STEP_SIZE``) and a decode-step advance (always
        exactly 1 token) can both flow through the SAME snapshot method
        without a parallel near-duplicate implementation. Every
        existing call site updated accordingly; no behavior change for
        the decode-only case (``{rid: 1 for rid in request_ids}`` is
        exactly the old frozenset-of-1s semantics).

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
        advancing = advancing or {}
        return tuple(
            BatchEntry(
                request_id=rid,
                cache_slot=rec.cache_slot,
                phase=rec.phase,
                expected_cache_len=rec.cache_len,
                n_tokens=advancing.get(rid, 0),
            )
            for rid, rec in sorted(
                self._requests.items(), key=lambda kv: kv[1].cache_slot
            )
        )

    def handle(self, event: Event) -> list[Command]:
        match event:
            case NewRequestEvent(request_id=rid, cache_slot=slot):
                return self._handle_new_request(rid, slot)
            case NewChunkedPrefillRequestEvent(
                request_id=rid, cache_slot=slot, total_prompt_tokens=total
            ):
                return self._handle_new_chunked_prefill_request(rid, slot, total)
            case TokenGeneratedEvent(request_ids=rids):
                return self._handle_tokens_generated(rids)
            case PrefillChunkAdvancedEvent(
                request_id=rid, n_tokens_this_chunk=n_tokens
            ):
                return self._handle_prefill_chunk_advanced(rid, n_tokens)
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

    def _handle_new_chunked_prefill_request(
        self, request_id: int, cache_slot: int, total_prompt_tokens: int
    ) -> list[Command]:
        """2026-08-06 (Phase 2 scoping session). Same slot-exclusivity/
        duplicate-id/max-concurrency invariants as ``_handle_new_request``
        (kept fully duplicated rather than factored into a shared helper
        with a phase parameter -- this module's own established style,
        see e.g. ``RequestDoneEvent``/``RequestAbortedEvent`` sharing one
        handler ONLY because their semantics are truly identical; here
        the two admission paths' error messages and PHASE OUTCOME
        genuinely differ, so keeping them as textually separate, easily-
        greppable methods is preferred over a shared helper with a
        branch inside it).

        Per module docstring's Phase 2 addendum and the design doc's
        own "at most ONE request mid-prefill at a time" scope decision:
        this does NOT enforce single-concurrent-prefill here -- that is
        a scheduling POLICY decision for the real ``tick()`` caller
        (which event to feed this core, and when), not a protocol
        INVARIANT this pure core should hard-enforce. A future policy
        change to allow N>1 concurrent prefills would not need to touch
        this module at all; deliberately kept orthogonal.
        """
        if request_id in self._requests:
            raise ProtocolViolationError(
                f"NewChunkedPrefillRequestEvent for request_id={request_id} "
                f"which is ALREADY active (slot="
                f"{self._requests[request_id].cache_slot}) -- duplicate "
                f"request_id, refusing to silently overwrite"
            )
        current_state = self._slot_state.get(cache_slot, SlotState.FREE)
        if current_state != SlotState.FREE:
            raise ProtocolViolationError(
                f"NewChunkedPrefillRequestEvent targets cache_slot="
                f"{cache_slot} which is {current_state.name}, not FREE -- "
                f"this is EXACTLY the slot-reuse-before-eviction-ack race "
                f"this module exists to prevent (see module docstring "
                f"point 4); refusing to assign a request to a slot that "
                f"hasn't been evicted yet"
            )
        if len(self._requests) >= self.max_concurrency:
            raise ProtocolViolationError(
                f"NewChunkedPrefillRequestEvent would exceed "
                f"max_concurrency={self.max_concurrency} (currently "
                f"{len(self._requests)} active) -- N>{self.max_concurrency} "
                f"concurrency is explicitly out of scope for this design "
                f"(design doc Section 10)"
            )
        self._requests[request_id] = _RequestRecord(
            cache_slot=cache_slot,
            phase=Phase.PREFILL,
            cache_len=0,
            total_prompt_tokens=total_prompt_tokens,
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
        # 2026-08-06 (Phase 2 scoping session): a request currently in
        # PREFILL phase cannot ALSO be advancing via a real decode step
        # -- these are mutually exclusive phases for one request at one
        # instant. Checked BEFORE mutating any state (same "validate
        # everything, then mutate" discipline as the missing-id check
        # above), and named explicitly per-request so the error is
        # actionable rather than a generic assertion failure.
        still_prefilling = [
            rid for rid in request_ids if self._requests[rid].phase == Phase.PREFILL
        ]
        if still_prefilling:
            raise ProtocolViolationError(
                f"TokenGeneratedEvent for request_ids={still_prefilling} "
                f"which {'are' if len(still_prefilling) > 1 else 'is'} "
                f"still in Phase.PREFILL, not DECODE -- a decode-step "
                f"advance cannot apply to a request still mid-prefill; "
                f"this would silently mix prefill and decode semantics "
                f"for the same request in the same step"
            )
        for rid in request_ids:
            rec = self._requests[rid]
            rec.cache_len = known_len_advance(rec.cache_len, n_tokens=1)
        return [
            SendStepCommand(
                StepMessage(
                    step_id=self._next_step_id(),
                    entries=self._active_batch_entries(
                        advancing={rid: 1 for rid in request_ids}
                    ),
                )
            )
        ]

    def _handle_prefill_chunk_advanced(
        self, request_id: int, n_tokens_this_chunk: int
    ) -> list[Command]:
        """2026-08-06 (Phase 2 scoping session). Advances ``request_id``'s
        prefill cache by ``n_tokens_this_chunk`` and, if that advance
        reaches ``total_prompt_tokens`` exactly, transitions the request
        PREFILL -> DECODE. Finality is DERIVED here (``cache_len ==
        total_prompt_tokens`` after the advance), never asserted by the
        event -- see ``PrefillChunkAdvancedEvent``'s own docstring for
        why (mirrors ``RankOneMirror``'s identical independent
        derivation, so both ranks compute the same fact from the same
        kind of ground truth rather than one side trusting the other's
        claim -- module docstring point 2).

        Per module docstring point 6 (off-by-one tripwire) and the
        design doc's Phase 2 entry's "final-chunk boundary" open item:
        this module's OWN convention is that ``total_prompt_tokens`` is
        the exact count of tokens this request's prefill must advance
        the cache by BEFORE the first real decode step runs -- i.e.
        ``total_prompt_tokens`` already reflects whatever "drop the
        last prompt token" convention the real prefill call site uses
        (``prefill()``'s own ``prompt_tokens[:-1]`` contract), NOT the
        raw prompt length. The caller (a future real ``tick()``
        integration, not yet built) is responsible for computing
        ``total_prompt_tokens`` consistently with that existing
        convention; this module only enforces internal self-consistency
        (the running sum matches what's claimed), not what the number
        SHOULD be relative to the real tokenized prompt -- that's a real
        integration-time cross-check to add when the actual ``tick()``
        wiring is built, not something this zero-I/O core can verify on
        its own.
        """
        rec = self._requests.get(request_id)
        if rec is None:
            raise ProtocolViolationError(
                f"PrefillChunkAdvancedEvent for request_id={request_id} "
                f"which is not active -- stale/duplicate event, refusing "
                f"to process (this would otherwise silently create a "
                f"phantom cache-length increment for a slot no request "
                f"owns)"
            )
        if rec.phase != Phase.PREFILL:
            raise ProtocolViolationError(
                f"PrefillChunkAdvancedEvent for request_id={request_id} "
                f"which is in Phase.{rec.phase.name}, not PREFILL -- a "
                f"chunk advance cannot apply to a request that has "
                f"already finished prefilling (or never started via "
                f"NewChunkedPrefillRequestEvent in the first place)"
            )
        assert rec.total_prompt_tokens is not None, (
            "internal invariant violation: a Phase.PREFILL record must "
            "always carry total_prompt_tokens -- this is a bug in this "
            "module itself (NewChunkedPrefillRequestEvent is the only "
            "path that ever sets phase=PREFILL, and it always sets this "
            "field), not a caller error"
        )
        new_len = known_len_advance(rec.cache_len, n_tokens_this_chunk)
        if new_len > rec.total_prompt_tokens:
            raise ProtocolViolationError(
                f"PrefillChunkAdvancedEvent for request_id={request_id} "
                f"claims n_tokens_this_chunk={n_tokens_this_chunk}, which "
                f"would advance cache_len from {rec.cache_len} to "
                f"{new_len} -- OVERSHOOTING total_prompt_tokens="
                f"{rec.total_prompt_tokens}. A chunk that reads past the "
                f"end of its own claimed prompt length is exactly the "
                f"off-by-one silent-corruption shape module docstring "
                f"point 6 exists to catch"
            )
        rec.cache_len = new_len
        if new_len == rec.total_prompt_tokens:
            # Finality DERIVED here, not asserted by the event -- see
            # this method's own docstring.
            rec.phase = Phase.DECODE
        return [
            SendStepCommand(
                StepMessage(
                    step_id=self._next_step_id(),
                    entries=self._active_batch_entries(
                        advancing={request_id: n_tokens_this_chunk}
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
        # 2026-08-06 (Phase 2 scoping session). Both keyed by cache_slot,
        # populated/cleared in lockstep with ``_slot_cache_len`` (same
        # admission -> ... -> evict-ack lifecycle) so a slot that is
        # EVICTED and REUSED by a later, different request starts this
        # tracking fresh -- Fable's consult-review gap #1: an earlier
        # draft of this design treated "PREFILL -> DECODE, once per
        # SLOT, ever" as the invariant, which is wrong the instant a
        # slot is evicted and reused by a second chunked-prefill
        # request; the real invariant is per-ADMISSION (this dict's
        # lifecycle), not per-slot-forever.
        self._slot_phase: dict[int, Phase] = {}
        self._slot_total_prompt_tokens: dict[int, int] = {}

    def record_prefill_admission(
        self, cache_slot: int, total_prompt_tokens: int
    ) -> None:
        """Call BEFORE the first ``StepMessage`` naming ``cache_slot`` in
        ``Phase.PREFILL`` arrives -- mirrors receiving a real
        ``PrefillMessage.n_prompt_tokens`` over the wire at admission
        time (2026-08-06, Phase 2 scoping session; module docstring's
        Phase 2 addendum). This is the independent ground truth
        ``validate_step`` derives prefill completion FROM -- it is
        NEVER inferred from a caller-claimed "this is the final chunk"
        signal (Fable's consult-review gap #2: trusting such a claim
        would defeat this class's entire "rank 1 never trusts, always
        independently re-derives" purpose -- see class docstring).
        """
        if cache_slot in self._slot_total_prompt_tokens:
            raise ProtocolViolationError(
                f"record_prefill_admission called for cache_slot="
                f"{cache_slot} which ALREADY has a tracked "
                f"total_prompt_tokens={self._slot_total_prompt_tokens[cache_slot]} "
                f"-- duplicate registration, refusing to silently "
                f"overwrite (if this slot was legitimately evicted and "
                f"reused, ``build_evict_ack`` must run first -- it clears "
                f"this tracking as part of the same evict lifecycle "
                f"``_slot_cache_len`` already goes through)"
            )
        if total_prompt_tokens < 1:
            raise ProtocolViolationError(
                f"record_prefill_admission(cache_slot={cache_slot}, "
                f"total_prompt_tokens={total_prompt_tokens}) -- must be "
                f">=1, matching NewChunkedPrefillRequestEvent's own "
                f"validation on rank 0's side"
            )
        self._slot_total_prompt_tokens[cache_slot] = total_prompt_tokens

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
        # 2026-08-06 (Phase 2 scoping session, Fable consult-review gap
        # #4): within ONE StepMessage, an advancing (n_tokens>0) PREFILL
        # entry may never be co-listed with any OTHER advancing entry --
        # per the design doc's "at most ONE request mid-prefill at a
        # time" + "separate alternating steps, not a mixed per-step
        # tensor" scope decisions, a real step is either a pure prefill-
        # chunk step (exactly one advancing PREFILL entry) or a pure
        # decode step (one or more advancing DECODE entries), never
        # both. Collected here, enforced after the per-entry loop below
        # (which still needs to run first to know each entry's
        # DERIVED phase, not just rank 0's claimed one).
        advancing_phases: list[Phase] = []
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
                # baseline (always 0: EITHER Phase-1's legacy decode-
                # only admission, which always starts already-prefilled
                # with an empty per-rank decode cache, OR a fresh
                # Phase-2 chunked-prefill admission, which also always
                # starts at cache_len=0 by construction -- see
                # ``NewChunkedPrefillRequestEvent``. A nonzero baseline
                # here would indicate a scheduling bug either way, not a
                # legitimate prefilled length, since this mirror has no
                # prior cache state to compare against yet).
                if entry.expected_cache_len != 0:
                    raise ProtocolViolationError(
                        f"StepMessage step_id={message.step_id} claims "
                        f"expected_cache_len={entry.expected_cache_len} for "
                        f"NEWLY-scheduled cache_slot={entry.cache_slot} "
                        f"(request_id={entry.request_id}), but this mirror "
                        f"has no prior cache state for this slot -- both "
                        f"Phase-1 decode-only and Phase-2 chunked-prefill "
                        f"admissions always start at cache_len=0; a "
                        f"nonzero claim here indicates a scheduler bug, "
                        f"not a legitimate prefilled length"
                    )
                if entry.phase == Phase.PREFILL and (
                    entry.cache_slot not in self._slot_total_prompt_tokens
                ):
                    raise ProtocolViolationError(
                        f"StepMessage step_id={message.step_id} schedules "
                        f"request_id={entry.request_id} on NEWLY-scheduled "
                        f"cache_slot={entry.cache_slot} in Phase.PREFILL, "
                        f"but this mirror has no total_prompt_tokens on "
                        f"record for this slot -- ``record_prefill_"
                        f"admission`` must be called (mirroring a real "
                        f"``PrefillMessage``'s arrival) BEFORE the first "
                        f"PREFILL-phase StepMessage for a slot, so this "
                        f"mirror has independent ground truth to derive "
                        f"completion from rather than trusting rank 0's "
                        f"claimed phase blindly"
                    )
                self._slot_state[entry.cache_slot] = SlotState.ACTIVE
                self._slot_cache_len[entry.cache_slot] = 0
                self._slot_phase[entry.cache_slot] = entry.phase
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
                # 2026-08-06 (Phase 2 scoping session, Fable consult-
                # review gap #2): DERIVE the phase this entry SHOULD be
                # in, from this mirror's OWN prior-tracked phase + the
                # newly-validated cache length, and cross-check it
                # against rank 0's claimed ``entry.phase`` -- never
                # trust the claim outright. This is what actually gives
                # the phase tracker teeth: a rank-0 bug that flips to
                # DECODE one chunk early (silently dropping prompt
                # tokens) is caught here even though the raw
                # expected_cache_len arithmetic above would pass, because
                # the DERIVED phase and the CLAIMED phase disagree.
                prior_phase = self._slot_phase.get(entry.cache_slot)
                if prior_phase is None:
                    raise ProtocolViolationError(
                        f"internal invariant violation: cache_slot="
                        f"{entry.cache_slot} is ACTIVE but has no tracked "
                        f"phase -- this is a bug in RankOneMirror itself"
                    )
                if prior_phase == Phase.DECODE:
                    derived_phase = Phase.DECODE
                else:
                    total = self._slot_total_prompt_tokens.get(entry.cache_slot)
                    if total is None:
                        raise ProtocolViolationError(
                            f"internal invariant violation: cache_slot="
                            f"{entry.cache_slot} is tracked as Phase.PREFILL "
                            f"but has no total_prompt_tokens on record -- "
                            f"this is a bug in RankOneMirror itself"
                        )
                    derived_phase = (
                        Phase.DECODE
                        if entry.expected_cache_len == total
                        else Phase.PREFILL
                    )
                if derived_phase != entry.phase:
                    raise ProtocolViolationError(
                        f"StepMessage step_id={message.step_id} claims "
                        f"phase=Phase.{entry.phase.name} for cache_slot="
                        f"{entry.cache_slot} (request_id={entry.request_id}), "
                        f"but this mirror INDEPENDENTLY DERIVES "
                        f"Phase.{derived_phase.name} from its own tracked "
                        f"state (prior_phase={prior_phase.name}, "
                        f"expected_cache_len={entry.expected_cache_len} vs "
                        f"total_prompt_tokens="
                        f"{self._slot_total_prompt_tokens.get(entry.cache_slot)}) "
                        f"-- a claimed phase that disagrees with this "
                        f"rank's own independent derivation is exactly the "
                        f"'trusting a caller-claimed final-chunk signal' "
                        f"hazard this design deliberately avoids (see "
                        f"``record_prefill_admission``'s docstring)"
                    )
                self._slot_phase[entry.cache_slot] = derived_phase
            # ``entry.expected_cache_len`` IS the ground-truth length
            # AFTER this step's token(s) are applied (validated above
            # against this mirror's own prior tracked length + the
            # claimed advance). Setting it here directly (not re-adding
            # ``entry.n_tokens`` again) is what keeps this mirror's
            # ground truth in sync with rank 0's, since the addition
            # already happened conceptually in the comparison above.
            self._slot_cache_len[entry.cache_slot] = entry.expected_cache_len
            if entry.n_tokens > 0:
                advancing_phases.append(entry.phase)
        if Phase.PREFILL in advancing_phases and len(advancing_phases) > 1:
            raise ProtocolViolationError(
                f"StepMessage step_id={message.step_id} co-lists an "
                f"ADVANCING Phase.PREFILL entry alongside "
                f"{len(advancing_phases) - 1} other advancing "
                f"entr{'y' if len(advancing_phases) == 2 else 'ies'} in "
                f"the SAME step -- per the design doc's 'separate "
                f"alternating steps, not a mixed per-step tensor' "
                f"scope decision, a real step is either a pure "
                f"prefill-chunk step (exactly one advancing PREFILL "
                f"entry) or a pure decode step (one or more advancing "
                f"DECODE entries), never both in the same step"
            )

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
        message the real transport layer would send back to rank 0.

        2026-08-06 (Phase 2 scoping session, Fable consult-review gap
        #1): also clears ``_slot_phase``/``_slot_total_prompt_tokens``
        for this slot, in the SAME lifecycle step as the pre-existing
        ``_slot_cache_len`` clear -- so a slot legitimately evicted and
        later reused by a DIFFERENT chunked-prefill request starts this
        tracking fresh, rather than a stale phase/total from the
        PREVIOUS occupant leaking into the new admission's validation.
        Uses ``.pop(..., None)`` (not ``del``) for both -- ``_slot_phase``
        is always populated (every admission path sets it in
        ``validate_step``'s FREE branch), but
        ``_slot_total_prompt_tokens`` is populated ONLY for a
        chunked-prefill admission (``record_prefill_admission``) -- a
        decode-only (Phase-1-style) admission never sets it, so evicting
        one must not raise a KeyError.
        """
        slot_state = self._slot_state.get(message.cache_slot)
        if slot_state != SlotState.DRAINING:
            raise ProtocolViolationError(
                f"build_evict_ack called for cache_slot={message.cache_slot} "
                f"which is {slot_state.name if slot_state else 'UNKNOWN'}, "
                f"not DRAINING -- call validate_evict first"
            )
        self._slot_state[message.cache_slot] = SlotState.FREE
        del self._slot_cache_len[message.cache_slot]
        self._slot_phase.pop(message.cache_slot, None)
        self._slot_total_prompt_tokens.pop(message.cache_slot, None)
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

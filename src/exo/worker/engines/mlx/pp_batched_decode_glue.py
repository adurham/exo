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
    a caller uses to fold an ALREADY-PREFILLED request's result into
    the batch -- pure in-memory queueing, zero wire I/O, cannot hang,
    cannot race anything.
  - `Rank0BatchedDecodeGlue.tick(...)` is the ONLY thing that ever
    touches the wire on rank 0 for this session, and it is called
    from EXACTLY ONE place: `ExoBatchGenerator.step()`'s existing
    per-cycle call (mirroring `_step_pp_spec`'s own call site). Each
    `tick()` call does AT MOST ONE of the following, in this fixed
    priority order (see the 2026-08-06 update below for why a THIRD
    branch, prefill announcement, was added ABOVE admission/decode):
    admit exactly one pending, already-prefilled request (if any is
    queued AND its slot is free), or run one real batched decode step
    for the current batch, or announce (via a real `PrefillMessage`
    send) that a new request's prefill may now begin. This keeps
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
    request's local cache half happens ENTIRELY LOCALLY on rank 1,
    but -- see the 2026-08-06 update below -- ONLY once rank 1's own
    `tick()` reactively receives rank 0's `PrefillMessage`, never
    independently. Rank 1's prefilled cache is NEVER sent over the
    wire (only rank 0's admission decision + the resulting
    `StepMessage` cross the wire); this glue just needs a place to
    STAGE that locally-prefilled cache, keyed by `request_id`, until
    the matching admission arrives reactively in a `StepMessage`.

UPDATE (2026-08-06, closes the real N=2 admission-race deadlock
documented in
``docs/batched-decode-n2-admission-handoff-2026-08-05.md`` and
``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 15):
the ORIGINAL design above assumed a request's prefill had ALREADY
happened (via the old, unchanged, per-rank-independent `prefill()`
call inside `submit()`) by the time `enqueue_admission`/
`stage_local_cache` were ever called -- i.e. `submit()` on BOTH ranks
ran its own local prefill, independently, on its own schedule, BEFORE
either side of this glue was ever touched. That is exactly the
uncoordinated cross-rank decision the design doc's Section 15 traced
the real hardware deadlock to: rank 0's `submit()` could start
issuing prefill's metaframe wire traffic in the same window rank 1's
`step()` was still mid-`tick()` issuing decode's `StepMessage`
traffic -- two structurally different wire shapes on the same link,
observed as `[jaccl] reliable_all_reduce_v2 deadline`.

The fix folds "start prefill for request X now" into this SAME
single-writer channel via a new `MSG_KIND_PREFILL` control message
(`PrefillMessage`, `pp_scheduler_wire.py`/`pp_scheduler_protocol.py`):
  - `submit()` no longer runs prefill unconditionally. It calls
    `Rank0BatchedDecodeGlue.enqueue_prefill(...)` (rank 0) -- pure
    in-memory queueing, zero wire I/O, mirroring
    `enqueue_admission`'s own no-I/O guarantee -- and returns without
    prefilling. Rank 1 does NOTHING at `submit()` time for this
    request; it has not yet been told to prefill anything.
  - `tick()` gains a NEW, HIGHEST-priority branch: if a pending
    prefill is queued, `tick()` (a) allocates ("reserves") the
    request's cache slot in the driver's OWN bookkeeping right here,
    atomically with the send, so a LATER `tick()` call in the SAME
    process can never grant a second prefill onto that slot before
    this one's `enqueue_admission` arrives (this is the fix for a
    real gap a `consult` review caught in an earlier draft of this
    design: without reserving the slot immediately, `tick()`'s
    admission branch and its prefill-announcement branch could race
    each other ACROSS iterations, not just across ranks); (b) sends a
    real `PrefillMessage` to rank 1 (the single wire-touching act for
    this branch); (c) returns a `PrefillGrant` telling the CALLER
    (still `ExoBatchGenerator`, one layer up) to now run the real
    prefill forward pass and report back via `enqueue_admission`.
    Because slot reservation happens IN `tick()`, admission (existing
    branch, now second-priority) is checked before a NEW prefill can
    be granted -- in-flight work finishes before new work starts,
    which is both the fix for the slot race and the right operational
    order (never start a heavier operation while cheaper pending work
    could clear a slot first).
  - `Rank1BatchedDecodeGlue.tick()` gains a matching THIRD branch on
    `header.msg_kind`: `MSG_KIND_PREFILL` reactively decodes the
    `PrefillMessage` and returns a `PrefillGrant` of its own. Rank 1
    NEVER independently decides "prefill this now" from its own local
    queue state anymore -- it can only ever learn this from a
    `PrefillGrant` `tick()` itself produced by receiving rank 0's
    message. This is what makes the fix actually close the race:
    both ranks now agree, by construction, on the exact tick boundary
    where a switch from decode-mode to prefill-mode collectives
    happens, because that boundary IS a message in the one ordered
    control stream both ranks already serialize on.
  - `PrefillMessage.flags` carries whether this request is eligible
    for the batched-decode path or must use the OLD single-request
    `MetaFramedPipelineFirstLayer`/`LastLayer` fallback -- rank 1
    reads this from the wire (`PrefillGrant.single_request_fallback`)
    rather than recomputing eligibility itself, because eligibility
    is a rank-0-only decision (it depends on request features rank 1
    never sees) and recomputing it independently on rank 1 would
    reopen exactly the same class of cross-rank-disagreement bug this
    whole message kind exists to close.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import mlx.core as mx
from loguru import logger

from exo.worker.engines.mlx.auto_parallel import flush_prefill_metaframe_sends
from exo.worker.engines.mlx.pp_batched_decode_adapter import (
    AdmitResponse,
    BatchedDecodeResponseAdapter,
    StepResponse,
)
from exo.worker.engines.mlx.pp_batched_decode_runtime import (
    BatchedDecodeSession,
    RankOneMirrorSession,
)
from exo.worker.engines.mlx.pp_metaframe import ForwardPhase
from exo.worker.engines.mlx.pp_prefill_session import ResumablePrefillSession
from exo.worker.engines.mlx.pp_scheduler_protocol import (
    PREFILL_FLAG_SINGLE_REQUEST_FALLBACK,
    PrefillAbortAckMessage,
    PrefillAdvanceMessage,
    PrefillChunkDoneAckMessage,
    PrefillMessage,
    PrefillReadyMessage,
)
from exo.worker.engines.mlx.pp_scheduler_wire import (
    MSG_KIND_EVICT,
    MSG_KIND_PREFILL,
    MSG_KIND_PREFILL_ABORT,
    MSG_KIND_PREFILL_ADVANCE,
    MSG_KIND_STEP,
    decode_evict_message,
    decode_prefill_abort_message,
    recv_header,
    recv_prefill_advance_body,
    recv_prefill_body,
    recv_prefill_chunk_done_ack_message,
    recv_prefill_ready_message,
    recv_step_table,
    send_evict_ack_message,
    send_prefill_abort_ack_message,
    send_prefill_advance_message,
    send_prefill_chunk_done_ack_message,
    send_prefill_message,
    send_prefill_ready_message,
    send_step_message,
)

if TYPE_CHECKING:
    from exo.worker.engines.mlx.pp_batched_decode_runtime import Sampler
    from exo.worker.engines.mlx.types import KVCacheType


class GlueError(RuntimeError):
    """Raised on any glue-layer invariant violation -- fail-stop by
    design, matching this whole session's established discipline."""


def exchange_prefill_peer_layer_count(
    *, local_layer_count: int, dst_rank: int, group: mx.distributed.Group
) -> int:
    """One-time, model-load-time exchange of each rank's REAL local
    layer count for the chunked-prefill completion-barrier fix
    (2026-08-06) -- called ONCE per model load (mirrors
    ``pp_metaframe.handshake_metaframe_protocol``'s own call-once-at-
    warmup discipline), NEVER per-request, and NEVER from inside
    ``tick()`` (this is explicitly NOT part of that method's "ONLY
    call site that ever touches mx.distributed.send/recv_like for
    this session's PER-REQUEST control traffic" invariant -- this
    runs before any session/glue object even exists).

    WHY THIS EXISTS: ``PrefillAdvanceMessage.max_layers`` is rank-
    agnostic (each rank advances its OWN local layer count) because
    the two ranks' real layer counts are CONFIRMED uneven on the real
    43-layer DSv4-Flash 2-rank topology (``src/exo/master/
    placement_utils.py``'s ``_allocate_and_validate_layers`` allocates
    layers per-node by MEMORY WEIGHT, not an even-split formula) --
    22 layers on rank 0, 21 on rank 1 being the real, confirmed
    numbers on the identical-hardware 2-node cluster this project
    targets. Without knowing the PEER's real layer count, rank 0 (the
    sole scheduler -- ``tick()``'s own single-writer design) cannot
    know how many real ``PrefillAdvanceMessage``s it must keep sending
    to fully advance rank 1's session for one chunk, even after rank
    0's OWN local session has already reached completion -- a
    consult-reviewed alternative to a heavier per-advance ack
    round-trip protocol (rejected: doubles wire round-trips on the
    prefill critical path for a purely session-lifetime problem that
    a one-time, load-time exchange closes for free).

    2026-08-07, REAL PRODUCTION INCIDENT, method changed from
    point-to-point send/recv to a scatter+``all_sum`` collective: the
    original point-to-point implementation (bare ``mx.distributed.send``
    immediately followed by ``mx.distributed.recv_like``, mirroring
    ``handshake_metaframe_protocol``'s call-once-at-warmup discipline
    but NOT its actual collective mechanism) deadlocked on real
    2-node hardware -- BOTH ranks call this function with the SAME
    source order (``send`` then ``recv_like``), so both ranks post a
    blocking ``send()`` before either has posted a matching ``recv()``.
    jaccl's reliable-send path has no buffering guarantee for this
    case; both ranks hit jaccl's own 15-second drain deadline
    simultaneously (confirmed via real cluster logs: both ranks raised
    ``[jaccl] send() deadline in drain`` at the identical millisecond).
    The caller's own ``try/except Exception`` (this function's only
    caller, ``ExoBatchGenerator.__post_init__``) caught the timeout and
    fell back to the non-batched path per its own documented contract
    -- but the in-flight, already-posted send on the wire was NOT
    cancelled by that fallback, and its payload (a lone int32, this
    rank's real layer count -- e.g. 22) landed on the SAME untagged
    stream ``recv_metaframe`` reads from, arriving just in time to be
    misconsumed as a corrupt metaframe header's version field by the
    OTHER protocol's very next real recv call during warmup --
    producing the exact ``MetaFrame protocol version mismatch:
    received 22, this build expects 3`` crash seen on the peer rank.

    Fix: use ``all_sum`` (the SAME collective ``handshake_metaframe_
    protocol`` already uses safely at this exact call site, warmup
    time, before ``EXO_PP_NO_COORD_COLLECTIVE`` gating applies to
    per-request traffic) instead of point-to-point send/recv. Each
    rank scatters its own value into its own slot of an otherwise-zero
    ``world_size``-length vector, then ``all_sum`` gives every rank the
    full vector in one synchronized collective call -- no per-rank
    send/recv ordering to get wrong, no possibility of one rank's send
    outliving a caught timeout and corrupting a later, unrelated recv,
    because a collective either fully completes on every rank or the
    SAME exception propagates identically on every rank (no partial/
    one-sided failure mode to begin with). Returns the PEER's local
    layer count -- the caller is responsible for storing it wherever
    the chunk-driving logic needs it (today: ``Rank0BatchedDecodeGlue``'s
    own construction).
    """
    if group.size() <= 1:
        return local_layer_count
    world_size = group.size()
    local_rank = group.rank()
    scatter = mx.zeros((world_size,), dtype=mx.int32)
    scatter[local_rank] = local_layer_count
    summed = mx.distributed.all_sum(scatter, group=group)
    mx.eval(summed)
    peer_layer_count = int(summed[dst_rank].item())
    # TEMP DIAGNOSTIC (2026-08-10, chasing the real mutual-deadlock bug
    # found at call_id=411 -- see design doc Section 39/handoff for the
    # full incident. Hypothesis: local_layer_count (measured via
    # len(get_layers(get_inner_model(model))) at the call site) may not
    # equal the real number of "layer" yields _forward_steps produces
    # for THIS rank's local segment, which would silently desync rank
    # 0's ceil(peer_prefill_layer_count/max_layers) advance budget from
    # what rank 1 actually needs to reach done=True). Remove once
    # root-caused.
    logger.info(
        f"[LAYER_COUNT_EXCHANGE] local_rank={local_rank} "
        f"local_layer_count={local_layer_count} dst_rank={dst_rank} "
        f"peer_layer_count={peer_layer_count}"
    )
    if peer_layer_count < 1:
        raise GlueError(
            f"exchange_prefill_peer_layer_count: received a peer layer "
            f"count of {peer_layer_count} from rank {dst_rank} -- a "
            f"zero/negative layer count is not a real occurrence; the "
            f"two ranks' handshake has desynced or one rank sent "
            f"malformed data. Refusing to proceed with an invalid "
            f"chunk-completion schedule."
        )
    return peer_layer_count


# 2026-08-06 fix for the prefill forward-pass race (see
# PrefillReadyMessage's own docstring in pp_scheduler_protocol.py for
# the full incident this closes). Bounds how long rank 0 will
# re-announce the SAME pending prefill after a NACK (rank 1 not yet
# locally registered for it) before failing loud rather than retrying
# forever -- matching this session's own root-cause-only,
# no-silent-mitigation standing rule: a bounded retry across REAL,
# DIFFERENT tick() calls (each one giving rank 1's own event loop a
# genuine, unblocked opportunity to drain its work queue and register
# the matching _DeferredPrefill) is a real resolution of a real,
# expected timing race -- not a disguised sleep loop.
#
# 2026-08-06 bug #7 fix (see
# docs/batched-decode-n2-admission-handoff-2026-08-05.md's
# "2026-08-06 root-cause analysis: Attempt 1 and Attempt 2 (bug #7)"
# section): this bound used to be a fixed RETRY COUNT
# (_PREFILL_READY_MAX_RETRIES=50), which real N=2 hardware testing
# proved tripped falsely -- rank 0 crashed after receiving 50
# CONSECUTIVE EXPLICIT NACKs (ready=False replies) from rank 1, which
# is 50 consecutive PROOFS OF LIVENESS, not evidence of a stalled or
# hung peer. The failure happened because rank 1's main thread was
# legitimately busy running a DIFFERENT request's own real prefill
# forward pass (real GPU compute, ~1.35s observed on hardware) on the
# SAME thread that would otherwise reach its own submit()/
# mark_prefill_registered() call for the new request -- an explicit
# NACK proves rank 1 is alive and responding, it just hasn't gotten
# to registration yet. A fixed retry count conflates "peer explicitly
# said not-yet, N times" (expected, benign, scales with unrelated
# concurrent work) with "peer went SILENT / timed out" (the actual
# dead/hung-peer signal this guard is meant to catch).
#
# Fixed by switching to a WALL-CLOCK deadline measured from the FIRST
# NACK for a given request_id, not a retry count: as long as rank 1
# keeps explicitly replying (even every reply being ready=False), the
# deadline is a real signal of "has genuinely too much wall-clock time
# passed" rather than "has an arbitrary number of round-trips
# happened" -- correctly generous under load (many ticks can complete
# quickly while other requests are mid-prefill) and correctly strict
# against a truly wedged peer (recv_prefill_ready_message's own
# underlying recv has no timeout, so a hung/crashed rank 1 would
# manifest as this call itself hanging, not as fast NACKs -- see that
# function's own docstring). Chosen generously: several times longer
# than any single real prefill this session's hardware testing has
# observed (worst case so far: low single-digit seconds even at
# moderate prompt lengths), while still catching a genuinely stuck
# rank 1 well before an operator would otherwise notice.
_PREFILL_READY_MAX_WAIT_SECONDS = 30.0


@dataclass
class _PendingAdmission:
    request_id: int
    cache_slot: int
    prefilled_cache: "KVCacheType"
    initial_token: int
    sampler: "Sampler"
    max_tokens: int


@dataclass(frozen=True)
class _PendingPrefill:
    """A rank-0-local, not-yet-announced request waiting for
    ``tick()`` to become its single-writer dispatch point for the
    "admit request B now, start its prefill" decision -- see
    ``Rank0BatchedDecodeGlue.enqueue_prefill``'s docstring for the
    full N=2 admission-race rationale this closes
    (``docs/batched-decode-n2-admission-handoff-2026-08-05.md``,
    ``PrefillMessage``'s own docstring in ``pp_scheduler_protocol.py``).

    Pure data, NO closure/callable -- deliberately mirrors
    ``_PendingAdmission``'s own "just data, caller does the real
    work" shape. This glue module has no access to (and must not
    need to know about) the tokenizer/vision-processor/kv-prefix-
    cache/logits-processor machinery the real ``prefill()`` call
    needs; it only needs enough to (a) decide FIFO admission order
    and (b) put ``request_id``/``cache_slot``/``n_prompt_tokens`` on
    the wire via a real ``PrefillMessage``. The caller (whichever of
    ``ExoBatchGenerator``'s two prefill code paths is in play) is
    responsible for actually running prefill once ``tick()`` returns
    a ``PrefillGrant`` naming this ``request_id``, then calling
    ``enqueue_admission`` with the real result to fold it into the
    batch on a LATER ``tick()``.
    """

    request_id: int
    cache_slot: int
    n_prompt_tokens: int
    single_request_fallback: bool


@dataclass(frozen=True)
class PrefillGrant:
    """``tick()``'s signal (on EITHER rank) that the caller must now
    run a real prefill forward pass for ``request_id`` at
    ``cache_slot`` -- and ONLY NOW, never independently, per the
    single-writer/single-decider design this whole module exists to
    enforce (module docstring, updated 2026-08-06 for the N=2
    admission-race fix).

    On rank 0: returned the tick a pending prefill was announced
    (the real ``PrefillMessage`` send already happened inside this
    same ``tick()`` call, BEFORE this grant is returned) -- the
    caller must now run its own real prefill (batched-metaframe or
    single-request-fallback path, per ``single_request_fallback``),
    then call ``Rank0BatchedDecodeGlue.enqueue_admission`` with the
    result so the NEXT ``tick()`` folds it into the batch.

    On rank 1: returned the tick a ``PrefillMessage`` was reactively
    received (matching rank 0's send) -- the caller must run ITS OWN
    local prefill (the receive side of the SAME real cross-rank
    forward pass rank 0 just announced and is now running), then call
    ``Rank1BatchedDecodeGlue.stage_local_cache`` with the result so
    the admission arriving in a later ``StepMessage`` has a staged
    cache to bind to (unchanged from the pre-existing reactive-
    admission mechanism -- only WHEN prefill itself runs changed,
    not what happens after).

    ``single_request_fallback``: mirrors
    ``PrefillMessage.flags``'s ``PREFILL_FLAG_SINGLE_REQUEST_FALLBACK``
    bit -- tells the caller which real layer stack
    (``BatchedMetaFramedPipelineFirstLayer``/``LastLayer`` vs the
    plain ``MetaFramedPipelineFirstLayer``/``LastLayer`` this
    request is ineligible for the batched path and must use instead)
    this prefill's forward pass will exercise. Both ranks always
    agree on this value because it travels on the wire inside the
    same ``PrefillMessage`` rank 1 reactively decoded to produce this
    grant -- it is never independently recomputed on rank 1.
    """

    request_id: int
    cache_slot: int
    n_prompt_tokens: int
    single_request_fallback: bool


@dataclass(frozen=True)
class PrefillAdvanceCompleted:
    """``tick()``'s signal (rank 0 only -- see
    ``register_prefill_session``'s own docstring for why rank 1 has
    no equivalent return path) that an in-flight
    ``ResumablePrefillSession`` this glue was driving just reached its
    own ``done=True`` -- i.e. ONE CHUNK's forward pass has fully
    completed. 2026-08-06, Phase 2 Stage 3.

    Deliberately narrow: carries only what a caller needs to decide
    "what happens next for this chunk's output" -- NOT a decision
    this module makes itself (mirrors ``PrefillGrant``'s own "this
    module does mechanics, not prefill-completion policy" boundary).
    A caller holding a multi-chunk prefill (the not-yet-built
    generator-ified ``pipeline_parallel_prefill``, Stage 2's remaining
    piece) is expected to inspect ``output``, decide whether more
    chunks remain, and either call ``register_prefill_session`` again
    for the NEXT chunk or -- once every chunk is done -- proceed to
    ``enqueue_admission`` exactly as the existing synchronous
    ``run_prefill()`` path already does today.
    """

    request_id: int
    output: mx.array


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
    # 2026-08-06, chunk-drive redesign: rank 1's REAL local layer
    # count, learned via a REAL one-time model-load-time handshake
    # (``exchange_prefill_peer_layer_count`` -- NOT a per-request wire
    # message; this must already be known by the time ANY chunked-
    # prefill session is registered). Required, no default -- an
    # un-set value here is a genuine construction bug, not a "not yet
    # in use" state to silently tolerate, since the whole point of
    # this field is to make the chunk-completion arithmetic possible
    # AT ALL (see ``_prefill_phase``'s own field comment for the full
    # two-phase design this enables).
    peer_prefill_layer_count: int
    _pending: list[_PendingAdmission] = field(default_factory=list)
    _pending_prefill: list[_PendingPrefill] = field(default_factory=list)
    # Slots a PrefillGrant has been issued for but whose matching
    # enqueue_admission() has not yet arrived -- i.e. "reserved but
    # not yet occupied." Separate from
    # ``session.driver.cache_router``'s own occupied/free bookkeeping
    # (which only tracks ADMITTED, post-prefill requests, per
    # ``BatchedDecodeDriver.admit_request``'s Phase-1 scope: Phase-1's
    # ``SchedulerCore`` only ever models requests that are already
    # decode-ready). Without this set, a SECOND ``tick()`` call in the
    # SAME process -- before the first grant's ``enqueue_admission``
    # arrives -- could see the slot as still "free" (the router has
    # no idea it was just promised to someone) and grant a prefill
    # for a DIFFERENT request onto the SAME slot, corrupting whichever
    # admission lands second. A `consult` review (2026-08-06) caught
    # this gap in an earlier draft; reserving here, atomically with
    # the ``PrefillMessage`` send inside ``tick()``, is what closes it
    # -- the reservation and the wire send happen in the SAME
    # single-writer call, so no other ``tick()`` invocation can ever
    # observe the slot as free in between.
    _reserved_slots: set[int] = field(default_factory=set)
    # Dedicated monotonic counter for PrefillMessage.step_id, kept
    # SEPARATE from SchedulerCore's own internal step counter (which
    # backs StepMessage/EvictMessage and is not exposed outside
    # pp_scheduler_protocol.py). A shared counter across ALL three
    # control-message kinds would be a nice future strengthening of
    # the wire's monotonicity tripwire (module docstring point 5 of
    # pp_scheduler_protocol.py), but is not required for correctness
    # here: PrefillMessage's own ordering is already enforced by this
    # being the single-writer wire call for ALL of this session's
    # traffic (rank 1 never receives two PrefillMessages out of
    # order, because rank 0 never sends two without an intervening
    # StepMessage/EvictMessage, by this very tick() call structure).
    _prefill_step_id: int = field(default=0, init=False)
    # 2026-08-06 fix for the prefill forward-pass race (see
    # PrefillReadyMessage's own docstring in pp_scheduler_protocol.py
    # for the full incident this closes).
    #
    # 2026-08-06 bug #7 fix (see _PREFILL_READY_MAX_WAIT_SECONDS's own
    # docstring above for the full rationale): tracks the WALL-CLOCK
    # deadline (an absolute `time.monotonic()` value) for each pending
    # request_id's FIRST NACK, not a retry count -- so the fail-stop
    # guard measures genuine elapsed time waiting on rank 1, correctly
    # tolerant of however many fast NACK round-trips happen along the
    # way while rank 1 is legitimately busy with other work. Cleared
    # on a successful (ready=True) grant.
    _prefill_ready_deadline: dict[int, float] = field(default_factory=dict)
    # 2026-08-06, Phase 2 Stage 3: the ONE in-flight chunked-prefill
    # ResumablePrefillSession this glue is currently driving, if any.
    # Deliberately a single Optional slot, not a dict/queue -- matches
    # the design doc's confirmed Phase 2 scope ("at most one request
    # mid-chunked-prefill at a time"; PrefillAdvanceMessage.request_id
    # is carried explicitly anyway so a future scope relaxation to
    # multiple concurrent chunked prefills doesn't need a wire-shape
    # change, per that message's own docstring). Set by
    # ``register_prefill_session``, cleared by ``tick()`` itself the
    # instant a driven advance reaches ``done=True`` (this glue never
    # holds a stale completed session across ticks).
    _active_prefill_session: tuple[int, ResumablePrefillSession] | None = field(
        default=None, init=False
    )
    # Per-request monotonic advance_seq counter for
    # PrefillAdvanceMessage -- see that dataclass's own docstring for
    # why this is a SEPARATE counter from step_id (the desync tripwire
    # a `consult` review specifically recommended). Reset (via simply
    # never being read again) once a session completes; the NEXT
    # session's first advance always starts back at 1, matching
    # PrefillAdvanceMessage's own documented 1-based convention.
    _prefill_advance_seq: int = field(default=0, init=False)
    # 2026-08-06, chunk-drive redesign: which chunk (0-based) this
    # glue's OWN local session belongs to -- tagged on every
    # PrefillAdvanceMessage this glue sends so rank 1 can distinguish
    # "an advance for the chunk it's currently working on" from a
    # stale/misrouted one.
    _prefill_chunk_index: int = field(default=0, init=False)
    # 2026-08-06, REAL bug found (not hypothetical) auditing this
    # mechanism before wiring it into live serving, closed by this
    # two-phase redesign: ``MetaFramedPipelineFirstLayer.__call__``
    # (pp_metaframe.py) BLOCKS on ``recv_metaframe`` as the literal
    # first op of rank 1's own first local layer -- rank 1 cannot make
    # ANY progress on a chunk until rank 0 has walked its ENTIRE local
    # layer stack for that chunk and its activation has actually been
    # flushed onto the wire (``flush_prefill_sends()`` -- a SECOND real
    # bug found in the same audit: nothing previously called it after
    # a session's queued send). The original design sent rank 1 a
    # ``PrefillAdvanceMessage`` on EVERY one of rank 0's own advance
    # ticks, which would have made rank 1's tick() block hard, inside
    # the SAME single-writer call site that must also service decode
    # and admission, for the ENTIRE remaining duration of rank 0's
    # local-layer traversal -- exactly the multi-second freeze this
    # whole mechanism exists to prevent, just relocated onto rank 1.
    #
    # Fix (consult-reviewed: message-arrival-driven progress, not a
    # shared tick counter, per that review's core principle -- adapted
    # to exo's existing synchronous tick() transport rather than a
    # full async/CQ-poll rewrite, which was explicitly out of scope):
    # split ONE chunk's drive into two REAL phases rank 0's tick()
    # walks through explicitly, never conflating them:
    #   RANK0_LOCAL: tick() advances ONLY this rank's own local
    #     session, sending NO PrefillAdvanceMessage at all -- rank 1
    #     has nothing to do yet and must not be told otherwise.
    #   HANDOFF: the instant rank 0's own session reaches done=True,
    #     tick() calls flush_prefill_sends() (closes bug #2) so the
    #     queued activation ACTUALLY reaches rank 1's blocking recv,
    #     THEN computes exactly how many advances rank 1's session
    #     needs from peer_layer_count (closes bug #1's the two ranks'
    #     layer counts differ problem) and transitions to draining.
    #   RANK1_DRAINING: tick() sends rank 1 EXACTLY the precomputed
    #     number of PrefillAdvanceMessages, one per real tick (still
    #     respecting the existing decode-interleave alternation) --
    #     rank 1's own recv_metaframe has ALREADY unblocked by this
    #     point (real data arrived in HANDOFF), so each advance now
    #     makes REAL progress instead of hanging.
    _prefill_phase: Literal["rank0_local", "handoff", "rank1_draining"] = field(
        default="rank0_local", init=False
    )
    # Set at the RANK0_LOCAL -> HANDOFF transition: the exact number
    # of PrefillAdvanceMessages RANK1_DRAINING must still send before
    # this chunk is genuinely complete on BOTH ranks. Decremented by
    # 1 on every advance sent; the chunk is done when this reaches 0
    # (NOT when rank 0's own local session reaches done -- that only
    # means rank 0's HALF is done, per the whole point of this fix).
    _prefill_rank1_advances_remaining: int = field(default=0, init=False)
    # Layer-segment size for EVERY advance() call this glue drives --
    # module docstring's own "this glue does mechanics, not real
    # scheduling-fairness tuning" boundary: a fixed default here, not
    # yet exposed as a real constructor-configurable knob, since no
    # real hardware layer-timing data exists yet to tune it against
    # (see the design doc's own "eval-boundary overhead: measured
    # locally, needs real cluster confirmation" gating note).
    _prefill_advance_max_layers: int = field(default=2, init=False)
    # Alternation state for the advance-vs-decode fairness policy
    # above -- True means "an advance just ran, decode gets priority
    # THIS tick if there's decode work"; False (the initial value)
    # means "decode just ran (or nothing has run yet), advance gets
    # priority this tick." A simple 2-state flip-flop, not a ratio --
    # see the tick() rung's own comment for why a smarter ratio is
    # deliberately left untuned here.
    _prefill_favor_decode_next: bool = field(default=False, init=False)

    def register_prefill_session(
        self, request_id: int, session: ResumablePrefillSession, *, chunk_index: int
    ) -> None:
        """Register a live, not-yet-advanced ``ResumablePrefillSession``
        for ``tick()`` to drive -- the real caller (the not-yet-built
        generator-ified ``pipeline_parallel_prefill``, Stage 2's
        remaining piece) constructs the session for ONE CHUNK's
        forward pass and hands it here INSTEAD of calling
        ``session.advance(...)`` itself, so every real layer-segment
        advance happens inside ``tick()``'s single-writer wire call --
        the same discipline every other piece of real wire traffic in
        this module already follows (module docstring's own "ONLY
        call site that ever touches mx.distributed.send/recv_like"
        invariant, now extended to cover chunked-prefill advances too).

        ``chunk_index``: the 0-based real chunk this session belongs
        to, matching ``_pipeline_parallel_prefill_steps``'s own chunk
        loop index exactly (the real caller drives that generator and
        MUST pass its own current chunk index here, not guess). Tagged
        onto every ``PrefillAdvanceMessage`` this glue sends for this
        session so rank 1 can distinguish "an advance for the chunk
        I'm currently working on" from a stale/misrouted one.

        Resets this glue's chunk-drive state machine to
        ``"rank0_local"`` -- see that phase's own field comment above
        for the full two-phase design (RANK0_LOCAL -> HANDOFF ->
        RANK1_DRAINING) this closes.

        Raises ``GlueError`` if a session is already registered and
        still in flight -- per the confirmed Phase 2 scope (module
        docstring above), at most ONE chunked prefill session may be
        active at a time; a second registration while one is already
        pending is a caller bug, not a real occurrence to silently
        queue or overwrite. Also raises if ``peer_layer_count`` was
        never set on this glue (via the real model-load-time
        ``exchange_prefill_peer_layer_count`` handshake) -- without it,
        RANK1_DRAINING has no way to compute how many real advances
        rank 1's session needs.
        """
        if self._active_prefill_session is not None:
            active_id, _ = self._active_prefill_session
            raise GlueError(
                f"register_prefill_session({request_id}): a session for "
                f"request_id={active_id} is already active -- at most one "
                f"chunked-prefill session may be in flight at a time "
                f"(confirmed Phase 2 scope), refusing to silently queue "
                f"or overwrite a second one"
            )
        if self.peer_prefill_layer_count < 1:
            raise GlueError(
                f"register_prefill_session({request_id}): "
                f"peer_prefill_layer_count="
                f"{self.peer_prefill_layer_count} -- the real cross-rank "
                f"layer-count handshake (exchange_prefill_peer_layer_count, "
                f"called ONCE at model-load time) must run and set this "
                f"field on the glue BEFORE any session is registered; "
                f"RANK1_DRAINING cannot compute rank 1's real advance "
                f"count without it."
            )
        self._active_prefill_session = (request_id, session)
        self._prefill_advance_seq = 0
        self._prefill_chunk_index = chunk_index
        self._prefill_phase = "rank0_local"
        self._prefill_rank1_advances_remaining = 0
        # TEMP DIAGNOSTIC (2026-08-08, chasing the real advance_seq
        # desync bug found at chunk 51/69 of a 100K-token prefill -- see
        # design doc Section 17/18). Remove once root-caused.
        logger.info(
            f"[PREFILL_REGISTER_R0] request_id={request_id} chunk_index={chunk_index}"
        )

    def has_active_prefill_session(self) -> bool:
        return self._active_prefill_session is not None

    def reset_chunk_drive_state_after_reconnect(self) -> int | None:
        """Purely LOCAL chunk-drive state reset after a jaccl in-place
        reconnect (2026-08-09, design doc Section 27/28 -- real hardware
        finding: a request whose prefill was mid-chunk-drive when a
        transport fault hit left this rank's ``_active_prefill_session``/
        ``_prefill_phase``/``_prefill_rank1_advances_remaining`` stuck
        exactly where the fault interrupted them, because
        ``ExoBatchGenerator.reset_after_reconnect()`` only ever cleared
        ``_active_tasks``/``_mlx_gen`` -- it never touched EITHER rank's
        glue chunk-drive state or ``_deferred_prefill_by_uid``'s
        orphaned ``.drive`` objects. The NEXT request's
        ``register_prefill_session`` call then either hit the
        "already active" guard against a session that no live client
        was ever going to complete, or -- worse, confirmed on real
        hardware -- proceeded into a corrupted ``_prefill_phase``/
        ``_prefill_rank1_advances_remaining`` combination that tripped
        the ``RANK1_DRAINING`` fail-loud guard
        (``GlueError: reached RANK1_DRAINING with
        _prefill_rank1_advances_remaining=0``).

        UNLIKE ``abort_prefill_session()``, this does NOT attempt any
        wire round-trip (no ``PrefillAbortMessage``/ack) -- the wire
        itself is exactly what just failed and is being rebuilt by the
        caller's ``grp.reconnect()``; any partial in-flight
        cross-rank prefill state is being unconditionally discarded on
        BOTH ranks by the SAME jaccl-fault recovery path (this method
        is called identically on rank 0 and rank 1's glue, from
        ``ExoBatchGenerator.reset_after_reconnect()``, immediately
        after ``grp.reconnect()`` succeeds), so there is no
        "one side thinks it's aborted, the other still expects a
        message" split-brain risk the way an isolated single-request
        cancel would have.

        Returns the request_id that was dropped, or None if no session
        was active."""
        if self._active_prefill_session is None:
            return None
        dropped_id, prefill_session = self._active_prefill_session
        try:
            prefill_session.abort()
        except Exception as e:
            logger.warning(
                f"reset_chunk_drive_state_after_reconnect: "
                f"session.abort() for request_id={dropped_id} raised "
                f"{e!r} -- continuing to clear local state regardless, "
                f"since the underlying MLX arrays are already stale "
                f"post-reconnect device-context rebuild"
            )
        self._active_prefill_session = None
        self._prefill_phase = "rank0_local"
        self._prefill_rank1_advances_remaining = 0
        self._prefill_favor_decode_next = False
        return dropped_id

    def enqueue_admission(
        self,
        request_id: int,
        cache_slot: int,
        prefilled_cache: "KVCacheType",
        initial_token: int,
        sampler: "Sampler",
        max_tokens: int,
    ) -> None:
        """Fold an ALREADY-PREFILLED request's result into the batch.
        Called either (a) directly by a caller that already ran
        prefill itself BEFORE this glue was ever involved (the
        original, pre-2026-08-06 shape -- still valid, e.g. for a
        test harness driving prefill out of band), or (b), the real
        production shape as of the 2026-08-06 fix, by
        ``ExoBatchGenerator`` AFTER it ran the real prefill a
        ``PrefillGrant`` returned by ``tick()`` asked for. Pure
        in-memory append -- no wire I/O, so it cannot hang or race
        the decode-step loop. The actual admission (which touches the
        wire) happens later, inside ``tick()``.
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

    def enqueue_prefill(
        self,
        request_id: int,
        cache_slot: int,
        n_prompt_tokens: int,
        single_request_fallback: bool,
    ) -> None:
        """Called from ``ExoBatchGenerator.submit()`` INSTEAD OF
        running prefill directly (2026-08-06 fix for the real N=2
        admission-race deadlock -- see module docstring's 2026-08-06
        update). Pure in-memory append, zero wire I/O -- mirrors
        ``enqueue_admission``'s own no-I/O guarantee, so this call
        can never hang or race the decode-step loop. The actual
        prefill announcement (a real ``PrefillMessage`` send) happens
        later, inside ``tick()``, which is the single place this
        whole module allows wire traffic to originate.
        """
        self._pending_prefill.append(
            _PendingPrefill(
                request_id=request_id,
                cache_slot=cache_slot,
                n_prompt_tokens=n_prompt_tokens,
                single_request_fallback=single_request_fallback,
            )
        )

    def has_pending_admissions(self) -> bool:
        return bool(self._pending)

    def has_pending_prefills(self) -> bool:
        return bool(self._pending_prefill)

    def has_admitted_request(self, request_id: int) -> bool:
        """True iff ``request_id`` is genuinely running in this
        session's steady-state batched decode (real per-request
        generation state exists -- sampler, next_token, cache slot).
        2026-08-09 (design doc Section 27, external-cancellation fix):
        distinguishes this from a uid that is merely queued in
        ``_pending``/``_pending_prefill`` (not yet admitted) -- a
        caller must route those two cases through different removal
        paths (``complete_request()`` for an admitted request,
        ``cancel_pending_prefill()`` for a queued one).
        """
        return self.session.has_request(request_id)

    def cancel_pending_prefill(self, request_id: int) -> bool:
        """Drop ``request_id`` from the pending-admission /
        pending-prefill queues if it is sitting in either one,
        WITHOUT having been admitted into the session yet. Pure
        in-memory removal -- no wire I/O, so this is safe to call
        from any thread/call site, unlike ``complete_request()``
        (which does real blocking wire I/O and must never be called
        for a request this method already removed). Returns True iff
        something was actually removed (the caller uses this to know
        whether it must ALSO call ``complete_request()`` for the same
        request_id, or whether removal from the queue was sufficient).

        2026-08-09 (design doc Section 27, external-cancellation fix):
        a request can be cancelled by the client at ANY point in its
        lifecycle -- including before ``tick()`` ever admits it (still
        sitting in ``_pending`` waiting for its target slot to free, or
        in ``_pending_prefill`` waiting for its PrefillMessage/
        PrefillReadyMessage round trip). Neither queue entry has a
        counterpart in ``self.session._requests`` yet, so
        ``complete_request()``'s real ``evict_request()``/
        ``EvictMessage`` protocol has nothing to evict and would raise
        ``BatchedDecodeSessionError``. Dropping the queue entry here is
        the correct, minimal removal for this case -- there is no
        cross-rank state to reconcile since rank 1 never received a
        PrefillMessage for a request still sitting in ``_pending``
        (the ``_pending`` queue is for ALREADY-PREFILLED admissions
        awaiting a free slot -- see ``enqueue_admission``'s own
        docstring), and if a ``PrefillMessage`` for a
        ``_pending_prefill`` entry has ALREADY been sent (this
        request is ``head`` and rank 1 already ACKed readiness), the
        caller is responsible for NOT calling this after that grant
        was issued -- ``tick()`` pops an entry from ``_pending_prefill``
        and hands the caller a ``PrefillGrant`` in the SAME call, so by
        the time the caller could observe the grant, the entry is
        already gone from this queue and this method is a correct
        no-op for it (the caller must instead let the deferred prefill
        run to completion and cancel via the NORMAL
        ``abort_prefill_session``/``complete_request`` paths once it
        has real session state).
        """
        removed = False
        before = len(self._pending)
        self._pending = [p for p in self._pending if p.request_id != request_id]
        removed = removed or len(self._pending) != before

        before = len(self._pending_prefill)
        # A reserved slot for a NOW-cancelled pending prefill must be
        # released too, or that slot leaks forever (never re-offered to
        # any future request) -- mirrors this class's own established
        # "every reservation has an explicit release" discipline (see
        # the ``_reserved_slots.discard`` call in the admission branch
        # of ``tick()`` above).
        for p in self._pending_prefill:
            if p.request_id == request_id:
                self._reserved_slots.discard(p.cache_slot)
        self._pending_prefill = [
            p for p in self._pending_prefill if p.request_id != request_id
        ]
        removed = removed or len(self._pending_prefill) != before

        self._prefill_ready_deadline.pop(request_id, None)
        return removed

    def tick(
        self, model: object
    ) -> tuple[
        dict[int, "AdmitResponse | StepResponse"],
        int | None,
        PrefillGrant | None,
        PrefillAdvanceCompleted | None,
    ]:
        """The ONLY call site that ever touches
        ``mx.distributed.send``/``recv_like`` for this session on
        rank 0. Called from EXACTLY ONE place:
        ``ExoBatchGenerator.step()`` (mirrors ``_step_pp_spec``'s own
        single call site).

        Does AT MOST ONE of the following per call, in this FIXED
        priority order (never more than one, never zero unless there
        is genuinely nothing to do):
          1. If a pending, ALREADY-PREFILLED admission exists AND its
             target slot is free: admit it (real ``StepMessage`` send
             to rank 1), return its first classified response keyed
             by ``request_id``. Checked FIRST so in-flight work
             (a request whose prefill already ran) finishes before
             any NEW prefill is granted -- both the fix for the
             slot-reservation race documented on ``_reserved_slots``
             above, and the right operational order.
          2. Else, if a pending prefill is queued AND its target slot
             is neither occupied (an active request) nor already
             reserved (a grant already issued, awaiting admission):
             reserve the slot, send a real ``PrefillMessage``
             announcing it to rank 1, and return a ``PrefillGrant``
             telling the caller to now run the real prefill forward
             pass. THIS is the branch that closes the N=2 admission
             race (module docstring 2026-08-06 update): the decision
             "start prefill for request X now" is made HERE, inside
             the single-writer wire call, instead of independently
             inside ``submit()`` on each rank's own schedule.
             CHECKED BEFORE decode (branch 3, not after) -- an
             earlier draft of this method placed decode first, which
             is a real bug: ``has_active_requests()`` stays True for
             the entire lifetime of ANY currently-decoding request, so
             if decode outranked a new prefill, a second request could
             NEVER be granted while the first was still generating --
             defeating N=2 concurrency's entire purpose. Prefill
             outranking decode costs exactly one decode-step tick's
             worth of latency for the currently-active batch each time
             a new request is admitted (bounded by
             ``max_concurrency`` -- at most that many consecutive
             non-decode ticks can ever happen back to back, since each
             one consumes a free slot), which is the correct,
             deliberate trade -- a `consult` review (2026-08-06)
             caught this ordering bug before it shipped.
          3. Else, if the session has active requests: run one real
             batched decode step, return every active request's
             classified response this step.
          4. Else: return an empty result (nothing to do this tick).

        The second element of the returned tuple is the
        ``request_id`` that was JUST admitted this tick (``None`` on
        every other kind of tick) -- callers need this to know which
        request's FIRST response came from admission (a different
        code path than steady-state decode) without re-deriving it
        from the dict's own contents. The third element is non-None
        ONLY on a prefill-announcement tick (branch 2) -- see
        ``PrefillGrant``'s own docstring for what the caller must do
        with it.
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
            self._reserved_slots.discard(pending.cache_slot)
            return {pending.request_id: admit_response}, pending.request_id, None, None

        # 2026-08-07 (Phase 2 live-wiring, priority-order guard -- per a
        # consult review flagging a real gap in this method's own fixed
        # priority order): a new prefill must NEVER be granted while a
        # chunked-prefill session is already active on THIS rank. Without
        # this guard, this branch (priority 2) would still fire ahead of
        # the chunk-drive branch (priority 3) below for a SECOND pending
        # request even while the FIRST request's multi-chunk drive is
        # mid-flight -- issuing a real PrefillMessage/forward pass for
        # request B while request A's chunk-drive still owns this rank's
        # real wire state. register_prefill_session()'s own "at most one
        # active session" guard would eventually catch this as a hard
        # GlueError once request B's caller tried to register ITS first
        # session, but that is a crash, not graceful backpressure --
        # skipping this branch here means request B's PrefillMessage
        # simply waits (stays queued in _pending_prefill) until request
        # A's chunk-drive genuinely completes and clears
        # _active_prefill_session, exactly matching this module's
        # confirmed Phase 2 scope ("at most one request mid-chunked-
        # prefill at a time").
        if self._pending_prefill and self._active_prefill_session is None:
            head = self._pending_prefill[0]
            # A slot already reserved FOR THIS EXACT PENDING REQUEST
            # (tracked by `head.request_id in self._prefill_ready_deadline`
            # -- i.e. this is a RETRY of an attempt that already
            # reserved it, not a fresh grant to a DIFFERENT request)
            # must NOT count as "busy" here, or a retry can never
            # proceed past its own first reservation. `_reserved_slots`
            # itself only tracks slot NUMBERS (not which request holds
            # the reservation) since it's a cross-request mutual-
            # exclusion guard (module docstring/field comment above);
            # this retry-dict check is what disambiguates "busy with
            # someone ELSE's reservation" from "busy with MY OWN,
            # still-pending reservation" without needing to widen
            # `_reserved_slots` into a slot->request_id map.
            is_own_retry = head.request_id in self._prefill_ready_deadline
            slot_busy = self.session.driver.cache_router.is_occupied(
                head.cache_slot
            ) or (head.cache_slot in self._reserved_slots and not is_own_retry)
            if not slot_busy:
                # Reserve the slot on the FIRST attempt for this
                # request only -- idempotent across retries (a NACK
                # leaves ``head`` at the front of ``_pending_prefill``
                # for the next tick() to retry, so this branch can run
                # again for the SAME request before it's ever popped;
                # ``head.cache_slot in self._reserved_slots`` above
                # already guards re-reservation, this comment just
                # makes that fact explicit at the call site).
                self._reserved_slots.add(head.cache_slot)
                flags = (
                    PREFILL_FLAG_SINGLE_REQUEST_FALLBACK
                    if head.single_request_fallback
                    else 0
                )
                self._prefill_step_id += 1
                prefill_message = PrefillMessage(
                    step_id=self._prefill_step_id,
                    request_id=head.request_id,
                    cache_slot=head.cache_slot,
                    n_prompt_tokens=head.n_prompt_tokens,
                    flags=flags,
                )
                send_prefill_message(
                    prefill_message, dst=self.dst_rank, group=self.group
                )
                # 2026-08-06 fix for the prefill forward-pass race (see
                # PrefillReadyMessage's own docstring in
                # pp_scheduler_protocol.py for the full incident this
                # closes): BLOCK for rank 1's real ack before running
                # (granting the caller permission to run) the real
                # prefill forward pass. This is the single change that
                # makes it safe for the caller to immediately follow a
                # returned PrefillGrant with real metaframe wire
                # traffic -- rank 1 has JUST confirmed, on this same
                # wire, that it is genuinely ready to run its own
                # matching forward pass right now.
                ack = recv_prefill_ready_message(self.dst_rank, group=self.group)
                if ack.request_id != head.request_id:
                    raise GlueError(
                        f"tick(): PrefillReadyMessage request_id mismatch "
                        f"-- sent PrefillMessage for request_id="
                        f"{head.request_id}, received ack for request_id="
                        f"{ack.request_id}. The two ranks' control-message "
                        f"streams have desynced."
                    )
                if not ack.ready:
                    # 2026-08-06 bug #7 fix (see
                    # _PREFILL_READY_MAX_WAIT_SECONDS's own docstring
                    # above for the full rationale): an explicit NACK
                    # is a real reply proving rank 1 is ALIVE -- it is
                    # not, by itself, evidence of a stall. Only genuine
                    # elapsed WALL-CLOCK time waiting for rank 1 to
                    # finish whatever else it's legitimately doing
                    # (e.g. another request's own real prefill forward
                    # pass) is the actual signal this guard should act
                    # on, not how many times rank 1 happened to reply
                    # "not yet" along the way.
                    now = time.monotonic()
                    deadline = self._prefill_ready_deadline.get(head.request_id)
                    if deadline is None:
                        deadline = now + _PREFILL_READY_MAX_WAIT_SECONDS
                        self._prefill_ready_deadline[head.request_id] = deadline
                    elif now > deadline:
                        waited = now - (deadline - _PREFILL_READY_MAX_WAIT_SECONDS)
                        raise GlueError(
                            f"tick(): rank 1 NACK'd PrefillMessage for "
                            f"request_id={head.request_id} for "
                            f"{waited:.1f}s (limit "
                            f"{_PREFILL_READY_MAX_WAIT_SECONDS:.1f}s) -- "
                            f"rank 1's own local _DeferredPrefill "
                            f"registration for this request never "
                            f"arrived. This should self-resolve within a "
                            f"handful of seconds under normal scheduling "
                            f"(even while rank 1 is busy with another "
                            f"request's own real prefill); this long a "
                            f"wait indicates a genuine stall on rank 1 "
                            f"(crashed, hung, or a real registration "
                            f"bug), not an expected timing race. "
                            f"Refusing to retry forever."
                        )
                    # Leave head at the front of _pending_prefill (NOT
                    # popped) and its slot reserved -- the next tick()
                    # will retry the SAME request. Returning here
                    # (rather than falling through to decode below)
                    # keeps this tick's OWN single "at most one thing
                    # per call" contract -- a NACK still counted as
                    # this tick's one real wire round-trip.
                    return {}, None, None, None
                self._prefill_ready_deadline.pop(head.request_id, None)
                self._pending_prefill.pop(0)
                return (
                    {},
                    None,
                    PrefillGrant(
                        request_id=head.request_id,
                        cache_slot=head.cache_slot,
                        n_prompt_tokens=head.n_prompt_tokens,
                        single_request_fallback=head.single_request_fallback,
                    ),
                    None,
                )

        # 2026-08-06, chunk-drive redesign (see _prefill_phase's own
        # field comment above for the full incident/design): drives
        # a chunked-prefill session through TWO REAL PHASES, never
        # conflating them -- RANK0_LOCAL (this rank's own layers only,
        # NO wire traffic to rank 1 yet) then RANK1_DRAINING (rank 1's
        # session, now that its recv_metaframe has REAL data to
        # unblock on). HANDOFF is the single-tick transition between
        # them: flush the queued activation send, compute exactly how
        # many advances rank 1 needs from the real cross-rank layer-
        # count handshake, and fall through into the first drain send
        # on the SAME tick (no wasted tick doing nothing).
        if self._active_prefill_session is not None:
            has_decode_work = self.session.has_active_requests()
            advance_this_tick = (
                not has_decode_work or not self._prefill_favor_decode_next
            )
            if advance_this_tick:
                self._prefill_favor_decode_next = True
                request_id, prefill_session = self._active_prefill_session
                just_handed_off = False

                if self._prefill_phase == "rank0_local":
                    # Advance ONLY this rank's own local session --
                    # rank 1 has nothing to do yet (its first layer's
                    # recv_metaframe has no data to unblock on) and
                    # sending it a PrefillAdvanceMessage here would
                    # make its tick() block hard for the ENTIRE rest
                    # of this rank's local-layer traversal -- the
                    # exact bug this redesign closes.
                    _layers_advanced, done = prefill_session.advance(
                        max_layers=self._prefill_advance_max_layers,
                        phase_for_pause=ForwardPhase.PREFILL_CONTINUE,
                    )
                    # TEMP DIAGNOSTIC (2026-08-10, chasing the real
                    # mutual-deadlock bug found at call_id=411 -- see
                    # design doc Section 39/handoff for the full
                    # incident). Confirms rank 0's own local layer
                    # count/yield-count agreement before the handoff
                    # arithmetic below ever runs. Remove once
                    # root-caused.
                    logger.info(
                        f"[RANK0_LOCAL_ADVANCE] chunk_index="
                        f"{self._prefill_chunk_index} "
                        f"requested_max_layers="
                        f"{self._prefill_advance_max_layers} "
                        f"layers_advanced={_layers_advanced} done={done} "
                        f"last_layer_index={prefill_session.last_layer_index}"
                    )
                    if done:
                        self._prefill_phase = "handoff"
                    else:
                        return {}, None, None, None
                    # Falls through to "handoff" handling below on
                    # THIS SAME tick -- no wasted tick transitioning.

                if self._prefill_phase == "handoff":
                    # 2026-08-07 (wire-ordering bug fix, found via a
                    # real 2-process subprocess test driving REAL
                    # wire-sending layers through this exact path for
                    # the first time -- no earlier test in this
                    # campaign ever exercised a real mx.distributed.send
                    # here, and the mlx-lm submodule pin's own missing
                    # _forward_steps split meant this code was 100%
                    # unreachable on real hardware regardless). This
                    # rank's own final local layer for this chunk
                    # ALREADY queued its header+table+activation
                    # together (ForwardStepInfo.defer_header=True, set
                    # by ResumablePrefillSession.advance's own
                    # set_forward_step_info call) instead of sending
                    # the header eagerly -- the flush that actually
                    # puts them on the wire happens further below,
                    # AFTER the first send_prefill_advance_message()
                    # call, not here. Sending the metaframe BEFORE any
                    # scheduler-wire control message announces it would
                    # land on rank 1's wire ahead of anything its
                    # tick() dispatcher has been told to expect --
                    # rank 1's own recv_header() unconditionally reads
                    # a scheduler-wire header FIRST on every tick(), so
                    # metaframe bytes must never arrive except INSIDE
                    # the handler that scheduler-wire message dispatches
                    # into. See flush_prefill_metaframe_sends()'s own
                    # docstring for the full invariant.
                    #
                    # Ceiling division: rank 1's session needs exactly
                    # this many real advance() calls to walk its own
                    # peer_prefill_layer_count layers at
                    # _prefill_advance_max_layers per call -- the real
                    # cross-rank layer-count handshake
                    # (exchange_prefill_peer_layer_count, model-load
                    # time) is what makes this computable AT ALL (see
                    # peer_prefill_layer_count's own field comment for
                    # why the two ranks' real layer counts differ).
                    self._prefill_rank1_advances_remaining = -(
                        -self.peer_prefill_layer_count
                        // self._prefill_advance_max_layers
                    )
                    # TEMP DIAGNOSTIC (2026-08-10, chasing the real
                    # mutual-deadlock bug found at call_id=411 -- see
                    # design doc Section 39/handoff for the full
                    # incident). The exact ceiling-division budget
                    # rank 0 computed for this chunk -- compare against
                    # how many [PREFILL_ADVANCE_APPLIED] lines rank 1
                    # actually logs with done=True to confirm/refute
                    # the layer-count-mismatch hypothesis. Remove once
                    # root-caused.
                    logger.info(
                        f"[HANDOFF_BUDGET] chunk_index="
                        f"{self._prefill_chunk_index} "
                        f"peer_prefill_layer_count="
                        f"{self.peer_prefill_layer_count} "
                        f"prefill_advance_max_layers="
                        f"{self._prefill_advance_max_layers} "
                        f"advances_budgeted="
                        f"{self._prefill_rank1_advances_remaining}"
                    )
                    self._prefill_phase = "rank1_draining"
                    just_handed_off = True
                    # Falls through to "rank1_draining" handling below
                    # on THIS SAME tick.

                # self._prefill_phase == "rank1_draining" (either
                # arrived here directly, or fell through from
                # "handoff" above on this same tick).
                if self._prefill_rank1_advances_remaining <= 0:
                    # Genuinely unreachable in correct operation --
                    # the transition into "rank1_draining" above
                    # always sets a >=1 value (peer_prefill_layer_count
                    # is validated >=1 at construction/handshake time),
                    # and this branch decrements to 0 and exits via the
                    # `done` return below on the SAME tick that reaches
                    # 0. A defensive fail-loud guard, not a real
                    # occurrence.
                    raise GlueError(
                        f"tick(): reached RANK1_DRAINING with "
                        f"_prefill_rank1_advances_remaining="
                        f"{self._prefill_rank1_advances_remaining} -- "
                        f"the chunk-drive state machine has a real bug, "
                        f"refusing to send a meaningless advance"
                    )
                self._prefill_advance_seq += 1
                advance_message = PrefillAdvanceMessage(
                    step_id=self._prefill_step_id + 1,
                    request_id=request_id,
                    advance_seq=self._prefill_advance_seq,
                    max_layers=self._prefill_advance_max_layers,
                    chunk_index=self._prefill_chunk_index,
                )
                self._prefill_step_id += 1
                # TEMP DIAGNOSTIC (2026-08-08, chasing the real advance_seq
                # desync bug found at chunk 51/69 of a 100K-token prefill --
                # see design doc Section 17/18). Remove once root-caused.
                logger.info(
                    f"[PREFILL_ADVANCE_SEND] chunk_index={self._prefill_chunk_index} "
                    f"advance_seq={self._prefill_advance_seq} "
                    f"remaining_before_send={self._prefill_rank1_advances_remaining} "
                    f"just_handed_off={just_handed_off}"
                )
                send_prefill_advance_message(
                    advance_message, dst=self.dst_rank, group=self.group
                )
                if just_handed_off:
                    # THE fix: the metaframe (header+table+activation)
                    # this rank's own final local layer queued back in
                    # RANK0_LOCAL is flushed HERE -- strictly AFTER the
                    # send_prefill_advance_message() call immediately
                    # above -- so it lands on the wire in the exact
                    # order rank 1's tick() dispatcher expects: the
                    # announcing scheduler-wire header first, THEN the
                    # metaframe. Only fires on the SAME tick that just
                    # transitioned handoff->rank1_draining (guarded by
                    # just_handed_off) -- every SUBSEQUENT
                    # rank1_draining advance within a multi-advance
                    # chunk has nothing new queued (this rank's own
                    # local forward pass for this chunk already fully
                    # completed back in RANK0_LOCAL; only rank 1 has
                    # remaining local layers to walk), so calling this
                    # unconditionally every rank1_draining tick would
                    # be a harmless no-op in practice (the queue would
                    # already be empty) but is guarded explicitly
                    # anyway to keep this call site's intent legible.
                    flush_prefill_metaframe_sends()
                self._prefill_rank1_advances_remaining -= 1
                if self._prefill_rank1_advances_remaining == 0:
                    # Rank 1 has now received exactly enough advances
                    # to have walked its own full local layer stack
                    # for this chunk -- the chunk is genuinely done on
                    # BOTH ranks, not just this one.
                    #
                    # 2026-08-08, real production incident fix (design
                    # doc Section 21/22, `consult`-reviewed decision:
                    # bounded blocking wait, not non-blocking polling
                    # -- see PrefillChunkDoneMessage's own docstring
                    # for the full incident and rationale). "Sent all
                    # 11 advances" does NOT mean "rank 1 finished
                    # PROCESSING them" -- rank 1's own local Metal
                    # compute for this chunk's tail layers can still
                    # genuinely be running (confirmed on real
                    # hardware: a 14-second Event::wait stall).
                    # BLOCK here for rank 1's real completion ack
                    # before declaring this chunk done -- rank 1 sends
                    # it EAGERLY, from inside its own tick() call,
                    # immediately after its own advance() forward pass
                    # genuinely finishes (see that call site's own
                    # comment), so this recv resolves as soon as rank
                    # 1's real compute is done, not on some separate
                    # round-trip. Bounded by the underlying jaccl
                    # transport's own real deadline mechanisms (the
                    # StallWatch/drain-deadline machinery already
                    # covering every other recv in this module) --
                    # deliberately no separate Python-level timeout
                    # wrapper here, mirroring how every other blocking
                    # recv in this class (e.g. the abort-ack round
                    # trip) already relies on that same backstop
                    # rather than reimplementing one.
                    ack = recv_prefill_chunk_done_ack_message(
                        src=self.dst_rank, group=self.group
                    )
                    if (
                        ack.request_id != request_id
                        or ack.chunk_index != self._prefill_chunk_index
                    ):
                        raise GlueError(
                            f"tick(): PrefillChunkDoneAckMessage mismatch "
                            f"-- expected request_id={request_id} "
                            f"chunk_index={self._prefill_chunk_index}, got "
                            f"request_id={ack.request_id} chunk_index="
                            f"{ack.chunk_index}. Refusing to proceed on a "
                            f"mismatched ack rather than silently "
                            f"registering the wrong chunk."
                        )
                    self._active_prefill_session = None
                    self._prefill_phase = "rank0_local"
                    return (
                        {},
                        None,
                        None,
                        PrefillAdvanceCompleted(
                            request_id=request_id, output=prefill_session.output()
                        ),
                    )
                return {}, None, None, None
            self._prefill_favor_decode_next = False

        if self.session.has_active_requests():
            prepared = self.session.prepare_step()
            send_step_message(prepared.message, dst=self.dst_rank, group=self.group)
            logits = self.session.run_forward(cast("_ModelLike", model), prepared)
            step_results = self.session.finish_step(prepared, logits)
            classified = self.adapter.classify_step_results(step_results)
            return dict(classified), None, None, None

        return {}, None, None, None

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

    def abort_prefill_session(self, request_id: int) -> None:
        """Caller (``ExoBatchGenerator.cancel()``) signals
        ``request_id``'s chunked-prefill drive must be abandoned right
        now -- 2026-08-07, real cancel/abort mechanism.

        MUST be called from the SAME cooperative caller context that
        drives ``tick()`` itself (mirrors ``complete_request``'s own
        "deliberately NOT folded into tick()" discipline exactly) --
        NEVER concurrently with an in-flight ``tick()`` call for this
        rank. This is a real constraint, not a formality: rank 1
        ALWAYS holds its own live mirrored session state the entire
        time this rank's is active (both ranks register independently
        at admission time, per a `consult` review correcting an
        earlier, wrong assumption that only ``RANK1_DRAINING`` implied
        rank-1 state) -- there is no local-only fast path. Every
        abort, regardless of ``_prefill_phase``, closes this rank's
        own session locally THEN sends a real, blocking-acked
        ``PrefillAbortMessage``/``PrefillAbortAckMessage`` round trip,
        exactly mirroring ``complete_request``'s own
        ``EvictMessage``/``EvictAckMessage`` pattern.

        Raises ``GlueError`` if no session is registered for
        ``request_id`` (caller bug -- ``ExoBatchGenerator.cancel()``
        must only route here for a uid it already confirmed has an
        active ``ChunkedPrefillDrive``, per that method's own
        docstring) or if the received ack's ``request_id`` mismatches
        (identical fail-stop rationale to ``complete_request``'s own
        ack-mismatch guard -- refusing to silently believe an ack for
        the wrong request).
        """
        if self._active_prefill_session is None:
            raise GlueError(
                f"abort_prefill_session({request_id}): no active prefill "
                f"session on this rank -- caller must only call this for "
                f"a request_id it already confirmed has an active "
                f"chunk-drive"
            )
        active_id, prefill_session = self._active_prefill_session
        if active_id != request_id:
            raise GlueError(
                f"abort_prefill_session({request_id}): this rank's active "
                f"session is for request_id={active_id}, not {request_id} "
                f"-- refusing to abort the wrong request's session"
            )

        from exo.worker.engines.mlx.pp_scheduler_protocol import PrefillAbortMessage
        from exo.worker.engines.mlx.pp_scheduler_wire import (
            recv_prefill_abort_ack_message,
            send_prefill_abort_message,
        )

        prefill_session.abort()
        self._active_prefill_session = None
        self._prefill_phase = "rank0_local"
        self._prefill_rank1_advances_remaining = 0
        self._prefill_step_id += 1
        abort_message = PrefillAbortMessage(
            step_id=self._prefill_step_id, request_id=request_id
        )
        send_prefill_abort_message(abort_message, dst=self.dst_rank, group=self.group)
        ack = recv_prefill_abort_ack_message(src=self.dst_rank, group=self.group)
        if ack.request_id != request_id:
            raise GlueError(
                f"abort_prefill_session({request_id}): PrefillAbortAckMessage "
                f"mismatch -- expected request_id={request_id}, got "
                f"request_id={ack.request_id}. Refusing to proceed on a "
                f"mismatched ack."
            )


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
    # 2026-08-06 fix for the prefill forward-pass race (see
    # PrefillReadyMessage's own docstring in pp_scheduler_protocol.py
    # for the full incident this closes -- a real
    # SchedulerWireProtocolError crash on N=2 hardware). Populated by
    # ``mark_prefill_registered`` (called from
    # ``ExoBatchGenerator.submit()`` the INSTANT this rank registers
    # its own local ``_DeferredPrefill`` for a request -- pure local
    # bookkeeping, zero wire I/O, matching every other *_registered/
    # *_staged method in this module). ``tick()``'s ``MSG_KIND_PREFILL``
    # branch checks membership here SYNCHRONOUSLY, inline, to decide
    # whether to ACK or NACK rank 0's ``PrefillMessage`` -- never a
    # callback into ``ExoBatchGenerator``, keeping this glue
    # self-contained and its "tick() is the ONLY wire I/O call site"
    # invariant intact.
    _registered_request_ids: set[int] = field(default_factory=set)
    # 2026-08-06, Phase 2 Stage 3: rank 1's OWN local
    # ResumablePrefillSession, mirroring rank 0's own
    # ``_active_prefill_session`` field -- each rank drives ITS OWN
    # session (holding ITS OWN half of the model's layers), advancing
    # in lockstep ONLY because both ranks react to the SAME
    # PrefillAdvanceMessage.advance_seq sequence on the wire, never
    # because either rank makes an independent local timing decision
    # (module docstring's own "rank 1 never independently decides"
    # discipline, now extended to advances the same way it already
    # covers admission).
    _active_prefill_session: tuple[int, ResumablePrefillSession] | None = field(
        default=None, init=False
    )
    # Rank 1's own view of the last advance_seq it processed for the
    # currently-active session -- the desync tripwire a `consult`
    # review specifically recommended (PrefillAdvanceMessage's own
    # docstring). None before any advance has been received for the
    # current session.
    _last_prefill_advance_seq: int | None = field(default=None, init=False)

    def register_prefill_session(
        self, request_id: int, session: ResumablePrefillSession
    ) -> None:
        """Register rank 1's OWN local ``ResumablePrefillSession`` for
        ``tick()`` to drive reactively -- called by the not-yet-built
        generator-ified ``pipeline_parallel_prefill`` on rank 1 the
        SAME way rank 0's own ``register_prefill_session`` is (see
        that method's own docstring for the full rationale); the two
        registrations are independent local calls on each rank, but
        stay in lockstep because BOTH ranks' ``tick()`` only ever
        advances a registered session in reaction to the SAME wire
        message (rank 0 sends it, rank 1 receives it) -- never on
        either rank's own local schedule.
        """
        if self._active_prefill_session is not None:
            active_id, _ = self._active_prefill_session
            raise GlueError(
                f"register_prefill_session({request_id}): a session for "
                f"request_id={active_id} is already active on this rank -- "
                f"at most one chunked-prefill session may be in flight at "
                f"a time (confirmed Phase 2 scope), refusing to silently "
                f"queue or overwrite a second one"
            )
        self._active_prefill_session = (request_id, session)
        self._last_prefill_advance_seq = None
        # TEMP DIAGNOSTIC (2026-08-08, chasing the real advance_seq
        # desync bug found at chunk 51/69 of a 100K-token prefill -- see
        # design doc Section 17/18). Remove once root-caused.
        logger.info(f"[PREFILL_REGISTER_R1] request_id={request_id}")

    def has_active_prefill_session(self) -> bool:
        return self._active_prefill_session is not None

    def reset_chunk_drive_state_after_reconnect(self) -> int | None:
        """Rank 1's mirror of ``Rank0BatchedDecodeGlue``'s own
        identically-named method -- see that method's docstring for
        the full incident/rationale (design doc Section 27/28, real
        hardware finding, 2026-08-09). Called identically on both
        ranks' glues from ``ExoBatchGenerator.reset_after_reconnect()``
        immediately after ``grp.reconnect()`` succeeds -- purely local,
        no wire round-trip (the wire is exactly what just failed and
        is being rebuilt by the caller).

        Returns the request_id that was dropped, or None if no session
        was active."""
        if self._active_prefill_session is None:
            return None
        dropped_id, prefill_session = self._active_prefill_session
        try:
            prefill_session.abort()
        except Exception as e:
            logger.warning(
                f"reset_chunk_drive_state_after_reconnect: "
                f"session.abort() for request_id={dropped_id} raised "
                f"{e!r} -- continuing to clear local state regardless, "
                f"since the underlying MLX arrays are already stale "
                f"post-reconnect device-context rebuild"
            )
        self._active_prefill_session = None
        self._last_prefill_advance_seq = None
        return dropped_id

    def mark_prefill_registered(self, request_id: int) -> None:
        """Called from ``ExoBatchGenerator.submit()`` the moment this
        rank registers its own local ``_DeferredPrefill`` closure for
        ``request_id`` -- BEFORE any ``PrefillMessage`` for it may
        have arrived (the common case) or possibly AFTER (the real,
        expected timing race two independently-scheduled per-rank
        event loops can hit). Either ordering is safe: this is pure
        set membership, checked fresh on whatever ``tick()`` call
        actually receives the matching ``PrefillMessage``, whenever
        that happens to be. Pure in-memory update, zero wire I/O."""
        self._registered_request_ids.add(request_id)

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

    def tick(
        self, model: object
    ) -> tuple[PrefillGrant | None, int | None, PrefillAdvanceCompleted | None]:
        """The ONLY call site that ever touches
        ``mx.distributed.send``/``recv_like`` for this session on
        rank 1. Receives exactly one header, branches on its real
        ``msg_kind`` (prefill announcement, decode step, or eviction
        -- matching ``pp_scheduler_wire.py``'s own documented
        dispatch contract: "production dispatch code that must
        handle ANY of the kinds arriving next should call
        ``recv_header`` directly and branch on ``.msg_kind``"), and
        drives exactly the matching local action.

        Returns a 2-tuple ``(grant, evicted_request_id)``:

        - ``grant``: a ``PrefillGrant`` ONLY when a ``PrefillMessage``
          was just received AND this rank confirmed (via a real
          ``PrefillReadyMessage`` ack -- see that dataclass's own
          docstring for the full 2026-08-06 race this handshake
          closes) it is ready to run the matching prefill forward
          pass RIGHT NOW; ``None`` on every other kind of tick,
          including a not-yet-ready ``PrefillMessage`` (see below).
        - ``evicted_request_id``: the ``request_id`` this tick just
          evicted, ONLY on an eviction tick (see the 2026-08-06 bug
          #7 fix below for why the caller needs this); ``None`` on
          every other kind of tick. Never both non-None in the same
          call -- ``tick()``'s "at most one real thing per call"
          contract still holds; a tick is either a prefill grant, an
          eviction, a decode step, or an idle/not-ready no-op.

        RANK 1 NEVER INDEPENDENTLY DECIDES TO PREFILL (2026-08-06 fix
        for the real N=2 admission-race deadlock -- see module
        docstring's 2026-08-06 update). This is the reactive half of
        that fix: the only way this glue ever learns "prefill request
        X now" is by receiving rank 0's ``PrefillMessage`` right here.

        2026-08-06 UPDATE (prefill forward-pass race fix -- see
        ``PrefillReadyMessage``'s own docstring in
        ``pp_scheduler_protocol.py`` for the full incident this
        closes): a received ``PrefillMessage`` no longer
        unconditionally returns a grant. This rank replies with a
        real ``PrefillReadyMessage`` FIRST -- ``ready=True`` only if
        ``mark_prefill_registered`` was already called for this
        ``request_id`` (this rank's own local ``_DeferredPrefill`` is
        genuinely ready to run). If not yet registered, this rank
        sends ``ready=False`` and returns ``None`` -- it does NOT
        block/wait for registration here (that would deadlock: the
        SAME main-thread runner loop that would run this wait is also
        the loop responsible for draining the work queue and calling
        ``submit()``, which is the only thing that could ever satisfy
        the wait). Rank 0 is responsible for retrying (a bounded
        number of times, on a LATER ``tick()`` cycle -- giving this
        rank's own event loop real, unblocked opportunities to
        process its pending ``submit()`` work in between attempts).

        2026-08-06 bug #7 fix (third root cause, see
        docs/batched-decode-n2-admission-handoff-2026-08-05.md's
        "2026-08-06 root-cause analysis" sections for the full
        hardware evidence): before this fix, ``_step_batched_decode``
        ALWAYS returned an empty response list on rank 1 -- meaning
        rank 1's own eviction handling (this method's
        ``MSG_KIND_EVICT`` branch, which correctly frees this glue's
        internal cache_router/mirror bookkeeping) never propagated a
        finish signal back up to ``runner.py``'s OWN, SEPARATE
        admission gate (``self.active_tasks``, capped by
        ``EXO_MAX_CONCURRENT_REQUESTS``). Real N=2 hardware testing
        confirmed the consequence directly: rank 1's `runner.py`
        never logged a second "runner ready" (which only fires when
        ``self.active_tasks`` empties) for the ENTIRE remainder of a
        session after the very first admission, even though the
        request had already completed and been forgotten by every
        OTHER part of the system (rank 0, the master, the client) --
        rank 1's admission gate monotonically filled up and could
        NEVER drain while batched-decode was active, so once
        ``EXO_MAX_CONCURRENT_REQUESTS`` slots were consumed, every
        subsequent request was deferred FOREVER on rank 1, not merely
        delayed by a resolvable timing race. Fixed by having THIS
        eviction branch return the evicted request_id so the caller
        (``_step_batched_decode``) can synthesize a real
        finish-classified response for it -- mirroring
        ``_step_pp_spec``'s own established convention of returning
        real classified responses unconditionally on BOTH ranks (not
        just rank 0), which is exactly how that path's own
        ``runner.py``-level admission gate already drains correctly.
        """
        from exo.worker.engines.mlx.pp_batched_decode_runtime import _ModelLike

        header = recv_header(self.src_rank, group=self.group)
        if header.msg_kind == MSG_KIND_PREFILL:
            prefill_message = recv_prefill_body(header, self.src_rank, group=self.group)
            ready = prefill_message.request_id in self._registered_request_ids
            send_prefill_ready_message(
                PrefillReadyMessage(
                    step_id=prefill_message.step_id,
                    request_id=prefill_message.request_id,
                    ready=ready,
                ),
                dst=self.src_rank,
                group=self.group,
            )
            if not ready:
                return None, None, None
            self._registered_request_ids.discard(prefill_message.request_id)
            return (
                PrefillGrant(
                    request_id=prefill_message.request_id,
                    cache_slot=prefill_message.cache_slot,
                    n_prompt_tokens=prefill_message.n_prompt_tokens,
                    single_request_fallback=bool(
                        prefill_message.flags & PREFILL_FLAG_SINGLE_REQUEST_FALLBACK
                    ),
                ),
                None,
                None,
            )
        if header.msg_kind == MSG_KIND_PREFILL_ADVANCE:
            # 2026-08-06, Phase 2 Stage 3: rank 1's reactive half of
            # the advance rung -- rank 0's tick() already decided
            # "advance, not decode" and sent this message; rank 1
            # NEVER independently decides to advance its own session
            # (module docstring's own "rank 1 never independently
            # decides" discipline, now extended here exactly as it
            # already covers admission and prefill announcement).
            advance_message = recv_prefill_advance_body(
                header, self.src_rank, group=self.group
            )
            if self._active_prefill_session is None:
                raise GlueError(
                    f"tick(): received PrefillAdvanceMessage for "
                    f"request_id={advance_message.request_id} but this "
                    f"rank has no active local prefill session registered "
                    f"-- register_prefill_session was never called for "
                    f"this request_id, or this rank's session already "
                    f"completed/was never started. The two ranks' "
                    f"control-message streams have desynced."
                )
            active_id, prefill_session = self._active_prefill_session
            if active_id != advance_message.request_id:
                raise GlueError(
                    f"tick(): PrefillAdvanceMessage request_id mismatch "
                    f"-- this rank's active session is for "
                    f"request_id={active_id}, received an advance for "
                    f"request_id={advance_message.request_id}. The two "
                    f"ranks' control-message streams have desynced."
                )
            # advance_seq desync tripwire -- per the consult review
            # that recommended this field: catch a cross-rank
            # divergence as an IMMEDIATE loud assertion here, rather
            # than as a silent multi-tick-later jaccl/RDMA hang.
            expected_seq = (self._last_prefill_advance_seq or 0) + 1
            # TEMP DIAGNOSTIC (2026-08-08, chasing the real advance_seq
            # desync bug found at chunk 51/69 of a 100K-token prefill --
            # see design doc Section 17/18). Remove once root-caused.
            logger.info(
                f"[PREFILL_ADVANCE_RECV] chunk_index={advance_message.chunk_index} "
                f"received_seq={advance_message.advance_seq} "
                f"expected_seq={expected_seq}"
            )
            if advance_message.advance_seq != expected_seq:
                raise GlueError(
                    f"tick(): PrefillAdvanceMessage.advance_seq="
                    f"{advance_message.advance_seq} for request_id="
                    f"{advance_message.request_id} does not match this "
                    f"rank's own expected next seq={expected_seq} -- "
                    f"the two ranks' local advance counts have "
                    f"desynchronized. Refusing to advance on a "
                    f"potentially-misaligned layer-segment boundary."
                )
            self._last_prefill_advance_seq = advance_message.advance_seq
            _layers_advanced, done = prefill_session.advance(
                max_layers=advance_message.max_layers,
                phase_for_pause=ForwardPhase.PREFILL_CONTINUE,
            )
            # TEMP DIAGNOSTIC (2026-08-10, chasing the real mutual-
            # deadlock bug found at call_id=411 -- see design doc
            # Section 39/handoff for the full incident). Confirms
            # whether rank 1's own layer-boundary yield count matches
            # what rank 0's advance budget (ceil(peer_prefill_layer_
            # count/max_layers)) assumed. Remove once root-caused.
            logger.info(
                f"[PREFILL_ADVANCE_APPLIED] chunk_index="
                f"{advance_message.chunk_index} advance_seq="
                f"{advance_message.advance_seq} "
                f"requested_max_layers={advance_message.max_layers} "
                f"layers_advanced={_layers_advanced} done={done} "
                f"last_layer_index={prefill_session.last_layer_index}"
            )
            if done:
                self._active_prefill_session = None
                self._last_prefill_advance_seq = None
                # 2026-08-08, real production incident fix (see
                # PrefillChunkDoneMessage's own docstring for the full
                # incident/design decision -- bounded blocking wait,
                # not non-blocking polling, per that decision's own
                # rationale). Send the ack EAGERLY, from right here,
                # inside this SAME tick() call -- prefill_session.
                # advance() above is a real, synchronous, blocking
                # Metal forward pass; by the time this line runs, this
                # rank's own compute for this chunk has GENUINELY
                # finished (not merely "message received"). No
                # separate round-trip needed on THIS rank's side --
                # rank 0 is the one that blocks, waiting to receive
                # this ack, mirroring the established
                # _PREFILL_READY_MAX_WAIT_SECONDS bounded-wait pattern
                # this same file already uses at the admission
                # handshake.
                send_prefill_chunk_done_ack_message(
                    PrefillChunkDoneAckMessage(
                        step_id=advance_message.step_id,
                        request_id=advance_message.request_id,
                        chunk_index=advance_message.chunk_index,
                    ),
                    dst=self.src_rank,
                    group=self.group,
                )
                return (
                    None,
                    None,
                    PrefillAdvanceCompleted(
                        request_id=advance_message.request_id,
                        output=prefill_session.output(),
                    ),
                )
            return None, None, None
        if header.msg_kind == MSG_KIND_PREFILL_ABORT:
            # 2026-08-07: real cancel/abort mechanism. Rank 0 has
            # ALREADY closed its own local session and decided this
            # request's chunk-drive is dead -- rank 1's job here is
            # purely reactive: close its OWN mirrored session (never
            # independently decided, matching every other branch in
            # this dispatch's "rank 1 never independently decides"
            # discipline) and ack. Per the `consult` review that
            # corrected this design's earlier wrong assumption: rank 1
            # ALWAYS has live session state once register_prefill_
            # session has run on both ranks (not merely during
            # RANK1_DRAINING), so this branch must be reachable
            # regardless of what _prefill_phase rank 0's OWN state
            # machine was in when it decided to abort -- there is no
            # narrower "only valid during X" precondition to check
            # here (unlike MSG_KIND_PREFILL_ADVANCE's phase-implicit
            # checks above, which rely on rank 0 having already walked
            # through rank0_local/handoff before ever sending an
            # advance).
            abort_message = decode_prefill_abort_message(header)
            if self._active_prefill_session is None:
                raise GlueError(
                    f"tick(): received PrefillAbortMessage for "
                    f"request_id={abort_message.request_id} but this "
                    f"rank has no active local prefill session registered "
                    f"-- the two ranks' control-message streams have "
                    f"desynced (rank 0 must never send an abort for a "
                    f"request this rank was never told to register)."
                )
            active_id, prefill_session = self._active_prefill_session
            if active_id != abort_message.request_id:
                raise GlueError(
                    f"tick(): PrefillAbortMessage request_id mismatch -- "
                    f"this rank's active session is for "
                    f"request_id={active_id}, received an abort for "
                    f"request_id={abort_message.request_id}. The two "
                    f"ranks' control-message streams have desynced."
                )
            prefill_session.abort()
            self._active_prefill_session = None
            self._last_prefill_advance_seq = None
            send_prefill_abort_ack_message(
                PrefillAbortAckMessage(
                    step_id=abort_message.step_id,
                    request_id=abort_message.request_id,
                ),
                dst=self.src_rank,
                group=self.group,
            )
            return None, None, None
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
            # 2026-08-06 bug fix: build_evict_ack (NOT a hand-built
            # EvictAckMessage) is what actually transitions this
            # rank's own RankOneMirror._slot_state from DRAINING back
            # to FREE -- see Rank0/RankOneMirrorDriver.build_evict_ack's
            # own docstring for the real production bug this closes
            # (nothing called it before; a slot stayed permanently
            # DRAINING after its first eviction, exploding the first
            # real N=2 slot-reuse under load with a
            # ProtocolViolationError). Called AFTER release_slot, per
            # this method's own "free the real cache state, then ack"
            # ordering -- release_slot only touches this glue's own
            # cache_router bookkeeping, build_evict_ack only touches
            # the SEPARATE mirror validator's _slot_state, so their
            # relative order doesn't matter for correctness, but
            # keeping "the thing that's true first" (cache genuinely
            # freed) before "the thing that says it's safe to reuse"
            # (mirror state FREE) is the more defensible sequencing.
            ack = self.session.build_evict_ack(evict_message)
            send_evict_ack_message(ack, dst=self.src_rank, group=self.group)
            # 2026-08-06 bug #7 fix: return the evicted request_id so
            # the caller can synthesize a real finish-classified
            # response -- see this method's own docstring for the
            # full incident this closes (runner.py's admission gate
            # on rank 1 could never drain without this).
            return None, evict_message.request_id, None
        else:
            raise GlueError(
                f"tick(): received unexpected msg_kind={header.msg_kind} "
                f"-- rank 1's glue only ever expects MSG_KIND_PREFILL, "
                f"MSG_KIND_PREFILL_ADVANCE, MSG_KIND_STEP, or MSG_KIND_EVICT "
                f"next on the wire for this session"
            )
        return None, None, None

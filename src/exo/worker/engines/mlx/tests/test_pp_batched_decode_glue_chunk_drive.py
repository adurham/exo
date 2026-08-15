# pyright: reportPrivateUsage=false, reportAny=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
"""Unit-level correctness test for Rank0BatchedDecodeGlue's chunk-
drive state machine (RANK0_LOCAL -> HANDOFF -> RANK1_DRAINING) --
2026-08-06, the fix for the REAL, hardware-relevant bug found before
this mechanism was ever wired into live serving:

  1. PrefillAdvanceMessage.max_layers was sent identically to both
     ranks, but the real 43-layer DSv4-Flash 2-rank topology gives
     rank 0 and rank 1 GENUINELY DIFFERENT local layer counts
     (memory-weighted placement allocation, not an even-split
     formula) -- meaning a naive "advance both ranks the same number
     of times" design would desync the two ranks' chunk-completion
     timing.
  2. MetaFramedPipelineFirstLayer.__call__ BLOCKS on recv_metaframe as
     the literal first op of rank 1's own first local layer -- rank 1
     cannot make ANY progress until rank 0 has walked its ENTIRE local
     stack and flushed its activation onto the wire. The original
     design sent rank 1 a PrefillAdvanceMessage on EVERY one of rank
     0's own advance ticks, which would have made rank 1's tick()
     block hard, inside the single-writer call site that must also
     service decode and admission, for the entire remaining duration
     of rank 0's local-layer traversal.

This test proves the fix WITHOUT needing a real 2-process/RDMA
transport: drives a REAL Rank0BatchedDecodeGlue (real tick() calls,
real PrefillAdvanceMessage encode/decode against an in-process fake
transport) with a DELIBERATELY UNEVEN peer_prefill_layer_count (the
whole point of the fix), and independently verifies -- by counting
actual messages sent and by mirroring the exact advance sequence into
a second, real ResumablePrefillSession standing in for rank 1's own
local session -- that:
  (a) NO PrefillAdvanceMessage is sent until rank 0's own local
      session has genuinely finished (closes bug #2: rank 1 is never
      told to advance before it has real data to make progress on).
  (b) The number of PrefillAdvanceMessages sent exactly matches what
      rank 1's own (genuinely different) layer count requires (closes
      bug #1: no desync from an even-split assumption).
  (c) Mirroring those exact messages into a second local session
      (standing in for rank 1's real one) reaches done=True on
      EXACTLY the last message, never early or late.
"""

from __future__ import annotations

from typing import Iterator, cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.pp_batched_decode_adapter import (
    BatchedDecodeResponseAdapter,
)
from exo.worker.engines.mlx.pp_batched_decode_glue import (
    GlueError,
    PrefillAdvanceCompleted,
    Rank0BatchedDecodeGlue,
    Rank1BatchedDecodeGlue,
)
from exo.worker.engines.mlx.pp_batched_decode_runtime import (
    BatchedDecodeSession,
    RankOneMirrorSession,
)
from exo.worker.engines.mlx.pp_metaframe import ForwardPhase, get_forward_step_info
from exo.worker.engines.mlx.pp_prefill_session import (
    ForwardStep,
    ResumablePrefillSession,
)
from exo.worker.engines.mlx.pp_scheduler_protocol import (
    PrefillAbortMessage,
    PrefillAdvanceMessage,
)

pytestmark = pytest.mark.filterwarnings("ignore")


class _CountingLayerModel:
    """Same structural shape as test_pp_prefill_session.py's own
    _FakeInterruptibleModel -- a real, working _forward_steps
    generator (not a mock), just with a caller-controlled layer
    count, so this test can construct rank 0's and rank 1's sessions
    with GENUINELY DIFFERENT n_layers, matching the real uneven-split
    scenario this fix closes."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers

    def _forward_steps(
        self,
        inputs: mx.array,
        cache: object = None,
        *,
        interruptible: bool = False,
    ) -> Iterator[ForwardStep]:
        h = inputs
        for i in range(self.n_layers):
            get_forward_step_info()  # exercised for real, not bypassed
            h = h + mx.array([1.0])
            mx.eval(h)
            if interruptible:
                yield ("layer", i, h)
        yield ("done", None, h)


class _RankGroupStub:
    """Minimal mx.distributed.Group stand-in for tick()'s send call --
    dst_rank/group are only ever read by send_header/mx.distributed.send,
    both globally monkeypatched below."""

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 2


def _make_rank0_glue(*, peer_prefill_layer_count: int) -> Rank0BatchedDecodeGlue:
    session = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session, eos_ids=frozenset())
    return Rank0BatchedDecodeGlue(
        session=session,
        adapter=adapter,
        dst_rank=1,
        group=cast(mx.distributed.Group, cast(object, _RankGroupStub())),
        peer_prefill_layer_count=peer_prefill_layer_count,
    )


def _make_rank1_glue() -> Rank1BatchedDecodeGlue:
    session = RankOneMirrorSession.new(max_concurrency=2)
    return Rank1BatchedDecodeGlue(
        session=session,
        src_rank=0,
        group=cast(mx.distributed.Group, cast(object, _RankGroupStub())),
    )


def _feed_prefill_abort_message(
    monkeypatch: pytest.MonkeyPatch, message: PrefillAbortMessage
) -> list[PrefillAbortMessage]:
    """Monkeypatch recv_header (the ONLY thing Rank1BatchedDecodeGlue.
    tick() calls to learn what arrived next) to return a real
    ``WireHeader`` encoding ``message`` -- then let the REAL
    decode_prefill_abort_message + PrefillAbortAckMessage send path
    run against a captured, in-process fake transport (same pattern
    as ``_capture_sent_advances``, applied to the abort ack instead of
    advance messages)."""
    import exo.worker.engines.mlx.pp_scheduler_wire as wire_mod

    fake_header = wire_mod.WireHeader(
        version=wire_mod.SCHEDULER_WIRE_PROTOCOL_VERSION,
        msg_kind=wire_mod.MSG_KIND_PREFILL_ABORT,
        step_id=message.step_id,
        field_d=message.request_id,
        field_e=0,
    )

    recv_header_call_count = [0]

    def _fake_recv_header(src: int, *, group: object) -> object:
        del src, group
        recv_header_call_count[0] += 1
        if recv_header_call_count[0] > 1:
            raise RuntimeError(
                "recv_header called a SECOND time -- the real wire has "
                "no second header queued for a MSG_KIND_PREFILL_ABORT "
                "branch to receive; this would deadlock on real "
                "hardware. A correct implementation must decode the "
                "abort message from the header tick() ALREADY received "
                "at the top of the method, never call recv_header again."
            )
        return fake_header

    sent_acks: list[PrefillAbortMessage] = []

    def _fake_send_header(header_arr: mx.array, dst: int, *, group: object) -> None:
        del dst, group
        mx.eval(header_arr)
        values = cast(list[int], header_arr.tolist())
        # [version, msg_kind, step_id, field_d, field_e]
        sent_acks.append(PrefillAbortMessage(step_id=values[2], request_id=values[3]))

    monkeypatch.setattr(wire_mod, "recv_header", _fake_recv_header)
    monkeypatch.setattr(wire_mod, "send_header", _fake_send_header)
    import exo.worker.engines.mlx.pp_batched_decode_glue as glue_mod

    monkeypatch.setattr(glue_mod, "recv_header", _fake_recv_header)
    return sent_acks


def _capture_sent_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> list[PrefillAdvanceMessage]:
    """Monkeypatch mx.distributed.send/send_header at the module level
    tick() actually calls them through, capturing every real
    PrefillAdvanceMessage this glue's tick() sends -- NOT a mock of
    tick() itself, the real send/recv_prefill_advance_body round-trip
    still runs, just against captured arrays instead of a real
    transport."""
    sent: list[PrefillAdvanceMessage] = []
    import exo.worker.engines.mlx.pp_scheduler_wire as wire_mod

    real_send_header = wire_mod.send_header
    real_mx_send = mx.distributed.send

    def _fake_mx_send(arr: mx.array, dst: int, *, group: object) -> mx.array:
        del dst, group
        mx.eval(arr)
        return arr

    def _fake_mx_recv_like(x: mx.array, src: int, *, group: object) -> mx.array:
        # 2026-08-08, real production incident fix (design doc
        # Section 21/22): tick()'s RANK1_DRAINING completion branch
        # now does a real bounded blocking recv for
        # PrefillChunkDoneAckMessage before returning
        # PrefillAdvanceCompleted -- this test harness previously
        # never needed to stub mx.distributed.recv_like at all
        # (rank 0's tick() was send-only during RANK1_DRAINING before
        # this fix). Echo back the request_id/chunk_index of the
        # MOST RECENTLY captured PrefillAdvanceMessage -- that IS the
        # advance whose completion triggered this recv (tick()'s own
        # call ordering: send the last advance, THEN recv its ack, on
        # the same call), so `sent[-1]` always holds exactly the
        # right values, correct across multiple chunks/requests
        # within one test with no separate stateful bookkeeping
        # needed here.
        del src, group
        last = sent[-1]
        header = mx.array(
            [
                wire_mod.SCHEDULER_WIRE_PROTOCOL_VERSION,
                wire_mod.MSG_KIND_PREFILL_CHUNK_DONE_ACK,
                last.step_id,
                last.request_id,
                last.chunk_index,
            ],
            dtype=x.dtype,
        )
        return header

    captured_headers: list[mx.array] = []

    def _fake_send_header(header: mx.array, dst: int, *, group: object) -> None:
        del dst, group
        mx.eval(header)
        captured_headers.append(header)

    monkeypatch.setattr(wire_mod, "send_header", _fake_send_header)
    monkeypatch.setattr(mx.distributed, "send", _fake_mx_send)
    monkeypatch.setattr(mx.distributed, "recv_like", _fake_mx_recv_like)

    orig_send_prefill_advance_message = wire_mod.send_prefill_advance_message

    def _capturing_send(
        message: PrefillAdvanceMessage, dst: int, *, group: object
    ) -> None:
        sent.append(message)
        orig_send_prefill_advance_message(
            message, dst, group=cast("mx.distributed.Group", group)
        )

    monkeypatch.setattr(wire_mod, "send_prefill_advance_message", _capturing_send)
    # tick() imports send_prefill_advance_message directly into its own
    # module namespace -- patch that reference too, matching how the
    # real production import shape works.
    import exo.worker.engines.mlx.pp_batched_decode_glue as glue_mod

    monkeypatch.setattr(glue_mod, "send_prefill_advance_message", _capturing_send)
    del real_send_header, real_mx_send
    return sent


def test_no_advance_sent_before_rank0_local_session_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug #2's direct regression test: rank 0's own session needs
    MULTIPLE real advance() calls to finish (n_layers=6 yields 3 real
    "layer" pauses at max_layers=2, THEN a separate 4th call is needed
    to consume the generator's own "done" sentinel -- ResumablePrefillSession's
    own documented contract) -- confirm ZERO PrefillAdvanceMessages
    are sent during ANY of those RANK0_LOCAL ticks, and the FIRST
    message only appears on the tick where rank 0's own session
    reaches done=True (that SAME tick falls through into
    HANDOFF+RANK1_DRAINING and sends the first real advance -- by
    design, to avoid wasting a tick on a pure state transition)."""
    sent = _capture_sent_advances(monkeypatch)
    glue = _make_rank0_glue(peer_prefill_layer_count=4)
    model = _CountingLayerModel(n_layers=6)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=session, chunk_index=0)

    # Tick while still in RANK0_LOCAL -- confirm ZERO messages sent on
    # every one of these ticks, whatever the exact count turns out to
    # be (this test's own point is "none until real completion", not
    # a specific tick count -- that arithmetic detail belongs to
    # ResumablePrefillSession's own tests, not duplicated here).
    result = glue.tick(model=cast("object", model))
    ticks_in_rank0_local = 1
    while glue._prefill_phase == "rank0_local":
        assert sent == [], "rank 1 must not be messaged before rank 0 finishes locally"
        assert result[3] is None  # no PrefillAdvanceCompleted yet
        result = glue.tick(model=cast("object", model))
        ticks_in_rank0_local += 1
        assert ticks_in_rank0_local < 20, "runaway loop -- phase never transitioned"

    # The tick where rank 0's own local session reaches done=True
    # falls through into HANDOFF and RANK1_DRAINING on the SAME tick,
    # sending the first real advance to rank 1.
    assert glue._prefill_phase == "rank1_draining"
    assert len(sent) == 1, (
        f"the tick where rank 0's own session completes must ALSO send "
        f"the first advance to rank 1 (same-tick fall-through, no "
        f"wasted tick) -- got {len(sent)} messages"
    )

    # peer_prefill_layer_count=4 at default max_layers=2 needs
    # floor(4/2)+1 = 3 total advances -- drain the remaining ones.
    #
    # UPDATED 2026-08-15 (design doc Section 45): this asserted
    # ceil(4/2)==2, which encoded the very off-by-one that caused the
    # multi-session mutual deadlock. advance() only reports done=True on
    # the call that consumes the generator's trailing ("done", ...)
    # sentinel, so when max_layers evenly divides the peer's layer count
    # (4 % 2 == 0 here) the layer-consuming calls all return done=False
    # and ONE MORE advance is genuinely required. Confirmed against the
    # real ResumablePrefillSession in
    # test_pp_prefill_advance_budget_parity.py.
    while result[3] is None:
        result = glue.tick(model=cast("object", model))
    assert len(sent) == 3, (
        f"peer_prefill_layer_count=4 at default max_layers=2 needs "
        f"floor(4/2)+1=3 real advances to rank 1 (the 3rd consumes the "
        f"'done' sentinel); got {len(sent)}"
    )


def test_advance_count_matches_uneven_peer_layer_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE core proof: rank 0's own n_layers (7) and rank 1's
    peer_prefill_layer_count (5) are DELIBERATELY DIFFERENT (mirrors
    the real confirmed 22 vs 21 split) -- confirm the number of
    PrefillAdvanceMessages sent is EXACTLY ceil(5/2)=3, driven purely
    by the peer's real layer count, never rank 0's own."""
    sent = _capture_sent_advances(monkeypatch)
    glue = _make_rank0_glue(peer_prefill_layer_count=5)
    model = _CountingLayerModel(n_layers=7)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=session, chunk_index=0)

    completed: PrefillAdvanceCompleted | None = None
    for _tick in range(20):
        result = glue.tick(model=cast("object", model))
        if result[3] is not None:
            completed = result[3]
            break

    assert completed is not None, "chunk never reported complete"
    assert completed.request_id == 1
    assert len(sent) == 3, f"expected ceil(5/2)=3 advances to rank 1, got {len(sent)}"
    assert not glue.has_active_prefill_session()


def test_mirrored_rank1_session_reaches_done_on_exactly_the_last_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drives the REAL sequence of PrefillAdvanceMessages this glue
    sends into a SECOND, independent ResumablePrefillSession (standing
    in for rank 1's own local session, exactly matching what
    Rank1BatchedDecodeGlue.tick()'s real MSG_KIND_PREFILL_ADVANCE
    branch does with each message's max_layers) -- confirms the mirror
    session reaches done=True on EXACTLY the final captured message,
    never early (which would mean rank 0 under-drove it) or late
    (which would mean rank 0 over-drove it past rank 1's real
    completion, wasting a message on an already-finished session --
    ResumablePrefillSession.advance's own fail-loud guard would catch
    that as a real PrefillSessionError)."""
    sent = _capture_sent_advances(monkeypatch)
    rank1_layer_count = 5
    glue = _make_rank0_glue(peer_prefill_layer_count=rank1_layer_count)
    rank0_model = _CountingLayerModel(n_layers=8)
    rank0_session = ResumablePrefillSession(
        inner_model=rank0_model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=rank0_session, chunk_index=0)

    for _tick in range(20):
        result = glue.tick(model=cast("object", rank0_model))
        if result[3] is not None:
            break

    assert len(sent) >= 1

    rank1_model = _CountingLayerModel(n_layers=rank1_layer_count)
    rank1_session = ResumablePrefillSession(
        inner_model=rank1_model, inputs=mx.array([0.0]), cache=[]
    )
    for i, message in enumerate(sent):
        is_last = i == len(sent) - 1
        _layers_advanced, done = rank1_session.advance(
            max_layers=message.max_layers,
            phase_for_pause=(
                ForwardPhase.PREFILL_FINAL if is_last else ForwardPhase.PREFILL_CONTINUE
            ),
        )
        if is_last:
            assert done, (
                f"mirror rank-1 session did NOT reach done=True on the "
                f"FINAL captured advance (message {i + 1}/{len(sent)}) -- "
                f"rank 0 under-drove rank 1's real layer count"
            )
        else:
            assert not done, (
                f"mirror rank-1 session reached done=True EARLY, on "
                f"message {i + 1}/{len(sent)} (not the final one) -- rank "
                f"0 would have kept sending advances to an "
                f"already-completed rank-1 session"
            )


def test_register_prefill_session_requires_peer_layer_count_set() -> None:
    """Construction-time guard: a glue built with
    peer_prefill_layer_count=0 (the invalid/unset sentinel) must
    refuse to register a session at all -- RANK1_DRAINING has no way
    to compute a real advance count without it."""
    session_obj = BatchedDecodeSession.new(max_concurrency=2)
    adapter = BatchedDecodeResponseAdapter(session=session_obj, eos_ids=frozenset())
    glue = Rank0BatchedDecodeGlue(
        session=session_obj,
        adapter=adapter,
        dst_rank=1,
        group=cast(mx.distributed.Group, cast(object, _RankGroupStub())),
        peer_prefill_layer_count=0,
    )
    model = _CountingLayerModel(n_layers=2)
    prefill_session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    with pytest.raises(GlueError, match="peer_prefill_layer_count"):
        glue.register_prefill_session(
            request_id=1, session=prefill_session, chunk_index=0
        )


def test_prefill_advance_messages_carry_the_registered_chunk_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every real PrefillAdvanceMessage sent for a session must carry
    THAT session's own registered chunk_index -- confirms the field
    threads through correctly, not just defaults to 0 by accident."""
    sent = _capture_sent_advances(monkeypatch)
    glue = _make_rank0_glue(peer_prefill_layer_count=3)
    model = _CountingLayerModel(n_layers=2)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=7, session=session, chunk_index=4)

    for _tick in range(10):
        result = glue.tick(model=cast("object", model))
        if result[3] is not None:
            break

    assert len(sent) >= 1
    assert all(m.chunk_index == 4 for m in sent)
    assert all(m.request_id == 7 for m in sent)


def test_msg_kind_prefill_advance_still_decodes_the_real_captured_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check tying this file's in-process capture back to the
    REAL wire encode/decode path (test_pp_scheduler_wire.py's own
    concern) -- confirms the messages captured here are genuinely
    well-formed PrefillAdvanceMessage instances, not just
    Python-level mock objects that happen to satisfy this test file's
    own assertions."""
    sent = _capture_sent_advances(monkeypatch)
    glue = _make_rank0_glue(peer_prefill_layer_count=3)
    model = _CountingLayerModel(n_layers=2)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=session, chunk_index=0)
    for _tick in range(10):
        result = glue.tick(model=cast("object", model))
        if result[3] is not None:
            break
    assert len(sent) >= 1
    for message in sent:
        assert isinstance(message, PrefillAdvanceMessage)
        assert message.max_layers >= 1
        assert message.advance_seq >= 1


def test_no_new_prefill_granted_while_chunk_drive_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-08-07 priority-order guard (Phase 2 live-wiring, found via
    a `consult` review of the live-wiring design BEFORE it was
    implemented): tick()'s fixed priority order checks "grant a new
    prefill" (branch 2) BEFORE "advance an active chunk-drive session"
    (branch 3) -- without an explicit guard, a SECOND pending prefill
    could be granted mid-drive, racing request A's still-in-flight
    chunk session against request B's brand-new one. This test proves
    the guard: a chunk session for request 1 is registered and
    genuinely mid-drive (NOT yet done), then a second request (2) is
    enqueued via enqueue_prefill -- every subsequent tick() must keep
    driving request 1's chunk session, NEVER granting request 2's
    prefill, until request 1's drive genuinely completes and clears
    _active_prefill_session.

    Verified load-bearing: reverting the guard (dropping the
    `self._active_prefill_session is None` condition from tick()'s
    pending-prefill branch) makes this test fail loudly -- tick()
    would grant request 2's PrefillMessage on an early tick, well
    before request 1's chunk session reaches done=True.
    """
    _sent = _capture_sent_advances(monkeypatch)
    glue = _make_rank0_glue(peer_prefill_layer_count=4)
    model = _CountingLayerModel(n_layers=6)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=session, chunk_index=0)
    glue.enqueue_prefill(
        request_id=2,
        cache_slot=1,
        n_prompt_tokens=4,
        single_request_fallback=False,
    )

    saw_completion = False
    for _tick in range(20):
        assert glue.has_pending_prefills(), (
            "request 2's prefill must remain pending -- the priority-"
            "order guard must never let tick() reach the grant branch "
            "while request 1's chunk-drive session is still active"
        )
        result = glue.tick(model=cast("object", model))
        grant = result[2]
        assert grant is None, (
            f"tick() granted a new prefill (request_id="
            f"{grant.request_id if grant else None}) while request 1's "
            f"chunk-drive session was still active -- this is exactly "
            f"the cross-request interleaving hazard the priority-order "
            f"guard exists to prevent"
        )
        if result[3] is not None:
            saw_completion = True
            break

    assert saw_completion, "request 1's chunk-drive session never completed"
    assert not glue.has_active_prefill_session()

    # NOW that request 1's drive is genuinely done, request 2's
    # pending prefill must finally be REACHABLE (no longer blocked by
    # the priority-order guard) -- checked via has_pending_prefills()
    # staying True (never popped) combined with the guard condition
    # itself now being satisfied, rather than driving the full
    # PrefillMessage/PrefillReadyMessage wire round-trip (a real
    # 2-rank handshake genuinely requiring a recv_prefill_ready_message
    # response this single-rank test has no peer to provide -- that
    # full handshake is already covered by
    # test_pp_admission_race_subprocess.py's real 2-process harness).
    assert glue.has_pending_prefills(), (
        "request 2's prefill must still be queued, ready to be granted "
        "on the very next tick now that the guard condition "
        "(active_prefill_session is None) is satisfied"
    )


def test_msg_kind_prefill_abort_closes_rank1_session_and_sends_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-08-07 real cancel/abort mechanism: rank 1's own
    ``Rank1BatchedDecodeGlue.tick()`` reactively closes ITS local
    session when a real ``PrefillAbortMessage`` arrives (never
    independently decides -- this module's own established
    discipline, per that branch's own docstring), then replies with a
    ``PrefillAbortAckMessage`` carrying the SAME ``request_id``.

    This is also the regression test for a REAL bug this test caught
    while being written: the first implementation called
    ``recv_prefill_abort_message(self.src_rank, group=self.group)``
    INSIDE the branch -- but ``tick()``'s dispatch loop had ALREADY
    consumed the header via its own earlier ``recv_header`` call at
    the top of the method (matching every other branch's established
    pattern: decode from the already-received header, e.g.
    ``decode_evict_message(header)``, never re-receive a second one).
    That bug would have deadlocked the real wire (rank 1 blocking on a
    SECOND header rank 0 never sends) -- caught here, not on real
    hardware, because this test drives the REAL ``tick()`` dispatch
    with a REAL (monkeypatched-transport) header already queued for
    ``recv_header`` to consume exactly once.
    """
    glue = _make_rank1_glue()
    model = _CountingLayerModel(n_layers=6)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=session)
    assert glue.has_active_prefill_session()

    sent_acks = _feed_prefill_abort_message(
        monkeypatch, PrefillAbortMessage(step_id=1, request_id=1)
    )

    grant, evicted_request_id, prefill_advance_completed = glue.tick(
        model=cast("object", model)
    )

    assert grant is None
    assert evicted_request_id is None
    assert prefill_advance_completed is None
    assert not glue.has_active_prefill_session(), (
        "rank 1's own session must be genuinely closed after receiving "
        "the abort -- glue._active_prefill_session must be cleared, not "
        "left permanently occupied"
    )
    assert len(sent_acks) == 1
    assert sent_acks[0].request_id == 1


def test_msg_kind_prefill_abort_for_unregistered_request_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-stop guard: rank 1 receiving a ``PrefillAbortMessage`` for
    a ``request_id`` it never registered a session for is a genuine
    cross-rank desync (rank 0 must never send an abort for a request
    rank 1 was never told to track) -- must raise ``GlueError``, not
    silently no-op."""
    glue = _make_rank1_glue()
    _feed_prefill_abort_message(
        monkeypatch, PrefillAbortMessage(step_id=1, request_id=999)
    )

    with pytest.raises(GlueError, match="no active local prefill session"):
        glue.tick(model=cast("object", None))


def test_chunk_done_ack_mismatch_raises_instead_of_silently_registering_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-08-08 real production incident fix (design doc Section
    21/22): rank 0's own "chunk complete" decision (all advances
    SENT) does not mean rank 1 has finished PROCESSING them -- a
    real, reproduced-on-hardware race where rank 1's own local Metal
    compute for a chunk's tail layers was still running when rank 0
    would previously have raced ahead into the next chunk. This test
    proves the actual fail-loud fix: if the ``PrefillChunkDoneAckMessage``
    rank 0 blocks for arrives with a MISMATCHED request_id/chunk_index
    (standing in for a genuinely desynced ack -- the class of bug a
    real transport bug or a genuine control-stream desync would
    produce), ``tick()`` must raise ``GlueError`` rather than silently
    trusting the wrong ack and declaring the wrong chunk done."""
    sent = _capture_sent_advances(monkeypatch)
    glue = _make_rank0_glue(peer_prefill_layer_count=4)
    model = _CountingLayerModel(n_layers=1)
    session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=session, chunk_index=0)

    import exo.worker.engines.mlx.pp_scheduler_wire as wire_mod

    def _mismatched_recv_like(x: mx.array, src: int, *, group: object) -> mx.array:
        # Deliberately wrong request_id -- standing in for a genuine
        # cross-rank desync, not the happy-path echo the shared
        # _capture_sent_advances helper normally returns.
        del src, group
        return mx.array(
            [
                wire_mod.SCHEDULER_WIRE_PROTOCOL_VERSION,
                wire_mod.MSG_KIND_PREFILL_CHUNK_DONE_ACK,
                sent[-1].step_id if sent else 0,
                999,  # wrong request_id
                sent[-1].chunk_index if sent else 0,
            ],
            dtype=x.dtype,
        )

    monkeypatch.setattr(mx.distributed, "recv_like", _mismatched_recv_like)

    with pytest.raises(GlueError, match="PrefillChunkDoneAckMessage mismatch"):
        for _tick in range(20):
            result = glue.tick(model=cast("object", model))
            if result[3] is not None:
                break


def test_reset_chunk_drive_state_after_reconnect_clears_stuck_rank0_session() -> None:
    """Regression test for the 2026-08-09 real-hardware finding (design
    doc Section 27/28): a jaccl transport fault mid-chunk-drive used to
    leave Rank0BatchedDecodeGlue's _active_prefill_session/_prefill_phase/
    _prefill_rank1_advances_remaining stuck exactly where the fault
    interrupted them, because ExoBatchGenerator.reset_after_reconnect()
    only ever cleared _active_tasks/_mlx_gen -- never either glue's own
    chunk-drive state. Confirmed on real hardware: the NEXT request's
    register_prefill_session() call then either hit the "already active"
    guard against a session no client would ever complete, or ran into a
    corrupted state combination that tripped the RANK1_DRAINING
    fail-loud guard (a full runner crash).

    This proves reset_chunk_drive_state_after_reconnect() genuinely
    clears the stuck state -- register_prefill_session() for a NEW
    request must succeed afterward instead of raising "already active"."""
    glue = _make_rank0_glue(peer_prefill_layer_count=4)
    model = _CountingLayerModel(n_layers=7)
    stuck_session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    # Simulate a fault hitting mid-drive: register a session and advance
    # it partway (RANK0_LOCAL, not yet handed off) -- standing in for
    # whatever real _prefill_phase a fault could interrupt at, since the
    # reset method's job is to unconditionally clear ALL of them, not
    # just this one.
    glue.register_prefill_session(request_id=1, session=stuck_session, chunk_index=0)
    assert glue.has_active_prefill_session()

    dropped_id = glue.reset_chunk_drive_state_after_reconnect()

    assert dropped_id == 1
    assert not glue.has_active_prefill_session()
    assert glue._prefill_phase == "rank0_local"
    assert glue._prefill_rank1_advances_remaining == 0

    # THE key proof: a genuinely NEW request can now register without
    # hitting the "already active" guard -- this is exactly what a
    # runner crash on real hardware looked like before this fix (the
    # guard either fired, or the corrupted state tripped RANK1_DRAINING).
    new_model = _CountingLayerModel(n_layers=7)
    new_session = ResumablePrefillSession(
        inner_model=new_model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=2, session=new_session, chunk_index=0)
    assert glue.has_active_prefill_session()


def test_reset_chunk_drive_state_after_reconnect_is_a_noop_when_idle() -> None:
    """No active session -- the reset must return None and touch nothing,
    matching the real call site's usage (called unconditionally on every
    jaccl reconnect, whether or not a chunk-drive was actually active)."""
    glue = _make_rank0_glue(peer_prefill_layer_count=4)
    assert not glue.has_active_prefill_session()

    dropped_id = glue.reset_chunk_drive_state_after_reconnect()

    assert dropped_id is None
    assert not glue.has_active_prefill_session()


def test_reset_chunk_drive_state_after_reconnect_clears_stuck_rank1_session() -> None:
    """Rank 1's mirror of the rank-0 test above."""
    glue = _make_rank1_glue()
    model = _CountingLayerModel(n_layers=5)
    stuck_session = ResumablePrefillSession(
        inner_model=model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=1, session=stuck_session)
    assert glue.has_active_prefill_session()

    dropped_id = glue.reset_chunk_drive_state_after_reconnect()

    assert dropped_id == 1
    assert not glue.has_active_prefill_session()

    new_model = _CountingLayerModel(n_layers=5)
    new_session = ResumablePrefillSession(
        inner_model=new_model, inputs=mx.array([0.0]), cache=[]
    )
    glue.register_prefill_session(request_id=2, session=new_session)
    assert glue.has_active_prefill_session()

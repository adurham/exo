# pyright: reportPrivateUsage=false
"""Phase 1 correctness tests for the wire-protocol state machine
(``pp_scheduler_protocol.py``).

See ``docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md`` Section 8
Risk #11 and Section 11's answered open question #1: logic-level
fuzzing of the batch-composition/cache-routing state machine, BEFORE
any real Phase 1 scheduler code exists, targeting exactly Risk #5/#11
(silent cross-request corruption, deadlock).

No ``hypothesis`` dependency -- not present in this project
(``uv.lock``/``pyproject.toml`` checked, confirmed absent) and adding
one is out of scope without the user's explicit go-ahead for a new
dependency. Instead: a hand-rolled, seeded PRNG-driven fuzzer using
only the stdlib ``random`` module. This has a real advantage over
``hypothesis`` for this specific use case anyway, per the `consult`
review's own framing -- deterministic, exactly-reproducible failures
(same seed -> same event sequence -> same failure, every time) matter
more here than ``hypothesis``'s automatic shrinking, since this state
machine is small enough that a failing seed's event log IS already a
minimal reproduction once printed.

Structure, per the consult review's guidance:
1. Directed unit tests for each individual invariant (fast, pinpoint
   the exact violation each raises).
2. A seeded random fuzzer running many event sequences through BOTH
   ``SchedulerCore`` (rank 0's view) and ``RankOneMirror`` (rank 1's
   view) in lockstep, asserting they never disagree.
3. TARGETED (not just uniform-random) generation of the two sequence
   shapes the consult explicitly flagged as under-sampled by uniform
   random event generation: abort-while-in-flight and
   abort-then-immediate-new-request -- the exact race this whole
   module exists to prevent (module docstring point 4).
"""

from __future__ import annotations

import random

import pytest

from exo.worker.engines.mlx.pp_scheduler_protocol import (
    BatchEntry,
    Command,
    EvictAckReceivedEvent,
    EvictMessage,
    NewRequestEvent,
    Phase,
    ProtocolViolationError,
    RankOneMirror,
    RequestAbortedEvent,
    RequestDoneEvent,
    SchedulerCore,
    SendEvictCommand,
    SendStepCommand,
    SlotState,
    StepMessage,
    TokenGeneratedEvent,
)

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Directed unit tests -- one per invariant, for a fast pinpoint failure
# when a future edit breaks a specific rule.
# ---------------------------------------------------------------------


def test_new_request_assigns_slot_and_emits_step() -> None:
    core = SchedulerCore(max_concurrency=2)
    commands = core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    assert len(commands) == 1
    cmd = commands[0]
    assert isinstance(cmd, SendStepCommand)
    assert cmd.message.step_id == 1
    assert len(cmd.message.entries) == 1
    entry = cmd.message.entries[0]
    assert entry.request_id == 1
    assert entry.cache_slot == 0
    assert entry.expected_cache_len == 0


def test_duplicate_request_id_raises() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    with pytest.raises(ProtocolViolationError, match="ALREADY active"):
        core.handle(NewRequestEvent(request_id=1, cache_slot=1))


def test_new_request_on_non_free_slot_raises() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    with pytest.raises(ProtocolViolationError, match="not FREE"):
        core.handle(NewRequestEvent(request_id=2, cache_slot=0))


def test_max_concurrency_enforced() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    core.handle(NewRequestEvent(request_id=2, cache_slot=1))
    with pytest.raises(ProtocolViolationError, match="max_concurrency"):
        core.handle(NewRequestEvent(request_id=3, cache_slot=2))


def test_token_generated_advances_cache_len() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    commands = core.handle(TokenGeneratedEvent(request_ids=(1,)))
    cmd = commands[0]
    assert isinstance(cmd, SendStepCommand)
    assert cmd.message.entries[0].expected_cache_len == 1


def test_token_generated_for_unknown_request_raises() -> None:
    core = SchedulerCore(max_concurrency=2)
    with pytest.raises(ProtocolViolationError, match="not active"):
        core.handle(TokenGeneratedEvent(request_ids=(99,)))


def test_token_generated_empty_request_ids_raises_at_construction() -> None:
    """TokenGeneratedEvent's own __post_init__ invariant -- an empty
    request_ids tuple is rejected before it ever reaches
    SchedulerCore.handle at all."""
    with pytest.raises(ProtocolViolationError, match="non-empty"):
        TokenGeneratedEvent(request_ids=())


def test_token_generated_batched_advances_both_requests_same_step() -> None:
    """THE real batched-decode case: N=2 requests advancing in ONE
    real forward pass must be represented as ONE TokenGeneratedEvent
    with both request_ids, not two separate events -- per the design
    rationale in TokenGeneratedEvent's own docstring (a consult
    review: splitting into N events would make RankOneMirror pass
    through intermediate states that never existed on rank 0)."""
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    core.handle(NewRequestEvent(request_id=2, cache_slot=1))
    commands = core.handle(TokenGeneratedEvent(request_ids=(1, 2)))
    assert len(commands) == 1
    cmd = commands[0]
    assert isinstance(cmd, SendStepCommand)
    entries_by_id = {e.request_id: e for e in cmd.message.entries}
    assert entries_by_id[1].expected_cache_len == 1
    assert entries_by_id[1].n_tokens == 1
    assert entries_by_id[2].expected_cache_len == 1
    assert entries_by_id[2].n_tokens == 1


def test_token_generated_batched_one_bad_id_rejects_atomically() -> None:
    """Validates ALL request_ids before mutating ANY state -- a bad id
    anywhere in the batch must leave every request's cache_len
    UNCHANGED (partial application would itself be a silent-corruption
    surface: request A's cache_len advancing while the event as a
    whole raises)."""
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    with pytest.raises(ProtocolViolationError, match="not active"):
        core.handle(TokenGeneratedEvent(request_ids=(1, 99)))
    # request 1's cache_len must NOT have advanced despite being valid
    # -- the whole batch failed atomically.
    commands = core.handle(TokenGeneratedEvent(request_ids=(1,)))
    cmd = commands[0]
    assert isinstance(cmd, SendStepCommand)
    assert cmd.message.entries[0].expected_cache_len == 1  # not 2


def test_request_done_transitions_slot_to_draining_and_emits_evict() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    commands = core.handle(RequestDoneEvent(request_id=1))
    assert len(commands) == 1
    cmd = commands[0]
    assert isinstance(cmd, SendEvictCommand)
    assert cmd.message.request_id == 1
    assert cmd.message.cache_slot == 0
    assert core._slot_state[0] == SlotState.DRAINING


def test_request_aborted_behaves_identically_to_done() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    commands = core.handle(RequestAbortedEvent(request_id=1))
    assert isinstance(commands[0], SendEvictCommand)
    assert core._slot_state[0] == SlotState.DRAINING


def test_evict_unknown_request_raises() -> None:
    core = SchedulerCore(max_concurrency=2)
    with pytest.raises(ProtocolViolationError, match="not active"):
        core.handle(RequestDoneEvent(request_id=99))


def test_reusing_draining_slot_before_ack_raises() -> None:
    """THE core invariant this module exists to enforce (module
    docstring point 4): a slot mid-eviction (DRAINING, no ack yet)
    must NOT be reassignable to a new request."""
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    core.handle(RequestDoneEvent(request_id=1))
    with pytest.raises(ProtocolViolationError, match="not FREE"):
        core.handle(NewRequestEvent(request_id=2, cache_slot=0))


def test_evict_ack_frees_slot_for_reuse() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    core.handle(RequestDoneEvent(request_id=1))
    core.handle(EvictAckReceivedEvent(request_id=1, cache_slot=0))
    assert core._slot_state[0] == SlotState.FREE
    # Now reusable -- must NOT raise.
    commands = core.handle(NewRequestEvent(request_id=2, cache_slot=0))
    assert isinstance(commands[0], SendStepCommand)


def test_evict_ack_without_prior_evict_raises() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    with pytest.raises(ProtocolViolationError, match="not DRAINING"):
        core.handle(EvictAckReceivedEvent(request_id=1, cache_slot=0))


def test_duplicate_evict_ack_raises() -> None:
    core = SchedulerCore(max_concurrency=2)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    core.handle(RequestDoneEvent(request_id=1))
    core.handle(EvictAckReceivedEvent(request_id=1, cache_slot=0))
    with pytest.raises(ProtocolViolationError, match="not DRAINING"):
        core.handle(EvictAckReceivedEvent(request_id=1, cache_slot=0))


# ---------------------------------------------------------------------
# RankOneMirror directed unit tests -- validating rank 1's independent
# reactive checks against rank 0's claims.
# ---------------------------------------------------------------------


def test_mirror_accepts_well_formed_step_sequence() -> None:
    core = SchedulerCore(max_concurrency=2)
    mirror = RankOneMirror()

    for cmd in core.handle(NewRequestEvent(request_id=1, cache_slot=0)):
        assert isinstance(cmd, SendStepCommand)
        mirror.validate_step(cmd.message)
    for cmd in core.handle(TokenGeneratedEvent(request_ids=(1,))):
        assert isinstance(cmd, SendStepCommand)
        mirror.validate_step(cmd.message)
    for cmd in core.handle(TokenGeneratedEvent(request_ids=(1,))):
        assert isinstance(cmd, SendStepCommand)
        mirror.validate_step(cmd.message)
    # No exception -- 2 decode steps processed cleanly.


def test_mirror_rejects_out_of_order_step_id() -> None:
    mirror = RankOneMirror()

    msg = StepMessage(
        step_id=5,  # should be 1 first
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=0,
                n_tokens=1,
            ),
        ),
    )
    with pytest.raises(ProtocolViolationError, match="out-of-order"):
        mirror.validate_step(msg)


def test_mirror_rejects_duplicate_slot_in_same_step() -> None:
    mirror = RankOneMirror()

    msg = StepMessage(
        step_id=1,
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=0,
                n_tokens=1,
            ),
            BatchEntry(
                request_id=2,
                cache_slot=0,  # SAME slot as request 1 -- must reject
                phase=Phase.DECODE,
                expected_cache_len=0,
                n_tokens=1,
            ),
        ),
    )
    with pytest.raises(ProtocolViolationError, match="MORE THAN ONCE"):
        mirror.validate_step(msg)


def test_mirror_rejects_scheduling_on_draining_slot() -> None:
    """THE critical corruption-vector test: a StepMessage referencing a
    DRAINING slot must be rejected loudly, not silently processed."""
    core = SchedulerCore(max_concurrency=2)
    mirror = RankOneMirror()

    for cmd in core.handle(NewRequestEvent(request_id=1, cache_slot=0)):
        assert isinstance(cmd, SendStepCommand)
        mirror.validate_step(cmd.message)

    evict_commands = core.handle(RequestDoneEvent(request_id=1))
    evict_cmd = evict_commands[0]
    assert isinstance(evict_cmd, SendEvictCommand)
    mirror.validate_evict(evict_cmd.message)  # slot now DRAINING on rank 1

    # Simulate a buggy/stale scheduler trying to schedule the DRAINING
    # slot before the eviction is acked -- must raise on rank 1's side,
    # not corrupt cache state.

    bad_msg = StepMessage(
        step_id=evict_cmd.message.step_id + 1,
        entries=(
            BatchEntry(
                request_id=2,
                cache_slot=0,  # still DRAINING
                phase=Phase.DECODE,
                expected_cache_len=0,
                n_tokens=1,
            ),
        ),
    )
    with pytest.raises(ProtocolViolationError, match="DRAINING"):
        mirror.validate_step(bad_msg)


def test_mirror_rejects_cache_len_mismatch() -> None:
    """Directly targets the consult's flagged 'off-by-one KV length
    from mixed prefill/decode batches' silent-corruption case."""
    core = SchedulerCore(max_concurrency=2)
    mirror = RankOneMirror()

    for cmd in core.handle(NewRequestEvent(request_id=1, cache_slot=0)):
        assert isinstance(cmd, SendStepCommand)
        mirror.validate_step(cmd.message)

    # Rank 0 (buggy) claims cache_len=5 when the real advance should be 1.

    bad_msg = StepMessage(
        step_id=2,
        entries=(
            BatchEntry(
                request_id=1,
                cache_slot=0,
                phase=Phase.DECODE,
                expected_cache_len=5,  # WRONG -- mirror tracked 0
                n_tokens=1,
            ),
        ),
    )
    with pytest.raises(ProtocolViolationError, match="disagrees"):
        mirror.validate_step(bad_msg)


def test_mirror_evict_ack_roundtrip() -> None:
    core = SchedulerCore(max_concurrency=2)
    mirror = RankOneMirror()

    for cmd in core.handle(NewRequestEvent(request_id=1, cache_slot=0)):
        assert isinstance(cmd, SendStepCommand)
        mirror.validate_step(cmd.message)

    evict_commands = core.handle(RequestDoneEvent(request_id=1))
    evict_cmd = evict_commands[0]
    assert isinstance(evict_cmd, SendEvictCommand)
    mirror.validate_evict(evict_cmd.message)
    ack = mirror.build_evict_ack(evict_cmd.message)
    assert ack.request_id == 1
    assert ack.cache_slot == 0
    core.handle(
        EvictAckReceivedEvent(request_id=ack.request_id, cache_slot=ack.cache_slot)
    )
    assert core._slot_state[0] == SlotState.FREE


def test_mirror_evict_without_prior_active_raises() -> None:
    mirror = RankOneMirror()
    msg = EvictMessage(step_id=1, request_id=1, cache_slot=0)
    with pytest.raises(ProtocolViolationError, match="not ACTIVE"):
        mirror.validate_evict(msg)


# ---------------------------------------------------------------------
# Seeded random fuzzer -- runs SchedulerCore + RankOneMirror in
# lockstep across many pseudo-random event sequences, asserting they
# never silently disagree (either both accept, or the mirror's own
# independent check must be the one to catch any real violation).
# ---------------------------------------------------------------------


class _FuzzHarness:
    """Drives SchedulerCore + RankOneMirror in lockstep for one
    pseudo-random event sequence. Any ProtocolViolationError raised by
    EITHER side during a well-formed sequence (i.e. one this harness
    itself only generates legal NewRequestEvent/TokenGeneratedEvent/
    RequestDoneEvent sequences for) is a genuine bug and must fail the
    test loudly with the exact seed + event log for reproduction.
    """

    def __init__(self, max_concurrency: int) -> None:
        self.core = SchedulerCore(max_concurrency=max_concurrency)
        self.mirror = RankOneMirror()
        self.active_request_ids: set[int] = set()
        self.free_slots: list[int] = list(range(max_concurrency))
        self.draining: dict[int, tuple[int, int]] = {}  # slot -> (req_id, step_id)
        self.next_request_id = 0
        self.event_log: list[str] = []

    def _apply(self, commands: list[Command]) -> None:
        for cmd in commands:
            if isinstance(cmd, SendStepCommand):
                self.mirror.validate_step(cmd.message)
            else:
                self.mirror.validate_evict(cmd.message)
                ack = self.mirror.build_evict_ack(cmd.message)
                self.draining[cmd.message.cache_slot] = (
                    cmd.message.request_id,
                    cmd.message.step_id,
                )
                # Immediately ack in this harness -- a SEPARATE targeted
                # generator below tests the "new request arrives BEFORE
                # the ack" race explicitly, rather than relying on
                # uniform-random timing to stumble into it (per the
                # consult's explicit warning that uniform sampling
                # rarely hits this race).
                self.core.handle(
                    EvictAckReceivedEvent(
                        request_id=ack.request_id, cache_slot=ack.cache_slot
                    )
                )
                del self.draining[cmd.message.cache_slot]
                self.free_slots.append(cmd.message.cache_slot)

    def new_request(self) -> None:
        if not self.free_slots:
            return
        slot = self.free_slots.pop()
        rid = self.next_request_id
        self.next_request_id += 1
        self.event_log.append(f"new_request(request_id={rid}, cache_slot={slot})")
        self._apply(self.core.handle(NewRequestEvent(request_id=rid, cache_slot=slot)))
        self.active_request_ids.add(rid)

    def token_generated(self, rng: random.Random) -> None:
        if not self.active_request_ids:
            return
        rid = rng.choice(sorted(self.active_request_ids))
        self.event_log.append(f"token_generated(request_id={rid})")
        self._apply(self.core.handle(TokenGeneratedEvent(request_ids=(rid,))))

    def evict_request(self, rng: random.Random, aborted: bool) -> None:
        if not self.active_request_ids:
            return
        rid = rng.choice(sorted(self.active_request_ids))
        self.active_request_ids.discard(rid)
        kind = "aborted" if aborted else "done"
        self.event_log.append(f"evict_request(request_id={rid}, kind={kind})")
        event = (
            RequestAbortedEvent(request_id=rid)
            if aborted
            else RequestDoneEvent(request_id=rid)
        )
        self._apply(self.core.handle(event))

    def dump_log(self) -> str:
        return "\n".join(self.event_log)


def test_fuzz_random_event_sequences_never_desync() -> None:
    """Uniform-random event sequences across many seeds -- baseline
    coverage. Deliberately does NOT rely on this alone (see the
    targeted tests below) since the consult review flagged that
    uniform sampling rarely hits the specific abort/reuse race this
    module exists to prevent."""
    for seed in range(2000):
        rng = random.Random(seed)
        harness = _FuzzHarness(max_concurrency=2)
        try:
            for _ in range(50):
                action = rng.choice(
                    ["new_request", "token_generated", "evict_done", "evict_aborted"]
                )
                if action == "new_request":
                    harness.new_request()
                elif action == "token_generated":
                    harness.token_generated(rng)
                elif action == "evict_done":
                    harness.evict_request(rng, aborted=False)
                else:
                    harness.evict_request(rng, aborted=True)
        except ProtocolViolationError as e:  # noqa: BLE001
            pytest.fail(
                f"seed={seed} produced a genuine desync/violation in a "
                f"well-formed random sequence -- this is a REAL bug, not "
                f"an expected rejection (this harness only generates "
                f"legal action sequences with proper ack-before-reuse):\n"
                f"{e}\n\nEvent log:\n{harness.dump_log()}"
            )


def test_fuzz_targeted_abort_while_slot_would_be_reused_immediately() -> None:
    """TARGETED sequence (per the consult's explicit recommendation,
    since uniform random rarely hits this): abort a request, then
    IMMEDIATELY try to reuse its slot for a new request BEFORE the
    evict ack is processed. Must raise ProtocolViolationError -- this
    is the exact race the DRAINING state exists to prevent, and this
    test proves it's actually enforced, not merely documented."""
    core = SchedulerCore(max_concurrency=1)
    core.handle(NewRequestEvent(request_id=1, cache_slot=0))
    commands = core.handle(RequestAbortedEvent(request_id=1))
    assert isinstance(commands[0], SendEvictCommand)
    # Do NOT process the evict ack -- attempt immediate slot reuse.
    with pytest.raises(ProtocolViolationError, match="not FREE"):
        core.handle(NewRequestEvent(request_id=2, cache_slot=0))


def test_fuzz_targeted_abort_then_new_request_after_proper_ack_succeeds() -> None:
    """The legal counterpart to the above: abort, ack, THEN reuse --
    must succeed cleanly. Proves the invariant is exactly "no reuse
    before ack", not an overly broad "never reuse a slot"."""
    core = SchedulerCore(max_concurrency=1)
    mirror = RankOneMirror()

    for cmd in core.handle(NewRequestEvent(request_id=1, cache_slot=0)):
        assert isinstance(cmd, SendStepCommand)
        mirror.validate_step(cmd.message)

    evict_commands = core.handle(RequestAbortedEvent(request_id=1))
    evict_cmd = evict_commands[0]
    assert isinstance(evict_cmd, SendEvictCommand)
    mirror.validate_evict(evict_cmd.message)
    ack = mirror.build_evict_ack(evict_cmd.message)
    core.handle(
        EvictAckReceivedEvent(request_id=ack.request_id, cache_slot=ack.cache_slot)
    )

    # Now legal -- must NOT raise.
    commands = core.handle(NewRequestEvent(request_id=2, cache_slot=0))
    assert isinstance(commands[0], SendStepCommand)


def test_fuzz_targeted_many_rapid_abort_reuse_cycles() -> None:
    """Stress the abort -> ack -> reuse cycle repeatedly on the SAME
    slot across many iterations with a seeded RNG interleaving other
    requests in between -- the consult specifically flagged
    'abort-while-in-flight' and 'abort-then-immediate-new-request' as
    under-sampled by uniform generation, so this generates ONLY those
    two shapes, many times, across many seeds."""
    for seed in range(1000):
        rng = random.Random(seed)
        core = SchedulerCore(max_concurrency=2)
        mirror = RankOneMirror()
        next_rid = 0
        active: dict[int, int] = {}  # slot -> request_id
        log: list[str] = []
        try:
            for _ in range(30):
                free_slots = [s for s in range(2) if s not in active]
                if free_slots and (not active or rng.random() < 0.5):
                    slot = rng.choice(free_slots)
                    rid = next_rid
                    next_rid += 1
                    log.append(f"new({rid},{slot})")
                    for cmd in core.handle(
                        NewRequestEvent(request_id=rid, cache_slot=slot)
                    ):
                        assert isinstance(cmd, SendStepCommand)
                        mirror.validate_step(cmd.message)
                    active[slot] = rid
                elif active:
                    slot = rng.choice(sorted(active.keys()))
                    rid = active.pop(slot)
                    log.append(f"abort({rid},{slot})")
                    evict_commands = core.handle(RequestAbortedEvent(request_id=rid))
                    evict_cmd = evict_commands[0]
                    assert isinstance(evict_cmd, SendEvictCommand)
                    mirror.validate_evict(evict_cmd.message)
                    ack = mirror.build_evict_ack(evict_cmd.message)
                    core.handle(
                        EvictAckReceivedEvent(
                            request_id=ack.request_id, cache_slot=ack.cache_slot
                        )
                    )
        except ProtocolViolationError as e:  # noqa: BLE001
            pytest.fail(
                f"seed={seed} raised in a well-formed abort/reuse-cycle "
                f"sequence -- genuine bug:\n{e}\n\nLog:\n" + "\n".join(log)
            )

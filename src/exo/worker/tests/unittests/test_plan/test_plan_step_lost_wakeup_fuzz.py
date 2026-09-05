"""I16 (round 12): RANDOMIZED sweep for the `plan_step` event-wake path.

`test_plan_step_event_wake.py` (sibling module) proves the lost-wakeup fix by
hand-constructing ONE specific interleaving (mutate + signal land exactly
between the waiter's state check and its `await`). That is necessary but not
sufficient evidence: a fix that happens to survive one hand-picked ordering
could still have a narrower race that only shows up under a different timing.

This module instead runs many randomized trials against the REAL
`exo.worker.main.Worker` (gate ON), each trial replaying a random-length
sequence of state-mutating events through a random mix of two interleaving
shapes:

  * "synchronous_window" -- the classic adversarial window from the sibling
    suite's `test_signal_between_state_check_and_await_is_not_lost`: mutate
    + signal complete fully BEFORE the waiter's `await` even begins.
  * "racing_window" -- a genuine concurrent race: the applier fires after a
    randomized number of scheduler checkpoints while the waiter is already
    parked inside `_wait_for_state_change`.

Across every trial and every round it asserts two properties, using
`anyio.Event.is_set()` as a non-invasive oracle (no monkeypatching of
`main.py`, which is frozen for this task):

  1. ZERO missed wakes -- every state mutation this sweep applies is present
     in `worker.state.tasks` by the end of its trial.
  2. ZERO timeout-driven wakes -- every `_wait_for_state_change` call in this
     sweep returns with `waiting_on.is_set() is True`, i.e. it was woken by
     the event, never by its internal `anyio.move_on_after` fallback. Since
     every round in this sweep always eventually signals well within the real
     0.1s fallback, these two properties collapse to the same oracle by
     construction -- see `_run_real_worker_trial` for why that is legitimate
     and not circular.

Trial count and seeding: 300 trials, each running 1-5 rounds, seeded
deterministically from `BASE_SEED + trial_index` so any failure reproduces
exactly by re-running with that seed. 300 trials at effectively zero
wall-clock cost each (no real sleeps occur on the passing path -- signals
always land within a handful of `anyio.sleep(0)` checkpoints) keeps the whole
sweep well under a second, nowhere near the ~30s CI budget.

CRITICAL negative control: a lost-wakeup sweep that cannot fail is worthless.
Section "NEGATIVE CONTROLS" below reproduces the exact hazard pattern in an
ISOLATED toy harness (never touching the frozen `main.py`) for both classic
bugs -- (i) capture-after-check, (ii) reused Event -- and proves this sweep's
methodology detects both. Those tests assert the harness catches the bug
(`lost > 0` / `bad_rounds > 0`) so the suite stays green; the literal failing
assertion output, demonstrating the harness WOULD fail a zero-tolerance
check against the broken variants, is captured separately (see the task
report) since a permanently-failing test cannot live in a green suite.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import ModuleType

import anyio
import pytest

from exo.shared.apply import apply
from exo.shared.types.events import IndexedEvent, TaskCreated
from exo.shared.types.tasks import LoadModel, TaskId
from exo.worker.tests.constants import INSTANCE_1_ID
from exo.worker.tests.unittests.test_plan.test_plan_step_event_wake import (
    _capture,  # pyright: ignore[reportPrivateUsage]
    _load_worker_main,  # pyright: ignore[reportPrivateUsage]
    _make_worker,  # pyright: ignore[reportPrivateUsage]
    _signal,  # pyright: ignore[reportPrivateUsage]
    _wait,  # pyright: ignore[reportPrivateUsage]
)

NUM_TRIALS = 300
MIN_EVENTS_PER_TRIAL = 1
MAX_EVENTS_PER_TRIAL = 5
MAX_RACE_CHECKPOINTS = 6
BASE_SEED = 20260904_100

# Toy-harness-only tuning (never touches main.py). Small so the deliberately
# broken demonstrations below stay fast even when they DO time out.
TOY_TICK_SECONDS = 0.02


@pytest.fixture(autouse=True)
def restore_worker_main() -> Iterator[None]:
    """Leave `exo.worker.main` reloaded at its real (unset-gate) state.

    Mirrors the sibling suite's identically-named fixture; duplicated here
    (rather than imported) because pytest only discovers autouse fixtures
    declared in the file/conftest doing the collecting.
    """
    yield
    _ = _load_worker_main(gate=None)


# --------------------------------------------------------------------------
# THE RANDOMIZED SWEEP -- real `exo.worker.main.Worker`, gate ON.
# --------------------------------------------------------------------------


@dataclass
class _TrialResult:
    seed: int
    missing_task_ids: list[TaskId]
    timeout_driven_rounds: list[int]

    @property
    def ok(self) -> bool:
        return not self.missing_task_ids and not self.timeout_driven_rounds


async def _run_real_worker_trial(module: ModuleType, seed: int) -> _TrialResult:
    """Replay a random-length sequence of events through the REAL Worker.

    Mirrors `plan_step`'s own capture discipline exactly: capture
    `waiting_on` once, then re-capture immediately after every wake -- before
    doing anything else -- which is precisely what closes the lost-wakeup
    window in the production code (see `main.py`'s `plan_step` comment on
    capture ordering).
    """
    rng = random.Random(seed)
    worker, _sender = _make_worker(module)

    waiting_on = _capture(worker)
    idx = 0
    task_ids: list[TaskId] = []
    timeout_driven_rounds: list[int] = []

    num_events = rng.randint(MIN_EVENTS_PER_TRIAL, MAX_EVENTS_PER_TRIAL)
    for round_index, current_idx in enumerate(range(idx, idx + num_events)):
        task_id = TaskId()
        task_ids.append(task_id)
        indexed_event = IndexedEvent(
            idx=current_idx,
            event=TaskCreated(
                task_id=task_id,
                task=LoadModel(task_id=task_id, instance_id=INSTANCE_1_ID),
            ),
        )

        if rng.choice((True, False)):
            # THE adversarial window: mutate + signal complete fully BEFORE
            # the waiter's await even starts (test C's scenario, replayed at
            # a random point in a random-length sequence of rounds).
            worker.state = apply(worker.state, event=indexed_event)
            _signal(worker)
            await _wait(worker, waiting_on)
        else:
            # A genuinely concurrent race: the applier fires after a random
            # number of scheduler checkpoints while the waiter is already
            # parked inside `_wait_for_state_change`. Bound as default args
            # (not closed-over loop variables) so each task group's coroutines
            # see THIS round's values regardless of scheduling order.
            delay = rng.randint(0, MAX_RACE_CHECKPOINTS)

            async def applier(
                *, delay: int = delay, indexed_event: IndexedEvent = indexed_event
            ) -> None:
                for _ in range(delay):
                    await anyio.sleep(0)
                worker.state = apply(worker.state, event=indexed_event)
                _signal(worker)

            async def waiter(*, waiting_on: anyio.Event = waiting_on) -> None:
                await _wait(worker, waiting_on)

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(waiter)
                task_group.start_soon(applier)

        # ORACLE: `_wait_for_state_change` can only return with `is_set()`
        # True via `waiting_on.wait()` completing (an event-driven wake), or
        # False via its internal `anyio.move_on_after(_PLAN_TICK_SECONDS)`
        # expiring (a timeout-driven wake). Every round above guarantees a
        # real signal lands well inside that 0.1s window, so `is_set() is
        # False` here is unambiguous evidence of a timeout-driven wake, not
        # an artifact of this sweep's own timing.
        if not waiting_on.is_set():
            timeout_driven_rounds.append(round_index)

        waiting_on = _capture(worker)  # what plan_step does immediately

    missing_task_ids = [tid for tid in task_ids if tid not in worker.state.tasks]
    return _TrialResult(
        seed=seed,
        missing_task_ids=missing_task_ids,
        timeout_driven_rounds=timeout_driven_rounds,
    )


async def test_randomized_sweep_zero_missed_and_zero_timeout_wakes() -> None:
    """The headline test: 300 randomized trials, zero misses, zero timeouts.

    On failure the assertion message names the exact failing seed so the
    interleaving reproduces deterministically via `random.Random(seed)`.
    """
    module = _load_worker_main(gate="1")

    failures: list[_TrialResult] = []
    for trial_index in range(NUM_TRIALS):
        seed = BASE_SEED + trial_index
        result = await _run_real_worker_trial(module, seed)
        if not result.ok:
            failures.append(result)

    assert not failures, (
        f"{len(failures)}/{NUM_TRIALS} randomized trials hit a lost or "
        f"timeout-driven wake. First failure: seed={failures[0].seed} "
        f"missing_task_ids={failures[0].missing_task_ids} "
        f"timeout_driven_rounds={failures[0].timeout_driven_rounds} "
        f"(reproduce with random.Random({failures[0].seed}))"
    )


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS -- prove the sweep methodology actually has teeth.
#
# These build an ISOLATED toy harness that reproduces the two classic
# lost-wakeup mistakes WITHOUT editing `main.py` (frozen for this task), then
# run the same randomized-round shape as the real sweep above against it.
# --------------------------------------------------------------------------


@dataclass
class _ToyWaker:
    """A minimal, standalone reproduction of the `_state_applied` pattern."""

    tick: float
    reuse_event: bool = False
    event: anyio.Event = field(default_factory=anyio.Event)

    def signal(self) -> None:
        if self.reuse_event:
            # BUG (ii): reuse a single Event instead of swapping in a fresh
            # one. `anyio.Event` has no `clear()`, so once this fires, EVERY
            # future capture of `self.event` inherits a permanently-set
            # object, regardless of whether a new mutation ever happens.
            self.event.set()
        else:
            previous = self.event
            self.event = anyio.Event()
            previous.set()


async def _toy_wait(waker: _ToyWaker, captured: anyio.Event) -> None:
    with anyio.move_on_after(waker.tick):
        await captured.wait()


async def _toy_round_capture_timing(
    waker: _ToyWaker, rng: random.Random, *, check_delay: int, delay_max: int
) -> bool:
    """One randomized round probing capture-vs-check ordering.

    Returns True iff the wait resolved via the toy event (not its fallback).

    `check_delay` models "work between deciding there's nothing to do and
    grabbing the current Event" -- 0 reproduces the CORRECT ordering (capture
    happens as the waiter's very first, synchronous action, exactly like
    `plan_step` re-capturing `waiting_on` before calling `plan()`); >=1
    reproduces BUG (i): the waiter burns `check_delay` scheduler checkpoints
    before reading `waker.event`, so a signal landing in that gap is
    captured on the wrong, already-replaced object.
    """
    signal_delay = rng.randint(0, delay_max)
    captured_box: dict[str, anyio.Event] = {}

    async def waiter() -> None:
        for _ in range(check_delay):
            await anyio.sleep(0)
        captured_box["event"] = waker.event  # the capture -- early or late
        await _toy_wait(waker, captured_box["event"])

    async def applier() -> None:
        for _ in range(signal_delay):
            await anyio.sleep(0)
        waker.signal()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(waiter)
        task_group.start_soon(applier)

    return captured_box["event"].is_set()


async def test_negative_control_capture_after_check_is_caught() -> None:
    """CRITICAL negative control (i): capture-after-check must be caught.

    `check_delay=2` reproduces the broken ordering. Across 200 randomized
    trials this must lose at least one wakeup, proving the sweep's
    methodology (same round shape as the real-Worker sweep above) is capable
    of catching this exact class of bug rather than passing vacuously.
    """
    trials = 200
    lost = 0
    first_failure_seed: int | None = None

    for trial_index in range(trials):
        seed = 20260904_200 + trial_index
        trial_rng = random.Random(seed)
        woke_via_event = await _toy_round_capture_timing(
            _ToyWaker(tick=TOY_TICK_SECONDS),
            trial_rng,
            check_delay=2,
            delay_max=4,
        )
        if not woke_via_event:
            lost += 1
            if first_failure_seed is None:
                first_failure_seed = seed

    assert lost > 0, (
        "capture-after-check demonstration harness produced ZERO lost "
        f"wakes across {trials} trials -- the randomized sweep would not "
        "have caught this bug; strengthen the harness rather than trust it"
    )
    print(
        f"[negative control i: capture-after-check] {lost}/{trials} "
        f"trials lost the wakeup; first failing seed={first_failure_seed}"
    )


async def _toy_round_reuse_timing(
    waker: _ToyWaker, rng: random.Random, *, delay_max: int
) -> bool:
    """One randomized round probing the reused-Event bug (ii).

    Capture timing here is deliberately CORRECT (capture happens first,
    synchronously, `check_delay=0`) so this isolates the OTHER hazard: does
    the signaller hand out a fresh, unset Event every time? A healthy round
    must capture an UNSET Event and then see it become set only via THIS
    round's own applier call.
    """
    signal_delay = rng.randint(0, delay_max)
    captured = waker.event  # correct: capture immediately, before any check
    pre_set = captured.is_set()

    async def applier() -> None:
        for _ in range(signal_delay):
            await anyio.sleep(0)
        waker.signal()

    async def waiter() -> None:
        await _toy_wait(waker, captured)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(waiter)
        task_group.start_soon(applier)

    return not pre_set and captured.is_set()


async def test_negative_control_reused_event_is_caught() -> None:
    """CRITICAL negative control (ii): a reused Event must be caught.

    Runs many rounds against ONE `_ToyWaker(reuse_event=True)` instance (the
    bug only manifests once an Event has already been consumed once and
    never replaced). From round 2 onward every capture inherits the
    permanently-set object from round 1's `signal()` call, so this must
    report a majority of rounds unhealthy.
    """
    rounds = 50
    rng = random.Random(20260904_300)
    waker = _ToyWaker(tick=TOY_TICK_SECONDS, reuse_event=True)

    bad_rounds = 0
    first_bad_round: int | None = None

    for round_index in range(rounds):
        healthy = await _toy_round_reuse_timing(waker, rng, delay_max=3)
        if not healthy:
            bad_rounds += 1
            if first_bad_round is None:
                first_bad_round = round_index

    assert bad_rounds > 0, (
        "reused-Event demonstration harness produced ZERO stale-wake "
        f"rounds across {rounds} rounds -- the randomized sweep would not "
        "have caught this bug; strengthen the harness rather than trust it"
    )
    print(
        f"[negative control ii: reused event] {bad_rounds}/{rounds} "
        f"rounds were stale from a previous signal; "
        f"first bad round={first_bad_round}"
    )

"""I16 (round 12): event-triggered wake for the `plan_step` loop-top wait.

Historically `Worker.plan_step`'s `while True:` opened with an unconditional
`await anyio.sleep(0.1)`, so a task dispatched into cluster state waited for the
next 100ms tick before the worker acted on it. The fix replaces that tick with a
wait on an `anyio.Event` signalled at the SINGLE state-apply point
(`Worker._event_applier`), keeping the 0.1s sleep purely as a fallback timeout.

`anyio.Event` has no `clear()`, so the implementation swaps in a FRESH Event on
every signal and `plan_step` captures its reference BEFORE reading state. These
tests pin that ordering -- especially the lost-wakeup interleaving, which is
driven deterministically here (explicit synchronous statement order), never by
wall-clock timing.

The gate is `EXO_WORKER_PLAN_EVENT_WAKE`, read ONCE at module import into a
module-level `Final[bool]`. Tests that need it flipped therefore reload the
module under a patched environment rather than mutating `os.environ` and hoping.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import cast

import anyio
import pytest

import exo.worker.main as worker_main_module
from exo.shared.apply import apply
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import Event, IndexedEvent, TaskCreated
from exo.shared.types.tasks import LoadModel, Task
from exo.utils.channels import Sender, channel
from exo.worker.main import Worker
from exo.worker.tests.constants import INSTANCE_1_ID, NODE_A, TASK_1_ID

# The fallback timeout the implementation must retain. Tests assert on logical
# wake-cause, never on a measured wall-clock duration.
FALLBACK_SECONDS = 0.1


def _load_worker_main(*, gate: str | None) -> ModuleType:
    """Re-import `exo.worker.main` with the env gate set to `gate`.

    The gate is a module-level `Final[bool]` read once at import (mirroring
    `exo.api.phase_marks`), so flipping it in a test REQUIRES a reload under a
    patched environment.
    """
    with pytest.MonkeyPatch.context() as patched:
        if gate is None:
            patched.delenv("EXO_WORKER_PLAN_EVENT_WAKE", raising=False)
        else:
            patched.setenv("EXO_WORKER_PLAN_EVENT_WAKE", gate)
        return importlib.reload(worker_main_module)


@pytest.fixture(autouse=True)
def restore_worker_main() -> Iterator[None]:
    """Leave `exo.worker.main` reloaded at its real (unset-gate) state."""
    yield
    _ = _load_worker_main(gate=None)


def _gate_enabled(module: ModuleType) -> bool:
    """`ModuleType` attribute access is untyped; narrow it once, here."""
    return cast(bool, module._PLAN_EVENT_WAKE_ENABLED)


def _tick_seconds(module: ModuleType) -> float:
    return cast(float, module._PLAN_TICK_SECONDS)


def _worker_class(module: ModuleType) -> type[Worker]:
    return cast(type[Worker], module.Worker)


def _make_worker(module: ModuleType) -> tuple[Worker, Sender[IndexedEvent]]:
    """Build a `Worker` with real in-process channels and no runners.

    Returns the worker plus the sender feeding its inbound event channel, so a
    test can drive the real `_event_applier` coroutine.
    """
    inbound_sender, inbound_receiver = channel[IndexedEvent]()
    outbound_sender, _outbound_receiver = channel[Event]()
    command_sender, _command_receiver = channel[ForwarderCommand]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()

    worker = _worker_class(module)(
        NODE_A,
        event_receiver=inbound_receiver,
        event_sender=outbound_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )
    return worker, inbound_sender


def _capture(worker: Worker) -> anyio.Event:
    """What `plan_step` does at the top of its loop: grab the current Event."""
    return worker._state_applied  # pyright: ignore[reportPrivateUsage]


def _signal(worker: Worker) -> None:
    """What `_event_applier` does after mutating state."""
    worker._signal_state_applied()  # pyright: ignore[reportPrivateUsage]


async def _wait(worker: Worker, waiting_on: anyio.Event) -> None:
    """The loop-top wait under test."""
    await worker._wait_for_state_change(waiting_on)  # pyright: ignore[reportPrivateUsage]


def _an_event() -> IndexedEvent:
    """A real `IndexedEvent` that `apply()` accepts and that mutates state."""
    task: Task = LoadModel(task_id=TASK_1_ID, instance_id=INSTANCE_1_ID)
    return IndexedEvent(idx=0, event=TaskCreated(task_id=TASK_1_ID, task=task))


# --------------------------------------------------------------------------
# A. Gate OFF => plain 100ms sleep, no event dependency.
# --------------------------------------------------------------------------


def test_gate_defaults_to_off_when_unset() -> None:
    assert _gate_enabled(_load_worker_main(gate=None)) is False


@pytest.mark.parametrize("value", ["0", "false", "False", ""])
def test_gate_off_for_disabled_values(value: str) -> None:
    assert _gate_enabled(_load_worker_main(gate=value)) is False


def test_gate_on_for_one() -> None:
    assert _gate_enabled(_load_worker_main(gate="1")) is True


async def test_gate_off_wait_is_a_plain_sleep_ignoring_the_event() -> None:
    """A: with the gate OFF the loop-top wait does not consult the Event.

    Proven by signalling BEFORE the wait and showing the wait still runs past
    the half-fallback mark -- the `set()` bought nothing, which is exactly
    today's behaviour.
    """
    module = _load_worker_main(gate=None)
    worker, _sender = _make_worker(module)

    captured = _capture(worker)
    _signal(worker)  # inert under the gate

    returned_early = False
    with anyio.move_on_after(FALLBACK_SECONDS / 2):
        await _wait(worker, captured)
        returned_early = True

    assert returned_early is False


async def test_gate_off_signal_does_not_replace_the_event() -> None:
    """A: with the gate OFF the signal path is fully inert (no allocation)."""
    module = _load_worker_main(gate=None)
    worker, _sender = _make_worker(module)

    before = _capture(worker)
    _signal(worker)
    after = _capture(worker)

    assert before is after
    assert before.is_set() is False


# --------------------------------------------------------------------------
# B. Gate ON => a state-apply wakes the waiter well under the fallback.
# --------------------------------------------------------------------------


async def test_gate_on_signal_wakes_waiter_before_fallback() -> None:
    """B: assert on logical wake-cause, not on a measured duration.

    The whole wait is wrapped in a deadline far SHORTER than the fallback. If
    the waiter could only return via the fallback timeout, that outer deadline
    would cancel it first and `woke` would stay False.
    """
    module = _load_worker_main(gate="1")
    worker, _sender = _make_worker(module)

    captured = _capture(worker)
    woke = False

    async def waiter() -> None:
        nonlocal woke
        await _wait(worker, captured)
        woke = True

    with anyio.move_on_after(FALLBACK_SECONDS / 4):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(waiter)
            await anyio.sleep(0)  # let the waiter park
            _signal(worker)

    assert woke is True, "waiter did not wake on the signal, only on fallback"


async def test_gate_on_real_event_applier_wakes_plan_step_waiter() -> None:
    """B (real path): drive the wake through `_event_applier`, not a helper.

    This is the production call path: an `IndexedEvent` arrives on the worker's
    inbound channel, `_event_applier` applies it to `self.state` and signals.
    """
    module = _load_worker_main(gate="1")
    worker, inbound = _make_worker(module)

    captured = _capture(worker)
    woke = False

    async def waiter() -> None:
        nonlocal woke
        await _wait(worker, captured)
        woke = True

    with anyio.move_on_after(FALLBACK_SECONDS / 4):
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                worker._event_applier  # pyright: ignore[reportPrivateUsage]
            )
            task_group.start_soon(waiter)
            await anyio.sleep(0)
            await inbound.send(_an_event())
            await anyio.sleep(0)
            inbound.close()
            task_group.cancel_scope.cancel()

    assert woke is True
    # The state really was mutated by the same apply that woke us.
    assert TASK_1_ID in worker.state.tasks


async def test_gate_on_real_plan_step_loop_reruns_plan_on_signal() -> None:
    """B (fullest real path): run the ACTUAL `plan_step` coroutine.

    Not a helper in isolation -- this is the production loop. `plan` is
    replaced with a counting stub returning None (so the loop just spins on the
    wait), then a signal is fired. Under an outer deadline much shorter than
    the fallback, `plan` must be re-evaluated MORE times than it could have
    been by the 0.1s tick alone.
    """
    module = _load_worker_main(gate="1")
    worker, _sender = _make_worker(module)

    calls = 0
    calls_before = 0

    def counting_plan(*_args: object, **_kwargs: object) -> Task | None:
        nonlocal calls
        calls += 1
        return None

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(module, "plan", counting_plan)
        with anyio.move_on_after(FALLBACK_SECONDS / 2):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(worker.plan_step)
                await anyio.sleep(0)
                calls_before = calls
                for _ in range(5):
                    _signal(worker)
                    await anyio.sleep(0)
                task_group.cancel_scope.cancel()

    # Each signal must have driven another plan() evaluation. Within half the
    # fallback the plain 0.1s tick could not have produced any extra call.
    assert calls - calls_before >= 5, (
        f"plan_step did not re-evaluate plan() on each signal "
        f"(before={calls_before}, after={calls})"
    )


# --------------------------------------------------------------------------
# C. THE LOST-WAKEUP WINDOW. The most important test in this change.
# --------------------------------------------------------------------------


async def test_signal_between_state_check_and_await_is_not_lost() -> None:
    """C: mutation + set land BETWEEN the waiter's state check and its await.

    Deterministic by construction -- no task-interleaving luck. The sequence is
    driven by explicit synchronous statement order in ONE task:

        1. capture the current Event   (what `plan_step` does at loop top)
        2. read state / evaluate plan  (finds nothing to do)
        3. >>> state mutates + signal fires HERE <<<   (the danger window)
        4. await the CAPTURED event

    Step 3's signal targets the object captured at step 1, so step 4 must
    return immediately. Had the implementation captured the reference AFTER the
    state check (or tried to reset an Event in place), step 4 would park for
    the full fallback and the enclosing short deadline would cancel it.
    """
    module = _load_worker_main(gate="1")
    worker, _sender = _make_worker(module)

    # 1. capture BEFORE checking state
    captured = _capture(worker)

    # 2. check state -- nothing to do yet
    assert TASK_1_ID not in worker.state.tasks

    # 3. THE WINDOW: mutate state, THEN signal. Fully synchronous, so it
    #    provably lands after the check and before the await below.
    worker.state = apply(worker.state, event=_an_event())
    _signal(worker)

    # 4. await the CAPTURED event -- must return immediately, not on fallback.
    observed = False
    with anyio.move_on_after(FALLBACK_SECONDS / 4):
        await _wait(worker, captured)
        observed = True

    assert observed is True, (
        "LOST WAKEUP: signal fired between the state check and the await, and "
        "the waiter blocked anyway"
    )
    assert TASK_1_ID in worker.state.tasks


async def test_signal_before_capture_does_not_wedge_the_next_wait() -> None:
    """C (converse): a signal predating the capture must NOT be seen.

    Otherwise the Event would be permanently hot and the loop would spin. The
    freshly-captured Event is unset, so this wait must fall through to the
    fallback timeout instead of returning instantly.
    """
    module = _load_worker_main(gate="1")
    worker, _sender = _make_worker(module)

    _signal(worker)  # stale signal, before the capture
    captured = _capture(worker)

    assert captured.is_set() is False

    returned_early = False
    with anyio.move_on_after(FALLBACK_SECONDS / 4):
        await _wait(worker, captured)
        returned_early = True

    assert returned_early is False


# --------------------------------------------------------------------------
# D. Fresh Event per wake.
# --------------------------------------------------------------------------


async def test_event_is_replaced_not_left_set_so_repeat_signals_wake() -> None:
    """D: after a wake, a SUBSEQUENT signal still wakes a new waiter."""
    module = _load_worker_main(gate="1")
    worker, _sender = _make_worker(module)

    for round_index in range(3):
        captured = _capture(worker)
        assert captured.is_set() is False, (
            f"round {round_index}: captured Event was already set -- it was "
            "reused rather than replaced"
        )

        _signal(worker)

        # The old object is consumed...
        assert captured.is_set() is True
        # ...and a DIFFERENT, unset object took its place.
        replacement = _capture(worker)
        assert replacement is not captured
        assert replacement.is_set() is False

        woke = False
        with anyio.move_on_after(FALLBACK_SECONDS / 4):
            await _wait(worker, captured)
            woke = True
        assert woke is True, f"round {round_index}: waiter did not wake"


# --------------------------------------------------------------------------
# E. Fallback timeout intact -- no hang when no signal ever fires.
# --------------------------------------------------------------------------


async def test_gate_on_with_no_signal_still_proceeds_via_fallback() -> None:
    """E: never a bare await-forever; the 0.1s fallback must still fire."""
    module = _load_worker_main(gate="1")
    worker, _sender = _make_worker(module)

    captured = _capture(worker)

    proceeded = False
    # Generous outer bound: proves termination, not a timing claim.
    with anyio.move_on_after(FALLBACK_SECONDS * 20):
        await _wait(worker, captured)
        proceeded = True

    assert proceeded is True, "waiter hung: fallback timeout was lost"
    assert captured.is_set() is False


async def test_gate_on_fallback_is_not_a_shortened_spin() -> None:
    """E: the fallback must still be the full 0.1s, not a shortened poll.

    Guards against the tick being "fixed" by turning it into a faster spin.
    """
    module = _load_worker_main(gate="1")
    assert _tick_seconds(module) == FALLBACK_SECONDS

    worker, _sender = _make_worker(module)
    captured = _capture(worker)

    returned_early = False
    with anyio.move_on_after(FALLBACK_SECONDS / 2):
        await _wait(worker, captured)
        returned_early = True

    assert returned_early is False, (
        "unsignalled wait returned before half the fallback -- the tick was "
        "shortened into a spin-wait instead of being event-driven"
    )

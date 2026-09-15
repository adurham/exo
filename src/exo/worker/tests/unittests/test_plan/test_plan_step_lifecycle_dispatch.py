"""I16 regression (2026-09-15): a lifecycle task dispatch must not be duplicated.

THE PRODUCTION OUTAGE THIS PINS
-------------------------------
With `EXO_WORKER_PLAN_EVENT_WAKE=1`, `plan_step` dispatches a task and then
re-enters `plan()` far sooner than the historical 100ms tick allowed. Any
dispatch that is NOT awaited to completion therefore has a window in which
`plan()` runs again while the runner has not yet published the status
transition that suppresses the task -- and `plan()` re-emits it.

That is exactly what happened on 2026-09-15. `ConnectToGroup` is constructed
fresh by `plan()` on every call with a NEW `task_id`, so no task_id-keyed
guard (neither the supervisor's `pending`/`completed` sets nor the runner's
`seen` set) can suppress the repeat. Two `ConnectToGroup` tasks for the SAME
instance were dispatched ~5ms apart:

    Starting task ConnectToGroup(task_id='b773da1c-...', instance_id='53a472e2-...')
    Starting task ConnectToGroup(task_id='ba6468b1-...', instance_id='53a472e2-...')

The runner processed the first (`RunnerIdle` -> connect -> `RunnerConnected`)
and then hit the second with `RunnerConnected` already current, fell through
to `handle_first_task`'s `case _:` arm and died:

    ValueError: Received ConnectToGroup outside of state machine
                in self.current_status=RunnerConnected()

SIGKILL-free process death -> RunnerFailed -> instance teardown and
re-placement -> the observed crash-loop. The identical arm
(`StartWarmup`) is guarded by `isinstance(runner.status, RunnerLoaded)`, so it
had the same defect.

THE FIX
-------
`Worker.plan_step`'s catch-all arm no longer routes lifecycle tasks through
`self._tg.start_soon(...)`. Generation tasks (stable task_id, dedupe-safe)
keep the non-blocking dispatch that the c>=2 batched-prefill rendezvous
requires; every other task is awaited, which serializes `plan_step` against
the runner's acknowledgement. Because each `handle_first_task` arm publishes
its new `RunnerStatusUpdated` BEFORE calling `acknowledge_task`,
`supervisor.start_task` cannot return until the suppressing status is already
in state -- so re-entering `plan()` cannot reproduce the task.

These tests are behavioural, not structural: they drive the REAL `plan_step`
coroutine with a `plan` stub that KEEPS RETURNING the lifecycle task (the
precise production hazard -- the guard status is not published until the
runner acks, so `plan()` would happily emit the task again), park the
dispatch in flight, deliver state-apply wakes, then cancel the loop and count
what was actually dispatched.

The in-flight dispatch is held on an event that is deliberately NEVER
released: the observation window is closed by cancelling the worker's task
group, so neither the loop nor the test can drift, and the count is read from
recorded state after the group has fully stopped.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import cast

import anyio
import pytest

import exo.worker.main as worker_main_module
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import Event, IndexedEvent
from exo.shared.types.tasks import (
    ConnectToGroup,
    StartWarmup,
    Task,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.utils.channels import channel
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    COMMAND_1_ID,
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
)


def _load_worker_main(*, gate: str | None) -> ModuleType:
    """Re-import `exo.worker.main` with the event-wake gate set to `gate`.

    The gate is a module-level `Final[bool]` read once at import, so flipping
    it REQUIRES a reload under a patched environment (same convention as
    `test_plan_step_event_wake.py`).
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


def _make_worker(module: ModuleType) -> Worker:
    _inbound_sender, inbound_receiver = channel[IndexedEvent]()
    outbound_sender, _outbound_receiver = channel[Event]()
    command_sender, _command_receiver = channel[ForwarderCommand]()
    download_sender, _download_receiver = channel[ForwarderDownloadCommand]()

    worker_class = cast(type[Worker], module.Worker)
    return worker_class(
        NODE_A,
        event_receiver=inbound_receiver,
        event_sender=outbound_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )


def _signal(worker: Worker) -> None:
    """What `_event_applier` does after EVERY state apply."""
    worker._signal_state_applied()  # pyright: ignore[reportPrivateUsage]


def _lifecycle_task(task_type: str) -> Task:
    if task_type == "ConnectToGroup":
        return ConnectToGroup(instance_id=INSTANCE_1_ID)
    return StartWarmup(instance_id=INSTANCE_1_ID)


def _text_generation_task() -> Task:
    return TextGeneration(
        instance_id=INSTANCE_1_ID,
        command_id=COMMAND_1_ID,
        task_status=TaskStatus.Pending,
        task_params=TextGenerationTaskParams(
            model=MODEL_A_ID,
            input=[InputMessage(role="user", content=InputMessageContent(""))],
        ),
    )


async def _drain_scheduler(checkpoints: int = 5) -> None:
    """Yield enough times for the worker's task group to reach a stable point."""
    for _ in range(checkpoints):
        await anyio.sleep(0)


# --------------------------------------------------------------------------
# THE REGRESSION: a lifecycle dispatch must be awaited, end to end.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task_type", ["ConnectToGroup", "StartWarmup"])
async def test_lifecycle_dispatch_is_blocked_so_a_wake_cannot_duplicate_it(
    task_type: str,
) -> None:
    """Gate ON + wakes during an in-flight lifecycle dispatch => ONE dispatch.

    The `plan` stub models the production hazard exactly: it keeps returning
    the lifecycle task, because the runner has not yet published the status
    transition (`RunnerConnecting` / `RunnerWarmingUp`) that would make
    `plan()` stop emitting it. Under the buggy `start_soon` dispatch the loop
    re-enters `plan()` on the very next wake and emits a SECOND task with a
    different `task_id` -- the 2026-09-15 crash-loop.
    """
    module = _load_worker_main(gate="1")
    worker = _make_worker(module)

    dispatched: list[Task] = []
    dispatch_entered = anyio.Event()
    hold_dispatch = anyio.Event()  # deliberately never released

    def plan_stub(*_args: object, **_kwargs: object) -> Task | None:
        return _lifecycle_task(task_type)

    async def recording_dispatch(task: Task) -> None:
        dispatched.append(task)
        dispatch_entered.set()
        # Park here for the whole hazard window: in production this is
        # `await event.wait()` inside `supervisor.start_task`, which only
        # returns once the runner acknowledges.
        await hold_dispatch.wait()

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(module, "plan", plan_stub)
        patched.setattr(worker, "_start_runner_task", recording_dispatch)

        async with worker._tg as worker_tg:  # pyright: ignore[reportPrivateUsage]
            worker_tg.start_soon(worker.plan_step)
            await _drain_scheduler()

            # Kick the loop so the first lifecycle task is dispatched.
            _signal(worker)
            await _drain_scheduler()

            assert dispatch_entered.is_set(), (
                f"the first {task_type} was never dispatched"
            )

            # THE HAZARD WINDOW: state-apply wakes arriving while the
            # lifecycle dispatch is still in flight. Under the old catch-all
            # `start_soon` arm each of these re-entered plan() and re-emitted
            # the task.
            for _ in range(5):
                _signal(worker)
                await _drain_scheduler()

            # Stop the loop; the in-flight dispatch is cancelled with it.
            worker_tg.cancel_scope.cancel()

    assert len(dispatched) == 1, (
        f"DUPLICATE DISPATCH: {len(dispatched)} {task_type} tasks were "
        f"dispatched for the same instance while the first was still in "
        f"flight. Distinct task_ids: {[str(t.task_id) for t in dispatched]}. "
        f"In production the runner accepts the first and dies on the second "
        f"with 'Received {task_type} outside of state machine' -- the "
        f"2026-09-15 crash-loop. A lifecycle dispatch must be awaited "
        f"(blocking), never routed through start_soon."
    )
    assert isinstance(dispatched[0], type(_lifecycle_task(task_type)))


@pytest.mark.parametrize("task_type", ["ConnectToGroup", "StartWarmup"])
async def test_lifecycle_dispatch_completes_before_the_loop_replans(
    task_type: str,
) -> None:
    """Ordering proof: `plan()` is never re-entered mid-dispatch.

    Records a statement order and asserts that between the first dispatch's
    START and its END, `plan()` never ran -- the serialization the fix relies
    on, stated independently of the duplicate count above.
    """
    module = _load_worker_main(gate="1")
    worker = _make_worker(module)

    order: list[str] = []
    plan_calls = 0
    dispatch_entered = anyio.Event()
    hold_dispatch = anyio.Event()  # deliberately never released
    plans_during_dispatch: list[str] = []

    def plan_stub(*_args: object, **_kwargs: object) -> Task | None:
        nonlocal plan_calls
        plan_calls += 1
        order.append(f"plan#{plan_calls}")
        return _lifecycle_task(task_type)

    async def recording_dispatch(_task: Task) -> None:
        order.append("dispatch:start")
        dispatch_entered.set()
        try:
            await hold_dispatch.wait()
        finally:
            # Recorded on the way out so a cancelled dispatch still closes
            # its span (the assertion below reads the order, not completion).
            order.append("dispatch:end")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(module, "plan", plan_stub)
        patched.setattr(worker, "_start_runner_task", recording_dispatch)

        async with worker._tg as worker_tg:  # pyright: ignore[reportPrivateUsage]
            worker_tg.start_soon(worker.plan_step)
            await _drain_scheduler()
            _signal(worker)
            await _drain_scheduler()
            assert dispatch_entered.is_set()
            for _ in range(5):
                _signal(worker)
                await _drain_scheduler()
                plans_during_dispatch.append(f"plan#{plan_calls}")
            worker_tg.cancel_scope.cancel()

    start_index = order.index("dispatch:start")
    end_index = order.index("dispatch:end", start_index)
    between = order[start_index + 1 : end_index]

    assert order[: start_index + 1] == ["plan#1", "dispatch:start"], (
        f"unexpected dispatch ordering: {order}"
    )
    assert between == [], (
        f"plan() ran {len(between)} time(s) inside an in-flight {task_type} "
        f"dispatch: {order}. The loop must stay parked in the dispatch until "
        f"the runner has published the status that suppresses the task. "
        f"(plan count seen from the test body: {plans_during_dispatch})"
    )


# --------------------------------------------------------------------------
# GUARD RAIL: the c>=2 prefill win must NOT be traded away for the fix.
# --------------------------------------------------------------------------


async def test_generation_dispatch_remains_non_blocking() -> None:
    """Generation tasks MUST keep the non-blocking dispatch.

    `db9b3384` made generation dispatch non-blocking so a 2nd request reaches
    the runner's work_queue inside the batched-prefill rendezvous window.
    Generation tasks carry a STABLE task_id and are dedupe-safe, so
    re-planning over an in-flight one is harmless -- the opposite of the
    lifecycle case. Making this arm blocking too would silently re-serialize
    c>=2 prefill.

    The assertion is the mirror image of the regression test: `plan()` MUST be
    re-entered while the generation dispatch is still in flight.
    """
    module = _load_worker_main(gate="1")
    worker = _make_worker(module)

    plan_calls = 0
    dispatch_entered = anyio.Event()
    hold_dispatch = anyio.Event()  # deliberately never released
    generation_task = _text_generation_task()
    calls_while_in_flight = 0

    def plan_stub(*_args: object, **_kwargs: object) -> Task | None:
        nonlocal plan_calls
        plan_calls += 1
        # Keep offering the SAME generation task (stable task_id), exactly as
        # cluster state does while the request is still Pending/Running.
        return generation_task

    async def recording_dispatch(_task: Task) -> None:
        dispatch_entered.set()
        await hold_dispatch.wait()

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(module, "plan", plan_stub)
        patched.setattr(worker, "_start_runner_task", recording_dispatch)

        async with worker._tg as worker_tg:  # pyright: ignore[reportPrivateUsage]
            worker_tg.start_soon(worker.plan_step)
            await _drain_scheduler()
            for _ in range(5):
                _signal(worker)
                await _drain_scheduler()
            assert dispatch_entered.is_set(), "generation task never dispatched"
            calls_while_in_flight = plan_calls
            worker_tg.cancel_scope.cancel()

    assert calls_while_in_flight > 1, (
        f"generation dispatch blocked the loop: plan() was called only "
        f"{calls_while_in_flight} time(s) while the first dispatch was still "
        f"in flight. Non-blocking dispatch is REQUIRED for the c>=2 "
        f"batched-prefill rendezvous (db9b3384) -- do not make this arm "
        f"blocking to fix the lifecycle hazard."
    )


# --------------------------------------------------------------------------
# LIVENESS: the newly-blocking lifecycle dispatch must not wedge plan_step.
# --------------------------------------------------------------------------


async def test_lifecycle_dispatch_unblocks_when_the_runner_dies_mid_dispatch() -> None:
    """A runner that dies mid-dispatch must not wedge `plan_step` forever.

    Making the lifecycle dispatch blocking introduces a liveness obligation
    that the previous non-blocking dispatch did not have: `plan_step` now
    sits inside `supervisor.start_task`'s ack wait. If that wait could never
    be released, the single `plan_step` driver would stop planning entirely --
    a strictly worse failure than the crash-loop it fixes.

    The production release path is `RunnerSupervisor._forward_events`'s
    `finally` block, which sets EVERY pending event when the runner's event
    stream ends (crash, SIGTERM, or a closed pipe), and `start_task`'s own
    `ClosedResourceError` handler, which drops the task instead of waiting.
    This test pins the contract at the `plan_step` level: whatever the
    supervisor does to release the wait, the loop must resume planning and
    the test must observe it. The stub below releases the ack the way a
    dying runner does -- the wait returns WITHOUT the guard status ever being
    published -- and the assertion is that the loop MOVES ON rather than
    hanging.
    """
    module = _load_worker_main(gate="1")
    worker = _make_worker(module)

    dispatched: list[Task] = []
    plan_calls = 0

    def plan_stub(*_args: object, **_kwargs: object) -> Task | None:
        nonlocal plan_calls
        plan_calls += 1
        # Keep offering the lifecycle task: a crashed runner never publishes
        # the guard status, so the real `plan()` would keep emitting it too.
        return _lifecycle_task("ConnectToGroup")

    async def dispatch_then_runner_dies(_task: Task) -> None:
        dispatched.append(_task)
        # Model the dead-runner release: `start_task` returns even though no
        # status transition was ever published. (In production this is the
        # supervisor's `_forward_events` finally-block setting `pending`, or
        # the `ClosedResourceError` drop path.)
        await anyio.sleep(0)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(module, "plan", plan_stub)
        patched.setattr(worker, "_start_runner_task", dispatch_then_runner_dies)

        async with worker._tg as worker_tg:  # pyright: ignore[reportPrivateUsage]
            worker_tg.start_soon(worker.plan_step)
            await _drain_scheduler()

            # Each wake must be able to run another planning iteration. If the
            # blocking dispatch could wedge, `plan_calls` would stall at 1 and
            # the loop would be parked forever.
            for _ in range(5):
                _signal(worker)
                await _drain_scheduler()

            assert plan_calls > 1, (
                f"plan_step stopped planning after {plan_calls} call(s) once "
                f"the runner released the dispatch without publishing the "
                f"guard status. The blocking dispatch must never be able to "
                f"wedge the single plan_step driver -- a dead runner has to "
                f"release the ack wait (supervisor._forward_events sets every "
                f"pending event; start_task drops the task on "
                f"ClosedResourceError), and the loop must then continue."
            )
            worker_tg.cancel_scope.cancel()

    assert len(dispatched) > 1, (
        "the loop did not keep dispatching after the runner died; it appears "
        "to have stalled inside the blocking dispatch"
    )

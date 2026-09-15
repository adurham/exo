"""I16 follow-up (2026-09-15): apply_task_created must not resurrect a
terminal task.

`apply_task_created` used to be an unconditional dict overwrite. Investigating
a deterministic-task_id alternative for I16 (never shipped -- rejected, see
PERFORMANCE_HISTORY.md) surfaced a real consequence of that: a task_id
collision lets a later TaskCreated flip an already-Complete (or otherwise
terminal) task back to an earlier status with no error anywhere. This does not
fix a currently-live bug -- task_ids are still random today, so a collision
cannot happen through any code path that exists right now -- but it closes the
generic hazard (any future source of a duplicate task_id would hit the same
silent corruption) at the one place that would actually stop it.
"""

from exo.shared.apply import apply_task_created
from exo.shared.types.events import TaskCreated
from exo.shared.types.state import State
from exo.shared.types.tasks import LoadModel, TaskId, TaskStatus
from exo.shared.types.worker.instances import InstanceId


def test_apply_task_created_adds_a_new_task():
    task_id = TaskId()
    instance_id = InstanceId()
    task = LoadModel(task_id=task_id, instance_id=instance_id)
    state = State()

    new_state = apply_task_created(TaskCreated(task_id=task_id, task=task), state)

    assert new_state.tasks[task_id] is task


def test_apply_task_created_ignores_a_repeat_for_an_already_complete_task():
    task_id = TaskId()
    instance_id = InstanceId()
    completed_task = LoadModel(
        task_id=task_id, instance_id=instance_id, task_status=TaskStatus.Complete
    )
    state = State(tasks={task_id: completed_task})

    # A stray/duplicate TaskCreated for the SAME task_id, as would happen if
    # anything ever collided task_ids (today: cannot happen: task_ids are
    # random; this pins the state layer against it regardless of source).
    resurrecting_task = LoadModel(
        task_id=task_id, instance_id=instance_id, task_status=TaskStatus.Pending
    )

    new_state = apply_task_created(
        TaskCreated(task_id=task_id, task=resurrecting_task), state
    )

    assert new_state.tasks[task_id] is completed_task
    assert new_state.tasks[task_id].task_status == TaskStatus.Complete


def test_apply_task_created_ignores_repeats_for_every_terminal_status():
    for terminal_status in (
        TaskStatus.Complete,
        TaskStatus.TimedOut,
        TaskStatus.Failed,
        TaskStatus.Cancelled,
    ):
        task_id = TaskId()
        instance_id = InstanceId()
        terminal_task = LoadModel(
            task_id=task_id, instance_id=instance_id, task_status=terminal_status
        )
        state = State(tasks={task_id: terminal_task})
        repeat = LoadModel(
            task_id=task_id, instance_id=instance_id, task_status=TaskStatus.Pending
        )

        new_state = apply_task_created(TaskCreated(task_id=task_id, task=repeat), state)

        assert new_state.tasks[task_id].task_status == terminal_status, (
            f"a repeat TaskCreated resurrected a {terminal_status} task"
        )


def test_apply_task_created_still_overwrites_a_non_terminal_task():
    """The guard must not become a blanket no-op for a second TaskCreated.

    A legitimate case exists: a non-terminal task (Pending/Running) can be
    updated by a fresh TaskCreated -- the guard is specifically about
    protecting TERMINAL state from being reverted, not about making
    apply_task_created idempotent in general.
    """
    task_id = TaskId()
    instance_id = InstanceId()
    pending_task = LoadModel(
        task_id=task_id, instance_id=instance_id, task_status=TaskStatus.Pending
    )
    state = State(tasks={task_id: pending_task})

    running_task = LoadModel(
        task_id=task_id, instance_id=instance_id, task_status=TaskStatus.Running
    )
    new_state = apply_task_created(
        TaskCreated(task_id=task_id, task=running_task), state
    )

    assert new_state.tasks[task_id] is running_task

# pyright: reportUnusedFunction=false, reportAny=false
"""Tests that a worker-side TaskStatus.Failed transition surfaces to the
HTTP client waiting on that command's stream, rather than hanging it
forever (see API._fail_stream_for_task's own docstring for the full
incident this closes -- confirmed on real 2-node hardware 2026-08-08:
a jaccl transport reconnect correctly failed the task at the worker
level, but the client that submitted the request never received a
response because TaskStatusUpdated(Failed) was previously never
translated into either an ErrorChunk or a queue close)."""

from unittest.mock import MagicMock

from exo.api.main import API
from exo.api.types import ImageGenerationTaskParams
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.state import State
from exo.shared.types.tasks import ImageGeneration, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId


def _make_api_with_state(state: State) -> API:
    """Create a minimal API instance with pre-set state."""
    api = object.__new__(API)
    api.state = state
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    return api


def _make_text_gen_task(
    instance_id: InstanceId,
    command_id: CommandId,
    *,
    task_status: TaskStatus = TaskStatus.Failed,
    error_message: str | None = "jaccl transport fault",
) -> TextGeneration:
    return TextGeneration(
        instance_id=instance_id,
        command_id=command_id,
        task_status=task_status,
        error_message=error_message,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        ),
    )


def test_fail_stream_sends_error_chunk_and_closes_text_generation_queue() -> None:
    """A Failed TextGeneration task pushes an ErrorChunk to the waiting
    client's stream, then closes and evicts the queue -- matching the real
    incident this fix closes (client was previously left awaiting a chunk
    that would never arrive)."""
    instance_id = InstanceId("inst-1")
    command_id = CommandId("cmd-1")
    task = _make_text_gen_task(instance_id, command_id)

    state = State(tasks={task.task_id: task})
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._text_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._fail_stream_for_task(task.task_id)  # pyright: ignore[reportPrivateUsage]

    sender.send_nowait.assert_called_once()
    (sent_chunk,) = sender.send_nowait.call_args.args
    assert isinstance(sent_chunk, ErrorChunk)
    assert sent_chunk.error_message == "jaccl transport fault"
    assert sent_chunk.finish_reason == "error"

    sender.close.assert_called_once()
    assert command_id not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_fail_stream_sends_error_chunk_and_closes_image_generation_queue() -> None:
    """Same behavior for a Failed ImageGeneration task."""
    instance_id = InstanceId("inst-img")
    command_id = CommandId("cmd-img")
    task = ImageGeneration(
        instance_id=instance_id,
        command_id=command_id,
        task_status=TaskStatus.Failed,
        error_message="runner crashed mid-generation",
        task_params=ImageGenerationTaskParams(prompt="a cat", model="test-model"),
    )

    state = State(tasks={task.task_id: task})
    api = _make_api_with_state(state)

    sender = MagicMock()
    api._image_generation_queues[command_id] = sender  # pyright: ignore[reportPrivateUsage]

    api._fail_stream_for_task(task.task_id)  # pyright: ignore[reportPrivateUsage]

    sender.send_nowait.assert_called_once()
    (sent_chunk,) = sender.send_nowait.call_args.args
    assert isinstance(sent_chunk, ErrorChunk)
    assert sent_chunk.error_message == "runner crashed mid-generation"

    sender.close.assert_called_once()
    assert command_id not in api._image_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_fail_stream_ignores_unrelated_commands() -> None:
    """Failing one task does NOT touch another command's still-open stream."""
    instance_id = InstanceId("inst-1")
    failed_command_id = CommandId("cmd-failed")
    other_command_id = CommandId("cmd-keep")
    failed_task = _make_text_gen_task(instance_id, failed_command_id)
    other_task = _make_text_gen_task(
        instance_id,
        other_command_id,
        task_status=TaskStatus.Running,
        error_message=None,
    )

    state = State(
        tasks={failed_task.task_id: failed_task, other_task.task_id: other_task}
    )
    api = _make_api_with_state(state)

    other_sender = MagicMock()
    api._text_generation_queues[other_command_id] = other_sender  # pyright: ignore[reportPrivateUsage]

    api._fail_stream_for_task(failed_task.task_id)  # pyright: ignore[reportPrivateUsage]

    other_sender.send_nowait.assert_not_called()
    other_sender.close.assert_not_called()
    assert other_command_id in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_fail_stream_is_a_noop_when_no_stream_is_open() -> None:
    """If the client already disconnected (no queue registered), failing
    the task must not raise -- just a no-op."""
    instance_id = InstanceId("inst-1")
    command_id = CommandId("cmd-1")
    task = _make_text_gen_task(instance_id, command_id)

    state = State(tasks={task.task_id: task})
    api = _make_api_with_state(state)

    # No queue registered for command_id -- must not raise.
    api._fail_stream_for_task(task.task_id)  # pyright: ignore[reportPrivateUsage]

    assert command_id not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


def test_fail_stream_is_a_noop_for_unknown_task_id() -> None:
    """An unknown task_id (already evicted from state) must not raise."""
    from exo.shared.types.tasks import TaskId

    state = State(tasks={})
    api = _make_api_with_state(state)

    api._fail_stream_for_task(TaskId("does-not-exist"))  # pyright: ignore[reportPrivateUsage]

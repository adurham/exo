"""Tests for the client-supplied ``correlation_id`` request-identification
surface.

BACKGROUND. ``PrefillCancelled`` covers the window where a client cancels
a request that is still prefilling. Before this field existed, the only
way a client could learn the ``CommandId`` of its own in-flight request
was to read the ``id`` of the FIRST STREAMED CHUNK -- which by definition
does not exist until prefill is over. So the mid-prefill cancel window
was unreachable from any client, and the ``PrefillCancelled`` path could
never be hardware-verified (see
bench/section85_prefill_cancel_hardware_test.py).

``correlation_id`` closes that gap: the client picks the id BEFORE
sending, and exo echoes it into cluster state as part of the task's
params, which the master indexes before prefill begins.

These tests pin the contract the hardware harness depends on:
  1. The wire type accepts it.
  2. The adapter propagates it into the internal task params verbatim.
  3. It survives the State serialization round-trip under the camelCase
     alias generator, at the exact JSON path the harness reads.
  4. Absence is fine and stays None (no behavioural change for every
     existing client).
"""

from __future__ import annotations

import json
from typing import Any, cast

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types.api import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import CommandId
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId


def _request(correlation_id: str | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=ModelId("test/model"),
        messages=[ChatCompletionMessage(role="user", content="hello")],
        correlation_id=correlation_id,
    )


async def test_chat_request_propagates_correlation_id() -> None:
    """The adapter must carry the client's id into the internal params
    untouched -- that object is what ends up in State."""
    params = await chat_request_to_text_generation(_request("harness-abc-123"))
    assert params.correlation_id == "harness-abc-123"


async def test_chat_request_without_correlation_id_stays_none() -> None:
    """Every pre-existing client omits the field; it must remain None and
    change nothing."""
    params = await chat_request_to_text_generation(_request(None))
    assert params.correlation_id is None


def test_correlation_id_is_visible_in_serialized_state() -> None:
    """The harness reads /state and looks for
    tasks[*].TextGeneration.taskParams.correlationId. Pin that exact
    path, alias included -- if the alias generator or field name drifts,
    the harness goes blind and would silently degrade to cancelling
    post-prefill, which is precisely the bug being fixed."""
    correlation_id = "harness-state-probe"
    task = TextGeneration(
        task_id=TaskId("task-1"),
        task_status=TaskStatus.Running,
        instance_id=InstanceId("instance-1"),
        command_id=CommandId("command-1"),
        task_params=TextGenerationTaskParams(
            model=ModelId("test/model"),
            input=[],
            correlation_id=correlation_id,
        ),
    )
    state = State(tasks={task.task_id: task})

    serialized = cast(dict[str, Any], json.loads(state.model_dump_json(by_alias=True)))

    tasks = cast(dict[str, Any], serialized["tasks"])
    body = cast(dict[str, Any], tasks["task-1"]["TextGeneration"])
    assert body["commandId"] == "command-1"
    # NOTE the asymmetric casing, verified against the live cluster's
    # /state: the Task envelope is camelCased by State's alias generator
    # ("taskId"/"commandId"/"taskParams"), but TextGenerationTaskParams is
    # a plain BaseModel with no alias generator, so its OWN fields stay
    # snake_case. The harness must therefore read
    # tasks[*].TextGeneration.taskParams.correlation_id -- getting this
    # wrong makes the harness silently blind, which degrades it into
    # cancelling post-prefill: the exact bug it exists to prevent.
    assert body["taskParams"]["correlation_id"] == correlation_id
    assert "correlationId" not in body["taskParams"]


def test_correlation_id_absent_serializes_as_null_not_missing() -> None:
    """The harness treats "field present but null" and "field missing"
    identically, but a missing key would also mask a serialization
    regression. Pin that the key is always emitted."""
    task = TextGeneration(
        task_id=TaskId("task-2"),
        task_status=TaskStatus.Pending,
        instance_id=InstanceId("instance-1"),
        command_id=CommandId("command-2"),
        task_params=TextGenerationTaskParams(model=ModelId("test/model"), input=[]),
    )
    state = State(tasks={task.task_id: task})
    serialized = cast(dict[str, Any], json.loads(state.model_dump_json(by_alias=True)))
    params = cast(
        dict[str, Any], serialized["tasks"]["task-2"]["TextGeneration"]["taskParams"]
    )
    assert "correlation_id" in params
    assert params["correlation_id"] is None

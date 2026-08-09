# pyright: reportUnusedFunction=false, reportAny=false
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.common import CommandId
from exo.shared.types.state import State


def _make_api() -> Any:
    """Create a minimal API instance with cancel route and error handler."""

    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    # 2026-08-09: cancel_command() now looks up the command's task_id via
    # self.state.tasks (see _find_task_for_command's own docstring) to poll
    # for the real cancellation path completing before falling back to an
    # eager sender.close() -- needs a real (even if empty) State object.
    # An empty State means _find_task_for_command always returns None for
    # these synthetic command_ids (no matching task was ever registered),
    # so cancel_command falls straight through to the pre-existing
    # immediate-close behavior these tests already assert on -- exactly
    # matching each test's own setup (a sender registered directly, with
    # no corresponding task ever added to state).
    api.state = State()
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/v1/cancel/{command_id}")(api.cancel_command)
    return api


def test_cancel_nonexistent_command_returns_404() -> None:
    """Cancel for an unknown command_id returns 404 in OpenAI error format."""
    api = _make_api()
    client = TestClient(api.app)

    response = client.post("/v1/cancel/nonexistent-id")
    assert response.status_code == 404
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Command not found or already completed"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_cancel_active_text_generation() -> None:
    """Cancel an active text generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("text-cmd-123")
    sender = MagicMock()
    api._text_generation_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


def test_cancel_active_image_generation() -> None:
    """Cancel an active image generation command: returns 200, sender.close() called."""
    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("img-cmd-456")
    sender = MagicMock()
    api._image_generation_queues[cid] = sender

    response = client.post(f"/v1/cancel/{cid}")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    assert data["command_id"] == str(cid)
    sender.close.assert_called_once()
    api._send.assert_called_once()
    task_cancelled = api._send.call_args[0][0]
    assert task_cancelled.cancelled_command_id == cid


def test_cancel_returns_immediately_once_task_leaves_tracked_state() -> None:
    """Regression test for the 2026-08-09 real-hardware fix (design doc
    Section 27): the pre-fix bug called sender.close() IMMEDIATELY after
    sending TaskCancelled, in the same breath -- racing ahead of the
    worker planner's own ~100ms poll tick and losing the cancellation on
    real hardware (confirmed: zero 'Worker plan: CancelTask' log lines
    across three live cancel attempts). The fix instead polls
    self.state.tasks for the task to genuinely leave tracked state (the
    real completion path: TaskCancelled -> planner -> CancelTask ->
    runner -> natural stream end -> TaskFinished -> TaskDeleted) before
    touching the sender at all.

    This test proves the NEW behavior end-to-end: when the real
    completion path removes the task from self.state.tasks WHILE
    cancel_command is still polling, the method returns success WITHOUT
    ever calling sender.close() itself -- proving it genuinely waited for
    (and deferred to) the real path instead of racing ahead of it."""
    import threading
    import time as time_module

    from exo.shared.types.common import ModelId
    from exo.shared.types.state import State
    from exo.shared.types.tasks import TextGeneration
    from exo.shared.types.text_generation import TextGenerationTaskParams
    from exo.shared.types.worker.instances import InstanceId

    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("text-cmd-real-completion")
    sender = MagicMock()
    api._text_generation_queues[cid] = sender

    task = TextGeneration(
        instance_id=InstanceId("instance-1"),
        command_id=cid,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"), input=[], stream=True
        ),
    )
    api.state = State(tasks={task.task_id: task})

    def _simulate_real_completion_path() -> None:
        # Simulates the REAL completion path (planner dispatches
        # CancelTask -> runner stops -> stream ends naturally ->
        # TaskFinished -> TaskDeleted) removing the task from tracked
        # state shortly after TaskCancelled was sent -- well within
        # CANCEL_ACK_TIMEOUT_SECONDS, so cancel_command's poll loop
        # should observe it and return WITHOUT ever force-closing.
        time_module.sleep(0.15)
        api.state = State(tasks={})

    completer = threading.Thread(target=_simulate_real_completion_path)
    completer.start()
    response = client.post(f"/v1/cancel/{cid}")
    completer.join(timeout=5.0)

    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert data["message"] == "Command cancelled."
    # THE KEY ASSERTION: sender.close() was never called by cancel_command
    # itself -- the real completion path (simulated above) is what
    # actually tore the stream down, exactly matching the fix's own
    # documented intent of deferring to the natural teardown path rather
    # than racing ahead of it.
    sender.close.assert_not_called()
    api._send.assert_called_once()


def test_cancel_falls_back_to_force_close_if_task_never_completes() -> None:
    """The safety-valve half of the same fix: if the real completion path
    never removes the task from tracked state within
    CANCEL_ACK_TIMEOUT_SECONDS (a genuinely hung/crashed runner), the
    method MUST still fall back to the old eager sender.close() so the
    client's HTTP call doesn't hang forever."""
    from exo.api.main import CANCEL_ACK_TIMEOUT_SECONDS
    from exo.shared.types.common import ModelId
    from exo.shared.types.state import State
    from exo.shared.types.tasks import TextGeneration
    from exo.shared.types.text_generation import TextGenerationTaskParams
    from exo.shared.types.worker.instances import InstanceId

    api = _make_api()
    client = TestClient(api.app)

    cid = CommandId("text-cmd-never-completes")
    sender = MagicMock()
    api._text_generation_queues[cid] = sender

    task = TextGeneration(
        instance_id=InstanceId("instance-1"),
        command_id=cid,
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"), input=[], stream=True
        ),
    )
    # The task is NEVER removed from state for the whole test -- simulates
    # a runner that never actually responds to the cancellation.
    api.state = State(tasks={task.task_id: task})

    response = client.post(f"/v1/cancel/{cid}")

    assert response.status_code == 200
    sender.close.assert_called_once()
    api._send.assert_called_once()
    # Sanity: the timeout constant this test relies on staying short
    # enough to run quickly is itself reasonable (a few seconds, not
    # minutes) -- if this ever changes to something huge, this test
    # would silently become extremely slow rather than catching a real
    # regression, so assert the assumption explicitly.
    assert 0 < CANCEL_ACK_TIMEOUT_SECONDS <= 30.0

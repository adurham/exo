# pyright: reportUnusedFunction=false, reportAny=false
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams


def test_http_exception_handler_formats_openai_style() -> None:
    """Test that HTTPException is converted to OpenAI-style error format."""

    app = FastAPI()

    # Setup exception handler
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    # Add test routes that raise HTTPException
    @app.get("/test-error")
    async def _test_error() -> None:
        raise HTTPException(status_code=500, detail="Test error message")

    @app.get("/test-not-found")
    async def _test_not_found() -> None:
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app)

    # Test 500 error
    response = client.get("/test-error")
    assert response.status_code == 500
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Test error message"
    assert data["error"]["type"] == "Internal Server Error"
    assert data["error"]["code"] == 500

    # Test 404 error
    response = client.get("/test-not-found")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["message"] == "Resource not found"
    assert data["error"]["type"] == "Not Found"
    assert data["error"]["code"] == 404


def test_degeneration_error_becomes_real_500_not_empty_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (2026-07-26): the non-streaming /v1/chat/completions
    branch used to wrap collect_chat_response's single-yield generator in
    a StreamingResponse. Once that response starts (status 200 committed),
    an exception raised later inside the generator -- e.g.
    collect_chat_response raising ValueError on an ErrorChunk, which is
    exactly what the degeneration kill-switch produces -- could no longer
    change the status code: the client got a "successful" empty 200 body
    with zero error signal. This test exercises the FULL route (not just
    collect_chat_response or the exception handler in isolation, which
    are already covered elsewhere) with a mocked chunk stream that yields
    an ErrorChunk, and asserts a real 500 with proper OpenAI-style error
    JSON comes back -- not an empty 200. If this regresses (e.g. someone
    reverts to StreamingResponse(collect_chat_response(...))), this test
    must fail.
    """
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    api._setup_routes()  # pyright: ignore[reportPrivateUsage]

    async def _fake_validate_model_has_instance(model_id: ModelId) -> ModelId:
        return model_id

    class _FakeCommand:
        def __init__(self) -> None:
            from exo.shared.types.common import CommandId

            self.command_id = CommandId("test-degen-cmd")

    async def _fake_send_text_generation_with_images(
        task_params: TextGenerationTaskParams,
    ) -> Any:
        return _FakeCommand()

    async def _fake_token_chunk_stream(
        command_id: Any,
    ) -> AsyncGenerator[
        PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
    ]:
        yield ErrorChunk(
            model=ModelId("mlx-community/DeepSeek-V4-Flash"),
            error_message=(
                "Generation terminated: repetition-loop degeneration detected"
            ),
        )

    monkeypatch.setattr(
        api, "_validate_model_has_instance", _fake_validate_model_has_instance
    )
    monkeypatch.setattr(
        api,
        "_send_text_generation_with_images",
        _fake_send_text_generation_with_images,
    )
    monkeypatch.setattr(api, "_token_chunk_stream", _fake_token_chunk_stream)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx-community/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )

    assert response.status_code == 500, (
        f"expected a real 500 on degeneration, got {response.status_code} "
        f"body={response.content!r} -- if this is 200 with an empty body, "
        "the StreamingResponse-wrapping-a-raising-generator bug is back"
    )
    data: dict[str, Any] = response.json()
    assert "error" in data
    assert "repetition-loop degeneration" in data["error"]["message"]

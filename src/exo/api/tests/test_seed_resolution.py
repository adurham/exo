"""Admission-time sampling-seed resolution (regression, 2026-07-27).

The engine seeds the global MLX PRNG per task with a FIXED 42 whenever
``task_params.seed`` is None (batch_generate.py submit/submit_batched,
generate.py) — every rank must seed identically or pipeline-parallel
sampling desyncs, so the seed cannot be invented rank-locally. But with the
API never resolving a seed, every seedless request was fully deterministic:
byte-identical completions for byte-identical requests (verified live —
two identical seedless /v1/chat/completions calls at temperature 0.8
returned identical tokens). Client-side retry-for-diversity strategies
were therefore no-ops: a retry of a degenerate draw replayed the SAME
degenerate draw with probability 1 (the 2026-07-27 hard_eval
empty-content failure loop, 7/21 trials).

Contract pinned here (mirrors _ensure_seed for image generation):
``_send_text_generation_with_images`` resolves a random seed at admission
when the client sent none, and preserves an explicit client seed exactly —
including 0, which a careless ``task.seed or 42`` silently replaced.
"""

import pytest

from exo.api.main import API
from exo.shared.types.common import ModelId
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)


def _make_api_with_send_capture() -> tuple[API, list[TextGeneration]]:
    api = object.__new__(API)
    sent: list[TextGeneration] = []

    async def _fake_send(command: TextGeneration) -> None:
        sent.append(command)

    api._send = _fake_send  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return api, sent


def _task_params(seed: int | None) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("mlx-community/DeepSeek-V4-Flash"),
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        seed=seed,
    )


@pytest.mark.anyio
async def test_seedless_request_gets_random_seed_resolved_at_admission() -> None:
    api, sent = _make_api_with_send_capture()
    command = await api._send_text_generation_with_images(_task_params(seed=None))  # pyright: ignore[reportPrivateUsage]
    resolved = command.task_params.seed
    assert resolved is not None, (
        "A seedless request must leave admission with a resolved seed — "
        "otherwise the engine's fixed-42 fallback makes generation "
        "deterministic and client retries replay the same degenerate draw."
    )
    assert 0 <= resolved <= 2**32 - 1
    assert sent, "command must still be dispatched"


@pytest.mark.anyio
async def test_two_seedless_requests_get_different_seeds() -> None:
    api, _ = _make_api_with_send_capture()
    seeds: set[int | None] = set()
    for _ in range(8):
        command = await api._send_text_generation_with_images(  # pyright: ignore[reportPrivateUsage]
            _task_params(seed=None)
        )
        seeds.add(command.task_params.seed)
    assert len(seeds) > 1, (
        "Consecutive seedless requests must not share one seed — retries "
        "need genuinely fresh draws."
    )


@pytest.mark.anyio
async def test_explicit_seed_is_preserved_exactly() -> None:
    api, _ = _make_api_with_send_capture()
    command = await api._send_text_generation_with_images(_task_params(seed=7))  # pyright: ignore[reportPrivateUsage]
    assert command.task_params.seed == 7


@pytest.mark.anyio
async def test_explicit_seed_zero_is_preserved_not_replaced() -> None:
    """Seed 0 is a valid client choice — guards the ``seed or 42`` bug class."""
    api, _ = _make_api_with_send_capture()
    command = await api._send_text_generation_with_images(_task_params(seed=0))  # pyright: ignore[reportPrivateUsage]
    assert command.task_params.seed == 0

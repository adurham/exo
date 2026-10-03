"""Wire-level tests for the reasoning-effort ceiling clamp.

A request carrying ``reasoning_effort`` above the wire ceiling (``max``,
``ultra``) must be clamped DOWN to the ceiling (``xhigh``) at the API boundary,
rather than being rejected with a 422 literal_error. Genuine typos must still
fail validation with the valid set named.
"""

import pydantic
import pytest

from exo.api.types.api import ChatCompletionRequest
from exo.api.types.openai_responses import ResponsesRequest
from exo.shared.types.text_generation import resolve_reasoning_params

_CHAT_BASE: dict[str, object] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


def _chat_request(**kwargs: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({**_CHAT_BASE, **kwargs})


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_chat_request_over_ceiling_clamps_to_xhigh(effort: str) -> None:
    request = _chat_request(reasoning_effort=effort)
    assert request.reasoning_effort == "xhigh"


@pytest.mark.parametrize(
    "effort", ["none", "minimal", "low", "medium", "high", "xhigh"]
)
def test_chat_request_at_or_below_ceiling_passes_through(effort: str) -> None:
    request = _chat_request(reasoning_effort=effort)
    assert request.reasoning_effort == effort


def test_chat_request_none_stays_none() -> None:
    request = _chat_request(reasoning_effort=None)
    assert request.reasoning_effort is None


def test_chat_request_unknown_effort_still_fails_validation() -> None:
    with pytest.raises(pydantic.ValidationError):
        _chat_request(reasoning_effort="bogus")


def test_clamped_chat_request_resolves_to_xhigh_thinking() -> None:
    request = _chat_request(reasoning_effort="max")
    assert resolve_reasoning_params(request.reasoning_effort, None) == ("xhigh", True)


def test_responses_request_over_ceiling_clamps_to_xhigh() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "hello",
            "reasoning": {"effort": "max"},
        }
    )
    assert request.reasoning is not None
    assert request.reasoning.effort == "xhigh"


def test_responses_request_unknown_effort_still_fails_validation() -> None:
    with pytest.raises(pydantic.ValidationError):
        ResponsesRequest.model_validate(
            {
                "model": "test-model",
                "input": "hello",
                "reasoning": {"effort": "bogus"},
            }
        )

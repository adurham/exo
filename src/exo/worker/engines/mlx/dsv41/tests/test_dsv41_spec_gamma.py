# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedFunction=false, reportMissingModuleSource=false, reportOptionalMemberAccess=false, reportMissingTypeArgument=false
"""Scoped tests for the per-request dsv41 ``spec_gamma`` wire field.

Pins three contracts for the ADDITIVE, default-safe change:

  (a) an absent / non-int ``spec_gamma`` resolves to ``None``, which the engine
      reads as "use the default gamma 3" -- byte-identical to the pre-change
      behavior;
  (b) an in-range integer passes through unchanged;
  (c) an out-of-range integer is CLAMPED into [1, 6] at the API boundary, and a
      typo / bool resolves to ``None`` (degrades to the engine default rather
      than 422-ing the request).

The engine side of the threading is also pinned: ``_rounds(spec_gamma=...)``
selects the ``GammaPolicy`` start gamma (None => ``self.gamma``), and the
per-position acceptance histogram accumulates (index-clamped at 6).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types.api import ChatCompletionRequest
from exo.shared.models.model_cards import ModelId

_CHAT_BASE: dict[str, object] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


def _req(**kwargs: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({**_CHAT_BASE, **kwargs})


# ----------------------------------------------------------- wire field (API)


def test_absent_spec_gamma_is_none() -> None:
    """Omitting the field is non-breaking: it resolves to the engine default."""
    assert _req().spec_gamma is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, 1), (3, 3), (4, 4), (6, 6), (0, 1), (7, 6), (-3, 1), (99, 6)],
)
def test_spec_gamma_in_range_passes_and_out_of_range_clamps(
    raw: int, expected: int
) -> None:
    """An int is clamped into the engine's supported [1, 6]."""
    assert _req(spec_gamma=raw).spec_gamma == expected


@pytest.mark.parametrize("bad", [None, True, False, "3", 2.5, "abc", [1]])
def test_spec_gamma_non_int_passes_through_as_none(bad: object) -> None:
    """A typo / bool degrades to None (engine default), never a 422."""
    assert _req(spec_gamma=bad).spec_gamma is None


async def test_adapter_forwards_spec_gamma() -> None:
    """The chat adapter copies the clamped value onto TextGenerationTaskParams."""
    params = await chat_request_to_text_generation(_req(spec_gamma=5))
    assert params.spec_gamma == 5


async def test_adapter_absent_spec_gamma_is_none() -> None:
    params = await chat_request_to_text_generation(_req())
    assert params.spec_gamma is None


async def test_adapter_clamps_out_of_range_spec_gamma() -> None:
    params = await chat_request_to_text_generation(_req(spec_gamma=42))
    assert params.spec_gamma == 6


# ------------------------------------------------- engine threading (stub)


class _SessionStub:
    """The surface ``_rounds`` touches when ``_one_round`` is stubbed out."""

    def __init__(self) -> None:
        self.cache = SimpleNamespace(cache=object())
        self.draft_state = None
        self.eos_id = -1

    def maybe_checkpoint(self) -> None:
        pass


def _engine_with_head() -> Any:
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded

    from ._stub_harness import StubBody

    body = StubBody()

    class _Sender:
        def send(self, item: object) -> None: ...

    class _Receiver:
        def collect(self) -> list[object]:
            return []

    loaded = Dsv41Loaded(
        model=body,
        tokenizer=None,  # type: ignore[arg-type]
        args=body.args,
        model_path=None,  # type: ignore[arg-type]
        built_layers=[0],
        full_stack=False,
        rank=0,
        world=1,
        load_seconds=0.0,
        head=object(),  # a head is present; _one_round is stubbed in the test
    )
    return eng_mod.Dsv41Engine(
        loaded=loaded,
        model_id=ModelId("x/y"),
        group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=_Sender(),  # type: ignore[arg-type]
        device_rank=0,
        speculative=True,
        prefill_chunk_size=4,
        max_kv_tokens=1024,
    )


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch, seen: list[Any], *, accepted: int
) -> None:
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    def _stub(
        _engine: Any, *, policy: Any, **_kw: Any
    ) -> tuple[list[int], float, int, int]:
        seen.append(policy)
        return [1], 0.0, accepted, 1

    monkeypatch.setattr(eng_mod, "_one_round", _stub)


def test_rounds_none_spec_gamma_uses_engine_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spec_gamma=None`` => policy gamma is the engine default (3)."""
    seen: list[Any] = []
    _install_recorder(monkeypatch, seen, accepted=0)
    engine = _engine_with_head()
    for _ in engine._rounds(_SessionStub(), anchor=1, max_tokens=1):
        pass
    assert seen[0] is not None
    assert seen[0].next() == 3


def test_rounds_explicit_spec_gamma_selects_policy_gamma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spec_gamma=4`` => policy gamma is 4."""
    seen: list[Any] = []
    _install_recorder(monkeypatch, seen, accepted=0)
    engine = _engine_with_head()
    for _ in engine._rounds(_SessionStub(), anchor=1, max_tokens=1, spec_gamma=4):
        pass
    assert seen[0] is not None
    assert seen[0].next() == 4


def test_rounds_accept_histogram_accumulates_and_clamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The histogram counts rounds by accepted-draft count, index-clamped at 6."""
    seen: list[Any] = []
    _install_recorder(monkeypatch, seen, accepted=99)
    engine = _engine_with_head()
    for _ in engine._rounds(_SessionStub(), anchor=1, max_tokens=1):
        pass
    assert engine._spec_accept_hist == [0, 0, 0, 0, 0, 0, 1]


def test_final_response_carries_accept_histogram() -> None:
    """The terminal stats carry the histogram through to GenerationStats."""
    from exo.worker.engines.mlx.dsv41.rounds import _final_response

    resp = _final_response(
        token=1,
        text="",
        prefill_tps=1.0,
        prompt_tokens=3,
        generated=0,
        reason="stop",
        mtp_cycles=5,
        mtp_accepted=2,
        mtp_accept_hist=[1, 0, 2, 0, 0, 0, 0],
    )
    assert resp.stats is not None
    assert resp.stats.mtp_accepted_histogram_cumulative == [1, 0, 2, 0, 0, 0, 0]

# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedFunction=false, reportMissingModuleSource=false, reportOptionalMemberAccess=false, reportMissingTypeArgument=false
"""Scoped tests for the per-request dsv41 ``round_prof`` wire field.

Pins the ADDITIVE, default-safe plumbing for the per-round instrumentation mode.
The mode is a documented NO-OP until the timer lands (a later change), so every
value must currently be byte-identical -- these tests pin the WIRE and the
RESOLUTION, not any timer behaviour.

Contracts:
  (a) an absent / non-int ``round_prof`` resolves to ``None`` (the engine
      default), and an in-range int passes through unchanged;
  (b) an out-of-range int is CLAMPED into [0, 2] at the API boundary, and a
      typo / bool resolves to ``None`` (degrades to the engine default rather
      than 422-ing the request);
  (c) the engine resolves per request: an explicit field wins over the
      import-time env default (``EXO_DSV41_ROUND_PROF``), ``None`` uses the env
      default, and a garbage env value falls back to 0;
  (d) ``_rounds`` hands the RESOLVED value to ``_one_round`` as a keyword, and
      ``_one_round`` accepts-and-ignores it (no-op).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types.api import ChatCompletionRequest
from exo.shared.models.model_cards import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams

_CHAT_BASE: dict[str, object] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


def _req(**kwargs: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({**_CHAT_BASE, **kwargs})


# ----------------------------------------------------------- wire field (API)


def test_absent_round_prof_is_none() -> None:
    """Omitting the field is non-breaking: it resolves to the engine default."""
    assert _req().round_prof is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 0), (1, 1), (2, 2), (-3, 0), (9, 2), (99, 2)],
)
def test_round_prof_in_range_passes_and_out_of_range_clamps(
    raw: int, expected: int
) -> None:
    """An int is clamped into the engine's supported [0, 2]."""
    assert _req(round_prof=raw).round_prof == expected


@pytest.mark.parametrize("bad", [None, True, False, "1", 1.5, "abc", [1]])
def test_round_prof_non_int_passes_through_as_none(bad: object) -> None:
    """A typo / bool degrades to None (engine default), never a 422."""
    assert _req(round_prof=bad).round_prof is None


async def test_adapter_forwards_round_prof() -> None:
    """The chat adapter copies the clamped value onto TextGenerationTaskParams."""
    params = await chat_request_to_text_generation(_req(round_prof=2))
    assert params.round_prof == 2


async def test_adapter_absent_round_prof_is_none() -> None:
    params = await chat_request_to_text_generation(_req())
    assert params.round_prof is None


async def test_adapter_clamps_out_of_range_round_prof() -> None:
    params = await chat_request_to_text_generation(_req(round_prof=42))
    assert params.round_prof == 2


# ----------------------------------------------------------- params field


def test_params_field_defaults_to_none() -> None:
    """The task-params field defaults to None (engine default applies)."""
    params = TextGenerationTaskParams(
        model=ModelId("x/y"),
        input=[],
    )
    assert params.round_prof is None


# ----------------------------------------------------------- env default


def test_read_round_prof_default_unset_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    monkeypatch.delenv("EXO_DSV41_ROUND_PROF", raising=False)
    assert eng_mod._read_round_prof_default() == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", 0), ("   ", 0), ("0", 0), ("1", 1), ("2", 2), ("9", 2), ("-3", 0)],
)
def test_read_round_prof_default_parses_and_clamps(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    monkeypatch.setenv("EXO_DSV41_ROUND_PROF", raw)
    assert eng_mod._read_round_prof_default() == expected


def test_read_round_prof_default_garbage_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed env value falls back to 0 (off) rather than crashing."""
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    monkeypatch.setenv("EXO_DSV41_ROUND_PROF", "yes")
    assert eng_mod._read_round_prof_default() == 0


def test_import_time_default_is_zero() -> None:
    """Regression guard: the process default is 0 with the env unset."""
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    assert eng_mod._ROUND_PROF_DEFAULT == 0


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
    monkeypatch: pytest.MonkeyPatch, seen: list[int]
) -> None:
    """Stub ``_one_round`` and record the ``round_prof`` it was handed."""
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    def _stub(
        _engine: Any,
        *,
        round_prof: int = -1,
        **_kw: Any,
    ) -> tuple[list[int], float, int, int]:
        seen.append(round_prof)
        return [1], 0.0, 1, 1

    monkeypatch.setattr(eng_mod, "_one_round", _stub)


def test_rounds_explicit_round_prof_reaches_one_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request's explicit mode is threaded through to ``_one_round``."""
    seen: list[int] = []
    _install_recorder(monkeypatch, seen)
    engine = _engine_with_head()
    for _ in engine._rounds(_SessionStub(), anchor=1, max_tokens=1, round_prof=2):
        pass
    assert seen == [2]


def test_rounds_none_uses_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``round_prof=None`` => the import-time env default."""
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    monkeypatch.setattr(eng_mod, "_ROUND_PROF_DEFAULT", 2)
    seen: list[int] = []
    _install_recorder(monkeypatch, seen)
    engine = _engine_with_head()
    for _ in engine._rounds(_SessionStub(), anchor=1, max_tokens=1):
        pass
    assert seen == [2]


def test_rounds_request_overrides_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit field WINS over the env default."""
    from exo.worker.engines.mlx.dsv41 import engine as eng_mod

    monkeypatch.setattr(eng_mod, "_ROUND_PROF_DEFAULT", 0)
    seen: list[int] = []
    _install_recorder(monkeypatch, seen)
    engine = _engine_with_head()
    for _ in engine._rounds(_SessionStub(), anchor=1, max_tokens=1, round_prof=1):
        pass
    assert seen == [1]


def test_rounds_no_field_is_zero_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: with nothing set, ``_one_round`` gets ``round_prof=0``.

    This is the byte-identical path -- the resolved mode is the off default and
    the rest of the call is unchanged (the stub is exercised through the same
    ``_rounds`` control flow the spec_gamma tests use).
    """
    seen: list[int] = []
    _install_recorder(monkeypatch, seen)
    engine = _engine_with_head()
    for _ in engine._rounds(_SessionStub(), anchor=1, max_tokens=1):
        pass
    assert seen == [0]


# ----------------------------------------------------------- no-op contract


def test_one_round_accepts_round_prof_keyword_with_zero_default() -> None:
    """``_one_round`` exposes ``round_prof`` with default 0 and ignores it."""
    import inspect

    from exo.worker.engines.mlx.dsv41.rounds import _one_round

    sig = inspect.signature(_one_round)
    assert "round_prof" in sig.parameters
    assert sig.parameters["round_prof"].default == 0

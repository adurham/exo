# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The branch-only ``DSV41_SPEC_GAMMA`` override: unset == production, set == arm.

WHY THIS FILE EXISTS. Round Q2-Gamma re-prices the speculative draft depth by
running the SAME binary under different ``gamma`` values, flipped by an env var
plus a relaunch (there is deliberately no request-level surface). The
load-bearing property is that the UNSET path is byte-identical to production:
the whole experiment is void if merely carrying the new code changes the default
arm (gamma 3) or adds log noise the log-parsing harness would trip on. These
tests pin exactly that, plus the hard-error contract -- a typo'd arm must fail
the boot, never silently run the default -- and that gamma 5 (the top of the
supported set) is accepted.

The engine is built over the shared CPU-only stub body (no checkpoint, no GPU)
exactly as ``test_dsv41_prefill_liveness`` does, so ``__post_init__`` -- the
point under test -- runs in full.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.dsv41 import engine as eng_mod
from exo.worker.engines.mlx.dsv41.errors import Dsv41ConfigError
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
from exo.worker.engines.mlx.dsv41.rounds import _spec_policy
from exo.worker.engines.mlx.dsv41.tests._stub_harness import StubBody

ENV = eng_mod.DSV41_SPEC_GAMMA_ENV


class _Sender:
    def __init__(self) -> None:
        self.events: list[object] = []

    def send(self, item: object) -> None:
        self.events.append(item)


class _Receiver:
    def collect(self) -> list[object]:
        return []


def _make_engine() -> eng_mod.Dsv41Engine:
    """A real ``Dsv41Engine`` over the stub body (no checkpoint, no GPU)."""
    body = StubBody()
    loaded = Dsv41Loaded(
        model=body, tokenizer=None, args=body.args, model_path=None,  # type: ignore[arg-type]
        built_layers=[0], full_stack=False, rank=0, world=1, load_seconds=0.0,
        head=None,
    )
    return eng_mod.Dsv41Engine(
        loaded=loaded, model_id=ModelId("x/y"), group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=_Sender(),  # type: ignore[arg-type]
        device_rank=0, speculative=False, prefill_chunk_size=4, max_kv_tokens=256,
    )


def test_supported_set_is_exactly_2_3_4_5() -> None:
    """Pin the derived set; changing it must be a deliberate, reviewed edit."""
    assert eng_mod._SPEC_GAMMA_SUPPORTED == (2, 3, 4, 5)


def test_default_gamma_field_is_three() -> None:
    """The dataclass default the unset path falls back to is 3."""
    fields = {f.name: f for f in dataclasses.fields(eng_mod.Dsv41Engine)}
    assert fields["gamma"].default == 3


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_empty_keeps_the_production_default(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    """UNSET == today: gamma 3, and the value flows to the loop's policy start."""
    if raw is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, raw)
    # The parser itself is a pure passthrough of the default when unset.
    assert eng_mod._spec_gamma_from_env(3) == 3
    engine = _make_engine()
    assert engine.gamma == 3
    # ...and the effective value reaches ``_spec_policy`` (GammaPolicy(start=)).
    assert _spec_policy(engine.gamma).g == 3


def test_unset_emits_no_override_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """UNSET must add no log noise beyond what already exists (req. 2)."""
    monkeypatch.delenv(ENV, raising=False)
    with caplog.at_level(logging.INFO):
        engine = _make_engine()
    assert engine.gamma == 3
    assert "spec gamma override" not in caplog.text


@pytest.mark.parametrize("value,expected", [("2", 2), ("3", 3), ("4", 4), ("5", 5)])
def test_valid_override_sets_the_effective_gamma(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    """Every supported value is accepted and lands on the engine + its policy."""
    monkeypatch.setenv(ENV, value)
    assert eng_mod._spec_gamma_from_env(3) == expected
    engine = _make_engine()
    assert engine.gamma == expected
    assert _spec_policy(engine.gamma).g == expected


def test_gamma_five_is_structurally_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """gamma 5 -> 6 verify rows, which VERIFY_MS covers (key 6). Accepted, not refused."""
    assert 5 in eng_mod._SPEC_GAMMA_SUPPORTED
    monkeypatch.setenv(ENV, "5")
    assert eng_mod._spec_gamma_from_env(3) == 5
    assert _make_engine().gamma == 5


@pytest.mark.parametrize("value", ["1", "0", "-1", "6", "7", "abc", "3.0", "4x"])
def test_out_of_set_or_non_integer_is_a_hard_error(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """HARD ERROR: a typo'd arm fails the boot; it is NEVER a silent fallback to 3."""
    monkeypatch.setenv(ENV, value)
    with pytest.raises(Dsv41ConfigError):
        eng_mod._spec_gamma_from_env(3)
    # ...and the engine construction path raises too, rather than producing an
    # engine that would quietly run gamma 3 (the poisoning failure mode).
    with pytest.raises(Dsv41ConfigError):
        _make_engine()

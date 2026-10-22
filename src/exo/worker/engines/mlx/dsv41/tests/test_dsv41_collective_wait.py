"""Tests for the per-round all_sum wait instrument (EXO_DSV41_ALLSUM_WAIT).

The instrument is a diagnostic, but its OFF path must be byte-identical (the
model keeps its original collective and nothing is evaluated on the host), its
PREFILL path must stay asynchronous (only an open decode round is timed), and
its ON path must attribute the wait to the right round. All three are asserted
here without a GPU, a model or a distributed group: the target "module" is a
stand-in with one ``all_sum`` attribute, exactly the shape the patch relies on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from exo.worker.engines.mlx.dsv41 import collective_wait as cw


class _Clock:
    """A perf_counter with a scripted sequence of readings."""

    def __init__(self, readings: list[float]) -> None:
        self._readings = readings

    def perf_counter(self) -> float:
        return self._readings.pop(0)


def _fake_module(result: Any) -> tuple[Any, list[tuple[Any, Any]]]:
    """A stand-in for ``mlx_lm.models.deepseek_v41.collective``."""
    calls: list[tuple[Any, Any]] = []

    def all_sum(x: Any, group: Any = None) -> Any:
        calls.append((x, group))
        return result

    return SimpleNamespace(all_sum=all_sum), calls


def test_the_gate_is_off_unless_the_env_var_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in (None, "", "0"):
        if value is None:
            monkeypatch.delenv(cw.ENV, raising=False)
        else:
            monkeypatch.setenv(cw.ENV, value)
        assert cw.enabled() is False
    monkeypatch.setenv(cw.ENV, "1")
    assert cw.enabled() is True


def test_install_is_a_no_op_when_the_gate_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFF: the model keeps its own collective and nothing is host-evaluated."""
    cw.uninstall()
    monkeypatch.delenv(cw.ENV, raising=False)
    module, calls = _fake_module(object())
    original = module.all_sum

    assert cw.install(module) is False
    assert module.all_sum is original, "OFF must not touch the model's collective"

    module.all_sum(1, group=2)
    assert calls == [(1, 2)]


def test_outside_a_round_the_collective_stays_asynchronous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ON but no open round (prefill): pass through, never host-evaluate."""
    cw.uninstall()
    monkeypatch.setenv(cw.ENV, "1")
    evaluated: list[Any] = []
    monkeypatch.setattr(cw.mx, "eval", lambda y: evaluated.append(y))
    module, calls = _fake_module("result")

    assert cw.install(module) is True
    cw.reset_rounds()
    assert module.all_sum("x", group="g") == "result"
    assert calls == [("x", "g")]
    assert evaluated == [], "prefill collectives must not be host-synced"
    assert cw.stats() == (0, 0.0, 0.0)

    cw.uninstall()


def test_install_times_the_host_wait_and_emit_round_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ON inside a round: each collective is evaluated and its wait is counted."""
    cw.uninstall()
    monkeypatch.setenv(cw.ENV, "1")
    evaluated: list[Any] = []
    monkeypatch.setattr(cw.mx, "eval", lambda y: evaluated.append(y))
    module, calls = _fake_module("result")

    assert cw.install(module) is True
    monkeypatch.setattr(cw, "time", _Clock([0.0, 0.010, 0.020, 0.030, 0.040]))
    cw.begin_round()

    assert module.all_sum("x", group="g") == "result"
    assert module.all_sum("x", group="g") == "result"
    assert calls == [("x", "g"), ("x", "g")]
    assert evaluated == ["result", "result"], "ON: the wait is measured by evaling"
    assert cw.stats() == (2, pytest.approx(20.0), pytest.approx(10.0))

    line = cw.emit_round(1, 612.3, 3)
    assert line is not None
    assert line.startswith("[ALLSUMWAIT] n=1 rank=1 calls=2 ")
    assert "sum_ms=20.0 " in line
    assert "max_ms=10.0 " in line
    assert line.endswith("ms=612.3 committed=3")
    assert cw.stats() == (0, 0.0, 0.0), "emit_round starts a fresh round"
    assert module.all_sum("x", group="g") == "result"
    assert evaluated == ["result", "result"], "the closed window stops the timing"

    cw.uninstall()


def test_the_line_format_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    cw.uninstall()
    monkeypatch.setenv(cw.ENV, "1")
    monkeypatch.setattr(cw.mx, "eval", lambda y: None)
    module, _calls = _fake_module("r")
    assert cw.install(module) is True
    cw.reset_rounds()

    line = cw.emit_round(1, 612.3, 3)
    assert line == (
        "[ALLSUMWAIT] n=1 rank=1 calls=0 sum_ms=0.0 max_ms=0.0 ms=612.3 committed=3"
    )
    cw.uninstall()


def test_emit_round_is_silent_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cw.uninstall()
    monkeypatch.delenv(cw.ENV, raising=False)
    assert cw.emit_round(0, 100.0, 1) is None


def test_install_is_idempotent_and_uninstall_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cw.uninstall()
    monkeypatch.setenv(cw.ENV, "1")
    monkeypatch.setattr(cw.mx, "eval", lambda y: None)
    module, _calls = _fake_module("r")
    original = module.all_sum

    assert cw.install(module) is True
    wrapped = module.all_sum
    assert cw.install(module) is True
    assert module.all_sum is wrapped, "a second install must not double-wrap"

    cw.uninstall()
    assert module.all_sum is original


def test_round_numbering_restarts_per_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cw.uninstall()
    monkeypatch.setenv(cw.ENV, "1")
    monkeypatch.setattr(cw.mx, "eval", lambda y: None)
    module, _calls = _fake_module("r")
    assert cw.install(module) is True

    cw.reset_rounds()
    assert cw.emit_round(0, 1.0, 1) is not None
    assert cw.emit_round(0, 1.0, 1) is not None
    cw.reset_rounds()
    line = cw.emit_round(0, 1.0, 1)
    assert line is not None and line.startswith("[ALLSUMWAIT] n=1 ")
    cw.uninstall()


def test_rank_of_survives_a_stand_in_engine() -> None:
    assert cw.rank_of(SimpleNamespace(device_rank=1)) == 1
    assert cw.rank_of(SimpleNamespace(device_rank=None)) == -1
    assert cw.rank_of(SimpleNamespace()) == -1

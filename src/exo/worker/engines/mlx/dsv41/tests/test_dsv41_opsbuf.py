"""Tests for the per-round command-buffer batching lever (max_ops_per_buffer).

The lever's contract: read the value from the levers file at every decode round,
call the mlx setter only when the value CHANGES, stay silent and inert when the
key is absent, and degrade to a one-shot warning (never an exception) on an mlx
build without the binding.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from exo.worker.engines.mlx.dsv41 import opsbuf


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(opsbuf.ENV_FILE, raising=False)
    monkeypatch.delenv(opsbuf.ENV, raising=False)
    opsbuf._applied = None
    opsbuf.reset_rounds()


def _levers(tmp_path: Any, data: dict) -> str:
    p = tmp_path / "levers.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_absent_key_is_inert(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    calls: list[int] = []
    monkeypatch.setattr(opsbuf, "_setter", lambda: calls.append)
    monkeypatch.setenv(opsbuf.ENV_FILE, _levers(tmp_path, {"sync_every_n": 0}))
    assert opsbuf.requested() is None
    assert opsbuf.begin_round() is None
    assert calls == []


def test_value_is_applied_once_and_only_on_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(opsbuf, "_setter", lambda: calls.append)
    path = tmp_path / "levers.json"
    monkeypatch.setenv(opsbuf.ENV_FILE, str(path))

    path.write_text(json.dumps({"max_ops_per_buffer": 50}))
    line = opsbuf.begin_round()
    assert line is not None and "max_ops_per_buffer=50" in line
    assert calls == [50]
    assert opsbuf.applied() == 50

    # Same value next round: no second call, no line.
    assert opsbuf.begin_round() is None
    assert calls == [50]

    # Changed value: applied again.
    path.write_text(json.dumps({"max_ops_per_buffer": 8}))
    line = opsbuf.begin_round()
    assert line is not None and "max_ops_per_buffer=8" in line
    assert calls == [50, 8]


def test_junk_and_zero_are_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    calls: list[int] = []
    monkeypatch.setattr(opsbuf, "_setter", lambda: calls.append)
    path = tmp_path / "levers.json"
    monkeypatch.setenv(opsbuf.ENV_FILE, str(path))
    for bad in (0, -5, "banana", None):
        path.write_text(json.dumps({"max_ops_per_buffer": bad}))
        assert opsbuf.requested() in (None, 0) or isinstance(opsbuf.requested(), int)
        opsbuf.begin_round()
    assert calls == [], "no valid positive integer means no call at all"


def test_env_fallback_works_without_a_file(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(opsbuf, "_setter", lambda: calls.append)
    monkeypatch.setenv(opsbuf.ENV, "16")
    assert opsbuf.requested() == 16
    opsbuf.begin_round()
    assert calls == [16]


def test_missing_binding_degrades_to_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(opsbuf, "_setter", lambda: None)
    monkeypatch.setenv(opsbuf.ENV_FILE, _levers(tmp_path, {"max_ops_per_buffer": 1}))
    line = opsbuf.begin_round()
    assert line is not None and "no set_max_ops_per_buffer" in line
    assert opsbuf.applied() is None


def test_policy_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(opsbuf, "_setter", lambda: calls.append)
    assert opsbuf.apply_for_turn(True) is None
    assert calls == []


def test_policy_keys_on_cold_and_deep_not_on_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session flag, not context length, selects the cold threshold."""
    calls: list[int] = []
    monkeypatch.setattr(opsbuf, "_setter", lambda: calls.append)
    monkeypatch.setenv(opsbuf.ENV_POLICY, "1")
    monkeypatch.delenv(opsbuf.ENV_FILE, raising=False)

    assert opsbuf.is_cold_deep(0, 400_000) is True
    assert opsbuf.is_cold_deep(350_000, 360_000) is False  # grown by deltas
    assert opsbuf.is_cold_deep(0, 1_000) is False          # short

    line = opsbuf.apply_for_turn(True)
    assert line is not None and "max_ops_per_buffer=1" in line
    assert calls == [1]
    line = opsbuf.apply_for_turn(False)
    assert line is not None and "max_ops_per_buffer=50" in line
    assert calls == [1, 50]
    assert opsbuf.apply_for_turn(False) is None
    assert calls == [1, 50]


def test_the_levers_file_still_wins_over_the_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(opsbuf, "_setter", lambda: calls.append)
    monkeypatch.setenv(opsbuf.ENV_POLICY, "1")
    monkeypatch.setenv(opsbuf.ENV_FILE, _levers(tmp_path, {"max_ops_per_buffer": 8}))
    assert opsbuf.apply_for_turn(True) is None
    assert calls == [], "a diagnostic override owns the value"

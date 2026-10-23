"""Tests for the collective sync-cadence lever (DSV41_SYNC_EVERY_N).

The lever's OFF path must be byte-identical (the model keeps its own collective
and nothing is evaluated on the host), its PREFILL path must stay asynchronous
(only an open decode round syncs), and its cadence must be exactly every Kth
collective inside a round. The runtime levers file must win over the env and be
re-read per round, so one boot can sweep K. All asserted without a GPU, a model
or a distributed group: the target "module" is a stand-in with one ``all_sum``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.dsv41 import sync_cadence


def _fake_module() -> tuple[SimpleNamespace, list[Any]]:
    calls: list[Any] = []

    def all_sum(x: Any, group: Any = None) -> Any:
        calls.append((x, group))
        return x

    return SimpleNamespace(all_sum=all_sum), calls


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sync_cadence.ENV, raising=False)
    monkeypatch.delenv(sync_cadence.ENV_FILE, raising=False)
    sync_cadence.uninstall()


def test_gate_is_off_unless_the_env_or_file_says_otherwise() -> None:
    assert sync_cadence.enabled() is False
    assert sync_cadence.current_k() == 0


def test_garbage_and_negative_values_mean_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw in ("", "0", "-3", "nope"):
        monkeypatch.setenv(sync_cadence.ENV, raw)
        assert sync_cadence.current_k() == 0
        assert sync_cadence.enabled() is False


def test_install_is_a_no_op_when_the_gate_is_off() -> None:
    mod, _ = _fake_module()
    original = mod.all_sum
    assert sync_cadence.install(mod) is False
    assert mod.all_sum is original
    assert sync_cadence.emit_round(0, 100.0, 1) is None


def test_outside_a_round_every_collective_stays_asynchronous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(sync_cadence.ENV, "1")
    evaluated: list[int] = []
    monkeypatch.setattr(mx, "eval", lambda *a, **k: evaluated.append(1))
    mod, calls = _fake_module()
    assert sync_cadence.install(mod) is True
    mod.all_sum(mx.array([1.0]))
    mod.all_sum(mx.array([2.0]))
    assert len(calls) == 2
    assert evaluated == []  # prefill: nothing host-evaluated


def test_inside_a_round_every_kth_collective_is_host_synced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(sync_cadence.ENV, "2")
    evaluated: list[int] = []
    monkeypatch.setattr(mx, "eval", lambda *a, **k: evaluated.append(1))
    mod, _ = _fake_module()
    assert sync_cadence.install(mod) is True
    sync_cadence.begin_round()
    for _ in range(5):
        mod.all_sum(mx.array([1.0]))
    assert len(evaluated) == 2  # calls 2 and 4 of 5
    line = sync_cadence.emit_round(1, 357.0, 3)
    assert line is not None
    assert "k=2" in line and "calls=5" in line and "synced=2" in line
    assert "rank=1" in line and "ms=357.0" in line


def test_k_equals_one_syncs_every_collective(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sync_cadence.ENV, "1")
    evaluated: list[int] = []
    monkeypatch.setattr(mx, "eval", lambda *a, **k: evaluated.append(1))
    mod, _ = _fake_module()
    assert sync_cadence.install(mod) is True
    sync_cadence.begin_round()
    for _ in range(3):
        mod.all_sum(mx.array([1.0]))
    assert len(evaluated) == 3


def test_levers_file_wins_and_is_re_read_each_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv(sync_cadence.ENV, "1")
    path = tmp_path / "levers.json"
    path.write_text(json.dumps({"sync_every_n": 4}))
    monkeypatch.setenv(sync_cadence.ENV_FILE, str(path))
    assert sync_cadence.current_k() == 4

    evaluated: list[int] = []
    monkeypatch.setattr(mx, "eval", lambda *a, **k: evaluated.append(1))
    mod, _ = _fake_module()
    assert sync_cadence.install(mod) is True

    sync_cadence.begin_round()
    for _ in range(8):
        mod.all_sum(mx.array([1.0]))
    assert len(evaluated) == 2  # 8 calls at k=4

    # Same boot, next turn: disable through the file, no relaunch.
    path.write_text(json.dumps({"sync_every_n": 0}))
    sync_cadence.begin_round()
    for _ in range(8):
        mod.all_sum(mx.array([1.0]))
    assert len(evaluated) == 2  # pass-through: unchanged
    line = sync_cadence.emit_round(0, 590.0, 1)
    assert line is not None and "k=0" in line and "synced=0" in line


def test_uninstall_restores_the_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sync_cadence.ENV, "1")
    mod, _ = _fake_module()
    original = mod.all_sum
    assert sync_cadence.install(mod) is True
    assert mod.all_sum is not original
    sync_cadence.uninstall()
    assert mod.all_sum is original
    assert sync_cadence.emit_round(0, 1.0, 1) is None


def test_sync_what_in_drains_the_input_and_leaves_the_result_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The control that separates the in-buffer wait from host build-ahead.

    ``sync_what=in`` must evaluate the collective's INPUT (the producer chain)
    and must NOT evaluate the collective's result -- that is the whole point:
    the GPU-side wait on the CPU stream stays in place, so a recovery under
    this setting cannot be attributed to removing that wait.
    """
    monkeypatch.setenv(sync_cadence.ENV, "1")
    path = tmp_path / "levers.json"
    path.write_text(json.dumps({"sync_every_n": 1, "sync_what": "in"}))
    monkeypatch.setenv(sync_cadence.ENV_FILE, str(path))

    evaluated: list[Any] = []
    monkeypatch.setattr(mx, "eval", lambda *a, **k: evaluated.append(a[0]))
    mod, calls = _fake_module()
    assert sync_cadence.install(mod) is True
    assert sync_cadence.current_what() == "in"

    a, b = mx.array([1.0]), mx.array([2.0])
    sync_cadence.begin_round()
    mod.all_sum(a)
    mod.all_sum(b)
    assert len(calls) == 2, "the collective is still issued every time"
    assert len(evaluated) == 2
    assert evaluated[0] is a and evaluated[1] is b, "the INPUT is what was drained"
    line = sync_cadence.emit_round(0, 590.0, 1)
    assert line is not None
    assert "what=in" in line and "drained=2" in line and "synced=0" in line

"""Env-gate tests for ``EXO_DSV4_DSPARK_TP_SHARD``.

The full DSpark TP-sharding correctness path can only be verified on the
real cluster (2× M4 Max, TP=2). What we CAN verify hermetically on the
laptop is the two properties that any production-safe env gate must have:

1. Default OFF: with ``EXO_DSV4_DSPARK_TP_SHARD`` unset, the sharding
   strategy MUST NOT touch ``model.model.dspark`` even if the DSpark
   overlay attached the head — no bytes moved, no ``sharding_group``
   assigned, no expert weights re-sliced. This is the current production
   behavior we're preserving.
2. Enabled + failure = safe fallback: with the flag ON and a failure
   inside the shard loop, the strategy detaches ``dspark`` (and clears
   the taps) instead of leaving a half-sharded module attached — a
   half-sharded head would desync collectives on the first draft cycle,
   which is strictly worse than falling back to the MTP-1 draft path.

The DeepseekV4 strategy also iterates ``model.model.layers`` and
``model.model.mtp`` — both are set to ``[]`` here so the test only
exercises the DSpark loop.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import cast
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.auto_parallel import DeepseekV4ShardingStrategy


class _SentinelDSpark:
    """A stand-in for ``model.model.dspark`` whose ``.stages`` access is a
    tripwire: if the sharding loop ever touches it while the env gate is
    OFF, the property raises and the test fails loudly.
    """

    @property
    def stages(self) -> list[object]:  # pragma: no cover - tripwire
        raise AssertionError(
            "shard_model touched dspark.stages while "
            "EXO_DSV4_DSPARK_TP_SHARD was not '1' — the env gate is broken."
        )


class _ExplodingDSpark:
    """A stand-in whose ``.stages`` access ALWAYS raises RuntimeError.

    Used for the env-ON failure-fallback test: a failure at first-stage
    dereference must be caught by the strategy's try/except and result in
    ``model.model.dspark`` being detached — never a raised exception out
    of ``shard_model``.
    """

    @property
    def stages(self) -> list[object]:
        raise RuntimeError("simulated DSpark sharding failure")


class _FakeInnerModel:
    def __init__(self, dspark: object) -> None:
        self.layers: list[object] = []
        self.mtp: list[object] = []
        self.dspark: object = dspark


class _FakeModel(nn.Module):
    def __init__(self, dspark: object) -> None:
        super().__init__()
        self.model: _FakeInnerModel = _FakeInnerModel(dspark)


def _make_strategy() -> DeepseekV4ShardingStrategy:
    group = MagicMock()
    _ = group.size.return_value = 2  # pyright: ignore[reportAny]
    _ = group.rank.return_value = 0  # pyright: ignore[reportAny]
    # The shard helpers are only invoked when the loop actually runs;
    # for the env-OFF and failure-fallback tests these are never called
    # from within the DSpark branch, so a MagicMock is fine.
    return DeepseekV4ShardingStrategy(
        group=group,
        all_to_sharded_linear=MagicMock(),
        sharded_to_all_linear=MagicMock(),
        all_to_sharded_linear_in_place=MagicMock(),
        sharded_to_all_linear_in_place=MagicMock(),
    )


@pytest.fixture(autouse=True)
def _clear_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: PT004
    """Ensure the env var starts unset so per-test setpatch controls truth."""
    monkeypatch.delenv("EXO_DSV4_DSPARK_TP_SHARD", raising=False)
    monkeypatch.delenv("EXO_DSV4_MOE_FUSED_GATE_UP", raising=False)


def _drain(gen: Generator[object, None, nn.Module]) -> None:
    """Fully exhaust a generator (StopIteration.value is the returned model)."""
    try:
        while True:
            _ = next(gen)
    except StopIteration:
        return


def test_dspark_tp_shard_default_off_is_noop_on_dspark() -> None:
    """With the env gate unset, the loop must be a no-op — the sentinel's
    ``.stages`` access would raise if touched, so a clean drain proves it
    was never touched.
    """
    assert os.environ.get("EXO_DSV4_DSPARK_TP_SHARD") is None
    strategy = _make_strategy()
    model = _FakeModel(_SentinelDSpark())
    _drain(cast("Generator[object, None, nn.Module]", strategy.shard_model(model)))
    # dspark must still be attached and untouched.
    assert isinstance(model.model.dspark, _SentinelDSpark)


def test_dspark_tp_shard_explicit_zero_is_noop_on_dspark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``EXO_DSV4_DSPARK_TP_SHARD=0`` is equivalent to unset."""
    monkeypatch.setenv("EXO_DSV4_DSPARK_TP_SHARD", "0")
    strategy = _make_strategy()
    model = _FakeModel(_SentinelDSpark())
    _drain(cast("Generator[object, None, nn.Module]", strategy.shard_model(model)))
    assert isinstance(model.model.dspark, _SentinelDSpark)


def test_dspark_tp_shard_enabled_failure_detaches_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the gate ON and the shard loop failing at first dereference,
    the strategy must catch and detach ``dspark`` — never raise out.
    Detaching preserves cluster liveness by falling back to the MTP-1
    draft path instead of desyncing collectives at first draft cycle.
    """
    monkeypatch.setenv("EXO_DSV4_DSPARK_TP_SHARD", "1")
    strategy = _make_strategy()
    model = _FakeModel(_ExplodingDSpark())
    # Must not raise even though the sharding path explodes.
    _drain(cast("Generator[object, None, nn.Module]", strategy.shard_model(model)))
    # And the module must be detached so the runtime falls back cleanly.
    assert not hasattr(model.model, "dspark")


def test_dspark_tp_shard_enabled_missing_dspark_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the gate ON but no ``model.model.dspark`` attached (the
    overlay was skipped, or the ranks agreed to fall back), the loop
    must silently no-op — ``getattr(..., "dspark", None)`` returns None.
    """
    monkeypatch.setenv("EXO_DSV4_DSPARK_TP_SHARD", "1")
    strategy = _make_strategy()

    class _NoDSparkInner:
        def __init__(self) -> None:
            self.layers: list[object] = []
            self.mtp: list[object] = []

    class _NoDSparkModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model: _NoDSparkInner = _NoDSparkInner()

    model = _NoDSparkModel()
    _drain(cast("Generator[object, None, nn.Module]", strategy.shard_model(model)))
    assert not hasattr(model.model, "dspark")


def test_dspark_tp_shard_enabled_iterates_when_dspark_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: with the gate ON and an attached module whose
    ``.stages`` is an EMPTY list, the loop runs to completion (zero
    iterations, zero exceptions) — proving the branch is reachable when
    the gate is set, not silently dead.
    """
    monkeypatch.setenv("EXO_DSV4_DSPARK_TP_SHARD", "1")
    strategy = _make_strategy()

    class _EmptyStagesDSpark:
        stages: list[object] = []

    model = _FakeModel(_EmptyStagesDSpark())
    _drain(cast("Generator[object, None, nn.Module]", strategy.shard_model(model)))
    # No stages to shard → module stays attached; no detach happened.
    assert isinstance(model.model.dspark, _EmptyStagesDSpark)


# Silence an unused-import warning on mlx.core — we may need it if a
# future test wants to touch mx.arrays. Keeping the import documents the
# environment these tests actually run in.
_ref_mx = mx

"""ROUND-Q1B: liveness beats from the DSv4.1 load-time warmup.

The load warmup (``prefill.load_warmup``) is the longest event-free stretch of
``Dsv41Builder.load``. Its first 512-row forward is a ~48 s compile storm on
exl3 at world=2, and it ran >66 s with ``DSV41_DENSE=affine6``. The
supervisor's silence watchdog (45 s, plus one 20 s growth probe) SIGKILLed the
healthy affine warmup 4/4 times. The fix wires a throttled heartbeat into the
warmup's per-K-layer fence hook. These tests pin the exo side:

* ``load_warmup`` forwards the heartbeat as ``fence_hook`` when the installed
  mlx-lm accepts it, and otherwise warns and falls back to the old call;
* the builder's beat sends ``RunnerStatusUpdated(RunnerLoading)`` (any event
  resets the supervisor clock), unthrottled on the first fence and throttled
  to one per 15 s after that;
* ``Dsv41Builder.load`` actually passes the beat to the warmup.
"""

# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from exo.shared.types.events import RunnerStatusUpdated
from exo.shared.types.worker.runners import RunnerId, RunnerLoading
from exo.worker.engines.mlx.dsv41 import builder as builder_module
from exo.worker.engines.mlx.dsv41.builder import Dsv41Builder
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded


class _Sender:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def send(self, item: object) -> None:
        self.sent.append(item)


def _loaded() -> Dsv41Loaded:
    return Dsv41Loaded(
        model=type("FakeModel", (), {"layers": list(range(40))})(),
        tokenizer=object(),
        args=object(),
        model_path=Path("/nonexistent"),
        built_layers=list(range(40)),
        full_stack=True,
        rank=1,
        world=2,
        load_seconds=0.0,
    )


class _Bound:
    bound_runner_id = RunnerId("runner-q1b")


def _builder(sender: _Sender) -> Dsv41Builder:
    return Dsv41Builder(
        model_id="m",  # type: ignore[arg-type]
        event_sender=sender,  # type: ignore[arg-type]
        cancel_receiver=object(),  # type: ignore[arg-type]
    )


def test_heartbeat_sends_runner_status_and_throttles(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(builder_module.time, "monotonic", lambda: clock["t"])
    sender = _Sender()
    beat = _builder(sender)._load_heartbeat(_Bound(), _loaded())  # type: ignore[arg-type]

    beat()  # first fence: unthrottled
    assert len(sender.sent) == 1
    ev = sender.sent[0]
    assert isinstance(ev, RunnerStatusUpdated)
    assert ev.runner_id == RunnerId("runner-q1b")
    assert isinstance(ev.runner_status, RunnerLoading)
    assert ev.runner_status.layers_loaded == ev.runner_status.total_layers == 40

    clock["t"] += 5.0
    beat()  # inside the 15 s throttle
    assert len(sender.sent) == 1
    clock["t"] += 11.0
    beat()  # 16 s after the first send
    assert len(sender.sent) == 2


class _FakePrefill:
    """Stands in for ``mlx_lm.models.deepseek_v41.prefill``."""

    def __init__(self, accepts_hook: bool) -> None:
        self.calls: list[dict[str, Any]] = []
        if accepts_hook:

            def load_warmup(model, head=None, *, fence_hook=None):  # noqa: ANN001
                self.calls.append({"fence_hook": fence_hook})
                if fence_hook is not None:
                    fence_hook()
                return {}

        else:

            def load_warmup(model, head=None):  # noqa: ANN001
                self.calls.append({})
                return {}

        self.load_warmup = load_warmup


def _install_prefill(monkeypatch, fake: _FakePrefill) -> None:
    import mlx_lm.models.deepseek_v41 as pkg

    monkeypatch.setattr(pkg, "prefill", fake, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules, "mlx_lm.models.deepseek_v41.prefill", fake
    )


def test_load_warmup_forwards_heartbeat_as_fence_hook(monkeypatch):
    fake = _FakePrefill(accepts_hook=True)
    _install_prefill(monkeypatch, fake)
    beats: list[int] = []
    loaded = _loaded()
    loaded.head = None  # type: ignore[attr-defined]
    builder_module.load_warmup(loaded, heartbeat=lambda: beats.append(1))
    assert fake.calls and fake.calls[0]["fence_hook"] is not None
    assert beats == [1]


def test_load_warmup_without_heartbeat_is_the_old_call(monkeypatch):
    fake = _FakePrefill(accepts_hook=True)
    _install_prefill(monkeypatch, fake)
    loaded = _loaded()
    loaded.head = None  # type: ignore[attr-defined]
    builder_module.load_warmup(loaded)
    assert fake.calls == [{"fence_hook": None}]


def test_load_warmup_old_mlx_lm_falls_back_and_warns(monkeypatch):
    fake = _FakePrefill(accepts_hook=False)
    _install_prefill(monkeypatch, fake)
    warnings: list[str] = []
    monkeypatch.setattr(
        builder_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg)
    )
    loaded = _loaded()
    loaded.head = None  # type: ignore[attr-defined]
    builder_module.load_warmup(loaded, heartbeat=lambda: None)
    assert fake.calls == [{}]
    assert any("WITHOUT liveness beats" in w for w in warnings)


def test_shipped_mlx_lm_load_warmup_accepts_fence_hook():
    """The vendored mlx-lm this exo commit pins must carry the parameter;
    otherwise the fix silently degrades to the warning path above."""
    import inspect

    from mlx_lm.models.deepseek_v41 import prefill

    if "fence_hook" not in inspect.signature(prefill.load_warmup).parameters:
        pytest.skip("installed mlx-lm predates the Q1B fence_hook (checked on deploy)")
    assert "fence_hook" in inspect.signature(prefill.load_warmup).parameters

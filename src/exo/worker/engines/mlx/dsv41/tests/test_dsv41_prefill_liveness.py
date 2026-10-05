"""CPU-only wiring tests for the Fix A fence-hook prefill liveness path.

The mlx-lm side (a sibling change) calls ``model._fence_hook`` after every
model-level ``mx.eval`` fence of a multi-row forward. This module pins the
EXO side of the contract, which is pure attribute plumbing and needs no GPU:

* ``engine_prefill`` installs ``model._fence_hook`` for the loop and restores
  the previous value in a ``finally`` (including after a forward raises), warns
  once when the resolved fence is disabled, and surfaces a latched
  ``_fence_hook_failed`` at CRITICAL;
* the engine's session store carries the hook through
  ``Dsv41Sessions`` -> ``Conversation`` -> (``functools.partial``) ->
  ``engine_prefill``, driven end-to-end over the shared stub body;
* ``Dsv41Engine._session_fence_heartbeat`` delegates to the throttled
  ``prefill_heartbeat`` and warns when the gap between fence callbacks exceeds
  the 30 s spacing watchdog (measured, never asserted).

The stub body never *calls* the hook (only the real ``Model.__call__`` does);
these tests assert the attribute is set/restored/forwarded, which is the whole
of the exo-side responsibility.
"""

# This is a WHITE-BOX test: it deliberately pokes the driver-engine's private
# attributes (``_fence_hook``, ``_fence_hook_failed``, ``_session_fence_heartbeat``,
# ``_last_fence_heartbeat_monotonic``), inspects the ``functools.partial`` a
# restored conversation carries, and uses duck-typed ``Any`` doubles that stand
# in for the real MLX model. None of that is a real defect, so these rules are
# disabled for the file rather than ignored line-by-line.
# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import engine as eng_mod
from exo.worker.engines.mlx.dsv41 import session as s_

# The shared stub body over the fork's REAL ModelCache: the same harness the
# session + park suites use, so the end-to-end wiring check reproduces their
# construction path rather than a bespoke one.
from ._stub_harness import StubBody

# --------------------------------------------------------------------------
# test doubles
# --------------------------------------------------------------------------


def _noop_hook() -> None:
    """A fence hook that does nothing: identity is what the tests assert."""


def _other_hook() -> None:
    """A distinct no-op, so a restore test can tell ``prev`` from replacement."""


class _FakeLogger:
    """Records (level, message) pairs instead of touching the loguru singleton."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def _record(self, level: str, message: str) -> None:
        self.messages.append((level, message))

    def info(self, message: str, *_args: object, **_kwargs: object) -> None:
        self._record("INFO", message)

    def warning(self, message: str, *_args: object, **_kwargs: object) -> None:
        self._record("WARNING", message)

    def critical(self, message: str, *_args: object, **_kwargs: object) -> None:
        self._record("CRITICAL", message)

    def of(self, level: str) -> list[str]:
        return [m for lvl, m in self.messages if lvl == level]


@contextmanager
def _capture(module: Any) -> Iterator[_FakeLogger]:
    """Swap ``module.logger`` for a recorder and restore it afterwards.

    ``module`` is deliberately ``Any``: the two modules under test
    (``session`` and ``engine``) are swapped in the same way, and neither
    exports a common logger-holder type.
    """
    cap = _FakeLogger()
    original = module.logger
    module.logger = cap
    try:
        yield cap
    finally:
        module.logger = original


class _OffsetCache:
    """Minimal cache stub: ``engine_prefill`` only reads/writes ``offset``."""

    def __init__(self) -> None:
        self.offset = 0


class _HookModel:
    """Fake body that records the ``_fence_hook`` it sees at each forward.

    Mirrors the real ``Model.__call__`` surface the driver uses and advances
    ``cache.offset`` like the body does, so the chunk loop terminates.
    ``raise_on_call`` injects a forward failure to exercise the ``finally``.

    The two driver-owned attributes are read back through *public* probes
    (``fence_now``/``hook_now``/``hook_failed``) so the test never touches the
    private names directly.
    """

    def __init__(
        self,
        fence_every: int = 0,
        hook: Callable[[], None] | None = None,
        raise_on_call: int | None = None,
    ) -> None:
        self._fence_every = int(fence_every)
        if hook is not None:
            self._fence_hook = hook
        self.raise_on_call = raise_on_call
        self.calls = 0
        self.hook_at_call: list[Callable[[], None] | None] = []
        #: What ``Model.__call__`` latches when the hook raises (production).
        self._fence_hook_failed = False

    def fence_now(self) -> int:
        return int(self._fence_every)

    def hook_now(self) -> Callable[[], None] | None:
        return getattr(self, "_fence_hook", None)

    def set_hook_failed(self, value: bool) -> None:
        self._fence_hook_failed = value

    def hook_failed(self) -> bool:
        return bool(self._fence_hook_failed)

    def __call__(self, ids: mx.array, cache: _OffsetCache, **_rest: Any) -> mx.array:
        self.calls += 1
        self.hook_at_call.append(self.hook_now())
        if self.raise_on_call is not None and self.calls == self.raise_on_call:
            raise RuntimeError("injected forward failure")
        cache.offset += int(ids.shape[-1])
        return mx.array([[0.0]])


class _HookRecordingBody(StubBody):
    """StubBody that records the fence hook it runs under (wiring check)."""

    def __init__(self) -> None:
        super().__init__()
        self._fence_every = 0
        self.hooks_seen: list[Callable[[], None] | None] = []

    def hook_now(self) -> Callable[[], None] | None:
        return getattr(self, "_fence_hook", None)

    def __call__(
        self,
        ids: Any,
        cache: Any,
        last_logit_only: bool = False,
        return_taps: bool = False,
        argmax: bool = False,
    ) -> Any:
        self.hooks_seen.append(self.hook_now())
        return super().__call__(
            ids, cache, last_logit_only, return_taps, argmax
        )


def _make_engine(device_rank: int = 0) -> eng_mod.Dsv41Engine:
    """A real ``Dsv41Engine`` over the stub body (no checkpoint, no GPU)."""
    from exo.shared.types.common import ModelId
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded

    body = StubBody()
    loaded = Dsv41Loaded(
        model=body, tokenizer=None, args=body.args, model_path=None,  # type: ignore[arg-type]
        built_layers=[0], full_stack=False, rank=0, world=1, load_seconds=0.0,
        head=None,
    )

    class _Sender:
        def __init__(self) -> None:
            self.events: list[object] = []

        def send(self, item: object) -> None:
            self.events.append(item)

    class _Receiver:
        def collect(self) -> list[object]:
            return []

    return eng_mod.Dsv41Engine(
        loaded=loaded, model_id=ModelId("x/y"), group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=_Sender(),  # type: ignore[arg-type]
        device_rank=device_rank, speculative=False, prefill_chunk_size=4,
        max_kv_tokens=256,
    )


# --------------------------------------------------------------------------
# (1) engine_prefill: set / restore / warn / CRITICAL
# --------------------------------------------------------------------------


def test_engine_prefill_sets_and_restores_the_fence_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    model = _HookModel()
    s_.engine_prefill(
        model, np.arange(10), _OffsetCache(), chunk=4, fence_hook=_noop_hook
    )

    assert model.calls, "no forward ran"
    assert all(h is _noop_hook for h in model.hook_at_call)
    # absent before -> restored to None (the model never had the attribute)
    assert model.hook_now() is None
    assert model.fence_now() == 0


def test_engine_prefill_restores_a_previous_fence_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    model = _HookModel(hook=_noop_hook)
    s_.engine_prefill(
        model, np.arange(10), _OffsetCache(), chunk=4, fence_hook=_other_hook
    )
    assert model.hook_now() is _noop_hook, "the prior hook must be restored"


def test_engine_prefill_restores_the_fence_hook_after_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forward that raises still restores BOTH the hook and the fence."""
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    model = _HookModel(fence_every=7, hook=_noop_hook, raise_on_call=2)
    with pytest.raises(RuntimeError, match="injected forward failure"):
        s_.engine_prefill(
            model, np.arange(10), _OffsetCache(), chunk=4, fence_hook=_other_hook
        )
    assert model.hook_now() is _noop_hook
    assert model.fence_now() == 7


def test_engine_prefill_controls_line_reports_fence_hook_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    model = _HookModel()
    with _capture(s_) as cap:
        s_.engine_prefill(
            model, np.arange(4), _OffsetCache(), chunk=4, fence_hook=_noop_hook
        )
    assert any("fence_hook=on" in m for m in cap.of("INFO")), cap.messages


def test_engine_prefill_controls_line_reports_fence_hook_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    model = _HookModel()
    with _capture(s_) as cap:
        s_.engine_prefill(model, np.arange(4), _OffsetCache(), chunk=4)
    assert any("fence_hook=off" in m for m in cap.of("INFO")), cap.messages


def test_engine_prefill_warns_when_the_fence_disables_the_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fence_every=0 means the hook can never fire: warn once, loudly."""
    monkeypatch.setenv(s_.PREFILL_FENCE_EVERY_ENV, "0")
    model = _HookModel()
    with _capture(s_) as cap:
        s_.engine_prefill(
            model, np.arange(4), _OffsetCache(), chunk=4, fence_hook=_noop_hook
        )
    assert any("liveness hook unavailable" in m for m in cap.of("WARNING")), (
        cap.messages
    )
    # ...and the controls line is honest about it.
    assert any("fence_hook=off" in m for m in cap.of("INFO")), cap.messages


def test_engine_prefill_does_not_warn_when_no_hook_was_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(s_.PREFILL_FENCE_EVERY_ENV, "0")
    model = _HookModel()
    with _capture(s_) as cap:
        s_.engine_prefill(model, np.arange(4), _OffsetCache(), chunk=4)
    assert not cap.of("WARNING"), cap.messages


def test_engine_prefill_surfaces_a_latched_hook_failure_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model-side latch (``_fence_hook_failed``) is reported and reset."""
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    model = _HookModel()
    model.set_hook_failed(True)  # what Model.__call__ latches on a hook bug
    with _capture(s_) as cap:
        s_.engine_prefill(
            model, np.arange(4), _OffsetCache(), chunk=4, fence_hook=_noop_hook
        )
    assert cap.of("CRITICAL"), cap.messages
    assert model.hook_failed() is False, "the flag must be reset"


def test_engine_prefill_no_critical_without_a_latched_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    model = _HookModel()
    with _capture(s_) as cap:
        s_.engine_prefill(
            model, np.arange(4), _OffsetCache(), chunk=4, fence_hook=_noop_hook
        )
    assert not cap.of("CRITICAL"), cap.messages


# --------------------------------------------------------------------------
# (2) the session store carries the hook through to engine_prefill
# --------------------------------------------------------------------------


def test_session_passes_the_fence_hook_through_to_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dsv41Sessions(fence_heartbeat=...) reaches the driver's forward loop."""
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    body = _HookRecordingBody()
    store = s_.Dsv41Sessions(
        body, None, max_seq_len=256, chunk=4, fence_heartbeat=_noop_hook
    )
    conv = store.get(list(range(10, 42)), "conv")
    conv.prefill(np.arange(10, 42, dtype=np.int64))

    assert body.hooks_seen, "prefill ran no forward"
    assert all(h is _noop_hook for h in body.hooks_seen), body.hooks_seen
    assert body.hook_now() is None, "restored after the turn"


def test_session_without_a_hook_leaves_the_model_unhooked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(s_.PREFILL_FENCE_EVERY_ENV, raising=False)
    body = _HookRecordingBody()
    store = s_.Dsv41Sessions(body, None, max_seq_len=256, chunk=4)
    conv = store.get(list(range(10, 42)), "conv")
    conv.prefill(np.arange(10, 42, dtype=np.int64))

    assert body.hooks_seen
    assert all(h is None for h in body.hooks_seen)


# --------------------------------------------------------------------------
# (3) Dsv41Engine._session_fence_heartbeat: delegate + spacing watchdog
# --------------------------------------------------------------------------


def test_session_fence_heartbeat_delegates_to_prefill_heartbeat() -> None:
    engine = _make_engine()
    hits: list[int] = []
    engine.heartbeat = lambda: hits.append(1)

    engine._session_fence_heartbeat()

    assert hits == [1], "the fence hook must re-emit the runner status"
    assert engine._last_fence_heartbeat_monotonic > 0.0


def test_session_fence_heartbeat_spacing_warning_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A >30 s gap between fence callbacks warns; a small one does not."""
    engine = _make_engine()
    engine.heartbeat = _noop_hook
    clock = {"t": 1000.0}
    # Patch the engine module's clock only (base.prefill_heartbeat keeps its own
    # real-time throttle; it is irrelevant to the spacing assertion).
    monkeypatch.setattr(eng_mod.time, "monotonic", lambda: clock["t"])

    with _capture(eng_mod) as cap:
        engine._session_fence_heartbeat()          # first call: never warns
        clock["t"] = 1020.0
        engine._session_fence_heartbeat()          # 20 s: under the 30 s bound
        assert not cap.of("WARNING"), cap.messages
        clock["t"] = 1060.0
        engine._session_fence_heartbeat()          # 40 s: over the bound

    warns = cap.of("WARNING")
    assert warns and "fence heartbeat spacing" in warns[-1], warns


def test_session_fence_heartbeat_spacing_boundary_is_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly 30.0 s is not a violation (the check is strictly greater)."""
    engine = _make_engine()
    engine.heartbeat = _noop_hook
    clock = {"t": 500.0}
    monkeypatch.setattr(eng_mod.time, "monotonic", lambda: clock["t"])

    with _capture(eng_mod) as cap:
        engine._session_fence_heartbeat()
        clock["t"] = 530.0
        engine._session_fence_heartbeat()

    assert not cap.of("WARNING"), cap.messages


def test_fence_heartbeat_constant_is_thirty_seconds() -> None:
    assert eng_mod.FENCE_HEARTBEAT_MAX_SPACING_SECONDS == 30.0


# --------------------------------------------------------------------------
# the engine wires the hook into its session store
# --------------------------------------------------------------------------


def test_engine_builds_sessions_with_the_fence_heartbeat() -> None:
    """__post_init__ hands ``_session_fence_heartbeat`` to the session store."""
    engine = _make_engine()
    assert engine._sessions._fence_heartbeat == engine._session_fence_heartbeat


# --------------------------------------------------------------------------
# a RESTORED parked conversation must carry a fresh hook too
# --------------------------------------------------------------------------


def test_restored_parked_conversation_carries_the_fence_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "Any",
) -> None:
    """The park store's rebuild path gets the hook (else a restored delta
    prefill would be silent on the fence and could be killed as hung).

    Drives the REAL ``Dsv41Sessions._park_store()`` build path (the one the
    production store is built from) rather than injecting a store, so this
    pins the wiring a restored conversation actually sees.
    """
    monkeypatch.setenv("EXO_DSV41_PARK_DIR", str(tmp_path))
    monkeypatch.setenv("EXO_DSV41_PARK_MIN_TOKENS", "0")
    store = s_.Dsv41Sessions(
        StubBody(initial_capacity=4), None, max_seq_len=256, chunk=4,
        long_chunk=4, long_threshold=10**9, max_sessions=2,
        fence_heartbeat=_noop_hook,
    )
    for name in ("A", "B", "C"):
        conv = store.get(np.arange(10, 40, dtype=np.int64).tolist(), name)
        conv.prefill(np.arange(10, 40, dtype=np.int64))
        conv.finish()
    # A is the oldest -> parked to SSD when C evicts it, not hard-discarded.
    assert store.stats["parked"] == 1, store.stats
    assert store.stats["evicted"] == 1, store.stats

    restored = store.get(np.arange(10, 40, dtype=np.int64).tolist(), "A")
    assert store.stats["restored"] == 1, store.stats
    # The hook the restored conversation will install on its next prefill.
    kwargs = dict(restored.cache._prefill.keywords)
    assert kwargs.get("fence_hook") is _noop_hook, kwargs



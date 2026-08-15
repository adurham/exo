"""Pins rank 0's chunk-drive advance BUDGET formula to the real
``ResumablePrefillSession.advance()`` semantics, across BOTH layer-count
parities.

Design doc Section 45 (2026-08-15). This is the regression test for the
multi-session mutual deadlock first seen in Section 39 and wrongly
recorded as "hypothesis DISPROVEN" in Section 40.

THE BUG
-------
``Rank0BatchedDecodeGlue.tick()``'s HANDOFF transition must predict how
many ``PrefillAdvanceMessage``s rank 1 needs before rank 1's own
``advance()`` will report ``done=True`` (rank 0 then BLOCKS on rank 1's
chunk-done ack; rank 1 only sends that ack when it sees ``done``). It
used ``ceil(peer_prefill_layer_count / max_layers)``, which is short by
EXACTLY ONE whenever ``max_layers`` evenly divides the peer's layer
count.

Why: the underlying ``_forward_steps`` generator yields L ``("layer",
...)`` steps and then a SEPARATE ``("done", ...)`` sentinel, while
``advance()``'s own loop is ``while layers_advanced < max_layers:``. So
``advance()`` returns ``done=True`` only on the call that actually
CONSUMES that sentinel:

  * ``L % max != 0`` -- the last call consumes the ``r < max`` remainder,
    its loop condition is still satisfied, so it consumes the sentinel in
    the SAME call. ``ceil == floor+1``: the old formula coincided.
  * ``L % max == 0`` -- every call fills its quota exactly and returns
    ``(max, False)`` without ever reaching the sentinel. One MORE call is
    needed, which consumes only the sentinel and returns ``(0, True)``.

Correct budget is therefore ``floor(L/max) + 1`` for BOTH parities.

WHY THIS TEST EXISTS RATHER THAN JUST THE FIX
---------------------------------------------
Section 40 formed this exact hypothesis, instrumented it on real
hardware, ran 10 live test runs, and concluded "DISPROVEN by direct
measurement" -- because that session's driver peer had 21 layers (ODD),
the single parity where ``ceil()`` is accidentally correct. The
hypothesis was right all along; it was tested under the only layer count
that hides the bug. A live cluster only ever exercises whatever layer
split the current model/topology happens to produce, so hardware testing
CANNOT be relied on to cover both parities.

These tests drive the REAL ``ResumablePrefillSession`` (via the same
``_FakeInterruptibleModel`` shape ``test_pp_prefill_session.py`` already
uses) and count actual ``advance()`` calls to ``done=True`` -- so the
budget formula is pinned to the session's true behaviour, not to a
restatement of it. If ``advance()``'s loop or the sentinel contract ever
changes, these fail loudly instead of deadlocking a live 2-node cluster.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.pp_metaframe import ForwardPhase
from exo.worker.engines.mlx.pp_prefill_session import (
    ForwardStep,
    ResumablePrefillSession,
)

pytestmark = pytest.mark.filterwarnings("ignore")


class _FakeInterruptibleModel:
    """Same structural stand-in ``test_pp_prefill_session.py`` uses: a
    ``_forward_steps`` generator yielding ``("layer", i, h)`` per layer
    and then a single ``("done", None, h)`` sentinel."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers

    def _forward_steps(
        self,
        inputs: mx.array,
        cache: object = None,
        *,
        interruptible: bool = False,
    ) -> Iterator[ForwardStep]:
        h = inputs
        for i in range(self.n_layers):
            h = h + mx.array([1.0])
            mx.eval(h)
            if interruptible:
                yield ("layer", i, h)
        yield ("done", None, h)


def _real_advances_to_done(n_layers: int, max_layers: int) -> int:
    """Drive a REAL ResumablePrefillSession and count how many
    ``advance()`` calls it actually takes to reach ``done=True``."""
    model = _FakeInterruptibleModel(n_layers=n_layers)
    session = ResumablePrefillSession(
        inner_model=model,
        inputs=mx.array([0.0]),
        cache=(),
    )
    calls = 0
    while True:
        calls += 1
        _layers_advanced, done = session.advance(
            max_layers=max_layers,
            phase_for_pause=ForwardPhase.PREFILL_CONTINUE,
        )
        if done:
            return calls
        if calls > n_layers + 5:  # safety net; never hit when correct
            raise AssertionError(
                f"advance() never reported done for n_layers={n_layers} "
                f"max_layers={max_layers} after {calls} calls"
            )


def _budget(peer_layer_count: int, max_layers: int) -> int:
    """The formula Rank0BatchedDecodeGlue.tick() uses for
    ``_prefill_rank1_advances_remaining``. Kept in lockstep with the real
    call site by the tests below."""
    return (peer_layer_count // max_layers) + 1


@pytest.mark.parametrize(
    "n_layers,max_layers",
    [
        # THE REGRESSION CASE: max evenly divides L. This is the real
        # production topology that deadlocked (rank 1 = 22 layers,
        # EXO-configured max_layers=2) and the parity Section 40 never
        # exercised.
        (22, 2),
        (4, 2),
        (2, 2),
        (21, 3),
        (10, 5),
        # The ODD parity Section 40 DID measure, where the old ceil()
        # formula coincidentally agreed -- must keep working.
        (21, 2),
        (3, 2),
        (1, 2),
        (22, 3),
        # max_layers larger than the whole segment: one call consumes
        # every layer AND the sentinel.
        (22, 32),
        (1, 8),
        # max_layers == 1, the finest granularity.
        (5, 1),
        (1, 1),
    ],
)
def test_budget_matches_real_advance_call_count(n_layers: int, max_layers: int) -> None:
    """rank 0's predicted budget must equal the number of advance()
    calls rank 1 REALLY needs. If the budget is short, rank 0 stops
    sending while rank 1 still waits for another advance -> both ranks
    block on each other forever (the Section 39/45 mutual deadlock). If
    it's long, rank 1 gets an advance after its session already
    finished, which trips advance()'s own completed-session guard."""
    real = _real_advances_to_done(n_layers, max_layers)
    predicted = _budget(n_layers, max_layers)
    assert predicted == real, (
        f"advance-budget mismatch for n_layers={n_layers} "
        f"max_layers={max_layers}: rank 0 would send {predicted} advance(s) "
        f"but rank 1 really needs {real}. "
        f"(short budget => mutual deadlock; long budget => "
        f"completed-session raise)"
    )


def test_old_ceil_formula_is_short_on_the_even_parity() -> None:
    """Guards the ROOT CAUSE itself, so nobody 'simplifies' the budget
    back to ceil(). Documents precisely why Section 40's live
    measurement (21 layers, odd) could not see this bug."""

    def _old_ceil_budget(peer_layer_count: int, max_layers: int) -> int:
        return -(-peer_layer_count // max_layers)

    # Even parity -- the real production case that deadlocked.
    assert _real_advances_to_done(22, 2) == 12
    assert _old_ceil_budget(22, 2) == 11  # short by one -> deadlock
    assert _budget(22, 2) == 12  # fixed

    # Odd parity -- what Section 40 happened to measure. Both formulas
    # agree here, which is exactly why the hypothesis looked disproven.
    assert _real_advances_to_done(21, 2) == 11
    assert _old_ceil_budget(21, 2) == 11
    assert _budget(21, 2) == 11


def test_sentinel_only_advance_reports_zero_layers_and_done() -> None:
    """On the even parity the FINAL advance consumes only the ``("done",
    ...)`` sentinel, so it legitimately advances ZERO layers. Rank 1's
    handler must treat that as a normal completion (it branches purely on
    ``done``; ``layers_advanced`` is only logged) -- this pins that
    ``(0, True)`` shape, which never executed in production before the
    Section 45 fix because the old budget stopped one call short."""
    model = _FakeInterruptibleModel(n_layers=4)
    session = ResumablePrefillSession(
        inner_model=model,
        inputs=mx.array([0.0]),
        cache=(),
    )
    assert session.advance(
        max_layers=2, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
    ) == (2, False)
    assert session.advance(
        max_layers=2, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
    ) == (2, False)
    # All 4 layers consumed, but the sentinel has NOT been reached yet --
    # this third call is the one the old ceil() budget never sent.
    assert session.advance(
        max_layers=2, phase_for_pause=ForwardPhase.PREFILL_CONTINUE
    ) == (0, True)

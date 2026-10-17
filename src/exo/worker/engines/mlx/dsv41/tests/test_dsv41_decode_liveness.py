"""CPU-only tests for the DSv4.1 DECODE-phase liveness beat + output-length caps.

Two engine fixes are pinned here:

* the DECODE-phase liveness beat: ``Dsv41Engine._rounds`` calls the 15 s
  throttled ``base.Engine.prefill_heartbeat`` once per COMPLETED round. Model
  LOAD and PREFILL already beat; the decode path emitted nothing the worker
  supervisor counts, and the runner-loop decode heartbeat cannot stand in --
  it needs ``step()`` to RETURN, while the output parser can hold a whole
  reasoning phase inside one round. Healthy, actively-decoding runners were
  therefore SIGKILLed as hung (22 kills observed live 2026-10-17). The control
  tests pin the converse: a round loop that never COMPLETES a round (no
  round at all, an empty batch, a collective that unwinds) still goes silent.
* the output-length resolution (``resolve_max_output_tokens``): a request that
  sent no length gets ``EXO_DSV41_DEFAULT_MAX_OUTPUT_TOKENS`` and every
  resolved value is clamped by ``EXO_DSV41_MAX_OUTPUT_TOKENS_CEILING``, so a
  runaway generation cannot pin the served slot (and both TP ranks with it).

The clock is FAKE in the heartbeat section: ``prefill_heartbeat`` throttles on
``time.monotonic``, so a simulated decode is what makes "at least once, and at
most once, per 15 s" checkable without wall-clock sleeps.
"""

# This is a WHITE-BOX test: it deliberately pokes the driver-engine's private
# attributes (``_rounds``, ``_last_prefill_heartbeat_monotonic``, ``heartbeat``)
# and swaps the engine module's ``_one_round``/``logger`` for doubles. None of
# that is a real defect, so these rules are disabled for the file rather than
# ignored line-by-line.
# pyright: reportPrivateUsage=false, reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import pytest

from exo.worker.engines import base as base_mod
from exo.worker.engines.mlx.constants import MAX_TOKENS
from exo.worker.engines.mlx.dsv41 import engine as eng_mod
from exo.worker.engines.mlx.dsv41.tests.test_dsv41_engine import (
    ANSWER_TEXT,
    _drain,
    _engine,
)

# Reuse the load-warmup/liveness suite's real-engine builder and its logger
# recorder (the same convention ``test_dsv41_reasoning_budget`` uses to import
# its harness from ``test_dsv41_engine``) rather than a bespoke construction.
from .test_dsv41_prefill_liveness import _capture, _make_engine

# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class _FakeSession:
    """The only session surface ``_rounds`` touches without ``_one_round``.

    ``_rounds`` passes ``session.cache.cache`` and ``session.draft_state`` to
    ``_one_round`` (which is patched out here), calls ``session.maybe_checkpoint``
    at every round boundary and reads ``session.eos_id``.
    """

    def __init__(self, eos_id: int = -1) -> None:
        self.cache = SimpleNamespace(cache=object())
        self.draft_state = object()
        self.eos_id = eos_id
        self.checkpoints = 0

    def maybe_checkpoint(self) -> None:
        self.checkpoints += 1


def _fake_rounds(
    monkeypatch: pytest.MonkeyPatch,
    clock: dict[str, float],
    *,
    step_seconds: float,
    tokens_per_round: int = 1,
) -> dict[str, int]:
    """Install a ``_one_round`` that commits a batch and advances the clock.

    Each round advances the FAKE clock by ``step_seconds`` and returns a
    non-empty batch of ids that can never be the EOS sentinel (``_rounds``
    stops the loop on a committed EOS).
    """
    calls = {"n": 0}

    def fake_one_round(engine: Any, **kwargs: Any) -> tuple[list[int], float, int, int]:
        del engine, kwargs
        calls["n"] += 1
        clock["t"] += step_seconds
        first = 100 + calls["n"] * tokens_per_round
        return [first + i for i in range(tokens_per_round)], 0.0, 0, 0

    monkeypatch.setattr(eng_mod, "_one_round", fake_one_round)
    return calls


def _beats(engine: Any, clock: dict[str, float]) -> list[float]:
    """Record the fake-clock time of every ``heartbeat`` the engine emits."""
    hits: list[float] = []
    engine.heartbeat = lambda: hits.append(clock["t"])
    return hits


# --------------------------------------------------------------------------
# (a) decode rounds beat at least once -- and at most once -- per 15 s
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step_seconds", "rounds", "expected_beats"),
    [
        # 12 x 5 s = 55 s of decoding after the first (immediate) beat:
        # one beat per 15 s boundary.
        (5.0, 12, 4),
        # A much tighter round cadence must NOT beat any more often: the
        # throttle is a wall-clock bound, not a per-round counter.
        (1.0, 40, 3),
    ],
)
def test_decode_rounds_beat_once_per_fifteen_seconds(
    monkeypatch: pytest.MonkeyPatch,
    step_seconds: float,
    rounds: int,
    expected_beats: int,
) -> None:
    engine = _make_engine()
    clock = {"t": 1000.0}
    beats = _beats(engine, clock)
    session = _FakeSession()
    _fake_rounds(monkeypatch, clock, step_seconds=step_seconds)
    monkeypatch.setattr(base_mod.time, "monotonic", lambda: clock["t"])

    batches = list(engine._rounds(session, 42, rounds))

    assert len(batches) == rounds, "every round committed and was yielded"
    assert len(beats) == expected_beats, beats
    # The FIRST completed round beats immediately (the throttle's sentinel is
    # 0.0, which real monotonic time is always past): a long reasoning phase
    # must be covered from its very first round, not after 15 s of silence.
    assert beats[0] == 1000.0 + step_seconds
    gaps = [b - a for a, b in pairwise(beats)]
    # At MOST once per 15 s...
    assert all(g >= 15.0 for g in gaps), gaps
    # ...and at least once per 15 s plus at most one round of overhead (the
    # design doc's corrected invariant: silence <= throttle + longest round).
    assert all(g <= 15.0 + step_seconds for g in gaps), gaps
    # The trailing silence after the last beat is bounded too: the turn's final
    # rounds are covered by the last beat.
    last_round_at = 1000.0 + step_seconds * rounds
    assert last_round_at - beats[-1] <= 15.0


def test_every_completed_round_offers_a_beat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The beat is *offered* on every completed round (the throttle decides).

    With exactly 15 s between rounds every offer is due, so the wiring -- one
    call per completed round, ON the round boundary -- is asserted independently
    of the cadence test above. (A FROZEN clock, by contrast, lands exactly one
    beat: that is the throttle working, and it is pinned by the frozen-clock
    assertion below.)
    """

    engine = _make_engine()
    clock = {"t": 5000.0}
    beats = _beats(engine, clock)
    session = _FakeSession()
    calls = _fake_rounds(monkeypatch, clock, step_seconds=15.0)
    monkeypatch.setattr(base_mod.time, "monotonic", lambda: clock["t"])

    list(engine._rounds(session, 42, 7))

    assert calls["n"] == 7
    assert len(beats) == 7, "one beat per completed round when the throttle is due"


def test_a_frozen_clock_lands_exactly_one_beat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check on the throttle itself: with time not advancing, a long
    run of completed rounds emits ONE beat -- the per-round offer is throttled,
    not forwarded blindly."""
    engine = _make_engine()
    clock = {"t": 5000.0}
    beats = _beats(engine, clock)
    _fake_rounds(monkeypatch, clock, step_seconds=0.0)
    monkeypatch.setattr(base_mod.time, "monotonic", lambda: clock["t"])

    list(engine._rounds(_FakeSession(), 42, 9))

    assert beats == [5000.0]


# --------------------------------------------------------------------------
# (b) controls: no completed round => NO beat (a wedged collective is silent)
# --------------------------------------------------------------------------


def test_a_loop_with_no_completed_round_produces_no_beat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_tokens=0``: the while loop body never runs, so nothing beats.

    This is the shape of a round loop wedged BEFORE its first collective -- the
    supervisor must still see the silence and kill it.
    """
    engine = _make_engine()
    clock = {"t": 1000.0}
    beats = _beats(engine, clock)
    calls = _fake_rounds(monkeypatch, clock, step_seconds=600.0)
    monkeypatch.setattr(base_mod.time, "monotonic", lambda: clock["t"])

    assert list(engine._rounds(_FakeSession(), 42, 0)) == []
    assert calls["n"] == 0
    assert beats == [], "a loop that completed no round must not beat"


def test_an_empty_round_raises_and_beats_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty-batch invariant also guarantees no beat: a round that
    committed nothing must never be reported as progress (fail loud instead)."""
    engine = _make_engine()
    clock = {"t": 1000.0}
    beats = _beats(engine, clock)

    def empty_round(engine_: Any, **kwargs: Any) -> tuple[list[int], float, int, int]:
        del engine_, kwargs
        clock["t"] += 120.0  # the round took two minutes and committed nothing
        return [], 0.0, 0, 0

    monkeypatch.setattr(eng_mod, "_one_round", empty_round)
    monkeypatch.setattr(base_mod.time, "monotonic", lambda: clock["t"])

    with pytest.raises(RuntimeError, match="committed no tokens"):
        list(engine._rounds(_FakeSession(), 42, 4))
    assert beats == []


def test_a_wedged_collective_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``_one_round`` that unwinds without committing beats nothing.

    A genuinely wedged collective never returns at all; the only way it can
    leave this loop is by raising, and even then no round COMPLETED -- so the
    supervisor's silence clock keeps running and the kill still lands.
    """
    engine = _make_engine()
    clock = {"t": 1000.0}
    beats = _beats(engine, clock)

    def wedged(engine_: Any, **kwargs: Any) -> tuple[list[int], float, int, int]:
        del engine_, kwargs
        clock["t"] += 900.0
        raise RuntimeError("collective never returned")

    monkeypatch.setattr(eng_mod, "_one_round", wedged)
    monkeypatch.setattr(base_mod.time, "monotonic", lambda: clock["t"])

    with pytest.raises(RuntimeError, match="never returned"):
        list(engine._rounds(_FakeSession(), 42, 8))
    assert beats == [], "a round that never completed must not look alive"


# --------------------------------------------------------------------------
# (c) resolve_max_output_tokens: request wins / default / ceiling / 0 disables
# --------------------------------------------------------------------------


def _clear_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(eng_mod.DEFAULT_MAX_OUTPUT_TOKENS_ENV, raising=False)
    monkeypatch.delenv(eng_mod.MAX_OUTPUT_TOKENS_CEILING_ENV, raising=False)


def test_the_knob_names_and_defaults_are_the_documented_ones() -> None:
    assert (
        eng_mod.DEFAULT_MAX_OUTPUT_TOKENS_ENV == "EXO_DSV41_DEFAULT_MAX_OUTPUT_TOKENS"
    )
    assert eng_mod.DEFAULT_MAX_OUTPUT_TOKENS == 8192
    assert (
        eng_mod.MAX_OUTPUT_TOKENS_CEILING_ENV == "EXO_DSV41_MAX_OUTPUT_TOKENS_CEILING"
    )
    assert eng_mod.MAX_OUTPUT_TOKENS_CEILING == 32768
    # The clamp must not touch the old default path: MAX_TOKENS stays below it.
    assert MAX_TOKENS == 32168 < eng_mod.MAX_OUTPUT_TOKENS_CEILING


def test_a_positive_request_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_knobs(monkeypatch)
    assert eng_mod.resolve_max_output_tokens(512) == 512
    assert eng_mod.resolve_max_output_tokens(MAX_TOKENS) == MAX_TOKENS


def test_an_absent_request_gets_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_knobs(monkeypatch)
    assert eng_mod.resolve_max_output_tokens(None) == eng_mod.DEFAULT_MAX_OUTPUT_TOKENS
    # <= 0 is the same "the client did not ask" case (the old ``or MAX_TOKENS``).
    assert eng_mod.resolve_max_output_tokens(0) == eng_mod.DEFAULT_MAX_OUTPUT_TOKENS
    assert eng_mod.resolve_max_output_tokens(-1) == eng_mod.DEFAULT_MAX_OUTPUT_TOKENS


def test_the_default_knob_overrides_the_absent_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_knobs(monkeypatch)
    monkeypatch.setenv(eng_mod.DEFAULT_MAX_OUTPUT_TOKENS_ENV, "4096")
    assert eng_mod.resolve_max_output_tokens(None) == 4096
    # ...and a client that DID ask is still honoured.
    assert eng_mod.resolve_max_output_tokens(64) == 64


def test_a_default_of_zero_keeps_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EXO_DSV41_DEFAULT_MAX_OUTPUT_TOKENS=0`` disables the default, keeping
    the pre-fix ``MAX_TOKENS`` behaviour byte-for-byte."""
    _clear_knobs(monkeypatch)
    monkeypatch.setenv(eng_mod.DEFAULT_MAX_OUTPUT_TOKENS_ENV, "0")
    assert eng_mod.resolve_max_output_tokens(None) == MAX_TOKENS
    assert eng_mod.resolve_max_output_tokens(0) == MAX_TOKENS


def test_an_oversized_request_is_clamped_to_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_knobs(monkeypatch)
    assert (
        eng_mod.resolve_max_output_tokens(100_000) == eng_mod.MAX_OUTPUT_TOKENS_CEILING
    )
    # Exactly the ceiling, and one below it, are untouched.
    assert eng_mod.resolve_max_output_tokens(32768) == 32768
    assert eng_mod.resolve_max_output_tokens(32767) == 32767


def test_the_ceiling_also_clamps_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling is applied to the RESOLVED value, default path included."""
    _clear_knobs(monkeypatch)
    monkeypatch.setenv(eng_mod.DEFAULT_MAX_OUTPUT_TOKENS_ENV, "50000")
    assert eng_mod.resolve_max_output_tokens(None) == eng_mod.MAX_OUTPUT_TOKENS_CEILING


def test_a_ceiling_of_zero_disables_the_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_knobs(monkeypatch)
    monkeypatch.setenv(eng_mod.MAX_OUTPUT_TOKENS_CEILING_ENV, "0")
    assert eng_mod.resolve_max_output_tokens(100_000) == 100_000
    assert eng_mod.resolve_max_output_tokens(None) == eng_mod.DEFAULT_MAX_OUTPUT_TOKENS


def test_the_knobs_are_overridable_to_other_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_knobs(monkeypatch)
    monkeypatch.setenv(eng_mod.DEFAULT_MAX_OUTPUT_TOKENS_ENV, "1024")
    monkeypatch.setenv(eng_mod.MAX_OUTPUT_TOKENS_CEILING_ENV, "2048")
    assert eng_mod.resolve_max_output_tokens(None) == 1024
    assert eng_mod.resolve_max_output_tokens(999_999) == 2048


def test_a_malformed_knob_warns_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher typo degrades to the code default, loudly, never crashes."""
    _clear_knobs(monkeypatch)
    monkeypatch.setenv(eng_mod.DEFAULT_MAX_OUTPUT_TOKENS_ENV, "lots")
    with _capture(eng_mod) as cap:
        assert (
            eng_mod.resolve_max_output_tokens(None) == eng_mod.DEFAULT_MAX_OUTPUT_TOKENS
        )
    assert any("non-integer" in m for m in cap.of("WARNING")), cap.messages


# --------------------------------------------------------------------------
# (c, continued) the engine logs the resolution only when it CHANGED the value
# --------------------------------------------------------------------------


def _resolve_lines(cap: Any) -> list[str]:
    return [m for m in cap.of("INFO") if "output length resolve" in m]


def test_a_clamped_request_logs_one_resolution_line() -> None:
    engine, _model, _tokenizer, _events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=100_000
    )
    with _capture(eng_mod) as cap:
        _drain(engine)
    lines = _resolve_lines(cap)
    assert len(lines) == 1, lines
    assert "requested=100000" in lines[0]
    assert f"resolved={eng_mod.MAX_OUTPUT_TOKENS_CEILING}" in lines[0]


def test_a_defaulted_request_logs_one_resolution_line() -> None:
    engine, _model, _tokenizer, _events = _engine([34, 1], text_of=ANSWER_TEXT)
    with _capture(eng_mod) as cap:
        _drain(engine)
    lines = _resolve_lines(cap)
    assert len(lines) == 1, lines
    assert "requested=None" in lines[0]
    assert f"resolved={eng_mod.DEFAULT_MAX_OUTPUT_TOKENS}" in lines[0]


def test_an_honoured_request_logs_nothing() -> None:
    """A client that asked for 8 and got 8 is not worth a log line per turn."""
    engine, _model, _tokenizer, _events = _engine(
        [34, 1], text_of=ANSWER_TEXT, max_output_tokens=8
    )
    with _capture(eng_mod) as cap:
        _drain(engine)
    assert not _resolve_lines(cap)

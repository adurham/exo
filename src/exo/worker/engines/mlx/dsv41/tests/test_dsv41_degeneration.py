"""Tests for the DSv4.1 degeneration (repetition-loop) guards.

Mirror of ``src/exo/worker/tests/unittests/test_runner/test_loop_detection.py``
(the generic MLX path's kill-switch tests), adapted to what is DIFFERENT about the
DSv4.1 engine:

* the engine feeds the guard from its own decode loop, and a decode ROUND commits
  a whole batch of tokens at once (DSpark draft + verify), so the stream under
  test is presented in round-sized batches as well as one token at a time;
* termination is asserted through the REAL engine + parser pipeline (the chunks an
  OpenAI-compatible client would actually see: ``ErrorChunk`` for the default
  ``"error"`` action, a clean ``"stop"`` terminal for the ``"stop"`` action), not
  by calling the detector in isolation;
* the SESSION contract on a guard-fire is asserted too -- the turn is rolled back,
  never checkpointed, and the conversation still serves the next turn;
* the per-request observability (first-token latency, the ~10 s throughput line,
  and the bounded committed-token dump on a guard-fire or a client cancel) is
  asserted, because "the next incident is diagnosable from disk" is half the point
  of this change: the incident that motivated it streamed ~3m41s of unbroken
  repetition and NOTHING of that text was ever recorded.

The thresholds themselves are NOT re-derived here (the generic path's suite owns
them); what is pinned is that this engine's driver reproduces them over a batched
feed, and that the two engines cannot silently disagree about the *policy*.
"""
# pyright: reportPrivateUsage=false
#   (deliberate: the suite drives the guard's private internals and the engine's
#   private session/state attributes, exactly as the generic loop-detection tests do.)

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mlx.core as mx
import pytest

from exo.shared.types.chunks import ErrorChunk, TokenChunk


def _loop_truncation(chunks):
    """The terminal of a guard-truncated turn (finish_reason=length + cause)."""
    finals = [
        c
        for c in chunks
        if isinstance(c, TokenChunk) and c.finish_reason == "length"
    ]
    assert finals, "the guard must truncate the degenerate turn"
    assert finals[-1].loop_cause is not None, "a truncation must carry its cause"
    assert finals[-1].loop_cause.finish_cause == "repetition_loop"
    assert finals[-1].loop_cause.retryable is False
    return finals[-1]
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.worker.engines.mlx.dsv41.degeneration import (
    DEFAULT_LOOP_ACTION,
    DEGENERATION_ERROR_TEXT,
    DUMP_DIR_ENV,
    DUMP_TOKEN_LIMIT,
    DegenerationGuard,
    assert_guard_env_agrees_across_ranks,
    guard_env_digest,
    guard_env_fingerprint,
    guard_env_report,
    guard_env_values,
    loop_action,
    write_degeneration_dump,
)
from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
from exo.worker.engines.mlx.dsv41.tests.conftest import FakeTokenizer, model_id
from exo.worker.engines.mlx.dsv41.tests.test_dsv41_engine import (
    ANSWER_TEXT,
    MockHead,
    ScriptedModel,
    _drain,
    _engine,
    _Receiver,
    _Sender,
)
from exo.worker.engines.mlx.generator.batch_generate import (
    _LOOP_DETECT_ACTION,
    _LOOP_DETECT_ENABLED,
    _LOOP_DETECT_LONG_BLOCK,
    _LOOP_DETECT_LONG_ENABLED,
    _LOOP_DETECT_LONG_MAX_PERIOD,
    _LOOP_DETECT_LONG_MIN_REPEATS,
    _LOOP_DETECT_MAX_PERIOD,
    _LOOP_DETECT_MIN_REPEATS,
    _LOOP_DETECT_WINDOW,
)

#: The same cycle the generic suite's period-2 case uses (``exo -> exo -> ...``).
CYCLE = [101, 102]

#: The REAL confirmed 14-token degeneration cycle from the generic path's own
#: regression test (" So N = 1,234,567,891,011." on DSv4's tokenizer). Copied
#: verbatim: if this engine's default knobs ever narrowed below the real failure's
#: period, that regression must fail HERE too.
REAL_14_TOKEN_CYCLE = [
    3016,
    471,
    438,
    223,
    19,
    14,
    14456,
    14,
    25601,
    14,
    31444,
    14,
    13407,
    16,
]


def _feed(seq: Any, guard: DegenerationGuard | None = None) -> tuple[int | None, Any]:
    """Feed a flat sequence one token at a time; returns ``(index, hit)``."""
    guard = guard if guard is not None else DegenerationGuard()
    for index, token_id in enumerate(seq):
        hit = guard.feed(int(token_id))
        if hit is not None:
            return index, hit
    return None, None


def _feed_batched(rounds: list[list[int]]) -> tuple[int | None, Any]:
    """Feed ROUND-sized batches (the engine's feed shape); returns ``(index, hit)``.

    The index returned is the flat committed-token index, so a batched feed and a
    token-at-a-time feed of the SAME stream are directly comparable.
    """
    guard = DegenerationGuard()
    index = -1
    for batch in rounds:
        for token_id in batch:
            index += 1
            hit = guard.feed(int(token_id))
            if hit is not None:
                return index, hit
    return None, None


def _block_stream(blocks: list[list[int]]) -> list[int]:
    return [token for block in blocks for token in block]


# ----------------------------------------------------------------- exact-token


class TestTightCycleDetector:
    def test_period_one_token_spam_fires(self):
        index, hit = _feed([5] * 10)
        assert hit is not None
        assert hit.detector == "token"
        assert hit.period == 1
        assert index == _LOOP_DETECT_MIN_REPEATS - 1

    def test_period_two_cycle_fires_at_min_repeats(self):
        index, hit = _feed(CYCLE * _LOOP_DETECT_MIN_REPEATS)
        assert hit is not None
        assert (hit.detector, hit.period) == ("token", 2)
        assert hit.repeats >= _LOOP_DETECT_MIN_REPEATS
        # The cycle is confirmed on the last token of the min_repeats-th cycle.
        assert index == 2 * _LOOP_DETECT_MIN_REPEATS - 1

    def test_below_min_repeats_does_not_fire(self):
        index, hit = _feed(CYCLE * (_LOOP_DETECT_MIN_REPEATS - 1))
        assert (index, hit) == (None, None)

    def test_max_period_boundary(self):
        at = list(range(_LOOP_DETECT_MAX_PERIOD)) * _LOOP_DETECT_MIN_REPEATS
        over = list(range(_LOOP_DETECT_MAX_PERIOD + 1)) * _LOOP_DETECT_MIN_REPEATS
        assert _feed(at)[1] is not None
        assert _feed(over)[1] is None

    def test_real_confirmed_14_token_cycle_fires(self):
        index, hit = _feed(REAL_14_TOKEN_CYCLE * 20)
        assert hit is not None, (
            "the real confirmed degeneration cycle must be detected with the "
            "current shared defaults -- if this fails the knobs were narrowed"
        )
        assert hit.detector == "token"
        assert hit.period == 14
        assert index == 14 * _LOOP_DETECT_MIN_REPEATS - 1

    def test_varied_stream_does_not_fire(self):
        random.seed(1)
        assert _feed([random.randint(0, 5000) for _ in range(64)])[1] is None

    def test_long_normal_generation_does_not_fire(self):
        """A genuinely varied 6000-token generation must not trip either detector."""
        random.seed(7)
        guard = DegenerationGuard()
        for _ in range(6000):
            if guard.feed(random.randint(0, 200_000)) is not None:
                pytest.fail("a varied generation is not degeneration")
        assert len(guard.recent_token_ids) <= _LOOP_DETECT_WINDOW

    def test_window_is_bounded(self):
        guard = DegenerationGuard()
        for token_id in range(1000):
            guard.feed(token_id)
        assert len(guard.recent_token_ids) == _LOOP_DETECT_WINDOW

    def test_cycle_only_at_the_tail_fires(self):
        index, hit = _feed([9, 8, 7, 6] + CYCLE * _LOOP_DETECT_MIN_REPEATS)
        assert hit is not None and hit.period == 2
        assert index == 4 + 2 * _LOOP_DETECT_MIN_REPEATS - 1

    def test_cycle_interrupted_before_the_threshold_is_not_counted(self):
        """The streaming feed fires the instant a cycle crosses the threshold, so
        the case that matters is a run that never GETS there: an interrupted cycle
        must not be counted, and a shorter second run must not inherit the first's
        repeats."""
        assert (
            _feed(
                CYCLE * (_LOOP_DETECT_MIN_REPEATS - 1)
                + [99]
                + CYCLE * (_LOOP_DETECT_MIN_REPEATS - 1)
            )[1]
            is None
        )

    def test_disabled_guard_never_fires(self):
        assert _feed(CYCLE * 50, DegenerationGuard(enabled=False))[1] is None


# -------------------------------------------------------------- long-period


class TestLongPeriodDetector:
    """Exact word-for-word repeated multi-sentence cycles, seen through block
    hashes. Blocks are distinct 24-token patterns so the tight detector (max
    period 24) can never be what fires."""

    def _blocks_for(self, seed: int, count: int) -> list[list[int]]:
        random.seed(seed)
        return [
            [random.randint(1000, 999_999) for _ in range(_LOOP_DETECT_LONG_BLOCK)]
            for _ in range(count)
        ]

    def test_two_block_cycle_fires(self):
        a, b, _ = self._blocks_for(3, 3)
        index, hit = _feed(_block_stream([a, b] * _LOOP_DETECT_LONG_MIN_REPEATS))
        assert hit is not None
        assert hit.detector == "long_period"
        assert hit.period == 2
        assert index == 2 * _LOOP_DETECT_LONG_MIN_REPEATS * _LOOP_DETECT_LONG_BLOCK - 1

    def test_three_block_cycle_fires(self):
        blocks = self._blocks_for(4, 3)
        _index, hit = _feed(_block_stream(blocks * _LOOP_DETECT_LONG_MIN_REPEATS))
        assert hit is not None and hit.period == 3

    def test_below_min_repeats_does_not_fire(self):
        a, b, _ = self._blocks_for(5, 3)
        assert (
            _feed(_block_stream([a, b] * (_LOOP_DETECT_LONG_MIN_REPEATS - 1)))[1]
            is None
        )

    def test_max_period_boundary(self):
        blocks = self._blocks_for(6, _LOOP_DETECT_LONG_MAX_PERIOD + 1)
        at = _block_stream(
            blocks[:_LOOP_DETECT_LONG_MAX_PERIOD] * _LOOP_DETECT_LONG_MIN_REPEATS
        )
        over = _block_stream(blocks * _LOOP_DETECT_LONG_MIN_REPEATS)
        assert _feed(at)[1] is not None
        assert _feed(over)[1] is None

    def test_one_token_difference_breaks_the_cycle(self):
        """Exact match only: a single differing token in one block means no cycle.

        The detector's hash scan is a CANDIDATE finder; this is the case the raw
        block re-verify exists for -- and it is also the false-positive guard for
        'looks repetitive but is not degenerate' long-form reasoning.
        """
        a, b, _ = self._blocks_for(8, 3)
        b_prime = list(b)
        b_prime[7] += 1
        assert _feed(_block_stream([a, b, a, list(b), a, b_prime]))[1] is None

    def test_varied_blocks_do_not_fire(self):
        random.seed(2)
        stream = _block_stream(self._blocks_for(9, 48))
        assert _feed(stream)[1] is None

    def test_incrementing_blocks_do_not_fire(self):
        """A numbered list whose blocks each differ must not fire (the tight
        detector needs 6 identical tokens in a row; each block here is an
        arithmetic run, so neither detector can fire)."""
        blocks = [
            [1000 * i + j for j in range(_LOOP_DETECT_LONG_BLOCK)] for i in range(20)
        ]
        assert _feed(_block_stream(blocks))[1] is None

    def test_block_window_is_bounded(self):
        guard = DegenerationGuard()
        for token_id in range(500):
            guard.feed(token_id)
        assert len(guard.block_hashes) <= guard.long_window_blocks


# -------------------------------------------- the dsv41 batched feed equivalence


class TestBatchedFeed:
    """A DSv4.1 round commits a whole batch at once. The guard must see the SAME
    stream, so the verdict must not depend on the batching."""

    def _mixed_stream(self) -> list[int]:
        random.seed(11)
        prefix = [random.randint(2000, 90_000) for _ in range(300)]
        return prefix + CYCLE * 12

    def test_round_batched_feed_matches_token_at_a_time(self):
        stream = self._mixed_stream()
        per_token = _feed(stream)
        # The same stream cut into round-sized batches (3, 1, 4, 2, ... tokens).
        rounds: list[list[int]] = []
        cursor = 0
        for size in [3, 1, 4, 2, 5]:
            while cursor < len(stream):
                rounds.append(stream[cursor : cursor + size])
                cursor += size
        batched = _feed_batched(rounds)
        assert per_token[0] == batched[0]
        assert per_token[1] == batched[1]
        assert batched[1] is not None and batched[1].detector == "token"

    def test_engine_terminates_at_the_index_the_guard_predicts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """End to end: the engine's own round-batched feed fires where the guard
        predicts, and the dump records exactly the committed stream up to it."""
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        stream = self._mixed_stream()
        expected_index, expected_hit = _feed(stream)
        assert expected_index is not None and expected_hit is not None

        engine, _model, _tokenizer, _events = _engine(
            stream + [1], text_of=ANSWER_TEXT, max_output_tokens=2000
        )
        chunks = _drain(engine)
        final = _loop_truncation(chunks)
        assert not any(isinstance(c, ErrorChunk) for c in chunks), (
            "a loop truncation must NOT be an error frame (an error is a retry "
            "signal and greedy decoding reproduces the same cycle)"
        )
        assert final.loop_cause.dropped_tokens >= 0

        dumps = list(tmp_path.glob("*.json"))
        assert len(dumps) == 1, "a guard-fire must dump the committed tail"
        payload = json.loads(dumps[0].read_text())
        # ``produced`` is [anchor] + every committed token, so the guard fires with
        # exactly expected_index + 1 committed ids behind it.
        assert payload["returned_tokens"] == expected_index + 1
        assert payload["token_ids"][-len(expected_hit.cycle_token_ids) :] == (
            expected_hit.cycle_token_ids
        )
        assert payload["reason"] == "guard-token"

    def test_the_gated_tail_is_dropped_not_streamed(self):
        """A TIGHT cycle: the emission gate holds the tail while a repeat is
        suspected, so the client's stream stops BEFORE the cycle rather than
        receiving it and then being told to ignore it."""
        text_of = {101: "AB", 102: "CD"}
        engine, _model, _tokenizer, _events = _engine(
            CYCLE * 12 + [1], text_of=text_of, max_output_tokens=500
        )
        chunks = _drain(engine)
        final = _loop_truncation(chunks)
        text = "".join(c.text for c in chunks if isinstance(c, TokenChunk))
        assert "ABCDABCDABCD" not in text, f"the cycle leaked into the stream: {text!r}"
        assert final.loop_cause.dropped_tokens >= 1

    def test_draft_tokens_are_never_fed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """With a head attached one round commits several tokens -- and a LYING
        draft must not shift the verdict, because only target-confirmed tokens
        are ever committed (and therefore only those are ever fed)."""
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        stream = self._mixed_stream()
        expected_index, _hit = _feed(stream)
        assert expected_index is not None

        engine = _engine_with_head(stream + [1], lie_at=1, max_output_tokens=2000)
        chunks = _drain(engine)
        _loop_truncation(chunks)
        # This round really did commit SEVERAL tokens at once (a draft was accepted),
        # so the feed under test is the batched one, not a degenerate 1-token stream.
        assert engine._spec_drafted > 0 and engine._spec_accepted > 0

        guard_dumps = list(tmp_path.glob("*guard-token*.json"))
        assert len(guard_dumps) == 1
        payload = json.loads(guard_dumps[0].read_text())
        assert payload["returned_tokens"] == expected_index + 1
        # The lie never appears in the fed/committed stream.
        assert 9999 not in payload["token_ids"]

    def test_termination_mid_round_keeps_the_session_reusable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A guard-fire inside a speculative round must leave the conversation at a
        point the cache actually holds, and the NEXT turn must still be served."""
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        monkeypatch.setenv("EXO_DSV41_PARK", "0")
        engine = _engine_with_head(CYCLE * 60 + [1], lie_at=1, max_output_tokens=500)
        chunks = _drain(engine)
        _loop_truncation(chunks)

        conversation = next(iter(engine._sessions._entries.values())).session
        # Rolled back to the pre-turn state, NOT checkpointed: the degenerate turn
        # contributes no rows to what the cache holds.
        assert conversation.offset == 0
        assert conversation.tokens == []
        assert conversation._inflight is False

        # ... and the conversation still serves the next turn normally.
        engine.submit(
            TextGeneration(
                task_id=TaskId("task-2"),
                command_id=CommandId("cmd-2"),
                task_params=TextGenerationTaskParams(
                    model=engine.model_id,
                    input=[
                        InputMessage(role="user", content=InputMessageContent("again"))
                    ],
                    max_output_tokens=4,
                    temperature=0.0,
                ),
                instance_id=InstanceId("inst-1"),
            )
        )
        engine._active = None  # the previous turn already finished
        pairs = list(engine.step())
        assert pairs, "the next turn must start"


# --------------------------------------------------------------- action policy


class TestActionPolicy:
    """The finish-reason policy is a deliberate choice (see the buildnote). What is
    pinned here is that it is the SAME policy as the generic path's by default, that
    it is always TERMINATING, and that a typo cannot disarm it."""

    def test_default_action_is_truncate(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("EXO_LOOP_DETECT_ACTION", raising=False)
        assert DEFAULT_LOOP_ACTION == "truncate"
        assert loop_action() == "truncate"

    def test_default_action_matches_the_generic_path(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("EXO_LOOP_DETECT_ACTION", raising=False)
        # DELIBERATE DIVERGENCE (2026-10-17): the dsv41 engine truncates a loop
        # (finish_reason="length" + a dsv41 cause) while the GENERIC path still
        # fails the turn as an error -- it has no truncation implementation, and
        # its "terminate" predicate only knows stop/error.
        assert loop_action() == "truncate"
        assert _LOOP_DETECT_ACTION == "error"

    def test_default_action_is_terminating(self):
        assert DEFAULT_LOOP_ACTION in ("stop", "error", "truncate")

    def test_env_selects_stop(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EXO_LOOP_DETECT_ACTION", "stop")
        assert loop_action() == "stop"

    def test_env_selects_warn(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EXO_LOOP_DETECT_ACTION", "WARN")
        assert loop_action() == "warn"

    def test_unknown_action_falls_back_to_the_terminating_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("EXO_LOOP_DETECT_ACTION", "banana")
        assert loop_action() == "truncate"

    def test_error_action_maps_to_an_error_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        monkeypatch.setenv("EXO_LOOP_DETECT_ACTION", "error")
        engine, _m, _t, _e = _engine(
            CYCLE * 40 + [1], text_of=ANSWER_TEXT, max_output_tokens=1000
        )
        chunks = _drain(engine)
        errors = [c for c in chunks if isinstance(c, ErrorChunk)]
        assert errors and errors[0].error_message == DEGENERATION_ERROR_TEXT
        assert not any(
            isinstance(c, TokenChunk) and c.finish_reason == "stop" for c in chunks
        )

    def test_stop_action_finishes_with_stop_and_no_error_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        monkeypatch.setenv("EXO_LOOP_DETECT_ACTION", "stop")
        engine, _m, _t, _e = _engine(
            CYCLE * 40 + [1], text_of=ANSWER_TEXT, max_output_tokens=1000
        )
        chunks = _drain(engine)
        assert not any(isinstance(c, ErrorChunk) for c in chunks)
        stops = [
            c for c in chunks if isinstance(c, TokenChunk) and c.finish_reason == "stop"
        ]
        assert stops and stops[-1].text == ""

    def test_warn_action_does_not_terminate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        monkeypatch.setenv("EXO_LOOP_DETECT_ACTION", "warn")
        engine, model, _t, _e = _engine(
            CYCLE * 40 + [1], text_of=ANSWER_TEXT, max_output_tokens=20
        )
        chunks = _drain(engine)
        # No error, and the turn ran to its own limit instead of being cut.
        assert not any(isinstance(c, ErrorChunk) for c in chunks)
        assert any(
            isinstance(c, TokenChunk) and c.finish_reason == "length" for c in chunks
        )
        assert len(model.calls) >= 20


# ---------------------------------------------------------- session consistency


class TestSessionConsistency:
    def test_guard_fire_rolls_back_instead_of_checkpointing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        engine, _m, _t, _e = _engine(
            CYCLE * 40 + [1], text_of=ANSWER_TEXT, max_output_tokens=1000
        )
        _drain(engine)
        conversation = next(iter(engine._sessions._entries.values())).session
        assert conversation.offset == 0
        assert conversation.tokens == []
        assert conversation.generated == []
        assert conversation._inflight is False

    def test_abort_turn_mirrors_the_failure_path(self):
        """``_abort_turn`` cancels and -- for a one-off conversation -- drops it,
        exactly like the engine's own failure path."""
        engine = _engine_with_head(CYCLE * 40 + [1])
        cancelled: list[int] = []
        dropped: list[Any] = []
        session = SimpleNamespace(cancel=lambda: cancelled.append(1))
        engine._sessions.drop = dropped.append  # type: ignore[method-assign]
        engine._abort_turn(session, True)
        assert cancelled == [1] and dropped == []
        engine._abort_turn(session, False)
        assert cancelled == [1, 1] and dropped == [session]

    def test_a_normal_turn_still_checkpoints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The guard must not change the NORMAL completion path: a healthy turn
        still commits its rows (this is the negative control for the rollback)."""
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        engine, _m, _t, _e = _engine(
            [34, 35, 36, 1], text_of=ANSWER_TEXT, max_output_tokens=8
        )
        chunks = _drain(engine)
        assert (
            "".join(c.text for c in chunks if isinstance(c, TokenChunk))
            == "hi there bye"
        )
        conversation = next(iter(engine._sessions._entries.values())).session
        assert conversation.offset > 0
        assert conversation.tokens != []
        assert conversation._inflight is False


# --------------------------------------------------------------- observability


class TestObservability:
    def test_env_report_names_every_knob_and_a_digest(self):
        report = guard_env_report(1, 2)
        assert "rank 1/2" in report
        assert f"digest={guard_env_digest()}" in report
        for name, value in guard_env_values():
            assert f"{name}={value}" in report

    def test_env_digest_tracks_the_action_knob(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("EXO_LOOP_DETECT_ACTION", raising=False)
        before = guard_env_digest()
        monkeypatch.setenv("EXO_LOOP_DETECT_ACTION", "stop")
        assert guard_env_digest() != before

    def test_env_report_reflects_the_generic_knobs(self):
        """The reported values are the EFFECTIVE ones (the shared knobs), not a
        second copy that could drift."""
        values = dict(guard_env_values())
        assert values["EXO_LOOP_DETECT"] == ("1" if _LOOP_DETECT_ENABLED else "0")
        assert values["EXO_LOOP_DETECT_WINDOW"] == str(_LOOP_DETECT_WINDOW)
        assert values["EXO_LOOP_DETECT_MAX_PERIOD"] == str(_LOOP_DETECT_MAX_PERIOD)
        assert values["EXO_LOOP_DETECT_MIN_REPEATS"] == str(_LOOP_DETECT_MIN_REPEATS)
        assert values["EXO_LOOP_DETECT_LONG"] == (
            "1" if _LOOP_DETECT_LONG_ENABLED else "0"
        )
        assert values["EXO_LOOP_DETECT_LONG_BLOCK"] == str(_LOOP_DETECT_LONG_BLOCK)
        assert values["EXO_LOOP_DETECT_LONG_MAX_PERIOD"] == str(
            _LOOP_DETECT_LONG_MAX_PERIOD
        )
        assert values["EXO_LOOP_DETECT_LONG_MIN_REPEATS"] == str(
            _LOOP_DETECT_LONG_MIN_REPEATS
        )

    def test_guard_fire_logs_the_detection_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        engine, _m, _t, _e = _engine(
            CYCLE * 40 + [1], text_of=ANSWER_TEXT, max_output_tokens=1000
        )
        records, sink = _capture_logs()
        try:
            _drain(engine)
        finally:
            _drop_sink(sink)
        text = "\n".join(records)
        assert "DEGENERATION DETECTED" in text
        assert "guard_env_digest=" in text
        assert "period=2" in text
        assert "action=truncate" in text

    def test_first_token_progress_and_turn_end_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        engine, _m, _t, _e = _engine(
            [34, 35, 36, 1], text_of=ANSWER_TEXT, max_output_tokens=8
        )
        records, sink = _capture_logs()
        try:
            _drain(engine)
        finally:
            _drop_sink(sink)
        text = "\n".join(records)
        assert "obs first-token" in text
        assert "obs turn-end" in text
        assert "tok_s=" in text

    def test_dump_written_on_a_client_cancel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A cancel mid-decode must capture the committed tail (the incident that
        motivated this port was cancelled by hand and its text was never recorded)."""
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        engine, _m, _t, _e = _engine(
            CYCLE * 60 + [1], text_of=ANSWER_TEXT, max_output_tokens=1000
        )
        # Drive a few rounds, then cancel this task and let the engine notice.
        for _ in range(3):
            list(engine.step())
        cast("Any", engine.cancel_receiver).items.append(TaskId("task-1"))
        for _ in range(50):
            list(engine.step())

        dumps = list(tmp_path.glob("*client-cancel*.json"))
        assert dumps, "a client cancel must dump the committed tail"
        payload = json.loads(dumps[0].read_text())
        assert payload["reason"] == "client-cancel"
        assert payload["token_ids"], "the dump must contain the committed ids"

    def test_nonzero_rank_writes_no_dump(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        engine, _m, _t, _e = _engine(
            CYCLE * 40 + [1], text_of=ANSWER_TEXT, max_output_tokens=1000, device_rank=1
        )
        _drain(engine)
        assert list(tmp_path.glob("*.json")) == []

    def test_dump_is_bounded_to_the_tail(self, tmp_path: Path):
        ids = list(range(5000))
        path = write_degeneration_dump(
            token_ids=ids,
            reason="unit",
            task_id="t-1",
            generated=5000,
            log_dir=tmp_path,
        )
        assert path is not None
        payload = json.loads(Path(path).read_text())
        assert payload["returned_tokens"] == DUMP_TOKEN_LIMIT
        assert payload["token_ids"] == ids[-DUMP_TOKEN_LIMIT:]
        assert payload["generated"] == 5000
        assert payload["guard_env_digest"] == guard_env_digest()

    def test_dump_failure_is_swallowed(self, tmp_path: Path):
        """A dump must never break serving: an unwritable target logs and returns
        ``None`` instead of raising."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        assert (
            write_degeneration_dump(
                token_ids=[1, 2, 3],
                reason="unit",
                task_id="t-1",
                generated=3,
                log_dir=blocker,
            )
            is None
        )


# ----------------------------------------------------- offline replay (real text)

#: Real recorded sessions on this machine (see the buildnote). The first is a
#: successful long agentic run (expected: ZERO detections). The second thrashed
#: and its final generation looped -- but its loop text was never persisted, so
#: what is replayed here is its STORED reasoning: prime false-positive material
#: (genuinely repetitive-looking long-form reasoning that must NOT fire).
STATE_DB = Path.home() / ".hermes" / "state.db"
REPLAY_SESSIONS = {
    "clean-long-run": "20261013_115838_90d257",
    "thrashing-session": "20261013_184432_180a8c",
}
TOKENIZER_CANDIDATES = (
    Path.home()
    / ".exo"
    / "models"
    / "deepseek-ai--DeepSeek-V4-Flash"
    / "tokenizer.json",
)
_REPLAY_CHAR_BUDGET = 300_000


def _local_tokenizer() -> Any | None:
    """The checkpoint family's tokenizer, if it is on this machine."""
    for candidate in TOKENIZER_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            from tokenizers import Tokenizer

            return Tokenizer.from_file(str(candidate))
        except Exception:  # noqa: BLE001 - a missing/broken tokenizer is a skip
            continue
    return None


def _session_texts(session_id: str) -> list[str]:
    """Every stored text field of one session, bounded, read-only."""
    con = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT content, reasoning, reasoning_content FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        con.close()
    out: list[str] = []
    total = 0
    for row in rows:
        for field in row:
            if isinstance(field, str) and field:
                out.append(field)
                total += len(field)
                if total >= _REPLAY_CHAR_BUDGET:
                    return out
    return out


@pytest.mark.skipif(
    _local_tokenizer() is None,
    reason="no local DSv4 tokenizer (residual risk: the real-text replay is not run)",
)
@pytest.mark.skipif(not STATE_DB.exists(), reason="no local Hermes session DB")
def test_real_recorded_sessions_do_not_false_positive():
    """Replay REAL session text through the guard at the shipped defaults.

    A detection here would be a FALSE POSITIVE on genuine output -- the exact
    failure mode that would make the guard unusable on this cluster.
    """
    tokenizer = _local_tokenizer()
    assert tokenizer is not None
    summary: list[str] = []
    for label, session_id in REPLAY_SESSIONS.items():
        texts = _session_texts(session_id)
        if not texts:
            summary.append(f"{label}: NO TEXT (skipped)")
            continue
        detections: list[str] = []
        tokens = 0
        for text in texts:
            ids = cast("list[int]", tokenizer.encode(text).ids)
            tokens += len(ids)
            guard = DegenerationGuard()
            for token_id in ids:
                hit = guard.feed(int(token_id))
                if hit is not None:
                    detections.append(
                        f"{hit.detector} period={hit.period} repeats={hit.repeats}"
                    )
                    break
        summary.append(
            f"{label}: {len(texts)} text fields, {tokens} tokens, "
            f"{len(detections)} detection(s) {detections}"
        )
    print("offline replay: " + " | ".join(summary))
    for line in summary:
        assert "0 detection(s) []" in line or "NO TEXT" in line, (
            f"the guard fired on real (non-degenerate) recorded text: {line}"
        )


# ------------------------------------------------- separator false-positive fix


class TestSeparatorExemption:
    """A run of one separator glyph -- a markdown horizontal rule, an ``=====``
    banner, a code-fence/ASCII-art row -- is legitimate formatting and used to
    fire the period-1 detector, killing the turn. The engine wires the
    tokenizer's decode in, so the tight detector can tell a separator run from a
    word cycle. Proven here through the REAL engine: the run must NOT error, and
    a SUSTAINED separator spam must still be caught (later) by the long-period
    detector, so termination is not lost -- only the false positive is.
    """

    def test_engine_does_not_fire_on_a_markdown_rule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        # 40 dashes (well past min_repeats=6) then EOS.
        engine, _m, _t, _e = _engine(
            [45] * 40 + [1],
            text_of={**ANSWER_TEXT, 45: "-"},
            max_output_tokens=60,
        )
        chunks = _drain(engine)
        assert not any(isinstance(c, ErrorChunk) for c in chunks)
        assert "".join(c.text for c in chunks if isinstance(c, TokenChunk)) == "-" * 40
        assert not list(tmp_path.glob("*guard-*.json")), "no guard-fire, no dump"

    def test_sustained_separator_spam_is_still_caught_by_the_long_detector(self):
        # The tight detector exempts the separator run; the block-hash detector
        # (72+ identical tokens = 3 identical 24-token blocks) still terminates it.
        guard = DegenerationGuard(token_text=lambda ids: "-" * len(ids))
        hit = None
        for _ in range(200):
            hit = guard.feed(45)
            if hit is not None:
                break
        assert hit is not None
        assert hit.detector == "long_period"
        assert hit.period == 1

    def test_engine_still_terminates_sustained_separator_spam(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        engine, _m, _t, _e = _engine(
            [45] * 80,
            text_of={**ANSWER_TEXT, 45: "-"},
            max_output_tokens=100,
        )
        chunks = _drain(engine)
        assert not any(isinstance(c, ErrorChunk) for c in chunks)
        _loop_truncation(chunks)


# ------------------------------------------------------- rank-env fail-fast


class TestRankEnvAgreement:
    """A guard knob that differs between the TP ranks is a hang class: one rank
    terminates a degenerate turn while its peer keeps decoding (and wedges in the
    model's TP all_sum). The startup check reduces the env fingerprint with a min
    AND a max over the coord subgroup and fails the boot on disagreement.
    """

    def test_no_group_is_a_noop(self):
        # single rank / PP mode / the unit tests: nothing to compare against.
        assert_guard_env_agrees_across_ranks(None)

    def test_fingerprint_is_nonnegative_and_int32_safe(self):
        fp = guard_env_fingerprint()
        assert 0 <= fp < 2**31

    def test_matching_env_passes(self, monkeypatch: pytest.MonkeyPatch):
        import exo.worker.engines.mlx.utils_mlx as um

        # A reduce that returns the value unchanged == every rank agrees.
        monkeypatch.setattr(um, "mx_min_int", lambda value, group: value)
        assert_guard_env_agrees_across_ranks(object())

    def test_mismatch_raises_loudly(self, monkeypatch: pytest.MonkeyPatch):
        import exo.worker.engines.mlx.utils_mlx as um

        # min != max: the ranks disagree.
        monkeypatch.setattr(um, "mx_min_int", lambda value, group: value - 1)
        with pytest.raises(RuntimeError, match="MISMATCH"):
            assert_guard_env_agrees_across_ranks(object())


# ------------------------------------------------------- feed alignment (anchor)


def test_guard_stream_starts_at_the_first_generated_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No 1-token offset vs the generic path.

    The generic path feeds the guard every GENERATED token from the first one.
    The engine's ``anchor`` IS that first generated token -- ``rounds._one_round``
    emits only the token PREDICTED from the anchor, never the anchor itself -- so
    ``produced = [anchor] + rounds`` is exactly "first generated token onward".
    This pins it: the anchor is the dump's first committed id, and the engine
    fires one flat position later than a token-at-a-time feed of the round stream
    (i.e. the anchor is fed, and nothing is fed twice).
    """
    monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
    stream = CYCLE * 40
    engine, _m, _t, _e = _engine(
        stream + [1], text_of=ANSWER_TEXT, max_output_tokens=1000
    )
    _drain(engine)
    dumps = list(tmp_path.glob("*guard-token*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text())
    # The first fed id is the anchor, which for this script is the first token of
    # the cycle -> the anchor is part of what the guard judged.
    assert payload["token_ids"][0] == CYCLE[0]
    # ... and the fire lands on the SAME flat token as a raw token-at-a-time feed
    # of the round stream (index expected_index), i.e. exactly one count for the
    # anchor, not a duplicate and not a skip.
    expected_index, _hit = _feed(stream)
    assert expected_index is not None
    assert payload["returned_tokens"] == expected_index + 1


# ------------------------------------------------------------------ test helpers


class _StubWindow:
    """The surface of a DSpark draft window ``Conversation`` keeps in lockstep.

    Only ``n_ctx`` (checked against the cache offset at every checkpoint, cadence
    and cancel) and ``win_kv`` (snapshotted by ``Conversation._checkpoint``) are
    read; the engine suite's ``MockHead`` returns a bare ``object()``, which is
    fine for its own round-unit tests but cannot satisfy a REAL Conversation's
    ``draft_ctx() == offset`` invariant over a multi-row prefill.
    """

    def __init__(self) -> None:
        self.n_ctx = 0
        self.win_kv = mx.zeros((1, 1, 4), dtype=mx.float32)


class _LockstepModel(ScriptedModel):
    """A ``ScriptedModel`` whose taps carry ONE ROW PER FED ROW.

    The engine suite's mock hardcodes a single tap row for a logits forward
    (``self._taps(1)``), which is right for the round-unit tests it was written for
    but wrong for the real engine: the prefill's per-chunk taps are what advance the
    draft window, so with one row per CHUNK the window falls behind the cache and
    the Conversation's lockstep invariant fires. ``__call__`` records the fed row
    count in ``self.calls`` before asking for taps, so use that.
    """

    def _taps(self, rows: int) -> dict[int, mx.array]:
        del rows
        fed_rows = int(self.calls[-1][1]) if self.calls else 1
        return {0: mx.zeros((1, fed_rows, 4), dtype=mx.float32)}


class _LockstepHead(MockHead):
    """A head whose draft window ADVANCES with the rows it is given.

    ``draft`` reads its position from the window (``n_ctx``), which is what the
    real DSpark head does implicitly, so a round's draft is the target's own
    continuation and acceptance is total -- i.e. a round really does commit several
    tokens at once, which is the feed shape these tests exist to cover.
    """

    def make_cache(self, bsz: int = 1) -> Any:
        return [_StubWindow()]

    def append_ctx(self, taps: mx.array, dsc: Any) -> None:
        rows = int(taps.shape[1])
        for window in dsc:
            window.n_ctx = int(window.n_ctx) + rows
        self.appended.append(rows)

    def draft(
        self, anchor: mx.array, embed: Any, head_lin: Any, dsc: Any, *, width: int
    ) -> Any:
        del anchor, embed, head_lin
        start = self.model._index(int(dsc[0].n_ctx) + 1)
        ids = list(self.model.script[start : start + width])
        if self.lie_at is not None and self.lie_at < len(ids):
            ids[self.lie_at] = 9999  # a token the target will not confirm
        return mx.array([ids], dtype=mx.int32), mx.ones((1, len(ids)))


def _engine_with_head(
    script: list[int],
    *,
    lie_at: int | None = None,
    max_output_tokens: int | None = None,
    device_rank: int = 0,
) -> Dsv41Engine:
    """A Dsv41Engine with a DSpark head attached, so rounds commit >1 token.

    ``_engine`` in the engine suite builds headless engines only (its head tests
    drive ``_one_round`` directly); the batched-feed tests here need a REAL engine
    whose rounds commit several tokens at once.
    """
    tokenizer = FakeTokenizer(text_of=ANSWER_TEXT)
    model = _LockstepModel(script)
    head = _LockstepHead(model, lie_at=lie_at)
    loaded = Dsv41Loaded(
        model=model,
        tokenizer=tokenizer,
        args=model.args,
        model_path=Path("/nonexistent-checkpoint"),
        built_layers=list(range(40)),
        full_stack=True,
        rank=0,
        world=1,
        load_seconds=0.0,
        head=head,
    )
    engine = Dsv41Engine(
        loaded=loaded,
        model_id=model_id(),
        group=None,
        cancel_receiver=_Receiver(),  # type: ignore[arg-type]
        event_sender=_Sender(),  # type: ignore[arg-type]
        device_rank=device_rank,
        speculative=True,
        prefill_chunk_size=2,
    )
    engine.submit(
        TextGeneration(
            task_id=TaskId("task-1"),
            command_id=CommandId("cmd-1"),
            task_params=TextGenerationTaskParams(
                model=model_id(),
                input=[InputMessage(role="user", content=InputMessageContent("hello"))],
                max_output_tokens=max_output_tokens,
                temperature=0.0,
            ),
            instance_id=InstanceId("inst-1"),
        )
    )
    return engine


def _capture_logs() -> tuple[list[str], int]:
    """Attach a loguru sink; returns ``(records, sink_id)``."""
    from loguru import logger as loguru_logger

    records: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(str(message)), level="INFO"
    )
    return records, sink_id


def _drop_sink(sink_id: int) -> None:
    from loguru import logger as loguru_logger

    loguru_logger.remove(sink_id)

"""Tests for DSv4 repetition-loop (decode degeneration) detection + kill-switch.

DSv4 occasionally collapses into a short token cycle that repeats forever
(observed 2026-06-15: an "exo → exo → … → hermes → hermes …" loop that ran for
minutes — a complete killer). A sampling penalty only lowers the PROBABILITY of
looping; it cannot guarantee termination (repetition_penalty=1.05 was tried and
empirically failed — mlx-lm's repetition_penalty is presence-based, applying the
penalty once regardless of repeat count). The deterministic guarantee is the
kill-switch: when ``_detect_token_loop`` finds a cycle, the generation is
force-terminated with finish_reason="stop" (EXO_LOOP_DETECT_ACTION="stop",
default on).

These tests cover the detector's threshold behavior and the action default.
"""

from exo.worker.engines.mlx.generator.batch_generate import (
    _detect_token_loop,
    _LOOP_DETECT_ACTION,
    _LOOP_DETECT_MAX_PERIOD,
    _LOOP_DETECT_MIN_REPEATS,
)


class TestDetectTokenLoop:
    def test_period_one_token_spam(self):
        # A single token repeated (e.g. BOS spam) is a period-1 loop.
        assert _detect_token_loop([5] * 10) == (1, 10)

    def test_period_two_cycle(self):
        # The screenshot's "exo → exo → …" is a period-2 cycle [tok, arrow].
        assert _detect_token_loop([101, 102] * 20) == (2, 20)

    def test_at_min_repeats_triggers(self):
        # Exactly min_repeats back-to-back cycles must trigger.
        ids = [1, 2] * _LOOP_DETECT_MIN_REPEATS
        result = _detect_token_loop(ids)
        assert result is not None
        assert result[0] == 2

    def test_below_min_repeats_does_not_trigger(self):
        # One fewer than min_repeats must NOT trigger (avoids false positives on
        # legitimate short repetition like "ha ha ha").
        ids = [1, 2] * (_LOOP_DETECT_MIN_REPEATS - 1)
        assert _detect_token_loop(ids) is None

    def test_max_period_boundary(self):
        # A cycle exactly at max_period triggers; one longer does not.
        at = list(range(_LOOP_DETECT_MAX_PERIOD)) * _LOOP_DETECT_MIN_REPEATS
        over = list(range(_LOOP_DETECT_MAX_PERIOD + 1)) * _LOOP_DETECT_MIN_REPEATS
        assert _detect_token_loop(at) is not None
        assert _detect_token_loop(over) is None

    def test_varied_text_does_not_trigger(self):
        import random

        random.seed(1)
        ids = [random.randint(0, 5000) for _ in range(64)]
        assert _detect_token_loop(ids) is None

    def test_loop_only_at_tail_counts(self):
        # Varied prefix then a loop at the tail: still detected (the cycle ends
        # at the most recent token, which is what runaway decode looks like).
        ids = [9, 8, 7, 6] + [3, 4] * _LOOP_DETECT_MIN_REPEATS
        result = _detect_token_loop(ids)
        assert result is not None
        assert result[0] == 2

    def test_broken_cycle_not_counted(self):
        # A cycle interrupted before the tail must not count the older run.
        ids = [1, 2] * _LOOP_DETECT_MIN_REPEATS + [99]
        # The tail is now [..., 2, 99]; no period-1/2 cycle ends at 99.
        assert _detect_token_loop(ids) is None


class TestLoopDetectAction:
    def test_default_action_is_stop(self):
        # The kill-switch must default ON — an infinite loop is a complete killer
        # and detection-only (legacy "warn") does not terminate it.
        assert _LOOP_DETECT_ACTION == "stop"

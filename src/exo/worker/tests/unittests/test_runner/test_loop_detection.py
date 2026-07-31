"""Tests for DSv4 repetition-loop (decode degeneration) detection + kill-switch.

DSv4 occasionally collapses into a short token cycle that repeats forever
(observed 2026-06-15: an "exo → exo → … → hermes → hermes …" loop that ran for
minutes — a complete killer). A sampling penalty only lowers the PROBABILITY of
looping; it cannot guarantee termination (repetition_penalty=1.05 was tried and
empirically failed — mlx-lm's repetition_penalty is presence-based, applying the
penalty once regardless of repeat count). The deterministic guarantee is the
kill-switch: when ``_detect_token_loop`` finds a cycle, the generation is
terminated.

By the time a cycle is confirmed the output is already degenerate — the tokens
leading INTO the loop are garbage too (observed 2026-06-16: DSv4 regurgitated
session_search result JSON into its reasoning, then looped on the `}"]` tail;
the surfaced "answer" was a 2-char `"]`). So the default action is "error":
fail the turn cleanly (finish_reason="error" -> ErrorChunk -> 500 -> hermes
retries) and REPLACE the degenerate partial with a diagnostic message, rather
than surface the broken remnant. "stop" (terminate but surface the partial) and
"warn" (log only) remain available via EXO_LOOP_DETECT_ACTION.

These tests cover the detector's threshold behavior and the action default.

TestDetectLongPeriodLoop covers a SECOND, separate detector added 2026-07-31:
DSv4's reasoning_content stuck 20+ minutes cycling the same handful of full
sentences WORD-FOR-WORD IDENTICAL each repeat, with a period far longer
(~150-300+ raw tokens) than _LOOP_DETECT_MAX_PERIOD (24) can reach. Rather
than raising max_period directly (which would multiply the tight-loop
detector's per-TOKEN cost by ~12x forever), _detect_long_period_loop chunks
the stream into fixed-size blocks, hashes each block once as it completes,
and runs the SAME periodicity algorithm over the much shorter list of block
hashes. See the "Long-period (multi-sentence) degeneration detection" comment
block in batch_generate.py for the full rationale.
"""

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import ModelId
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.generator.batch_generate import (
    _DEGENERATION_ERROR_TEXT,
    _LOOP_DETECT_ACTION,
    _LOOP_DETECT_LONG_MAX_PERIOD,
    _LOOP_DETECT_LONG_MIN_REPEATS,
    _LOOP_DETECT_MAX_PERIOD,
    _LOOP_DETECT_MIN_REPEATS,
    _detect_long_period_loop,
    _detect_token_loop,
)
from exo.worker.runner.llm_inference.model_output_parsers import (
    map_responses_to_chunks,
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

    def test_real_confirmed_14_token_degeneration_cycle(self):
        """Regression (2026-07-26): live hard_eval run against DSv4-Flash
        with all speculative decoding off (math_largest_prime_factor task)
        produced a genuine degeneration loop -- the model got stuck
        re-verifying the same multi-digit multiplication ("So N =
        1,234,567,891,011.") over 100 times back-to-back, burning the
        entire 8192-token budget. This exact phrase tokenizes to a 14-TOKEN
        cycle on DSv4's tokenizer (confirmed via direct encode) -- which
        the OLD max_period=8 default made mathematically impossible to
        detect (not a tuning-too-conservative issue, a hard ceiling below
        the real failure's period). max_period/window were widened
        specifically so this class is caught."""
        # The actual confirmed token ids for " So N = 1,234,567,891,011."
        cycle = [3016, 471, 438, 223, 19, 14, 14456, 14, 25601, 14, 31444, 14, 13407, 16]
        ids = cycle * 20
        result = _detect_token_loop(ids)
        assert result is not None, (
            "the real confirmed degeneration cycle must be detected with "
            "current defaults -- if this fails, max_period/window were "
            "narrowed back below the real failure's period"
        )
        period, repeats = result
        assert period == 14
        assert repeats >= _LOOP_DETECT_MIN_REPEATS


class TestLoopDetectAction:
    def test_default_action_is_error(self):
        # The kill-switch must default to a TERMINATING action — an infinite
        # loop is a complete killer and detection-only ("warn") does not stop
        # it. The default is "error" (clean retryable fail), not "stop"
        # (surface the partial), because the pre-loop tokens are already
        # degenerate and surfacing them hands the user broken output.
        assert _LOOP_DETECT_ACTION == "error"

    def test_error_action_is_terminating(self):
        # Both "error" and "stop" must be terminating actions; only "warn" is
        # log-only. (Guards the detection-site membership check.)
        assert _LOOP_DETECT_ACTION in ("error", "stop")

    def test_degeneration_error_text_is_meaningful(self):
        # The diagnostic that REPLACES the degenerate partial must be non-empty
        # and mention degeneration so it is useful in the ErrorChunk / logs.
        assert _DEGENERATION_ERROR_TEXT
        assert "degeneration" in _DEGENERATION_ERROR_TEXT.lower()


class TestDegenerationErrorSurfacing:
    """The surfacing contract for the "error" action: a degeneration response
    (finish_reason="error", text=diagnostic) must map to an ErrorChunk carrying
    the diagnostic — NOT a TokenChunk that surfaces the degenerate remnant.
    This is what stops the morning-2026-06-16 `"]` leak reaching the user."""

    def test_error_response_maps_to_error_chunk(self):
        # This is exactly what batch_generate.step emits when the kill-switch
        # fires with action="error": text replaced by the diagnostic,
        # finish_reason="error".
        resp = GenerationResponse(
            text=_DEGENERATION_ERROR_TEXT,
            token=0,
            finish_reason="error",
            usage=None,
        )
        chunk = map_responses_to_chunks(resp, ModelId("mlx-community/DeepSeek-V4-Flash"))
        assert isinstance(chunk, ErrorChunk)
        # The diagnostic is carried as the error message; the degenerate partial
        # is never surfaced as displayable content.
        assert chunk.error_message == _DEGENERATION_ERROR_TEXT


class TestDetectLongPeriodLoop:
    """Detector for multi-sentence, exact-word-for-word repetition cycles
    whose period is far beyond _LOOP_DETECT_MAX_PERIOD (24). Operates over
    BLOCK hashes, not raw token ids — see module docstring for rationale.
    """

    def test_two_block_cycle_triggers(self):
        # Simulates the real incident's shape: two distinct ~block-sized
        # "sentences" (represented here as distinct hash values) alternating.
        hashes = [111, 222] * _LOOP_DETECT_LONG_MIN_REPEATS
        result = _detect_long_period_loop(hashes)
        assert result is not None
        assert result[0] == 2  # period, in blocks
        assert result[1] >= _LOOP_DETECT_LONG_MIN_REPEATS

    def test_three_block_cycle_triggers(self):
        # The real incident cycled THREE distinct paragraphs, not two.
        hashes = [111, 222, 333] * _LOOP_DETECT_LONG_MIN_REPEATS
        result = _detect_long_period_loop(hashes)
        assert result is not None
        assert result[0] == 3

    def test_at_min_repeats_triggers(self):
        hashes = [1, 2] * _LOOP_DETECT_LONG_MIN_REPEATS
        result = _detect_long_period_loop(hashes)
        assert result is not None

    def test_below_min_repeats_does_not_trigger(self):
        hashes = [1, 2] * (_LOOP_DETECT_LONG_MIN_REPEATS - 1)
        assert _detect_long_period_loop(hashes) is None

    def test_max_period_boundary(self):
        at = list(range(_LOOP_DETECT_LONG_MAX_PERIOD)) * _LOOP_DETECT_LONG_MIN_REPEATS
        over = (
            list(range(_LOOP_DETECT_LONG_MAX_PERIOD + 1))
            * _LOOP_DETECT_LONG_MIN_REPEATS
        )
        assert _detect_long_period_loop(at) is not None
        assert _detect_long_period_loop(over) is None

    def test_varied_blocks_do_not_trigger(self):
        # Distinct block hashes throughout (no repeating cycle) must not
        # false-positive — this is the guard against flagging genuinely
        # varied long-form reasoning as degeneration.
        import random

        random.seed(2)
        hashes = [random.randint(0, 10**9) for _ in range(48)]
        assert _detect_long_period_loop(hashes) is None

    def test_broken_cycle_not_counted(self):
        hashes = [1, 2] * _LOOP_DETECT_LONG_MIN_REPEATS + [99]
        assert _detect_long_period_loop(hashes) is None

    def test_shortest_period_preferred(self):
        # If both a period-1 AND period-2 interpretation would satisfy
        # min_repeats at the tail, the scan (period ascending) returns the
        # shortest — same contract as _detect_token_loop.
        hashes = [7] * (_LOOP_DETECT_LONG_MIN_REPEATS * 3)
        result = _detect_long_period_loop(hashes)
        assert result is not None
        assert result[0] == 1

    def test_does_not_trigger_on_legitimately_repetitive_but_varied_content(self):
        # A numbered-list-style pattern where each block differs (e.g. by an
        # incrementing index folded into the hash) must not trigger, same
        # false-positive guard the tight-loop detector already relies on.
        hashes = [hash((i, "list item")) for i in range(20)]
        assert _detect_long_period_loop(hashes) is None

# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Tests for make_reasoning_budget_limiter (reasoning-loop invariant).

Regression coverage for the 2026-07-26 finding: DeepSeek-V4-Flash can reach
a correct answer inside its reasoning, then fall into a long-period
self-doubt loop ("But earlier I got X? Actually I got X? Let's recheck...")
that re-derives the same result verbatim (with wording drift each cycle)
indefinitely, consuming the ENTIRE max_tokens budget on reasoning_content
and leaving `content` empty. Confirmed via hard_eval.py against the live
exo cluster (tasks math_digit_sum / math_largest_prime_factor /
math_binom_mod), reproduced identically at temp=0 greedy AND
temp=1.0/top_p=0.95 real sampling, and identically with MTP/DSpark
speculative decoding fully on or off (rules out decode-path causes). Not
caught by the existing degeneration kill-switch (period<=8 exact-repeat
detector; this loop's period is 60-400+ tokens with paraphrase drift).

make_reasoning_budget_limiter enforces an invariant instead of pattern-
matching the loop shape: reasoning must not be allowed to consume the
entire generation budget and leave the answer channel empty. Once a
still-open thinking block exceeds its token budget, the only viable next
token is forced to be think_end_id.
"""
import mlx.core as mx

from exo.worker.engines.mlx.generator.generate import (
    make_reasoning_budget_limiter,
    safe_think_token_id,
)

THINK_START = 0
THINK_END = 1
VOCAB = 10


def _logits(shape: tuple[int, ...] = (1, VOCAB)) -> mx.array:
    return mx.zeros(shape)


def _is_forced_to_think_end(out: mx.array) -> bool:
    forced = bool((out[..., THINK_END] > 1e8).item())
    others_banned = bool((out[..., 0] < -1e8).item())
    return forced and others_banned


class TestReasoningBudgetLimiterConstruction:
    def test_none_when_think_start_id_missing(self):
        assert make_reasoning_budget_limiter(None, THINK_END, 10) is None

    def test_none_when_think_end_id_missing(self):
        assert make_reasoning_budget_limiter(THINK_START, None, 10) is None

    def test_none_when_budget_is_zero(self):
        assert make_reasoning_budget_limiter(THINK_START, THINK_END, 0) is None

    def test_none_when_budget_is_negative(self):
        assert make_reasoning_budget_limiter(THINK_START, THINK_END, -5) is None

    def test_active_when_all_inputs_valid(self):
        assert make_reasoning_budget_limiter(THINK_START, THINK_END, 10) is not None


class TestReasoningBudgetLimiterBehavior:
    def test_no_think_start_seen_is_noop_regardless_of_length(self):
        """Never entered a thinking block at all -- must never fire, even
        on a very long history (this is the plain-chat, no-thinking case)."""
        proc = make_reasoning_budget_limiter(THINK_START, THINK_END, budget_tokens=5)
        assert proc is not None
        history = mx.array([5, 6, 7, 8, 9] * 50)
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_in_thinking_under_budget_is_noop(self):
        proc = make_reasoning_budget_limiter(THINK_START, THINK_END, budget_tokens=10)
        assert proc is not None
        history = mx.array([THINK_START] + [5, 6, 7])
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_in_thinking_over_budget_forces_think_end(self):
        proc = make_reasoning_budget_limiter(THINK_START, THINK_END, budget_tokens=10)
        assert proc is not None
        history = mx.array([THINK_START] + [5] * 15)
        out = proc(history, _logits())
        assert _is_forced_to_think_end(out)

    def test_in_thinking_exactly_at_budget_forces_think_end(self):
        """Boundary: elapsed == budget_tokens must also force (>=, not >)."""
        proc = make_reasoning_budget_limiter(THINK_START, THINK_END, budget_tokens=10)
        assert proc is not None
        history = mx.array([THINK_START] + [5] * 10)
        out = proc(history, _logits())
        assert _is_forced_to_think_end(out)

    def test_already_closed_is_noop_even_with_very_long_trailing_history(self):
        """The common well-formed case: thinking closed properly, then a
        long answer follows. Must never fire regardless of how long the
        post-close content runs."""
        proc = make_reasoning_budget_limiter(THINK_START, THINK_END, budget_tokens=10)
        assert proc is not None
        history = mx.array(
            [THINK_START] + [5] * 15 + [THINK_END] + [6] * 500
        )
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_second_reopened_thinking_block_over_budget_forces_close(self):
        """A model that legitimately closes thinking once, then re-opens a
        SECOND thinking block that itself runs long, must be evaluated
        against the most recent (second) think_start, not the first."""
        proc = make_reasoning_budget_limiter(THINK_START, THINK_END, budget_tokens=10)
        assert proc is not None
        history = mx.array(
            [THINK_START] + [1, 2, 3] + [THINK_END]
            + [9, 9, 9]
            + [THINK_START] + [4] * 15
        )
        out = proc(history, _logits())
        assert _is_forced_to_think_end(out)

    def test_second_reopened_thinking_block_under_budget_is_noop(self):
        """Symmetric sanity: the re-opened block under its own budget must
        not be falsely forced just because a lot of total history exists."""
        proc = make_reasoning_budget_limiter(THINK_START, THINK_END, budget_tokens=10)
        assert proc is not None
        history = mx.array(
            [THINK_START] + [1, 2, 3] + [THINK_END]
            + [9] * 200
            + [THINK_START] + [4, 5]
        )
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)


class TestSafeThinkTokenId:
    class _RaisingTokenizer:
        @property
        def think_start_id(self):
            raise ValueError("multi-token delimiter")

    class _PlainTokenizer:
        think_start_id = 42

    class _MissingAttrTokenizer:
        pass

    def test_swallows_value_error_from_property(self):
        assert safe_think_token_id(self._RaisingTokenizer(), "think_start_id") is None

    def test_returns_value_when_present(self):
        assert safe_think_token_id(self._PlainTokenizer(), "think_start_id") == 42

    def test_returns_none_when_attribute_missing(self):
        assert (
            safe_think_token_id(self._MissingAttrTokenizer(), "think_start_id")
            is None
        )

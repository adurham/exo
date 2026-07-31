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


class TestReasoningBudgetLimiterPromptBakedInThinking:
    """Regression for the 2026-07-26 no-op bug: DeepSeek-V4's chat template
    appends a literal <think> suffix to the PROMPT itself -- the model
    never generates an opening <think> token of its own, so a scan of only
    the GENERATED history never finds think_start_id and the limiter
    silently never engaged (confirmed live: an 8192-token request ran its
    entire budget on a self-doubt loop with the fix never firing).
    starts_in_thinking + prompt_token_count are the fix: they mirror the
    same signal the existing stream parser (parse_thinking_models) already
    receives via detect_thinking_prompt_suffix()."""

    def test_starts_in_thinking_with_no_explicit_open_token_still_forces_close(self):
        """The exact production scenario: prompt ends with <think>, no
        think_start_id ever appears in the generated stream, budget must
        still be enforced from the start of GENERATION."""
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10,
            starts_in_thinking=True, prompt_token_count=5,
        )
        assert proc is not None
        # 5 prompt tokens (no THINK_START among them -- it's implicit/not
        # literally present) + 15 generated tokens, still no think_end.
        history = mx.array([100, 101, 102, 103, 104] + [5] * 15)
        out = proc(history, _logits())
        assert _is_forced_to_think_end(out)

    def test_starts_in_thinking_under_budget_is_noop(self):
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10,
            starts_in_thinking=True, prompt_token_count=5,
        )
        assert proc is not None
        history = mx.array([100, 101, 102, 103, 104] + [5, 6, 7])
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_starts_in_thinking_prompt_length_does_not_eat_the_budget(self):
        """A long prompt must not count against the reasoning budget --
        the window anchors to the start of GENERATION, not token 0."""
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10,
            starts_in_thinking=True, prompt_token_count=500,
        )
        assert proc is not None
        # 500 prompt tokens + only 3 generated tokens: nowhere near budget.
        history = mx.array(list(range(500)) + [5, 6, 7])
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_starts_in_thinking_but_explicit_close_seen_is_noop(self):
        """If the model DOES eventually emit a real think_end (even with no
        explicit think_start), that must still correctly close reasoning."""
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10,
            starts_in_thinking=True, prompt_token_count=5,
        )
        assert proc is not None
        history = mx.array(
            [100, 101, 102, 103, 104] + [5, 6] + [THINK_END] + [7] * 500
        )
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_starts_in_thinking_false_with_no_open_token_is_still_noop(self):
        """Sanity: starts_in_thinking=False (the default / non-prompt-baked
        case) must behave exactly as before -- no explicit <think> seen
        means genuinely never entered thinking."""
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10,
            starts_in_thinking=False, prompt_token_count=5,
        )
        assert proc is not None
        history = mx.array([100, 101, 102, 103, 104] + [5] * 50)
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_explicit_think_start_takes_priority_over_starts_in_thinking(self):
        """If an explicit think_start_id IS found in the stream, use that as
        the anchor (existing behavior), not the prompt_token_count fallback
        -- covers a model that both starts pre-opened AND later legitimately
        re-opens a second explicit block."""
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10,
            starts_in_thinking=True, prompt_token_count=5,
        )
        assert proc is not None
        # Explicit close then explicit re-open, well past prompt_token_count.
        history = mx.array(
            [100, 101, 102, 103, 104] + [1, 2] + [THINK_END]
            + [9] * 200
            + [THINK_START] + [4] * 15
        )
        out = proc(history, _logits())
        assert _is_forced_to_think_end(out)


class TestReasoningBudgetLimiterMaxSeconds:
    """Regression coverage for the 2026-07-31 fix: budget_tokens alone could
    not bound wall-clock time when max_output_tokens resolved to a large
    default (no explicit client value), letting the token-only trigger take
    15-30+ minutes to fire. max_seconds is an INDEPENDENT second trigger --
    whichever of budget_tokens or max_seconds fires first forces the close.
    """

    def test_none_when_both_budget_and_time_disabled(self):
        # budget_tokens<=0 AND max_seconds disabled -> no processor at all
        # (same "zero cost when inapplicable" contract as the token-only path).
        assert (
            make_reasoning_budget_limiter(THINK_START, THINK_END, 0, max_seconds=None)
            is None
        )
        assert (
            make_reasoning_budget_limiter(THINK_START, THINK_END, 0, max_seconds=0)
            is None
        )
        assert (
            make_reasoning_budget_limiter(THINK_START, THINK_END, -5, max_seconds=-1)
            is None
        )

    def test_active_with_only_time_trigger_enabled(self):
        # budget_tokens<=0 but max_seconds>0: must still construct (time-only
        # mode), not collapse to None just because the token trigger is off.
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=0, max_seconds=60
        )
        assert proc is not None

    def test_time_trigger_fires_after_max_seconds_elapsed(self, monkeypatch):
        fake_time = [1000.0]
        monkeypatch.setattr(
            "exo.worker.engines.mlx.generator.generate.time.monotonic",
            lambda: fake_time[0],
        )
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END,
            budget_tokens=10_000,  # token trigger far out of reach
            max_seconds=5,
        )
        assert proc is not None
        history = mx.array([THINK_START] + [5] * 3)
        # First call: starts the clock, well under budget/time -- no-op.
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)
        # Advance wall clock past max_seconds without adding many tokens
        # (simulates a slow/degraded cluster: few tokens generated, lots of
        # real time elapsed -- exactly the gap this fix closes).
        fake_time[0] += 6.0
        history2 = mx.array([THINK_START] + [5] * 4)
        out2 = proc(history2, _logits())
        assert _is_forced_to_think_end(out2)

    def test_time_trigger_does_not_fire_before_max_seconds(self, monkeypatch):
        fake_time = [2000.0]
        monkeypatch.setattr(
            "exo.worker.engines.mlx.generator.generate.time.monotonic",
            lambda: fake_time[0],
        )
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10_000, max_seconds=60
        )
        assert proc is not None
        history = mx.array([THINK_START] + [5] * 3)
        proc(history, _logits())  # starts the clock
        fake_time[0] += 30.0  # well under the 60s cap
        out = proc(history, _logits())
        assert not _is_forced_to_think_end(out)

    def test_token_trigger_still_fires_before_time_trigger_when_faster(self):
        # The two triggers are independent -- whichever fires FIRST wins.
        # Here the token budget is tiny and time is generous, so the token
        # trigger must fire without needing any wall-clock to elapse.
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=5, max_seconds=3600
        )
        assert proc is not None
        history = mx.array([THINK_START] + [5] * 10)
        out = proc(history, _logits())
        assert _is_forced_to_think_end(out)

    def test_clock_resets_on_close_for_reopened_thinking_block(self, monkeypatch):
        fake_time = [3000.0]
        monkeypatch.setattr(
            "exo.worker.engines.mlx.generator.generate.time.monotonic",
            lambda: fake_time[0],
        )
        proc = make_reasoning_budget_limiter(
            THINK_START, THINK_END, budget_tokens=10_000, max_seconds=5
        )
        assert proc is not None
        # First thinking block: starts clock, closes cleanly before firing.
        history_open = mx.array([THINK_START] + [1, 2, 3])
        proc(history_open, _logits())
        fake_time[0] += 2.0
        history_closed = mx.array([THINK_START] + [1, 2, 3] + [THINK_END] + [9] * 5)
        out_closed = proc(history_closed, _logits())
        assert not _is_forced_to_think_end(out_closed)
        # A SECOND thinking block re-opens well after the first block's
        # clock would have expired (2s + more elapsed > 5s), but since it's
        # a fresh block the clock must have reset on close -- must NOT fire
        # immediately just because stale elapsed time from the first block
        # would have exceeded max_seconds.
        fake_time[0] += 4.0
        history_reopened = mx.array(
            [THINK_START] + [1, 2, 3] + [THINK_END] + [9] * 5
            + [THINK_START] + [4]
        )
        out_reopened = proc(history_reopened, _logits())
        assert not _is_forced_to_think_end(out_reopened)


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

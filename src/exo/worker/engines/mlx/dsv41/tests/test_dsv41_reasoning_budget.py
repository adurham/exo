"""Tests for the DSv4.1 reasoning-budget limiter (the round-batched port).

Mirror of ``src/exo/worker/engines/mlx/tests/test_reasoning_budget_limiter.py``
(the generic MLX path's logits-processor limiter -- ``make_reasoning_budget_limiter``
in ``generator/generate.py``), adapted to what is DIFFERENT about the custom DSv4.1
engine:

* the engine has no per-token sampler: ``rounds._one_round`` argmaxes a whole
  DSpark draft + verify BATCH at once, so the limiter is fed a round's committed
  batch and CUTS it at the threshold -- the generic path (one token per forward)
  would simply never have produced the tokens past it;
* the forced ``think_end`` is a plain single-row STEP whose output is overridden,
  and its OWN KV is computed by the FOLLOWING round (treating it as already-cached
  is an off-by-one that corrupts the stream);
* the force decision must be a PURE function of the committed ids: both TP ranks
  commit byte-identical tokens, so they agree with no collective. The generic
  path's wall-clock ``max_seconds`` trigger is deliberately NOT ported (a rank-local
  clock is a hang class here).

The thresholds themselves are the generic suite's; this suite pins the round-batched
DRIVER and the engine wiring, and that the two engines cannot silently disagree
about the invariant (reasoning must not eat the whole budget and leave the answer
empty).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.worker.engines.mlx.dsv41 import reasoning_budget as rb
from exo.worker.engines.mlx.dsv41.degeneration import (
    DEGENERATION_ERROR_TEXT,
    DUMP_DIR_ENV,
    REPETITION_LOOP_ERROR_CODE,
)
from exo.worker.engines.mlx.dsv41.reasoning_budget import (
    ReasoningBudgetLimiter,
    budget_tokens_from_request,
    make_reasoning_budget_limiter,
)
from exo.worker.engines.mlx.dsv41.rounds import _one_round, _spec_policy
from exo.worker.engines.mlx.dsv41.tests.conftest import (
    THINK_END,
    THINK_END_ID,
    THINK_START,
    THINK_START_ID,
    FakeTokenizer,
    model_id,
)
from exo.worker.engines.mlx.dsv41.tests.test_dsv41_engine import (
    ANSWER_TEXT,
    MockCache,
    MockHead,
    ScriptedModel,
    _drain,
    _engine,
    _EngineStub,
    _text_of,
)
from exo.worker.engines.mlx.generator.generate import safe_think_token_id

START = THINK_START_ID
END = THINK_END_ID
#: ``ANSWER_TEXT``'s ids are NOT think markers, so a script of these counts as
#: reasoning for the limiter (the fake prompt starts inside thinking): a convenient
#: "pure reasoning" stream with no natural close.
REASONING = 34  # -> "hi"


def _varied(n: int, start: int = 200) -> list[int]:
    """``n`` DISTINCT ids that are not think markers and never repeat.

    Engine-level tests need a long reasoning run that does NOT close on its own --
    but a run of one repeated id would trip the DEGENERATION guard's cycle
    detectors instead (which fire first and abort the turn). A strictly increasing
    stream is provably cycle-free (``_detect_token_loop`` needs a repeated period),
    so only the reasoning-budget limiter is under test.
    """
    return [start + i for i in range(n)]


#: The fake decode table must map the CHECKPOINT's real think ids to their marker
#: STRINGS, or the thinking parser (which matches on text) cannot see the forced
#: end delimiter and every token stays ``is_thinking``.
MARKER_TEXT: dict[int, str] = {THINK_START_ID: THINK_START, THINK_END_ID: THINK_END}
TEXT_WITH_MARKERS: dict[int, str] = {**ANSWER_TEXT, **MARKER_TEXT}


def _limiter(
    budget: int, *, starts_in_thinking: bool = False
) -> ReasoningBudgetLimiter:
    lim = make_reasoning_budget_limiter(
        START, END, budget, starts_in_thinking=starts_in_thinking
    )
    assert lim is not None
    return lim


# ------------------------------------------------------------- factory / knobs


class TestFactory:
    def test_none_when_think_start_id_missing(self):
        assert make_reasoning_budget_limiter(None, END, 10) is None

    def test_none_when_think_end_id_missing(self):
        assert make_reasoning_budget_limiter(START, None, 10) is None

    def test_none_when_budget_is_zero(self):
        assert make_reasoning_budget_limiter(START, END, 0) is None

    def test_none_when_budget_is_negative(self):
        assert make_reasoning_budget_limiter(START, END, -5) is None

    def test_active_when_all_inputs_valid(self):
        assert make_reasoning_budget_limiter(START, END, 10) is not None

    def test_a_multi_token_think_end_disables_the_limiter(self):
        """``think_end_id`` raises ValueError when the delimiter is >1 token;
        ``safe_think_token_id`` must swallow that (no budget limiting rather than
        a crash mid-generation)."""

        class _Multi:
            @property
            def think_end_id(self):
                raise ValueError("more than 1 token")

        assert safe_think_token_id(_Multi(), "think_end_id") is None

    def test_no_wall_clock_parameter_is_ported(self):
        """Rank-local time must never gate the force: the generic ``max_seconds``
        knob is deliberately absent from the round-batched port."""
        params = inspect.signature(make_reasoning_budget_limiter).parameters
        assert "max_seconds" not in params
        assert (
            "prompt_token_count" not in params
        )  # generated tokens are counted directly


class TestBudgetResolution:
    def test_request_value_wins(self):
        assert budget_tokens_from_request(50, 1000) == 50

    def test_absence_falls_back_to_the_shared_default(self):
        expected = min(
            int(1000 * rb.REASONING_BUDGET_FRACTION), rb.REASONING_BUDGET_MAX_TOKENS
        )
        assert budget_tokens_from_request(None, 1000) == expected

    def test_non_positive_request_falls_back_to_the_default(self):
        for value in (0, -7):
            assert budget_tokens_from_request(value, 1000) == min(
                int(1000 * rb.REASONING_BUDGET_FRACTION), rb.REASONING_BUDGET_MAX_TOKENS
            )

    def test_fraction_knob_at_or_below_zero_disables_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(rb, "REASONING_BUDGET_FRACTION", 0.0)
        assert budget_tokens_from_request(None, 1000) <= 0
        # ... but an explicit request value is still honoured.
        assert budget_tokens_from_request(20, 1000) == 20

    def test_default_is_capped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(rb, "REASONING_BUDGET_MAX_TOKENS", 64)
        assert budget_tokens_from_request(None, 10_000) == 64


# ------------------------------------------------------------- phase over the stream


class TestFeedPhase:
    def test_no_open_block_never_forces(self):
        """Plain chat (never entered reasoning): no force, however long."""
        lim = _limiter(5)  # starts_in_thinking=False, no think_start ever seen
        for _ in range(50):
            assert lim.feed([5, 6, 7, 8, 9]) == [5, 6, 7, 8, 9]
            assert lim.next_force is None

    def test_in_think_under_budget_is_noop(self):
        lim = _limiter(10, starts_in_thinking=True)
        assert lim.feed([5, 6, 7]) == [5, 6, 7]
        assert lim.next_force is None

    def test_crossing_exactly_at_the_last_token_latches_the_force(self):
        lim = _limiter(3, starts_in_thinking=True)
        kept = lim.feed([5, 6, 7])
        assert kept == [5, 6, 7]  # all admitted: the cut is AFTER them
        assert lim.next_force == END

    def test_crossing_mid_batch_cuts_the_batch_at_the_threshold(self):
        """The token that reaches the budget is the LAST admitted; the rest --
        including the round's bonus -- are dropped."""
        lim = _limiter(2, starts_in_thinking=True)
        kept = lim.feed([5, 6, 7, 8])
        assert kept == [5, 6]
        assert lim.next_force == END
        assert lim.elapsed == 2

    def test_a_natural_close_before_the_threshold_never_forces(self):
        lim = _limiter(10, starts_in_thinking=True)
        kept = lim.feed([5, END, 6, 7, 8])
        assert kept == [5, END, 6, 7, 8]
        assert lim.next_force is None

    def test_a_natural_close_after_the_threshold_is_dropped(self):
        """A close the model emits PAST the budget point is too late: the force
        wins, and the batch is cut at the threshold."""
        lim = _limiter(2, starts_in_thinking=True)
        kept = lim.feed([5, 6, END, 7])
        assert kept == [5, 6]
        assert lim.next_force == END

    def test_closed_block_is_noop_even_with_long_trailing_history(self):
        lim = _limiter(10, starts_in_thinking=True)
        lim.feed([1, 2, 3, END])
        for _ in range(20):
            assert lim.feed([9] * 5) == [9] * 5
        assert lim.next_force is None

    def test_reopened_block_is_measured_from_its_own_open(self):
        """A second block the model opens itself is evaluated against the SECOND
        think_start, exactly as the generic limiter's "most recent think_start"."""
        lim = _limiter(3, starts_in_thinking=True)
        lim.feed([1, 2, END])  # first block, closed
        kept = lim.feed([START, 5, 6, 7, 8])  # second block crosses mid-batch
        assert kept == [START, 5, 6, 7]
        assert lim.next_force == END

    def test_reopened_block_under_its_own_budget_is_noop(self):
        lim = _limiter(10, starts_in_thinking=True)
        lim.feed([1, 2, END])
        assert lim.feed([START, 5, 6]) == [START, 5, 6]
        assert lim.next_force is None

    def test_starts_in_thinking_counts_the_anchor(self):
        """The prompt opens the block, so the FIRST generated token is reasoning
        token #1 -- the generic limiter counts it too."""
        lim = _limiter(4, starts_in_thinking=True)
        lim.feed([REASONING])  # the anchor
        assert lim.elapsed == 1
        assert lim.feed([REASONING, REASONING]) == [REASONING, REASONING]
        assert lim.elapsed == 3
        assert lim.next_force is None
        lim.feed([REASONING])
        assert lim.next_force == END

    def test_force_is_pure_and_rank_invariant(self):
        """Two limiters fed the same committed stream make the same decision --
        the only cross-rank requirement is byte-identical committed ids."""
        stream = [REASONING] * 12
        rounds = [stream[0:1], stream[1:4], stream[4:9], stream[9:12]]
        a, b = (
            _limiter(7, starts_in_thinking=True),
            _limiter(7, starts_in_thinking=True),
        )
        assert [a.feed(x) for x in rounds] == [b.feed(x) for x in rounds]
        assert a.next_force == b.next_force == END


# ------------------------------------------------------------- round-level cut


class TestRoundCut:
    """The dsv41-specific half: a round's BATCH is cut and the cache/draft rollback
    follows the SHORTENED prefix."""

    def _spec_model(self) -> tuple[ScriptedModel, MockHead, MockCache]:
        model = ScriptedModel([34, 35, 36, 34, 35, 1], prompt_tokens=0)
        model.round_entry_offset = 1  # the round feeds the anchor as row 0
        return model, MockHead(model), MockCache(64)

    def test_a_cut_shortens_the_committed_batch_and_the_cache_rollback(self):
        model, head, cache = self._spec_model()
        engine = _EngineStub(_draft_windows={0: object()})
        lim = _limiter(2, starts_in_thinking=True)
        committed, _ms, accepted, gamma = _one_round(
            engine,  # type: ignore[arg-type]
            model=model,
            cache=cache,
            token=model.script[0],
            head=head,
            policy=_spec_policy(3),
            limiter=lim,
        )
        # The round WOULD have committed 1 + gamma tokens; the limiter cuts it to
        # the threshold (2 reasoning tokens).
        assert gamma == 3
        assert committed == model.script[1:3]
        assert accepted == 3  # the verify's real acceptance is unchanged
        assert lim.next_force == END
        # The cache and the draft window both follow the SHORTENED prefix.
        assert cache.offset == len(committed)
        assert head.appended == [len(committed)]

    def test_without_a_limiter_the_round_is_unchanged(self):
        model, head, cache = self._spec_model()
        engine = _EngineStub(_draft_windows={0: object()})
        committed, _ms, accepted, gamma = _one_round(
            engine,  # type: ignore[arg-type]
            model=model,
            cache=cache,
            token=model.script[0],
            head=head,
            policy=_spec_policy(3),
        )
        assert committed == model.script[1:5]
        assert (accepted, gamma) == (3, 3)
        assert cache.offset == 1 + gamma

    def test_a_forced_round_is_one_plain_step_that_emits_think_end(self):
        """The forced round takes NO draft (the cold start) and commits exactly the
        think-END id; its own KV is deliberately NOT computed here."""
        model, head, cache = self._spec_model()
        engine = _EngineStub(_draft_windows={0: object()})
        lim = _limiter(4, starts_in_thinking=True)
        committed, _ms, accepted, gamma = _one_round(
            engine,  # type: ignore[arg-type]
            model=model,
            cache=cache,
            token=model.script[0],
            head=head,
            policy=_spec_policy(3),
            limiter=lim,
            force_think_end=END,
        )
        assert committed == [END]
        assert (accepted, gamma) == (0, 1)
        # Plain single-row forward: no argmax batch, no draft consulted.
        assert model.argmax_calls == []
        assert cache.offset == 1
        # The draft window is advanced one row (the anchor the forced token sits on).
        assert head.appended == [1]
        # The phase closed: no further force is pending.
        assert lim.next_force is None
        assert lim.in_think is False

    def test_a_forced_round_without_a_head_still_costs_one_row(self):
        model = ScriptedModel([34, 35, 36, 1], prompt_tokens=0)
        cache = MockCache(64)
        committed, _ms, accepted, gamma = _one_round(
            _EngineStub(),  # type: ignore[arg-type]
            model=model,
            cache=cache,
            token=model.script[0],
            head=None,
            policy=None,
            limiter=_limiter(4, starts_in_thinking=True),
            force_think_end=END,
        )
        assert committed == [END]
        assert cache.offset == 1


# ------------------------------------------------------------- engine wiring


class TestEngineWiring:
    def test_limiter_forces_think_end_after_the_budget(self):
        """End to end: a request whose reasoning runs past its budget emits a
        think_end at exactly the budget point, and content follows it."""
        script = _varied(40) + [1]
        # budget = int(8 * 0.75) = 6 reasoning tokens.
        engine, _model, _tokenizer, _events = _engine(
            script, text_of=TEXT_WITH_MARKERS, max_output_tokens=8
        )
        chunks = _drain(engine)
        thinking = [
            c for c in chunks if isinstance(c, TokenChunk) and c.is_thinking and c.text
        ]
        content = [
            c
            for c in chunks
            if isinstance(c, TokenChunk) and not c.is_thinking and c.text
        ]
        assert len(thinking) == 6, "exactly the budget's worth of reasoning tokens"
        assert content, "the forced think_end must let content through"
        conversation = next(iter(engine._sessions._entries.values())).session
        produced = conversation.generated
        assert END in produced
        # The forced think_end sits at the budget point: `budget` reasoning tokens
        # precede it (0-based index == budget).
        assert produced.index(END) == 6
        assert produced[:6] == script[:6]
        assert any(
            isinstance(c, TokenChunk) and c.finish_reason == "length" for c in chunks
        )

    def test_think_end_kv_is_computed_by_the_following_round(self):
        """The forced token is committed but NOT fed in its own round: the next
        round feeds it as its anchor (off-by-one guard)."""
        script = _varied(40) + [1]
        engine, _model, _tokenizer, _events = _engine(
            script, text_of=TEXT_WITH_MARKERS, max_output_tokens=8
        )
        _drain(engine)
        conversation = next(iter(engine._sessions._entries.values())).session
        produced = conversation.generated
        # The cache holds every committed row EXCEPT the last -- which is the
        # forced think_end's row, fed by the next round (or the next turn).
        assert conversation.offset == 3 + len(produced) - 1
        # The token after think_end came from a round that FED think_end as its
        # anchor (the forced id REPLACED the token the model would have emitted, so
        # the next committed id is the following script entry).
        idx = produced.index(END)
        assert produced[idx] == END
        assert produced[idx + 1] == script[idx + 1]

    def test_request_budget_tokens_lowers_the_force_point(self):
        engine, _model, _tokenizer, _events = _engine(
            _varied(40) + [1],
            text_of=TEXT_WITH_MARKERS,
            max_output_tokens=100,
            reasoning_budget_tokens=3,
        )
        _drain(engine)
        produced = next(iter(engine._sessions._entries.values())).session.generated
        assert produced.index(END) == 3

    def test_absent_request_budget_uses_the_default(self):
        engine, _model, _tokenizer, _events = _engine(
            _varied(120) + [1],
            text_of=TEXT_WITH_MARKERS,
            max_output_tokens=100,
        )
        _drain(engine)
        produced = next(iter(engine._sessions._entries.values())).session.generated
        assert produced.index(END) == min(
            int(100 * 0.75), rb.REASONING_BUDGET_MAX_TOKENS
        )

    def test_bench_requests_skip_the_limiter(self):
        """Parity with the generic call site: bench bans EOS to force a fixed
        decode length, so an early think_end would only perturb the measurement."""
        engine, _model, _tokenizer, _events = _engine(
            _varied(40) + [1],
            text_of=TEXT_WITH_MARKERS,
            max_output_tokens=8,
            bench=True,
        )
        chunks = _drain(engine)
        # With no limiter the reasoning block never closes: every emitted token is
        # reasoning, no content follows, and the cap ends the turn.
        assert not any(
            isinstance(c, TokenChunk) and not c.is_thinking and c.text for c in chunks
        )
        assert any(
            isinstance(c, TokenChunk) and c.finish_reason == "length" for c in chunks
        )

    def test_tokenizer_without_single_token_delimiters_skips_the_limiter(self):
        """A tokenizer whose think delimiter is >1 vocab token can never be forced
        by a single id, so the limiter must collapse to None (not crash)."""
        fake = FakeTokenizer(text_of=ANSWER_TEXT)
        fake.__dict__["_think_end_tokens"] = (THINK_END_ID, 999999)
        assert safe_think_token_id(fake, "think_end_id") is None
        assert (
            make_reasoning_budget_limiter(
                safe_think_token_id(fake, "think_start_id"),
                safe_think_token_id(fake, "think_end_id"),
                10,
            )
            is None
        )


# ------------------------------------------------------------- structured code


class TestGuardCleanFailHasAStructuredCode:
    def test_guard_error_chunk_carries_repetition_loop_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        # A tight period-2 cycle: the GUARD fires (well before the reasoning budget).
        stream = [101, 102] * 12
        engine, _model, _tokenizer, _events = _engine(
            stream + [1], text_of=ANSWER_TEXT, max_output_tokens=500
        )
        chunks = _drain(engine)
        errors = [c for c in chunks if isinstance(c, ErrorChunk)]
        assert errors, "the guard must fail the degenerate turn"
        assert errors[0].error_message == DEGENERATION_ERROR_TEXT
        assert errors[0].error_code == REPETITION_LOOP_ERROR_CODE == "repetition_loop"

    def test_a_clean_turn_carries_no_error_code(self):
        engine, _model, _tokenizer, _events = _engine(
            [34, 1], text_of=ANSWER_TEXT, max_output_tokens=8
        )
        chunks = _drain(engine)
        assert not any(isinstance(c, ErrorChunk) for c in chunks)
        assert _text_of(chunks) == "hi"


# ------------------------------------------------------------- session reuse


class TestSessionHistory:
    def test_a_following_request_still_applies_the_limiter(self):
        """The limiter is PER-REQUEST state, initialized from that request's rendered
        prompt -- a second request on the engine builds a fresh one and fires again."""
        engine, _model, _tokenizer, _events = _engine(
            _varied(40) + [1],
            text_of=TEXT_WITH_MARKERS,
            max_output_tokens=8,
        )
        _drain(engine)
        first = next(iter(engine._sessions._entries.values())).session.generated
        assert END in first

        engine.submit(
            TextGeneration(
                task_id=TaskId("task-2"),
                command_id=CommandId("cmd-2"),
                task_params=TextGenerationTaskParams(
                    model=model_id(),
                    input=[
                        InputMessage(role="user", content=InputMessageContent("again"))
                    ],
                    max_output_tokens=8,
                    temperature=0.0,
                ),
                instance_id=InstanceId("inst-1"),
            )
        )
        _drain(engine)
        second = next(iter(engine._sessions._entries.values())).session.generated
        # A fresh per-request limiter still fires on the follow-up request.
        assert END in second

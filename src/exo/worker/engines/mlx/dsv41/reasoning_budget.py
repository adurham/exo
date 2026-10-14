"""Reasoning-budget limiter for the DSv4.1 serve path (round-batched decode).

WHAT THIS IS (PHASE 2 of the degeneration guard). The GENERIC MLX path owns
``make_reasoning_budget_limiter`` (``exo.worker.engines.mlx.generator.generate``):
a logits processor that, once a still-open reasoning block has run for
``budget_tokens`` tokens, forces the model's NEXT sampled token to be the
think-END delimiter -- so reasoning cannot consume the whole generation budget
and leave the answer channel empty. The failure it addresses is confirmed and
source-traced in that function's docstring (DeepSeek-V4-Flash reaches a correct
answer inside reasoning, then falls into a long-period self-doubt loop that
re-derives the same result verbatim; the short-cycle degeneration guard cannot
see it because its period is 60-400+ tokens with paraphrase drift).

The custom DSv4.1 engine deliberately bypasses ``mlx_generate`` /
``ExoBatchGenerator`` (see ``dsv41/engine.py``'s module docstring): it has no
per-token logits-processor pipeline and no sampler at all -- ``rounds._one_round``
argmaxes a whole DSpark draft + verify BATCH at once. So this module is the
round-batched PORT of that limiter, and the two must agree about the invariant
(reasoning must not eat the whole budget), not about the mechanism.

WHY A ROUND NEEDS A DIFFERENT MECHANISM. A round commits a whole batch of tokens
in one go (DSpark draft + verify). If the budget is crossed in the MIDDLE of
that batch, the generic path (one token per forward) would simply never have
produced the tokens past the threshold. Here they have ALREADY been computed and
their rows are in the cache, so the round must:

* commit only the accepted tokens UP TO the threshold,
* DISCARD the remaining accepted tokens AND the round's bonus token (the target's
  own argmax at the first mismatch), rolling the cache back to the committed
  prefix, and
* ensure the NEXT committed token is the think-END delimiter.

THINK_END IS NOT COMMITTED HERE. This module only decides WHERE the batch is cut
and whether a force is pending; ``rounds._one_round`` enacts the rollback (cache
via ``spec.rollback``, draft window via ``head.append_ctx``) and runs the force
as a plain single-row greedy step whose output is overridden to ``think_end_id``.
That is deliberate and load-bearing: the forced ``think_end`` token's OWN KV must
be computed as the FIRST input of the FOLLOWING round (which feeds it as its
anchor), never as already-cached -- treating it as already-cached is an off-by-one
that corrupts the stream silently. The forced round is therefore a COLD START (no
speculative draft), which is also what the DSpark draft window needs: it conditions
on target hidden states, so the first round after a mid-batch cut re-primes rather
than drafting from context a truncated forward no longer reflects.

PURITY / RANK AGREEMENT. ``feed`` is a PURE function of the committed token ids
plus the phase, which is itself initialised from the RENDERED prompt (never
inferred from seeing think markers in output: DeepSeek-V4's chat template appends a
literal `` thinking`` suffix to the prompt, so the model never generates an open
marker of its own). Both TP ranks commit byte-identical tokens under greedy, so
both compute an identical cut and an identical force -- the decision needs no
collective. This is also WHY the generic path's wall-clock ``max_seconds`` trigger
is DELIBERATELY NOT ported: a rank-local clock would let one rank force while its
peer kept decoding, wedging the peer in the model's own TP ``all_sum``. Token-only,
exactly like the 0.75 x budget cap itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from exo.worker.engines.mlx.generator.batch_generate import (
    _REASONING_BUDGET_FRACTION,
    _REASONING_BUDGET_MAX_TOKENS,
)

__all__ = [
    "ReasoningBudgetLimiter",
    "budget_tokens_from_request",
    "make_reasoning_budget_limiter",
]

#: The env knobs the generic call site uses, re-exported so both engines read ONE
#: source of truth for the DEFAULT budget (the request value, when present, wins
#: -- see :func:`budget_tokens_from_request`).
REASONING_BUDGET_FRACTION = _REASONING_BUDGET_FRACTION
REASONING_BUDGET_MAX_TOKENS = _REASONING_BUDGET_MAX_TOKENS


def budget_tokens_from_request(request_value: int | None, max_tokens: int) -> int:
    """The reasoning budget for a request: the request's own cap, else the default.

    An explicit, positive per-request value WINS (a client that sends a cap means
    it). With no request value the default mirrors the generic call site exactly:
    ``min(max_tokens * EXO_REASONING_BUDGET_FRACTION, EXO_REASONING_BUDGET_MAX_TOKENS)``
    -- the fraction keeps real headroom for the answer, the absolute ceiling keeps
    the worst case bounded when ``max_tokens`` resolved to the engine default.
    ``<= 0`` (the fraction knob set to a non-positive value) means DISABLED; the
    caller collapses that to ``limiter is None``, the same convention the generic
    path uses for a disabled knob.
    """
    if request_value is not None and int(request_value) > 0:
        return int(request_value)
    return min(
        int(int(max_tokens) * REASONING_BUDGET_FRACTION),
        int(REASONING_BUDGET_MAX_TOKENS),
    )


@dataclass
class ReasoningBudgetLimiter:
    """Per-request reasoning-budget limiter over the COMMITTED token stream.

    Construct one per request (via :func:`make_reasoning_budget_limiter`) and
    :meth:`feed` it every round's committed batch, in order, WITH the anchor
    first. ``feed`` returns the tokens the round may actually commit (a prefix of
    its batch) and latches :attr:`next_force` when the budget point was reached
    inside that batch -- the caller then runs the next round as a plain step whose
    output is forced to ``think_end_id``.

    Fields are plain so a test can pin them without touching the process env.
    """

    think_start_id: int
    think_end_id: int
    budget_tokens: int
    starts_in_thinking: bool = False
    #: Committed reasoning tokens in the currently-open block (0 when closed).
    elapsed: int = field(default=0, init=False)
    #: Whether a thinking block is currently open (prompt-implied or an explicit
    #: think_start committed since).
    in_think: bool = field(default=False, init=False)
    #: ``think_end_id`` when the NEXT round's output must be forced to it, else None.
    next_force: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # The rendered prompt may already sit inside a reasoning block (the chat
        # template's `` thinking`` suffix); the caller computes that once and hands
        # it in. The anchor is the first GENERATED token, so it is fed via
        # :meth:`feed`, not counted here.
        self.in_think = bool(self.starts_in_thinking)
        self.elapsed = 0

    def _apply(self, token: int) -> None:
        """Update the phase for ONE committed token (no budget check)."""
        if token == self.think_end_id:
            if self.in_think:
                self.in_think = False
                self.elapsed = 0
            return
        if token == self.think_start_id:
            # An explicit open (or a re-open): the window restarts here, exactly as
            # the generic limiter's "scan for the most recent think_start" does.
            self.in_think = True
            self.elapsed = 0
            return
        if self.in_think:
            self.elapsed += 1

    def feed(self, batch: list[int]) -> list[int]:
        """Admit ONE round's committed batch; returns the prefix that may commit.

        Latches :attr:`next_force` to ``think_end_id`` iff the reasoning budget is
        reached inside this batch with the block STILL open. The token at the
        threshold is the LAST admitted; every token after it -- including the
        round's bonus token when the cut lands on it -- is dropped, and the NEXT
        round must emit the forced ``think_end``.
        """
        self.next_force = None
        kept: list[int] = []
        for raw in batch:
            token = int(raw)
            if (
                self.in_think
                and self.elapsed >= self.budget_tokens
                and token != self.think_end_id
                and token != self.think_start_id
            ):
                # Budget spent and the block is still open while the batch is
                # still producing reasoning tokens: drop this token and the rest.
                # (Unreachable while the force is always consumed by the caller;
                # kept as a guard so a caller that ignores next_force cannot let
                # reasoning run past its budget.)
                self.next_force = self.think_end_id
                return kept
            # A natural think_end closes the block (and is admissible); a
            # think_start restarts the window -- both are handled by _apply.
            self._apply(token)
            kept.append(token)
            if self.in_think and self.elapsed >= self.budget_tokens:
                # The budget point is reached AT this token, so it is the last
                # reasoning token to commit: everything after it in the batch is
                # dropped and the NEXT committed token is forced to think_end.
                self.next_force = self.think_end_id
                return kept
        return kept


def make_reasoning_budget_limiter(
    think_start_id: int | None,
    think_end_id: int | None,
    budget_tokens: int,
    starts_in_thinking: bool = False,
) -> ReasoningBudgetLimiter | None:
    """A round-batched reasoning-budget limiter, or None when inapplicable.

    Mirrors the generic factory's "collapse a disabled/inapplicable knob to None"
    contract: None when the tokenizer exposes no single-token think delimiters, or
    when ``budget_tokens <= 0`` -- so the caller skips the limiter entirely at zero
    cost, exactly as the generic path skips adding its processor.

    ``prompt_token_count`` is NOT a parameter here (unlike the generic factory):
    this limiter counts GENERATED tokens from the anchor by construction, so a long
    prompt cannot eat into the budget in the first place.
    """
    if think_start_id is None or think_end_id is None:
        return None
    if budget_tokens <= 0:
        return None
    return ReasoningBudgetLimiter(
        think_start_id=int(think_start_id),
        think_end_id=int(think_end_id),
        budget_tokens=int(budget_tokens),
        starts_in_thinking=bool(starts_in_thinking),
    )

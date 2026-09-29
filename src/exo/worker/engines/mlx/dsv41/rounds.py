"""Request plumbing for the DSv4.1 engine: gating, response construction and
the decode round.

WHY THIS IS ITS OWN MODULE. ``engine.py`` is written against these helpers --
its call sites read ``_refuse_unsupported(params, ...)``,
``_one_round(self, model=..., ...)``, ``_final_response(...)``, etc. -- but the
scaffolding commit that introduced the engine never committed them, so the
module was complete only as a sketch. They live here rather than inline in
``engine.py`` so that the engine keeps short, readable call sites, and so the
round/gating logic stays reviewable and unit-testable without a GPU or a
checkpoint. ``engine.py`` pulls them into its namespace with one explicit
import, which is also the seam that keeps ``_one_round`` from needing a
circular import back into the engine class.

ROUND STRUCTURE mirrors the harnesses that validated DSpark on this
checkpoint (``p56``/``p63``: draft -> ONE verify forward -> accept the
matching prefix -> roll back -> feed taps), so the engine and the harness
cannot drift apart silently: the same ``mlx_lm.models.deepseek_v41.spec``
``snap``/``stashes``/``rollback`` pair is used, in the same order.

GREEDY VS SPECULATIVE. With no draft head the round is a single-row forward
and one token (this is byte-for-byte plain greedy decode). With a head the
round drafts ``gamma`` tokens and verifies them in one ``gamma + 1``-row
forward; only tokens the target model itself confirms are ever emitted, so
speculation can change the text relative to greedy only through the
documented chunk-shape effect of the body, never through an unverified draft.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.dsv41.errors import Dsv41UnsupportedFeature
from exo.worker.engines.mlx.dsv41.session import Dsv41Sessions, TurnOutcome
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids a circular import)
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine

#: Fallback cache capacity when neither the instance cap nor the checkpoint's
#: ``max_seq_len`` is available. Deliberately small: a wrong-large value would
#: let a request build a cache the machine cannot hold.
_DEFAULT_CACHE_TOKENS = 8192

#: Key for the single DSpark draft window kept alive for a request. The head's
#: context is continuous over the request's tokens, so one window per request
#: is what the round needs -- see ``_one_round``.
_DRAFT_KEY = 0


# --------------------------------------------------------------- request gating


def _refuse_unsupported(
    params: TextGenerationTaskParams, *, vision_available: bool
) -> None:
    """Refuse, loudly, anything this build cannot honour honestly.

    Every branch here is a feature exo has and DSv4.1 does not (yet). Silently
    ignoring one would corrupt the client's assumptions invisibly -- an image
    dropped from the prompt, logprobs that are not computed -- so each one fails
    the request with a reason the client can act on.

    Prefix reuse is NOT refused any more: it is served by the conversation
    sessions in ``dsv41/session.py`` (one live ``ModelCache`` per conversation),
    which is what ``use_prefix_cache`` means for this engine.
    """
    if params.images and not vision_available:
        raise Dsv41UnsupportedFeature(
            "DSv4.1: this request carries image(s), but no vision tower is "
            "attached to this instance. The checkpoint has one (a DeepSeek-ViT "
            "tower plus an aligner, and a real <|deepseek_image|> placeholder "
            "token), but the exo-side wiring is not part of this build, so the "
            "honest answer is to refuse rather than silently answer from text "
            "only. Drop the images or route to an instance with vision."
        )
    if params.logprobs or params.top_logprobs:
        raise Dsv41UnsupportedFeature(
            "DSv4.1: logprobs were requested, but the DSv4.1 decode path "
            "returns no per-token logprobs (the round resolves argmaxes). "
            "Refusing rather than returning an empty logprobs block that a "
            "client would read as 'no alternatives'."
        )


def _cache_capacity(engine: Dsv41Engine) -> int:
    """Tokens this instance's cache may hold.

    Order: the instance's own cap (``max_kv_tokens``, set by placement from
    the card/instance config), else the checkpoint's ``max_seq_len`` (1M for
    this model), else a conservative default. The engine needs this BEFORE
    allocating the cache: ``make_cache`` preallocates to the requested length,
    so asking for the checkpoint maximum on a small machine is an OOM at
    request time instead of a clean refusal.
    """
    cap = getattr(engine, "max_kv_tokens", None)
    if cap is not None and int(cap) > 0:
        return int(cap)
    args = getattr(getattr(engine, "loaded", None), "args", None)
    model_max = getattr(args, "max_seq_len", 0) or 0
    return int(model_max) if int(model_max) > 0 else _DEFAULT_CACHE_TOKENS


# --------------------------------------------------------------- prompt / stream


def _queue_of(
    source: Generator[GenerationResponse],
) -> Generator[GenerationResponse | None]:
    """Adapt the engine's token generator to the parser pipeline's shape.

    The parser pipeline (thinking split, DSML block detection, tool-call
    accumulation) is written against the production stream's shape: the
    engine's queue yields ``None`` whenever it is momentarily empty, which is
    what tells the parsers "this is a flush point, emit what you are holding"
    (``GeneratorQueue.gen`` in ``batch_generate.py``). Interleaving a ``None``
    after every real response reproduces exactly that contract, which matters
    for more than tidiness: without the sentinel, a partially-buffered marker
    would never be flushed, and ``step()`` drains the parser until it sees a
    ``None`` -- so a run of buffered tokens would be held to end-of-turn
    instead of streamed.
    """
    for response in source:
        yield response
        yield None


def _stop_sequences(params: TextGenerationTaskParams) -> tuple[str, ...]:
    """Client stop strings, normalized (same semantics as the MLX generator)."""
    stop = params.stop
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


def _stop_index(text: str, sequences: tuple[str, ...]) -> int | None:
    """Index of the EARLIEST stop-sequence occurrence in ``text``, else None.

    Earliest, not first-listed: when several stop strings are set, the one the
    model actually reaches first is what ends the turn, whichever order the
    client listed them in.
    """
    best: int | None = None
    for sequence in sequences:
        if not sequence:
            continue
        found = text.find(sequence)
        if found != -1 and (best is None or found < best):
            best = found
    return best


# --------------------------------------------------------------- response shape


def _mid_response(token: int, text: str, task_id: TaskId | None) -> GenerationResponse:
    """A non-terminal response: text for the client, no usage.

    Usage/stats are attached only to the terminal response, exactly as exo's
    own generators do -- the OpenAI adapter reads usage off the final chunk,
    and a mid-stream response carrying a half-counted usage would be worse
    than none.
    """
    del task_id  # only used by the runner's per-task chunk correlation
    return GenerationResponse(text=text, token=token, usage=None)


def _final_response(
    *,
    token: int,
    text: str,
    prefill_tps: float,
    prompt_tokens: int,
    generated: int,
    reason: FinishReason,
    task_id: TaskId | None = None,
    round_stats: Any | None = None,
    reused_tokens: int = 0,
    prefill_tokens: int | None = None,
) -> GenerationResponse:
    """The terminal response for a request, with usage and stats attached.

    ``prompt_tokens`` is the WHOLE conversation this turn belongs to;
    ``reused_tokens`` is the part that came from the session's live cache and
    ``prefill_tokens`` the rows actually fed (``None`` => the whole prompt).
    That split is what a client reads to see the multi-turn win, and it is also
    what ``GenerationStats.prompt_tps`` is computed against.
    """
    del task_id, prefill_tokens  # the split is reported through usage/stats below
    hit = "partial" if reused_tokens > 0 else "none"
    if round_stats is not None:
        stats = round_stats.stats(prefill_tps, prompt_tokens, generated).model_copy(
            update={"prefix_cache_hit": hit}
        )
    else:
        stats = GenerationStats(
            prompt_tps=prefill_tps,
            generation_tps=0.0,
            prompt_tokens=prompt_tokens,
            generation_tokens=generated,
            peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
            prefix_cache_hit=hit,
        )
    return GenerationResponse(
        text=text,
        token=token,
        finish_reason=reason,
        stats=stats,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=generated,
            total_tokens=prompt_tokens + generated,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=reused_tokens),
            # count_reasoning_tokens patches reasoning_tokens in on the way
            # out; it starts at 0 here like every other exo generator.
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        ),
    )


# --------------------------------------------------------------- decode rounds


def rows_fed(token: int, committed: list[int], accepted: int, head: Any | None) -> list[int]:
    """The rows one round actually fed and kept, in order.

    The verify forward feeds the anchor plus EVERY drafted token, then rolls the
    rejected suffix back, so the rows that survive are the anchor plus the first
    ``accepted`` drafts -- which is exactly ``len(committed) - 1`` tokens (the
    round also commits the target's own token at the first mismatch, which is NOT
    a fed row: the next round feeds it). With no draft head the round feeds one
    row and commits one token.
    """
    if head is None or accepted >= len(committed):
        return [int(token)]
    return [int(token), *[int(t) for t in committed[:accepted]]]


def _spec_policy(gamma: int) -> Any:
    """Adaptive gamma policy for the speculative round (see ``spec.GammaPolicy``)."""
    from mlx_lm.models.deepseek_v41.spec import GammaPolicy

    return GammaPolicy(start=gamma)


def _one_round(
    engine: Dsv41Engine,
    *,
    model: Any,
    cache: Any,
    token: int,
    head: Any | None,
    policy: Any | None,
    draft_state: Any | None = None,
) -> tuple[list[int], float, int, int]:
    """One decode round: returns ``(tokens, ms, accepted, gamma)``.

    ``tokens`` are the tokens COMMITTED by this round (always the target
    model's own argmaxes -- never an unverified draft), which is what the
    engine emits. Greedy when ``head`` is None; otherwise DSpark draft +
    chunk verify.

    Rows fed vs tokens committed: a round always feeds its anchor (``token``) and
    keeps a prefix of its drafts. The caller gets that split from
    :func:`rows_fed` so its token history can be kept in step with the cache
    across the rollback.

    Draft-context priming. The DSpark head drafts from its own window of
    context taps, and the harnesses prime it from the PREFILL forward's taps
    (``return_taps=True``). This engine's prefill deliberately does not carry
    taps (it streams a fenced chunk loop and only ever keeps the last logits),
    so the first round with a head loaded primes the window from that round's
    own taps and drafts from the second round on: one warm round per request,
    no correctness consequence (the target verifies every draft) and no
    change to the prefill path. If a future revision passes the prefill taps
    in, the only change here is to append them before the first draft.
    """
    started = time.perf_counter()
    anchor = mx.array([token], dtype=mx.int32).reshape(1, 1)
    if head is None:
        logits = model(anchor, cache, last_logit_only=True)
        next_token = int(mx.argmax(logits.reshape(-1), axis=-1).item())
        return [next_token], (time.perf_counter() - started) * 1e3, 1, 1

    from mlx_lm.models.deepseek_v41 import spec as SP

    taps_ids: list[int] = list(model.args.dspark_target_layer_ids)

    def tapcat(taps: dict[int, mx.array]) -> mx.array:
        return mx.concatenate([taps[layer] for layer in taps_ids], axis=-1)

    # ``draft_state`` is the conversation's draft window when the caller owns one
    # (``session.Conversation``, which keeps it in step with the body cache); a
    # bare engine falls back to the per-request window in ``_draft_windows``.
    if draft_state is None:
        draft_state = engine._draft_windows.get(_DRAFT_KEY)

    if draft_state is None:
        # Round 1 with a head: no draft window yet. Step plainly, but keep the
        # taps so round 2 can draft.
        logits, taps = model(anchor, cache, last_logit_only=True, return_taps=True)
        next_token = int(mx.argmax(logits.reshape(-1), axis=-1).item())
        mx.eval(next_token)
        draft_state = head.make_cache(1)
        head.append_ctx(tapcat(taps), draft_state)
        engine._draft_windows[_DRAFT_KEY] = draft_state
        logger.info(
            "[DSV41] DSpark draft window primed from the first decode step; "
            "drafting from the next round (the prefill forward carries no taps "
            "-- see rounds._one_round)."
        )
        return [next_token], (time.perf_counter() - started) * 1e3, 1, 1

    gamma = int(policy.next()) if policy is not None else 1
    position = int(cache.offset)
    drafted = head.draft(
        anchor,
        getattr(model, "embed", None),
        getattr(model, "head", None),
        draft_state,
        width=gamma,
    )
    drafted = drafted.astype(mx.int32)
    verify_in = mx.concatenate([anchor.reshape(1, 1), drafted.reshape(1, gamma)], axis=1)
    snapshot = SP.snap(cache, position)
    logits, taps = model(verify_in, cache, return_taps=True, argmax=True)
    mx.eval(logits)
    stashes = SP.stashes(cache)
    target = [int(v) for v in logits[0]]
    draft = [int(v) for v in drafted[0]]

    accepted = 0
    while accepted < gamma and accepted < len(target) and target[accepted] == draft[accepted]:
        accepted += 1
    # The token at the first mismatch is the target's own argmax -- it is
    # committed along with the accepted drafts, so a round always commits at
    # least one token and the cache lands on a position the target produced.
    committed = draft[:accepted] + [target[accepted]]
    committed_position = position + accepted + 1
    SP.rollback(cache, snapshot, committed_position, stashes)
    head.append_ctx(tapcat(taps)[:, : accepted + 1], draft_state)
    return (
        committed,
        (time.perf_counter() - started) * 1e3,
        accepted,
        gamma,
    )

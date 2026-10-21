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
from exo.worker.engines.mlx.dsv41.sampling import (
    Sampling,
    sample_one,
    sample_prefix,
    sampled_logprobs,
)
from exo.worker.engines.mlx.dsv41.no_repeat import (
    NEG_INF,
    BanTurnStats,
    NoRepeatNgram,
    argmax_after_ban_detailed,
)
from exo.worker.runner.bootstrap import logger

from mlx_lm.profiler import span

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids a circular import)
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
    from exo.worker.engines.mlx.dsv41.reasoning_budget import (
        ReasoningBudgetLimiter,
    )

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


def _mid_response(
    token: int, text: str, task_id: TaskId | None, logprob: Any = None
) -> GenerationResponse:
    """A non-terminal response: text for the client, no usage.

    Usage/stats are attached only to the terminal response, exactly as exo's
    own generators do -- the OpenAI adapter reads usage off the final chunk,
    and a mid-stream response carrying a half-counted usage would be worse
    than none.
    """
    del task_id  # only used by the runner's per-task chunk correlation
    sel, top = logprob if logprob is not None else (None, None)
    return GenerationResponse(
        text=text, token=token, usage=None, logprob=sel, top_logprobs=top
    )


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
    logprob: Any = None,
    mtp_cycles: int = 0,
    mtp_accepted: int = 0,
    error_code: str | None = None,
    loop_cause: Any | None = None,
) -> GenerationResponse:
    """The terminal response for a request, with usage and stats attached.

    ``prompt_tokens`` is the WHOLE conversation this turn belongs to;
    ``reused_tokens`` is the part that came from the session's live cache and
    ``prefill_tokens`` the rows actually fed (``None`` => the whole prompt).
    That split is what a client reads to see the multi-turn win, and it is also
    what ``GenerationStats.prompt_tps`` is computed against.

    ``mtp_cycles``/``mtp_accepted`` are the session-cumulative speculative
    counters (same meaning as the batch-generator's ``mtp_*_cumulative``:
    deltas across successive requests give the live acceptance rate).

    ``error_code`` is the OPTIONAL OpenAI-style structured code carried on an
    ``finish_reason="error"`` terminal (``map_responses_to_chunks`` copies it
    onto the ``ErrorChunk``). None for an ordinary terminal, so clients that
    only read the message are unaffected.
    """
    del task_id, prefill_tokens  # the split is reported through usage/stats below
    hit = "partial" if reused_tokens > 0 else "none"
    if round_stats is not None:
        stats = round_stats.stats(prefill_tps, prompt_tokens, generated).model_copy(
            update={
                "prefix_cache_hit": hit,
                "mtp_cycles_cumulative": mtp_cycles,
                "mtp_accepted_drafts_cumulative": mtp_accepted,
            }
        )
    else:
        stats = GenerationStats(
            prompt_tps=prefill_tps,
            generation_tps=0.0,
            prompt_tokens=prompt_tokens,
            generation_tokens=generated,
            peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
            prefix_cache_hit=hit,
            mtp_cycles_cumulative=mtp_cycles,
            mtp_accepted_drafts_cumulative=mtp_accepted,
        )
    sel, top = logprob if logprob is not None else (None, None)
    return GenerationResponse(
        loop_cause=loop_cause,
        text=text,
        token=token,
        logprob=sel,
        top_logprobs=top,
        finish_reason=reason,
        stats=stats,
        error_code=error_code,
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


def _row_logprobs(logits: mx.array, k: int, lp_out: list[Any] | None) -> None:
    """Log-probs of the greedy token of one full logits row, into ``lp_out``."""
    if not k or lp_out is None:
        return
    from mlx_lm.models.deepseek_v41 import logprobs as _lp

    _ids, sel, tid, tlp = _lp.from_logits(logits.reshape(1, -1), k)
    mx.eval(sel, tid, tlp)
    lp_out.append((float(sel[0].item()), tid[0].tolist(), tlp[0].tolist()))


def _rows_logprobs(lp: dict[str, mx.array], n: int, lp_out: list[Any]) -> None:
    """The first ``n`` verify rows' log-probs (batch row 0), into ``lp_out``."""
    sel = lp["selected"].reshape(-1)[:n]
    tid = lp["top_ids"].reshape(-1, lp["top_ids"].shape[-1])[:n]
    tlp = lp["top_logprobs"].reshape(-1, lp["top_logprobs"].shape[-1])[:n]
    mx.eval(sel, tid, tlp)
    for s_, i_, l_ in zip(sel.tolist(), tid.tolist(), tlp.tolist(), strict=True):
        lp_out.append((float(s_), i_, l_))


def _spec_policy(gamma: int) -> Any:
    """Adaptive gamma policy for the speculative round (see ``spec.GammaPolicy``)."""
    from mlx_lm.models.deepseek_v41.spec import GammaPolicy

    return GammaPolicy(start=gamma)


def _row_penalties(
    ban: NoRepeatNgram | None, history: list[int] | None
) -> dict[int, float] | None:
    """The escape's per-row penalties for ONE row (None when the knob is off)."""
    if ban is None or not ban.enabled:
        return None
    return ban.penalties(history or []) or None


def _sample_one_row(
    logits: mx.array,
    sampling: Any,
    ban: NoRepeatNgram | None,
    history: list[int] | None,
    stats: BanTurnStats | None = None,
) -> int:
    """One sampled token (non-speculative row), with the escape's penalties applied."""
    penalties = _row_penalties(ban, history)
    token = sample_one(logits, sampling, banned=penalties)
    if stats is not None and penalties:
        stats.masked_total += len(penalties)
        raw = int(mx.argmax(logits.reshape(-1), axis=-1).item())
        stats.rows_changed += int(token != raw)
    return token


def _sampled_row_logprobs(
    logits: mx.array,
    committed: list[int],
    sampling: Any,
    ban: NoRepeatNgram | None,
    history: list[int] | None,
    k: int,
    lp_out: list[Any] | None,
) -> None:
    """Log-probs of SAMPLED tokens, read off the transformed row."""
    if not k or lp_out is None:
        return
    lp_out.extend(
        sampled_logprobs(
            logits,
            committed,
            sampling,
            banned_per_row=[_row_penalties(ban, history)],
            k=k,
        )
    )


def _greedy_token(
    logits: mx.array,
    ban: NoRepeatNgram | None,
    history: list[int] | None,
    stats: BanTurnStats | None = None,
) -> int:
    """This row's greedy token: plain argmax, or the argmax over BANNED logits.

    With the ban disabled (``ban`` None, or ``n < MIN_NGRAM``) this is EXACTLY the
    pre-ban expression ``int(mx.argmax(logits.reshape(-1), axis=-1).item())``, so
    the default path stays byte-identical. With the ban on, the single row is
    masked (``-inf`` on every id that would close a repeated n-gram of the produced
    history) before the argmax -- see ``dsv41/no_repeat.py`` for the why.

    ``stats`` (optional) accumulates the turn telemetry: one committed row is
    counted, its mask size added, and ``rows_changed`` bumped only when the ban
    actually moved the argmax. The raw argmax used for that comparison is computed
    inside the ban helper, never here.
    """
    if ban is None or not ban.enabled:
        return int(mx.argmax(logits.reshape(-1), axis=-1).item())
    banned = [ban.penalties(history or [])]
    ids, _masked, changed, masked_counts = argmax_after_ban_detailed(
        logits.reshape(1, -1), banned
    )
    if stats is not None:
        stats.rows_changed += int(changed[0])
        stats.masked_total += int(masked_counts[0])
    return int(ids[0])


def _assert_full_vocab_row(logits: mx.array, model: Any) -> None:
    """The ban path must run on the FULL vocab row on EVERY rank.

    With the ban on, ``_one_round`` asks the verify forward for ``argmax=False``
    rows -- the path that rebuilds the full vocab row on each rank instead of the
    sharded ``combine_argmax``. A shard (narrower last dim than the model's vocab)
    cannot be masked identically across rank boundaries without a collective, which
    is the one thing this ban exists to avoid, so FAIL LOUD rather than silently
    mask a partial row (a wrong-vocab mask would ban the wrong ids per rank).
    """
    vocab = int(getattr(getattr(model, "args", None), "vocab_size", 0) or 0)
    width = int(logits.shape[-1])
    if vocab and width != vocab:
        raise RuntimeError(
            f"DSV4.1: the no-repeat n-gram ban needs the FULL vocab row on every "
            f"rank (the argmax=False verify forward), but it returned {width} "
            f"columns for a vocab of {vocab}. A sharded row cannot be masked "
            f"identically across ranks without a collective -- refusing to decode."
        )


def _ensure_capacity_for_round(cache: Any, rows: int) -> None:
    """Grow the mlx-lm cache BEFORE this round's forward graph is built.

    ``cache`` is the raw ``mlx_lm.models.deepseek_v41.cache.ModelCache``
    (``session.cache.cache``). Growth reallocates the compressed/index/engram
    buffers, so it must happen at an eval-clean boundary -- before the round
    builds its forward, never mid-graph. ``rows`` is the worst case this round
    can write: 1 for the greedy single-row forward, ``1 + gamma`` for the
    draft + verify forward (the verify feeds the anchor plus every drafted row
    before the rollback trims the rejected suffix).

    A cache double / older fork without growth support is a no-op; the mlx-lm
    ``CapacityError`` (and any allocator failure, which ``ensure_capacity``
    re-raises as one) maps to the engine's ``Dsv41UnsupportedFeature`` refusal
    so a too-long request fails cleanly instead of crashing the runner.
    """
    ensure = getattr(cache, "ensure_capacity", None)
    if ensure is None:
        return
    from mlx_lm.models.deepseek_v41.cache import CapacityError as _CapErr

    try:
        ensure(int(cache.offset) + int(rows))
    except _CapErr as e:
        raise Dsv41UnsupportedFeature(
            f"DSV4.1 out of context: cache cannot hold "
            f"{int(cache.offset) + int(rows)} tokens ({e})"
        ) from e


def _one_round(
    engine: Dsv41Engine,
    *,
    model: Any,
    cache: Any,
    token: int,
    head: Any | None,
    policy: Any | None,
    draft_state: Any | None = None,
    logprobs: int = 0,
    lp_out: list[Any] | None = None,
    limiter: "ReasoningBudgetLimiter | None" = None,
    force_think_end: int | None = None,
    ban: NoRepeatNgram | None = None,
    history: list[int] | None = None,
    ban_stats: BanTurnStats | None = None,
    sampling: Sampling | None = None,
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

    ``limiter`` (the round-batched reasoning-budget limiter, see
    ``dsv41/reasoning_budget.py``): when set, this round's committed batch is
    ADMITTED through it and may be cut short at the reasoning-budget threshold
    (the tokens past the cut, including the round's bonus token, are dropped and
    the cache/draft rollback below follows the SHORTENED batch). ``force_think_end``
    (a token id) makes THIS round a plain single-row STEP -- no speculative draft,
    the cold start the limiter requires after a cut -- whose output is overridden
    to that id. The forced token's own KV is deliberately NOT computed here: the
    NEXT round feeds it as its anchor (see the module docstring's off-by-one note).

    Draft-context priming. The DSpark head drafts from its own window of
    context taps, and the harnesses prime it from the PREFILL forward's taps
    (``return_taps=True``). This engine's prefill deliberately does not carry
    taps (it streams a fenced chunk loop and only ever keeps the last logits),
    so the first round with a head loaded primes the window from that round's
    own taps and drafts from the second round on: one warm round per request,
    no correctness consequence (the target verifies every draft) and no
    change to the prefill path. If a future revision passes the prefill taps
    in, the only change here is to append them before the first draft.

    ``ban``/``history`` (the no-repeat n-gram escape, ``dsv41/no_repeat.py``):
    ``history`` is the request's produced-token list so far, ENDING with
    ``token`` (the row this round feeds); ``ban`` is the resolved knob. When the
    ban is on, every row that the round can commit argmaxes over logits masked
    (-inf) on the ids that would close a repeated n-gram of that row's own
    history -- applied BEFORE the argmax, over the whole verify batch. THE TARGET
    CHOICE IS ALWAYS THE MASKED ARGMAX: the draft-accept comparison, the token
    committed at the first mismatch and the all-accepted bonus all read the same
    masked ``target``, and the log-prob of a committed row is read off the MASKED
    row too. No raw/sharded argmax is consulted on the enabled path, so an
    absorbing loop whose draft agrees with the UNMASKED target is still rejected --
    otherwise the ban would never fire. ``ban_stats`` accumulates the turn
    telemetry (see :func:`_greedy_token`); both are optional and default to "off":
    with ``ban`` None (or ``n < MIN_NGRAM``) this round is byte-for-byte the
    pre-ban round.
    """
    started = time.perf_counter()
    anchor = mx.array([token], dtype=mx.int32).reshape(1, 1)

    if force_think_end is not None:
        # FORCED round: one plain greedy row whose output is the think-END
        # delimiter. It is a COLD START (no DSpark draft) on purpose -- the draft
        # window conditions on target hidden states, and the round that tripped the
        # budget was cut mid-batch, so the first post-force round re-primes rather
        # than drafting from context a truncated forward no longer reflects.
        _ensure_capacity_for_round(cache, 1)
        if head is None:
            model(anchor, cache, last_logit_only=True)
        else:
            _, taps = model(anchor, cache, last_logit_only=True, return_taps=True)
            taps_ids = list(model.args.dspark_target_layer_ids)
            cat = mx.concatenate([taps[layer] for layer in taps_ids], axis=-1)
            if draft_state is None:
                draft_state = engine._draft_windows.get(_DRAFT_KEY)
            if draft_state is None:
                draft_state = head.make_cache(1)
                engine._draft_windows[_DRAFT_KEY] = draft_state
            # Keep the draft window in step with the committed prefix (this row's
            # tap is the anchor the forced token sits on) -- one row, no draft.
            head.append_ctx(cat[:, :1], draft_state)
        if limiter is not None:
            # Feed the forced token so the phase CLOSES the block and next_force is
            # cleared; it does not re-trigger (think_end is always admissible).
            limiter.feed([int(force_think_end)])
        return [int(force_think_end)], (time.perf_counter() - started) * 1e3, 0, 1

    if head is None:
        # Plain greedy: one row in, one token out. Grow for that row first.
        _ensure_capacity_for_round(cache, 1)
        logits = model(anchor, cache, last_logit_only=True)
        _sampling_on = sampling is not None and not sampling.greedy
        if _sampling_on:
            next_token = _sample_one_row(logits, sampling, ban, history, ban_stats)
            _sampled_row_logprobs(
                logits, [next_token], sampling, ban, history, logprobs, lp_out
            )
        else:
            next_token = _greedy_token(logits, ban, history, ban_stats)
            _row_logprobs(logits, logprobs, lp_out)
        committed = [next_token]
        if limiter is not None:
            committed = limiter.feed(committed)
        return committed, (time.perf_counter() - started) * 1e3, 1, 1

    from mlx_lm.models.deepseek_v41 import spec

    taps_ids = list(model.args.dspark_target_layer_ids)

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
        _ensure_capacity_for_round(cache, 1)
        logits, taps = model(anchor, cache, last_logit_only=True, return_taps=True)
        _sampling_on = sampling is not None and not sampling.greedy
        if _sampling_on:
            next_token = _sample_one_row(logits, sampling, ban, history, ban_stats)
            _sampled_row_logprobs(
                logits, [next_token], sampling, ban, history, logprobs, lp_out
            )
        else:
            next_token = _greedy_token(logits, ban, history, ban_stats)
            _row_logprobs(logits, logprobs, lp_out)
        draft_state = head.make_cache(1)
        head.append_ctx(tapcat(taps), draft_state)
        engine._draft_windows[_DRAFT_KEY] = draft_state
        logger.info(
            "[DSV41] DSpark draft window primed from the first decode step; "
            "drafting from the next round (the prefill forward carries no taps "
            "-- see rounds._one_round)."
        )
        committed = [next_token]
        if limiter is not None:
            committed = limiter.feed(committed)
        return committed, (time.perf_counter() - started) * 1e3, 1, 1

    gamma = int(policy.next()) if policy is not None else 1
    # Speculative verify feeds the anchor + EVERY drafted row in ONE forward
    # (1 + gamma rows) before the rollback trims the rejected suffix, so grow
    # for that whole worst case here, at the eval-clean boundary.
    with span("round.cache_grow"):
        _ensure_capacity_for_round(cache, 1 + gamma)
    position = int(cache.offset)
    with span("round.draft"):
        drafted = head.draft(
            anchor.reshape(-1),  # DSparkHead.draft takes [b] anchor ids
            getattr(model, "embed", None),
            getattr(model, "head", None),
            draft_state,
            width=gamma,
        )
    # The real DSparkHead.draft returns (tokens, per-position confidence).
    if isinstance(drafted, tuple):
        drafted = drafted[0]
    drafted = drafted.astype(mx.int32)
    verify_in = mx.concatenate(
        [anchor.reshape(1, 1), drafted.reshape(1, gamma)], axis=1
    )
    # 1 + gamma: the verify forward feeds the anchor plus EVERY drafted row, and
    # the snapshot must cover the ring slots of all of them or a rejection cannot
    # put the overwritten slots back (spec.snap/rollback).
    with span("round.snap"):
        snapshot = spec.snap(cache, position, 1 + gamma)
    # ── NO-REPEAT N-GRAM: ask for the RAW rows when the ban is on ───────────
    # The mask has to land BETWEEN the head and the argmax, so the ban path
    # requests logits (argmax=False) instead of ids. That returns the full vocab
    # row on EVERY rank -- ``argmax=True`` would take the sharded ``combine_argmax``
    # path instead -- which is exactly what makes the two TP ranks mask the same
    # row and argmax the same id with NO collective. The extra memory is bounded:
    # (1 + gamma) rows of one vocab row, only while the knob is enabled.
    ban_policy = ban if (ban is not None and ban.enabled) else None
    sampling_on = sampling is not None and not sampling.greedy
    with span("round.verify"):
        if ban_policy is not None or sampling_on:
            # Both the escape and sampling need the FULL vocab rows (argmax=False):
            # the mask/penalty has to land before the argmax or the distribution, and
            # the full row is what makes the two TP ranks agree with no collective.
            logits, taps = model(verify_in, cache, return_taps=True, argmax=False)
            lp = None
        elif logprobs:
            logits, taps, lp = model(verify_in, cache, return_taps=True, argmax=True,
                                     logprobs=logprobs)
        else:
            logits, taps = model(verify_in, cache, return_taps=True, argmax=True)
            lp = None
        with span("round.eval"):
            mx.eval(logits)
    stashes = spec.stashes(cache)
    draft = [int(v) for v in drafted[0]]
    masked_rows: mx.array | None = None
    changed: list[bool] = []
    masked_counts: list[int] = []
    if ban_policy is not None:
        # The ban path runs on the FULL vocab row (argmax=False). A shard here
        # means the model took the wrong forward -- fail loud before masking.
        _assert_full_vocab_row(logits, model)
        # Per-row ban over the whole verify batch (row 0 after the produced
        # history's last id, row i after the i-th fed draft token -- see
        # ``NoRepeatNgram.banned_per_row``), applied BEFORE the argmax. A row
        # whose every candidate is banned falls back to its unmasked argmax, so a
        # round can never emit an invalid token. ``target`` is THE masked argmax:
        # the accept loop, the first-mismatch commit and the bonus all read it.
        target, masked_rows, changed, masked_counts = argmax_after_ban_detailed(
            logits[0], ban_policy.penalties_per_row(history or [], draft)
        )
    else:
        target = [int(v) for v in logits[0]]

    if sampling_on:
        # Rejection sampling over the verify batch: accept a draft with
        # probability p(draft); at the first rejection draw the residual.
        penalties = (
            ban_policy.penalties_per_row(history or [], draft)
            if ban_policy is not None
            else None
        )
        engine._sampling_step += 1
        committed, accepted = sample_prefix(
            logits[0], draft, sampling=sampling, banned_per_row=penalties,
            step=engine._sampling_step,
        )
        if ban_policy is not None and ban_stats is not None:
            raw_ids = [
                int(v) for v in mx.argmax(logits[0], axis=-1).tolist()
            ][: len(committed)]
            ban_stats.masked_total += sum(
                len(p) for p in (penalties or [])[: len(committed)]
            )
            ban_stats.rows_changed += sum(
                1
                for i, t in enumerate(committed)
                if i < len(raw_ids) and t != raw_ids[i]
            )
        if logprobs and lp_out is not None:
            lp_out.extend(
                sampled_logprobs(
                    logits[0],
                    committed,
                    sampling,
                    banned_per_row=penalties,
                    k=logprobs,
                )
            )
    else:
        accepted = 0
        while (
            accepted < gamma
            and accepted < len(target)
            and target[accepted] == draft[accepted]
        ):
            accepted += 1
    # The token at the first mismatch is the target's own argmax -- it is
    # committed along with the accepted drafts, so a round always commits at
    # least one token and the cache lands on a position the target produced. A
    # draft that ran out of script (``len(draft) < gamma``) is exhausted: the
    # round then commits what it did produce plus the target's answer for the next
    # position, so the decoded count still advances by one.
    if not sampling_on:
        committed = draft[:accepted] + [target[min(accepted, len(target) - 1)]]
    if limiter is not None:
        # Reasoning-budget cut: admit only up to the threshold. A SHORTER batch
        # means the rollback target and the draft-window append below both follow
        # the shortened committed prefix (the tokens past the cut -- including the
        # bonus -- are dropped, never committed, and rolled out of the cache).
        committed = limiter.feed(committed)
    if ban_policy is not None and ban_stats is not None:
        # Turn telemetry over the COMMITTED rows only (the rows this round can
        # actually emit): how many saw the ban move the argmax, and how many ids
        # the ban removed in total. Diagnostics only -- the tokens above do not
        # depend on these counters.
        for i in range(min(len(committed), len(masked_counts))):
            ban_stats.rows_changed += int(changed[i])
            ban_stats.masked_total += int(masked_counts[i])
    if sampling_on:
        pass
    elif lp is not None and lp_out is not None:
        # committed[i] == target[i] (accepted drafts match the target), so row
        # i of the verify forward carries committed token i's log-probs.
        _rows_logprobs(lp, len(committed), lp_out)
    elif ban_policy is not None and logprobs and lp_out is not None:
        # The ban steers the ARGMAX, so the log-prob of a committed row must be
        # read for the MASKED committed id, not the raw argmax: read it off the
        # MASKED row, which makes ``from_logits`` select exactly that id (and its
        # top-k are taken over the masked row, so a banned id never appears as a
        # competitor). A fully-banned row fell back to its RAW argmax, so its raw
        # row is the faithful read there.
        # NOTE the (correct) semantic shift: log_softmax(masked)[id] >=
        # log_softmax(raw)[id] at a ban-changed position, because the banned mass
        # is renormalised onto the survivors -- so a client comparing logprobs ON
        # vs OFF sees the number move at exactly the rows the ban moved. When the
        # ban is off this branch does not run and the value is byte-identical.
        assert masked_rows is not None
        for i in range(len(committed)):
            row = masked_rows[i]
            if float(mx.max(row)) == NEG_INF:
                row = logits[0, i]
            _row_logprobs(row, logprobs, lp_out)
    committed_position = position + len(committed)
    with span("round.rollback"):
        spec.rollback(cache, snapshot, committed_position, stashes)
    with span("round.draft_append"):
        head.append_ctx(tapcat(taps)[:, : len(committed)], draft_state)
    return (
        committed,
        (time.perf_counter() - started) * 1e3,
        accepted,
        gamma,
    )

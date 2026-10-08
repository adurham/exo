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

import json
import os
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, TextIO

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
    mtp_accept_hist: list[int] | None = None,
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
    ``mtp_accept_hist`` is the session-cumulative per-position acceptance
    histogram (index ``k`` = rounds that accepted exactly ``k`` drafts); deltas
    across successive requests give the per-round p1..pk survival curve.
    """
    del task_id, prefill_tokens  # the split is reported through usage/stats below
    hit = "partial" if reused_tokens > 0 else "none"
    if round_stats is not None:
        stats = round_stats.stats(prefill_tps, prompt_tokens, generated).model_copy(
            update={
                "prefix_cache_hit": hit,
                "mtp_cycles_cumulative": mtp_cycles,
                "mtp_accepted_drafts_cumulative": mtp_accepted,
                "mtp_accepted_histogram_cumulative": mtp_accept_hist,
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
            mtp_accepted_histogram_cumulative=mtp_accept_hist,
        )
    sel, top = logprob if logprob is not None else (None, None)
    return GenerationResponse(
        text=text,
        token=token,
        logprob=sel,
        top_logprobs=top,
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


def _round_prof_hook_for(
    engine: Dsv41Engine, round_prof: int
) -> _RoundProf | None:
    """The engine's per-worker ``_RoundProf``, attached lazily -- or ``None``.

    ``mode < 1`` returns ``None`` *immediately*: the caller's only trace of
    ``round_prof`` is then its ``if hook is None`` tests, so the off path
    builds no object and takes no timer branch (byte-identical).

    The timer object is cached on the engine (one per worker), so the JSONL
    file is opened once and survives across requests.
    """
    if round_prof < 1:
        return None
    hook = engine._round_prof  # pyright: ignore[reportPrivateUsage]
    if not isinstance(hook, _RoundProf):
        hook = _RoundProf(round_prof, rank=engine.device_rank)
        engine._round_prof = hook  # pyright: ignore[reportPrivateUsage]
    elif hook.mode != round_prof:
        hook.mode = round_prof
    return hook


class _RoundProf:
    """Opt-in per-round phase timer (PROF 1 host-wallclock, 2 eval-fenced).

    One instance per worker, created lazily by
    :func:`_round_prof_hook_for`. It holds the wallclock readings a round
    takes at its natural sync points and appends them, one JSONL line per
    round, to ``EXO_DSV41_ROUND_PROF_PATH`` (default
    ``/tmp/dsv41_round_prof.<rank>.jsonl``, ``<rank>`` filled with the
    worker's ``device_rank``), flushed per line so a killed runner still
    leaves every round it completed.

    Lifecycle per round, split across the two halves of the round's span:
      * ``enter_fields`` -- called at the END of ``_one_round`` with that
        round's bracket milliseconds (``draft_build_ms``,
        ``verify_block_ms``, ``tail_bookkeep_ms``) plus ``gamma`` and
        ``n_accepted``. It stamps ``round_idx`` (a per-worker counter) and
        stages the row.
      * ``flush_pending`` -- called by the engine's ``_rounds`` AFTER the
        consumer drained the round's batch (i.e. after the generator resumed
        from its ``yield``), adding ``round_total_ms`` and the ``emit_ms``
        span measured across that suspension, then writing one line.

    IT MUST NEVER KILL A RUNNER. The file ``open`` and the ``write``/``flush``
    are each wrapped: a failure is logged ONCE and the hook self-disables
    (``_disabled``), returning to a pure no-op, rather than propagating into
    the decode loop. Every row also carries a ``rank`` tag so the two TP
    workers' files can be told apart when merged.
    """

    __slots__ = ("mode", "rank", "path", "_fh", "_disabled", "_n_rounds", "_pending")

    def __init__(self, mode: int, rank: int = 0) -> None:
        self.mode = mode
        self.rank = rank
        self.path = _round_prof_path_from_env().replace("<rank>", str(rank))
        self._fh: TextIO | None = None
        self._disabled = False
        self._n_rounds = 0
        self._pending: dict[str, float | int] | None = None
        try:
            self._fh = open(self.path, "a", buffering=1)  # noqa: SIM115
        except Exception as exc:  # noqa: BLE001 - instrumentation must never raise
            self._disabled = True
            logger.warning(  # pyright: ignore[reportUnknownMemberType]
                f"[DSV41] round_prof disabled: cannot open {self.path} "
                f"({type(exc).__name__}: {exc})"
            )

    def enter_fields(
        self,
        *,
        gamma: int,
        n_accepted: int,
        draft_build_ms: float,
        verify_block_ms: float,
        tail_bookkeep_ms: float,
    ) -> None:
        """Stage one round's bracket fields; ``round_idx`` is assigned here.

        The engine-side spans (``round_total_ms``, ``emit_ms``, ``rank``) are
        added later by :meth:`flush_pending`, because ``emit_ms`` only exists
        after the generator resumes from the round's ``yield``.
        """
        self._n_rounds += 1
        self._pending = {
            "round_idx": self._n_rounds,
            "gamma": gamma,
            "n_accepted": n_accepted,
            "draft_build_ms": draft_build_ms,
            "verify_block_ms": verify_block_ms,
            "tail_bookkeep_ms": tail_bookkeep_ms,
        }

    def flush_pending(self, *, round_total_ms: float, emit_ms: float) -> None:
        """Complete the staged row with the engine-side spans and write it."""
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        pending["round_total_ms"] = round_total_ms
        pending["emit_ms"] = emit_ms
        pending["rank"] = self.rank
        if self._disabled or self._fh is None:
            return
        try:
            self._fh.write(json.dumps(pending, separators=(",", ":")) + "\n")
            self._fh.flush()
        except Exception as exc:  # noqa: BLE001 - instrumentation must never raise
            self._disabled = True
            logger.warning(  # pyright: ignore[reportUnknownMemberType]
                f"[DSV41] round_prof disabled: write to {self.path} failed "
                f"({type(exc).__name__}: {exc})"
            )


def _round_prof_path_from_env() -> str:
    """JSONL dump path for the per-round timer (``<rank>`` filled per worker)."""
    raw = os.environ.get("EXO_DSV41_ROUND_PROF_PATH")
    if raw is None or raw.strip() == "":
        return "/tmp/dsv41_round_prof.<rank>.jsonl"
    return raw


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
    round_prof: int = 0,
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

    ``round_prof`` selects a per-round instrumentation mode. 0 (or unset)
    is OFF and this round is byte-identical: the only trace of the mode on
    that path is one ``hook is None`` test. 1 = host-wallclock timer (no
    added ``mx.eval``; the numbers are taken at sync points the round
    already has, so it is non-perturbing). 2 = the same timer with an
    ``mx.eval`` of each deferred graph at its bracket end, so every phase is
    billed to itself instead of draining inside the NEXT round's verify.

    The mode is identical on both TP ranks (it rides in the task params), so
    a per-request mode can never diverge across ranks -- if it did, a
    mismatched number of ``mx.eval``s would deadlock the collectives.

    PROF=1 brackets (see ``docs``/phase-1 design Q6):
      * ``draft_build`` -- around ``head.draft(...)`` + the ``verify_in``
        concat: the draft's HOST graph-build only (sub-ms); its GPU work is
        reached by the verify graph and billed to ``verify_block``.
      * ``verify_block`` -- around the verify forward through its single
        ``mx.eval(logits)``: the one honest blocking bracket. It absorbs the
        verify forward, the draft's GPU compute, the compiler's per-layer
        host round-trips, AND the PREVIOUS round's deferred rollback/
        append_ctx tail (both queue lazy work with no eval and drain here).
      * ``tail_bookkeep`` -- the post-eval accept/rollback/append_ctx
        bookkeeping: host graph-build + lazy queueing, eval-free (its
        compute is billed to the next round's ``verify_block``).
    PROF=2 adds an ``mx.eval`` at the close of ``draft_build`` and of the
    ``tail_bookkeep`` bracket (which spans both ``spec.rollback`` and
    ``head.append_ctx``), so each drains its own deferred work. PROF=2 is
    deliberately SERIALIZING: never quote a PROF=2 round time as the
    production round time.
    A single JSON line is emitted per round by the engine's ``_rounds``
    span (``_RoundProf``), carrying every bracket plus ``round_total``.
    """
    hook = _round_prof_hook_for(engine, round_prof)
    started = time.perf_counter()
    anchor = mx.array([token], dtype=mx.int32).reshape(1, 1)
    if head is None:
        # Plain greedy: one row in, one token out. Grow for that row first.
        _ensure_capacity_for_round(cache, 1)
        logits = model(anchor, cache, last_logit_only=True)
        next_token = int(mx.argmax(logits.reshape(-1), axis=-1).item())
        _row_logprobs(logits, logprobs, lp_out)
        elapsed = (time.perf_counter() - started) * 1e3
        if hook is not None:
            hook.enter_fields(
                gamma=1, n_accepted=1,
                draft_build_ms=0.0, verify_block_ms=elapsed, tail_bookkeep_ms=0.0,
            )
        return [next_token], elapsed, 1, 1

    from mlx_lm.models.deepseek_v41 import spec

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
        _ensure_capacity_for_round(cache, 1)
        logits, taps = model(anchor, cache, last_logit_only=True, return_taps=True)
        next_token = int(mx.argmax(logits.reshape(-1), axis=-1).item())
        _row_logprobs(logits, logprobs, lp_out)
        draft_state = head.make_cache(1)
        head.append_ctx(tapcat(taps), draft_state)
        engine._draft_windows[_DRAFT_KEY] = draft_state
        logger.info(
            "[DSV41] DSpark draft window primed from the first decode step; "
            "drafting from the next round (the prefill forward carries no taps "
            "-- see rounds._one_round)."
        )
        elapsed = (time.perf_counter() - started) * 1e3
        if hook is not None:
            # The priming round has no draft: it steps plainly and feeds the
            # draft window. Its append_ctx graph is lazy, so -- like every
            # deferred tail -- it drains in the NEXT round's verify_block.
            hook.enter_fields(
                gamma=1, n_accepted=1,
                draft_build_ms=0.0, verify_block_ms=elapsed, tail_bookkeep_ms=0.0,
            )
        return [next_token], elapsed, 1, 1

    gamma = int(policy.next()) if policy is not None else 1
    # Speculative verify feeds the anchor + EVERY drafted row in ONE forward
    # (1 + gamma rows) before the rollback trims the rejected suffix, so grow
    # for that whole worst case here, at the eval-clean boundary.
    _ensure_capacity_for_round(cache, 1 + gamma)
    position = int(cache.offset)
    t_draft = time.perf_counter()
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
    draft_build_ms = 0.0
    if hook is not None:
        # PROF=1: host graph-build only (sub-ms); the draft's GPU work is
        # reached by the verify graph and billed to verify_block. PROF=2:
        # force the draft + its window writes so the draft stage is drained
        # here. (The per-stage ``draft_stage{0,1,2}`` split from the design's
        # Q6b is a follow-up: it needs an eval hook inside ``DSparkHead``.)
        if hook.mode >= 2:
            # Force the draft + its window writes (R1 force expressions) so the
            # draft stage is billed here. Untyped mlx objects -> local ignores.
            mx.eval(
                drafted,  # pyright: ignore[reportUnknownArgumentType]
                [w.win_kv for w in draft_state],  # pyright: ignore[reportAny]
            )
        draft_build_ms = (time.perf_counter() - t_draft) * 1e3
    snapshot = spec.snap(cache, position)
    t_verify = time.perf_counter()
    if logprobs:
        logits, taps, lp = model(verify_in, cache, return_taps=True, argmax=True,
                                 logprobs=logprobs)
    else:
        logits, taps = model(verify_in, cache, return_taps=True, argmax=True)
        lp = None
    mx.eval(logits)
    verify_block_ms = 0.0
    if hook is not None:
        verify_block_ms = (time.perf_counter() - t_verify) * 1e3
    stashes = spec.stashes(cache)
    target = [int(v) for v in logits[0]]
    draft = [int(v) for v in drafted[0]]

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
    committed = draft[:accepted] + [target[min(accepted, len(target) - 1)]]
    if lp is not None and lp_out is not None:
        # committed[i] == target[i] (accepted drafts match the target), so row
        # i of the verify forward carries committed token i's log-probs.
        _rows_logprobs(lp, len(committed), lp_out)
    t_tail = time.perf_counter()
    committed_position = position + accepted + 1
    spec.rollback(cache, snapshot, committed_position, stashes)
    if hook is not None and hook.mode >= 2:
        # rollback is lazy (O(1) carry rebuild, no eval); force its destination
        # so its compute is billed here, not to the NEXT round's verify.
        mx.eval(
            [lc.comp_state.kv_state for lc in cache.layers  # pyright: ignore[reportAny]
             if lc.comp_state is not None]  # pyright: ignore[reportAny]
        )
    head.append_ctx(tapcat(taps)[:, : accepted + 1], draft_state)
    tail_bookkeep_ms = 0.0
    if hook is not None:
        if hook.mode >= 2:
            # append_ctx is the main_proj GEMM + each stage's ring write; force
            # the rings so this stage drains here too.
            mx.eval(
                [w.win_kv for w in draft_state]  # pyright: ignore[reportAny]
            )
        tail_bookkeep_ms = (time.perf_counter() - t_tail) * 1e3
        hook.enter_fields(
            gamma=gamma, n_accepted=accepted,
            draft_build_ms=draft_build_ms,
            verify_block_ms=verify_block_ms,
            tail_bookkeep_ms=tail_bookkeep_ms,
        )
    return (
        committed,
        (time.perf_counter() - started) * 1e3,
        accepted,
        gamma,
    )

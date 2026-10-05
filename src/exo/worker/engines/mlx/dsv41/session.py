"""Multi-turn session wiring for the DSv4.1 engine.

WHAT THIS IS. The engine's conversation store, and the piece that turns a
request's prompt into a turn on a LIVE cache instead of a fresh one. Without it
``use_prefix_cache`` would be a lie: the engine would re-prefill the whole prompt
on every request. With it, turn N+1 of a conversation prefills only the delta
(``turn-2 prefill < 10% of turn 1``), which is the multi-turn acceptance bar.

It is deliberately NOT ``serve.Session``. That class owns a whole turn -- prefill
AND the speculative decode loop via ``spec.generate`` -- while this engine drives
decode with ``rounds._one_round``, whose contract the engine tests pin (fenced
prefill chunks, one anchor, tap-fed draft window). So this module takes the half
that is about STATE from the mlx-lm fork's ``session_cache.SessionCache``
(prefix match, rewind, checkpoint, token history) and leaves the decode loop to
the engine:

* :class:`Conversation` wraps one ``SessionCache`` -- the live ``ModelCache``
  (window ring + compressed KV + compressor carry + index keys + engram id
  history) -- plus the DSpark draft window that must stay in step with it.
* :meth:`Conversation.prefill` feeds a turn's NEW rows with the engine's own
  chunk loop (:func:`engine_prefill`: ``last_logit_only`` per chunk, a
  model-level eval fence every K layers, no post-prefill decode-prime probe, one
  sync per chunk) and returns the anchor row's logits. The prefill's per-chunk
  taps are pushed into the draft window here, so its ``n_ctx`` tracks
  ``cache.offset`` exactly.
* :meth:`Conversation.finish` checkpoints the whole turn (cache + history) so the
  next turn can extend it and :meth:`Conversation.cancel` can roll it back.

Reuse is exact: ``SessionCache`` rewinds to the newest checkpoint at or below the
common prefix and re-feeds only the delta, so a reused prefix is bitwise the
state a full prefill would have produced for the same chunk boundaries (the
fork's own suite measures cache-state |delta| = 0 with matching chunk plans).

MEASURED: see ``bench/pV7_engine_e2e.py`` for the command and the numbers.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

from exo.worker.engines.mlx.dsv41.errors import Dsv41UnsupportedFeature
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:  # pragma: no cover - typing only (park imports this module)
    from exo.worker.engines.mlx.dsv41.park import ParkedStore

__all__ = [
    "Conversation",
    "Dsv41Sessions",
    "GreedySession",
    "TurnOutcome",
    "choose_prefill_step",
    "engine_prefill",
]

#: Rows of shared prefix required before a resident conversation is reused for a
#: prompt that is not an exact extension of it. Deliberately not 0: every pair of
#: chat prompts shares the system header, and rewinding a big conversation to
#: "fix" a twenty-token overlap is worse than starting cold.
MIN_REUSE_TOKENS = 64

#: SSD parking of idle conversations (see ``dsv41/park.py``). Enabled by
#: default; ``EXO_DSV41_PARK=0`` disables it and makes eviction hard-discard, as
#: it was before. The park store itself owns the directory / budget / min-token
#: gates (``EXO_DSV41_PARK_DIR`` / ``EXO_DSV41_PARK_MAX_GB`` /
#: ``EXO_DSV41_PARK_MIN_TOKENS``).
PARK_ENV = "EXO_DSV41_PARK"


def _park_enabled() -> bool:
    """Whether SSD parking is on (default on; ``0``/``false``/``no``/``off`` off)."""
    return os.environ.get(PARK_ENV, "1").strip().lower() not in (
        "0", "false", "no", "off")


#: Layer spacing for the model-level eval fence on the engine prefill path. The
#: mlx-lm ``Model.__call__`` commits the chunk every ``_fence_every`` layers of a
#: multi-row forward (same ops, same dtypes -- fenced and unfenced are
#: bit-identical), which releases a chunk's O(context) indexer/compressor
#: transients as it progresses instead of holding them until the final
#: ``mx.eval``. ``0`` disables.
PREFILL_FENCE_EVERY_ENV = "EXO_PREFILL_FENCE_EVERY"
DEFAULT_PREFILL_FENCE_EVERY = 2

#: Per-chunk transient budget in DECIMAL megabytes for the chunk-size policy:
#: the indexer's worst-case score row for a chunk is ``step * offset * 4`` bytes
#: (fp32, [1, n, nb] with nb ~= end_pos/ratio), so the chunk shrinks as the
#: context grows to keep that row within the budget.
PREFILL_TRANSIENT_BUDGET_ENV = "EXO_PREFILL_TRANSIENT_BUDGET_MB"
DEFAULT_PREFILL_TRANSIENT_BUDGET_MB = 2048
#: Bytes per decimal MB (the budget env is named in MB, sizes are bytes).
_MBYTES_PER_MB = 1_000_000

#: Periodic checkpoint ladder during a turn's delta prefill. ``SessionCache.plan``
#: rewinds to the newest checkpoint at or below the prompt's longest common
#: prefix, so with checkpoints only at offset 0 and turn ends a ONE-row
#: undershoot below the newest turn end rewinds to 0 and re-feeds the whole
#: context (measured live: a 30033-row follow-up re-fed all 30007 rows, 112.5 s
#: instead of seconds). A ladder checkpoint every ``spacing`` rows bounds that
#: re-feed to ~``spacing`` rows. ``0`` disables the ladder. Explicit
#: ``checkpoint_spacing`` on :class:`Conversation` overrides the env.
CHECKPOINT_SPACING_ENV = "EXO_DSV41_CHECKPOINT_SPACING_ROWS"
DEFAULT_CHECKPOINT_SPACING_ROWS = 1024

#: End-anchored margin rung: force one checkpoint within ``margin`` rows of a
#: turn's delta END, so the confirmed 1-3-row BPE-seam undershoot at the seam
#: costs ~``margin`` rows of refeed instead of a full context. Grid rungs alone
#: would miss the seam. ``0`` disables it; a call no longer than ``margin`` is
#: skipped (its prompt-end checkpoint already anchors it).
CHECKPOINT_MARGIN_ENV = "EXO_DSV41_CHECKPOINT_MARGIN_ROWS"
DEFAULT_CHECKPOINT_MARGIN_ROWS = 512

#: Retained checkpoints per conversation (body rings + carries + draft window).
#: Retention bounds the ladder's memory; the ladder (spacing) bounds worst-case
#: re-feed. Never prune offset 0 (the sibling mlx-lm ``snapshot()`` change keeps
#: it): pure-newest-N silently degrades an old-prefix rewind to rewind-to-0.
CHECKPOINT_KEEP_ENV = "EXO_DSV41_CHECKPOINT_KEEP"
DEFAULT_CHECKPOINT_KEEP = 32


def _read_int_env(name: str, default: int) -> int:
    """Integer env override, falling back to ``default`` on a bad value.

    A malformed override must not crash a worker at engine-construction time;
    the fallback keeps the node serving with the documented default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"[DSV41] ignoring non-integer {name}={raw!r}; using {default}"
        )
        return int(default)


def _indexer_row_bytes() -> int:
    """Bytes per element of the indexer score row, probed ONCE at module import.

    The row is fp32 (4 B) in the classic build and bf16 (2 B) when the deployed
    mlx-lm has the bf16 score row active (``DSV41_INDEXER_ROW_BF16``, read at
    mlx-lm import; process-lifetime constant). The chunk policy must follow the
    ACTUAL dtype so it never over-counts: at depth a hard-coded 4-B charge
    halves the chunk under the same byte budget, doubling the per-chunk
    drain/collective overhead per row. Falls back to fp32's 4 on import trouble.
    """
    import importlib

    try:
        indexer = importlib.import_module("mlx_lm.models.deepseek_v41.indexer")
        dtype = getattr(indexer, "_ROW_DTYPE")  # noqa: B009 - set at mlx-lm import
        return int(dtype.size)  # pyright: ignore[reportAny] - mlx dtype is untyped
    except Exception:  # noqa: BLE001 - a dtype probe must never crash the worker
        return 4


_INDEXER_ROW_BYTES = _indexer_row_bytes()


def choose_prefill_step(
    offset: int,
    total: int,
    base: int,
    budget_bytes: int,
    floor: int = 128,
    row_bytes: int = 4,
) -> int:
    """Rows for the next prefill chunk under a transient-memory budget.

    Pure policy (no model, no MLX), so the schedule is unit-testable on the CPU.
    ``offset`` is the absolute cache position the chunk starts at and ``total``
    the absolute row this feed ends at (so ``total - offset`` rows remain). The
    indexer's worst-case score row for a chunk grows with ``step * offset`` (its
    logical length ``nb`` is the END position, which is ~offset here); that row
    costs ``step * offset * row_bytes`` bytes (``row_bytes`` = the deployed
    score-row element size: 4 fp32, 2 bf16). We therefore keep the full ``base``
    chunk while that row fits in ``budget_bytes`` and shrink as the context
    grows. ``base`` is also the ceiling (only shrinking, never growing), and
    ``floor`` is the smallest chunk ever chosen (throughput does not profit from
    going lower). The result is clamped to the rows remaining and to at least 1,
    so ``total <= offset`` (nothing left) yields 1.
    """
    offset = max(int(offset), 0)
    remaining = int(total) - offset
    if remaining <= 0:
        return 1
    worst_row = max(1, int(row_bytes)) * max(offset, 1)  # bytes per row of the score row
    rows = min(int(base), max(int(floor), int(budget_bytes) // worst_row))
    return max(1, min(rows, remaining))


def _resolve_transient_budget_bytes(transient_budget_mb: int | None) -> int:
    """Transient budget in bytes: explicit MB wins, else the env, else default."""
    mb = (
        _read_int_env(PREFILL_TRANSIENT_BUDGET_ENV, DEFAULT_PREFILL_TRANSIENT_BUDGET_MB)
        if transient_budget_mb is None
        else int(transient_budget_mb)
    )
    return max(1, mb) * _MBYTES_PER_MB


def _resolve_fence_every(fence_every: int | None) -> int:
    """Fence spacing: explicit value wins, else the env, else the default."""
    if fence_every is None:
        return max(0, _read_int_env(PREFILL_FENCE_EVERY_ENV, DEFAULT_PREFILL_FENCE_EVERY))
    return max(0, int(fence_every))


def engine_prefill(
    model: Any,
    ids: Any,
    cache: Any,
    *,
    chunk: int | None = None,
    long_chunk: int | None = None,
    long_threshold: int | None = None,
    fence_every: int | None = None,
    transient_budget_bytes: int | None = None,
    last_logit_only: bool = True,
    argmax: bool = False,
    return_taps: bool = False,
    taps_out: Any = None,
    progress: Any = None,
    taps_cb: Any = None,  # pyright: ignore[reportAny]
    fence_hook: Callable[[], None] | None = None,
    **rest: Any,
) -> Any:
    """The engine's prefill loop, as a driver for ``SessionCache``.

    Same signature and chunk policy as ``prefill.prefill``/``chunked_prefill`` so
    ``SessionCache`` can take it as ``prefill_fn``, but with the engine's own
    shape: every chunk is one ``last_logit_only`` forward (intermediate chunks
    need no full-row head projection), one sync each, no periodic pool clears
    and no post-prefill decode-prime probe. A reused-prefix turn and a cold turn
    run this SAME loop, so session reuse cannot change the tokens it produces.

    Transient bound (a 1M-token prefill must not OOM a 128 GiB node):

    * **eval fences** -- the whole loop runs with ``model._fence_every`` set
      (from ``fence_every``, else ``EXO_PREFILL_FENCE_EVERY``, default 2), so
      ``Model.__call__`` commits each multi-row chunk in K-layer command buffers
      instead of one giant lazy graph. That releases each layer's O(context)
      indexer score rows / compressor ``kv_all`` copies as the chunk progresses
      instead of pinning them until the final ``mx.eval``. Fences change no op
      and no dtype (fenced and unfenced runs are bit-identical); single-row
      (decode) forwards ignore them. The previous value is restored in a
      ``finally``.
    * **transient-budget chunking** -- unless the caller pins ``long_threshold``
      (legacy fixed-crossover behaviour, kept for compatibility and for
      ``SessionCache._prefill_planned``), the chunk size comes from
      :func:`choose_prefill_step`: the indexer's worst-case score row for a
      chunk is ``step * offset * 4`` bytes, so the chunk stays at ``chunk`` while
      that row fits ``transient_budget_bytes`` (else
      ``EXO_PREFILL_TRANSIENT_BUDGET_MB``, default 2048 MB) and shrinks to a
      128-row floor as the context grows.

    ``taps_out`` collects the per-chunk DSpark taps (the draft window's context
    feed); ``progress`` is ``fn(chunks, rows_done, elapsed_s)`` and doubles as
    the engine's cancellation point. ``taps_cb`` -- optional -- is
    ``fn(chunk_taps_dict)`` called at each chunk END, right after that chunk's
    taps are committed (``mx.eval``), so a caller can feed them into the draft
    window immediately and keep ``draft_ctx == offset`` at every chunk boundary
    (which is what makes a mid-prefill checkpoint a consistent pair). Taps are
    still collected into ``taps_out`` for compatibility; the two are
    independent and a caller that feeds per chunk must NOT also batch-feed
    ``taps_out`` (double-apply).

    ``fence_hook`` is the fence-point liveness callback: it is installed on the
    model as ``model._fence_hook`` for the duration of the loop (the sibling
    mlx-lm change calls it after each model-level ``mx.eval`` fence), so each
    multi-row chunk re-emits the runner's status at a point backed by K layers
    of actually-committed compute. The previous value is restored in a
    ``finally``, exactly like the fence spacing. It only reaches the model
    thread; this driver never calls it directly.
    """
    del rest  # tolerated, unused driver keywords (forward compatibility)
    if isinstance(ids, mx.array):
        ids_mx = ids if ids.ndim == 2 else ids[None]
    else:
        ids_mx = mx.array(np.asarray(ids, dtype=np.int64)[None])
    total = int(ids_mx.shape[1])
    if total == 0:
        raise ValueError("engine_prefill: empty ids")

    base = 512 if chunk is None else int(chunk)
    long_step = 128 if long_chunk is None else int(long_chunk)
    # ``long_threshold`` is None for the engine's own path (budget policy); a
    # caller that sets it explicitly (e.g. SessionCache's forced chunk plan)
    # gets the legacy fixed-crossover branch, which takes precedence.
    threshold = None if long_threshold is None else int(long_threshold)
    # ``transient_budget_bytes`` is already bytes (the partial in Conversation
    # resolved the MB env); absent => resolve the env/default here.
    budget_bytes = (
        _resolve_transient_budget_bytes(None)
        if transient_budget_bytes is None
        else int(transient_budget_bytes)
    )
    want_taps = bool(return_taps or taps_out is not None)

    # Log the EFFECTIVE transient controls once per prefill call: both knobs
    # were previously read from env but never forwarded by the launcher, so the
    # deployed values were invisible. One line per call is noise-free and makes
    # the live config checkable from the runner log.
    _fence_eff = _resolve_fence_every(fence_every)
    hook_on = bool(_fence_eff > 0 and fence_hook is not None)
    logger.info(
        f"[DSV41] prefill controls: fence_every={_fence_eff} "
        f"transient_budget_mb={budget_bytes // _MBYTES_PER_MB} "
        f"score_row_bytes={_indexer_row_bytes()} "
        f"fence_hook={'on' if hook_on else 'off'} "
        f"(rows={total}, base={base})"
    )
    if fence_hook is not None and _fence_eff == 0:
        # The hook rides the model-level eval fence; with no fence it is never
        # called, so a long prefill emits nothing and the supervisor's hang
        # watchdog may SIGKILL a healthy runner. Loud once per prefill.
        logger.warning(  # pyright: ignore[reportUnknownMemberType]
            "[DSV41] liveness hook unavailable: fence_every=0 "
            "(EXO_PREFILL_FENCE_EVERY resolved to 0), so no fence point fires"
        )

    t0 = time.perf_counter()
    out = None
    done = 0
    nchunks = 0
    last_taps = None
    fence_prev = getattr(model, "_fence_every", None)
    hook_prev = getattr(model, "_fence_hook", None)  # pyright: ignore[reportAny]
    model._fence_every = _fence_eff
    model._fence_hook = fence_hook
    try:
        while done < total:
            offset: int = int(cache.offset)
            if threshold is not None:
                step = long_step if offset >= threshold else base
            else:
                step = choose_prefill_step(
                    offset, total, base, budget_bytes, row_bytes=_indexer_row_bytes()
                )
            stop = min(done + step, total)
            piece = ids_mx[:, done:stop]
            last = stop == total
            res = model(
                piece,
                cache,
                last_logit_only=True if not last else last_logit_only,
                return_taps=want_taps,
                argmax=argmax if last else False,
            )
            handle, taps = res if isinstance(res, tuple) else (res, None)
            if taps is not None and taps_out is not None:
                taps_out.append(taps)
                last_taps = taps
            mx.eval(handle, *(taps.values() if taps else []))
            if taps_cb is not None and taps is not None:
                # Per-chunk draft feed: BEFORE the caller's progress point (and
                # therefore before any checkpoint it drives), so the draft window
                # sits at the same row as the body cache at that boundary.
                taps_cb(taps)
            if last:
                out = handle
            done = stop
            nchunks += 1
            if progress is not None:
                progress(nchunks, done, time.perf_counter() - t0)
    finally:
        model._fence_every = fence_prev if fence_prev is not None else 0
        model._fence_hook = hook_prev
    if getattr(model, "_fence_hook_failed", False):  # pyright: ignore[reportAny]
        # The model-side wrapper latches this after a hook bug (it disables the
        # hook and logs once at WARNING). Surface it at CRITICAL: from here on
        # the runner is silent during this prefill even though it is healthy,
        # which is exactly the false-hang class this hook exists to prevent.
        logger.critical(  # pyright: ignore[reportUnknownMemberType]
            "[DSV41] prefill fence liveness hook FAILED and was disabled for "
            "subsequent forwards; a long prefill may now go silent (see the "
            "model-side warning) and a healthy runner may be killed"
        )
        model._fence_hook_failed = False
    if return_taps:
        return out, (last_taps if last_taps is not None else {})
    return out


@dataclass
class TurnOutcome:
    """One turn's fed rows plus the numbers the engine reports.

    ``tokens`` is filled in by the engine once its decode rounds have run; the
    prefill half of the turn only produces ``anchor_logits``.
    """

    anchor_logits: mx.array | None
    prompt_tokens: int
    prefill_tokens: int
    reused_tokens: int
    cache_offset: int
    tokens: list[int] = field(default_factory=list)
    hit: bool = False
    prefill_seconds: float = 0.0
    rewound_from: int | None = None
    committed: bool = False

    @property
    def reuse_ratio(self) -> float:
        return self.prefill_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def __str__(self) -> str:
        return (
            f"prompt={self.prompt_tokens} prefill={self.prefill_tokens} "
            f"reuse={self.reused_tokens} cache={self.cache_offset}"
            + (f" rewind={self.rewound_from}" if self.rewound_from is not None else "")
            + ("" if self.committed else " UNCOMMITTED")
        )


def _ids_of(ids: Any) -> np.ndarray:
    arr = np.array(ids) if isinstance(ids, mx.array) else np.asarray(ids)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"session ids must be [n] or [1, n], got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _newest_boundary(cache: Any) -> int:  # pyright: ignore[reportAny]
    """Newest checkpoint position of a ``SessionCache`` (>= 0; offset 0 exists).

    The mlx-lm cache is dynamic, so strict pyright sees ``boundaries`` as
    ``list[Unknown]``; this local helper keeps that noise out of the call sites.
    """
    return int(max(cache.boundaries))  # pyright: ignore[reportAny]


def _has_boundary(cache: Any, pos: int) -> bool:  # pyright: ignore[reportAny]
    """Whether ``pos`` is already a checkpoint of ``cache`` (dedupe test)."""
    return bool(pos in cache.boundaries)  # pyright: ignore[reportAny]


class Conversation:
    """One conversation's live cache, draft window and token history.

    Serves both decode modes: the engine's round drafts and verifies when a head
    is attached (``uses_draft``), and steps one row at a time otherwise. The
    session state is identical either way.
    """

    def __init__(
        self,
        model: Any,
        head: Any | None,
        *,
        max_seq_len: int,
        chunk: int | None = None,
        long_chunk: int | None = None,
        long_threshold: int | None = None,
        transient_budget_mb: int | None = None,
        max_snapshots: int | None = None,
        checkpoint_spacing: int | None = None,
        checkpoint_margin: int | None = None,
        eos_id: int = 1,
        progress: Any = None,
        fence_hook: Callable[[], None] | None = None,
    ) -> None:
        from mlx_lm.models.deepseek_v41 import session_cache as _sc

        self.model = model
        self.head = head
        self.uses_draft = head is not None
        self.eos_id = int(eos_id)
        # Checkpoint-ladder cadence. Explicit values win (tests); otherwise the
        # env, else the documented defaults. ``spacing`` 0 disables the grid
        # ladder and ``margin`` 0 disables the end-anchored rung; ``keep`` is the
        # retained-snapshot cap handed to the SessionCache (which also keeps
        # offset 0 -- see the sibling snapshot() change).
        self._spacing = max(0, _read_int_env(CHECKPOINT_SPACING_ENV,
                                             DEFAULT_CHECKPOINT_SPACING_ROWS)
                            if checkpoint_spacing is None else int(checkpoint_spacing))
        self._margin = max(0, _read_int_env(CHECKPOINT_MARGIN_ENV,
                                            DEFAULT_CHECKPOINT_MARGIN_ROWS)
                           if checkpoint_margin is None else int(checkpoint_margin))
        keep = (_read_int_env(CHECKPOINT_KEEP_ENV, DEFAULT_CHECKPOINT_KEEP)
                if max_snapshots is None else int(max_snapshots))
        #: One warning per conversation when the draft window is out of lockstep
        #: at a cadence point; we then SKIP the checkpoint rather than snapshot
        #: an inconsistent pair.
        self._lockstep_warned = False
        #: ``total`` of the prefill call in flight (the end-anchored rung's
        #: reference); set/cleared around ``cache.append_turn``.
        self._prefill_total: int | None = None
        #: The end-anchored margin rung fires once per prefill call.
        self._margin_rung_done = False
        # Transient bound: run the prefill driver under an eval fence, and pass
        # the per-chunk byte budget the driver's chunk-size policy honours.
        # All three are driver-only keywords SessionCache does not forward, so
        # they ride ``prefill_fn`` via ``functools.partial``; ``inspect.signature``
        # on a partial keeps the underlying ``**rest``, so SessionCache still
        # sees the ``chunk``/``long_chunk``/``long_threshold`` capabilities.
        # ``fence_hook`` (the fence-point liveness callback) rides the same
        # partial: it is the ONLY path by which it reaches ``engine_prefill``,
        # since the mlx-lm-side SessionCache is untouched by this change (see
        # the design's exo driver contract). Both the conv/general
        # (``SessionCache._prefill_call``) and the planned
        # (``SessionCache._prefill_planned``) feed paths call this partial, so
        # the hook is installed on both.
        prefill_fn = functools.partial(
            engine_prefill,
            fence_every=None,  # None => the driver reads EXO_PREFILL_FENCE_EVERY
            transient_budget_bytes=_resolve_transient_budget_bytes(transient_budget_mb),
            fence_hook=fence_hook,
            # Feed each chunk's taps into the draft window as that chunk
            # completes -- the precondition for a mid-prefill checkpoint being a
            # CONSISTENT (body, draft) pair. The old post-delta batch feed is
            # therefore skipped in prefill().
            taps_cb=self._on_chunk_taps,
        )
        try:
            self.cache = _sc.SessionCache(
                model,
                max_seq_len=int(max_seq_len),
                max_snapshots=keep,
                prefill_fn=prefill_fn,
                chunk=chunk,
                long_chunk=long_chunk,
                long_threshold=long_threshold,
                progress=progress,
            )
        except _sc.CapacityError as e:  # pragma: no cover - defensive
            raise Dsv41UnsupportedFeature(
                f"DSV4.1 conversation does not fit this instance's cache: {e}"
            ) from e
        #: The DSpark draft window (``head.make_cache(1)``): primed from the
        #: prefill's per-chunk taps, appended to by the engine's verify rounds
        #: (``rounds._one_round``), and checkpointed with the body cache.
        self.draft_state: Any | None = None
        #: Draft-window copies keyed by cache checkpoint position: a cache rewind
        #: must rewind the draft window to the same row, or the two drift apart.
        self._draft_snaps: dict[int, list[tuple[Any, int]] | None] = {0: None}
        #: Next-token logits at each prompt-end checkpoint, so an exact repeat
        #: of a prompt (zero new rows) still has an anchor.
        self._anchor_at: dict[int, Any] = {}
        self._gen: list[int] = []
        #: Cache rows already in place when the CURRENT turn started; the
        #: generated history is aligned to position ``_base + k`` for entry ``k``.
        self._base = 0
        self.n_turns = 0
        self._inflight = False

    # -- introspection ----------------------------------------------------
    @property
    def offset(self) -> int:
        return int(self.cache.offset)

    @property
    def tokens(self) -> list[int]:
        return [int(t) for t in self.cache.tokens]

    @property
    def generated(self) -> list[int]:
        return list(self._gen)

    def draft_ctx(self) -> int:
        """Rows the draft window holds (must equal ``offset`` between turns)."""
        try:
            return int(self.draft_state[0].n_ctx)
        except Exception:  # pragma: no cover - no draft window
            return -1

    def summary(self) -> str:
        return (
            f"Conversation(turns={self.n_turns} cache={self.offset} "
            f"generated={len(self._gen)} draft_ctx={self.draft_ctx()}"
            + (" INFLIGHT" if self._inflight else "")
            + ")"
        )

    # -- turn -------------------------------------------------------------
    def prefill(self, prompt_ids: Any, *, chunk_plan: Any = None) -> TurnOutcome:
        """Feed this turn's NEW rows onto the live cache; returns the anchor.

        ``anchor_logits`` is the last fed row's logits: its argmax is this turn's
        first token, and it is what the engine hands to its first round.
        """
        from mlx_lm.models.deepseek_v41 import session_cache as _sc

        ids = _ids_of(prompt_ids)
        if ids.shape[0] == 0:
            raise ValueError("session turn: empty prompt")
        taps: list = []
        t0 = time.perf_counter()
        # The end-anchored margin rung's reference and its once-per-call latch;
        # cleared in the finally so a failed append cannot leave them stale.
        self._prefill_total = int(ids.shape[0])
        self._margin_rung_done = False
        try:
            # Rewind the draft window to the row the body cache will rewind to
            # BEFORE the delta is fed. The per-chunk callback advances the draft
            # as each chunk lands, so it must start from the same base the body
            # starts from -- otherwise feeding inside ``append_turn`` and
            # restoring after it would fight (the restore would wipe the feed,
            # or leave the draft a whole delta behind).
            _, draft_base, _ = self.cache.plan(ids)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if self.draft_state is not None and self.draft_ctx() not in (-1, draft_base):
                self._restore_draft(draft_base)  # pyright: ignore[reportUnknownArgumentType]
            res = self.cache.append_turn(
                ids,
                argmax=False,
                return_taps=False,
                taps_out=taps,
                chunk_plan=chunk_plan,
                checkpoint=False,
            )
        except _sc.CapacityError as e:
            raise Dsv41UnsupportedFeature(
                f"DSV4.1 session out of context: {e}"
            ) from e
        except _sc.RollbackError as e:
            raise Dsv41UnsupportedFeature(f"DSV4.1 session rewind failed: {e}") from e
        finally:
            self._prefill_total = None
            self._margin_rung_done = False
        pre_s = time.perf_counter() - t0
        logits = res.logits
        if logits is None and int(res.tokens_prefilled) == 0:
            logits = self._anchor_at.get(self.offset)
        if logits is None:
            raise Dsv41UnsupportedFeature(
                "DSV4.1 session: this turn fed no new rows (the prompt is an exact "
                "cache hit) and the cache holds no anchor logits for it; append the "
                "new user text (or the previous reply) and retry."
            )
        self._base = self.offset - int(res.tokens_prefilled)
        # The delta's taps were fed PER CHUNK (``taps_cb``, before any ladder
        # checkpoint), so the draft window already sits at ``offset``. The
        # collected ``taps`` list is retained for compatibility but MUST NOT be
        # batch-fed again (double-apply). Only a driver that did NOT run the
        # per-chunk callback leaves ``draft_ctx`` behind and needs the fallback.
        if self.draft_ctx() not in (-1, self.offset):
            self._feed_taps(taps)  # pyright: ignore[reportUnknownMemberType]
        if self.draft_state is not None and self.draft_ctx() != self.offset:
            raise RuntimeError(
                f"DSV4.1 session: the draft window holds {self.draft_ctx()} rows "
                f"but the cache is at {self.offset}: they must advance together."
            )
        # Checkpoint the prompt end: a follow-up turn whose re-rendered reply
        # differs from the generated tokens then reuses this whole prompt.
        self._checkpoint()
        self._anchor_at[self.offset] = logits
        self._inflight = True
        return TurnOutcome(
            anchor_logits=logits,
            prompt_tokens=int(ids.shape[0]),
            prefill_tokens=int(res.tokens_prefilled),
            reused_tokens=int(res.tokens_reused),
            cache_offset=self.offset,
            hit=bool(res.hit),
            prefill_seconds=pre_s,
            rewound_from=res.rolled_back_from,
        )

    def _feed_taps(self, taps: list) -> None:  # pyright: ignore[reportMissingTypeArgument]
        """Push the prefill's per-chunk DSpark taps into the draft window.

        ``taps`` is a list of per-chunk tap dicts (the batch path); the
        per-chunk cadence path feeds one chunk at a time through here too.
        """
        if self.head is None or not taps:
            return
        ids = list(self.model.args.dspark_target_layer_ids)
        if self.draft_state is None:
            self.draft_state = self.head.make_cache(1)
        for chunk_taps in taps:
            cat = mx.concatenate([chunk_taps[layer] for layer in ids], axis=-1)
            self.head.append_ctx(cat, self.draft_state)

    def _feed_chunk_taps(self, chunk_taps: Any) -> None:  # pyright: ignore[reportAny]
        """Feed ONE chunk's taps into the draft window immediately."""
        self._feed_taps([chunk_taps])  # pyright: ignore[reportUnknownMemberType]

    def _on_chunk_taps(self, chunk_taps: Any) -> None:  # pyright: ignore[reportAny]
        """Per-chunk callback from ``engine_prefill``: draft feed + cadence.

        ORDER IS LOAD-BEARING. The chunk's taps are fed FIRST (this chunk's
        rows), so ``draft_ctx == offset`` holds when the margin and grid checks
        below decide to checkpoint; a checkpoint at this boundary therefore
        snapshots a CONSISTENT (body, draft) pair. ``engine_prefill`` calls this
        at each chunk end, after that chunk's taps are evaluated.
        """
        self._feed_chunk_taps(chunk_taps)
        self._margin_rung()
        self.maybe_checkpoint()

    def _margin_rung(self) -> None:
        """Force ONE checkpoint within ``margin`` rows of the prefill's end.

        Grid rungs land on multiples of ``spacing`` and can leave the delta end
        up to ``spacing`` rows past the newest rung -- exactly where the
        confirmed 1-3-row BPE-seam undershoot bites. This rung puts a checkpoint
        at the first chunk boundary at/after ``total - margin``, so the seam
        undershoot costs ~``margin`` rows of refeed. Skipped for calls no longer
        than ``margin`` (their prompt-end checkpoint already anchors them), and
        fires at most once per prefill call.
        """
        if self._margin <= 0 or self._margin_rung_done:
            return
        total = self._prefill_total
        if total is None or total <= self._margin:
            return
        if self.offset < total - self._margin:
            return
        self._margin_rung_done = True
        self.maybe_checkpoint(force=True)

    def maybe_checkpoint(self, *, force: bool = False) -> bool:
        """Take a ladder checkpoint when the cadence says so (or ``force``).

        Returns whether a snapshot was taken. The grid cadence fires when the
        newest existing boundary is at least ``spacing`` rows behind the cache
        offset (``spacing == 0`` disables it). Never snapshots when a boundary
        already equals ``self.offset`` (dedupe -- a refeed re-crossing a rung
        must not add a duplicate). Verifies the draft/body lockstep invariant
        first: with a draft window present, ``draft_ctx`` must equal ``offset``,
        else the checkpoint is SKIPPED with a one-time warning rather than
        snapshotting an inconsistent pair.
        """
        if not force:
            if self._spacing <= 0:
                return False
            newest = _newest_boundary(self.cache)
            if self.offset - newest < self._spacing:
                return False
        if _has_boundary(self.cache, self.offset):
            return False
        if self.draft_state is not None and self.draft_ctx() != self.offset:
            if not self._lockstep_warned:
                self._lockstep_warned = True
                logger.warning(  # pyright: ignore[reportUnknownMemberType]
                    "[DSV41] checkpoint cadence skipped: the draft window holds "
                    f"{self.draft_ctx()} rows but the cache is at {self.offset}; "
                    "refusing to checkpoint an inconsistent (body, draft) pair"
                )
            return False
        self._checkpoint()
        return True

    def mark_rows(self, ids: Any) -> None:
        """Book-keeping append for rows the engine's decode loop fed itself."""
        arr = _ids_of(ids)
        self.cache.mark_seen(arr)
        self._gen.extend(int(t) for t in arr)

    def finish(self, *, checkpoint: bool = True) -> None:
        """Close the turn: checkpoint the cache for the next one.

        ``checkpoint=False`` leaves it uncommitted so :meth:`cancel` has something
        real to roll back to (the engine always checkpoints).
        """
        if checkpoint:
            self._checkpoint()
            self._inflight = False
        self.n_turns += 1

    def _checkpoint(self) -> None:
        """Snapshot the body cache and the draft window at the same row."""
        self.cache.snapshot()
        if self.draft_state is not None:
            saved = [(w.win_kv + 0, int(w.n_ctx)) for w in self.draft_state]
            mx.eval([kv for kv, _ in saved])
            self._draft_snaps[self.offset] = saved
        self._prune_stale_snaps()

    def _prune_stale_snaps(self) -> None:
        """Drop draft/anchor checkpoints with no body boundary (kept in lockstep).

        ``SessionCache`` prunes the body snapshots itself (retention + rewind);
        the draft-window and anchor dicts must follow, or a rewind to a dropped
        boundary leaves orphan keys behind. Called after every checkpoint and
        after a cancel/rewind.
        """
        keep = set(self.cache.boundaries)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        for pos in [p for p in self._draft_snaps if p not in keep]:
            del self._draft_snaps[pos]
        for pos in [p for p in self._anchor_at if p not in keep]:
            del self._anchor_at[pos]

    def _restore_draft(self, pos: int) -> None:
        """Put the draft window back to the row the body cache was rewound to."""
        if pos not in self._draft_snaps:
            raise RuntimeError(
                f"DSV4.1 session: the cache rewound to {pos} but no draft-window "
                f"checkpoint exists there (have {sorted(self._draft_snaps)})."
            )
        saved = self._draft_snaps[pos]
        if saved is None or self.draft_state is None:
            self.draft_state = None
            return
        for w, (kv, n_ctx) in zip(self.draft_state, saved, strict=True):
            w.win_kv = kv + 0
            w.n_ctx = n_ctx

    def sync_history(self, prompt_ids: Any, turn: "TurnOutcome") -> None:
        """Align the token history with the rows the cache ACTUALLY holds.

        The engine's decode rounds feed rows with ``model(...)`` directly (they
        bypass the prefill driver), so ``SessionCache``'s own history never sees
        them. Reconcile it here, once per turn: after the turn the cache holds the
        prompt's rows plus every generated token except the LAST one (that token is
        the next turn's anchor and has not been fed). The write is direct because
        ``SessionCache.mark_seen`` refuses to be used as a catch-up (it requires the
        two views to already agree), and this module owns the engine side of that
        contract.
        """
        prompt = _ids_of(prompt_ids)
        rows = np.asarray(turn.tokens[: max(0, len(turn.tokens) - 1)], dtype=np.int64)
        # Absolute rows: the full prompt (reused prefix + this turn's prefill)
        # plus every generated token but the last.
        want = int(prompt.shape[0]) + int(rows.shape[0])
        held = int(self.cache.offset)
        if held != want:
            raise RuntimeError(
                f"DSV4.1 session: the cache holds {held} rows but this turn fed "
                f"{want} (prefill={turn.prefill_tokens}, reuse={turn.reused_tokens}); "
                "the decode rounds and the cache are out of step."
            )
        with contextlib.suppress(Exception):  # pragma: no cover - no stand-in history
            self.cache._ids = np.ascontiguousarray(np.concatenate([prompt, rows]))
        self._gen = [int(t) for t in turn.tokens]

    def cancel(self) -> int:
        """Roll the in-flight turn back: cache, and the generated history.

        ``_base`` is the offset the turn started from, so the history is truncated
        to the rows the cache still holds (``offset - _base`` entries).
        """
        if not self._inflight:
            return 0
        # Back to the turn START (the prompt-end checkpoint is newer than it).
        if self._base in self.cache.boundaries:  # pyright: ignore[reportUnknownMemberType]
            dropped = int(self.cache.rewind(self._base))
        else:
            dropped = int(self.cache.cancel())
        if self.draft_ctx() != -1:
            self._restore_draft(self.offset)
        # The rewind dropped every body snapshot above the target; the draft and
        # anchor dicts must drop the same keys (no orphans > the rewind target).
        self._prune_stale_snaps()
        keep = max(0, self.offset - self._base)
        if len(self._gen) > keep:
            del self._gen[keep:]
        self._inflight = False
        return dropped

    def close(self) -> None:
        self.cache.del_cache()
        self.draft_state = None


class GreedySession(Conversation):
    """A conversation with no DSpark head: same protocol, no drafting."""


@dataclass
class _Entry:
    session: Any
    last_used: float = field(default_factory=time.monotonic)


class Dsv41Sessions:
    """The engine's conversation store: bounded LRU of conversations.

    ``get(prompt_ids, key=...)`` returns the conversation this request belongs
    to: the one named by ``key`` when the client supplied a conversation id, else
    the resident conversation sharing the longest prefix with ``prompt_ids`` (at
    least :data:`MIN_REUSE_TOKENS` rows), else a new cold one.
    """

    def __init__(
        self,
        model: Any,
        head: Any | None,
        *,
        max_seq_len: int,
        chunk: int | None = None,
        long_chunk: int | None = None,
        long_threshold: int | None = None,
        transient_budget_mb: int | None = None,
        max_sessions: int = 2,
        max_snapshots: int | None = None,
        checkpoint_spacing: int | None = None,
        checkpoint_margin: int | None = None,
        eos_id: int = 1,
        use_draft: bool = True,
        progress: Callable[[int, int, float], None] | None = None,
        fence_heartbeat: Callable[[], None] | None = None,
        park_store: Any = None,
    ) -> None:
        self.model = model
        self.head = head if use_draft else None
        self.max_seq_len = int(max_seq_len)
        self.max_sessions = max(1, int(max_sessions))
        self.kw = dict(
            chunk=chunk,
            long_chunk=long_chunk,
            long_threshold=long_threshold,
            transient_budget_mb=transient_budget_mb,
            max_snapshots=max_snapshots,
            checkpoint_spacing=checkpoint_spacing,
            checkpoint_margin=checkpoint_margin,
            eos_id=eos_id,
        )
        #: Forwarded to Conversation -> SessionCache (kept OUT of ``self.kw`` so
        #: its value type does not widen the dict and break ``**`` unpacking).
        #: Without it the engine's per-chunk prefill-progress hook is unreachable
        #: and a prefill longer than the supervisor's hang-watchdog window emits
        #: no events, so a healthy runner is SIGKILLed mid-prefill.
        self._progress: Callable[[int, int, float], None] | None = progress
        #: Forwarded to Conversation -> SessionCache -> engine_prefill as the
        #: fence-point liveness hook (``model._fence_hook``). Kept OUT of
        #: ``self.kw`` for the same type-reason as ``_progress``: it is a
        #: callable, not a construction scalar, and must not reach the parked
        #: store's ``conv_kw`` (a restored conversation takes a fresh hook).
        self._fence_heartbeat: Callable[[], None] | None = fence_heartbeat
        self._entries: "collections.OrderedDict[str, _Entry]" = collections.OrderedDict()
        self.stats = collections.Counter()
        #: An injected park store (the tests pass one rooted at ``tmp_path``);
        #: ``None`` builds the default store lazily on the first eviction.
        self._park: "ParkedStore | None" = park_store
        #: Latched once a park-store build fails, so it is not retried per eviction.
        self._park_failed = False

    # -- key resolution ---------------------------------------------------
    @staticmethod
    def key_for(prompt_ids: Any, key: str | None) -> str:
        from mlx_lm.models.deepseek_v41 import session_cache as _sc

        if key:
            return f"id:{key}"
        return "p:" + _sc.prefix_hash(_ids_of(prompt_ids))

    def get(self, prompt_ids: Any, key: str | None = None) -> Conversation:
        from mlx_lm.models.deepseek_v41 import session_cache as _sc

        ids = _ids_of(prompt_ids)
        if key:
            entry = self._entries.get(f"id:{key}")
            if entry is not None:
                self._entries.move_to_end(f"id:{key}")
                self.stats["reuse"] += 1
                return entry.session
            # Not resident: a parked session with the same conversation key is
            # the best possible match (no prefix threshold -- the client named it).
            restored = self._restore_parked(ids, key=key)
            if restored is not None:
                return restored
            return self._open(f"id:{key}")

        best, best_lcp = None, 0
        for k, entry in self._entries.items():
            lcp = _sc.common_prefix_len(ids, entry.session.tokens)
            if lcp > best_lcp:
                best, best_lcp = k, lcp
        if best is not None and best_lcp >= MIN_REUSE_TOKENS:
            self._entries.move_to_end(best)
            self.stats["reuse"] += 1
            logger.info(
                f"[DSV41] session reuse: this {len(ids)}-token prompt matches a "
                f"resident conversation on {best_lcp} rows"
            )
            return self._entries[best].session
        # No resident match: consult the parked store before paying a cold
        # prefill. A hit restores the conversation (and its cache) from SSD.
        restored = self._restore_parked(ids, key=None)
        if restored is not None:
            return restored
        self.stats["cold"] += 1
        return self._open("p:" + _sc.prefix_hash(ids))

    # -- parking ----------------------------------------------------------

    def _park_store(self) -> "ParkedStore | None":
        """The park store (injected by tests, else built once from the env).

        Built lazily: a worker that never overflows its store never pays for it,
        and a store that cannot be constructed is not retried per eviction.
        """
        if self._park is not None:
            return self._park
        if self._park_failed:
            return None
        try:
            from exo.worker.engines.mlx.dsv41.park import ParkedStore

            self._park = ParkedStore(
                self.model, self.head,
                # ``fence_hook`` must ride the store's own conversation-build
                # path (like ``progress``): a restored parked conversation gets
                # a FRESH liveness hook, and without it a long delta prefill on
                # a restored conversation would emit no fence events and could
                # be SIGKILLed as hung. It is carried here, NOT in ``conv_kw``
                # (which is the bare construction-scalar dict).
                conv_kw=dict(self.kw, fence_hook=self._fence_heartbeat),
                progress=self._progress,
            )
            return self._park
        except Exception as exc:  # noqa: BLE001 - parking is best-effort
            logger.warning(
                f"[DSV41] park store unavailable ({type(exc).__name__}: {exc}); "
                "eviction will hard-discard"
            )
            self._park_failed = True
            return None

    def _restore_parked(self, ids: Any, *, key: str | None) -> Conversation | None:
        """Try to restore a parked session for ``ids``; ``None`` on any miss."""
        if not _park_enabled():
            return None
        store = self._park_store()
        if store is None:
            return None
        try:
            conv = store.try_restore(ids, key=key)
        except Exception as exc:  # noqa: BLE001 - restore never breaks a request
            logger.warning(
                f"[DSV41] parked restore failed ({type(exc).__name__}: {exc}); "
                "falling back to a cold open")
            self.stats["park_failed"] += 1
            return None
        if conv is None:
            return None
        store_key = self.key_for(ids, key)
        self._entries[store_key] = _Entry(conv)
        self.stats["restored"] += 1
        logger.info(
            f"[DSV41] parked session restored: {len(conv.tokens)} rows from SSD "
            f"(key={store_key})"
        )
        return conv

    def _open(self, key: str) -> Conversation:
        session = Conversation(
            self.model,
            self.head,
            max_seq_len=self.max_seq_len,
            progress=self._progress,
            fence_hook=self._fence_heartbeat,
            **self.kw,
        )
        self._entries[key] = _Entry(session)
        while len(self._entries) > self.max_sessions:
            victim_key, victim = self._entries.popitem(last=False)
            # Preserve the conversation id (if any) so a parked session can be
            # found by key later; prefix-keyed entries carry no client id.
            ckey = victim_key[3:] if victim_key.startswith("id:") else None
            self._evict(victim.session, key=ckey)
        return session

    def _evict(self, session: Any, *, key: str | None = None) -> None:
        """Park an idle conversation to SSD if possible, else close it outright.

        Parking is best-effort and must never raise: any failure (disabled,
        too short, mid-flight, unsafe draft window, disk error) falls back to the
        original hard ``close()``. The path is deliberately conservative --
        ``drop``/``close``/``cancel_all`` never reach here, so explicit
        discards stay hard.
        """
        parked = False
        if _park_enabled():
            store = self._park_store()
            if store is not None:
                try:
                    parked = bool(store.park(session, key=key))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"[DSV41] park of an evicted session failed "
                        f"({type(exc).__name__}: {exc}); discarding"
                    )
                    self.stats["park_failed"] += 1
                    parked = False
        if parked:
            self.stats["parked"] += 1
            logger.info(
                f"[DSV41] parked an idle conversation to SSD "
                f"({len(session.tokens)} rows)"
            )
        else:
            logger.info("[DSV41] session store evicting an idle conversation")
        session.close()
        self.stats["evicted"] += 1

    def drop(self, session: Any) -> None:
        """Forget + close a conversation (a one-off request has finished).

        The cache is released immediately, so a later request cannot be answered
        from a conversation nobody asked to keep (an image request, or one that
        disabled prefix reuse).
        """
        for k, entry in list(self._entries.items()):
            if entry.session is session:
                del self._entries[k]
                entry.session.close()
                self.stats["dropped"] += 1
                return

    def cancel_all(self) -> int:
        """Cancel every in-flight turn (the post-reconnect / cancel-all path)."""
        dropped = 0
        for entry in self._entries.values():
            dropped += int(entry.session.cancel())
        return dropped

    def close(self) -> None:
        for entry in self._entries.values():
            entry.session.close()
        self._entries.clear()

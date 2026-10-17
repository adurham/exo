"""The DSv4.1 (EXL3) serving engine.

WHAT THIS IS. A dedicated :class:`exo.worker.engines.base.Engine` for
``dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`` that drives the mlx-lm
fork's ``deepseek_v41`` model (EXL3 trellis kernels + DSpark draft head) and
reuses exo's tokenizer, chat template, DSML tool parser and chunk plumbing. It
deliberately does NOT route through ``mlx_generate`` / ``ExoBatchGenerator`` /
``KVPrefixCache``: DSv4.1 has its own incremental cache object
(``deepseek_v41.ModelCache``), its own speculative round, and no batched decode.

REUSED FROM EXO (unchanged):
  * tokenizer loading + ``apply_chat_template`` (a model id containing
    ``deepseek-v4`` already routes through the vendored DSv4 encoder),
  * ``parse_thinking_models`` / ``parse_deepseek_v4`` / ``map_responses_to_chunks``
    (see ``dsv41/output.py``),
  * task/cancel agreement (``mx_any`` / ``mx_all_gather_tasks``, coord subgroup),
  * ``GenerationStats`` / ``Usage`` / ``GenerationResponse`` and the runner's
    status heartbeat.

DELIBERATELY NOT REUSED (and why):
  * ``mlx_generate``        -- assumes an mlx-lm cache + ``stream_generate``.
  * ``KVPrefixCache``       -- DSv4.1's cache is a custom per-layer structure
    (window ring + compressed KV + compressor carry + index keys + engram id
    history); prefix reuse is ``dsv41/session.py`` (one conversation per
    ``ModelCache``) instead.
  * ``tensor_auto_parallel``-- TP is built into the loader.
  * sampling                -- v1 is GREEDY ONLY (the DSpark verify loop is
    greedy). A request that asks for a non-greedy temperature is REFUSED rather
    than silently served greedy; ``_resolve_sampler`` is workstream F's hook.
  * ``KVPrefixCache``       -- prefix reuse is ``dsv41/session.py`` (see below).

VISION. A request carrying images renders the DSv4.1 image placeholder into the
prompt, expands it through ``deepseek_v41.image_processor.prepare_vl_inputs``,
and splices the merged ``(1, s, dim)`` embedding tensor (token lookup with each
image span overwritten by its ViT/aligner block) in as ``model.embed`` for the
prompt prefill (``dsv41/vision.py``). The reference's rule that an image span
must be prefilled in ONE forward from position 0 is enforced by the splice.

SESSIONS. Multi-turn requests join a conversation session
(``dsv41/session.py``): turn N+1 prefills only the delta on the previous turn's
live cache (+ DSpark draft windows), which is what makes a follow-up turn cheap.
An image-carrying request, or one that disables prefix reuse, runs as a cold
one-off session that is dropped when the request finishes.

REQUEST PATH. ``step()`` serves one request at a time (batch size 1): prompt ->
chunked fenced prefill -> first token -> greedy rounds (speculative when the
draft head is loaded) -> EOS / stop / length. Tokens are emitted as soon as
their round resolves, detokenized with the same streaming detokenizer mlx-lm
uses.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, BinaryIO

import mlx.core as mx

from exo.api.types import (
    FinishReason,
    GenerationStats,
    TopLogprobItem,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import GenerationTask, TaskId, TextGeneration
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.cache import encode_prompt
from exo.worker.engines.mlx.constants import MAX_TOKENS
from exo.worker.engines.mlx.dsv41.agreement import RankAgreement
from exo.worker.engines.mlx.dsv41.coord import ParkCoordinator
from exo.worker.engines.mlx.dsv41.degeneration import (
    DEGENERATION_ERROR_TEXT,
    REPETITION_LOOP_ERROR_CODE,
    DegenerationGuard,
    LoopAction,
    LoopHit,
    assert_guard_env_agrees_across_ranks,
    guard_env_digest,
    guard_env_report,
    loop_action,
    write_degeneration_dump,
)
from exo.worker.engines.mlx.dsv41.errors import (
    Dsv41InvalidRequest,
    Dsv41UnsupportedFeature,
    reclassify_input_error,
)
from exo.worker.engines.mlx.dsv41.fingerprint import compute_fingerprint
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
from exo.worker.engines.mlx.dsv41.no_repeat import (
    BanTurnStats,
    NoRepeatNgram,
    format_ban_line,
)
from exo.worker.engines.mlx.dsv41.output import _starts_in_thinking, dsv41_output_parser
from exo.worker.engines.mlx.dsv41.reasoning_budget import (
    ReasoningBudgetLimiter,
    budget_tokens_from_request,
    make_reasoning_budget_limiter,
)
from exo.worker.engines.mlx.dsv41.rounds import (
    _cache_capacity,
    _final_response,
    _mid_response,
    _one_round,
    _queue_of,
    _refuse_unsupported,
    _row_logprobs,
    _spec_policy,
    _stop_index,
    _stop_sequences,
)
from exo.worker.engines.mlx.dsv41.session import (
    Dsv41Sessions,
    TurnOutcome,
    ckpt_tick_seconds,
)
from exo.worker.engines.mlx.dsv41.vision import (
    Dsv41Vision,
    build_embeddings,
    image_spans,
    prompt_tokens_for_request,
    render_prompt,
    splice_embeddings,
)
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    safe_think_token_id,
)
from exo.worker.engines.mlx.utils_mlx import apply_chat_template, get_coord_group
from exo.worker.runner.bootstrap import logger


def _maybe_install_next18_capture() -> None:
    """Install the next18 real-tensor capture hook when the operator opted in.

    The R1 ship precondition A2 / R2 hard gate needs the DSv4.1 indexer's REAL
    decode/verify tensors recorded *inside the model-server process*, which the
    offline synthetic suite cannot give. ``bench/next18_capture.py`` provides the
    hook, but nothing in the stock server imports it, so a capture path exported
    in the operator's shell is (correctly) inert.

    This is the ONE bootstrap point: the DSv4.1 engine is constructed on every
    rank of every node exactly once, before any request is served, and only for
    this model. The hook is installed here IFF both hold:

      * ``DSV41_NEXT18_CAPTURE`` is set to a non-empty output path (the operator
        opted in), AND
      * the ``next18_capture`` module is importable (the harness file is staged on
        ``sys.path`` by the operator's ``PYTHONPATH`` -- e.g. the R1 capture run
        copies ``bench/next18_capture.py`` to each node's ``/tmp`` and exports
        ``PYTHONPATH`` accordingly, mirroring the existing ``EXO_PYTHONPATH_DIAG``
        mechanism in ``start_cluster.sh``).

    WHY IT NEVER FIRES IN PRODUCTION. With ``DSV41_NEXT18_CAPTURE`` unset this
    function returns immediately -- no import, no monkeypatch, no cost -- so the
    production path is byte-for-byte unchanged. The capture path is a separate
    process concern (its own module on ``PYTHONPATH``), never linked into exo or
    mlx-lm, so nothing about a production boot can reach it.

    ``next18_capture`` auto-installs on import when the env names a path; we log
    the confirmed install so the runner log is the on-node evidence the capture
    is live (``grep '[next18_capture]'``).
    """
    import os

    path = os.environ.get("DSV41_NEXT18_CAPTURE")
    if not path:
        return                                    # production: inert, zero cost
    try:
        import next18_capture as cap              # operator-staged on PYTHONPATH
    except Exception as e:  # noqa: BLE001 - capture must never break serving
        logger.warning(
            f"[DSV41] DSV41_NEXT18_CAPTURE={path} set but next18_capture is not "
            f"importable ({e!r}); capture DISABLED. Stage bench/next18_capture.py "
            f"on PYTHONPATH (see start_cluster.sh DSV41_NEXT18_CAPTURE* forwards)."
        )
        return
    try:
        if getattr(cap, "_CAP", None) is None:
            cap.install(path)                     # idempotent; env default path
        logger.info(f"[DSV41] next18 capture hook ACTIVE -> {path}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[DSV41] next18 capture auto-install failed: {e!r}")

#: Chunk size for the chunked prefill loop. PREFILL IS THE KNOWN BLOCKER for
#: this model (74 tok/s at 8K, Metal GPU-timeout above 16K -- exo phase 19), and
#: workstream C owns the fix. Until then the engine keeps chunks small so a long
#: prompt cannot build one enormous lazy graph, fences each multi-row forward
#: (``dsv41.session.engine_prefill``'s model-level eval fence) and shrinks the
#: chunk further as the context grows (the transient-budget policy there).
DEFAULT_PREFILL_CHUNK = 512

#: Warmup generation length. Short on purpose: the point is to compile the EXL3
#: kernels, not to measure anything (the first forward is the expensive one --
#: exo phase 19 measured 206 s for a 2K first chunk).
DEFAULT_WARMUP_TOKENS = 8


def _dtype_of(loaded: Any, model: Any) -> str:
    """Best-effort KV storage dtype for the park build fingerprint (plan §G2).

    Source, in order: an explicit ``dtype`` on the loaded record, then on the
    model args, then on the first built layer/cache the model exposes. When none
    is reachable the stable string ``"unknown"`` is used -- deterministic across
    restarts of the same build, so the layout fingerprint still gates a real
    layout change; a change to the KV storage dtype is already caught by
    ``kv_code_fp`` (the fork's ``cache`` module, which fixes the grid dtype, is
    allowlisted).
    """
    for obj in (
        loaded,
        getattr(loaded, "args", None),
        getattr(model, "args", None),
    ):
        dt = getattr(obj, "dtype", None)
        if dt is not None:
            return str(dt)
    layers = getattr(model, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        dt = getattr(layers[0], "dtype", None)
        if dt is not None:
            return str(dt)
    return "unknown"


#: Prompt used for warmup; deliberately a plain chat turn.
_WARMUP_PROMPT = "Reply with the single word: ready"

#: Max tolerated gap between fence-point liveness callbacks. The hook fires at
#: every model-level eval fence (``fence_every`` layers of committed compute),
#: and ``prefill_heartbeat`` re-emits the runner status at most every 15 s, so
#: the runner's silence clock is bounded by ~15 s + the longest between-fence
#: region. A spacing above this is the failure PREDICTOR (a fence gap growing
#: toward the supervisor's 45 s window): it is measured and warned, never
#: asserted (see the design doc's corrected invariant).
FENCE_HEARTBEAT_MAX_SPACING_SECONDS = 30.0

#: Exhaustion marker for step()'s parser pull (the parser also yields None).
_END = object()

#: Per-request observability: how often the committed-token count + tokens/s
#: line is emitted while a turn is generating. Deliberately wall-clock spaced
#: (not token spaced): a degenerate loop and a healthy long generation both need
#: to be readable from the runner log at a predictable cadence.
OBS_RATE_INTERVAL_SECONDS = 10.0

#: Canonical OpenAI error code for a request whose prompt exceeds the served
#: window. Emitted on the wire so OpenAI-compatible clients (and Hermes'
#: error classifier, which maps ``context_length_exceeded`` ->
#: _V_CONTEXT_OVERFLOW) route it to compaction instead of blind retries.
CONTEXT_LENGTH_EXCEEDED_CODE = "context_length_exceeded"

# ---------------------------------------------------------------------------
# output-length defaults (decode liveness' sibling: bound the runaway turn)
# ---------------------------------------------------------------------------
# A request that sends NO ``max_output_tokens`` (or a non-positive one) used to
# fall straight through to ``MAX_TOKENS`` (32168, mlx/constants.py), so a
# runaway generation could hold the one served slot -- and, at TP=2, BOTH
# ranks, which are locked in the same collectives -- for 32K tokens with the
# client long gone. The DEFAULT knob caps that at a size that still leaves room
# for a long reasoning turn; the CEILING knob clamps ANY resolved value
# (including an explicit client maximum) so one request cannot pin the cluster.
# Both are read at RESOLVE time, like this package's other knobs, so a test can
# monkeypatch.setenv; both take 0 as "disabled": 0 for DEFAULT keeps the old
# ``MAX_TOKENS`` behaviour byte-for-byte, 0 for CEILING removes the clamp.
DEFAULT_MAX_OUTPUT_TOKENS_ENV = "EXO_DSV41_DEFAULT_MAX_OUTPUT_TOKENS"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
MAX_OUTPUT_TOKENS_CEILING_ENV = "EXO_DSV41_MAX_OUTPUT_TOKENS_CEILING"
MAX_OUTPUT_TOKENS_CEILING = 32768


def _read_int_env(name: str, default: int) -> int:
    """Read an integer knob, warning (never raising) on a malformed value.

    Same contract as ``dsv41.park._env_int`` / ``dsv41.session._read_float_env``:
    a typo in the launcher must not take the engine down at request time.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[DSV41] ignoring non-integer {name}={raw!r}; using {default}")
        return int(default)


def resolve_max_output_tokens(requested: int | None) -> int:
    """Resolve a request's output cap from the request and the two env knobs.

    Pure in the sense that matters here: no engine state is touched, and the
    result is a deterministic function of (``requested``, the environment).

      * a POSITIVE ``requested`` wins -- the client's own number is honoured;
      * ``None``/0/negative means "the client did not ask" -> the
        ``EXO_DSV41_DEFAULT_MAX_OUTPUT_TOKENS`` default (0 = keep
        ``MAX_TOKENS``);
      * the result is then clamped by ``EXO_DSV41_MAX_OUTPUT_TOKENS_CEILING``
        (0 = no clamp), which bounds a runaway generation even when the client
        explicitly asked for more than the instance should ever serve.
    """
    default = _read_int_env(DEFAULT_MAX_OUTPUT_TOKENS_ENV, DEFAULT_MAX_OUTPUT_TOKENS)
    ceiling = _read_int_env(MAX_OUTPUT_TOKENS_CEILING_ENV, MAX_OUTPUT_TOKENS_CEILING)
    if requested is None or requested <= 0:
        resolved = MAX_TOKENS if default <= 0 else default
    else:
        resolved = int(requested)
    if ceiling > 0:
        resolved = min(resolved, ceiling)
    return resolved


# ---------------------------------------------------------------------------
# no-repeat n-gram ban (greedy-only engine: the deterministic loop ESCAPE)
# ---------------------------------------------------------------------------
# See ``dsv41/no_repeat.py`` for the WHY. This engine refuses temperature > 0 and
# has no repetition-penalty pipeline, so an argmax attractor like the observed
# "Let me write." family is an ABSORBING state -- cloud stacks escape such states
# by sampling, and a no-repeat n-gram ban is the cheapest deterministic escape
# that keeps the engine greedy. EXO_DSV41_NO_REPEAT_NGRAM is the n-gram length:
# 0 or unset (the DEFAULT) = DISABLED = byte-identical behaviour, and only
# n >= no_repeat.MIN_NGRAM (3) has any effect; no_repeat.SUGGESTED_N (6) is the
# suggested value when enabled. EXO_DSV41_NO_REPEAT_NGRAM_WINDOW bounds how far
# back an earlier occurrence may sit, in PRODUCED tokens. Both are read at RESOLVE
# time, like this package's other knobs, so a test can monkeypatch.setenv.
NO_REPEAT_NGRAM_ENV = "EXO_DSV41_NO_REPEAT_NGRAM"
NO_REPEAT_NGRAM_DEFAULT = 0
NO_REPEAT_NGRAM_WINDOW_ENV = "EXO_DSV41_NO_REPEAT_NGRAM_WINDOW"
NO_REPEAT_NGRAM_WINDOW_DEFAULT = 2048
#: Comma-separated EXTRA control-token ids that must never be banned. OPTIONAL
#: and unset by default: the EOS and think start/end ids this engine already
#: resolves are exempt WITHOUT it (see resolve_no_repeat_ngram_exempt_ids).
NO_REPEAT_NGRAM_EXEMPT_ENV = "EXO_DSV41_NO_REPEAT_NGRAM_EXEMPT_IDS"


def resolve_no_repeat_ngram_exempt_ids(
    tokenizer: Any | None = None,
) -> frozenset[int]:
    """Control-token ids the no-repeat n-gram ban must NEVER mask (pure).

    Repetition of special/control tokens is normal: EOS ends a turn and the think
    start/end delimiters open/close a reasoning block, so banning one could stop
    the model closing a block or a tool call. The DEFAULT exempt set is the
    tokenizer's EOS ids plus the think start/end ids this engine already resolves
    for the reasoning-budget limiter (``safe_think_token_id``, which degrades to
    None rather than raising). The comma-separated
    ``EXO_DSV41_NO_REPEAT_NGRAM_EXEMPT_IDS`` EXTENDS the set; a malformed entry
    warns and is skipped, never crashes.

    Pure in the same sense as the other resolvers: a deterministic function of the
    environment and the tokenizer alone, so both TP ranks resolve the identical
    set with no collective.
    """
    ids: set[int] = set()
    raw = os.environ.get(NO_REPEAT_NGRAM_EXEMPT_ENV)
    if raw is not None and raw.strip():
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                logger.warning(
                    f"[DSV41] ignoring non-integer {NO_REPEAT_NGRAM_EXEMPT_ENV} "
                    f"entry {part!r}"
                )
    if tokenizer is not None:
        for eos in getattr(tokenizer, "eos_token_ids", None) or ():
            try:
                ids.add(int(eos))
            except (TypeError, ValueError):
                continue
        for attr in ("think_start_id", "think_end_id"):
            tid = safe_think_token_id(tokenizer, attr)
            if tid is not None:
                ids.add(int(tid))
    return frozenset(ids)


def resolve_no_repeat_ngram(tokenizer: Any | None = None) -> NoRepeatNgram:
    """Resolve the no-repeat n-gram ban from the environment (pure).

    No engine state is touched: the result is a deterministic function of the two
    env knobs (and, when the knob is ON, the tokenizer's control-token ids) alone,
    exactly like ``resolve_max_output_tokens``. An unset or malformed knob
    resolves to the disabled default (``n=0``), so a launcher typo cannot turn the
    ban on by accident -- and a knob that turns it on is still inert below
    ``no_repeat.MIN_NGRAM``. The control-token exemption is installed ONLY when the
    ban is enabled, so the disabled path touches neither the tokenizer nor the env.
    """
    n = _read_int_env(NO_REPEAT_NGRAM_ENV, NO_REPEAT_NGRAM_DEFAULT)
    window = _read_int_env(
        NO_REPEAT_NGRAM_WINDOW_ENV, NO_REPEAT_NGRAM_WINDOW_DEFAULT
    )
    ban = NoRepeatNgram(n=n, window=window)
    if ban.enabled:
        ban = ban.with_exempt(resolve_no_repeat_ngram_exempt_ids(tokenizer))
    return ban


class Dsv41ContextLengthExceeded(Dsv41UnsupportedFeature):  # noqa: N818 - name
    """The request's prompt exceeds this instance's served context window.

    A deterministic CLIENT-INPUT refusal -- the same request will never fit --
    so it is a distinct type. It is still a
    :class:`Dsv41UnsupportedFeature` (same catch sites, same fail-one-request
    semantics) but carries the canonical ``context_length_exceeded`` code so
    the API layer can hand clients a classifiable error instead of a generic
    "internal error" stream event.
    """

    #: OpenAI-style structured code carried to the client.
    code: str = CONTEXT_LENGTH_EXCEEDED_CODE


def _context_length_message(prompt_len: int, max_tokens: int, capacity: int) -> str:
    """The refusal text, phrased in OpenAI's standard context-length wording.

    ``maximum context length is {N} tokens`` is the exact phrase OpenAI's own
    overflow 400 uses, so any client that pattern-matches context overflow
    (including wording-only clients that ignore the structured code) classifies
    it correctly. The leading ``DSV4.1:`` tag is retained for operator logs,
    and the numeric breakdown is preserved (no semantic change).
    """
    return (
        f"DSV4.1: this model's maximum context length is {capacity} tokens. "
        f"However, your request resulted in a prompt of {prompt_len} prompt "
        f"tokens plus {max_tokens} max_output_tokens, which exceeds the "
        f"{capacity}-token cache this instance was configured for (prompt + "
        f"max_output_tokens + 8 > capacity; max_kv_tokens / card "
        f"context_length). Please reduce the length of the messages."
    )


def _image_span_end(image_inputs: Any) -> int:
    """Last token row any image span covers (0 when there are no images)."""
    return max((end for _start, end in image_spans(image_inputs)), default=0)


def _first_chunk_covers(total: int, chunk: int, span_end: int) -> list[int]:
    """A prefill chunk plan whose FIRST piece covers every image span.

    The DSv4 vision path's ``plan_prefill_chunks`` makes the same point: an
    image span is only valid inside the forward whose cache offset is 0, so a
    boundary that lands inside a span is a hard error rather than a slow path.
    Returns the piece sizes for ``SessionCache.append_turn(chunk_plan=...)``.
    """
    first = max(int(chunk), int(span_end), 1)
    if first >= total:
        return [total]
    rest = total - first
    return [first] + [int(chunk)] * ((rest + int(chunk) - 1) // int(chunk))


def _warmup_messages() -> list[InputMessage]:
    """The warmup request's messages (one plain user turn)."""
    return [InputMessage(role="user", content=_WARMUP_PROMPT)]


@dataclass
class _RoundStats:
    rounds: int = 0
    accepted: int = 0
    gammas: int = 0
    ms: float = 0.0

    def add(self, ms: float, accepted: int, gamma: int) -> None:
        self.rounds += 1
        self.accepted += accepted
        self.gammas += gamma
        self.ms += ms

    def stats(
        self, prompt_tps: float, prompt_tokens: int, generated: int
    ) -> GenerationStats:
        elapsed = self.ms / 1e3
        return GenerationStats(
            prompt_tps=prompt_tps,
            generation_tps=(generated / elapsed) if elapsed > 0 else 0.0,
            prompt_tokens=prompt_tokens,
            generation_tokens=generated,
            peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
            prefix_cache_hit="none",
        )


def _command_of(params: TextGenerationTaskParams) -> TaskId | None:
    """TaskId for prefill-progress events.

    ``TextGenerationTaskParams`` carries no command id; the runner correlates
    progress chunks by task id, and the engine only has it while a request is
    active. ``_Active`` therefore stamps progress events itself (see
    ``_generate``), so this helper exists only for the warmup path where there
    is no client to notify.
    """
    del params
    return None


class _NullTokenizer:
    """Unused placeholder kept out of the type graph (see _Active)."""


@dataclass
class _Active:
    task: TextGeneration
    generator: Generator[GenerationResponse]
    output_generator: Iterator[GenerationChunk | None]
    #: Rows this request's prefill will feed (progress chunk totals).
    prefill_total: int = 0


@dataclass(eq=False)
class Dsv41Engine(Engine):
    """Batch-size-1, greedy, speculative-decode engine for DeepSeek-V4.1."""

    loaded: Dsv41Loaded
    model_id: ModelId
    group: mx.distributed.Group | None
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    device_rank: int
    max_kv_tokens: int | None = None
    prefill_chunk_size: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    #: Speculative decode: ``gamma`` draft length, adaptive policy per round.
    speculative: bool = True
    gamma: int = 3
    adaptive_gamma: bool = True
    #: Sampling hook (workstream F). None => greedy, the only mode implemented.
    sampler: Any | None = None
    #: The loaded DSv4.1 vision tower (``dsv41.vision.Dsv41Vision``), or None
    #: when this instance serves text only. An image-carrying request against a
    #: None tower is refused rather than answered about nothing.
    vision_processor: Dsv41Vision | None = None
    #: Resident conversation sessions (each holds a full cache + draft window).
    max_sessions: int = 2
    #: Prefill chunking once a conversation passes ``long_threshold`` rows.
    #: ``long_threshold`` defaults to None: the engine's own prefill uses the
    #: transient-budget policy (:func:`dsv41.session.choose_prefill_step`), and
    #: a caller that sets a finite threshold reverts to the fixed-crossover
    #: ``long_chunk`` behaviour.
    long_chunk: int | None = None
    long_threshold: int | None = None
    #: Per-chunk transient budget (MB) for the prefill chunk-size policy; None
    #: => ``EXO_PREFILL_TRANSIENT_BUDGET_MB`` (default 2048 MB).
    prefill_transient_budget_mb: int | None = None
    #: Instance-level prefill cap, resolved from
    #: ``EXO_PREFILL_STEP_SIZE``/instance metadata by the builder.
    prefill_heartbeat_seconds: float = 15.0

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _agreement: RankAgreement = field(init=False)
    _active: _Active | None = field(default=None, init=False)
    #: The DSpark draft window for the request (``rounds._DRAFT_KEY``). The
    #: head's context is continuous over the request's tokens, so one window
    #: lives here for the whole request: primed from the first decode step's
    #: taps and appended to on every verify round (see ``rounds._one_round``).
    _draft_windows: dict[int, Any] = field(default_factory=dict, init=False)
    _stop_sequences: tuple[str, ...] = field(default=(), init=False)
    #: Conversation store: each session owns one ``ModelCache`` (+ the DSpark
    #: draft windows) and serves the turns of one conversation, which is what
    #: makes turn N+1's prefill only the delta. Created in ``__post_init__``.
    _sessions: Dsv41Sessions = field(init=False)
    #: Build fingerprint (plan §G2), computed ONCE in ``__post_init__`` from the
    #: checkpoint identity (model path/tokenizer/tensor-parallel layout). Tags
    #: every parked manifest this engine writes and gates every restore.
    _fingerprint: Any = field(init=False)
    #: Cumulative speculative counters since process start (rounds started,
    #: draft tokens accepted, draft tokens attempted). Deltas across
    #: successive requests give the live acceptance rate -- the decode-depth
    #: ladder reads these via the API's ``mtp_*`` stats fields.
    _spec_rounds: int = field(default=0, init=False)
    _spec_accepted: int = field(default=0, init=False)
    _spec_drafted: int = field(default=0, init=False)
    #: Monotonic timestamp of the previous ``_session_fence_heartbeat`` call,
    #: for the >30 s spacing watchdog. ``0.0`` (never called) is treated as
    #: "first call" and never warns.
    _last_fence_heartbeat_monotonic: float = field(default=0.0, init=False)
    #: Monotonic time of the last ``step()`` entry. The §G1 idle tick waits
    #: ``EXO_DSV41_CKPT_IDLE_S`` past it, so a peer still finishing the same
    #: request (identical stream, deterministic finish) is long out of
    #: ``step()`` before this rank issues any idle-tick collective.
    _last_step_monotonic: float = field(default=0.0, init=False)
    #: §G3: set once a shutdown is armed. ``_activate_next`` refuses to start a
    #: queued request while this is True, so "stop admitting new requests" holds
    #: even while the runner is still draining an in-flight turn.
    _shutting_down: bool = field(default=False, init=False)

    # ---------------------------------------------------------------- lifecycle

    def __post_init__(self) -> None:
        # Real-tensor capture (R1 ship precondition A2): install the next18 hook
        # when the operator opted in. No-op with DSV41_NEXT18_CAPTURE unset.
        _maybe_install_next18_capture()
        self._agreement = RankAgreement(get_coord_group(self.group))
        if self.loaded.head is None and self.speculative:
            logger.warning(
                "[DSV41] speculative decode requested but no draft head was "
                "loaded; serving plain greedy decode."
            )
        # Prefill fence + progress accounting: this checkpoint needs a small
        # chunk (see DEFAULT_PREFILL_CHUNK) unless the instance says otherwise.
        self._chunk = self.prefill_chunk_size or DEFAULT_PREFILL_CHUNK
        long_chunk = self.long_chunk or self._chunk
        # Plan §G2: compute the build fingerprint ONCE here -- at startup, never
        # lazily at shutdown and never per-park. It identifies the checkpoint
        # (model/tokenizer), the KV/forward code and the MLX build, and the TP
        # layout (world size, built layer range, dtype), so a parked cache
        # written by a different build is refused instead of silently served.
        #
        # The park layout DELIBERATELY OMITS ``rank``. The DeepSeek MLA cache
        # stores a SINGLE shared latent per token (``win_kv``/``comp_kv`` are
        # ``(bsz, ..., head_dim)``, NOT ``n_heads*head_dim``; ``attn.wkv`` is
        # built full on both ranks), so rank is NOT a cache-CONTENT axis. The two
        # nodes' TP device ranks are assigned by launch order and SWAP across a
        # graceful relaunch, so gating a park on rank made every parked dir
        # foreign after a swap (``park.skip reason=foreign-rank`` -> cold on both
        # nodes). Excluding rank makes the park fingerprint rank-INVARIANT and a
        # swapped-rank restore content-safe. (``fingerprint.layout_fingerprint``
        # itself stays rank-sensitive; only this park layout drops the axis.)
        self._fingerprint = compute_fingerprint(
            model_path=self.loaded.model_path,
            tokenizer=self.loaded.tokenizer,
            model=self.loaded.model,
            layout={
                "world_size": self.loaded.world,
                "layers": self.loaded.built_layers,
                "dtype": _dtype_of(self.loaded, self.loaded.model),
            },
        )
        self._sessions = Dsv41Sessions(
            self.loaded.model,
            self.loaded.head,
            max_seq_len=_cache_capacity(self),
            chunk=self._chunk,
            long_chunk=long_chunk,
            long_threshold=self.long_threshold,
            transient_budget_mb=self.prefill_transient_budget_mb,
            max_sessions=self.max_sessions,
            use_draft=bool(self.speculative and self.loaded.head is not None),
            # Wire the per-chunk progress hook through to the session cache: it
            # emits PrefillProgressChunk (rank 0) / heartbeats (rank != 0) during
            # a long prefill. Unwired, a prefill longer than the supervisor's
            # hang-watchdog window emits no events and kills a healthy runner.
            progress=self._session_progress,
            # Fence-point liveness: a hook keyed to the model's eval fence, so
            # an event is re-emitted at every K layers of committed compute even
            # when no prefill chunk boundary (the coarser ``progress`` signal)
            # is reached for minutes. Forwarded Dsv41Sessions -> Conversation ->
            # (the prefill_fn partial) -> engine_prefill; active on ALL ranks.
            fence_heartbeat=self._session_fence_heartbeat,
            # The build fingerprint, forwarded to the lazily-built park store.
            fingerprint=self._fingerprint,
            # Plan §G4: park/restore become CLUSTER decisions over the same
            # QP-less TCP coord subgroup RankAgreement uses (never the model TP
            # group). ``None`` (single rank / PP mode) keeps local-only parking.
            coordinator=ParkCoordinator.from_group(
                get_coord_group(self.group), rank=self.loaded.rank),
            rank=self.loaded.rank,
        )
        # Degeneration-guard configuration, logged ONCE per rank at startup: the
        # guard's verdict is a pure function of the committed token ids (which are
        # byte-identical on both TP ranks), so the ONLY way the two ranks could
        # disagree about whether to kill a turn is a knob that differs between the
        # nodes -- and the launcher builds each runner's environment from an
        # explicit allow-list, so an unlisted variable is silently absent. This
        # line makes the EFFECTIVE values checkable (and diffable between the two
        # runner logs) instead of assumed.
        logger.info(guard_env_report(self.loaded.rank, self.loaded.world))
        # ... and FAIL THE BOOT if the ranks disagree. A knob that differs
        # between ranks desyncs the guard: one rank terminates a degenerate turn
        # while its peer keeps decoding, and the peer then wedges in the model's
        # TP all_sum. The log line above is only checkable by eye; this makes the
        # mismatch fatal at the existing construction-point coord collective
        # (get_coord_group, called by both ranks in matching order two lines up).
        assert_guard_env_agrees_across_ranks(self._agreement.group)

    def warmup(self) -> None:
        """A short real generation, to compile the EXL3/Metal kernels.

        The first forward of this model pays a large kernel-compile cost (exo
        phase 19: 206 s for a 2K first chunk). Paying it here keeps it out of
        the first client request's latency.
        """
        params = TextGenerationTaskParams(
            model=self.model_id,
            input=_warmup_messages(),
            max_output_tokens=DEFAULT_WARMUP_TOKENS,
            temperature=0.0,
            enable_thinking=False,
        )
        prompt = apply_chat_template(self.loaded.tokenizer, params)
        started = time.perf_counter()
        tokens = 0
        for _response in self._generate(params, prompt, task_id=None):
            tokens += 1
        logger.info(
            f"[DSV41] warmup: {tokens} token(s) in "
            f"{time.perf_counter() - started:.1f}s (kernel compile); "
            f"peak={mx.get_peak_memory() / 1e9:.1f} GB"
        )
        mx.clear_cache()

    def submit(self, task: GenerationTask) -> None:
        assert isinstance(task, TextGeneration)
        self._agreement.submit(task)

    # ------------------------------------------------------- §G1 idle checkpoint

    def idle_tick_seconds(self) -> float | None:
        """Runner idle-tick cadence; ``None`` unless ``EXO_DSV41_CKPT=1``."""
        return ckpt_tick_seconds()

    def idle_tick(self, request_pending: Any = None) -> Any:
        """One between-requests tick: maybe disk-checkpoint ONE idle conversation.

        Runs on the runner/engine thread (the runner calls it from its idle wait),
        so every MLX call stays on the thread that owns the streams. Sequence:

        1. LOCAL gate, no collectives: no request active/queued on this rank and
           ``Dsv41Sessions.checkpoint_candidate`` names a conversation (between
           turns, ``>= PARK_MIN_TOKENS``, dirty ``>= CKPT_MIN_DELTA`` or never on
           disk, idle ``>= CKPT_IDLE_S``). Nothing qualifies -> return; the
           common idle tick costs nothing and issues NO collective.
        2. ``agree_on_tasks()`` -- the SAME collective prefix ``step()`` issues
           with no active request. If ANY rank has a pending task (the peer is in
           ``step()``), the exchange pairs with the peer's own
           ``agree_on_tasks`` (this rank contributes no task, so nothing is
           agreed) and the tick stands down; the runner then picks up the task
           and both ranks meet in ``step()``.
        3. Only when every rank is task-free (so the peer is in its own idle
           tick at the same program point) does the coordinated checkpoint run
           (``Dsv41Sessions.maybe_checkpoint`` -> IDLE agreement, G4 agreed N,
           per-unit cancellation polls, PARK_ACK).

        ``request_pending`` (from the runner: its work queue is non-empty) is
        polled between units; any rank seeing a request makes every rank abort
        within one unit. Never raises on LOCAL errors; a coord transport fault
        propagates exactly as it does from ``step()``.
        """
        if self._active is not None or self._agreement.queue or self._agreement.maybe_queue:
            return None  # not idle on this rank: never enter a collective here
        if time.monotonic() - self._last_step_monotonic < self._sessions.ckpt_idle_s:
            return None  # this rank served a request too recently
        try:
            cand = self._sessions.checkpoint_candidate()
        except Exception as exc:  # noqa: BLE001 - the gate must never raise
            logger.warning(f"[DSV41] checkpoint gate failed ({type(exc).__name__}: {exc})")
            return None
        if cand is None:
            return None
        if self._agreement.agree_on_tasks() or self._agreement.queue:
            # A peer has a task (it is in step()); this rank contributed none, so
            # nothing was agreed. Stand down and let the runner fetch the task.
            return None
        local_pending = request_pending if request_pending is not None else (lambda: False)

        def pending() -> bool:
            return bool(local_pending()) or bool(self._agreement.maybe_queue)

        try:
            return self._sessions.maybe_checkpoint(request_pending=pending, candidate=cand)
        finally:
            # Return the freed CPU staging copies to the OS between requests, as
            # the runner's own idle reclaim does; never inside the decode loop.
            mx.clear_cache()

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        self._last_step_monotonic = time.monotonic()
        if self._active is None:
            self._agreement.agree_on_tasks()
            self._activate_next(output)
            if self._active is None:
                output.extend(
                    (task_id, CancelledResponse()) for task_id in self._cancelled_tasks
                )
                return iter(output)

        active = self._active
        assert active is not None
        try:
            # Pull ONLY through the parser pipeline (it wraps active.generator);
            # a direct next(active.generator) here would steal a token per step.
            # None = flush point (end of this step); _END = stream exhausted.
            while True:
                parsed = next(active.output_generator, _END)
                if parsed is _END:
                    raise StopIteration
                if parsed is None:
                    # A flush point with nothing released (parser still holding a
                    # marker): keep pulling so a step never comes back empty mid-turn.
                    if output:
                        break
                    continue
                output.append((active.task.task_id, parsed))
        except (Dsv41UnsupportedFeature, Dsv41InvalidRequest) as e:
            # A request this build cannot serve (unsupported feature) or whose
            # own input is invalid (e.g. the literal image placeholder token in
            # message text): fail the REQUEST with the reason, keep the runner
            # (both ranks raise the same refusal for the same params, so they
            # stay in step).
            self._fail_request(active.task, e, output)
            self._activate_next(output)
        except (StopIteration, PrefillCancelled):
            # The parser pipeline BUFFERS (thinking split / DSML detection hold
            # tokens until they are disambiguated), so at end-of-stream whatever
            # it is still holding must be drained before the turn is closed --
            # otherwise the last tokens of a turn are silently dropped and the
            # client sees a truncated answer (observed: a 4-token turn arriving
            # as 2 tokens, with the rest stuck in the parser's buffer).
            while (parsed := next(active.output_generator, None)) is not None:
                output.append((active.task.task_id, parsed))
            output.append((active.task.task_id, FinishedResponse()))
            self._agreement.forget(active.task.task_id)
            self._active = None
            self._activate_next(output)
        except Exception as e:
            self._send_error(active.task, e)
            self._active = None
            raise
        return iter(output)

    def _fail_request(
        self,
        task: TextGeneration,
        e: Exception,
        output: list[
            tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
        ],
    ) -> None:
        """Fail one request, emit its terminal response, and forget it.

        Shared by step()'s refusal handlers and by ``_activate_next``, whose
        eager ``_render_prompt`` can raise before a request ever becomes active
        (the placeholder-in-text case that used to crash the runner): both must
        produce the SAME terminal shape -- an error chunk (rank 0 only, per
        ``_send_error``) plus FinishedResponse -- so the runner closes the task
        instead of waiting forever on a stream that will never start, and the
        next queued task can run.
        """
        self._send_error(task, e)
        output.append((task.task_id, FinishedResponse()))
        self._agreement.forget(task.task_id)
        self._active = None

    def _activate_next(
        self,
        output: list[
            tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
        ],
    ) -> None:
        """Start the next queued request, skipping ones whose START fails.

        A task whose own start (render/validation) fails with a request-level
        error is failed here and skipped; a genuine internal failure still
        propagates loudly. The loop matters as much as the catch:
        ``_activate_next`` is also called by step()'s handlers, so a bad task
        sitting at the head of the queue must not be able to crash the step
        that just finished a turn -- and with several bad tasks queued, one
        step must not raise on the first.
        """
        while self._agreement.queue:
            if self._shutting_down:
                # §G3 step 1: a shutdown is armed; do not start a new turn.
                return
            task = self._agreement.queue.popleft()
            try:
                self._active = self._start(task)
            except Dsv41InvalidRequest as e:
                logger.warning(
                    f"[DSV41] invalid request {task.task_id}: {e}; "
                    "failing the request and serving the next queued task."
                )
                self._fail_request(task, e, output)
                continue
            except Exception as e:
                # A start failure that is NOT request input is an engine bug: it
                # must crash loudly (the supervisor re-creates the runner), with
                # the client still told why -- exactly the pre-fix behaviour.
                self._send_error(task, e)
                raise
            return

    def close(self) -> None:
        self._active = None
        self._agreement.reset()
        self._draft_windows.clear()
        self._sessions.close()
        del self.loaded

    def reset_after_reconnect(self) -> list[int]:
        """Drop in-flight requests after a jaccl transport fault.

        Same contract as ``ExoBatchGenerator.reset_after_reconnect``: the model
        stays resident, and the dropped task ids are returned so the runner can
        fail them (clients retry). DSv4.1's cache lives only for the lifetime of
        one request, so there is nothing to rebuild.
        """
        dropped: list[int] = []
        if self._active is not None:
            dropped.append(int(self._active.task.task_id))
            self._active = None
        dropped.extend(int(t.task_id) for t in self._agreement.queue)
        self._agreement.reset()
        self._draft_windows.clear()
        # A session's cache belongs to the conversation, not the request, so it
        # survives the reconnect -- but the turn in flight is rolled back, so the
        # next request sees the pre-turn state instead of a half-fed cache.
        self._sessions.cancel_all()
        return dropped

    # ------------------------------------------------- §G3 park-on-shutdown

    def begin_shutdown(self) -> None:
        """§G3 step 1: stop admitting new requests.

        After this, ``_activate_next`` will not start a queued request, so the
        only turn that can still run is the one already in flight (drained by
        the runner). Idempotent.
        """
        self._shutting_down = True

    def shutdown_park(
        self,
        *,
        signal_time: float,
        budget: Any = None,
        now: Any = time.monotonic,
    ) -> Any:
        """§G3 steps (3)+(4) on the engine/runner thread: abort, then park.

        Called from the runner's ``main()`` finally while the model is still
        resident. Uses the EXISTING §G1/§G4 coordinated checkpoint path
        (``sessions.maybe_checkpoint`` -> ``coord.run_checkpoint``) so the final
        force checkpoint stays rank-invariant and COLD-safe on any disagreement —
        no new collective protocol.

        Releases every in-flight reference BEFORE the checkpoints (drop the
        ``_active`` request and its session from residency; clear the draft
        windows) so no MLX array is held across the next forward — the §G1
        donation trap does not apply to the coordination, but the abort itself
        must not pin the graph either.
        """
        from exo.worker.engines.mlx.dsv41 import shutdown as _sh

        self._shutting_down = True
        b = budget if budget is not None else _sh.shutdown_budget()
        # (3) release the active request; abort_inflight drops its conversation.
        self._active = None
        self._draft_windows.clear()

        def precheck() -> bool:
            # Mirror the §G1 idle tick: pair an agree_on_tasks with a peer that
            # may be inside step(), and stand down if either side holds a task.
            # (agree_on_tasks is ALWAYS issued — its collective must run.)
            return not (
                self._agreement.agree_on_tasks() or self._agreement.queue
            )

        return _sh.run_shutdown_park(
            self._sessions,
            signal_time=signal_time,
            budget=b,
            now=now,
            precheck=precheck,
        )

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise Dsv41UnsupportedFeature(
            "DSv4.1 does not implement disaggregated prefill: a remote prefill "
            "would have to ship this model's cache (window ring + compressed KV "
            "+ compressor carry + index keys + engram id history) over the wire, "
            "which is not built. Disable remote prefill for this model."
        )

    # ------------------------------------------------------------------ requests

    def _start(self, task: TextGeneration) -> _Active:
        """Build the request's generator + parser pipeline (eager render).

        The eager ``_render_prompt`` here runs outside any active-request try
        block, so a request-input ValueError (the literal image placeholder
        token in message text, an unreadable image block) has already been
        translated to :class:`Dsv41InvalidRequest` at the render seam, which is
        what lets ``_activate_next`` fail just this request and move on. Any
        other exception escapes unchanged -- a genuine engine bug must still
        crash loudly.
        """
        generator = self._build_generator(task)
        output_generator = dsv41_output_parser(
            _queue_of(generator),
            self.loaded.tokenizer,
            self._render_prompt(task.task_params),
            self.model_id,
        )
        return _Active(task, generator, output_generator)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            # A context-length refusal carries the canonical OpenAI
            # ``context_length_exceeded`` code so the API layer can hand the
            # client a classifiable error; every other failure stays a plain
            # message (no code).
            error_code = getattr(e, "code", None)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                        error_code=error_code,
                    ),
                )
            )

    def _render_prompt(self, params: TextGenerationTaskParams) -> str:
        """Chat-templated prompt for a request.

        Text-only requests go through exo's shared ``apply_chat_template``; an
        image-carrying request needs ``vision.render_prompt``, because the shared
        one keeps only the ``type == "text"`` content parts and would flatten the
        image block (and therefore the model's placeholder) away.

        Rendering is the EAGER step of starting a request (it runs before the
        generator is pulled), so a request-input validation failure from the
        vendored DSv4 encoder -- the literal image placeholder token typed into
        message text is the live one -- is reclassified to
        :class:`Dsv41InvalidRequest` here, at the source. ValueErrors that are
        not request input (engine bugs) re-raise unchanged and still crash the
        runner loudly.
        """
        try:
            if params.images and self.vision_processor is not None:
                return render_prompt(
                    self.loaded.tokenizer, params, self.vision_processor.placeholder
                )
            return apply_chat_template(self.loaded.tokenizer, params)
        except ValueError as e:
            reclassify_input_error(e)
            raise

    def _build_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        params = task.task_params
        prompt = self._render_prompt(params)
        return self._generate(params, prompt, task_id=task.task_id)

    def _conversation_key(self, params: TextGenerationTaskParams) -> str | None:
        """Client-visible conversation id, when the request carries one."""
        key = getattr(params, "correlation_id", None)
        return key if isinstance(key, str) and key else None

    # --------------------------------------------------------------- progress

    def _session_progress(self, chunks: int, rows_done: int, elapsed: float) -> None:
        """Per-prefill-chunk hook: cancellation + heartbeat + progress chunks.

        Called from inside the prefill driver while the request's generator is
        being advanced, which is exactly where the engine can still act on a
        cancellation: raising here unwinds the session turn, and the session
        rolls its own cache back.
        """
        del elapsed
        active = self._active
        task_id = active.task.task_id if active is not None else None
        self._check_cancel(task_id)
        if active is not None and self.device_rank == 0 and chunks > 1:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=active.task.command_id,
                    chunk=PrefillProgressChunk(
                        model=self.model_id,
                        processed_tokens=rows_done,
                        total_tokens=active.prefill_total or rows_done,
                    ),
                )
            )
        else:
            self.prefill_heartbeat()

    def _session_fence_heartbeat(self) -> None:
        """Fence-point liveness hook installed as ``model._fence_hook``.

        The mlx-lm body calls this after each model-level ``mx.eval`` fence
        (every ``_fence_every`` layers of a multi-row forward), so it re-emits
        the runner status at a point backed by actually-committed compute --
        granularity far finer than the per-CHUNK ``_session_progress`` (a chunk
        at deep context takes minutes, which is what let a healthy prefill go
        silent past the supervisor's window). Runs on the model thread: no
        locks, no model-state mutation, and the delegate is the same 15 s
        throttled ``prefill_heartbeat`` the progress path uses.

        The spacing watchdog only measures: it warns when the gap since the
        previous invocation exceeds ``FENCE_HEARTBEAT_MAX_SPACING_SECONDS``,
        the predictor of a fence gap growing toward the hang window.
        """
        now = time.monotonic()
        last = self._last_fence_heartbeat_monotonic
        # ``last == 0.0`` means no fence has fired yet in THIS request (reset at
        # turn start), so a stale timestamp from a previous request's last fence
        # cannot masquerade as a long gap here.
        if last > 0.0 and (now - last) > FENCE_HEARTBEAT_MAX_SPACING_SECONDS:
            logger.warning(  # pyright: ignore[reportUnknownMemberType]
                f"[DSV41] fence heartbeat spacing {now - last:.1f}s exceeds "
                f"{FENCE_HEARTBEAT_MAX_SPACING_SECONDS:.0f}s (rank="
                f"{self.device_rank}): a between-fence region is running long; "
                "supervisor silence may be approaching its window"
            )
        self._last_fence_heartbeat_monotonic = now
        self.prefill_heartbeat()

    # ------------------------------------------------------------------ generate

    def _generate(
        self,
        params: TextGenerationTaskParams,
        prompt: str,
        *,
        task_id: TaskId | None,
    ) -> Generator[GenerationResponse]:
        """Serve one request: prompt -> token ids -> session turn -> chunks.

        One turn, end to end: the conversation's delta prefill (through the
        session, so a follow-up turn only feeds what is new), then the engine's
        decode rounds from the prefill's anchor, then the detokenized stream.
        """
        # Wall-clock origin for this request's observability lines (time to first
        # committed token, the ~10 s throughput cadence, and the turn summary).
        request_started = time.perf_counter()
        model = self.loaded.model
        tokenizer = self.loaded.tokenizer
        vision = self.vision_processor
        _refuse_unsupported(params, vision_available=vision is not None)

        max_tokens = resolve_max_output_tokens(params.max_output_tokens)
        # One INFO line ONLY when the resolution actually changed the number
        # (a client that sent 512 is served 512 and logs nothing): an operator
        # can see a defaulted/clamped request without grepping every turn, and
        # both knobs are visible in the runner's startup env banner.
        if max_tokens != params.max_output_tokens:
            logger.info(
                f"[DSV41] output length resolve turn={task_id} "
                f"requested={params.max_output_tokens} resolved={max_tokens}"
            )
        eos_ids = set(tokenizer.eos_token_ids)
        stop_sequences = _stop_sequences(params)
        capacity = _cache_capacity(self)

        # -- prompt -> token ids. With images this is prepare_vl_inputs'
        #    expansion, which is NOT what tokenizer.encode(prompt) returns.
        image_inputs = None
        token_types: list[int] | None = None
        embeddings: mx.array | None = None
        if params.images:
            assert vision is not None  # _refuse_unsupported guarantees this
            tokens_list, token_types, image_inputs = prompt_tokens_for_request(
                vision, prompt, params.images, tokenizer
            )
            prompt_len = len(tokens_list)
            embeddings = build_embeddings(model, vision, tokens_list, image_inputs)
            mx.eval(embeddings)
            logger.info(
                f"[DSV41] prompt expanded with {len(params.images)} image(s): "
                f"{prompt_len} tokens, span(s)={image_spans(image_inputs)}"
            )
        else:
            prompt_tokens = encode_prompt(tokenizer, prompt)
            prompt_len = int(prompt_tokens.shape[0])
            # .tolist() converts in one call; iterating an mx.array and int()-ing
            # each element costs ~0.17 s per 16K prompt (identical list).
            tokens_list = [int(t) for t in prompt_tokens.tolist()]
        if prompt_len == 0:
            raise ValueError("DSV4.1: empty prompt after chat templating")
        if params.max_output_tokens is None:
            # Client did not ask for a length: generate up to what the cache holds.
            max_tokens = min(max_tokens, capacity - prompt_len - 8)
        if max_tokens < 1 or prompt_len + max_tokens + 8 > capacity:
            # Standard OpenAI wording + the canonical ``context_length_exceeded``
            # code (see ``_context_length_error`` / ``ErrorChunk.error_code``).
            # This is a deterministic CLIENT-INPUT refusal -- the prompt alone
            # exceeds the served window -- so it must be classifiable by any
            # OpenAI-compatible client (which keys on the wording/code and
            # compacts rather than burning retries on an opaque stream error).
            raise Dsv41ContextLengthExceeded(
                _context_length_message(prompt_len, max_tokens, capacity)
            )

        # -- conversation: every request joins one. The store reuses a resident
        #    conversation when this prompt extends it (that IS the multi-turn
        #    feature) or when the client names it, else it opens a cold one. A
        #    benchmark request that did not ask for reuse, and any image-carrying
        #    request (whose token stream must not join a cached conversation),
        #    drop their conversation when the request finishes.
        keep = not (params.bench and not params.use_prefix_cache) and embeddings is None
        # The per-request draft-window slot is only for engines with no session
        # (``rounds._one_round``'s fallback); a conversation owns its own.
        self._draft_windows.clear()
        # An image-carrying request must NOT reuse a resident conversation.
        # ``embeddings is not None`` is exactly the signal that this prompt's
        # prefill carries image spans -- it is what logs "prompt expanded with
        # N image(s)" just above. Reuse would delta-prefill from a nonzero cache
        # offset, but the vision splice requires the whole image span inside the
        # forward whose offset is 0, so it raises and (before this) killed the
        # runner: the warm-session image crash. Force a cold one-off session;
        # ``keep`` is already False for these requests, so it is dropped at end.
        session = self._sessions.get(
            tokens_list,
            self._conversation_key(params),
            reuse=embeddings is None,
        )
        if self._active is not None:
            self._active.prefill_total = prompt_len
        try:
            turn, anchor = self._start_turn(
                session,
                tokens_list,
                embeddings=embeddings,
                image_span_end=_image_span_end(image_inputs),
                token_types=token_types,
            )
        except PrefillCancelled:
            session.cancel()
            raise
        except Exception:
            if not keep:
                self._sessions.drop(session)
            raise
        finally:
            if self._active is not None:
                self._active.prefill_total = 0

        # -- token stream: emitted round by round as the rounds commit them.
        #    ``produced`` is every committed token (the cache holds all of
        #    them but the last), even past a stop point inside one round.
        produced: list[int] = [anchor]
        # Log-probs: computed only when asked (an extra small all_sum per round).
        top_n = int(params.top_logprobs or 0)
        lp_k = max(top_n, 1) if (params.logprobs or top_n) else 0
        anchor_lp: list[Any] = []
        _row_logprobs(turn.anchor_logits, lp_k, anchor_lp)

        # ── reasoning-budget limiter (PHASE 2 of the degeneration guard) ──
        # The generic MLX path forces a clean end-of-thinking transition once a
        # still-open reasoning block has run for 0.75 x the generation budget
        # (generate.make_reasoning_budget_limiter), so a self-doubt loop cannot
        # consume the whole max_tokens budget and leave the answer channel empty.
        # This engine has no logits-processor pipeline (its round argmaxes a whole
        # verify batch at once), so its round-batched port
        # (dsv41/reasoning_budget.py) is threaded into _rounds/_one_round instead.
        # Skipped for bench requests (parity with the generic call site: bench bans
        # EOS to force a fixed decode length, so an early think_end would only
        # perturb what bench exists to measure), and whenever the tokenizer exposes
        # no single-token think delimiters or the budget resolves <= 0.
        limiter: ReasoningBudgetLimiter | None = None
        if not params.bench:
            limiter = make_reasoning_budget_limiter(
                think_start_id=safe_think_token_id(tokenizer, "think_start_id"),
                think_end_id=safe_think_token_id(tokenizer, "think_end_id"),
                budget_tokens=budget_tokens_from_request(
                    params.reasoning_budget_tokens, max_tokens
                ),
                starts_in_thinking=_starts_in_thinking(prompt, tokenizer),
            )
        if limiter is not None:
            # The anchor IS the first generated token, and the generic limiter
            # counts it too -- feed it before the first round so the reasoning
            # window starts where the model's does (never inferred from markers).
            limiter.feed([anchor])
            logger.info(
                f"[DSV41] reasoning-budget limiter armed turn={task_id}: "
                f"budget={limiter.budget_tokens} think_start={limiter.think_start_id} "
                f"think_end={limiter.think_end_id} "
                f"starts_in_thinking={limiter.starts_in_thinking}"
            )

        def committed_tokens() -> Iterator[tuple[int, Any]]:
            yield anchor, (anchor_lp[0] if anchor_lp else None)
            for batch, lps in self._rounds(
                session, anchor, max_tokens - 1, logprobs=lp_k, limiter=limiter
            ):
                produced.extend(batch)
                for i, t in enumerate(batch):
                    yield t, (lps[i] if i < len(lps) else None)

        def as_logprob(entry: Any) -> Any:
            if entry is None:
                return None
            sel, ids, vals = entry
            items = [
                TopLogprobItem(
                    token=(s := tokenizer.decode([int(i)])),
                    logprob=float(v),
                    bytes=list(s.encode("utf-8")),
                )
                for i, v in zip(ids[:top_n], vals[:top_n], strict=True)
            ]
            return (float(sel), items)

        detokenizer = tokenizer.detokenizer
        emitted = 0
        accumulated = ""
        final_reason: FinishReason | None = None
        prefill_tps = (
            turn.prefill_tokens / turn.prefill_seconds
            if turn.prefill_seconds > 0
            else 0.0
        )
        last_tid = anchor
        # ── per-request observability + the degeneration (repetition-loop) guard ──
        # The guard is fed EVERY committed token, in order, as it is committed --
        # never a draft token (an unconfirmed draft never reaches the client and
        # must not shape the verdict) and never only a round boundary (one
        # speculative round commits a whole batch at once, while the detectors'
        # cycle alignment is defined over the raw committed stream). Both TP ranks
        # commit byte-identical tokens, so this verdict is rank-invariant and the
        # two ranks always stop together -- the per-request verdict needs no
        # collective (the startup knob-equality check in __init__ is separate).
        guard = DegenerationGuard(token_text=tokenizer.decode)
        guard_action: LoopAction = loop_action()

        def log_turn_end(reason: str, committed: int | None = None) -> None:
            """One bounded summary line per request (see OBS_RATE_INTERVAL_SECONDS).

            ``committed`` defaults to ``len(produced)`` -- every id the cache
            holds, which for a normal turn INCLUDES a final round's overshoot past
            a stop point (those ids were committed to the cache, just not
            emitted). The guard path passes ``guard_consumed`` instead: the
            aborted round's tail is rolled back and never committed, so the count
            the guard actually judged is the honest one there.
            """
            elapsed = time.perf_counter() - request_started
            committed_count = len(produced) if committed is None else committed
            rate = committed_count / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"[DSV41] obs turn-end turn={task_id} committed={committed_count} "
                f"emitted={emitted} elapsed={elapsed:.1f}s tok_s={rate:.1f} "
                f"reason={reason}"
            )

        first_token_logged = False
        last_rate_log = request_started
        #: Committed tokens actually handed to the guard (``produced`` is extended by
        #: a whole ROUND before its tokens are yielded, so it runs ahead of the fire
        #: point; the dump must be the stream the guard JUDGED, not the round's tail).
        guard_consumed = 0
        try:
            for tid, lp_entry in committed_tokens():
                last_tid = tid
                # Parity with the generic path, which never feeds the token that
                # carries the terminating finish_reason: the EOS token ENDS the
                # turn, it is not part of the generated content the guard judges.
                # (A single EOS could not form a cycle anyway, but skipping it
                # keeps the fed stream byte-identical to the generic path's.)
                if tid in eos_ids:
                    final_reason = "stop"
                    break
                guard_consumed += 1
                hit = guard.feed(tid)
                if hit is not None:
                    # A confirmed repetition cycle: log it (all ranks, identical
                    # apart from the env digest), dump the tail on rank 0, and --
                    # for a terminating action -- close the turn through the
                    # engine's EXISTING terminal path. ``tid`` is deliberately NOT
                    # detokenized: the token that completes the cycle is the one
                    # token we can still not send.
                    self._record_degeneration(
                        hit=hit,
                        action=guard_action,
                        tokenizer=tokenizer,
                        task_id=task_id,
                        produced=produced[:guard_consumed],
                        emitted=emitted,
                        prompt_tokens=turn.prompt_tokens,
                    )
                    if guard_action in ("stop", "error"):
                        self._abort_turn(session, keep)
                        log_turn_end(
                            f"degeneration-{guard_action}", committed=guard_consumed
                        )
                        yield _final_response(
                            token=tid,
                            text=(
                                DEGENERATION_ERROR_TEXT
                                if guard_action == "error"
                                else ""
                            ),
                            prefill_tps=prefill_tps,
                            prompt_tokens=turn.prompt_tokens,
                            generated=emitted,
                            reason=("error" if guard_action == "error" else "stop"),
                            task_id=task_id,
                            reused_tokens=turn.reused_tokens,
                            # Structured, classifiable code on the clean-fail so a
                            # client can react without string-matching the message.
                            error_code=(
                                REPETITION_LOOP_ERROR_CODE
                                if guard_action == "error"
                                else None
                            ),
                            mtp_cycles=self._spec_drafted,
                            mtp_accepted=self._spec_accepted,
                        )
                        return
                if not first_token_logged:
                    first_token_logged = True
                    logger.info(
                        f"[DSV41] obs first-token turn={task_id} "
                        f"latency={time.perf_counter() - request_started:.2f}s "
                        f"(prefill {turn.prefill_seconds:.2f}s, "
                        f"{turn.prompt_tokens} prompt tokens, "
                        f"reuse={turn.reused_tokens})"
                    )
                now = time.perf_counter()
                if now - last_rate_log >= OBS_RATE_INTERVAL_SECONDS:
                    done = emitted + 1
                    elapsed = now - request_started
                    logger.info(
                        f"[DSV41] obs progress turn={task_id} committed={done} "
                        f"elapsed={elapsed:.1f}s "
                        f"tok_s={(done / elapsed if elapsed > 0 else 0.0):.1f}"
                    )
                    last_rate_log = now
                detokenizer.add_token(tid)
                text = detokenizer.last_segment
                emitted += 1
                accumulated += text
                stop_hit = _stop_index(accumulated, stop_sequences)
                if stop_hit is not None:
                    text = text[: max(0, len(text) - (len(accumulated) - stop_hit))]
                    final_reason = "stop"
                if final_reason is None and emitted >= max_tokens:
                    final_reason = "length"
                if final_reason is not None:
                    self._end_turn(session, tokens_list, turn, produced, keep)
                    log_turn_end(str(final_reason))
                    yield _final_response(
                        token=tid,
                        text=text,
                        prefill_tps=prefill_tps,
                        prompt_tokens=turn.prompt_tokens,
                        generated=emitted,
                        reason=final_reason,
                        task_id=task_id,
                        reused_tokens=turn.reused_tokens,
                        logprob=as_logprob(lp_entry),
                        mtp_cycles=self._spec_drafted,
                        mtp_accepted=self._spec_accepted,
                    )
                    return
                yield _mid_response(tid, text, task_id, as_logprob(lp_entry))
        except GeneratorExit:
            # The consumer abandoned the generator (a close/GC), which is NOT a
            # client cancel: keep the pre-existing rollback semantics but do not
            # treat it as an incident worth dumping -- nothing was cut short.
            session.cancel()
            if not keep:
                self._sessions.drop(session)
            raise
        except BaseException as exc:
            # A client cancel (or any other unwind) mid-turn: capture the
            # committed tail before it is rolled away. This is the observability
            # half of the incident that motivated the guard -- a runaway
            # generation was cancelled by hand and NOTHING of its text was
            # recorded anywhere. The reason carries the exception type so the dump
            # is self-explanatory.
            reason = (
                "client-cancel"
                if isinstance(exc, PrefillCancelled)
                else f"aborted-{type(exc).__name__}"
            )
            self._dump_short_turn(
                produced,
                reason=reason,
                task_id=task_id,
                generated=emitted,
            )
            # A cut-short turn still gets its one summary line (the guard-fire and
            # normal paths already do), so every turn is accounted for; the count
            # is what was judged before the unwind, not the rolled-back round tail.
            log_turn_end(reason, committed=guard_consumed)
            session.cancel()
            if not keep:
                self._sessions.drop(session)
            raise
        final_reason = final_reason or ("stop" if emitted == 0 else "length")
        log_turn_end(str(final_reason))
        self._end_turn(session, tokens_list, turn, produced, keep)
        # the stream ended on an EOS token, or on the conversation's budget
        yield _final_response(
            token=last_tid,
            text="",
            prefill_tps=prefill_tps,
            prompt_tokens=turn.prompt_tokens,
            generated=emitted,
            # EOS break sets final_reason="stop"; only budget exhaustion is "length".
            reason=final_reason,
            task_id=task_id,
            reused_tokens=turn.reused_tokens,
            mtp_cycles=self._spec_drafted,
            mtp_accepted=self._spec_accepted,
        )

    def _end_turn(
        self,
        session: Any,
        tokens: list[int],
        turn: TurnOutcome,
        produced: list[int],
        keep: bool,
    ) -> None:
        """Close a turn: history in step with the cache, checkpoint, maybe drop."""
        turn.tokens = list(produced)
        session.sync_history(tokens, turn)
        session.finish(checkpoint=True)
        turn.committed = True
        if not keep:
            self._sessions.drop(session)

    def _abort_turn(self, session: Any, keep: bool) -> None:
        """Roll an in-flight turn back instead of committing it.

        Mirrors the engine's FAILURE path exactly (``session.cancel()`` plus the
        drop of a one-off conversation) rather than its completion path, for the
        one caller that exists today: a degeneration guard-fire. That is
        deliberate. ``_end_turn`` would ``sync_history`` + ``finish(checkpoint=True)``
        the whole (degenerate) turn, publishing a collapsed generation as a
        reusable checkpoint and parking it to SSD with the conversation -- the
        next turn on that conversation would then extend a corrupt state. After
        this rollback the cache, the draft window and the token history all sit at
        the pre-turn point the cache genuinely holds, which is what ``Conversation.cancel``
        re-establishes (it rewinds to ``_base`` and prunes the orphaned draft/anchor
        snapshots the decode-side ladder took above it).
        """
        session.cancel()
        if not keep:
            self._sessions.drop(session)

    def _record_degeneration(
        self,
        *,
        hit: LoopHit,
        action: LoopAction,
        tokenizer: Any,
        task_id: TaskId | None,
        produced: list[int],
        emitted: int,
        prompt_tokens: int,
    ) -> None:
        """Log a confirmed repetition cycle and dump the committed tail.

        The line is emitted on EVERY rank (identical but for
        ``guard_env_digest``, which is a pure function of the guard knobs): the
        two ranks can be diffed directly on a guard-fire, which is the cross-rank
        knob check for this feature. The dump itself is rank 0 only -- one copy on
        the API/log node rather than two.
        """
        try:
            cycle_text = tokenizer.decode(hit.cycle_token_ids)
        except Exception:  # noqa: BLE001 - a diagnostic must never break serving
            cycle_text = "<decode-failed>"
        units = (
            "raw-token cycle"
            if hit.detector == "token"
            else "block cycle (x24-token blocks)"
        )
        logger.warning(  # pyright: ignore[reportUnknownMemberType]
            f"[DSV41] DEGENERATION DETECTED turn={task_id} "
            f"at committed={len(produced)} emitted={emitted}: "
            f"{units} period={hit.period} repeated>={hit.repeats}x "
            f"action={action} guard_env_digest={guard_env_digest()} "
            f"prompt_tokens~{prompt_tokens} "
            f"cycle_token_ids={hit.cycle_token_ids} cycle_text={cycle_text!r}"
        )
        self._dump_short_turn(
            produced,
            reason=f"guard-{hit.detector}",
            task_id=task_id,
            generated=emitted,
            extra={
                "detector": hit.detector,
                "period": hit.period,
                "repeats": hit.repeats,
                "action": str(action),
            },
        )

    def _dump_short_turn(
        self,
        produced: list[int],
        *,
        reason: str,
        task_id: TaskId | None,
        generated: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Rank 0: write the last ~2048 committed ids to the exo log dir.

        Bounded and best-effort (``degeneration.write_degeneration_dump`` swallows
        its own errors), so this can sit on a cancel/guard path without a cost or a
        failure mode. The point is that the NEXT incident's loop text is on disk:
        the one that motivated this port streamed ~3m41s of unbroken repetition and
        its text was never captured anywhere.
        """
        if self.device_rank != 0:
            return
        path = write_degeneration_dump(
            token_ids=produced,
            reason=reason,
            task_id=str(task_id),
            generated=generated,
            extra=extra,
        )
        if path is not None:
            logger.warning(f"[DSV41] committed-token dump written: {path}")

    def _run_turn(
        self,
        session: Any,
        tokens: list[int],
        max_tokens: int,
        *,
        embeddings: mx.array | None,
        image_span_end: int = 0,
        token_types: list[int] | None = None,
    ) -> TurnOutcome:
        """Run one conversation turn: delta prefill, then decode rounds.

        ``image_span_end`` is the last row any image span covers; the delta
        prefill is forced into one piece that covers it, because the reference
        requires the whole span inside the forward whose cache offset is 0. The
        splice itself passes every other ``embed`` read through to the real table
        (the draft head reads it on every draft), so it is safe to leave
        installed for the whole turn.
        """
        turn, anchor = self._start_turn(
            session,
            tokens,
            embeddings=embeddings,
            image_span_end=image_span_end,
            token_types=token_types,
        )
        produced = [anchor]
        for batch, _lps in self._rounds(session, anchor, max_tokens - 1):
            produced.extend(batch)
        turn.tokens = produced
        session.sync_history(tokens, turn)
        session.finish(checkpoint=True)
        turn.committed = True
        return turn

    def _start_turn(
        self,
        session: Any,
        tokens: list[int],
        *,
        embeddings: mx.array | None,
        image_span_end: int = 0,
        token_types: list[int] | None = None,
    ) -> tuple[TurnOutcome, int]:
        """Delta prefill; returns the turn and its anchor (first generated token).

        ``token_types`` (image requests) marks the image-span rows so the model
        routes them with the VL bias and keeps them out of the engram n-grams,
        as the reference does; the mask is installed for this prefill only.
        """
        total = len(tokens)
        plan = (
            _first_chunk_covers(total, self._chunk, image_span_end)
            if embeddings is not None
            else None
        )
        # A stale fence timestamp from the PREVIOUS request's last fence must not
        # make this request's FIRST fence look like a >30 s gap (observed live:
        # a spurious "spacing 429.8s exceeds 30s" on the first fence after an
        # idle gap between requests). Reset it here, right before the prefill;
        # ``_session_fence_heartbeat`` treats 0.0 as "no previous call this
        # request" and skips the spacing warning entirely.
        self._last_fence_heartbeat_monotonic = 0.0
        if embeddings is None:
            turn = session.prefill(tokens, chunk_plan=plan)
        else:
            with splice_embeddings(
                self.loaded.model, embeddings, 0, image_span_end, token_types
            ):
                turn = session.prefill(tokens, chunk_plan=plan)
        if turn.reused_tokens:  # pyright: ignore[reportAny]
            logger.info(f"[DSV41] turn reuse: {turn}")
            if turn.prefill_tokens > 256:  # pyright: ignore[reportAny]
                # A reuse that still re-fed more than a small chunk means the
                # prefix collapsed to a much older checkpoint (the 1-2 row
                # undershoot below the newest boundary used to rewind all the
                # way to 0). Make it loud: the "delta" claims were once wrong
                # and went undetected.
                logger.warning(  # pyright: ignore[reportUnknownMemberType]
                    f"[DSV41] reuse undershoot: refed={turn.prefill_tokens} rows "  # pyright: ignore[reportAny]
                    f"(reused={turn.reused_tokens})"  # pyright: ignore[reportAny]
                )
        anchor = int(mx.argmax(turn.anchor_logits.reshape(-1), axis=-1).item())
        return turn, anchor

    def _rounds(
        self,
        session: Any,
        anchor: int,
        max_tokens: int,
        *,
        logprobs: int = 0,
        limiter: "ReasoningBudgetLimiter | None" = None,
    ) -> Iterator[tuple[list[int], list[Any]]]:
        """Decode rounds from the anchor; yields each round's committed tokens
        and (when ``logprobs`` > 0) their log-prob entries.

        Stops once ``max_tokens`` tokens after the anchor are committed or a
        round commits EOS. A round's whole batch is yielded: the cache already
        holds its rows (the emitter applies EOS / stop / the cap).

        ``limiter`` is the per-request reasoning-budget limiter (see
        ``dsv41/reasoning_budget.py``). ``_one_round`` admits each round's batch
        through it, which may CUT the batch short at the reasoning-budget
        threshold; the limiter then latches ``next_force``, so the FOLLOWING round
        steps plainly and emits the forced ``think_end`` (the cold start the cut
        requires). Both ranks compute the same latch from byte-identical committed
        ids, so the two stay in step with no collective.

        The no-repeat n-gram ban (``dsv41/no_repeat.py``) is threaded here too:
        resolved once per turn from the env knobs (DEFAULT OFF) and passed to every
        ``_one_round`` together with the produced-token ``history`` this loop keeps
        in lockstep with the cache. It is a pure function of that history, so both
        ranks build the identical ban and stay in step with no collective, exactly
        like the limiter above. When it is ON, the resolver also installs the
        CONTROL-TOKEN exemption (EOS + think start/end + ``_EXEMPT_IDS``), and this
        loop emits ONE INFO telemetry line at turn end (``format_ban_line``).
        """
        head = self.loaded.head if self.speculative else None
        policy = (
            _spec_policy(self.gamma)
            if (head is not None and self.adaptive_gamma)
            else None
        )
        n = 0
        token = anchor
        # The no-repeat n-gram ban (dsv41/no_repeat.py). DISABLED unless
        # EXO_DSV41_NO_REPEAT_NGRAM >= 3, so the default path is untouched. The
        # produced-token history it reads is built here, one round at a time, in
        # lockstep with the cache: it ENDS with ``token`` (the row the next round
        # feeds), which is exactly the sequence position the round predicts from.
        # The tokenizer is passed only so the control-token exemption (EOS + think
        # start/end) resolves when -- and only when -- the knob is enabled.
        ban = resolve_no_repeat_ngram(getattr(self.loaded, "tokenizer", None))
        # Turn-cumulative ban telemetry (None == knob off == never allocated, so the
        # disabled path does no extra work and logs nothing).
        ban_stats: BanTurnStats | None = BanTurnStats() if ban.enabled else None
        history: list[int] = [int(anchor)]
        # The anchor is fed to the limiter by the caller (_generate) before the
        # first round, so it can ALREADY have latched a force (a budget small enough
        # to be spent by the anchor itself): honour it on the very first round.
        force: int | None = limiter.next_force if limiter is not None else None
        try:
            while n < max_tokens:
                active = self._active
                self._check_cancel(active.task.task_id if active is not None else None)
                lps: list[Any] = []
                committed, _round_ms, _accepted, _gamma = _one_round(
                    self,
                    model=self.loaded.model,
                    cache=session.cache.cache,
                    token=token,
                    head=head,
                    policy=policy,
                    draft_state=session.draft_state,
                    logprobs=logprobs,
                    lp_out=lps if logprobs else None,
                    limiter=limiter,
                    force_think_end=force,
                    ban=ban,
                    history=history,
                    ban_stats=ban_stats,
                )
                # Cumulative spec counters (deltas across requests = live
                # acceptance rate; _accepted == _gamma means a fully-accepted round).
                if head is not None:
                    self._spec_rounds += 1
                    self._spec_accepted += int(_accepted)
                    self._spec_drafted += int(_gamma)
                batch = [int(t) for t in committed]
                if not batch:
                    # A round ALWAYS commits >= 1 token: the limiter admits the
                    # threshold token itself and its forced round emits one. An empty
                    # round would spin this loop, so fail loud rather than hang.
                    raise RuntimeError(
                        "DSV4.1: a decode round committed no tokens "
                        "(reasoning-budget limiter invariant violated)"
                    )
                n += len(batch)
                token = batch[-1]
                # Keep the ban's history in step with the cache: this round committed
                # every id in ``batch`` (its rows are in the cache), so the NEXT round
                # predicts from a sequence ending in ``batch[-1]``. Trim to the scan
                # window -- older ids can never be reached by the ban -- so the list
                # does not grow with the whole turn.
                history.extend(batch)
                if ban.enabled:
                    keep = int(ban.window) + int(ban.n)
                    if keep > 0 and len(history) > 2 * keep:
                        del history[:-keep]
                # ── DECODE-PHASE LIVENESS BEAT ──────────────────────────────────
                # Re-emit the runner status once per COMPLETED round, below the
                # empty-batch guard, so every beat is backed by a finished
                # collective AND a non-empty committed batch. Rationale: the engine
                # beats every <=15 s during LOAD and PREFILL (the loading progress
                # path and the fence hook), but the DECODE path emitted nothing the
                # supervisor counts. The runner-loop decode heartbeat
                # (runner.py handle_generation_tasks) cannot stand in for it: it
                # needs step() to RETURN, and the output parser can hold tokens back
                # for a whole reasoning phase, so step() stays inside one round for
                # minutes and its 15 s re-emit never runs. The supervisor's hang
                # watchdog then saw >45 s of silence from runners that were
                # decoding HEALTHY -- committed tokens advancing every ~10 s right
                # up to each kill (22 SIGKILLs observed live 2026-10-17). This is
                # the design doctrine of
                # docs/dsv41-prefill-liveness-and-hang-guard-design-2026-10-05.md:
                # signal real progress, never widen the timeout. A genuinely WEDGED
                # round (a collective that never returns) still goes SILENT -- it
                # never reaches this line, so no beat. Active on BOTH TP ranks (each
                # rank's own supervisor must see its own runner's events).
                # ``prefill_heartbeat`` is internally throttled to one call per
                # 15 s and is installed by the runner
                # (``self.generator.heartbeat = lambda: self.update_status(...)``).
                self.prefill_heartbeat()
                # The limiter latches the force for the round AFTER the one that
                # tripped the budget; pass it in on the next iteration.
                force = limiter.next_force if limiter is not None else None
                # Decode-side ladder: the draft window is in step with the cache at
                # a round boundary (``_one_round`` appends exactly the accepted
                # rows), so the same cadence guard applies. Without this, an
                # assistant-turn seam undershoot rewinds to the previous turn start
                # = unbounded refeed.
                session.maybe_checkpoint()  # pyright: ignore[reportAny]
                yield batch, lps
                if session.eos_id in batch:
                    return
        finally:
            # ── NO-REPEAT N-GRAM TELEMETRY ──────────────────────────────────
            # ONE INFO line per turn, ONLY when the knob is on (the disabled path
            # allocates no counters and logs nothing -- byte-identical behaviour).
            # Emitted from a ``finally`` so a turn that ends early (an EOS
            # ``return`` above, or a consumer that abandons this generator once it
            # has its tokens) still reports its counts. Fields and spelling are
            # pinned by no_repeat.format_ban_line.
            if ban.enabled and ban_stats is not None:
                logger.info(format_ban_line(ban, ban_stats))

    # ------------------------------------------------------------------ rounds

    def _check_cancel(self, task_id: TaskId | None) -> None:
        """Collect + agree on cancellations; raise PrefillCancelled when ours."""
        self._agreement.agree_on_cancellations_fast(self.cancel_receiver.collect())
        if task_id is not None and self._agreement.should_cancel(task_id):
            raise PrefillCancelled()

    def _command_id(self, task_id: TaskId) -> Any:
        """Command id for a task id (needed by ChunkGenerated)."""
        task = self._agreement.all_tasks.get(task_id)
        if task is None:
            raise RuntimeError(f"DSv4.1: no active task {task_id} for progress chunk")
        return task.command_id

    # ------------------------------------------------------------------ sampling

    def _resolve_sampler(self, params: TextGenerationTaskParams) -> Any | None:
        temperature = params.temperature
        if temperature is None:
            temperature = self.default_temperature
        if temperature is not None and temperature > 0:
            raise Dsv41UnsupportedFeature(
                "DSv4.1 serving is greedy-only in this build: speculative "
                "sampling is not implemented yet, so a request with "
                f"temperature={temperature} cannot be honoured honestly. Send "
                "temperature=0, or leave sampling unset with no card default."
            )
        return self.sampler

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
from exo.worker.engines.mlx.dsv41.errors import (
    Dsv41InvalidRequest,
    Dsv41UnsupportedFeature,
    reclassify_input_error,
)
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
from exo.worker.engines.mlx.dsv41.output import dsv41_output_parser
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
from exo.worker.engines.mlx.dsv41.session import Dsv41Sessions, TurnOutcome
from exo.worker.engines.mlx.dsv41.vision import (
    Dsv41Vision,
    build_embeddings,
    image_spans,
    prompt_tokens_for_request,
    render_prompt,
    splice_embeddings,
)
from exo.worker.engines.mlx.generator.generate import PrefillCancelled
from exo.worker.engines.mlx.utils_mlx import apply_chat_template, get_coord_group
from exo.worker.runner.bootstrap import logger

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

#: Prompt used for warmup; deliberately a plain chat turn.
_WARMUP_PROMPT = "Reply with the single word: ready"
#: Exhaustion marker for step()'s parser pull (the parser also yields None).
_END = object()

#: Canonical OpenAI error code for a request whose prompt exceeds the served
#: window. Emitted on the wire so OpenAI-compatible clients (and Hermes'
#: error classifier, which maps ``context_length_exceeded`` ->
#: _V_CONTEXT_OVERFLOW) route it to compaction instead of blind retries.
CONTEXT_LENGTH_EXCEEDED_CODE = "context_length_exceeded"


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

    # ---------------------------------------------------------------- lifecycle

    def __post_init__(self) -> None:
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
        )

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

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        if self._active is None:
            self._agreement.agree_on_tasks()
            self._activate_next(output)
            if self._active is None:
                output.extend(
                    (task_id, CancelledResponse())
                    for task_id in self._cancelled_tasks
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
        model = self.loaded.model
        tokenizer = self.loaded.tokenizer
        vision = self.vision_processor
        _refuse_unsupported(params, vision_available=vision is not None)

        max_tokens = params.max_output_tokens or MAX_TOKENS
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
        session = self._sessions.get(tokens_list, self._conversation_key(params))
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

        def committed_tokens() -> Iterator[tuple[int, Any]]:
            yield anchor, (anchor_lp[0] if anchor_lp else None)
            for batch, lps in self._rounds(session, anchor, max_tokens - 1, logprobs=lp_k):
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
        try:
            for tid, lp_entry in committed_tokens():
                last_tid = tid
                if tid in eos_ids:
                    final_reason = "stop"
                    break
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
                    )
                    return
                yield _mid_response(tid, text, task_id, as_logprob(lp_entry))
        except BaseException:
            session.cancel()
            if not keep:
                self._sessions.drop(session)
            raise
        self._end_turn(session, tokens_list, turn, produced, keep)
        # the stream ended on an EOS token, or on the conversation's budget
        yield _final_response(
            token=last_tid,
            text="",
            prefill_tps=prefill_tps,
            prompt_tokens=turn.prompt_tokens,
            generated=emitted,
            # EOS break sets final_reason="stop"; only budget exhaustion is "length".
            reason=final_reason or ("stop" if emitted == 0 else "length"),
            task_id=task_id,
            reused_tokens=turn.reused_tokens,
        )

    def _end_turn(
        self, session: Any, tokens: list[int], turn: TurnOutcome,
        produced: list[int], keep: bool,
    ) -> None:
        """Close a turn: history in step with the cache, checkpoint, maybe drop."""
        turn.tokens = list(produced)
        session.sync_history(tokens, turn)
        session.finish(checkpoint=True)
        turn.committed = True
        if not keep:
            self._sessions.drop(session)

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
        plan = _first_chunk_covers(total, self._chunk, image_span_end) if embeddings is not None else None
        if embeddings is None:
            turn = session.prefill(tokens, chunk_plan=plan)
        else:
            with splice_embeddings(
                self.loaded.model, embeddings, 0, image_span_end, token_types
            ):
                turn = session.prefill(tokens, chunk_plan=plan)
        if turn.reused_tokens:
            logger.info(f"[DSV41] turn reuse: {turn}")
        anchor = int(mx.argmax(turn.anchor_logits.reshape(-1), axis=-1).item())
        return turn, anchor

    def _rounds(
        self, session: Any, anchor: int, max_tokens: int, *, logprobs: int = 0
    ) -> Iterator[tuple[list[int], list[Any]]]:
        """Decode rounds from the anchor; yields each round's committed tokens
        and (when ``logprobs`` > 0) their log-prob entries.

        Stops once ``max_tokens`` tokens after the anchor are committed or a
        round commits EOS. A round's whole batch is yielded: the cache already
        holds its rows (the emitter applies EOS / stop / the cap).
        """
        head = self.loaded.head if self.speculative else None
        policy = (
            _spec_policy(self.gamma)
            if (head is not None and self.adaptive_gamma)
            else None
        )
        n = 0
        token = anchor
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
            )
            batch = [int(t) for t in committed]
            n += len(batch)
            token = batch[-1]
            yield batch, lps
            if session.eos_id in batch:
                return

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

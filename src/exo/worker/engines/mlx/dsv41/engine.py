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
    history). Session reuse is workstream E's deliverable; a request that asks
    for it is refused loudly instead of silently re-prefilling.
  * ``tensor_auto_parallel``-- TP is built into the loader.
  * sampling                -- v1 is GREEDY ONLY (the DSpark verify loop is
    greedy). A request that asks for a non-greedy temperature is REFUSED rather
    than silently served greedy; ``_resolve_sampler`` is workstream F's hook.
  * vision                  -- workstream H; images are dropped LOUDLY.

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
from exo.worker.engines.mlx.dsv41.errors import Dsv41UnsupportedFeature
from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded
from exo.worker.engines.mlx.dsv41.output import dsv41_output_parser
from exo.worker.engines.mlx.dsv41.rounds import (
    _cache_capacity,
    _final_response,
    _mid_response,
    _one_round,
    _queue_of,
    _refuse_unsupported,
    _spec_policy,
    _stop_index,
    _stop_sequences,
)
from exo.worker.engines.mlx.generator.generate import PrefillCancelled
from exo.worker.engines.mlx.utils_mlx import apply_chat_template, get_coord_group
from exo.worker.runner.bootstrap import logger

#: Chunk size for the chunked prefill loop. PREFILL IS THE KNOWN BLOCKER for
#: this model (74 tok/s at 8K, Metal GPU-timeout above 16K -- exo phase 19), and
#: workstream C owns the fix. Until then the engine keeps chunks small so a long
#: prompt cannot build one enormous lazy graph, and fences every chunk.
DEFAULT_PREFILL_CHUNK = 512

#: Warmup generation length. Short on purpose: the point is to compile the EXL3
#: kernels, not to measure anything (the first forward is the expensive one --
#: exo phase 19 measured 206 s for a 2K first chunk).
DEFAULT_WARMUP_TOKENS = 8

#: Prompt used for warmup; deliberately a plain chat turn.
_WARMUP_PROMPT = "Reply with the single word: ready"


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
            prefix_cache_hit="none",  # DSv4.1 has no prefix cache (stream E)
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
    #: Vision hook (workstream H). None => no vision tower for this checkpoint.
    vision_processor: Any | None = None
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
        if self._active is None:
            self._agreement.agree_on_tasks()
            if self._agreement.queue:
                self._start_next()
            else:
                return iter(
                    [
                        (task_id, CancelledResponse())
                        for task_id in self._cancelled_tasks
                    ]
                )

        active = self._active
        assert active is not None
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        try:
            next(active.generator)
            # Drain every chunk currently available: the parse pipeline buffers
            # (thinking split, DSML detection) and only releases on flush points.
            self._active = active  # keep alive across the drain
            while (parsed := next(active.output_generator, None)) is not None:
                output.append((active.task.task_id, parsed))
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
            if self._agreement.queue:
                self._start_next()
        except Exception as e:
            self._send_error(active.task, e)
            self._active = None
            raise
        return iter(output)

    def close(self) -> None:
        self._active = None
        self._agreement.reset()
        self._draft_windows.clear()
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
        return dropped

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise Dsv41UnsupportedFeature(
            "DSv4.1 does not implement disaggregated prefill: a remote prefill "
            "would have to ship this model's cache (window ring + compressed KV "
            "+ compressor carry + index keys + engram id history) over the wire, "
            "which is not built. Disable remote prefill for this model."
        )

    # ------------------------------------------------------------------ requests

    def _start_next(self) -> None:
        task = self._agreement.queue.popleft()
        try:
            generator = self._build_generator(task)
            output_generator = dsv41_output_parser(
                _queue_of(generator),
                self.loaded.tokenizer,
                apply_chat_template(self.loaded.tokenizer, task.task_params),
                self.model_id,
            )
        except Exception as e:
            self._send_error(task, e)
            raise
        self._active = _Active(task, generator, output_generator)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _build_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        params = task.task_params
        prompt = apply_chat_template(self.loaded.tokenizer, params)
        return self._generate(params, prompt, task_id=task.task_id)

    # ------------------------------------------------------------------ generate

    def _generate(
        self,
        params: TextGenerationTaskParams,
        prompt: str,
        *,
        task_id: TaskId | None,
    ) -> Generator[GenerationResponse]:
        model = self.loaded.model
        tokenizer = self.loaded.tokenizer
        _refuse_unsupported(params, vision_available=self.vision_processor is not None)

        max_tokens = params.max_output_tokens or MAX_TOKENS
        eos_ids = set(tokenizer.eos_token_ids)
        stop_sequences = _stop_sequences(params)
        capacity = _cache_capacity(self)

        prompt_tokens = encode_prompt(tokenizer, prompt)
        prompt_len = int(prompt_tokens.shape[0])
        if prompt_len == 0:
            raise ValueError("DSv4.1: empty prompt after chat templating")
        if prompt_len + max_tokens + 8 > capacity:
            raise Dsv41UnsupportedFeature(
                f"DSv4.1: prompt {prompt_len} + max_output_tokens {max_tokens} "
                f"needs more than the {capacity}-token cache this instance was "
                "configured for (max_kv_tokens / card context_length)."
            )

        cache = model.make_cache(1, max_seq_len=capacity)

        # --- chunked, fenced prefill. Each chunk is evaluated (and the cache
        # advanced) before the next is built, so the lazy graph stays bounded --
        # the whole reason the Metal GPU timeout is survivable at all here.
        started = time.perf_counter()
        last_logits: mx.array | None = None
        chunk = self._chunk
        for start in range(0, prompt_len, chunk):
            self._check_cancel(task_id)
            if task_id is not None and self.device_rank == 0 and start > 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=self._command_id(task_id),
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=min(start + chunk, prompt_len),
                            total_tokens=prompt_len,
                        ),
                    )
                )
            else:
                self.prefill_heartbeat()
            last_logits = model(
                prompt_tokens[start : start + chunk][None], cache, last_logit_only=True
            )
            mx.eval(last_logits)
        prefill_seconds = time.perf_counter() - started
        prefill_tps = prompt_len / prefill_seconds if prefill_seconds > 0 else 0.0
        if last_logits is None:
            raise RuntimeError("DSv4.1: prefill produced no logits")
        first_token = int(mx.argmax(last_logits.reshape(-1), axis=-1).item())

        detokenizer = tokenizer.detokenizer
        emitted = 0
        if first_token in eos_ids:
            yield _final_response(
                token=first_token,
                text="",
                prefill_tps=prefill_tps,
                prompt_tokens=prompt_len,
                generated=0,
                reason="stop",
                task_id=task_id,
            )
            return
        detokenizer.add_token(first_token)
        emitted = 1
        yield _mid_response(first_token, detokenizer.last_segment, task_id)

        head = self.loaded.head if self.speculative else None
        policy = (
            _spec_policy(self.gamma)
            if (head is not None and self.adaptive_gamma)
            else None
        )
        round_stats = _RoundStats()
        token = first_token
        accumulated = ""
        final_reason: FinishReason | None = None

        while emitted < max_tokens:
            self._check_cancel(task_id)
            drafted, round_ms, accepted, gamma = _one_round(
                self,
                model=model,
                cache=cache,
                token=token,
                head=head,
                policy=policy,
            )
            round_stats.add(round_ms, accepted, gamma)
            stop_at = None
            for index, tid in enumerate(drafted):
                if tid in eos_ids:
                    # End the turn HERE. The draft batch can contain tokens that
                    # follow the EOS (the verify window does not know it ended),
                    # and the earlier version of this loop only set the reason
                    # and broke out of the inner loop -- so the outer loop
                    # re-drafted forever and the request never finished.
                    final_reason = "stop"
                    stop_at = index
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
                    yield _final_response(
                        token=tid,
                        text=text,
                        prefill_tps=prefill_tps,
                        prompt_tokens=prompt_len,
                        generated=emitted,
                        reason=final_reason,
                        task_id=task_id,
                        round_stats=round_stats,
                    )
                    return
                yield _mid_response(tid, text, task_id)
                token = tid
            if final_reason is not None:
                yield _final_response(
                    token=drafted[stop_at] if stop_at is not None else token,
                    text="",
                    prefill_tps=prefill_tps,
                    prompt_tokens=prompt_len,
                    generated=emitted,
                    reason=final_reason,
                    task_id=task_id,
                    round_stats=round_stats,
                )
                return
        if final_reason is None:
            # Only reachable when max_tokens was 0-ish; stay honest about it.
            yield _final_response(
                token=token,
                text="",
                prefill_tps=prefill_tps,
                prompt_tokens=prompt_len,
                generated=emitted,
                reason="length",
                task_id=task_id,
                round_stats=round_stats,
            )

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

import itertools
import os
import time
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_DSV4_BATCHED_PREFILL, EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
from exo.worker.engines.mlx.disaggregated.serve import run_prefill_for_request
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    mlx_generate,
    warmup_inference,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    get_coord_group,
    mx_all_gather_tasks,
    mx_any,
    mx_min_int,
    prewarm_coord_group,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger

from .model_output_parsers import apply_all_parsers, map_responses_to_chunks
from .tool_parsers import ToolParser


class GeneratorQueue[T]:
    def __init__(self):
        self._q = deque[T]()

    def push(self, t: T):
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input."""
    from exo.worker.engines.mlx.utils_mlx import mlx_force_oom

    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class SequentialGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    check_for_cancel_every: int = 50
    max_kv_tokens: int | None = None
    prefill_step_size: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_presence_penalty: float | None = None
    default_repetition_penalty: float | None = None
    default_frequency_penalty: float | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: (
        tuple[
            TextGeneration,
            # mlx generator that does work
            Generator[GenerationResponse],
            # queue that the 1st generator should push to and 3rd generator should pull from
            GeneratorQueue[GenerationResponse],
            # generator to get parsed outputs
            Iterator[GenerationChunk | None],
        ]
        | None
    ) = field(default=None, init=False)

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )
        # Pre-warm the distributed collective path so the first real
        # agree_on_tasks doesn't pay a ~1s cold-start penalty.
        self.agree_on_tasks()
        self.agree_on_cancellations()
        # Eager coord-subgroup split + verification probe so the
        # split() runs in lockstep at a known sync point. See
        # `prewarm_coord_group` in utils_mlx.
        prewarm_coord_group(self.group)

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all).

        Fast path: a single mx_any of `bool(self._maybe_queue)` decides if ANY
        rank has new submissions. In the common decode-loop case (no rank has
        anything queued), we skip the 2× all_gather entirely. This pattern
        mirrors `agree_on_cancellations_fast` and is necessary because the
        unconditional call site fires at decode rate (~30 Hz); driving JACCL
        all_gather at that rate has been observed to corrupt return buffers
        (max_tasks bit-flipped to ~1B → 152 GB metal::malloc OOM).

        Coord subgroup: runs on the sibling subgroup so this small
        mx_any / all_gather doesn't share the model TP group's
        `next_call_id_` counter. Cross-stream call_id race fixed
        2026-05-07. See `get_coord_group` in utils_mlx.
        """
        coord = get_coord_group(self.group)
        if not mx_any(len(self._maybe_queue) > 0, coord):
            return
        agreed, different = mx_all_gather_tasks(self._maybe_queue, coord)
        # Extend from `agreed` (sorted by task_id on all ranks) to guarantee
        # every rank enqueues tasks in the same order, preventing TP collective
        # deadlocks (upstream PR #2048). DO NOT filter through self._maybe_queue
        # — that preserves OUR submission order, which differs across ranks
        # and produces the c=2 duplicated-token output mode.
        self._queue.extend(agreed)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel.

        Uses the coord subgroup, see ``agree_on_tasks`` rationale.
        """
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        coord = get_coord_group(self.group)
        if mx_any(has_cancel_all, coord):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, coord)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def agree_on_cancellations_fast(self) -> None:
        """Lightweight cancellation check for use during prefill.

        Uses a single mx_any to check if ANY rank has cancellations. Only runs
        the expensive all_gather if someone actually has something to cancel.
        Saves ~4 distributed ops per call in the common (no-cancel) case.

        Coord subgroup, see ``agree_on_tasks`` rationale.
        """
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        coord = get_coord_group(self.group)
        has_anything = has_cancel_all or len(self._maybe_cancel) > 0
        if not mx_any(has_anything, coord):
            return  # Fast path: no rank has cancellations — 1 collective op total

        # Slow path: at least one rank has cancels — full protocol
        if mx_any(has_cancel_all, coord):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, coord)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        if self._active is None:
            self.agree_on_tasks()

            if self._queue:
                self._start_next()
            else:
                return map(
                    lambda task: (task, CancelledResponse()), self._cancelled_tasks
                )

        assert self._active is not None

        task, gen, queue, output_generator = self._active
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        try:
            response = next(gen)
            queue.push(response)
            # drain potentially many responses every time
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

        except (StopIteration, PrefillCancelled):
            output.append((task.task_id, FinishedResponse()))
            self._active = None
            if self._queue:
                self._start_next()

        except Exception as e:
            self._send_error(task, e)
            self._active = None
            raise

        return itertools.chain(
            output,
            map(lambda task: (task, CancelledResponse()), self._cancelled_tasks),
        )

    def _start_next(self) -> None:
        task = self._queue.popleft()
        try:
            gen = self._build_generator(task)
        except Exception as e:
            self._send_error(task, e)
            raise
        queue = GeneratorQueue[GenerationResponse]()

        if task.task_params.bench:
            output_generator: Iterator[GenerationChunk | None] = map(
                lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
            )
        else:
            output_generator = apply_all_parsers(
                queue.gen(),
                apply_chat_template(self.tokenizer, task.task_params),
                self.tool_parser,
                self.tokenizer,
                type(self.model),
                self.model_id,
                task.task_params.tools,
            )
        self._active = (task, gen, queue, output_generator)

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
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )
            else:
                self.prefill_heartbeat()

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations_fast()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    raise PrefillCancelled()

                self.agree_on_tasks()

        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task.task_params,
            prompt=prompt,
            kv_prefix_cache=self.kv_prefix_cache,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            group=self.group,
            vision_processor=self.vision_processor,
            max_kv_tokens=self.max_kv_tokens,
            prefill_step_size=self.prefill_step_size,
            instance_temperature=self.default_temperature,
            instance_top_p=self.default_top_p,
            instance_top_k=self.default_top_k,
            instance_min_p=self.default_min_p,
            instance_presence_penalty=self.default_presence_penalty,
            instance_repetition_penalty=self.default_repetition_penalty,
            instance_frequency_penalty=self.default_frequency_penalty,
        )

    def close(self) -> None:
        del self.model, self.tokenizer, self.group

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )


@dataclass(eq=False)
class BatchGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int = 50
    vision_processor: VisionProcessor | None = None
    max_kv_tokens: int | None = None
    prefill_step_size: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_presence_penalty: float | None = None
    default_repetition_penalty: float | None = None
    default_frequency_penalty: float | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _gen: ExoBatchGenerator = field(init=False)
    _active_tasks: dict[
        int,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Iterator[GenerationChunk | None],
        ],
    ] = field(default_factory=dict, init=False)
    _jaccl_step_count: int = field(default=0, init=False)
    # Section 43 continued (2026-08-10): uids whose cancel() was flagged
    # but whose underlying PP-spec generator (ExoBatchGenerator's
    # _pp_spec_gen_by_uid) hasn't actually finished yet. See
    # _apply_cancellations()'s docstring for the full race this closes --
    # a naive "flag cancel, immediately report CancelledResponse" let the
    # RUNNER (runner.py's `while self.active_tasks:`) go idle before the
    # generator was ever driven to its own next agreed-cancel checkpoint,
    # leaving _pp_spec_gen_by_uid[uid] permanently populated and every
    # subsequent request for that runner rejected with
    # PPSpecAlreadyActiveError. Maps uid -> the TaskId to report once the
    # generator actually closes.
    _pending_pp_spec_cancel: dict[int, TaskId] = field(
        default_factory=dict, init=False
    )
    # Section 46 (2026-08-15): the batched-decode FOLLOWER analogue of
    # _pending_pp_spec_cancel. A cancelled uid whose per-request state can
    # only be released by the DRIVER's incoming EvictMessage is parked here
    # (kept in _active_tasks, so runner.py's loop stays alive and this rank
    # keeps pumping tick() to service that message) instead of being
    # finalized immediately. Maps uid -> the TaskId to report once the evict
    # has genuinely been handled.
    _pending_batched_decode_evict: dict[int, TaskId] = field(
        default_factory=dict, init=False
    )
    _jaccl_step_handle: BinaryIO | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._gen = ExoBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            vision_processor=self.vision_processor,
            model_id=self.model_id,
            max_kv_tokens=self.max_kv_tokens,
            prefill_step_size=self.prefill_step_size,
            default_temperature=self.default_temperature,
            default_top_p=self.default_top_p,
            default_top_k=self.default_top_k,
            default_min_p=self.default_min_p,
            default_presence_penalty=self.default_presence_penalty,
            default_repetition_penalty=self.default_repetition_penalty,
            default_frequency_penalty=self.default_frequency_penalty,
        )

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )
        # Pre-warm the distributed collective path so the first real
        # agree_on_tasks doesn't pay a ~1s cold-start penalty.
        self.agree_on_tasks()
        self.agree_on_cancellations()
        # Eager coord-subgroup split + verification probe so the
        # split() runs in lockstep at a known sync point. See
        # `prewarm_coord_group` in utils_mlx.
        prewarm_coord_group(self.group)

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all).

        Fast path: a single mx_any of `bool(self._maybe_queue)` decides if ANY
        rank has new submissions. In the common decode-loop case (no rank has
        anything queued), we skip the 2× all_gather entirely. This pattern
        mirrors `agree_on_cancellations_fast` and is necessary because the
        unconditional call site fires at decode rate (~30 Hz); driving JACCL
        all_gather at that rate has been observed to corrupt return buffers
        (max_tasks bit-flipped to ~1B → 152 GB metal::malloc OOM at c=2).

        Coord group: use the sibling subgroup so this small mx_any /
        all_gather doesn't share the model TP group's `next_call_id_`
        counter. Cross-stream call_id race fixed 2026-05-07. See
        `get_coord_group` in utils_mlx.
        """
        coord = get_coord_group(self.group)
        if not mx_any(len(self._maybe_queue) > 0, coord):
            return
        _t0 = time.perf_counter()
        agreed, different = mx_all_gather_tasks(self._maybe_queue, coord)
        # Extend from `agreed` (sorted by task_id on all ranks) — upstream
        # PR #2048 fix. See the other agree_on_tasks above for rationale.
        self._queue.extend(agreed)
        self._maybe_queue = list(different)
        _dt = time.perf_counter() - _t0
        if _dt > 0.005 and os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1"):
            logger.info(f"[PROF] agree_on_tasks={_dt*1000:.1f}ms")

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel.

        Uses the coord subgroup, see ``agree_on_tasks`` rationale.
        """
        _t0 = time.perf_counter()
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        coord = get_coord_group(self.group)
        if mx_any(has_cancel_all, coord):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, coord)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)
        _dt = time.perf_counter() - _t0
        if _dt > 0.005 and os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1"):
            logger.info(f"[PROF] agree_on_cancellations={_dt*1000:.1f}ms (mx_any + 2x all_gather)")

    def agree_on_cancellations_fast(self) -> None:
        """Lightweight cancellation check for use during prefill.

        Uses a single mx_any to check if ANY rank has cancellations. Only runs
        the expensive all_gather if someone actually has something to cancel.
        Saves ~4 distributed ops per call in the common (no-cancel) case.

        Coord group, see ``agree_on_tasks`` rationale.
        """
        has_cancel_all = False
        _pc: list[str] = []
        _pd: list[str] = []
        for task_id in self.cancel_receiver.collect():
            _pc.append(str(task_id))
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])
            else:
                _pd.append(str(task_id))
        if _pc:
            logger.info(
                f"CANCELPROBE[bg.fast] collected={_pc} dropped={_pd} "
                f"maybe={len(self._maybe_cancel)} all_tasks={[str(k) for k in self._all_tasks]}"
            )

        coord = get_coord_group(self.group)
        has_anything = has_cancel_all or len(self._maybe_cancel) > 0
        if not mx_any(has_anything, coord):
            if _pc:
                logger.info("CANCELPROBE[bg.fast] EXIT mx_any fastpath")
            return  # Fast path: no rank has cancellations — 1 collective op total

        # Slow path: at least one rank has cancels — full protocol
        if mx_any(has_cancel_all, coord):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, coord)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)
        logger.info(
            f"CANCELPROBE[bg.fast] AFTER agreed={[str(t.task_id) for t in agreed]} "
            f"cancelled={[str(t) for t in self._cancelled_tasks]}"
        )

    def _jaccl_dump_step(self, label: str) -> None:
        """Snapshot per-rank Python state once per step() call when
        JACCL_TRACE_STEP=1. Writes one JSON line per call to
        /tmp/jaccl_step_rank_${rank}_pid${pid}.log so cross-rank diff
        can identify the first asymmetric state on the prefix-cache
        share path. No collectives — pure local reads. Off by default;
        zero overhead when env unset.
        Memory: next_session_plan_jaccl_c2_prefix_cache.md.
        """
        if os.environ.get("JACCL_TRACE_STEP") != "1":
            return

        import json
        from collections import deque as _deque
        from typing import Sized, cast

        if self._jaccl_step_handle is None:
            rank = self.group.rank() if self.group is not None else 0
            path = f"/tmp/jaccl_step_rank_{rank}_pid{os.getpid()}.log"
            self._jaccl_step_handle = open(path, "ab", buffering=0)  # noqa: SIM115

        rec: dict[str, object] = {
            "step": self._jaccl_step_count,
            "label": label,
            "active_uids": sorted(self._active_tasks.keys()),
            "queue_len": len(self._queue),
            "maybe_queue_len": len(self._maybe_queue),
        }

        mlx_gen: object = getattr(self._gen, "_mlx_gen", None)
        if mlx_gen is not None:
            gen_batch: object = getattr(mlx_gen, "_generation_batch", None)
            if gen_batch is not None:
                uids = cast(list[int], getattr(gen_batch, "uids", []))
                num_tokens = cast(list[int], getattr(gen_batch, "_num_tokens", []))
                max_tokens = cast(list[int], getattr(gen_batch, "max_tokens", []))
                rec["gen_batch_uids"] = list(uids)
                rec["gen_batch_num_tokens"] = list(num_tokens)
                rec["gen_batch_max_tokens"] = list(max_tokens)

            unprocessed = cast(
                "_deque[object] | None",
                getattr(mlx_gen, "_unprocessed_sequences", None),
            )
            rec["unprocessed_count"] = (
                len(unprocessed) if unprocessed is not None else 0
            )
            prompt_batch = cast(
                "Sized | None", getattr(mlx_gen, "_prompt_batch", None)
            )
            rec["prompt_batch_len"] = (
                len(prompt_batch) if prompt_batch is not None else 0
            )

            tb = cast(
                "dict[int, _deque[object]] | None",
                getattr(mlx_gen, "_token_buffer", None),
            )
            if tb is not None:
                rec["token_buffer"] = {
                    str(uid): len(buf) for uid, buf in sorted(tb.items())
                }
            mtp_prefilled = cast(
                "set[int] | None", getattr(mlx_gen, "_mtp_prefilled", None)
            )
            if mtp_prefilled is not None:
                rec["mtp_prefilled"] = sorted(mtp_prefilled)

        line = (json.dumps(rec, default=str) + "\n").encode("utf-8")
        self._jaccl_step_handle.write(line)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ]:
        from exo.worker.engines.mlx.trace import T

        self._jaccl_dump_step("step_entry")
        self._jaccl_step_count += 1

        # agree_on_tasks() is a collective (mx.distributed.all_gather). Both
        # ranks must call it together — gating on per-rank `self._queue` lets
        # one rank skip it while the other waits forever inside, deadlocking
        # the cluster on the next iteration's all_reduce. The collective itself
        # short-circuits cheaply (utils_mlx.py:1102) when no rank has new
        # tasks, so calling unconditionally is safe.
        with T("batch_gen.agree_on_tasks"):
            self.agree_on_tasks()

        # 2026-08-09 (design doc Section 29, real-hardware finding): also
        # run the FAST cancellation check unconditionally, right here,
        # every single step() call — mirroring agree_on_tasks()'s own
        # unconditional-with-internal-mx_any-gate pattern immediately
        # above (proven safe/cheap at decode rate on this exact call
        # site). This is the ONLY place in this class provably symmetric
        # across both ranks every step (agree_on_tasks already lives
        # here), unlike the per-decode-token callback closures inside
        # ExoBatchGenerator's response loop, which are NOT reliably
        # symmetric for the batched-decode path (rank 1's mirror-only
        # tick() never builds real per-request response/callback state
        # the way rank 0's does) — a per-rank-asymmetric collective call
        # site is a deadlock risk, strictly worse than the bug being
        # fixed here (a `consult` review flagged this distinction before
        # landing; the token-callback's own counter-gated
        # agree_on_cancellations() calls are left in place as a harmless,
        # redundant backstop, not removed).
        #
        # Root cause this closes: `check_for_cancel_every` (up to 100) is
        # calibrated ONCE at warmup time against a fast, near-empty-
        # context decode. Real decode throughput under a large resident
        # KV cache (30K+ tokens, PP + batched-decode forward pass) can be
        # 30-100x slower per token than that warmup measurement — so the
        # OLD counter-gated check (full agree_on_cancellations(), fired
        # only every `check_for_cancel_every` decode tokens from inside
        # the per-token callback) could leave a real client cancellation
        # sitting unobserved for 80+ decode tokens, measured directly at
        # 133.7s/136.4s (rank0/rank1) on real hardware before this fix —
        # the runner kept burning real GPU compute serving a request the
        # client had already disconnected from, matching the shape of
        # the now-fixed 717523cb6/94fc04a4d bugs closely enough to be
        # mistaken for a regression of either. This fix bounds
        # cancellation-observation latency to ~1 decode step (the
        # request's uid is flagged in `self._cancelled_tasks` here, then
        # actually evicted via `_apply_cancellations()` at the end of
        # THIS SAME step() call after `self._gen.step()` returns — so at
        # most one extra forward pass runs post-cancel, not up to
        # `check_for_cancel_every`).
        with T("batch_gen.agree_on_cancellations_fast"):
            self.agree_on_cancellations_fast()

        # Submit any queued tasks to the engine. When EXO_DSV4_BATCHED_PREFILL
        # is on AND the queue has 2+ tasks, run a single batched-prefill pass
        # over them so they share the prefill phase instead of serializing.
        # The legacy submit() path runs prefill SYNCHRONOUSLY per task; at c=2
        # long context that meant stream 0 monopolized the runner main loop
        # for ~6 min while stream 1 sat idle, then stream 1 ran another ~6 min
        # sequential prefill, collapsing per-stream throughput to ~7.7 tok/s.
        # See ``ExoBatchGenerator.submit_batched`` for the heterogeneity
        # gating (bench-only, no vision, no remote prefill, no MTP/PP-spec).
        if EXO_DSV4_BATCHED_PREFILL:
            # COLLECTIVE GATE. The batched-prefill branch issues a single TP
            # forward across the popped tasks (``_batched_start_task`` ->
            # ``submit_batched`` -> model all_reduce). Both ranks MUST enter
            # the branch together AND pop the SAME number of tasks, or the TP
            # collective sees mismatched batch shapes and JACCL barrier-hangs
            # the cluster forever (the 2026-06-11 concurrent-request wedge).
            #
            # ``available_slots`` (from ``len(self._active_tasks)``) and
            # ``len(self._queue)`` are per-rank: after ``agree_on_tasks`` the
            # queue *contents* match, but cancellations / finishes can leave
            # ``_active_tasks`` transiently asymmetric, so the raw local
            # decision is not safe to branch on. Reduce both to their MIN
            # across ranks first so every rank computes an identical
            # ``batch_count`` and takes the identical path. Mirrors the
            # collective gates at step()'s ``has_work`` check and
            # batch_generate.py:1900. Coord subgroup so this control-plane
            # reduce never shares the model TP ``next_call_id_`` counter.
            coord = get_coord_group(self.group)
            local_slots = EXO_MAX_CONCURRENT_REQUESTS - len(self._active_tasks)
            agreed_slots = mx_min_int(local_slots, coord)
            agreed_queue_len = mx_min_int(len(self._queue), coord)
            if agreed_slots > 1 and agreed_queue_len >= 2:
                batch_count = min(agreed_slots, agreed_queue_len)
                tasks_batch: list[TextGeneration] = []
                for _ in range(batch_count):
                    tasks_batch.append(self._queue.popleft())

                try:
                    with T("batch_gen.batched_start_task"):
                        uids = self._batched_start_task(tasks_batch)
                except PrefillCancelled:
                    # Treat batched prefill cancellation as "all tasks
                    # cancelled" — they all go back through the cancellation
                    # path. Phase 5 stress-tests this.
                    # Same greppable marker as the step() handler below.
                    logger.info(
                        "PREFILL_CANCELLED_PATH: prefill cancelled inside "
                        "_batched_start_task (batched submit path)"
                    )
                    uids = []
                except Exception as e:
                    # Surface the error against the first task in the batch;
                    # raise so the runner sees the failure and the cluster
                    # cleanly tears down rather than stalling.
                    if tasks_batch:
                        self._send_error(tasks_batch[0], e)
                    raise

                for task, uid in zip(tasks_batch, uids, strict=True):
                    queue = GeneratorQueue[GenerationResponse]()
                    if task.task_params.bench:
                        output_generator: Iterator[GenerationChunk | None] = map(
                            lambda r: map_responses_to_chunks(r, self.model_id),
                            queue.gen(),
                        )
                    else:
                        output_generator = apply_all_parsers(
                            queue.gen(),
                            apply_chat_template(self.tokenizer, task.task_params),
                            self.tool_parser,
                            self.tokenizer,
                            type(self.model),
                            self.model_id,
                            task.task_params.tools,
                        )
                    self._active_tasks[uid] = (task, queue, output_generator)

        while self._queue and len(self._active_tasks) < EXO_MAX_CONCURRENT_REQUESTS:
            task = self._queue.popleft()
            try:
                with T("batch_gen.start_task"):
                    uid = self._start_task(task)
            except PrefillCancelled:
                # Same greppable marker as the step() handler below.
                logger.info(
                    "PREFILL_CANCELLED_PATH: prefill cancelled inside "
                    "_start_task (single-task submit path)"
                )
                continue
            except Exception as e:
                self._send_error(task, e)
                raise

            queue = GeneratorQueue[GenerationResponse]()
            if task.task_params.bench:
                output_generator: Iterator[GenerationChunk | None] = map(
                    lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
                )
            else:
                output_generator = apply_all_parsers(
                    queue.gen(),
                    apply_chat_template(self.tokenizer, task.task_params),
                    self.tool_parser,
                    self.tokenizer,
                    type(self.model),
                    self.model_id,
                    task.task_params.tools,
                )
            self._active_tasks[uid] = (task, queue, output_generator)

        # `self._gen.has_work` is per-rank state (active_tasks +
        # gen_batch + prompt_batch). Without a collective gate, ranks
        # can take divergent paths here: the rank with work calls
        # self._gen.step() (issuing TP all_reduces from the model
        # forward) while the rank without work returns early. The
        # active rank then busy-polls JACCL forever for a peer send
        # that never comes and the cluster wedges.
        # Memory: jaccl_phase_a_finding_2026_05_05.md, hermes_wedge_root_cause_2026_05_04.md.
        # Coord group: gate fires per-step at decode rate, must not
        # share call_id space with the model TP forward. Race fix
        # 2026-05-07 — see get_coord_group in utils_mlx.
        if not mx_any(self._gen.has_work, get_coord_group(self.group)):
            return self._apply_cancellations()

        self._jaccl_dump_step("pre_gen_step")

        try:
            results = self._gen.step()
        except PrefillCancelled:
            # Section 58/61: under batched decode, prefill is DEFERRED --
            # it no longer runs inside submit(), it runs inside
            # tick()-granted work reached from step()
            # (_run_deferred_prefill_for_grant). PrefillCancelled is
            # deliberately a BaseException (see the contrasting comment
            # on PPSpecAlreadyActiveError in batch_generate.py) so the
            # runner's generic `except Exception` handling does NOT
            # catch it -- which was correct while the only way to raise
            # it was from a submit path that already handled it
            # explicitly. Both of those handlers still exist
            # (_batched_start_task and _start_task above), but nothing
            # followed the exception when prefill's execution moved
            # into step(), so a client cancelling during an in-flight
            # deferred prefill killed the whole runner process with an
            # uncaught exception. Real traceback captured on hardware,
            # design doc Section 58.
            #
            # The correct outcome is the one both submit-path handlers
            # already chose: the cancelled request goes away, the runner
            # stays alive. _apply_cancellations() is the existing
            # mechanism for exactly that -- it is what step()'s own
            # no-work path returns a few lines above, and it already
            # knows how to defer finalization for requests whose
            # generator/glue still holds state (PP-spec parking, the
            # batched-decode follower evict handshake) rather than
            # reporting them cancelled too early.
            #
            # Cross-rank safety: this is NOT a unilateral local
            # decision. The raise site
            # (distributed_prompt_progress_callback) calls
            # agree_on_cancellations_fast() -- a collective -- BEFORE
            # checking should_cancel(), so both ranks reach the same
            # verdict on the same request and take this path in
            # lockstep. Swallowing it on one rank only would desync;
            # swallowing it after an agreed collective does not.
            # The "PREFILL_CANCELLED_PATH" token is a stable, greppable
            # marker: it is the ONLY proof available to an off-box harness
            # that a cancel landed inside the mid-prefill window rather than
            # after prefill completed. Asserted on (in ~/exo.log) by
            # bench/section85_prefill_cancel_hardware_test.py -- do not
            # reword it without updating that harness.
            logger.info(
                "PREFILL_CANCELLED_PATH: prefill cancelled during a "
                "deferred (batched-decode) prefill inside step(); "
                "reporting the cancellation and keeping the runner alive"
            )
            return self._apply_cancellations()

        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        for uid, response in results:
            if uid not in self._active_tasks:
                # should we error here?
                logger.warning(f"{uid=} not found in active tasks")
                continue

            task, queue, output_generator = self._active_tasks[uid]
            queue.push(response)
            # If a generator fails to parse for some reason and returns early, we should not crash
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

            # check if original response was terminal and append a Finished()
            #
            # Section 43 continued (2026-08-10): if this uid is parked in
            # self._pending_pp_spec_cancel (cancel() was flagged, waiting
            # for the generator to actually agree-and-close), do NOT emit
            # FinishedResponse or delete _active_tasks here even though
            # finish_reason is now set -- that would race with
            # _drain_pending_pp_spec_cancellations() below (called via
            # step()'s return chain) emitting a CancelledResponse for the
            # SAME task_id once it observes the generator gone from
            # _pp_spec_gen_by_uid, producing two contradictory terminal
            # responses for one request. The pending-cancel path is the
            # sole finalizer for these uids; this branch just lets the
            # response continue flowing through queue/output_generator
            # above (so any partial streamed output up to this point is
            # still captured) without also terminating the task here.
            if (
                response.finish_reason is not None
                and uid not in self._pending_pp_spec_cancel
                and uid not in getattr(self, "_pending_batched_decode_evict", {})
            ):
                output.append((task.task_id, FinishedResponse()))
                del self._active_tasks[uid]

        return itertools.chain(
            output,
            self._drain_pending_pp_spec_cancellations(),
            self._drain_pending_batched_decode_evictions(),
            self._apply_cancellations(),
        )

    def _drain_pending_batched_decode_evictions(
        self,
    ) -> Iterator[tuple[TaskId, CancelledResponse]]:
        """Finalize batched-decode FOLLOWER cancellations whose driver-sent
        EvictMessage has actually been serviced since they were parked.

        Section 46 (2026-08-15). Companion to
        ``_drain_pending_pp_spec_cancellations`` -- same deferral pattern,
        different completion signal.

        THE BUG THIS CLOSES (confirmed on real 2-node hardware via
        simultaneous faulthandler stacks from both ranks): a client cancel
        reaches BOTH ranks. On the DRIVER, ``ExoBatchGenerator.cancel()``
        calls ``complete_request()`` -> ``send_evict_message()`` and then
        BLOCKS waiting for the EvictAck. On the FOLLOWER, this class used to
        immediately report ``CancelledResponse`` and drop the uid from
        ``_active_tasks`` -- which empties ``runner.py``'s
        ``while self.active_tasks:`` loop, so the follower exits it and goes
        idle ("runner idle" logged 5x). But the follower's ONLY handler for
        ``MSG_KIND_EVICT`` lives inside ``Rank1BatchedDecodeGlue.tick()``,
        which that loop is what drives. So the evict is never serviced, the
        driver's ``send_header`` blocks forever, and the follower's KV/cache
        slot is never released:

            driver:   cancel -> complete_request -> send_evict_message
                      -> send_header                      [BLOCKED FOREVER]
            follower: runner.py:317 main -> queue.get     [IDLE, gone home]

        Note the follower genuinely NEEDS that evict -- it is not redundant
        post-cancel. ``cancel()``'s batched-decode branches are all gated on
        ``_batched_decode_rank0_glue is not None`` (driver-only), so the
        follower frees NOTHING locally; only the driver's EvictMessage runs
        ``session.evict()`` + ``release_slot()`` on it. Skipping the evict
        would trade a deadlock for a per-cancel KV-slot leak.

        Fix: park such uids (keeping them in ``_active_tasks`` so the runner
        loop keeps pumping ``tick()``), and finalize here once the evict has
        genuinely been handled. The completion signal is the glue's own
        existing ``evicted_request_id`` -> synthesized ``finish_reason``
        path, which already fires exactly when ``MSG_KIND_EVICT`` is
        serviced -- so this needs no new cross-rank protocol, only the
        deferral. ``ExoBatchGenerator.has_work`` already reports True while
        the follower glue holds the request, so the loop stays alive on its
        own for the extra step()s this takes to converge.
        """
        # getattr-guarded: some tests construct BatchGenerator via
        # object.__new__ (bypassing __post_init__/dataclass field defaults),
        # so this attribute can legitimately be absent. Matches the
        # defensive getattr already used for _gen's optional hooks.
        if not getattr(self, "_pending_batched_decode_evict", None):
            return iter([])

        still_held = getattr(
            self._gen, "batched_decode_follower_awaiting_evict", None
        )
        results: list[tuple[TaskId, CancelledResponse]] = []
        for uid, task_id in list(self._pending_batched_decode_evict.items()):
            # Finalize once the follower no longer holds the request -- i.e.
            # the driver's evict landed and tick() released the slot.
            if still_held is None or not still_held(uid):
                results.append((task_id, CancelledResponse()))
                self._pending_batched_decode_evict.pop(uid, None)
                self._active_tasks.pop(uid, None)
        return iter(results)

    def _apply_cancellations(
        self,
    ) -> Iterator[tuple[TaskId, CancelledResponse]]:
        """Report CancelledResponse for tasks the client asked to cancel.

        Section 43 continued (2026-08-10) -- PP-spec deferred-finalize fix:
        for a uid whose PP speculative-decode generator is still live
        (``self._gen._pp_spec_gen_by_uid``), do NOT immediately report
        CancelledResponse or delete it from ``self._active_tasks`` here.
        ``self._gen.cancel(uids)`` only sets a cheap local flag
        (``_pp_spec_cancel_requested``) -- the generator itself only
        actually observes that flag and finishes (via its own
        ``pipeline_agree_cancel()`` cross-rank checkpoint, then
        StopIteration, then ``_close_pp_spec_gen()`` popping it from
        ``_pp_spec_gen_by_uid``) on the NEXT call to ``self._gen.step()``.

        Reporting CancelledResponse before that happens is a real bug, not
        just early: runner.py's `while self.active_tasks:` loop pops its
        own tracking entry the moment it sees CancelledResponse and, once
        empty, exits immediately -- so the runner goes idle ("runner idle:
        reclaimed MLX allocator pool") while the generator is still
        alive and never driven, permanently occupying
        ``_pp_spec_gen_by_uid[uid]``. Every subsequent request then fails
        with PPSpecAlreadyActiveError, confirmed on real 2-node hardware.

        Fix: park such uids in ``self._pending_pp_spec_cancel`` instead of
        finalizing them here. ``step()`` checks that dict on every call
        (after ``self._gen.step()`` has run, so the generator has had a
        chance to reach its checkpoint) and finalizes any uid whose
        generator has actually finished (no longer in
        ``_pp_spec_gen_by_uid``) via ``_drain_pending_pp_spec_cancellations()``.
        This also keeps ``self._active_tasks`` (and therefore runner.py's
        own tracking dict) populated for as many extra step() calls as it
        takes the generator to agree-and-close, which is exactly what
        keeps the runner's outer loop alive long enough to drive it there
        -- no separate has_work/loop-condition change needed, since
        ``ExoBatchGenerator.has_work`` already returns True while
        ``_pp_spec_gen_by_uid`` is non-empty.

        Non-PP-spec uids (TP/plain batched decode) are unaffected --
        finalized immediately, exactly as before.
        """
        if not self._cancelled_tasks:
            return iter([])

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        uids_to_cancel: list[int] = []
        results: list[tuple[TaskId, CancelledResponse]] = []
        pp_spec_gen_by_uid = getattr(self._gen, "_pp_spec_gen_by_uid", {})
        # Section 46 (2026-08-15): on the FOLLOWER rank of a batched-decode
        # session, finalizing here is the same "runner goes idle while the
        # peer still needs it" bug this method's own docstring describes for
        # the PP-spec path -- see _follower_awaiting_evict()'s docstring for
        # the full mechanism and the live stack traces that proved it.
        follower_awaiting_evict = getattr(
            self._gen, "batched_decode_follower_awaiting_evict", None
        )

        for uid, (task, _, _) in list(self._active_tasks.items()):
            if task.task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(uid)
                if uid in pp_spec_gen_by_uid:
                    # Defer: park it, DO NOT delete from _active_tasks or
                    # report CancelledResponse yet -- the generator hasn't
                    # actually finished.
                    self._pending_pp_spec_cancel[uid] = task.task_id
                elif follower_awaiting_evict is not None and follower_awaiting_evict(
                    uid
                ):
                    # Same deferral, different reason: this rank is the
                    # FOLLOWER and still holds real per-request state
                    # (KV/cache slot) that only the DRIVER's incoming
                    # EvictMessage releases. That message is serviced by
                    # Rank1BatchedDecodeGlue.tick()'s MSG_KIND_EVICT branch,
                    # which only runs while runner.py's `while
                    # self.active_tasks:` loop is still alive. Finalizing now
                    # exits that loop and strands the driver blocked forever
                    # in its own send_evict_message(). Park instead; the
                    # existing evicted_request_id -> finish_reason drain path
                    # (see _step_batched_decode's rank-1 branch) finalizes it
                    # once the evict has ACTUALLY been serviced.
                    self._pending_batched_decode_evict[uid] = task.task_id
                else:
                    results.append((task.task_id, CancelledResponse()))
                    del self._active_tasks[uid]

        if uids_to_cancel:
            self._gen.cancel(uids_to_cancel)

        already_cancelled = {tid for tid, _ in results}
        pending_task_ids = set(self._pending_pp_spec_cancel.values()) | set(
            getattr(self, "_pending_batched_decode_evict", {}).values()
        )
        for tid in self._cancelled_tasks:
            if (
                tid != CANCEL_ALL_TASKS
                and tid not in already_cancelled
                and tid not in pending_task_ids
            ):
                results.append((tid, CancelledResponse()))

        self._cancelled_tasks.clear()
        return iter(results)

    def _drain_pending_pp_spec_cancellations(
        self,
    ) -> Iterator[tuple[TaskId, CancelledResponse]]:
        """Finalize any parked PP-spec cancellations whose generator has
        actually finished since they were flagged. See
        ``_apply_cancellations``'s docstring for the full race this closes.
        Called every ``step()`` (after ``self._gen.step()`` has run, giving
        the generator a chance to reach its agreed-cancel checkpoint).
        """
        if not self._pending_pp_spec_cancel:
            return iter([])
        pp_spec_gen_by_uid = getattr(self._gen, "_pp_spec_gen_by_uid", {})
        done: list[tuple[int, TaskId]] = [
            (uid, tid)
            for uid, tid in self._pending_pp_spec_cancel.items()
            if uid not in pp_spec_gen_by_uid
        ]
        results: list[tuple[TaskId, CancelledResponse]] = []
        for uid, tid in done:
            del self._pending_pp_spec_cancel[uid]
            self._active_tasks.pop(uid, None)
            results.append((tid, CancelledResponse()))
        return iter(results)

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

    def _start_task(self, task: TextGeneration) -> int:
        from exo.worker.engines.mlx.trace import request_trace, T

        _check_for_debug_prompts(task.task_params)
        with T("start_task.apply_chat_template"):
            prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )
            else:
                self.prefill_heartbeat()

        def distributed_prompt_progress_callback() -> None:
            t0 = time.perf_counter()
            self.agree_on_cancellations_fast()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()
            request_trace.record("prefill.distributed_callback", t0)

        def prefill_cancel_requested() -> bool:
            """Cancellation probe for the INTERRUPTIBLE chunk drive
            (design doc Section 96).

            ``distributed_prompt_progress_callback`` above is the cancel
            hook for the EAGER generator loop. Proven on hardware that
            the interruptible drive does not use it: one 40K-token
            prefill logged 79 ``PREFILL_ADVANCE_APPLIED`` against only 6
            callback firings, ALL of them before the cancel arrived, so
            ``PrefillCancelled`` was never raised and
            ``PREFILL_CANCELLED_PATH`` never fired. Delivery, collection
            and cross-rank agreement were all verified healthy in the
            same run -- the id genuinely reaches ``_cancelled_tasks``;
            nothing was asking.

            Returns a plain bool and NEVER raises: the chunk driver is
            responsible for choosing the quiescent moment to act, and an
            exception thrown mid-drive is exactly what Section 93 rules
            out. Local-only and non-blocking (a set membership test plus
            a ``receive_nowait`` drain), so it adds no cross-rank traffic
            to the prefill hot path.
            """
            # DELIBERATELY does NOT drain cancel_receiver here.
            #
            # 2026-08-16 (Section 99): an earlier version of this probe
            # called ``self.cancel_receiver.collect()``, which bottoms
            # out in ``multiprocessing.Queue.get(block=False)`` --
            # acquiring the queue's INTERNAL LOCK. This probe runs on
            # the generator/chunk-drive thread while the supervisor
            # thread drains the SAME queue, so the two contend that
            # lock. Reproduced on hardware: the runner wedged with two
            # threads parked in ``lock_PyThread_acquire_lock ->
            # _PyMutex_LockTimed -> _PySemaphore_Wait``, no MLX and no
            # jaccl frames anywhere in the stack, GPU idle, zero log
            # output for 5+ minutes.
            #
            # ``_cancelled_tasks`` is already populated by
            # ``agree_on_cancellations_fast()`` on the owning thread
            # (verified on hardware: CANCELPROBE showed the id landing
            # in the set). A plain set-membership read is lock-free and
            # sufficient -- the drain was never this probe's job.
            return self.should_cancel(task.task_id)

        # Register the probe on the generator rather than threading it
        # through submit(): the chunk drive lives behind
        # submit() -> _submit_batched_decode_deferred() -> the glue, and
        # the glue is what owns both the quiescent chunk boundary AND
        # the already-built bilateral abort
        # (Rank0BatchedDecodeGlue.abort_prefill_session -> blocking
        # PrefillAbortMessage/ack round trip). A parameter would have to
        # cross four layers to reach a component that can already do the
        # work; a registered probe reaches it in one.
        self._gen.set_prefill_cancel_probe(prefill_cancel_requested)

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            # Section 48 (2026-08-15): observe a LOCALLY-known cancel every
            # token, instead of only at the every-100-tokens collective
            # checkpoint below.
            #
            # Measured on real hardware: with the collective check as the
            # ONLY observation point, a cancel issued at token ~16 was not
            # noticed until the request had ground out the remaining ~84
            # tokens of the interval -- ~97 SECONDS at 30K context. The
            # runner did stop correctly and log "runner idle" afterwards
            # (cancellation was never broken), it was just ~97s late, which
            # is both bad UX and what made section27_cancel_abort_test's 90s
            # convergence window fail.
            #
            # This is deliberately LOCAL-ONLY and adds no cross-rank
            # traffic: cancel_receiver.collect() is non-blocking
            # (receive_nowait until WouldBlock) and should_cancel() is a
            # plain set membership test (engines/base.py). Both PP ranks
            # independently receive the same TaskCancelled command, so each
            # one can observe it locally at the same logical point without
            # needing to agree first. The existing collective
            # agree_on_cancellations()/agree_on_tasks() cadence below is
            # UNCHANGED -- it remains the authority for cross-rank
            # agreement; this only shortens the latency before a rank
            # notices a cancel it has already been told about directly.
            for _cancelled_id in self.cancel_receiver.collect():
                self._cancelled_tasks.add(_cancelled_id)
            if self.should_cancel(task.task_id):
                self._cancelled_tasks.add(task.task_id)

            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                t0 = time.perf_counter()
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)
                self.agree_on_tasks()
                request_trace.record("decode.agree_on_cancel_and_tasks", t0)

        with T("start_task.mlx_gen_submit"):
            return self._gen.submit(
                task_params=task.task_params,
                prompt=prompt,
                on_prefill_progress=on_prefill_progress,
                distributed_prompt_progress_callback=distributed_prompt_progress_callback,
                on_generation_token=on_generation_token,
            )

    def _batched_start_task(
        self, tasks: list[TextGeneration]
    ) -> list[int]:
        """Build per-task callbacks and run a SINGLE batched prefill across tasks.

        Mirrors ``_start_task`` per task (chat template, prefill-progress event,
        cancellation polling, generation-token callback) but bundles all of them
        into one ``ExoBatchGenerator.submit_batched`` call so the prefill phase
        runs at shape (B, L_chunk) instead of serializing per task. Returns one
        uid per input task, in the same order.
        """
        from exo.worker.engines.mlx.trace import T, request_trace

        bundle: list[tuple[
            TextGenerationTaskParams,
            str,
            "object",
            "object",
            "object",
        ]] = []

        for task in tasks:
            _check_for_debug_prompts(task.task_params)

            with T("batched_start_task.apply_chat_template"):
                prompt = apply_chat_template(self.tokenizer, task.task_params)

            # Closures bind to ``task`` so each task's prefill-progress
            # event and cancellation poll target its own task_id.
            def _make_on_prefill_progress(_task: TextGeneration):
                def on_prefill_progress(processed: int, total: int) -> None:
                    if self.device_rank == 0:
                        self.event_sender.send(
                            ChunkGenerated(
                                command_id=_task.command_id,
                                chunk=PrefillProgressChunk(
                                    model=self.model_id,
                                    processed_tokens=processed,
                                    total_tokens=total,
                                ),
                            )
                        )
                    else:
                        self.prefill_heartbeat()

                return on_prefill_progress

            def _make_distributed_callback(_task: TextGeneration):
                def distributed_prompt_progress_callback() -> None:
                    t0 = time.perf_counter()
                    # Poll cancellations across both ranks. We DON'T raise
                    # PrefillCancelled here even if the per-task is
                    # cancelled — the batched prefill processes all
                    # streams together and we'd waste the rest of the
                    # batch's compute. Instead, the cancellation is
                    # recorded in ``_cancelled_tasks`` and applied after
                    # prefill completes via ``_apply_cancellations``.
                    self.agree_on_cancellations_fast()
                    logger.info(
                        f"CANCELPROBE[prefill_batched.cb] task={_task.task_id} "
                        f"should_cancel={self.should_cancel(_task.task_id)} "
                        f"(DEFERRED BY DESIGN - does not raise)"
                    )
                    request_trace.record(
                        "prefill_batched.distributed_callback", t0
                    )

                return distributed_prompt_progress_callback

            def _make_on_generation_token(_task: TextGeneration):
                tokens_since_cancel_check = self.check_for_cancel_every

                def on_generation_token() -> None:
                    nonlocal tokens_since_cancel_check
                    tokens_since_cancel_check += 1
                    if tokens_since_cancel_check >= self.check_for_cancel_every:
                        tokens_since_cancel_check = 0
                        t0 = time.perf_counter()
                        self.agree_on_cancellations()
                        if self.should_cancel(_task.task_id):
                            self._cancelled_tasks.add(_task.task_id)
                        self.agree_on_tasks()
                        request_trace.record(
                            "decode.agree_on_cancel_and_tasks", t0
                        )

                return on_generation_token

            bundle.append((
                task.task_params,
                prompt,
                _make_on_prefill_progress(task),
                _make_distributed_callback(task),
                _make_on_generation_token(task),
            ))

        with T("batched_start_task.mlx_gen_submit_batched"):
            return self._gen.submit_batched(bundle)  # type: ignore[arg-type]

    def close(self) -> None:
        self._gen.close()
        del self.model, self.tokenizer, self.group

    def reset_after_reconnect(self) -> list[int]:
        return self._gen.reset_after_reconnect()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )

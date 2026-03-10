import itertools
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TextGeneration
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    mlx_generate,
    warmup_inference,
)
from exo.shared.constants import EXO_TRACING_ENABLED
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    mx_all_gather_tasks,
    mx_any,
)
from exo.worker.runner.bootstrap import logger

from .model_output_parsers import apply_all_parsers
from .tool_parsers import ToolParser


class Cancelled:
    pass


class Finished:
    pass


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


class InferenceGenerator(ABC):
    _cancelled_tasks: set[TaskId]

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: TextGeneration,
    ) -> None: ...

    @abstractmethod
    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, ToolCallResponse | GenerationResponse | Cancelled | Finished]
    ]: ...

    @abstractmethod
    def close(self) -> None: ...


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
class SequentialGenerator(InferenceGenerator):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    heartbeat: object | None = None

    def _touch_heartbeat(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.value = time.monotonic()  # pyright: ignore[reportAttributeAccessIssue]

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: (
        tuple[
            TextGeneration,
            # mlx generator that does work
            Generator[GenerationResponse],
            # queue that the 1st generator should push to and 3rd generator should pull from
            GeneratorQueue[GenerationResponse],
            # generator to get parsed outputs
            Generator[GenerationResponse | ToolCallResponse | None],
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

    def submit(
        self,
        task: TextGeneration,
    ) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        if EXO_TRACING_ENABLED:
            t0 = time.perf_counter()
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        self._queue.extend(task for task in self._maybe_queue if task in agreed)
        self._maybe_queue = [task for task in self._maybe_queue if task in different]
        if EXO_TRACING_ENABLED:
            logger.info(f"agree_on_tasks took {(time.perf_counter() - t0) * 1000:.1f}ms")

    def _drain_local_cancellations(self) -> None:
        """Drain cancel pipe locally — no collective ops here.

        Cancel signals are drained from the local pipe without any
        collective ops.  The actual cancel *decision* is made collectively
        via mx_any in step() so that all TP peers agree at the same step
        boundary and no peer moves to agree_on_tasks while another is
        still in a decode all_reduce.
        """
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            else:
                self._cancelled_tasks.add(task_id)

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        # Drain cancel pipe locally — no collective ops to avoid TP deadlock.
        self._drain_local_cancellations()

        if self._active is None:
            self.agree_on_tasks()

            if self._queue:
                self._start_next()
            else:
                return self._flush_cancelled()

        assert self._active is not None

        task, mlx_gen, queue, output_generator = self._active

        # Use mx_any so ALL TP peers agree on cancellation at the same step
        # boundary.  Without this, one peer may cancel and move to
        # agree_on_tasks (all_gather) while the other is still in a decode
        # all_reduce → mismatched collective ops → deadlock.
        if mx_any(self.should_cancel(task.task_id), self.group):
            self._cancelled_tasks.discard(task.task_id)
            self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
            self._active = None
            if self._queue:
                self._start_next()
            return itertools.chain(
                [(task.task_id, Cancelled())],
                self._flush_cancelled(),
            )

        response = None
        try:
            queue.push(next(mlx_gen))
            response = next(output_generator)
        except PrefillCancelled:
            response = Cancelled()
            self._active = None
            self._cancelled_tasks.discard(task.task_id)
            self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
            if self._queue:
                self._start_next()
        except StopIteration:
            response = Finished()
            self._active = None
            if self._queue:
                self._start_next()
        except ValueError as e:
            logger.warning(f"Task {task.task_id} rejected: {e}")
            self._send_error(task, e)
            self._active = None
            response = Finished()
        except Exception as e:
            self._send_error(task, e)
            self._active = None
            raise
        return itertools.chain(
            [] if response is None else [(task.task_id, response)],
            self._flush_cancelled(),
        )

    def _flush_cancelled(self) -> Iterable[tuple[TaskId, Cancelled]]:
        cancelled = list(self._cancelled_tasks)
        self._cancelled_tasks.clear()
        return ((tid, Cancelled()) for tid in cancelled)

    def _start_next(self) -> None:
        task = self._queue.popleft()
        try:
            mlx_gen = self._build_generator(task)
        except ValueError as e:
            logger.warning(f"Task {task.task_id} rejected during build: {e}")
            self._send_error(task, e)
            return
        except Exception as e:
            self._send_error(task, e)
            raise
        queue = GeneratorQueue[GenerationResponse]()

        if task.task_params.bench:
            output_generator = queue.gen()
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
        self._active = (task, mlx_gen, queue, output_generator)

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

        def distributed_prompt_progress_callback() -> None:
            # Heartbeat only — no collective ops during prefill.
            # Collective ops (all_sum/all_gather) before the model's
            # mx.eval() can corrupt JACCL RDMA state and deadlock
            # the subsequent all_reduce ops in TP layers.
            self._touch_heartbeat()

        def on_generation_token() -> None:
            # Heartbeat only — no collective ops during decode.
            # mx.async_eval() of the next token may be running
            # concurrently on the GPU (all_reduce for TP), so any
            # collective op here (all_sum for mx_any) conflicts with
            # JACCL RDMA and can deadlock.
            # Cancel is checked between requests in step() instead.
            self._touch_heartbeat()

        # Touch heartbeat before mlx_generate — tokenization, cache lookup,
        # and deepcopy can take tens of seconds on large prompts.
        self._touch_heartbeat()

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
        )

    def close(self) -> None:
        del self.model, self.tokenizer, self.group


@dataclass(eq=False)
class BatchGenerator(InferenceGenerator):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    heartbeat: object | None = None
    check_for_cancel_every: int = 50

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _mlx_gen: ExoBatchGenerator = field(init=False)
    _active_tasks: dict[
        int,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Generator[GenerationResponse | ToolCallResponse | None],
        ],
    ] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._mlx_gen = ExoBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
        )

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(
        self,
        task: TextGeneration,
    ) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        self._queue.extend(task for task in self._maybe_queue if task in agreed)
        self._maybe_queue = [task for task in self._maybe_queue if task in different]

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def _drain_local_cancellations(self) -> None:
        """Drain cancel pipe locally — no collective ops.

        Same rationale as SequentialGenerator: collective ops in callbacks
        or between RDMA-heavy steps can collide with JACCL.
        """
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            else:
                self._cancelled_tasks.add(task_id)

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        self._drain_local_cancellations()

        # Use mx_any so ALL TP peers agree on cancellation before
        # the next model forward pass (all_reduce).  Without this,
        # one peer may cancel and exit the generation loop while the
        # other is still in all_reduce → deadlock.
        #
        # Only consider cancels for tasks still active in the MLX batch —
        # stale cancels (for already-completed tasks) must not trigger
        # CANCEL_ALL on other peers, or they'll kill unrelated active tasks.
        active_task_ids = {
            task.task_id for _, (task, _, _) in self._active_tasks.items()
        }
        has_cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks
        has_relevant_cancel = has_cancel_all or bool(
            self._cancelled_tasks & active_task_ids
        )

        if mx_any(has_relevant_cancel, self.group):
            if not has_relevant_cancel:
                # Another peer has relevant cancels but we don't.
                # In TP mode all peers share the same active task set,
                # so cancelling everything keeps them in sync.
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            early_cancelled = self._apply_cancellations()
        else:
            # Discard stale cancels that don't match any active task.
            self._cancelled_tasks &= active_task_ids | {CANCEL_ALL_TASKS}
            early_cancelled = []

        if not self._queue:
            self.agree_on_tasks()

        # Submit any queued tasks to the engine
        while self._queue and len(self._active_tasks) < EXO_MAX_CONCURRENT_REQUESTS:
            task = self._queue.popleft()
            try:
                uid = self._start_task(task)
            except PrefillCancelled:
                continue
            except ValueError as e:
                logger.warning(f"Task {task.task_id} rejected: {e}")
                self._send_error(task, e)
                continue
            except Exception as e:
                self._send_error(task, e)
                raise

            queue = GeneratorQueue[GenerationResponse]()
            if task.task_params.bench:
                output_generator = queue.gen()
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

        if not self._mlx_gen.has_work:
            return early_cancelled

        results = self._mlx_gen.step()

        output: list[
            tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
        ] = []
        for uid, response in results:
            if uid not in self._active_tasks:
                # should we error here?
                logger.warning(f"{uid=} not found in active tasks")
                continue

            task, queue, output_generator = self._active_tasks[uid]
            queue.push(response)
            parsed = next(output_generator)

            if parsed is not None:
                output.append((task.task_id, parsed))

            if response.finish_reason is not None:
                output.append((task.task_id, Finished()))
                del self._active_tasks[uid]

        return itertools.chain(early_cancelled, output, self._apply_cancellations())

    def _apply_cancellations(
        self,
    ) -> list[tuple[TaskId, Cancelled]]:
        if not self._cancelled_tasks:
            return []

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        uids_to_cancel: list[int] = []
        results: list[tuple[TaskId, Cancelled]] = []

        for uid, (task, _, _) in list(self._active_tasks.items()):
            if task.task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(uid)
                results.append((task.task_id, Cancelled()))
                del self._active_tasks[uid]

        if uids_to_cancel:
            self._mlx_gen.cancel(uids_to_cancel)

        already_cancelled = {tid for tid, _ in results}
        for tid in self._cancelled_tasks:
            if tid != CANCEL_ALL_TASKS and tid not in already_cancelled:
                results.append((tid, Cancelled()))

        self._cancelled_tasks.clear()
        return results

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

        def distributed_prompt_progress_callback() -> None:
            # Heartbeat only — no collective ops during prefill.
            # Pipeline parallel prefill uses its own RDMA collective ops
            # (all_gather for TP layers); injecting additional collective
            # ops (all_sum/all_gather for cancellation) collides with
            # JACCL RDMA and causes wc status=1 errors + deadlock.
            if self.heartbeat is not None:
                self.heartbeat.value = time.monotonic()  # pyright: ignore[reportAttributeAccessIssue]

        def on_generation_token() -> None:
            # Heartbeat only — same RDMA collision risk during decode.
            if self.heartbeat is not None:
                self.heartbeat.value = time.monotonic()  # pyright: ignore[reportAttributeAccessIssue]

        return self._mlx_gen.submit(
            task_params=task.task_params,
            prompt=prompt,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
        )

    def close(self) -> None:
        self._mlx_gen.close()
        del self.model, self.tokenizer, self.group

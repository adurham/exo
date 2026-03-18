import os
import time
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx
from anyio import WouldBlock
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_TRACING_ENABLED
from exo.shared.models.model_cards import ModelTask
from exo.shared.types.chunks import (
    DraftChunk,
    ErrorChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import (
    ConnectToGroup,
    DraftGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    InferenceGenerator,
    SequentialGenerator,
)

from .batch_generator import Cancelled, Finished
from .tool_parsers import make_mlx_parser


class ExitCode(str, Enum):
    AllTasksComplete = "AllTasksComplete"
    Continue = "Continue"
    Shutdown = "Shutdown"


class Runner:
    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
        *,
        heartbeat: object | None = None,
        heartbeat_timeout: object | None = None,
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        self.instance, self.runner_id, self.shard_metadata = (
            self.bound_instance.instance,
            self.bound_instance.bound_runner_id,
            self.bound_instance.bound_shard,
        )
        self.model_id = self.shard_metadata.model_card.model_id
        self.device_rank = self.shard_metadata.device_rank

        logger.info("hello from the runner")
        if getattr(self.shard_metadata, "immediate_exception", False):
            raise Exception("Fake exception - runner failed to spin up.")
        if timeout := getattr(self.shard_metadata, "should_timeout", 0):
            time.sleep(timeout)

        self.setup_start_time = time.time()

        self.generator: Builder | InferenceGenerator = Builder(
            self.model_id, self.event_sender, self.cancel_receiver,
            heartbeat=heartbeat,
            heartbeat_timeout=heartbeat_timeout,
        )

        self.seen: set[TaskId] = set()
        self.active_tasks: dict[
            TaskId,
            TextGeneration,
        ] = {}

        logger.info("runner created")
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus):
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus):
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task):
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self):
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_first_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    def handle_first_task(self, task: Task):
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                assert isinstance(self.generator, Builder)
                logger.info("runner connecting")
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)

                if EXO_TRACING_ENABLED:
                    t_connect = time.time()
                self.generator.group = initialize_mlx(self.bound_instance)

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())
                if EXO_TRACING_ENABLED:
                    logger.info(
                        f"runner connected in {time.time() - t_connect:.2f}s"
                    )
                else:
                    logger.info("runner connected")

            # we load the model if it's connected with a group, or idle without a group. we should never tell a model to connect if it doesn't need to
            case LoadModel() if isinstance(self.generator, Builder) and (
                (
                    isinstance(self.current_status, RunnerConnected)
                    and self.generator.group is not None
                )
                or (
                    isinstance(self.current_status, RunnerIdle)
                    and self.generator.group is None
                )
            ):
                total_layers = (
                    self.shard_metadata.end_layer - self.shard_metadata.start_layer
                )
                if EXO_TRACING_ENABLED:
                    t_load = time.time()
                logger.info("runner loading")

                self.update_status(
                    RunnerLoading(layers_loaded=0, total_layers=total_layers)
                )
                self.acknowledge_task(task)

                def on_model_load_timeout() -> None:
                    self.update_status(
                        RunnerFailed(error_message="Model loading timed out")
                    )
                    time.sleep(0.5)

                def on_layer_loaded(layers_loaded: int, total: int) -> None:
                    self.update_status(
                        RunnerLoading(layers_loaded=layers_loaded, total_layers=total)
                    )

                assert (
                    ModelTask.TextGeneration in self.shard_metadata.model_card.tasks
                ), f"Incorrect model task(s): {self.shard_metadata.model_card.tasks}"
                self.generator.inference_model, self.generator.tokenizer, self.generator.draft_model = (
                    load_mlx_items(
                        self.bound_instance,
                        self.generator.group,
                        on_timeout=on_model_load_timeout,
                        on_layer_loaded=on_layer_loaded,
                    )
                )

                # Speculative decoding: if the instance has a draft_model configured,
                # create a DraftClient that queries the draft instance's node directly.
                # We resolve the draft node's IP from cluster state at setup time.
                _draft_model_id = self.bound_instance.instance.draft_model
                _draft_tokens = self.bound_instance.instance.draft_tokens
                # All TP ranks must know about speculative decode for generator selection
                if _draft_model_id:
                    self.generator.use_speculative = True
                if _draft_model_id and self.device_rank == 0:
                    try:
                        from exo.worker.engines.mlx.draft_client import DraftClient
                        _draft_url = self._resolve_draft_url(_draft_model_id)
                        if _draft_url:
                            self.generator.draft_model = DraftClient(
                                server_url=_draft_url,
                                num_draft_tokens=_draft_tokens,
                                draft_model=_draft_model_id,
                            )
                            logger.info(f"Draft client ready (url={_draft_url}, model={_draft_model_id}, K={_draft_tokens})")
                        else:
                            logger.warning(f"Could not resolve draft node for {_draft_model_id}")
                    except Exception as e:
                        logger.warning(f"Failed to init draft client: {e}")

                self.generator.kv_prefix_cache = KVPrefixCache(self.generator.group)
                self.generator = self.generator.build()

                # Start direct draft server for low-latency draft requests
                if self.device_rank == 0:
                    self._start_draft_server()

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())
                if EXO_TRACING_ENABLED:
                    logger.info(f"runner loaded in {time.time() - t_load:.2f}s")
                else:
                    logger.info("runner loaded")

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                assert isinstance(self.generator, InferenceGenerator)
                logger.info("runner warming up")

                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)

                self.generator.warmup()

                logger.info(
                    f"runner initialized in {time.time() - self.setup_start_time} seconds"
                )

                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())
                logger.info("runner ready")

            case TextGeneration() if isinstance(self.current_status, RunnerReady):
                return_code = self.handle_generation_tasks(starting_task=task)
                if return_code == ExitCode.Shutdown:
                    return

            case DraftGeneration() if isinstance(self.current_status, (RunnerReady, RunnerRunning)):
                self.handle_draft_task(task)

            case Shutdown():
                self.shutdown(task)
                return

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                )

    def handle_draft_task(self, task: DraftGeneration):
        """Handle a draft generation request with persistent KV cache."""
        assert isinstance(self.generator, InferenceGenerator)
        self.acknowledge_task(task)
        self.send_task_status(task.task_id, TaskStatus.Running)

        t0 = time.perf_counter()

        # Lazy-init the draft cache
        if not hasattr(self, '_draft_cache'):
            from mlx_lm.models.cache import make_prompt_cache
            self._draft_cache = make_prompt_cache(self.generator.model)
            self._draft_cache_len = 0

        try:
            if task.action == "reset":
                from mlx_lm.models.cache import make_prompt_cache
                self._draft_cache = make_prompt_cache(self.generator.model)
                self._draft_cache_len = 0
                result_tokens: list[int] = []

            elif task.action == "prefill":
                if task.prefill_token_ids:
                    tokens = mx.array([task.prefill_token_ids])
                    logits = self.generator.model(tokens, cache=self._draft_cache)
                    mx.eval(logits)
                    self._draft_cache_len += len(task.prefill_token_ids)
                result_tokens = []

            else:  # "draft"
                from mlx_lm.models.cache import trim_prompt_cache
                if task.trim > 0:
                    trim_prompt_cache(self._draft_cache, task.trim)
                    self._draft_cache_len = max(0, self._draft_cache_len - task.trim)

                result_tokens = []
                tok = task.token_id
                for _ in range(task.num_tokens):
                    current = mx.array([[tok]])
                    logits = self.generator.model(current, cache=self._draft_cache)
                    mx.eval(logits)
                    self._draft_cache_len += 1
                    tok = logits[0, -1].argmax().item()
                    result_tokens.append(tok)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=DraftChunk(
                        model=self.model_id,
                        tokens=result_tokens,
                        cache_len=self._draft_cache_len,
                        elapsed_ms=elapsed_ms,
                    ),
                )
            )
        except Exception as e:
            logger.warning(f"Draft generation failed: {e}")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        error_message=str(e),
                    ),
                )
            )

        self.send_task_status(task.task_id, TaskStatus.Complete)

    def _resolve_draft_url(self, draft_model_id: str) -> str | None:
        """Find the API URL of the node running the draft model instance.

        Retries up to 30s since the draft node's network info may not have
        propagated via gossip when the primary model finishes loading.
        """
        import json
        import urllib.request
        for attempt in range(10):
            try:
                url = self._resolve_draft_url_once(draft_model_id)
                if url:
                    return url
                if attempt < 9:
                    logger.info(f"Draft URL not yet resolvable (attempt {attempt + 1}/10), retrying in 3s...")
                    time.sleep(3)
            except Exception as e:
                if attempt < 9:
                    logger.info(f"Draft URL resolution failed (attempt {attempt + 1}/10): {e}, retrying in 3s...")
                    time.sleep(3)
                else:
                    logger.warning(f"Failed to resolve draft node URL after 10 attempts: {e}")
        return None

    def _resolve_draft_url_once(self, draft_model_id: str) -> str | None:
        """Single attempt to find the draft node URL from cluster state."""
        import json
        import urllib.request
        with urllib.request.urlopen("http://localhost:52415/state", timeout=5) as resp:
            state = json.loads(resp.read())
        for inst in state.get("instances", {}).values():
            # Handle tagged union format: {"MlxRingInstance": {...}} or {"MlxJacclInstance": {...}}
            inst_data = inst
            for key in ("MlxRingInstance", "MlxJacclInstance"):
                if key in inst:
                    inst_data = inst[key]
                    break
            model_id = inst_data.get("shardAssignments", {}).get("modelId", "")
            if model_id != draft_model_id:
                continue
            node_to_runner = inst_data.get("shardAssignments", {}).get("nodeToRunner", {})
            draft_node_id = next(iter(node_to_runner.keys()), None)
            if not draft_node_id:
                continue
            for node_id, net_info in state.get("nodeNetwork", {}).items():
                if node_id != draft_node_id:
                    continue
                tb_ip: str | None = None
                fallback_ip: str | None = None
                for iface in net_info.get("interfaces", []):
                    ip = iface.get("ipAddress", "")
                    if not ip or ":" in ip or ip.startswith("127.") or ip.startswith("fe80") or ip.startswith("169.254"):
                        continue
                    if iface.get("interfaceType", "") == "thunderbolt":
                        tb_ip = ip
                        break
                    if fallback_ip is None:
                        fallback_ip = ip
                best = tb_ip or fallback_ip
                if best:
                    draft_port = int(os.environ.get("EXO_DRAFT_DIRECT_PORT", "52416"))
                    return f"http://{best}:{draft_port}"
        return None

    def _start_draft_server(self) -> None:
        """Start a lightweight HTTP server for direct draft requests, bypassing the task pipeline."""
        import json
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        runner = self
        model = self.generator.model  # pyright: ignore[reportAttributeAccessIssue]
        lock = threading.Lock()

        # Shared draft cache state
        cache = make_prompt_cache(model)
        cache_len = 0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/v1/draft/health":
                    self._json({"status": "ok", "cache_len": cache_len, "model": runner.model_id})
                else:
                    self.send_error(404)

            def do_POST(self):
                nonlocal cache, cache_len
                body = self._read()
                if self.path == "/v1/draft":
                    token_id = body.get("token_id", 1)
                    num_tokens = body.get("num_tokens", 10)
                    trim = body.get("trim", 0)
                    t0 = time.perf_counter()
                    with lock:
                        if trim > 0:
                            trim_prompt_cache(cache, trim)
                            cache_len = max(0, cache_len - trim)
                        tokens = []
                        tok = token_id
                        for _ in range(num_tokens):
                            logits = model(mx.array([[tok]]), cache=cache)
                            mx.eval(logits)
                            cache_len += 1
                            tok = logits[0, -1].argmax().item()
                            tokens.append(tok)
                    self._json({"tokens": tokens, "elapsed_ms": (time.perf_counter() - t0) * 1000})
                elif self.path == "/v1/draft/prefill":
                    token_ids = body.get("token_ids", body.get("token_ids", []))
                    t0 = time.perf_counter()
                    with lock:
                        if token_ids:
                            logits = model(mx.array([token_ids]), cache=cache)
                            mx.eval(logits)
                            cache_len += len(token_ids)
                    self._json({"cache_len": cache_len, "elapsed_ms": (time.perf_counter() - t0) * 1000})
                elif self.path == "/v1/draft/reset":
                    with lock:
                        cache = make_prompt_cache(model)
                        cache_len = 0
                    self._json({"status": "ok"})
                else:
                    self.send_error(404)

            def _read(self) -> dict:
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n)) if n else {}

            def _json(self, data: dict):
                out = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, format, *args):
                pass  # suppress per-request logging

        port = int(os.environ.get("EXO_DRAFT_DIRECT_PORT", "52416"))
        try:
            server = HTTPServer(("0.0.0.0", port), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            logger.info(f"Draft direct server started on port {port}")
        except OSError as e:
            logger.warning(f"Could not start draft direct server on port {port}: {e}")

    def shutdown(self, task: Task):
        logger.info("runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)
        if isinstance(self.generator, InferenceGenerator):
            self.generator.close()
        mx.clear_cache()
        import gc

        gc.collect()
        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())

    def submit_text_generation(self, task: TextGeneration) -> bool:
        assert isinstance(self.generator, InferenceGenerator)
        try:
            self.active_tasks[task.task_id] = task
            self.generator.submit(task)
            return True
        except ValueError as e:
            logger.warning(f"Task {task.task_id} rejected: {e}")
            self.active_tasks.pop(task.task_id, None)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        error_message=str(e),
                        model=self.model_id,
                    ),
                )
            )
            self.send_task_status(task.task_id, TaskStatus.Complete)
            return False

    def _try_connect_draft(self) -> None:
        """Lazy DraftClient init — retries if the draft instance wasn't ready at load time."""
        assert isinstance(self.generator, InferenceGenerator)
        if self.generator.draft_model is not None and hasattr(self.generator.draft_model, 'prefill'):
            return  # Already connected with TCP draft
        _draft_model_id = self.bound_instance.instance.draft_model
        _draft_tokens = self.bound_instance.instance.draft_tokens
        if not _draft_model_id or self.device_rank != 0:
            return
        try:
            from exo.worker.engines.mlx.draft_client import DraftClient
            _draft_url = self._resolve_draft_url(_draft_model_id)
            if _draft_url:
                self.generator.draft_model = DraftClient(
                    server_url=_draft_url,
                    num_draft_tokens=_draft_tokens,
                    draft_model=_draft_model_id,
                )
                logger.info(f"Draft client connected (url={_draft_url}, model={_draft_model_id}, K={_draft_tokens})")
                if isinstance(self.generator, BatchGenerator):
                    logger.warning(
                        "Draft client connected but BatchGenerator is active — "
                        "speculative decoding requires SequentialGenerator (restart needed)"
                    )
        except Exception as e:
            logger.debug(f"Draft client not yet available: {e}")

    def handle_generation_tasks(self, starting_task: TextGeneration):
        assert isinstance(self.current_status, RunnerReady)
        assert isinstance(self.generator, InferenceGenerator)

        self._try_connect_draft()

        logger.info(f"received chat request: {starting_task}")
        self.update_status(RunnerRunning())
        logger.info("runner running")
        self.acknowledge_task(starting_task)
        self.seen.add(starting_task.task_id)

        if not self.submit_text_generation(starting_task):
            self.update_status(RunnerReady())
            return ExitCode.Continue

        while self.active_tasks:
            results = self.generator.step()

            finished: list[TaskId] = []
            for task_id, result in results:
                match result:
                    case Cancelled():
                        finished.append(task_id)
                    case Finished():
                        self.send_task_status(task_id, TaskStatus.Complete)
                        finished.append(task_id)
                    case _:
                        task = self.active_tasks.get(task_id)
                        if task is None:
                            # Task was already cancelled/finished but the
                            # generator still yielded a trailing response.
                            continue
                        self.send_response(result, task.command_id)

            for task_id in finished:
                self.active_tasks.pop(task_id, None)

            try:
                task = self.task_receiver.receive_nowait()

                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)

                match task:
                    case TextGeneration():
                        self.acknowledge_task(task)
                        self.submit_text_generation(task)
                    case Shutdown():
                        self.shutdown(task)
                        return ExitCode.Shutdown
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {self.current_status=}"
                        )

            except WouldBlock:
                pass

        # Send shutdown signal to draft node if we're the TP master
        if (hasattr(self.generator, 'draft_model') and self.generator.draft_model is not None
                and hasattr(self.generator.draft_model, 'shutdown')):
            try:
                self.generator.draft_model.shutdown()
                logger.info("Sent shutdown signal to draft node")
            except Exception:
                pass

        self.update_status(RunnerReady())
        logger.info("runner ready")

        return ExitCode.AllTasksComplete

    def send_response(
        self, response: GenerationResponse | ToolCallResponse, command_id: CommandId
    ):
        match response:
            case GenerationResponse():
                if self.device_rank == 0 and response.finish_reason == "error":
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ErrorChunk(
                                error_message=response.text,
                                model=self.model_id,
                            ),
                        )
                    )

                elif self.device_rank == 0:
                    assert response.finish_reason not in (
                        "error",
                        "tool_calls",
                        "function_call",
                    )
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=TokenChunk(
                                model=self.model_id,
                                text=response.text,
                                token_id=response.token,
                                usage=response.usage,
                                finish_reason=response.finish_reason,
                                stats=response.stats,
                                logprob=response.logprob,
                                top_logprobs=response.top_logprobs,
                                is_thinking=response.is_thinking,
                            ),
                        )
                    )
            case ToolCallResponse():
                if self.device_rank == 0:
                    self.event_sender.send(
                        ChunkGenerated(
                            command_id=command_id,
                            chunk=ToolCallChunk(
                                tool_calls=response.tool_calls,
                                model=self.model_id,
                                usage=response.usage,
                                stats=response.stats,
                            ),
                        )
                    )


class _TpDraftWrapper:
    """Wraps a DraftClient (rank 0 only) with TP broadcast for speculative decoding.

    All TP ranks must call the model with identical draft tokens. Rank 0 fetches
    from the TCP DraftClient, then broadcasts the result to all other ranks via
    mx.distributed.all_gather before returning.
    """

    def __init__(self, draft_client: object | None, group: mx.distributed.Group):
        self._client = draft_client  # DraftClient on rank 0, None on other ranks
        self._group = group
        self.num_draft_tokens: int = getattr(draft_client, 'num_draft_tokens', 10) if draft_client else 10
        self.server_url: str = getattr(draft_client, 'server_url', '') if draft_client else ''
        self._num_to_trim: int = 0
        self._result: list[int] | None = None

    def prefill(self, token_ids: list[int]) -> int | None:
        """Prefill draft cache (rank 0 only)."""
        if self._client is not None and hasattr(self._client, 'prefill'):
            return self._client.prefill(token_ids)
        return None

    def _broadcast_draft(self, token_id: int, num: int, trim: int) -> list[int]:
        """Fetch draft tokens on rank 0, broadcast to all TP ranks, return result."""
        if self._group.rank() == 0 and self._client is not None:
            result = self._client.fetch_draft_sync(token_id, num, trim=trim)
            padded = result[:num] + [0] * max(0, num - len(result))
            draft_array = mx.array([len(result)] + padded, dtype=mx.int32)
        else:
            draft_array = mx.zeros(num + 1, dtype=mx.int32)

        gathered = mx.distributed.all_gather(draft_array.reshape(1, -1), group=self._group)
        mx.eval(gathered)
        rank0_data = gathered[0].tolist()
        actual_len = rank0_data[0]
        return rank0_data[1:1 + actual_len] if actual_len > 0 else []

    def fetch_draft_sync(self, token_id: int, num_tokens: int = 0, trim: int = 0) -> list[int]:
        """Blocking draft with TP broadcast. Used by generate.py's _tcp_draft_fn."""
        num = num_tokens or self.num_draft_tokens
        return self._broadcast_draft(token_id, num, trim)

    def request_draft(self, token_id: int, num_tokens: int = 0) -> None:
        """Fetch draft tokens on rank 0, broadcast to all TP ranks."""
        num = num_tokens or self.num_draft_tokens
        trim = self._num_to_trim
        self._num_to_trim = 0
        self._result = self._broadcast_draft(token_id, num, trim)

    def reset_cache(self) -> None:
        if self._client is not None and hasattr(self._client, 'reset_cache'):
            self._client.reset_cache()

    def shutdown(self) -> None:
        if self._client is not None and hasattr(self._client, 'shutdown'):
            self._client.shutdown()


@dataclass
class Builder:
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    heartbeat: object | None = None
    heartbeat_timeout: object | None = None
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    draft_model: Model | None = None
    is_draft_node: bool = False
    use_speculative: bool = False

    def build(
        self,
    ) -> InferenceGenerator:
        assert self.model_id
        assert self.inference_model
        assert self.tokenizer

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)

        device_rank = 0 if self.group is None else self.group.rank()
        _use_sequential = bool(os.environ.get("EXO_NO_BATCH"))
        # If a draft model is configured, ALL TP ranks must use SequentialGenerator
        # with a TP-aware draft_fn that broadcasts draft tokens from rank 0.
        if not _use_sequential and self.use_speculative:
            _use_sequential = True
            if self.group is not None and self.group.size() > 1:
                # TP mode: wrap the DraftClient in a broadcast-based draft_fn
                # so all ranks call the model with the same draft tokens.
                self.draft_model = _TpDraftWrapper(self.draft_model, self.group)
            logger.info("using SequentialGenerator (TCP draft model requires speculative path)")
        if _use_sequential:
            logger.info("using SequentialGenerator")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                heartbeat=self.heartbeat,
                heartbeat_timeout=self.heartbeat_timeout,
                draft_model=self.draft_model,
                is_draft_node=self.is_draft_node,
            )
        logger.info("using BatchGenerator")
        return BatchGenerator(
            model=self.inference_model,
            tokenizer=self.tokenizer,
            group=self.group,
            tool_parser=tool_parser,
            kv_prefix_cache=kv_prefix_cache,
            model_id=self.model_id,
            device_rank=device_rank,
            cancel_receiver=self.cancel_receiver,
            event_sender=self.event_sender,
            heartbeat=self.heartbeat,
            heartbeat_timeout=self.heartbeat_timeout,
            is_draft_node=self.is_draft_node,
        )

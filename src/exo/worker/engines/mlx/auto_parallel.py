import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Any, Iterator, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    shard_inplace,
    shard_linear,
    sum_gradients,
)
from mlx_lm.models.base import (
    scaled_dot_product_attention,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3MLP
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.deepseek_v32 import DeepseekV32MLP
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe import MoE
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer, Glm4MoeLiteMLP
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel
from mlx_lm.models.gpt_oss import GptOssMoeModel
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.kimi_k25 import Model as KimiK25Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.minimax import MiniMaxAttention
from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.ministral3 import Model as Ministral3Model
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen3_5DecoderLayer
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel as Qwen3_5TextModelInner
from mlx_lm.models.qwen3_5 import SparseMoeBlock as Qwen3_5SparseMoeBlock
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from mlx_lm.models.step3p5 import Model as Step35Model
from mlx_lm.models.step3p5 import Step3p5MLP as Step35MLP
from mlx_lm.models.step3p5 import Step3p5Model as Step35InnerModel

from exo.shared.constants import EXO_TRACING_ENABLED
from exo.shared.logging import logger
from exo.shared.types.worker.shards import HybridShardMetadata, PipelineShardMetadata

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache

TimeoutCallback = Callable[[], None]
LayerLoadedCallback = Callable[[int, int], None]  # (layers_loaded, total_layers)


_pending_prefill_sends: list[tuple[mx.array, int, mx.distributed.Group]] = []


@dataclass
class PipelineTimings:
    """Accumulates per-operation timing for pipeline parallel layers.

    All times are in microseconds. Accumulates across multiple steps
    (tokens for decode, chunks for prefill) and should be reset between phases.
    """

    recv_eval_us: int = 0  # eval(x) before recv_like (waiting for prev compute to finish)
    recv_us: int = 0  # recv_like + eval(x) (waiting for data from previous rank)
    compute_us: int = 0  # layer forward + eval(output) (all GPU compute for this rank)
    send_us: int = 0  # send + depends + eval (dispatching to next rank)
    all_gather_us: int = 0  # all_gather + eval (decode only — gathering tokens)
    flush_sends_us: int = 0  # flush_prefill_sends (prefill only — async send dispatch)
    step_count: int = 0

    def reset(self) -> None:
        self.recv_eval_us = 0
        self.recv_us = 0
        self.compute_us = 0
        self.send_us = 0
        self.all_gather_us = 0
        self.flush_sends_us = 0
        self.step_count = 0

    def log_and_reset(self, label: str, rank: int) -> None:
        if self.step_count == 0:
            return
        total = (
            self.recv_eval_us
            + self.recv_us
            + self.compute_us
            + self.send_us
            + self.all_gather_us
            + self.flush_sends_us
        )
        avg = total / self.step_count
        parts = [
            f"recv_eval={self.recv_eval_us / 1000:.1f}ms",
            f"recv={self.recv_us / 1000:.1f}ms",
            f"compute={self.compute_us / 1000:.1f}ms",
            f"send={self.send_us / 1000:.1f}ms",
        ]
        if self.all_gather_us > 0:
            parts.append(f"all_gather={self.all_gather_us / 1000:.1f}ms")
        if self.flush_sends_us > 0:
            parts.append(f"flush_sends={self.flush_sends_us / 1000:.1f}ms")
        logger.info(
            f"[R{rank}] {label} pipeline ({self.step_count} steps, "
            f"{total / 1000:.1f}ms total, {avg / 1000:.2f}ms/step): {', '.join(parts)}"
        )
        self.reset()


_pipeline_timings = PipelineTimings()


def get_pipeline_timings() -> PipelineTimings:
    return _pipeline_timings


_dist_trace_enabled = False
_dist_trace_rank = -1


def enable_distributed_tracing(rank: int) -> None:
    """Monkey-patch mx.distributed ops to log every call with rank, op, shape, and group info."""
    global _dist_trace_enabled, _dist_trace_rank
    if _dist_trace_enabled:
        return
    _dist_trace_enabled = True
    _dist_trace_rank = rank

    _orig_all_sum = mx.distributed.all_sum
    _orig_all_gather = mx.distributed.all_gather
    _orig_send = mx.distributed.send
    _orig_recv = mx.distributed.recv
    _orig_recv_like = mx.distributed.recv_like

    _call_count: dict[str, int] = {"all_sum": 0, "all_gather": 0, "send": 0, "recv": 0, "recv_like": 0}

    def _traced_all_sum(x: mx.array, *, group: mx.distributed.Group | None = None, **kw: object) -> mx.array:
        n = _call_count["all_sum"]
        _call_count["all_sum"] = n + 1
        g_size = group.size() if group else "default"
        g_rank = group.rank() if group else "default"
        logger.info(f"[dist R{_dist_trace_rank}] all_sum #{n} shape={x.shape} group=(rank={g_rank},size={g_size})")
        return _orig_all_sum(x, group=group, **kw)  # type: ignore

    def _traced_all_gather(x: mx.array, *, group: mx.distributed.Group | None = None, **kw: object) -> mx.array:
        n = _call_count["all_gather"]
        _call_count["all_gather"] = n + 1
        g_size = group.size() if group else "default"
        g_rank = group.rank() if group else "default"
        logger.info(f"[dist R{_dist_trace_rank}] all_gather #{n} shape={x.shape} group=(rank={g_rank},size={g_size})")
        result = _orig_all_gather(x, group=group, **kw)  # type: ignore
        logger.info(f"[dist R{_dist_trace_rank}] all_gather #{n} returned shape={result.shape}")
        return result

    def _traced_send(x: mx.array, dst: int, *, group: mx.distributed.Group | None = None, **kw: object) -> mx.array:
        n = _call_count["send"]
        _call_count["send"] = n + 1
        g_size = group.size() if group else "default"
        logger.info(f"[dist R{_dist_trace_rank}] send #{n} dst={dst} shape={x.shape} group_size={g_size}")
        return _orig_send(x, dst, group=group, **kw)  # type: ignore

    def _traced_recv(x: mx.array, src: int, *, group: mx.distributed.Group | None = None, **kw: object) -> mx.array:
        n = _call_count["recv"]
        _call_count["recv"] = n + 1
        g_size = group.size() if group else "default"
        logger.info(f"[dist R{_dist_trace_rank}] recv #{n} src={src} shape={x.shape} group_size={g_size}")
        return _orig_recv(x, src, group=group, **kw)  # type: ignore

    def _traced_recv_like(x: mx.array, src: int, *, group: mx.distributed.Group | None = None, **kw: object) -> mx.array:
        n = _call_count["recv_like"]
        _call_count["recv_like"] = n + 1
        g_size = group.size() if group else "default"
        logger.info(f"[dist R{_dist_trace_rank}] recv_like #{n} src={src} shape={x.shape} group_size={g_size}")
        return _orig_recv_like(x, src, group=group, **kw)  # type: ignore

    mx.distributed.all_sum = _traced_all_sum  # type: ignore
    mx.distributed.all_gather = _traced_all_gather  # type: ignore
    mx.distributed.send = _traced_send  # type: ignore
    mx.distributed.recv = _traced_recv  # type: ignore
    mx.distributed.recv_like = _traced_recv_like  # type: ignore
    logger.info(f"[dist R{rank}] distributed op tracing enabled")


def flush_prefill_sends() -> None:
    if EXO_TRACING_ENABLED:
        t0 = time.perf_counter()
    for output, dst, group in _pending_prefill_sends:
        sent = mx.distributed.send(output, dst, group=group)
        mx.async_eval(sent)
    _pending_prefill_sends.clear()
    if EXO_TRACING_ENABLED:
        _pipeline_timings.flush_sends_us += int((time.perf_counter() - t0) * 1_000_000)


def clear_prefill_sends() -> None:
    # Discard pending sends (e.g. on cancellation).
    _pending_prefill_sends.clear()


def eval_with_timeout(
    mlx_item: Any,  # pyright: ignore[reportAny]
    timeout_seconds: float = 60.0,
    on_timeout: TimeoutCallback | None = None,
) -> None:
    """Evaluate MLX item with a hard timeout.

    If on_timeout callback is provided, it will be called before terminating
    the process. This allows the runner to send a failure event before exit.
    """
    completed = threading.Event()

    def watchdog() -> None:
        if not completed.wait(timeout=timeout_seconds):
            logger.error(
                f"mlx_item evaluation timed out after {timeout_seconds:.0f}s. "
                "This may indicate an issue with FAST_SYNCH and tensor parallel sharding. "
                "Terminating process."
            )
            if on_timeout is not None:
                on_timeout()
            os._exit(1)

    watchdog_thread = threading.Thread(target=watchdog, daemon=True)
    watchdog_thread.start()

    try:
        mx.eval(mlx_item)  # pyright: ignore[reportAny]
    finally:
        completed.set()


DISTRIBUTED_OP_TIMEOUT = 60.0


class DistributedOpWatchdog:
    """Kills the runner process if any distributed mx.eval exceeds the timeout.

    Uses a single daemon thread that polls every 2s.  Pipeline layers call
    arm()/disarm() (via ``guarded()``) around blocking ``mx.eval()`` calls
    that depend on distributed ops (recv, send, all_gather, all_sum).

    When the deadline is exceeded the watchdog logs an error and calls
    ``os._exit(1)`` to hard-terminate the runner process.  The supervisor
    will then detect the death and propagate ``RunnerFailed``.
    """

    def __init__(self, timeout: float = DISTRIBUTED_OP_TIMEOUT):
        self._timeout = timeout
        self._deadline: float = float("inf")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def arm(self) -> None:
        self._deadline = time.monotonic() + self._timeout

    def disarm(self) -> None:
        self._deadline = float("inf")

    @contextmanager
    def guarded(self) -> Iterator[None]:
        self.arm()
        try:
            yield
        finally:
            self.disarm()

    def _run(self) -> None:
        while True:
            time.sleep(2.0)
            if time.monotonic() > self._deadline:
                logger.error(
                    f"Distributed operation timed out after {self._timeout:.0f}s — "
                    "peer process likely dead. Terminating runner to allow recovery."
                )
                os._exit(1)


_distributed_watchdog: DistributedOpWatchdog | None = None


def init_distributed_watchdog(timeout: float = DISTRIBUTED_OP_TIMEOUT) -> None:
    """Initialise the per-process distributed-op watchdog (idempotent)."""
    global _distributed_watchdog
    if _distributed_watchdog is None:
        _distributed_watchdog = DistributedOpWatchdog(timeout)


def get_distributed_watchdog() -> DistributedOpWatchdog | None:
    return _distributed_watchdog


def _guarded_eval(*args: Any) -> None:  # pyright: ignore[reportAny]
    """``mx.eval`` protected by the distributed-operation watchdog.

    If the watchdog has been initialised, arms/disarms the deadline around the
    eval.  Otherwise falls back to a plain ``mx.eval``.
    """
    wd = _distributed_watchdog
    if wd is not None:
        wd.arm()
        try:
            mx.eval(*args)  # pyright: ignore[reportAny]
        finally:
            wd.disarm()
    else:
        mx.eval(*args)  # pyright: ignore[reportAny]


class _LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class CustomMlxLayer(nn.Module):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _LayerCallable):
        super().__init__()
        dict.__setitem__(self, "_original_layer", original_layer)  # pyright: ignore[reportUnknownMemberType]

    @property
    def original_layer(self) -> _LayerCallable:
        return cast(_LayerCallable, self["_original_layer"])

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = cast(_LayerCallable, self["_original_layer"])
                return getattr(original_layer, name)


class TracingLayerWrapper(CustomMlxLayer):
    """Wraps a layer to log forward-pass boundaries for deadlock diagnosis.

    NOTE: Per-layer timing only measures graph-build time (MLX is lazy).
    Actual GPU time is measured by the per-decode-step timer in generate.py.
    Only the first and last layer log, to show forward-pass start/end.
    """

    # Shared across all layers on this rank to track forward-pass boundaries
    _num_layers: int = 0

    def __init__(self, original_layer: _LayerCallable, rank: int, layer_idx: int):
        super().__init__(original_layer)
        self._trace_rank = rank
        self._trace_idx = layer_idx

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self._trace_idx == 0:
            logger.info(f"[trace R{self._trace_rank}] forward pass START x.shape={x.shape}")
        result = self.original_layer(x, *args, **kwargs)
        if self._trace_idx == TracingLayerWrapper._num_layers - 1:
            logger.info(f"[trace R{self._trace_rank}] forward pass END (layer {self._trace_idx})")
        return result


_TP_EVAL_INTERVAL: int = int(os.environ.get("EXO_TP_EVAL_INTERVAL", "8"))
_USE_EXPERT_PARALLEL: bool = os.environ.get("EXO_EXPERT_PARALLEL", "0") == "1"
"""Force a guarded eval every N transformer layers in pure TP mode.

Without this, the entire model forward pass (62 layers for MiniMax) builds a
single lazy graph with ~186 collective operations.  Evaluating all of them in
one ``mx.eval`` can deadlock the JACCL RDMA layer when both ranks block
waiting for buffers that neither can release.  Inserting periodic evals limits
the outstanding collectives and lets each rank make progress.
"""


class DistributedEvalBarrier(CustomMlxLayer):
    """Forces ``_guarded_eval`` after a transformer block in pure TP mode.

    Inserted every ``_TP_EVAL_INTERVAL`` layers by the sharding strategies to
    break the lazy graph into smaller chunks, preventing RDMA deadlocks from
    accumulated collective operations.

    Only active during prefill.  During decode (1 token) the graph is small
    and the barriers add unnecessary synchronization that prevents GPU
    pipelining of compute and RDMA communication.
    """

    is_prefill: bool = True

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        result = self.original_layer(x, *args, **kwargs)
        if self.is_prefill:
            _guarded_eval(result)
        return result


class PipelineFirstLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        group: mx.distributed.Group,
        recv_from: int | None = None,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.recv_from: int = recv_from if recv_from is not None else (r - 1)
        self.group = group
        self.is_prefill: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            if EXO_TRACING_ENABLED:
                t0 = time.perf_counter()
            mx.eval(x)
            if EXO_TRACING_ENABLED:
                t1 = time.perf_counter()
            x = mx.distributed.recv_like(x, self.recv_from, group=self.group)
            _guarded_eval(x)
            if EXO_TRACING_ENABLED:
                _pipeline_timings.recv_eval_us += int((t1 - t0) * 1_000_000)
                _pipeline_timings.recv_us += int((time.perf_counter() - t1) * 1_000_000)
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group
        self.original_layer_signature = signature(self.original_layer.__call__)
        self.is_prefill: bool = False
        self.queue_sends: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        cache = self.original_layer_signature.bind_partial(
            x, *args, **kwargs
        ).arguments.get("cache", None)

        if EXO_TRACING_ENABLED:
            t0 = time.perf_counter()
        output: mx.array = self.original_layer(x, *args, **kwargs)

        # Eval layer output to materialize it before send — this splits the graph
        # so the send is isolated and the receiving rank's recv can complete.
        _guarded_eval(output)
        if EXO_TRACING_ENABLED:
            _pipeline_timings.compute_us += int(
                (time.perf_counter() - t0) * 1_000_000
            )

        if self.r != self.s - 1:
            if EXO_TRACING_ENABLED:
                t_send = time.perf_counter()
            if self.queue_sends:
                _pending_prefill_sends.append(
                    (output, (self.r + 1) % self.s, self.group)
                )
            else:
                output = mx.distributed.send(
                    output, (self.r + 1) % self.s, group=self.group
                )
            if cache is not None:
                _cache = cache[0] if hasattr(cache, "caches") else cache  # type: ignore
                if hasattr(_cache, "keys"):  # pyright: ignore[reportAny]
                    _cache.keys = mx.depends(_cache.keys, output)  # type: ignore
            _guarded_eval(output)
            if cache is not None and hasattr(_cache, "keys"):  # type: ignore
                _guarded_eval(_cache.keys)  # type: ignore
            if EXO_TRACING_ENABLED:
                _pipeline_timings.send_us += int(
                    (time.perf_counter() - t_send) * 1_000_000
                )

        if not self.is_prefill:
            if EXO_TRACING_ENABLED:
                t_ag = time.perf_counter()
            output = mx.distributed.all_gather(output, group=self.group)[
                -output.shape[0] :
            ]
            _guarded_eval(output)
            if EXO_TRACING_ENABLED:
                _pipeline_timings.all_gather_us += int(
                    (time.perf_counter() - t_ag) * 1_000_000
                )

        if EXO_TRACING_ENABLED:
            _pipeline_timings.step_count += 1
        return output


class HybridPipelineLastLayer(CustomMlxLayer):
    """Last layer of the TP-master in hybrid mode — sends output to PP tail."""

    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group,
        send_to: int,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group
        self.send_to: int = send_to
        self.original_layer_signature = signature(self.original_layer.__call__)
        self.is_prefill: bool = False
        self.queue_sends: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        cache = self.original_layer_signature.bind_partial(
            x, *args, **kwargs
        ).arguments.get("cache", None)

        if EXO_TRACING_ENABLED:
            t0 = time.perf_counter()
        output: mx.array = self.original_layer(x, *args, **kwargs)
        _guarded_eval(output)
        if EXO_TRACING_ENABLED:
            _pipeline_timings.compute_us += int(
                (time.perf_counter() - t0) * 1_000_000
            )

        if EXO_TRACING_ENABLED:
            t_send = time.perf_counter()
        if self.queue_sends:
            _pending_prefill_sends.append(
                (output, self.send_to, self.group)
            )
        else:
            output = mx.distributed.send(
                output, self.send_to, group=self.group
            )
        if cache is not None:
            _cache = cache[0] if hasattr(cache, "caches") else cache  # type: ignore
            if hasattr(_cache, "keys"):  # pyright: ignore[reportAny]
                _cache.keys = mx.depends(_cache.keys, output)  # type: ignore
        _guarded_eval(output)
        if cache is not None and hasattr(_cache, "keys"):  # type: ignore
            _guarded_eval(_cache.keys)  # type: ignore
        if EXO_TRACING_ENABLED:
            _pipeline_timings.send_us += int(
                (time.perf_counter() - t_send) * 1_000_000
            )

        # Decode: all ranks must participate in all_gather for token sync
        if not self.is_prefill:
            if EXO_TRACING_ENABLED:
                t_ag = time.perf_counter()
            output = mx.distributed.all_gather(output, group=self.group)[
                -output.shape[0] :
            ]
            _guarded_eval(output)
            if EXO_TRACING_ENABLED:
                _pipeline_timings.all_gather_us += int(
                    (time.perf_counter() - t_ag) * 1_000_000
                )

        if EXO_TRACING_ENABLED:
            _pipeline_timings.step_count += 1
        return output


class HybridPipelinePassthroughLayer(CustomMlxLayer):
    """Last layer of a TP non-master node — no send, just all_gather for decode sync."""

    def __init__(
        self,
        original_layer: _LayerCallable,
        group: mx.distributed.Group,
    ):
        super().__init__(original_layer)
        self.group = group
        self.is_prefill: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if EXO_TRACING_ENABLED:
            t0 = time.perf_counter()
        output: mx.array = self.original_layer(x, *args, **kwargs)
        _guarded_eval(output)
        if EXO_TRACING_ENABLED:
            _pipeline_timings.compute_us += int(
                (time.perf_counter() - t0) * 1_000_000
            )

        if not self.is_prefill:
            if EXO_TRACING_ENABLED:
                t_ag = time.perf_counter()
            output = mx.distributed.all_gather(output, group=self.group)[
                -output.shape[0] :
            ]
            _guarded_eval(output)
            if EXO_TRACING_ENABLED:
                _pipeline_timings.all_gather_us += int(
                    (time.perf_counter() - t_ag) * 1_000_000
                )

        if EXO_TRACING_ENABLED:
            _pipeline_timings.step_count += 1
        return output


def _unwrap_tracing(layer: Any) -> Any:  # pyright: ignore[reportAny]
    """Unwrap TracingLayerWrapper to get the actual pipeline layer underneath."""
    if isinstance(layer, TracingLayerWrapper):
        return layer.original_layer
    return layer


def set_pipeline_prefill(model: nn.Module, is_prefill: bool) -> None:
    for layer in model.layers:  # type: ignore
        inner = _unwrap_tracing(layer)
        if isinstance(inner, (PipelineFirstLayer, PipelineLastLayer, HybridPipelineLastLayer, HybridPipelinePassthroughLayer, DistributedEvalBarrier)):
            inner.is_prefill = is_prefill


def set_pipeline_queue_sends(model: nn.Module, queue_sends: bool) -> None:
    for layer in model.layers:  # type: ignore
        inner = _unwrap_tracing(layer)
        if isinstance(inner, (PipelineLastLayer, HybridPipelineLastLayer)):
            inner.queue_sends = queue_sends


def get_inner_model(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "transformer", None)
    if isinstance(inner, nn.Module):
        return inner

    inner = getattr(model, "language_model", None)
    if isinstance(inner, nn.Module):
        inner_inner = getattr(inner, "model", None)
        if isinstance(inner_inner, nn.Module):
            return inner_inner

    raise ValueError("Model must either have a 'model' or 'transformer' attribute")


def get_layers(inner_model_instance: nn.Module) -> list[_LayerCallable]:
    # Handle both model.layers and model.h cases
    layers: list[_LayerCallable]
    if hasattr(inner_model_instance, "layers"):
        layers = cast(list[_LayerCallable], inner_model_instance.layers)
    elif hasattr(inner_model_instance, "h"):
        layers = cast(list[_LayerCallable], inner_model_instance.h)
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")

    return layers


def _patch_qwen35_cache(
    model: Qwen3_5TextModel,
    fa_idx: int,
    has_full_attn: bool,
    ssm_idx: int,
    has_linear: bool,
) -> None:
    # Hacks to make make_mask happy.
    original = model.make_cache

    def patched() -> list[ArraysCache | KVCache]:
        cache: list[ArraysCache | KVCache] = original()
        if not has_full_attn:
            entry = cache[fa_idx]
            orig_make_mask = entry.make_mask
            entry.make_mask = lambda n, **_kw: orig_make_mask(n)  # type: ignore
        if not has_linear:
            orig_ssm_make_mask = cache[ssm_idx].make_mask
            cache[ssm_idx].make_mask = (  # type: ignore
                lambda n, **kw: orig_ssm_make_mask(n, **kw) if kw else None  # type: ignore
            )
        return cache

    model.make_cache = patched


def pipeline_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: PipelineShardMetadata,
    on_layer_loaded: LayerLoadedCallback | None,
) -> nn.Module:
    """
    Automatically parallelize a model across multiple devices.
    Args:
    model: The model to parallelize (must have a 'layers' or 'h' property)
    model_shard_meta: The metadata for the model shard
    Returns:
    The parallelized model
    """
    inner_model_instance: nn.Module = get_inner_model(model)

    layers = get_layers(inner_model_instance)

    start_layer, end_layer = model_shard_meta.start_layer, model_shard_meta.end_layer
    device_rank, world_size = model_shard_meta.device_rank, model_shard_meta.world_size

    layers = layers[start_layer:end_layer]
    total = len(layers)
    for i, layer in enumerate(layers):
        mx.eval(layer)  # type: ignore
        if on_layer_loaded is not None:
            on_layer_loaded(i, total)

    layers[0] = PipelineFirstLayer(layers[0], device_rank, group=group)
    layers[-1] = PipelineLastLayer(
        layers[-1],
        device_rank,
        world_size,
        group=group,
    )

    if isinstance(inner_model_instance, GptOssMoeModel):
        inner_model_instance.layer_types = inner_model_instance.layer_types[  # type: ignore
            start_layer:end_layer
        ]
        # We can assume the model has at least one layer thanks to placement.
        # If a layer type doesn't exist, we can set it to 0.
        inner_model_instance.swa_idx = (
            0
            if "sliding_attention" not in inner_model_instance.layer_types  # type: ignore
            else inner_model_instance.layer_types.index(  # type: ignore
                "sliding_attention"
            )
        )
        inner_model_instance.ga_idx = (
            0
            if "full_attention" not in inner_model_instance.layer_types  # type: ignore
            else inner_model_instance.layer_types.index(  # type: ignore
                "full_attention"
            )
        )

    if isinstance(inner_model_instance, Step35InnerModel):
        inner_model_instance.num_layers = len(layers)
        sliding_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_sliding", False)
        ]
        full_layers = [
            i
            for i, layer in enumerate(layers)
            if not getattr(layer, "is_sliding", True)
        ]
        inner_model_instance._swa_idx = 0 if not sliding_layers else sliding_layers[0]
        inner_model_instance._full_idx = 0 if not full_layers else full_layers[0]

    if isinstance(inner_model_instance, Qwen3_5TextModelInner):
        full_attn_layers = [
            i for i, layer in enumerate(layers) if not getattr(layer, "is_linear", True)
        ]
        linear_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_linear", False)
        ]
        inner_model_instance.fa_idx = full_attn_layers[0] if full_attn_layers else 0
        inner_model_instance.ssm_idx = linear_layers[0] if linear_layers else 0
        if not full_attn_layers or not linear_layers:
            _patch_qwen35_cache(
                cast(Qwen3_5TextModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=bool(full_attn_layers),
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=bool(linear_layers),
            )

    _set_layers(model, layers)

    assert isinstance(layers, list), (
        "Expected a list of layers after auto-parallel initialisation"
    )

    return patch_pipeline_model(model, group)


def hybrid_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: HybridShardMetadata,
    timeout_seconds: float,
    on_timeout: TimeoutCallback | None,
    on_layer_loaded: LayerLoadedCallback | None,
) -> nn.Module:
    """Hybrid tensor + pipeline parallelism.

    TP nodes (tp_rank >= 0) share the same layer range and shard weights via
    all-reduce within a TP sub-group created by group.split().
    The PP tail (tp_rank == -1) owns a disjoint layer range and communicates
    with the TP-master via send/recv.
    """
    init_distributed_watchdog()

    is_tp_node = model_shard_meta.tp_rank >= 0
    tp_color = 0 if is_tp_node else 1

    logger.info(
        f"[hybrid] rank={model_shard_meta.device_rank} tp_rank={model_shard_meta.tp_rank} "
        f"pp_rank={model_shard_meta.pp_rank} is_tp={is_tp_node} color={tp_color} "
        f"layers=[{model_shard_meta.start_layer},{model_shard_meta.end_layer}) "
        f"send_to={model_shard_meta.pipeline_send_to} recv_from={model_shard_meta.pipeline_recv_from}"
    )
    logger.info(f"[hybrid] rank={model_shard_meta.device_rank} calling group.split(color={tp_color})")
    tp_group = group.split(tp_color)
    logger.info(
        f"[hybrid] rank={model_shard_meta.device_rank} split complete: "
        f"tp_group.size()={tp_group.size()} tp_group.rank()={tp_group.rank()}"
    )

    # Apply tensor parallelism to TP nodes
    if is_tp_node and tp_group.size() > 1:
        logger.info(f"[hybrid] rank={model_shard_meta.device_rank} applying tensor_auto_parallel")
        model = tensor_auto_parallel(
            model, tp_group, timeout_seconds, on_timeout, on_layer_loaded
        )
        logger.info(f"[hybrid] rank={model_shard_meta.device_rank} tensor_auto_parallel complete")

    # Draft node: no primary model layers to process. The draft model is
    # loaded separately in load_mlx_items. Just return the model as-is.
    # BUT: must still call group.split() because it's a collective operation
    # that ALL nodes in the group must participate in.
    if model_shard_meta.draft_model_id is not None:
        logger.info(
            f"[hybrid] rank={model_shard_meta.device_rank} is draft node "
            f"(model={model_shard_meta.draft_model_id}), skipping layer sharding"
        )
        # Participate in group.split() (collective — all nodes must call it)
        tp_color = 1  # not in TP group
        _draft_subgroup = group.split(tp_color)
        logger.info(f"[hybrid] draft node split complete (subgroup size={_draft_subgroup.size()})")
        mx.eval(model)
        return model

    inner_model_instance = get_inner_model(model)
    all_layers = get_layers(inner_model_instance)

    start_layer = model_shard_meta.start_layer
    end_layer = model_shard_meta.end_layer
    layers = all_layers[start_layer:end_layer]

    # PP tail needs to eval its own layers (TP nodes already eval'd in tensor_auto_parallel)
    if not is_tp_node:
        full_model_layers = len(all_layers)
        for i, layer in enumerate(layers):
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(start_layer + i, full_model_layers)

    # Wrap pipeline boundaries
    # PP tail's first layer receives from TP-master
    if model_shard_meta.pipeline_recv_from is not None:
        layers[0] = PipelineFirstLayer(
            layers[0],
            model_shard_meta.device_rank,
            group=group,
            recv_from=model_shard_meta.pipeline_recv_from,
        )

    # Skip ALL pipeline wrapping when draft mode is active.
    # Draft node doesn't participate in all_gather/send/recv during model forward.
    # Draft token exchange is handled separately by RDMADraftClient.
    _has_real_pp_tail = (model_shard_meta.pipeline_send_to is not None
                         or model_shard_meta.pipeline_recv_from is not None)

    # TP-master's last layer sends to PP tail + participates in decode all_gather
    if model_shard_meta.pipeline_send_to is not None and _has_real_pp_tail:
        layers[-1] = HybridPipelineLastLayer(
            layers[-1],
            model_shard_meta.device_rank,
            model_shard_meta.world_size,
            group=group,
            send_to=model_shard_meta.pipeline_send_to,
        )
    # TP non-master: no send, but must participate in decode all_gather
    elif is_tp_node and _has_real_pp_tail:
        layers[-1] = HybridPipelinePassthroughLayer(
            layers[-1],
            group=group,
        )

    # PP tail's last layer is the final pipeline stage — needs all_gather for decode
    if model_shard_meta.pp_rank == model_shard_meta.pp_size - 1 and _has_real_pp_tail:
        # Only wrap if not already wrapped by one of the above
        if not isinstance(layers[-1], (HybridPipelineLastLayer, HybridPipelinePassthroughLayer)):
            layers[-1] = PipelineLastLayer(
                layers[-1],
                model_shard_meta.device_rank,
                model_shard_meta.world_size,
                group=group,
            )

    logger.info(
        f"[hybrid] rank={model_shard_meta.device_rank} setup complete: "
        f"{len(layers)} layers [{start_layer}..{end_layer}), "
        f"first={type(layers[0]).__name__}, last={type(layers[-1]).__name__}"
    )

    _set_layers(model, layers)
    return patch_pipeline_model(model, group)


def patch_pipeline_model[T](model: T, group: mx.distributed.Group) -> T:
    # Patch __call__ on the model's class
    cls = model.__class__
    original_call = cls.__call__  # type :ignore
    call_signature = signature(original_call)  # type :ignore

    def patched_call(
        self: T,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        logits: mx.array = original_call(self, *args, **kwargs)  # type: ignore
        cache = call_signature.bind_partial(self, *args, **kwargs).arguments.get(
            "cache", None
        )

        # Add dependency to last cache entry to ensure distributed ops are evaluated
        if cache is not None:
            last = cache[-1]  # type: ignore
            dep_cache = last[0] if hasattr(last, "caches") else last  # type: ignore
            if hasattr(dep_cache, "keys") and dep_cache.keys is not None:  # type: ignore
                dep_cache.keys = mx.depends(dep_cache.keys, logits)  # type: ignore

        return logits

    cls.__call__ = patched_call
    return model


def patch_tensor_model[T](model: T) -> T:
    """Patch model's __call__ to ensure distributed ops sync during inference.

    When EXO_COMPILE_DECODE=1, the patched_call is still installed but the
    mx.depends cache ordering is skipped during compiled decode steps.
    The compile flag is checked per-call via a model attribute so that
    uncompiled paths (prefill, warmup) still get the dependency.
    """
    cls = model.__class__
    original_call = cls.__call__
    call_signature = signature(original_call)

    def patched_call(
        self: T,
        *args: object,
        **kwargs: object,
    ) -> mx.array:
        logits: mx.array = original_call(self, *args, **kwargs)  # pyright: ignore[reportAny]

        # Skip cache dependency during decode (1 token) — the lazy graph is
        # tiny and MLX's natural ordering handles it.  Only needed during
        # prefill where large graphs can reorder distributed ops.
        # Also skip for compiled decode (mx.compile can't trace mx.depends).
        if getattr(self, "_compiled_decode_active", False):
            return logits

        # Check if this is a decode step (single token input)
        if len(args) > 0 and hasattr(args[0], "shape"):
            seq_len = args[0].shape[1] if len(args[0].shape) > 1 else args[0].shape[0]  # type: ignore
            if seq_len <= 1:
                return logits

        cache = call_signature.bind_partial(self, *args, **kwargs).arguments.get(
            "cache", None
        )

        # Add dependency to last cache entry to ensure distributed ops are evaluated
        if cache is not None and len(cache) > 0:  # pyright: ignore[reportAny]
            last = cache[-1]  # pyright: ignore[reportAny]
            dep_cache = last[0] if hasattr(last, "caches") else last  # pyright: ignore[reportAny]
            dep_cache.keys = mx.depends(dep_cache.keys, logits)  # pyright: ignore[reportAny,reportUnknownMemberType]

        return logits

    cls.__call__ = patched_call
    return model


def tensor_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    timeout_seconds: float,
    on_timeout: TimeoutCallback | None,
    on_layer_loaded: LayerLoadedCallback | None,
) -> nn.Module:
    init_distributed_watchdog()

    all_to_sharded_linear = partial(
        shard_linear,
        sharding="all-to-sharded",
        group=group,
    )
    sharded_to_all_linear = partial(
        shard_linear,
        sharding="sharded-to-all",
        group=group,
    )

    segments: int = 1

    def _all_to_sharded(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - all to sharded")
            return weight.ndim - 1, segments
        return max(weight.ndim - 2, 0), segments

    all_to_sharded_linear_in_place = partial(
        shard_inplace,
        sharding=_all_to_sharded,  # type: ignore
        group=group,
    )

    n = group.size()

    def _sharded_to_all(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - sharded to all")
            weight /= n
            return None
        return -1, segments

    sharded_to_all_linear_in_place = partial(
        shard_inplace,
        sharding=_sharded_to_all,  # type: ignore
        group=group,
    )

    if isinstance(model, (LlamaModel, Ministral3Model)):
        tensor_parallel_sharding_strategy = LlamaShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, MiniMaxModel):
        tensor_parallel_sharding_strategy = MiniMaxShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GLM4MoeLiteModel):
        tensor_parallel_sharding_strategy = GLM4MoeLiteShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Glm4MoeModel):
        tensor_parallel_sharding_strategy = Glm4MoeShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(
        model, (Qwen3MoeModel, Qwen3NextModel, Qwen3_5TextModel, Qwen3_5MoeModel)
    ):
        tensor_parallel_sharding_strategy = QwenShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GptOssModel):
        tensor_parallel_sharding_strategy = GptOssShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Step35Model):
        tensor_parallel_sharding_strategy = Step35ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    model = tensor_parallel_sharding_strategy.shard_model(
        model, timeout_seconds, on_timeout, on_layer_loaded
    )

    # Insert eval barriers every N layers to prevent RDMA deadlocks in pure TP.
    # Also wrap the LAST layer unconditionally so no tail chunk falls through
    # to the unguarded mx.eval in generate_step.
    if _TP_EVAL_INTERVAL > 0:
        inner = get_inner_model(model)
        if hasattr(inner, "layers"):
            n = len(inner.layers)
            for i in range(n):
                if (i + 1) % _TP_EVAL_INTERVAL == 0 or i == n - 1:
                    inner.layers[i] = DistributedEvalBarrier(inner.layers[i])  # type: ignore

    return patch_tensor_model(model)


class TensorParallelShardingStrategy(ABC):
    def __init__(
        self,
        group: mx.distributed.Group,
        all_to_sharded_linear: Callable[..., nn.Linear],
        sharded_to_all_linear: Callable[..., nn.Linear],
        all_to_sharded_linear_in_place: Callable[..., None],
        sharded_to_all_linear_in_place: Callable[..., None],
    ):
        self.all_to_sharded_linear = all_to_sharded_linear
        self.sharded_to_all_linear = sharded_to_all_linear
        self.all_to_sharded_linear_in_place = all_to_sharded_linear_in_place
        self.sharded_to_all_linear_in_place = sharded_to_all_linear_in_place
        self.group = group
        self.N = group.size()

    @abstractmethod
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module: ...


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(LlamaModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            # Force load weights before sharding to avoid FAST_SYNCH deadlock
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            if layer.self_attn.n_kv_heads is not None:
                layer.self_attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


def _set_layers(model: nn.Module, layers: list[_LayerCallable]) -> None:
    inner_model_instance = get_inner_model(model)
    if hasattr(inner_model_instance, "layers"):
        inner_model_instance.layers = layers

        # Update DeepSeek V3 specific parameters when layers are shrunk
        if isinstance(
            model, (DeepseekV3Model, DeepseekV32Model, Glm4MoeModel, KimiK25Model)
        ) and hasattr(inner_model_instance, "num_layers"):
            logger.info(
                f"Setting num_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.start_idx = 0
            inner_model_instance.end_idx = len(layers)
            inner_model_instance.num_layers = len(layers)
        elif isinstance(model, Qwen3MoeModel):
            logger.info(
                f"Setting num_hidden_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.num_hidden_layers = len(layers)
    elif hasattr(inner_model_instance, "h"):
        inner_model_instance.h = layers
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")


class DeepSeekShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(DeepseekV3Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)

            # Shard the self attention
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )

            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            # Logic from upstream mlx
            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            # Shard the MLP
            if isinstance(layer.mlp, (DeepseekV3MLP, DeepseekV32MLP)):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            # Shard the MoE.
            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)

        return model


class ShardedMoE(CustomMlxLayer):
    """Wraps any MoE layer with distributed sum_gradients / all_sum."""

    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        # sum_gradients is identity during inference (no backward pass).
        # Skip it to avoid unnecessary wrapper overhead.
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class ExpertParallelMoE(nn.Module):
    """Expert Parallelism: each node owns a subset of full experts.

    Instead of sharding each expert's weights across nodes (TP), this pins
    whole experts to specific nodes.  Local experts compute with contiguous
    memory reads.  Remote experts are handled by exchanging activations via
    point-to-point RDMA (all_gather + scatter) instead of all_sum.

    Benefits over ShardedMoE:
    - Contiguous expert weight reads (no scatter across hidden dim)
    - Eliminates the per-layer MoE all_sum barrier
    """

    def __init__(
        self,
        moe_block: nn.Module,
        group: mx.distributed.Group,
    ):
        super().__init__()
        # Store the full MoE block (gate + switch_mlp)
        dict.__setitem__(self, "_moe_block", moe_block)
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()

        # Total experts and per-node split
        self.num_experts: int = moe_block.num_experts  # type: ignore
        self.experts_per_node = self.num_experts // self.world_size
        self.local_start = self.rank * self.experts_per_node
        self.local_end = self.local_start + self.experts_per_node

    @property
    def moe_block(self) -> nn.Module:
        return self["_moe_block"]  # type: ignore

    def _split_expert_weights(self) -> None:
        """Trim switch_mlp weights to only this node's experts.

        Called once after construction to free memory for remote experts.
        Immediately evaluates and makes contiguous to avoid lazy view
        mismatches between TP ranks during eval_with_timeout.
        """
        s = self.local_start
        e = self.local_end
        mlp = self.moe_block.switch_mlp  # type: ignore

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, proj_name)
            # QuantizedSwitchLinear stores weight, scales, biases
            if hasattr(proj, "scales"):
                proj.weight = mx.contiguous(proj.weight[s:e])
                proj.scales = mx.contiguous(proj.scales[s:e])
                if proj.biases is not None:
                    proj.biases = mx.contiguous(proj.biases[s:e])
            else:
                proj.weight = mx.contiguous(proj.weight[s:e])
            if "bias" in proj:
                proj.bias = mx.contiguous(proj.bias[s:e])

        # Force evaluation so the trimmed weights are materialized before
        # eval_with_timeout runs on the layer.  Without this, different
        # ranks have different lazy graphs and FAST_SYNCH times out.
        mx.eval(mlp.parameters())

    _ep_logged: bool = False

    def __call__(self, x: mx.array) -> mx.array:
        moe = self.moe_block

        # 1. Gate computation — identical on all nodes (full gate weights).
        gates = moe.gate(x)  # type: ignore
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = moe.top_k  # type: ignore
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if moe.norm_topk_prob:  # type: ignore
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        if not ExpertParallelMoE._ep_logged:
            ExpertParallelMoE._ep_logged = True
            logger.info(
                f"EP forward: rank={self.rank} local=[{self.local_start},{self.local_end}) "
                f"inds shape={inds.shape} "
                f"gate_proj.weight.shape={moe.switch_mlp.gate_proj.weight.shape}"  # type: ignore
            )

        # 2. Remap and mask
        is_local = (inds >= self.local_start) & (inds < self.local_end)
        local_inds = mx.where(is_local, inds - self.local_start, 0)
        local_scores = mx.where(is_local, scores, 0.0)

        # 3. Compute via switch_mlp with remapped indices.
        # Remote slots use index 0 with score 0. gather_qmm still loads
        # expert-0 weights for those slots, but the zeroed score eliminates
        # their contribution. This wastes ~50% of expert compute but is
        # correct and avoids Metal kernel changes.
        y = moe.switch_mlp(x, local_inds)  # type: ignore

        # 4. Weighted sum — only local experts contribute.
        local_result = (y * local_scores[..., None]).sum(axis=-2)

        # 5. All-reduce to combine local results from all nodes.
        result = mx.distributed.all_sum(local_result, group=self.group)

        return result


class GLM4MoeLiteShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(GLM4MoeLiteModel, model)
        total = len(model.layers)  # type: ignore
        for i, layer in enumerate(model.layers):  # type: ignore
            layer = cast(Glm4MoeLiteDecoderLayer, layer)
            eval_with_timeout(
                layer.parameters(),
                timeout_seconds / total,
                on_timeout,
            )
            if layer.self_attn.q_lora_rank is None:  # type: ignore
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )

            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            # Logic from upstream mlx
            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, Glm4MoeLiteMLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group  # type: ignore
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)

        return model


class WrappedMiniMaxAttention(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable, group: mx.distributed.Group):
        super().__init__(layer)
        self.group = group

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: "Cache | None" = None,
    ) -> mx.array:
        batch_dim, seq_dim, _ = x.shape

        self._original_layer = cast(MiniMaxAttention, self.original_layer)  # type: ignore

        queries: mx.array = self._original_layer.q_proj(x)
        keys: mx.array = self._original_layer.k_proj(x)
        values: mx.array = self._original_layer.v_proj(x)

        if getattr(self, "use_qk_norm", False):
            q_dim = queries.shape[-1]
            k_dim = keys.shape[-1]
            n = self.group.size()

            qk = mx.concatenate(
                [queries, keys], axis=-1
            )  # (batch_dim, seq_dim, q_dim + k_dim)
            qk = mx.distributed.all_gather(
                qk, group=self.group
            )  # (n*batch_dim, seq_dim, q_dim + k_dim)

            qk = qk.reshape(n, batch_dim, seq_dim, q_dim + k_dim).transpose(1, 2, 0, 3)
            queries = qk[..., :q_dim].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * q_dim)
            keys = qk[..., q_dim:].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * k_dim)

            queries = self._original_layer.q_norm(queries)
            keys = self._original_layer.k_norm(keys)

            # Split back and take this rank's portion
            queries = mx.split(queries, n, axis=-1)[self.group.rank()]
            keys = mx.split(keys, n, axis=-1)[self.group.rank()]

        queries = queries.reshape(
            batch_dim, seq_dim, self._original_layer.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self._original_layer.rope(queries, offset=cache.offset)
            keys = self._original_layer.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self._original_layer.rope(queries)
            keys = self._original_layer.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self._original_layer.scale,  # type: ignore
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_dim, seq_dim, -1)

        return self._original_layer.o_proj(output)


class MiniMaxShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(MiniMaxModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            # Shard the self attention
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N

            layer.self_attn = WrappedMiniMaxAttention(layer.self_attn, self.group)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]

            # Shard the MoE.
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.gate_proj
            )
            self.sharded_to_all_linear_in_place(
                layer.block_sparse_moe.switch_mlp.down_proj
            )
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.up_proj
            )
            layer.block_sparse_moe = ShardedMoE(layer.block_sparse_moe)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            layer.block_sparse_moe.sharding_group = self.group  # pyright: ignore[reportAttributeAccessIssue]
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class QwenShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(
            Qwen3MoeModel | Qwen3NextModel | Qwen3_5TextModel | Qwen3_5MoeModel, model
        )
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            # Shard the self attention
            if isinstance(layer, Qwen3MoeDecoderLayer):
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj
                )
                layer.self_attn.n_heads //= self.N
                layer.self_attn.n_kv_heads //= self.N
            else:
                assert isinstance(layer, (Qwen3NextDecoderLayer, Qwen3_5DecoderLayer))
                if hasattr(layer, "linear_attn"):
                    linear_attn = layer.linear_attn

                    if isinstance(linear_attn, Qwen3NextGatedDeltaNet):
                        # Qwen3-Next: combined projections
                        linear_attn.in_proj_qkvz = self.all_to_sharded_linear(
                            linear_attn.in_proj_qkvz
                        )
                        linear_attn.in_proj_ba = self.all_to_sharded_linear(
                            linear_attn.in_proj_ba
                        )
                    else:
                        # Qwen3.5: separate projections
                        # in_proj_qkv has sections [q(key_dim), k(key_dim), v(value_dim)]
                        # that must be split section-aware, not as a contiguous block
                        key_dim = linear_attn.key_dim
                        value_dim = linear_attn.value_dim
                        linear_attn.in_proj_qkv = shard_linear(
                            linear_attn.in_proj_qkv,
                            "all-to-sharded",
                            segments=[key_dim, key_dim + key_dim],
                            group=self.group,
                        )
                        linear_attn.in_proj_z = self.all_to_sharded_linear(
                            linear_attn.in_proj_z
                        )
                        linear_attn.in_proj_b = self.all_to_sharded_linear(
                            linear_attn.in_proj_b
                        )
                        linear_attn.in_proj_a = self.all_to_sharded_linear(
                            linear_attn.in_proj_a
                        )
                    linear_attn.out_proj = self.sharded_to_all_linear(
                        linear_attn.out_proj
                    )

                    # Shard conv1d: depthwise conv with non-contiguous channel slicing.
                    # Channel layout is [q(key_dim), k(key_dim), v(value_dim)].
                    # Each rank takes its head-slice from each of the three sections.
                    rank = self.group.rank()
                    key_dim = linear_attn.key_dim
                    value_dim = linear_attn.value_dim
                    key_dim_shard = key_dim // self.N
                    value_dim_shard = value_dim // self.N

                    q_idx = mx.arange(rank * key_dim_shard, (rank + 1) * key_dim_shard)
                    k_idx = mx.arange(
                        key_dim + rank * key_dim_shard,
                        key_dim + (rank + 1) * key_dim_shard,
                    )
                    v_idx = mx.arange(
                        2 * key_dim + rank * value_dim_shard,
                        2 * key_dim + (rank + 1) * value_dim_shard,
                    )
                    conv_indices = mx.concatenate([q_idx, k_idx, v_idx])
                    linear_attn.conv1d.weight = linear_attn.conv1d.weight[conv_indices]
                    new_conv_dim = key_dim_shard * 2 + value_dim_shard
                    linear_attn.conv1d.groups = new_conv_dim

                    num_v_shard = linear_attn.num_v_heads // self.N
                    v_start = rank * num_v_shard
                    v_end = v_start + num_v_shard
                    linear_attn.A_log = linear_attn.A_log[v_start:v_end]
                    linear_attn.dt_bias = linear_attn.dt_bias[v_start:v_end]

                    linear_attn.num_k_heads //= self.N
                    linear_attn.num_v_heads //= self.N
                    linear_attn.key_dim = (
                        linear_attn.head_k_dim * linear_attn.num_k_heads
                    )
                    linear_attn.value_dim = (
                        linear_attn.head_v_dim * linear_attn.num_v_heads
                    )
                    linear_attn.conv_dim = (
                        linear_attn.key_dim * 2 + linear_attn.value_dim
                    )
                else:
                    layer.self_attn.q_proj = self.all_to_sharded_linear(
                        layer.self_attn.q_proj
                    )
                    layer.self_attn.k_proj = self.all_to_sharded_linear(
                        layer.self_attn.k_proj
                    )
                    layer.self_attn.v_proj = self.all_to_sharded_linear(
                        layer.self_attn.v_proj
                    )
                    layer.self_attn.o_proj = self.sharded_to_all_linear(
                        layer.self_attn.o_proj
                    )
                    layer.self_attn.num_attention_heads //= self.N
                    layer.self_attn.num_key_value_heads //= self.N

            # Shard the MoE.
            if isinstance(
                layer.mlp,
                (
                    Qwen3MoeSparseMoeBlock,
                    Qwen3NextSparseMoeBlock,
                    Qwen3_5SparseMoeBlock,
                ),
            ):
                if _USE_EXPERT_PARALLEL and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                    # Expert Parallelism: each node owns a subset of FULL experts.
                    # Contiguous weight reads, no per-expert sharding.
                    ep = ExpertParallelMoE(layer.mlp, self.group)
                    ep._split_expert_weights()
                    layer.mlp = ep  # type: ignore
                else:
                    # Standard TP: shard each expert's weights across nodes.
                    self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                    self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                    self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                    if isinstance(
                        layer.mlp, (Qwen3NextSparseMoeBlock, Qwen3_5SparseMoeBlock)
                    ):
                        self.all_to_sharded_linear_in_place(
                            layer.mlp.shared_expert.gate_proj
                        )
                        self.sharded_to_all_linear_in_place(
                            layer.mlp.shared_expert.down_proj
                        )
                        self.all_to_sharded_linear_in_place(layer.mlp.shared_expert.up_proj)
                    layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                    layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class Glm4MoeShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(Glm4MoeModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            layer.self_attn.n_kv_heads //= self.N

            if isinstance(layer.mlp, MoE):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class GptOssShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(GptOssMoeModel, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads
                // layer.self_attn.num_key_value_heads
            )

            layer.self_attn.sinks = layer.self_attn.sinks[
                layer.self_attn.num_attention_heads
                * self.group.rank() : layer.self_attn.num_attention_heads
                * (self.group.rank() + 1)
            ]

            self.all_to_sharded_linear_in_place(layer.mlp.experts.gate_proj)
            self.sharded_to_all_linear_in_place(layer.mlp.experts.down_proj)
            self.all_to_sharded_linear_in_place(layer.mlp.experts.up_proj)

            layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
            layer.mlp.sharding_group = self.group  # pyright: ignore[reportAttributeAccessIssue]
            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model


class Step35ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
        timeout_seconds: float,
        on_timeout: TimeoutCallback | None,
        on_layer_loaded: LayerLoadedCallback | None,
    ) -> nn.Module:
        model = cast(Step35Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            eval_with_timeout(layer.parameters(), timeout_seconds / total, on_timeout)
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_heads //= self.N
            layer.self_attn.num_kv_heads //= self.N

            if getattr(layer.self_attn, "use_head_wise_attn_gate", False):
                layer.self_attn.g_proj = self.all_to_sharded_linear(
                    layer.self_attn.g_proj
                )

            if isinstance(layer.mlp, Step35MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            else:
                layer.mlp.sharding_group = self.group
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.up_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.share_expert.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)

            mx.eval(layer)
            if on_layer_loaded is not None:
                on_layer_loaded(i, total)
        return model

from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx
from mlx_lm.generate import GenerationBatch

_PRECOMPUTE_TOP_K = 20


@dataclass
class BatchTopKLogprobs:
    uids: list[int] = field(default_factory=list)
    indices: mx.array | None = None
    values: mx.array | None = None
    selected: mx.array | None = None
    _uid_to_row: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._uid_to_row = {uid: i for i, uid in enumerate(self.uids)}

    def for_uid(self, uid: int) -> tuple[list[int], list[float], float] | None:
        if self.indices is None or self.values is None or self.selected is None:
            return None
        row = self._uid_to_row.get(uid)
        if row is None:
            return None
        return (
            cast(list[int], self.indices[row].tolist()),
            cast(list[float], self.values[row].tolist()),
            float(self.selected[row].item()),
        )


@dataclass
class _TopKBuffer:
    needs_topk: bool = False
    pending: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)
    ready: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)


def _get_buffer(batch: GenerationBatch) -> _TopKBuffer:
    buf = getattr(batch, "_topk_buffer", None)
    if buf is None:
        buf = _TopKBuffer()
        batch._topk_buffer = buf  # pyright: ignore[reportAttributeAccessIssue]
    return buf


def set_needs_topk(batch: GenerationBatch, needed: bool) -> None:
    _get_buffer(batch).needs_topk = needed


def take_ready_topk(batch: GenerationBatch) -> BatchTopKLogprobs:
    return _get_buffer(batch).ready


def _patched_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    # GPU-utilization probe (optional). Same pattern as the upstream mlx-lm
    # probe in mlx_lm/generate.py:GenerationBatch._step, copied here because
    # this monkey-patch replaces _step at import time so the upstream probe
    # never runs. Env-gated (MLX_GPU_TIME=1) — fast no-op when unset.
    import os as _os
    import sys as _sys
    import time as _time
    _gpu_probe = bool(_os.environ.get("MLX_GPU_TIME"))
    if _gpu_probe:
        _gpu_log_every = int(_os.environ.get("MLX_GPU_TIME_LOG_EVERY", "32"))
        _wall_start = _time.perf_counter()
        _gpu_ns_start = mx.metal.gpu_time_ns()

    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens
    assert inputs is not None, "_step requires initialized _next_tokens"

    buf = _get_buffer(self)
    buf.ready = buf.pending
    buf.pending = BatchTopKLogprobs()

    if _gpu_probe:
        _t_pre_forward = _time.perf_counter()
    logits = self.model(inputs[:, None], cache=self.prompt_cache)
    logits = logits[:, -1, :]
    if _gpu_probe:
        _t_post_forward = _time.perf_counter()

    if self.logits_processors is not None and any(self.logits_processors):
        processed_logits: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_logits = logits[e : e + 1]
            for processor in self.logits_processors[e]:
                sample_logits = processor(mx.array(self.tokens[e]), sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    if self.samplers is not None and any(self.samplers):
        all_samples: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_sampler = self.samplers[e] or self.fallback_sampler
            all_samples.append(sample_sampler(logprobs[e : e + 1]))
        sampled = mx.concatenate(all_samples, axis=0)
    else:
        sampled = self.fallback_sampler(logprobs)

    self._next_tokens = sampled
    self._next_logprobs = logprobs

    if _gpu_probe:
        _t_pre_async_eval = _time.perf_counter()
    if buf.needs_topk:
        batch_size = len(self.uids)
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        pending_indices = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        pending_values = mx.take_along_axis(logprobs, pending_indices, axis=1)
        sort_order = mx.argsort(-pending_values, axis=1)
        pending_indices = mx.take_along_axis(pending_indices, sort_order, axis=1)
        pending_values = mx.take_along_axis(pending_values, sort_order, axis=1)
        pending_selected = logprobs[mx.arange(batch_size), sampled]
        buf.pending = BatchTopKLogprobs(
            uids=list(self.uids),
            indices=pending_indices,
            values=pending_values,
            selected=pending_selected,
        )
        mx.async_eval(
            self._next_tokens,
            self._next_logprobs,
            pending_indices,
            pending_values,
            pending_selected,
        )
    else:
        mx.async_eval(self._next_tokens, self._next_logprobs)

    if _gpu_probe:
        _t_post_async_eval = _time.perf_counter()

    current_lp = self._current_logprobs
    if isinstance(current_lp, mx.array):
        mx.eval(inputs, current_lp)
    elif current_lp:
        mx.eval(inputs, *current_lp)
    else:
        mx.eval(inputs)
    if _gpu_probe:
        _t_post_eval = _time.perf_counter()

    # NOTE: previous attempts to bound the per-step ArrayDesc leak via
    # mx.detach (single-node + recursive subgraph), mx.synchronize, and
    # gc.collect all failed (the leaked ArrayDescs aren't held by the
    # array graph, the worker queue, or Python ref cycles). Compile
    # cache clearing is a candidate but mx.clear_compile_cache crashes
    # the runner ("PyThreadState_Get: GIL released") — even after
    # holding the GIL the per-thread cache lifetime model is fragile.
    # Use MLX_DISABLE_COMPILE=1 at startup to test the compile-cache
    # hypothesis safely (paid for by re-tracing on every call).

    token_list = cast(list[int], inputs.tolist())
    for sti, ti in zip(self.tokens, token_list, strict=True):
        sti.append(ti)

    if isinstance(current_lp, mx.array):
        current_lp = list(current_lp)

    if _gpu_probe:
        _t_step_end = _time.perf_counter()
        _wall_ns = int((_t_step_end - _wall_start) * 1e9)
        _gpu_ns_delta = mx.metal.gpu_time_ns() - _gpu_ns_start
        _ns = lambda a, b: int((b - a) * 1e9)
        _pre_fwd_ns = _ns(_wall_start, _t_pre_forward)
        _fwd_build_ns = _ns(_t_pre_forward, _t_post_forward)
        _sample_build_ns = _ns(_t_post_forward, _t_pre_async_eval)
        _async_ns = _ns(_t_pre_async_eval, _t_post_async_eval)
        _eval_block_ns = _ns(_t_post_async_eval, _t_post_eval)
        _post_eval_ns = _ns(_t_post_eval, _t_step_end)

        cnt = getattr(self, "_gpu_probe_cnt", 0) + 1
        self._gpu_probe_cnt = cnt
        self._gpu_probe_sum_wall = getattr(self, "_gpu_probe_sum_wall", 0) + _wall_ns
        self._gpu_probe_sum_gpu = getattr(self, "_gpu_probe_sum_gpu", 0) + _gpu_ns_delta
        self._gpu_probe_sum_pre_fwd = getattr(self, "_gpu_probe_sum_pre_fwd", 0) + _pre_fwd_ns
        self._gpu_probe_sum_fwd_build = getattr(self, "_gpu_probe_sum_fwd_build", 0) + _fwd_build_ns
        self._gpu_probe_sum_sample = getattr(self, "_gpu_probe_sum_sample", 0) + _sample_build_ns
        self._gpu_probe_sum_async = getattr(self, "_gpu_probe_sum_async", 0) + _async_ns
        self._gpu_probe_sum_eval = getattr(self, "_gpu_probe_sum_eval", 0) + _eval_block_ns
        self._gpu_probe_sum_post = getattr(self, "_gpu_probe_sum_post", 0) + _post_eval_ns
        if cnt % _gpu_log_every == 0:
            avg = lambda x: x / cnt / 1e6
            B = inputs.shape[0] if hasattr(inputs, "shape") else len(inputs)
            pct = (self._gpu_probe_sum_gpu / self._gpu_probe_sum_wall * 100.0) if self._gpu_probe_sum_wall > 0 else 0.0
            _sys.stderr.write(
                f"[GPU_TIME pid={_os.getpid()}] "
                f"steps={cnt} B={B} "
                f"wall={avg(self._gpu_probe_sum_wall):.2f} "
                f"gpu={avg(self._gpu_probe_sum_gpu):.2f} "
                f"pct={pct:.1f} "
                f"pre_fwd={avg(self._gpu_probe_sum_pre_fwd):.3f} "
                f"fwd_build={avg(self._gpu_probe_sum_fwd_build):.2f} "
                f"sample={avg(self._gpu_probe_sum_sample):.3f} "
                f"async={avg(self._gpu_probe_sum_async):.3f} "
                f"eval={avg(self._gpu_probe_sum_eval):.2f} "
                f"post={avg(self._gpu_probe_sum_post):.2f}\n"
            )
            _sys.stderr.flush()

    return token_list, current_lp


def apply_batch_gen_patch() -> None:
    GenerationBatch._step = _patched_step

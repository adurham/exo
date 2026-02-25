import os
import time
from copy import deepcopy
import functools
from typing import Callable, Generator, cast, get_args, Optional, Tuple, List, Union, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import BatchGenerator
from mlx_lm.utils import does_model_support_input_embeddings
from mlx_lm.models.cache import ArraysCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.api import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill, set_pipeline_cache
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
    normalize_prompt_for_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_TO_UPDATE = 1000
_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5
_DEFAULT_COMPLETION_BATCH_SIZE = 8
_DEFAULT_PREFILL_BATCH_SIZE = 8
_DEFAULT_PREFILL_STEP_SIZE = 32  # Must be small for hybrid TP: 59 layers × all-reduce per layer. 48 tokens = 766ms, 177 = GPU timeout


from mlx.utils import tree_reduce
import contextlib

@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        max_rec_size = mx.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            # logger.warning(...) 
            pass
        old_limit = mx.set_wired_limit(max_rec_size)
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()
            mx.set_wired_limit(old_limit)


def _env_int(name: str, default: int) -> int:
    try:
        return int((__import__("os").environ.get(name) or "").strip() or default)
    except Exception:
        return default


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def mlx_batch_generate(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    tasks: list[TextGenerationTaskParams],
    prompts: list[str],
    group: mx.distributed.Group | None = None,
) -> Generator[tuple[int, GenerationResponse]]:
    """Batch-generate tokens for multiple prompts and multiplex results.

    This is a best-effort batching helper intended for high-throughput,
    non-tool-calling, non-logprobs use cases. Callers should ensure tasks
    are compatible (same sampling params, no stop sequences, etc.).
    """
    assert len(tasks) == len(prompts)
    if not tasks:
        return

    mx.reset_peak_memory()
    seed = tasks[0].seed or 42
    mx.random.seed(seed)

    # All tasks are assumed to have identical sampling params (caller checked).
    sampler = make_sampler(
        temp=tasks[0].temperature if tasks[0].temperature is not None else 0.7,
        top_p=tasks[0].top_p if tasks[0].top_p is not None else 1.0,
        top_k=tasks[0].top_k if tasks[0].top_k is not None else 0,
    )

    prompt_tokens_batch: list[list[int]] = []
    max_tokens_batch: list[int] = []
    for task, prompt in zip(tasks, prompts, strict=True):
        toks_arr = encode_prompt(tokenizer, prompt)
        toks_arr = fix_unmatched_think_end_tokens(toks_arr, tokenizer)
        prompt_tokens_batch.append([int(x) for x in toks_arr.tolist()])
        max_tokens_batch.append(int(task.max_output_tokens or MAX_TOKENS))

    completion_batch_size = max(
        1, _env_int("EXO_BATCH_COMPLETION_SIZE", _DEFAULT_COMPLETION_BATCH_SIZE)
    )
    prefill_batch_size = max(
        1, _env_int("EXO_BATCH_PREFILL_SIZE", _DEFAULT_PREFILL_BATCH_SIZE)
    )

    mx_barrier(group)

    gen = BatchGenerator(
        model=model,
        sampler=sampler,
        completion_batch_size=completion_batch_size,
        prefill_batch_size=prefill_batch_size,
        prefill_step_size=2048,
    )
    uids = gen.insert(prompt_tokens_batch, max_tokens=max_tokens_batch)
    uid_to_index: dict[int, int] = {int(uid): i for i, uid in enumerate(uids)}
    finished: set[int] = set()

    while len(finished) < len(uids):
        responses = gen.next()
        for r in responses:
            uid = int(r.uid)
            if uid in finished:
                continue

            idx = uid_to_index[uid]
            token_id = int(r.token)
            text = tokenizer.decode([token_id])

            finish_reason: FinishReason | None = cast(
                FinishReason | None, r.finish_reason
            )
            if finish_reason is not None:
                finished.add(uid)

            yield idx, GenerationResponse(
                text=text,
                token=token_id,
                finish_reason=finish_reason
                if finish_reason in get_args(FinishReason)
                else None,
                stats=None,
                usage=None,
            )

    mx_barrier(group)

class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        (tokens_per_sec, num_tokens, snapshots)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[int] = []  # deferred: stores token counts, real snapshot taken at rollback time

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            # Snapshot is O(layers × seq_len) due to deepcopy.  For 16K tokens
            # across 55 layers this takes ~11s — far too expensive to do on
            # every progress callback.  We defer snapshots: record token count
            # now and take the real deepcopy lazily when needed (line ~276).
            snapshots.append(processed)

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    set_pipeline_prefill(model, is_prefill=True)
    set_pipeline_cache(model, cache)

    mx_barrier(group)
    logger.info("Starting prefill")

    # Use max_tokens=1 because max_tokens=0 does not work.
    # We just throw away the generated token - we only care about filling the cache
    try:
        for _ in stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_tokens,
            max_tokens=1,
            sampler=sampler,
            prompt_cache=cache,
            prefill_step_size=_env_int("EXO_PREFILL_STEP_SIZE", _DEFAULT_PREFILL_STEP_SIZE),
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
            prompt_progress_callback=progress_callback,
        ):
            break  # Stop after first iteration - cache is now filled
    except PrefillCancelled:
        set_pipeline_prefill(model, is_prefill=False)
        raise

    set_pipeline_prefill(model, is_prefill=False)

    # stream_generate added 1 extra generated token to the cache, so we should trim it.
    # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
    for c in cache:
        if isinstance(c, ArraysCache):
            # ArraysCache (SSM state) can't be trimmed, reset to empty
            c.state = [None] * len(c.state)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        elif hasattr(c, 'trim'):
            c.trim(2)  # pyright: ignore[reportUnknownMemberType]

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Return deferred snapshots as CacheSnapshots for prefix cache compatibility
    result_snapshots: list[CacheSnapshot] = []
    if has_ssm and snapshots:
        for tc in snapshots[:-1]:  # Exclude the last snapshot
            result_snapshots.append(CacheSnapshot(states=[], token_count=tc))
    return tokens_per_sec, num_tokens, result_snapshots


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None = None,
) -> int:
    content = "Prompt to warm up the inference engine. Repeat this."

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        task_params=TextGenerationTaskParams(
            model=ModelId(""),
            input=[InputMessage(role="user", content=content)],
        ),
    )

    tokens_generated = 0

    cache = make_kv_cache(
        model=model,
    )

    # Use a default sampler for warmup
    sampler = make_sampler(temp=0.0)

    mx_barrier(group)

    logger.info("Generating warmup tokens")
    set_pipeline_prefill(model, is_prefill=True)
    for _r in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        max_tokens=50,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=2048,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        logger.info("Generated warmup token: " + str(_r.text))
        tokens_generated += 1
    set_pipeline_prefill(model, is_prefill=False)

    # Save warmup KV cache to prefix cache for potential reuse
    if kv_prefix_cache is not None:
        warmup_tokens = encode_prompt(tokenizer, warmup_prompt)
        kv_prefix_cache.add_kv_cache(warmup_tokens, cache)
        logger.info(f"Warmup KV cache saved to prefix cache ({len(warmup_tokens)} tokens)")

    logger.info("Generated ALL warmup tokens")

    mx_barrier(group)

    return tokens_generated


def ban_token_ids(token_ids: list[int]) -> Callable[[mx.array, mx.array], mx.array]:
    token_ids = [int(t) for t in token_ids]

    def proc(_history: mx.array, logits: mx.array) -> mx.array:
        for tid in token_ids:
            logits[..., tid] = -1e9
        return logits

    return proc


def eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def extract_top_logprobs(
    logprobs: mx.array,
    tokenizer: TokenizerWrapper,
    top_logprobs: int,
    selected_token: int,
) -> tuple[float, list[TopLogprobItem]]:
    """Extract the selected token's logprob and top alternative tokens.

    Args:
        logprobs: Full vocabulary logprobs array from MLX
        tokenizer: Tokenizer for decoding token IDs to strings
        top_logprobs: Number of top alternatives to return
        selected_token: The token ID that was actually sampled

    Returns:
        Tuple of (selected_token_logprob, list of TopLogprobItem for top alternatives)
    """
    # Get the logprob of the selected token
    selected_logprob = float(logprobs[selected_token].item())

    # Get top indices (most probable tokens)
    # mx.argpartition gives indices that would partition the array
    # We negate logprobs since argpartition finds smallest, and we want largest
    top_logprobs = min(top_logprobs, logprobs.shape[0])  # Don't exceed vocab size
    top_indices = mx.argpartition(-logprobs, top_logprobs)[:top_logprobs]

    # Get the actual logprob values for these indices
    top_values = logprobs[top_indices]

    # Sort by logprob (descending) for consistent ordering
    sort_order = mx.argsort(-top_values)
    top_indices = top_indices[sort_order]
    top_values = top_values[sort_order]

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for i in range(top_logprobs):
        token_id = int(top_indices[i].item())
        token_logprob = float(top_values[i].item())
        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        # Get byte representation
        token_bytes = list(token_str.encode("utf-8"))
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=token_bytes,
            )
        )

    return selected_logprob, top_logprob_items


    return parser


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())
DEFAULT_QUANTIZED_KV_START = 5000

def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    
    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(
                f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                f"must match the sequence length of the prompt ({len(prompt)}), or the "
                "prompt must be empty."
            )
    elif len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = make_kv_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            out = model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            out = model(input_tokens, cache=prompt_cache)
        
        return out

    # Pre-compute hybrid pipeline layer list once (avoid per-step getattr + iteration)
    _hybrid_layers: list = []
    try:
        from exo.worker.engines.mlx.auto_parallel import _HybridPipelineLastLayer, _get_layers
        inner_model = getattr(model, 'model', model)
        _all_layers = _get_layers(inner_model)
        _hybrid_layers = [l for l in _all_layers if isinstance(l, _HybridPipelineLastLayer)]
    except (ValueError, ImportError):
        pass

    def _drain_pending_sends():
        """Collect and clear any deferred sends from _HybridPipelineLastLayer."""
        pending = []
        for layer in _hybrid_layers:
            pending.extend(layer._pending_sends)
            layer._pending_sends = []
        return pending

    _step_counter = [0]

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens
        _step_id = _step_counter[0]
        _step_counter[0] += 1

        with mx.stream(generation_stream):
            _st0 = _time.perf_counter()
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )
            _st1 = _time.perf_counter()

            # Robust reshaping for 2D logits
            if len(logits.shape) == 2 and logits.shape[0] == input_tokens.shape[0]:
                logits = logits[None, :, :]
            elif len(logits.shape) == 2:
                logits = logits[:, None, :]

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            _st2 = _time.perf_counter()
            quantize_cache_fn(prompt_cache)
            _st3 = _time.perf_counter()

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            _st4 = _time.perf_counter()

            # Hybrid pipeline token sync
            hybrid_group = getattr(model, '_hybrid_pipeline_group', None)
            decode_mode = getattr(model, '_hybrid_decode_mode', False)
            if hybrid_group is not None and decode_mode:
                is_pp_tail = getattr(model, '_hybrid_pipeline_is_pp_tail', False)

                _pending = _drain_pending_sends()
                _st5 = _time.perf_counter()
                mx.eval(sampled, *_pending)
                _st6 = _time.perf_counter()

                if is_pp_tail:
                    contribution = sampled
                else:
                    contribution = mx.zeros_like(sampled)
                sampled = mx.distributed.all_sum(contribution, group=hybrid_group)
                mx.eval(sampled)
                _st7 = _time.perf_counter()

                logger.info(
                    f"[STEP {_step_id}] tokens={input_tokens.shape[0]} "
                    f"model={(_st1-_st0)*1000:.0f}ms "
                    f"reshape={(_st2-_st1)*1000:.0f}ms "
                    f"quantize={(_st3-_st2)*1000:.0f}ms "
                    f"sample={(_st4-_st3)*1000:.0f}ms "
                    f"eval={(_st6-_st5)*1000:.0f}ms "
                    f"token_sync={(_st7-_st6)*1000:.0f}ms "
                    f"total={(_st7-_st0)*1000:.0f}ms"
                )
            elif hybrid_group is not None:
                _pending = _drain_pending_sends()
                _st5 = _time.perf_counter()
                if _pending:
                    mx.eval(*_pending)
                _st6 = _time.perf_counter()

                logger.info(
                    f"[STEP {_step_id}] tokens={input_tokens.shape[0]} "
                    f"model={(_st1-_st0)*1000:.0f}ms "
                    f"reshape={(_st2-_st1)*1000:.0f}ms "
                    f"quantize={(_st3-_st2)*1000:.0f}ms "
                    f"sample={(_st4-_st3)*1000:.0f}ms "
                    f"drain={(_st6-_st5)*1000:.0f}ms "
                    f"total={(_st6-_st0)*1000:.0f}ms (prefill_mode)"
                )
            else:
                _st5 = _time.perf_counter()
                mx.eval(sampled)
                _st6 = _time.perf_counter()
                logger.info(
                    f"[STEP {_step_id}] tokens={input_tokens.shape[0]} "
                    f"model={(_st1-_st0)*1000:.0f}ms "
                    f"reshape={(_st2-_st1)*1000:.0f}ms "
                    f"quantize={(_st3-_st2)*1000:.0f}ms "
                    f"sample={(_st4-_st3)*1000:.0f}ms "
                    f"eval={(_st6-_st5)*1000:.0f}ms "
                    f"total={(_st6-_st0)*1000:.0f}ms"
                )

            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        import time as _time
        _prefill_chunk_idx = 0

        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            logger.info(f"PREFILL loop: starting chunk {_prefill_chunk_idx}, remaining={remaining}, n_to_process={n_to_process}")

            _t0 = _time.perf_counter()
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            _t1 = _time.perf_counter()
            quantize_cache_fn(prompt_cache)
            _t2 = _time.perf_counter()

            # Eval cache states + pending sends together
            _num_caches = len(prompt_cache)
            _current_sends = _drain_pending_sends()
            all_states = [_c.state for _c in prompt_cache]
            if _current_sends:
                mx.eval(*all_states, *_current_sends)
            else:
                mx.eval(*all_states)
            _slow_caches = []
            _t3 = _time.perf_counter()

            _kv_len = prompt_cache[0].offset if hasattr(prompt_cache[0], 'offset') else '?'
            logger.info(
                f"PREFILL chunk {_prefill_chunk_idx}: "
                f"tokens={n_to_process}, kv_len={_kv_len}, "
                f"model={(_t1-_t0)*1000:.0f}ms, "
                f"quantize={(_t2-_t1)*1000:.0f}ms, "
                f"eval={(_t3-_t2)*1000:.0f}ms ({_num_caches} layers), "
                f"sends={len(_current_sends)}, "
                f"total={(_t3-_t0)*1000:.0f}ms"
            )
            _prefill_chunk_idx += 1

            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            _t4 = _time.perf_counter()
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            _t5 = _time.perf_counter()
            logger.info(
                f"PREFILL loop end: "
                f"model={(_t1-_t0)*1000:.0f}ms "
                f"quantize={(_t2-_t1)*1000:.0f}ms "
                f"eval={(_t3-_t2)*1000:.0f}ms "
                f"callback={(_t4-_t3)*1000:.0f}ms "
                f"slice={(_t5-_t4)*1000:.0f}ms "
                f"total={(_t5-_t0)*1000:.0f}ms"
            )

        _loop_exit = _time.perf_counter()
        logger.info(f"PREFILL loop exited, prompt_remaining={len(prompt)}, chunks={_prefill_chunk_idx}")

        # Clear cache once after prefill completes, not per-chunk
        _gap_t0 = _time.perf_counter()
        mx.clear_cache()
        _gap_t1 = _time.perf_counter()

        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)
        _gap_t2 = _time.perf_counter()

    mx.async_eval(y, logprobs)
    _gap_t3 = _time.perf_counter()
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n == 0:
            mx.eval(y)
            _gap_t4 = _time.perf_counter()
            logger.info(
                f"PREFILL->DECODE gap: "
                f"clear_cache={(_gap_t1-_gap_t0)*1000:.0f}ms "
                f"first_step={(_gap_t2-_gap_t1)*1000:.0f}ms "
                f"async_eval={(_gap_t3-_gap_t2)*1000:.0f}ms "
                f"eval_y={(_gap_t4-_gap_t3)*1000:.0f}ms "
                f"total={(_gap_t4-_gap_t0)*1000:.0f}ms"
            )
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if n == max_tokens:
            break
        yield y.item(), logprobs
        if n % 1024 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1


def stream_generate(
    model: nn.Module,
    tokenizer: Union[TokenizerWrapper, Any],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    return_logprobs: bool = False,
    top_logprobs_k: Optional[int] = None,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Try to infer if special tokens are needed
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    kwargs["max_tokens"] = max_tokens

    if draft_model is None:
        kwargs.pop("num_draft_tokens", None)
        token_generator = generate_step(prompt, model, **kwargs)
        # from_draft always false for non-speculative generation
        token_generator = (
            (token, logprobs, False) for token, logprobs in token_generator
        )
    else:
        raise NotImplementedError("Speculative decoding not supported in debug mode")

    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        logprobs = None
        token = -1 
        for n, (token_mx, logprobs, from_draft) in enumerate(token_generator):
            if isinstance(token_mx, mx.array):
                token = token_mx.item()
            else:
                token = token_mx
             
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            # Extract logprobs if requested
            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if return_logprobs:
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=logprobs,
                    tokenizer=tokenizer,
                    top_logprobs=top_logprobs_k or DEFAULT_TOP_LOGPROBS,
                    selected_token=token.item() if hasattr(token, "item") else token,
                )
            
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprob=logprob,
                top_logprobs=top_logprobs,
                stats=GenerationStats(
                    prompt_tps=prompt_tps,
                    generation_tps=(n + 1) / (time.perf_counter() - tic),
                    prompt_tokens=prompt.size,
                    generation_tokens=n + 1,
                    peak_memory_usage=Memory.from_bytes(int(mx.get_peak_memory()))
                ),
                finish_reason=None,
                usage=None, 
            )

        detokenizer.finalize()
        peak_mem_bytes = int(mx.get_peak_memory())
        stats = GenerationStats(
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            prompt_tokens=prompt.size,
            generation_tokens=n + 1,
            peak_memory_usage=Memory.from_bytes(peak_mem_bytes)
        )
        # Re-extract logprobs for final token if needed
        # Use simple variable names to avoid conflict
        final_logprob: float | None = None
        final_top_logprobs: list[TopLogprobItem] | None = None
        
        if return_logprobs and logprobs is not None:
             final_logprob, final_top_logprobs = extract_top_logprobs(
                logprobs=logprobs,
                tokenizer=tokenizer,
                top_logprobs=top_logprobs_k or DEFAULT_TOP_LOGPROBS,
                selected_token=token.item() if hasattr(token, "item") else token,
            )

        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprob=final_logprob,
            top_logprobs=final_top_logprobs,
            stats=stats,
            finish_reason=cast(FinishReason, "stop" if token in tokenizer.eos_token_ids else "length"),
            usage=None, 
        )

def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)

    # Normalize prompt for cache comparison (strip volatile patterns like cch=)
    normalized_prompt = normalize_prompt_for_cache(prompt)
    if normalized_prompt != prompt:
        normalized_tokens = encode_prompt(tokenizer, normalized_prompt)
        normalized_tokens = fix_unmatched_think_end_tokens(normalized_tokens, tokenizer)
        logger.info(
            f"Prompt normalized for cache: {len(all_prompt_tokens)} -> {len(normalized_tokens)} tokens "
            f"(stripped {len(all_prompt_tokens) - len(normalized_tokens)} volatile tokens)"
        )
    else:
        normalized_tokens = None

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    if kv_prefix_cache is None:
        caches = make_kv_cache(model=model)
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, all_prompt_tokens, normalized_tokens=normalized_tokens
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = []
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)]

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    mx_barrier(group)
    logger.debug("Ready to prefill")
    # Prefill cache with all tokens except the last one
    prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
        model,
        tokenizer,
        sampler,
        prompt_tokens[:-1],
        caches,
        group,
        on_prefill_progress,
    )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    # stream_generate starts from the last token
    last_token = prompt_tokens[-2:]

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    in_thinking = False
    reasoning_tokens = 0
    think_start = tokenizer.think_start
    think_end = tokenizer.think_end

    logger.info("Starting decode")
    mx_barrier(group)

    logger.debug(f"Tokenizer EOS IDs: {getattr(tokenizer, 'eos_token_ids', 'Not Set')}")
    logger.info("Starting stream_generate loop...")
    for completion_tokens, out in enumerate(
        stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=last_token,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=caches,
            prefill_step_size=1,
            kv_group_size=KV_GROUP_SIZE,
            kv_bits=KV_BITS,
            return_logprobs=task.logprobs,
            top_logprobs_k=task.top_logprobs,
        ),
        start=1,
    ):
        logger.debug(f"Gen token [{completion_tokens}]: {out.token} | {out.text}")
        generated_text_parts.append(out.text)
        accumulated_text += out.text
        
        # Check for stop tokens manually if needed
        if out.token in getattr(tokenizer, 'eos_token_ids', []):
             logger.debug(f"Hit stop token: {out.token}")

        if think_start is not None and out.text == think_start:
            in_thinking = True
        elif think_end is not None and out.text == think_end:
            in_thinking = False
        if in_thinking:
            reasoning_tokens += 1

        # Check for stop sequences
        text = out.text
        finish_reason: FinishReason | None = cast(
            FinishReason | None, out.finish_reason
        )
        stop_matched = False

        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in accumulated_text:
                    # Trim text to just before the stop sequence
                    stop_index = accumulated_text.find(stop_seq)
                    text_before_stop = accumulated_text[:stop_index]
                    chunk_start = len(accumulated_text) - len(out.text)
                    text = text_before_stop[chunk_start:]
                    finish_reason = "stop"
                    stop_matched = True
                    break

        is_done = finish_reason is not None
        
        stats: GenerationStats | None = None
        if is_done and out.stats is not None:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps),
                generation_tps=float(out.stats.generation_tps),
                prompt_tokens=int(prefill_tokens + out.stats.prompt_tokens),
                generation_tokens=int(out.stats.generation_tokens),
                peak_memory_usage=out.stats.peak_memory_usage,
            )
            if not stop_matched and out.finish_reason not in get_args(FinishReason):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

            total_prompt_tokens = len(all_prompt_tokens)
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
            )

        # Extract logprobs from the full vocabulary logprobs array
        # Logprobs are now populated inside stream_generate
        pass

        if is_done:
            # Log generation stats
            generation_elapsed = time.perf_counter() - generation_start_time
            generated_tokens = len(generated_text_parts)
            generation_tps = (
                generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
            )
            logger.debug(
                f"Generation complete: prefill {prompt_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )
            if kv_prefix_cache is not None:
                generated_tokens_array = mx.array(
                    tokenizer.encode(
                        "".join(generated_text_parts), add_special_tokens=False
                    )
                )
                full_prompt_tokens = mx.concatenate(
                    [all_prompt_tokens, generated_tokens_array]
                )
                hit_ratio = (
                    prefix_hit_length / len(all_prompt_tokens)
                    if len(all_prompt_tokens) > 0
                    else 0.0
                )
                if (
                    matched_index is not None
                    and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
                ):
                    kv_prefix_cache.update_kv_cache(
                        matched_index,
                        all_prompt_tokens,
                        caches,
                        cache_snapshots,
                        restore_pos=prefix_hit_length,
                        normalized_tokens=normalized_tokens,
                    )
                else:
                    kv_prefix_cache.add_kv_cache(
                        all_prompt_tokens, caches, cache_snapshots,
                        normalized_tokens=normalized_tokens,
                    )

        yield GenerationResponse(
            text=text,
            token=out.token,
            logprob=out.logprob,
            top_logprobs=out.top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
        )

        if is_done:
            mx_barrier(group)
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]

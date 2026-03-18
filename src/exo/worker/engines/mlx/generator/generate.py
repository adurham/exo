import functools
import math
import os
import time
from typing import Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models.cache import ArraysCache, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_TRACING_ENABLED
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
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx.auto_parallel import (
    HybridPipelineLastLayer,
    HybridPipelinePassthroughLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
    _guarded_eval,
    clear_prefill_sends,
    flush_prefill_sends,
    get_pipeline_timings,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
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
    MAX_PREFILL_CHUNK,
    MAX_TOKENS,
    PREFILL_STEP_SIZE,
)

# Self-speculative decoding: use the same model with layer skipping as draft.
# EXO_SPECULATIVE_DECODE=N means skip factor N (e.g., 10 = use every 10th layer).
# 0 or unset = disabled.
_SPECULATIVE_SKIP = int(os.environ.get("EXO_SPECULATIVE_DECODE", "0"))
_SPECULATIVE_DRAFT_TOKENS = 10  # fallback only; instance config draft_tokens is authoritative
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5


class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


def _truncate_prompt_tokens(tokens: mx.array, max_tokens: int) -> mx.array:
    """Truncate prompt tokens to fit within max_tokens by removing the middle.

    Keeps the first 25% (system prompt / BOS) and the last 75% (recent turns),
    which preserves the most relevant context for subagent-style requests.
    """
    if len(tokens) <= max_tokens:
        return tokens
    keep_start = max_tokens // 4
    keep_end = max_tokens - keep_start
    return mx.concatenate([tokens[:keep_start], tokens[-keep_end:]])


def _has_pipeline_communication_layer(model: Model):
    for layer in model.layers:
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer, HybridPipelineLastLayer, HybridPipelinePassthroughLayer)):
            return True
    return False


def pipeline_parallel_prefill(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
    distributed_prompt_progress_callback: Callable[[], None] | None,
    group: mx.distributed.Group,
) -> None:
    """Prefill the KV cache for pipeline parallel with overlapping stages.

    Each rank processes the full prompt through its real cache, offset by leading
    and trailing dummy iterations.

    Total iterations per rank = N_real_chunks + world_size - 1:
      - rank r leading dummies  (skip_pipeline_io, throwaway cache)
      - N_real_chunks real      (pipeline IO active, real cache)
      - (world_size-1-r) trailing dummies (skip_pipeline_io, throwaway cache)

    e.g.
    Timeline (2 ranks, 3 chunks of 10240 tokens @ step=4096):
        iter 0: R0 real[0:4096]     R1 dummy
        iter 1: R0 real[4096:8192]  R1 real[0:4096]
        iter 2: R0 real[8192:10240] R1 real[4096:8192]
        iter 3: R0 dummy            R1 real[8192:10240]

    This function is designed to match mlx_lm's stream_generate exactly in terms of
    side effects (given the same prefill step size)
    """
    prefill_step_size = prefill_step_size // min(4, group.size())

    quantize_cache_fn: Callable[..., None] = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=0,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    _prompt_cache: KVCacheType = prompt_cache
    rank = group.rank()
    world_size = group.size()

    # Build list of real prompt chunk sizes
    total = len(prompt)
    real_chunk_sizes: list[int] = []
    remaining = total - 1
    while remaining:
        n = min(prefill_step_size, remaining)
        real_chunk_sizes.append(n)
        remaining -= n
    n_real = len(real_chunk_sizes)

    # Each rank does: [rank leading dummies] [N real chunks] [world_size-1-rank trailing dummies]
    n_leading = rank
    n_trailing = world_size - 1 - rank
    n_total = n_leading + n_real + n_trailing

    t_start = time.perf_counter()
    processed = 0
    logger.info(
        f"[R{rank}] Pipeline prefill: {n_real} real + {n_leading} leading + {n_trailing} trailing = {n_total} iterations"
    )
    clear_prefill_sends()
    if EXO_TRACING_ENABLED:
        get_pipeline_timings().reset()

    # Initial callback matching generate_step
    prompt_progress_callback(0, total)

    try:
        with mx.stream(generation_stream):
            for _ in range(n_leading):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                model(
                    prompt[processed : processed + chunk_size][None],
                    cache=_prompt_cache,
                )
                quantize_cache_fn(_prompt_cache)
                processed += chunk_size

                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

                flush_prefill_sends()

                # Force incremental eval to prevent lazy graph accumulation.
                # Pipeline tail ranks have no pending sends, so flush_prefill_sends
                # is a no-op — without this, all computation defers to the final
                # mx.eval, causing 90s+ blocks and heartbeat timeouts.
                mx.async_eval(*[c.state for c in _prompt_cache])  # type: ignore

                prompt_progress_callback(processed, total)

            for _ in range(n_trailing):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

    finally:
        clear_prefill_sends()

    # Post-loop: process remaining 1 token + add +1 entry to match stream_generate.
    if EXO_TRACING_ENABLED:
        t_post = time.perf_counter()
    for _ in range(2):
        with mx.stream(generation_stream):
            model(prompt[-1:][None], cache=_prompt_cache)
            quantize_cache_fn(_prompt_cache)
        flush_prefill_sends()

    # Touch heartbeat: model() calls above can take seconds.
    if distributed_prompt_progress_callback is not None:
        distributed_prompt_progress_callback()

    assert _prompt_cache is not None
    if EXO_TRACING_ENABLED:
        t_cache_eval = time.perf_counter()
    from exo.worker.engines.mlx.auto_parallel import _guarded_eval

    _guarded_eval([c.state for c in _prompt_cache])  # type: ignore
    if EXO_TRACING_ENABLED:
        cache_eval_ms = (time.perf_counter() - t_cache_eval) * 1000
        post_ms = (time.perf_counter() - t_post) * 1000
        logger.info(
            f"[R{rank}] Prefill post-loop: {post_ms:.1f}ms (cache eval: {cache_eval_ms:.1f}ms)"
        )

    # Touch heartbeat: mx.eval on cache states can take 10s+ seconds.
    if distributed_prompt_progress_callback is not None:
        distributed_prompt_progress_callback()

    # Final callback matching generate_step
    prompt_progress_callback(total, total)

    prefill_elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[R{rank}] Prefill: {n_real} real + {n_leading}+{n_trailing} dummy iterations, "
        f"Processed {processed} tokens in {prefill_elapsed_ms:.1f}ms "
        f"({processed / (prefill_elapsed_ms / 1000):.1f} tok/s)"
    )
    if EXO_TRACING_ENABLED:
        get_pipeline_timings().log_and_reset("prefill", rank)


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
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
    snapshots: list[CacheSnapshot] = []

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            snapshots.append(snapshot_ssm_states(cache))

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    def combined_progress_callback(processed: int, total: int) -> None:
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    set_pipeline_prefill(model, is_prefill=True)

    # Flush ALL Metal streams before the prefill to ensure no residual RDMA
    # state from the previous request's generation_stream interferes with the
    # new collective operations.  mx_barrier alone syncs the default stream,
    # but the model forward pass runs on generation_stream — stale async work
    # on that stream can cause JACCL RDMA deadlocks.
    if group is not None:
        mx.synchronize()

    if EXO_TRACING_ENABLED:
        t_barrier = time.perf_counter()
    mx_barrier(group)
    if EXO_TRACING_ENABLED:
        barrier_ms = (time.perf_counter() - t_barrier) * 1000
        logger.info(f"Pre-prefill barrier: {barrier_ms:.1f}ms")
    if EXO_TRACING_ENABLED:
        logger.info("Starting prefill")

    # Touch heartbeat: mx_barrier can block for seconds waiting for other ranks.
    if distributed_prompt_progress_callback is not None:
        distributed_prompt_progress_callback()

    is_pipeline = _has_pipeline_communication_layer(model)

    prefill_step_size = PREFILL_STEP_SIZE

    # Cap non-pipeline prefill chunks so heartbeat callbacks fire between
    # chunks.  Pipeline parallel manages its own heartbeat touches per
    # iteration, so only cap the stream_generate path.
    capped_step_size = min(prefill_step_size, MAX_PREFILL_CHUNK)

    try:
        if is_pipeline and num_tokens >= prefill_step_size:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            pipeline_parallel_prefill(
                model=model,
                prompt=prompt_tokens,
                prompt_cache=cache,
                prefill_step_size=prefill_step_size,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                prompt_progress_callback=progress_callback,
                distributed_prompt_progress_callback=distributed_prompt_progress_callback,
                group=group,
            )
        else:
            # Use max_tokens=1 because max_tokens=0 does not work.
            # We just throw away the generated token - we only care about filling the cache
            for _ in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,
                max_tokens=1,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=capped_step_size,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
                prompt_progress_callback=combined_progress_callback,
            ):
                break  # Stop after first iteration - cache is now filled
    except PrefillCancelled:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise

    set_pipeline_queue_sends(model, queue_sends=False)
    set_pipeline_prefill(model, is_prefill=False)

    # stream_generate added 1 extra generated token to the cache, so we should trim it.
    # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
    if EXO_TRACING_ENABLED:
        t_trim = time.perf_counter()
    pre_gen = snapshots[-2] if has_ssm else None
    for i, c in enumerate(cache):
        if has_ssm and isinstance(c, (ArraysCache, RotatingKVCache)):
            assert pre_gen is not None
            if pre_gen.states[i] is not None:
                cache[i] = pre_gen.states[i]  # type: ignore
        else:
            assert not isinstance(c, (ArraysCache, RotatingKVCache))
            c.trim(2)
    if EXO_TRACING_ENABLED:
        logger.info(f"Cache trim took {(time.perf_counter() - t_trim) * 1000:.1f}ms")

    # Touch heartbeat: cache trim above can take seconds at large context.
    if distributed_prompt_progress_callback is not None:
        distributed_prompt_progress_callback()

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    # Exclude the last snapshot
    return tokens_per_sec, num_tokens, snapshots[:-1] if snapshots else []


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
    is_draft_node: bool = False,
) -> int:
    logger.info(f"warming up inference for instance: {model_id}")
    t = time.monotonic()

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

    if EXO_TRACING_ENABLED:
        t_barrier = time.perf_counter()
    mx_barrier(group)
    if EXO_TRACING_ENABLED:
        logger.info(
            f"Warmup pre-barrier: {(time.perf_counter() - t_barrier) * 1000:.1f}ms"
        )

    if is_draft_node:
        # Draft node: warm up the draft model standalone (no TP/PP communication)
        logger.info("Draft node: warming up draft model standalone")
        tokens = mx.array([[1, 2, 3]])
        logits = model(tokens)
        mx.eval(logits)
        tokens_generated = 1
        logger.info("Draft node warmup complete")
    else:
        if EXO_TRACING_ENABLED:
            t_warmup_gen = time.perf_counter()
        prompt_len = len(warmup_prompt) if isinstance(warmup_prompt, list) else len(warmup_prompt)  # type: ignore
        logger.info(f"Generating warmup tokens (prompt_len={prompt_len}, max_tokens=50)")
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
            tokens_generated += 1

        if EXO_TRACING_ENABLED:
            warmup_gen_ms = (time.perf_counter() - t_warmup_gen) * 1000
            logger.info(
                f"Generated ALL warmup tokens: {tokens_generated} in {warmup_gen_ms:.1f}ms"
            )
        else:
            logger.info("Generated ALL warmup tokens")

    if EXO_TRACING_ENABLED:
        t_barrier = time.perf_counter()
    mx_barrier(group)
    if EXO_TRACING_ENABLED:
        logger.info(
            f"Warmup post-barrier: {(time.perf_counter() - t_barrier) * 1000:.1f}ms"
        )

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    check_for_cancel_every = min(
        math.ceil(tokens_generated / min(time.monotonic() - t, 0.001)), 100
    )
    if group is not None:
        gathered = mx.distributed.all_gather(
            mx.array([check_for_cancel_every]),
            group=group,
        )
        _guarded_eval(gathered)
        check_for_cancel_every = int(mx.max(gathered).item())

    logger.info(
        f"runner checking for cancellation every {check_for_cancel_every} tokens"
    )

    return check_for_cancel_every


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
        if math.isnan(token_logprob):
            continue

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


def draft_provider_loop(
    model: Model,
    group: mx.distributed.Group,
    recv_from: int,
    send_to: int,
    num_draft_tokens: int = 10,
    heartbeat: object | None = None,
) -> None:
    """Draft provider loop: receives tokens via RDMA, generates predictions, sends back.

    Runs on the MacBook as part of a hybrid JACCL group.
    The TP master (Studio) sends a control message, the draft node generates
    predictions and sends them back — all over RDMA.

    Protocol per step:
      1. Recv (2,) int32 from TP master: [accepted_token_id, num_to_trim]
         - accepted_token_id: the verified token to continue from
         - num_to_trim: how many draft tokens were rejected (trim KV cache)
         - accepted_token_id < 0 means shutdown
      2. Trim draft KV cache by num_to_trim (KV cache desync fix)
      3. Feed accepted token through draft model
      4. Generate num_draft_tokens predictions
      5. Send (num_draft_tokens,) int32 back to TP master
    """
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    cache = make_prompt_cache(model)
    logger.info(
        f"Draft provider loop started (recv_from={recv_from}, "
        f"send_to={send_to}, K={num_draft_tokens})"
    )

    step = 0
    while True:
        # Touch heartbeat before blocking recv — prefill on Studios can take 10s+
        if heartbeat is not None:
            heartbeat.value = time.monotonic()  # pyright: ignore[reportAttributeAccessIssue]

        # Receive control message: [token_id, num_to_trim]
        ctrl = mx.distributed.recv(shape=(2,), dtype=mx.int32, src=recv_from, group=group)
        mx.eval(ctrl)
        token_id = ctrl[0].item()
        num_to_trim = ctrl[1].item()

        if token_id < 0:
            logger.info("Draft provider: received shutdown signal")
            break

        # Touch heartbeat after recv completes
        if heartbeat is not None:
            heartbeat.value = time.monotonic()  # pyright: ignore[reportAttributeAccessIssue]

        # Trim KV cache for rejected tokens (KV cache desync fix)
        if num_to_trim > 0:
            trim_prompt_cache(cache, num_to_trim)

        # Feed the accepted token through draft model
        t0 = time.perf_counter()
        current = mx.array([[token_id]])
        logits = model(current, cache=cache)
        mx.eval(logits)

        # Generate K draft tokens
        draft_tokens = []
        for _ in range(num_draft_tokens):
            next_token = logits[0, -1].argmax()
            mx.eval(next_token)
            draft_tokens.append(next_token.item())
            current = next_token.reshape(1, 1)
            logits = model(current, cache=cache)
            mx.eval(logits)

        # Send draft predictions via RDMA
        result = mx.array(draft_tokens, dtype=mx.int32)
        mx.distributed.send(result, send_to, group=group)
        mx.eval(result)

        step += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if step <= 5 or step % 50 == 0:
            logger.info(
                f"Draft step {step}: generated {num_draft_tokens} tokens in {elapsed_ms:.1f}ms "
                f"(token_id={token_id}, trim={num_to_trim})"
            )


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
    on_token_count_known: Callable[[int], None] | None = None,
    draft_model: Model | None = None,
) -> Generator[GenerationResponse]:
    from exo.worker.engines.mlx.cache import (
        MEMORY_THRESHOLD,
        get_memory_used_percentage,
    )

    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Normalize volatile patterns (session IDs, hashes) before encoding so that
    # identical conversations produce identical tokens regardless of session metadata.
    prompt = normalize_prompt_for_cache(prompt)

    # Encode prompt once at the top and fix unmatched think tags.
    # Touch heartbeat: tokenization of large prompts (90K+ tokens) can take seconds.
    if distributed_prompt_progress_callback is not None:
        distributed_prompt_progress_callback()
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)

    # Strip the generation prefix (e.g. <think>\n) from the cache comparison
    # key.  The chat template appends this to prime the model, but it won't
    # appear at the same position in the next request's prompt (the actual
    # response replaces it).  Keeping it causes a 2-3 token trailing mismatch
    # on every cache lookup.
    think_start_id: int | None = tokenizer.think_start_id
    cache_key_tokens = all_prompt_tokens
    if think_start_id is not None:
        toks = all_prompt_tokens.tolist()
        # Template typically appends <think>\n — strip both if present
        if len(toks) >= 2 and toks[-2] == think_start_id and toks[-1] == 10:
            cache_key_tokens = all_prompt_tokens[:-2]
            logger.info(
                f"Stripped generation prefix <think>\\n from cache key: "
                f"{len(all_prompt_tokens)} -> {len(cache_key_tokens)} tokens"
            )
        elif len(toks) >= 1 and toks[-1] == think_start_id:
            cache_key_tokens = all_prompt_tokens[:-1]
            logger.info(
                f"Stripped generation prefix <think> from cache key: "
                f"{len(all_prompt_tokens)} -> {len(cache_key_tokens)} tokens"
            )

    # Update heartbeat timeout with actual token count now that we know it.
    if on_token_count_known is not None:
        on_token_count_known(len(all_prompt_tokens))

    effective_max_context = task.max_context_tokens

    if effective_max_context is not None and len(all_prompt_tokens) > effective_max_context:
        original_len = len(all_prompt_tokens)
        all_prompt_tokens = _truncate_prompt_tokens(all_prompt_tokens, effective_max_context)
        logger.warning(
            f"Prompt truncated: {original_len} -> {len(all_prompt_tokens)} tokens "
            f"(limit {effective_max_context})"
        )

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench:
        kv_prefix_cache = None

    # Try KV cache lookup BEFORE memory check — reusing the cache means we
    # only need to prefill the delta, which requires far less memory than a
    # full prefill after eviction.
    # Touch heartbeat: cache lookup deepcopies KV state which can be slow.
    if distributed_prompt_progress_callback is not None:
        distributed_prompt_progress_callback()
    prefix_hit_length = 0
    matched_index: int | None = None
    cache_from_prefix = False
    if kv_prefix_cache is None:
        caches = make_kv_cache(model=model)
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index = kv_prefix_cache.get_kv_cache(
            model, all_prompt_tokens
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if prefix_hit_length > 0:
            hit_pct = 100 * prefix_hit_length / len(all_prompt_tokens)
            if hit_pct < 1.0:
                # Near-total cache miss (e.g. compaction changed the system
                # prompt).  The matched prefix is too short to be useful and
                # the stale entry wastes memory.  Evict it and start fresh.
                logger.info(
                    f"KV cache near-miss: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                    f"({hit_pct:.1f}%) — evicting stale entry and using fresh cache"
                )
                kv_prefix_cache.clear()
                mx.clear_cache()
                caches = make_kv_cache(model=model)
                prompt_tokens = all_prompt_tokens
                prefix_hit_length = 0
                matched_index = None
            else:
                logger.info(
                    f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({hit_pct:.1f}%)"
                )
                cache_from_prefix = True

    # Memory check.  If memory is above threshold, evict stored KV cache
    # entries to free memory.  We keep our already-copied prefix match if
    # eviction frees enough — only fall back to a full reprefill when
    # memory is still too high after eviction.
    mem_used = get_memory_used_percentage()
    if mem_used > MEMORY_THRESHOLD:
        if kv_prefix_cache is not None:
            kv_prefix_cache.clear()
            mx.clear_cache()
            mem_used = get_memory_used_percentage()
            logger.info(
                f"Evicted KV cache under memory pressure, now at {mem_used:.0%}"
            )
        if cache_from_prefix:
            # The matched entry was evicted from the store, so the
            # matched_index is stale.  Clear it so the save path uses
            # add_kv_cache instead of update_kv_cache.
            matched_index = None
            if mem_used > MEMORY_THRESHOLD:
                # Still too tight — discard the prefix match entirely.
                caches = make_kv_cache(model=model)
                prompt_tokens = all_prompt_tokens
                prefix_hit_length = 0
                cache_from_prefix = False
            else:
                logger.info(
                    f"Retained prefix hit ({prefix_hit_length} tokens) after eviction"
                )
        if mem_used > MEMORY_THRESHOLD:
            raise ValueError(
                f"memory pressure too high ({mem_used:.0%} used, threshold {MEMORY_THRESHOLD:.0%}): "
                f"cannot accept new request"
            )

    # Touch heartbeat: cache lookup deepcopy of large KV state can take 10s+ seconds.
    if distributed_prompt_progress_callback is not None:
        distributed_prompt_progress_callback()

    # Pre-size KV cache step to avoid reallocation spikes during decode.
    # Default step=256 causes a full-cache copy every 256 tokens; at 100K
    # context each copy costs ~160ms. Setting step to cover the full expected
    # sequence eliminates all mid-generation reallocations.
    max_output = task.max_output_tokens or MAX_TOKENS
    if effective_max_context is not None:
        max_output = min(max_output, effective_max_context - len(all_prompt_tokens))
    max_tokens_estimate = max_output + len(all_prompt_tokens)
    for c in caches:
        if hasattr(c, "step"):
            c.step = max_tokens_estimate

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
        make_logits_processors(
            repetition_penalty=task.repetition_penalty,
            repetition_context_size=task.repetition_context_size,
        )
    )
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)] + logits_processors

    sampler = make_sampler(
        temp=task.temperature if task.temperature is not None else 0.7,
        top_p=task.top_p if task.top_p is not None else 1.0,
        min_p=task.min_p if task.min_p is not None else 0.05,
        top_k=task.top_k if task.top_k is not None else 0,
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    # Start draft prefill in background thread (overlaps with primary prefill)
    # IMPORTANT: use all_prompt_tokens, not prompt_tokens — the latter may be
    # truncated by KV prefix cache hits, but the draft model needs the full prompt.
    _draft_prefill_thread = None
    if draft_model is not None and hasattr(draft_model, 'prefill'):
        import threading
        _draft_client_ref = draft_model
        _draft_client_ref.reset_cache()
        _prompt_token_ids = all_prompt_tokens.tolist() if isinstance(all_prompt_tokens, mx.array) else list(all_prompt_tokens)
        def _bg_draft_prefill():
            try:
                _draft_client_ref.prefill(_prompt_token_ids)
            except Exception as e:
                logger.warning(f"Draft prefill failed: {e}")
        _draft_prefill_thread = threading.Thread(target=_bg_draft_prefill, daemon=True)
        _draft_prefill_thread.start()
        logger.info(f"Draft prefill started in background ({len(_prompt_token_ids)} tokens)")

    # Prefill cache with all tokens except the last one
    prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
        model,
        tokenizer,
        sampler,
        prompt_tokens[:-1],
        caches,
        group,
        on_prefill_progress,
        distributed_prompt_progress_callback,
    )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    # Save the KV cache immediately after prefill so the work is preserved
    # even if generation is cancelled before completion. This is critical for
    # large-context requests (/compact) that may take 5+ minutes to prefill
    # and then get cancelled during decode — without this, the retry has to
    # re-prefill from scratch.
    if kv_prefix_cache is not None:
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
                normalized_tokens=cache_key_tokens,
            )
        else:
            kv_prefix_cache.add_kv_cache(
                all_prompt_tokens, caches, cache_snapshots,
                normalized_tokens=cache_key_tokens,
            )

    # stream_generate starts from the last token
    last_token = prompt_tokens[-2:]

    max_tokens = task.max_output_tokens or MAX_TOKENS
    if effective_max_context is not None:
        remaining = effective_max_context - len(all_prompt_tokens)
        if remaining < max_tokens:
            logger.info(f"Clamping max_tokens from {max_tokens} to {remaining} (context budget)")
            max_tokens = max(1, remaining)
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    total_prompt_tokens = len(all_prompt_tokens)
    first_token_emitted = False
    in_thinking = False
    reasoning_tokens = 0
    think_start = tokenizer.think_start
    think_end = tokenizer.think_end

    # Touch heartbeat: prefill may have taken many seconds.
    if on_generation_token is not None:
        on_generation_token()

    # Flush all streams before decode — same rationale as pre-prefill sync.
    if group is not None:
        mx.synchronize()

    if EXO_TRACING_ENABLED:
        get_pipeline_timings().reset()
        t_barrier = time.perf_counter()
    mx_barrier(group)
    if EXO_TRACING_ENABLED:
        decode_barrier_ms = (time.perf_counter() - t_barrier) * 1000
        logger.info(f"Pre-decode barrier: {decode_barrier_ms:.1f}ms")

    # Touch heartbeat: pre-decode barrier can block waiting for other ranks.
    if on_generation_token is not None:
        on_generation_token()

    if EXO_TRACING_ENABLED:
        logger.info("Starting decode")

    def _log_generation_stats(generated_text_parts: list[str], reason: str) -> None:
        """Log generation stats and save KV cache. Called on both normal completion and abort."""
        generation_elapsed = time.perf_counter() - generation_start_time
        generated_tokens = len(generated_text_parts)
        generation_tps = (
            generated_tokens / generation_elapsed if generation_elapsed > 0 else 0.0
        )
        if EXO_TRACING_ENABLED:
            logger.info(
                f"Generation {reason}: prefill {prefill_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, decoded {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s in {generation_elapsed * 1000:.1f}ms "
                f"({generation_elapsed * 1000 / generated_tokens:.2f}ms/tok)"
                if generated_tokens > 0
                else f"Generation {reason}: prefill {prefill_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, 0 tokens decoded"
            )
            rank = group.rank() if group is not None else 0
            get_pipeline_timings().log_and_reset("decode", rank)
        else:
            logger.debug(
                f"Generation {reason}: prefill {prefill_tokens} tokens @ "
                f"{prefill_tps:.1f} tok/s, generated {generated_tokens} tokens @ "
                f"{generation_tps:.1f} tok/s"
            )

    def _save_kv_cache(generated_text_parts: list[str]) -> None:
        """Save KV cache with generated tokens. Called on both normal completion and abort."""
        if kv_prefix_cache is None or not generated_text_parts:
            return
        # Don't save cache when speculative decode was active — the sequential
        # verify path may leave the cache in a state that doesn't exactly match
        # the token sequence (off-by-one from draft/verify transitions).
        if _draft_fn is not None:
            return
        try:
            if EXO_TRACING_ENABLED:
                t_cache_update = time.perf_counter()
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
                    full_prompt_tokens,
                    caches,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                    normalized_tokens=cache_key_tokens,
                )
            else:
                kv_prefix_cache.add_kv_cache(
                    full_prompt_tokens, caches, cache_snapshots,
                    normalized_tokens=cache_key_tokens,
                )
            if EXO_TRACING_ENABLED:
                logger.info(
                    f"KV prefix cache update took "
                    f"{(time.perf_counter() - t_cache_update) * 1000:.1f}ms"
                )
        except Exception:
            logger.warning("Failed to save KV cache after generation", exc_info=True)

    # Speculative decoding: TCP draft server or CPU draft engine
    _draft_fn = None
    _cpu_draft = None
    if draft_model is not None and hasattr(draft_model, 'prefill'):
        # TCP DraftClient — use speculative_generate_step via draft_fn
        _draft_client = draft_model
        # Use instance-configured draft tokens (from UI) rather than env var default
        _tcp_draft_tokens = getattr(_draft_client, 'num_draft_tokens', _SPECULATIVE_DRAFT_TOKENS)
        # Wait for background draft prefill to finish before decode starts
        if _draft_prefill_thread is not None:
            _draft_prefill_thread.join()
            logger.info("Draft prefill complete (was overlapped with primary prefill)")
        def _tcp_draft_fn(token_id: int, num_tokens: int, trim: int = 0) -> list:
            """Blocking call to remote draft server."""
            return _draft_client.fetch_draft_sync(token_id, num_tokens, trim=trim)
        _draft_fn = _tcp_draft_fn
        logger.info(
            f"TCP speculative decode active: server={_draft_client.server_url}, "
            f"K={_tcp_draft_tokens}"
        )
    elif draft_model is not None and hasattr(draft_model, 'draft_sync'):
        _cpu_draft = draft_model
        logger.info(
            f"CPU-pipelined speculative decode active: "
            f"layers={_cpu_draft._n_layers}, draft_tokens={_SPECULATIVE_DRAFT_TOKENS}"
        )

    _generation_logged = False
    try:
      _gen_kwargs = dict(
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
      )
      if _draft_fn is not None:
          _gen_kwargs["draft_fn"] = _draft_fn
          _gen_kwargs["num_draft_tokens"] = _tcp_draft_tokens
      elif _cpu_draft is not None:
          _cpu_draft.reset_cache()
          # Feed prompt tokens to CPU draft model's cache
          if isinstance(last_token, mx.array):
              for t in last_token.tolist():
                  _cpu_draft._forward_one(t)
          _gen_kwargs["cpu_draft_fn"] = _cpu_draft.draft_sync
          _gen_kwargs["num_draft_tokens"] = _SPECULATIVE_DRAFT_TOKENS

      _accepted_draft_tokens = 0
      for completion_tokens, out in enumerate(
        stream_generate(**_gen_kwargs),
        start=1,
      ):
        if out.from_draft:
            _accepted_draft_tokens += 1
        generated_text_parts.append(out.text)
        accumulated_text += out.text

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

        # Stop generation if total context (prompt + completion) exceeds limit.
        if (
            effective_max_context is not None
            and finish_reason is None
            and total_prompt_tokens + completion_tokens >= effective_max_context
        ):
            finish_reason = "context_window_exceeded"

        is_done = finish_reason is not None

        stats: GenerationStats | None = None
        if is_done:
            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
            )
            if not stop_matched and out.finish_reason not in get_args(FinishReason):
                logger.warning(
                    f"Model generated unexpected finish_reason: {out.finish_reason}"
                )

            if _accepted_draft_tokens > 0:
                logger.info(
                    f"Speculative decode: {_accepted_draft_tokens}/{completion_tokens} tokens "
                    f"from draft ({_accepted_draft_tokens / completion_tokens * 100:.1f}%)"
                )

            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens,
                    accepted_prediction_tokens=_accepted_draft_tokens,
                ),
            )

        # Emit prompt_tokens on the first token so streaming adapters can
        # populate input_tokens in message_start without waiting for the final chunk.
        if not first_token_emitted and not is_done:
            first_token_emitted = True
            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=0,
                total_tokens=total_prompt_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=prefix_hit_length
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0
                ),
            )

        # Extract logprobs from the full vocabulary logprobs array
        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if task.logprobs:
            logprob, top_logprobs = extract_top_logprobs(
                logprobs=out.logprobs,
                tokenizer=tokenizer,
                top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                selected_token=out.token,
            )

        if is_done:
            _log_generation_stats(generated_text_parts, "complete")
            _save_kv_cache(generated_text_parts)
            _generation_logged = True

        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(
            text=text,
            token=out.token,
            logprob=logprob,
            top_logprobs=top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
            is_thinking=in_thinking,
        )

        # Reset usage after first-token emission so only the first and final
        # tokens carry Usage objects. This avoids per-token overhead.
        if usage is not None and not is_done:
            usage = None

        if is_done:
            # Touch heartbeat: KV cache update above can take 10s+ seconds.
            if on_generation_token is not None:
                on_generation_token()
            if EXO_TRACING_ENABLED:
                t_barrier = time.perf_counter()
            mx_barrier(group)
            if EXO_TRACING_ENABLED:
                logger.info(
                    f"Post-decode barrier: {(time.perf_counter() - t_barrier) * 1000:.1f}ms"
                )
            # Touch heartbeat: post-decode barrier can block waiting for other ranks.
            if on_generation_token is not None:
                on_generation_token()
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
    except GeneratorExit:
        if not _generation_logged and generated_text_parts:
            _log_generation_stats(generated_text_parts, "aborted (GeneratorExit)")
            _save_kv_cache(generated_text_parts)
            _generation_logged = True
    finally:
        if not _generation_logged and generated_text_parts:
            _log_generation_stats(generated_text_parts, "aborted")
            _save_kv_cache(generated_text_parts)

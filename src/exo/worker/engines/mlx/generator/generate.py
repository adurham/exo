import time
from copy import deepcopy
from typing import Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, stream_generate
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
from exo.worker.engines.mlx.auto_parallel import set_pipeline_prefill
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
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
_DEFAULT_COMPLETION_BATCH_SIZE = 8
_DEFAULT_PREFILL_BATCH_SIZE = 8


def _env_int(name: str, default: int) -> int:
    try:
        return int((__import__("os").environ.get(name) or "").strip() or default)
    except Exception:
        return default


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


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    Returns:
        tokens_per_sec
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []

    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            snapshots.append(snapshot_ssm_states(cache))

    set_pipeline_prefill(model, is_prefill=True)

    # Use max_tokens=1 because max_tokens=0 does not work.
    # We just throw away the generated token - we only care about filling the cache
    for _ in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=1,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=512,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
        prompt_progress_callback=progress_callback,
    ):
        break  # Stop after first iteration - cache is now filled

    set_pipeline_prefill(model, is_prefill=False)

    # stream_generate added 1 extra generated token to the cache, so we should trim it.
    # Because of needing to roll back arrays cache, we will generate on 2 tokens so trim 1 more.
    pre_gen = deepcopy(snapshots[-2]) if has_ssm else None
    for i, c in enumerate(cache):
        if has_ssm and isinstance(c, (ArraysCache, RotatingKVCache)):
            assert pre_gen is not None
            if pre_gen.states[i] is not None:
                cache[i] = deepcopy(pre_gen.states[i])  # type: ignore
        else:
            assert not isinstance(c, (ArraysCache, RotatingKVCache))
            c.trim(2)  # pyright: ignore[reportUnknownMemberType]

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
    group: mx.distributed.Group | None = None,
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


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None = None,
    group: mx.distributed.Group | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # TODO: Randomise task seed and set in taskparams, instead of hard coding as 42.
    seed = task.seed or 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)

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
            model, all_prompt_tokens
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
    )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    # stream_generate starts from the last token
    y = prompt_tokens[-1:]
    
    # Custom compiled step function for faster generation
    @mx.compile
    def step(input_tokens, cache):
        logits = model(input_tokens, cache=cache)
        return logits[:, -1, :]

    def sample(logits):
        return sampler(logits)

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None
    in_thinking = False
    reasoning_tokens = 0
    think_start = tokenizer.think_start
    think_end = tokenizer.think_end
    
    # Maintain full token history for logits processors
    # Note: prompt_tokens includes the last token which we are just about to process?
    # No, stream_generate takes prefilled cache and the *last token* as input prompt to start generation.
    # The cache contains everything UP TO the last token.
    # So 'y' is the last token of the prompt.
    tokens = prompt_tokens 

    mx_barrier(group)

    logger.debug(f"Tokenizer EOS IDs: {getattr(tokenizer, 'eos_token_ids', 'Not Set')}")
    logger.debug("Starting compiled generation loop...")
    
    completion_tokens = 0
    while True:
        if completion_tokens >= max_tokens:
            finish_reason = "length"
            break
            
        # 1. Forward pass (compiled)
        # cache is updated in-place
        logits = step(y[None], cache=caches)
        
        # 2. Logits processing (uncompiled, usually fast)
        # Logits processors expect (tokens, logits). 'tokens' should include the *new* token?
        # In mlx_lm: "tokens = mx.concat([tokens, input_tokens])" before processing
        # But here 'y' is the input token for this step.
        # So we append y to tokens history?
        # mlx_lm does: tokens = concat(tokens, input_tokens) if tokens else input_tokens
        # We start with prompt_tokens.
        # Check if we need to update tokens array. Logits processors might need it.
        # For ban_token_ids it doesn't use history.
        if logits_processors:
            # Only construct full history if needed (optimization)
            # But tokens array growth is expensive? mlx_lm does it.
            # We already have 'tokens' as array.
            # tokens = mx.concatenate([tokens, y]) # This is slow if done every step?
            # actually mlx_lm does it.
            pass 

        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        
        # 3. Sampling
        # Sampler returns (token, logprobs) or just token? 
        # exo make_sampler returns a function that does:
        #   return mx.random.categorical(logits * (1/temp)) ...
        # logic from mlx_lm.sample_utils.make_sampler:
        #   def sample(logits): ... return token
        # It does NOT return logprobs.
        new_token_id = sample(logprobs) # index
        
        # 4. Detokenization and yielding
        # We need the text.
        token_val = new_token_id.item()
        out_text = tokenizer.decode([token_val])
        
        # Update history
        y = new_token_id.reshape(1)
        tokens = mx.concatenate([tokens, y])
        completion_tokens += 1
        
        # 5. Stop conditions
        finish_reason = None
        if token_val in getattr(tokenizer, 'eos_token_ids', []):
            finish_reason = "stop"
        
        # Yield response
        logger.debug(f"Gen token [{completion_tokens}]: {token_val} | {out_text}")
        generated_text_parts.append(out_text)
        accumulated_text += out_text
        
        if think_start is not None and out_text == think_start:
            in_thinking = True
        elif think_end is not None and out_text == think_end:
            in_thinking = False
        if in_thinking:
            reasoning_tokens += 1
            
        # Check stop sequences
        stop_matched = False
        if stop_sequences:
            for stop_seq in stop_sequences:
                 if stop_seq in accumulated_text:
                    finish_reason = "stop"
                    stop_matched = True
                    # Trim logic... matches original code
                    stop_index = accumulated_text.find(stop_seq)
                    text_before_stop = accumulated_text[:stop_index]
                    chunk_start = len(accumulated_text) - len(out_text)
                    out_text = text_before_stop[chunk_start:] # Correct the output chunk if needed
                    break
        
        is_done = finish_reason is not None
        
        # Logprobs extraction
        # Reuse existing logic
        r_logprob = None
        r_top_logprobs = None
        if task.logprobs:
            r_logprob, r_top_logprobs = extract_top_logprobs(
                logprobs=logprobs[0], # unbatch
                tokenizer=tokenizer,
                top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                selected_token=new_token_id
            )

        # Stats
        stats = None
        if is_done:
            # We need prompt_tps etc.
            # Calculate them
            generation_elapsed = time.perf_counter() - generation_start_time
            generation_tps = completion_tokens / generation_elapsed if generation_elapsed > 0 else 0
            stats = GenerationStats(
                prompt_tps=float(prefill_tps),
                generation_tps=float(generation_tps),
                prompt_tokens=int(prefill_tokens + completion_tokens), # total?
                generation_tokens=int(completion_tokens),
                peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
            )
            usage = Usage(
                prompt_tokens=int(prefill_tokens),
                completion_tokens=completion_tokens,
                total_tokens=int(prefill_tokens) + completion_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=prefix_hit_length),
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning_tokens),
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
                if (
                    matched_index is not None
                    and prefix_hit_length >= _MIN_PREFIX_HIT_TO_UPDATE
                ):
                    kv_prefix_cache.update_kv_cache(
                        matched_index,
                        full_prompt_tokens,
                        caches,
                        cache_snapshots,
                        restore_pos=prefix_hit_length,
                    )
                else:
                    kv_prefix_cache.add_kv_cache(
                        full_prompt_tokens, caches, cache_snapshots
                    )

        yield GenerationResponse(
            text=out_text,
            token=token_val,
            logprob=r_logprob,
            top_logprobs=r_top_logprobs,
            finish_reason=finish_reason,
            stats=stats,
            usage=usage,
        )

        if is_done:
            mx_barrier(group)
            break
            
        # Update cache for next step happens implicitly in model() call?
        # Yes, caches are mutable and updated in step().
        
        # Limit accumulated_text
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]

        # Eval to keep loop non-blocking but synced?
        # Assumed mx.compile handles sync or we use mx.async_eval(y)
        # stream_generate uses mx.async_eval(y, logprobs)
        mx.async_eval(y)

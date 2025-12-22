from typing import Any, Callable, Generator, cast, get_args

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper

# from exo.engines.mlx.cache import KVPrefixCache
from exo.shared.types.api import ChatCompletionMessage, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.constants import KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    make_kv_cache,
    mx_barrier,
)
from exo.worker.runner.bootstrap import logger

try:
    import psutil
except ImportError:
    psutil = None

generation_stream = mx.new_stream(mx.default_device())


def maybe_quantize_kv_cache(
    prompt_cache: list[KVCache | Any],
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: int | None,
) -> None:
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if (
            hasattr(c, "to_quantized") and c.offset >= quantized_kv_start  # type: ignore
        ):
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
) -> int:
    import time
    import os
    
    # Get rank info for logging
    try:
        import mlx.core as mx
        rank = mx.distributed.rank() if mx.distributed.is_available() else -1
        world_size = mx.distributed.size() if mx.distributed.is_available() else -1
        logger.info(f"[RANK {rank}/{world_size}] ========== WARMUP STARTING ==========")
    except:
        logger.info("[RANK UNKNOWN] ========== WARMUP STARTING ==========")
        rank = -1
        world_size = -1
    
    content = "Prompt to warm up the inference engine. Repeat this."

    logger.info(f"[RANK {rank}] Starting warmup: applying chat template")
    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=ChatCompletionTaskParams(
            model="",
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content=content,
                )
            ],
        ),
    )
    logger.info(f"[RANK {rank}] Warmup prompt created (length: {len(warmup_prompt)})")

    tokens_generated = 0

    logger.info(f"[RANK {rank}] Creating KV cache for warmup")
    cache = make_kv_cache(
        model=model,
    )
    logger.info(f"[RANK {rank}] KV cache created, starting token generation")

    warmup_start_time = time.time()
    logger.info(f"[RANK {rank}] Generating warmup tokens (max 50)")
    
    # Log memory and performance info before warmup
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        swap_info = psutil.swap_memory()
        logger.info(
            f"[RANK {rank}] Pre-warmup memory: RSS={mem_info.rss / 1024 / 1024 / 1024:.2f}GB, "
            f"VMS={mem_info.vms / 1024 / 1024 / 1024:.2f}GB. "
            f"Swap used: {swap_info.used / 1024 / 1024 / 1024:.2f}GB"
        )
        if swap_info.used > 0:
            logger.warning(f"[RANK {rank}] SWAP DETECTED before warmup! This can severely impact performance.")
    
    logger.info(f"[RANK {rank}] Creating stream_generate iterator...")
    iterator = stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        max_tokens=50,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=65536,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    )
    logger.info(f"[RANK {rank}] Created stream_generate iterator, starting iteration loop")
    
    token_times = []
    last_log_time = warmup_start_time
    iteration_count = 0
    last_iteration_time = warmup_start_time
    
    logger.info(f"[RANK {rank}] ========== ENTERING ITERATOR LOOP ==========")
    
    # In pipeline parallelism, stream_generate only yields on Rank 0.
    # Non-rank-0 ranks need to manually drive the iterator to participate in forward passes.
    # We'll use next() with a sentinel to handle the case where non-rank-0 ranks don't get results.
    max_iterations = 50
    iterator_exhausted = False
    
    try:
        while tokens_generated < max_iterations and not iterator_exhausted:
            iteration_count += 1
            iteration_start_time = time.time()
            time_since_last_iter = iteration_start_time - last_iteration_time
            
            logger.info(
                f"[RANK {rank}] Attempting iteration #{iteration_count} "
                f"(tokens_generated={tokens_generated}/{max_iterations}, "
                f"time_since_last={time_since_last_iter:.3f}s)"
            )
            
            try:
                # Try to get next result from iterator
                # On Rank 0, this will yield a result
                # On non-rank-0 ranks, this will block until Rank 0's iteration completes
                # but may not yield a result (depending on mlx_lm implementation)
                _r = next(iterator)
                iteration_time = time.time()
                time_since_last_iter = iteration_time - last_iteration_time
                last_iteration_time = iteration_time
                
                # Log every iteration with full details
                has_text = hasattr(_r, 'text') and _r.text
                finish_reason = _r.finish_reason if hasattr(_r, 'finish_reason') else 'N/A'
                text_preview = _r.text[:50] if has_text else 'N/A'
                
                logger.info(
                    f"[RANK {rank}] ========== ITERATION #{iteration_count} SUCCESS =========="
                )
                logger.info(
                    f"[RANK {rank}] Iteration #{iteration_count} details: "
                    f"text='{text_preview}', finish_reason={finish_reason}, "
                    f"time_since_last={time_since_last_iter:.3f}s, "
                    f"elapsed={iteration_time - warmup_start_time:.2f}s"
                )
                
                tokens_generated += 1
                token_time = iteration_time
                elapsed = token_time - warmup_start_time
                time_since_last_log = token_time - last_log_time
                token_times.append(token_time)
                
                # Log every token, and also log memory every 10 tokens
                # Only log token text if it exists (Rank 0 will have text, other ranks may not)
                if has_text:
                    logger.info(
                        f"[RANK {rank}] Generated warmup token #{tokens_generated}: '{_r.text}' "
                        f"(finish_reason={_r.finish_reason}, elapsed: {elapsed:.2f}s, "
                        f"time_since_last: {time_since_last_log:.2f}s)"
                    )
                else:
                    logger.info(
                        f"[RANK {rank}] Processed warmup activation #{tokens_generated} "
                        f"(no token text, finish_reason={finish_reason}, elapsed: {elapsed:.2f}s, "
                        f"time_since_last: {time_since_last_log:.2f}s)"
                    )
                last_log_time = token_time
                
            except StopIteration:
                logger.info(f"[RANK {rank}] Iterator exhausted (StopIteration) at iteration #{iteration_count}")
                iterator_exhausted = True
                break
            except Exception as e:
                logger.error(f"[RANK {rank}] Error getting next iteration: {e}", exc_info=True)
                raise
            
            # Log if we're taking too long between iterations
            if time_since_last_iter > 1.0:
                logger.warning(
                    f"[RANK {rank}] WARNING: Long delay between iterations! "
                    f"{time_since_last_iter:.2f}s since last iteration"
                )
            
            # Log memory every 10 tokens
            if psutil is not None and tokens_generated % 10 == 0:
                mem_info = process.memory_info()
                swap_info = psutil.swap_memory()
                logger.info(
                    f"[RANK {rank}] Token #{tokens_generated} memory: RSS={mem_info.rss / 1024 / 1024 / 1024:.2f}GB. "
                    f"Swap used: {swap_info.used / 1024 / 1024 / 1024:.2f}GB"
                )
                if swap_info.used > 0:
                    logger.warning(
                        f"[RANK {rank}] Token #{tokens_generated} SWAP USAGE DETECTED: {swap_info.used / 1024 / 1024 / 1024:.2f}GB"
                    )
            
            # Stop when we've generated max_tokens
            if tokens_generated >= max_iterations:
                logger.info(f"[RANK {rank}] Reached max_tokens limit ({tokens_generated}), exiting loop")
                break
                
    except Exception as e:
        logger.error(f"[RANK {rank}] Error during warmup token generation: {e}", exc_info=True)
        raise

    logger.info(f"[RANK {rank}] ========== EXITED ITERATOR LOOP ==========")
    logger.info(f"[RANK {rank}] Exited stream_generate loop after {tokens_generated} tokens (total iterations: {iteration_count})")
    generation_time = time.time() - warmup_start_time
    
    # Calculate token generation rate
    if len(token_times) > 1:
        avg_token_time = sum(token_times[i] - token_times[i-1] for i in range(1, len(token_times))) / (len(token_times) - 1)
        logger.info(f"[RANK {rank}] Average time per token: {avg_token_time:.3f}s ({1/avg_token_time:.2f} tokens/sec)")
    
    logger.info(
        f"[RANK {rank}] Generated ALL {tokens_generated} warmup tokens in {generation_time:.2f}s, "
        f"waiting for barrier"
    )
    
    # Log memory before barrier
    if psutil is not None:
        mem_info = process.memory_info()
        swap_info = psutil.swap_memory()
        logger.info(
            f"[RANK {rank}] Pre-barrier memory: RSS={mem_info.rss / 1024 / 1024 / 1024:.2f}GB. "
            f"Swap used: {swap_info.used / 1024 / 1024 / 1024:.2f}GB"
        )
        if swap_info.used > 0:
            logger.warning(
                f"[RANK {rank}] Pre-barrier SWAP USAGE DETECTED: {swap_info.used / 1024 / 1024 / 1024:.2f}GB"
            )
    
    barrier_start_time = time.time()
    logger.info(f"[RANK {rank}] ========== ENTERING BARRIER ==========")
    logger.info(f"[RANK {rank}] Entering barrier at {barrier_start_time:.2f}")
    try:
        mx_barrier()
        barrier_time = time.time() - barrier_start_time
        logger.info(f"[RANK {rank}] ========== EXITED BARRIER ==========")
        logger.info(f"[RANK {rank}] Warmup barrier completed in {barrier_time:.2f}s")
    except Exception as e:
        logger.error(f"[RANK {rank}] ERROR in barrier: {e}", exc_info=True)
        raise

    total_time = time.time() - warmup_start_time
    logger.info(f"[RANK {rank}] ========== WARMUP COMPLETED ==========")
    logger.info(f"[RANK {rank}] Warmup completed: {tokens_generated} tokens in {total_time:.2f}s total")

    return tokens_generated


def mlx_generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse]:
    # Currently we support chat-completion tasks only.
    logger.info(f"task_params: {task}")

    prompt = apply_chat_template(
        tokenizer=tokenizer,
        chat_task_data=task,
    )

    caches = make_kv_cache(model=model)

    max_tokens = task.max_tokens or MAX_TOKENS
    token_count = 0
    logger.info(f"Starting generation: max_tokens={max_tokens}, prompt_length={len(prompt)}")
    for out in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=caches,
        prefill_step_size=65536,
        kv_group_size=KV_GROUP_SIZE,
        kv_bits=KV_BITS,
    ):
        token_count += 1
        logger.info(f"Generated token #{token_count}: text='{out.text}', token_id={out.token}, finish_reason={out.finish_reason}")
        if out.finish_reason is not None and out.finish_reason not in get_args(
            FinishReason
        ):
            # We don't throw here as this failure case is really not all that bad
            # Just log the error and move on
            logger.warning(
                f"Model generated unexpected finish_reason: {out.finish_reason}"
            )

        yield GenerationResponse(
            text=out.text,
            token=out.token,
            finish_reason=cast(FinishReason | None, out.finish_reason),
        )

        if out.finish_reason is not None:
            break

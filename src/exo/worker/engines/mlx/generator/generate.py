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
    
    content = "Prompt to warm up the inference engine. Repeat this."

    logger.info("Starting warmup: applying chat template")
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
    logger.info(f"Warmup prompt created (length: {len(warmup_prompt)})")

    tokens_generated = 0

    logger.info("Creating KV cache for warmup")
    cache = make_kv_cache(
        model=model,
    )
    logger.info("KV cache created, starting token generation")

    warmup_start_time = time.time()
    logger.info("Generating warmup tokens (max 50)")
    
    # Log memory and performance info before warmup
    import os
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        swap_info = process.memory_full_info() if hasattr(process, 'memory_full_info') else None
        logger.info(
            f"Pre-warmup memory: RSS={mem_info.rss / 1024 / 1024 / 1024:.2f}GB, "
            f"VMS={mem_info.vms / 1024 / 1024 / 1024:.2f}GB"
        )
        if swap_info:
            logger.info(
                f"Pre-warmup swap: {swap_info.swap / 1024 / 1024 / 1024:.2f}GB"
            )
    
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
    logger.info("Created stream_generate iterator, starting iteration")
    
    token_times = []
    try:
        for _r in iterator:
            tokens_generated += 1
            token_time = time.time()
            elapsed = token_time - warmup_start_time
            token_times.append(token_time)
            
            # Log memory every 10 tokens
            if psutil is not None and tokens_generated % 10 == 0:
                mem_info = process.memory_info()
                swap_info = process.memory_full_info() if hasattr(process, 'memory_full_info') else None
                logger.info(
                    f"Token #{tokens_generated} memory: RSS={mem_info.rss / 1024 / 1024 / 1024:.2f}GB"
                )
                if swap_info and swap_info.swap > 0:
                    logger.warning(
                        f"Token #{tokens_generated} SWAP USAGE DETECTED: {swap_info.swap / 1024 / 1024 / 1024:.2f}GB"
                    )
            
            logger.info(
                f"Generated warmup token #{tokens_generated}: '{_r.text}' "
                f"(finish_reason={_r.finish_reason}, elapsed: {elapsed:.2f}s)"
            )
            # Stop when we've generated max_tokens
            if tokens_generated >= 50:
                logger.info(f"Reached max_tokens limit ({tokens_generated}), exiting loop")
                break
    except StopIteration:
        logger.info(f"Iterator exhausted (StopIteration) after {tokens_generated} tokens")
    except Exception as e:
        logger.error(f"Error during warmup token generation: {e}", exc_info=True)
        raise

    logger.info(f"Exited stream_generate loop after {tokens_generated} tokens")
    generation_time = time.time() - warmup_start_time
    
    # Calculate token generation rate
    if len(token_times) > 1:
        avg_token_time = sum(token_times[i] - token_times[i-1] for i in range(1, len(token_times))) / (len(token_times) - 1)
        logger.info(f"Average time per token: {avg_token_time:.3f}s ({1/avg_token_time:.2f} tokens/sec)")
    
    logger.info(
        f"Generated ALL {tokens_generated} warmup tokens in {generation_time:.2f}s, "
        f"waiting for barrier"
    )
    
    # Log memory before barrier
    if psutil is not None:
        mem_info = process.memory_info()
        swap_info = process.memory_full_info() if hasattr(process, 'memory_full_info') else None
        logger.info(
            f"Pre-barrier memory: RSS={mem_info.rss / 1024 / 1024 / 1024:.2f}GB"
        )
        if swap_info and swap_info.swap > 0:
            logger.warning(
                f"Pre-barrier SWAP USAGE DETECTED: {swap_info.swap / 1024 / 1024 / 1024:.2f}GB"
            )
    
    barrier_start_time = time.time()
    logger.info(f"Entering barrier at {barrier_start_time:.2f}")
    mx_barrier()
    barrier_time = time.time() - barrier_start_time
    logger.info(f"Warmup barrier completed in {barrier_time:.2f}s")

    total_time = time.time() - warmup_start_time
    logger.info(f"Warmup completed: {tokens_generated} tokens in {total_time:.2f}s total")

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

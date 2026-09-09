# The `logger` re-exported by `exo.worker.runner.bootstrap` is annotated
# `loguru.Logger`, but loguru ships no stubs in this environment, so
# basedpyright resolves it as unknown. Repo-wide pre-existing condition (see
# deepseek_v4_vision.py's identical note); suppressed narrowly, logger only.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
import time

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    cache_length,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.generator.generate import prefill as mlx_prefill
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.bootstrap import logger


class RemotePrefillVisionUnsupportedError(RuntimeError):
    """A vision request was routed to a remote prefill server.

    Phase 4d residual gap (2026-09-09). This is not a chunk-scheduling problem
    that a guard could schedule around -- the disaggregated prefill path has no
    vision capability at all, and the missing piece is the image EMBEDDINGS
    rather than their token positions.

    On the client, a vision request's real image embeddings are produced by the
    vision tower into ``VisionResult.embeddings`` and spliced into the forward
    pass by the ``patch_embed_tokens`` context manager, which monkeypatches
    ``embed_tokens`` on the LOCAL model object. ``PrefillRequest`` carries
    ``token_ids`` and ``start_pos`` and nothing else, so a prefill server
    receives no embeddings, cannot reconstruct them, and never installs that
    patch. Under DeepSeek-V4's scheme the image tokens are sentinel ids at
    ``vocab_size + {0..4}``, deliberately OUTSIDE the embedding table, so the
    server's ``embed_tokens`` would gather out-of-range rows; under the mlx-vlm
    scheme they are an in-vocabulary placeholder whose row is meaningless.

    Neither case raises on its own. The out-of-range gather returns zeros in
    practice (see ``deepseek_v4_vision.build_embeddings``' note that this is
    observed MLX behaviour, not a documented guarantee), so without this check
    the server would return a confidently wrong KV cache and the client's
    ``except Exception: fall back to local prefill`` would never fire.

    ``should_use_remote_prefill`` is where this is actually PREVENTED -- it
    refuses to route a vision request here in the first place. This error is
    the boundary check behind it, so a client that bypasses that routing rule
    fails loudly instead of silently corrupting the cache.
    """


def _embedding_table_size(model: Model) -> int | None:
    """Row count of the model's input embedding table, or ``None``.

    ``None`` means "could not determine", not "no limit": test doubles and any
    architecture ``get_inner_model`` cannot walk both land here. The caller
    treats that as "cannot check" and says so in the log rather than inventing
    a bound -- ``should_use_remote_prefill`` is the guarantee, this is defence
    in depth behind it.
    """
    from exo.worker.engines.mlx.vision import get_inner_model

    try:
        inner = get_inner_model(model)  # type: ignore[reportUnknownArgumentType]
        weight = inner.embed_tokens.weight  # type: ignore[reportUnknownMemberType]
        # Quantized embeddings pack along the FEATURE axis only, so axis 0 is
        # the vocabulary count for both the plain and quantized layers.
        return int(weight.shape[0])  # type: ignore[reportUnknownArgumentType]
    except Exception:
        return None


def _reject_if_vision_request(model: Model, request: PrefillRequest) -> None:
    """Refuse a request whose tokens cannot be embedded by this server.

    The test is the exact failure mechanism rather than a proxy for it: a token
    id at or past the embedding table's row count is precisely the out-of-range
    gather that silently returns zeros. DeepSeek-V4's image sentinels
    (``vocab_size + {0..4}``) are that by construction.

    DETECTION LIMIT, stated rather than papered over: the mlx-vlm vision scheme
    uses an IN-vocabulary ``image_token_id``, which no embedding-table bound can
    detect. Those models have the identical embedding-transport problem, and
    they are covered by ``should_use_remote_prefill``'s ``has_vision``
    exclusion, which is keyed on "is this a vision request at all" and is
    therefore scheme-agnostic. Closing this second gap authoritatively would
    mean putting the image spans (and ultimately the embeddings) on the wire,
    which is the feature this is deliberately not.
    """
    vocabulary_size = _embedding_table_size(model)
    if vocabulary_size is None:
        logger.debug(
            "Prefill server could not resolve an embedding table size for "
            f"{type(model).__name__}; skipping the vision-request check for "
            f"request_id={request.request_id}"
        )
        return

    out_of_range = max(request.token_ids, default=-1)
    if out_of_range < vocabulary_size:
        return

    offending = [t for t in request.token_ids if t >= vocabulary_size]
    raise RemotePrefillVisionUnsupportedError(
        f"prefill request {request.request_id!r} carries "
        f"{len(offending)} token id(s) at or past the embedding table's "
        f"{vocabulary_size} rows (highest: {out_of_range}). Those are image "
        "sentinel tokens, and remote prefill cannot serve them: the vision "
        "tower's embeddings are computed on the requesting node and spliced "
        "in by patch_embed_tokens, and PrefillRequest has no field to carry "
        "them. Prefill this request locally -- should_use_remote_prefill "
        "already declines to route vision requests here, so reaching this "
        "means a client bypassed it."
    )


def run_prefill_for_request(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None,
    request: PrefillRequest,
) -> KVCacheType:
    # Phase 4d residual gap: fail loudly BEFORE touching the prefix cache or
    # running a forward pass. Both would otherwise consume the unembeddable
    # tokens -- and `get_kv_cache` is called here without `media_regions`, so
    # the image-span clamp added to it cannot fire on this side either.
    _reject_if_vision_request(model, request)

    prompt_tokens = mx.array(request.token_ids)
    prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
    n_tokens = int(prompt_tokens.shape[0])
    t0 = time.perf_counter()

    matched_index: int | None = None
    prefix_hit_length = 0
    if kv_prefix_cache is not None:
        cache, remaining, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        prefix_hit_length = n_tokens - int(remaining.shape[0])
    else:
        cache = make_kv_cache(model)
        remaining = prompt_tokens

    target_offset = max(0, n_tokens - 2)
    new_tokens = max(0, target_offset - prefix_hit_length)
    prefill_input = remaining[:new_tokens]
    if int(prefill_input.shape[0]) > 0:
        sampler = make_sampler(temp=1.0)
        _ = mlx_prefill(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            prompt_tokens=prefill_input,
            cache=cache,
            group=group,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

    if kv_prefix_cache is not None:
        try:
            # target_offset is the real absolute token position this cache
            # holds after the prefill call above (prefix_hit_length carried
            # over + new_tokens just prefilled) -- the authoritative count,
            # not a cache-internal .size() derivation (see
            # snapshot_ssm_states' docstring for why that used to be wrong
            # for CacheList-composed non-sliceable layers).
            cache_snapshots = [snapshot_ssm_states(cache, target_offset)]
            hit_ratio = prefix_hit_length / n_tokens if n_tokens > 0 else 0.0
            if matched_index is not None and hit_ratio >= 0.5:
                kv_prefix_cache.update_kv_cache(
                    matched_index,
                    prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                )
            else:
                kv_prefix_cache.add_kv_cache(prompt_tokens, cache, cache_snapshots)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to save prefix cache on prefill server"
            )

    elapsed = time.perf_counter() - t0
    final_offset = cache_length(cache)
    logger.info(
        f"Prefill: request_id={request.request_id} "
        f"{n_tokens} tokens (prefix_hit={prefix_hit_length}, "
        f"final_offset={final_offset}) in {elapsed * 1000:.0f}ms"
    )
    return cache

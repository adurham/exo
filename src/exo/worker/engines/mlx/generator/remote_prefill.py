import time
from collections.abc import Callable
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.protocol import Header, KVChunk
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import CacheSnapshot, snapshot_ssm_states
from exo.worker.engines.mlx.disaggregated.client import (
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger

#: Below this many uncached prompt tokens, shipping a KV cache over the wire
#: costs more than prefilling locally. Previously duplicated as a literal in
#: both ``generator/generate.py`` and ``generator/batch_generate.py``; both now
#: re-export this one definition so the routing rule has a single home
#: alongside ``should_use_remote_prefill``.
REMOTE_PREFILL_MIN_TOKENS = 1000


def should_use_remote_prefill(
    *,
    uncached_token_count: int,
    prefill_endpoint: str | None,
    has_vision: bool,
) -> bool:
    """Whether this request may be prefilled on a remote prefill server.

    Phase 4d residual gap (2026-09-09). ``has_vision`` is the NEW term, and it
    is a hard exclusion rather than a preference: **remote prefill cannot serve
    a vision request at all**, and the reason is structural, not a missing
    chunk guard.

    A vision request's image tokens carry no usable embedding of their own.
    Under the DeepSeek-V4 scheme they are five SENTINEL ids at
    ``vocab_size + {0..4}``, deliberately outside the embedding table; under
    the mlx-vlm scheme they are a repeated in-vocabulary placeholder whose
    table row is meaningless. Either way the REAL embeddings are computed by
    the vision tower into ``VisionResult.embeddings`` and spliced into the
    forward pass by the ``patch_embed_tokens`` context manager, which patches
    ``embed_tokens`` on the LOCAL model object.

    ``PrefillRequest`` carries ``token_ids`` and ``start_pos`` and nothing
    else, so the prefill server receives no embeddings, cannot reconstruct
    them, and never installs ``patch_embed_tokens``. Its ``embed_tokens`` call
    would therefore look up rows that are either out-of-range (DSv4 sentinels)
    or meaningless (mlx-vlm placeholder) and prefill a KV cache the client then
    ingests as authoritative. Nothing on either side raises: the caller's
    ``except Exception: fall back to local prefill`` never fires, because there
    is no exception -- just wrong tokens.

    So the fix is to not route there, decided HERE, at the point that owns the
    local-vs-remote choice. ``run_prefill_for_request`` additionally rejects
    such a request server-side, so a client that bypasses this fails loudly
    rather than silently; see that function for the boundary check.

    Supporting vision remotely means putting the embedding tensor on the wire,
    installing ``patch_embed_tokens`` server-side, and threading
    ``media_regions`` into both the server's chunk planner and its prefix
    cache. That is a feature, not this gap.
    """
    if has_vision:
        return False
    return uncached_token_count > REMOTE_PREFILL_MIN_TOKENS and (
        prefill_endpoint is not None
    )


def remote_prefill(
    prompt_tokens: mx.array,
    cache: KVCacheType,
    on_prefill_progress: Callable[[int, int], None] | None,
    *,
    endpoint: str,
    request_id: str,
    model_id: str,
    start_pos: int = 0,
) -> tuple[float, int, list[CacheSnapshot]]:
    t0 = time.perf_counter()
    total_prompt_tokens = int(prompt_tokens.shape[0])
    num_layers: int = 0

    def _on_header(header: Header) -> None:
        nonlocal num_layers
        num_layers = header.num_layers

    def _on_chunk(_chunk: KVChunk, chunks_received: int) -> None:
        nonlocal num_layers
        if on_prefill_progress is None:
            return
        if num_layers > 0 and chunks_received % num_layers == 0:
            tokens_so_far = chunks_received // num_layers
            on_prefill_progress(
                min(tokens_so_far, total_prompt_tokens),
                total_prompt_tokens,
            )

    request = PrefillRequest(
        model_id=model_id,
        token_ids=cast(list[int], prompt_tokens.tolist()),
        start_pos=start_pos,
        request_id=request_id,
    )
    result = remote_prefill_fetch(
        endpoint, request, on_header=_on_header, on_kv_chunk=_on_chunk
    )
    t_received = time.perf_counter()

    caches = cast(list[KVCache | RotatingKVCache | ArraysCache], list(cache))
    final_offset = ingest_into_mlx_cache(result, caches, start_pos=start_pos)
    t_done = time.perf_counter()

    num_tokens = final_offset - start_pos
    tps = num_tokens / max(t_done - t0, 0.001)

    logger.info(
        f"Remote prefill: {num_tokens} tokens (start_pos={start_pos}, "
        f"final_offset={final_offset}) at {tps:.0f} tok/s, "
        f"transfer={(t_received - t0) * 1000:.0f}ms, "
        f"inject={(t_done - t_received) * 1000:.0f}ms"
    )
    # final_offset is the real absolute token position after ingest -- the
    # authoritative count (see snapshot_ssm_states' docstring for why a
    # cache-internal .size() derivation is wrong for CacheList-composed
    # non-sliceable layers).
    return tps, num_tokens, [snapshot_ssm_states(cache, final_offset)]

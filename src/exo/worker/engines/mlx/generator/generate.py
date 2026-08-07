import contextlib
import functools
import math
import os
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterator, Literal, cast, get_args

import mlx.core as mx
from mlx_lm.generate import (
    maybe_quantize_kv_cache,
    stream_generate,
)
from mlx_lm.models.cache import ArraysCache, CacheList, RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
)
from exo.worker.engines.mlx.auto_parallel import (
    PipelineFirstLayer,
    PipelineLastLayer,
    clear_prefill_sends,
    flush_prefill_sends,
    set_pipeline_prefill,
    set_pipeline_queue_sends,
)
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.sampling import card_sampling_values, resolve_sampling
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.pp_metaframe import (
    MetaFramedPipelineFirstLayer,
    MetaFramedPipelineLastLayer,
)
from exo.worker.engines.mlx.pp_prefill_session import (
    PrefillSessionError,
    ResumablePrefillSession,
    supports_chunked_prefill_interruption,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    fix_unmatched_think_end_tokens,
    get_coord_group,
    mx_barrier,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    get_inner_model,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

REMOTE_PREFILL_MIN_TOKENS = 1000


def _heap_census_mx_arrays(top_n: int = 15) -> str:
    """Live mx.array census via gc: find the largest live arrays and, for each,
    name a referrer chain so we can identify WHICH Python object holds them.
    This is the instrument for the DSv4 'active memory pinned at idle, not in
    any cache' leak. Gated by EXO_DSV4_HEAPCENSUS=1 (off by default — gc.walk
    is expensive). Read-only; never mutates; never raises.
    """
    import gc
    from collections import defaultdict
    try:
        arrays = []
        total = 0
        count = 0
        # Group ALL arrays by (shape,dtype) to find accumulating small ones.
        by_shape: dict = defaultdict(lambda: [0, 0])  # key -> [count, bytes]
        for obj in gc.get_objects():
            try:
                if isinstance(obj, mx.array):
                    nb = obj.nbytes
                    total += nb
                    count += 1
                    key = (tuple(obj.shape), str(obj.dtype))
                    by_shape[key][0] += 1
                    by_shape[key][1] += nb
                    if nb >= 16 * 1024 * 1024:
                        arrays.append((nb, obj))
            except Exception:
                continue
        arrays.sort(key=lambda t: -t[0])
        # Top shape-groups by TOTAL bytes (these reveal accumulating small arrays).
        top_groups = sorted(by_shape.items(), key=lambda kv: -kv[1][1])[:12]
        lines = [f"live mx.arrays: total={total/1024**3:.2f}GB, count={count}, big(>=16MB)={len(arrays)}"]
        lines.append("  top shape-groups by total bytes (count x shape dtype = GB):")
        for (shp, dt), (cnt, b) in top_groups:
            lines.append(f"    {cnt:>5d} x {shp} {dt} = {b/1024**3:.2f}GB")
        # Name holders of the SUSPECT accumulating class: (1, 7936/var, 512|128) bf16.
        # Walk the referrer chain UP TO ROOT (until we hit a frame, module, or a
        # named object) so we identify the EXACT owner, not just the immediate
        # list. Stops at the first frame (names the function) or repeats.
        lines.append("  holder trace-to-root for ONE suspect bf16 (1,*,512/128) array:")
        import types as _types
        def _describe(o):
            t = type(o).__name__
            if isinstance(o, _types.FrameType):
                return f"FRAME[{o.f_code.co_name}@{o.f_code.co_filename.split('/')[-1]}:{o.f_lineno}]"
            if isinstance(o, _types.ModuleType):
                return f"MODULE[{getattr(o,'__name__','?')}]"
            if t not in ("list", "tuple", "dict", "cell"):
                mod = getattr(type(o), "__module__", "")
                return f"{mod}.{t}" if mod else t
            return t
        traced = 0
        for obj in gc.get_objects():
            if traced >= 3:
                break
            try:
                if not isinstance(obj, mx.array):
                    continue
                shp = tuple(obj.shape)
                if not (len(shp) == 3 and shp[0] == 1 and shp[2] in (512, 128)
                        and str(obj.dtype) == "mlx.core.bfloat16"):
                    continue
                # BFS up the referrer graph to the first frame/module/named obj.
                chain = []
                cur = obj
                visited = set()
                for _hop in range(10):
                    refs = [r for r in gc.get_referrers(cur)
                            if id(r) not in visited and r is not chain]
                    visited.update(id(r) for r in refs)
                    # prefer a frame/module/named owner; else follow the first container
                    pick = None
                    for r in refs:
                        if isinstance(r, (_types.FrameType, _types.ModuleType)):
                            pick = r; break
                        if type(r).__name__ not in ("list", "tuple", "dict", "cell", "list_iterator"):
                            pick = r; break
                    if pick is None:
                        pick = refs[0] if refs else None
                    if pick is None:
                        chain.append("(no referrer / root)")
                        break
                    chain.append(_describe(pick))
                    if isinstance(pick, (_types.FrameType, _types.ModuleType)):
                        break
                    cur = pick
                lines.append(f"    {shp} -> " + " -> ".join(chain))
                traced += 1
            except Exception as e:
                lines.append(f"    trace error: {e}")
                break
        for nb, a in arrays[:top_n]:
            try:
                shp = getattr(a, "shape", "?")
                dt = str(getattr(a, "dtype", "?"))
            except Exception:
                shp, dt = "?", "?"
            # name the holders: types of objects that reference this array
            holders = []
            try:
                for r in gc.get_referrers(a):
                    rt = type(r).__name__
                    if rt in ("list", "tuple", "dict"):
                        # one more hop: who holds the container?
                        for rr in gc.get_referrers(r):
                            holders.append(f"{type(rr).__name__}.{rt}")
                            if len(holders) >= 3:
                                break
                    else:
                        holders.append(rt)
                    if len(holders) >= 3:
                        break
            except Exception:
                pass
            lines.append(f"  {nb/1024**2:.0f}MB shape={shp} {dt} <- {holders[:3]}")
        return "\n".join(lines)
    except Exception as e:
        return f"heap census failed: {e}"


def _profile_cache_bytes(cache_list: Any) -> dict[str, float]:
    """Sum live mx.array bytes in a per-layer cache list, grouped by the
    cache class that owns them. Read-only memory attribution for the DSv4
    'where do the GB go' investigation. Walks CacheList nesting and the
    known array-bearing attrs of each cache type (RotatingKVCache keys/values,
    PoolingCache _pool_storage/buf_kv/buf_gate, ArraysCache state). Returns
    {class_name: MB}. Defensive: any unexpected shape is skipped, never raises.
    """
    acc: dict[str, float] = {}

    def _arr_bytes(a: Any) -> int:
        try:
            if isinstance(a, mx.array):
                return a.nbytes
        except Exception:
            pass
        return 0

    def _add(cls: str, b: int) -> None:
        if b:
            acc[cls] = acc.get(cls, 0.0) + b / 1024**2

    def _walk(obj: Any) -> None:
        if obj is None:
            return
        sub = getattr(obj, "caches", None)
        if sub is not None:
            for c in sub:
                _walk(c)
            return
        cls = type(obj).__name__
        for attr in (
            "keys", "values",
            "_pool_storage", "buf_kv", "buf_gate",
        ):
            _add(cls, _arr_bytes(getattr(obj, attr, None)))
        st = getattr(obj, "state", None)
        if isinstance(st, (list, tuple)):
            for s in st:
                _add(cls, _arr_bytes(s))

    try:
        if isinstance(cache_list, (list, tuple)):
            for c in cache_list:
                _walk(c)
        else:
            _walk(cache_list)
    except Exception:
        pass
    return acc


def _log_cache_profile(tag: str, cache_list: Any) -> None:
    """Emit a one-line [MEMPROF] breakdown of cache bytes by owning class."""
    try:
        prof = _profile_cache_bytes(cache_list)
        if not prof:
            logger.info(f"[MEMPROF] {tag}: (no cache arrays found)")
            return
        total = sum(prof.values())
        parts = ", ".join(
            f"{k}={v/1024:.2f}GB" for k, v in sorted(prof.items(), key=lambda kv: -kv[1])
        )
        logger.info(f"[MEMPROF] {tag}: total_cache={total/1024:.2f}GB | {parts}")
    except Exception as e:
        logger.info(f"[MEMPROF] {tag}: profile failed: {e}")

generation_stream = mx.new_stream(mx.default_device())

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5

# Retain at most this many chunk-boundary cache snapshots per request.
# Trade-off: more snaps = wider partial-prefix-hit coverage across requests
# at the cost of memory per leaf (each snap deep-copies pooled state, which
# scales with prefill depth). At 16K-avg Hermes prompts, each snap is ~180 MB
# of pooled state, so 4 snaps × 4 leaves = ~3 GB — comfortable headroom.
# 4 still covers ~1K tokens of partial-hit window, enough for typical agent
# turn extension (gen ~150 tok + tool-result ~500 tok).
# Lowered 4→2 to give long-context Hermes sessions ~3-5 GB headroom for
# Hermes auxiliary calls (memory_manager.sync_all, title_generation) that
# otherwise pile concurrent KV pressure and trigger macOS thrash near the
# 128 GB ceiling. 2 still covers a single agent-turn extension.
_SNAPSHOT_RETENTION = 2


@contextlib.contextmanager
def patch_embed_tokens(
    model: Model,
    embeddings: mx.array,
    start_offset: int = 0,
    token_count: int = 0,
    image_token_id: int | None = None,
) -> Generator[None]:
    inner = get_inner_model(model)  # type: ignore
    original_embed = inner.embed_tokens  # type: ignore
    end_offset = start_offset + token_count
    offset = [start_offset]

    def _inject(input_ids: mx.array) -> mx.array:
        chunk_start = offset[0]
        chunk_len = input_ids.shape[-1]
        chunk_end = chunk_start + chunk_len
        offset[0] = chunk_end

        # The injection window is [start_offset, end_offset).
        if chunk_end <= start_offset or chunk_start >= end_offset:
            return original_embed(input_ids)  # type: ignore

        # Mixed chunk: splice the pre-computed embeddings for the overlap
        # into `original_embed(input_ids)` for any text-only fringes.
        overlap_start = max(chunk_start, start_offset)
        overlap_end = min(chunk_end, end_offset)
        dst_start = overlap_start - chunk_start
        dst_end = overlap_end - chunk_start
        text_embeds: mx.array = original_embed(input_ids)  # type: ignore
        return mx.concatenate(
            [
                text_embeds[:, :dst_start, :],
                embeddings[:, overlap_start:overlap_end, :],
                text_embeds[:, dst_end:, :],
            ],
            axis=1,
        )

    for attr in dir(original_embed):  # type: ignore
        if not attr.startswith("_") and not hasattr(_inject, attr):
            with contextlib.suppress(AttributeError, TypeError):
                setattr(_inject, attr, getattr(original_embed, attr))  # type: ignore

    inner.embed_tokens = _inject

    # Gemma 4 (e2b/e4b) has a second, independent embedding table that produces
    # per-layer conditioning signals via self.embed_tokens_per_layer(input_ids).
    # The injected vision embeddings live in the main residual stream only, so
    # if image_token_id positions are passed through as-is the per-layer table
    # produces garbage signals at those positions (the `<image>` token was never
    # trained to have meaningful per-layer inputs).
    original_per_layer = getattr(inner, "embed_tokens_per_layer", None)  # type: ignore
    if original_per_layer is not None and image_token_id is not None:

        def _clean_per_layer(input_ids: mx.array) -> mx.array:
            clean_ids = mx.where(
                input_ids == image_token_id, mx.zeros_like(input_ids), input_ids
            )
            return original_per_layer(clean_ids)  # type: ignore

        inner.embed_tokens_per_layer = _clean_per_layer

    try:
        yield
    finally:
        inner.embed_tokens = original_embed
        if original_per_layer is not None and image_token_id is not None:
            inner.embed_tokens_per_layer = original_per_layer


class PrefillCancelled(BaseException):
    """Raised when prefill is cancelled via the progress callback."""


def _has_pipeline_communication_layer(model: Model):
    """Detects whether ``model`` has ANY real cross-rank pipeline
    communication layer installed -- gates ``prefill()``'s choice of
    ``pipeline_parallel_prefill()`` (real distributed chunked prefill)
    vs. ``stream_generate()`` (single-rank path).

    2026-08-06 REAL bug found (not hypothetical) auditing this
    mechanism before wiring the chunk-drive fix into live serving:
    this check ONLY matched the LEGACY ``PipelineFirstLayer``/
    ``PipelineLastLayer`` classes, never
    ``MetaFramedPipelineFirstLayer``/``MetaFramedPipelineLastLayer``
    (``pp_metaframe.py``) or their ``Batched*`` subclasses
    (``pp_batched_decode_layers.py``) -- because
    ``MetaFramedPipelineFirstLayer`` does NOT subclass the legacy
    ``PipelineFirstLayer`` (confirmed via ``class
    MetaFramedPipelineFirstLayer(CustomMlxLayer)`` -- a DIFFERENT base
    class entirely). This meant ``is_pipeline`` was ALWAYS ``False``
    under ``EXO_PP_METAFRAME=1``/``EXO_PP_BATCHED_DECODE=1``, so
    ``prefill()`` ALWAYS routed through ``stream_generate()`` and
    ``pipeline_parallel_prefill()`` -- the ONLY thing that can ever
    yield real chunk boundaries for the whole chunked-prefill
    interruption mechanism built this session -- was NEVER reached,
    regardless of anything else fixed. Confirmed SAFE to broaden (not
    just convenient): the SAME session's earlier regression fix
    (``set_pipeline_prefill``/``set_pipeline_queue_sends`` now also
    updating the metaframe ``ForwardStepInfo`` contextvar, not just
    the dead legacy ambient flags) means
    ``pipeline_parallel_prefill``'s existing forward-pass calls
    (``model(chunk_tokens, cache=...)``) ALREADY correctly drive
    metaframe layers once this gate lets them run -- this fix was
    genuinely blocked on that earlier one, not independent of it.
    """
    for layer in model.layers:
        if isinstance(
            layer,
            (
                PipelineFirstLayer,
                PipelineLastLayer,
                MetaFramedPipelineFirstLayer,
                MetaFramedPipelineLastLayer,
            ),
        ):
            return True
    return False


def _pipeline_parallel_prefill_steps(
    model: Model,
    prompt: mx.array,
    prompt_cache: KVCacheType,
    prefill_step_size: int,
    kv_group_size: int | None,
    kv_bits: int | None,
    prompt_progress_callback: Callable[[int, int], None],
    distributed_prompt_progress_callback: Callable[[], None] | None,
    group: mx.distributed.Group,
    *,
    interruptible: bool = False,
) -> "Iterator[tuple[Literal['chunk'], int, mx.array] | tuple[Literal['done'], None, None]]":
    """2026-08-06 (Phase 2 Stage 4, generator-core split, consult-
    reviewed before implementation): the ORIGINAL ``pipeline_parallel_
    prefill`` body, byte-for-byte unchanged, EXCEPT for one new yield
    point per REAL chunk (only reached when ``interruptible=True``)
    and the final fall-through becoming ``yield ("done", None, None)``.
    ``pipeline_parallel_prefill`` (below) stays a thin eager wrapper
    that drains this generator to completion -- EVERY existing caller
    goes through it unchanged and is structurally atomic, mirroring
    ``DeepseekV4Model._forward_steps``/``__call__``'s own Stage 1b
    split exactly (same rationale: a generator's yield that a given
    call path never reaches simply never fires).

    Per a consult review: this function's own chunk/dummy-iteration
    pipeline-bubble-fill bookkeeping (leading/trailing dummy
    iterations for N-rank overlap, ``real_chunk_sizes``, ``processed``
    offset tracking) is the load-bearing state a "just swap the inner
    model(...) call" shortcut would have silently duplicated or
    desynced -- this split keeps ALL of that bookkeeping exactly where
    it already lives, unmodified, and only changes what happens AT
    each real chunk boundary.

    ``interruptible=True`` does NOT itself drive a
    ``ResumablePrefillSession`` -- it only marks each real chunk's
    boundary as a yield point (``("chunk", i, chunk_tokens)``,
    yielding the chunk's TOKENS, not output -- pipeline_parallel_
    prefill's own call to ``model(...)`` discards the output already;
    only the cache-population side effect matters). The CALLER
    receives this yield, decides HOW to actually run that chunk's
    forward pass (eagerly via ``model(...)``, unchanged, OR via a
    ``ResumablePrefillSession`` it constructs and drives to completion
    across real ``tick()`` calls), and resumes this generator via
    ``send(None)`` once the chunk's cache-populating side effect has
    genuinely happened -- this generator does NOT touch the cache
    itself when interruptible, that becomes the caller's
    responsibility for that one chunk. Every OTHER per-chunk step
    (quantize, flush_prefill_sends, eval_cache, contiguous-breaking,
    memory logging, progress callback) still runs HERE, unchanged,
    immediately after the resume -- only the forward pass itself
    moves to the caller when interruptible.
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

    # Initial callback matching generate_step
    prompt_progress_callback(0, total)

    from exo.worker.engines.mlx.trace import request_trace

    try:
        with mx.stream(generation_stream):
            for _ in range(n_leading):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

            for i in range(n_real):
                chunk_size = real_chunk_sizes[i]
                _t_fwd = time.perf_counter()
                # 2026-08-06 (Phase 2 Stage 4): the ONLY yield point in
                # this generator. Reached after EVERY real chunk's
                # forward pass when interruptible=True -- the caller
                # is responsible for having ALREADY run (eagerly or
                # via a ResumablePrefillSession) the forward pass that
                # populates _prompt_cache for THIS chunk before
                # send()-ing control back here; this generator resumes
                # straight into quantize/flush/eval, exactly as if the
                # eager model(...) call below had just returned. Never
                # reached at all when interruptible=False (every
                # existing eager caller), so this line changes NOTHING
                # about their behavior.
                chunk_tokens = prompt[processed : processed + chunk_size][None]
                if interruptible:
                    yield ("chunk", i, chunk_tokens)
                else:
                    model(chunk_tokens, cache=_prompt_cache)
                quantize_cache_fn(_prompt_cache)
                request_trace.record(f"prefill.chunk{i}.forward({chunk_size}tok)", _t_fwd)
                processed += chunk_size

                if distributed_prompt_progress_callback is not None:
                    _t_cb = time.perf_counter()
                    distributed_prompt_progress_callback()
                    request_trace.record(f"prefill.chunk{i}.distributed_cb", _t_cb)

                _t_flush = time.perf_counter()
                flush_prefill_sends()
                request_trace.record(f"prefill.chunk{i}.flush_sends", _t_flush)

                _t_eval = time.perf_counter()
                mx.eval([c.state for c in _prompt_cache])  # type: ignore
                request_trace.record(f"prefill.chunk{i}.eval_cache", _t_eval)

                # Break shared-buffer references in DeltaNet (ArraysCache) entries.
                # NOTE: This MUST be a separate eval from the cache state eval above.
                # On R1, eval_cache materializes the cache (~313ms) during the pipeline
                # bubble while R0 waits. Merging contiguous into that eval moves the
                # work into the forward time, making R1's forward slower and increasing
                # the pipeline bubble — net 4s regression on 16K prefill.
                _t_contig = time.perf_counter()
                for _c in _prompt_cache:
                    if isinstance(_c, ArraysCache):
                        _c.cache = [mx.contiguous(x) if x is not None else x for x in _c.cache]
                        mx.eval(*[x for x in _c.cache if x is not None])
                request_trace.record(f"prefill.chunk{i}.contiguous", _t_contig)

                # Log memory every 5 chunks for profiling
                if i % 5 == 0 or i == n_real - 1:
                    active_gb = mx.metal.get_active_memory() / 1024**3
                    peak_gb = mx.metal.get_peak_memory() / 1024**3
                    logger.info(f"[MEM] prefill chunk {i+1}/{n_real} ({processed} tokens): active={active_gb:.2f} GB, peak={peak_gb:.2f} GB")

                prompt_progress_callback(processed, total)

            for _ in range(n_trailing):
                if distributed_prompt_progress_callback is not None:
                    distributed_prompt_progress_callback()

    finally:
        clear_prefill_sends()

    # Post-loop: process the remaining 1 token not covered by the chunk loop.
    # The chunk loop processes total-1 tokens; this handles the last one.
    # (Previously did 2 forward passes to match stream_generate's extra generated
    # token, but that's unnecessary — prefill() trims conditionally now.)
    _t_post = time.perf_counter()
    with mx.stream(generation_stream):
        model(prompt[-1:][None], cache=_prompt_cache)
        quantize_cache_fn(_prompt_cache)
    flush_prefill_sends()
    request_trace.record("prefill.post_loop_token", _t_post)

    assert _prompt_cache is not None
    with mx.stream(generation_stream):
        mx.eval([c.state for c in _prompt_cache])  # type: ignore

    # Final callback matching generate_step
    prompt_progress_callback(total, total)

    logger.info(
        f"[R{rank}] Prefill: {n_real} real + {n_leading}+{n_trailing} dummy iterations, "
        f"Processed {processed} tokens in {(time.perf_counter() - t_start) * 1000:.1f}ms"
    )
    yield ("done", None, None)


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
    """Thin eager wrapper (2026-08-06, Phase 2 Stage 4, consult-
    reviewed) -- drains ``_pipeline_parallel_prefill_steps`` to
    completion with ``interruptible=False`` (the default), matching
    this function's ORIGINAL (pre-refactor) behavior and side effects
    exactly, byte-for-byte. Every existing caller (the four real
    ``prefill()``/``prefill_batched()`` call sites) is unaffected by
    this refactor -- see ``_pipeline_parallel_prefill_steps``'s own
    docstring for the full rationale, mirroring
    ``DeepseekV4Model.__call__``'s identical Stage 1b split."""
    for _kind, _idx, _payload in _pipeline_parallel_prefill_steps(
        model,
        prompt,
        prompt_cache,
        prefill_step_size,
        kv_group_size,
        kv_bits,
        prompt_progress_callback,
        distributed_prompt_progress_callback,
        group,
        interruptible=False,
    ):
        pass


def prefill(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
    prefill_step_size: int | None = None,
    snapshot_offset: int = 0,
) -> tuple[float, int, list[CacheSnapshot]]:
    """Prefill the KV cache with prompt tokens.

    This runs the model over the prompt tokens to populate the cache,
    then trims off the extra generated token.

    ``snapshot_offset``: callers on a KV-prefix-cache HIT pass in only the
    REMAINING (post-hit) suffix of the prompt as ``prompt_tokens`` -- the
    already-cached prefix is never re-run through the model. Internally,
    each chunk-boundary snapshot is captured with
    ``snapshot_ssm_states(cache, processed)``, where ``processed`` counts
    progress through THIS CALL's local ``prompt_tokens`` starting at 0.
    Without ``snapshot_offset``, a partial-hit prefill's snapshots are
    stamped with a ``token_count`` that is wrong by exactly
    ``prefix_hit_length`` -- too small, relative to the trie's absolute
    token positions.

    CROSS-REQUEST CONTAMINATION BUG (2026-07-28, round 3): this offset was
    missing entirely on the local/serial prefill path (this function),
    while the remote-prefill sibling path (``remote_prefill.py``) already
    computed and applied it correctly via its own ``start_pos`` parameter
    (see ``remote_prefill``'s ``final_offset = ingest_into_mlx_cache(...,
    start_pos=start_pos)`` then ``snapshot_ssm_states(cache, final_offset)``
    -- an absolute position). The asymmetry meant every LOCAL partial-hit
    prefill silently mis-stamped its snapshots' token_count too low. A
    later, unrelated request whose OWN match_length happened to coincide
    with that wrong (too-small) token_count could then have
    ``_find_nearest_snapshot``/``_resolve_restore_position`` pick and
    restore a snapshot that actually encodes a much deeper position in a
    PRIOR, unrelated request's own generation -- non-sliceable
    (SSM/PoolingCache) layer state well past the shared boilerplate
    boundary, from a completely different task's unique content -- while
    the trie's sliceable-layer bookkeeping (which only ever tracks
    genuinely shared prefix bytes) stayed correct. This produced the
    observed symptom: a request effectively resuming generation from deep
    inside an unrelated earlier request's own answer, yielding a coherent,
    complete, CORRECTLY-FORMATTED response to the WRONG task. Fixed by
    threading the real ``prefix_hit_length`` through as ``snapshot_offset``
    so every snapshot's token_count is always an absolute prompt position,
    matching remote_prefill's existing (correct) contract.

    Returns:
        (tokens_per_sec, num_tokens, snapshots)
    """
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return 0.0, 0, []

    # Start a fresh trace for this prefill so the per-chunk spans (forward,
    # eval_cache, flush_sends, all_sum, indexer, contiguous) are scoped to this
    # request. Without reset() the span list accumulates across all requests
    # forever; without the matching dump() at the end the spans are recorded
    # but never logged. Gated on EXO_TRACING_ENABLED (no-op when off), so this
    # is free in production and gives a full per-chunk timeline when tracing on
    # — used to diagnose the high-context prefill stall (~12s per 128-tok chunk
    # at 460K ctx). See exo.worker.engines.mlx.trace.
    from exo.worker.engines.mlx.trace import request_trace as _rt

    _rt.reset()

    # Reset the model-side span profiler (EXO_PROFILER=spans) at prefill start so
    # its per-span totals (attn.indexer, attn.sdpa, attn.compressor, moe.*,
    # attn.all_sum) are scoped to THIS prefill — dumped at the end below. This
    # replaces the fragile SIGUSR1-based dump (signaling a live MLX/RDMA process
    # can crash it). No-op when no profiler hook is registered.
    try:
        from mlx_lm import profiler as _prof

        _ph: Any = _prof.get()
        if _ph is not None and hasattr(_ph, "dump"):
            _ph.dump(reset=True)  # discard warmup/prior; start clean
    except Exception:
        _ph = None

    logger.debug(f"Prefilling {num_tokens} tokens...")
    start_time = time.perf_counter()
    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []
    _diag = os.environ.get("EXO_PREFIX_CACHE_DIAG") == "1"
    _diag_rank = group.rank() if group is not None else 0
    if _diag:
        logger.info(
            f"[PREFIX_DIAG rank={_diag_rank}] prefill() ENTRY num_tokens={num_tokens} "
            f"has_ssm={has_ssm}"
        )

    # TODO(evan): kill the callbacks/runner refactor
    def progress_callback(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start_time
        tok_per_sec = processed / elapsed if elapsed > 0 else 0
        logger.debug(
            f"Prefill progress: {processed}/{total} tokens ({tok_per_sec:.1f} tok/s)"
        )
        if has_ssm:
            # Keep up to _SNAPSHOT_RETENTION most-recent chunk-boundary
            # snapshots. Original "last 2 only" was too tight (rollback-
            # only — broke cross-request prefix cache); "keep all" OOM'd
            # m4-2 at 99% memory after a few Hermes turns because each
            # snapshot deep-copies pooled-attention state that grows
            # with prefill depth.
            #
            # 8 retains: rollback uses [-2] (immediate prior chunk),
            # cross-request hits land for any match_length within
            # ~7 × prefill_step_size of the snapshotted leaf's end.
            # For DSv4 chunk_size=256 that's ~1.8K tokens of partial-
            # hit coverage from a leaf's tail. Multi-turn Hermes flows
            # where each turn extends by <1.8K tokens hit cleanly.
            snapshots.append(snapshot_ssm_states(cache, snapshot_offset + processed))
            if _diag:
                logger.info(
                    f"[PREFIX_DIAG rank={_diag_rank}] snapshot appended "
                    f"processed={processed}/{total} "
                    f"absolute_token_count={snapshot_offset + processed} "
                    f"len(snapshots)={len(snapshots)}"
                )
            if len(snapshots) > _SNAPSHOT_RETENTION:
                snapshots.pop(0)

        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    def combined_progress_callback(processed: int, total: int) -> None:
        if distributed_prompt_progress_callback is not None:
            distributed_prompt_progress_callback()
        progress_callback(processed, total)

    from exo.worker.engines.mlx.trace import request_trace, T

    set_pipeline_prefill(model, is_prefill=True)

    # Release any cached Metal buffers before prefill to maximize headroom
    # for the forward pass intermediates during long context prefills.
    with T("prefill.clear_cache"):
        mx.clear_cache()

    with T("prefill.barrier"):
        mx_barrier(group)

    # Memory checkpoint before prefill
    with T("prefill.mem_checkpoint"):
        mx.eval(mx.zeros(1))
        active_gb = mx.metal.get_active_memory() / 1024**3
        peak_gb = mx.metal.get_peak_memory() / 1024**3
        cache_gb = mx.metal.get_cache_memory() / 1024**3
    logger.info(f"[MEM] before prefill ({num_tokens} tokens): active={active_gb:.2f} GB, peak={peak_gb:.2f} GB, cache={cache_gb:.2f} GB")
    logger.info("Starting prefill")

    is_pipeline = _has_pipeline_communication_layer(model)

    if prefill_step_size is None:
        prefill_step_size = int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))

    try:
        if is_pipeline and num_tokens >= prefill_step_size:
            set_pipeline_queue_sends(model, queue_sends=True)
            assert group is not None, "Pipeline prefill requires a distributed group"
            with T("prefill.pipeline_parallel"):
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
            with T("prefill.stream_generate"):
                # THE LEAK FIX: stream_generate is a GENERATOR. The old
                # `for _ in stream_generate(...): break` pulled one item then
                # broke WITHOUT closing it, leaving the generator's frame
                # suspended and alive every turn. That frame pins its internal
                # iteration over the per-layer prompt_cache (a list_iterator
                # over the DSv4 sparse layers), so one set of pooled
                # (1, P, 512)/(1, P, 128) bf16 tensors leaked PER SPARSE LAYER
                # PER TURN (+21/turn, ~0.2-0.4 GB/turn; ~29GB over a long
                # session). Verified via gc heap census: holder was
                # PoolingCache-in-a-list reached via list_iterator. Wrapping in
                # contextlib.closing() guarantees .close() runs on break, which
                # unwinds the suspended frame and releases those references.
                _sg = stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_tokens,
                    max_tokens=1,
                    sampler=sampler,
                    prompt_cache=cache,
                    prefill_step_size=prefill_step_size,
                    kv_group_size=KV_GROUP_SIZE,
                    kv_bits=KV_BITS,
                    prompt_progress_callback=combined_progress_callback,
                )
                with contextlib.closing(_sg):
                    for _ in _sg:
                        break
    except PrefillCancelled:
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise

    set_pipeline_queue_sends(model, queue_sends=False)
    set_pipeline_prefill(model, is_prefill=False)

    # Trim extra entries from the cache so the decode path can reprocess last_tokens[-2:].
    # - stream_generate path: generated 1 extra token → trim(2) (generated + last prompt token)
    # - pipeline_parallel path: 1 post-loop token, no generation → trim(1)
    # SSM/ArraysCache layers are rolled back to snapshots[-2] (state after last real chunk).
    _trim_n = 1 if (is_pipeline and num_tokens >= prefill_step_size) else 2
    with T("prefill.cache_trim_and_rollback"):
        pre_gen = deepcopy(snapshots[-2]) if has_ssm else None
        for i, c in enumerate(cache):
            if has_ssm and isinstance(c, (ArraysCache, RotatingKVCache)):
                assert pre_gen is not None
                if pre_gen.states[i] is not None:
                    cache[i] = deepcopy(pre_gen.states[i])  # type: ignore
            else:
                assert not isinstance(c, (ArraysCache, RotatingKVCache))
                c.trim(_trim_n)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
    try:
        _a = mx.metal.get_active_memory() / 1024**3
        logger.info(f"[MEM] after prefill ({num_tokens} tok): active={_a:.2f} GB")
        _log_cache_profile(f"after prefill ({num_tokens} tok)", cache)
        if os.environ.get("EXO_DSV4_HEAPCENSUS") == "1":
            logger.info(f"[HEAPCENSUS] after prefill ({num_tokens} tok):\n{_heap_census_mx_arrays()}")
    except Exception:
        pass
    # Emit the per-chunk span timeline (no-op unless EXO_TRACING_ENABLED). This
    # names exactly which sub-step (forward / eval_cache / flush_sends / all_sum
    # / indexer / contiguous) consumes wall time per chunk — the diagnostic for
    # the high-context prefill stall.
    _rt.dump()
    # Dump the model-side span breakdown for THIS prefill (attn.indexer,
    # attn.sdpa, attn.compressor, moe.*, attn.all_sum). Signal-free — avoids the
    # SIGUSR1-to-a-live-MLX-process crash. No-op when EXO_PROFILER!=spans.
    if _ph is not None:
        try:
            _ph.dump(reset=True)
        except Exception:
            pass
    # Exclude the last snapshot
    _final_snapshots = snapshots[:-1] if snapshots else []
    if _diag:
        logger.info(
            f"[PREFIX_DIAG rank={_diag_rank}] prefill() RETURN "
            f"len(snapshots) before trim={len(snapshots)} "
            f"len(returned)={len(_final_snapshots)}"
        )
    return tokens_per_sec, num_tokens, _final_snapshots


@dataclass
class ChunkedPrefillDrive:
    """Live cross-tick state for ONE request's real chunked-prefill drive
    (2026-08-07, Phase 2 live-wiring). Bundles the outer
    ``_pipeline_parallel_prefill_steps(interruptible=True)`` generator
    handle plus everything ``prefill()``'s own tail logic needs once the
    LAST real chunk's ``ResumablePrefillSession`` reaches ``done=True`` --
    so a caller driving this across many ``ExoBatchGenerator.step()`` calls
    gets back the SAME ``(tokens_per_sec, num_tokens, snapshots)`` shape
    ``prefill()`` returns synchronously today, indistinguishable to any
    downstream consumer.

    Constructed by ``prefill_interruptible_start()``; advanced one real
    prompt-chunk at a time by ``prefill_interruptible_advance()``, called
    ONLY after the caller's current ``session`` reaches ``done=True`` (a
    caller-side invariant this class does not itself enforce -- mirrors
    ``ResumablePrefillSession.advance``'s own \"caller must check .done\"
    discipline).
    """

    model: Model
    outer_gen: "Generator[tuple[Literal['chunk'], int, mx.array] | tuple[Literal['done'], None, None], None, None]"
    session: "ResumablePrefillSession"
    chunk_index: int
    cache: KVCacheType
    num_tokens: int
    start_time: float
    has_ssm: bool
    snapshots: list[CacheSnapshot]


def prefill_interruptible_start(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens: mx.array,
    cache: KVCacheType,
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
    prefill_step_size: int | None = None,
) -> "ChunkedPrefillDrive | None":
    """Sibling to ``prefill()`` (2026-08-07, Phase 2 live-wiring),
    mirroring the SAME eligibility gate ``prefill()`` itself uses
    (``is_pipeline and num_tokens >= prefill_step_size``) plus two
    additional, narrower checks this interruptible path specifically
    needs -- returns ``None`` (never raises) for every case where the
    caller must fall back to the existing, unmodified, synchronous
    ``prefill()`` instead:

    - Not eligible for ``pipeline_parallel_prefill`` at all (small
      prompt, non-pipeline sharding) -- identical to ``prefill()``'s own
      ``is_pipeline and num_tokens >= prefill_step_size`` branch
      condition.
    - ``group.size() != 2`` -- this campaign's confirmed real scope is
      N=2 only (design doc Section 13.3); the chunk-drive machinery this
      function feeds (``Rank0BatchedDecodeGlue``'s two-phase
      RANK0_LOCAL/HANDOFF/RANK1_DRAINING state machine) is built and
      tested ONLY for a 2-rank point-to-point pair. Silently falling
      back rather than misbehaving at an untested topology.
    - The loaded model's inner model does not structurally support
      chunked-prefill interruption (``supports_chunked_prefill_
      interruption`` -- i.e. no real ``_forward_steps`` generator
      exists on this checkpoint's model class). THIS is the current,
      real, production state: the ``mlx-lm`` submodule pin as of
      2026-08-07 does not yet include DSv4's ``_forward_steps`` split
      (it exists only on the fork's ``pp-layer-segment-wip`` branch,
      not yet rebased onto the fork's current ``main`` and not yet
      pinned) -- so on TODAY's real hardware this function always
      returns ``None`` and every real request takes the unmodified
      synchronous ``prefill()`` path, a provable, tested no-op until
      that follow-up work lands (see design doc's 2026-08-07 entry).

    Deliberately does NOT call ``mx_barrier(group)`` (unlike ``prefill()``,
    which keeps it, unmodified, for every other caller) -- per a
    ``consult`` review: for exactly N=2 (enforced above), the caller's
    OWN pre-existing point-to-point ``PrefillMessage``/
    ``PrefillReadyMessage`` handshake (``Rank0BatchedDecodeGlue``/
    ``Rank1BatchedDecodeGlue.tick()``, already run before either rank
    ever reaches this function) already provides the identical
    rank-pair synchronization guarantee a collective ``all_sum`` barrier
    would -- adding a SECOND, redundant collective specifically on this
    path would reintroduce a real deadlock hazard (rank 1 blocking on a
    collective before it has sent the very ack rank 0 is waiting to
    receive). ``mx.clear_cache()`` IS kept (purely local Metal-allocator
    hygiene, zero synchronization content, real memory-regression risk
    on long prompts if dropped)."""
    num_tokens = len(prompt_tokens)
    if num_tokens == 0:
        return None
    is_pipeline = _has_pipeline_communication_layer(model)
    if prefill_step_size is None:
        prefill_step_size = int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))
    if not (is_pipeline and num_tokens >= prefill_step_size):
        return None
    if group is None or group.size() != 2:
        return None

    inner_model = cast("object", get_inner_model(model))
    if not supports_chunked_prefill_interruption(inner_model):
        return None

    has_ssm = has_non_kv_caches(cache)
    snapshots: list[CacheSnapshot] = []
    start_time = time.perf_counter()

    def progress_callback(processed: int, total: int) -> None:
        if has_ssm:
            snapshots.append(snapshot_ssm_states(cache, processed))
            if len(snapshots) > _SNAPSHOT_RETENTION:
                snapshots.pop(0)
        if on_prefill_progress is not None:
            on_prefill_progress(processed, total)

    mx.clear_cache()

    set_pipeline_prefill(model, is_prefill=True)
    set_pipeline_queue_sends(model, queue_sends=True)

    outer_gen = cast(
        "Generator[tuple[Literal['chunk'], int, mx.array] | tuple[Literal['done'], None, None], None, None]",
        _pipeline_parallel_prefill_steps(
            model,
            prompt_tokens,
            cache,
            prefill_step_size,
            KV_GROUP_SIZE,
            KV_BITS,
            progress_callback,
            distributed_prompt_progress_callback,
            group,
            interruptible=True,
        ),
    )
    try:
        kind, idx, chunk_tokens = next(outer_gen)
    except StopIteration as exc:
        # Genuinely unreachable given the is_pipeline/num_tokens>=
        # prefill_step_size check above (which guarantees at least one
        # real chunk) -- fail loud rather than silently swallow a
        # contract violation in _pipeline_parallel_prefill_steps.
        set_pipeline_queue_sends(model, queue_sends=False)
        set_pipeline_prefill(model, is_prefill=False)
        raise RuntimeError(
            "prefill_interruptible_start: _pipeline_parallel_prefill_steps "
            "yielded no chunks despite num_tokens >= prefill_step_size -- "
            "this violates that generator's own documented contract"
        ) from exc
    assert kind == "chunk" and idx == 0, (
        f"prefill_interruptible_start: expected first yield to be "
        f"('chunk', 0, ...), got ({kind!r}, {idx!r}, ...)"
    )

    session = ResumablePrefillSession(
        inner_model=inner_model,  # type: ignore[reportArgumentType]
        inputs=cast("mx.array", chunk_tokens),
        cache=cache,
    )
    return ChunkedPrefillDrive(
        model=model,
        outer_gen=outer_gen,
        session=session,
        chunk_index=0,
        cache=cache,
        num_tokens=num_tokens,
        start_time=start_time,
        has_ssm=has_ssm,
        snapshots=snapshots,
    )


def prefill_interruptible_advance(
    drive: ChunkedPrefillDrive,
) -> "ResumablePrefillSession | tuple[float, int, list[CacheSnapshot]]":
    """Resume ``drive.outer_gen`` after ``drive.session`` reached
    ``done=True`` (caller-enforced invariant -- see class docstring).
    Two real outcomes, matching ``_pipeline_parallel_prefill_steps``'s
    own documented yield shape:

    - Another real chunk remains: constructs and returns a NEW
      ``ResumablePrefillSession`` for it (also updates ``drive.session``/
      ``drive.chunk_index`` in place) -- the caller must register this
      new session (``Rank0/Rank1BatchedDecodeGlue.register_prefill_
      session``) before this same tick's control returns, per Hazard
      2's fix (see design doc's 2026-08-07 entry: since ``tick()`` is
      the only recv call site on either rank, running this
      synchronously and inline -- never scheduled/deferred -- is what
      makes registering-before-the-next-advance-arrives provably true,
      not merely likely).
    - The drive is genuinely done: the outer generator's OWN trailing
      code (the post-loop single final-token forward pass, final
      flush/eval) has ALREADY run, byte-for-byte identical to
      ``prefill()``'s own eager path, BEFORE this function ever sees
      the ``(\"done\", None, None)`` yield -- see
      ``_pipeline_parallel_prefill_steps``'s own tail. This function
      then runs the SAME correctness-critical tail ``prefill()`` itself
      runs (pipeline-flag reset, cache trim/rollback, tps calculation)
      and returns the identical ``(tokens_per_sec, num_tokens,
      snapshots)`` shape -- indistinguishable to any caller from what
      the synchronous ``prefill()`` path would have produced for this
      SAME request."""
    if not drive.session.done:
        raise PrefillSessionError(
            "prefill_interruptible_advance: called with drive.session.done="
            "False -- the caller must finish advancing the CURRENT "
            "session (via its own advance() calls) before resuming the "
            "outer generator; resuming early would desync the outer "
            "generator's own per-chunk bookkeeping from the real forward "
            "pass state"
        )
    try:
        kind, idx, payload = drive.outer_gen.send(None)
    except StopIteration as exc:
        raise RuntimeError(
            "prefill_interruptible_advance: _pipeline_parallel_prefill_steps "
            "raised StopIteration without first yielding a ('done', ...) "
            "step -- this violates that generator's own documented contract"
        ) from exc

    if kind == "chunk":
        chunk_tokens = cast("mx.array", payload)
        new_session = ResumablePrefillSession(
            inner_model=drive.session.inner_model,
            inputs=chunk_tokens,
            cache=drive.cache,
        )
        drive.session = new_session
        drive.chunk_index = cast(int, idx)
        return new_session

    assert kind == "done"
    set_pipeline_queue_sends(drive.model, queue_sends=False)
    set_pipeline_prefill(drive.model, is_prefill=False)

    # Cache trim/rollback: mirrors prefill()'s own tail EXACTLY for the
    # is_pipeline+chunked branch, where _trim_n is unconditionally 1 (the
    # eligibility gate in prefill_interruptible_start already guarantees
    # is_pipeline and num_tokens >= prefill_step_size, so prefill()'s own
    # `1 if (is_pipeline and num_tokens >= prefill_step_size) else 2`
    # always resolves to 1 on this path).
    with mx.stream(generation_stream):
        pre_gen = deepcopy(drive.snapshots[-2]) if drive.has_ssm else None
        for i, c in enumerate(drive.cache):
            if drive.has_ssm and isinstance(c, (ArraysCache, RotatingKVCache)):
                assert pre_gen is not None
                if pre_gen.states[i] is not None:
                    drive.cache[i] = deepcopy(pre_gen.states[i])  # type: ignore
            else:
                assert not isinstance(c, (ArraysCache, RotatingKVCache))
                c.trim(1)  # type: ignore[reportUnknownMemberType]

    elapsed = time.perf_counter() - drive.start_time
    tokens_per_sec = drive.num_tokens / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"Chunked prefill complete: {drive.num_tokens} tokens in "
        f"{elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )
    final_snapshots = drive.snapshots[:-1] if drive.snapshots else []
    return tokens_per_sec, drive.num_tokens, final_snapshots


def prefill_batched(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens_list: list[mx.array],
    cache_list: list[KVCacheType],
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
    prefill_step_size: int | None = None,
) -> tuple[list[float], list[int], list[KVCacheType], list[list[CacheSnapshot]]]:
    """TP-aware batched prefill: process N prompts together at shape (B, L_chunk).

    The serial ``prefill()`` path was the proven c=2 long-context bottleneck —
    one prefilling stream blocked the runner main loop for the full duration
    while the other sat in the queue, so c=2 100K MTP=0 collapsed to ~7.7
    tok/s/stream. This primitive merges per-stream caches into a batched cache,
    runs the model at (B, L_chunk) so DSv4's TP collectives fire normally, and
    extracts per-stream caches at the end.

    Caller passes ``prompt_tokens[:-1]`` per stream (matching ``prefill()``'s
    contract); returned caches land at offset ``len(prompt[:-1]) - 1`` so the
    decode path can pick up from ``prompt_tokens[-2:]`` as it does today.

    Returns ``(per_stream_tps, per_stream_token_count, per_stream_cache,
    per_stream_snapshots)``. Snapshots are empty for the DSv4 cache shape (no
    ArraysCache layers); SSM/DeltaNet models go through the serial fallback.
    """
    n_streams = len(prompt_tokens_list)
    assert n_streams == len(cache_list), "prompt and cache list lengths must match"
    assert n_streams >= 1, "prefill_batched requires at least one stream"

    # ArraysCache (DeltaNet/SSM) needs per-chunk snapshot rollback machinery
    # that the serial ``prefill()`` is built around. Batched prefill doesn't
    # snapshot per-chunk; if any stream's cache holds an ArraysCache layer,
    # fall back to serial. RotatingKVCache + PoolingCache are fine here —
    # both have merge() / extract() and run cleanly in the batched path
    # (DSv4 uses CacheList(RotatingKVCache, PoolingCache, PoolingCache)).
    def _has_arrays_cache(layers: KVCacheType) -> bool:
        for layer in layers:
            if isinstance(layer, ArraysCache):
                return True
            if isinstance(layer, CacheList):
                for sub in layer.caches:
                    if isinstance(sub, ArraysCache):
                        return True
        return False

    if any(_has_arrays_cache(c) for c in cache_list):
        logger.info(
            "prefill_batched: ArraysCache/SSM detected — falling back to serial prefill"
        )
        return _serial_prefill_fallback(
            model,
            tokenizer,
            sampler,
            prompt_tokens_list,
            cache_list,
            group,
            on_prefill_progress,
            distributed_prompt_progress_callback,
            prefill_step_size,
        )

    # Caller passes prompt[:-1] (matching ``prefill()``'s contract: cache lands
    # at len-2 so decode can pick up from prompt[-2:]). We process one fewer
    # token internally — i.e., effectively prompt[:-2] — so the cache offset
    # ends at len-2 with NO trim needed. Serial prefill uses snapshot+restore
    # for RotatingKVCache because trim doesn't roll back rotation state, and
    # PoolingCache's `trim` only affects its remainder buffer (not the
    # pooled entries). Skipping the last token avoids needing either, since
    # the cache simply stops one short.
    full_lengths = [int(p.shape[0]) for p in prompt_tokens_list]
    if any(length <= 1 for length in full_lengths):
        logger.warning(
            "prefill_batched: prompt length <= 1 — falling back to serial prefill"
        )
        return _serial_prefill_fallback(
            model,
            tokenizer,
            sampler,
            prompt_tokens_list,
            cache_list,
            group,
            on_prefill_progress,
            distributed_prompt_progress_callback,
            prefill_step_size,
        )

    # Drop the last token of each prompt — the cache will land at
    # full_lengths[i] - 1 = len(caller's input) - 1 = len(original prompt) - 2.
    process_tokens_list = [p[:-1] for p in prompt_tokens_list]
    lengths = [int(p.shape[0]) for p in process_tokens_list]
    max_length = max(lengths)
    padding = [max_length - length for length in lengths]
    has_padding = any(p > 0 for p in padding)

    # Right-pad prompts to max_length with token=0. The padded positions
    # are masked out via cache.prepare(right_padding=...) below and rolled
    # off via cache.finalize() at the end.
    if has_padding:
        padded_lists = [
            p.tolist() + [0] * (max_length - int(p.shape[0]))
            for p in process_tokens_list
        ]
        padded_tokens = mx.array(padded_lists)
    else:
        padded_tokens = mx.stack(list(process_tokens_list), axis=0)

    if prefill_step_size is None:
        prefill_step_size = int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))

    from exo.worker.engines.mlx.trace import T

    set_pipeline_prefill(model, is_prefill=True)

    with T("prefill_batched.clear_cache"):
        mx.clear_cache()
    with T("prefill_batched.barrier"):
        mx_barrier(group)

    with T("prefill_batched.mem_checkpoint"):
        mx.eval(mx.zeros(1))
        active_gb = mx.metal.get_active_memory() / 1024**3
        peak_gb = mx.metal.get_peak_memory() / 1024**3
        cache_gb = mx.metal.get_cache_memory() / 1024**3
    logger.info(
        f"[MEM] before batched prefill (B={n_streams} max_L={max_length} "
        f"lengths={lengths}): active={active_gb:.2f} GB, peak={peak_gb:.2f} GB, "
        f"cache={cache_gb:.2f} GB"
    )
    logger.info(
        f"Starting batched prefill: B={n_streams} max_L={max_length} "
        f"step={prefill_step_size}"
    )

    # mlx-lm's ``_merge_caches`` handles each cache type's merge protocol —
    # QuantizedKVCache → BatchKVCache (dequant), RotatingKVCache →
    # BatchRotatingKVCache, CacheList recurses. Per-stream left_padding is
    # set inside the merge to align differing-length history.
    from mlx_lm.generate import _merge_caches  # type: ignore

    with T("prefill_batched.merge_caches"):
        batched_cache = _merge_caches(cache_list)

    # Tell the merged cache about right-padding so the per-chunk attention
    # mask zeroes out the padded tail and ``finalize()`` can roll those
    # entries off after prefill.
    if has_padding:
        with T("prefill_batched.prepare"):
            for c in batched_cache:
                c.prepare(lengths=lengths, right_padding=padding)

    start_time = time.perf_counter()

    if on_prefill_progress is not None:
        on_prefill_progress(0, max_length)

    try:
        with mx.stream(generation_stream):
            offset = 0
            chunk_idx = 0
            while offset < max_length:
                n_to_process = min(prefill_step_size, max_length - offset)
                _t_fwd = time.perf_counter()
                model(padded_tokens[:, offset : offset + n_to_process], cache=batched_cache)
                from exo.worker.engines.mlx.trace import request_trace

                request_trace.record(
                    f"prefill_batched.chunk{chunk_idx}.forward({n_to_process}tok)",
                    _t_fwd,
                )

                # TP-rank synchronization point — same pattern as the
                # serial ``prefill()`` chunk loop. Guards against rank
                # drift before the next chunk's all_sum collectives fire.
                _t_barrier = time.perf_counter()
                mx_barrier(group)
                request_trace.record(
                    f"prefill_batched.chunk{chunk_idx}.barrier", _t_barrier
                )

                if distributed_prompt_progress_callback is not None:
                    _t_cb = time.perf_counter()
                    distributed_prompt_progress_callback()
                    request_trace.record(
                        f"prefill_batched.chunk{chunk_idx}.distributed_cb", _t_cb
                    )

                _t_eval = time.perf_counter()
                mx.eval([c.state for c in batched_cache])
                request_trace.record(
                    f"prefill_batched.chunk{chunk_idx}.eval_cache", _t_eval
                )
                mx.clear_cache()

                offset += n_to_process
                chunk_idx += 1
                if on_prefill_progress is not None:
                    on_prefill_progress(offset, max_length)

                if chunk_idx % 5 == 0 or offset >= max_length:
                    active_gb = mx.metal.get_active_memory() / 1024**3
                    peak_gb = mx.metal.get_peak_memory() / 1024**3
                    logger.info(
                        f"[MEM] batched prefill chunk {chunk_idx} ({offset}/{max_length}): "
                        f"active={active_gb:.2f} GB, peak={peak_gb:.2f} GB"
                    )

        if has_padding:
            with T("prefill_batched.finalize"):
                for c in batched_cache:
                    c.finalize()
                mx.eval([c.state for c in batched_cache])
                mx.clear_cache()
    except PrefillCancelled:
        set_pipeline_prefill(model, is_prefill=False)
        raise

    set_pipeline_prefill(model, is_prefill=False)

    try:
        _log_cache_profile(
            f"after batched prefill (B={n_streams} L={max_length})", batched_cache
        )
    except Exception:
        pass

    # Extract per-stream caches. Cache offset is already at
    # ``full_lengths[i] - 1`` (= caller's prompt[:-1] length minus 1)
    # because we processed one fewer token than passed in. No trim needed.
    with T("prefill_batched.extract"):
        per_stream_caches: list[KVCacheType] = []
        for i in range(n_streams):
            per_stream = [c.extract(i) for c in batched_cache]
            per_stream_caches.append(per_stream)
        mx.eval([
            c.state for stream_cache in per_stream_caches for c in stream_cache  # type: ignore
        ])

    # Release the merged batched cache + Metal allocator pool. batched_cache
    # holds the FULL merged KV (B streams × full_length × hidden) — after
    # extract() copies per-stream slices into per_stream_caches, the merged
    # buffer is dead. Without explicit del + clear_cache, it lingers until
    # function return (Python GC) AND its Metal buffers stay in the
    # allocator cache pool. Over many successive prefills on a growing leaf
    # (the Hermes reasoning pattern), this accumulates: +1-2 GB per prefill,
    # climbing 13->19 GB excess over ~5 min (fact 778). Force-free here.
    del batched_cache
    with T("prefill_batched.clear_cache_post_extract"):
        mx.clear_cache()

    elapsed = time.perf_counter() - start_time
    per_stream_tps = [
        (length / elapsed) if elapsed > 0 else 0.0 for length in lengths
    ]
    # Empty SSM snapshots — DSv4 doesn't use ArraysCache. Maintained for
    # API parity with ``prefill()``'s return signature.
    per_stream_snapshots: list[list[CacheSnapshot]] = [[] for _ in range(n_streams)]

    logger.info(
        f"Batched prefill: B={n_streams} max_L={max_length} done in "
        f"{elapsed:.2f}s ({sum(lengths) / elapsed:.1f} tok/s aggregate)"
    )

    return per_stream_tps, lengths, per_stream_caches, per_stream_snapshots


def _serial_prefill_fallback(
    model: Model,
    tokenizer: TokenizerWrapper,
    sampler: Callable[[mx.array], mx.array],
    prompt_tokens_list: list[mx.array],
    cache_list: list[KVCacheType],
    group: mx.distributed.Group | None,
    on_prefill_progress: Callable[[int, int], None] | None,
    distributed_prompt_progress_callback: Callable[[], None] | None,
    prefill_step_size: int | None,
) -> tuple[list[float], list[int], list[KVCacheType], list[list[CacheSnapshot]]]:
    """Fallback when batched prefill can't be applied (SSM caches, empty prompt).

    Runs the original ``prefill()`` per stream in sequence. Caller still gets
    the same return shape as the batched path.
    """
    per_stream_tps: list[float] = []
    per_stream_tokens: list[int] = []
    per_stream_snapshots: list[list[CacheSnapshot]] = []
    for prompt_tokens, cache in zip(prompt_tokens_list, cache_list, strict=True):
        tps, tokens, snapshots = prefill(
            model,
            tokenizer,
            sampler,
            prompt_tokens,
            cache,
            group,
            on_prefill_progress,
            distributed_prompt_progress_callback,
            prefill_step_size=prefill_step_size,
        )
        per_stream_tps.append(tps)
        per_stream_tokens.append(tokens)
        per_stream_snapshots.append(snapshots)
    return per_stream_tps, per_stream_tokens, list(cache_list), per_stream_snapshots


def warmup_inference(
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    model_id: ModelId,
) -> int:
    logger.info(f"warming up inference for instance: {model_id}")

    content = InputMessageContent(
        "Prompt to warm up the inference engine. Repeat this."
    )

    warmup_task_params = TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=content)],
        max_output_tokens=50,
        temperature=0.0,
    )

    warmup_prompt = apply_chat_template(
        tokenizer=tokenizer,
        task_params=warmup_task_params,
    )

    tokens_generated = 0

    mx_barrier(group)

    logger.info("Generating warmup tokens")

    t = time.monotonic()

    for _r in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=warmup_task_params,
        prompt=warmup_prompt,
        kv_prefix_cache=None,
        group=group,
        is_warmup=True,
    ):
        tokens_generated += 1

    check_for_cancel_every = min(
        math.ceil(tokens_generated / min(time.monotonic() - t, 0.001)), 100
    )

    mx_barrier(group)

    logger.info(f"warmed up by generating {tokens_generated} tokens")
    if group is not None:
        # Control-plane sync: run on the coord subgroup (isolated call_id
        # counter + QPs) like every other non-model-forward collective, so
        # the model group's data QP carries ONLY all_sums. Required by the
        # jaccl reliable-optimistic path's standing recv pool, and correct
        # hygiene regardless (matches agree_on_tasks / mx_min_int).
        check_for_cancel_every = int(
            mx.max(
                mx.distributed.all_gather(
                    mx.array([check_for_cancel_every]),
                    group=get_coord_group(group),
                )
            ).item()
        )

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


def safe_think_token_id(tokenizer: TokenizerWrapper, attr: str) -> int | None:
    """Read tokenizer.think_start_id / tokenizer.think_end_id defensively.

    Both properties raise ValueError (not AttributeError) when the model's
    think delimiter tokenizes to MORE than one vocab token, so a plain
    getattr(tokenizer, attr, None) does NOT catch that case -- it only
    catches the attribute not existing at all. Callers that want "no
    reasoning-budget limiting for this tokenizer" as the safe fallback
    (rather than an unhandled crash) should go through this helper instead
    of touching the property directly.
    """
    try:
        value = getattr(tokenizer, attr, None)
    except Exception:  # noqa: BLE001 - deliberately broad; any failure to
        # resolve a think-token id must degrade to "disable the reasoning
        # budget limiter", never crash generation.
        return None
    return cast(int, value) if value is not None else None


def make_reasoning_budget_limiter(
    think_start_id: int | None,
    think_end_id: int | None,
    budget_tokens: int,
    starts_in_thinking: bool = False,
    prompt_token_count: int = 0,
    max_seconds: float | None = None,
) -> Callable[[mx.array, mx.array], mx.array] | None:
    """Force a clean end-of-thinking transition once a reasoning block has run
    for ``budget_tokens`` tokens -- OR ``max_seconds`` of wall-clock time --
    without closing on its own, whichever fires first.

    Root cause this addresses (confirmed live 2026-07-26, hard_eval.py against
    the exo cluster, tasks math_digit_sum / math_largest_prime_factor /
    math_binom_mod): DeepSeek-V4-Flash sometimes reaches a correct answer
    inside its reasoning, then falls into a self-doubt loop ("But earlier I
    got X? Actually I got X? Let's recheck...") that re-derives the same
    result verbatim (with wording drift each cycle) indefinitely, consuming
    the entire max_tokens budget on reasoning_content and leaving `content`
    empty. Confirmed NOT a decode-path artifact: reproduces identically on
    the fully non-speculative decode path (EXO_SPECULATIVE=0, no MTP, no
    DSpark, real temp=0 sampling engaged, no jaccl/transport issues) --
    a genuine model attractor state, not caught by the existing short-cycle
    degeneration kill-switch (EXO_LOOP_DETECT_MAX_PERIOD=8 tokens; this
    loop's period is 60-400+ tokens with paraphrase drift each cycle, not an
    exact repeat).

    Deliberately NOT a pattern-matcher for the self-doubt loop shape itself
    (that would be overfit to one observed prompt). Instead enforces an
    invariant that holds regardless of *why* reasoning is running long:
    reasoning must not be allowed to consume the entire generation budget
    and leave the answer channel empty. In the confirmed failure case the
    model reaches the correct answer well before the loop starts, so forcing
    an early close salvages a correct response instead of an empty one.

    ``max_seconds`` (added 2026-07-31): the token-only trigger has a real
    gap -- ``budget_tokens`` is typically ``max_output_tokens * FRACTION``
    (see ``_REASONING_BUDGET_FRACTION`` / ``_REASONING_BUDGET_MAX_TOKENS`` in
    batch_generate.py), and when a client sends no explicit
    ``max_output_tokens`` this falls through to the engine's hardcoded
    default (32168), yielding a ~24K-token budget. At this cluster's
    realistic decode throughput (~15-30 tok/s, occasionally slower under an
    unrelated known PP throughput-variance issue), exhausting a token-only
    budget that large can take 15-30+ minutes wall-clock -- confirmed via a
    real 20+ minute incident (2026-07-31) where this mechanism WAS engaged
    and WOULD have eventually intervened, just far too slowly to be useful
    protection. A token count alone cannot bound wall-clock time across a
    throughput range that varies this much (plus the separate known
    degradation case), so ``max_seconds`` is an INDEPENDENT second trigger
    checked on the same per-token callback (near-zero added cost): whichever
    of budget_tokens or max_seconds is reached first forces the close. This
    keeps the token cap as the primary, cheap, deterministic-given-token-
    count mechanism (good for reproducibility/eval consistency -- the same
    prompt gets the same reasoning depth under normal load) while adding a
    real ceiling on worst-case intervention latency regardless of transient
    cluster slowness. Do NOT make the whole mechanism purely time-based --
    reasoning depth becoming a function of transient load (same prompt cut
    off at a different token count depending on how busy the cluster is)
    would hurt eval reproducibility; time is a backstop, not the primary
    trigger.

    CRITICAL (found 2026-07-26 after this shipped as a no-op): DeepSeek-V4's
    chat template appends a literal <think> suffix to the PROMPT itself --
    the model is never expected to (and does not) generate an opening
    <think> token of its own. A naive "scan generated history for
    think_start_id" never finds one, so this processor silently never
    engaged. This mirrors exactly what the existing stream parser
    (parse_thinking_models) already handles via its own starts_in_thinking
    parameter, computed once via detect_thinking_prompt_suffix(prompt,
    tokenizer) before generation starts -- callers of THIS function must
    thread the same signal through, since a stateless (history, logits) ->
    logits function has no access to the original prompt text on its own.

    Args:
        think_start_id: vocab id of the model's think-open delimiter (e.g.
            TokenizerWrapper.think_start_id), or None if the model/tokenizer
            doesn't expose one (no-op).
        think_end_id: vocab id of the think-close delimiter, or None (no-op).
        budget_tokens: once this many tokens have elapsed since the most
            recent (still-open) think_start_id -- or since generation start,
            when starts_in_thinking=True and no explicit open token exists
            in the stream -- force think_end_id as the only viable next
            token. Must be > 0 for the processor to do anything; <= 0 is
            treated as "no budget" (no-op, matching the existing convention
            of collapsing disabled knobs to None before they reach
            make_logits_processors-style call sites).
        starts_in_thinking: True when the prompt itself already ends with
            the opening think delimiter (detect_thinking_prompt_suffix()),
            so the model will never generate one. When True and no
            think_start_id is ever found in the stream, thinking is treated
            as open starting at prompt_token_count (the first generated
            token) rather than never-entered.
        prompt_token_count: number of prompt tokens in ``history`` before
            generation begins. Only used when starts_in_thinking=True and no
            explicit think_start_id is found -- anchors the budget window to
            the start of GENERATION, not the start of the prompt (a long
            prompt must not eat into the reasoning budget).
        max_seconds: wall-clock ceiling on how long thinking may stay open,
            independent of budget_tokens -- see rationale above. None or
            <= 0 disables the time-based trigger (token-only, matching the
            pre-2026-07-31 behavior). The clock starts on the processor's
            FIRST invocation for this generation (not construction time),
            so it measures actual decode wall-clock, not queueing/setup
            time before the first token.

    Returns:
        A logits processor, or None if thinking isn't supported by this
        tokenizer/model or budget_tokens <= 0 AND max_seconds is disabled
        (so callers can skip adding it to the processor list entirely --
        zero cost when both triggers are inapplicable, same pattern as
        repetition_penalty==1.0 collapsing to None).
    """
    if think_start_id is None or think_end_id is None:
        return None
    _time_enabled = max_seconds is not None and max_seconds > 0
    if budget_tokens <= 0 and not _time_enabled:
        return None

    _start_wall: list[float] = []  # single-element mutable cell (closure)

    def proc(history: mx.array, logits: mx.array) -> mx.array:
        # history is the full token sequence generated so far for this
        # stream (prompt + completion). Find the most recent think_start;
        # if a think_end occurs AFTER it, thinking has already closed and
        # this is a no-op (covers the normal single-block case AND the
        # legitimate re-entry case where the model reopens <think> itself).
        ids: list[int] = [int(t) for t in history.reshape(-1).tolist()]  # type: ignore
        start_idx = -1
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == think_start_id:
                start_idx = i
                break
        if start_idx < 0:
            if not starts_in_thinking:
                return logits  # never entered thinking, and the prompt
                # didn't imply an implicit open either -- genuine no-op.
            # Prompt-implied thinking (no literal <think> token exists in
            # the stream to find): anchor the window to the first
            # GENERATED token, not token 0 of the whole prompt+completion
            # sequence, so a long prompt doesn't eat into the budget.
            start_idx = max(prompt_token_count - 1, -1)
        if think_end_id in ids[start_idx + 1:]:
            _start_wall.clear()  # closed -- reset clock for any re-entry
            return logits  # already closed after that open -- no-op
        elapsed = (len(ids) - 1) - start_idx
        over_token_budget = budget_tokens > 0 and elapsed >= budget_tokens
        over_time_budget = False
        if _time_enabled:
            now = time.monotonic()
            if not _start_wall:
                _start_wall.append(now)
            elif (now - _start_wall[0]) >= cast(float, max_seconds):
                over_time_budget = True
        if not over_token_budget and not over_time_budget:
            return logits
        # Over budget (token or time) and still open: force think_end_id as
        # the only viable token. Ban everything else rather than just
        # boosting think_end_id, so this is a hard guarantee, not a strong
        # nudge a determined sampler could still route around.
        forced = mx.full(logits.shape, -1e9, dtype=logits.dtype)
        forced[..., think_end_id] = 1e9
        return forced

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
    precomputed_indices: list[int] | None = None,
    precomputed_values: list[float] | None = None,
    precomputed_selected: float | None = None,
) -> tuple[float, list[TopLogprobItem]]:
    if (
        precomputed_indices is not None
        and precomputed_values is not None
        and precomputed_selected is not None
    ):
        top_indices_list: list[int] = precomputed_indices[:top_logprobs]
        top_values_list: list[float] = precomputed_values[:top_logprobs]
        selected_logprob = precomputed_selected
    else:
        selected_logprob_arr = logprobs[selected_token]
        top_logprobs = min(top_logprobs, logprobs.shape[0] - 1)
        top_indices = mx.argpartition(-logprobs, top_logprobs)[:top_logprobs]
        top_values = logprobs[top_indices]
        sort_order = mx.argsort(-top_values)
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]
        mx.eval(selected_logprob_arr, top_indices, top_values)
        selected_logprob = float(selected_logprob_arr.item())
        top_indices_list = top_indices.tolist()  # type: ignore
        top_values_list = top_values.tolist()  # type: ignore

    # Convert to list of TopLogprobItem
    top_logprob_items: list[TopLogprobItem] = []
    for token_id, token_logprob in zip(top_indices_list, top_values_list, strict=True):
        if math.isnan(token_logprob):
            continue

        # Decode token ID to string
        token_str = tokenizer.decode([token_id])
        top_logprob_items.append(
            TopLogprobItem(
                token=token_str,
                logprob=token_logprob,
                bytes=list(token_str.encode("utf-8")),
            )
        )

    return selected_logprob, top_logprob_items


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
    vision_processor: VisionProcessor | None = None,
    is_warmup: bool = False,
    max_kv_tokens: int | None = None,
    prefill_step_size: int | None = None,
    instance_temperature: float | None = None,
    instance_top_p: float | None = None,
    instance_top_k: int | None = None,
    instance_min_p: float | None = None,
    instance_presence_penalty: float | None = None,
    instance_repetition_penalty: float | None = None,
    instance_frequency_penalty: float | None = None,
) -> Generator[GenerationResponse]:
    # Ensure that generation stats only contains peak memory for this generation
    mx.reset_peak_memory()
    # API-admitted requests always arrive with a resolved seed (random when
    # the client sent none — see _send_text_generation_with_images in
    # api/main.py, mirroring _ensure_seed's "distributed consistency"
    # contract for images). The fixed 42 below is only reachable for
    # engine-internal/bench constructions that bypass the API, where
    # reproducibility is desirable. `is not None` (not `or`): an explicit
    # seed of 0 is a valid client choice and must not silently become 42.
    seed = task.seed if task.seed is not None else 42
    mx.random.seed(seed)

    # Encode prompt once at the top and fix unmatched think tags
    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    all_prompt_tokens = fix_unmatched_think_end_tokens(all_prompt_tokens, tokenizer)
    min_prefix_hit_length = max(1000, system_prompt_token_count(task, tokenizer))

    vision: VisionResult | None = None
    if vision_processor is not None:
        try:
            vision = prepare_vision(
                images=task.images,
                chat_template_messages=task.chat_template_messages,
                vision_processor=vision_processor,
                tokenizer=tokenizer,
                model=model,
                model_id=task.model,
                task_params=task,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Vision processing failed, falling back to text-only"
            )
    if vision is not None:
        all_prompt_tokens = vision.prompt_tokens
    media_regions: list[MediaRegion] = vision.media_regions if vision else []

    # Do not use the prefix cache if we are trying to do benchmarks.
    is_bench = task.bench
    if is_bench and not task.use_prefix_cache:
        kv_prefix_cache = None

    # Use prefix cache if available, otherwise create fresh cache
    prefix_hit_length = 0
    matched_index: int | None = None
    is_exact_hit = False
    if kv_prefix_cache is None:
        caches = make_kv_cache(model=model, max_kv_size=max_kv_tokens)
        prompt_tokens = all_prompt_tokens
    else:
        caches, prompt_tokens, matched_index, is_exact_hit = (
            kv_prefix_cache.get_kv_cache(
                model, all_prompt_tokens, media_regions=media_regions
            )
        )
        prefix_hit_length = len(all_prompt_tokens) - len(prompt_tokens)
        if prefix_hit_length > 0:
            logger.info(
                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
            )

    _card = card_sampling_values(task.model, task.enable_thinking)
    _resolved = resolve_sampling(
        request_temperature=task.temperature,
        request_top_p=task.top_p,
        request_top_k=task.top_k,
        request_min_p=task.min_p,
        request_presence_penalty=task.presence_penalty,
        request_repetition_penalty=task.repetition_penalty,
        request_frequency_penalty=task.frequency_penalty,
        instance_temperature=instance_temperature,
        instance_top_p=instance_top_p,
        instance_top_k=instance_top_k,
        instance_min_p=instance_min_p,
        instance_presence_penalty=instance_presence_penalty,
        instance_repetition_penalty=instance_repetition_penalty,
        instance_frequency_penalty=instance_frequency_penalty,
        card_temperature=_card.temperature if _card else None,
        card_top_p=_card.top_p if _card else None,
        card_top_k=_card.top_k if _card else None,
        card_min_p=_card.min_p if _card else None,
        card_presence_penalty=_card.presence_penalty if _card else None,
        card_repetition_penalty=_card.repetition_penalty if _card else None,
        card_frequency_penalty=_card.frequency_penalty if _card else None,
    )

    # 1.0 is a no-op for repetition_penalty — collapse to None so mlx-lm
    # skips the processor instead of running per-token mul-by-1 work.
    # Passing context_size=None to mlx-lm's processor crashes inside it
    # (`tokens[-None:]`), so always coerce to its default 20.
    _rp = _resolved["repetition_penalty"]
    if _rp == 1.0:
        _rp = None
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
        make_logits_processors(
            repetition_penalty=_rp,
            repetition_context_size=task.repetition_context_size or 20,
            presence_penalty=_resolved["presence_penalty"],
            presence_context_size=task.presence_context_size or 20,
            frequency_penalty=_resolved["frequency_penalty"],
        )
    )
    if is_bench:
        # Only sample length eos tokens
        eos_ids = eos_ids_from_tokenizer(tokenizer)
        logits_processors = [ban_token_ids(eos_ids)] + logits_processors

    sampler = make_sampler(
        temp=_resolved["temp"],
        top_p=_resolved["top_p"],
        top_k=_resolved["top_k"],
        min_p=_resolved["min_p"],
    )

    # Normalize stop sequences to a list
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None
        else []
    )
    max_stop_len = max((len(s) for s in stop_sequences), default=0)

    maybe_vision_ctx = (
        patch_embed_tokens(
            model,
            vision.embeddings,
            prefix_hit_length,
            len(prompt_tokens) - 1,
            image_token_id=vision.image_token_id,
        )
        if vision is not None
        else contextlib.nullcontext()
    )
    use_remote = (
        len(prompt_tokens) > REMOTE_PREFILL_MIN_TOKENS
        and task.prefill_endpoint is not None
    )
    remote_prefilled = False
    prefill_tps = 0.0
    prefill_tokens = 0
    ssm_snapshots_list: list[CacheSnapshot] = []
    with maybe_vision_ctx:
        if use_remote and task.prefill_endpoint is not None:
            try:
                prefill_tps, prefill_tokens, ssm_snapshots_list = remote_prefill(
                    prompt_tokens[:-1],
                    caches,
                    on_prefill_progress,
                    endpoint=task.prefill_endpoint,
                    request_id=str(uuid.uuid4()),
                    model_id=str(task.model),
                    start_pos=prefix_hit_length,
                )
                remote_prefilled = True
            except Exception:
                logger.opt(exception=True).warning(
                    "Remote prefill failed, falling back to local prefill"
                )
        if not remote_prefilled:
            # Keep prefill_step_size kwarg — required for our DSv4-Flash
            # `EXO_DSV4_PREFILL_STEP_SIZE=256` tuning (per dsv4_prefill_chunk_size_curve memory).
            prefill_tps, prefill_tokens, ssm_snapshots_list = prefill(
                model,
                tokenizer,
                sampler,
                prompt_tokens[:-1],
                caches,
                group,
                on_prefill_progress,
                distributed_prompt_progress_callback,
                prefill_step_size=prefill_step_size,
                snapshot_offset=prefix_hit_length,
            )
    cache_snapshots: list[CacheSnapshot] | None = ssm_snapshots_list or None

    if kv_prefix_cache is not None and matched_index is not None and is_exact_hit:
        prefill_tps = kv_prefix_cache.prefill_tps[matched_index]

    if kv_prefix_cache is not None:
        hit_ratio = (
            prefix_hit_length / len(all_prompt_tokens)
            if len(all_prompt_tokens) > 0
            else 0.0
        )
        if matched_index is not None and (
            prefix_hit_length >= min_prefix_hit_length
            and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
        ):
            kv_prefix_cache.update_kv_cache(
                matched_index,
                all_prompt_tokens,
                caches,
                cache_snapshots,
                restore_pos=prefix_hit_length,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
                low_priority=task.low_priority,
                high_priority=task.high_priority,
            )
        else:
            kv_prefix_cache.add_kv_cache(
                all_prompt_tokens,
                caches,
                cache_snapshots,
                media_regions=media_regions,
                prefill_tps=prefill_tps,
                low_priority=task.low_priority,
                high_priority=task.high_priority,
            )

    # stream_generate starts from the last token
    last_token = prompt_tokens[-2:]

    max_tokens = task.max_output_tokens or MAX_TOKENS
    accumulated_text = ""
    generated_text_parts: list[str] = []
    generation_start_time = time.perf_counter()
    usage: Usage | None = None

    # Memory checkpoint after prefill, before decode
    mx.eval(mx.zeros(1))
    active_gb = mx.metal.get_active_memory() / 1024**3
    peak_gb = mx.metal.get_peak_memory() / 1024**3
    cache_gb = mx.metal.get_cache_memory() / 1024**3
    logger.info(f"[MEM] after prefill, before decode: active={active_gb:.2f} GB, peak={peak_gb:.2f} GB, cache={cache_gb:.2f} GB")
    try:
        _log_cache_profile("after prefill (serial cache)", caches)
    except Exception:
        pass
    logger.info("Starting decode")
    mx_barrier(group)

    # --- PP idle-time speculation (skipped during warmup) ---
    _pp_spec_gen = None
    _pp_draft = getattr(model, "_pp_draft_model", None)
    _pp_draft_cache = getattr(model, "_pp_draft_cache", None)
    # Both ranks must enter the speculation loop — check env var, not model attribute
    # (draft model is only on rank 0, but rank 1 must participate in the protocol)
    _has_pp_draft = bool(os.environ.get("EXO_PP_DRAFT_MODEL", ""))
    logger.info(f"PP spec check: is_warmup={is_warmup}, has_draft_env={_has_pp_draft}, "
                f"draft_model={'yes' if _pp_draft else 'no'}, "
                f"group={'size=' + str(group.size()) if group else 'None'}")
    if (not is_warmup
        and _has_pp_draft
        and group is not None
        and group.size() > 1):
        try:
            from ..pp_speculation import (
                get_pipeline_info,
                pp_speculative_decode_loop,
                _install_spec_layers,
                _configure_layers,
            )
            pp_info = get_pipeline_info(model)
            logger.info(f"PP spec: get_pipeline_info returned {pp_info}")
            if pp_info is not None:
                pp_rank, pp_world_size, pp_group = pp_info
                inner = getattr(model, "language_model", model)
                _install_spec_layers(inner)

                # Prefill draft cache with tail of prompt (rank 0 only, instant — no PP needed)
                # The draft model uses a RotatingKVCache, so only recent tokens matter.
                if pp_rank == 0 and _pp_draft is not None:
                    _draft_kv_window = int(os.environ.get("EXO_DRAFT_KV_WINDOW", "4096"))
                    _draft_prompt = all_prompt_tokens[-_draft_kv_window:]
                    _draft_chunk = 512
                    for i in range(0, len(_draft_prompt), _draft_chunk):
                        _pp_draft(_draft_prompt[i:i + _draft_chunk][None], cache=_pp_draft_cache)
                        mx.eval([c.state if hasattr(c, 'state') else c for c in _pp_draft_cache])
                    logger.info(f"Draft model prefilled with {len(_draft_prompt)} tokens (of {len(all_prompt_tokens)} total)")

                # First token via standard PP (both ranks, synchronized)
                _first_gen = stream_generate(
                    model=model, tokenizer=tokenizer, prompt=last_token,
                    max_tokens=1, sampler=sampler, logits_processors=logits_processors,
                    prompt_cache=caches, prefill_step_size=1,
                    kv_group_size=KV_GROUP_SIZE, kv_bits=KV_BITS,
                )
                _first_out = next(_first_gen)
                first_y = mx.array([_first_out.token])
                mx.eval(first_y)

                logger.info(f"PP speculation active: rank={pp_rank}")

                def _spec_token_gen():
                    from mlx_lm.generate import GenerationResponse
                    _detok = tokenizer.detokenizer
                    gen_start = time.perf_counter()
                    # Clear finish_reason from max_tokens=1 — this is just the first token
                    _first_fixed = GenerationResponse(
                        text=_first_out.text, token=_first_out.token,
                        logprobs=_first_out.logprobs, from_draft=False,
                        prompt_tokens=_first_out.prompt_tokens,
                        prompt_tps=_first_out.prompt_tps,
                        generation_tokens=_first_out.generation_tokens,
                        generation_tps=_first_out.generation_tps,
                        peak_memory=_first_out.peak_memory,
                        finish_reason=None,
                    )
                    yield _first_fixed

                    for tok_id, lp in pp_speculative_decode_loop(
                        model=model, draft_model=_pp_draft,
                        prompt_cache=caches, draft_cache=_pp_draft_cache,
                        sampler=sampler, logits_processors=logits_processors,
                        first_y=first_y, first_logprobs=mx.zeros(1),
                        max_tokens=max_tokens - 1,
                        pp_rank=pp_rank, pp_world_size=pp_world_size,
                        pp_group=pp_group,
                    ):
                        if tok_id in tokenizer.eos_token_ids:
                            elapsed = time.perf_counter() - gen_start
                            yield GenerationResponse(
                                text="", token=tok_id, logprobs=lp, from_draft=False,
                                prompt_tokens=len(last_token), prompt_tps=prefill_tps or 0.0,
                                generation_tokens=1, generation_tps=1.0/elapsed if elapsed > 0 else 0,
                                peak_memory=mx.get_peak_memory()/1e9, finish_reason="stop",
                            )
                            return
                        _detok.add_token(tok_id)
                        elapsed = time.perf_counter() - gen_start
                        yield GenerationResponse(
                            text=_detok.last_segment, token=tok_id, logprobs=lp, from_draft=False,
                            prompt_tokens=len(last_token), prompt_tps=prefill_tps or 0.0,
                            generation_tokens=1, generation_tps=1.0/elapsed if elapsed > 0 else 0,
                            peak_memory=mx.get_peak_memory()/1e9,
                        )

                _pp_spec_gen = _spec_token_gen()
        except Exception as e:
            sys.stderr.write(f"[PP speculation] setup failed: {e}\n")
            sys.stderr.flush()
            _pp_spec_gen = None

    _decode_gen = _pp_spec_gen if _pp_spec_gen is not None else stream_generate(
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

    # EXO_DECODE_PROBE: aggregate wall + GPU time over windows of N tokens.
    # gpu_time_ns() is async — populated by Metal completion handlers, so
    # per-iter deltas read ~0 immediately after enqueue. We instead capture
    # gpu_time_ns + wall at every Nth token and compute the WINDOW delta,
    # which gives accurate "GPU% busy" because by token K+N the GPU has
    # actually completed the cycle-K command buffer.
    _exo_probe = bool(os.environ.get("EXO_DECODE_PROBE"))
    _exo_probe_every = int(os.environ.get("EXO_DECODE_PROBE_EVERY", "16"))
    _exo_window_t0 = time.perf_counter() if _exo_probe else 0.0
    _exo_window_g0 = mx.metal.gpu_time_ns() if _exo_probe else 0
    _exo_cnt = 0

    for completion_tokens, out in enumerate(
        _decode_gen,
        start=1,
    ):
        if _exo_probe:
            _exo_cnt += 1
            if _exo_cnt % _exo_probe_every == 0:
                _t = time.perf_counter()
                _g = mx.metal.gpu_time_ns()
                _wall_ms = (_t - _exo_window_t0) * 1000.0
                _gpu_ms = (_g - _exo_window_g0) / 1e6
                _per_wall = _wall_ms / _exo_probe_every
                _per_gpu = _gpu_ms / _exo_probe_every
                _pct = _per_gpu / _per_wall * 100 if _per_wall > 0 else 0.0
                import sys as _sys
                _sys.stderr.write(
                    f"[EXO_DECODE_PROBE pid={os.getpid()}] tokens={_exo_cnt} "
                    f"wall_ms={_per_wall:.2f} gpu_ms={_per_gpu:.2f} gpu_pct={_pct:.1f}\n"
                )
                _sys.stderr.flush()
                _exo_window_t0 = _t
                _exo_window_g0 = _g
        generated_text_parts.append(out.text)
        accumulated_text += out.text

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
        if is_done:
            # Classify prefix-cache outcome from the request setup
            # earlier in this function. Field was previously left at
            # default "none" so the metric always read 0% even when
            # the cache was hitting.
            if is_exact_hit:
                prefix_cache_kind: Literal["none", "partial", "exact"] = "exact"
            elif prefix_hit_length > 0:
                prefix_cache_kind = "partial"
            else:
                prefix_cache_kind = "none"

            stats = GenerationStats(
                prompt_tps=float(prefill_tps or out.prompt_tps),
                generation_tps=float(out.generation_tps),
                prompt_tokens=int(prefill_tokens + out.prompt_tokens),
                generation_tokens=int(out.generation_tokens),
                peak_memory_usage=Memory.from_gb(out.peak_memory),
                prefix_cache_hit=prefix_cache_kind,
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
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
            )

        # Extract logprobs from the full vocabulary logprobs array
        logprob: float | None = None
        top_logprobs: list[TopLogprobItem] | None = None
        if task.logprobs:
            with mx.stream(generation_stream):
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=out.logprobs,
                    tokenizer=tokenizer,
                    top_logprobs=task.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=out.token,
                )

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
        )

        if is_done:
            mx_barrier(group)
            break

        # Limit accumulated_text to what's needed for stop sequence detection
        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]

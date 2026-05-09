import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Literal, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.generate import (
    GenerationBatch,
    generation_stream,
    stream_generate,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import (
    DEFAULT_TOP_LOGPROBS,
    KV_BITS,
    KV_GROUP_SIZE,
    MAX_TOKENS,
)
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    patch_embed_tokens,
    prefill,
    prefill_batched,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    set_needs_topk,
    take_ready_topk,
)
from exo.worker.engines.mlx.sampling import card_sampling_values, resolve_sampling
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    get_coord_group,
    mx_any,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5
REMOTE_PREFILL_MIN_TOKENS = 1000


_MEM_PROFILE_PATH = os.environ.get("EXO_MEMORY_PROFILE_PATH")
_MEM_PROFILE_INTERVAL = int(os.environ.get("EXO_MEMORY_PROFILE_INTERVAL", "256"))

# Periodic mx.clear_cache() to release MLX's caching allocator pool back
# to the OS. Without this, the allocator holds freed GPU buffers for reuse
# indefinitely; IOGPU/Metal residency descriptors track all of them and
# count toward process RSS even when the buffers themselves aren't
# "active" from MLX's perspective. On long Think-mode decode (>50K tokens)
# this is the dominant source of RSS growth and ultimately what OOMs the
# runner — not the bf16 KV cache scaling we initially suspected.
#
# Trade-off: clearing forces subsequent allocations to come from a cold
# pool, costing decode tok/s. Empirical sweet spot TBD; defaults to off.
_MLX_CLEAR_CACHE_INTERVAL = int(
    os.environ.get("EXO_MLX_CLEAR_CACHE_INTERVAL", "0")
)

# Periodic gc.collect() to break Python ref cycles on the MLX array graph.
# `heap` snapshot during DSv4 long decode showed 2.4M std::__shared_ptr_emplace
# <mlx::core::array::ArrayDesc> instances accumulating (~240 per decode token,
# never freed) even though `mlx::core::eval` calls `array::detach()` after
# evaluation. The ArrayDescs were held alive by Python wrapper references
# trapped in ref cycles until Python's gen-0/gen-1/gen-2 collectors finally
# kicked in. By that point steady-state had drifted significantly.
#
# Cost: gc.collect() walks the full heap. Cheap once, but expensive if called
# per decode step. Default 256 is a balance between leak control and GC
# overhead. Set to 0 to disable.
import gc  # noqa: E402  (placed after env reads for clarity)
_GC_COLLECT_INTERVAL = int(os.environ.get("EXO_GC_COLLECT_INTERVAL", "0"))

# Periodic macOS malloc_zone_pressure_relief() to force freed-but-cached
# chunks back to the OS. The MLX C++ side is correctly releasing
# ArrayDesc shared_ptr instances on eval+detach (verified via the live
# atomic counter — oscillates around 500K, doesn't grow), but the
# underlying libsystem_malloc holds the freed control-block chunks for
# reuse and doesn't shrink RSS until pressure_relief() is called or
# memory pressure forces eviction. Without this, process RSS grows at
# ~155 KB/decoded-token even though *live* allocations are bounded.
#
# malloc_zone_pressure_relief(NULL, 0) asks ALL malloc zones to munmap
# their freed pages. Returns bytes released. Cost: walks each zone's
# free list (microseconds typically; can be milliseconds under heavy
# fragmentation).
import ctypes  # noqa: E402
_MALLOC_RELIEF_INTERVAL = int(os.environ.get("EXO_MALLOC_RELIEF_INTERVAL", "0"))
_libsystem_malloc: Any = None
_malloc_zone_pressure_relief: Any = None
if _MALLOC_RELIEF_INTERVAL > 0:
    try:
        _libsystem_malloc = ctypes.CDLL("/usr/lib/system/libsystem_malloc.dylib")
        _malloc_zone_pressure_relief = (
            _libsystem_malloc.malloc_zone_pressure_relief
        )
        _malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _malloc_zone_pressure_relief.restype = ctypes.c_size_t
    except Exception as exc:
        logger.warning(
            f"libsystem_malloc.malloc_zone_pressure_relief unavailable: {exc}"
        )
        _malloc_zone_pressure_relief = None

# tracemalloc-based Python heap top-allocator dump. Enabled when
# EXO_TRACEMALLOC_PATH is set. Captures a snapshot every
# EXO_TRACEMALLOC_INTERVAL decode steps and writes the top-N growers
# (compared to the prior snapshot) to the path. Massive overhead — only
# enable for memory-leak hunts, not perf measurement.
_TRACEMALLOC_PATH = os.environ.get("EXO_TRACEMALLOC_PATH")
_TRACEMALLOC_INTERVAL = int(os.environ.get("EXO_TRACEMALLOC_INTERVAL", "2000"))
_TRACEMALLOC_TOP_N = int(os.environ.get("EXO_TRACEMALLOC_TOP_N", "20"))
_tracemalloc_prev_snapshot: Any = None

if _TRACEMALLOC_PATH:
    import tracemalloc as _tracemalloc
    _tracemalloc.start(25)


def _tracemalloc_dump(profile_path: str, step: int, tokens: int) -> None:
    """Diff current tracemalloc snapshot against the previous one and write
    the top-N growers (filename:lineno → bytes-grown) to a JSONL log.

    First call records the baseline and emits no diff. Subsequent calls
    write the growth between consecutive intervals — exactly what we need
    to spot per-token leaks.
    """
    global _tracemalloc_prev_snapshot
    try:
        snap = _tracemalloc.take_snapshot()
        if _tracemalloc_prev_snapshot is None:
            _tracemalloc_prev_snapshot = snap
            return
        stats = snap.compare_to(_tracemalloc_prev_snapshot, "lineno")[
            :_TRACEMALLOC_TOP_N
        ]
        record = {
            "ts": time.time(),
            "step": step,
            "tokens": tokens,
            "top_growers": [
                {
                    "loc": str(s.traceback[0]) if s.traceback else "?",
                    "size_diff_bytes": int(s.size_diff),
                    "count_diff": int(s.count_diff),
                    "size_bytes": int(s.size),
                    "count": int(s.count),
                }
                for s in stats
            ],
        }
        with open(profile_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        _tracemalloc_prev_snapshot = snap
    except Exception as exc:
        logger.warning(f"tracemalloc dump failed: {exc}")


def _mem_profile_record(
    profile_path: str,
    step_count: int,
    total_tokens: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a memory snapshot to the profile JSONL.

    Captures GPU active and peak memory (Metal-side), then resets peak so
    each window's `peak_bytes` reflects the high-water mark since the
    previous snapshot — that's what tells us about transient spikes.
    `active_bytes` is the currently-allocated steady-state.

    `extra` lets callers stamp event-specific metadata (e.g. `phase=startup`,
    `phase=after_prefill`) for offline analysis.
    """
    try:
        active = int(mx.metal.get_active_memory())
        peak = int(mx.metal.get_peak_memory())
        cache = int(mx.metal.get_cache_memory())
        record: dict[str, Any] = {
            "ts": time.time(),
            "step": step_count,
            "tokens": total_tokens,
            "active_bytes": active,
            "peak_bytes": peak,
            "cache_bytes": cache,
        }
        # Fork-only: live ArrayDesc count. Sampled here so the heap snapshot
        # taken at the same wall-clock can be cross-referenced. Guarded
        # because upstream MLX and older fork pins don't expose it.
        live_arraydesc_fn = cast(
            "Callable[[], int] | None",
            getattr(mx.metal, "live_array_desc_count", None),
        )
        if live_arraydesc_fn is not None:
            try:
                record["live_array_desc"] = int(live_arraydesc_fn())
            except Exception:
                pass
        # Fork-only: per-primitive-type live ArrayDesc breakdown. Empty when
        # MLX_PER_TYPE_TRACK / MLX_PER_TYPE_DUMP_INTERVAL are unset. The map
        # is small (~50-100 entries) and the snapshot is mutex-protected;
        # keep this inside the existing throttled record path.
        per_type_fn = cast(
            "Callable[[], dict[str, int]] | None",
            getattr(mx.metal, "live_array_desc_count_by_type", None),
        )
        if per_type_fn is not None:
            try:
                snapshot = per_type_fn()
                if snapshot:
                    record["live_array_desc_by_type"] = {
                        k: int(v) for k, v in snapshot.items() if v > 0
                    }
            except Exception:
                pass
        try:
            import psutil
            mi = psutil.Process().memory_info()
            record["rss_bytes"] = int(mi.rss)
            record["vms_bytes"] = int(mi.vms)
        except Exception:
            pass
        if extra:
            record.update(extra)
        with open(profile_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        mx.metal.reset_peak_memory()
    except Exception as exc:
        logger.warning(f"memory profile write failed: {exc}")


def _mlx_gen_elapsed_seconds(mlx_gen: Any) -> float:
    """Best-effort cumulative generation time for an mlx-lm BatchGenerator.

    Older mlx-lm forks kept a ``_stats.generation_time`` counter. The new
    BatchGenerator tracks timing through a ``stats()`` context manager and
    exposes only a monotonic ``_steps_counter``. Fall through a known-good
    order; if nothing fits, use wall clock — tok/s stays meaningful.
    """
    stats = getattr(mlx_gen, "_stats", None)
    if stats is not None:
        gen_time = getattr(stats, "generation_time", None)
        if gen_time is not None:
            return float(gen_time)
    return time.perf_counter()


def _stop_sequences(task_params: TextGenerationTaskParams) -> list[str]:
    if task_params.stop is None:
        return []
    if isinstance(task_params.stop, str):
        return [task_params.stop]
    return task_params.stop


@dataclass
class _EngineTask:
    uid: int
    task_params: TextGenerationTaskParams
    all_prompt_tokens: mx.array
    prefix_hit_length: int
    matched_index: int | None
    cache_snapshots: list[CacheSnapshot] | None
    detokenizer: StreamingDetokenizer
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    potential_stop_sequence_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    generation_time_at_start: float = 0.0
    in_thinking: bool = False
    reasoning_tokens: int = 0
    prefill_tps: float = 0.0
    media_regions: list[MediaRegion] = field(default_factory=list)
    # Whether the radix-trie returned an exact-match (full prompt
    # already cached). Distinguishes "exact" from "partial" hits when
    # populating GenerationStats.prefix_cache_hit.
    is_exact_hit: bool = False


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None
    model_id: str = ""
    max_kv_tokens: int | None = None
    prefill_step_size: int | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_presence_penalty: float | None = None
    default_repetition_penalty: float | None = None
    default_frequency_penalty: float | None = None

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)
    _pp_spec_active: bool = field(init=False, default=False)
    _pp_spec_gen: Generator[tuple[int, mx.array], None, None] | None = field(init=False, default=None)
    _pp_spec_uid: int | None = field(init=False, default=None)
    _pp_spec_eos: set[int] = field(init=False, default_factory=set)
    _uid_counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        use_speculative = os.environ.get("EXO_SPECULATIVE", "0") == "1"
        stop_tokens = set(eos_ids_from_tokenizer(self.tokenizer))
        # mlx-lm's new BatchGenerator expects stop_tokens as Sequence[Sequence[int]]
        # — one "sequence" per stop. Wrap each EOS id so state-machine setup gets
        # the shape it expects.
        stop_tokens_seq = [[t] for t in stop_tokens]

        prefill_step_size = self.prefill_step_size or 4096

        if use_speculative:
            try:
                # DSv4 path: MTP module is part of the loaded model
                # (mlx-lm's deepseek_v4.sanitize keeps mtp.* keys when
                # num_nextn_predict_layers > 0). No separate weights
                # file needed; the spec generator constructs a thin
                # predictor wrapper over `model.model.mtp[0]`.
                inner = getattr(self.model, "model", None)
                is_dsv4_with_mtp = (
                    inner is not None
                    and type(inner).__name__ == "DeepseekV4Model"
                    and hasattr(inner, "mtp")
                    and len(inner.mtp) > 0
                )

                if is_dsv4_with_mtp:
                    from exo.worker.engines.mlx.speculative.dsv4_mtp import (
                        DSv4MTPBatchGenerator,
                        DSv4MTPPredictor,
                    )

                    gamma = int(os.environ.get("EXO_SPECULATIVE_GAMMA", "2"))
                    temp = float(os.environ.get("EXO_SPECULATIVE_TEMP", "0.0"))
                    alpha = float(os.environ.get("EXO_SPECULATIVE_ALPHA", "1.0"))
                    mtp_pred = DSv4MTPPredictor(self.model, mtp_idx=0)
                    self._mlx_gen = DSv4MTPBatchGenerator(
                        model=self.model,
                        mtp_predictor=mtp_pred,
                        gamma=gamma,
                        temp=temp,
                        alpha=alpha,
                        stop_tokens=stop_tokens_seq,
                        prefill_step_size=prefill_step_size,
                    )
                    logger.info(
                        f"DSv4 MTP speculative decoding enabled (γ={gamma}, T={temp})"
                    )
                else:
                    # Qwen3.5-style path: separate MTP weights file.
                    from exo.worker.engines.mlx.speculative.mtp_batch_generator import (
                        MTPBatchGenerator,
                    )
                    from exo.worker.engines.mlx.speculative.mtp_module import (
                        MTPPredictor,
                    )

                    mtp_weights = self._resolve_mtp_weights()
                    gamma = int(os.environ.get("EXO_SPECULATIVE_GAMMA", "2"))

                    if mtp_weights:
                        mtp = MTPPredictor(self.model, mtp_weights, quantize=False)
                        temp = float(os.environ.get("EXO_SPECULATIVE_TEMP", "0.7"))
                        alpha = float(os.environ.get("EXO_SPECULATIVE_ALPHA", "1.0"))
                        self._mlx_gen = MTPBatchGenerator(
                            model=self.model,
                            mtp_predictor=mtp,
                            gamma=gamma,
                            temp=temp,
                            alpha=alpha,
                            stop_tokens=stop_tokens_seq,
                            prefill_step_size=prefill_step_size,
                        )
                        logger.info(
                            f"MTP speculative decoding enabled (γ={gamma}, T={temp})"
                        )
                        # Skip warmup — OOMs on 397B (Metal abort, uncatchable)
                    else:
                        logger.warning(
                            "EXO_SPECULATIVE=1 but could not find MTP weights. Falling back."
                        )
                        self._mlx_gen = MlxBatchGenerator(
                            model=self.model,
                            stop_tokens=stop_tokens_seq,
                            prefill_step_size=prefill_step_size,
                        )
            except Exception as e:
                logger.warning(f"Failed to init MTP speculative: {e}. Falling back.")
                self._mlx_gen = MlxBatchGenerator(
                    model=self.model,
                    stop_tokens=stop_tokens_seq,
                    prefill_step_size=prefill_step_size,
                )
        else:
            self._mlx_gen = MlxBatchGenerator(
                model=self.model,
                stop_tokens=stop_tokens_seq,
                prefill_step_size=prefill_step_size,
            )

        self._mlx_gen._needs_topk = False  # pyright: ignore[reportAttributeAccessIssue]
        self._pp_spec_eos = set(eos_ids_from_tokenizer(self.tokenizer))

        if _MEM_PROFILE_PATH:
            _mem_profile_record(
                _MEM_PROFILE_PATH,
                step_count=0,
                total_tokens=0,
                extra={"phase": "post_init"},
            )
            logger.info(
                f"memory profile enabled → {_MEM_PROFILE_PATH} "
                f"(interval={_MEM_PROFILE_INTERVAL} steps)"
            )

        # Enable PP speculation if draft model is configured and we're in PP mode
        draft_path = os.environ.get("EXO_PP_DRAFT_MODEL", "")
        if draft_path and self.group is not None and self.group.size() > 1:
            try:
                from ..pp_speculation import get_pipeline_info
                if get_pipeline_info(self.model) is not None:
                    self._pp_spec_active = True
                    logger.info("PP speculation enabled in BatchGenerator")
                    # Load MTP for PP speculation (prefer pre-quantized 4-bit, 3.7GB)
                    if use_speculative:
                        mtp_weights = self._resolve_mtp_weights()
                        if mtp_weights:
                            try:
                                self._pp_mtp = MTPPredictor(
                                    self.model, mtp_weights, quantize=False,
                                )
                                logger.info(f"PP MTP loaded from {mtp_weights}")
                            except Exception as e:
                                import traceback
                                logger.warning(f"PP MTP load failed: {e}\n{traceback.format_exc()}")
                                self._pp_mtp = None
                        else:
                            self._pp_mtp = None
                    else:
                        self._pp_mtp = None
            except Exception:
                pass

    def _model_hidden_size(self) -> int | None:
        """Return the hidden_size of the loaded model, or None if undetectable.

        Used to validate MTP weight compatibility — MTP weights from a
        different model architecture (e.g. 397B's MTP loaded for 35B-A3B)
        will have a mismatched pre_fc_norm_hidden weight and crash at the
        first inference call.
        """
        try:
            args: Any = getattr(self.model, "args", None)
            if args is not None:
                tc: Any = getattr(args, "text_config", None)
                if isinstance(tc, dict) and "hidden_size" in tc:
                    return int(tc["hidden_size"])  # pyright: ignore[reportUnknownArgumentType]
                hs: Any = getattr(args, "hidden_size", None)
                if hs is not None:
                    return int(hs)
            inner: Any = getattr(self.model, "language_model", None) or self.model
            inner_args: Any = getattr(inner, "args", None)
            if inner_args is not None:
                hs2: Any = getattr(inner_args, "hidden_size", None)
                if hs2 is not None:
                    return int(hs2)
        except Exception:
            pass
        return None

    @staticmethod
    def _peek_mtp_hidden_size(weights_path: str) -> int | None:
        """Peek at an MTP safetensors file and return the hidden_size it
        was trained for, without loading the full file.

        Reads only the safetensors JSON header (a few KB), not the
        weight data. The MTP module's `pre_fc_norm_hidden` weight is a
        1-D tensor whose length equals the hidden_size of the model the
        MTP was distilled from. If that doesn't match the loaded model's
        hidden_size, the weights are incompatible.
        """
        import json
        import struct
        try:
            with open(weights_path, "rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return None
                header_size: int = struct.unpack("<Q", header_size_bytes)[0]
                header_bytes = f.read(header_size)
            header = cast(dict[str, Any], json.loads(header_bytes))
            for key in (
                "mtp.pre_fc_norm_hidden.weight",
                "mtp.norm.weight",
                "mtp.pre_fc_norm_embedding.weight",
            ):
                entry = header.get(key)
                if isinstance(entry, dict):
                    shape = cast(dict[str, Any], entry).get("shape")
                    if isinstance(shape, list) and shape:
                        first = cast(list[Any], shape)[0]
                        return int(first)
        except Exception:
            return None
        return None

    def _mtp_compatible_with_model(self, weights_path: str) -> bool:
        """Verify a candidate MTP weights file matches the loaded model.

        Logs a warning and returns False on mismatch so the caller can
        skip to the next candidate (or fall back to vanilla decoding).
        Returns True when shapes match OR when either side cannot be
        determined (best-effort — preserves prior behavior for unusual
        models where the check would otherwise be a false negative).
        """
        model_hidden = self._model_hidden_size()
        mtp_hidden = self._peek_mtp_hidden_size(weights_path)
        if model_hidden is None or mtp_hidden is None:
            return True
        if model_hidden != mtp_hidden:
            logger.warning(
                f"Skipping MTP weights at {weights_path}: "
                f"hidden_size {mtp_hidden} != model hidden_size {model_hidden}. "
                f"These weights are for a different model architecture."
            )
            return False
        return True

    def _resolve_mtp_weights(self) -> str | None:
        """Find MTP weights: explicit path, local model dir, or HF repo extraction.

        Detection order:
        1. EXO_MTP_WEIGHTS env var (explicit path)
        2. Pre-quantized cache (~/.cache/exo/mtp_weights/mtp_*_q4.safetensors)
        3. Bf16 cache (~/.cache/exo/mtp_weights/mtp_*.safetensors)
        4. Local model directory (check weight index for mtp.* keys)
        5. HF repo download (selective shard download)

        Every candidate is validated against the loaded model's hidden_size
        before being returned, so a stale cache from a previous run with a
        different model is rejected automatically.
        """
        import hashlib
        from pathlib import Path

        # 1. Explicit path
        explicit_path = os.environ.get("EXO_MTP_WEIGHTS", "")
        if explicit_path and os.path.exists(explicit_path):
            if self._mtp_compatible_with_model(explicit_path):
                return explicit_path
            return None

        # Determine source HF repo for MTP weights
        mtp_repo = os.environ.get("EXO_MTP_MODEL", "")
        if not mtp_repo:
            mtp_repo = self._detect_mtp_repo()
        if not mtp_repo:
            return None

        # 2-3. Check cache
        cache_dir = Path.home() / ".cache" / "exo" / "mtp_weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(mtp_repo.encode()).hexdigest()[:12]

        q4_path = cache_dir / f"mtp_{cache_key}_q4.safetensors"
        if q4_path.exists() and self._mtp_compatible_with_model(str(q4_path)):
            logger.info(f"Using pre-quantized MTP weights: {q4_path}")
            return str(q4_path)

        bf16_path = cache_dir / f"mtp_{cache_key}.safetensors"
        if bf16_path.exists() and self._mtp_compatible_with_model(str(bf16_path)):
            logger.info(f"Using cached MTP weights: {bf16_path}")
            return str(bf16_path)

        # 4. Check local model directory for MTP weights
        if self.model_id:
            from exo.download.download_utils import build_model_path
            from exo.shared.types.common import ModelId
            local_path = self._extract_mtp_from_local(
                build_model_path(ModelId(self.model_id)), cache_dir, cache_key
            )
            if local_path and self._mtp_compatible_with_model(local_path):
                return local_path

        # 5. Download from HF repo
        try:
            dl_path = self._extract_mtp_from_hf(mtp_repo)
            if dl_path and self._mtp_compatible_with_model(dl_path):
                return dl_path
        except Exception as e:
            logger.warning(f"Failed to extract MTP weights from {mtp_repo}: {e}")
        return None

    def _detect_mtp_repo(self) -> str:
        """Detect the HF repo containing MTP weights for this model.

        Checks model args for mtp_num_hidden_layers and model_type to determine
        which HF repo has the MTP weights. Returns '' if MTP not supported.

        Note: there is intentionally NO fallback to a hardcoded "default"
        repo per model_type. The qwen3_5_moe family contains multiple
        architectures with different hidden_sizes (e.g. 397B uses 4096,
        35B-A3B uses 2048), and silently picking one would load
        architecturally incompatible weights — see the rms_norm crash
        when 397B's MTP file was loaded into a 35B-A3B model.
        """
        try:
            inner = getattr(self.model, 'model', None) or self.model.language_model.model
            args = (getattr(self.model, 'args', None)
                    or getattr(inner, 'args', None)
                    or getattr(getattr(inner, 'model', None), 'args', None))
            model_type = getattr(args, 'model_type', '') if args else ''
            if not model_type and args and hasattr(args, 'text_config'):
                model_type = args.text_config.get('model_type', '')

            # Check if model has MTP layers configured
            has_mtp = args and getattr(args, 'mtp_num_hidden_layers', 0) > 0

            if has_mtp or 'qwen3_5' in model_type:
                # Map model_id to original HF repo (strip mlx-community prefix + quant suffix)
                repo = self._model_id_to_hf_repo(self.model_id) if self.model_id else ''
                if repo:
                    logger.info(f"Auto-detected MTP repo: {repo} (model_type={model_type})")
                    return repo
                logger.info(
                    f"No MTP repo derivable for model_id={self.model_id!r} "
                    f"(model_type={model_type}); falling back to vanilla decoding."
                )
        except Exception as e:
            logger.warning(f"MTP detection failed: {e}")
        return ''

    @staticmethod
    def _model_id_to_hf_repo(model_id: str) -> str:
        """Map an MLX model ID to the original HF repo containing MTP weights.

        e.g. 'mlx-community/Qwen3.5-397B-A17B-4bit' → 'Qwen/Qwen3.5-397B-A17B'
        """
        # Strip common MLX community prefixes
        name = model_id
        if name.startswith('mlx-community/'):
            name = name[len('mlx-community/'):]

        # Strip quantization suffixes
        for suffix in ['-4bit', '-8bit', '-bf16', '-fp16', '-MLX-4bit', '-MLX-8bit', '-MLX']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        # Map to original Qwen repo
        if name.startswith('Qwen3'):
            return f"Qwen/{name}"

        return ''

    def _extract_mtp_from_local(self, model_dir, cache_dir, cache_key) -> str | None:
        """Check local model directory for MTP weights and extract if found."""
        import json
        from pathlib import Path

        model_dir = Path(model_dir)
        idx_path = model_dir / "model.safetensors.index.json"
        if not idx_path.exists():
            return None

        with open(idx_path) as f:
            idx = json.load(f)

        mtp_keys = [k for k in idx["weight_map"] if k.startswith("mtp.")]
        if not mtp_keys:
            return None

        mtp_shards = sorted({idx["weight_map"][k] for k in mtp_keys})
        logger.info(f"Found {len(mtp_keys)} MTP weights in local model ({len(mtp_shards)} shards)")

        # Extract MTP tensors from local shards
        from safetensors.torch import load_file, save_file
        mtp_tensors = {}
        for shard_name in mtp_shards:
            shard_path = model_dir / shard_name
            if not shard_path.exists():
                return None  # shard missing, can't extract locally
            tensors = load_file(str(shard_path))
            for k, v in tensors.items():
                if k.startswith("mtp."):
                    mtp_tensors[k] = v

        if not mtp_tensors:
            return None

        cached_path = cache_dir / f"mtp_{cache_key}.safetensors"
        save_file(mtp_tensors, str(cached_path))
        logger.info(f"Extracted {len(mtp_tensors)} MTP tensors from local model → {cached_path}")
        return str(cached_path)

    @staticmethod
    def _auto_quantize_mtp(bf16_path, q4_path) -> str | None:
        """Auto-quantize bf16 MTP weights to 4-bit. Returns q4 path or None on failure."""
        try:
            import mlx.core as mx
            import mlx.nn as nn

            logger.info(f"Auto-quantizing MTP weights → {q4_path}")
            weights = mx.load(str(bf16_path))
            q_weights = {}
            for k, v in weights.items():
                if v.ndim == 2 and min(v.shape) >= 64:
                    lin = nn.Linear(v.shape[1], v.shape[0], bias=False)
                    lin.weight = v
                    ql = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
                    q_weights[k] = ql.weight
                    q_weights[k.replace('.weight', '.scales')] = ql.scales
                    q_weights[k.replace('.weight', '.biases')] = ql.biases
                    mx.eval(ql.weight, ql.scales, ql.biases)
                    del lin, ql
                else:
                    q_weights[k] = v
            mx.save_safetensors(str(q4_path), q_weights)
            logger.info(f"Auto-quantized MTP: {len(q_weights)} tensors → {q4_path}")
            return str(q4_path)
        except Exception as e:
            logger.warning(f"Auto-quantize MTP failed: {e}")
            return None

    def _extract_mtp_from_hf(self, repo_id: str) -> str:
        """Download MTP tensors from HF repo and cache as a single safetensors file.

        Uses the weight index to only download shards containing MTP weights,
        avoiding a full model download for large models (e.g. 397B = 220GB).
        """
        import hashlib
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file, save_file

        cache_dir = Path.home() / ".cache" / "exo" / "mtp_weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(repo_id.encode()).hexdigest()[:12]
        cached_path = cache_dir / f"mtp_{cache_key}.safetensors"

        logger.info(f"Downloading MTP weights from {repo_id}...")

        # Use weight index to find only the shards containing MTP weights
        try:
            idx_path = hf_hub_download(repo_id, "model.safetensors.index.json")
            with open(idx_path) as f:
                idx = json.load(f)
            mtp_shards = {
                shard for key, shard in idx["weight_map"].items()
                if key.startswith("model.mtp.") or key.startswith("mtp.")
            }
            logger.info(f"MTP weights span {len(mtp_shards)} of {len(set(idx['weight_map'].values()))} shards")
        except Exception:
            mtp_shards = None

        mtp_tensors = {}

        if mtp_shards:
            # Download only the shards containing MTP weights
            for shard_name in sorted(mtp_shards):
                shard_path = hf_hub_download(repo_id, shard_name)
                tensors = load_file(shard_path)
                for k, v in tensors.items():
                    if k.startswith("model.mtp."):
                        mtp_tensors[k.replace("model.mtp.", "")] = v
                    elif k.startswith("mtp."):
                        mtp_tensors[k] = v
        else:
            # Fallback: download all safetensors (small models)
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json"])
            model_path = Path(model_dir)
            for sf_file in sorted(model_path.glob("*.safetensors")):
                tensors = load_file(str(sf_file))
                for k, v in tensors.items():
                    if k.startswith("model.mtp."):
                        mtp_tensors[k.replace("model.mtp.", "")] = v

        if not mtp_tensors:
            raise ValueError(f"No MTP tensors found in {repo_id}")

        save_file(mtp_tensors, str(cached_path))
        logger.info(f"Cached {len(mtp_tensors)} MTP tensors to {cached_path}")
        return str(cached_path)

    def warmup_speculative(self, model, tokenizer) -> None:
        """Warm up the speculative decoding path (MTP draft + verify kernels)."""
        if not hasattr(self._mlx_gen, 'mtp'):
            return

        from mlx_lm.models import cache as cache_mod

        from exo.worker.engines.mlx.speculative.mtp_module import (
            draft_tokens,
            speculative_forward,
        )

        logger.info("Warming up speculative decoding kernels...")
        mtp = self._mlx_gen.mtp
        gamma = self._mlx_gen.gamma

        warmup_prompt = tokenizer.encode("Warm up speculative decoding.")
        cache = cache_mod.make_prompt_cache(model)
        mtp.reset_cache()

        pre_norm, logits = speculative_forward(model, mx.array([warmup_prompt]), cache)
        mx.eval(pre_norm, logits)
        next_token = mx.argmax(logits[0, -1], axis=-1).item()

        if pre_norm.shape[1] > 1:
            _ = mtp.predict(pre_norm[:, :-1, :], mx.array([warmup_prompt[1:]]))
            mx.eval(_)

        last_pn = pre_norm[:, -1:, :]
        next_arr = mx.array([[next_token]])
        for _ in range(3):
            draft_ids, _ = draft_tokens(mtp, last_pn, next_arr, gamma, 0.0)
            draft_concat = mx.concatenate([d.reshape(1, 1) for d in draft_ids], axis=1)
            verify_input = mx.concatenate([next_arr, draft_concat], axis=1)
            vpn, vl = speculative_forward(model, verify_input, cache, speculative=True)
            all_next = mx.argmax(vl[0], axis=-1)
            mx.eval(vpn, all_next)
            next_arr = all_next[0].reshape(1, 1)
            last_pn = vpn[:, 0:1, :]
            for i, c in enumerate(cache):
                if hasattr(c, 'base'):
                    cache[i] = c.base

        logger.info("Speculative warmup complete")

    @property
    def has_work(self) -> bool:
        # New mlx-lm split BatchGenerator into _prompt_batch + _generation_batch
        # with _unprocessed_sequences. Keep fallbacks to the old names so this
        # module still works against older mlx-lm checkouts if someone pins
        # back.
        unprocessed = getattr(
            self._mlx_gen, "_unprocessed_sequences", None
        )
        if unprocessed is None:
            unprocessed = getattr(self._mlx_gen, "unprocessed_prompts", None)
        has_unprocessed = bool(unprocessed) if unprocessed is not None else False

        gen_batch = getattr(self._mlx_gen, "_generation_batch", None)
        if gen_batch is not None:
            has_generation = len(gen_batch) > 0
        else:
            has_generation = getattr(self._mlx_gen, "active_batch", None) is not None

        return (
            bool(self._active_tasks)
            or has_unprocessed
            or has_generation
            or self._pp_spec_gen is not None
        )

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> int:
        from exo.worker.engines.mlx.trace import T

        with T("submit.encode_prompt"):
            all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
            all_prompt_tokens = fix_unmatched_think_end_tokens(
                all_prompt_tokens, self.tokenizer
            )

        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []

        if self.vision_processor is not None:
            try:
                with T("submit.vision"):
                    vision = prepare_vision(
                        images=task_params.images,
                        chat_template_messages=task_params.chat_template_messages,
                        vision_processor=self.vision_processor,
                        tokenizer=self.tokenizer,
                        model=self.model,
                        model_id=task_params.model,
                        task_params=task_params,
                    )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )

        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
            media_regions = vision.media_regions

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        is_exact_hit = False
        prompt_tokens = all_prompt_tokens

        with T("submit.kv_prefix_cache_lookup"):
            if self.kv_prefix_cache is not None and not is_bench:
                cache, remaining_tokens, matched_index, is_exact_hit = (
                    self.kv_prefix_cache.get_kv_cache(
                        self.model, all_prompt_tokens, media_regions=media_regions
                    )
                )
                prefix_hit_length = len(all_prompt_tokens) - len(remaining_tokens)
                if prefix_hit_length > 0:
                    logger.info(
                        f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                        f"cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
                    )
                    prompt_tokens = remaining_tokens
                else:
                    cache = make_kv_cache(self.model, max_kv_size=self.max_kv_tokens)
            else:
                cache = make_kv_cache(self.model, max_kv_size=self.max_kv_tokens)

        seed = task_params.seed if task_params.seed is not None else 42
        mx.random.seed(seed)

        _card = card_sampling_values(task_params.model, task_params.enable_thinking)
        _resolved = resolve_sampling(
            request_temperature=task_params.temperature,
            request_top_p=task_params.top_p,
            request_top_k=task_params.top_k,
            request_min_p=task_params.min_p,
            request_presence_penalty=task_params.presence_penalty,
            request_repetition_penalty=task_params.repetition_penalty,
            request_frequency_penalty=task_params.frequency_penalty,
            instance_temperature=self.default_temperature,
            instance_top_p=self.default_top_p,
            instance_top_k=self.default_top_k,
            instance_min_p=self.default_min_p,
            instance_presence_penalty=self.default_presence_penalty,
            instance_repetition_penalty=self.default_repetition_penalty,
            instance_frequency_penalty=self.default_frequency_penalty,
            card_temperature=_card.temperature if _card else None,
            card_top_p=_card.top_p if _card else None,
            card_top_k=_card.top_k if _card else None,
            card_min_p=_card.min_p if _card else None,
            card_presence_penalty=_card.presence_penalty if _card else None,
            card_repetition_penalty=_card.repetition_penalty if _card else None,
            card_frequency_penalty=_card.frequency_penalty if _card else None,
        )
        with T("submit.make_sampler"):
            sampler = make_sampler(
                temp=_resolved["temp"],
                top_p=_resolved["top_p"],
                top_k=_resolved["top_k"],
                min_p=_resolved["min_p"],
            )

        vision_ctx = (
            patch_embed_tokens(
                self.model, vision.embeddings, prefix_hit_length, len(prompt_tokens) - 1
            )
            if vision is not None
            else contextlib.nullcontext()
        )
        uncached_count = len(prompt_tokens)
        use_remote = (
            uncached_count > REMOTE_PREFILL_MIN_TOKENS
            and task_params.prefill_endpoint is not None
        )

        _prefill_tps: float = 0.0
        _prefill_tokens: int = 0
        cache_snapshots: list[CacheSnapshot] = []
        remote_prefilled = False
        with vision_ctx, T("submit.prefill"):
            if use_remote and task_params.prefill_endpoint is not None:
                try:
                    _prefill_tps, _prefill_tokens, cache_snapshots = remote_prefill(
                        prompt_tokens[:-1],
                        cache,
                        on_prefill_progress,
                        endpoint=task_params.prefill_endpoint,
                        request_id=str(uuid.uuid4()),
                        model_id=str(task_params.model),
                        start_pos=prefix_hit_length,
                    )
                    remote_prefilled = True
                except Exception:
                    logger.opt(exception=True).warning(
                        "Remote prefill failed, falling back to local prefill"
                    )

            if not remote_prefilled:
                # Keep prefill_step_size kwarg for our DSv4-Flash tuning
                # (DSV4_PREFILL_STEP_SIZE=256 per memory).
                _prefill_tps, _prefill_tokens, cache_snapshots = prefill(
                    self.model,
                    self.tokenizer,
                    sampler,
                    prompt_tokens[:-1],
                    cache,
                    self.group,
                    on_prefill_progress,
                    distributed_prompt_progress_callback,
                    prefill_step_size=self.prefill_step_size,
                )

        # We need to clamp rotating kv caches to max size so that mlx lm's _merge_caches behaves
        with T("submit.clamp_rotating_caches"):
            for c in cache:
                if (
                    isinstance(c, RotatingKVCache)
                    and c.keys is not None
                    and c.values is not None
                    and c.keys.shape[2] > c.max_size
                ):
                    trim_size = c.keys.shape[2] - c.max_size
                    c.keys = c._trim(trim_size, c.keys)
                    c.values = c._trim(trim_size, c.values)
                    c._idx = c.max_size

        with T("submit.save_prefix_cache"):
            if not is_bench:
                min_prefix_hit_length = max(
                    1000, system_prompt_token_count(task_params, self.tokenizer)
                )
                self._save_prefix_cache(
                    all_prompt_tokens,
                    list(cache),
                    cache_snapshots,
                    prefix_hit_length,
                    matched_index,
                    min_prefix_hit_length,
                    media_regions,
                )

        last_tokens = prompt_tokens[-2:]

        with T("submit.make_logits_processors"):
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
                    repetition_context_size=task_params.repetition_context_size or 20,
                    presence_penalty=_resolved["presence_penalty"],
                    presence_context_size=task_params.presence_context_size or 20,
                    frequency_penalty=_resolved["frequency_penalty"],
                )
            )
            if is_bench:
                eos_ids = eos_ids_from_tokenizer(self.tokenizer)
                logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        max_tokens = task_params.max_output_tokens or MAX_TOKENS

        if self._pp_spec_active:
            with T("submit.pp_spec_setup"):
                return self._submit_pp_spec(
                    task_params, all_prompt_tokens, prefix_hit_length, matched_index,
                    is_exact_hit, cache_snapshots, cache, last_tokens, sampler,
                    logits_processors, max_tokens, on_generation_token, _prefill_tps,
            )

        uids = self._mlx_gen.insert(
            prompts=[last_tokens.tolist()],
            max_tokens=[max_tokens],
            caches=[list(cache)],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        assert len(uids) == 1

        uid = uids[0]

        # MTP prefill: build MTP cache from prompt hidden states
        if hasattr(self._mlx_gen, 'mtp'):
            prompt_pre_norm = self._mlx_gen._captured.get('prompt_pre_norm')
            if prompt_pre_norm is not None:
                mx.eval(prompt_pre_norm)
                self._mlx_gen.mtp.reset_cache()
                S_pre = prompt_pre_norm.shape[1]
                if S_pre > 1:
                    toks_list = all_prompt_tokens.tolist() if hasattr(all_prompt_tokens, 'tolist') else list(all_prompt_tokens)
                    mtp_tokens = toks_list[1:S_pre]
                    _ = self._mlx_gen.mtp.predict(
                        prompt_pre_norm[:, :-1, :],
                        mx.array([mtp_tokens])
                    )
                    mx.eval(_)
                    logger.info(f"MTP cache prefilled ({S_pre} positions)")

        # Set per-request temperature for speculative. EXO_SPECULATIVE_TEMP
        # overrides everything; otherwise fall through the same resolution
        # chain (request → instance → cluster → hardcoded) as the sampler.
        if hasattr(self._mlx_gen, '_request_temp'):
            env_temp = os.environ.get("EXO_SPECULATIVE_TEMP")
            if env_temp is not None:
                self._mlx_gen._request_temp[uid] = float(env_temp)
            else:
                self._mlx_gen._request_temp[uid] = resolve_sampling(
                    request_temperature=task_params.temperature,
                    instance_temperature=self.default_temperature,
                )["temp"]

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            is_exact_hit=is_exact_hit,
            cache_snapshots=cache_snapshots or None,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=_prefill_tps,
            # New mlx-lm BatchGenerator removed `_stats` cumulative counters
            # (generation_time lived on a stats context manager instead). Wall
            # clock is an acceptable proxy for tok/s here since the active task
            # is almost always generating or waiting on agree_on_tasks.
            generation_time_at_start=_mlx_gen_elapsed_seconds(self._mlx_gen),
            media_regions=media_regions,
        )

        return uid

    def submit_batched(  # noqa: C901
        self,
        tasks: list[tuple[
            TextGenerationTaskParams,
            str,
            Callable[[int, int], None] | None,
            Callable[[], None] | None,
            Callable[[], None] | None,
        ]],
    ) -> list[int]:
        """Submit multiple tasks with a SINGLE batched prefill pass.

        Each tuple is ``(task_params, prompt_str, on_prefill_progress,
        distributed_prompt_progress_callback, on_generation_token)``. Returns
        the list of uids in the same order as ``tasks``.

        The serial ``submit()`` path was the proven c=2 long-context bottleneck:
        stream 0 monopolized the runner main loop for the full prefill duration
        while stream 1 sat in the queue, so c=2 100K MTP=0 collapsed to ~7.7
        tok/s/stream. This batched path runs ``prefill_batched`` once across
        all tasks so they prefill TOGETHER at shape (B, L_chunk).

        Falls back to per-task ``submit()`` for any task that can't be batched
        (vision, remote-prefill endpoint, MTP-active, PP-spec). When only one
        task is in the batch list, also falls back to ``submit()`` to avoid
        the unnecessary merge/extract overhead.
        """
        from exo.worker.engines.mlx.trace import T

        # Single-task fast path / wedge guard: degenerates to existing submit()
        # so c=1 deployments don't pay any new code-path cost.
        if len(tasks) <= 1:
            return [self.submit(*t) for t in tasks]

        # PP-spec or MTP path: batched prefill not yet wired through these.
        # Fall back to per-task submit() to keep them working.
        if self._pp_spec_active or hasattr(self._mlx_gen, "mtp"):
            logger.info(
                "submit_batched: PP-spec or MTP active — falling back to per-task submit()"
            )
            return [self.submit(*t) for t in tasks]

        # Heterogeneity guard: any task that needs the per-task path (vision,
        # remote prefill, prefix-cache hit) goes through ``submit()`` so the
        # batched fast path stays simple. Phase 4 ships behind opt-in env;
        # Phase 5+ extends coverage if needed.
        eligible: list[int] = []
        ineligible: list[int] = []
        for i, (task_params, _prompt, _opp, _dppc, _ogt) in enumerate(tasks):
            is_bench = task_params.bench
            has_remote = task_params.prefill_endpoint is not None
            has_vision = bool(task_params.images) or any(
                getattr(msg, "content", None) is not None
                and not isinstance(msg.content, str)
                for msg in task_params.input
            )
            if not is_bench or has_remote or has_vision:
                ineligible.append(i)
            else:
                eligible.append(i)

        if len(eligible) < 2:
            return [self.submit(*t) for t in tasks]

        # Run the eligible tasks through the batched path; the rest go
        # through the legacy per-task path. Preserve the input ordering
        # in the returned uids.
        uids: list[int | None] = [None] * len(tasks)

        with T("submit_batched.batched_path"):
            batch_uids = self._submit_batched_eligible(
                [tasks[i] for i in eligible]
            )
        for i, uid in zip(eligible, batch_uids, strict=True):
            uids[i] = uid

        for i in ineligible:
            uids[i] = self.submit(*tasks[i])

        # All slots filled by construction.
        return [u for u in uids if u is not None]

    def _submit_batched_eligible(  # noqa: C901
        self,
        tasks: list[tuple[
            TextGenerationTaskParams,
            str,
            Callable[[int, int], None] | None,
            Callable[[], None] | None,
            Callable[[], None] | None,
        ]],
    ) -> list[int]:
        """Run a SINGLE batched-prefill pass across `tasks` (all eligible).

        Sequence per task: encode prompt → resolve sampling → make logits
        processors → fresh cache. Then ONE ``prefill_batched`` call across all
        tasks. Then per-task ``_mlx_gen.insert`` + ``_active_tasks`` registration.

        Caller must ensure all tasks are eligible (bench=True, no vision, no
        remote prefill, no MTP/PP-spec). See ``submit_batched`` for the gate.
        """
        from exo.worker.engines.mlx.trace import T

        n = len(tasks)
        assert n >= 2, "_submit_batched_eligible requires >= 2 tasks"

        # ---- Per-task preprocessing (cheap, sequential is fine) ----
        all_prompt_tokens_list: list[mx.array] = []
        prompt_tokens_list: list[mx.array] = []
        cache_list: list[Any] = []
        sampler_list: list[Callable[[mx.array], mx.array]] = []
        logits_processors_list: list[list[Callable[[mx.array, mx.array], mx.array]]] = []
        max_tokens_list: list[int] = []
        prefix_hit_lengths: list[int] = [0] * n
        matched_indices: list[int | None] = [None] * n
        is_exact_hits: list[bool] = [False] * n

        for task_params, prompt, _opp, _dppc, _ogt in tasks:
            with T("submit_batched.encode_prompt"):
                tokens = encode_prompt(self.tokenizer, prompt)
                tokens = fix_unmatched_think_end_tokens(tokens, self.tokenizer)
            all_prompt_tokens_list.append(tokens)

            # Eligibility gate enforced bench=True so kv_prefix_cache is bypassed
            # (matches the existing submit() behavior at line 946: prefix cache
            # is skipped when ``is_bench``). Fresh cache per task.
            with T("submit_batched.make_kv_cache"):
                cache = make_kv_cache(self.model, max_kv_size=self.max_kv_tokens)
            cache_list.append(list(cache))
            prompt_tokens_list.append(tokens)

            seed = task_params.seed if task_params.seed is not None else 42
            mx.random.seed(seed)

            _card = card_sampling_values(task_params.model, task_params.enable_thinking)
            _resolved = resolve_sampling(
                request_temperature=task_params.temperature,
                request_top_p=task_params.top_p,
                request_top_k=task_params.top_k,
                request_min_p=task_params.min_p,
                request_presence_penalty=task_params.presence_penalty,
                request_repetition_penalty=task_params.repetition_penalty,
                request_frequency_penalty=task_params.frequency_penalty,
                instance_temperature=self.default_temperature,
                instance_top_p=self.default_top_p,
                instance_top_k=self.default_top_k,
                instance_min_p=self.default_min_p,
                instance_presence_penalty=self.default_presence_penalty,
                instance_repetition_penalty=self.default_repetition_penalty,
                instance_frequency_penalty=self.default_frequency_penalty,
                card_temperature=_card.temperature if _card else None,
                card_top_p=_card.top_p if _card else None,
                card_top_k=_card.top_k if _card else None,
                card_min_p=_card.min_p if _card else None,
                card_presence_penalty=_card.presence_penalty if _card else None,
                card_repetition_penalty=_card.repetition_penalty if _card else None,
                card_frequency_penalty=_card.frequency_penalty if _card else None,
            )
            sampler = make_sampler(
                temp=_resolved["temp"],
                top_p=_resolved["top_p"],
                top_k=_resolved["top_k"],
                min_p=_resolved["min_p"],
            )
            sampler_list.append(sampler)

            _rp = _resolved["repetition_penalty"]
            if _rp == 1.0:
                _rp = None
            lp_list: list[Callable[[mx.array, mx.array], mx.array]] = (
                make_logits_processors(
                    repetition_penalty=_rp,
                    repetition_context_size=task_params.repetition_context_size or 20,
                    presence_penalty=_resolved["presence_penalty"],
                    presence_context_size=task_params.presence_context_size or 20,
                    frequency_penalty=_resolved["frequency_penalty"],
                )
            )
            # bench mode: ban EOS so length is the only stop signal — same as
            # the existing submit() codepath at line 1097.
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            lp_list = [ban_token_ids(eos_ids)] + lp_list
            logits_processors_list.append(lp_list)

            max_tokens_list.append(task_params.max_output_tokens or MAX_TOKENS)

        # ---- Pick a single shared prefill progress callback ----
        # The serial path uses a per-task closure. For batched prefill we use
        # the FIRST task's callbacks for cluster-side progress reporting and
        # cancellation polling (the cancellation callback fires
        # agree_on_cancellations_fast, which is collective and shared across
        # tasks anyway). Per-task progress events still come back via the
        # decode-side step() path.
        on_prefill_progress = tasks[0][2]
        distributed_prompt_progress_callback = tasks[0][3]

        # Adapter: the serial prefill expects (prompt_tokens[:-1]) so the
        # cache lands at len-2. prefill_batched mirrors that contract.
        prompt_inputs = [p[:-1] for p in prompt_tokens_list]

        with T("submit_batched.prefill"):
            (
                per_stream_tps,
                _per_stream_tokens,
                per_stream_caches,
                _per_stream_snapshots,
            ) = prefill_batched(
                self.model,
                self.tokenizer,
                sampler_list[0],
                prompt_inputs,
                cache_list,
                self.group,
                on_prefill_progress,
                distributed_prompt_progress_callback,
                prefill_step_size=self.prefill_step_size,
            )

        # ---- Insert each task's last-2 tokens into mlx-lm BatchGenerator ----
        uids: list[int] = []
        for i, (task_params, _prompt, _opp, _dppc, on_generation_token) in enumerate(tasks):
            last_tokens = prompt_tokens_list[i][-2:]
            with T("submit_batched.insert"):
                inserted = self._mlx_gen.insert(
                    prompts=[last_tokens.tolist()],
                    max_tokens=[max_tokens_list[i]],
                    caches=[per_stream_caches[i]],
                    samplers=[sampler_list[i]],
                    logits_processors=[logits_processors_list[i]],
                )
            assert len(inserted) == 1
            uid = inserted[0]
            uids.append(uid)

            # Per-request temperature for the speculative decode path
            # (no-op when not running speculative). Mirrors submit().
            if hasattr(self._mlx_gen, "_request_temp"):
                env_temp = os.environ.get("EXO_SPECULATIVE_TEMP")
                if env_temp is not None:
                    self._mlx_gen._request_temp[uid] = float(env_temp)
                else:
                    self._mlx_gen._request_temp[uid] = resolve_sampling(
                        request_temperature=task_params.temperature,
                        instance_temperature=self.default_temperature,
                    )["temp"]

            self._active_tasks[uid] = _EngineTask(
                uid=uid,
                task_params=task_params,
                all_prompt_tokens=all_prompt_tokens_list[i],
                prefix_hit_length=prefix_hit_lengths[i],
                matched_index=matched_indices[i],
                is_exact_hit=is_exact_hits[i],
                cache_snapshots=None,  # SSM/snapshots not used at bench-only batched path
                detokenizer=self.tokenizer.detokenizer,
                on_generation_token=on_generation_token,
                generation_start_time=time.perf_counter(),
                prefill_tps=per_stream_tps[i],
                # Match submit()'s bookkeeping: capture wall after prefill so
                # gen_tps reflects decode-only timing, not prefill.
                generation_time_at_start=_mlx_gen_elapsed_seconds(self._mlx_gen),
                media_regions=[],
            )

        return uids

    def _submit_pp_spec(
        self,
        task_params: TextGenerationTaskParams,
        all_prompt_tokens: mx.array,
        prefix_hit_length: int,
        matched_index: int | None,
        is_exact_hit: bool,
        cache_snapshots: list[CacheSnapshot] | None,
        cache: list[Any],
        last_tokens: mx.array,
        sampler: Callable,
        logits_processors: list[Callable],
        max_tokens: int,
        on_generation_token: Callable[[], None] | None,
        prefill_tps: float,
    ) -> int:
        """Set up PP speculative decode for this task."""
        from exo.worker.engines.mlx.trace import T, request_trace

        from ..pp_speculation import (
            _install_spec_layers,
            get_pipeline_info,
            pp_speculative_decode_loop,
        )

        with T("pp_spec.get_pipeline_info"):
            pp_info = get_pipeline_info(self.model)
            assert pp_info is not None
            pp_rank, pp_world_size, pp_group = pp_info

        with T("pp_spec.install_spec_layers"):
            inner = getattr(self.model, "language_model", self.model)
            _install_spec_layers(inner)

        _pp_draft = getattr(self.model, "_pp_draft_model", None)
        _pp_draft_cache = getattr(self.model, "_pp_draft_cache", None)

        # Prefill draft cache with tail of prompt (rank 0 only)
        # The draft model uses a RotatingKVCache, so only recent tokens matter.
        if pp_rank == 0 and _pp_draft is not None:
            with T("pp_spec.draft_prefill"):
                _draft_kv_window = int(os.environ.get("EXO_DRAFT_KV_WINDOW", "4096"))
                _draft_tokens = all_prompt_tokens[-_draft_kv_window:]
                _draft_chunk = 512
                for i in range(0, len(_draft_tokens), _draft_chunk):
                    _pp_draft(_draft_tokens[i:i + _draft_chunk][None], cache=_pp_draft_cache)
                    mx.eval([c.state if hasattr(c, 'state') else c for c in _pp_draft_cache])
                mx.clear_cache()
                logger.info(f"Draft model prefilled with {len(_draft_tokens)} tokens (of {len(all_prompt_tokens)} total)")

        # First token via standard PP
        with T("pp_spec.first_token"):
            _first_gen = stream_generate(
                model=self.model, tokenizer=self.tokenizer, prompt=last_tokens,
                max_tokens=1, sampler=sampler, logits_processors=logits_processors,
                prompt_cache=cache, prefill_step_size=1,
                kv_group_size=KV_GROUP_SIZE, kv_bits=KV_BITS,
            )
            _first_out = next(_first_gen)
            first_y = mx.array([_first_out.token])
            mx.eval(first_y)

        logger.info(f"PP speculation active: rank={pp_rank}")

        # Get PP MTP predictor (lightweight, skip_mlp=True)
        _pp_mtp = getattr(self, '_pp_mtp', None)
        if _pp_mtp is not None:
            logger.info("PP speculation using MTP for drafting")

        # Create the spec decode generator
        request_trace.mark("pp_spec.decode_loop_start")
        self._pp_spec_gen = pp_speculative_decode_loop(
            model=self.model, draft_model=_pp_draft,
            prompt_cache=cache, draft_cache=_pp_draft_cache,
            sampler=sampler, logits_processors=logits_processors,
            first_y=first_y, first_logprobs=mx.zeros(1),
            max_tokens=max_tokens - 1,
            pp_rank=pp_rank, pp_world_size=pp_world_size,
            pp_group=pp_group,
            mtp_predictor=_pp_mtp,
        )

        self._uid_counter += 1
        uid = self._uid_counter
        self._pp_spec_uid = uid

        # Store first token to yield on first step()
        self._pp_first_token = _first_out.token

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            is_exact_hit=is_exact_hit,
            cache_snapshots=cache_snapshots or None,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=prefill_tps,
        )

        return uid

    def _step_pp_spec(self) -> list[GenerationBatch.Response]:
        """Get next token from PP speculative decode loop."""
        uid = self._pp_spec_uid
        assert uid is not None

        # Yield the first token if we haven't yet
        if hasattr(self, '_pp_first_token'):
            tok = self._pp_first_token
            del self._pp_first_token
            is_eos = tok in self._pp_spec_eos
            return [GenerationBatch.Response(
                uid=uid, token=tok, logprobs=mx.zeros(1),
                finish_reason="stop" if is_eos else None,
                current_state=None, match_sequence=None,
                prompt_cache=None, all_tokens=None,
            )]

        assert self._pp_spec_gen is not None
        try:
            tok_id, lp = next(self._pp_spec_gen)
            is_eos = tok_id in self._pp_spec_eos
            return [GenerationBatch.Response(
                uid=uid, token=tok_id, logprobs=lp,
                finish_reason="stop" if is_eos else None,
                current_state=None, match_sequence=None,
                prompt_cache=None, all_tokens=None,
            )]
        except StopIteration:
            # max_tokens reached
            self._pp_spec_gen = None
            self._pp_spec_uid = None
            return [GenerationBatch.Response(
                uid=uid, token=0, logprobs=mx.zeros(1),
                finish_reason="length",
                current_state=None, match_sequence=None,
                prompt_cache=None, all_tokens=None,
            )]

    def step(self) -> list[tuple[int, GenerationResponse]]:
        # `has_work` is per-rank state. In TP mode the outer gate at
        # batch_generator.step():564 is already collective, but mlx-lm's
        # step here may also be called from non-TP paths — keep this
        # gate collective when a sharding group is present so all ranks
        # branch identically. Memory: jaccl_phase_a_finding_2026_05_05.md.
        # Coord subgroup: this gate fires once per step at decode rate
        # alongside the model TP forward; without isolation the small
        # mx_any all_sum interleaves with model bf16 all_sums in the
        # encoder queue and the call_id counter races (2026-05-07
        # diagnosis via JACCL_TRACE_HASH). See get_coord_group.
        local_has_work = self.has_work
        _gpu_probe_local = bool(os.environ.get("MLX_GPU_TIME"))
        _t_mx_any_start = time.perf_counter() if _gpu_probe_local else 0.0
        if self.group is not None and self.group.size() > 1:
            if not mx_any(local_has_work, get_coord_group(self.group)):
                return []
        elif not local_has_work:
            return []
        if _gpu_probe_local:
            _t_mx_any_ns = int((time.perf_counter() - _t_mx_any_start) * 1e9)
            self._gpu_probe_mx_any_total = getattr(self, "_gpu_probe_mx_any_total", 0) + _t_mx_any_ns
            self._gpu_probe_mx_any_count = getattr(self, "_gpu_probe_mx_any_count", 0) + 1

        _trace = os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1")
        from exo.worker.engines.mlx.trace import request_trace

        # Use PP speculation decode if active
        if self._pp_spec_gen is not None:
            _step_tic = time.perf_counter()
            responses = self._step_pp_spec()
            _next_elapsed = time.perf_counter() - _step_tic
            request_trace.record("decode.step.mlx_next", _step_tic)
        else:
            self._mlx_gen._needs_topk = any(  # pyright: ignore[reportAttributeAccessIssue]
                t.task_params.logprobs for t in self._active_tasks.values()
            )
            _step_tic = time.perf_counter()
            # New mlx-lm BatchGenerator.next() returns (prompt_resps, gen_resps).
            # Use next_generated() to keep the old list-of-generations semantics.
            if hasattr(self._mlx_gen, "next_generated"):
                responses = self._mlx_gen.next_generated()
            else:
                responses = self._mlx_gen.next()  # legacy fork fallback
            _next_elapsed = time.perf_counter() - _step_tic
            request_trace.record("decode.step.mlx_next", _step_tic)

        results: list[tuple[int, GenerationResponse]] = []

        # per-token profiling accumulators
        _t_callback_total = 0.0
        _t_detok_total = 0.0
        _t_stop_total = 0.0
        _t_logprobs_total = 0.0
        _t_response_build_total = 0.0

        for response in responses:
            if response.uid not in self._active_tasks:
                logger.warning(
                    f"response uid {response.uid} was not found - should be active"
                )
                continue

            state = self._active_tasks[response.uid]

            # ── on_generation_token callback (agree_on_cancellations + agree_on_tasks every N tokens) ──
            _t0 = time.perf_counter()
            if state.on_generation_token is not None:
                state.on_generation_token()
            _t_callback_total += time.perf_counter() - _t0

            # ── detokenization ──
            _t0 = time.perf_counter()
            if response.finish_reason != "stop":
                state.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            state.generated_text_parts.append(text)
            state.potential_stop_sequence_text += text

            think_start = self.tokenizer.think_start
            think_end = self.tokenizer.think_end
            if think_start is not None and text == think_start:
                state.in_thinking = True
            elif think_end is not None and text == think_end:
                state.in_thinking = False
            if state.in_thinking:
                state.reasoning_tokens += 1
            _t_detok_total += time.perf_counter() - _t0

            # ── stop sequence check ──
            _t0 = time.perf_counter()
            finish_reason: FinishReason | None = cast(
                FinishReason | None, response.finish_reason
            )
            task_params = state.task_params
            stop_sequences = _stop_sequences(task_params)
            max_stop_len = max((len(s) for s in stop_sequences), default=0)

            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in state.potential_stop_sequence_text:
                        stop_index = state.potential_stop_sequence_text.find(stop_seq)
                        text_before_stop = state.potential_stop_sequence_text[
                            :stop_index
                        ]
                        chunk_start = len(state.potential_stop_sequence_text) - len(
                            text
                        )
                        text = text_before_stop[chunk_start:]
                        finish_reason = "stop"
                        break
            _t_stop_total += time.perf_counter() - _t0

            is_done = finish_reason is not None

            # ── logprobs extraction ──
            _t0 = time.perf_counter()
            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if task_params.logprobs and os.environ.get("EXO_DISABLE_LOGPROBS") != "1":
                with mx.stream(generation_stream):
                    logprob, top_logprobs = extract_top_logprobs(
                        logprobs=response.logprobs,
                        tokenizer=self.tokenizer,
                        top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                        selected_token=response.token,
                        precomputed_indices=getattr(response, "_topk_indices", None),
                        precomputed_values=getattr(response, "_topk_values", None),
                        precomputed_selected=getattr(
                            response, "_selected_logprob", None
                        ),
                    )
            _t_logprobs_total += time.perf_counter() - _t0

            # ── response building ──
            _t0 = time.perf_counter()
            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                if self._pp_spec_gen is not None or self._pp_spec_uid is not None:
                    gen_elapsed = time.perf_counter() - state.generation_start_time
                    generation_tps = (
                        state.completion_tokens / gen_elapsed
                        if gen_elapsed > 0
                        else 0.0
                    )
                    # Clean up spec state
                    self._pp_spec_gen = None
                    self._pp_spec_uid = None
                else:
                    gen_time_delta = (
                        _mlx_gen_elapsed_seconds(self._mlx_gen)
                        - state.generation_time_at_start
                    )
                    generation_tps = (
                        state.completion_tokens / gen_time_delta
                        if gen_time_delta > 0
                        else 0.0
                    )

                # MTP self-spec cumulative counters from the generator
                # if it's a DSv4MTPBatchGenerator. Master diffs successive
                # completions to drive Prometheus.
                mtp_cycles_cum = int(
                    getattr(self._mlx_gen, "_spec_cycles", 0) or 0
                )
                mtp_accepted_cum = int(
                    getattr(self._mlx_gen, "_spec_total_accepted", 0) or 0
                )

                # Classify the prefix-cache outcome for this request.
                # Field was previously never assigned anywhere — every
                # request defaulted to "none" so the metric always read
                # 0% hit rate even when the cache was working.
                if state.is_exact_hit:
                    prefix_cache_kind: Literal["none", "partial", "exact"] = "exact"
                elif state.prefix_hit_length > 0:
                    prefix_cache_kind = "partial"
                else:
                    prefix_cache_kind = "none"

                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    # mx.get_peak_memory() returns bytes. Convert to GiB
                    # (binary 1024^3) — matches the rest of the codebase
                    # (generate.py:290/403/845) and the actual physical
                    # RAM the cluster nodes have. Previously divided by
                    # 1e9 (decimal SI gigabytes), which made the metric
                    # read ~7% larger than actual usage and confused
                    # near-OOM diagnosis.
                    peak_memory_usage=Memory.from_gb(
                        mx.get_peak_memory() / 1024**3
                    ),
                    prefix_cache_hit=prefix_cache_kind,
                    mtp_cycles_cumulative=mtp_cycles_cum,
                    mtp_accepted_drafts_cumulative=mtp_accepted_cum,
                )
                total_prompt_tokens = len(state.all_prompt_tokens)
                usage = Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=state.completion_tokens,
                    total_tokens=total_prompt_tokens + state.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=state.prefix_hit_length
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=state.reasoning_tokens
                    ),
                )

            results.append(
                (
                    response.uid,
                    GenerationResponse(
                        text=text,
                        token=response.token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                        finish_reason=finish_reason,
                        stats=stats,
                        usage=usage,
                    ),
                )
            )
            _t_response_build_total += time.perf_counter() - _t0

            if is_done:
                del self._active_tasks[response.uid]
            elif (
                max_stop_len > 0
                and len(state.potential_stop_sequence_text) > max_stop_len
            ):
                state.potential_stop_sequence_text = state.potential_stop_sequence_text[
                    -max_stop_len:
                ]

        _step_end = time.perf_counter()
        _step_elapsed = _step_end - _step_tic
        _overhead = _step_elapsed - _next_elapsed
        _post_total = _t_callback_total + _t_detok_total + _t_stop_total + _t_logprobs_total + _t_response_build_total
        request_trace.record("decode.step.post_process", _step_tic + _next_elapsed, _step_end)

        # _next_count was added by the old fast_next patch; on vanilla
        # BatchGenerator (new mlx-lm) it doesn't exist — fall back to our own
        # cumulative step counter for logging cadence.
        _mlx_next_count = getattr(
            self._mlx_gen, "_next_count", None
        )
        if _mlx_next_count is None:
            _mlx_next_count = getattr(self, "_step_counter", 0) + 1
            self._step_counter = _mlx_next_count  # pyright: ignore[reportAttributeAccessIssue]
        if _mlx_next_count % 64 == 0 and responses:
            _gpu_probe_local = bool(os.environ.get("MLX_GPU_TIME"))
            _mxa_total = getattr(self, "_gpu_probe_mx_any_total", 0)
            _mxa_count = getattr(self, "_gpu_probe_mx_any_count", 0)
            _mxa_avg_ms = (_mxa_total / _mxa_count / 1e6) if (_gpu_probe_local and _mxa_count > 0) else 0.0
            logger.debug(
                f"step overhead: {_overhead * 1000:.2f}ms (next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms)"
            )
            if _gpu_probe_local:
                logger.info(
                    f"[STEP_TIMING] mx_any_avg_ms={_mxa_avg_ms:.3f} (n={_mxa_count}) "
                    f"next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms"
                )
        if _trace and _mlx_next_count % 64 == 0 and responses:
            logger.info(
                f"[PROF step] mlx_next={_next_elapsed * 1000:.2f}ms "
                f"callback={_t_callback_total * 1000:.2f}ms "
                f"detok={_t_detok_total * 1000:.2f}ms "
                f"stop_check={_t_stop_total * 1000:.2f}ms "
                f"logprobs={_t_logprobs_total * 1000:.2f}ms "
                f"response_build={_t_response_build_total * 1000:.2f}ms "
                f"total={_step_elapsed * 1000:.2f}ms"
            )

        if _MEM_PROFILE_PATH and _mlx_next_count % _MEM_PROFILE_INTERVAL == 0:
            _total_tokens = sum(
                int(t.completion_tokens) for t in self._active_tasks.values()
            )
            _mem_profile_record(
                _MEM_PROFILE_PATH,
                step_count=int(_mlx_next_count),
                total_tokens=_total_tokens,
                extra={"phase": "decode"},
            )

        if _TRACEMALLOC_PATH and _mlx_next_count % _TRACEMALLOC_INTERVAL == 0:
            _total_tokens_t = sum(
                int(t.completion_tokens) for t in self._active_tasks.values()
            )
            _tracemalloc_dump(
                _TRACEMALLOC_PATH,
                step=int(_mlx_next_count),
                tokens=_total_tokens_t,
            )

        if (
            _MLX_CLEAR_CACHE_INTERVAL > 0
            and _mlx_next_count % _MLX_CLEAR_CACHE_INTERVAL == 0
        ):
            mx.clear_cache()

        if (
            _GC_COLLECT_INTERVAL > 0
            and _mlx_next_count % _GC_COLLECT_INTERVAL == 0
        ):
            gc.collect()

        if (
            _MALLOC_RELIEF_INTERVAL > 0
            and _malloc_zone_pressure_relief is not None
            and _mlx_next_count % _MALLOC_RELIEF_INTERVAL == 0
        ):
            try:
                released = _malloc_zone_pressure_relief(None, 0)
                if released > 0 and _mlx_next_count % (_MALLOC_RELIEF_INTERVAL * 10) == 0:
                    logger.info(
                        f"[mem] malloc pressure_relief released {released / (1024**2):.1f} MB"
                    )
            except Exception:
                pass

        return results

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._mlx_gen.close()
        mx.clear_cache()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
        min_prefix_hit_length: int = 1000,
        media_regions: list[MediaRegion] | None = None,
    ) -> None:
        if self.kv_prefix_cache is None:
            return

        try:
            hit_ratio = (
                prefix_hit_length / len(all_prompt_tokens)
                if len(all_prompt_tokens) > 0
                else 0.0
            )
            if matched_index is not None and (
                prefix_hit_length >= min_prefix_hit_length
                and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                self.kv_prefix_cache.update_kv_cache(
                    matched_index,
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                    media_regions=media_regions,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)

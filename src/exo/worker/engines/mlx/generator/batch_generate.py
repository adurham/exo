import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, cast

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
    ChunkedPrefillDrive,
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    make_reasoning_budget_limiter,
    patch_embed_tokens,
    prefill,
    prefill_batched,
    safe_think_token_id,
)
from exo.worker.engines.mlx.generator.remote_prefill import remote_prefill
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    set_needs_topk,
    take_ready_topk,
)
from exo.worker.engines.mlx.sampling import card_sampling_values, resolve_sampling
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
    fix_unmatched_think_end_tokens,
    get_coord_group,
    mx_any,
    pipeline_agree_prefix_hit_length,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from exo.worker.engines.mlx.pp_batched_decode_glue import (
        PrefillGrant,
        Rank0BatchedDecodeGlue,
        Rank1BatchedDecodeGlue,
    )

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
_MLX_CLEAR_CACHE_INTERVAL = int(os.environ.get("EXO_MLX_CLEAR_CACHE_INTERVAL", "0"))

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

# ── Degeneration (repetition-loop) detection ──
# Catches decode collapse: the model emitting a short token cycle forever
# (e.g. "the user is on 1. the user is on 1." or BOS spam). Pure observability
# — logs ONCE per request when a loop crosses threshold, never alters output.
# On by default (cheap: a small deque scan per token). Disable with
# EXO_LOOP_DETECT=0. Tunables: window = how many recent token ids to inspect;
# min_repeats = how many back-to-back cycle repetitions trigger the warning.
#
# max_period/window WIDENED 2026-07-26: confirmed live (hard_eval.py,
# math_largest_prime_factor / math_binom_mod, all-MTP-off decode) that this
# kill-switch was structurally blind to a real, severe degeneration class --
# a sentence-level arithmetic-verification loop with an exact 14-TOKEN
# period (e.g. " So N = 1,234,567,891,011." tokenizes to 14 ids on DSv4's
# tokenizer), repeating 100+ times back-to-back, burning the entire
# max_tokens budget. The old max_period=8 cap made this mathematically
# impossible to detect -- not a tuning-too-conservative issue, a hard
# ceiling below the real failure's period. 24 covers this with real margin
# (confirmed period was 14) while the false-positive risk stays low: 6
# EXACT back-to-back repeats of a run(=1..24)-token cycle is already strong
# evidence of degeneration regardless of period length (legitimate
# intentionally-repetitive content -- "write X 5 times", boilerplate rows --
# essentially never repeats byte-for-byte AND stays perfectly periodic for
# 6+ contiguous cycles; real instances differ by at least one token per
# repeat). Window widened to comfortably exceed max_period*min_repeats
# (24*6=144) with slack for detection alignment.
_LOOP_DETECT_ENABLED = os.environ.get("EXO_LOOP_DETECT", "1") != "0"
_LOOP_DETECT_WINDOW = int(os.environ.get("EXO_LOOP_DETECT_WINDOW", "160"))
_LOOP_DETECT_MAX_PERIOD = int(os.environ.get("EXO_LOOP_DETECT_MAX_PERIOD", "24"))
_LOOP_DETECT_MIN_REPEATS = int(os.environ.get("EXO_LOOP_DETECT_MIN_REPEATS", "6"))
# What to DO when a repetition loop is detected:
#   "error" (default) — fail the turn cleanly with finish_reason="error"
#     (-> ErrorChunk -> 500 -> hermes classifies it retryable and retries).
#     This is the RIGHT default because by the time a loop is confirmed the
#     output is already degenerate: the tokens leading INTO the cycle are
#     garbage too (observed 2026-06-16: DSv4 regurgitated session_search result
#     JSON into its reasoning, then looped on the `}"]` tail — the surfaced
#     "answer" was a 2-char `"]`). Surfacing that remnant (what "stop" does)
#     hands the user broken output; failing cleanly lets the turn be retried.
#     The degenerate partial text is REPLACED with a diagnostic message so the
#     ErrorChunk carries a useful reason and the garbage never reaches display.
#   "stop" — force finish_reason="stop": terminate the runaway but SURFACE the
#     partial output already produced. Use only when you'd rather keep a
#     possibly-coherent prefix than retry. Note: this leaks the pre-collapse
#     wander (see the 2026-06-16 case), so it is no longer the default.
#   "warn" — legacy behavior: log once, never alter output (no termination).
# Either terminating mode is a hard guarantee an infinite loop needs — sampling
# penalties only lower the PROBABILITY of looping, they can't guarantee it ends.
# A loop of period<=8 repeated >=6x is unambiguous degeneration, never
# legitimate content, so terminating it is safe.
# Set EXO_LOOP_DETECT_MIN_REPEATS higher if you want a longer leash before the
# stop fires (default 6 cycles = caught fast, ~well before it wastes minutes).
_LOOP_DETECT_ACTION = os.environ.get("EXO_LOOP_DETECT_ACTION", "error")
# Diagnostic text that REPLACES the degenerate partial output when the action
# is "error". Kept short and specific so it is useful in logs / retries.
_DEGENERATION_ERROR_TEXT = (
    "Generation terminated: repetition-loop degeneration detected "
    "(the model collapsed into a repeating token cycle). Failing the turn "
    "cleanly so it can be retried."
)

# Reasoning-budget cap: force a clean end-of-thinking transition once
# reasoning has run for this FRACTION of the request's max_output_tokens,
# if it hasn't closed on its own. Confirmed 2026-07-26 (hard_eval.py against
# the exo cluster, tasks math_digit_sum / math_largest_prime_factor /
# math_binom_mod): DeepSeek-V4-Flash can reach a correct answer inside
# reasoning, then fall into a self-doubt loop ("But earlier I got X? Actually
# I got X? Let's recheck...") that re-derives the same result verbatim (with
# wording drift each cycle) indefinitely, consuming the ENTIRE max_tokens
# budget on reasoning_content and leaving `content` empty (client falls back
# to a bare BOS token). Confirmed NOT a decode-path artifact -- identical at
# temp=0 greedy AND temp=1.0/top_p=0.95 real sampling; identical with
# MTP/DSpark speculative decoding fully on or off. NOT caught by the
# degeneration kill-switch above: that detector is period<=8 exact-repeat,
# this loop's period is 60-400+ tokens with paraphrase drift each cycle.
# Deliberately an INVARIANT (reasoning must not consume the whole budget and
# leave the answer empty), not a pattern-match on the self-doubt loop shape
# itself -- see make_reasoning_budget_limiter's docstring for the full
# rationale. 0.75 chosen from the one confirmed failure reaching its correct
# answer at ~55% of budget before looping -- gives real headroom for
# legitimately long reasoning while still guaranteeing some budget remains
# for the answer. Set EXO_REASONING_BUDGET_FRACTION<=0 to disable (no-op,
# same convention as repetition_penalty==1.0 collapsing to None).
#
# FIXED 2026-07-31 (was flagged as a known gap here, now closed): originally
# budget_tokens = max_output_tokens * this fraction with no ceiling. When a
# client sends no explicit max_output_tokens (e.g. hermes-agent's exo
# provider does not by default), it fell through to MAX_TOKENS=32168
# (constants.py) -> budget ~24,126 tokens. At this cluster's realistic
# decode throughput (~15-30 tok/s, sometimes slower under an unrelated known
# PP throughput-variance issue), exhausting that could take 15-30+ minutes
# -- this mechanism WAS engaged and WOULD have eventually intervened on the
# 2026-07-31 20+-minute incident, just far too slowly to be useful
# protection. Fixed with TWO independent additions (both applied at the
# call site in submit(), not here -- this fraction is unchanged):
# (1) _REASONING_BUDGET_MAX_TOKENS: an absolute ceiling so budget_tokens
#     never exceeds this regardless of how large max_output_tokens resolves
#     to (budget_tokens = min(fraction * max_tokens, this_cap)). Default
#     16384 -- comfortably above the ~13-14K tokens the one confirmed
#     self-doubt-loop failure needed to reach its correct answer before
#     looping (~55% of the OLD ~24,126 budget), so legitimate long reasoning
#     isn't clipped, while still meaningfully improving the worst case.
# (2) _REASONING_BUDGET_MAX_SECONDS: an INDEPENDENT wall-clock backstop
#     (make_reasoning_budget_limiter's new max_seconds parameter) -- a token
#     count alone cannot bound wall-clock time across a throughput range
#     that varies 2x, plus the separate known PP degradation case. Default
#     360s (6 min). Deliberately a BACKSTOP, not the primary trigger --
#     making the whole mechanism purely time-based would make reasoning
#     depth a function of transient cluster load (same prompt cut off at a
#     different token count depending on how busy the cluster is), hurting
#     eval reproducibility. The token cap still governs the common case;
#     time only fires if something is unusually slow.
# See make_reasoning_budget_limiter's docstring in generate.py for the full
# mechanism-level rationale of both triggers.
_REASONING_BUDGET_FRACTION = float(
    os.environ.get("EXO_REASONING_BUDGET_FRACTION", "0.75")
)
_REASONING_BUDGET_MAX_TOKENS = int(
    os.environ.get("EXO_REASONING_BUDGET_MAX_TOKENS", "16384")
)
_REASONING_BUDGET_MAX_SECONDS = float(
    os.environ.get("EXO_REASONING_BUDGET_MAX_SECONDS", "360")
)

# ── Long-period (multi-sentence) degeneration detection ──
# _detect_token_loop above catches TIGHT exact-token cycles (period<=24,
# e.g. "the user is on 1. the user is on 1."). It is structurally blind to a
# DIFFERENT collapse shape observed 2026-07-31: DSv4's reasoning_content
# stuck for 20+ minutes cycling the same handful of full sentences
# WORD-FOR-WORD IDENTICAL each repeat (not paraphrased -- confirmed from the
# raw stream), just with a period far longer than 24 tokens (each cycle was
# ~150-300+ tokens, several full sentences). Raising _LOOP_DETECT_MAX_PERIOD
# to cover that directly would blow up the existing detector's cost (it scans
# EVERY candidate period 1..max_period on EVERY token -- O(max_period*window)
# per token; going from 24 to ~300 is a ~12x per-token cost increase forever,
# on the hot path, to catch a comparatively rare failure mode).
#
# This is DELIBERATELY a second, separate mechanism from
# make_reasoning_budget_limiter above, not a replacement -- that limiter
# targets PARAPHRASE-DRIFT self-doubt loops (period 60-400+ tokens, wording
# differs each cycle) by salvaging an early answer once a real one was
# already found; this detector targets EXACT-repeat long-period loops (the
# repeated content is byte-identical, a stronger "genuinely stuck, zero
# forward progress" signal) by terminating fast regardless of what
# max_output_tokens/budget_tokens happens to resolve to for a given request.
# They cover non-overlapping failure signatures and can both be active on
# the same request; whichever fires first wins.
#
# Instead: chunk the token stream into fixed-size non-overlapping blocks
# (_LOOP_DETECT_LONG_BLOCK tokens each), hash each block once as it completes,
# and run the SAME period-scan algorithm as _detect_token_loop but over the
# short deque of BLOCK HASHES rather than raw token ids. This only runs once
# per block (not once per token) and the periodicity scan itself is over a
# short list of small ints, so total added cost is negligible -- strictly
# cheaper per-token than the existing tight-loop detector -- while covering
# periods up to _LOOP_DETECT_LONG_MAX_PERIOD * _LOOP_DETECT_LONG_BLOCK tokens
# (default 12*24 = 288, comfortably above the observed ~150-300 token cycles).
#
# Deliberately does NOT use fuzzy/similarity matching (e.g. SimHash) even
# though the observed loop involved slightly different reasoning en route
# to each repeated sentence -- the REPEATED sentences themselves were
# byte-identical, so an exact block-hash match already catches this class
# without the false-positive risk of a similarity threshold. On a hash match,
# the underlying raw token blocks are compared exactly (not just hashes)
# before terminating, eliminating hash-collision risk entirely -- this
# verification only runs on the rare detection path, so it's free.
#
# Explicitly NOT a repetition_penalty / sampling-side fix: repetition_penalty
# was tried project-wide and reverted (2026-07-24, commit 4b4309d56) after a
# controlled test found it caused a 23.3% SILENT tool-call corruption rate
# (penalizing tokens a verbatim-copy task needs to reproduce exactly, e.g.
# building a file path from context). This detector never touches sampling;
# it is a pure kill-switch, same guarantee model as _detect_token_loop --
# terminating (action="error"/"stop", same EXO_LOOP_DETECT_ACTION knob), not
# salvaging like make_reasoning_budget_limiter, because an exact repeated
# cycle is a stronger "stuck" signal than paraphrase drift: the tokens
# leading into a confirmed EXACT cycle are already degenerate too (same
# reasoning as the tight-loop detector's own action="error" default above),
# so forcing an early answer out of that reasoning risks a confident-but-
# wrong response rather than a clean retryable failure.
_LOOP_DETECT_LONG_ENABLED = os.environ.get("EXO_LOOP_DETECT_LONG", "1") != "0"
_LOOP_DETECT_LONG_BLOCK = int(os.environ.get("EXO_LOOP_DETECT_LONG_BLOCK", "24"))
_LOOP_DETECT_LONG_MAX_PERIOD = int(
    os.environ.get("EXO_LOOP_DETECT_LONG_MAX_PERIOD", "12")
)
# Lower than the tight-loop detector's min_repeats=6: each "period" here is
# already ~150-300+ raw tokens (several full sentences), so waiting for 6
# full repeats before terminating would burn proportionally far more
# wall-clock than the exact-token case before intervening -- exactly the
# thing that made the real incident run 20+ minutes with no other signal.
_LOOP_DETECT_LONG_MIN_REPEATS = int(
    os.environ.get("EXO_LOOP_DETECT_LONG_MIN_REPEATS", "3")
)
# Block-hash deque only needs to hold enough blocks to see
# max_period * min_repeats back; a little slack for scan alignment.
_LOOP_DETECT_LONG_WINDOW_BLOCKS = (
    _LOOP_DETECT_LONG_MAX_PERIOD * _LOOP_DETECT_LONG_MIN_REPEATS + 2
)

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
        _malloc_zone_pressure_relief = _libsystem_malloc.malloc_zone_pressure_relief
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


def _detect_token_loop(
    token_ids: list[int],
    max_period: int = _LOOP_DETECT_MAX_PERIOD,
    min_repeats: int = _LOOP_DETECT_MIN_REPEATS,
) -> tuple[int, int] | None:
    """Detect a repeating token cycle at the TAIL of ``token_ids``.

    Returns ``(period, repeats)`` for the shortest period (1..max_period)
    whose cycle repeats back-to-back at least ``min_repeats`` times ending
    at the most recent token, else ``None``. Cheap: O(max_period * window),
    no allocation beyond slices. Used only for logging — never gates output.
    """
    n = len(token_ids)
    if n < min_repeats:
        return None
    for period in range(1, max_period + 1):
        if n < period * min_repeats:
            break
        cycle = token_ids[n - period :]
        repeats = 1
        # walk backwards in blocks of `period`, counting identical cycles
        pos = n - period
        while pos - period >= 0 and token_ids[pos - period : pos] == cycle:
            repeats += 1
            pos -= period
        if repeats >= min_repeats:
            return period, repeats
    return None


def _detect_long_period_loop(
    block_hashes: list[int],
    max_period: int = _LOOP_DETECT_LONG_MAX_PERIOD,
    min_repeats: int = _LOOP_DETECT_LONG_MIN_REPEATS,
) -> tuple[int, int] | None:
    """Detect a repeating BLOCK-hash cycle at the tail of ``block_hashes``.

    Same shortest-period-first back-to-back-repeat algorithm as
    ``_detect_token_loop``, but operating over a short list of block hashes
    (one entry per ``_LOOP_DETECT_LONG_BLOCK`` raw tokens) instead of raw
    token ids. This is what lets it reach periods of hundreds of raw tokens
    (period * block_size) without the O(max_period * window) per-TOKEN cost
    the tight-loop detector would pay for the same reach. See the
    "Long-period (multi-sentence) degeneration detection" comment block
    above for the full rationale (why block hashing, why exact match not
    fuzzy, why this doesn't touch sampling).

    Returns ``(period_in_blocks, repeats)`` on a match, else ``None``.
    Caller is responsible for re-verifying the raw token blocks match
    exactly before terminating (hash collisions are astronomically
    unlikely with a 64-bit hash over ~24-token blocks, but the raw
    comparison is free on this rare detection-only path, so there is no
    reason not to make the guarantee airtight).
    """
    n = len(block_hashes)
    if n < min_repeats:
        return None
    for period in range(1, max_period + 1):
        if n < period * min_repeats:
            break
        cycle = block_hashes[n - period :]
        repeats = 1
        pos = n - period
        while pos - period >= 0 and block_hashes[pos - period : pos] == cycle:
            repeats += 1
            pos -= period
        if repeats >= min_repeats:
            return period, repeats
    return None


class PPSpecAlreadyActiveError(RuntimeError):
    """Raised when a second PP speculative-decode task is submitted while
    one is already in flight on this rank's ExoBatchGenerator.

    See the long comment on ExoBatchGenerator._pp_spec_gen_by_uid (2026-07-31)
    for the full rationale: PP's shared SpecPipelineFirstLayer/
    SpecPipelineLastLayer mode-flag state represents the ONE physical
    rank0<->rank1 wire link, so genuinely concurrent PP-spec decoding isn't
    safe with today's architecture -- this replaces a prior silent-clobber
    data-corruption bug (second submit() overwrote the first request's
    generator reference, orphaning its task forever) with an explicit,
    immediately-visible rejection instead. A plain RuntimeError subclass
    (not BaseException, unlike PrefillCancelled in generate.py) so it's
    caught by the runner's existing `except Exception` handling around
    generation-task dispatch and surfaces as a clean task failure the
    caller can retry, not an uncaught crash.
    """


@dataclass
class _DeferredPrefill:
    """A single-request's prefill work, deferred until
    ``Rank0BatchedDecodeGlue``/``Rank1BatchedDecodeGlue.tick()`` grants
    it via a real ``PrefillGrant`` (2026-08-06 fix for the N=2
    admission-race deadlock -- see ``pp_batched_decode_glue.py``'s
    module docstring for the full rationale).

    ``run_prefill`` is a zero-argument closure built inside
    ``submit()``, capturing exactly the same local state (vision
    context, remote-vs-local prefill choice, cache-snapshot
    collection, RotatingKVCache clamping, prefix-cache write-back)
    that used to run INLINE, immediately, before this fix. Deferring
    it to a closure -- rather than re-deriving these steps a second
    time inside ``_step_batched_decode`` -- keeps this fix a pure
    reordering of WHEN that existing, already-correct code runs, not
    a reimplementation of it.

    ``drive`` (2026-08-07, Phase 2 live-wiring): when
    ``_run_deferred_prefill_for_grant`` decides this request's real
    prefill IS chunk-interruptible (``prefill_interruptible_start``
    returned non-``None``), the resulting ``ChunkedPrefillDrive`` is
    stored HERE rather than in a second, separately-keyed dict --
    per a ``consult`` review flagging the two-dict-per-request shape
    as a real desync risk (cancellation/failure paths only need to
    reason about ONE bookkeeping structure per request_id/uid). Kept
    on THIS object (not popped into ``_active_tasks``) specifically
    because ``_DeferredPrefill`` already IS this request's single
    source of truth for "prefill in progress, not yet admitted" --
    ``None`` here means either "prefill hasn't started" or "this
    request's prefill is NOT chunk-interruptible and ran/will run
    synchronously via ``run_prefill()`` instead," matching the
    existing ``run_prefill`` closure's own unconditional presence.
    """

    run_prefill: Callable[[], tuple[float, int, list[CacheSnapshot], "KVCacheType"]]
    try_start_chunked_prefill: Callable[[], "ChunkedPrefillDrive | None"]
    finalize_prefill: Callable[
        [float, int, list[CacheSnapshot]],
        tuple[float, int, list[CacheSnapshot], "KVCacheType"],
    ]
    task_params: TextGenerationTaskParams
    last_tokens: mx.array
    sampler: Callable[[mx.array], mx.array]
    max_tokens: int
    on_generation_token: Callable[[], None] | None
    cache_slot: int
    drive: "ChunkedPrefillDrive | None" = field(default=None, init=False)


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
    # ── degeneration (repetition-loop) detection ──
    # Rolling window of the most recent generated token ids, used to detect
    # decode collapse (the model emitting a short cycle forever). Pure
    # observability: never alters sampling/output. See _detect_token_loop.
    recent_token_ids: list[int] = field(default_factory=list)
    degeneration_warned: bool = False
    # ── long-period (multi-sentence) degeneration detection ──
    # Raw tokens accumulating toward the next _LOOP_DETECT_LONG_BLOCK-sized
    # block (cleared once the block completes and is hashed). Kept as raw
    # tokens (not just the running hash) so the terminating path can
    # re-verify a hash match against actual token equality before firing.
    long_loop_pending_tokens: list[int] = field(default_factory=list)
    # Completed blocks: parallel lists of (hash, raw tokens) so a hash-match
    # candidate can be verified exactly. Bounded to
    # _LOOP_DETECT_LONG_WINDOW_BLOCKS by the caller (deque semantics via
    # manual trim, kept as plain lists to match recent_token_ids' style).
    long_loop_block_hashes: list[int] = field(default_factory=list)
    long_loop_block_tokens: list[list[int]] = field(default_factory=list)
    long_loop_degeneration_warned: bool = False
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
    # PER-TASK KEYED, not a singular attribute (fixed 2026-07-31 -- see
    # PPSpecAlreadyActiveError below and the entry guard in
    # _submit_pp_spec). Historically these were bare `Generator | None` /
    # `int | None` instance attributes, clobbered wholesale by a second
    # submit() before the first request's generator was exhausted: the
    # first request's task orphaned forever (its generator reference lost,
    # nothing ever resumes it, the runner wedges waiting on a response
    # that never comes) and the second request silently inherited
    # whatever mid-flight state the overwrite left behind. Keying by uid
    # makes that contract explicit and, combined with the entry guard,
    # converts the failure mode from silent data corruption into a loud,
    # immediately-visible rejection.
    #
    # IMPORTANT -- this is a SAFETY fix, not a concurrency feature. PP's
    # SpecPipelineFirstLayer/SpecPipelineLastLayer (pp_speculation.py) are
    # singular objects installed ONCE onto the model's real, persistent
    # layer list; their _pp_recv/_pp_send/_speculative flags represent
    # the state of the ONE physical rank0<->rank1 wire link this step,
    # reconfigured via _configure_layers() at the start of every request
    # and reset in the generator's `finally:`. Two PP-spec generators
    # genuinely interleaved via step() would each reconfigure that SAME
    # shared link with no atomicity between "configure" and "use it" --
    # exactly the stale-mode-flag bug a 2026-07-20 fix already patched
    # for the disconnect/exception-path case, except as the NORMAL path
    # instead of a rare edge case. True concurrent PP-spec decoding needs
    # either per-request wire-protocol multiplexing or a real scheduler
    # over the shared layer objects -- separate, larger architectural
    # work, not this fix. EXO_MAX_CONCURRENT_REQUESTS stays capped at 1
    # for Pipeline mode; this dict never holds more than one entry in
    # today's architecture, by design (the entry guard enforces it).
    _pp_spec_gen_by_uid: dict[int, Generator[tuple[int, mx.array], None, None]] = field(
        default_factory=dict, init=False
    )
    _pp_spec_eos: set[int] = field(init=False, default_factory=set)
    _uid_counter: int = field(init=False, default=0)
    # Monotonic per-process counter, incremented once per submit() call,
    # used to tag the pipeline_agree_prefix_hit_length() exchange so a
    # tag mismatch on receipt is detectable (protocol-invariant check,
    # not a real request id — never sent anywhere else).
    _prefix_hit_agree_tag: int = field(init=False, default=0)

    # Phase 1 batched-decode (design doc, Section 9) -- opt-in,
    # EXO_PP_BATCHED_DECODE=1 + real BatchedMetaFramedPipelineFirstLayer/
    # LastLayer already installed at model-load time (see utils_mlx.py's
    # EXO_PP_BATCHED_DECODE branch). LATCHED ONCE at __post_init__ (see
    # this class's own docstring convention for _pp_spec_active) -- a
    # mid-session flip of this flag is not supported, matching every
    # other engine-selection flag in this class. Default OFF: when
    # False, this entire subsystem is never constructed and submit()/
    # step() take their existing, unmodified code paths -- see the
    # single `if self._batched_decode_active:` branch point in each.
    _batched_decode_active: bool = field(init=False, default=False)
    # Exactly one of these two is non-None when _batched_decode_active
    # is True (rank 0 gets the admitting/sampling glue, rank 1 gets the
    # mirroring-only glue) -- never both, matching
    # BatchedDecodeSession/RankOneMirrorSession's own rank-exclusive
    # design.
    _batched_decode_rank0_glue: "Rank0BatchedDecodeGlue | None" = field(
        init=False, default=None
    )
    _batched_decode_rank1_glue: "Rank1BatchedDecodeGlue | None" = field(
        init=False, default=None
    )
    _batched_decode_eos: set[int] = field(init=False, default_factory=set)
    # Rank-0-only: prefill work deferred until Rank0BatchedDecodeGlue's
    # tick() grants it (2026-08-06 fix, see _DeferredPrefill's own
    # docstring). Keyed by uid so _step_batched_decode can look up the
    # exact deferred work a PrefillGrant's request_id refers to -- a
    # PrefillGrant only carries request_id/cache_slot/n_prompt_tokens
    # (the wire-transmissible subset), not the full closure, so this
    # dict is where the rest of a submit() call's captured state lives
    # between enqueue_prefill() and the grant actually arriving.
    _deferred_prefill_by_uid: dict[int, "_DeferredPrefill"] = field(
        init=False, default_factory=dict
    )

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
                elif inner is not None and type(inner).__name__ == "DeepseekV4Model":
                    # DeepSeek-V4 models NEVER use the Qwen-style separate MTP
                    # weights file mechanism (below) -- they either use the
                    # checkpoint-bundled classic MTP head (is_dsv4_with_mtp
                    # branch above, gated by EXO_DSV4_MTP=1) or DSpark, which
                    # is dispatched entirely separately at generate-time via
                    # pp_dspark_decode_loop (see the PP-spec request path
                    # further down this file) -- this __post_init__ runs
                    # unconditionally at generator construction, before PP
                    # vs non-PP is even decided, so it never sees DSpark.
                    # Skipping straight to the non-speculative fallback here
                    # avoids a doomed _resolve_mtp_weights() HF-repo lookup
                    # that can never succeed for DSv4 and previously logged a
                    # misleading "could not find MTP weights" warning on
                    # every DSv4 launch (2026-08-03, DeepSeek-V4-Flash-0731
                    # deploy -- looked like a real problem, was actually a
                    # guaranteed-to-fail probe for an irrelevant code path).
                    self._mlx_gen = MlxBatchGenerator(
                        model=self.model,
                        stop_tokens=stop_tokens_seq,
                        prefill_step_size=prefill_step_size,
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
                    # Per-model gamma for the Qwen3.5-style MTP path. Qwen3.6's
                    # dedicated head is trained with block_size=3, so it
                    # sustains a deeper draft chain than DSv4's depth-1 head.
                    # Default γ=3 here, set ONLY by EXO_QWEN_SPECULATIVE_GAMMA
                    # — independent of the DSv4 EXO_SPECULATIVE_GAMMA so the two
                    # models can run different chain depths concurrently.
                    gamma = int(os.environ.get("EXO_QWEN_SPECULATIVE_GAMMA", "3"))

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

        # Enable PP speculation if draft model is configured and we're in PP
        # mode. EXO_SPECULATIVE is the intended master kill-switch for ALL
        # speculative decoding (Tensor-mode MTP AND PP-mode speculation of
        # every kind -- DSpark, native MTP, chained MTP, classic draft-model)
        # -- previously it only gated whether DSv4's OWN native MTP head got
        # loaded ON TOP of an already-active PP-spec session, while
        # EXO_PP_DRAFT_MODEL's mere non-emptiness (default: a real path)
        # silently kept _pp_spec_active True regardless. That mismatch bit a
        # 2026-07-26 investigation: a test run set EXO_SPECULATIVE=0 assuming
        # it fully disabled speculation, but PP-mode classic draft-model
        # speculation (and therefore, transitively, DSpark -- see the
        # _pp_spec_active gate at this method's submit() call site) was
        # still active the whole time. Fixed by making EXO_SPECULATIVE a
        # real master switch here, which also makes the old EXO_PP_SPEC_
        # DISABLE flag (removed from start_cluster.sh the same day) genuinely
        # redundant rather than "removed but still needed via a workaround".
        draft_path = os.environ.get("EXO_PP_DRAFT_MODEL", "")
        if (
            use_speculative
            and draft_path
            and self.group is not None
            and self.group.size() > 1
        ):
            try:
                from ..pp_speculation import get_pipeline_info

                if get_pipeline_info(self.model) is not None:
                    self._pp_spec_active = True
                    logger.info("PP speculation enabled in BatchGenerator")
                    # Load MTP for PP speculation. BUG FIX (2026-07-17): this
                    # used to unconditionally construct mtp_module.MTPPredictor
                    # (the Qwen3.5-style loader that resolves a SEPARATE MTP
                    # weights file via _resolve_mtp_weights()) via a bare name
                    # reference with no import in this scope. For any model
                    # whose native checkpoint already carries an MTP head
                    # (DSv4-Flash's model.model.mtp[0] -- the is_dsv4_with_mtp
                    # branch above in this same method), the deferred
                    # `from ...mtp_module import MTPPredictor` never executes
                    # (it's ONLY imported in the sibling Qwen3.5-style else
                    # branch), so this raised UnboundLocalError on every PP+MTP
                    # attempt for DSv4-Flash specifically -- caught by the
                    # inner except below, logged, and silently downgraded to
                    # draft-model-only PP speculation (no MTP) every time.
                    # Mirror the same is_dsv4_with_mtp detection block A uses:
                    # DSv4's MTP head needs no separate weights file at all
                    # (DSv4MTPPredictor wraps the already-loaded, already-
                    # sharded model.model.mtp[0] directly), so skip
                    # _resolve_mtp_weights()'s whole cache/HF-download
                    # resolution pipeline for that case.
                    if use_speculative:
                        _inner_pp = getattr(self.model, "model", None)
                        _is_dsv4_with_mtp = (
                            _inner_pp is not None
                            and type(_inner_pp).__name__ == "DeepseekV4Model"
                            and hasattr(_inner_pp, "mtp")
                            and len(_inner_pp.mtp) > 0
                        )
                        if _is_dsv4_with_mtp:
                            try:
                                from exo.worker.engines.mlx.speculative.dsv4_mtp import (
                                    DSv4MTPPredictor,
                                )

                                self._pp_mtp = DSv4MTPPredictor(self.model, mtp_idx=0)
                                logger.info(
                                    "PP MTP loaded (DSv4 native head, mtp_idx=0)"
                                )
                            except Exception as e:
                                import traceback

                                logger.warning(
                                    f"PP MTP load failed: {e}\n{traceback.format_exc()}"
                                )
                                self._pp_mtp = None
                        else:
                            mtp_weights = self._resolve_mtp_weights()
                            if mtp_weights:
                                try:
                                    from exo.worker.engines.mlx.speculative.mtp_module import (
                                        MTPPredictor,
                                    )

                                    self._pp_mtp = MTPPredictor(
                                        self.model,
                                        mtp_weights,
                                        quantize=False,
                                    )
                                    logger.info(f"PP MTP loaded from {mtp_weights}")
                                except Exception as e:
                                    import traceback

                                    logger.warning(
                                        f"PP MTP load failed: {e}\n{traceback.format_exc()}"
                                    )
                                    self._pp_mtp = None
                            else:
                                self._pp_mtp = None
                    else:
                        self._pp_mtp = None
            except Exception:
                pass

        # Phase 1 batched-decode (design doc Section 9): construct the
        # rank-appropriate glue ONLY when explicitly opted in
        # (EXO_PP_BATCHED_DECODE=1) AND the batched pipeline layers are
        # actually installed on this model (get_batched_pipeline_info
        # returns non-None only when utils_mlx.py's EXO_PP_BATCHED_
        # DECODE branch actually ran at load time -- see that module's
        # own gate). Mutually exclusive with PP-spec by construction:
        # get_batched_pipeline_info only ever finds
        # BatchedMetaFramedPipelineLastLayer, a DIFFERENT layer class
        # than PP-spec's SpecPipelineLastLayer
        # (get_pipeline_info/pp_speculation.py), so a model can never
        # satisfy both checks at once. Latched HERE, once, for the
        # lifetime of this generator instance -- mirrors _pp_spec_active's
        # own single-construction-time-decision convention; no
        # mid-session flag flips are supported anywhere in this class.
        if os.environ.get("EXO_PP_BATCHED_DECODE", "0") == "1":
            try:
                from exo.worker.engines.mlx.pp_batched_decode_layers import (
                    get_batched_pipeline_info,
                )

                pipeline_info = get_batched_pipeline_info(self.model)
                if pipeline_info is not None:
                    rank, _world_size, group = pipeline_info
                    self._batched_decode_eos = set(
                        eos_ids_from_tokenizer(self.tokenizer)
                    )
                    # 2026-08-06, chunk-drive redesign: BOTH ranks need
                    # to know the PEER's real local layer count for the
                    # chunk-completion arithmetic (see
                    # Rank0BatchedDecodeGlue.peer_prefill_layer_count's
                    # own field comment for the full incident this
                    # closes -- the two ranks' real layer counts are
                    # CONFIRMED uneven on the real 43-layer DSv4-Flash
                    # topology). Run this ONE-TIME handshake here, at
                    # model-load time, regardless of which rank this
                    # is -- both sides of the exchange must run or the
                    # other blocks forever on its own recv.
                    from exo.worker.engines.mlx.auto_parallel import (
                        get_inner_model,
                        get_layers,
                    )
                    from exo.worker.engines.mlx.pp_batched_decode_glue import (
                        exchange_prefill_peer_layer_count,
                    )

                    local_layer_count = len(get_layers(get_inner_model(self.model)))
                    peer_layer_count = exchange_prefill_peer_layer_count(
                        local_layer_count=local_layer_count,
                        dst_rank=1 if rank == 0 else 0,
                        group=group,
                    )
                    if rank == 0:
                        from exo.worker.engines.mlx.pp_batched_decode_adapter import (
                            BatchedDecodeResponseAdapter,
                        )
                        from exo.worker.engines.mlx.pp_batched_decode_glue import (
                            Rank0BatchedDecodeGlue,
                        )
                        from exo.worker.engines.mlx.pp_batched_decode_runtime import (
                            BatchedDecodeSession,
                        )

                        session = BatchedDecodeSession.new(max_concurrency=2)
                        adapter = BatchedDecodeResponseAdapter(
                            session=session, eos_ids=frozenset(self._batched_decode_eos)
                        )
                        self._batched_decode_rank0_glue = Rank0BatchedDecodeGlue(
                            session=session,
                            adapter=adapter,
                            dst_rank=1,
                            group=group,
                            peer_prefill_layer_count=peer_layer_count,
                        )
                        logger.info(
                            "Phase 1 batched-decode ENABLED (rank 0, "
                            "admission+decode glue constructed)"
                        )
                    else:
                        from exo.worker.engines.mlx.pp_batched_decode_glue import (
                            Rank1BatchedDecodeGlue,
                        )
                        from exo.worker.engines.mlx.pp_batched_decode_runtime import (
                            RankOneMirrorSession,
                        )

                        mirror_session = RankOneMirrorSession.new(max_concurrency=2)
                        self._batched_decode_rank1_glue = Rank1BatchedDecodeGlue(
                            session=mirror_session, src_rank=0, group=group
                        )
                        logger.info(
                            "Phase 1 batched-decode ENABLED (rank 1, "
                            "mirror glue constructed)"
                        )
                    self._batched_decode_active = True
                else:
                    logger.info(
                        "EXO_PP_BATCHED_DECODE=1 but "
                        "get_batched_pipeline_info found no installed "
                        "batched pipeline layers on this model -- "
                        "batched-decode path stays OFF for this "
                        "generator instance (falls through to the "
                        "existing submit()/step() paths unmodified)."
                    )
            except Exception:
                logger.opt(exception=True).warning(
                    "Phase 1 batched-decode construction failed -- "
                    "falling back to the existing submit()/step() "
                    "paths unmodified for this generator instance."
                )
                self._batched_decode_active = False
                self._batched_decode_rank0_glue = None
                self._batched_decode_rank1_glue = None

    def _submit_batched_decode(
        self,
        task_params: TextGenerationTaskParams,
        cache: KVCacheType,
        last_tokens: mx.array,
        sampler: Callable[[mx.array], mx.array],
        max_tokens: int,
        on_generation_token: Callable[[], None] | None,
        prefill_tps: float,
    ) -> int:
        """Admission path for the Phase 1 batched-decode session
        (design doc Section 9). Mirrors ``submit()``'s own uid
        allocation and ``_active_tasks`` bookkeeping exactly -- the
        SAME uid space, the SAME ``_EngineTask`` shape -- so every
        downstream consumer (``step()``'s response-processing loop,
        cancellation, stats) works completely unmodified regardless
        of which engine produced a given uid.

        RANK-DEPENDENT (found + fixed 2026-08-05 after the second
        real cluster run): rank 0 calls
        ``Rank0BatchedDecodeGlue.enqueue_admission`` -- per that
        module's own docstring and the `consult` review behind its
        design, this is PURE in-memory queueing with ZERO wire I/O,
        so this branch cannot hang or race the decode-step loop. The
        actual admission (real wire send) happens later, inside
        ``step()``'s ``tick()`` call -- the single-writer rule this
        whole subsystem is built around.

        Rank 1 calls ``Rank1BatchedDecodeGlue.stage_local_cache``
        instead -- rank 1's OWN local ``prefill()`` call (the
        unmodified, already-run call immediately before this method,
        same as every other request) already produced a real
        prefilled KV cache for this request; that cache is never
        sent over the wire (only rank 0's activations/metadata are),
        so it must be staged locally, keyed by ``request_id``, for
        ``Rank1BatchedDecodeGlue.tick()`` to pick up reactively when
        this request's admission arrives over the wire from rank 0
        (see that glue's own docstring for the full reactive-
        admission-detection design). Symmetric ``uid``/``cache_slot``
        derivation with rank 0's branch is REQUIRED here: both ranks
        process the identical, globally-ordered stream of eligible
        submissions (this fork's own event-sourcing architecture
        guarantees this -- see the design doc's "why no new
        broadcast-admission protocol is needed" analysis), so
        ``self._uid_counter``/``len(self._active_tasks)`` grow
        identically on both ranks call-for-call, making the SAME
        ``uid``/``cache_slot`` pair available on rank 1 at its own
        ``submit()`` time without any cross-rank message -- verified
        against the real wire protocol's own slot-assignment
        invariant (``pp_scheduler_protocol.py``'s ``SchedulerCore``:
        a slot can never move DRAINING->FREE->reassigned without an
        explicit eviction-ack round trip, so rank 0's decision and
        rank 1's independently-derived value cannot silently diverge
        onto different physical slots for the same request_id).
        """
        uid = self._uid_counter
        self._uid_counter += 1

        # cache_slot: Phase 1 scope is max_concurrency=2 (see
        # BatchedDecodeSession.new's own default) -- reuse
        # len(_active_tasks) at admission time as a simple 2-slot
        # allocator (0 or 1), matching the max_concurrency this
        # session/glue pair was constructed with above. Computed
        # BEFORE the rank-dependent branch below so both ranks derive
        # it identically from the same (symmetric) counter state.
        cache_slot = len(self._active_tasks) % 2

        if self._batched_decode_rank1_glue is not None:
            self._batched_decode_rank1_glue.stage_local_cache(
                request_id=uid, cache_slot=cache_slot, prefilled_cache=cache
            )
        else:
            assert self._batched_decode_rank0_glue is not None
            self._batched_decode_rank0_glue.enqueue_admission(
                request_id=uid,
                cache_slot=cache_slot,
                prefilled_cache=cache,
                initial_token=int(last_tokens[-1].item()),
                sampler=sampler,
                max_tokens=max_tokens,
            )

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=last_tokens,
            prefix_hit_length=0,
            matched_index=None,
            is_exact_hit=False,
            cache_snapshots=None,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=prefill_tps,
            generation_time_at_start=0.0,
            media_regions=[],
        )
        self._update_fence_arming()
        return uid

    def _submit_batched_decode_deferred(
        self,
        *,
        task_params: TextGenerationTaskParams,
        prompt: str,
        prompt_tokens: mx.array,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        sampler: Callable[[mx.array], mx.array],
        max_tokens: int,
        on_generation_token: Callable[[], None] | None,
        on_prefill_progress: Callable[[int, int], None] | None,
        distributed_prompt_progress_callback: Callable[[], None] | None,
        vision: VisionResult | None,
        media_regions: list[MediaRegion],
        prefix_hit_length: int,
        matched_index: int | None,
        is_bench: bool,
    ) -> int:
        """Admission path for the Phase 1 batched-decode session,
        2026-08-06 update (N=2 admission-race fix -- see
        ``pp_batched_decode_glue.py``'s module docstring for the full
        rationale). Supersedes ``_submit_batched_decode`` (kept above,
        unused, as a documented reference for the OLD pre-fix shape --
        a future cleanup pass may delete it once this path has proven
        itself on real hardware) for the ELIGIBLE-request case: instead
        of running prefill immediately and folding an already-prefilled
        cache into the batch, this method DEFERS the real prefill
        forward pass until ``Rank0BatchedDecodeGlue``/
        ``Rank1BatchedDecodeGlue.tick()`` grants it -- the fix for the
        real, hardware-confirmed deadlock where each rank's own
        independently-scheduled ``submit()`` call could issue prefill's
        wire traffic in the same window the peer rank was still
        mid-``tick()`` issuing decode's wire traffic.

        Builds a ``run_prefill`` closure that is a byte-for-byte
        relocation of what USED to run inline in ``submit()`` (vision
        context, remote-vs-local prefill choice, cache-snapshot
        collection, RotatingKVCache clamping, prefix-cache write-back)
        -- none of that logic is reimplemented here, only its EXECUTION
        TIME moves, from "immediately, inside submit()" to "later,
        inside ``_run_deferred_prefill_for_grant``, when a
        ``PrefillGrant`` says it's this rank's turn."

        uid/cache_slot derivation is IDENTICAL to
        ``_submit_batched_decode``'s own symmetric-counter scheme (see
        that method's docstring for the full "why no new broadcast-
        admission protocol is needed" rationale) -- unaffected by this
        fix, since both ranks still process the identical, globally-
        ordered stream of eligible submissions and still grow
        ``self._uid_counter``/``len(self._active_tasks)`` identically,
        call-for-call.
        """
        uid = self._uid_counter
        self._uid_counter += 1
        cache_slot = len(self._active_tasks) % 2

        last_tokens = prompt_tokens[-2:]

        def _finalize_prefill(
            _prefill_tps: float,
            _prefill_tokens: int,
            cache_snapshots: list[CacheSnapshot],
        ) -> tuple[float, int, list[CacheSnapshot], KVCacheType]:
            """Shared tail for BOTH the synchronous (``run_prefill``) and
            chunked-interruptible (``try_start_chunked_prefill`` +
            ``_advance_chunked_prefill_drive``) real prefill paths --
            2026-08-07, extracted verbatim (byte-for-byte, zero logic
            change) from this closure's own pre-existing tail so both
            paths apply the IDENTICAL RotatingKVCache clamp and
            prefix-cache write-back, never duplicated or allowed to
            drift apart.
            """
            from exo.worker.engines.mlx.trace import T

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
                        task_params.low_priority,
                        task_params.high_priority,
                    )

            return _prefill_tps, _prefill_tokens, cache_snapshots, cache

        def run_prefill() -> tuple[float, int, list[CacheSnapshot], KVCacheType]:
            from exo.worker.engines.mlx.trace import T

            vision_ctx = (
                patch_embed_tokens(
                    self.model,
                    vision.embeddings,
                    prefix_hit_length,
                    len(prompt_tokens) - 1,
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
                    # See _submit_batched_decode_deferred's own docstring
                    # and the identical block this was relocated from
                    # (submit()'s old inline prefill, pre-2026-08-06) for
                    # the full snapshot_offset/cross-request-contamination
                    # rationale -- unchanged here, only relocated.
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
                        snapshot_offset=prefix_hit_length,
                    )

            return _finalize_prefill(_prefill_tps, _prefill_tokens, cache_snapshots)

        def try_start_chunked_prefill() -> "ChunkedPrefillDrive | None":
            """2026-08-07, Phase 2 live-wiring: mirrors ``run_prefill``'s
            OWN vision/remote-prefill eligibility computation exactly --
            chunked interruption only ever applies to the LOCAL
            ``pipeline_parallel_prefill`` path (remote prefill has no
            layer-segment concept at all; a vision request's
            ``patch_embed_tokens`` context manager scoping isn't
            threaded through ``prefill_interruptible_start`` and isn't
            needed in practice since batched-decode eligibility already
            excludes ``has_images`` requests upstream -- checked again
            HERE, defensively, rather than trusting that upstream gate
            alone). Returns ``None`` for every ineligible case (never
            raises) -- ``_run_deferred_prefill_for_grant`` falls back to
            the unmodified, synchronous ``run_prefill()`` above whenever
            this returns ``None``, so this function's own eligibility
            mistakes fail SAFE (slower, synchronous path) rather than
            unsafe.
            """
            if vision is not None:
                return None
            uncached_count = len(prompt_tokens)
            use_remote = (
                uncached_count > REMOTE_PREFILL_MIN_TOKENS
                and task_params.prefill_endpoint is not None
            )
            if use_remote:
                return None
            from exo.worker.engines.mlx.generator.generate import (
                prefill_interruptible_start,
            )

            return prefill_interruptible_start(
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

        self._deferred_prefill_by_uid[uid] = _DeferredPrefill(
            run_prefill=run_prefill,
            try_start_chunked_prefill=try_start_chunked_prefill,
            finalize_prefill=_finalize_prefill,
            task_params=task_params,
            last_tokens=last_tokens,
            sampler=sampler,
            max_tokens=max_tokens,
            on_generation_token=on_generation_token,
            cache_slot=cache_slot,
        )

        if self._batched_decode_rank0_glue is not None:
            self._batched_decode_rank0_glue.enqueue_prefill(
                request_id=uid,
                cache_slot=cache_slot,
                n_prompt_tokens=len(prompt_tokens) - 1,
                single_request_fallback=False,
            )
        else:
            assert self._batched_decode_rank1_glue is not None
            # 2026-08-06 fix for the prefill forward-pass race (see
            # PrefillReadyMessage's own docstring in
            # pp_scheduler_protocol.py for the full incident this
            # closes, and this method's own module for the earlier
            # "parking" mechanism this REPLACES): rank 1 registers its
            # own local readiness the INSTANT this _DeferredPrefill
            # exists -- pure local bookkeeping, zero wire I/O. If
            # rank 0's PrefillMessage for this uid already arrived
            # (rank 0's tick() can race ahead of this rank's own
            # submit(), a real and expected timing skew between two
            # independently-scheduled per-rank event loops), rank 0
            # already got a NACK for that attempt and will retry on a
            # LATER tick() -- this mark_prefill_registered() call is
            # what makes that RETRY succeed; there is no "already
            # arrived, service it now" special case to handle here
            # anymore (the old _parked_prefill_grants mechanism this
            # replaced tried to service a grant synchronously the
            # instant registration happened, WITHOUT rank 0 having any
            # way to know whether that succeeded -- which is exactly
            # the race that produced the real
            # SchedulerWireProtocolError crash on N=2 hardware; see the
            # handoff doc's "2026-08-06 finding" section for the full
            # incident writeup).
            self._batched_decode_rank1_glue.mark_prefill_registered(uid)

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=last_tokens,
            prefix_hit_length=0,
            matched_index=None,
            is_exact_hit=False,
            cache_snapshots=None,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=0.0,
            generation_time_at_start=0.0,
            media_regions=[],
        )
        self._update_fence_arming()
        return uid

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
            inner = (
                getattr(self.model, "model", None) or self.model.language_model.model
            )
            args = (
                getattr(self.model, "args", None)
                or getattr(inner, "args", None)
                or getattr(getattr(inner, "model", None), "args", None)
            )
            model_type = getattr(args, "model_type", "") if args else ""
            if not model_type and args and hasattr(args, "text_config"):
                model_type = args.text_config.get("model_type", "")

            # Check if model has MTP layers configured
            has_mtp = args and getattr(args, "mtp_num_hidden_layers", 0) > 0

            if has_mtp or "qwen3_5" in model_type:
                # Prefer a dedicated, MLX-ready MTP drafter repo
                # (mlx-community/<base>-MTP-bf16) when one is published — its
                # norm weights are already +1-shifted for MLX and the proj
                # weights are bit-identical to the upstream full-model MTP, so
                # using it directly removes the fragile runtime +1 shift
                # heuristic. Falls back to the upstream Qwen/<base> full repo
                # (MTP tensors extracted from its shards) when no dedicated
                # drafter exists.
                dedicated = (
                    self._dedicated_mtp_repo(self.model_id) if self.model_id else ""
                )
                if dedicated:
                    logger.info(
                        f"Auto-detected dedicated MTP drafter: {dedicated} (model_type={model_type})"
                    )
                    return dedicated
                # Map model_id to original HF repo (strip mlx-community prefix + quant suffix)
                repo = self._model_id_to_hf_repo(self.model_id) if self.model_id else ""
                if repo:
                    logger.info(
                        f"Auto-detected MTP repo: {repo} (model_type={model_type})"
                    )
                    return repo
                logger.info(
                    f"No MTP repo derivable for model_id={self.model_id!r} "
                    f"(model_type={model_type}); falling back to vanilla decoding."
                )
        except Exception as e:
            logger.warning(f"MTP detection failed: {e}")
        return ""

    @staticmethod
    def _model_id_to_hf_repo(model_id: str) -> str:
        """Map an MLX model ID to the original HF repo containing MTP weights.

        e.g. 'mlx-community/Qwen3.5-397B-A17B-4bit' → 'Qwen/Qwen3.5-397B-A17B'
        """
        # Strip common MLX community prefixes
        name = model_id
        if name.startswith("mlx-community/"):
            name = name[len("mlx-community/") :]

        # Strip quantization suffixes
        for suffix in [
            "-4bit",
            "-8bit",
            "-bf16",
            "-fp16",
            "-MLX-4bit",
            "-MLX-8bit",
            "-MLX",
        ]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        # Map to original Qwen repo
        if name.startswith("Qwen3"):
            return f"Qwen/{name}"

        return ""

    @staticmethod
    def _dedicated_mtp_repo(model_id: str) -> str:
        """Return mlx-community/<base>-MTP-bf16 if it exists on HF, else ''.

        The dedicated drafter repos publish the MTP head standalone, already
        +1-norm-shifted for MLX. We prefer them over extracting from the
        upstream full-model repo. An EXO_MTP_NO_DEDICATED=1 escape hatch
        disables this preference. The HF existence check is cached per repo
        so we don't hit the network on every detection.
        """
        import os as _os

        if _os.environ.get("EXO_MTP_NO_DEDICATED", "") == "1":
            return ""
        # Strip mlx-community/ prefix and a quant suffix to get <base>.
        name = model_id
        if name.startswith("mlx-community/"):
            name = name[len("mlx-community/") :]
        for suffix in [
            "-4bit",
            "-8bit",
            "-6bit",
            "-5bit",
            "-bf16",
            "-fp16",
            "-MLX-4bit",
            "-MLX-8bit",
            "-MLX",
        ]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        candidate = f"mlx-community/{name}-MTP-bf16"
        # Cache existence checks on the class to avoid repeat network calls.
        cache = ExoBatchGenerator.__dict__.get("_dedicated_mtp_cache")
        if cache is None:
            cache = {}
            ExoBatchGenerator._dedicated_mtp_cache = cache  # type: ignore[attr-defined]
        if candidate in cache:
            return cache[candidate]
        exists = ""
        try:
            from huggingface_hub import HfApi

            if HfApi().repo_exists(candidate):
                exists = candidate
        except Exception as e:
            logger.debug(f"Dedicated MTP repo check for {candidate} failed: {e}")
            exists = ""
        cache[candidate] = exists
        return exists

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
        logger.info(
            f"Found {len(mtp_keys)} MTP weights in local model ({len(mtp_shards)} shards)"
        )

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
        logger.info(
            f"Extracted {len(mtp_tensors)} MTP tensors from local model → {cached_path}"
        )
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
                    q_weights[k.replace(".weight", ".scales")] = ql.scales
                    q_weights[k.replace(".weight", ".biases")] = ql.biases
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
        """Download MTP tensors from an HF repo and cache as one safetensors file.

        Handles two repo shapes:
          * FULL model repo (e.g. Qwen/Qwen3.6-35B-A3B): MTP tensors are a
            subset of the sharded weights, keyed ``model.mtp.*``. We use the
            weight index to download only the shards that carry them.
          * DEDICATED drafter repo (mlx-community/<base>-MTP-bf16): a single
            ``model.safetensors`` whose keys are BARE (``fc.weight``,
            ``layers.0.*``, ``norm.weight``, ``pre_fc_norm_*``) and whose norm
            weights are ALREADY +1-shifted for MLX.

        All kept tensors are normalized to the ``mtp.`` prefix the MTP loader
        expects. For dedicated (pre-shifted) heads we drop a ``.noshift``
        marker next to the cache so the loader skips its runtime +1 shift.
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

        # The dedicated drafter repos are the entire (small) model with bare
        # keys and already-shifted norms. Detect them by name/model_type so we
        # take the bare-key + no-shift path.
        is_dedicated = repo_id.endswith("-MTP-bf16") or repo_id.endswith("-MTP")
        if not is_dedicated:
            try:
                cfg_path = hf_hub_download(repo_id, "config.json")
                with open(cfg_path) as f:
                    _cfg = json.load(f)
                if str(_cfg.get("model_type", "")).endswith("_mtp"):
                    is_dedicated = True
            except Exception:
                pass

        def _norm_key(k: str) -> str | None:
            """Map a raw tensor key to the canonical ``mtp.*`` form, or None to drop."""
            if k.startswith("model.mtp."):
                return "mtp." + k[len("model.mtp.") :]
            if k.startswith("mtp."):
                return k
            if is_dedicated:
                # Bare drafter keys (fc., layers., norm., pre_fc_norm_*).
                return "mtp." + k
            return None

        mtp_tensors: dict = {}

        # Use the weight index (full repos) to fetch only MTP shards.
        mtp_shards = None
        if not is_dedicated:
            try:
                idx_path = hf_hub_download(repo_id, "model.safetensors.index.json")
                with open(idx_path) as f:
                    idx = json.load(f)
                mtp_shards = {
                    shard
                    for key, shard in idx["weight_map"].items()
                    if key.startswith("model.mtp.") or key.startswith("mtp.")
                }
                logger.info(
                    f"MTP weights span {len(mtp_shards)} of {len(set(idx['weight_map'].values()))} shards"
                )
            except Exception:
                mtp_shards = None

        if mtp_shards:
            for shard_name in sorted(mtp_shards):
                shard_path = hf_hub_download(repo_id, shard_name)
                for k, v in load_file(shard_path).items():
                    nk = _norm_key(k)
                    if nk is not None:
                        mtp_tensors[nk] = v
        else:
            # Dedicated drafter repo, or a small full repo with no index:
            # download every safetensors and keep the MTP/bare tensors.
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                repo_id, allow_patterns=["*.safetensors", "*.json"]
            )
            for sf_file in sorted(Path(model_dir).glob("*.safetensors")):
                for k, v in load_file(str(sf_file)).items():
                    nk = _norm_key(k)
                    if nk is not None:
                        mtp_tensors[nk] = v

        if not mtp_tensors:
            raise ValueError(f"No MTP tensors found in {repo_id}")

        save_file(mtp_tensors, str(cached_path))
        # Marker: dedicated heads ship pre-shifted norms — tell the loader to
        # skip its +1 shift so we don't double-shift.
        marker = cached_path.with_suffix(".noshift")
        if is_dedicated:
            marker.write_text(repo_id)
            logger.info(f"Dedicated MTP head — wrote no-shift marker {marker.name}")
        elif marker.exists():
            marker.unlink()
        logger.info(
            f"Cached {len(mtp_tensors)} MTP tensors to {cached_path} "
            f"(dedicated={is_dedicated})"
        )
        return str(cached_path)

    def warmup_speculative(self, model, tokenizer) -> None:
        """Warm up the speculative decoding path (MTP draft + verify kernels)."""
        if not hasattr(self._mlx_gen, "mtp"):
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
                if hasattr(c, "base"):
                    cache[i] = c.base

        logger.info("Speculative warmup complete")

    @property
    def has_work(self) -> bool:
        # New mlx-lm split BatchGenerator into _prompt_batch + _generation_batch
        # with _unprocessed_sequences. Keep fallbacks to the old names so this
        # module still works against older mlx-lm checkouts if someone pins
        # back.
        unprocessed = getattr(self._mlx_gen, "_unprocessed_sequences", None)
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
            or bool(self._pp_spec_gen_by_uid)
        )

    def _set_fence_async_engine(self, arm: bool) -> None:
        """Set the "engine" arming key of the DSv4 c=1 async decode fence.

        Disarming drains any deferred async graph (mx.synchronize) so a
        newly admitted request's forwards can't interleave with in-flight
        deferred collectives from the current stream (the 2026-07-02 c=2
        corruption). No-op for models without the side channel.
        """
        try:
            from mlx_lm.models.deepseek_v4 import _set_fence_async_ok
        except ImportError:
            return
        if not arm:
            _set_fence_async_ok(False, key="engine")
            mx.synchronize()
        else:
            _set_fence_async_ok(True, key="engine")

    def _update_fence_arming(self) -> None:
        """Arm the engine key iff the active-request count is within the
        async-fence limit (1 unless EXO_DSV4_FENCE_ASYNC_C2 raises it)."""
        limit = max(1, int(os.environ.get("EXO_DSV4_FENCE_ASYNC_C2", "0") or "0") or 1)
        self._set_fence_async_engine(
            1 <= len(self._active_tasks) <= limit and not self._pp_spec_active
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

        # Disarm the async fence before ANY forward for this request —
        # its prefill must not interleave with deferred graphs from an
        # already-decoding stream.
        self._set_fence_async_engine(False)

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

        # Seed + sampler are set up here (BEFORE the batched-decode routing
        # decision below) so both the batched-decode and serial paths can
        # share a single construction. Order: seed derivation from
        # task_params has no dependency on cache state, so hoisting it
        # above the KVPrefixCache lookup is safe and lets us avoid
        # duplicating the sampler-construction block across the two
        # dispatch paths.
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

        # 2026-08-06 cross-rank eligibility divergence fix (bug #6, see
        # docs/batched-decode-n2-admission-handoff-2026-08-05.md, section
        # "2026-08-06 fix: eliminate cross-rank eligibility divergence"):
        # decide batched-decode routing HERE, BEFORE any per-rank
        # KVPrefixCache lookup runs. The eligibility inputs used at this
        # point (has_images/has_tools/uses_speculative_decode/
        # sharding_is_pipeline/batched_decode_enabled) are ALL either
        # request-derived (identical on every rank via the broadcast/queue)
        # or static cluster-wide startup config -- so both ranks are
        # mathematically guaranteed to compute the identical verdict with
        # zero wire coordination. Previously the eligibility check ran
        # AFTER the prefix-cache lookup and consumed an
        # ``is_prefix_cache_hit`` input; because each rank's KVPrefixCache
        # is an independent per-process radix trie that can genuinely
        # diverge across ranks (differing timing of when a prior request's
        # tokens fold into each rank's trie), rank 0 and rank 1 could
        # compute different verdicts for the SAME request -- producing the
        # hardware-observed 50x PrefillReadyMessage NACK-retry storm on
        # real N=2. Removing the parameter from is_eligible_for_batched_
        # decode() and routing BEFORE any trie touch structurally
        # eliminates the divergence source.
        #
        # TRADEOFF (intentional, accepted): requests that are shape-
        # eligible for batched-decode will never get prefix-cache-hit
        # benefits while EXO_PP_BATCHED_DECODE is enabled, even for
        # trivial chat-template-boilerplate prefix hits. The prior design
        # already treated ANY non-zero prefix hit as making a request
        # INELIGIBLE for batched-decode, so this doesn't change behavior
        # for genuinely cache-benefiting requests being pulled INTO
        # batched-decode -- it only prevents the divergence at the cost
        # of the trivial-prefix opportunity that the (buggy) old
        # is_prefix_cache_hit input made per-rank-inconsistent anyway.
        # Shape-INELIGIBLE requests (vision/tools/speculative/non-PP) fall
        # through to the serial path below UNCHANGED and still enjoy full
        # prefix-cache behavior (via pipeline_agree_prefix_hit_length()).
        if self._batched_decode_active and (
            self._batched_decode_rank0_glue is not None
            or self._batched_decode_rank1_glue is not None
        ):
            from exo.worker.engines.mlx.pp_batched_decode_eligibility import (
                is_eligible_for_batched_decode,
            )

            early_eligibility = is_eligible_for_batched_decode(
                has_images=bool(task_params.images),
                has_tools=bool(task_params.tools),
                uses_speculative_decode=hasattr(self._mlx_gen, "mtp"),
                sharding_is_pipeline=self.group is not None and self.group.size() > 1,
                batched_decode_enabled=True,
            )
            if early_eligibility.eligible:
                # Batched-eligible: skip KVPrefixCache lookup entirely
                # (do not read, do not mutate -- no trie touch at all)
                # and dispatch to the deferred-prefill batched path with
                # a fresh cold cache. Seed + sampler were built above,
                # shared with the serial path.
                cache_bd = make_kv_cache(self.model, max_kv_size=self.max_kv_tokens)
                with T("submit.batched_decode_enqueue_prefill"):
                    return self._submit_batched_decode_deferred(
                        task_params=task_params,
                        prompt=prompt,
                        prompt_tokens=all_prompt_tokens,
                        all_prompt_tokens=all_prompt_tokens,
                        cache=cache_bd,
                        sampler=sampler,
                        max_tokens=task_params.max_output_tokens or MAX_TOKENS,
                        on_generation_token=on_generation_token,
                        on_prefill_progress=on_prefill_progress,
                        distributed_prompt_progress_callback=(
                            distributed_prompt_progress_callback
                        ),
                        vision=vision,
                        media_regions=media_regions,
                        prefix_hit_length=0,
                        matched_index=None,
                        is_bench=is_bench,
                    )
            logger.debug(
                f"batched-decode ineligible, falling back to serial submit(): "
                f"{early_eligibility.reason}"
            )

        # Multi-rank Pipeline-Parallel serving with coord collectives disabled
        # (EXO_PP_NO_COORD_COLLECTIVE=1, the standard PP launch config) has no
        # channel left to make the prefix-cache hit/miss DECISION collective
        # via the usual mx_any/mx_min_int/get_coord_group path (that path is
        # deliberately unavailable in this mode -- see get_coord_group()'s
        # docstring: a coord all_sum queued behind a blocked p2p recv on the
        # shared transport can deadlock). Each rank's independent,
        # per-process kv_prefix_cache computes its own local hit-length
        # first; pipeline_agree_prefix_hit_length() then makes that
        # agreement WITHOUT a coord collective, by reusing the same raw p2p
        # send/recv_like primitives (and mx.eval discipline) already used
        # for the per-layer hidden-state handoff and for PP+MTP's
        # token/tag exchange -- run as a discrete pre-step strictly before
        # any prefill chunk sends begin, so it can't queue behind or pair
        # with in-flight prefill traffic. "Unanimous or cold": if every
        # rank's local hit-length agrees, use it; on ANY mismatch (a rank
        # evicted a leaf the other didn't, a crash/reconnect left one rank
        # cold, etc.) fall back to hit_length=0 on every rank, identically,
        # rather than trying to reconcile to some smaller shared depth --
        # reconciling risks the exact asymmetric non-sliceable-layer
        # (RotatingKVCache/ArraysCache) restore mismatch this whole
        # subsystem exists to avoid (see references/
        # jaccl-reconnect-crash-loop-and-git-reset-trap.md bug #3). See
        # pipeline_agree_prefix_hit_length()'s docstring for the full
        # protocol (linear reduce + broadcast, int32 wire, tag-checked).
        pp_no_coord_collective = (
            os.environ.get("EXO_PP_NO_COORD_COLLECTIVE") == "1"
            and self.group is not None
            and self.group.size() > 1
        )

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
                local_hit_length = len(all_prompt_tokens) - len(remaining_tokens)

                if pp_no_coord_collective:
                    if local_hit_length == 0:
                        # 2026-08-06 (N=2 admission-race follow-up, see
                        # docs/batched-decode-n2-admission-handoff-2026-08-05.md):
                        # the cross-rank agreement in
                        # pipeline_agree_prefix_hit_length() computes
                        # min(every rank's local hit-length) -- when THIS
                        # rank's own local hit-length is already 0, the
                        # agreed result is MATHEMATICALLY GUARANTEED to
                        # also be 0 (min() over non-negative values can
                        # never exceed the smallest input), independent
                        # of what any other rank reports. Skip the real
                        # wire call entirely in this case -- it would be
                        # both wasted round-trip latency AND, more
                        # importantly under N=2 concurrency, a real
                        # collision risk: this function's own docstring
                        # states its per-process monotonic tag protocol
                        # is only safe when
                        # EXO_MAX_CONCURRENT_REQUESTS=1 (strict
                        # one-request-at-a-time lockstep), a precondition
                        # the 2026-08-06 batched-decode admission-race fix
                        # relaxes to 2 for eligible requests. A real
                        # cluster N=2 test surfaced exactly this: this
                        # call's own p2p traffic raced against the NEW
                        # tick()-driven StepMessage/PrefillMessage
                        # traffic from a concurrently-decoding OTHER
                        # request, producing a real tag mismatch
                        # ("pipeline_agree_prefix_hit_length: tag
                        # mismatch on rank 0") and a jaccl recv deadline.
                        # This fix removes the wire call for the common
                        # case (a genuinely cold request, which is the
                        # normal case for two independent concurrent
                        # conversations sharing no prompt prefix) without
                        # weakening the real cross-rank agreement's own
                        # safety guarantee for the REMAINING case (a
                        # local hit_length > 0 candidate) -- that case
                        # still calls the real wire agreement below,
                        # unchanged, and is a documented, SEPARATE,
                        # scoped-out follow-up gap (racing candidate
                        # cache-hit requests under N=2 concurrency are
                        # NOT yet closed; see this method's own
                        # SCOPE NOTE above and the handoff doc).
                        prefix_hit_length = 0
                        cache = make_kv_cache(
                            self.model, max_kv_size=self.max_kv_tokens
                        )
                    else:
                        self._prefix_hit_agree_tag += 1
                        with T("submit.pp_prefix_hit_agreement"):
                            prefix_hit_length = pipeline_agree_prefix_hit_length(
                                local_hit_length,
                                self.group,
                                self._prefix_hit_agree_tag,
                            )
                        if prefix_hit_length != local_hit_length:
                            # This rank's local match was rejected by cross-rank
                            # agreement (mismatch, or another rank forced 0)
                            # -- release the eviction guard on the leaf we
                            # matched (if any) and fall through to a cold
                            # prefill built fresh below, identically to every
                            # other rank.
                            if local_hit_length > 0:
                                logger.debug(
                                    "PP prefix-cache hit REJECTED by cross-rank "
                                    f"agreement: local={local_hit_length} "
                                    f"agreed={prefix_hit_length} -- falling back "
                                    "to cold prefill on every rank."
                                )
                                self.kv_prefix_cache.release_active_leaf()
                            prompt_tokens = all_prompt_tokens
                            matched_index = None
                            is_exact_hit = False
                            cache = make_kv_cache(
                                self.model, max_kv_size=self.max_kv_tokens
                            )
                        elif prefix_hit_length > 0:
                            logger.info(
                                f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} "
                                f"tokens cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%) "
                                "[PP cross-rank agreed]"
                            )
                            prompt_tokens = remaining_tokens
                        else:
                            cache = make_kv_cache(
                                self.model, max_kv_size=self.max_kv_tokens
                            )
                else:
                    prefix_hit_length = local_hit_length
                    if prefix_hit_length > 0:
                        logger.info(
                            f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                            f"cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
                        )
                        prompt_tokens = remaining_tokens
                    else:
                        cache = make_kv_cache(
                            self.model, max_kv_size=self.max_kv_tokens
                        )
            else:
                cache = make_kv_cache(self.model, max_kv_size=self.max_kv_tokens)

        # API-admitted requests always carry a resolved seed (random when the
        # client sent none — api/main.py resolves it at admission so every PP
        # rank seeds identically). The 42 fallback below is only reachable for
        # engine-internal/bench constructions that bypass the API.
        # (Seed + sampler are now built ABOVE, near the batched-decode routing
        # decision, so both paths share the same construction; they are no
        # longer duplicated here. See the block near "cross-rank eligibility
        # divergence fix" above.)

        # 2026-08-06 cross-rank eligibility divergence fix (bug #6): the
        # batched-decode routing decision has moved EARLIER in this method
        # (before the KVPrefixCache lookup) -- see the block near the top
        # of submit() marked "cross-rank eligibility divergence fix".
        # Any request reaching THIS point is guaranteed to be running the
        # serial-path prefill (either shape-ineligible for batched-decode,
        # or batched-decode not active on this generator). No second
        # eligibility check is needed here.

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
                # snapshot_offset=prefix_hit_length: see prefill()'s docstring
                # (2026-07-28 cross-request contamination bug, round 3) --
                # without this, a partial-hit prefill's snapshots are stamped
                # with a token_count relative to this call's local (post-hit)
                # prompt_tokens instead of the absolute prompt position,
                # letting a later unrelated request's match_length coincide
                # with and restore a stale, wrong-task snapshot.
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
                    snapshot_offset=prefix_hit_length,
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
                    task_params.low_priority,
                    task_params.high_priority,
                )

        last_tokens = prompt_tokens[-2:]

        max_tokens = task_params.max_output_tokens or MAX_TOKENS

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
            # Bench mode (/bench/chat/completions) deliberately bans EOS to
            # force exactly max_tokens of output for consistent throughput
            # timing (see bench/METHODOLOGY.md) and does not grade output
            # correctness at all -- the reasoning-budget limiter has nothing
            # to protect there and forcing an early think_end would only
            # perturb the very decode-length/timing profile bench mode
            # exists to measure. Skip it for bench requests.
            if not is_bench:
                _reasoning_budget = make_reasoning_budget_limiter(
                    think_start_id=safe_think_token_id(
                        self.tokenizer, "think_start_id"
                    ),
                    think_end_id=safe_think_token_id(self.tokenizer, "think_end_id"),
                    budget_tokens=min(
                        int(max_tokens * _REASONING_BUDGET_FRACTION),
                        _REASONING_BUDGET_MAX_TOKENS,
                    ),
                    max_seconds=_REASONING_BUDGET_MAX_SECONDS,
                    starts_in_thinking=detect_thinking_prompt_suffix(
                        prompt, self.tokenizer
                    ),
                    prompt_token_count=int(all_prompt_tokens.size),
                )
                if _reasoning_budget is not None:
                    logits_processors = logits_processors + [_reasoning_budget]
            if is_bench:
                eos_ids = eos_ids_from_tokenizer(self.tokenizer)
                logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        if self._pp_spec_active:
            with T("submit.pp_spec_setup"):
                return self._submit_pp_spec(
                    task_params,
                    all_prompt_tokens,
                    prefix_hit_length,
                    matched_index,
                    is_exact_hit,
                    cache_snapshots,
                    cache,
                    last_tokens,
                    sampler,
                    logits_processors,
                    max_tokens,
                    on_generation_token,
                    _prefill_tps,
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

        # MTP prefill: build MTP cache from prompt hidden states.
        #
        # 2026-05-20 c>=2 fix: each new submission gets a FRESH RotatingKVCache
        # (B=1) via reset_cache(), prefills into it, then snapshots it under
        # this uid via snapshot_for_uid(uid). The shared `mtp._cache` is now
        # ONLY ever a single-stream cache between submit() and the next
        # spec-cycle entry. DSv4MTPBatchGenerator rebuilds the active
        # multi-stream cache from per-uid snapshots at every BS-transition.
        # The old code's `mtp.reset_cache()` (no snapshot) clobbered stream
        # 1's MTP K/V whenever stream 2 arrived → catastrophic c>=2 regression.
        if hasattr(self._mlx_gen, "mtp"):
            prompt_pre_norm = self._mlx_gen._captured.get("prompt_pre_norm")
            if prompt_pre_norm is not None:
                mx.eval(prompt_pre_norm)
                # submit() handles exactly ONE stream, so the MTP cache it
                # prefills + snapshots MUST be batch-1. The shared `_captured`
                # dict holds the pre_norm from the most recent prefill forward;
                # when two independent requests arrive close enough that their
                # prefills batch, that capture is (N>1, S, hidden). Feeding a
                # batched pre_norm into predict() builds a batch-N MTP cache,
                # whose snapshot later fails BatchRotatingKVCache.merge with
                # `Cannot broadcast (N,1,2,512) into (1,1,2,512)` and crashes
                # the runner (fatal c>=2 concurrent regression). Take this
                # stream's own (last) row. No-op at c=1: a (1,S,H) tensor
                # slices to itself, so the champion path stays bit-exact.
                if prompt_pre_norm.shape[0] != 1:
                    prompt_pre_norm = prompt_pre_norm[-1:, :, :]
                self._mlx_gen.mtp.reset_cache()
                S_pre = prompt_pre_norm.shape[1]
                if S_pre > 1:
                    toks_list = (
                        all_prompt_tokens.tolist()
                        if hasattr(all_prompt_tokens, "tolist")
                        else list(all_prompt_tokens)
                    )
                    mtp_tokens = toks_list[1:S_pre]
                    _ = self._mlx_gen.mtp.predict(
                        prompt_pre_norm[:, :-1, :], mx.array([mtp_tokens])
                    )
                    mx.eval(_)
                    logger.info(f"MTP cache prefilled ({S_pre} positions)")
                # Snapshot for this uid. Safe even when S_pre==1 (empty cache
                # is a valid snapshot — subsequent draft will start from
                # zero MTP history, same as the pre-fix c=1 short-prompt case).
                if hasattr(self._mlx_gen.mtp, "snapshot_for_uid"):
                    self._mlx_gen.mtp.snapshot_for_uid(uid)

        # DSpark ctx warm-up + per-request reset. The draft module's rotating
        # ctx-KV window must (a) NOT carry the previous request's context —
        # stale ctx never breaks correctness (verification rejects bad
        # drafts) but silently depresses acceptance — and (b) start warm
        # from this prompt's tail: the capture side channel holds the
        # hc-means from the most recent prefill forward (the last chunk /
        # the incremental suffix on a prefix-cache hit). RoPE positions are
        # window-relative (constant absolute shift cancels in q·k), matching
        # how the per-cycle appends extend the same cache.
        _dspark_mod = getattr(getattr(self._mlx_gen, "model", None), "model", None)
        _dspark_mod = getattr(_dspark_mod, "dspark", None)
        if _dspark_mod is not None:
            from mlx_lm.models.deepseek_v4 import get_dspark_ctx

            self._mlx_gen._dspark_caches = _dspark_mod.make_cache()
            _ds_ctx = get_dspark_ctx(_dspark_mod.target_layer_ids)
            if _ds_ctx is not None:
                if _ds_ctx.shape[0] != 1:
                    _ds_ctx = _ds_ctx[-1:, :, :]
                _dspark_mod.append_ctx(_ds_ctx, self._mlx_gen._dspark_caches)
                mx.eval([c.keys for c in self._mlx_gen._dspark_caches])
                logger.info(f"DSpark ctx warmed ({_ds_ctx.shape[1]} positions)")

        # Set per-request temperature for speculative. EXO_SPECULATIVE_TEMP
        # overrides everything; otherwise fall through the same resolution
        # chain (request → instance → cluster → hardcoded) as the sampler.
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
        self._update_fence_arming()

        return uid

    def submit_batched(  # noqa: C901
        self,
        tasks: list[
            tuple[
                TextGenerationTaskParams,
                str,
                Callable[[int, int], None] | None,
                Callable[[], None] | None,
                Callable[[], None] | None,
            ]
        ],
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
        # Disarm the async fence before ANY forward for these requests
        # (batched prefill must not interleave with deferred graphs from
        # an already-decoding stream).
        self._set_fence_async_engine(False)
        if len(tasks) <= 1:
            # 2026-05-21 diag: log if we got len 1 despite rendezvous gathering 2.
            logger.info(f"submit_batched fallback gate-1: len(tasks)={len(tasks)}")
            return [self.submit(*t) for t in tasks]

        # 2026-08-06 bug #7 fix (Attempt 1 root cause, see
        # docs/batched-decode-n2-admission-handoff-2026-08-05.md's
        # "2026-08-06 root-cause analysis: Attempt 1 and Attempt 2 (bug
        # #7)" section): Phase 1 batched-decode path active -- fall back
        # to per-task submit() (each task's own eligibility check there
        # routes it through the pp_batched_decode_glue single-writer
        # tick()-gated channel, or the safe serial fallback for
        # ineligible requests). EXO_DSV4_BATCHED_PREFILL's rendezvous
        # batching here is a SEPARATE, older mechanism that sends real
        # metaframe prefill traffic directly, with zero awareness of the
        # tick()-gated admission protocol bugs #1/#5/#6 were built
        # around -- letting it run while batched-decode is active lets
        # rank 0 send genuine prefill wire bytes while rank 1 is mid-tick
        # on the decode-step channel, hitting the EXACT
        # SchedulerWireProtocolError version-mismatch crash bug #5's fix
        # was supposed to have already closed (confirmed via real N=2
        # hardware, 2026-08-06). Mirrors the existing `_pp_spec_active`
        # check immediately below -- same shape, same reason (a
        # DIFFERENT admission mechanism must not race the glue's own
        # single-writer wire channel).
        if self._batched_decode_active:
            logger.info(
                "submit_batched: Phase 1 batched-decode active — falling "
                "back to per-task submit() (each task routes through the "
                "pp_batched_decode_glue single-writer channel or the "
                "safe serial fallback)"
            )
            return [self.submit(*t) for t in tasks]

        # PP-spec path: batched prefill not yet wired. Keep falling back.
        # MTP path: 2026-05-20 — was previously falling back to per-task
        # submit(), which serialized prefill across streams (stream 2
        # waited for stream 1's ~6min prefill to complete before starting
        # its own). At c=2 100K MTP-on this collapsed agg t/s to 6 because
        # per_req timing started at submit-done = post-individual-prefill,
        # so stream 1's "decode time" included stream 2's prefill. Now
        # the batched path handles MTP-on too: per-stream MTP cache
        # prefill happens after the batched main prefill, using slices
        # of the captured `prompt_pre_norm`.
        if self._pp_spec_active:
            logger.info(
                "submit_batched: PP-spec active — falling back to per-task submit()"
            )
            return [self.submit(*t) for t in tasks]

        # Heterogeneity guard: any task that needs the per-task path (vision,
        # remote prefill, prefix-cache hit) goes through ``submit()`` so the
        # batched fast path stays simple. Vision is detected via
        # ``task_params.images`` only — the per-message content is always
        # ``InputMessageContent`` (never raw str), so an isinstance check on
        # message content would falsely flag every text request as vision.
        eligible: list[int] = []
        ineligible: list[int] = []
        for i, (task_params, _prompt, _opp, _dppc, _ogt) in enumerate(tasks):
            is_bench = task_params.bench
            has_remote = task_params.prefill_endpoint is not None
            has_vision = bool(task_params.images)
            # OPT-11 (2026-06-24): removed is_bench gate. Batched prefill
            # now works for regular /v1/chat/completions requests, not just
            # /bench. The is_bench gate was a validation guard from the
            # original batched-prefill rollout. The batched path is now
            # proven stable at B=2 with the OPT-10 SDPA fix and
            # MLX_MAX_MB_PER_BUFFER=200. Only exclude remote-prefill and
            # vision tasks (heterogeneous paths not yet wired).
            if has_remote or has_vision:
                ineligible.append(i)
            else:
                eligible.append(i)

        if len(eligible) < 2:
            # 2026-05-21 diag: log which condition caused fallback for
            # diagnosing c=2 batched-prefill rendezvous race.
            for i, (task_params, _prompt, _opp, _dppc, _ogt) in enumerate(tasks):
                logger.info(
                    f"submit_batched fallback gate-3: task[{i}] "
                    f"bench={task_params.bench} "
                    f"has_remote={task_params.prefill_endpoint is not None} "
                    f"has_vision={bool(task_params.images)}"
                )
            return [self.submit(*t) for t in tasks]

        # Run the eligible tasks through the batched path; the rest go
        # through the legacy per-task path. Preserve the input ordering
        # in the returned uids.
        uids: list[int | None] = [None] * len(tasks)

        with T("submit_batched.batched_path"):
            batch_uids = self._submit_batched_eligible([tasks[i] for i in eligible])
        for i, uid in zip(eligible, batch_uids, strict=True):
            uids[i] = uid

        for i in ineligible:
            uids[i] = self.submit(*tasks[i])

        # All slots filled by construction.
        return [u for u in uids if u is not None]

    def _submit_batched_eligible(  # noqa: C901
        self,
        tasks: list[
            tuple[
                TextGenerationTaskParams,
                str,
                Callable[[int, int], None] | None,
                Callable[[], None] | None,
                Callable[[], None] | None,
            ]
        ],
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
        logits_processors_list: list[
            list[Callable[[mx.array, mx.array], mx.array]]
        ] = []
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

            # See submit(): API-admitted requests arrive with a resolved
            # seed; the 42 fallback is engine-internal/bench only.
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
        # 2026-05-20: batched-prefill path now also handles MTP-on. The
        # batched prefill above wrote each stream's main-model KV. Now
        # for each stream, we also need to prefill the per-stream MTP
        # cache (so MTP draft has the right history). The serial submit()
        # path captures `prompt_pre_norm` from the FINAL forward of each
        # stream's prefill (the _CapturingNorm wrapper); the batched
        # prefill's FINAL forward processed all N streams at once, so
        # _captured["prompt_pre_norm"] holds shape (N, last_chunk, hidden).
        # We slice per-stream and prefill each uid's MTP cache.
        captured_prompt_pre_norm = None
        if hasattr(self._mlx_gen, "mtp"):
            captured_prompt_pre_norm = self._mlx_gen._captured.get("prompt_pre_norm")
            if captured_prompt_pre_norm is not None:
                mx.eval(captured_prompt_pre_norm)
            # The per-stream slicing below assumes the capture came from ONE
            # final forward over ALL N streams — (N, last_chunk, hidden).
            # When the batched prefill's last forward covered fewer streams
            # (ragged lengths / per-stream chunking), the capture batch dim
            # is smaller: slicing stream i then yields the WRONG stream's
            # hidden rows for i < B (silent draft-cache contamination) and a
            # ZERO-batch tensor for i >= B, which crashed the runner in
            # mtp_module._combine ("reshape 841 into (0,841)", 2026-07-10).
            # Guard: skip MTP prefill entirely on a batch-dim mismatch —
            # the draft caches start cold (perf-only; verify corrects).
            if captured_prompt_pre_norm is not None and captured_prompt_pre_norm.shape[
                0
            ] != len(tasks):
                logger.warning(
                    "prompt_pre_norm capture batch {} != {} streams; "
                    "skipping MTP cache prefill (cold draft caches)".format(
                        captured_prompt_pre_norm.shape[0], len(tasks)
                    )
                )
                captured_prompt_pre_norm = None

        # ---- Per-stream insert + MTP cache prefill (sequential) ----
        # Each stream's MTP-cache prefill is wall-time non-trivial (~5-15s at
        # 100K). We do this sequentially before registering ANY task in
        # _active_tasks. Once all MTP caches are snapshotted, decode can
        # actually begin — so we capture a single shared "decode wall starts
        # here" timestamp AFTER this loop, and assign it to every stream's
        # generation_time_at_start. Otherwise stream 0's gen_time_delta at
        # completion would absorb the wall stream 1 spent prefilling its MTP
        # cache (~15s skew on outlier iters), since decode is batched and
        # neither stream actually produces tokens until BOTH MTP caches are
        # ready. See Phase 14 Plan A for the 2026-05-21 forensics.
        uids: list[int] = []
        per_stream_meta: list[
            tuple[
                int,  # uid
                TextGenerationTaskParams,  # task_params
                Callable[[], None] | None,  # on_generation_token
            ]
        ] = []
        for i, (task_params, _prompt, _opp, _dppc, on_generation_token) in enumerate(
            tasks
        ):
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

            # MTP per-stream cache prefill. Mirrors submit() line 1135+,
            # but uses the batched-prefill's captured `prompt_pre_norm`
            # slice (B=N → 1) instead of the per-task forward's capture.
            # Each call to mtp.predict() advances the SHARED self._cache,
            # but snapshot_for_uid stashes it under this uid so
            # activate_for_uids can reconstruct it later.
            if hasattr(self._mlx_gen, "mtp") and captured_prompt_pre_norm is not None:
                # Slice this stream's pre_norm: (N, last_chunk, hidden) → (1, last_chunk, hidden).
                stream_pre_norm = captured_prompt_pre_norm[i : i + 1]
                self._mlx_gen.mtp.reset_cache()
                S_pre = stream_pre_norm.shape[1]
                # Ragged batch: the captured chunk is RIGHT-padded to the
                # batch max_L, so a shorter stream's true rows are the FIRST
                # k_i = len_i - (max_L - S_pre) rows; its tail rows are
                # padding. Pairing the uniform S_pre-row capture with this
                # stream's unpadded token tail crashed the runner on any
                # rendezvous batch of unequal-length prompts
                # ([broadcast_shapes] (1,21,4096) vs (1,72,4096), 2026-07-02).
                # Slice both sides to the stream's true rows; if fewer than
                # 2 true rows land in the captured chunk, skip the prefill —
                # the draft cache just starts cold for this stream.
                max_l = max(len(t) for t in all_prompt_tokens_list)
                toks_list = all_prompt_tokens_list[i].tolist()
                n_total = len(toks_list)
                k_i = min(S_pre, n_total - (max_l - S_pre))
                if k_i > 1:
                    # hidden rows 0..k_i-2 (positions n-k_i .. n-2) pair with
                    # tokens n-k_i+1 .. n-1 — same offset semantics as the
                    # serial submit() path.
                    mtp_tokens = toks_list[n_total - k_i + 1 : n_total]
                    _ = self._mlx_gen.mtp.predict(
                        stream_pre_norm[:, : k_i - 1, :],
                        mx.array([mtp_tokens]),
                    )
                    mx.eval(_)
                    logger.info(
                        f"MTP cache prefilled ({k_i} positions, batched stream {i})"
                    )
                if hasattr(self._mlx_gen.mtp, "snapshot_for_uid"):
                    self._mlx_gen.mtp.snapshot_for_uid(uid)

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

            per_stream_meta.append((uid, task_params, on_generation_token))

        # Release the captured prompt_pre_norm + the _captured dict. The
        # batched-prefill's final forward captured (N, last_chunk, hidden)
        # activations for MTP cache prefill above. With all MTP caches now
        # snapshotted, the capture is dead. Without del + clear, it lingers
        # in self._mlx_gen._captured across turns — a retained intermediate
        # that contributes to the prefill working-set accumulation (fact 778).
        del captured_prompt_pre_norm
        if hasattr(self._mlx_gen, "_captured"):
            self._mlx_gen._captured.clear()
        mx.clear_cache()

        # All per-stream MTP caches are now snapshotted. Decode is about to
        # start for every stream simultaneously, so capture ONE shared
        # generation-start wall and assign it to every _active_tasks entry.
        common_generation_start = time.perf_counter()
        common_generation_time_at_start = _mlx_gen_elapsed_seconds(self._mlx_gen)

        for i, (uid, task_params, on_generation_token) in enumerate(per_stream_meta):
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
                generation_start_time=common_generation_start,
                prefill_tps=per_stream_tps[i],
                # Match submit()'s bookkeeping: capture wall after prefill so
                # gen_tps reflects decode-only timing, not prefill. For batched
                # prefill we use a SHARED capture across all streams (see the
                # block above) so per-stream gen_time_delta measures from the
                # same anchor — eliminates the per-stream skew that produced
                # the 16.7/22.5 outliers in the 10-iter c=2 validation.
                generation_time_at_start=common_generation_time_at_start,
                media_regions=[],
            )
            self._update_fence_arming()

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
        # Entry guard (2026-07-31): reject a second concurrent PP-spec
        # submission explicitly instead of silently clobbering the first
        # request's generator. See PPSpecAlreadyActiveError's docstring and
        # the long comment on _pp_spec_gen_by_uid for the full rationale --
        # PP's shared wire-link state genuinely can't run two of these at
        # once yet; this converts what used to be silent data corruption
        # into a loud, immediately-visible rejection. Checked FIRST, before
        # any shared layer state (_install_spec_layers/_configure_layers
        # below) is touched, so a rejected second request has zero side
        # effects on the first request's in-flight decode.
        if self._pp_spec_gen_by_uid:
            _active_uid = next(iter(self._pp_spec_gen_by_uid))
            raise PPSpecAlreadyActiveError(
                f"PP speculative decode already active for uid={_active_uid}; "
                "a second concurrent PP-spec request is not supported by "
                "today's architecture (shared rank0<->rank1 wire-link state "
                "in SpecPipelineFirstLayer/SpecPipelineLastLayer -- see "
                "PPSpecAlreadyActiveError's docstring). Reject and let the "
                "caller retry once the active request completes, rather "
                "than silently corrupting either request's state."
            )
        from exo.worker.engines.mlx.trace import T, request_trace

        from ..pp_speculation import (
            _configure_layers,
            _install_spec_layers,
            get_pipeline_info,
            pp_chained_decode_loop,
            pp_dspark_decode_loop,
            pp_speculative_decode_loop,
        )

        with T("pp_spec.get_pipeline_info"):
            pp_info = get_pipeline_info(self.model)
            assert pp_info is not None
            pp_rank, pp_world_size, pp_group = pp_info

        with T("pp_spec.install_spec_layers"):
            # FIX (2026-07-17): must resolve to the TRUE inner model (the
            # object whose `.layers` is the actual persistent list the
            # forward pass iterates), not `language_model`/`model` alone.
            # For DeepSeek-V4 (no `language_model` attr), the outer `Model`
            # class exposes a `.layers` PROPERTY that returns
            # `self.model.pipeline_layers` -- itself a property computing
            # `self.layers[start_idx:end_idx]`, i.e. a BRAND NEW list
            # object on every single access. _install_spec_layers used to
            # be called with `inner = getattr(self.model, "language_model",
            # self.model)` (= self.model itself here, since no
            # language_model attr exists), so `inner.layers` resolved to
            # that disposable property-computed slice -- mutating it had
            # ZERO effect on the model's real, persistent layer list.
            # SpecPipelineFirstLayer/SpecPipelineLastLayer were correctly
            # constructed and "installed" into a list that was discarded
            # the instant this function returned, so the model's actual
            # forward pass never called them -- confirmed via unconditional
            # canary prints in both classes' __call__ that never fired
            # despite _install_spec_layers appearing to succeed. Resolve
            # one level deeper (mirroring pp_speculation.py's own
            # `inner_model = getattr(inner, "model", inner)` pattern) so
            # this mutates the actual persistent `.layers` list.
            inner = getattr(self.model, "language_model", self.model)
            inner_model = getattr(inner, "model", inner)
            _spec_first, _spec_last = _install_spec_layers(inner_model)
            # DEFENSIVE RESET (2026-07-20, jaccl transport-fault root
            # cause): _spec_first/_spec_last are the SAME persistent
            # layer OBJECTS across every request once installed
            # (_install_spec_layers' isinstance guard never re-wraps
            # them) -- their _pp_recv/_pp_send/_speculative mode flags
            # are mutable instance state that can OUTLIVE the PREVIOUS
            # request's decode-loop generator if that generator's
            # `finally: _configure_layers(...)` reset never ran (e.g. a
            # client disconnect / exception path that bypasses
            # _close_pp_spec_gen(), or a race between one rank's
            # generator being suspended mid-recv and its close() call).
            # Live-traced 2026-07-20: the plain (non-speculative)
            # first-token stream_generate() call immediately below ran
            # with rank1's SpecPipelineFirstLayer still armed
            # _pp_recv=True from a PRIOR request's decode loop, taking
            # the speculative recv branch during plain prefill while
            # rank0's corresponding layer correctly showed
            # _pp_send=False (FALLTHROUGH TO BASE) -- an asymmetric
            # mode mismatch that left rank1 blocked on a recv rank0
            # never sends, eventually tripping jaccl's 15s drain
            # deadline (`[jaccl] recv() deadline in drain — clean
            # re-place`). Reset unconditionally here, BEFORE the
            # plain-PP first-token call below, so this entry point's
            # correctness never depends on any OTHER code path's
            # cleanup having succeeded on every rank. Cheap (attribute
            # writes only, no wire traffic) -- safe to call even when
            # spec_first/spec_last end up unused this request (e.g. a
            # non-DSpark PP path is about to be taken instead).
            _configure_layers(_spec_first, _spec_last)

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
                    _pp_draft(
                        _draft_tokens[i : i + _draft_chunk][None], cache=_pp_draft_cache
                    )
                    mx.eval(
                        [c.state if hasattr(c, "state") else c for c in _pp_draft_cache]
                    )
                mx.clear_cache()
                logger.info(
                    f"Draft model prefilled with {len(_draft_tokens)} tokens (of {len(all_prompt_tokens)} total)"
                )

        # First token via standard PP
        with T("pp_spec.first_token"):
            _first_gen = stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=last_tokens,
                max_tokens=1,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=cache,
                prefill_step_size=1,
                kv_group_size=KV_GROUP_SIZE,
                kv_bits=KV_BITS,
            )
            _first_out = next(_first_gen)
            first_y = mx.array([_first_out.token])
            mx.eval(first_y)

        logger.info(f"PP speculation active: rank={pp_rank}")

        # Get PP MTP predictor (lightweight, skip_mlp=True)
        _pp_mtp = getattr(self, "_pp_mtp", None)
        if _pp_mtp is not None:
            logger.info("PP speculation using MTP for drafting")

        # Create the spec decode generator
        request_trace.mark("pp_spec.decode_loop_start")
        # DSpark -- highest priority when available: a dedicated 3-stage
        # semi-autoregressive draft head, strictly more capable than either
        # the plain single-MTP-head chained path below or the original
        # single-token MTP path. See pp_speculation.py's module-level
        # comment above pp_dspark_decode_loop for the full rank1-owned
        # draft+verify design and why DSpark's context-conditioning taps +
        # lm_head being rank1-resident inverts the rank0-drafts-during-
        # idle-time assumption the other two paths use.
        #
        # Gating: attach-at-load (EXO_DSV4_DSPARK=1, utils_mlx.py) is the
        # ONLY control surface -- there used to be a second, redundant
        # EXO_PP_DSPARK flag gating "use it as the PP decode loop", but the
        # two were never toggled independently in practice (EXO_PP_DSPARK
        # requiring EXO_DSV4_DSPARK=1 to do anything was called out as a
        # hard dependency in start_cluster.sh's own comments) and having
        # both was just confusing (2026-07-26). _has_dspark below is the
        # correct single source of truth: it reflects the actual post-load
        # runtime state, including the rank-consistency guard that can
        # detach DSpark group-wide if any rank's overlay failed -- so this
        # naturally falls back correctly even when the load-time flag was
        # set but attachment didn't actually succeed.
        _inner_pp_for_dspark = getattr(self.model, "model", None)
        _has_dspark = (
            _inner_pp_for_dspark is not None
            and getattr(_inner_pp_for_dspark, "dspark", None) is not None
        )
        # Opt-in k-token chained MTP draft + batched verify (see
        # pp_speculation.py's module-level comment above _PP_MTP_CHAIN_K
        # for the full design). Only reachable when: (a) explicitly
        # requested via EXO_PP_MTP_CHAIN_K>1, and (b) a native DSv4 MTP
        # predictor is active (the generic draft-model fallback and the
        # Qwen3.5-style MTPPredictor don't implement predict()'s
        # return_hidden=True chaining contract this path depends on).
        # Falls back to the proven single-token path otherwise --
        # default (EXO_PP_MTP_CHAIN_K unset or =1) is UNCHANGED behavior.
        _chain_k = int(os.environ.get("EXO_PP_MTP_CHAIN_K", "1"))
        # Stash pp_rank on self (2026-07-19) purely for diagnostic logging in
        # _step_pp_spec/_close_pp_spec_gen -- the loop-building code above
        # only has it as a local, but the stream-hang investigation needs
        # rank context on every finish-decision log line to correlate the
        # two ranks' independent EOS/max_tokens decisions against each
        # other (see EXO_PP_SPEC_FINISH_LOG below).
        self._pp_rank_for_log = pp_rank
        if _has_dspark:
            logger.info("PP speculation using DSpark (rank1-owned draft+verify)")
            _pp_spec_gen = pp_dspark_decode_loop(
                model=self.model,
                prompt_cache=cache,
                first_y=first_y,
                max_tokens=max_tokens - 1,
                pp_rank=pp_rank,
                pp_world_size=pp_world_size,
                pp_group=pp_group,
            )
        elif _chain_k > 1 and _pp_mtp is not None and hasattr(_pp_mtp, "predict"):
            logger.info(f"PP speculation using chained MTP draft, k={_chain_k}")
            _pp_spec_gen = pp_chained_decode_loop(
                model=self.model,
                prompt_cache=cache,
                sampler=sampler,
                first_y=first_y,
                first_logprobs=mx.zeros(1),
                max_tokens=max_tokens - 1,
                pp_rank=pp_rank,
                pp_world_size=pp_world_size,
                pp_group=pp_group,
                mtp_predictor=_pp_mtp,
                chain_k=_chain_k,
            )
        else:
            _pp_spec_gen = pp_speculative_decode_loop(
                model=self.model,
                draft_model=_pp_draft,  # type: ignore
                prompt_cache=cache,
                draft_cache=_pp_draft_cache,  # type: ignore
                sampler=sampler,
                logits_processors=logits_processors,
                first_y=first_y,
                first_logprobs=mx.zeros(1),
                max_tokens=max_tokens - 1,
                pp_rank=pp_rank,
                pp_world_size=pp_world_size,
                pp_group=pp_group,
                mtp_predictor=_pp_mtp,
            )

        self._uid_counter += 1
        uid = self._uid_counter
        # Dict write (2026-07-31, see _pp_spec_gen_by_uid's class-level
        # comment): the entry guard at the top of this method already
        # ensures the dict is empty here, so this is always a fresh
        # single-entry insert, never an overwrite.
        self._pp_spec_gen_by_uid[uid] = _pp_spec_gen

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
        self._update_fence_arming()

        return uid

    def _step_pp_spec(self) -> list[GenerationBatch.Response]:
        """Get next token from PP speculative decode loop."""
        # Single active entry, guaranteed by _submit_pp_spec's entry guard
        # (see _pp_spec_gen_by_uid's class-level comment) -- this dict
        # never holds more than one uid in today's architecture.
        uid = next(iter(self._pp_spec_gen_by_uid))

        # Diagnostic instrumentation (2026-07-19, EXO_PP_SPEC_FINISH_LOG=1,
        # default off): investigating the 2026-07-18 stream-never-closed
        # hang. Two distinct failure shapes are consistent with the
        # reported symptom (decode completes server-side, GPUs idle,
        # RunnerRunning never transitions):
        #   (a) TRUE DEADLOCK -- one rank blocked inside an mx.distributed
        #       send/recv waiting for a message its peer never sends. The
        #       existing stall sampler (EXO_STALL_SAMPLER_SECONDS, ON by
        #       default) already catches this: _step_beat stops advancing,
        #       full thread stacks dump to ~/exo_stall_dumps after 10s.
        #   (b) SILENT MISCOUNT -- both ranks keep calling step() and
        #       returning normally (so the stall sampler NEVER fires --
        #       _step_beat keeps refreshing), but one rank's finish
        #       decision (EOS membership test, or `n >= max_tokens` inside
        #       the inner generator) diverges from its peer's, so one rank
        #       cleanly finishes+closes while the other's generator blocks
        #       on the next cycle's wire recv forever. Nothing existing
        #       today would catch this shape at all -- there is no log
        #       line anywhere that records the actual is_eos/finish_reason
        #       decision alongside a per-rank cycle counter, so a diverging
        #       decision between ranks is invisible until you're staring at
        #       a wedged runner with no causal trail.
        # This block adds exactly that: a monotonic call counter (call_n)
        # and, only on a finish decision, one log line with rank + call_n +
        # token id + which branch fired. Gated behind its own env var
        # (distinct from EXO_TRACING_ENABLED) so it can be enabled without
        # also paying for the (much noisier) per-cycle wire-protocol trace
        # in pp_speculation.py's _log(). Zero cost when unset -- the
        # counter increment is one int add, and the env lookup is cached
        # via getattr on first call.
        _fin_log = getattr(self, "_pp_spec_finish_log_enabled", None)
        if _fin_log is None:
            _fin_log = os.environ.get("EXO_PP_SPEC_FINISH_LOG", "0") == "1"
            self._pp_spec_finish_log_enabled = _fin_log
        _call_n = getattr(self, "_pp_spec_call_n", 0) + 1
        self._pp_spec_call_n = _call_n
        _rank_for_log = getattr(self, "_pp_rank_for_log", "?")

        # Yield the first token if we haven't yet
        if hasattr(self, "_pp_first_token"):
            tok = self._pp_first_token
            del self._pp_first_token
            # int() normalization (2026-07-19). VERIFIED NOT THE BUG: traced
            # both source sites in pp_speculation.py -- rank1's
            # `bonus_token = _all_next_list[n_accepted]` and rank0's
            # `bonus_token = _wire2_list[1]` are BOTH already plain Python
            # ints (`[int(v) for v in ...tolist()]` casts happen at the
            # source on both paths already). This int() call is therefore a
            # no-op here, kept only as cheap defensive belt-and-suspenders in
            # case a future edit reintroduces an mx-scalar leak -- it is NOT
            # a fix for the reported stream-hang and should not be reported
            # as one. The actual hang root cause is still open; see the
            # handoff notes.
            is_eos = int(tok) in self._pp_spec_eos
            if _fin_log:
                logger.info(
                    f"[PP_SPEC_FINISH] rank={_rank_for_log} call_n={_call_n} "
                    f"branch=first_token tok={tok} is_eos={is_eos}"
                )
            if is_eos:
                # EOS on the very first token: the PP decode-loop generator
                # was already created (and, for pp_dspark_decode_loop, has
                # already done its cold-start draft() + cache mutation) but
                # never entered its `while n < max_tokens:` body via next().
                # Explicitly close it rather than relying on the later plain
                # `self._pp_spec_gen = None` assignment in step() (2026-07-19
                # hardening -- see the matching comment below for why bare
                # refcount-drop finalization is not safe to depend on here).
                self._close_pp_spec_gen(uid)
            return [
                GenerationBatch.Response(
                    uid=uid,
                    token=tok,
                    logprobs=mx.zeros(1),
                    finish_reason="stop" if is_eos else None,
                    prompt_cache=None,
                    all_tokens=None,
                )
            ]

        _pp_spec_gen = self._pp_spec_gen_by_uid[uid]
        try:
            tok_id, lp = next(_pp_spec_gen)
            # int() normalization -- verified NOT the bug, kept as harmless
            # defensive belt-and-suspenders. See the matching comment on the
            # first-token branch above for the full trace.
            is_eos = int(tok_id) in self._pp_spec_eos
            if _fin_log:
                logger.info(
                    f"[PP_SPEC_FINISH] rank={_rank_for_log} call_n={_call_n} "
                    f"branch=steady_state tok={tok_id} is_eos={is_eos}"
                )
            if is_eos:
                # BUG HARDENING (2026-07-19, investigating the stream-never-
                # closed hang from last session): on EOS this used to just
                # return finish_reason="stop" and leave self._pp_spec_gen
                # ALIVE and suspended mid-`while` -- the only cleanup was a
                # later bare `self._pp_spec_gen = None` in step() (search
                # "Clean up spec state"), which finalizes the generator via
                # CPython refcounting, NOT an explicit close(). That's fine
                # IF the generator has zero reference cycles back to `self`
                # (e.g. via the `_captured` hidden-state dict closure or the
                # model/cache args bound into the frame) -- but if a cycle
                # exists, finalization (and therefore the generator's
                # `finally:` block, which resets `_configure_layers` pipeline
                # send/recv state) is deferred to the next cyclic-GC pass
                # instead of running deterministically right here. That is a
                # plausible contributor to a wedged/stuck pipeline-layer
                # config surviving into the runner's idle-transition path.
                # Close explicitly and immediately instead of trusting GC
                # timing; StopIteration/GeneratorExit from an already-primed
                # but not-yet-iterated generator is expected and swallowed by
                # close() itself, so no try/except is needed here.
                self._close_pp_spec_gen(uid)
            return [
                GenerationBatch.Response(
                    uid=uid,
                    token=tok_id,
                    logprobs=lp,
                    finish_reason="stop" if is_eos else None,
                    prompt_cache=None,
                    all_tokens=None,
                )
            ]
        except StopIteration:
            # max_tokens reached
            if _fin_log:
                logger.info(
                    f"[PP_SPEC_FINISH] rank={_rank_for_log} call_n={_call_n} "
                    f"branch=stop_iteration (max_tokens reached)"
                )
            self._close_pp_spec_gen(uid)
            return [
                GenerationBatch.Response(
                    uid=uid,
                    token=0,
                    logprobs=mx.zeros(1),
                    finish_reason="length",
                    prompt_cache=None,
                    all_tokens=None,
                )
            ]

    def _close_pp_spec_gen(self, uid: int) -> None:
        """Deterministically finalize the PP spec-decode generator for ``uid``.

        Calls .close() explicitly (runs its `finally:` block synchronously,
        right here) instead of just dropping the reference and hoping
        CPython's refcounter (not cyclic GC) gets to it immediately. See the
        EOS-path comment in _step_pp_spec for the full rationale -- this is
        a defensive hardening pass, not a confirmed fix for the 2026-07-18
        stream-never-closed hang (that needs live re-validation once the
        cluster is reachable again).

        Takes an explicit uid (2026-07-31, see _pp_spec_gen_by_uid's
        class-level comment) rather than reading a singular instance
        attribute -- pops exactly the entry this call site is finishing,
        so it can never accidentally clear a DIFFERENT uid's state.
        """
        gen = self._pp_spec_gen_by_uid.pop(uid, None)
        if gen is not None:
            try:
                gen.close()
            except Exception:
                logger.debug("pp spec-decode generator close() raised", exc_info=True)

    def _run_deferred_prefill_for_grant(
        self, grant: "PrefillGrant", *, is_rank1: bool
    ) -> None:
        """Shared grant-fulfillment logic for both ranks
        (2026-08-06 N=2 admission-race fix): looks up the
        ``_DeferredPrefill`` this rank registered at ``submit()``
        time for ``grant.request_id`` and runs its real prefill --
        via one of TWO paths, decided fresh on every grant (2026-08-07,
        Phase 2 live-wiring):

        1. Chunk-interruptible (``deferred.try_start_chunked_prefill()``
           returns non-None): registers the FIRST real chunk's
           ``ResumablePrefillSession`` with this rank's own glue
           (``register_prefill_session``) and returns WITHOUT calling
           ``enqueue_admission``/``stage_local_cache`` -- Hazard 1's
           fix (caller-assumed-completion): nothing downstream can
           mistake "a session was registered" for "this request's
           prefill is done and ready to decode" because this method's
           own return carries no such signal; ``_deferred_prefill_by_uid``
           keeps this request's entry (NOT popped) exactly because its
           prefill is still genuinely in progress across FUTURE ticks.
           ``_step_batched_decode`` only calls ``enqueue_admission``/
           ``stage_local_cache`` for THIS request once
           ``prefill_interruptible_advance`` -- called from
           ``_advance_chunked_prefill_drive`` -- reaches the real
           ("done", ...) outer-generator yield, many ticks later.

        2. Synchronous (chunking ineligible -- small prompt, remote
           prefill, or the loaded model doesn't structurally support
           ``_forward_steps`` yet): runs ``deferred.run_prefill()`` to
           completion NOW, exactly as this method did before
           2026-08-07, and immediately calls ``enqueue_admission``/
           ``stage_local_cache`` -- zero behavior change for this case
           (the ONLY case that has ever run on real production
           hardware to date, since the mlx-lm submodule pin as of
           2026-08-07 doesn't yet include DSv4's real ``_forward_steps``
           split -- see this session's design doc entry for the
           follow-up that closes that gap).

        Both paths pop ``deferred`` from ``_deferred_prefill_by_uid``
        ONLY once this request's real prefill has genuinely, fully
        completed -- path 2 pops immediately (below); path 1's pop
        happens later, inside ``_advance_chunked_prefill_drive``, at
        the SAME point ``enqueue_admission``/``stage_local_cache``
        finally runs for that request.

        2026-08-06 UPDATE (prefill forward-pass race fix -- see
        ``PrefillReadyMessage``'s own docstring in
        ``pp_scheduler_protocol.py`` for the full incident this
        closes): this method is now ONLY EVER CALLED with a grant for
        a request whose matching ``_DeferredPrefill`` is genuinely
        registered. The real-hardware-confirmed race this used to
        paper over with a "parking" mechanism (rank 1 receiving a
        grant before its own registration existed, then silently
        deferring the real forward pass to a LATER, unsynchronized
        call -- while rank 0 had ALREADY started sending real
        metaframe bytes, assuming rank 1 was running its matching
        side right now) is now closed one level up, in
        ``Rank1BatchedDecodeGlue.tick()``'s own ``PrefillReadyMessage``
        ack/NACK handshake: rank 1 NEVER returns a grant for an
        unregistered request in the first place (it NACKs instead,
        and rank 0 retries on a later tick until registration lands).
        A missing ``_DeferredPrefill`` here is therefore now
        structurally impossible on EITHER rank -- fail loud
        immediately rather than silently parking, matching rank 0's
        own pre-existing fail-loud behavior for the identical case.
        """
        deferred = self._deferred_prefill_by_uid.get(grant.request_id)
        if deferred is None:
            from exo.worker.engines.mlx.pp_batched_decode_glue import GlueError

            raise GlueError(
                f"_run_deferred_prefill_for_grant: rank "
                f"{1 if is_rank1 else 0} received a PrefillGrant for "
                f"request_id={grant.request_id} but has no matching "
                f"_DeferredPrefill registered. Since the 2026-08-06 "
                f"PrefillReadyMessage ack/NACK fix, this is "
                f"structurally impossible on EITHER rank -- rank 1 "
                f"only ever returns a grant for a request its own "
                f"mark_prefill_registered() already covered (it NACKs "
                f"and rank 0 retries otherwise), and rank 0's grant "
                f"origin is the SAME call path as its own "
                f"enqueue_prefill(). Refusing to guess at a prefill "
                f"to run."
            )

        drive = deferred.try_start_chunked_prefill()
        if drive is not None:
            deferred.drive = drive
            if is_rank1:
                assert self._batched_decode_rank1_glue is not None
                self._batched_decode_rank1_glue.register_prefill_session(
                    grant.request_id, drive.session
                )
            else:
                assert self._batched_decode_rank0_glue is not None
                self._batched_decode_rank0_glue.register_prefill_session(
                    grant.request_id, drive.session, chunk_index=drive.chunk_index
                )
            return

        # Path 2 (synchronous, unchanged behavior) -- pop NOW, this
        # request's real prefill is about to run to full completion in
        # this same call.
        self._deferred_prefill_by_uid.pop(grant.request_id, None)

        # cache_snapshots is already consumed inside run_prefill()'s own
        # closure (it calls _save_prefix_cache with it before returning) --
        # not needed again here, hence the underscore-prefixed discard.
        prefill_tps, _prefill_tokens, _cache_snapshots, prefilled_cache = (
            deferred.run_prefill()
        )
        self._admit_completed_prefill(
            grant,
            deferred,
            prefill_tps=prefill_tps,
            prefilled_cache=prefilled_cache,
            is_rank1=is_rank1,
        )

    def _admit_completed_prefill(
        self,
        grant: "PrefillGrant",
        deferred: "_DeferredPrefill",
        *,
        prefill_tps: float,
        prefilled_cache: "KVCacheType",
        is_rank1: bool,
    ) -> None:
        """Shared tail (2026-08-07) for BOTH real-prefill-completion
        paths (the synchronous path in ``_run_deferred_prefill_for_grant``
        and the chunk-drive path in ``_advance_chunked_prefill_drive``):
        folds a genuinely-completed prefill's result into the batch via
        ``enqueue_admission`` (rank 0) / ``stage_local_cache`` (rank 1)
        -- extracted so both paths apply the IDENTICAL admission logic,
        never duplicated or allowed to drift apart.
        """
        active_task = self._active_tasks.get(grant.request_id)
        if active_task is not None:
            active_task.prefill_tps = prefill_tps

        if is_rank1:
            assert self._batched_decode_rank1_glue is not None
            self._batched_decode_rank1_glue.stage_local_cache(
                request_id=grant.request_id,
                cache_slot=grant.cache_slot,
                prefilled_cache=prefilled_cache,
            )
        else:
            assert self._batched_decode_rank0_glue is not None
            self._batched_decode_rank0_glue.enqueue_admission(
                request_id=grant.request_id,
                cache_slot=grant.cache_slot,
                prefilled_cache=prefilled_cache,
                initial_token=int(deferred.last_tokens[-1].item()),
                sampler=deferred.sampler,
                max_tokens=deferred.max_tokens,
            )

    def _advance_chunked_prefill_drive(
        self, request_id: int, *, is_rank1: bool
    ) -> None:
        """Called from ``_step_batched_decode`` ONLY when this rank's
        ``tick()`` just returned a non-None ``PrefillAdvanceCompleted``
        for ``request_id`` (2026-08-07, Phase 2 live-wiring) -- i.e. the
        CURRENT chunk's ``ResumablePrefillSession`` (owned by the glue,
        driven entirely inside ``tick()``'s own single-writer call) just
        reached ``done=True``. Resumes the outer
        ``_pipeline_parallel_prefill_steps`` generator via
        ``prefill_interruptible_advance`` and handles its two real
        outcomes:

        - Another real chunk remains: registers the NEW
          ``ResumablePrefillSession`` with this rank's glue
          SYNCHRONOUSLY, INLINE, before this method returns -- never
          scheduled or deferred to a later loop turn. This is Hazard
          2's fix for the inner-chunk-boundary case: since ``tick()``
          is the ONLY real recv call site on either rank, and this
          method runs to completion strictly between one ``tick()``
          return and the next ``tick()`` call in the SAME runner event
          loop, rank 1 physically cannot observe chunk i+1's FIRST
          ``PrefillAdvanceMessage`` before registering chunk i+1's own
          session, given ordered wire delivery -- provably, not merely
          probably, true.
        - The drive is genuinely done: ``prefill_interruptible_advance``
          returns the real ``(tokens_per_sec, num_tokens, snapshots)``
          tuple (the outer generator's own trailing code -- final
          forward pass, flush, eval -- already ran before this point).
          Runs ``deferred.finalize_prefill`` (the SAME RotatingKVCache
          clamp + prefix-cache write-back tail the synchronous path
          runs) and THEN calls ``_admit_completed_prefill`` -- the
          request is NOW, for the first time, genuinely ready to
          decode. Pops ``deferred`` from ``_deferred_prefill_by_uid``
          here -- the SAME point ``_run_deferred_prefill_for_grant``'s
          synchronous path pops it, so both paths free this bookkeeping
          at the identical logical moment ("prefill fully done"), never
          earlier.
        """
        from exo.worker.engines.mlx.generator.generate import (
            prefill_interruptible_advance,
        )
        from exo.worker.engines.mlx.pp_batched_decode_glue import (
            GlueError,
            PrefillGrant,
        )

        deferred = self._deferred_prefill_by_uid.get(request_id)
        if deferred is None or deferred.drive is None:
            raise GlueError(
                f"_advance_chunked_prefill_drive: rank "
                f"{1 if is_rank1 else 0} tick() reported "
                f"PrefillAdvanceCompleted for request_id={request_id} "
                f"but no matching in-flight ChunkedPrefillDrive is "
                f"registered -- this rank's own glue and "
                f"ExoBatchGenerator's drive bookkeeping have desynced"
            )
        drive = deferred.drive
        result = prefill_interruptible_advance(drive)

        if isinstance(result, tuple):
            prefill_tps, num_tokens, cache_snapshots = result
            _prefill_tps, _prefill_tokens, _cache_snapshots, prefilled_cache = (
                deferred.finalize_prefill(prefill_tps, num_tokens, cache_snapshots)
            )
            deferred.drive = None
            self._deferred_prefill_by_uid.pop(request_id, None)
            grant = PrefillGrant(
                request_id=request_id,
                cache_slot=deferred.cache_slot,
                n_prompt_tokens=num_tokens,
                single_request_fallback=False,
            )
            self._admit_completed_prefill(
                grant,
                deferred,
                prefill_tps=_prefill_tps,
                prefilled_cache=prefilled_cache,
                is_rank1=is_rank1,
            )
            return

        # Another real chunk remains: `result` is the NEW
        # ResumablePrefillSession -- register it synchronously, inline,
        # per this method's own docstring (Hazard 2 fix).
        if is_rank1:
            assert self._batched_decode_rank1_glue is not None
            self._batched_decode_rank1_glue.register_prefill_session(request_id, result)
        else:
            assert self._batched_decode_rank0_glue is not None
            self._batched_decode_rank0_glue.register_prefill_session(
                request_id, result, chunk_index=drive.chunk_index
            )

    def _step_batched_decode(self) -> list[GenerationBatch.Response]:
        """One step of the Phase 1 batched-decode session (design doc
        Section 9). Rank-appropriate: rank 0 calls ``tick()`` on its
        admitting/sampling glue and translates the result into real
        ``GenerationBatch.Response`` objects (the SAME contract
        ``_step_pp_spec`` returns, so the rest of ``step()``'s
        response-processing loop works completely unmodified); rank 1
        calls ``tick()`` on its mirror-only glue and returns an empty
        list on every tick EXCEPT an eviction tick (see the 2026-08-06
        bug #7 fix below) -- it has no DECODE output of its own to
        report to a client (matches ``RankOneMirrorSession``'s own
        zero-decision-logic design and the pp_spec path's existing
        rank1-produces-no-CONTENT convention: ``runner.py``'s
        ``send_chunk`` already discards everything on non-zero ranks,
        so the response's actual token/content is never observed by a
        client either way).

        Eviction (a request hitting EOS/max_tokens/degeneration) is
        driven from HERE, sequentially AFTER ``tick()`` returns --
        never concurrently with it. This stays inside the same
        single-writer call chain the whole glue subsystem is built
        around (``step()`` -> this method -> ``tick()`` then,
        separately, ``complete_request()``), matching how PP's
        existing send/recv-based decode loop elsewhere in this class
        already blocks synchronously rank-to-rank every step -- this
        is not new risk beyond what every other PP code path in this
        file already accepts.

        2026-08-06 UPDATE (N=2 admission-race fix): ``tick()`` on
        EITHER rank may now also return a non-None ``PrefillGrant``
        (see that dataclass's own docstring). When it does, THIS
        method runs the real, previously-deferred prefill for that
        request synchronously (via ``_run_deferred_prefill_for_grant``)
        before returning -- still inside this same single ``step()``
        call, so the prefill's own wire I/O (the batched-metaframe
        forward pass) and the control-message tick() that granted it
        never straddle two separate runner-loop iterations with other
        work interleaved in between.

        2026-08-06 bug #7 fix (third root cause -- see
        docs/batched-decode-n2-admission-handoff-2026-08-05.md's
        "2026-08-06 root-cause analysis" sections for the full
        hardware evidence): rank 1 unconditionally returning an empty
        list meant ``runner.py``'s OWN admission gate
        (``self.active_tasks``, capped by
        ``EXO_MAX_CONCURRENT_REQUESTS``) could NEVER learn a request
        had finished while batched-decode was active -- the gate
        monotonically filled up and never drained, so every request
        past the concurrency cap deferred FOREVER, confirmed directly
        on real N=2 hardware. Fixed by translating
        ``Rank1BatchedDecodeGlue.tick()``'s new ``evicted_request_id``
        return value into a real finish-classified
        ``GenerationBatch.Response`` -- this rank's own
        ``batch_generator.py`` wrapper already pops the matching uid
        out of its ``_active_tasks`` on any non-None ``finish_reason``
        (the SAME mechanism ``_step_pp_spec`` already relies on for
        rank 1's own admission-gate draining on that path), and
        ``runner.py``'s outer gate follows suit. The actual
        token/``finish_reason`` VALUE reported here is never observed
        by a client (rank 1 never emits chunks -- see this method's
        own docstring above); only the fact that SOME finish_reason is
        set matters, to trigger the existing drain path.
        """
        if self._batched_decode_rank1_glue is not None:
            grant, evicted_request_id, prefill_advance_completed = (
                self._batched_decode_rank1_glue.tick(self.model)
            )
            if grant is not None:
                self._run_deferred_prefill_for_grant(grant, is_rank1=True)
            if prefill_advance_completed is not None:
                self._advance_chunked_prefill_drive(
                    prefill_advance_completed.request_id, is_rank1=True
                )
            if evicted_request_id is not None:
                return [
                    GenerationBatch.Response(
                        uid=evicted_request_id,
                        token=0,
                        logprobs=mx.zeros(1),
                        finish_reason="stop",
                        prompt_cache=None,
                        all_tokens=None,
                    )
                ]
            return []

        assert self._batched_decode_rank0_glue is not None
        classified, _admitted_id, grant, prefill_advance_completed = (
            self._batched_decode_rank0_glue.tick(self.model)
        )
        if grant is not None:
            self._run_deferred_prefill_for_grant(grant, is_rank1=False)
            return []
        if prefill_advance_completed is not None:
            self._advance_chunked_prefill_drive(
                prefill_advance_completed.request_id, is_rank1=False
            )
            return []

        responses: list[GenerationBatch.Response] = []
        to_evict: list[int] = []
        for request_id, result in classified.items():
            finish_reason = result.finish_reason
            responses.append(
                GenerationBatch.Response(
                    uid=request_id,
                    token=result.token,
                    logprobs=mx.zeros(1),
                    finish_reason=finish_reason,
                    prompt_cache=None,
                    all_tokens=None,
                )
            )
            if finish_reason is not None:
                to_evict.append(request_id)

        # Eviction happens AFTER building this step's responses (never
        # while classified.items() is still being iterated) -- mirrors
        # step()'s own established "evict after the loop" discipline
        # for the degeneration kill-switch / stop-sequence paths below.
        for request_id in to_evict:
            self._batched_decode_rank0_glue.complete_request(request_id)

        return responses

    def step(self) -> list[tuple[int, GenerationResponse]]:
        # EXO_DECODE_PROBE: measure wall + GPU time per step() call (= per token
        # in single-stream mode). Aggregates over EXO_DECODE_PROBE_EVERY tokens
        # then logs to stderr. gpu_time_ns() is async — populated by Metal
        # completion handlers — so we read it at window boundaries instead of
        # per-iter where deltas read 0.
        _exo_probe = getattr(self, "_exo_probe_init", None)
        if _exo_probe is None:
            self._exo_probe_init = bool(os.environ.get("EXO_DECODE_PROBE"))
            self._exo_probe_every = int(os.environ.get("EXO_DECODE_PROBE_EVERY", "16"))
            self._exo_window_t0 = time.perf_counter()
            self._exo_window_g0 = mx.metal.gpu_time_ns()
            self._exo_cnt = 0
            _exo_probe = self._exo_probe_init
        if _exo_probe:
            self._exo_cnt += 1
            if self._exo_cnt % self._exo_probe_every == 0:
                _t = time.perf_counter()
                _g = mx.metal.gpu_time_ns()
                _per_wall = (_t - self._exo_window_t0) * 1000.0 / self._exo_probe_every
                _per_gpu = (_g - self._exo_window_g0) / 1e6 / self._exo_probe_every
                _pct = _per_gpu / _per_wall * 100 if _per_wall > 0 else 0.0
                import sys as _sys

                _sys.stderr.write(
                    f"[BG_DECODE_PROBE pid={os.getpid()}] step={self._exo_cnt} "
                    f"wall_ms={_per_wall:.2f} gpu_ms={_per_gpu:.2f} gpu_pct={_pct:.1f}\n"
                )
                _sys.stderr.flush()
                self._exo_window_t0 = _t
                self._exo_window_g0 = _g

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
            self._gpu_probe_mx_any_total = (
                getattr(self, "_gpu_probe_mx_any_total", 0) + _t_mx_any_ns
            )
            self._gpu_probe_mx_any_count = (
                getattr(self, "_gpu_probe_mx_any_count", 0) + 1
            )

        _trace = os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1")
        from exo.worker.engines.mlx.trace import request_trace

        # Use PP speculation decode if active. Captured as a local BEFORE
        # calling _step_pp_spec() (2026-07-19): that call now deterministically
        # pops the uid's entry from self._pp_spec_gen_by_uid on EOS/max_tokens
        # (see _close_pp_spec_gen), not just on the pre-existing "Clean up spec
        # state" branch further below in this same step() -- so re-reading
        # self._pp_spec_gen_by_uid after the call to decide which stats path
        # this response belongs to would silently misclassify the FINAL
        # response of every PP-spec completion. was_pp_spec_step is that
        # decision, frozen at the top of this call.
        was_pp_spec_step = bool(self._pp_spec_gen_by_uid)
        if self._batched_decode_active:
            _step_tic = time.perf_counter()
            responses = self._step_batched_decode()
            _next_elapsed = time.perf_counter() - _step_tic
            request_trace.record("decode.step.mlx_next", _step_tic)
        elif was_pp_spec_step:
            _step_tic = time.perf_counter()
            responses = self._step_pp_spec()
            _next_elapsed = time.perf_counter() - _step_tic
            request_trace.record("decode.step.mlx_next", _step_tic)
        else:
            self._mlx_gen._needs_topk = any(  # pyright: ignore[reportAttributeAccessIssue]
                t.task_params.logprobs for t in self._active_tasks.values()
            )
            _step_tic = time.perf_counter()
            # Bounded step (task #25). Call next() — exactly ONE _next() pass —
            # instead of next_generated(). next_generated()'s while-loop keeps
            # iterating INSIDE the generator whenever a pass produces prompt
            # work but no generation responses; during a mid-decode admission
            # that can hold step() open for many seconds, so the runner never
            # returns to its loop, no event (or heartbeat) is emitted, and the
            # supervisor hang watchdog SIGKILLs a healthy decoding runner.
            # Returning after every pass keeps the semantics identical (the
            # runner re-calls step() immediately; prompt-only passes yield [])
            # while making the runner-level liveness heartbeat reachable.
            _prompt_responses, responses = self._mlx_gen.next()
            _next_elapsed = time.perf_counter() - _step_tic
            request_trace.record("decode.step.mlx_next", _step_tic)

        results: list[tuple[int, GenerationResponse]] = []
        # uids the degeneration kill-switch terminated this step. mlx-lm's
        # BatchGenerator only drops a sequence when ITS OWN logic sets a
        # finish_reason (EOS / length); a finish_reason we INJECT (kill-switch
        # stop/error) is invisible to it, so it keeps emitting that uid every
        # step forever while we've already removed it from _active_tasks → the
        # "response uid N was not found - should be active" spam (observed
        # 48k+ times across a 2h run; present since the kill-switch shipped,
        # for BOTH stop and error actions). We must explicitly evict these
        # uids from the generator AFTER the response loop (can't mutate the
        # generator while iterating its responses). Covers BOTH injected-finish
        # cases — the degen kill-switch AND string-level stop-sequence matches —
        # since neither is visible to mlx-lm's native matcher, so it would keep
        # decoding the uid (a phantom stream) otherwise.
        _evict_from_generator_uids: list[int] = []

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
                # Guard against out-of-range token ids before handing them to the
                # streaming detokenizer. DSv4-Flash's lm_head emits vocab_size
                # (129280) logits but the tokenizer's BPE tokenmap only covers
                # len(vocab) (128000) ids; a sampled special/reserved id in the
                # gap — or a negative sentinel that can leak from the MTP accept
                # path under temp>0 sampling — indexes the tokenmap out of range
                # and crashes the runner (tokenizer_utils.py:208,
                # IndexError: list index out of range). A token with no tokenmap
                # entry has no text to stream, so skip detok for it rather than
                # die; the token id is still recorded in completion_tokens below.
                _tok = response.token
                _tokmap = getattr(state.detokenizer, "tokenmap", None)
                _tok_ok = (
                    isinstance(_tok, int)
                    and _tok >= 0
                    and (_tokmap is None or _tok < len(_tokmap))
                )
                if _tok_ok:
                    state.detokenizer.add_token(response.token)
                else:
                    logger.warning(
                        f"skipping detok of out-of-range token id {_tok!r} "
                        f"(tokenmap len={len(_tokmap) if _tokmap is not None else 'n/a'}) "
                        f"for uid {response.uid}"
                    )
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            state.generated_text_parts.append(text)
            state.potential_stop_sequence_text += text

            # ── degeneration (repetition-loop) detection ──
            # Detects decode collapse (a short token cycle repeating forever).
            # With EXO_LOOP_DETECT_ACTION="error" (default) we FAIL the turn
            # cleanly (retryable); "stop" terminates but surfaces the partial;
            # "warn" keeps the legacy log-only behavior.
            degeneration_terminate = False
            if _LOOP_DETECT_ENABLED and response.finish_reason != "stop":
                ids = state.recent_token_ids
                ids.append(int(response.token))
                if len(ids) > _LOOP_DETECT_WINDOW:
                    del ids[: len(ids) - _LOOP_DETECT_WINDOW]
                # Once we've decided to terminate, no need to keep scanning.
                # Otherwise scan every token (not just until first warn) so the
                # termination fires the instant the cycle crosses threshold.
                loop = (
                    _detect_token_loop(ids) if not state.degeneration_warned else None
                )
                if loop is not None:
                    period, repeats = loop
                    state.degeneration_warned = True
                    degeneration_terminate = _LOOP_DETECT_ACTION in ("stop", "error")
                    cycle_ids = ids[len(ids) - period :]
                    try:
                        cycle_text = self.tokenizer.decode(cycle_ids)
                    except Exception:
                        cycle_text = "<decode-failed>"
                    tp = state.task_params
                    # ── DEGEN PROBE: correlate this collapse with the most
                    # recent MTP BS-transition cache swap for THIS uid. The
                    # hypothesis under test: degeneration onset clusters
                    # tightly AFTER a per-stream cache swap (small
                    # ms_since_transition) ⇒ the swap corrupts the shared
                    # cache the target verify-forward reads. Lazy import +
                    # only on the (rare) degeneration path = zero hot cost.
                    _degen_transition = "probe-off"
                    try:
                        from exo.worker.engines.mlx.speculative.dsv4_mtp import (
                            _DEGEN_LAST_TRANSITION,
                            _DEGEN_PROBE_ENABLED,
                            _degen_probe_write,
                        )

                        if _DEGEN_PROBE_ENABLED:
                            import time as _t

                            _stamp = _DEGEN_LAST_TRANSITION.get(int(response.uid))
                            if _stamp is not None:
                                _ms = (_t.perf_counter_ns() - _stamp["wall_ns"]) / 1e6
                                _degen_transition = (
                                    f"ms_since_swap={_ms:.1f} "
                                    f"last_swap_bs_gt1={_stamp['bs_gt1']} "
                                    f"swap_uids={_stamp['to_uids']}"
                                )
                            else:
                                _degen_transition = "no_swap_seen_this_uid"
                            _degen_probe_write(
                                {
                                    "event": "degeneration",
                                    "uid": int(response.uid),
                                    "completion_token": int(state.completion_tokens),
                                    "period": int(period),
                                    "repeats": int(repeats),
                                    "cycle_text": cycle_text,
                                    "in_thinking": bool(state.in_thinking),
                                    "prompt_tokens": int(state.all_prompt_tokens.size),
                                    "prefix_hit": int(state.prefix_hit_length),
                                    "last_transition": _stamp,
                                    "wall_ns": _t.perf_counter_ns(),
                                }
                            )
                    except Exception:
                        _degen_transition = "probe-err"
                    # Pre-format into one string: the runner's logger is
                    # loguru ({}-style), so %s args are NOT interpolated.
                    # f-string keeps this logger-agnostic.
                    logger.warning(
                        f"DEGENERATION DETECTED uid={response.uid} "
                        f"at completion_token={state.completion_tokens}: "
                        f"token cycle period={period} repeated>={repeats}x. "
                        f"action={_LOOP_DETECT_ACTION} "
                        f"cycle_token_ids={cycle_ids} cycle_text={cycle_text!r} "
                        f"in_thinking={state.in_thinking} | sampling: "
                        f"temp={tp.temperature} top_p={tp.top_p} "
                        f"top_k={tp.top_k} min_p={tp.min_p} "
                        f"rep_pen={tp.repetition_penalty} "
                        f"prompt_tokens~{int(state.all_prompt_tokens.size)} "
                        f"prefix_hit={state.prefix_hit_length} "
                        f"gen_engine={type(self._mlx_gen).__name__} "
                        f"| degen_probe: {_degen_transition}"
                    )

            # ── long-period (multi-sentence) degeneration detection ──
            # See the "Long-period (multi-sentence) degeneration detection"
            # comment block near _LOOP_DETECT_LONG_ENABLED for the full
            # rationale. Runs independently of the tight-loop detector above
            # (different window, different granularity) and can terminate on
            # its own even if the tight-loop detector never fires — this is
            # exactly the case it exists for (long word-for-word-identical
            # sentence cycles the tight detector's max_period can't reach).
            if (
                not degeneration_terminate
                and _LOOP_DETECT_LONG_ENABLED
                and response.finish_reason != "stop"
                and not state.long_loop_degeneration_warned
            ):
                pending = state.long_loop_pending_tokens
                pending.append(int(response.token))
                if len(pending) >= _LOOP_DETECT_LONG_BLOCK:
                    block_tokens = pending[:_LOOP_DETECT_LONG_BLOCK]
                    state.long_loop_pending_tokens = pending[_LOOP_DETECT_LONG_BLOCK:]
                    block_hash = hash(tuple(block_tokens))
                    hashes = state.long_loop_block_hashes
                    blocks = state.long_loop_block_tokens
                    hashes.append(block_hash)
                    blocks.append(block_tokens)
                    if len(hashes) > _LOOP_DETECT_LONG_WINDOW_BLOCKS:
                        overflow = len(hashes) - _LOOP_DETECT_LONG_WINDOW_BLOCKS
                        del hashes[:overflow]
                        del blocks[:overflow]

                    long_loop = _detect_long_period_loop(hashes)
                    if long_loop is not None:
                        long_period, long_repeats = long_loop
                        # Re-verify the raw token blocks match exactly before
                        # trusting a hash match — cheap here since this only
                        # runs on the rare detection path (see
                        # _detect_long_period_loop's docstring).
                        tail_blocks = blocks[len(blocks) - long_period :]
                        verify_pos = len(blocks) - long_period
                        verified_repeats = 1
                        while (
                            verify_pos - long_period >= 0
                            and blocks[verify_pos - long_period : verify_pos]
                            == tail_blocks
                        ):
                            verified_repeats += 1
                            verify_pos -= long_period
                        if verified_repeats >= _LOOP_DETECT_LONG_MIN_REPEATS:
                            state.long_loop_degeneration_warned = True
                            degeneration_terminate = _LOOP_DETECT_ACTION in (
                                "stop",
                                "error",
                            )
                            cycle_tokens = [t for blk in tail_blocks for t in blk]
                            try:
                                cycle_text = self.tokenizer.decode(cycle_tokens)
                            except Exception:
                                cycle_text = "<decode-failed>"
                            tp = state.task_params
                            logger.warning(
                                f"LONG-PERIOD DEGENERATION DETECTED "
                                f"uid={response.uid} "
                                f"at completion_token={state.completion_tokens}: "
                                f"block cycle period={long_period} "
                                f"({long_period * _LOOP_DETECT_LONG_BLOCK} raw "
                                f"tokens) repeated>={verified_repeats}x. "
                                f"action={_LOOP_DETECT_ACTION} "
                                f"cycle_text={cycle_text!r} "
                                f"in_thinking={state.in_thinking} | sampling: "
                                f"temp={tp.temperature} top_p={tp.top_p} "
                                f"top_k={tp.top_k} min_p={tp.min_p} "
                                f"rep_pen={tp.repetition_penalty} "
                                f"prompt_tokens~{int(state.all_prompt_tokens.size)} "
                                f"prefix_hit={state.prefix_hit_length} "
                                f"gen_engine={type(self._mlx_gen).__name__}"
                            )

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
                        # This "stop" is INJECTED by our string-level matcher —
                        # mlx-lm's native token matcher never saw it, so it keeps
                        # the uid in its batch and decodes it to max_tokens
                        # (phantom stream: "response uid N was not found" spam, an
                        # occupied slot that inflates B and disarms the B==1-gated
                        # fence/spec). Evict it after the loop, same as the degen
                        # kill-switch below. Rank-safe: this loop runs identically
                        # on all ranks (verified — both ranks emit byte-identical
                        # step responses), so the eviction is symmetric.
                        _evict_from_generator_uids.append(response.uid)
                        break
            # Degeneration kill-switch: a detected repetition loop terminates the
            # generation here. This is the deterministic guarantee against an
            # infinite/runaway loop that a sampling penalty alone cannot provide.
            #   action="error" (default): fail the turn cleanly — the degenerate
            #     partial text is REPLACED with a diagnostic message and
            #     finish_reason="error" (-> ErrorChunk -> 500 -> hermes retries),
            #     so the pre-collapse garbage never reaches display.
            #   action="stop": terminate but surface the partial (legacy) — the
            #     current token's text is emitted and finish_reason="stop".
            if degeneration_terminate:
                if _LOOP_DETECT_ACTION == "error":
                    text = _DEGENERATION_ERROR_TEXT
                    finish_reason = "error"
                else:
                    finish_reason = "stop"
                # Evict from the MLX generator after the loop — our injected
                # finish_reason won't make mlx-lm drop the sequence on its own.
                _evict_from_generator_uids.append(response.uid)
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
                if was_pp_spec_step:
                    # was_pp_spec_step (captured at the top of step(), before
                    # _step_pp_spec() ran) replaces the old
                    # "self._pp_spec_gen is not None or self._pp_spec_uid is
                    # not None" check here (2026-07-19) -- that check now
                    # reads stale/cleared state, since _step_pp_spec's EOS and
                    # StopIteration paths already deterministically pop the
                    # uid's entry from self._pp_spec_gen_by_uid via
                    # _close_pp_spec_gen() before we get here.
                    gen_elapsed = time.perf_counter() - state.generation_start_time
                    generation_tps = (
                        state.completion_tokens / gen_elapsed
                        if gen_elapsed > 0
                        else 0.0
                    )
                    # Spec state is already cleared by _close_pp_spec_gen()
                    # inside _step_pp_spec() -- nothing left to do here.
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
                mtp_cycles_cum = int(getattr(self._mlx_gen, "_spec_cycles", 0) or 0)
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
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1024**3),
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
                self._update_fence_arming()
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
        _post_total = (
            _t_callback_total
            + _t_detok_total
            + _t_stop_total
            + _t_logprobs_total
            + _t_response_build_total
        )
        request_trace.record(
            "decode.step.post_process", _step_tic + _next_elapsed, _step_end
        )

        # _next_count was added by the old fast_next patch; on vanilla
        # BatchGenerator (new mlx-lm) it doesn't exist — fall back to our own
        # cumulative step counter for logging cadence.
        _mlx_next_count = getattr(self._mlx_gen, "_next_count", None)
        if _mlx_next_count is None:
            _mlx_next_count = getattr(self, "_step_counter", 0) + 1
            self._step_counter = _mlx_next_count  # pyright: ignore[reportAttributeAccessIssue]
        if _mlx_next_count % 64 == 0 and responses:
            _gpu_probe_local = bool(os.environ.get("MLX_GPU_TIME"))
            _mxa_total = getattr(self, "_gpu_probe_mx_any_total", 0)
            _mxa_count = getattr(self, "_gpu_probe_mx_any_count", 0)
            _mxa_avg_ms = (
                (_mxa_total / _mxa_count / 1e6)
                if (_gpu_probe_local and _mxa_count > 0)
                else 0.0
            )
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

        if _GC_COLLECT_INTERVAL > 0 and _mlx_next_count % _GC_COLLECT_INTERVAL == 0:
            gc.collect()

        if (
            _MALLOC_RELIEF_INTERVAL > 0
            and _malloc_zone_pressure_relief is not None
            and _mlx_next_count % _MALLOC_RELIEF_INTERVAL == 0
        ):
            try:
                released = _malloc_zone_pressure_relief(None, 0)
                if (
                    released > 0
                    and _mlx_next_count % (_MALLOC_RELIEF_INTERVAL * 10) == 0
                ):
                    logger.info(
                        f"[mem] malloc pressure_relief released {released / (1024**2):.1f} MB"
                    )
            except Exception:
                pass

        # Evict any injected-finish sequences (degen kill-switch OR string-level
        # stop-sequence match) from the MLX generator so it stops emitting them
        # next step — the injected finish_reason alone does not make mlx-lm drop
        # the uid. Done here, after iterating this step's responses, to avoid
        # mutating the generator mid-iteration. Rank-safe: this list is populated
        # inside the response loop which runs identically on every rank, so the
        # remove() is symmetric. Best-effort: removal failure must not break the
        # decode loop.
        # Wedge INJECTOR (EXO_DSV4_WEDGE_INJECT=<step>): deterministically
        # reproduce the c>=2 degen-kill B-transition WITHOUT waiting for a random
        # degen. At the given step, if the generator batch is B>=2, force-evict
        # the highest active uid — SYMMETRICALLY on every rank (the step counter
        # advances in lockstep, verified: 2103==2103). If a provably-symmetric
        # eviction still wedges, the cause is the BS-transition mechanics (async
        # graph vs cache rebuild), not eviction asymmetry. Diagnostic only.
        _wt_inject = os.environ.get("EXO_DSV4_WEDGE_INJECT")
        if _wt_inject:
            _inj_step = getattr(self, "_wedge_inject_step", 0) + 1
            self._wedge_inject_step = _inj_step
            _inj_gb = getattr(self._mlx_gen, "_generation_batch", None)
            _inj_b = len(_inj_gb) if _inj_gb is not None else -1
            if _inj_b >= 2 and _inj_step == int(_wt_inject) and self._active_tasks:
                _inj_uid = max(self._active_tasks.keys())
                if _inj_uid not in _evict_from_generator_uids:
                    _evict_from_generator_uids.append(_inj_uid)
                    logger.info(
                        f"[WEDGE_INJECT] rank="
                        f"{self.group.rank() if self.group is not None else 0} "
                        f"step={_inj_step} B={_inj_b} force-evict uid={_inj_uid}"
                    )

        # Wedge tracer (EXO_DSV4_WEDGE_TRACE=1): the c>=2 degen-kill wedge is a
        # rank desync on a mid-batch eviction. Log, per rank, the batch size the
        # generator sees each step and every eviction, so a cross-rank diff shows
        # whether the eviction (and the resulting B-transition) is symmetric.
        if os.environ.get("EXO_DSV4_WEDGE_TRACE") == "1":
            _wt_rank = self.group.rank() if self.group is not None else 0
            _wt_step = getattr(self, "_wedge_trace_step", 0) + 1
            self._wedge_trace_step = _wt_step
            _wt_gb = getattr(self._mlx_gen, "_generation_batch", None)
            _wt_b = len(_wt_gb) if _wt_gb is not None else -1
            if _wt_b >= 2 or _evict_from_generator_uids:
                logger.info(
                    f"[WEDGE_TRACE] rank={_wt_rank} step={_wt_step} B={_wt_b} "
                    f"active={sorted(self._active_tasks.keys())} "
                    f"evict={sorted(_evict_from_generator_uids)}"
                )

        if _evict_from_generator_uids:
            try:
                self._mlx_gen.remove(_evict_from_generator_uids)
            except Exception as _evict_err:
                logger.warning(
                    f"injected-finish generator-evict failed for "
                    f"{_evict_from_generator_uids}: {_evict_err}"
                )
            if os.environ.get("EXO_DSV4_WEDGE_TRACE") == "1":
                _wt_gb2 = getattr(self._mlx_gen, "_generation_batch", None)
                _wt_b2 = len(_wt_gb2) if _wt_gb2 is not None else -1
                _wt_rank2 = self.group.rank() if self.group is not None else 0
                logger.info(
                    f"[WEDGE_TRACE] rank={_wt_rank2} POST-EVICT B={_wt_b2} "
                    f"removed={sorted(_evict_from_generator_uids)}"
                )

        return results

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)
        self._update_fence_arming()

    def reset_after_reconnect(self) -> list[int]:
        """Drop ALL in-flight sequences after an in-place jaccl reconnect.

        The wedged collective aborted mid-batch, so every active sequence's
        partial forward/KV state is discarded (the requests are failed and the
        clients retry). The model weights stay resident — only per-request state
        is cleared. Returns the uids that were dropped.
        """
        uids = list(self._active_tasks.keys())
        if uids:
            try:
                self._mlx_gen.remove(uids)
            except Exception as e:
                logger.warning(f"reset_after_reconnect: generator remove failed: {e!r}")
            for uid in uids:
                self._active_tasks.pop(uid, None)
            self._update_fence_arming()
        return uids

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
        low_priority: bool = False,
        high_priority: bool = False,
    ) -> None:
        if self.kv_prefix_cache is None:
            return

        if os.environ.get("EXO_PREFIX_CACHE_DIAG") == "1":
            _diag_rank = self.group.rank() if self.group is not None else 0
            logger.info(
                f"[PREFIX_DIAG rank={_diag_rank}] _save_prefix_cache ENTRY "
                f"cache_snapshots={'None' if cache_snapshots is None else len(cache_snapshots)} "
                f"prefix_hit_length={prefix_hit_length} matched_index={matched_index}"
            )

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
                    low_priority=low_priority,
                    high_priority=high_priority,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                    low_priority=low_priority,
                    high_priority=high_priority,
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)

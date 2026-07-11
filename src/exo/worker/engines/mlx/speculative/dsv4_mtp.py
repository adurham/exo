"""DSv4-specific MTP self-speculative decode glue.

Sits alongside the Qwen3.5-flavored ``mtp_module.py`` /
``mtp_batch_generator.py`` and reuses their machinery (the
``draft_tokens`` chain helper, the BatchGenerator scaffolding,
the per-uid pre-norm capture). The DSv4-specific pieces are:

* :class:`DSv4MTPPredictor` — thin wrapper around the model's
  pre-loaded ``model.model.mtp[mtp_idx]`` module. Unlike Qwen3.5's
  :class:`~.mtp_module.MTPPredictor`, weights are NOT loaded
  dynamically here; they live on the model already (via the
  modified ``deepseek_v4.sanitize`` and ``DeepseekV4Model.mtp``
  ModuleList). The predictor only adapts the call signature to
  the one ``draft_tokens`` expects.
* :func:`dsv4_speculative_forward` — verify-pass forward through
  the target. DSv4 uses standard KV caches (RotatingKVCache +
  PoolingCache wrapped in CacheList) with rollback via ``trim()``,
  so no GDN bookkeeping is needed; this is a thin wrapper over
  the model's ``__call__`` that hands back the captured pre-final-
  norm hidden via the ``MTPBatchGenerator``'s wrapped-norm side
  channel.
* :class:`DSv4MTPBatchGenerator` — overrides ``_speculative_next``
  to use the DSv4 helpers. Inherits per-uid pre-norm capture,
  token buffering, and BS=1 dispatch logic from the base
  :class:`~.mtp_batch_generator.MTPBatchGenerator`.

BS>1 batched-MTP is NOT yet enabled here — that's Phase 5.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, BinaryIO, Optional, Sequence, cast

import mlx.core as mx

from mlx_lm.models.cache import (
    BatchPoolingCache,
    BatchRotatingKVCache,
    CacheList,
    PerStreamBatchRotatingKVCache,
    PoolingCache,
)

from .mtp_batch_generator import MTPBatchGenerator
from . import mtp_module as _mtp_module  # for _TREE_ALPHA_PROBE_STEPS drain
from .mtp_module import broadcast_from_canonical, draft_tokens, draft_tokens_topk
from exo.worker.engines.mlx.utils_mlx import get_coord_group, mx_any


# Token-tree alpha probe writer state. Rank-0 only writes the JSONL so we
# don't fight two ranks for one file. File path is parametric to PID so
# multiple worker processes on the same host don't collide. Zero-cost when
# EXO_DSV4_TREE_ALPHA_PROBE is unset.
_TREE_ALPHA_PROBE_HANDLE: Optional[BinaryIO] = None
_TREE_ALPHA_PROBE_RECS: int = 0


def _tree_alpha_probe_write(record: dict) -> None:
    """Append one record to the rank-0 probe JSONL file."""
    global _TREE_ALPHA_PROBE_HANDLE, _TREE_ALPHA_PROBE_RECS
    if _TREE_ALPHA_PROBE_HANDLE is None:
        path = f"/tmp/dsv4_tree_alpha_probe_pid{os.getpid()}.jsonl"
        _TREE_ALPHA_PROBE_HANDLE = open(path, "ab", buffering=0)  # noqa: SIM115
    import json as _json
    line = (_json.dumps(record) + "\n").encode("utf-8")
    _TREE_ALPHA_PROBE_HANDLE.write(line)
    _TREE_ALPHA_PROBE_RECS += 1


def _upgrade_cache_to_per_stream(cache_obj: Any) -> None:
    """In-place class swap: BatchRotatingKVCache → PerStreamBatchRotatingKVCache.

    Walks the cache structure (handling CacheList nesting) and swaps
    the class pointer on every BatchRotatingKVCache it finds. The
    instance state is preserved verbatim — the per-stream subclass
    only adds methods (``trim_per_stream``) and overrides
    ``_update_in_place`` / ``make_mask``; no new attributes are
    needed at construction time.
    """
    if isinstance(cache_obj, CacheList):
        for sub in cache_obj.caches:
            _upgrade_cache_to_per_stream(sub)
        return
    if isinstance(cache_obj, BatchRotatingKVCache) and not isinstance(
        cache_obj, PerStreamBatchRotatingKVCache
    ):
        # Before swapping the class pointer, normalize the base ring buffer
        # into temporal (un-rotated) order so the per-stream subclass — which
        # treats the physical buffer as a contiguous ring indexed by logical
        # offset mod max_size — starts from a known layout. Without this, a
        # buffer that the base class had rotated (write head wrapped after a
        # long prefill) is reinterpreted by the per-stream code with the
        # wrong slot↔position mapping, scrambling 100K of context →
        # garbage/BOS-spam at c>=2 long context. _temporal_order is a no-op
        # when the buffer never rotated (short context), so this is safe and
        # cheap in the common case.
        if getattr(cache_obj, "rotated", False):
            cache_obj._temporal_order()
        cache_obj.__class__ = PerStreamBatchRotatingKVCache
        # Bootstrap the per-stream ring bookkeeping from the base state so
        # the first update_and_fetch / make_mask after the swap reads
        # consistent values (rather than the lazy-fallback that mistook the
        # physical buffer index for the logical offset).
        cache_obj._bootstrap_per_stream_ring()

logger = logging.getLogger(__name__)

# Log acceptance distribution every N spec cycles when EXO_DSV4_MTP_LOG=1.
# Set to 0 / unset to silence.
_LOG_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_LOG_INTERVAL", "0"))


# Token-tree drafting feature gate. When 1, _speculative_next routes to
# the K=2 gamma=2 tree path instead of the linear gamma=2 chain.
# K is set by EXO_DSV4_TREE_K (default 2). gamma comes from the standard
# EXO_SPECULATIVE_GAMMA (must be <=2; v1 supports gamma in {1,2}).
TREE_DRAFT = os.environ.get("EXO_DSV4_TREE_DRAFT") == "1"
TREE_K = int(os.environ.get("EXO_DSV4_TREE_K", "2"))


def _build_tree_mask_and_positions(
    parent_idx: list[int],
    depth: list[int],
    mask_cache: Any,
    dtype: Any = None,
) -> tuple[Any, Any]:
    """Build the per-tree-node attention mask and RoPE positions.

    Args:
        parent_idx: list[int] length n_nodes. parent_idx[i] is the index
            of node i's parent (-1 for the root).
        depth: list[int] length n_nodes. depth[i] is the depth of node i
            (0 = root, 1 = depth-1, etc).
        mask_cache: the RotatingKVCache used for mask construction (= the
            first child of the first layer's prompt_cache CacheList). Has
            `.offset` and `.max_size` (= sliding_window). We let it build
            the base causal mask and then OVERWRITE the (L_q, L_q) tail
            block with our tree-ancestor pattern.
        dtype: mx dtype for the mask additive (-inf, 0). Defaults to bf16.

    Returns: (mask, positions)
        mask: shape (L_q, kv_window + L_q) additive: 0.0 at attend, large
            negative value at do-not-attend. `kv_window` = whatever the
            cache's own mask machinery clamped to (typically sliding_window
            for RotatingKVCache; full offset only for prefill stages).
        positions: shape (L_q,) int32 -- RoPE positions. Node at depth d
            gets position offset + d (using the REAL un-clamped offset --
            RoPE is unaffected by sliding window).

    The kv_window-clamping is critical: the DSv4 SparseCompressedAttention
    local-attention branch reads only `local_kv.shape[2] = min(offset, sw)
    + L_q` KV positions. If the mask is wider than that, the SDPA scores
    @ mask broadcast fails (we crashed 2026-05-19 with
    Shapes (1,1,7,69328) and (1,64,7,134) cannot be broadcast).
    """
    n_nodes = len(parent_idx)
    if dtype is None:
        dtype = mx.bfloat16

    # Real (un-clamped) cache offset for RoPE positions.
    real_offset = int(mask_cache.offset)

    # Let the cache build the base causal mask. It will clamp the kv-axis
    # to sliding_window when offset > sliding_window. Boolean array of
    # shape (L_q, kv_window + L_q) where True = attend, False = mask out.
    base = mask_cache.make_mask(
        n_nodes,
        window_size=getattr(mask_cache, "max_size", None),
        return_array=True,
    )
    if base is None:
        # n_nodes==1 fast path -- shouldn't happen for tree verify, but
        # cover the edge case by emitting an all-attend mask.
        full_w = real_offset + n_nodes
        base = mx.ones((n_nodes, full_w), dtype=mx.bool_)

    # Normalise mask rank: RotatingKVCache.make_mask returns a 2D (L_q, kv)
    # mask, BUT BatchRotatingKVCache emits a 4D (B, n_heads, L_q, kv)
    # shape via mx.expand_dims for left_padding handling. Squeeze any
    # leading singleton axes so we work in a 2D space; we'll re-broadcast
    # to 4D when the side channel handles the mask in DeepseekV4Model.
    while base.ndim > 2:
        if base.shape[0] != 1:
            raise RuntimeError(
                f"_build_tree_mask_and_positions: unexpected mask shape "
                f"{base.shape}; expected leading dims to be 1."
            )
        base = base[0]

    # Convert boolean -> additive (0 attend, -1e9 mask).
    NEG = -1.0e9
    mask_add = mx.where(
        base,
        mx.array(0.0, dtype=dtype),
        mx.array(NEG, dtype=dtype),
    )  # (L_q, kv_window + L_q)

    # Overwrite the (L_q, L_q) tail block with our tree-ancestor sub-mask.
    # Ancestor sets: walk each node's parent chain.
    ancestors: list[set[int]] = [set() for _ in range(n_nodes)]
    for i in range(n_nodes):
        cur = parent_idx[i]
        while cur != -1:
            ancestors[i].add(cur)
            cur = parent_idx[cur]

    sub_mask = [[NEG] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        sub_mask[i][i] = 0.0
        for a in ancestors[i]:
            sub_mask[i][a] = 0.0
    sub_arr = mx.array(sub_mask, dtype=dtype)             # (L_q, L_q)

    # Splice: leftmost kv_window cols from base (the prefill attend pattern),
    # rightmost L_q cols replaced by tree sub-mask.
    kv_window = mask_add.shape[1] - n_nodes
    left = mask_add[:, :kv_window]
    mask = mx.concatenate([left, sub_arr], axis=1)         # (L_q, kv_window + L_q)

    # mlx-lm RotatingKVCache.make_mask can return a sequence dimension > max_size
    # when offset < max_size but offset + L_q > max_size. Force clamp the
    # sequence length (axis 1) so it matches the physical KV cache size
    # yielded by update_and_fetch, preventing SDPA broadcast crashes.
    max_s = getattr(mask_cache, "max_size", None)
    if max_s is not None and mask.shape[1] > max_s:
        mask = mask[:, -max_s:]


    positions = mx.array(
        [real_offset + d for d in depth], dtype=mx.int32
    )
    return mask, positions

# Per-cycle phase timing. When EXO_DSV4_MTP_PROFILE > 0, brackets the
# draft / verify / accept phases with mx.eval + perf_counter, summarising
# every N cycles. Inserts evals at phase boundaries which serialises
# pipelining — measurements are upper bounds on real production walls.
_PROFILE_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_PROFILE", "0"))

# BS>1 MIN-ACCEPTANCE (2026-07-02, default ON). The per-stream acceptance
# migration let per-stream rotating-KV offsets diverge from the
# batch-UNIFORM pooling/indexer caches (uniform trim + whole-batch
# restore_meta): after the first cross-stream acceptance divergence the
# pool consumes the wrong ring rows for one stream and the sparse indexer
# retrieves shifted blocks — the deterministic BS=2 degeneration loop
# (2026-06-17 note at the residual-sampling branch; isolated 2026-07-02:
# MTP-off c=2 clean, greedy c=2 corrupt). Clamping every stream to
# n_min = min(n_accepted) keeps ALL caches in lockstep by construction —
# the class docstring's original, 2026-05-22-validated strategy. Streams
# that accepted beyond n_min stage their accepted draft at position n_min
# as the next-cycle token, so no valid work is discarded beyond the
# documented min-strategy cost. EXO_DSV4_BS_MIN_ACCEPT=0 reverts to
# per-stream acceptance (known-corrupt at c=2) for A/B.
_BS_MIN_ACCEPT = os.environ.get("EXO_DSV4_BS_MIN_ACCEPT", "1") != "0"

# GREEDY ACCEPT-RULE ALIGNMENT (2026-07-10, default OFF until the
# byte-equality gate passes). At temp=0 the plain (MTP-off) generator picks
# argmax over LOGSUMEXP-NORMALIZED logprobs in native bf16 (mlx-lm
# generate.py GenerationBatch._step — the batched path has NO fp32 cast),
# while this file's accept/bonus picked argmax over RAW verify_logits.
# Round-to-nearest is monotone, so the bf16 subtraction can collapse two
# near-tied logits onto the SAME value; first-index argmax then picks the
# lower id — a DIFFERENT token than raw argmax, with bitwise-identical
# logits. Attribution harness (~/scratch/ulp_gen_vs_model.py, m4-1
# 2026-07-10): every construction factor (stream ctx, input dtype/laziness,
# pre_norm hook, full genstep replica) is bitwise EQUAL; the ONLY
# divergence between the generator step and a raw model() call is this
# decision rule (E3: first trajectory split at step 30 with logits
# bitwise-equal; E2: 4/64 raw-vs-normalized argmax flips). With
# EXO_DSV4_MTP_ACCEPT_LOGPROBS=1 the greedy accept/bonus argmaxes are taken
# over the SAME normalized logprobs the generator samples from, making
# MTP-on token selection rule-identical to MTP-off. Supersedes the
# tie-break fix (EXO_DSV4_MTP_TIEBREAK_FIX, already OFF in prod): that
# masked ties by picking lowest-id-within-eps; this removes the mismatch.
_ACCEPT_LOGPROBS = os.environ.get("EXO_DSV4_MTP_ACCEPT_LOGPROBS", "0") == "1"

# REGIME-B DOUBLE-ROLLBACK FIX (2026-07-10, default OFF until the
# byte-equality gate + battery pass). In the pool-flush rollback path
# (regime b), the code restore_meta()s every SNAPSHOTTED pool to its
# pre-verify state and THEN runs the blanket trim(γ+1) over all caches —
# but CacheList.trim recurses into those same pools, subtracting up to γ+1
# from the remainder that restore_meta just rewound. The pool's remainder
# buffer rows then misalign (the commit-forward overwrites the wrong row),
# so every flush-straddling rejection corrupts the compressed-context pool
# by a row — surfacing as ulp-scale logit drift vs sequential decode that
# accumulates over cycles (the residual MTP-on vs MTP-off byte-inequality
# after the accept-rule fix; localized by ~/scratch/ldiff_cycles.py:
# accept-only chains BITWISE CLEAN, reject cycles DRIFT from the first
# snapshotted rollback; trim-first-then-restore = BITWISE CLEAN across
# 36/24 regime-b cycles). With EXO_DSV4_POOL_RESTORE_AFTER_TRIM=1 the
# blanket trim runs FIRST (rewinding ring KV and the NON-snapshotted
# pools, which do need it) and restore_meta runs AFTER, so the restored
# state is what the commit-forward actually sees. Applies to the B=1 and
# batch rollback paths; the tree path freezes pools during verify and has
# no restore_meta.
_POOL_RESTORE_AFTER_TRIM = (
    os.environ.get("EXO_DSV4_POOL_RESTORE_AFTER_TRIM", "0") == "1"
)

# Per-request cycle statistics (EXO_DSV4_MTP_CYCLE_STATS=1): one WARNING
# line per finished stream with cycle/rejection/regime-b counts and the
# committed-token index of the first rejection and first regime-b repair —
# lets a serving-level trajectory fork be correlated with cycle events
# without any unsound trim+refeed instrumentation. B=1 path only.
_CYCLE_STATS = os.environ.get("EXO_DSV4_MTP_CYCLE_STATS", "0") == "1"

# BATCH-POOL SNAPSHOT FIX (2026-07-10, default OFF until gates pass). The
# batched generator converts every PoolingCache to BatchPoolingCache at
# insert (mlx-lm generate._make_cache / _merge_caches) — and
# BatchPoolingCache subclasses _BaseCache, NOT PoolingCache. So
# _collect_pooling_caches's isinstance(sub, PoolingCache) has returned []
# in serving at EVERY concurrency since that hierarchy existed: no pool is
# ever snapshotted, pool_flushed is always False, and regime b (the
# 2026-05-29 pool-contamination repair) is INERT — every flush-straddling
# rejection bakes rejected-draft rows into the compressed pools. Confirmed
# live 2026-07-10 via EXO_DSV4_MTP_CYCLE_STATS: 78 cycles, 58 rejections,
# regime_b=0. This is the residual serving-only ulp drift (onset at the
# first cycles) behind the MTP-on vs MTP-off byte-inequality.
# EXO_DSV4_POOL_SNAPSHOT_BATCH=1 collects BatchPoolingCache too and uses
# class-appropriate snapshot predicates and flush detection (per-stream
# lengths + pending bumps — comparing _pool_lengths alone false-positives
# whenever the verify's commit_pending applies a PRIOR staged bump).
# Pair with EXO_DSV4_POOL_RESTORE_AFTER_TRIM=1: activating snapshots with
# the legacy restore-then-trim order would re-introduce the double-rollback.
_POOL_SNAPSHOT_BATCH = (
    os.environ.get("EXO_DSV4_POOL_SNAPSHOT_BATCH", "0") == "1"
)

# UNIFIED SPEC-STATE ROLLBACK (2026-07-10, default OFF until gates pass).
# trim() is NOT rollback-safe on a ROTATED ring: the verify's draft writes
# physically overwrite the ring's oldest rows (destroying historical KV that
# sequential decode still has), decrement left_padding, and can wrap _idx —
# none of which trim() undoes. ldiff_cycles.py: reject cycles are bitwise
# clean at short unwrapped ctx but drift the moment the ring rotates
# (P=41 drifts exactly at abs 127; P=4096 immediately; up to 0.6 logits
# with junk drafts). With EXO_DSV4_SPEC_STATE_RESTORE=1 the B=1 path takes
# an O(1) reference snapshot of every ring (slice-assign rebinds
# keys/values, so pre-verify refs preserve pre-verify contents) plus a
# save_meta of every pool BEFORE verify; on ANY rejection it restores all
# of them wholesale and re-commits [y]+accepted with one small forward
# (bitwise = sequential under rowseq+rowmask). This supersedes the
# regime-a/-b distinction entirely in this mode. Cost: ring-donation
# blocking during verify + pool buf copies per cycle + one commit-forward
# per rejection — the price of exactness; default path unchanged.
_SPEC_STATE_RESTORE = (
    os.environ.get("EXO_DSV4_SPEC_STATE_RESTORE", "0") == "1"
)

# EXO_DSV4_SPEC_CACHE_ROLLBACK=1 (requires _SPEC_STATE_RESTORE): replace the
# per-rejection commit-forward with cache-level exact undo. The verify's
# cache writes for COMMITTED rows are already bitwise-sequential (rowseq +
# rowmask), so re-running the whole model on [y]+accepted only regenerates
# rows the verify computed — at prod reject rates that extra forward cost
# −41% decode t/s (27.3 → 16.0 measured 2026-07-10). Instead: rings/pools
# stash the rows pushed during verify (arm_spec_stash) and on rejection are
# rolled back at the cache level (restore snapshot + re-push committed rows
# through the sequential-decode write path — see cache.py). Pools that
# cannot roll back cache-level (B>1, or multi-flush at gamma+1 > ratio)
# report spec_can_rollback=False and the whole rejection falls back to the
# commit-forward path.
_SPEC_CACHE_ROLLBACK = (
    os.environ.get("EXO_DSV4_SPEC_CACHE_ROLLBACK", "0") == "1"
)

# EXO_DSV4_SPEC_CACHE_ROLLBACK_C2=1 (default OFF): cache-level pool undo
# for the BS>1 min-acceptance path. With POOL_SNAPSHOT_BATCH now active,
# the c>=2 contamination path (b) pays a batched commit-forward on every
# flush-rejection (ratio-4 pools flush ~every 4th token, and B streams
# multiply the odds) — measured c=2 cost 14.9 -> 10.8 t/s/stream. This
# gate replaces path (b) with: path (a)'s validated per-stream ring trims
# + pool spec_rollback (trim when every stream keeps its flushes; restore
# + re-accumulate the batch-uniform committed prefix — keep = n_min + 1 —
# when no stream's committed prefix flushed; mixed attribution falls back
# to the commit-forward). Rings keep verify-row values for committed
# tokens instead of commit-forward recomputes — the same batched-M
# approximation class (c>=2 has no bitwise gate; the bar is
# no-contamination, which the pool undo preserves exactly).
_SPEC_CACHE_ROLLBACK_C2 = (
    os.environ.get("EXO_DSV4_SPEC_CACHE_ROLLBACK_C2", "0") == "1"
)


def _pool_flushed_since(pc: Any, snap: Any) -> bool:
    """True iff this pool flushed a NEW entry since ``snap`` (save_meta).

    Uses the TOTAL (visible lengths/offset + staged pending bumps): the
    deferred path's commit_pending() at the top of the verify forward moves
    a PRIOR staged bump from pending into the visible count (total
    unchanged), so comparing the visible count alone false-positives. A
    real flush this cycle increases the total.
    """
    if hasattr(pc, "_pool_lengths"):  # BatchPoolingCache: per-stream lists
        cur = [
            int(l) + int(p)
            for l, p in zip(pc._pool_lengths, pc._pending_bumps, strict=True)
        ]
        pre = [int(l) + int(p) for l, p in zip(snap[0], snap[5], strict=True)]
        return cur != pre
    return (pc._pool_offset + pc._pending_offset_bump) != (snap[0] + snap[1])


def _pool_may_flush(pc: Any, verify_len: int) -> bool:
    """Snapshot predicate: could the next ``verify_len``-token forward flush
    this pool? (Snapshotting is the expensive part — buf copies — so only
    pools that can flush are snapshotted.)"""
    rem = pc.remainder
    if isinstance(rem, list):  # BatchPoolingCache: per-stream
        rem = max(rem) if rem else 0
    return rem + verify_len >= pc.ratio


# Max concurrent streams the async decode fence may stay armed for
# (cache-owner key). 1 = validated c=1-only arming. Raised by
# EXO_DSV4_FENCE_ASYNC_C2=N to extend async fencing to batched decode —
# must match the model-side gate (deepseek_v4._FENCE_ASYNC_MAX_B) and the
# engine-side limit (batch_generate._update_fence_arming).
_FENCE_ASYNC_MAX_STREAMS = max(
    1, int(os.environ.get("EXO_DSV4_FENCE_ASYNC_C2", "0") or "0") or 1
)


class _PhaseTimer:
    """Per-cycle phase timer for MTP profiling, sliced by batch size.

    Keeps a per-(B, phase) buffer of samples and emits ``{mean, min,
    max}`` in milliseconds every ``_PROFILE_INTERVAL`` cycles. Active
    only when that env var is non-zero.
    """

    def __init__(self) -> None:
        self.cycles: int = 0
        self.cycles_by_b: dict[int, int] = {}
        # samples[B][phase] = list of ms.
        self.samples: dict[int, dict[str, list[float]]] = {}
        self._pending: dict[str, float] = {}

    def record(self, phase: str, ms: float) -> None:
        self._pending[phase] = ms

    def end_cycle(self, batch_size: int) -> None:
        self.cycles += 1
        self.cycles_by_b[batch_size] = self.cycles_by_b.get(batch_size, 0) + 1
        bucket = self.samples.setdefault(batch_size, {})
        for phase, ms in self._pending.items():
            bucket.setdefault(phase, []).append(ms)
        self._pending = {}
        if _PROFILE_INTERVAL > 0 and self.cycles % _PROFILE_INTERVAL == 0:
            self.dump()

    def dump(self) -> None:
        bs_summary = ",".join(
            f"B={b}:{c}" for b, c in sorted(self.cycles_by_b.items())
        )
        logger.warning(f"[MTP-PROF] cycles={self.cycles} {bs_summary}")
        for b in sorted(self.samples.keys()):
            for phase in (
                "draft", "verify", "accept", "commit", "rollback", "total",
            ):
                xs = self.samples[b].get(phase)
                if not xs:
                    continue
                mean = sum(xs) / len(xs)
                logger.warning(
                    f"[MTP-PROF]   B={b} {phase:10s} mean={mean:6.2f}ms "
                    f"min={min(xs):6.2f}ms max={max(xs):6.2f}ms n={len(xs)}"
                )


_phase_timer = _PhaseTimer() if _PROFILE_INTERVAL > 0 else None


# ─── C2 bistability tracer ─────────────────────────────────────────────────
#
# When EXO_DSV4_C2_TRACE=1, _draft_tokens_batched emits a per-cycle,
# per-chain-step JSONL trace to /tmp/dsv4_c2_trace_pid${pid}.jsonl. The
# tracer is built to investigate γ≥2 c=2 bistability where iter-N+1 sees
# one stream collapse (~3% of normal rate) while the other stays at
# expected ~20 t/s — a per-stream tail at the verify-side that ISN'T
# captured by aggregate metrics or PhaseTimer mean/min/max summaries.
#
# IMPORTANT: tracer inserts mx.eval() at every chain-step boundary so
# perf_counter timestamps reflect real GPU/comm latency, not lazy
# command-buffer fill. The eval() insertions THEMSELVES act like the
# proposed per-step fence fix, so a trace run does NOT validate the
# fix. Workflow: (1) trace ON to FIND the mechanism; (2) trace OFF +
# targeted fence in place to VALIDATE the fix.
#
# Per-step record fields:
#   cycle: spec cycle number (per-rank, monotonic)
#   step: chain step index (0..γ-1)
#   B: batch size (number of streams) for this cycle
#   gamma: full chain depth this cycle
#   pid, rank: process / rank identifiers
#   ts_*_ns: monotonic perf_counter_ns at each boundary
#       step_start, after_eagle_install, after_predict, after_argmax,
#       after_broadcast, step_end
#   tok_arr_per_stream: list[int] — the post-broadcast tokens, length B.
#       Useful for spotting BOS spam / repetition before timing diverges.
#   prev_logits_argmax_per_stream: list[int] — what tok_arr WOULD be if
#       broadcast weren't applied. Mismatch with tok_arr indicates which
#       rank "won" the broadcast.
#   metal_active_mb, metal_peak_mb: mx.metal memory at step_end. Growing
#       peak across steps in same cycle = lazy graph not draining.
#   eagle_installed: bool
#
# Per-cycle summary record fields (one per cycle, written after the
# γ step records):
#   cycle, pid, rank, B, gamma
#   ts_cycle_start_ns, ts_cycle_end_ns, cycle_wall_ms
#   per_step_wall_ms: list[float] length γ
#   bistability_flag: True if any step's wall > 2× cycle median step
#
_C2_TRACE_ENABLED = os.environ.get("EXO_DSV4_C2_TRACE") == "1"
_C2_TRACE_HANDLE: Optional[BinaryIO] = None
_C2_TRACE_RECS: int = 0

# ── Degeneration ⇄ per-stream-cache-swap correlation probe ──────────────
# Decisive test for the hypothesis: DSv4 degeneration (repeating-token
# collapse, ~7% of requests 2026-06-16) is caused by the BS>1 per-stream
# cache swap (`activate_for_uids` extract+merge → `_bootstrap_per_stream_ring`
# modular-ring remap) corrupting the SHARED prompt_cache that the TARGET
# verify-forward reads — i.e. the loss is NOT in the draft (which is
# target-verified) but in the target's own context after a swap.
#
# The existing EXO_DSV4_MTP_REFCHECK can't test this: it is gated temp==0
# (production verify runs temp=1.0) AND both its forwards share the same
# (possibly-scrambled) cache, so a scramble makes them AGREE.
#
# This probe is orthogonal: it stamps every BS-transition (uid set change in
# activate_for_uids) with the current spec-cycle + logs the extracted
# phys_rows-vs-size() consistency of each single (the exact scramble tell from
# the 2026-06-15 extract/merge bug). The batch_generate degeneration detector
# reads the stamp and records `cycles_since_last_transition` for the
# degenerating uid. If degeneration onset clusters tightly after a transition
# (small cycles_since), the swap is causal; if it's uniformly distributed,
# it's not. Works at temp=1.0 (the failing config). Cost when OFF: zero.
_DEGEN_PROBE_ENABLED = os.environ.get("EXO_DSV4_DEGEN_PROBE") == "1"
_DEGEN_PROBE_HANDLE: Optional[BinaryIO] = None
# Per-uid (spec_cycle, wall_ns) of the most recent BS-transition swap that
# touched that uid. Read cross-module by batch_generate's degeneration
# detector to compute cycles_since_last_transition at the moment of collapse.
# Module-level (not instance) so the detector can reach it without a handle
# to the generator. Bounded: one entry per active uid, pruned on dropout.
_DEGEN_LAST_TRANSITION: dict[int, dict[str, Any]] = {}


def _degen_probe_write(record: dict[str, Any]) -> None:
    """Append one JSONL record to the per-pid degeneration-probe file.

    Lazy unbuffered append so a crash leaves a complete record prefix.
    Only called when _DEGEN_PROBE_ENABLED (zero cost otherwise).
    """
    global _DEGEN_PROBE_HANDLE
    if _DEGEN_PROBE_HANDLE is None:
        path = f"/tmp/dsv4_degen_probe_pid{os.getpid()}.jsonl"
        _DEGEN_PROBE_HANDLE = open(path, "ab", buffering=0)  # noqa: SIM115
        header = {
            "type": "header",
            "schema_version": 1,
            "pid": os.getpid(),
            "ts_open_ns": time.perf_counter_ns(),
            "env": {
                "EXO_SPECULATIVE_GAMMA": os.environ.get(
                    "EXO_SPECULATIVE_GAMMA", "?"
                ),
                "EXO_DSV4_MTP": os.environ.get("EXO_DSV4_MTP", "?"),
                "EXO_SPECULATIVE": os.environ.get("EXO_SPECULATIVE", "?"),
            },
        }
        _DEGEN_PROBE_HANDLE.write((json.dumps(header) + "\n").encode("utf-8"))
    _DEGEN_PROBE_HANDLE.write((json.dumps(record) + "\n").encode("utf-8"))


def _c2_trace_rank() -> int:
    """Best-effort rank lookup. Returns -1 if mx.distributed unavailable.

    Used as a hint only — the tracer keys records by PID + auto-detected
    rank when possible, otherwise PID alone."""
    try:
        return int(mx.distributed.init().rank())
    except Exception:
        return -1


def _c2_trace_write(record: dict) -> None:
    """Append one JSONL record to the per-pid trace file.

    Lazy file open + unbuffered append so a crash mid-cycle still leaves
    a complete prefix of records on disk. Cost when tracer is OFF: zero
    (this function isn't called unless _C2_TRACE_ENABLED).
    """
    global _C2_TRACE_HANDLE, _C2_TRACE_RECS
    if _C2_TRACE_HANDLE is None:
        path = f"/tmp/dsv4_c2_trace_pid{os.getpid()}.jsonl"
        _C2_TRACE_HANDLE = open(path, "ab", buffering=0)  # noqa: SIM115
        # Write a header record so consumers can identify schema.
        import json as _json
        header = {
            "type": "header",
            "schema_version": 1,
            "pid": os.getpid(),
            "rank": _c2_trace_rank(),
            "ts_open_ns": time.perf_counter_ns(),
            "env": {
                "EXO_SPECULATIVE_GAMMA": os.environ.get(
                    "EXO_SPECULATIVE_GAMMA", "?"
                ),
                "EXO_DSV4_FENCE_EVERY_N_LAYERS": os.environ.get(
                    "EXO_DSV4_FENCE_EVERY_N_LAYERS", "?"
                ),
                "EXO_DSV4_MTP_EAGLE_K": os.environ.get(
                    "EXO_DSV4_MTP_EAGLE_K", "?"
                ),
                "EXO_DSV4_MTP": os.environ.get("EXO_DSV4_MTP", "?"),
                "EXO_DSV4_INDEX_TOPK": os.environ.get(
                    "EXO_DSV4_INDEX_TOPK", "?"
                ),
            },
        }
        _C2_TRACE_HANDLE.write((_json.dumps(header) + "\n").encode("utf-8"))
    import json as _json
    _C2_TRACE_HANDLE.write((_json.dumps(record) + "\n").encode("utf-8"))
    _C2_TRACE_RECS += 1


def _c2_trace_metal_mb() -> tuple[float, float]:
    """Return (active_mb, peak_mb) from mx.metal, or (-1, -1) on error."""
    try:
        active = float(mx.metal.get_active_memory()) / (1024.0 * 1024.0)
        peak = float(mx.metal.get_peak_memory()) / (1024.0 * 1024.0)
        return active, peak
    except Exception:
        return -1.0, -1.0


def _build_low_bit_lm_head_copy(lm_head: Any, bits: int) -> Optional[Any]:
    """Build a lower-bit quantized COPY of ``lm_head`` for draft-only use.

    Returns None when there is nothing to gain (already at or below the
    requested bit width) or the module type is unsupported. The copy is
    affine group-64 (the qmv-supported layout). Deterministic: identical
    input weights on every rank produce identical copies, so the
    cross-rank draft-determinism contract is unchanged.
    """
    import mlx.nn as nn

    if isinstance(lm_head, nn.QuantizedLinear):
        if lm_head.bits <= bits:
            return None
        w = mx.dequantize(
            lm_head.weight,
            lm_head.scales,
            lm_head.biases,
            group_size=lm_head.group_size,
            bits=lm_head.bits,
        )
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
        lin.weight = w
        q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
        mx.eval(q.parameters())
        del w, lin
        return q
    if isinstance(lm_head, nn.Linear):
        q = nn.QuantizedLinear.from_linear(lm_head, group_size=64, bits=bits)
        mx.eval(q.parameters())
        return q
    return None


class DSv4MTPPredictor:
    """Thin wrapper around ``model.model.mtp[mtp_idx]`` that exposes the
    ``predict`` API expected by :func:`mtp_module.draft_tokens`.

    Unlike Qwen3.5's ``MTPPredictor`` (which builds the MTP module
    dynamically from a separate weight file), DSv4's MTP module is
    already part of the loaded model object — it gets sharded by
    :class:`DeepseekV4ShardingStrategy.shard_model` alongside the main
    layers, and its weights come in via the main checkpoint (with
    ``deepseek_v4.sanitize`` keeping ``mtp.*`` after the patch landed
    in mlx-lm@<commit>). Construction here is essentially free.
    """

    def __init__(self, model: Any, mtp_idx: int = 0) -> None:
        self.model = model
        # For DSv4: outer `Model` owns `lm_head`; inner `model.model`
        # (DeepseekV4Model) owns `embed_tokens`, `norm`, and `mtp`.
        # Don't conflate the two — Qwen3.5's MTPPredictor used the
        # same name for both because that model's lm_head lives on
        # the inner model.
        inner = getattr(model, "model", None) or model.language_model.model
        self._inner = inner
        self.embed_tokens = inner.embed_tokens
        self.final_norm = inner.norm
        # lm_head: try outer model first (DSv4 layout), fall back to
        # inner / language_model (Qwen3.5-style layouts).
        self.lm_head = (
            getattr(model, "lm_head", None)
            or getattr(inner, "lm_head", None)
            or getattr(getattr(model, "language_model", None), "lm_head", None)
        )
        if self.lm_head is None:
            raise RuntimeError(
                "DSv4MTPPredictor: could not locate lm_head on the model. "
                "Checked model.lm_head, model.model.lm_head, model.language_model.lm_head."
            )

        if not hasattr(inner, "mtp") or len(inner.mtp) <= mtp_idx:
            raise RuntimeError(
                f"Model has no MTP module at index {mtp_idx}. Checkpoint "
                f"may be missing mtp.* weights, or num_nextn_predict_layers "
                f"is set to 0 in ModelArgs."
            )
        self.mtp_module = inner.mtp[mtp_idx]
        self._cache: Optional[Any] = None
        # Per-uid prefilled MTP cache snapshots. Each value is a single-
        # stream RotatingKVCache holding the post-prefill MTP K/V for
        # that uid. Populated by batch_generate.submit() via
        # snapshot_for_uid(); consumed by activate_for_uids() to rebuild
        # the active `_cache` as a PerStreamBatchRotatingKVCache (B=N)
        # at every BS-transition. Dropped via drop_uid() on stream finish.
        #
        # This is the c>=2 MTP-on fix from 2026-05-20: the old code did
        # `mtp.reset_cache()` on every submit, clobbering the SHARED
        # cache when stream 2 arrived while stream 1 was still running.
        # Stream 1's drafts then ran with stream 2's prefill state →
        # 0% acceptance → catastrophic perf regression (5.8 agg t/s vs
        # 30 t/s c=1). Per-uid snapshots + just-in-time rebuild
        # preserves per-stream draft state across concurrent requests.
        self._cache_per_uid: dict[int, Any] = {}
        # Track which uid set is currently in self._cache (so we can skip
        # rebuilds when activate_for_uids is called with the same set).
        self._active_uids: tuple[int, ...] = ()
        # Eagle-style soft-embedding for chained MTP draft steps.
        # EXO_DSV4_MTP_EAGLE_K = 0 (default) disables the path — the
        # standard hard-argmax embedding feeds every chained predict()
        # call. K > 0 enables: at every chain step beyond the first,
        # the input embedding is replaced with a probability-weighted
        # mixture of the previous step's top-K vocab embeddings. Phase
        # 14 Plan B.2: targets step-1 P(top-1) lift to raise MTP gamma>=2
        # acceptance. Read once at predictor construction.
        self.eagle_k = int(os.environ.get("EXO_DSV4_MTP_EAGLE_K", "0"))
        # Soft-emb logit temperature (default 1.0 = raw softmax). T<1.0
        # sharpens the mixture toward top-1 → better directional match
        # to the hard embed the MTP head was trained on → acceptance up.
        self.eagle_t = float(os.environ.get("EXO_DSV4_MTP_EAGLE_T", "1.0"))

        # Draft-only low-bit lm_head (2026-07-07). The draft chain reads
        # the full replicated lm_head (~925 MB affine8 at 129K vocab) once
        # per draft step — the dominant share of the ~4.9 ms context-flat
        # draft phase. A 4-bit COPY used ONLY for draft predict() calls
        # halves that read. Output-distribution safety is exact by the
        # rejection-sampling property: the draft's probs q are the softmax
        # of the SAME low-bit logits it samples from (a valid proposal
        # distribution), and the verify/accept side keeps the full-
        # precision target lm_head — at temp=0 accepted tokens are the
        # target's own argmax regardless of the draft head. Only the
        # acceptance RATE can move (validated via the 4K decode band).
        # lm_head is replicated per rank (DSv4 shards only MoE) and the
        # dequant→requant is deterministic, so cross-rank draft
        # determinism is preserved. Gate: EXO_DSV4_MTP_DRAFT_LMHEAD_BITS
        # (0 = off/default, 4 = build a 4-bit affine g64 copy).
        self.draft_lm_head: Optional[Any] = None
        _draft_bits_env = os.environ.get("EXO_DSV4_MTP_DRAFT_LMHEAD_BITS", "0")
        try:
            _draft_bits = int(_draft_bits_env or "0")
        except ValueError:
            _draft_bits = 0
        if _draft_bits in (4, 8):
            try:
                self.draft_lm_head = _build_low_bit_lm_head_copy(
                    self.lm_head, _draft_bits
                )
                if self.draft_lm_head is not None:
                    logger.info(
                        "DSv4MTPPredictor: draft lm_head copy at "
                        f"{_draft_bits}-bit built"
                    )
            except Exception as exc:  # noqa: BLE001 — draft head is optional
                logger.warning(
                    f"DSv4MTPPredictor: draft lm_head build failed ({exc}); "
                    "drafting with the full-precision lm_head"
                )
                self.draft_lm_head = None

    def set_eagle_soft_emb(self, emb: Optional[mx.array]) -> None:
        """Install or clear the Eagle soft-embedding side channel on the
        underlying DSv4 MTP module. ``emb`` must be shape ``(B, S,
        hidden_size)`` — same as ``embed_tokens(token_ids)`` for the
        same B/S — and is consumed by the very next ``predict()`` call.
        Pass ``None`` to clear so subsequent forwards revert to the
        hard-embed path.

        Routed through the module-level ``_EAGLE_CTX`` side channel in
        ``mlx_lm.models.deepseek_v4`` (mirror of ``_TREE_VERIFY_CTX``)
        so we don't have to thread a kwarg through every layer signature.
        Single-threaded per worker process — same constraint as the
        tree-verify channel.
        """
        from mlx_lm.models.deepseek_v4 import _set_eagle_soft_emb
        _set_eagle_soft_emb(emb)

    def reset_cache(self, batch_size: int = 1) -> None:
        """Reset the MTP attention's KV cache. Call when the target
        model's prompt cache resets or when starting a new sequence.

        At ``batch_size > 1`` constructs a
        :class:`PerStreamBatchRotatingKVCache` so spec rollback can
        per-stream-trim the MTP cache the same way the target caches
        do. A shared scalar-offset cache at B>1 leaves rejected drafts
        in the buffer for the lower-acceptance stream and tanks
        downstream draft quality.
        """
        self._set_fence_async(False)
        if batch_size > 1:
            self._cache = PerStreamBatchRotatingKVCache(
                max_size=self.mtp_module.config.sliding_window,
                left_padding=[0] * batch_size,
            )
        else:
            self._cache = self.mtp_module.make_cache()
        self._active_uids = ()

    def _set_fence_async(self, arm: bool) -> None:
        """Set the "cache" arming key of the c=1 async decode fence.

        Disarming ALSO drains any deferred async graph (mx.synchronize)
        so cache merges/rebuilds can't race in-flight GPU work — the
        2026-07-02 c=2 join corruption. The fence only actually goes
        async when the request-level "engine" key (batch_generate) is
        ALSO set; this method never overrides that owner.
        """
        from mlx_lm.models.deepseek_v4 import _set_fence_async_ok

        if not arm:
            _set_fence_async_ok(False, key="cache")
            mx.synchronize()
        else:
            _set_fence_async_ok(True, key="cache")

    def snapshot_for_uid(self, uid: int) -> None:
        """Snapshot the current single-stream ``self._cache`` under ``uid``.

        Called by ``batch_generate.submit()`` AFTER the per-stream MTP
        prefill forward has populated ``self._cache``. The snapshot is
        a reference to the single-stream RotatingKVCache that
        ``reset_cache()`` just built and ``predict()`` just prefilled.

        This is a JOIN-TIME snapshot only — it captures the new stream's
        prefilled state so the next ``activate_for_uids()`` call can
        merge it into the multi-stream cache. After activation the live
        ``_cache`` is the authoritative state; the snapshot is no
        longer consulted.
        """
        if self._cache is None:
            raise RuntimeError(
                f"snapshot_for_uid({uid}): _cache is None — call "
                f"reset_cache() + prefill before snapshot"
            )
        self._set_fence_async(False)
        self._cache_per_uid[uid] = self._cache
        # Single uid: snapshot doubles as the active cache (c=1 fast path).
        self._active_uids = (uid,)
        self._set_fence_async(True)

    def activate_for_uids(self, uids: "Sequence[int]") -> None:
        """Rebuild ``self._cache`` so it covers exactly ``uids`` in order.

        This is called on every BS-transition (uid joining or leaving).
        It must preserve each remaining uid's MTP K/V state across
        transitions — extracting from the live cache before reassembling.

        Pseudocode:
          if uids == active_uids: no-op.
          if active is None or empty: build from join-time snapshots.
          if active is single-stream and we're staying single-stream:
            no-op (same uid).
          else:
            # Multi-uid or transition.
            extract per-uid caches from the current live cache,
            ADD any uid in `uids` that's not in active (= just joined)
            using its join-time snapshot,
            then BatchRotatingKVCache.merge -> reclass as PerStream.

        At c=1 with the same uid (no transition), this is a no-op and
        preserves the c=1 fast path bit-exactly.
        """
        from mlx_lm.models.cache import (
            BatchRotatingKVCache,
            PerStreamBatchRotatingKVCache as _PerStream,
        )

        uids_t = tuple(uids)
        if uids_t == self._active_uids:
            # No transition — keep the fence arming in sync with the
            # (unchanged) stream count.
            self._set_fence_async(len(uids_t) <= _FENCE_ASYNC_MAX_STREAMS)
            return  # Already active for this uid set.

        # Disarm the async fence AND drain any deferred graph before
        # touching caches — a merge racing in-flight async GPU work is
        # the 2026-07-02 c=2 join corruption.
        self._set_fence_async(False)

        # Extract each uid's current state. Sources:
        #   - uid in active_uids: extract from live cache (carries decode-time state).
        #   - uid NEW (in uids_t but not active): use the join-time snapshot.
        live = self._cache
        new_singles: list[Any] = []
        for u in uids_t:
            if live is not None and u in self._active_uids:
                idx = self._active_uids.index(u)
                if len(self._active_uids) == 1:
                    # Live is already a single-stream cache for this uid.
                    new_singles.append(live)
                else:
                    # Live is batched. Extract this uid's slice.
                    extracted = live.extract(idx)
                    new_singles.append(extracted)
            else:
                # Newly joining uid — use the join-time snapshot.
                snap = self._cache_per_uid.get(u)
                if snap is None:
                    raise RuntimeError(
                        f"activate_for_uids({uids_t}): no MTP snapshot "
                        f"for uid {u}; submit() must call snapshot_for_uid "
                        f"after prefill"
                    )
                new_singles.append(snap)

        if len(uids_t) == 1:
            # Single-uid path: skip the merge, just use the single cache.
            self._cache = new_singles[0]
        else:
            batched = BatchRotatingKVCache.merge(new_singles)
            # Reclass in place so per-stream trim/make_mask logic fires.
            # (lazy PerStream bootstrap happens on first update_and_fetch.)
            batched.__class__ = _PerStream
            self._cache = batched

        # Re-arm the async fence only when back at single-stream.
        self._set_fence_async(len(uids_t) <= _FENCE_ASYNC_MAX_STREAMS)

        # ── DEGEN PROBE: stamp this BS-transition per uid + log the
        # extract phys-vs-size consistency tell (zero cost when OFF). ──
        if _DEGEN_PROBE_ENABLED:
            _now_ns = time.perf_counter_ns()
            _prev = tuple(self._active_uids or ())
            _is_bs_gt1 = len(uids_t) > 1
            # phys_rows-vs-size() per extracted single: the exact scramble
            # tell. extract() that returns fewer physical rows than size()
            # claims is the 2026-06-15 ragged bug; a non-crashing version
            # (right row count, wrong temporal mapping) would show
            # phys==size yet still be scrambled — so we log BOTH the count
            # consistency AND each single's (offset, _idx, rotated) so a
            # mismatch in the modular-ring remap is visible post-hoc.
            _singles_meta: list[dict[str, Any]] = []
            for _s_any in new_singles:
                _s: Any = _s_any
                try:
                    _phys = (
                        int(_s.keys.shape[2]) if getattr(_s, "keys", None)
                        is not None else 0
                    )
                    _sz = int(_s.size()) if hasattr(_s, "size") else -1
                    _singles_meta.append({
                        "phys_rows": _phys,
                        "size": _sz,
                        "consistent": bool(_phys == _sz),
                        "idx": int(getattr(_s, "_idx", -1)),
                        "rotated": bool(getattr(_s, "rotated", False)),
                    })
                except Exception:
                    _singles_meta.append({"meta_err": True})
            for _u in uids_t:
                _DEGEN_LAST_TRANSITION[int(_u)] = {
                    "wall_ns": _now_ns,
                    "to_uids": list(uids_t),
                    "from_uids": list(_prev),
                    "bs_gt1": _is_bs_gt1,
                }
            # Prune stamps for uids no longer active (bounded dict).
            for _stale in list(_DEGEN_LAST_TRANSITION.keys()):
                if _stale not in uids_t:
                    _DEGEN_LAST_TRANSITION.pop(_stale, None)
            _degen_probe_write({
                "event": "bs_transition",
                "wall_ns": _now_ns,
                "from_uids": list(_prev),
                "to_uids": list(uids_t),
                "bs_gt1": _is_bs_gt1,
                "extracted_singles": _singles_meta,
                "any_inconsistent": any(
                    not m.get("consistent", True) for m in _singles_meta
                ),
            })

        self._active_uids = uids_t

    def drop_uid(self, uid: int) -> None:
        """Forget this uid's join-time snapshot. Call on stream finish.

        Note: this does NOT modify the live ``_cache``. The next
        ``activate_for_uids()`` with the new (smaller) uid set will
        extract the remaining uids and rebuild.
        """
        self._cache_per_uid.pop(uid, None)
        # _active_uids will be reconciled on next activate_for_uids call.

    def predict(
        self,
        hidden_state: mx.array,
        token_ids: mx.array,
        return_hidden: bool = False,
        draft_mode: bool = False,
    ) -> Any:
        """Match the call signature of :meth:`MTPPredictor.predict`.

        Args:
            hidden_state: ``(B, S, D)`` post-hc_head, pre-final-norm
                hidden state captured from the target model (or from a
                previous chained MTP step).
            token_ids: ``(B, S)`` int — the token sampled at each
                position whose logits we want to predict the successor
                of.
            return_hidden: if True, also return the post-MTP-block,
                pre-final-norm hidden state for the next chained step.
            draft_mode: when True and a low-bit draft lm_head copy was
                built (EXO_DSV4_MTP_DRAFT_LMHEAD_BITS), project logits
                through it instead of the full-precision lm_head. Draft
                chain only — the verify/accept side never sees it.

        Returns:
            ``logits`` (B, S, vocab) when ``return_hidden=False``,
            ``(logits, hidden)`` when True. If ``S == 1`` the returned
            logits are squeezed to ``(B, vocab)`` to match the Qwen3.5
            convention.
        """
        _head = self.lm_head
        if draft_mode and self.draft_lm_head is not None:
            _head = self.draft_lm_head

        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        B, S = token_ids.shape
        if hidden_state.ndim == 2:
            hidden_state = hidden_state.reshape(B, S, -1)

        # The MTP cache is persistent across requests in the predictor
        # instance. If batch size changes (request set added / removed
        # / c=1↔c=2 transition), the cached KV state is stale and would
        # crash a concatenate at the next attention update. Reset on
        # any batch-size mismatch — the next chained-draft step will
        # repopulate from scratch (one MTP forward of overhead, much
        # cheaper than a wrong-shape crash).
        if self._cache is not None:
            cached_keys = getattr(self._cache, "keys", None)
            if cached_keys is not None and getattr(cached_keys, "shape", None):
                cache_B = cached_keys.shape[0]
                if cache_B != B:
                    self._cache = None
            # Class-mismatch reset: per-stream class needed at B>1 but
            # we currently hold a plain RotatingKVCache (or vice versa).
            elif (B > 1) != isinstance(self._cache, PerStreamBatchRotatingKVCache):
                self._cache = None

        if self._cache is None:
            self.reset_cache(batch_size=B)

        # Per-stream cache returns keys padded out to max(offset) per
        # the layout described in cache.py:2256. Without a per-stream
        # right-padding mask the lower-offset streams attend to stale
        # / cross-stream KV at positions past their own offset. Build
        # the mask once here and thread it through the MTP block.
        mtp_mask: Optional[mx.array] = None
        if isinstance(self._cache, PerStreamBatchRotatingKVCache):
            _ps_cache = self._cache
            mtp_mask = _ps_cache.make_mask(
                S, window_size=self.mtp_module.config.sliding_window, return_array=True
            )
            # Clamp the mask's KV axis to the cache's ACTUAL fetched width, not
            # the bare sliding_window. PerStreamBatchRotatingKVCache stores K/V
            # in a WIDE ring of ``max_size + _RING_SLACK`` (e.g. 128 + 8 = 136)
            # and update_and_fetch returns that full ring width, so SDPA sees
            # (B, H, S, ring_width). Clamping to sliding_window (128) made the
            # mask NARROWER than the KV → "[broadcast_shapes] (2,1,1,128) and
            # (2,64,1,136) cannot be broadcast" crash during MTP batched draft.
            # The ring width is the correct ceiling; the window itself is still
            # enforced by the mask's boolean content from make_mask.
            ring_width = _ps_cache.max_size + _ps_cache._RING_SLACK
            if mtp_mask is not None and mtp_mask.shape[-1] > ring_width:
                mtp_mask = mtp_mask[..., -ring_width:]

        out = self.mtp_module(
            prev_hidden=hidden_state,
            next_token=token_ids,
            embed_tokens=self.embed_tokens,
            final_norm=self.final_norm,
            lm_head=_head,
            mask=mtp_mask,
            cache=self._cache,
            return_hidden=return_hidden,
        )

        if return_hidden:
            logits, pre_norm_out = out
        else:
            logits = out
            pre_norm_out = None

        if S == 1:
            logits = logits.squeeze(1)

        if return_hidden:
            return logits, pre_norm_out
        return logits


def dsv4_speculative_forward(
    model: Any,
    inputs: mx.array,
    cache: Sequence[Any],
    captured: dict[str, mx.array],
    speculative: bool = False,
) -> tuple[mx.array, mx.array]:
    """Verify-pass forward for DSv4. Returns ``(pre_norm, logits)``.

    DSv4's caches are :class:`RotatingKVCache` + :class:`PoolingCache`
    (+ optional indexer pool) wrapped in :class:`CacheList`. Rollback
    on rejected drafts is via ``cache.trim(n_rejected)`` on each
    top-level cache (CacheList recursively trims its children) — no
    GDN bookkeeping needed, no manual layer iteration, no kernel
    monkey-patching. ``speculative`` is accepted for parity with the
    Qwen3.5 helper but currently ignored — callers do rollback after
    this returns.

    The pre-final-norm hidden is captured as a side effect of the
    wrapped final-norm in
    :meth:`MTPBatchGenerator._setup_hidden_capture`; this function
    just reads it back from the supplied ``captured`` dict.
    """
    del speculative  # rollback is caller's responsibility; trim() is enough

    # Diagnostic (EXO_DSV4_FP32_ACT debug): capture the ROOT exception + full
    # traceback of the verify forward to a file, so the fp32 c=2 crash cause
    # survives log rotation on the runner respawn.
    if os.environ.get("EXO_DSV4_VERIFY_TRACE") == "1":
        try:
            logits = model(inputs, cache=cache)
            mx.eval(logits)
        except Exception:
            import traceback as _tb
            with open("/tmp/dsv4_verify_crash.txt", "a") as _f:
                _f.write("=== verify forward crash: inputs shape=%s dtype=%s ===\n"
                         % (tuple(inputs.shape), inputs.dtype))
                _f.write(_tb.format_exc() + "\n")
            raise
    else:
        logits = model(inputs, cache=cache)
    pre_norm = captured.get("pre_norm")
    if pre_norm is None:
        raise RuntimeError(
            "dsv4_speculative_forward: captured['pre_norm'] is empty — the "
            "MTPBatchGenerator's wrapped-norm side channel was not "
            "exercised. Did model() run without going through "
            "DeepseekV4Model.norm?"
        )
    return pre_norm, logits


class DSv4MTPBatchGenerator(MTPBatchGenerator):
    """``MTPBatchGenerator`` specialized for DSv4. Overrides
    :meth:`_speculative_next` to use the DSv4 verify-forward + cache
    rollback path. Adds **BS>1** support so c=2+ benefits from MTP.

    BS>1 strategy: **min-acceptance.** All caches share a single scalar
    offset, so different per-uid acceptance counts can't be rolled
    back independently without per-stream cache state (which DSv4's
    cache classes don't expose). Instead we take ``n_min = min(n_accepted)``
    across uids, advance every cache by ``n_min + 1`` (drafts beyond
    ``n_min`` are wasted), and yield ``n_min + 1`` tokens per uid.

    Throughput math at BS=2 with per-stream accept p=0.85, gamma=3:
      single-stream tokens/cycle  ≈ 1 + p + p² + p³ = 3.18
      min-of-2 tokens/cycle       ≈ 2 × E[min(γ_1, γ_2)] + 2 ≈ 2 × 1.7 + 2 = 5.4
      vs naive (no spec) at BS=2  = 2
      → ~2.7× tokens/step at BS=2 (vs 3.18× single-stream MTP).

    Quality is preserved by construction: discarded "extra" accepted
    drafts simply aren't yielded; the cache state matches what the
    target model would produce after ``n_min + 1`` non-spec steps.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Acceptance-rate counters. Always updated — read by the
        # opt-in EXO_DSV4_MTP_LOG_INTERVAL warning, by the
        # EXO_DSV4_MTP_PROFILE phase timer, and by GenerationStats's
        # cumulative MTP fields that flow to master Prometheus.
        self._spec_cycles: int = 0
        self._spec_total_accepted: int = 0
        self._spec_accept_hist: list[int] = [0] * (self.gamma + 1)
        self._cached_sharding_group: Optional[mx.distributed.Group] = None
        self._jaccl_spec_handle: Optional[BinaryIO] = None
        self._mtp_drift_handle: Optional[BinaryIO] = None
        self._mtp_trace_handle: Optional[BinaryIO] = None
        self._mtp_trace_seq: int = 0
        # EXO_DSV4_SPEC_TRACE=1: per-cycle committed-token + cache-offset
        # divergence trace for the system-prompt degeneration hunt
        # (2026-05-29). One JSONL record per spec cycle on rank 0. Cost
        # when OFF: a single env.get + bool check per cycle.
        self._spec_trace_enabled: bool = (
            os.environ.get("EXO_DSV4_SPEC_TRACE") == "1"
        )
        self._spec_trace_handle: Optional[BinaryIO] = None
        self._spec_trace_cycle: int = 0
        # MTP SAMPLING-PARITY (2026-06-18): the MTP speculative path historically
        # honored only temp + min_p — NOT top_p and NOT repetition_penalty —
        # while the main (non-spec) sampler applies all four. Under MTP-on at
        # temp=1.0 with no top_p / no rep-penalty, long & concurrent generations
        # drift into single-token loops + structured-output (DSML/tool-call)
        # corruption → dead turns under real hermes load. Close the gap: apply
        # top_p (nucleus), a repetition penalty, AND a frequency penalty on the
        # verify-logit sampling so MTP-on draws from the same truncated,
        # repetition-damped distribution as the main sampler. Defaults:
        # top_p=0.95 (DSv4 card), rep_pen=1.3, freq_pen=0.5, ctx=64. The
        # frequency penalty (additive, COUNT-SCALED over the recent ring) is the
        # tool that actually kills period-1 single-character loops — the fixed
        # multiplicative rep_pen alone (applied once per distinct token) is too
        # weak to dislodge a token the model is very confident about at long
        # context (observed: ' `'/'‑' loops at 30K ctx with rep_pen=1.1).
        # Override via EXO_DSV4_MTP_TOP_P / _REP_PEN / _FREQ_PEN / _REP_CTX;
        # set 1.0 / 0 to disable a component for A/B.
        # top_p default OFF (1.0) in the MTP hot path: the per-token nucleus
        # needs a full-vocab sort/partition (~0.5ms/call on top of the ~0.39ms
        # penalty+min_p, ×(gamma+1) calls/cycle) which measurably eats the
        # decode budget, while min_p already provides a confidence-adaptive tail
        # clip and the rep/freq penalties handle loops. Set EXO_DSV4_MTP_TOP_P<1
        # to re-enable for quality A/B.
        self._mtp_top_p: float = float(
            os.environ.get("EXO_DSV4_MTP_TOP_P", "1.0")
        )
        self._mtp_rep_pen: float = float(
            os.environ.get("EXO_DSV4_MTP_REP_PEN", "1.3")
        )
        self._mtp_freq_pen: float = float(
            os.environ.get("EXO_DSV4_MTP_FREQ_PEN", "1.0")
        )
        self._mtp_rep_ctx: int = int(
            os.environ.get("EXO_DSV4_MTP_REP_CTX", "64")
        )
        # Confidence-gated greedy commit for the MTP bonus/correction. When the
        # TARGET distribution's max prob >= this, commit argmax instead of
        # sampling — protects rigid structured output (DSML tool-call tags,
        # where p≈1.0) from a single corrupting tail draw that fails the whole
        # turn. 0 disables (pure sampling). See _mtp_confident_argmax.
        self._mtp_greedy_p: float = float(
            os.environ.get("EXO_DSV4_MTP_GREEDY_P", "0.85")
        )
        # min_p precomputed (value + its log) so the per-token filter avoids an
        # os.environ.get + mx.log every call.
        self._mtp_min_p: float = float(
            os.environ.get("EXO_DSV4_MTP_MIN_P", "0.05")
        )
        import math as _math
        self._mtp_log_min_p: float = (
            _math.log(self._mtp_min_p) if self._mtp_min_p > 0.0 else 0.0
        )
        # Per-uid recent committed-token ring for the repetition penalty.
        # Updated at each cycle end from the committed (broadcast-synced)
        # token ids, so it is identical across TP ranks.
        self._recent_tokens: dict[int, deque[int]] = {}

    def _mtp_filter_logits(self, logits1d: mx.array, uid: int) -> mx.array:
        """Apply repetition+frequency penalty → top_p → min_p to a 1-D logit vec.

        Brings the MTP verify-logit sampling to parity with the main
        make_sampler/make_logits_processors path. Order mirrors mlx-lm:
        repetition penalty (multiplicative, per distinct recent token) +
        frequency penalty (additive, scaled by how many times each recent token
        occurred — this is what actually kills period-1 single-char loops),
        then nucleus (top_p), then min_p tail-clip. Pure on identical
        (logits, history) inputs, so per-rank deterministic — only rank-0's
        broadcast token commits. ``logits1d`` is already temperature-scaled.
        """
        lg = logits1d
        recent = self._recent_tokens.get(uid)
        # 1. Repetition penalty (sign-aware multiplicative, mlx-lm semantics:
        # logits[tokens] = where(<0, *penalty, /penalty)) applied once per
        # DISTINCT recent token, PLUS a frequency penalty (additive, COUNT-
        # scaled): logits[t] -= freq_pen * count[t]. The frequency term grows
        # with repeat count, so a token stuck in a period-1 loop is driven down
        # hard — the fixed multiplicative rep_pen alone is too weak for that.
        rp = self._mtp_rep_pen
        fp = self._mtp_freq_pen
        if recent and ((rp and rp != 1.0) or (fp and fp != 0.0)):
            from collections import Counter as _Counter
            counts = _Counter(recent)
            uniq = sorted(counts)
            idx = mx.array(uniq, dtype=mx.int32)
            sel = lg[idx]
            if rp and rp != 1.0:
                sel = mx.where(sel < 0, sel * rp, sel / rp)
            if fp and fp != 0.0:
                cnt = mx.array([counts[t] for t in uniq], dtype=lg.dtype)
                sel = sel - fp * cnt
            # Functional scatter (unique idx → exact; no source mutation).
            lg = lg.at[idx].add(sel - lg[idx])
        # 2. min_p tail-clip FIRST (cheap O(V): one max + compare, no sort).
        #    Doing this before top_p removes the bulk of the ~129K vocab so the
        #    nucleus step works on a far smaller candidate set. Relative to the
        #    (penalized) peak. _mtp_log_min_p is precomputed in __init__ to keep
        #    this off the per-token hot path.
        mp = self._mtp_min_p
        if mp > 0.0:
            lmax = mx.max(lg)
            lg = mx.where(lg >= lmax + self._mtp_log_min_p, lg, -float("inf"))
        # 3. top_p (nucleus): keep the smallest set of top tokens whose softmax
        #    mass >= top_p. Use a bounded top_k instead of a full-vocab argsort
        #    (the old O(V log V) sort over 129K tokens was ~0.8ms/call and the
        #    decode hot path calls this gamma+1 times/cycle). The nucleus for
        #    sane top_p never needs more than a few hundred tokens, so top_k=512
        #    is a safe superset; we sort only those and scatter the keep-mask.
        tp = self._mtp_top_p
        if 0.0 < tp < 1.0:
            k = min(512, lg.shape[-1])
            top_idx = mx.argpartition(-lg, k - 1)[:k]
            top_lg = lg[top_idx]
            order = mx.argsort(-top_lg)
            ord_idx = top_idx[order]
            sorted_p = mx.softmax(lg, axis=-1)[ord_idx]
            cum = mx.cumsum(sorted_p, axis=-1)
            keep_sorted = cum - sorted_p < tp  # include the crossing token
            mask = mx.full(lg.shape, False)
            mask = mask.at[ord_idx].add(keep_sorted.astype(mx.bool_))  # type: ignore[attr-defined]
            lg = mx.where(mask, lg, -float("inf"))
        return lg

    def _mtp_confident_argmax(
        self, target_logits1d: mx.array
    ) -> Optional[int]:
        """Return the argmax token id IFF the target distribution is highly
        confident (max softmax prob >= EXO_DSV4_MTP_GREEDY_P, default 0.85),
        else None.

        Why: DSv4 emits rigid structured output (DSML tool-call invoke /
        parameter tags). At those positions the target is ~deterministic
        (argmax p≈1.0). The MTP bonus/correction still SAMPLES there, and a
        single low-probability tail draw inside an <invoke>/<parameter> block
        corrupts the syntax → "unterminated/malformed invoke" → the parser
        fails the whole turn (the dead-turn symptom under interactive tool use,
        even after min_p/top_p/rep_pen — those narrow the tail but don't
        guarantee the structural token). When the model is this confident,
        sampling buys nothing and only risks corruption, so commit the argmax.
        Genuinely uncertain positions (prose, p below threshold) fall through
        to normal sampling and keep their diversity. Deterministic on identical
        logits → per-rank consistent (only rank-0's broadcast commits anyway).
        """
        gp = self._mtp_greedy_p
        if gp <= 0.0:
            return None
        # max prob = softmax(lg).max(); compute via logsumexp for stability.
        lmax = mx.max(target_logits1d)
        lse = mx.logsumexp(target_logits1d)
        max_prob = float(mx.exp(lmax - lse).item())
        if max_prob >= gp:
            return int(mx.argmax(target_logits1d).item())
        return None

    @property
    def _mtp_penalties_active(self) -> bool:
        """True if any repetition/frequency penalty needs the token ring."""
        return self._mtp_rep_ctx > 0 and (
            self._mtp_rep_pen != 1.0 or self._mtp_freq_pen != 0.0
        )

    def _mtp_record_tokens(self, uid: int, token_ids: "Sequence[int]") -> None:
        """Append committed token ids to the uid's recent-token ring."""
        if not self._mtp_penalties_active:
            return
        ring = self._recent_tokens.get(uid)
        if ring is None:
            ring = cast("deque[int]", deque(maxlen=self._mtp_rep_ctx))
            self._recent_tokens[uid] = ring
        for t in token_ids:
            ring.append(int(t))

    def _filter_finished_uid(self, uid: int) -> None:
        """Override to log when a uid is filtered (per-rank, EOS or
        length). Used to capture asymmetric filter timing across ranks
        at BS-transition. Trace is gated on
        EXO_DSV4_MTP_TRANSITION_TRACE=1.

        Also drops the MTP cache snapshot for the finished uid (2026-05-20
        c>=2 fix). The active MTP cache is reconciled on the next spec
        cycle via activate_for_uids().
        """
        if os.environ.get("EXO_DSV4_MTP_TRANSITION_TRACE") == "1":
            gen_batch = self._generation_batch
            self._mtp_trace_log("filter_finished_uid", {
                "filter_uid": uid,
                "uids_before": list(gen_batch.uids),
                "num_tokens": list(gen_batch._num_tokens),
            })
        if _CYCLE_STATS:
            _st = getattr(self, "_cycle_stats", {}).pop(uid, None)
            if _st is not None:
                logger.warning(
                    f"[CYCLE-STATS] uid={uid} cycles={_st[0]} "
                    f"rejects={_st[1]} regime_b={_st[2]} "
                    f"first_reject_tok={_st[3]} first_regime_b_tok={_st[4]} "
                    f"committed={_st[5]} "
                    f"cache_rb={_st[6] if len(_st) > 6 else 0}"
                )
        if hasattr(self.mtp, "drop_uid"):
            self.mtp.drop_uid(uid)
        self._recent_tokens.pop(uid, None)
        super()._filter_finished_uid(uid)

    def _mtp_trace_log(self, event: str, data: dict[str, Any]) -> None:
        """Append a JSONL line tagged with rank+seq+event to the
        per-rank transition-trace file. Gated on
        EXO_DSV4_MTP_TRANSITION_TRACE=1; no-op otherwise.
        """
        if self._mtp_trace_handle is None:
            sg = self._get_sharding_group()
            rank = sg.rank() if sg is not None else 0
            path = f"/tmp/dsv4_mtp_trace_rank_{rank}_pid{os.getpid()}.log"
            self._mtp_trace_handle = open(path, "ab", buffering=0)  # noqa: SIM115
        import json as _json
        self._mtp_trace_seq += 1
        rec = {"seq": self._mtp_trace_seq, "event": event, **data}
        line = (_json.dumps(rec, default=str) + "\n").encode("utf-8")
        self._mtp_trace_handle.write(line)
        self._mtp_drift_step: int = 0

    def _collect_pooling_caches(self, gen_batch: Any) -> list[Any]:
        """Return every PoolingCache in the prompt cache (recursing into
        CacheList children).

        These are the caches whose ``trim()`` cannot undo a
        speculative-verify mutation (it only shrinks the remainder buffer,
        not the flushed pool). The linear/tree spec paths snapshot + restore
        them around verify to keep the compressed-attention context
        consistent with the committed token stream. See PoolingCache.save_meta.
        """
        pool_classes: tuple[type, ...] = (
            (PoolingCache, BatchPoolingCache)
            if _POOL_SNAPSHOT_BATCH
            else (PoolingCache,)
        )
        pools: list[Any] = []
        for c in gen_batch.prompt_cache:
            if isinstance(c, CacheList):
                for sub in c.caches:
                    if isinstance(sub, pool_classes):
                        pools.append(sub)
            elif isinstance(c, pool_classes):
                pools.append(c)
        return pools

    def _spec_trace_offsets(self, gen_batch: Any) -> dict[str, Any]:
        """Snapshot every cache's scalar offset (post-rollback) so a
        divergence cycle can be correlated with cache-state drift.

        Returns a dict mapping a stable cache label -> offset int. Walks
        CacheList children recursively. Best-effort: any cache without an
        ``.offset`` (or a getter) is reported as None. Used only by the
        EXO_DSV4_SPEC_TRACE path; never on the hot path when OFF.
        """
        def _off(c: Any) -> Any:
            o: Any = getattr(c, "offset", None)
            if o is not None:
                try:
                    return int(o)
                except Exception:
                    return str(o)
            getter: Any = getattr(c, "get_offset", None)
            if callable(getter):
                try:
                    return int(getter())
                except Exception:
                    return None
            return None

        offsets: dict[str, Any] = {}
        for ci, c in enumerate(gen_batch.prompt_cache):
            if isinstance(c, CacheList):
                for si, sub in enumerate(c.caches):
                    offsets[f"c{ci}.{type(sub).__name__}{si}"] = _off(sub)
            else:
                offsets[f"c{ci}.{type(c).__name__}"] = _off(c)
        mtp_cache: Any = getattr(self.mtp, "_cache", None)
        if mtp_cache is not None:
            offsets[f"mtp.{type(mtp_cache).__name__}"] = _off(mtp_cache)
        return offsets

    def _spec_trace_cycle_dump(
        self,
        uids: Sequence[int],
        gen_batch: Any,
        gamma: int,
        verify_input: mx.array,
        draft_concat: mx.array,
        target_tokens: mx.array,
        n_accepted_per: Sequence[int],
        bonus_vals: Sequence[int],
        all_tokens_per: Sequence[Sequence[tuple[int, Any]]],
    ) -> None:
        """Write one JSONL record per spec cycle (rank 0 only).

        Captures, for the degeneration hunt: the exact tokens COMMITTED
        this cycle (the ground-truth-comparable stream), the draft vs
        target argmax that drove acceptance, n_accepted, the staged bonus,
        the verify-input chunk, and all post-rollback cache offsets. A
        downstream diff script aligns the committed-token stream against a
        plain-greedy (MTP-off) capture of the same prompt and reports the
        first divergence cycle + that cycle's offsets/acceptance.

        Rank-gated to rank 0 (canonical), since all ranks commit the same
        tokens after the n_accepted broadcast — one stream is enough and
        avoids interleaved files.
        """
        sg = self._get_sharding_group()
        rank = sg.rank() if sg is not None else 0
        if rank != 0:
            return
        if self._spec_trace_handle is None:
            path = f"/tmp/dsv4_spec_trace_pid{os.getpid()}.jsonl"
            self._spec_trace_handle = open(  # noqa: SIM115
                path, "ab", buffering=0
            )
        import json as _json
        self._spec_trace_cycle += 1
        try:
            vi = cast(list[list[int]], verify_input.tolist())
            dc = cast(list[list[int]], draft_concat.tolist())
            tt = cast(list[list[int]], target_tokens.tolist())
        except Exception:
            vi = dc = tt = []
        rec = {
            "cycle": self._spec_trace_cycle,
            "pid": os.getpid(),
            "gamma": gamma,
            "uids": list(uids),
            # committed[n] = the ordered token ids yielded this cycle for
            # stream n: [next_token, accepted drafts..., bonus]. This is
            # the sequence that MUST match plain greedy.
            "committed": [
                [int(tid) for (tid, _lp) in row] for row in all_tokens_per
            ],
            "n_accepted": list(n_accepted_per),
            "bonus": [int(b) for b in bonus_vals],
            "draft": dc,            # (N, gamma) the drafted ids
            "target_argmax": tt,    # (N, gamma) verify argmax over drafts
            "verify_input": vi,     # (N, gamma+1)
            "offsets": self._spec_trace_offsets(gen_batch),
        }
        self._spec_trace_handle.write(
            (_json.dumps(rec, default=str) + "\n").encode("utf-8")
        )

    def _mtp_drift_dump(
        self,
        cycle_n: int,
        chain_step: int,
        hidden_pre: mx.array,
        logits: mx.array,
        tok_pre_sync: mx.array,
        tok_post_sync: mx.array,
    ) -> None:
        """Per-MTP-chain-step drift trace when EXO_MTP_DRIFT_DUMP=1.

        Cross-rank diff of this file localises which MTP forward step
        first produces divergent logits across ranks — i.e. when the
        ~1ulp drift accumulates enough to flip an argmax. Captures four
        per-step quantities:

        * ``hidden_hash`` — uint32 of all hidden_state values, hashed.
          Phase E.1 evidence says this is bit-exact across ranks at
          cycle 5 entry; this trace re-verifies on cluster.
        * ``logits_top16_argmax`` — argmax of top-16 logit values.
          Drift first manifests here: when two near-tied logits flip
          order across ranks, argmax flips.
        * ``logits_max_abs_delta`` — bf16 ulp count between the two
          largest logit values; sized so cross-rank comparison shows
          how close the contest was.
        * ``tok_pre_sync`` / ``tok_post_sync`` — argmax before vs
          after the broadcast_from_canonical sync. Equal pre⇒
          drift didn't flip an argmax this step. Different pre⇒
          drift flipped an argmax (the case Phase F catches).

        Default off; zero overhead when env unset.
        """
        if os.environ.get("EXO_MTP_DRIFT_DUMP") != "1":
            return

        import json

        if self._mtp_drift_handle is None:
            group = self._get_sharding_group()
            rank = group.rank() if group is not None else 0
            path = f"/tmp/mtp_drift_rank_{rank}_pid{os.getpid()}.log"
            self._mtp_drift_handle = open(  # noqa: SIM115
                path, "ab", buffering=0
            )

        # Materialise so .item() / .tolist() reads don't pay async-eval
        # roundtrips at cluster scale; the dump is opt-in so the cost
        # only matters when it's enabled.
        hidden_hash_arr = mx.sum(
            hidden_pre.astype(mx.float32) * 1e6
        ).astype(mx.int64)
        top_logits, top_idx = mx.topk(logits.astype(mx.float32), k=16, axis=-1)
        # Range of top-16 logits as a proxy for "how tied was the argmax".
        logit_spread = (top_logits[..., 0] - top_logits[..., 15]).astype(
            mx.float32
        )
        mx.eval(
            hidden_hash_arr, top_idx, logit_spread, tok_pre_sync, tok_post_sync
        )

        rec = {
            "cycle": cycle_n,
            "chain_step": chain_step,
            "hidden_hash": int(hidden_hash_arr.item()),
            "top_idx": cast(list[int], top_idx.tolist()),
            "logit_spread": cast(list[float], logit_spread.tolist())
            if logit_spread.ndim > 0
            else float(logit_spread.item()),
            "tok_pre_sync": cast(list[int], tok_pre_sync.tolist()),
            "tok_post_sync": cast(list[int], tok_post_sync.tolist()),
        }
        line = (json.dumps(rec) + "\n").encode("utf-8")
        self._mtp_drift_handle.write(line)

    def _jaccl_dump_spec(
        self,
        uids: "Sequence[int]",
        target_tokens_lst: list[list[int]],
        draft_lst: list[list[int]],
        matches_lst: list[list[bool]],
        n_accepted_per: list[int],
        next_tokens_int: list[int],
        bonus_vals: list[int],
    ) -> None:
        """Log one spec cycle when JACCL_TRACE_STEP=1. Cross-rank diff
        of this file pinpoints which mx.array first goes non-bit-exact
        across ranks — target_tokens (verify forward), draft_concat
        (MTP forward), or just matches (one of the above is off).
        Memory: next_session_plan_jaccl_c2_prefix_cache.md.
        """
        if os.environ.get("JACCL_TRACE_STEP") != "1":
            return

        import json

        if self._jaccl_spec_handle is None:
            group = self._get_sharding_group()
            rank = group.rank() if group is not None else 0
            path = f"/tmp/jaccl_spec_rank_{rank}_pid{os.getpid()}.log"
            self._jaccl_spec_handle = open(path, "ab", buffering=0)  # noqa: SIM115

        rec = {
            "uids": list(uids),
            "next_tokens": next_tokens_int,
            "target": target_tokens_lst,
            "draft": draft_lst,
            "matches": [[bool(b) for b in row] for row in matches_lst],
            "n_acc": n_accepted_per,
            "bonus": bonus_vals,
        }
        line = (json.dumps(rec) + "\n").encode("utf-8")
        self._jaccl_spec_handle.write(line)

    def _get_sharding_group(self) -> "Optional[mx.distributed.Group]":
        """Return the TP sharding group used by FFN/attention all_sums.

        Caches the lookup; falls back to None when the model isn't
        sharded. Used by `_next` to gate the buffer-drain branch
        collectively so ranks stay in lock-step under JACCL c=2.
        """
        if self._cached_sharding_group is not None:
            return self._cached_sharding_group
        model: Any = self.model  # type: ignore[reportUnknownMemberType]
        inner: Any = (
            getattr(model, "model", None)
            or getattr(model, "language_model", None)
        )
        if inner is None:
            return None
        layers: Any = getattr(inner, "layers", None) or getattr(
            getattr(inner, "model", None), "layers", None
        )
        if not layers:
            return None
        layer0: Any = layers[0]
        for attr in ("ffn", "mlp", "block_sparse_moe"):
            sub: Any = getattr(layer0, attr, None)
            if sub is not None:
                g: Any = getattr(sub, "sharding_group", None)
                if g is not None:
                    cast_g = cast("mx.distributed.Group", g)
                    self._cached_sharding_group = cast_g
                    return cast_g
        return None

    def _record_acceptance(self, n_accepted: int) -> None:
        """Record one MTP draft-acceptance sample.

        At BS=1 this is one call per spec cycle. At BS>1 the batched
        path calls this once PER STREAM (B times per cycle), so
        counter semantics are "per cycle×stream sample":

        * ``self._spec_cycles`` = cycles × concurrent streams
        * ``self._spec_total_accepted`` = sum of accepted drafts
          across all stream-cycles
        * histogram bins each stream's per-cycle acceptance count

        Mean acceptance per stream per cycle = total_accepted / cycles.
        """
        self._spec_cycles += 1
        self._spec_total_accepted += n_accepted
        if 0 <= n_accepted <= self.gamma:
            self._spec_accept_hist[n_accepted] += 1
        if _LOG_INTERVAL <= 0:
            return
        if self._spec_cycles % _LOG_INTERVAL == 0:
            mean = self._spec_total_accepted / self._spec_cycles
            hist = ",".join(
                f"{i}:{c}" for i, c in enumerate(self._spec_accept_hist)
            )
            logger.warning(
                f"[MTP] cycles={self._spec_cycles} "
                f"mean_accept={mean:.3f}/{self.gamma} "
                f"hist={hist}"
            )

    # ── BS>1 dispatch ──────────────────────────────────────────────────

    def _next(self, *args: Any, **kwargs: Any):
        """Override base BS=1-only spec dispatch with BS-N support.

        Eligibility for spec: at least one uid in gen_batch, no
        prefill pending, no unprocessed sequences. The BS=1 path
        keeps the parent's behavior; BS>1 takes the new batched
        path.
        """
        gen_batch = self._generation_batch

        # BS-transition diagnostic. Dumps per-cycle state to
        # /tmp/dsv4_mtp_trace_rank_${rank}_pid${pid}.log when
        # EXO_DSV4_MTP_TRANSITION_TRACE=1 is set. Used to localize
        # cross-rank divergence at the c=2→c=1 transition (one stream
        # hits EOS, other keeps going). Memory:
        # jaccl_ack_qp_fix_2026_05_07.md — open issue at session end.
        if os.environ.get("EXO_DSV4_MTP_TRANSITION_TRACE") == "1":
            self._mtp_trace_log("_next ENTER", {
                "uids": list(gen_batch.uids),
                "num_tokens": list(gen_batch._num_tokens),
                "buffered_uids": [
                    u for u, b in self._token_buffer.items() if b
                ],
                "prefilled": sorted(self._mtp_prefilled),
                "unprocessed": len(self._unprocessed_sequences),
                "prompt_batch": len(self._prompt_batch),
            })

        # Collective gen_batch sync. Empirically (Phase E.1 trace
        # comparison 2026-05-05) MTP draft state diverges across ranks
        # at long context after several spec cycles even though every
        # collective output hash matches. Drafts diverge for one uid
        # while another stays bit-exact, then `n_accepted_per` differs,
        # then per-uid `_num_tokens` and yielded tokens differ, then
        # length-finish (max_tokens) and stop-finish (state machine
        # match) fire on different ranks at different steps, then
        # `_filter_finished_uid` removes different uids per rank, then
        # the next forward all_sum has different message size on each
        # rank → JACCL LEN_ERR cluster wedge.
        #
        # Two-step sync at the top of every `_next` keeps collective
        # forwards in lock-step across ranks:
        #   1) gen_batch.uids ← intersection across ranks (any rank
        #      that filters a uid causes ALL ranks to filter it).
        #   2) gen_batch._num_tokens ← max across ranks (any rank that
        #      hits length-finish causes ALL ranks to hit it).
        # Yielded tokens still come from one rank only (master picks
        # device_rank=0); the other rank's yields are discarded by the
        # outer master event router. Memory:
        # next_session_plan_jaccl_c2_prefix_cache.md (Phase E).
        sync_group = self._get_sharding_group()
        # Coord subgroup: this upstream sync fires per-_next call
        # (~25-60 Hz at decode), interleaving with the model TP
        # forward's bf16 all_sums. Sharing `next_call_id_` with the
        # model group caused the c=2 cross-rank op-mismatch race
        # observed at call_id 4058 via JACCL_TRACE_HASH=1 on
        # 2026-05-07 (rank 0 saw a 1-byte all_sum, rank 1 saw a
        # 16384-byte bf16 all_sum). Routing this collective onto
        # the coord sibling subgroup gives it an isolated call_id
        # space and its own UC FIFO so cross-stream call_ids can no
        # longer collide. See get_coord_group in utils_mlx.
        coord_group = get_coord_group(sync_group)
        if coord_group is not None and coord_group.size() > 1 and len(gen_batch) >= 1:
            # Step 1 — uid intersection. Encode each rank's uid set as
            # a presence bitmask of bounded size; all_sum to count how
            # many ranks have each uid; keep only uids present on ALL.
            uid_bound = 1024
            local_presence: list[int] = [0] * uid_bound
            for _u in gen_batch.uids:
                if 0 <= _u < uid_bound:
                    local_presence[_u] = 1
            presence_arr = mx.array(local_presence, dtype=mx.int32)
            counted = mx.distributed.all_sum(presence_arr, group=coord_group)
            mx.eval(counted)
            n_ranks = coord_group.size()
            counted_lst = cast(list[int], counted.tolist())
            keep_uids: set[int] = {
                _u for _u, _c in enumerate(counted_lst) if _c == n_ranks
            }
            keep_indices: list[int] = [
                _i for _i, _u in enumerate(gen_batch.uids) if _u in keep_uids
            ]
            if os.environ.get("EXO_DSV4_MTP_TRANSITION_TRACE") == "1":
                # Counted entries that were non-zero (peer thought uid
                # was alive at any point). Helps spot rank divergence.
                nonzero_uids = {
                    _u: _c for _u, _c in enumerate(counted_lst) if _c > 0
                }
                self._mtp_trace_log("upstream_sync", {
                    "local_uids": list(gen_batch.uids),
                    "counted_uids": nonzero_uids,
                    "n_ranks": n_ranks,
                    "keep_uids": sorted(keep_uids),
                    "will_filter": len(keep_indices) < len(gen_batch.uids),
                })
            if len(keep_indices) < len(gen_batch.uids):
                drop_uids = [u for u in gen_batch.uids if u not in keep_uids]
                gen_batch.filter(keep_indices)
                # Clean up MTP-side state for dropped uids on ranks
                # that hadn't filtered them locally. _cleanup_uid pops
                # _mtp_pre_norm, _mtp_prefilled, _token_buffer,
                # _request_temp.
                for _u in drop_uids:
                    self._cleanup_uid(_u)

            # Step 2 — _num_tokens max. After filter, both ranks have
            # the same uid set in the same order, so the int array is
            # length-equivalent for all_max.
            if len(gen_batch) >= 1:
                synced = mx.distributed.all_max(
                    mx.array(gen_batch._num_tokens), group=coord_group
                )
                mx.eval(synced)
                gen_batch._num_tokens = cast(list[int], synced.tolist())

        # Drain buffered tokens first (one per call), per the parent
        # API. At BS>1 we may have multiple uids with buffered tokens
        # — drain whichever has them first; the gen_batch still
        # advances normally on subsequent calls.
        #
        # In TP mode the per-rank `_token_buffer` can diverge if any
        # earlier collective wasn't bit-exact across ranks (e.g. JACCL
        # c=2 transport corruption); without a collective gate, ranks
        # take different branches here and yield asymmetrically — the
        # downstream `on_generation_token` → `agree_on_cancellations`
        # callbacks then issue different op streams across ranks and
        # JACCL detects the mismatch as call_id LEN_ERR. Drain only
        # when ALL ranks have buffer for the uid so yield counts stay
        # symmetric. If any rank lacks buffer, fall through to MTP
        # uniformly. Memory: jaccl_phase_a_finding_2026_05_05.
        # Buffer-drain rules:
        #  * Single-rank or non-TP: drain freely.
        #  * TP at c=1: drain freely. There's exactly one uid by
        #    definition, so the per-uid loop runs once on every rank
        #    and `_filter_finished_uid` can't fire asymmetrically (the
        #    one uid either finishes on all ranks together via the same
        #    state-machine match, or on none). Yield counts stay
        #    cross-rank-symmetric.
        #  * TP at c>1: skip drain. `for uid in gen_batch.uids` length
        #    can diverge across ranks at long contexts when
        #    `_filter_finished_uid` fires asymmetrically (rank 0 has
        #    1 uid, rank 1 has 2). Mismatched per-uid loop counts ⇒
        #    each rank issues a different number of downstream
        #    collective sequences ⇒ JACCL LEN_ERR / wedge. The cost is
        #    that the MTP-yields-multiple-tokens-per-cycle gain is lost
        #    on TP c>1 — buffered drafts get clobbered on the next
        #    cycle (line 717: assign-not-append). Worth it to keep the
        #    cluster up at long context. Memory:
        #    jaccl_phase_a_finding_2026_05_05.md.
        #
        # Earlier code skipped drain in *all* TP modes including c=1,
        # which silently dropped accepted drafts at single-stream
        # generation — the source of the broken-English MTP=1
        # regression observed 2026-05-06.
        if len(gen_batch) >= 1:
            # 2026-05-20 c>=2 fix: prior code conditionally skipped drain
            # at TP c>1 to avoid asymmetric `_filter_finished_uid` across
            # ranks. The cost was severe: combined with the buffer
            # assignment-clobber (line 1416), each spec cycle wrote γ
            # new tokens on top of γ old tokens → c=2 MTP-on agg t/s
            # collapsed from expected ~30 to 5.7 at 100K.
            #
            # Drain IS safe at TP c>1 because the UNCONDITIONAL
            # broadcast at line 1213 (broadcast_from_canonical of
            # n_accepted_per + bonus_vals via coord_group) replaces
            # per-rank values with rank-0's POST-divergence. So even at
            # temp>0 where draft sampling diverges per rank, the
            # post-broadcast yielded tokens are bit-identical, so
            # _filter_finished_uid fires symmetrically (stop conditions
            # = function of yielded tokens) and per-stream buffer
            # growth is symmetric. The historical drain-skip was a
            # pre-broadcast defensive measure; with the broadcast in
            # place it's now a perf-killer with no safety benefit.
            for uid in gen_batch.uids:
                if uid in self._token_buffer and self._token_buffer[uid]:
                    return [], self._yield_buffered(uid)

        spec_eligible = (
            self.gamma > 0
            and len(gen_batch) >= 1
            and len(self._prompt_batch) == 0
            and len(self._unprocessed_sequences) == 0
        )
        # C≥2 high-context MTP degeneration gate REMOVED (2026-06-24). The
        # degeneration root cause — _bootstrap_per_stream_ring rebasing the
        # absolute position via self._offset (ring cursor) instead of
        # self.offset (logical position) — is fixed (mlx-lm 48a4a3c). MTP-on
        # at c≥2 high context now produces clean quality through 500K
        # (verified: b2 200K/300K/500K all_needles=True). The gate that
        # disabled spec at c≥2 high context (EXO_DSV4_MTP_C2_MAX_CTX) is no
        # longer needed; removing it so MTP-on is the default at c≥2. The
        # env var is still read for backward-compat safety but no longer has
        # a default-threshold effect (set to 0 to re-disable if a regression
        # ever surfaces).
        if spec_eligible and len(gen_batch) >= 2:
            _c2_max = int(os.environ.get("EXO_DSV4_MTP_C2_MAX_CTX", "0"))
            if _c2_max == 0:
                # 0 = no gate (default now that the root cause is fixed).
                # Kept as an explicit opt-in so a future regression can be
                # bandaged by setting a real threshold without a code change.
                pass
            else:
                _max_ctx = 0
                for _c in gen_batch.prompt_cache:
                    _subs = _c.caches if hasattr(_c, "caches") else [_c]
                    for _sub in _subs:
                        try:
                            _off = _sub.offset
                            if hasattr(_off, "shape"):
                                _off = int(mx.max(_off))
                            else:
                                _off = int(_off)
                            if _off > _max_ctx:
                                _max_ctx = _off
                        except Exception:
                            pass
                if _max_ctx > _c2_max:
                    spec_eligible = False
        # ALL-C high-context MTP gate (2026-07-09). At c=1, 100K+ ctx the
        # batched L>1 verify forward drifts vs a clean single-token forward
        # by up to ~1.7 logits (REFCHECK delta_128822 on the hermes-replay
        # corpus; same unattributed family as the qL=3 sdpa residuals up to
        # 2.8 seen in the c2 pooling-skew work). Near-tied structural tokens
        # then flip and corrupt rigid DSML blocks (the ``</｜DSML｜inv>``
        # class). The margin-gated tie re-verify (step 4) catches only the
        # flips where the VERIFY row itself shows a small gap — drift larger
        # than the gap escapes one-sided gating. Until the drift is
        # attributed and fixed, cap MTP by context at ANY concurrency:
        # above the threshold, sequential decode (bitwise the MTP-off path,
        # the known-clean configuration). 0 disables the gate.
        if spec_eligible:
            _allc_max = int(os.environ.get("EXO_DSV4_MTP_MAX_CTX", "0"))
            if _allc_max > 0:
                _max_ctx_allc = 0
                for _c in gen_batch.prompt_cache:
                    _subs = _c.caches if hasattr(_c, "caches") else [_c]
                    for _sub in _subs:
                        try:
                            _off = _sub.offset
                            if hasattr(_off, "shape"):
                                _off = int(mx.max(_off))
                            else:
                                _off = int(_off)
                            if _off > _max_ctx_allc:
                                _max_ctx_allc = _off
                        except Exception:
                            pass
                if _max_ctx_allc > _allc_max:
                    spec_eligible = False
        if os.environ.get("EXO_DSV4_MTP_TRANSITION_TRACE") == "1":
            self._mtp_trace_log("dispatch_decision", {
                "spec_eligible": spec_eligible,
                "gamma": self.gamma,
                "n_uids": len(gen_batch),
                "n_prompt_batch": len(self._prompt_batch),
                "n_unprocessed": len(self._unprocessed_sequences),
                "uids": list(gen_batch.uids),
            })
        if spec_eligible:
            # All uids must be prefilled (have a captured pre_norm).
            uids = list(gen_batch.uids)
            need_first_step = [u for u in uids if u not in self._mtp_prefilled]
            if need_first_step:
                return self._first_step_and_capture_batch(uids)
            # 2026-05-20 c>=2 MTP fix: sync the MTP cache layout to the
            # current uid set. At c=1 with the same uid as last cycle
            # this is a no-op. At BS-transitions (uid joining or
            # leaving) this rebuilds the cache by extracting per-uid
            # state from the live cache + merging in join-time
            # snapshots for newcomers. Without this the MTP cache stays
            # at whatever shape the prior submit() left it in, which
            # for c=2 was a single-stream cache for the LAST submit
            # only -- catastrophic regression for stream 1.
            if hasattr(self.mtp, "activate_for_uids"):
                self.mtp.activate_for_uids(uids)
            if len(uids) == 1:
                return [], self._speculative_next(uids[0])
            return [], self._speculative_next_batch(uids)

        # Fallback: standard non-spec path. Drop prefill flags so the
        # next BS=1 idle window re-captures from a clean forward.
        result = super(MTPBatchGenerator, self)._next(*args, **kwargs)
        for uid in self._generation_batch.uids:
            self._mtp_prefilled.discard(uid)
        return result

    def _first_step_and_capture_batch(
        self, uids: "Sequence[int]"
    ):
        """Run a standard decode step at BS=N and stash per-uid pre_norms.

        The wrapped final-norm captures ``(B, L, hidden)``; we slice
        the last position per batch entry into the per-uid dict.
        Prefilled uids whose pre_norm was already captured stay in
        the dict; freshly-prefilled ones get added.

        At BS>1 entry, also upgrades each ``BatchRotatingKVCache`` in
        ``gen_batch.prompt_cache`` to :class:`PerStreamBatchRotatingKVCache`
        in-place (class pointer swap; state is preserved). From this
        point forward writes are per-stream physical and rollback is
        per-stream — what makes spec at BS>1 actually work without
        the min-strategy regression.
        """
        prompt_responses, generation_responses = super(
            MTPBatchGenerator, self
        )._next()
        if not generation_responses:
            return prompt_responses, generation_responses

        gen_batch = self._generation_batch
        if len(gen_batch) > 1:
            for c in gen_batch.prompt_cache:
                _upgrade_cache_to_per_stream(c)
            # ── DEGEN PROBE (main prompt_cache): the MTP-cache probe in
            # activate_for_uids measures the DRAFT cache, whose scramble is
            # harmless (target-verified). THIS measures the SHARED prompt_cache
            # the TARGET verify-forward reads — the only cache whose
            # per-stream-layout corruption could actually poison output and
            # cause degeneration. For each batch entry, extract its single and
            # compare phys_rows vs size() (the scramble tell). Logged per
            # BS>1 step so we can correlate an inconsistent MAIN cache with a
            # degeneration on the same uid/time. Zero cost when probe OFF.
            if _DEGEN_PROBE_ENABLED:
                try:
                    _mc_meta: list[dict[str, Any]] = []
                    _uids_now = list(gen_batch.uids)
                    for _bi, _u in enumerate(_uids_now):
                        # Inspect the first cache layer's per-stream view for
                        # this batch index (representative of the layout;
                        # all layers share offset/_idx bookkeeping).
                        _layer0: Any = gen_batch.prompt_cache[0]
                        _sub: Any = (
                            _layer0.caches[0]
                            if hasattr(_layer0, "caches") else _layer0
                        )
                        _off_arr = getattr(_sub, "offset", None)
                        try:
                            _off = (
                                int(_off_arr[_bi].item())
                                if _off_arr is not None else -1
                            )
                        except Exception:
                            _off = -1
                        _maxsz = int(getattr(_sub, "max_size", -1))
                        _valid = (
                            min(_off, _maxsz)
                            if _off >= 0 and _maxsz > 0 else -1
                        )
                        _keys = getattr(_sub, "keys", None)
                        _phys = (
                            int(_keys.shape[2]) if _keys is not None else 0
                        )
                        _mc_meta.append({
                            "uid": int(_u),
                            "batch_idx": _bi,
                            "offset": _off,
                            "valid_expected": _valid,
                            "ring_phys": _phys,
                            "rotated": bool(getattr(_sub, "rotated", False)),
                            "perstream_idx": int(getattr(_sub, "_idx", -1)),
                        })
                    _degen_probe_write({
                        "event": "main_cache_bs_transition",
                        "wall_ns": time.perf_counter_ns(),
                        "uids": _uids_now,
                        "main_cache": _mc_meta,
                    })
                except Exception as _mc_err:
                    _degen_probe_write({
                        "event": "main_cache_probe_err",
                        "err": str(_mc_err),
                    })

        decode_pre_norm = self._captured.get("pre_norm")
        if decode_pre_norm is not None:
            mx.eval(decode_pre_norm)
            B = decode_pre_norm.shape[0]
            for b_idx, uid in enumerate(gen_batch.uids):
                if b_idx >= B:
                    break
                self._mtp_pre_norm[uid] = decode_pre_norm[b_idx : b_idx + 1, -1:, :]
                mx.eval(self._mtp_pre_norm[uid])
                self._mtp_prefilled.add(uid)
        return prompt_responses, generation_responses

    def _speculative_next_batch(self, uids: "Sequence[int]"):
        """One verify/accept cycle at BS>1 using min-acceptance strategy.

        For each uid, draft γ tokens chained through a separate MTP
        cache. Stack drafts into a (N, γ) batch; build the verify
        input (N, γ+1) by prepending each uid's last sampled token;
        run a single batched verify forward; per-uid acceptance check;
        roll back caches by ``γ - n_min`` where ``n_min`` is the
        minimum acceptance count across uids; yield ``n_min + 1``
        tokens per uid (extras buffered for the parent's next-call
        contract).
        """
        gen_batch = self._generation_batch
        N = len(uids)
        gamma = self.gamma
        sync_group = self._get_sharding_group()
        # Coord subgroup for ALL non-model-forward collectives in the
        # spec cycle (broadcast_from_canonical for draft / n_accepted
        # / bonus sync). Keeps these small int32 collectives off the
        # model TP next_call_id_ counter — the c=2 race fix from
        # 2026-05-07. See get_coord_group.
        coord_group = get_coord_group(sync_group)
        sync_drafts = sync_group is not None and sync_group.size() > 1

        # PER-STREAM TEMP (2026-06-18 root-cause fix for mixed-temp degeneration).
        # Hermes runs the main turn and its aux tasks at DIFFERENT temps (e.g.
        # main=1.0, title/utility=0.3); they rendezvous-batch together (BS>1).
        # The earlier fix collapsed the batch to a single scalar temp = MAX of
        # the streams — so a 0.3 stream got its drafts sampled AND its verify
        # accept-test run at 1.0, accepting tokens the 0.3 distribution would
        # never pick → incoherent (non-cyclic) output that the repetition
        # kill-switch does not catch. Fix: carry the per-stream temp through the
        # draft, the accept ratio (p and q MUST use the same per-stream temp for
        # the rejection-sampling guarantee), and the bonus/correction sample.
        # stream_temps[n] is uid n's own temp; _tvec is the (N,1) broadcast form
        # used for the batched draft sampling (clamped to >0 there).
        stream_temps = [self._request_temp.get(u, self.temp) for u in uids]
        # The pure-greedy fast path is taken only when EVERY stream is temp==0.
        # Any positive temp → sampling branch, where each stream uses its OWN
        # temp (a temp==0 stream inside a mixed batch is handled per-stream).
        all_greedy = all(t == 0 for t in stream_temps)
        _tvec = mx.array(
            [[max(t, 1e-6)] for t in stream_temps], dtype=mx.float32
        )  # (N,1) for per-stream draft scaling; clamp avoids div-by-zero

        # Verify each uid has the prerequisites.
        ys: list[mx.array] = []
        y_logprobs_list: list[Any] = []
        pre_norms: list[mx.array] = []
        for uid in uids:
            y = gen_batch._next_tokens
            if y is None or not gen_batch._next_logprobs:
                # Fallback: parent's standard path.
                return gen_batch.next()
            pre = self._mtp_pre_norm.get(uid)
            if pre is None:
                return gen_batch.next()
            pre_norms.append(pre)

        # gen_batch._next_tokens is shape (N,) — the tokens just
        # sampled by the previous step (one per uid, in uids order).
        next_tokens_arr = gen_batch._next_tokens.reshape(N, 1)  # (N, 1)
        # gen_batch._next_logprobs is a list of length N.
        y_logprobs_list = list(gen_batch._next_logprobs)

        prof = _phase_timer
        if prof is not None:
            t_cycle_start = time.perf_counter()

        # 1. Draft γ tokens — chained, batched across uids.
        # Stack pre_norms into (N, 1, hidden_size).
        stacked_pre_norm = mx.concatenate(pre_norms, axis=0)  # (N, 1, hidden)
        draft_ids_list, draft_probs_list = self._draft_tokens_batched(
            stacked_pre_norm, next_tokens_arr, gamma, _tvec, all_greedy
        )
        # draft_ids_list[i] is mx.array shape (N,) — uid b's i-th draft.
        # draft_probs_list[i] is mx.array shape (N, vocab) at temp>0, else None.

        if prof is not None:
            mx.eval(*draft_ids_list)
            t_after_draft = time.perf_counter()
            prof.record("draft", (t_after_draft - t_cycle_start) * 1000.0)

        # 2. Build verify input (N, γ+1) and run a single batched
        #    target forward. Cache rollback uses trim() afterwards.
        draft_concat = mx.concatenate(
            [d.reshape(N, 1) for d in draft_ids_list], axis=1
        )  # (N, γ)
        verify_input = mx.concatenate(
            [next_tokens_arr, draft_concat], axis=1
        )  # (N, γ+1)
        # Snapshot the compressed POOL state before the verify forward.
        # The γ+1 verify pushes draft tokens through accumulate_windows +
        # update_and_fetch on each BatchPoolingCache; on a window flush this
        # bakes draft-derived summaries into the long-range pool. Rejected
        # drafts must be rolled back or the sparse indexer retrieves
        # contaminated blocks → coherent-but-wrong output (needle miss at
        # 100K) even though the local rotating cache is rolled back. The
        # local ring fix (mlx-lm 5533423) cleared the gross BOS-spam; this
        # clears the residual long-range retrieval corruption. Snapshot is
        # cheap (a few list copies + the small ratio-wide buf tensors).
        _pool_caches = self._collect_pooling_caches(gen_batch)
        _pool_snaps = [
            pc.save_meta() if hasattr(pc, "save_meta") else None
            for pc in _pool_caches
        ]
        if _SPEC_CACHE_ROLLBACK_C2:
            for pc in _pool_caches:
                if hasattr(pc, "arm_spec_stash"):
                    pc.arm_spec_stash()

        verify_pre_norm, verify_logits = dsv4_speculative_forward(
            self.model,
            verify_input,
            gen_batch.prompt_cache,
            self._captured,
        )
        # verify_pre_norm: (N, γ+1, hidden), verify_logits: (N, γ+1, vocab)
        if _SPEC_CACHE_ROLLBACK_C2:
            # Disarm before any other forward (the commit-forward fallback
            # must not append to the stash); stashed rows stay bound for
            # the rollback below.
            for pc in _pool_caches:
                if hasattr(pc, "disarm_spec_stash"):
                    pc.disarm_spec_stash()

        if prof is not None:
            mx.eval(verify_pre_norm, verify_logits)
            t_after_verify = time.perf_counter()
            prof.record("verify", (t_after_verify - t_after_draft) * 1000.0)

        # 3. Per-uid acceptance check (min-strategy).
        target_tokens = mx.argmax(
            verify_logits[:, :gamma, :], axis=-1
        )  # (N, γ)

        if all_greedy:
            logprobs_all = verify_logits - mx.logsumexp(
                verify_logits, axis=-1, keepdims=True
            )  # (N, γ+1, vocab)
            if _ACCEPT_LOGPROBS:
                # Same decision rule as the MTP-off generator: argmax over
                # bf16-normalized logprobs (see _ACCEPT_LOGPROBS above).
                target_tokens = mx.argmax(logprobs_all[:, :gamma, :], axis=-1)
                all_next = mx.argmax(logprobs_all, axis=-1)  # (N, γ+1)
            else:
                all_next = mx.argmax(verify_logits, axis=-1)  # (N, γ+1)
            matches = mx.equal(target_tokens, draft_concat)  # (N, γ)
            mx.async_eval(matches, all_next, logprobs_all, verify_pre_norm)
            # Per-uid n_accepted = first-mismatch index.
            matches_arr = matches.tolist()  # list[list[bool]] (N, γ)
            target_arr = cast(list[list[int]], target_tokens.tolist())
            n_accepted_per: list[int] = []
            for n in range(N):
                k = 0
                while k < gamma and matches_arr[n][k]:
                    k += 1
                n_accepted_per.append(k)

            # Compute LOCAL bonus_vals using LOCAL n_accepted_per
            # (pre-broadcast). bonus_lps stay local — they're only
            # used for response.logprobs (informational; master picks
            # rank 0 only).
            draft_int = draft_concat.tolist()
            all_next_arr = all_next.tolist()
            next_tokens_int = next_tokens_arr.reshape(N).tolist()
            bonus_vals: list[int] = []
            bonus_lps: list[Any] = []
            for n in range(N):
                acc = n_accepted_per[n]
                bonus_vals.append(int(all_next_arr[n][acc]))
                bonus_lps.append(logprobs_all[n, acc])

            # Cross-rank n_accepted_per + bonus_vals broadcast —
            # UNCONDITIONAL, COMBINED. Earlier "matches at temp=0 is
            # bit-exact across ranks" assumption is wrong: MLX's TP
            # verify forward has ~1ulp drift in verify_logits;
            # tied/near-tied positions flip argmax across ranks →
            # matches diverges → n_accepted diverges → per-uid yield
            # count diverges → cross-rank _num_tokens drift (papered
            # over by next-cycle all_max but per-rank _token_buffer
            # deque depths stay drifted) → wedge at BS-transition.
            # Trace 2026-05-08 confirmed divergence on uid 5
            # num_tokens by 1 within same cycle.
            #
            # Combined into a single 2N-int32 broadcast: saves one
            # ACK barrier round-trip per cycle (~50-100µs on coord QP).
            if sync_drafts:
                combined_arr = broadcast_from_canonical(
                    mx.array(n_accepted_per + bonus_vals, dtype=mx.int32),
                    coord_group,
                )
                combined = cast(list[int], combined_arr.tolist())
                n_accepted_per = combined[:N]
                bonus_vals = combined[N:]

            # BS>1 min-acceptance clamp (see _BS_MIN_ACCEPT). A stream cut
            # below its own acceptance stages its ACCEPTED draft at n_min
            # as next cycle's token (at greedy that position was a match,
            # so the draft IS the target argmax); the min stream keeps its
            # canonical bonus. Deterministic on post-broadcast data, so
            # rank-consistent.
            if N > 1 and _BS_MIN_ACCEPT:
                _n_min_c = min(n_accepted_per)
                bonus_vals = [
                    int(draft_int[n][_n_min_c])
                    if n_accepted_per[n] > _n_min_c
                    else bonus_vals[n]
                    for n in range(N)
                ]
                bonus_lps = [logprobs_all[n, _n_min_c] for n in range(N)]
                n_accepted_per = [_n_min_c] * N

            # Build all_tokens_per using canonical n_accepted_per so
            # accepted-draft prefix length matches across ranks.
            all_tokens_per: list[list[tuple[int, mx.array]]] = []
            for n in range(N):
                acc = n_accepted_per[n]
                row: list[tuple[int, mx.array]] = [
                    (next_tokens_int[n], y_logprobs_list[n])
                ]
                for k in range(acc):
                    row.append((int(draft_int[n][k]), logprobs_all[n, k]))
                all_tokens_per.append(row)

            self._jaccl_dump_spec(
                uids,
                target_arr,
                cast(list[list[int]], draft_int),
                cast(list[list[bool]], matches_arr),
                n_accepted_per,
                cast(list[int], next_tokens_int),
                bonus_vals,
            )
        else:
            # Temp>0 stochastic acceptance, per-stream.
            #
            # Per-rank divergence sources at temp>0:
            #   * draft_probs_list[i] (q) is per-rank — same MLX numerical
            #     drift as drafts (~1ulp on softmax of MTP logits).
            #   * mx.random.uniform / mx.random.categorical are per-rank
            #     RNG, so samples differ across ranks even with bit-exact
            #     inputs.
            # ⇒ accept_ratios diverge (q drift) AND comparison with
            # uniforms diverges (RNG mismatch), so per-rank `k` differs.
            # We compute `k` per-rank then broadcast rank-0's value via
            # masked all_sum; same trick for the bonus sample which is
            # also drawn per-rank. After the broadcast all ranks build
            # all_tokens_per / bonus_vals from identical inputs.
            logprobs_all = verify_logits - mx.logsumexp(
                verify_logits, axis=-1, keepdims=True
            )
            k_local: list[int] = []
            _accept_probe = self._spec_trace_enabled
            _probe_rows: list[dict[str, Any]] = []
            for n, uid in enumerate(uids):
                # Per-stream temp: this uid's verify distribution MUST be scaled
                # by ITS OWN temp (same as its draft q) or the accept ratio is
                # garbage and the rejection-sampling guarantee breaks.
                _tn = max(stream_temps[n], 1e-6)
                accept_ratios: list[mx.array] = []
                for i in range(gamma):
                    p = mx.softmax(verify_logits[n, i] / _tn, axis=-1)
                    q = draft_probs_list[i]
                    p_di = p[draft_concat[n, i]]
                    q_di = q[n, draft_concat[n, i]]
                    ratio = p_di / mx.maximum(q_di, 1e-10)
                    accept_ratios.append(mx.minimum(ratio**self.alpha, 1.0))
                uniforms = mx.random.uniform(shape=(gamma,))
                # Structured-output accept gate (see single-uid path): at a
                # confident target position accept the draft ONLY if it is the
                # target argmax, else stop — prevents a stochastically-accepted
                # non-argmax token from corrupting a rigid DSML block.
                k = 0
                while k < gamma:
                    _confk = self._mtp_confident_argmax(verify_logits[n, k])
                    if _confk is not None:
                        if int(draft_concat[n, k].item()) == _confk:
                            k += 1
                            continue
                        break
                    if uniforms[k].item() < accept_ratios[k].item():
                        k += 1
                        continue
                    break
                k_local.append(k)
                # ACCEPT PROBE (EXO_DSV4_SPEC_TRACE=1): per-stream per-draft
                # p_di / q_di / accept_ratio / uniform / drafted-token. Lets a
                # downstream diff see whether the residual BS>1 temp>0 repetition
                # comes from inflated accept ratios (p/q mismatch) on the
                # prompt-echo tokens. Rank-0 only; trace-gated, zero hot-path cost.
                if _accept_probe:
                    _sg = self._get_sharding_group()
                    if _sg is None or _sg.rank() == 0:
                        for i in range(gamma):
                            _di = int(draft_concat[n, i].item())
                            _probe_rows.append({
                                "uid": int(uid),
                                "n": n,
                                "i": i,
                                "draft_tok": _di,
                                "p_di": float(
                                    mx.softmax(
                                        verify_logits[n, i] / _tn,
                                        axis=-1,
                                    )[draft_concat[n, i]].item()
                                ),
                                "q_di": float(
                                    draft_probs_list[i][n, draft_concat[n, i]].item()
                                ),
                                "accept_ratio": float(accept_ratios[i].item()),
                                "uniform": float(uniforms[i].item()),
                                "accepted": i < k,
                            })
            if _accept_probe and _probe_rows:
                try:
                    import json as _json
                    _ap_path = f"/tmp/dsv4_accept_probe_pid{os.getpid()}.jsonl"
                    with open(_ap_path, "ab") as _apf:
                        _apf.write(
                            (_json.dumps({
                                "cycle": int(self._spec_cycles),
                                "rows": _probe_rows,
                            }) + "\n").encode("utf-8")
                        )
                except Exception:
                    pass

            # Compute LOCAL bonus_local using LOCAL k_local (per-rank
            # random sample at per-rank acceptance index), then combine
            # n_accepted_per + bonus_vals into ONE broadcast — saves
            # one ACK barrier round-trip per cycle vs the prior two
            # separate broadcasts.
            # MTP SAMPLING PARITY — BATCH PATH. The per-stream bonus/correction
            # is routed through _mtp_filter_logits (rep_pen → top_p → min_p) so
            # it draws from the same truncated, repetition-damped distribution
            # as the main sampler. Per-rank determinism preserved: the filter is
            # deterministic on identical (logits, recent-token) inputs and only
            # rank-0's broadcast bonus_vals commit (combined broadcast below).
            bonus_lps = []
            bonus_local: list[int] = []
            next_tokens_int_t = next_tokens_arr.reshape(N).tolist()
            draft_concat_int = draft_concat.tolist()
            for n in range(N):
                uid_n = uids[n]
                _tn = max(stream_temps[n], 1e-6)  # this stream's own temp
                k = k_local[n]
                if k == gamma:
                    # FULL ACCEPTANCE — the replacement is the genuine BONUS
                    # token sampled from the target p at the post-draft
                    # position. Structured-output guard first: if the target is
                    # highly confident (DSML tag etc.), commit argmax so a tail
                    # draw can't corrupt the block. Else full sampling parity.
                    _cb = self._mtp_confident_argmax(verify_logits[n, k])
                    if _cb is not None:
                        bonus_local.append(_cb)
                    else:
                        _bl = verify_logits[n, k] * (1.0 / _tn)
                        _bl = self._mtp_filter_logits(_bl, uid_n)
                        bonus_local.append(int(mx.random.categorical(_bl).item()))
                else:
                    # REJECTION at position k — the replacement MUST be drawn
                    # from the RESIDUAL distribution max(p - q, 0), NOT raw p.
                    # This is the Leviathan/Chen distribution-preserving
                    # speculative-sampling correction the single-uid path does
                    # (~2630-2643); the batch path previously sampled raw p
                    # here, which is a genuine correctness defect (the corrected
                    # sample was biased toward the draft proposal q). q is the
                    # per-stream draft proposal at step k: draft_probs_list is
                    # per-draft-step, each entry shape (N, vocab) at temp>0.
                    #
                    # NOTE (2026-06-17): this is a real correctness hardening but
                    # it does NOT fix the deterministic BS=2 period-6 degeneration
                    # loop. Trace analysis (cycles 1177-1185) shows that loop runs
                    # at FULL acceptance (k==gamma) with draft==target_argmax — the
                    # target verify_logits argmax IS the loop — so this rejection
                    # branch is off the degeneration path. Root cause is elsewhere
                    # (suspected per-stream verify mask/offset in the L>1 batched
                    # forward); investigation ongoing.
                    # Structured-output guard: confident target → commit argmax.
                    _cr = self._mtp_confident_argmax(verify_logits[n, k])
                    if _cr is not None:
                        bonus_local.append(_cr)
                    else:
                        p = mx.softmax(verify_logits[n, k] / _tn, axis=-1)
                        q = draft_probs_list[k][n]
                        residual = mx.maximum(p - q, 0.0)
                        # Sample from the residual in LOG space, routed through
                        # the same parity filter (rep_pen → top_p → min_p) as
                        # the main sampler so the correction token is also
                        # repetition-damped and tail-clipped. log(residual)
                        # keeps the residual's shape; the filter re-weights/masks.
                        _rl = mx.log(residual + 1e-10)
                        _rl = self._mtp_filter_logits(_rl, uid_n)
                        bonus_local.append(
                            int(mx.random.categorical(_rl).item())
                        )
                bonus_lps.append(logprobs_all[n, k])

            if sync_drafts:
                combined_arr = broadcast_from_canonical(
                    mx.array(k_local + bonus_local, dtype=mx.int32),
                    coord_group,
                )
                combined = cast(list[int], combined_arr.tolist())
                n_accepted_per = combined[:N]
                bonus_vals = combined[N:]
            else:
                n_accepted_per = k_local
                bonus_vals = bonus_local

            # BS>1 min-acceptance clamp (see _BS_MIN_ACCEPT). A stream cut
            # below its own acceptance stages its ACCEPTED draft at n_min
            # as next cycle's token (that position passed its rejection
            # test, so committing it preserves the sampling distribution);
            # the min stream keeps its canonical correction/bonus sample.
            if N > 1 and _BS_MIN_ACCEPT:
                _n_min_c = min(n_accepted_per)
                bonus_vals = [
                    int(draft_concat_int[n][_n_min_c])
                    if n_accepted_per[n] > _n_min_c
                    else bonus_vals[n]
                    for n in range(N)
                ]
                bonus_lps = [logprobs_all[n, _n_min_c] for n in range(N)]
                n_accepted_per = [_n_min_c] * N

            # Build all_tokens_per using canonical n_accepted_per.
            all_tokens_per = []
            for n in range(N):
                k = n_accepted_per[n]
                row: list[tuple[int, mx.array]] = [
                    (int(next_tokens_int_t[n]), y_logprobs_list[n])
                ]
                for kk in range(k):
                    row.append(
                        (int(draft_concat_int[n][kk]), logprobs_all[n, kk])
                    )
                all_tokens_per.append(row)

        # Record per-stream MTP acceptance for telemetry. One sample
        # per uid per cycle; histogram bins each stream's per-cycle
        # acceptance count, and totals reflect cycle×stream samples.
        for _n_acc in n_accepted_per:
            self._record_acceptance(_n_acc)

        # Update each uid's recent-token ring with the tokens committed THIS
        # cycle (post-broadcast, canonical → identical across ranks) so the
        # repetition penalty in _mtp_filter_logits sees real generation history.
        if self._mtp_penalties_active:
            for n, uid in enumerate(uids):
                self._mtp_record_tokens(
                    uid, [int(tid) for (tid, _lp) in all_tokens_per[n]]
                )

        if prof is not None:
            t_after_accept = time.perf_counter()
            prof.record("accept", (t_after_accept - t_after_verify) * 1000.0)

        # 4. Per-stream cache rollback. Each stream b rolls back by
        #    γ - n_accepted_per[b]. Pass the Python int list straight
        #    to trim_per_stream so it does its arithmetic without
        #    syncing self.offset — at 43+ layers that saves ~6ms per
        #    spec cycle on cluster.
        #
        # ASYNC-FENCE DRAIN (2026-07-03 c=2 deep-degeneration root fix).
        # With EXO_DSV4_FENCE_ASYNC_C2 armed, the verify forward's
        # per-layer commits are mx.async_eval. mx.eval(verify_logits)
        # only waits for the LOGITS' dependency chain — pool/indexer
        # side-chain writes consumed by FUTURE forwards (pooled
        # summaries, indexer state) can still be in flight here. The
        # rollback below then mutates those same buffers
        # (trim_per_stream ring scatters, restore_meta, pool trims) and
        # races the deferred graph → stochastically corrupted pooled
        # state → deep-generation repetition loops (' *'/' **'/' his',
        # onset 1400-3900 tok, ~30%/4000-tok pair) and occasionally a
        # rank-desynced collective wedge. A single drain per cycle here
        # keeps the intra-forward async pipelining (the +25% c=2 win)
        # while making the mutations safe. B=1 is unaffected: its trims
        # are pure offset decrements (no buffer mutation), which is why
        # c=1 async has always been clean. Re-armed after the MTP-cache
        # trim below. (_set_fence_async lives on the predictor module.)
        self.mtp._set_fence_async(False)
        rollback_per_stream_py = [gamma - acc for acc in n_accepted_per]
        n_min = min(n_accepted_per)
        n_min_rollback = gamma - n_min  # uniform amount for non-per-stream caches
        any_rejection = any(acc < gamma for acc in n_accepted_per)

        # Pool-consistency discipline, mirroring the single-uid path
        # (2026-07-02 c=2 degeneration ROOT CAUSE). The old batched flow
        # trimmed the caches and then UNCONDITIONALLY restore_meta'd the
        # pools to their pre-verify snapshot on any rejection — assuming
        # committed tokens would "re-enter naturally" next cycle. They
        # never do (only [bonus, new drafts] enter the next forward), so
        # every rejecting cycle silently deleted the committed prefix
        # [y, *accepted] from the pool token stream. The pooled summaries
        # then desync from true positions, the compressed/indexer
        # attention reads shifted blocks, and output degenerates into the
        # prompt-echo/repetition loop within a few cycles.
        #
        # Correct discipline (same as the c=1 path at ~3404):
        #   (a) no pool flushed during verify -> rejected rows still sit
        #       in the pool remainder tail; plain trim removes exactly
        #       them (remainder grew by γ+1 >= rollback this cycle).
        #   (b) a pool flushed -> the flush may have baked rejected
        #       drafts into pooled summaries; restore pools to the
        #       pre-verify snapshot, roll back ALL γ+1 verify rows from
        #       every cache, and re-add the committed prefix with a small
        #       plain commit-forward. Under BS>1 min-acceptance the
        #       committed rows are batch-uniform (n_min + 1), so the
        #       commit-forward is a single (N, n_min+1) batched call.
        pool_flushed = any_rejection and any(
            snap is not None and _pool_flushed_since(pc, snap)
            for pc, snap in zip(_pool_caches, _pool_snaps, strict=True)
        )

        # Cache-level pool undo (see _SPEC_CACHE_ROLLBACK_C2): when every
        # pool can roll back at the cache level, take path (a)'s validated
        # per-stream trims for rings/other caches and spec_rollback the
        # pools — no commit-forward. keep is batch-uniform (min-acceptance).
        _c2_cache_level = False
        if pool_flushed and _SPEC_CACHE_ROLLBACK_C2:
            _c2_keep = n_min + 1
            _c2_pushed = gamma + 1
            _c2_cache_level = all(
                snap is not None
                and hasattr(pc, "spec_can_rollback")
                and pc.spec_can_rollback(snap, _c2_keep, _c2_pushed)
                for pc, snap in zip(_pool_caches, _pool_snaps, strict=True)
            )
            if _c2_cache_level:
                for c in gen_batch.prompt_cache:
                    subs = c.caches if isinstance(c, CacheList) else [c]
                    for sub in subs:
                        if isinstance(sub, PerStreamBatchRotatingKVCache):
                            sub.trim_per_stream(rollback_per_stream_py)
                        elif any(sub is pc for pc in _pool_caches):
                            continue  # handled by spec_rollback below
                        elif hasattr(sub, "trim"):
                            sub.trim(n_min_rollback)
                        elif hasattr(sub, "offset"):
                            sub.offset -= n_min_rollback
                for pc, snap in zip(_pool_caches, _pool_snaps, strict=True):
                    pc.spec_rollback(snap, _c2_keep)

        if _c2_cache_level:
            pass  # rollback fully handled above
        elif not pool_flushed:
            # (a) Cheap path (also the full-acceptance path): per-stream
            # rotating trim + uniform pool remainder trim.
            for c in gen_batch.prompt_cache:
                if isinstance(c, CacheList):
                    for sub in c.caches:
                        if isinstance(sub, PerStreamBatchRotatingKVCache):
                            sub.trim_per_stream(rollback_per_stream_py)
                        elif hasattr(sub, "trim"):
                            sub.trim(n_min_rollback)
                        elif hasattr(sub, "offset"):
                            sub.offset -= n_min_rollback
                elif isinstance(c, PerStreamBatchRotatingKVCache):
                    c.trim_per_stream(rollback_per_stream_py)
                elif hasattr(c, "trim"):
                    c.trim(n_min_rollback)
                elif hasattr(c, "offset"):
                    c.offset -= n_min_rollback
        else:
            # (b) Contamination path: restore snapshotted pools, drop all
            # γ+1 verify rows everywhere, re-add committed rows.
            if not _POOL_RESTORE_AFTER_TRIM:
                # LEGACY ORDER (double-rollback bug, see
                # _POOL_RESTORE_AFTER_TRIM): the blanket trim below re-trims
                # the pools restore_meta just rewound.
                for pc, snap in zip(_pool_caches, _pool_snaps):
                    if snap is not None:
                        pc.restore_meta(snap)
            _full = gamma + 1
            _full_per_stream = [_full] * N
            for c in gen_batch.prompt_cache:
                if isinstance(c, CacheList):
                    for sub in c.caches:
                        if isinstance(sub, PerStreamBatchRotatingKVCache):
                            sub.trim_per_stream(_full_per_stream)
                        elif hasattr(sub, "trim"):
                            sub.trim(_full)
                        elif hasattr(sub, "offset"):
                            sub.offset -= _full
                elif isinstance(c, PerStreamBatchRotatingKVCache):
                    c.trim_per_stream(_full_per_stream)
                elif hasattr(c, "trim"):
                    c.trim(_full)
                elif hasattr(c, "offset"):
                    c.offset -= _full
            if _POOL_RESTORE_AFTER_TRIM:
                # FIXED ORDER: restore AFTER the blanket trim so snapshotted
                # pools enter the commit-forward at exactly their pre-verify
                # state (see _POOL_RESTORE_AFTER_TRIM).
                for pc, snap in zip(_pool_caches, _pool_snaps, strict=True):
                    if snap is not None:
                        pc.restore_meta(snap)
            # Commit-forward: rows are the committed tokens per stream,
            # batch-uniform length n_min + 1 (min-acceptance).
            commit_rows = [
                [int(tid) for (tid, _lp) in all_tokens_per[n]]
                for n in range(N)
            ]
            commit_input = mx.array(commit_rows, dtype=mx.int32)  # (N, n_min+1)
            _commit_logits = self.model(
                commit_input, cache=gen_batch.prompt_cache
            )
            mx.eval(_commit_logits)
            del _commit_logits

        # MTP-side cache: also per-stream when possible.
        mtp_cache = self.mtp._cache
        if mtp_cache is not None:
            if isinstance(mtp_cache, PerStreamBatchRotatingKVCache):
                mtp_cache.trim_per_stream(rollback_per_stream_py)
            elif hasattr(mtp_cache, "trim"):
                mtp_cache.trim(n_min_rollback)
            elif hasattr(mtp_cache, "offset"):
                mtp_cache.offset -= n_min_rollback

        # Re-arm the async fence for the next cycle's forwards (same
        # B-gate as activate_for_uids). All cache mutations for this
        # cycle are done.
        self.mtp._set_fence_async(N <= _FENCE_ASYNC_MAX_STREAMS)

        # ── BATCHED REFERENCE-FORWARD REFCHECK (env-gated diagnostic) ──
        # EXO_DSV4_MTP_REFCHECK_BATCH=<jsonl path>: every cycle, trim the
        # last committed token from every stream and re-feed it as an
        # (N, 1) batched L=1 forward — the non-spec batched decode shape
        # that is clean at depth (MTP-off battery 4/4). Its logits are
        # P(next | committed prefix) per stream. Compare argmax against
        # the verify forward's bonus row (verify_logits[n, acc]) — a
        # disagreement pinpoints the FIRST cycle the B=2 L>1 verify (or
        # its async-fence interplay) diverges from a clean forward, and
        # the top-2 margins say HOW wrong (1ulp tie flip vs gross).
        # TP SAFETY: the trigger is the env var alone, so every rank runs
        # the identical trim + forward (collective) + refeed; only rank 0
        # writes. Runs AFTER the fence re-arm so production arming (and
        # the race under investigation) stays live during the refeed.
        # trim(1)+refeed is the same primitive the pool-flush commit
        # forward relies on; cycles where a pool flushed are tagged (the
        # refeed may cross a flush boundary and add noise).
        _refcheck_batch_path = os.environ.get("EXO_DSV4_MTP_REFCHECK_BATCH")
        if _refcheck_batch_path and not getattr(
            self, "_refcheck_batch_err_logged", False
        ):
            try:
                _rc_ones = [1] * N
                for c in gen_batch.prompt_cache:
                    if isinstance(c, CacheList):
                        for sub in c.caches:
                            if isinstance(sub, PerStreamBatchRotatingKVCache):
                                sub.trim_per_stream(_rc_ones)
                            elif hasattr(sub, "trim"):
                                sub.trim(1)
                            elif hasattr(sub, "offset"):
                                sub.offset -= 1
                    elif isinstance(c, PerStreamBatchRotatingKVCache):
                        c.trim_per_stream(_rc_ones)
                    elif hasattr(c, "trim"):
                        c.trim(1)
                    elif hasattr(c, "offset"):
                        c.offset -= 1
                _rc_last = [int(all_tokens_per[n][-1][0]) for n in range(N)]
                _rc_input = mx.array([[t] for t in _rc_last], dtype=mx.int32)
                _rc_logits = self.model(_rc_input, cache=gen_batch.prompt_cache)
                _rc_rows = _rc_logits[:, 0, :].astype(mx.float32)
                _rc_top2 = mx.topk(_rc_rows, 2, axis=-1)
                _rc_arg = mx.argmax(_rc_rows, axis=-1)
                _v_rows = mx.stack(
                    [verify_logits[n, n_accepted_per[n]] for n in range(N)]
                ).astype(mx.float32)
                _v_top2 = mx.topk(_v_rows, 2, axis=-1)
                _v_arg = mx.argmax(_v_rows, axis=-1)
                mx.eval(_rc_arg, _rc_top2, _v_arg, _v_top2)
                _rc_arg_l = [int(x) for x in _rc_arg.tolist()]
                _v_arg_l = [int(x) for x in _v_arg.tolist()]
                _rc_cyc = getattr(self, "_refcheck_batch_cycle", 0) + 1
                self._refcheck_batch_cycle = _rc_cyc
                _sg = self._get_sharding_group()
                if (_sg is None or _sg.rank() == 0) and (
                    _rc_arg_l != _v_arg_l or _rc_cyc % 100 == 1
                ):
                    _rc_t2 = cast(list[list[float]], _rc_top2.tolist())
                    _v_t2 = cast(list[list[float]], _v_top2.tolist())
                    with open(_refcheck_batch_path, "a") as _rcf:
                        _rcf.write(json.dumps({
                            "cycle": _rc_cyc,
                            "uids": list(uids),
                            "n_accepted": list(n_accepted_per),
                            "verify_argmax": _v_arg_l,
                            "ref_argmax": _rc_arg_l,
                            "agree": _rc_arg_l == _v_arg_l,
                            # mx.topk value order is not guaranteed
                            "v_margin": [abs(r[0] - r[1]) for r in _v_t2],
                            "r_margin": [abs(r[0] - r[1]) for r in _rc_t2],
                            "bonus_vals": [int(b) for b in bonus_vals],
                            "pool_flushed": bool(pool_flushed),
                        }) + "\n")
            except Exception as _rc_err:  # diagnostics must never kill decode
                if not getattr(self, "_refcheck_batch_err_logged", False):
                    self._refcheck_batch_err_logged = True
                    logger.warning(f"REFCHECK_BATCH failed (disabled): {_rc_err}")

        # 5. Update per-uid pre_norm to each stream's first-rejection
        #    position in verify_pre_norm.
        for n, uid in enumerate(uids):
            acc = n_accepted_per[n]
            self._mtp_pre_norm[uid] = verify_pre_norm[n : n + 1, acc : acc + 1, :]
            mx.eval(self._mtp_pre_norm[uid])

        # 6. Stage bonus tokens for next call.
        gen_batch._next_tokens = mx.array(bonus_vals)
        gen_batch._next_logprobs = bonus_lps
        mx.async_eval(gen_batch._next_tokens)

        # 6b. Degeneration-hunt trace (EXO_DSV4_SPEC_TRACE=1, rank 0).
        # Post-rollback so cache offsets reflect committed state. target_tokens
        # exists only on the temp=0 path; pass a zero placeholder at temp>0.
        if self._spec_trace_enabled:
            _tt = (
                target_tokens
                if all_greedy
                else mx.zeros_like(draft_concat)
            )
            self._spec_trace_cycle_dump(
                uids,
                gen_batch,
                gamma,
                verify_input,
                draft_concat,
                _tt,
                n_accepted_per,
                bonus_vals,
                all_tokens_per,
            )

        if prof is not None:
            t_after_rollback = time.perf_counter()
            prof.record(
                "rollback", (t_after_rollback - t_after_accept) * 1000.0
            )
            prof.record(
                "total", (t_after_rollback - t_cycle_start) * 1000.0
            )
            prof.end_cycle(N)

        # 7. Bookkeeping.
        total_yielded = sum(len(t) for t in all_tokens_per)
        self._gen_tokens_counter += total_yielded
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
            mx.clear_cache()

        # 8. State machine + length checks per yielded token, per uid.
        responses_per: list[list[Any]] = []
        for n, uid in enumerate(uids):
            idx = gen_batch.uids.index(uid)
            responses_per.append(
                self._build_yielded_responses(uid, idx, all_tokens_per[n])
            )

        # 2026-05-20 c>=2 fix v2 (drain-elimination):
        #
        # Previously this returned only the FIRST response per uid and
        # buffered the remaining γ tokens for later drain via
        # _yield_buffered. At TP c>1 each drain cycle invokes the full
        # _next() framework — including the mx_any(has_work, coord_group)
        # collective at batch_generate.py:1694 and on_generation_token
        # callbacks. Per spec cycle that meant γ extra _next() calls at
        # ~50ms each — drain overhead alone consumed half the c=2 wall.
        #
        # Fix: return ALL responses from this spec cycle in one call.
        # mlx-lm's BatchGenerator.next_generated() returns whatever list
        # we hand back, so multi-response returns are supported.
        # batch_generate._step() iterates `for response in responses` —
        # each response gets full processing (detok, stop check,
        # on_generation_token) regardless of list length.
        #
        # _build_yielded_responses already stops at first finish_reason
        # per uid (line 399-400 in mtp_batch_generator.py), so streams
        # that finish mid-cycle don't yield their post-finish drafts.
        # We still need to call _filter_finished_uid + _cleanup_uid for
        # those uids so the parent generation_batch shrinks correctly.
        all_responses: list[Any] = []
        for n, uid in enumerate(uids):
            row = responses_per[n]
            if not row:
                continue
            all_responses.extend(row)
            # Filter the uid if its LAST yielded response carries a
            # finish_reason. Streams that finished mid-cycle have
            # finish_reason on the last entry in `row` (since
            # _build_yielded_responses breaks immediately after).
            last = row[-1]
            if last.finish_reason is not None:
                self._filter_finished_uid(uid)
                self._cleanup_uid(uid)

        return all_responses

    def _draft_tokens_batched(
        self,
        stacked_pre_norm: mx.array,
        next_tokens_arr: mx.array,
        gamma: int,
        tvec: mx.array,
        all_greedy: bool,
    ) -> tuple[list[mx.array], list[Any]]:
        """γ chained MTP forwards at batch_size=N.

        Returns ``(draft_ids, draft_probs)`` matching the contract of
        :func:`mtp_module.draft_tokens` but with batch dim N.

        Uses ``self.mtp.predict(...)`` once per draft step — each call
        runs the MTP module forward at shape (N, 1).

        Cross-rank determinism: the MTP module is unsharded (full
        weights on every rank, sharding_group=None) so its forward
        SHOULD be bit-exact across ranks given bit-exact inputs. In
        practice (Phase E.1 trace 2026-05-05) MLX produces tiny logit
        drift between ranks at cycle 5+, flipping argmax for tied/
        near-tied positions. At temp>0 the per-rank RNG state also
        produces fully divergent samples even with bit-exact logits.
        Without intervention the divergent draft tokens cascade through
        the verify forward into asymmetric n_accepted_per → asymmetric
        _num_tokens → asymmetric _filter_finished_uid → JACCL LEN_ERR
        cluster wedge AND garbled output even when the wedge is
        bandaged downstream.

        Force determinism by broadcasting rank-0's post-argmax (temp=0)
        or post-categorical (temp>0) token IDs to every rank each step
        (masked all_sum: rank 0 contributes its actual int32 ids, all
        others contribute zeros, all_sum gives every rank rank-0's
        values). Cost is one small int32 collective per draft step
        (~4*N bytes); negligible vs the bf16 model forwards. Memory:
        jaccl_phase_f_outcome.md.
        """
        draft_ids: list[mx.array] = []
        draft_probs: list[Any] = []
        h = stacked_pre_norm
        tok_arr = next_tokens_arr  # (N, 1)
        sync_group = self._get_sharding_group()
        # Coord group for draft broadcasts: keeps the int32 token-sync
        # collectives off the model TP next_call_id_ counter
        # (2026-05-07 c=2 race fix). The collective rate scales with
        # γ × cycles, so isolation matters here. See get_coord_group.
        coord_group = get_coord_group(sync_group)
        sync_drafts = sync_group is not None and sync_group.size() > 1

        drift_dump = os.environ.get("EXO_MTP_DRIFT_DUMP") == "1"

        # C2 bistability tracer — see module-level docstring above the
        # _c2_trace_write definition. We capture per-step timestamps,
        # per-stream post-broadcast tokens, and metal memory deltas to
        # localize the iter-N+1 stream collapse seen at γ≥2 c=2. Each
        # mx.eval() inside the tracer adds a chain-step fence; on
        # purpose, since (a) we need the timestamps to be REAL and not
        # lazy-graph-fill, (b) we WANT the trace to either confirm or
        # falsify whether per-step draining alone fixes bistability.
        _c2_trace = _C2_TRACE_ENABLED and sync_drafts  # cluster-mode only
        _c2_cycle_start_ns = time.perf_counter_ns() if _c2_trace else 0
        _c2_step_walls_ms: list[float] = []
        _c2_per_step_records: list[dict[str, Any]] = []
        _c2_cycle_n = int(self._spec_cycles) if _c2_trace else 0
        _c2_b_size = int(next_tokens_arr.shape[0]) if _c2_trace else 0
        # Pre-init the per-step timestamp scratch slots so static
        # analysis sees them as always-bound across loop iterations.
        # Runtime semantics are unchanged: they're only READ inside
        # the same `if _c2_trace:` branch that WRITES them.
        _c2_t_step_start_ns: int = 0
        _c2_t_after_eagle_install_ns: int = 0
        _c2_t_after_predict_ns: int = 0
        _c2_metal_active_at_start: float = -1.0
        _c2_metal_peak_at_start: float = -1.0

        # Eagle soft-embedding chain (c=2 batched variant). Mirrors the
        # c=1 logic in mtp_module.draft_tokens: when EXO_DSV4_MTP_EAGLE_K>0
        # on the predictor, replace every chained predict() input
        # embedding (i>=1) with a probability-weighted top-K mixture
        # built from the previous step's logits. Soft-emb shape here is
        # (N, 1, hidden) since logits is (N, vocab) squeezed.
        _eagle_k = int(getattr(self.mtp, "eagle_k", 0) or 0)
        _eagle_embed = (
            getattr(self.mtp, "embed_tokens", None) if _eagle_k > 0 else None
        )
        _eagle_active = _eagle_k > 0 and _eagle_embed is not None
        prev_logits: Optional[mx.array] = None

        for i in range(gamma):
            # C2 trace: per-step START timestamp (after any prior-iter
            # eval() drains). NOT a fence — just records when the step
            # actually begins on the Python side.
            if _c2_trace:
                _c2_t_step_start_ns = time.perf_counter_ns()
                _c2_metal_active_at_start, _c2_metal_peak_at_start = (
                    _c2_trace_metal_mb()
                )
            hidden_for_dump = h if drift_dump else None
            _eagle_installed = _eagle_active and prev_logits is not None
            if _eagle_installed:
                # Cross-rank determinism: prev_logits is rank-local and
                # MLX's documented per-rank drift (see comment at lines
                # 1501-1521 above) can flip argmax for tied/near-tied
                # positions, making any rank-local soft-emb diverge
                # between ranks. The hard-embed path masks this via the
                # post-argmax broadcast at line 1586 below; reuse that
                # broadcast (tok_arr was already broadcast at the end of
                # iter i-1) instead of putting another large bf16
                # collective on the chain critical path.
                #
                # Why NOT broadcast_from_canonical(soft_emb, ...): an
                # all_gather on the bf16 (B, 1, hidden=4096) tensor is
                # 16 KB and is the *first* dependency of the next MTP
                # __call__ (it feeds enorm(emb) at deepseek_v4.py:2505).
                # That comm round-trip has zero overlap potential —
                # every i>=1 predict() would stall on a 16 KB UC RDMA
                # round-trip, turning the chain into comm/compute
                # serialization. Empirically verified to cause ~17x
                # slowdown (commit 21ba40db forensics:
                # .hermes/plans/2026-05-22 reports).
                assert prev_logits is not None  # narrow for type-checker
                assert _eagle_embed is not None  # _eagle_active gate
                if _eagle_k == 1:
                    # K=1: soft-emb collapses algebraically to
                    # embed_tokens(argmax(prev_logits)). tok_arr is
                    # exactly broadcast(argmax(prev_logits)) from the
                    # end of iter i-1, so embed_tokens(tok_arr) IS the
                    # K=1 soft-emb, cross-rank-deterministic, with NO
                    # new collective.
                    soft_emb = _eagle_embed(tok_arr)  # (B, 1, hidden)
                else:
                    # K>1: broadcast the tiny topk_ids + topk_probs
                    # tensors (single-digit bytes at K=2, B=2) and
                    # reconstruct the mixture locally on every rank
                    # from identical inputs. Avoids the 16 KB bf16
                    # critical-path collective; payload is dominated
                    # by per-call collective overhead, not bytes.
                    _logits3d = (
                        prev_logits
                        if prev_logits.ndim == 3
                        else prev_logits[:, None, :]
                    )
                    _probs = mx.softmax(_logits3d, axis=-1)
                    _topk_ids = mx.argsort(-_logits3d, axis=-1)[..., :_eagle_k]
                    _topk_probs = mx.take_along_axis(
                        _probs, _topk_ids, axis=-1
                    )
                    _topk_probs = _topk_probs / _topk_probs.sum(
                        axis=-1, keepdims=True
                    )
                    if sync_drafts:
                        _topk_ids = broadcast_from_canonical(
                            _topk_ids.astype(mx.int32), coord_group
                        )
                        _topk_probs = broadcast_from_canonical(
                            _topk_probs, coord_group
                        )
                    _topk_embs = _eagle_embed(_topk_ids)
                    soft_emb = (_topk_embs * _topk_probs[..., None]).sum(
                        axis=-2
                    )
                self.mtp.set_eagle_soft_emb(soft_emb)
            # C2 trace: post-eagle-install boundary. We mx.eval the
            # soft_emb (if installed) so the time spent in the assembly +
            # any K>1 collective is attributable to the install step
            # rather than getting folded into the predict() wall.
            if _c2_trace:
                if _eagle_installed:
                    try:
                        mx.eval(soft_emb)  # type: ignore[has-type]
                    except Exception:
                        pass
                _c2_t_after_eagle_install_ns = time.perf_counter_ns()
            try:
                logits, h = self.mtp.predict(
                    h, tok_arr, return_hidden=True,
                    draft_mode=(
                        getattr(self.mtp, "draft_lm_head", None) is not None
                    ),
                )
            finally:
                if _eagle_installed:
                    self.mtp.set_eagle_soft_emb(None)
            # C2 trace: post-predict boundary. mx.eval(logits, h)
            # forces the MoE forward to actually complete (including all
            # TP all_sum collectives) before we record the timestamp.
            # WITHOUT this eval, the timestamp captures lazy command-
            # buffer fill time only — useless for diagnosing comm tails.
            # WITH this eval, we get true GPU/comm wall per step. The
            # cost (one extra sync per chain step) is the same as the
            # proposed bistability fence and is acceptable here BECAUSE
            # we're diagnosing, not validating.
            if _c2_trace:
                try:
                    mx.eval(logits, h)
                except Exception:
                    pass
                _c2_t_after_predict_ns = time.perf_counter_ns()
            prev_logits = logits
            # logits: at S=1, predict() squeezes to (N, vocab).
            if all_greedy:
                tok_pre_sync = mx.argmax(logits, axis=-1).reshape(-1, 1)
                if sync_drafts:
                    tok_arr = broadcast_from_canonical(
                        tok_pre_sync.astype(mx.int32), coord_group
                    )
                else:
                    tok_arr = tok_pre_sync
                if drift_dump and hidden_for_dump is not None:
                    self._mtp_drift_dump(
                        cycle_n=self._spec_cycles,
                        chain_step=i,
                        hidden_pre=hidden_for_dump,
                        logits=logits,
                        tok_pre_sync=tok_pre_sync,
                        tok_post_sync=tok_arr,
                    )
                draft_ids.append(tok_arr.reshape(-1))
                draft_probs.append(None)
            else:
                # PER-STREAM temp: tvec is (N,1); logits/tvec scales each row by
                # ITS OWN temp so q (and the categorical draw) match the stream's
                # real temperature. A near-zero-temp stream (clamped 1e-6) gets a
                # near-one-hot q → effectively greedy, while a 1.0 stream samples
                # normally — within the same batched draw. q is returned per
                # stream and reused in the accept ratio (same temp → valid).
                q = mx.softmax(logits / tvec, axis=-1)  # (N, vocab)
                tok_pre_sync = mx.random.categorical(
                    logits / tvec
                ).reshape(-1, 1)  # (N, 1)
                if sync_drafts:
                    tok_arr = broadcast_from_canonical(
                        tok_pre_sync.astype(mx.int32), coord_group
                    )
                else:
                    tok_arr = tok_pre_sync
                if drift_dump and hidden_for_dump is not None:
                    self._mtp_drift_dump(
                        cycle_n=self._spec_cycles,
                        chain_step=i,
                        hidden_pre=hidden_for_dump,
                        logits=logits,
                        tok_pre_sync=tok_pre_sync,
                        tok_post_sync=tok_arr,
                    )
                draft_ids.append(tok_arr.reshape(-1))
                draft_probs.append(q)
            # h returned by predict at S=1 has shape (N, 1, hidden) —
            # fed straight into the next predict() call.

            # Per-step fence — direct port of mtp_module.py::draft_tokens:786
            # into the c=2 batched path. Without this, γ chained predict()
            # calls queue γ lazy `all_sum`s (one per MTP MoE forward) into
            # the GPU/comm-stream command buffer; each subsequent all_sum
            # is gated on the previous one's CQE delivery. At γ=2 c=2 100K
            # this manifested as the iter-N+1 stream collapse (bistability);
            # at γ=3 c=2 100K it ALSO produces '<｜begin▁of▁sentence｜>'
            # token spam from iter 1 onward — a quality regression that
            # the pre-c=2-aware quality_probe_dsv4.py never caught because
            # it fired single-request which routes through the c=1
            # mtp_module.draft_tokens path that already has this fence.
            # The c=1 path has been stable BECAUSE of this fence since
            # commit ce61e46b (2026-05-17). The c=2 batched path was
            # missing it; the γ=2 c=2 mitigation in commit 2e708e19 was
            # FENCE_EVERY_N_LAYERS=4 (a verify-side knob, orthogonal to
            # the draft-side jaccl chain). With γ=3 we outran that mask.
            # Cost: one extra GPU sync per chain step (~µs). When
            # EXO_DSV4_C2_TRACE=1 the tracer's later mx.eval(tok_arr,
            # tok_pre_sync) finds tok_arr already evaluated and becomes
            # essentially free.
            if i + 1 < gamma:
                mx.eval(tok_arr)

            # C2 trace: post-broadcast / step-end boundary. mx.eval the
            # tok_arr to force the int32 broadcast collective to drain,
            # AND the pre-sync argmax to drain (it would otherwise be
            # left lazy if sync_drafts is True). At this point all comm
            # work for the step is complete; subsequent steps will start
            # from a clean queue.
            if _c2_trace:
                try:
                    mx.eval(tok_arr, tok_pre_sync)  # type: ignore[has-type]
                except Exception:
                    pass
                _c2_t_step_end_ns = time.perf_counter_ns()
                # Materialize per-stream tokens for the record. tok_arr
                # is shape (N, 1) int32 post-broadcast. tok_pre_sync is
                # the rank-local arg before broadcast.
                try:
                    _c2_tok_post = [
                        int(x) for x in tok_arr.reshape(-1).tolist()
                    ]
                except Exception:
                    _c2_tok_post = []
                try:
                    _c2_tok_pre = [
                        int(x) for x in tok_pre_sync.reshape(-1).tolist()
                    ]
                except Exception:
                    _c2_tok_pre = []
                _c2_active_mb_end, _c2_peak_mb_end = _c2_trace_metal_mb()
                step_wall_ms = (
                    _c2_t_step_end_ns - _c2_t_step_start_ns
                ) / 1e6
                _c2_step_walls_ms.append(step_wall_ms)
                _c2_per_step_records.append({
                    "type": "step",
                    "cycle": _c2_cycle_n,
                    "step": int(i),
                    "B": _c2_b_size,
                    "gamma": int(gamma),
                    "pid": os.getpid(),
                    "ts_step_start_ns": _c2_t_step_start_ns,
                    "ts_after_eagle_install_ns": (
                        _c2_t_after_eagle_install_ns
                    ),
                    "ts_after_predict_ns": _c2_t_after_predict_ns,
                    "ts_step_end_ns": _c2_t_step_end_ns,
                    "wall_step_ms": step_wall_ms,
                    "wall_eagle_install_ms": (
                        _c2_t_after_eagle_install_ns - _c2_t_step_start_ns
                    ) / 1e6,
                    "wall_predict_ms": (
                        _c2_t_after_predict_ns
                        - _c2_t_after_eagle_install_ns
                    ) / 1e6,
                    "wall_argmax_broadcast_ms": (
                        _c2_t_step_end_ns - _c2_t_after_predict_ns
                    ) / 1e6,
                    "eagle_installed": bool(_eagle_installed),
                    "temp_zero": bool(all_greedy),
                    "tok_post_broadcast_per_stream": _c2_tok_post,
                    "tok_pre_broadcast_per_stream": _c2_tok_pre,
                    "metal_active_mb_start": (
                        _c2_metal_active_at_start
                    ),
                    "metal_peak_mb_start": _c2_metal_peak_at_start,
                    "metal_active_mb_end": _c2_active_mb_end,
                    "metal_peak_mb_end": _c2_peak_mb_end,
                })

        # C2 trace: per-cycle summary + flush per-step records.
        if _c2_trace:
            _c2_cycle_end_ns = time.perf_counter_ns()
            _c2_cycle_wall_ms = (
                _c2_cycle_end_ns - _c2_cycle_start_ns
            ) / 1e6
            # Bistability heuristic: any step > 2× the median step wall.
            # At γ=3 a normal cycle should have all 3 steps within ~30%
            # of each other; a step taking 2×+ the others is the iter-N+1
            # collapse fingerprint.
            if _c2_step_walls_ms:
                _c2_walls_sorted = sorted(_c2_step_walls_ms)
                _c2_median_ms = _c2_walls_sorted[len(_c2_walls_sorted) // 2]
                _c2_max_ms = _c2_walls_sorted[-1]
                _c2_bistability = (
                    _c2_max_ms > 2.0 * max(_c2_median_ms, 1e-3)
                )
            else:
                _c2_median_ms = 0.0
                _c2_max_ms = 0.0
                _c2_bistability = False
            for rec in _c2_per_step_records:
                _c2_trace_write(rec)
            _c2_trace_write({
                "type": "cycle",
                "cycle": _c2_cycle_n,
                "pid": os.getpid(),
                "B": _c2_b_size,
                "gamma": int(gamma),
                "ts_cycle_start_ns": _c2_cycle_start_ns,
                "ts_cycle_end_ns": _c2_cycle_end_ns,
                "cycle_wall_ms": _c2_cycle_wall_ms,
                "per_step_wall_ms": list(_c2_step_walls_ms),
                "median_step_wall_ms": _c2_median_ms,
                "max_step_wall_ms": _c2_max_ms,
                "bistability_flag": bool(_c2_bistability),
            })

        return draft_ids, draft_probs

    # ── BS=1 path (preserved) ──────────────────────────────────────────

    def _speculative_next(self, uid: int):
        """One verify/accept cycle, DSv4 flavor.

        Mirrors :meth:`MTPBatchGenerator._speculative_next` but uses:
          * :func:`dsv4_speculative_forward` for the verify pass
            (no GDN handling, no kernel patching)
          * ``cache.trim(rollback)`` on each prompt-cache entry for
            rejected-draft rollback (CacheList recursively trims its
            inner caches)

        Token-tree drafting gate: when EXO_DSV4_TREE_DRAFT=1, route to
        :meth:`_speculative_next_tree` which uses K-way top-K MTP
        expansion + tree-attention verify instead of the linear chain.
        Only temp=0 greedy is supported in tree v1; temp>0 falls back
        to the linear path.
        """
        if TREE_DRAFT:
            temp = self._request_temp.get(uid, self.temp)
            if temp == 0:
                return self._speculative_next_tree(uid)
            # temp>0 fall-through to linear path below.
        gen_batch = self._generation_batch
        idx = gen_batch.uids.index(uid)
        sync_group = self._get_sharding_group()
        # Coord subgroup for non-model-forward collectives in this cycle
        # (draft sync inside draft_tokens, n_accepted broadcast, bonus
        # broadcast). Isolated next_call_id_ counter from the model TP
        # group — c=2 race fix 2026-05-07.
        coord_group = get_coord_group(sync_group)
        sync_drafts = sync_group is not None and sync_group.size() > 1

        y = gen_batch._next_tokens
        if y is None or not gen_batch._next_logprobs:
            return gen_batch.next()

        pre_norm = self._mtp_pre_norm.get(uid)
        if pre_norm is None:
            return gen_batch.next()

        y_val = int(y[0].item())
        y_logprobs = gen_batch._next_logprobs[0]

        gamma = self.gamma
        temp = self._request_temp.get(uid, self.temp)
        alpha = self.alpha

        prof = _phase_timer
        if prof is not None:
            t_cycle_start = time.perf_counter()

        # 1. Draft γ tokens via MTP — chained, fully lazy.
        # Pass coord_group so each chained-predict step's tok_arr
        # broadcast goes through the isolated coord subgroup (separate
        # next_call_id_ from the model TP group). Without this the
        # per-rank MTP forward drifts by ~1ulp at cycle 5+ and drafts
        # diverge across ranks; at temp>0 the per-rank RNG also
        # diverges. The acceptance count and bonus token below are
        # broadcast separately at temp>0; at temp=0 the post-draft
        # token sequence and verify forward are bit-exact downstream so
        # only draft sync is needed.
        next_token_arr = y.reshape(1, 1)
        draft_ids, draft_probs = draft_tokens(
            self.mtp, pre_norm, next_token_arr, gamma, temp,
            fast_lm_head=getattr(self.mtp, "draft_lm_head", None) is not None,
            sync_group=coord_group,
        )

        if prof is not None:
            mx.eval(*draft_ids)
            t_after_draft = time.perf_counter()
            prof.record("draft", (t_after_draft - t_cycle_start) * 1000.0)

        # 2. Verify forward over [y, draft_0, ..., draft_{γ-1}] through
        #    the target. DSv4 has no GDN — just a vanilla forward.
        draft_concat = mx.concatenate(
            [d.reshape(1, 1) for d in draft_ids], axis=1
        )  # (1, γ)
        verify_input = mx.concatenate(
            [next_token_arr, draft_concat], axis=1
        )  # (1, γ+1)

        # POOL-CONTAMINATION FIX (2026-05-29): the verify forward processes
        # [y, draft_0..draft_{γ-1}] in ONE prompt-mode pass, which mutates
        # each PoolingCache via accumulate_windows — flushing pooled windows
        # / advancing the remainder using DRAFT tokens that may be rejected.
        # PoolingCache.trim() (called in the rollback below) only shrinks the
        # remainder; it CANNOT un-flush a pooled entry built from a rejected
        # draft. The contaminated pool then desyncs the compressed-attention
        # context from the real KV cache → output starts correct then
        # collapses into BOS-spam / loops, with a prefix-length-dependent
        # trigger (the system-prompt degeneration). The tree path already
        # guards this by FREEZING the pool during verify (deepseek_v4.py
        # ~1509); the linear path never did. Snapshot here, restore + re-pool
        # only committed tokens on any rejection (step 5b below).
        _pool_caches = self._collect_pooling_caches(gen_batch)
        # Only snapshot pools that WILL flush during the γ+1-token verify
        # forward. A pool flushes when remainder + (γ+1) >= ratio. Pools
        # that won't flush → rejected drafts sit in the remainder tail →
        # trim(rollback) handles them correctly → no snapshot needed.
        # This avoids 41 × mx.array() sync copies on cycles that don't
        # need them (the vast majority: ratio=4 → flush every 4th cycle,
        # ratio=128 → flush every 128th cycle).
        _verify_len = gamma + 1
        _ring_caches: list[Any] = []
        _ring_snaps: list[Any] = []
        if _SPEC_STATE_RESTORE:
            # Unified rollback: snapshot EVERYTHING the verify can mutate
            # (all pools — every pool's remainder grows on every row — and
            # every ring, by O(1) reference).
            _pool_snaps = [pc.save_meta() for pc in _pool_caches]
            for c in gen_batch.prompt_cache:
                subs = c.caches if hasattr(c, "caches") else [c]
                for sub in subs:
                    if hasattr(sub, "save_spec_state"):
                        _ring_caches.append(sub)
                        _ring_snaps.append(sub.save_spec_state())
            if _SPEC_CACHE_ROLLBACK:
                # Stash the rows the verify pushes so a rejection can be
                # undone at the cache level (no commit-forward).
                for pc in _pool_caches:
                    pc.arm_spec_stash()
                for rc in _ring_caches:
                    rc.arm_spec_stash()
        else:
            _pool_snaps = [
                pc.save_meta() if _pool_may_flush(pc, _verify_len) else None
                for pc in _pool_caches
            ]

        verify_pre_norm, verify_logits = dsv4_speculative_forward(
            self.model,
            verify_input,
            gen_batch.prompt_cache,
            self._captured,
        )

        if _SPEC_STATE_RESTORE and _SPEC_CACHE_ROLLBACK:
            # Disarm BEFORE any other forward (the commit-forward fallback
            # must not append to the stash); the stashed rows themselves
            # stay bound for the rollback below.
            for pc in _pool_caches:
                pc.disarm_spec_stash()
            for rc in _ring_caches:
                rc.disarm_spec_stash()

        if prof is not None:
            mx.eval(verify_pre_norm, verify_logits)
            t_after_verify = time.perf_counter()
            prof.record("verify", (t_after_verify - t_after_draft) * 1000.0)

        # 3. Acceptance check (lazy until item() in step 4).
        target_tokens = mx.argmax(verify_logits[:, :gamma, :], axis=-1)

        accept_ratios: list[mx.array] = []
        uniforms: Optional[mx.array] = None
        corrections: list[mx.array] = []
        bonus_token: Optional[mx.array] = None
        matches: Optional[mx.array] = None
        all_next: Optional[mx.array] = None

        if temp == 0:
            logprobs_all = verify_logits[0] - mx.logsumexp(
                verify_logits[0], axis=-1, keepdims=True
            )
            if _ACCEPT_LOGPROBS:
                # Same decision rule as the MTP-off generator: argmax over
                # bf16-normalized logprobs (see _ACCEPT_LOGPROBS above).
                target_tokens = mx.argmax(logprobs_all[:gamma], axis=-1)[None]
                all_next = mx.argmax(logprobs_all, axis=-1)
            else:
                all_next = mx.argmax(verify_logits[0], axis=-1)
            matches = mx.equal(target_tokens, draft_concat).squeeze(0)
            # TIE-BREAK LOSSLESSNESS FIX (gated by EXO_DSV4_MTP_TIEBREAK_FIX=1).
            # The batched verify forward differs from a sequential single-token
            # decode by ~1ulp; at temp 0 that flips TIED tokens, and one flipped
            # tie early cascades the whole generation onto a different (often
            # degenerate / repetition) trajectory — the spurious-</think> bug.
            #
            # FIX (v2): apply a DETERMINISTIC tie-break to the bonus-token
            # selection only — among tokens within `eps` logits of the max,
            # pick the LOWEST token id. This is stable across the batched-vs-
            # sequential ~1ulp difference (both see the same tied set and pick
            # the same member), so it stops the cascade at its source. It also
            # naturally suppresses spurious high-id specials like </think>
            # (128822) at ties while leaving clear-margin (legitimate) picks
            # untouched. Applied to `all_next` (which feeds the BONUS token).
            # SAFETY: the bonus is NOT in the KV cache (it becomes next cycle's
            # y, written next cycle), so changing its value mutates no cache
            # state; pre_norm is a hidden state independent of token choice.
            # We do NOT tie-break accepted drafts (those ARE in the cache —
            # rewriting them would desync KV). v1's failure was an extra
            # cache-advancing forward; this version touches no cache.
            #
            # DEFAULT ON (2026-06-03): validated to restore MTP-on correctness
            # to 100% (matches MTP-off) on the aistupid suite with no leaks /
            # no repetition. Set EXO_DSV4_MTP_TIEBREAK_FIX=0 to opt out.
            if os.environ.get("EXO_DSV4_MTP_TIEBREAK_FIX", "1") != "0":
                _tb_eps = float(
                    os.environ.get("EXO_DSV4_MTP_TIEBREAK_EPS", "0.5")
                )
                _vl0 = verify_logits[0]  # (gamma+1, vocab)
                _maxlogit = mx.max(_vl0, axis=-1, keepdims=True)
                # Mask: tokens within eps of the per-position max are "tied".
                _tied = _vl0 >= (_maxlogit - _tb_eps)
                # Among tied tokens pick the lowest id: set untied to a huge id,
                # tied to their own id, then argmin.
                _vocab = _vl0.shape[-1]
                _ids = mx.arange(_vocab, dtype=mx.int32)
                _big = mx.array(_vocab, dtype=mx.int32)
                _cand = mx.where(_tied, _ids, _big)
                all_next = mx.argmin(_cand, axis=-1).astype(mx.int32)
                mx.async_eval(matches, all_next, logprobs_all, verify_pre_norm)
            else:
                mx.async_eval(matches, all_next, logprobs_all, verify_pre_norm)

            # Token-tree alpha probe drain. Rank 0 only. Reads the per-step
            # MTP top-5 records that draft_tokens populated, joins with the
            # verify target argmax at the matching position, and writes one
            # JSONL row per draft step. Both MTP top-5 and target argmax are
            # evaled here (cheap; both are already in flight via async_eval
            # above plus argmax dispatch).
            if _mtp_module.TREE_ALPHA_PROBE:
                _is_rank0 = sync_group is None or sync_group.rank() == 0
                steps = _mtp_module._TREE_ALPHA_PROBE_STEPS
                if _is_rank0 and steps:
                    # target_tokens shape: (1, gamma) -- argmax of verify
                    # logits at the gamma draft positions. We materialise
                    # it (and the top-5 arrays) once, then iterate.
                    tgt_list = cast(
                        list[int], target_tokens.reshape(-1).tolist()
                    )
                    for rec in steps:
                        step_i = rec["step"]
                        if step_i >= len(tgt_list):
                            continue
                        # NOTE: as of 2026-05-19 the queue holds pre-
                        # materialised Python int lists in "top5_ids"
                        # (see draft_tokens). Earlier "top5_idx" lazy-array
                        # version corrupted the MTP forward — do NOT revive.
                        top5_list = cast(list[int], rec["top5_ids"])
                        target_tok = tgt_list[step_i]
                        _tree_alpha_probe_write({
                            "uid": int(uid),
                            "cycle": int(self._spec_cycles),
                            "step": int(step_i),
                            "gamma": int(gamma),
                            "mtp_top5": top5_list,
                            "target": int(target_tok),
                            "match_top1": bool(target_tok == top5_list[0]),
                            "match_top2": bool(target_tok in top5_list[:2]),
                            "match_top3": bool(target_tok in top5_list[:3]),
                            "match_top4": bool(target_tok in top5_list[:4]),
                            "match_top5": bool(target_tok in top5_list[:5]),
                        })
                # Drain unconditionally so a subsequent cycle starts fresh,
                # even on non-rank-0 ranks (no-op there since draft_tokens
                # appends on all ranks but only rank 0 writes).
                steps.clear()
        else:
            _su_probe_rows: list[dict[str, Any]] = []
            for i in range(gamma):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i]
                p_di = p[draft_ids[i].squeeze()]
                q_di = q[0, draft_ids[i].squeeze()]
                ratio = p_di / mx.maximum(q_di, 1e-10)
                accept_ratios.append(mx.minimum(ratio**alpha, 1.0))
                # SINGLE-UID ACCEPT PROBE (EXO_DSV4_SPEC_TRACE=1): BS=1 baseline
                # for the BS>1 probe in _speculative_next_batch. Lets a B=1-vs-B=2
                # diff confirm whether the batched verify forward over-weights the
                # prompt-echo tokens (per-stream verify-logit bias). Rank-0 only.
                if self._spec_trace_enabled:
                    _sg2 = self._get_sharding_group()
                    if _sg2 is None or _sg2.rank() == 0:
                        _su_probe_rows.append({
                            "uid": int(uid),
                            "i": i,
                            "draft_tok": int(draft_ids[i].squeeze().item()),
                            "p_di": float(p_di.item()),
                            "q_di": float(q_di.item()),
                            "accept_ratio": float(accept_ratios[i].item()),
                        })
            if self._spec_trace_enabled and _su_probe_rows:
                try:
                    import json as _json
                    with open(
                        f"/tmp/dsv4_accept_probe_b1_pid{os.getpid()}.jsonl", "ab"
                    ) as _apf:
                        _apf.write(
                            (_json.dumps({
                                "cycle": int(self._spec_cycles),
                                "rows": _su_probe_rows,
                            }) + "\n").encode("utf-8")
                        )
                except Exception:
                    pass
            # MTP MIN_P FIX (2026-06-16). The MTP speculative path never honored
            # the configured min_p/top_p — draft/accept/correction/bonus all
            # sampled from the RAW untruncated distribution, unlike the main
            # make_sampler (which clips the tail). At temp=1.0 the correction
            # categorical(log(residual)) and bonus categorical then occasionally
            # commit an extreme-tail token (audit-proven: p_bonus down to 0.0036
            # while argmax p≈1.0), which on rigid structured output (DSML tags)
            # cascades into a malformed close (</file>/bare </parameter>) ->
            # unterminated invoke -> dead turn. This is why MTP-off is clean (main
            # sampler clips the tail) and why min_p/top_p on the card only PARTLY
            # helped (the accept-test p shifts, but the correction/bonus draws
            # ignored the floor entirely). FIX: apply the same min_p tail-clip to
            # the correction residual and the bonus logits so MTP draws from the
            # same truncated distribution as the main sampler — MTP-on now matches
            # MTP-off distributionally. Default 0.05 (the DSv4 card value); set
            # EXO_DSV4_MTP_MIN_P=0 to disable (A/B). top_p is intentionally NOT
            # applied here — min_p alone removes the absolute-garbage tail that
            # seeds the cascade; adding top_p would further narrow but min_p is
            # the targeted, distribution-confidence-adaptive clip.
            # MTP SAMPLING PARITY — single-uid path. Route both the rejection
            # correction (residual) and the bonus through _mtp_filter_logits
            # (rep_pen → top_p → min_p) so MTP-on matches the main sampler.
            uniforms = mx.random.uniform(shape=(gamma,))
            for i in range(gamma):
                # Structured-output guard: if the target is highly confident at
                # this position (DSML tag etc.), commit its argmax — never let a
                # tail draw corrupt the rigid block. Else do the normal residual
                # correction sample.
                _ca = self._mtp_confident_argmax(verify_logits[0, i])
                if _ca is not None:
                    corrections.append(mx.array(_ca, dtype=mx.uint32))
                    continue
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i][0]
                residual = mx.maximum(p - q, 0.0)
                _rl = self._mtp_filter_logits(mx.log(residual + 1e-10), uid)
                corrections.append(mx.random.categorical(_rl))
            # Bonus token (post-draft position): same confidence guard first.
            _cb = self._mtp_confident_argmax(verify_logits[0, gamma])
            if _cb is not None:
                bonus_token = mx.array(_cb, dtype=mx.uint32)
            else:
                _bonus_logits = verify_logits[0, gamma] * (1.0 / temp)
                _bonus_logits = self._mtp_filter_logits(_bonus_logits, uid)
                bonus_token = mx.random.categorical(_bonus_logits)
            logprobs_all = verify_logits[0] - mx.logsumexp(
                verify_logits[0], axis=-1, keepdims=True
            )
            mx.async_eval(
                accept_ratios,
                uniforms,
                corrections,
                bonus_token,
                logprobs_all,
                verify_pre_norm,
                draft_concat,
            )

        # 4. Determine acceptance count.
        # NOTE: the tie-break fix (v2) lives in the all_next selection above
        # (deterministic lowest-id-within-eps for the bonus). Acceptance here
        # is UNCHANGED — accepted drafts are already in the KV cache, so they
        # must not be altered. Draft acceptance still uses exact-equality vs
        # the (untouched) batched target argmax.
        # Structured-output accept gate: at a position where the TARGET is
        # highly confident (DSML tool-call tag etc., max prob >= GREEDY_P), the
        # rejection-sampling test could still STOCHASTICALLY accept a drafted
        # token that ISN'T the target argmax (accept_ratio>0 for any token the
        # target gives nonzero mass). That commits a wrong structural token the
        # bonus/correction greedy-gate never sees → corrupt <invoke> block →
        # dead turn. So at confident positions, accept the draft ONLY if it IS
        # the target argmax; otherwise stop (the greedy-gated correction then
        # commits the right token). Non-confident positions: unchanged sampling.
        _draft_ids_flat = cast(list[int], draft_concat.reshape(-1).tolist())
        n_accepted = 0
        for i in range(gamma):
            if temp == 0:
                assert matches is not None
                if matches[i].item():
                    n_accepted += 1
                else:
                    break
            else:
                assert uniforms is not None
                _conf = self._mtp_confident_argmax(verify_logits[0, i])
                if _conf is not None:
                    # Confident target: only the argmax draft may be accepted.
                    if _draft_ids_flat[i] == _conf:
                        n_accepted += 1
                    else:
                        break
                elif uniforms[i].item() < accept_ratios[i].item():
                    n_accepted += 1
                else:
                    break

        # ── TIE RE-VERIFY GATE (EXO_DSV4_MTP_TIE_REVERIFY=1, temp=0) ────
        # The batched L>1 verify forward carries small numeric drift vs a
        # clean single-token forward; at NEAR-TIED logits that flips the
        # argmax and a wrong token gets committed (REFCHECK-proven
        # 2026-07-09 at 115K ctx: emitted 270, clean-ref argmax 566, top-2
        # gap 0.5 — the DSML-corruption class, e.g. ``</｜DSML｜inv>``).
        # Clear-margin rows are immune (drift ≪ gap), so scan the rows that
        # decide this cycle's commits (0..n_accepted inclusive: accepted
        # drafts + the bonus/correction row). At the FIRST row whose top-2
        # gap < eps, stop trusting the verify forward: truncate acceptance
        # there (the standard rollback below then trims the cache exactly
        # like a natural rejection) and mark the cycle so the post-rollback
        # re-verify block emits the CLEAN single-token forward's argmax at
        # that position instead. Unlike the rolled-back 2026-06 lowest-id
        # tie-break heuristic, the replacement is ground truth by
        # definition — it IS sequential decode's pick. The decision folds
        # into the existing canonical broadcast (no extra round-trip); the
        # extra 1-token forward runs only on tie cycles.
        _tie_reverify = 0
        if temp == 0 and os.environ.get(
            "EXO_DSV4_MTP_TIE_REVERIFY", "0"
        ) == "1":
            _trv_eps = float(
                os.environ.get("EXO_DSV4_MTP_TIE_REVERIFY_EPS", "1.0")
            )
            _trv_rows = verify_logits[0, : n_accepted + 1]
            _trv_top2 = mx.topk(_trv_rows, 2, axis=-1)
            _trv_gaps = mx.abs(_trv_top2[..., -1] - _trv_top2[..., -2])
            for _trv_i, _trv_g in enumerate(
                cast(list[float], _trv_gaps.tolist())
            ):
                if _trv_g < _trv_eps:
                    _tie_reverify = 1
                    if _trv_i < n_accepted:
                        n_accepted = _trv_i
                    break

        # Compute local bonus_val using local n_accepted, BEFORE the
        # cross-rank broadcast. This lets us combine n_accepted +
        # bonus_val into ONE broadcast (one ACK barrier round-trip
        # instead of two). Each rank's local values may differ — we
        # only commit the canonical (rank 0) pair below.
        if n_accepted == gamma:
            if temp == 0:
                assert all_next is not None
                bonus_val = int(all_next[gamma].item())
            else:
                assert bonus_token is not None
                bonus_val = int(bonus_token.item())
            bonus_lp = logprobs_all[gamma]
        else:
            if temp == 0:
                assert all_next is not None
                bonus_val = int(all_next[n_accepted].item())
            else:
                bonus_val = int(corrections[n_accepted].item())
            bonus_lp = logprobs_all[n_accepted]

        # Cross-rank n_accepted + bonus_val broadcast — UNCONDITIONAL,
        # COMBINED. Earlier code skipped at temp=0 assuming target_tokens
        # (argmax verify_logits) was bit-exact across ranks; MLX's TP
        # verify forward has ~1ulp drift, tied positions flip argmax,
        # `matches` diverges → n_accepted diverges → yield-count drift
        # → BS-transition wedge (trace 2026-05-08).
        #
        # Combined into a single 2-int32 broadcast: each ACK barrier
        # round-trip on the dedicated coord ACK QP costs ~50-100µs;
        # at γ=2 with ~30 cycles/sec/stream this saves one round-trip
        # per cycle = ~3% wall time per stream.
        if sync_drafts:
            # Third slot: the tie re-verify flag. Rank-local gap scans can
            # disagree at the very ties they detect (~1ulp TP drift), so the
            # canonical (rank 0) decision must drive every rank — a rank-
            # divergent branch around the re-verify forward would deadlock
            # the TP collective. Same 1-broadcast budget as before.
            combined_arr = broadcast_from_canonical(
                mx.array(
                    [n_accepted, bonus_val, _tie_reverify], dtype=mx.int32
                ),
                coord_group,
            )
            combined = cast(list[int], combined_arr.tolist())
            n_accepted = combined[0]
            bonus_val = combined[1]
            _tie_reverify = combined[2]
        # bonus_lp stays local — only used for response.logprobs
        # (informational, master picks rank 0's response only).

        self._record_acceptance(n_accepted)

        # ── VERIFY AUDIT (env-gated, temp=0 only) ────────────────────────
        # Diagnostic for the MTP losslessness break: the linear verify forward
        # is supposed to be bit-equivalent to a clean greedy forward, so every
        # emitted token must equal true-greedy argmax. We can't cheaply run a
        # reference forward inline, but we CAN catch the smoking gun: whenever a
        # special token (think_end / eos) is drafted, accepted, or chosen as the
        # bonus, dump the full per-pool state + flush/snapshot status + the
        # verify-logit margin at that position. If the spurious </think> wins by
        # a healthy margin => the verify pool/context is corrupted (target
        # "really" predicts </think> given drifted context). If it wins by a
        # hair => numerical. Correlating the cycle with pool flush state pins
        # the mechanism. Rank-0 only, JSONL to EXO_DSV4_MTP_VERIFY_AUDIT path.
        _audit_path = os.environ.get("EXO_DSV4_MTP_VERIFY_AUDIT")
        if _audit_path and temp == 0 and all_next is not None:
            _is_rank0 = sync_group is None or sync_group.rank() == 0
            if _is_rank0:
                try:
                    _special = {128822, 128821}  # </think>, <think>
                    _draft_list = [int(v) for v in draft_concat[0].tolist()]
                    _tgt_list = [int(v) for v in target_tokens[0].tolist()]
                    _all_next_list = [int(v) for v in all_next.tolist()]
                    _bonus_special = bonus_val in _special
                    _draft_special = any(d in _special for d in _draft_list)
                    _tgt_special = any(t in _special for t in _tgt_list)
                    if _bonus_special or _draft_special or _tgt_special:
                        _bpos = gamma if n_accepted == gamma else n_accepted
                        _vl = verify_logits[0, _bpos]
                        _top2 = mx.topk(_vl, 2)
                        _top2v = [float(x) for x in _top2.tolist()]
                        _argmax_bonus = int(mx.argmax(_vl).item())
                        _margin = (
                            abs(_top2v[-1] - _top2v[-2])
                            if len(_top2v) >= 2 else None
                        )
                        _pools = []
                        for _pc, _snap in zip(_pool_caches, _pool_snaps):
                            _pools.append({
                                "off": int(getattr(_pc, "_pool_offset", -1)),
                                "rem": int(getattr(_pc, "remainder", -1)),
                                "pend": int(getattr(_pc, "_pending_offset_bump", 0)),
                                "ratio": int(getattr(_pc, "ratio", -1)),
                                "snap": _snap is not None,
                            })
                        _rec = {
                            "cycle": int(self._spec_cycles),
                            "uid": int(uid),
                            "gamma": int(gamma),
                            "n_accepted": int(n_accepted),
                            "draft": _draft_list,
                            "target_argmax": _tgt_list,
                            "all_next": _all_next_list,
                            "bonus": int(bonus_val),
                            "bonus_pos": int(_bpos),
                            "bonus_argmax": _argmax_bonus,
                            "bonus_top2_logits": _top2v,
                            "bonus_margin": _margin,
                            "bonus_special": bool(_bonus_special),
                            "draft_special": bool(_draft_special),
                            "pools": _pools,
                        }
                        with open(_audit_path, "a") as _f:
                            _f.write(json.dumps(_rec) + "\n")
                except Exception as _audit_err:  # never break generation
                    logger.warning(f"verify-audit failed: {_audit_err}")

        # ── VERIFY AUDIT (temp>0 path) ───────────────────────────────────
        # The temp==0 audit above cannot run at temp>0 (all_next is None; the
        # bonus/corrections come from stochastic sampling). But our production
        # DSv4 runs at temp=1.0, and that is exactly where the long-tool-call
        # degeneration (spurious </think>/</file> cascade) was observed live
        # (2026-06-16). Capture the SAME discriminator at temp>0 so we can tell
        # the two candidate mechanisms apart:
        #   * pool contamination  -> the special token wins by a HEALTHY margin
        #     (verify context is genuinely corrupted; argmax really is </think>)
        #   * numerical / no temp>0 tiebreak -> wins by a HAIR (near-tied flip)
        # We log whenever a special token appears in the drafts, the batched
        # verify target argmax, or the chosen bonus. Rank-0 only, env-gated,
        # wrapped so it can never break generation. NOTE: at temp>0 the emitted
        # bonus is sampled (bonus_token / corrections), but the verify-logit
        # ARGMAX + margin at the bonus position is still the right corruption
        # signal — a corrupted context shifts the argmax mass, visible here.
        # The original gate (special-token only: </think>/<think>) MISSED the
        # real degeneration: live captures show the corruption is in ORDINARY
        # text tokens — the model generates the literal text `</file>` and bare
        # `</parameter>` (BPE tokens, NOT special ids), so a special-token-only
        # audit logs nothing. Token-id map (resolved from DSv4 tokenizer.json):
        # </think>=128822 <think>=128821 ｜DSML｜=128825; the DSML tag bodies and
        # </file> are plain text. New hypothesis being tested: at temp>0 a
        # REJECTION samples the correction token from the residual (p-q) dist
        # via mx.random.categorical over per-rank-drifted verify logits; with
        # the open tail (temp=1.0) that can draw a LOW-PROBABILITY garbage token
        # that seeds the cascade. (Corroborated: min_p/top_p — which clip that
        # residual tail — reduced degen freq 4->2.) So capture EVERY rejection
        # past a cycle threshold: the chosen bonus/correction token AND its
        # probability mass under the verify dist, to see if corrections draw
        # low-prob tail tokens on long generations. Rank-0 only, env-gated,
        # wrapped so it can never break generation.
        if _audit_path and temp != 0:
            _is_rank0 = sync_group is None or sync_group.rank() == 0
            _rejected = n_accepted < gamma
            _cyc = int(self._spec_cycles)
            # Capture all rejections after cycle 20 (degeneration is a
            # long-generation phenomenon), plus any special-token involvement.
            _special = {128822, 128821, 128825}  # </think>, <think>, ｜DSML｜
            _draft_list = [int(v) for v in draft_concat[0].tolist()]
            _tgt_list = [int(v) for v in target_tokens[0].tolist()]
            _bonus_special = int(bonus_val) in _special
            _special_seen = (
                _bonus_special
                or any(d in _special for d in _draft_list)
                or any(t in _special for t in _tgt_list)
            )
            if _is_rank0 and ((_rejected and _cyc >= 20) or _special_seen):
                try:
                    _bpos = gamma if n_accepted == gamma else n_accepted
                    _vl = verify_logits[0, _bpos]
                    # Verify-dist probability of the COMMITTED bonus token —
                    # the smoking gun for "correction sampled garbage": a tiny
                    # p_bonus on a rejection means we committed a tail token.
                    _vl_lse = float(mx.logsumexp(_vl).item())
                    _bonus_logit = float(_vl[int(bonus_val)].item())
                    _p_bonus = float(mx.exp(mx.array(_bonus_logit - _vl_lse)).item())
                    _argmax_bonus = int(mx.argmax(_vl).item())
                    _p_argmax = float(
                        mx.exp(mx.array(float(_vl[_argmax_bonus].item()) - _vl_lse)).item()
                    )
                    # MECHANISM CONFIRMATION: on a rejection the correction is
                    # sampled from residual = max(p - q, 0) via categorical(
                    # log(residual + 1e-10)). Hypothesis: when draft q and target
                    # p agree strongly, residual collapses to ~0 everywhere, the
                    # +1e-10 floor makes log() ~uniform over 128k vocab, and the
                    # categorical draws a near-uniform GARBAGE token. Predicts
                    # residual_sum ~= 0 on the garbage corrections. Recompute the
                    # residual at the rejection position to capture its mass.
                    _resid_sum = None
                    _resid_max = None
                    if _rejected and _bpos < gamma and draft_probs[_bpos] is not None:
                        _p = mx.softmax(verify_logits[0, _bpos] / temp, axis=-1)
                        _q = draft_probs[_bpos][0]
                        _resid = mx.maximum(_p - _q, 0.0)
                        _resid_sum = float(mx.sum(_resid).item())
                        _resid_max = float(mx.max(_resid).item())
                    _rec = {
                        "temp_gt0": True,
                        "temp": float(temp),
                        "cycle": _cyc,
                        "uid": int(uid),
                        "gamma": int(gamma),
                        "n_accepted": int(n_accepted),
                        "rejected": bool(_rejected),
                        "draft": _draft_list,
                        "target_argmax": _tgt_list,
                        "bonus": int(bonus_val),
                        "bonus_pos": int(_bpos),
                        # residual mass at rejection: ~0 => collapsed residual =>
                        # +1e-10 floor => uniform garbage correction (the bug).
                        "residual_sum": _resid_sum,
                        "residual_max": _resid_max,
                        "bonus_argmax": _argmax_bonus,
                        # p of committed bonus vs p of argmax under verify dist:
                        # p_bonus << p_argmax on a rejection = garbage correction.
                        "p_bonus": _p_bonus,
                        "p_argmax": _p_argmax,
                        "bonus_special": bool(_bonus_special),
                        "special_seen": bool(_special_seen),
                    }
                    with open(_audit_path, "a") as _f:
                        _f.write(json.dumps(_rec) + "\n")
                except Exception as _audit_err:  # never break generation
                    logger.warning(f"verify-audit(temp>0) failed: {_audit_err}")

        if prof is not None:
            t_after_accept = time.perf_counter()
            prof.record("accept", (t_after_accept - t_after_verify) * 1000.0)

        # 5. Roll back the target's KV caches for the rejected drafts.
        #    DSv4's CacheList implements trim() recursively; raw caches
        #    expose offset for the legacy path.
        #
        # POOL-CONTAMINATION FIX (2026-05-29). The verify forward processed
        # [y, draft_0..draft_{γ-1}] in one prompt-mode pass, advancing each
        # PoolingCache's remainder and — IF the γ+1-token span crossed a
        # compress_ratio boundary — FLUSHING a pooled window built partly
        # from draft tokens that may be rejected.
        #
        # Two regimes on a rejection (rollback > 0):
        #   (a) NO pool flushed during verify (the common case — pools flush
        #       only every `ratio` tokens). The rejected drafts merely sit at
        #       the TAIL of each pool's remainder buffer, so the original
        #       `trim(rollback)` removes exactly them and is fully correct.
        #       Cheap path, no extra forward.
        #   (b) A pool DID flush. trim() cannot un-flush a pooled entry
        #       (it only shrinks `remainder`), so a rejected draft would stay
        #       baked into the compressed long-range summary → compressed-
        #       attention desyncs from KV → output collapses (the system-
        #       prompt degeneration). Here we restore each pool to its
        #       pre-verify snapshot, drop ALL γ+1 verify-written KV entries,
        #       and re-pool ONLY the committed tokens [y, *accepted] via a
        #       small commit-forward (vanilla, no spec side channels) — same
        #       discipline as the tree slow-path. This rare path is the only
        #       one paying the extra forward.
        draft_int_values = [int(v) for v in draft_concat[0].tolist()]
        rollback = gamma - n_accepted
        if rollback > 0 and _SPEC_STATE_RESTORE:
            # Unified faithful rollback (see _SPEC_STATE_RESTORE): wholesale
            # restore of every ring + pool to the pre-verify state, then one
            # commit-forward re-plays [y, *accepted] exactly as sequential
            # decode would have.
            #
            # ASYNC-FENCE DRAIN: with EXO_DSV4_FENCE_ASYNC armed, the verify
            # forward's per-layer commits are mx.async_eval — pool/ring
            # side-chain writes can still be in flight here. The legacy B=1
            # rollback was safe because its trims were pure offset
            # decrements; THIS mode rebinds keys/values/pool buffers, which
            # would race the deferred graph. Drain first, re-arm after the
            # commit-forward (mirrors the batch path's discipline).
            _stats_regime_b = True
            _keep = _verify_len - rollback  # committed rows: [y]+accepted
            _cache_level = (
                _SPEC_CACHE_ROLLBACK
                and all(
                    rc.spec_pushed_rows() == _verify_len for rc in _ring_caches
                )
                and all(
                    pc.spec_can_rollback(psnap, _keep, _verify_len)
                    for pc, psnap in zip(_pool_caches, _pool_snaps, strict=True)
                )
            )
            _stats_cache_rb = _cache_level
            self.mtp._set_fence_async(False)
            if _cache_level:
                # Cache-level exact undo (see _SPEC_CACHE_ROLLBACK): restore
                # each ring and re-push the committed rows through the
                # sequential-decode write path; pools trim or restore+
                # re-accumulate per their flush attribution. No model
                # forward — the verify already computed every committed
                # row's cache state bitwise-sequentially.
                for rc, rsnap in zip(_ring_caches, _ring_snaps, strict=True):
                    rc.rollback_spec_write(rsnap, _keep)
                for pc, psnap in zip(_pool_caches, _pool_snaps, strict=True):
                    pc.spec_rollback(psnap, _keep)
            else:
                for rc, rsnap in zip(_ring_caches, _ring_snaps, strict=True):
                    rc.restore_spec_state(rsnap)
                for pc, psnap in zip(_pool_caches, _pool_snaps, strict=True):
                    if psnap is not None:
                        pc.restore_meta(psnap)
                commit_tokens = [y_val] + draft_int_values[:n_accepted]
                commit_input = mx.array(commit_tokens, dtype=mx.int32).reshape(
                    1, -1
                )
                _commit_logits = self.model(
                    commit_input, cache=gen_batch.prompt_cache
                )
                mx.eval(_commit_logits)
                del _commit_logits
            mtp_cache = self.mtp._cache
            if mtp_cache is not None:
                if hasattr(mtp_cache, "trim"):
                    mtp_cache.trim(rollback)
                elif hasattr(mtp_cache, "offset"):
                    mtp_cache.offset -= rollback
            self.mtp._set_fence_async(True)
        elif rollback > 0:
            # Detect whether any pool flushed a NEW entry during verify.
            # Use the TOTAL (visible offset + staged pending bump): the
            # deferred path's commit_pending() runs at the top of the verify
            # forward and moves the PRIOR step's staged bump from pending into
            # offset (total unchanged), so comparing _pool_offset alone would
            # false-positive. A real flush this cycle increases the total.
            # snap = (pool_offset, pending_bump, remainder, buf_kv, buf_gate).
            # Only pools that were snapshotted (snap is not None = flush
            # predicted) can have actually flushed. Check those.
            pool_flushed = any(
                snap is not None and _pool_flushed_since(pc, snap)
                for pc, snap in zip(_pool_caches, _pool_snaps, strict=True)
            )

            if not pool_flushed:
                # (a) Cheap correct path: rejected drafts are the remainder
                # tail; trim removes them. Identical to pre-fix behavior.
                for c in gen_batch.prompt_cache:
                    if hasattr(c, "trim"):
                        c.trim(rollback)
                    elif hasattr(c, "offset"):
                        c.offset -= rollback
            else:
                # (b) Contamination path: restore snapshotted pools, leave
                # unsnapshotted ones to trim() (they didn't flush).
                _stats_regime_b = True
                if not _POOL_RESTORE_AFTER_TRIM:
                    # LEGACY ORDER (double-rollback bug, see
                    # _POOL_RESTORE_AFTER_TRIM): the blanket trim below
                    # re-trims the pools restore_meta just rewound.
                    for pc, snap in zip(_pool_caches, _pool_snaps):
                        if snap is not None:
                            pc.restore_meta(snap)
                # Trim γ+1 (root y included) so the commit-forward re-adds
                # [y, *accepted] without double-counting y.
                for c in gen_batch.prompt_cache:
                    if hasattr(c, "trim"):
                        c.trim(gamma + 1)
                    elif hasattr(c, "offset"):
                        c.offset -= gamma + 1
                if _POOL_RESTORE_AFTER_TRIM:
                    # FIXED ORDER: restore AFTER the blanket trim so the
                    # snapshotted pools enter the commit-forward at exactly
                    # their pre-verify state (bitwise-sequential, proven by
                    # ldiff_cycles.py).
                    for pc, snap in zip(_pool_caches, _pool_snaps, strict=True):
                        if snap is not None:
                            pc.restore_meta(snap)
                commit_tokens = [y_val] + draft_int_values[:n_accepted]
                commit_input = mx.array(
                    commit_tokens, dtype=mx.int32
                ).reshape(1, -1)
                _commit_logits = self.model(
                    commit_input, cache=gen_batch.prompt_cache
                )
                mx.eval(_commit_logits)
                del _commit_logits

            # MTP cache: roll back by the rejected count (unchanged
            # semantics — n_accepted MTP steps seed the next pre_norm).
            mtp_cache = self.mtp._cache
            if mtp_cache is not None:
                if hasattr(mtp_cache, "trim"):
                    mtp_cache.trim(rollback)
                elif hasattr(mtp_cache, "offset"):
                    mtp_cache.offset -= rollback

        if _CYCLE_STATS:
            _stats = getattr(self, "_cycle_stats", None)
            if _stats is None:
                _stats = {}
                self._cycle_stats = _stats
            # [cycles, rejects, regime_b, first_reject_tok,
            #  first_regime_b_tok, committed_tokens, cache_rb]
            _st = _stats.setdefault(uid, [0, 0, 0, -1, -1, 0, 0])
            _st[0] += 1
            if rollback > 0:
                _st[1] += 1
                if _st[3] < 0:
                    _st[3] = _st[5]
                if locals().get("_stats_regime_b", False):
                    _st[2] += 1
                    if _st[4] < 0:
                        _st[4] = _st[5]
                if locals().get("_stats_cache_rb", False):
                    _st[6] += 1
            _st[5] += n_accepted + 1

        # ── TIE RE-VERIFY (paired with the gate in step 4) ───────────────
        # ⚠ RETIRED FROM PROD (2026-07-10, keep gated OFF): the trim+refeed
        # primitive this block (and the refcheck below) trusts is UNSOUND at
        # pool-flush cycles — PoolingCache.trim() cannot un-flush and the
        # refeed re-flushes, so the "clean reference forward" itself computes
        # over corrupted pool state whenever the trimmed token flushed a pool
        # (ratio-4 layers flush every 4th token). Superseded by the real fix:
        # EXO_DSV4_VERIFY_ROWSEQ in deepseek_v4.py makes the verify forward
        # bitwise-sequential, removing the drift this tried to patch.
        # Canonical flag says some committed row this cycle was a near-tie:
        # the cache now ends at exactly the committed prefix
        # [..., y, drafts[:n_accepted]], so reconstruct the CLEAN sequential
        # next-token distribution with the same trim+refeed primitive the
        # refcheck below uses (trim one token, re-run just it through the
        # target — restores the cache offset as a side effect) and commit
        # ITS argmax. TP SAFETY: every rank reaches this block (flag is from
        # the canonical broadcast) and runs the identical forward; the
        # rank-local argmax is then canonicalized with one extra broadcast —
        # paid only on tie cycles. bonus_lp intentionally keeps the verify
        # row's logprob (informational only; master reads rank 0's text).
        if _tie_reverify and temp == 0:
            _trv_committed = [y_val] + draft_int_values[:n_accepted]
            _trv_last = _trv_committed[-1]
            for _trv_c in gen_batch.prompt_cache:
                if hasattr(_trv_c, "trim"):
                    _trv_c.trim(1)
                elif hasattr(_trv_c, "offset"):
                    _trv_c.offset -= 1
            _trv_in = mx.array([_trv_last], dtype=mx.int32).reshape(1, 1)
            _trv_out = self.model(_trv_in, cache=gen_batch.prompt_cache)
            _trv_row = _trv_out[0, 0]
            mx.eval(_trv_row)
            _trv_pick = int(mx.argmax(_trv_row).item())
            if sync_drafts:
                _trv_arr = broadcast_from_canonical(
                    mx.array([_trv_pick], dtype=mx.int32), coord_group
                )
                _trv_pick = int(cast(list[int], _trv_arr.tolist())[0])
            self._tie_reverify_cycles = (
                getattr(self, "_tie_reverify_cycles", 0) + 1
            )
            if _trv_pick != bonus_val:
                self._tie_reverify_flips = (
                    getattr(self, "_tie_reverify_flips", 0) + 1
                )
            # Flip instrument: stdlib logger INFO is swallowed in the runner
            # (no handler; only loguru sinks reach exo.log — the reason the
            # refcheck writes JSONL). Same idiom here, opt-in via env path.
            _trv_log = os.environ.get("EXO_DSV4_MTP_TIE_REVERIFY_LOG")
            if _trv_log and (sync_group is None or sync_group.rank() == 0):
                try:
                    with open(_trv_log, "a") as _trv_f:
                        _trv_f.write(json.dumps({
                            "cycle": int(self._spec_cycles),
                            "uid": int(uid),
                            "n_accepted": int(n_accepted),
                            "verify_pick": int(bonus_val),
                            "clean_pick": int(_trv_pick),
                            "flipped": bool(_trv_pick != bonus_val),
                            "reverified": self._tie_reverify_cycles,
                            "flips": getattr(self, "_tie_reverify_flips", 0),
                        }) + "\n")
                except Exception as _trv_err:
                    logger.warning(f"tie-reverify log failed: {_trv_err}")
            bonus_val = _trv_pick

        # ── REFERENCE-FORWARD REFCHECK (env-gated, temp=0) ───────────────
        # ⚠ JUDGE BIAS (2026-07-10): the trim(1)+refeed below is only sound
        # when the trimmed token did NOT flush a pool — trim() cannot
        # un-flush and the refeed re-flushes (duplicate pooled entry), so on
        # flush cycles the "clean" reference row is itself corrupted and a
        # logged disagreement may be the INSTRUMENT's fault, not the
        # verify's. Interpret refcheck rows at non-flush cycles only; the
        # trustworthy end-to-end judges are the ldiff harness
        # (~/scratch/ldiff_seq_vs_batched.py, bitwise) and MTP-on vs MTP-off
        # output comparison.
        # Decisive losslessness test for the spurious-</think> bug. At this
        # point gen_batch.prompt_cache has been rolled back to EXACTLY the
        # committed prefix [.., y, *accepted_drafts] (rejected drafts trimmed
        # in step 5/6; if n_accepted==gamma nothing was trimmed and the cache
        # already ends at the committed prefix). The verify forward chose the
        # bonus as argmax(verify_logits[0, bonus_pos]) — a logit it computed
        # WHILE the in-flight forward also carried the (later, now-trimmed)
        # rejected-draft positions. Causally the bonus position cannot attend
        # to those later positions, and trim() is a pure offset decrement, so
        # a CLEAN single-token forward run over the same committed prefix MUST
        # reproduce the same next-token distribution iff the cache/mask path
        # is lossless. We reconstruct that clean distribution cheaply: trim
        # ONE more token (the last committed token) off every prompt-cache
        # entry, then re-run JUST that token through the target. The forward's
        # single-position logits == P(next | committed prefix), computed from
        # a cache that, at the moment those logits are produced, has NEVER
        # held a rejected draft past the bonus position. Re-feeding the token
        # restores the cache offset, so generation is unaffected.
        #
        # trim+refeed is the SAME primitive the production pool-flush branch
        # (step 5b above) already relies on for correctness, so no clone of
        # the 100K-context cache is needed (a clone via copy_rotating_kv_cache
        # would cost a per-layer numpy round-trip — far more expensive).
        #
        # TP SAFETY: self.model(...) drives the tensor-parallel collective.
        # ALL ranks MUST run it or the cluster deadlocks. The trigger below is
        # derived ONLY from cross-rank-canonical values: bonus_val (broadcast
        # in step 4) and draft_int_values (drafts are synced across ranks via
        # coord_group in draft_tokens). We deliberately do NOT trigger on the
        # locally-argmaxed target_tokens/all_next (those carry ~1ulp TP drift
        # and can disagree across ranks at tied positions, which would make
        # some ranks run the extra forward and others not → hang). So every
        # rank takes the same branch and runs the identical trim+forward+eval;
        # only rank 0 writes the JSONL row.
        # NOTE: extended to temp>0 (2026-06-16). The clean-ref-vs-verify-forward
        # ARGMAX comparison is temp-INDEPENDENT (argmax of logits doesn't depend
        # on the sampling temp), so this decisively answers the open question for
        # our temp=1.0 production path: does the verify forward DIVERGE from a
        # clean single-token forward over the same committed prefix (→ verify
        # computation / cache-rollback bug) or AGREE (→ the context fed into the
        # verify forward was already corrupted upstream). At temp>0 we compare
        # ref_argmax vs verify_argmax (NOT the sampled bonus_val, which is
        # stochastic). Use EXO_DSV4_MTP_REFCHECK_ALL=1 to check every cycle.
        _refcheck_path = os.environ.get("EXO_DSV4_MTP_REFCHECK")
        if _refcheck_path:
            try:
                # One-time liveness marker so "empty file" can be distinguished
                # from "instrument never ran". Rank-0 writes it on first entry.
                if not getattr(self, "_refcheck_live_logged", False):
                    self._refcheck_live_logged = True
                    if sync_group is None or sync_group.rank() == 0:
                        with open(_refcheck_path, "a") as _f:
                            _f.write(json.dumps({
                                "marker": "INSTRUMENT_ACTIVE",
                                "cycle": int(self._spec_cycles),
                                "all_mode": os.environ.get(
                                    "EXO_DSV4_MTP_REFCHECK_ALL") == "1",
                            }) + "\n")
                _special = {128822, 128821}  # </think>, <think>
                _committed = [y_val] + draft_int_values[:n_accepted]
                # EVERY-CYCLE mode: a losslessness break is, by definition,
                # verify-argmax != clean-greedy-argmax at ANY position — not
                # only special-token cycles. When EXO_DSV4_MTP_REFCHECK_ALL=1
                # we run the reference forward every cycle and log only the
                # cycles where they DISAGREE (plus a sparse heartbeat). This
                # catches the FIRST divergence deterministically and tells us
                # whether the drift is within-request (a single long gen
                # diverges) or cross-request (only after warmup/batch), without
                # depending on the flaky special-token trigger.
                _refcheck_all = os.environ.get("EXO_DSV4_MTP_REFCHECK_ALL") == "1"
                _trigger = _refcheck_all or (
                    bonus_val in _special
                    or any(d in _special for d in draft_int_values)
                )
                if _trigger:
                    _last_committed = _committed[-1]
                    _bpos = gamma if n_accepted == gamma else n_accepted
                    # Verify-forward bonus logit row (the suspect).
                    _verify_row = verify_logits[0, _bpos]
                    # Roll the committed prefix back by its last token on
                    # EVERY rank (cache is per-rank-local; keep them in step).
                    for _c in gen_batch.prompt_cache:
                        if hasattr(_c, "trim"):
                            _c.trim(1)
                        elif hasattr(_c, "offset"):
                            _c.offset -= 1
                    # Clean single-token reference forward — ALL ranks. This
                    # re-adds the last committed token, restoring the cache.
                    _ref_in = mx.array(
                        [_last_committed], dtype=mx.int32
                    ).reshape(1, 1)
                    _ref_out = self.model(
                        _ref_in, cache=gen_batch.prompt_cache
                    )
                    _ref_row = _ref_out[0, 0]
                    mx.eval(_ref_row, _verify_row)
                    # Rank-0-only logging; the forward above already ran on
                    # all ranks so the collective is balanced.
                    _is_rank0 = sync_group is None or sync_group.rank() == 0
                    if _is_rank0:
                        _ref_argmax = int(mx.argmax(_ref_row).item())
                        _ref_top2 = mx.topk(_ref_row, 2)
                        _ref_top2v = [float(x) for x in _ref_top2.tolist()]
                        _verify_argmax = int(mx.argmax(_verify_row).item())
                        # Temp-independent divergence test: clean reference
                        # forward argmax vs verify forward argmax. At temp==0
                        # bonus_val IS the verify argmax so this also reproduces
                        # the original test; at temp>0 we MUST compare argmax-to-
                        # argmax (bonus_val is sampled, not argmax). DISAGREE here
                        # = the verify forward computes a different distribution
                        # than a clean forward over the same committed prefix
                        # (verify/cache bug). AGREE but ref_argmax is garbage
                        # (e.g. 128825/128822) = upstream context already corrupt.
                        _agree = bool(_ref_argmax == _verify_argmax)
                        _delta_128822 = float(
                            (_verify_row[128822] - _ref_row[128822]).item()
                        )
                        # In ALL mode log only DIVERGENCES (the bug) plus a
                        # sparse heartbeat every 500 cycles so we can confirm
                        # the instrument is live. In special-token mode log
                        # every triggered cycle as before.
                        _heartbeat = (
                            _refcheck_all and (int(self._spec_cycles) % 500 == 0)
                        )
                        _should_log = (
                            (not _refcheck_all) or (not _agree) or _heartbeat
                        )
                        if _should_log:
                            _rec = {
                                "cycle": int(self._spec_cycles),
                                "uid": int(uid),
                                "gamma": int(gamma),
                                "n_accepted": int(n_accepted),
                                "bonus_pos": int(_bpos),
                                # committed continuation this cycle (tail).
                                "committed_prefix_tail": _committed[-8:],
                                "bonus_token": int(bonus_val),
                                "bonus_argmax_from_verify": _verify_argmax,
                                "refcheck_argmax": _ref_argmax,
                                "refcheck_top2_logits": _ref_top2v,
                                "agree": _agree,
                                "delta_128822_verify_minus_ref": _delta_128822,
                                "ref_picks_128822": bool(_ref_argmax == 128822),
                                "heartbeat": bool(_heartbeat and _agree),
                            }
                            with open(_refcheck_path, "a") as _f:
                                _f.write(json.dumps(_rec) + "\n")
            except Exception as _refcheck_err:  # never break generation
                logger.warning(f"mtp-refcheck failed: {_refcheck_err}")

        # 7. Update MTP pre_norm to the verify-pass hidden at the
        #    accepted position, ready for the next cycle.
        pos = gamma if n_accepted == gamma else n_accepted
        self._mtp_pre_norm[uid] = verify_pre_norm[:, pos : pos + 1, :]
        mx.eval(self._mtp_pre_norm[uid])

        # 8. Build all yielded tokens: [y, accepted drafts...].
        all_tokens: list[tuple[int, mx.array]] = [(y_val, y_logprobs)]
        for i in range(n_accepted):
            all_tokens.append((draft_int_values[i], logprobs_all[i]))

        # Update the uid's recent-token ring for the repetition penalty
        # (the staged bonus becomes y_val next cycle and is recorded then).
        if self._mtp_penalties_active:
            self._mtp_record_tokens(uid, [int(tid) for (tid, _lp) in all_tokens])

        # 9. Stage the bonus for the next call.
        gen_batch._next_tokens = mx.array([bonus_val])
        gen_batch._next_logprobs = [bonus_lp]
        mx.async_eval(gen_batch._next_tokens)

        # 9b. Degeneration-hunt trace (EXO_DSV4_SPEC_TRACE=1, rank 0).
        # c=1 single-stream path. Reshape the scalar n_accepted / bonus /
        # all_tokens into the (N=1) list shapes the dumper expects, and
        # build target_tokens as (1, gamma) for parity with the batch path.
        if self._spec_trace_enabled:
            _tt = (
                target_tokens
                if temp == 0
                else mx.zeros_like(draft_concat)
            )
            self._spec_trace_cycle_dump(
                [uid],
                gen_batch,
                gamma,
                verify_input,
                draft_concat,
                _tt,
                [n_accepted],
                [bonus_val],
                [all_tokens],
            )

        if prof is not None:
            t_after_rollback = time.perf_counter()
            prof.record(
                "rollback", (t_after_rollback - t_after_accept) * 1000.0
            )
            prof.record(
                "total", (t_after_rollback - t_cycle_start) * 1000.0
            )
            prof.end_cycle(1)

        # 10. Bookkeeping.
        self._gen_tokens_counter += len(all_tokens)
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
            mx.clear_cache()

        # 11. State machine + length checks per yielded token.
        # 2026-05-20 drain-elimination: return ALL responses in one
        # call instead of buffering γ for later drain. See rationale at
        # _speculative_next_batch (~line 1418). At c=1 this is also a
        # win because each drain `_next()` was paying the same per-call
        # overhead as a spec cycle.
        responses = self._build_yielded_responses(uid, idx, all_tokens)
        if responses and responses[-1].finish_reason is not None:
            self._filter_finished_uid(uid)
            self._cleanup_uid(uid)
        return responses

    def _speculative_next_tree(self, uid: int):
        """K-way tree-drafting verify/accept cycle (DSv4, BS=1, temp=0 only).

        Phase 4 of the token-tree drafting plan
        (.hermes/plans/2026-05-19_token_tree_drafting.md). Replaces the
        gamma chained linear-draft with a K^gamma TREE: each MTP forward
        emits top-K candidates per parent; verify processes ALL n_nodes
        tree tokens in one batched (1, n_nodes) forward; accept walks the
        tree to find the deepest matching path.

        Wall budget vs linear:
          - draft: K + K^2 = 6 MTP forwards (K=2 g=2) vs gamma=2 forwards
            linear. ~3x draft wall.
          - verify: L_q = 7 vs L_q = 3. ~+10% verify wall (KV-bandwidth
            bound at L_kv=100K).
          - tokens/cycle: ~2.32 vs 2.11 (+10%).
          - Net t/s: ~33 (vs 30 baseline) at the measured alpha.

        See plan Phase 4 + Phase 1.2 findings.
        """
        from mlx_lm.models import deepseek_v4 as _dsv4_model_mod

        gen_batch = self._generation_batch
        idx = gen_batch.uids.index(uid)
        sync_group = self._get_sharding_group()
        coord_group = get_coord_group(sync_group)
        sync_drafts = sync_group is not None and sync_group.size() > 1

        y = gen_batch._next_tokens
        if y is None or not gen_batch._next_logprobs:
            return gen_batch.next()

        pre_norm = self._mtp_pre_norm.get(uid)
        if pre_norm is None:
            return gen_batch.next()

        y_val = int(y[0].item())
        y_logprobs = gen_batch._next_logprobs[0]

        gamma = self.gamma
        K = TREE_K
        prof = _phase_timer
        if prof is not None:
            t_cycle_start = time.perf_counter()

        # 1. Draft tree. Returns Python int lists for tree_tokens,
        # parent_idx, depth. Length n_nodes = 1 + K + K^2 (for g=2).
        next_token_arr = y.reshape(1, 1)
        tree_tokens, parent_idx, depth_list = draft_tokens_topk(
            self.mtp, pre_norm, next_token_arr, gamma, K,
            sync_group=coord_group,
        )
        n_nodes = len(tree_tokens)

        if prof is not None:
            t_after_draft = time.perf_counter()
            prof.record("draft", (t_after_draft - t_cycle_start) * 1000.0)

        # 2. Build verify input (1, n_nodes) + the tree mask & positions.
        # The mask must be sized based on the RotatingKVCache's clamped
        # kv-window (= sliding_window), not the raw offset -- otherwise
        # DSv4 local-attention SDPA scores can't broadcast against it.
        first_cache = gen_batch.prompt_cache[0]
        from mlx_lm.models.cache import CacheList
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList)
            else first_cache
        )

        verify_input = mx.array(tree_tokens, dtype=mx.int32).reshape(1, n_nodes)
        tree_mask, tree_positions = _build_tree_mask_and_positions(
            parent_idx, depth_list, mask_cache,
        )

        # Install the tree-verify side channel for the upcoming model
        # forward. Must clear unconditionally afterwards.
        _dsv4_model_mod._set_tree_verify_ctx(tree_mask, tree_positions)
        # Once-per-process diagnostic so we know the tree path is firing
        # in production. EXO_DSV4_TREE_DEBUG=1 enables; default off.
        if os.environ.get("EXO_DSV4_TREE_DEBUG") == "1" and not getattr(
            self, "_tree_debug_logged", False
        ):
            logger.warning(
                f"[TREE-DEBUG] first cycle: n_nodes={n_nodes} "
                f"tree_tokens={tree_tokens} parent_idx={parent_idx} "
                f"depth={depth_list} mask.shape={tree_mask.shape} "
                f"positions={tree_positions.tolist()}"
            )
            self._tree_debug_logged = True
        try:
            verify_pre_norm, verify_logits = dsv4_speculative_forward(
                self.model,
                verify_input,
                gen_batch.prompt_cache,
                self._captured,
            )
        finally:
            _dsv4_model_mod._set_tree_verify_ctx(None, None)

        if prof is not None:
            mx.eval(verify_pre_norm, verify_logits)
            t_after_verify = time.perf_counter()
            prof.record("verify", (t_after_verify - t_after_draft) * 1000.0)

        # 3. Per-tree-node verify argmax. all_next[i] is what target argmax
        # says should be the NEXT token after node i.
        logprobs_all = verify_logits[0] - mx.logsumexp(
            verify_logits[0], axis=-1, keepdims=True
        )
        if _ACCEPT_LOGPROBS:
            # Same decision rule as the MTP-off generator (see
            # _ACCEPT_LOGPROBS above).
            all_next = mx.argmax(logprobs_all, axis=-1)  # (n_nodes,)
        else:
            all_next = mx.argmax(verify_logits[0], axis=-1)  # (n_nodes,)

        # First-cycle diagnostic: dump verify_logits' argmax so we can
        # tell whether the tree-verify forward gives the right next-token
        # predictions. Gated by EXO_DSV4_TREE_DEBUG=1; one-shot.
        if os.environ.get("EXO_DSV4_TREE_DEBUG") == "1" and not getattr(
            self, "_tree_verify_logged", False
        ):
            _next_list = all_next.tolist()
            logger.warning(
                f"[TREE-DEBUG] verify_argmax (per-node next): {_next_list}\n"
                f"             tree_tokens (children to compare): {tree_tokens}\n"
                f"             parent_idx: {parent_idx}\n"
                f"             expected acceptance check: for each i with parent p, "
                f"accept iff all_next[p]==tree_tokens[i]\n"
                f"             y_val (bonus from prev cycle): {y_val}"
            )
            self._tree_verify_logged = True

        # Cross-rank broadcast: with TP>1, ranks may produce slightly
        # different argmax at the same position. Broadcast the canonical
        # rank-0 all_next vector so all ranks agree on the accept walk.
        if sync_drafts:
            all_next = broadcast_from_canonical(
                all_next.astype(mx.int32), coord_group,
            )

        mx.async_eval(all_next, logprobs_all, verify_pre_norm)

        # 4. Tree-accept walk. For each leaf in the tree, walk from root
        # down to the leaf; accept while target_argmax[parent] matches
        # the child's tree token. Keep the deepest accepted path.
        all_next_list = cast(list[int], all_next.tolist())

        # Build children map for fast leaf detection.
        children: dict[int, list[int]] = {}
        for i in range(n_nodes):
            p = parent_idx[i]
            if p >= 0:
                children.setdefault(p, []).append(i)
        leaves = [i for i in range(n_nodes) if i not in children]

        # Find best path. best_path is the list of accepted node ids
        # (deepest first), best_depth is its length, best_end is the
        # deepest accepted node.
        best_depth = 0
        best_end_node = 0  # root: nothing accepted from drafts, bonus only
        best_path: list[int] = []
        for leaf in leaves:
            # Walk leaf -> root, then reverse to get root-first path.
            path: list[int] = []
            cur = leaf
            while cur != -1:
                path.append(cur)
                cur = parent_idx[cur]
            path.reverse()  # path[0] = root, path[-1] = leaf
            # Walk: accept while target_argmax[parent] == tree_tokens[child].
            accepted = 0
            prev = path[0]  # root
            for nxt in path[1:]:
                if all_next_list[prev] == tree_tokens[nxt]:
                    accepted += 1
                    prev = nxt
                else:
                    break
            if accepted > best_depth:
                best_depth = accepted
                best_end_node = prev
                best_path = path[1 : 1 + accepted]  # accepted child node ids

        n_accepted = best_depth
        bonus_val = int(all_next_list[best_end_node])
        bonus_lp = logprobs_all[best_end_node]

        self._record_acceptance(n_accepted)

        if prof is not None:
            t_after_accept = time.perf_counter()
            prof.record("accept", (t_after_accept - t_after_verify) * 1000.0)

        # 5. Roll back KV caches + (re)write the accepted-path KV.
        #
        # The verify forward wrote n_nodes tokens at local_cache cols
        # L_kv..L_kv+n_nodes in tree (BFS) order: col 0 = root (= y),
        # cols 1..K = depth-1 siblings, cols K+1..K+K^2 = depth-2
        # siblings (siblings of node 1 first, then node 2, etc).
        #
        # The ACCEPTED path is `[root] + best_path` -> col indices
        # `[0] + best_path`. For the next cycle's forward to see the
        # right KV-history, we need local_cache to end at offset
        # L_kv + n_accepted + 1 with cols [L_kv..L_kv+n_accepted]
        # holding the accepted-path KV.
        #
        # Two cases:
        #   (a) FAST PATH: accepted path is already at contiguous prefix
        #       cols [0, 1, ..., n_accepted] in tree-input order. True
        #       iff `best_path == [1..n_accepted]`. With DFS-prefix
        #       expansion (draft_tokens_topk reorders BFS -> DFS), the
        #       top-1 chain at every depth lives at contiguous cols, so
        #       most cycles hit this path. Just trim the trailing
        #       siblings from local_cache -> no commit forward needed.
        #
        #   (b) SLOW PATH: accepted path is non-contiguous (e.g.,
        #       best_path=[1, 3] for n_accepted=2 with top-2 d2). Trim
        #       the full tree and re-commit linearly via step 5b.
        #
        # MTP cache: always trim by `gamma - n_accepted`, mirroring
        # the linear baseline's post-accept semantics. K/V at L_kv+1
        # is always the root forward (shared across siblings); for
        # n_accepted >= 2 the K/V at L_kv+2 may be from the wrong
        # depth-1 sibling (the LAST one in BFS order). That only
        # affects the next cycle's MTP draft quality, not user
        # output: the verify forward catches any divergence.
        # TODO(v1.1): rebuild MTP cache from accepted path for
        # acceptance >= 90%.
        contiguous_accept = list(best_path) == list(range(1, n_accepted + 1))
        if contiguous_accept:
            # FAST PATH: trim only the rejected suffix from local_cache;
            # cols 0..n_accepted already hold correct accepted-path KV.
            trim_local = n_nodes - (n_accepted + 1)
        else:
            # SLOW PATH: trim the entire tree; step 5b re-writes y +
            # accepted drafts via a commit forward.
            trim_local = n_nodes

        if trim_local > 0:
            for c in gen_batch.prompt_cache:
                if hasattr(c, "trim"):
                    c.trim(trim_local)
                elif hasattr(c, "offset"):
                    c.offset -= trim_local

        mtp_cache = self.mtp._cache
        mtp_rollback = gamma - n_accepted
        if mtp_cache is not None and mtp_rollback > 0:
            if hasattr(mtp_cache, "trim"):
                mtp_cache.trim(mtp_rollback)
            elif hasattr(mtp_cache, "offset"):
                mtp_cache.offset -= mtp_rollback

        # 5b. Commit forward (SLOW PATH only). Run a small linear
        # forward `[y, *accepted_drafts]` (length n_accepted+1) without
        # the tree side channel; this writes correct contiguous KV at
        # local_cache cols L_kv..L_kv+n_accepted. The side channel was
        # cleared in step 2's finally block, so this is a vanilla
        # forward (tree mask off, standard RoPE, normal Compressor
        # pool updates). Without it the model has no KV record of the
        # accepted tokens it just yielded -> next cycle runs
        # context-blind (the bug fixed in 2026-05-20 3caffad7).
        #
        # Skipped on the fast path because the tree verify already wrote
        # the correct KV at cols [0..n_accepted] in BFS layout.
        if prof is not None:
            t_before_commit = time.perf_counter()
        if not contiguous_accept:
            commit_tokens = [y_val] + [tree_tokens[nid] for nid in best_path]
            commit_input = mx.array(commit_tokens, dtype=mx.int32).reshape(1, -1)
            _commit_logits = self.model(commit_input, cache=gen_batch.prompt_cache)
            # Force the forward to actually run; otherwise lazy mlx can
            # leave local_cache.update_and_fetch dangling and the next
            # cycle reads stale state.
            mx.eval(_commit_logits)
            del _commit_logits
        if prof is not None:
            t_after_commit = time.perf_counter()
            prof.record("commit", (t_after_commit - t_before_commit) * 1000.0)

        # 6. Update MTP pre_norm seed for the next cycle. Use the verify
        # pre_norm at the bonus node's position (where the next cycle's
        # draft will start from).
        self._mtp_pre_norm[uid] = verify_pre_norm[:, best_end_node : best_end_node + 1, :]
        mx.eval(self._mtp_pre_norm[uid])

        # 7. Build yielded tokens: [y, accepted-path drafts...].
        all_tokens: list[tuple[int, mx.array]] = [(y_val, y_logprobs)]
        for node_id in best_path:
            tok = tree_tokens[node_id]
            # Use the parent's logprobs as a proxy for the draft's
            # logprob. Cheap and only used for response metadata.
            parent_node = parent_idx[node_id]
            all_tokens.append((tok, logprobs_all[parent_node]))

        # 8. Stage the bonus for next call.
        gen_batch._next_tokens = mx.array([bonus_val])
        gen_batch._next_logprobs = [bonus_lp]
        mx.async_eval(gen_batch._next_tokens)

        if prof is not None:
            t_after_rollback = time.perf_counter()
            # "rollback" here excludes the commit forward (separately
            # recorded above as "commit") so the buckets don't double-
            # count. End-to-end wall is "total".
            prof.record(
                "rollback", (t_after_rollback - t_after_commit) * 1000.0
            )
            prof.record(
                "total", (t_after_rollback - t_cycle_start) * 1000.0
            )
            prof.end_cycle(1)

        # 9. Bookkeeping + state machine.
        self._gen_tokens_counter += len(all_tokens)
        self._steps_counter += 1
        if self._steps_counter % 512 == 0:
            mx.clear_cache()

        # 2026-05-20 drain-elimination: see _speculative_next_batch comment.
        responses = self._build_yielded_responses(uid, idx, all_tokens)
        if responses and responses[-1].finish_reason is not None:
            self._filter_finished_uid(uid)
            self._cleanup_uid(uid)
        return responses

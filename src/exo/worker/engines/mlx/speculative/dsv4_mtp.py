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

import logging
import os
import time
from collections import deque
from typing import Any, BinaryIO, Optional, Sequence, cast

import mlx.core as mx

from mlx_lm.models.cache import (
    BatchRotatingKVCache,
    CacheList,
    PerStreamBatchRotatingKVCache,
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
        cache_obj.__class__ = PerStreamBatchRotatingKVCache

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

    positions = mx.array(
        [real_offset + d for d in depth], dtype=mx.int32
    )
    return mask, positions

# Per-cycle phase timing. When EXO_DSV4_MTP_PROFILE > 0, brackets the
# draft / verify / accept phases with mx.eval + perf_counter, summarising
# every N cycles. Inserts evals at phase boundaries which serialises
# pipelining — measurements are upper bounds on real production walls.
_PROFILE_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_PROFILE", "0"))


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
        if batch_size > 1:
            self._cache = PerStreamBatchRotatingKVCache(
                max_size=self.mtp_module.config.sliding_window,
                left_padding=[0] * batch_size,
            )
        else:
            self._cache = self.mtp_module.make_cache()
        self._active_uids = ()

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
        self._cache_per_uid[uid] = self._cache
        # Single uid: snapshot doubles as the active cache (c=1 fast path).
        self._active_uids = (uid,)

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
            return  # Already active for this uid set.

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
            draft_mode: ignored for DSv4 (no truncated lm_head shortcut
                yet — the gain on the cluster is small and adds quant
                bookkeeping; can be added later if profiling justifies).

        Returns:
            ``logits`` (B, S, vocab) when ``return_hidden=False``,
            ``(logits, hidden)`` when True. If ``S == 1`` the returned
            logits are squeezed to ``(B, vocab)`` to match the Qwen3.5
            convention.
        """
        del draft_mode  # not yet implemented for DSv4

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
            mtp_mask = self._cache.make_mask(
                S, window_size=self.mtp_module.config.sliding_window, return_array=True
            )

        out = self.mtp_module(
            prev_hidden=hidden_state,
            next_token=token_ids,
            embed_tokens=self.embed_tokens,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
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
        if hasattr(self.mtp, "drop_uid"):
            self.mtp.drop_uid(uid)
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

        decode_pre_norm = self._captured.get("pre_norm")
        if decode_pre_norm is not None:
            mx.eval(decode_pre_norm)
            B = decode_pre_norm.shape[0]
            for b_idx, uid in enumerate(gen_batch.uids):
                if b_idx >= B:
                    break
                self._mtp_pre_norm[uid] = decode_pre_norm[b_idx : b_idx + 1, -1:, :]
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
            stacked_pre_norm, next_tokens_arr, gamma, self.temp
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
        verify_pre_norm, verify_logits = dsv4_speculative_forward(
            self.model,
            verify_input,
            gen_batch.prompt_cache,
            self._captured,
        )
        # verify_pre_norm: (N, γ+1, hidden), verify_logits: (N, γ+1, vocab)

        if prof is not None:
            mx.eval(verify_pre_norm, verify_logits)
            t_after_verify = time.perf_counter()
            prof.record("verify", (t_after_verify - t_after_draft) * 1000.0)

        # 3. Per-uid acceptance check (min-strategy).
        target_tokens = mx.argmax(
            verify_logits[:, :gamma, :], axis=-1
        )  # (N, γ)

        if self.temp == 0:
            matches = mx.equal(target_tokens, draft_concat)  # (N, γ)
            all_next = mx.argmax(verify_logits, axis=-1)  # (N, γ+1)
            logprobs_all = verify_logits - mx.logsumexp(
                verify_logits, axis=-1, keepdims=True
            )  # (N, γ+1, vocab)
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
            for n, uid in enumerate(uids):
                accept_ratios: list[mx.array] = []
                for i in range(gamma):
                    p = mx.softmax(verify_logits[n, i] / self.temp, axis=-1)
                    q = draft_probs_list[i]
                    p_di = p[draft_concat[n, i]]
                    q_di = q[n, draft_concat[n, i]]
                    ratio = p_di / mx.maximum(q_di, 1e-10)
                    accept_ratios.append(mx.minimum(ratio**self.alpha, 1.0))
                uniforms = mx.random.uniform(shape=(gamma,))
                k = 0
                while k < gamma and uniforms[k].item() < accept_ratios[k].item():
                    k += 1
                k_local.append(k)

            # Compute LOCAL bonus_local using LOCAL k_local (per-rank
            # random sample at per-rank acceptance index), then combine
            # n_accepted_per + bonus_vals into ONE broadcast — saves
            # one ACK barrier round-trip per cycle vs the prior two
            # separate broadcasts.
            bonus_lps = []
            bonus_local: list[int] = []
            next_tokens_int_t = next_tokens_arr.reshape(N).tolist()
            draft_concat_int = draft_concat.tolist()
            for n in range(N):
                k = k_local[n]
                bonus_local.append(
                    int(
                        mx.random.categorical(
                            verify_logits[n, k] * (1.0 / self.temp)
                        ).item()
                    )
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

        if prof is not None:
            t_after_accept = time.perf_counter()
            prof.record("accept", (t_after_accept - t_after_verify) * 1000.0)

        # 4. Per-stream cache rollback. Each stream b rolls back by
        #    γ - n_accepted_per[b]. Pass the Python int list straight
        #    to trim_per_stream so it does its arithmetic without
        #    syncing self.offset — at 43+ layers that saves ~6ms per
        #    spec cycle on cluster.
        rollback_per_stream_py = [gamma - acc for acc in n_accepted_per]
        n_min = min(n_accepted_per)
        n_min_rollback = gamma - n_min  # uniform amount for non-per-stream caches

        for c in gen_batch.prompt_cache:
            if isinstance(c, CacheList):
                # Inside CacheList: rotating KV is per-stream-aware,
                # pooling/indexer are uniform (advance ≤1 entry per
                # cycle at γ << compress_ratio so min-strategy is
                # essentially exact).
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

        # MTP-side cache: also per-stream when possible.
        mtp_cache = self.mtp._cache
        if mtp_cache is not None:
            if isinstance(mtp_cache, PerStreamBatchRotatingKVCache):
                mtp_cache.trim_per_stream(rollback_per_stream_py)
            elif hasattr(mtp_cache, "trim"):
                mtp_cache.trim(n_min_rollback)
            elif hasattr(mtp_cache, "offset"):
                mtp_cache.offset -= n_min_rollback

        # 5. Update per-uid pre_norm to each stream's first-rejection
        #    position in verify_pre_norm.
        for n, uid in enumerate(uids):
            acc = n_accepted_per[n]
            self._mtp_pre_norm[uid] = verify_pre_norm[n : n + 1, acc : acc + 1, :]

        # 6. Stage bonus tokens for next call.
        gen_batch._next_tokens = mx.array(bonus_vals)
        gen_batch._next_logprobs = bonus_lps
        mx.async_eval(gen_batch._next_tokens)

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
        temp: float,
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

        for i in range(gamma):
            hidden_for_dump = h if drift_dump else None
            logits, h = self.mtp.predict(
                h, tok_arr, return_hidden=True, draft_mode=False
            )
            # logits: at S=1, predict() squeezes to (N, vocab).
            if temp == 0:
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
                q = mx.softmax(logits / temp, axis=-1)  # (N, vocab)
                tok_pre_sync = mx.random.categorical(
                    logits * (1.0 / temp)
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
        verify_pre_norm, verify_logits = dsv4_speculative_forward(
            self.model,
            verify_input,
            gen_batch.prompt_cache,
            self._captured,
        )

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
            matches = mx.equal(target_tokens, draft_concat).squeeze(0)
            all_next = mx.argmax(verify_logits[0], axis=-1)
            logprobs_all = verify_logits[0] - mx.logsumexp(
                verify_logits[0], axis=-1, keepdims=True
            )
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
            for i in range(gamma):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i]
                p_di = p[draft_ids[i].squeeze()]
                q_di = q[0, draft_ids[i].squeeze()]
                ratio = p_di / mx.maximum(q_di, 1e-10)
                accept_ratios.append(mx.minimum(ratio**alpha, 1.0))
            uniforms = mx.random.uniform(shape=(gamma,))
            for i in range(gamma):
                p = mx.softmax(verify_logits[0, i] / temp, axis=-1)
                q = draft_probs[i][0]
                residual = mx.maximum(p - q, 0.0)
                corrections.append(mx.random.categorical(mx.log(residual + 1e-10)))
            bonus_token = mx.random.categorical(verify_logits[0, gamma] * (1.0 / temp))
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
                if uniforms[i].item() < accept_ratios[i].item():
                    n_accepted += 1
                else:
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
            combined_arr = broadcast_from_canonical(
                mx.array([n_accepted, bonus_val], dtype=mx.int32),
                coord_group,
            )
            combined = cast(list[int], combined_arr.tolist())
            n_accepted = combined[0]
            bonus_val = combined[1]
        # bonus_lp stays local — only used for response.logprobs
        # (informational, master picks rank 0's response only).

        self._record_acceptance(n_accepted)

        if prof is not None:
            t_after_accept = time.perf_counter()
            prof.record("accept", (t_after_accept - t_after_verify) * 1000.0)

        # 5. Roll back the target's KV caches for the rejected drafts.
        #    DSv4's CacheList implements trim() recursively; raw caches
        #    expose offset for the legacy path.
        rollback = gamma - n_accepted
        if rollback > 0:
            for c in gen_batch.prompt_cache:
                if hasattr(c, "trim"):
                    c.trim(rollback)
                elif hasattr(c, "offset"):
                    c.offset -= rollback

            # Also roll back the MTP module's own cache by the same
            # amount: each rejected draft cycle advanced the MTP cache
            # one step too. n_accepted MTP steps land in the next
            # cycle's pre_norm; the rest are wasted.
            mtp_cache = self.mtp._cache
            if mtp_cache is not None:
                if hasattr(mtp_cache, "trim"):
                    mtp_cache.trim(rollback)
                elif hasattr(mtp_cache, "offset"):
                    mtp_cache.offset -= rollback

        # 7. Update MTP pre_norm to the verify-pass hidden at the
        #    accepted position, ready for the next cycle.
        pos = gamma if n_accepted == gamma else n_accepted
        self._mtp_pre_norm[uid] = verify_pre_norm[:, pos : pos + 1, :]

        # 8. Build all yielded tokens: [y, accepted drafts...].
        draft_int_values = [int(v) for v in draft_concat[0].tolist()]
        all_tokens: list[tuple[int, mx.array]] = [(y_val, y_logprobs)]
        for i in range(n_accepted):
            all_tokens.append((draft_int_values[i], logprobs_all[i]))

        # 9. Stage the bonus for the next call.
        gen_batch._next_tokens = mx.array([bonus_val])
        gen_batch._next_logprobs = [bonus_lp]
        mx.async_eval(gen_batch._next_tokens)

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
        all_next = mx.argmax(verify_logits[0], axis=-1)  # (n_nodes,)
        logprobs_all = verify_logits[0] - mx.logsumexp(
            verify_logits[0], axis=-1, keepdims=True
        )

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

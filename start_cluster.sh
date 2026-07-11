#!/bin/bash

# Standardized Exo Cluster Startup Script
# Usage: ./start_cluster.sh
# Detects the current host and sets up the appropriate environment for the 3-node M4 cluster.

# ── Tunable defaults (override via environment before running) ──
: "${EXO_FAST_SYNCH:=1}"
: "${EXO_DISABLE_METAL_TIMEOUT:=1}"
# GPU command buffer queue depth. Lower = more headroom for macOS
# WindowServer to slot 60Hz frames in between our dispatches; saturated
# queue has triggered WindowServer userspace watchdog (40s stuck in
# IOGPUFamily) which kills WindowServer and aborts our runner via the
# Metal completion handler. Default upstream is 10. Forked default 5
# is the first stop on the bisect.
: "${EXO_MAX_ACTIVE_TASKS:=5}"
# DSv4 indexer sliding-window. 0 (the model default when this env var is
# unset) makes the indexer attend over the full cumulative pooled —
# decode rate halves by ~20K tokens, memory grows linearly, and the runner
# eventually silent-SIGABRTs around the 12-minute mark on long generations.
# 8192 was the validated "winning combo" per the dsv4 sliding-indexer plan.
# Unset by default = model default (0, unbounded lookback). W=8192 caps
# the indexer at ~65K raw tokens of lookback — for Think Max territory
# (200K+ context), the original prompt would fall outside that window.
# Decode rate without bounding takes a hit; recover via lower index_topk.
: "${EXO_DSV4_INDEXER_WINDOW:=}"
# index_topk is the sparse top-K width for the DSv4 indexer attention.
# Model documented default = 512. Earlier tuning ran at 192 (validated)
# or 160 (unvalidated speed bet, +7% decode). Reverted to model default
# 2026-05-18 14:20 — user flagged that we'd been comparing perf against
# a quality-compromised config without realizing.
# To override: set EXO_DSV4_INDEX_TOPK in the shell env before launch.
: "${EXO_DSV4_INDEX_TOPK:=512}"
# 2026-06-04: libp2p -> zenoh migration (exo #2132) renamed this env var.
# main.py hard-errors if the old EXO_LIBP2P_NAMESPACE is even present.
: "${EXO_ZENOH_NAMESPACE:=MAC_STUDIO_CLUSTER}"
: "${EXO_PP_DRAFT_MODEL=$HOME/.exo/models/mlx-community--Qwen3.5-0.8B-MLX-8bit}"
# DSv4-Flash sweet spot is 256 (251 tok/s vs 152 at 4096) per
# dsv4_prefill_chunk_size_curve memory. Smaller chunks also produce
# more chunk-boundary cache snapshots, which is what the prefix-cache
# uses to serve mid-prompt partial-prefix hits across requests.
# Lowered 256 -> 128 on 2026-05-11: c=1 100K sweep showed -38% wall and
# +65% prefill tok/s vs 256, decode unchanged, quality (needle-in-haystack)
# preserved. See docs/fork-notes.md "DSv4 c=1 100K tuning May 11".
# 2026-06-13: REVERTED to 128. The 256 default was committed off a measurement
# artifact (the prefill harness scraped a stray 'Prefill complete' log line and
# reported a phantom ~317 tok/s). Re-measured with a fixed, wall-cross-checked
# harness: chunk 256 = ~118-172 tok/s (a REGRESSION), chunk 128 + seq-split =
# ~274-288 tok/s (the real, verified win). So 128 stays the default. OPT-4's
# sparse-SDPA tiling code remains (gated, harmless at tile 128) but the 256
# super-chunk is NOT a win on this hardware. Override via env.
# 2026-07-01: RE-RAISED to 256 — the regression above predates PREFILL_
# ARGPARTITION. With argpartition ON (default below), the indexer topk no
# longer argsorts (B,256,P) rows per chunk, which was the L-scaling term that
# made 256 lose. Client-side WALL-CLOCK A/B at 100K ctx (bench/
# ab_probe_tier1.py, time-to-first-token over 76213 real tokens, no log
# scraping): 128 = 289 t/s, 256 = 353 t/s (+22%), 512 = 293 t/s (regression
# — OPT-4 gathered-tensor tiling overhead dominates). Needle + BOS-spam
# gates clean. 512 stays a non-default override.
: "${EXO_PREFILL_STEP_SIZE:=1024}"  # session-3 validated (+24% prefill; 2048 wedges)
# Context-adaptive prefill chunk sizing (2026-06-21). At LOW context, larger
# chunks (256) amortize the ~390ms per-chunk fixed overhead (43 layers x kernel
# launches x RDMA all_sum x eval x clear_cache) — +39% throughput at 100K. But
# at HIGH context the indexer scores transient (B, H=64, L, P) scales with both
# chunk size L and pooled P, so 256-chunk is -30% at 380K vs 128. The adaptive
# logic in mlx_lm stream_generate starts at EXO_PREFILL_STEP_SIZE (low ctx) and
# shrinks to EXO_PREFILL_STEP_SIZE_HIGH_CTX past EXO_PREFILL_STEP_SIZE_CROSSOVER.
# Defaults: high-ctx unset => fixed step (unchanged behavior). Set both to enable.
: "${EXO_PREFILL_STEP_SIZE_HIGH_CTX:=}"
: "${EXO_PREFILL_STEP_SIZE_CROSSOVER:=}"
# DSv4-Flash seq-split (all_gather batch-safe fix, mlx-lm 8a9cdee).
# Default ON — verified clean quality + throughput at B=2 100K-500K
# (docs/b2-mtp-resolution-2026-06-24.md). Without it, B>1 concurrent
# prefill scrambles streams via the rank-major all_gather reshape.
: "${EXO_DSV4_SEQ_SPLIT:=1}"
# Batched prefill across all queued tasks. Default ON 2026-05-08 after Phase 5
# cluster validation: c=2 100K MTP=0 went from 7.7 → 13.0 tok/s/stream (+69%)
# with both streams symmetric (no serialization tax remaining). c=1 path falls
# through the heterogeneity gate untouched. Memory: c2_batched_prefill_results_2026_05_08.md.
: "${EXO_DSV4_BATCHED_PREFILL:=1}"
# Rendezvous window: how long the runner waits for additional concurrent
# tasks after the first one arrives, before kicking off prefill. Adds the
# same delay to c=1 first-token. 200ms is the empirically-validated default
# (50ms / 100ms: cross-rank libp2p broadcast jitter caused m4-1 and m4-2
# to rendezvous on different iterations, so agree_on_tasks gated to the
# intersection and batched never fired). Set to 0 to disable rendezvous.
: "${EXO_BATCHED_PREFILL_RENDEZVOUS_MS:=200}"
# Optional mlx-lm profiler hook. Comma-separated variants:
#   spans         — per-span wall-time accumulator (was EXO_MINIMAX_TRACE)
#   layer_memory  — per-layer Metal memory snapshots (was EXO_PROFILE_LAYERS;
#                   EXO_PROFILER_LEVEL=2 also snapshots before each layer)
# Unset ⇒ all hook calls in mlx-lm short-circuit to no-ops.
: "${EXO_PROFILER:=}"
: "${EXO_PROFILER_LEVEL:=1}"
: "${EXO_MEMORY_PROFILE_PATH:=}"
: "${EXO_MEMORY_PROFILE_INTERVAL:=256}"
: "${EXO_TRACEMALLOC_PATH:=}"
: "${EXO_TRACEMALLOC_INTERVAL:=2000}"
: "${EXO_TRACEMALLOC_TOP_N:=20}"
: "${EXO_MLX_CLEAR_CACHE_INTERVAL:=0}"
: "${EXO_GC_COLLECT_INTERVAL:=0}"
: "${EXO_MALLOC_RELIEF_INTERVAL:=0}"
: "${EXO_LAYER_EVAL_INTERVAL:=1}"
: "${EXO_DRAFT_KV_WINDOW:=4096}"
: "${EXO_TURBOQUANT:=}"
# KV_CACHE_BITS and TURBOQUANT are mutually exclusive — TurboQuant does its own quantization
# Default 0 (bf16 KV, no quantization) — chosen 2026-05-09 for QUALITY safety:
# 4-bit KV introduces quantization noise into cached K/V values; even though
# SDPA scores are computed in higher precision, cached K/V quant errors
# compound across attention. Per `feedback_kv_cache_quality_risk.md`, prod
# deployments must use bf16 KV; the perf delta vs 4-bit is ~4% at c=2 100K
# DSv4 sparse-attention (4-bit faster on bandwidth pressure but quant
# noise hurts accuracy). Override to a positive N only for memory-
# constrained deploys near the 124 GB wired-limit ceiling.
if [ -n "$EXO_TURBOQUANT" ]; then
    EXO_KV_CACHE_BITS=""
else
    : "${EXO_KV_CACHE_BITS:=0}"
fi
: "${EXO_COMPUTE_DTYPE:=bf16}"
# 2-pass SDPA block count override (MLX PR #3455). Empirical sweet-spot
# 88 on MiniMax (+6.5% decode at 50K), but slight regression on DSv4
# (-0.5% across c=1..8) — DSv4's sparse-index compressor attention
# has a different access pattern than MiniMax's full SDPA. Leave unset
# globally; benchmark and override per-model when needed.
: "${MLX_SDPA_BLOCKS:=}"
: "${EXO_SPECULATIVE_GAMMA:=2}"
# Per-model gamma override for the Qwen3.5-style MTP path (Qwen3.6). Its
# dedicated head is trained with block_size=3, so it sustains a deeper draft
# chain than DSv4's depth-1 head — default γ=3, independent of the DSv4
# EXO_SPECULATIVE_GAMMA above.
: "${EXO_QWEN_SPECULATIVE_GAMMA:=3}"
# Eagle K (MTP top-K soft-emb mixture, K=1 fast-path / K>1 mixture path).
# Promoted to K=8 default 2026-05-24 after the no-renorm fix at
# mtp_module.py:715-729 (commit 2d8d5efc) lifted c=1 100K decode from
# 28.80 ± 0.10 t/s to 29.04 ± 0.07 t/s (+0.83%, Welch t=6.19, p<0.001).
# Quality preserved (needle ✓, BOS=0, bistability=0). See
# .hermes/plans/2026-05-24_w3_K8_norenorm_results.md.
: "${EXO_DSV4_MTP_EAGLE_K:=8}"
# MTP tie-break losslessness fix: DEFAULT OFF (2026-06-09). This was a BAND-AID
# for an upstream bug that is now fixed at the root, and on the canonical affine
# DeepSeek-V4-Flash checkpoint it CORRUPTS output.
#
# History: the batched L>1 spec-VERIFY forward used to be numerically inaccurate
# (hand-rolled bf16 split-softmax), so its argmax flipped vs a sequential decode
# at tied logits → spurious </think> / degeneration. The tie-break (deterministic
# lowest-id-within-eps on the bonus token, eps=0.5) papered over that on the OLD
# bundled-MTP checkpoint, whose logits were sharp enough (top-2 gap > 0.5) that
# the eps window only ever caught genuine ~1ulp ties.
#
# Two things changed: (1) upstream mlx-lm 491f6fe/5b00004 routes the L>1 verify
# through the accurate fp32 fused sparse SDPA — verify argmax is now bit-faithful
# to a clean reference forward (confirmed cycle-by-cycle via EXO_DSV4_MTP_REFCHECK:
# verify_argmax == reference_argmax every cycle). The flip the tie-break existed
# to fix no longer happens. (2) The canonical affine checkpoint has FLATTER logits
# (observed top-2 gaps 0.375–0.5), so eps=0.5 now captures the true argmax PLUS
# lower-id distractor tokens and the "pick lowest id" rule emits the WRONG token
# every cycle (e.g. true argmax 111467 → emitted 68599), cascading into total
# garbage ("petabits", Burmese unicode) at 0% draft acceptance.
#
# Root-cause fix = trust the now-accurate verify argmax; leave the tie-break off.
# (If a genuine ~1ulp tie cascade ever resurfaces, re-enable with an eps on the
# order of a bf16 ulp at logit scale, ~1e-2 — NOT 0.5.)
: "${EXO_DSV4_MTP_TIEBREAK_FIX:=0}"
: "${EXO_DSV4_MTP_TIEBREAK_EPS:=0.5}"
# Greedy accept-rule alignment (2026-07-10): argmax over logsumexp-normalized
# logprobs (the MTP-off generator's rule) instead of raw verify_logits for the
# temp=0 accept/bonus decisions — removes the near-tie trajectory divergence
# between MTP-on and MTP-off (the mismatch the retired tie-break fix papered
# over). Default OFF until the byte-equality gate + DSML battery pass.
: "${EXO_DSV4_MTP_ACCEPT_LOGPROBS:=1}"
# Regime-b double-rollback fix (2026-07-10): in the pool-flush rollback path,
# restore snapshotted pools AFTER the blanket trim (the legacy order let
# CacheList.trim re-trim the just-restored pools, corrupting the compressed
# pool by a row on every flush-straddling rejection). Default OFF until the
# byte-equality gate + battery pass.
: "${EXO_DSV4_POOL_RESTORE_AFTER_TRIM:=1}"
: "${EXO_DSV4_POOL_SNAPSHOT_BATCH:=1}"
# Rowseq per-row decode masks + unified spec-state rollback (2026-07-10):
# with these + POOL_SNAPSHOT_BATCH + ACCEPT_LOGPROBS, the MTP verify cycle
# is bitwise-faithful to sequential decode on the REAL batch cache classes
# (ldiff_cycles 9/9 CLEAN incl. wrapped rings). Default OFF until the
# serving byte-equality gate + battery pass.
: "${EXO_DSV4_ROWSEQ_ROWMASK:=1}"
: "${EXO_DSV4_SPEC_STATE_RESTORE:=1}"
: "${EXO_DSV4_SPEC_CACHE_ROLLBACK:=1}"
: "${EXO_DSV4_SPEC_CACHE_ROLLBACK_C2:=0}"
: "${EXO_DSV4_ROWSEQ_FULLBLOCK:=1}"
: "${EXO_DSV4_ROWSEQ_FULLBLOCK_MOE:=0}"
: "${EXO_DSV4_MOE_PARTS_ROWSEQ:=gate}"
# Long-ctx verify losslessness (2026-07-10, supersedes the 07-09 MTP_MAX_CTX
# =65536 + TIE_REVERIFY stopgap). Root cause of the DSML tool-call corruption
# (</｜DSML｜inv> class): an L>1 batched verify forward is NOT equivalent to
# L sequential steps — rotating-KV in-place writes truncate earlier rows'
# windows, pool flushes become visible to all rows at once, and the indexer
# score GEMM's M=L reduction order flips near-cutoff top-k membership;
# cumulative drift reached ~1.7 logits @115K and flipped near-tied structural
# tokens. REAL FIX: EXO_DSV4_VERIFY_ROWSEQ (deepseek_v4.py) runs verify
# attention per row with per-row cache updates — bitwise-sequential (ldiff
# harness: 0.0 divergence @4K/32K/131K). Row-seq costs ~1.6x vs classic
# batched verify at short ctx (extra dispatches + TP all_reduces), so it
# arms only at ROWSEQ_MIN_CTX+, where the drift demonstrably corrupts and
# where attention compute dominates the dispatch overhead.
# TIE_REVERIFY stays default-OFF: its trim+refeed primitive is unsound at
# pool-flush cycles (see dsv4_mtp.py step-5 warning); superseded by row-seq.
: "${EXO_DSV4_MTP_MAX_CTX:=0}"
: "${EXO_DSV4_MTP_TIE_REVERIFY:=0}"
: "${EXO_DSV4_VERIFY_ROWSEQ:=1}"
: "${EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX:=0}"
# EXO_SPECULATIVE default is set after DSV4_ENABLED is known — see below.
# Runner QoS pin — disabled by default. Benchmarking showed that pinning
# all runners to user_initiated causes Metal command-queue contention at
# c>=6, producing a 16% per-request regression and 25% bad-run rate.
# macOS dynamically adjusts priorities better than a static pin.
: "${EXO_RUNNER_QOS:=off}"
: "${LOG_LEVEL:=INFO}"  # DEBUG measurably slows serving (session 3)

# DeepSeek V4 Flash (~100 GB/rank at 6-bit): 158B total / 13B activated, hybrid
# Compressed Sparse Attention + sliding-window=128, 1 KV head, 1M context via
# YARN. The cluster's primary model. Set DSV4_ENABLED=0 to skip.
: "${DSV4_MODEL_ID:=mlx-community/DeepSeek-V4-Flash}"
: "${DSV4_ENABLED:=1}"
# Speculative decoding uses EXO_PP_DRAFT_MODEL (Qwen3.5 0.8B) as the draft;
# its tokenizer is incompatible with DSv4's, so drafted tokens come back as
# gibberish when verified through DSv4 logits. Force speculation off whenever
# DSv4 is the active model. Re-enable later if a DSv4-compatible draft (or
# MTP weights) becomes available.
if [ "${DSV4_ENABLED}" = "1" ]; then
    # MTP self-spec at γ=2 with per-stream cache + compaction beats
    # the non-MTP baseline at both c=1 (+49%) and c=2 (+10%) on
    # cluster, with bit-identical outputs at temp=0 (rejection-sampling
    # guarantee). See dsv4_mtp_session_2026_05_03_v2 memory.
    : "${EXO_SPECULATIVE:=1}"
    # DSv4 FUSION/COMPILE PATHS — DISABLED 2026-06-18 (correctness > ~3-4% perf).
    # All three batch-mis-specialize at batch_size>1: with any of them ON, a
    # concurrent (BS>1) MTP verify forward produces repetition-biased logits,
    # collapsing one stream into a deterministic period-6 prompt-echo loop that
    # trips the degeneration kill-switch (HTTP 500 on every concurrent request).
    # Proven: with all three =0, BS=2 MTP-on is CLEAN (0 errors, 0 degeneration);
    # with them ON, BS=2 degenerates every iteration. Combined perf they buy is
    # only ~3-4% (FUSED_MOE +1.2%/+1.1%, COMPILE_FFN +1.3% c=2, COMPILE_LAYER
    # incremental) — not worth breaking concurrent serving. Implementation is
    # left intact (env-gated, dormant) pending a batch-correct rework; set any
    # of these =1 to re-enable for single-stream-only experiments.
    # Full diagnosis: skills/.../references/dsv4-mtp-batch-degeneration-and-
    # diagnosis-2026-06-17.md (UPDATE9). Also fixed two real sampling bugs this
    # session (per-request temp 1b443098, residual correction 6cd30df9).
    : "${EXO_DSV4_FUSED_MOE:=0}"
    : "${EXO_DSV4_COMPILE_FFN:=0}"
    : "${EXO_DSV4_COMPILE_LAYER:=0}"
    # Cross-rank fence cadence during decode.
    #
    # 2026-05-17 update — REQUIRED for gamma>=2 MTP stability:
    #   fence=43 (one fence per forward) on gamma>=2 builds up an 86-deep
    #   chained-collective dependency in the GPU/comm-stream command buffer
    #   (43 layers x 2 all_sums per layer). Each chained all_sum is gated
    #   on the previous one peer CQE; tail-stall probability accumulates per
    #   cycle and collapses gamma=2 decode into ~30-50% bistability at iter 2+.
    #   fence=8 (~5 fences per 43-layer forward) breaks the chain into shorter
    #   independent segments. Verified 2026-05-17 by 5-iter gamma=2 c=1 100K
    #   bench: mean=31.5 t/s sigma=0.15, bad_rate=0/5. Beats gamma=1 champion
    #   (30.4 mean) by +1.1 t/s while maintaining stability.
    #
    # Throughput vs stability tradeoff sweep (2026-05-11 historical, gamma=1):
    #   fence=1  -> 15.4 tok/s   fence=4  -> 16.9   fence=8  -> 17.0
    #   fence=16 -> 17.3         fence=43 -> 17.4 (asymptote)
    # Note: those numbers predate the current decode champion (30.5).
    #
    # gamma=2 stability sweep (2026-05-17):
    #   fence=1  -> 28.2 stable    (high fence overhead)
    #   fence=8  -> 31.5 stable    (c=1 best stable throughput)
    #   fence=43 -> bistable (10.9-31.6 t/s, ~30% stall rate)
    #
    # gamma=2 c=2 100K stability sweep (2026-05-22):
    #   fence=2  -> bistable (iter 1: 22.90)        (over-fenced)
    #   fence=4  -> 34.18 mean sigma=0.05 (4/5 clean)  <-- SELECTED for c=2
    #   fence=8  -> bistable (2-3/5 iters at 22 agg)
    # The c=2 stable-decode regime needs SHORTER fence segments than c=1
    # because B=2 doubles per-collective payload size, raising peer-CQE
    # tail probability per call. fence=4 keeps the 43-layer chain in
    # ~11 fence segments (vs ~6 at fence=8), enough to push per-cycle
    # stall probability below threshold. Cost: ~0.7 t/s on c=1 (29.7 ->
    # ~29.0) from extra GPU syncs — acceptable to unlock c=2 stability.
    #
    # Set to 1 to revert to per-layer fences (slower but maximally stable).
    # Set to 8 to recover c=1 ceiling at the cost of c=2 bistability.
    # Set to 43 only when running gamma=1 (where the chain depth is harmless).
    : "${EXO_DSV4_FENCE_EVERY_N_LAYERS:=4}"
    # Metal command buffer size limit. M4 Max defaults to 50MB, but the
    # DSv4 43-layer B=2 batched prefill forward produces >50MB of intermediate
    # results. The 50MB limit causes mid-forward command buffer flushes that
    # create bimodal stalls (fast chunks 0.77s, slow chunks 2.3s, 3x ratio).
    # Setting 200MB lets the full B=2 forward fit in one command buffer,
    # eliminating the stalls. Measured: B=2 aggregate 317 t/s (was 144 t/s).
    : "${MLX_MAX_MB_PER_BUFFER:=200}"
    : "${MLX_MAX_OPS_PER_BUFFER:=200}"
    # MTP self-spec gate. ON by default — activates when (a) the
    # checkpoint contains mtp.* weights (mlx-community variants strip
    # them; use scripts/patch_dsv4_mtp.py to add them back from
    # upstream) AND (b) EXO_DSV4_MTP is non-zero. Set to 0 to fall back
    # to non-MTP decode.
    : "${EXO_DSV4_MTP:=1}"
    # Use the dedicated mlx-community/DeepSeek-V4-Flash-MTP-bf16 head instead of
    # the checkpoint-bundled MTP weights. DEFAULT ON (2026-06-07): measured
    # ~68% single-stream speedup (18.8 → 31.6 t/s) from better draft acceptance
    # (the dedicated repo's quant/packing suits the draft path). Overlaid onto
    # mtp[0] before sharding in utils_mlx._overlay_dsv4_dedicated_mtp. Set =0 to
    # fall back to the native checkpoint-bundled MTP head.
    : "${EXO_DSV4_MTP_DEDICATED:=1}"
else
    : "${EXO_SPECULATIVE:=1}"
fi
# Prefix cache: DSv4's sliding-window-128 means the prefix-cache slicing
# benefit is limited (RotatingKVCache becomes non-sliceable after rotation).
# DSv4 only serves real multi-turn conversations (aux/background tasks route to
# Qwen3.6, not here), so the realistic working set is ~2 live conversations.
#
# History: was 8, dropped to 1 on 2026-06-15 to fight memory pressure while the
# DSv4 multi-turn snapshot leak was still unfixed (each retained session crept
# +0.2-0.4 GB/turn forever, so holding several was dangerous). But 1 means ANY
# second DSv4 caller (a new tab, a delegated child, a concurrent conversation)
# evicts the live session — forcing a full multi-10K-token re-prefill on its
# very next turn (observed 2026-06-29: a 76,551-tok live session evicted under
# "session cap 1", then re-prefilled at 78,137 tok from scratch). The leak is
# now fixed (947d7e50: leaf_snapshots bounded, node snapshots dropped), so each
# retained session is memory-bounded again.
#
# Sized to 2 (NOT 4): this box CO-HOSTS Qwen3.6 (~20 GB), which the naive
# DSv4-only headroom math ignored. Full budget: DSv4 weights ~77 + Qwen ~20 +
# OS overhead leaves ~25-30 GB for DSv4 KV under the 124 GB wired limit. At
# ~2-3 GB/session (100K ctx), 4 sessions (~12 GB) plus a couple concurrent
# in-flight prefills got uncomfortably tight (free dropped toward the floor
# 2026-06-29). 2 keeps the live session safe from a single concurrent caller —
# the actual bug — while worst-casing at only ~6 GB of retained KV.
: "${DSV4_MAX_PREFIX_SESSIONS:=2}"
: "${DSV4_MAX_KV_TOKENS:=}"
: "${DSV4_MAX_PREFIX_BYTES:=}"
# Per-leaf KV snapshot retention for DSv4's non-sliceable (PoolingCache /
# RotatingKVCache) layers. Snapshots are now retained EVENLY SPACED across the
# leaf's token range (commit 3c7b700f) so a below-tip divergence restores from a
# nearby snapshot instead of cold re-prefilling the whole prompt. Each snapshot
# deep-copies pooled-attention state (~0.72 GB at ~108K ctx, measured), so the
# count is the per-leaf memory knob: 4 -> ~3 GB/leaf, 3 -> ~2.2 GB/leaf. Lowered
# 4->3 (2026-06-30) to reclaim ~0.7-1.5 GB/leaf of headroom on the co-hosted
# 128 GB box where DSv4 active hits ~85 GB/node at 140K ctx; 3 still gives
# endpoints + one interior rung (worst-case re-prefill ~= range/2).
: "${EXO_LEAF_SNAPSHOT_RETENTION:=3}"
# DSv4 sparse-index attention materializes a (B, n_heads, L, L×k) score buffer
# at prefill — cubic in L until k saturates at index_topk=512, so any single
# chunk above ~1.2K tokens crashes Metal allocation. Cap at 512 for safety
# margin until upstream PR #1192 lands the query-grouped sparse-SDPA fix.
# See dsv4_prefill_blowup memory + docs/upstream-prs.md.
# Lowered 256 -> 128 on 2026-05-11; re-raised to 256 on 2026-07-01 with
# argpartition ON — see EXO_PREFILL_STEP_SIZE note above.
: "${DSV4_PREFILL_STEP_SIZE:=}"  # empty = inherit EXO_PREFILL_STEP_SIZE env (single source of truth)
# Prefill indexer top-k via argpartition O(P) instead of argsort O(P log P)
# (2026-07-01, quality-equivalent: top-k SET identical, downstream gathered-KV
# softmax is order-invariant). MIN_P=8192 keeps argsort at small pools where
# argpartition's launch overhead loses (measured 295->163 t/s at P=500).
# Wall-clock A/B at 100K: baseline 255 t/s -> 289 t/s (+13%). Set
# EXO_DSV4_PREFILL_ARGPARTITION=0 to disable.
: "${EXO_DSV4_PREFILL_ARGPARTITION:=1}"
: "${EXO_DSV4_ARGPARTITION_MIN_P:=8192}"
# lm_head last-row-only during prefill chunks (2026-07-01). Prefill discards
# the logits; projecting 129K-vocab x chunk rows is pure waste. Gated to
# L > EXO_DSV4_LMHEAD_LASTROW_MIN_L (default 32) inside deepseek_v4.py so the
# MTP verify forward (L=gamma+1, consumes ALL rows) keeps full projection —
# the bare L>1 gate caused verify-slicing degeneration (mlx-lm 7c721d9).
# Decode probes: 29.0 t/s mean, quality gates clean.
: "${EXO_DSV4_LMHEAD_LASTROW:=1}"
# KV cache quantization (bits). With 1 KV head + head_dim=512, KV per token
# per layer is 2 × 1 × 512 × 2 B = 2 KiB at bf16. 4-bit halves that for tight
# 1M-context budgets; bf16 is fine at typical 50-200K usage.
# 0 = explicitly disable. rlt's V4Cache wraps a sliding-window RotatingKVCache
# whose interaction with fork's QuantizedKVCache wrapper hasn't been validated;
# leaving it bf16 until we confirm the quant path round-trips cleanly through
# the compressor branches.
#
# 2026-05-20: This default was accidentally 4 (commit history unclear). Per
# the skill canonical constraint (`feedback_kv_cache_quality_risk.md`) AND
# the comment block immediately above, prod is bf16. Reverting to 0 also
# unlocks the fast SDPA "causal" string path at c>1 (4-bit forces the
# dequantize+array-mask slow path, which made BS=2 verify 168ms vs 43ms
# at bf16). Override to a positive int only for memory-pressed deploys.
: "${DSV4_KV_CACHE_BITS:=0}"
# Sampling defaults — official DeepSeek V4 Flash card recommends
# temperature=1.0, top_p=1.0 for local deployment.
: "${DSV4_TEMPERATURE:=1.0}"
: "${DSV4_TOP_P:=1.0}"
: "${DSV4_TOP_K:=}"
: "${DSV4_MIN_P:=}"
: "${DSV4_PRESENCE_PENALTY:=}"
# DSv4 at spec (temp=1.0/top_p=1.0, no top_k/min_p/penalty) occasionally
# degenerates into repetition loops on structured/repetitive content — observed
# 2026-06-15: "8-bit-8-bit-8-bit…" in a markdown table, "Do NOT reach…", and an
# "exo → exo → … → hermes → hermes …" infinite loop that ran for minutes.
# repetition_penalty=1.05 was tried and EMPIRICALLY FAILED (the exo→hermes loop
# happened after it deployed, confirmed live): mlx-lm's repetition_penalty is
# PRESENCE-based (penalizes a repeated token's logit ONCE regardless of how many
# times it repeats — `logits[:, tokens] = …` overwrites, doesn't accumulate), so
# in a tight loop the whole window is the same few tokens each getting only a
# trivial ÷1.05 nudge. Reverted to no penalty (1.0 = no-op anyway). The real
# guarantee is the DEGENERATION KILL-SWITCH in batch_generate.py
# (EXO_LOOP_DETECT_ACTION="stop", default on): a detected token cycle
# (period<=8, repeated>=6x) force-terminates the generation with
# finish_reason="stop" — a hard stop a sampling penalty can't provide.
#
# UPDATE 2026-07-01: the presence-based limitation was fixed at the source.
# mlx-lm's make_repetition_penalty (adurham fork b6b7434) is now COUNT-AWARE:
# it applies penalty**count over the window (via scatter-add occurrence
# counts) instead of a single ÷penalty write, so a token dominating a loop is
# penalized geometrically per step while a token appearing once is unchanged
# (count==1 -> factor==penalty, bit-identical to the old behaviour). This makes
# a modest repetition_penalty actually able to break the loops that 1.05
# previously could not. Set to 1.1 (Ollama's proven universal default;
# count==1 collateral is a gentle ÷1.1, loop tokens get ÷1.1**count). Tune with
# a real long-context quality probe (needle-in-haystack 100K+, long structured
# generation, long reasoning) — NOT a short "paris" prompt. The kill-switch
# remains the deterministic backstop.
: "${DSV4_REPETITION_PENALTY:=1.1}"

# Qwen3.6-35B-A3B (MoE, ~17.5GB/rank at 8-bit across a 2-node TP shard). Small
# enough to run ALONGSIDE DeepSeek-V4-Flash (~74GB/rank): 74 + 17.5 = ~91.5GB
# of weights/rank, leaving ~32GB under the 124GB wired limit for KV + activations
# across both models. That headroom is TIGHT, so Qwen3.6's prefix/KV cache is
# hard-capped (see QWEN36_MAX_PREFIX_SESSIONS / _BYTES / KV_CACHE_BITS below) —
# without those caps its prefix cache grew unbounded and ate into the shared
# budget, pushing the box into swap. Served as a 2-node Tensor + MlxJaccl
# instance (same RDMA path as DSv4). DISABLED by default (2026-06-20): co-hosting
# alongside DSv4 left only ~21GB/node headroom and pushed the box into sustained
# memory compression. DSv4 runs solo by default; load Qwen on-demand via UI/API,
# or set QWEN36_ENABLED=1 to co-host at launch.
# Sampling: Qwen3.6 is thinking-mode by default — upstream thinking
# recommendation (temp 0.6 / top_p 0.95 / top_k 20 / min_p 0).
: "${QWEN36_MODEL_ID:=mlx-community/Qwen3.6-35B-A3B-8bit}"
# DISABLED 2026-07-01: Qwen3.6 no longer launched at cluster start — JIT
# (EXO_JIT_ENABLED path) loads/unloads it on demand, so co-hosting it eagerly
# just wastes ~17.5 GB/node of DSv4 headroom. Set QWEN36_ENABLED=1 to restore
# eager co-hosting at launch.
: "${QWEN36_ENABLED:=0}"
: "${QWEN36_TEMPERATURE:=0.6}"
: "${QWEN36_TOP_P:=0.95}"
: "${QWEN36_TOP_K:=20}"
: "${QWEN36_MIN_P:=0.0}"
: "${QWEN36_PRESENCE_PENALTY:=0.0}"
: "${QWEN36_REPETITION_PENALTY:=1.0}"
# KV / prefix-cache caps. Qwen3.6 is the AUX + image model: it only serves
# one-shot, independent calls (title-gen, skill review, lightweight background
# tasks, image gen) — never long multi-turn DSv4-style conversations. Those
# calls share no prompt prefix with each other, so a persistent prefix cache
# earns nothing and previously grew UNBOUNDED (maxPrefixSessions defaulted to
# None = no eviction; confirmed live 2026-06-15 holding sessions that never
# freed). That floor is pure waste under the tight ~32GB headroom we have
# co-hosting alongside DSv4. Cap it hard:
#   - 1 prefix session: aux calls share the SAME system-prompt prefix, so one
#     retained session still gives a prefill hit on the common prefix; anything
#     beyond that is unreusable. (0 would disable prefix reuse entirely.)
#   - 1 GiB byte ceiling: belt-and-suspenders so it can never balloon.
#   - 8-bit KV quant: halves Qwen's active KV vs bf16. Aux/image output quality
#     is non-critical, so trading a little precision for footprint is the right
#     call here. (DSv4 stays at 0/bf16: it's the quality-critical main model.)
: "${QWEN36_MAX_PREFIX_SESSIONS:=1}"
: "${QWEN36_MAX_PREFIX_BYTES:=1073741824}"
: "${QWEN36_KV_CACHE_BITS:=8}"
# Per-stream KV-token cap and prefill step left at instance defaults (empty);
# set these if a single aux/image call ever needs bounding.
: "${QWEN36_MAX_KV_TOKENS:=}"
: "${QWEN36_PREFILL_STEP_SIZE:=}"

# --- JIT model lifecycle (LM-Studio-style auto load/unload, in exo) ----------
# When EXO_JIT_ENABLED=1, a chat request for a model with no resident instance
# causes exo to auto-place it (transparent to the client), serve it, and unload
# it after an idle window. Default OFF (kill-switch) until proven on this box.
# The motivating use: keep DSv4-Flash interactive while Qwen3.6 (aux) loads on
# demand and frees its ~17.5 GB/node when idle. Set QWEN36_ENABLED=0 to NOT
# co-host Qwen at launch and let JIT bring it up only when an aux task needs it.
# ENABLED 2026-07-01: eager Qwen3.6 placement was disabled (see QWEN36 block)
# on the assumption JIT would load aux models on demand, but this switch was
# left at 0 — so every Qwen-targeted aux task (curator/memory_extraction/
# title_generation) 404'd with "No instance found". Turning JIT on closes that
# gap: aux tasks auto-place Qwen when needed and it unloads after idle.
: "${EXO_JIT_ENABLED:=1}"
# Per-node free-memory reserve (GB) an auto-load must leave on EVERY node, on
# top of its own weight share. Protects the resident interactive model whose
# KV/working-set GROWS with context (storage_size is weights-only). 18 GB ≈
# DSv4-Flash measured growth headroom. 0 disables the reserve (pre-JIT behavior).
: "${EXO_JIT_MEMORY_RESERVE_GB:=18.0}"
# Max seconds an auto-load may take to reach RunnerReady before the request
# gets a clean 503 (NOT a hang). ~30-60s is typical for Qwen3.6.
: "${EXO_JIT_LOAD_TIMEOUT_SECONDS:=120}"
# Idle seconds after which a JIT-placed instance is auto-unloaded. Only JIT
# instances with zero in-flight requests are reaped; the interactive model
# (placed explicitly, jit=False) is immune.
: "${EXO_JIT_IDLE_UNLOAD_SECONDS:=300}"
# Window (seconds) the API polls node memory when a JIT placement is blocked
# ONLY by ram_available before 503ing. Covers the post-kill reclaim window:
# relaunch exo, first JIT request used to fail instantly with "no admissible
# placement" while macOS reclaimed the previous runners' wired pages.
# Non-memory blockers still hard-fail immediately.
# VALIDATED 2026-07-10 (kill/relaunch cycling, model resident): the graceful
# SIGTERM path now releases the full ~85 GB in ~1 s (bootstrap pre-exit
# release), so this wait rarely engages — it remains as the guard for the
# pathological stuck-memory mode and slow-reclaim edge cases. Gate result:
# 5 cycles, zero placement 503s, kill→served 54 s (incl. 38 s model load).
: "${EXO_JIT_PLACEMENT_WAIT_SECONDS:=120}"

# --- Post-kill memory reclaim check (item 1b of the same plan) ---------------
# After killing exo, macOS normally returns the runners' ~60-80 GB of wired
# Metal pages within ~1 min. If (wired + compressor) has not recovered by
# EXO_RECLAIM_DEADLINE_SECONDS after the kill, the node is in the pathological
# stuck state (AGX pages orphaned by a SIGKILL mid-GPU-op; m4-2 2026-07-09:
# 61 GB stuck in the compressor with no owning process) — alert EXPLICITLY
# before launch instead of letting the session die in mysterious placement
# 503s. Warn-only: the launch proceeds so the operator decides. Escape hatch
# for the stuck state is a reboot; userspace cannot reclaim those pages.
: "${EXO_RECLAIM_CHECK:=1}"
: "${EXO_RECLAIM_RESIDUAL_MAX_GB:=25}"
: "${EXO_RECLAIM_DEADLINE_SECONDS:=180}"

# Cluster-wide sampling defaults (apply when neither request nor instance specifies).
# Unset by default — instance and hardcoded fallbacks take over.
: "${EXO_DEFAULT_TEMPERATURE:=}"
: "${EXO_DEFAULT_TOP_P:=}"
: "${EXO_DEFAULT_TOP_K:=}"
: "${EXO_DEFAULT_MIN_P:=}"

export IBV_FORK_SAFE=1
export PYTHONUNBUFFERED=1

# Define Node Constants
M4_1_IP="192.168.86.201"
M4_1_PEER_ID="12D3KooWDGQKAJUYpqTHzBhVpGzYxQagWRwFqJPzkEYzHxt3SSUg"
M4_2_IP="192.168.86.202"
M4_2_PEER_ID="12D3KooWQDzFqvjsgFRfheeV7uvtVUP1gruphpgoVELP9pkHBses"
MBP_IP="192.168.86.203"
MBP_PEER_ID="12D3KooWGtRYJcQpFLQBc3AFbES1A3BrFy55GyNLMNLNm64bHv16"

# Get current IPs (check all interfaces to correctly identify the node)
CURRENT_IPS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}')

IS_M4_1=false
IS_M4_2=false
IS_MBP=false

for IP in $CURRENT_IPS; do
    if [ "$IP" == "$M4_1_IP" ]; then
        IS_M4_1=true
        break
    fi
    if [ "$IP" == "$M4_2_IP" ]; then
        IS_M4_2=true
        break
    fi
    if [ "$IP" == "$MBP_IP" ]; then
        IS_MBP=true
        break
    fi
done

if [ "$IS_M4_1" = true ]; then
    echo "Detected M4-1 ($M4_1_IP)"
    # Peer with M4-2
    export EXO_DISCOVERY_PEERS="/ip4/$M4_2_IP/tcp/52415/p2p/$M4_2_PEER_ID"
elif [ "$IS_M4_2" = true ]; then
    echo "Detected M4-2 ($M4_2_IP)"
    # Peer with M4-1
    export EXO_DISCOVERY_PEERS="/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID"
elif [ "$IS_MBP" = true ]; then
    echo "Detected MacBook Pro ($MBP_IP)"
    # Peer with M4-1
    export EXO_DISCOVERY_PEERS="/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID"
else
    echo "Unknown host IPs: $CURRENT_IPS. Running as remote controller."
fi

# Full cluster setup — always runs regardless of which machine launches the script
echo "Starting cluster setup..."
echo "-----------------------------------------------------"

# Define nodes to start (using SSH config aliases)
NODES=("macstudio-m4-1" "macstudio-m4-2")
# NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")

    # Thunderbolt Connectivity Check
    echo "Discovering active Thunderbolt IPs..."

get_node_tb_ips() {
    local node=$1
    # 1. Ask the node for its Thunderbolt device names (e.g., en1, en2)
    local devices=$(ssh "$node" "networksetup -listallhardwareports" | awk '/Hardware Port: Thunderbolt/{getline; print $2}')
    
    # 2. Iterate through them locally, asking the node about each one individually
    for dev in $devices; do
        if ssh "$node" "ifconfig $dev" 2>/dev/null | grep -q "status: active"; then
            ssh "$node" "ifconfig $dev" | awk '/inet / && !/127\.0\.0\.1/{print $2}'
        fi
    done
}

find_shared_ip() {
    local target_ips=$1
    local peer_ips=$2
    for tip in $target_ips; do
        local t_subnet=$(echo "$tip" | awk -F. '{print $1"."$2"."$3}')
        for pip in $peer_ips; do
            local p_subnet=$(echo "$pip" | awk -F. '{print $1"."$2"."$3}')
            if [ "$t_subnet" == "$p_subnet" ]; then
                echo "$tip"
                return 0
            fi
        done
    done
    return 1
}

echo "Fetching active Thunderbolt IPs from all nodes..."
TB_M4_1_IPS=$(get_node_tb_ips "macstudio-m4-1")
TB_M4_2_IPS=$(get_node_tb_ips "macstudio-m4-2")
# TB_MBP_IPS=$(get_node_tb_ips "macbook-m4")

# Match IPs by their shared broadcast domains
M4_1_TO_M4_2=$(find_shared_ip "$TB_M4_1_IPS" "$TB_M4_2_IPS")
# M4_1_TO_MBP=$(find_shared_ip "$TB_M4_1_IPS" "$TB_MBP_IPS")

M4_2_TO_M4_1=$(find_shared_ip "$TB_M4_2_IPS" "$TB_M4_1_IPS")
# M4_2_TO_MBP=$(find_shared_ip "$TB_M4_2_IPS" "$TB_MBP_IPS")

# MBP_TO_M4_1=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_1_IPS")
# MBP_TO_M4_2=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_2_IPS")

echo "macstudio-m4-1 routes: -> M4-2 ($M4_1_TO_M4_2)"
echo "macstudio-m4-2 routes: -> M4-1 ($M4_2_TO_M4_1)"

# Verify Studio-to-Studio connection was discovered
if [ -z "$M4_1_TO_M4_2" ] || [ -z "$M4_2_TO_M4_1" ]; then
    echo "CRITICAL ERROR: Could not map Studio-to-Studio Thunderbolt topology!"
    exit 1
fi

# Validate each Studio has at least 1 active Thunderbolt interface
echo "Verifying direct Thunderbolt links..."
for node in macstudio-m4-1 macstudio-m4-2; do
    active_count=$(echo "$(get_node_tb_ips "$node")" | grep -c '.')
    if [ "$active_count" -lt 1 ]; then
        echo "CRITICAL ERROR: $node has no active Thunderbolt interfaces."
        echo "Check physical Thunderbolt cable connections!"
        exit 1
    fi
    echo "  $node: $active_count active TB interfaces ✓"
done

# Direct-link pings — clear any stale cross-subnet routes from previous runs first,
# then ping. Without routes, pings can only succeed over direct physical links — no relay.
echo "Testing direct-link connectivity (clearing stale routes first)..."
for node in macstudio-m4-1 macstudio-m4-2; do
    ssh "$node" "for r in \$(netstat -rn | awk '/192\.168\.(200|201|202)\./{print \$1}' | sort -u); do sudo route delete -net \$r 2>/dev/null; done" &> /dev/null
done

# Direct-link TB MTU. The KNOWN-GOOD BASELINE is 9000 — verified 2026-06-08
# from the macOS network config (preferences.plist on both Studios: en3
# Ethernet MTU=9000, Manual IP, full-duplex/autoselect). The cluster ran
# stably across hundreds of restarts at MTU 9000. The "8000 required / 9000
# corrupts weights" conclusion from 2026-06-07 was a MISDIAGNOSIS: the actual
# fault was (a) a macOS migration that wiped m4-2's Thunderbolt network
# services, and (b) forcing en3 to a non-default MTU (8000), which breaks
# Thunderbolt XDomain link RE-ENUMERATION after an RDMA queue teardown — the
# link then can't recover without a reboot. MTU is configured out-of-band via
# networksetup (NOT here); this script must NEVER run `ifconfig <tbiface> mtu`.
# If you ever need to restore the baseline:
#   sudo networksetup -setMTU 'Thunderbolt 2' 9000   (en3 on both Studios)
# EXO_TB_MTU is reference-only — it is NOT applied by this script.
: "${EXO_TB_MTU:=9000}"

# Repair the DIRECT-LINK connected route + set the desired MTU. Observed
# 2026-06-07: after some reboots/crashes macOS leaves the TB interface UP with
# the right /24 IP but WITHOUT a working connected route — m4-2 had no
# 192.168.204 route at all and m4-1's route carried the `!` (reject) flag. The
# cable was physically present (ARP resolved) but IP traffic black-holed → the
# ping below failed with a misleading "Check cable!". A warm reboot does NOT
# fix it (macOS recreates the same broken route at boot); reinstalling the
# connected route does. So before pinging, force a clean connected route (and
# the desired MTU) on the interface that owns the link IP.
#   $1 = node, $2 = the LOCAL link IP on that node (e.g. 192.168.204.1)
repair_direct_route() {
    local node="$1" local_ip="$2"
    [ -z "$local_ip" ] && return 0
    local subnet="${local_ip%.*}.0/24"
    # NOTE (2026-06-07): do NOT touch the interface MTU here. Running
    # `ifconfig <tbiface> mtu N` forces the Thunderbolt PHY to renegotiate,
    # and on this hardware that renegotiation intermittently lands in a
    # "No device connected" / link-dead state that survives until reboot.
    # We were the cause of the repeated TB drops this session. Only the
    # connected ROUTE is repaired here (passive — does not reset the PHY).
    # If a specific MTU is genuinely required, set it ONCE by hand when the
    # link is known-healthy; never bounce it on every boot.
    ssh "$node" "
        iface=\$(ifconfig 2>/dev/null | awk -v ip='$local_ip' '/^[a-z0-9]+: /{i=substr(\$1,1,length(\$1)-1)} \$1==\"inet\" && \$2==ip {print i; exit}')
        [ -z \"\$iface\" ] && exit 0
        sudo route -n delete -net $subnet 2>/dev/null
        sudo route -n add -net $subnet -interface \$iface 2>/dev/null
    " &> /dev/null
}
# $M4_*_TO_M4_* are the PEER IPs; the local IP on each node is the other one.
# Do NOT preemptively repair on every boot — a healthy link must be left
# untouched (route churn is passive but we keep the boot path minimal). The
# conditional repair below only fires when a ping actually fails.

# M4-1 ↔ M4-2 (direct link). Retry once after a fresh route repair before
# declaring a real cable fault, so a transient missing-route doesn't abort boot.
direct_link_ok() {
    local from="$1" to_ip="$2"
    ssh "$from" "ping -c 1 -W 1 $to_ip" &> /dev/null
}
if ! direct_link_ok macstudio-m4-1 "$M4_2_TO_M4_1"; then
    echo "  Direct link m4-1→m4-2 down; repairing route and retrying..."
    repair_direct_route macstudio-m4-1 "$M4_1_TO_M4_2"
    repair_direct_route macstudio-m4-2 "$M4_2_TO_M4_1"
    sleep 1
    if ! direct_link_ok macstudio-m4-1 "$M4_2_TO_M4_1"; then
        echo "ERROR: macstudio-m4-1 cannot directly reach M4-2 ($M4_2_TO_M4_1) even after route repair. Check cable!"; exit 1
    fi
fi
if ! direct_link_ok macstudio-m4-2 "$M4_1_TO_M4_2"; then
    echo "  Direct link m4-2→m4-1 down; repairing route and retrying..."
    repair_direct_route macstudio-m4-2 "$M4_2_TO_M4_1"
    repair_direct_route macstudio-m4-1 "$M4_1_TO_M4_2"
    sleep 1
    if ! direct_link_ok macstudio-m4-2 "$M4_1_TO_M4_2"; then
        echo "ERROR: macstudio-m4-2 cannot directly reach M4-1 ($M4_1_TO_M4_2) even after route repair. Check cable!"; exit 1
    fi
fi

echo "All direct Thunderbolt links verified ✓"

# RoCEv2 (RDMA) Per-Device Port State Check
# Only checks RDMA ports corresponding to TB interfaces that have active IPs (in use).
# PORT_DOWN on unconnected TB ports is expected and harmless.
echo "Checking per-device RDMA port states (active TB interfaces only)..."
RDMA_HEALTHY=true
for NODE in macstudio-m4-1 macstudio-m4-2; do
    echo -n "  $NODE: "
    # Get only the RDMA devices whose underlying interface has an active IP
    PORT_STATUS=$(ssh "$NODE" 'for dev in rdma_en1 rdma_en2 rdma_en3 rdma_en4 rdma_en5; do
        iface=${dev#rdma_}
        # Only check if this interface has a 192.168.x.x IP (active TB link)
        ip=$(ifconfig "$iface" 2>/dev/null | awk "/inet / && /192\.168\./{print \$2}")
        if [ -n "$ip" ]; then
            state=$(ibv_devinfo -d $dev 2>/dev/null | grep "state:" | head -1 | awk "{print \$2}")
            if [ -n "$state" ]; then
                echo -n "$dev($iface=$ip)=$state "
            fi
        fi
    done' 2>/dev/null)

    if [ -z "$PORT_STATUS" ]; then
        echo "no active RDMA devices found"
        RDMA_HEALTHY=false
    else
        echo "$PORT_STATUS"
        if echo "$PORT_STATUS" | grep -q "PORT_DOWN"; then
            echo "    ERROR: PORT_DOWN on an active TB interface on $NODE! Check cable."
            RDMA_HEALTHY=false
        fi
    fi
done

if [ "$RDMA_HEALTHY" = false ]; then
    echo ""
    if [ "${SKIP_RDMA_PORT_CHECK:-0}" = "1" ]; then
        echo "WARNING: RDMA ports are DOWN but SKIP_RDMA_PORT_CHECK=1 is set. Continuing (fresh boot assumed)."
    else
        echo "ERROR: One or more active-link RDMA ports are DOWN. Fix cables and retry."
        exit 1
    fi
else
    echo "  All active-link RDMA ports active \u2713"
fi

# RoCEv2 (RDMA) support is already verified by the per-device PORT_ACTIVE
# check above (ibv_devinfo), which queries the RoCE port state WITHOUT
# allocating anything. The previous "PD allocation" probe here did
# `timeout 2 ... mx.distributed.init(jaccl)` — but jaccl's Mesh ctor opens the
# ibv device + allocates a PD + creates/activates QPs (RTR/RTS) BEFORE the
# bootstrap barrier it then hangs on. `timeout` SIGTERMs the hung process, so
# Connection::~Connection() (destroy_qp/dealloc_pd/close_device) NEVER runs —
# leaking an active QP/PD on the Thunderbolt RoCE NIC EVERY boot. Accumulated
# orphaned RDMA state wedges the Apple TB stack to "No device connected"
# (needs a full OS reboot to clear). Root-caused 2026-06-07; the leaking probe
# is removed — PORT_ACTIVE from ibv_devinfo is the non-destructive RDMA-layer
# health signal. (warm-mem fact 524)

# Cross-subnet IP forwarding + routes are only needed for the 3-node mesh
# (m4-1 + m4-2 + MacBook): there each node has 2 direct links and must relay
# traffic to the subnet it is NOT directly on, so it has to act as a router.
# With 2 nodes on a single direct Thunderbolt link (both on the same /24) there
# is nothing to forward and no 3rd subnet to route to — the direct connected
# route (see repair_direct_route above) is all that's required. Enabling
# net.inet.ip.forwarding here would just turn the Macs into routers for traffic
# that never needs relaying, so it's left off for the 2-node topology.
echo "Skipping IP forwarding (not needed for 2-node direct Thunderbolt link)..."

SUBNET_M4_1_M4_2=$(echo "$M4_1_TO_M4_2" | awk -F. '{print $1"."$2"."$3".0/24"}')
# SUBNET_M4_1_MBP=$(echo "$M4_1_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')
# SUBNET_M4_2_MBP=$(echo "$M4_2_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')

# 3-node mesh only — enable IP forwarding so each node can relay to the subnet
# it isn't on. Disabled for the 2-node direct-link topology (nothing to relay).
# for NODE in macstudio-m4-1 macstudio-m4-2; do
#     ssh "$NODE" "sudo sysctl -w net.inet.ip.forwarding=1" &> /dev/null
# done

# Cross-subnet routes only needed for 3-node mesh (MacBook disconnected)
# ssh macstudio-m4-1 "sudo route delete -net $SUBNET_M4_2_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_2_MBP $M4_2_TO_M4_1" &> /dev/null
# ssh macstudio-m4-2 "sudo route delete -net $SUBNET_M4_1_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_1_MBP $M4_1_TO_M4_2" &> /dev/null
# ssh macbook-m4 "sudo route delete -net $SUBNET_M4_1_M4_2 2>/dev/null; sudo route add -net $SUBNET_M4_1_M4_2 $M4_1_TO_MBP" &> /dev/null

echo "Cross-subnet routes skipped (2-node direct link)."

# 0. Pre-deploy push check — verify local HEAD is on origin/$TARGET_BRANCH
PUSH_CHECK_BRANCH="${EXO_TARGET_BRANCH:-main}"
echo "Verifying local commits are pushed to origin/$PUSH_CHECK_BRANCH..."
LOCAL_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "none")
git fetch origin --quiet 2>/dev/null || true
ORIGIN_TARGET=$(git rev-parse "origin/$PUSH_CHECK_BRANCH" 2>/dev/null || echo "none")
if [ "$LOCAL_HEAD" = "none" ]; then
    echo "WARNING: Not in a git repo on controller. Skipping push check."
elif [ "$ORIGIN_TARGET" = "none" ]; then
    echo "WARNING: Could not fetch origin/$PUSH_CHECK_BRANCH. Skipping push check."
elif ! git merge-base --is-ancestor "$LOCAL_HEAD" "origin/$PUSH_CHECK_BRANCH" 2>/dev/null; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  WARNING: Local HEAD is NOT on origin/$PUSH_CHECK_BRANCH!"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║  Local HEAD:               $LOCAL_HEAD"
    echo "║  origin/$PUSH_CHECK_BRANCH: $(git rev-parse --short origin/$PUSH_CHECK_BRANCH)"
    echo "║                                                              ║"
    echo "║  Cluster nodes will reset to origin/$PUSH_CHECK_BRANCH, so your local"
    echo "║  commits will NOT be deployed. Push first:                   ║"
    echo "║    git push origin $PUSH_CHECK_BRANCH                       "
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "  Local HEAD ($LOCAL_HEAD) is on origin/$PUSH_CHECK_BRANCH ✓"
fi

# Residual (wired + compressor) GB on a node — the memory classes exo's dead
# runners leave behind. Healthy idle macOS sits well under the threshold;
# freshly-killed exo takes ~1 min to drain back under it.
node_residual_gb() {
    ssh "$1" "vm_stat" | awk '
        /page size of/                 { page_size = $8 }
        /Pages wired down:/            { wired = $4 }
        /Pages occupied by compressor:/ { compressor = $5 }
        END { printf "%d", (wired + compressor) * page_size / 1e9 }
    '
}

# Reclaim-curve check: poll a node's residual memory from its kill timestamp
# until it recovers or EXO_RECLAIM_DEADLINE_SECONDS pass, then alert
# explicitly (warn-only). See the EXO_RECLAIM_* block up top.
wait_for_memory_reclaim() {
    local NODE=$1 KILL_EPOCH=$2
    [ "$EXO_RECLAIM_CHECK" == "1" ] || return 0
    [ -n "$KILL_EPOCH" ] || return 0
    local DEADLINE=$(( KILL_EPOCH + EXO_RECLAIM_DEADLINE_SECONDS ))
    local RESIDUAL
    while :; do
        RESIDUAL=$(node_residual_gb "$NODE")
        if [ -n "$RESIDUAL" ] && [ "$RESIDUAL" -le "$EXO_RECLAIM_RESIDUAL_MAX_GB" ]; then
            echo "  Memory reclaim on $NODE complete (wired+compressor ${RESIDUAL} GB <= ${EXO_RECLAIM_RESIDUAL_MAX_GB} GB)."
            return 0
        fi
        if [ "$(date +%s)" -ge "$DEADLINE" ]; then
            echo ""
            echo "  ============================ STUCK MEMORY ============================"
            echo "  WARNING: $NODE wired+compressor is still ${RESIDUAL:-?} GB"
            echo "  (> ${EXO_RECLAIM_RESIDUAL_MAX_GB} GB) ${EXO_RECLAIM_DEADLINE_SECONDS}s after the exo kill."
            echo "  This is the pathological post-SIGKILL state: AGX/Metal wired pages"
            echo "  orphaned mid-GPU-op (m4-2 2026-07-09: 61 GB stuck in the compressor"
            echo "  with no owning process). Model placement on this node will refuse /"
            echo "  JIT requests will 503 until it clears."
            echo "  ESCAPE HATCH: reboot $NODE — userspace cannot reclaim these pages."
            echo "  ======================================================================"
            echo ""
            return 1
        fi
        echo "  Waiting for memory reclaim on $NODE (wired+compressor ${RESIDUAL:-?} GB > ${EXO_RECLAIM_RESIDUAL_MAX_GB} GB)..."
        sleep 5
    done
}

# 1. Cleanup, Update, and Build
for NODE in "${NODES[@]}"; do
    echo "Preparing $NODE..."
    echo "Setting Metal memory limit on $NODE..."
    if [[ "$NODE" == *"macbook"* ]]; then
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=32000"
    else
        # 115000 (was 124000), lowered 2026-06-29 to PREVENT the Metal-allocator
        # wedge root-caused this session. Mechanism: under a memory-pressure event
        # (compressed memory spiking — observed 16 GB) the MLX Metal allocator
        # falls off its fast pooled-reuse path into a stuck wired-malloc state;
        # prefill then drops ~5x (256→58 tok/s, GPU ~9W/idle, CPU pegged in
        # mlx::core::Fence::wait + MetalAllocator::malloc) and does NOT recover
        # without a full reboot (wired memory is pinned; relaunch can't reclaim
        # it). Trigger condition: 124000 on a 137 GB node co-hosting DSv4 (~79 GB
        # wired steady) + Qwen3.6 left only ~13 GB for OS + transient prefill
        # scratch, so a deep-context prefill's scratch spike pushed the system
        # into compression pressure. 115000 gives ~22 GB non-wired headroom —
        # DSv4+Qwen+worst-case retained KV still fit, but a transient spike can no
        # longer drive the OS into the pressure state that degrades the allocator.
        # Override via DSV4_WIRED_LIMIT_MB if a future footprint needs more.
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=${DSV4_WIRED_LIMIT_MB:-115000}"
    fi
    
    echo "Killing existing Exo processes on $NODE..."
    # GRACEFUL FIRST. Exo runners hold live RoCE/RDMA queue pairs (jaccl TP).
    # SIGKILL (-9) skips the C++ static-duration destructors that call
    # destroy_qp/dealloc_pd/close_device — leaking active QPs on the
    # Thunderbolt NIC, which accumulates and wedges the Apple TB stack to
    # "No device connected" (needs an OS reboot). So SIGTERM and WAIT for a
    # clean exit (static destructors run on normal interpreter exit → QPs
    # freed), and only escalate to -9 as a last resort. (root cause: warm-mem
    # fact 526; 2026-06-08)
    ssh "$NODE" "pkill -TERM -f 'python.*exo' 2>/dev/null || true; pkill -TERM -f 'exo.main' 2>/dev/null || true"
    # Wait up to ~15s for graceful exit so jaccl tears down RDMA cleanly.
    _gone=false
    for i in {1..15}; do
        if ssh "$NODE" "pgrep -f 'python.*exo'" > /dev/null 2>&1; then
            sleep 1
        else
            _gone=true
            break
        fi
    done
    if [ "$_gone" = false ]; then
        echo "  WARNING: Exo on $NODE did not exit on SIGTERM after 15s — escalating to SIGKILL (may leak RDMA QPs; reboot if TB wedges)."
        ssh "$NODE" "lsof -ti:52415,52416 | xargs kill -9 2>/dev/null || true"
        ssh "$NODE" "pkill -9 -f 'exo.main' || true"
        ssh "$NODE" "pkill -9 -f 'python.*exo' || true"
        sleep 1
    fi
    
    ssh "$NODE" "screen -wipe || true"

    # Timestamp the kill for the reclaim-curve check before launch (bash 3.2:
    # no associative arrays, so one dynamically-named scalar per node).
    printf -v "KILL_EPOCH_${NODE//-/_}" '%s' "$(date +%s)"

    echo "Ensuring Xcode developer directory on $NODE..."
    ssh "$NODE" "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer || true"
    
    # Update and Build Logic
    # main IS the verified production config (fix/c2-serving-hardening merged
    # 2026-07-07). Override with EXO_TARGET_BRANCH for experiments only.
    TARGET_BRANCH="${EXO_TARGET_BRANCH:-main}"
    ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && git fetch origin && git reset --hard && git checkout $TARGET_BRANCH && git reset --hard origin/$TARGET_BRANCH && git submodule update --init --recursive'" || { echo "Failed to update repo on $NODE"; exit 1; }
    
    echo "Ensuring build dependencies on $NODE..."
    ssh "$NODE" "/opt/homebrew/bin/brew install cmake 2>/dev/null || true"

    # Sync dependencies (mlx and mlx-lm are pulled from git via uv sources).
    # mlx + mlx-lm + mlx-vlm + torch live in the `mlx` extra (upstream's
    # 2026-05-25 restructure moved them from base deps to optional-dependencies),
    # so we need `--extra mlx` to actually install them on darwin. Otherwise the
    # runner crashes at import-time with `ModuleNotFoundError: No module named 'mlx.nn'`.
    # --all-packages installs workspace members too (exo-tools used by bench scripts).
    echo "Syncing dependencies on $NODE..."
    ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=/opt/homebrew/bin:\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/repos/exo && uv sync --extra mlx --all-packages'" || { echo "Failed to sync on $NODE"; exit 1; }

    # Pin mlx-lm to the vendored ./mlx-lm submodule (uv sync installs a stale copy).
    #
    # uv sync resolves mlx-lm from uv.lock, which pins an EXACT commit SHA
    # (the `git = ".../mlx-lm.git?branch=main#<sha>"` entry). When an exo
    # commit bumps the ./mlx-lm submodule gitlink WITHOUT also running
    # `uv lock --upgrade-package mlx-lm` + committing the new uv.lock, the
    # lockfile lags the vendored source. Worse, the package version string
    # doesn't change between mlx-lm commits, so uv reports "already satisfied"
    # and never reinstalls — the runner then imports OLD mlx_lm code while the
    # checked-out submodule has the fix. This bit us with the affine-DSv4
    # warmup crash: "[quantized_matmul] Scale type must be uint8 but received
    # type bfloat16" (make_quantization_config forced mode=mxfp8 onto affine
    # attention weights; the fix that gates the override on on-disk scale dtype
    # was in the submodule but not in the lock-pinned venv copy).
    #
    # The submodule gitlink is the source of truth for which mlx-lm THIS exo
    # commit was reviewed against, so force the venv to match it exactly. This
    # is deterministic and offline (no re-resolution against remote branch HEAD,
    # which line ~721's `git reset --hard` would clobber every run anyway).
    # Same idiom as the Rust-bindings rebuild below.
    echo "Pinning mlx-lm to vendored submodule on $NODE..."
    ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=/opt/homebrew/bin:\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/repos/exo && uv pip install --no-deps --force-reinstall ./mlx-lm'" || { echo "Failed to pin vendored mlx-lm on $NODE"; exit 1; }

    # Rebuild Rust pyo3 bindings from source (uv sync installs a stale pre-compiled version)
    echo "Rebuilding Rust pyo3 bindings on $NODE..."
    ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && uv pip install maturin 2>/dev/null && uv run maturin develop --release -m rust/exo_rs/Cargo.toml'" || { echo "Failed to rebuild Rust bindings on $NODE"; exit 1; }


    echo "Building dashboard on $NODE..."
    ssh "$NODE" "zsh -l -c 'source ~/.zshrc; cd ~/repos/exo/dashboard && npm install && npm run build'" || { echo "Failed to build dashboard on $NODE"; exit 1; }

done

# 2. Inter-Node Git Sync Check (M4-1 vs M4-2 vs MBP)
echo "Verifying commit consistency between nodes..."
COMMIT_M4_1=$(ssh macstudio-m4-1 "cd ~/repos/exo && git rev-parse --short HEAD")
COMMIT_M4_2=$(ssh macstudio-m4-2 "cd ~/repos/exo && git rev-parse --short HEAD")
# COMMIT_MBP=$(ssh macbook-m4 "cd ~/repos/exo && git rev-parse --short HEAD")

if [ "$COMMIT_M4_1" != "$COMMIT_M4_2" ]; then
    echo "CRITICAL ERROR: Cluster out of sync!"
    echo "macstudio-m4-1: $COMMIT_M4_1"
    echo "macstudio-m4-2: $COMMIT_M4_2"
    exit 1
fi
echo "Nodes synchronized on commit $COMMIT_M4_1."

# 3. Start Exo on each node
for NODE in "${NODES[@]}"; do
    echo "Starting Exo on $NODE..."

    # Reclaim-curve check (item 1b): make stuck post-kill memory an explicit,
    # named alert at launch time instead of a mysterious placement 503 later.
    # Warn-only — a failed check does not abort the launch.
    _kill_epoch_var="KILL_EPOCH_${NODE//-/_}"
    wait_for_memory_reclaim "$NODE" "${!_kill_epoch_var}" || true

    # Build the node environment string
    # AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1: Apple GPU driver env var that relaxes
    # the command buffer context store timeout. Without it, macOS's IOGPU
    # watchdog kills our process with kIOGPUCommandBufferCallbackErrorImpacting-
    # Interactivity when long ML workloads block WindowServer compositing —
    # silent SIGABRT that MLX's check_error never gets to log because the
    # kernel kills us first. Recommended by MLX maintainer in mlx#3267 and
    # confirmed working by reporter.
    # IOGPU silent SIGABRT root-caused (ResidencySet::insert calling
    # IOGPUMetalResidencySet::addAllocation which unconditionally aborts on
    # certain conditions). Fixed in fork mlx/backend/metal/resident.cpp by
    # routing all allocations through unwired_set_. The DYLD interposer that
    # caught it is no longer needed.
    EXO_ENV="PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 IBV_FORK_SAFE=1 AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1"
    EXO_ENV="$EXO_ENV EXO_ZENOH_NAMESPACE=$EXO_ZENOH_NAMESPACE"
    EXO_ENV="$EXO_ENV EXO_FAST_SYNCH=$EXO_FAST_SYNCH"
    EXO_ENV="$EXO_ENV EXO_MAX_ACTIVE_TASKS=$EXO_MAX_ACTIVE_TASKS"
    : "${EXO_PP_DRAFT_MODEL:=$HOME/.exo/models/mlx-community--Qwen3.5-0.8B-MLX-8bit}"
    EXO_ENV="$EXO_ENV EXO_PP_DRAFT_MODEL=$EXO_PP_DRAFT_MODEL"
    # Tracing default OFF in prod (session-3 A/B); export EXO_TRACING_ENABLED=true to enable.
    [ "${EXO_TRACING_ENABLED:-false}" = "true" ] && EXO_ENV="$EXO_ENV EXO_TRACING_ENABLED=true"
    [ "${EXO_TRACING_ENABLED:-false}" != "true" ] && EXO_ENV="$EXO_ENV EXO_TRACING_ENABLED=false"
    [ "${MLX_DISABLE_COMPILE:-}" = "1" ] && EXO_ENV="$EXO_ENV MLX_DISABLE_COMPILE=1"
    [ "${MALLOC_STACK_LOGGING:-}" = "1" ] && EXO_ENV="$EXO_ENV MallocStackLogging=1 MallocStackLoggingNoCompact=1"
    [ -n "${MLX_LOG_NEW_BUFFER_PATH:-}" ] && EXO_ENV="$EXO_ENV MLX_LOG_NEW_BUFFER_PATH=$MLX_LOG_NEW_BUFFER_PATH"
    # EXO_RUNNER_HANG_TIMEOUT_SECONDS: raise for the reliable ARQ path — large
    # prefill all_reduces are slow (4KB stop-and-wait; UC can't do fast large or
    # concurrent sends) and can legitimately run past the default 45s.
    [ -n "${EXO_RUNNER_HANG_TIMEOUT_SECONDS:-}" ]  && EXO_ENV="$EXO_ENV EXO_RUNNER_HANG_TIMEOUT_SECONDS=$EXO_RUNNER_HANG_TIMEOUT_SECONDS"
    # MLX_JACCL_RELIABLE_INFLIGHT: reliable-path pipeline depth. Depth 8 is
    # validated for sz<=2 chunks (<=16KB concurrent UC sends are clean; the old
    # MUST-be-1 note predates the 2026-07-06 pipelining patch, mlx 452fbebf).
    : "${MLX_JACCL_RELIABLE_INFLIGHT:=8}"
    [ -n "${MLX_JACCL_RELIABLE_INFLIGHT:-}" ]      && EXO_ENV="$EXO_ENV MLX_JACCL_RELIABLE_INFLIGHT=$MLX_JACCL_RELIABLE_INFLIGHT"
    [ -n "${MLX_LOG_ARRAY_DESC_COUNT_INTERVAL:-}" ] && EXO_ENV="$EXO_ENV MLX_LOG_ARRAY_DESC_COUNT_INTERVAL=$MLX_LOG_ARRAY_DESC_COUNT_INTERVAL"
    [ -n "${MLX_PER_TYPE_DUMP_INTERVAL:-}" ] && EXO_ENV="$EXO_ENV MLX_PER_TYPE_DUMP_INTERVAL=$MLX_PER_TYPE_DUMP_INTERVAL"
    [ -n "${MLX_PER_TYPE_TRACK:-}" ] && EXO_ENV="$EXO_ENV MLX_PER_TYPE_TRACK=$MLX_PER_TYPE_TRACK"
    [ -n "${MLX_LM_EAGER_EVAL_CACHES:-}" ] && EXO_ENV="$EXO_ENV MLX_LM_EAGER_EVAL_CACHES=$MLX_LM_EAGER_EVAL_CACHES"
    [ -n "${MLX_LM_CLEAR_COMPILE_CACHE_INTERVAL:-}" ] && EXO_ENV="$EXO_ENV MLX_LM_CLEAR_COMPILE_CACHE_INTERVAL=$MLX_LM_CLEAR_COMPILE_CACHE_INTERVAL"
    [ -n "${MLX_LM_SYNC_AFTER_STEP:-}" ] && EXO_ENV="$EXO_ENV MLX_LM_SYNC_AFTER_STEP=$MLX_LM_SYNC_AFTER_STEP"
    [ -n "${MLX_GPU_TIME:-}" ] && EXO_ENV="$EXO_ENV MLX_GPU_TIME=$MLX_GPU_TIME"
    [ -n "${EXO_DECODE_PROBE:-}" ] && EXO_ENV="$EXO_ENV EXO_DECODE_PROBE=$EXO_DECODE_PROBE"
    [ -n "${EXO_DECODE_PROBE_EVERY:-}" ] && EXO_ENV="$EXO_ENV EXO_DECODE_PROBE_EVERY=$EXO_DECODE_PROBE_EVERY"
    [ -n "${EXO_DSV4_DEGEN_PROBE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_DEGEN_PROBE=$EXO_DSV4_DEGEN_PROBE"
    [ -n "${MLX_GPU_TIME_LOG_EVERY:-}" ] && EXO_ENV="$EXO_ENV MLX_GPU_TIME_LOG_EVERY=$MLX_GPU_TIME_LOG_EVERY"
    [ -n "${MLX_SDPA_BLOCKS:-}" ] && EXO_ENV="$EXO_ENV MLX_SDPA_BLOCKS=$MLX_SDPA_BLOCKS"
    [ -n "${MLX_LM_SDPA_ROWSPLIT:-}" ] && EXO_ENV="$EXO_ENV MLX_LM_SDPA_ROWSPLIT=$MLX_LM_SDPA_ROWSPLIT"
    [ -n "${MLX_BUILD_PROBE:-}" ] && EXO_ENV="$EXO_ENV MLX_BUILD_PROBE=$MLX_BUILD_PROBE"
    [ -n "${MLX_BUILD_PROBE_LOG_EVERY:-}" ] && EXO_ENV="$EXO_ENV MLX_BUILD_PROBE_LOG_EVERY=$MLX_BUILD_PROBE_LOG_EVERY"
    [ -n "${MLX_OP_PROBE:-}" ] && EXO_ENV="$EXO_ENV MLX_OP_PROBE=$MLX_OP_PROBE"
    [ -n "${MLX_MAX_OPS_PER_BUFFER:-}" ] && EXO_ENV="$EXO_ENV MLX_MAX_OPS_PER_BUFFER=$MLX_MAX_OPS_PER_BUFFER"
    [ -n "${MLX_MAX_MB_PER_BUFFER:-}" ] && EXO_ENV="$EXO_ENV MLX_MAX_MB_PER_BUFFER=$MLX_MAX_MB_PER_BUFFER"
    # M-batched sorted-run gather qmv kill switch / tuning (mlx cb539cda).
    [ -n "${MLX_GATHER_QMV_RHS:-}" ] && EXO_ENV="$EXO_ENV MLX_GATHER_QMV_RHS=$MLX_GATHER_QMV_RHS"
    [ -n "${MLX_GATHER_QMV_RHS_TILE:-}" ] && EXO_ENV="$EXO_ENV MLX_GATHER_QMV_RHS_TILE=$MLX_GATHER_QMV_RHS_TILE"
    [ -n "${MLX_GATHER_QMV_RHS_RPS:-}" ] && EXO_ENV="$EXO_ENV MLX_GATHER_QMV_RHS_RPS=$MLX_GATHER_QMV_RHS_RPS"
    EXO_ENV="$EXO_ENV EXO_PREFILL_STEP_SIZE=$EXO_PREFILL_STEP_SIZE"
    [ -n "${EXO_PREFILL_STEP_SIZE_HIGH_CTX:-}" ] && EXO_ENV="$EXO_ENV EXO_PREFILL_STEP_SIZE_HIGH_CTX=$EXO_PREFILL_STEP_SIZE_HIGH_CTX"
    [ -n "${EXO_PREFILL_STEP_SIZE_CROSSOVER:-}" ] && EXO_ENV="$EXO_ENV EXO_PREFILL_STEP_SIZE_CROSSOVER=$EXO_PREFILL_STEP_SIZE_CROSSOVER"
    EXO_ENV="$EXO_ENV EXO_DSV4_BATCHED_PREFILL=$EXO_DSV4_BATCHED_PREFILL"
    EXO_ENV="$EXO_ENV EXO_BATCHED_PREFILL_RENDEZVOUS_MS=$EXO_BATCHED_PREFILL_RENDEZVOUS_MS"
    [ -n "$EXO_PROFILER" ]       && EXO_ENV="$EXO_ENV EXO_PROFILER=$EXO_PROFILER"
    [ -n "$EXO_PROFILER_SYNC_SPANS" ] && EXO_ENV="$EXO_ENV EXO_PROFILER_SYNC_SPANS=$EXO_PROFILER_SYNC_SPANS"
    [ -n "$EXO_DSV4_SECTION_TIME" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SECTION_TIME=$EXO_DSV4_SECTION_TIME"
    [ -n "$EXO_DSV4_SECTION_TIME_LOG_EVERY" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SECTION_TIME_LOG_EVERY=$EXO_DSV4_SECTION_TIME_LOG_EVERY"
    [ -n "$EXO_DSV4_PREFILL_ARGPARTITION" ] && EXO_ENV="$EXO_ENV EXO_DSV4_PREFILL_ARGPARTITION=$EXO_DSV4_PREFILL_ARGPARTITION"
    [ -n "${EXO_DSV4_ARGPARTITION_MIN_P:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_ARGPARTITION_MIN_P=$EXO_DSV4_ARGPARTITION_MIN_P"
    [ -n "$EXO_DSV4_LMHEAD_LASTROW" ] && EXO_ENV="$EXO_ENV EXO_DSV4_LMHEAD_LASTROW=$EXO_DSV4_LMHEAD_LASTROW"
    [ -n "$EXO_DSV4_SEQ_SPLIT" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SEQ_SPLIT=$EXO_DSV4_SEQ_SPLIT"
    # JIT model lifecycle (master + API read these; always forwarded so the
    # kill-switch and reserve are explicit in the runner env).
    EXO_ENV="$EXO_ENV EXO_JIT_ENABLED=$EXO_JIT_ENABLED"
    EXO_ENV="$EXO_ENV EXO_JIT_MEMORY_RESERVE_GB=$EXO_JIT_MEMORY_RESERVE_GB"
    EXO_ENV="$EXO_ENV EXO_JIT_LOAD_TIMEOUT_SECONDS=$EXO_JIT_LOAD_TIMEOUT_SECONDS"
    EXO_ENV="$EXO_ENV EXO_JIT_IDLE_UNLOAD_SECONDS=$EXO_JIT_IDLE_UNLOAD_SECONDS"
    EXO_ENV="$EXO_ENV EXO_JIT_PLACEMENT_WAIT_SECONDS=$EXO_JIT_PLACEMENT_WAIT_SECONDS"
    [ -n "${EXO_DSV4_HEAPCENSUS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_HEAPCENSUS=$EXO_DSV4_HEAPCENSUS"
    [ -n "$EXO_DSV4_SEQ_SPLIT_MIN_L" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SEQ_SPLIT_MIN_L=$EXO_DSV4_SEQ_SPLIT_MIN_L"
    [ -n "$EXO_PREFIX_CACHE_TRACE" ] && EXO_ENV="$EXO_ENV EXO_PREFIX_CACHE_TRACE=$EXO_PREFIX_CACHE_TRACE"
    [ -n "$EXO_RUNNER_COREDUMP" ] && EXO_ENV="$EXO_ENV EXO_RUNNER_COREDUMP=$EXO_RUNNER_COREDUMP"
    [ -n "$EXO_MEMORY_PROFILE_PATH" ] && EXO_ENV="$EXO_ENV EXO_MEMORY_PROFILE_PATH=$EXO_MEMORY_PROFILE_PATH EXO_MEMORY_PROFILE_INTERVAL=$EXO_MEMORY_PROFILE_INTERVAL"
    [ -n "$EXO_TRACEMALLOC_PATH" ] && EXO_ENV="$EXO_ENV EXO_TRACEMALLOC_PATH=$EXO_TRACEMALLOC_PATH EXO_TRACEMALLOC_INTERVAL=$EXO_TRACEMALLOC_INTERVAL EXO_TRACEMALLOC_TOP_N=$EXO_TRACEMALLOC_TOP_N"
    [ "$EXO_MLX_CLEAR_CACHE_INTERVAL" != "0" ] && EXO_ENV="$EXO_ENV EXO_MLX_CLEAR_CACHE_INTERVAL=$EXO_MLX_CLEAR_CACHE_INTERVAL"
    EXO_ENV="$EXO_ENV EXO_GC_COLLECT_INTERVAL=$EXO_GC_COLLECT_INTERVAL"
    [ "$EXO_MALLOC_RELIEF_INTERVAL" != "0" ] && EXO_ENV="$EXO_ENV EXO_MALLOC_RELIEF_INTERVAL=$EXO_MALLOC_RELIEF_INTERVAL"
    [ -n "$EXO_PROFILER_LEVEL" ] && EXO_ENV="$EXO_ENV EXO_PROFILER_LEVEL=$EXO_PROFILER_LEVEL"
    EXO_ENV="$EXO_ENV EXO_LAYER_EVAL_INTERVAL=$EXO_LAYER_EVAL_INTERVAL"
    EXO_ENV="$EXO_ENV EXO_DRAFT_KV_WINDOW=$EXO_DRAFT_KV_WINDOW"
    EXO_ENV="$EXO_ENV EXO_KV_CACHE_BITS=$EXO_KV_CACHE_BITS"
    if [ -n "$EXO_TURBOQUANT" ]; then
        EXO_ENV="$EXO_ENV EXO_TURBOQUANT=$EXO_TURBOQUANT"
    fi
    EXO_ENV="$EXO_ENV EXO_SPECULATIVE=$EXO_SPECULATIVE"
    EXO_ENV="$EXO_ENV EXO_SPECULATIVE_GAMMA=$EXO_SPECULATIVE_GAMMA"
    [ -n "${EXO_QWEN_SPECULATIVE_GAMMA:-}" ] && EXO_ENV="$EXO_ENV EXO_QWEN_SPECULATIVE_GAMMA=$EXO_QWEN_SPECULATIVE_GAMMA"
    EXO_ENV="$EXO_ENV EXO_COMPUTE_DTYPE=$EXO_COMPUTE_DTYPE"
    EXO_ENV="$EXO_ENV EXO_RUNNER_QOS=$EXO_RUNNER_QOS"
    EXO_ENV="$EXO_ENV LOG_LEVEL=$LOG_LEVEL"

    # Cluster-wide sampling defaults (only export if explicitly set).
    [ -n "$EXO_DEFAULT_TEMPERATURE" ] && EXO_ENV="$EXO_ENV EXO_DEFAULT_TEMPERATURE=$EXO_DEFAULT_TEMPERATURE"
    [ -n "$EXO_DEFAULT_TOP_P" ]       && EXO_ENV="$EXO_ENV EXO_DEFAULT_TOP_P=$EXO_DEFAULT_TOP_P"
    [ -n "$EXO_DEFAULT_TOP_K" ]       && EXO_ENV="$EXO_ENV EXO_DEFAULT_TOP_K=$EXO_DEFAULT_TOP_K"
    [ -n "$EXO_DEFAULT_MIN_P" ]       && EXO_ENV="$EXO_ENV EXO_DEFAULT_MIN_P=$EXO_DEFAULT_MIN_P"

    # DSv4 fused MoE gate+up (single gather_qmm dispatch). Off by default
    # while we validate decode quality vs unfused.
    [ -n "$EXO_DSV4_FUSED_MOE" ]       && EXO_ENV="$EXO_ENV EXO_DSV4_FUSED_MOE=$EXO_DSV4_FUSED_MOE"
    [ -n "${EXO_DSV4_MOE_FUSED_GATE_UP:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MOE_FUSED_GATE_UP=$EXO_DSV4_MOE_FUSED_GATE_UP"
    [ -n "${EXO_DSV4_COMPILE_FFN:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_COMPILE_FFN=$EXO_DSV4_COMPILE_FFN"
    [ -n "${EXO_DSV4_COMPILE_LAYER:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_COMPILE_LAYER=$EXO_DSV4_COMPILE_LAYER"
    [ -n "${EXO_DSV4_FENCE_EVERY_N_LAYERS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_FENCE_EVERY_N_LAYERS=$EXO_DSV4_FENCE_EVERY_N_LAYERS"
    [ -n "${EXO_DSV4_ALLSUM_PROBE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_ALLSUM_PROBE=$EXO_DSV4_ALLSUM_PROBE"
    [ -n "${EXO_DSV4_ALLSUM_PROBE_LOG_EVERY:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_ALLSUM_PROBE_LOG_EVERY=$EXO_DSV4_ALLSUM_PROBE_LOG_EVERY"
    # Decode fence overlap (2026-07-02): non-blocking per-layer fence
    # (mx.async_eval) at the DSv4 MoE all_sum site, armed ONLY at c=1
    # steady state via a two-key side channel (batch_generate "engine" key:
    # exactly one active request; dsv4_mtp "cache" key: single-uid steady,
    # disarm+synchronize around cache transitions). Measured: c=1 decode
    # 28.9 -> 37.0 t/s (+28%), outputs byte-identical (blocking-fence
    # parity), needle/BOS clean. At c>=2 the fence stays blocking (keys
    # disarmed).
    # NOTE (history): the 2026-07-02 c=2 JOIN corruption was the per-stream
    # ring bootstrap (fixed, mlx-lm 8b7b5f9) — unrelated to this c=1 fence.
    # The 2026-07-03 c=2 DEEP-generation corruption is a different bug,
    # implicated on FENCE_ASYNC_C2 (see below); this B==1-gated arming is
    # not affected.
    : "${EXO_DSV4_FENCE_ASYNC:=1}"
    [ -n "${EXO_DSV4_FENCE_ASYNC:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_FENCE_ASYNC=$EXO_DSV4_FENCE_ASYNC"
    # c=2 decode levers. Per-stream acceptance (BS_MIN_ACCEPT=0) kept from
    # the 2026-07-02 A/B (+24% vs min-clamp, exonerated for corruption by
    # the 2026-07-03 deep battery).
    #
    # FENCE_ASYNC_C2 DEFAULT 0 (2026-07-03): async fencing at c=2 is
    # IMPLICATED in the deep-generation corruption. 4000-token divergent
    # c=2 pairs with FENCE_ASYNC_C2=2 degenerate at ~27%/pair (repetition
    # loops at tokens 1400-3900, kill-switch fires, then the mid-batch
    # kill rank-desyncs a collective -> 100% CPU spin wedge needing
    # kill -9). Same battery with FENCE_ASYNC_C2=0: 12/12 clean.
    # Exonerated by direct A/B: MLX_LM_SDPA_ROWSPLIT, BS_MIN_ACCEPT,
    # MTP-off (clean -> spec path), preceded-by-c1, xctrace attach.
    # Part 6's 800-token validation was too shallow to catch this.
    # Cost: c=2 drops ~24.5 -> ~19 t/s/stream; c=1 keeps its +28% async
    # win (FENCE_ASYNC=1 arming is B==1-gated and validated deep).
    # Re-enable only with a fix for the steady-state batched-cycle race
    # (per-stream ring trims vs in-flight deferred async graphs) plus a
    # 4000-token c=2 battery. See MOE_KERNEL_HANDOFF.md 2026-07-03.
    : "${EXO_DSV4_FENCE_ASYNC_C2:=0}"
    : "${EXO_DSV4_BS_MIN_ACCEPT:=0}"
    [ -n "${EXO_DSV4_FENCE_ASYNC_C2:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_FENCE_ASYNC_C2=$EXO_DSV4_FENCE_ASYNC_C2"
    [ -n "${EXO_DSV4_BS_MIN_ACCEPT:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_BS_MIN_ACCEPT=$EXO_DSV4_BS_MIN_ACCEPT"
    [ -n "${EXO_DSV4_ROUTE_HIST:-}" ]   && EXO_ENV="$EXO_ENV EXO_DSV4_ROUTE_HIST=$EXO_DSV4_ROUTE_HIST"
    [ -n "${EXO_DSV4_ROUTE_HIST_DECODE_ONLY:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_ROUTE_HIST_DECODE_ONLY=$EXO_DSV4_ROUTE_HIST_DECODE_ONLY"
    [ -n "${EXO_DSV4_TOPK_FUSED:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_TOPK_FUSED=$EXO_DSV4_TOPK_FUSED"
    [ -n "$EXO_DSV4_INDEX_TOPK" ]      && EXO_ENV="$EXO_ENV EXO_DSV4_INDEX_TOPK=$EXO_DSV4_INDEX_TOPK"
    [ -n "${EXO_DSV4_MTP:-}" ]         && EXO_ENV="$EXO_ENV EXO_DSV4_MTP=$EXO_DSV4_MTP"
    # c>=2 MTP spec gate: =1 => spec-off at c>=2 (clean, non-spec batched
    # decode). INTERIM as of 2026-07-04 pending the batch-invariant bf16
    # kernel fix. The residual c>=2 corruption is NOT the ring-bootstrap bug
    # (that's fixed, mlx-lm 8b7b5f9); it is batch-dependent bf16 rounding
    # DRIFT in the decode: on-cluster spec-trace showed a c=2 stream match
    # its canonical c=1 trajectory BITWISE for 75 tokens then flip a near-tie,
    # which cascades into a repetition attractor (~23% of deep temp-1.0 c=2
    # pairs). fp32 activations fix it (batch-invariant, proven: 0 server degen)
    # but reliably crash this cluster's jaccl/RDMA transport at ~2 c=2 pairs
    # (EXO_DSV4_FP32_ACT, off by default). The real fix is batch-invariant
    # bf16 kernels (fixed reduction order). Until then: spec-off at c>=2.
    # Set =0 to re-enable c>=2 spec (fast but ~23% corrupt).
    : "${EXO_DSV4_MTP_C2_MAX_CTX:=0}"  # 0 = MTP ON at all c>=2 ctx (pooling B-invariance fix, validated)
    [ -n "${EXO_DSV4_MTP_C2_MAX_CTX:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_C2_MAX_CTX=$EXO_DSV4_MTP_C2_MAX_CTX"
    [ -n "${EXO_DSV4_MTP_C2_GATE_DEBUG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_C2_GATE_DEBUG=$EXO_DSV4_MTP_C2_GATE_DEBUG"
    # jaccl start-of-collective ACK barrier (mesh_impl.h ack_sync_pre). Keeps
    # the two TP ranks in lockstep at each collective boundary so a leading
    # rank's data/ack SEND never arrives before the peer posts its RECV WR —
    # the UC-silent-drop that wedges drain_acks/the data loop under c>=2. The
    # dedicated ACK QP (mesh.cpp 2026-05-17) fixes ack_sync_POST; this closes
    # the remaining PRE-side data-path race. Gated OFF upstream for A/B.
    [ -n "${MLX_JACCL_ACK_SYNC_PRE:-}" ] && EXO_ENV="$EXO_ENV MLX_JACCL_ACK_SYNC_PRE=$MLX_JACCL_ACK_SYNC_PRE"
    [ -n "${JACCL_POLL_INSTRUMENT:-}" ] && EXO_ENV="$EXO_ENV JACCL_POLL_INSTRUMENT=$JACCL_POLL_INSTRUMENT"
    [ -n "${JACCL_POLL_INSTRUMENT_THRESHOLD_US:-}" ] && EXO_ENV="$EXO_ENV JACCL_POLL_INSTRUMENT_THRESHOLD_US=$JACCL_POLL_INSTRUMENT_THRESHOLD_US"
    [ -n "${JACCL_TRACE_PROGRESS:-}" ] && EXO_ENV="$EXO_ENV JACCL_TRACE_PROGRESS=$JACCL_TRACE_PROGRESS"
    # Runner hang watchdog (supervisor _check_hang). Default 45s SIGKILLs a
    # runner wedged in a native jaccl collective under c>=2 load (self-heal).
    # Raise it for diagnostics to widen the sampling window before the kill.
    [ -n "${EXO_RUNNER_HANG_TIMEOUT_SECONDS:-}" ] && EXO_ENV="$EXO_ENV EXO_RUNNER_HANG_TIMEOUT_SECONDS=$EXO_RUNNER_HANG_TIMEOUT_SECONDS"
    # Batch-invariant matmul (mlx-lm deepseek_v4): per-row gemv for small M so
    # c>=2 decode bitwise-matches c=1 — the bf16 batch-drift corruption fix.
    [ -n "${EXO_DSV4_BATCH_INVARIANT_MM:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_BATCH_INVARIANT_MM=$EXO_DSV4_BATCH_INVARIANT_MM"
    [ -n "${EXO_DSV4_BATCH_INVARIANT_MM_MAX_M:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_BATCH_INVARIANT_MM_MAX_M=$EXO_DSV4_BATCH_INVARIANT_MM_MAX_M"
    [ -n "${EXO_DSV4_BATCHED_PREFILL_DEBUG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_BATCHED_PREFILL_DEBUG=$EXO_DSV4_BATCHED_PREFILL_DEBUG"
    [ -n "${EXO_HC_USE_OPS:-}" ]       && EXO_ENV="$EXO_ENV EXO_HC_USE_OPS=$EXO_HC_USE_OPS"
    [ -n "${EXO_DSV4_ACT_PROBE:-}" ]   && EXO_ENV="$EXO_ENV EXO_DSV4_ACT_PROBE=$EXO_DSV4_ACT_PROBE"
    [ -n "${EXO_DSV4_MTP_DEDICATED:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_DEDICATED=$EXO_DSV4_MTP_DEDICATED"
    [ -n "${EXO_LEAF_SNAPSHOT_RETENTION:-}" ] && EXO_ENV="$EXO_ENV EXO_LEAF_SNAPSHOT_RETENTION=$EXO_LEAF_SNAPSHOT_RETENTION"
    [ -n "${EXO_ARRAYSCACHE_DIAG:-}" ] && EXO_ENV="$EXO_ENV EXO_ARRAYSCACHE_DIAG=$EXO_ARRAYSCACHE_DIAG"
    # Eagle soft-embedding for chained MTP draft (Phase 14 Plan B.2).
    # Default OFF (0): mlx-lm's DeepseekV4MTPModule.__call__ uses the
    # hard-argmax embed_tokens() lookup — bit-exact with prior behavior.
    # When > 0: at every chained draft step beyond the first, the input
    # embedding is replaced with a probability-weighted top-K mixture
    # built from the previous step's logits. Targets step-1 P(top-1)
    # acceptance lift. Requires the _EAGLE_CTX side channel, which is on
    # adurham/mlx-lm@main (cluster's pin since 2026-05-29; formerly the
    # eagle-soft-emb branch, now merged into main). Recommended K=8.
    [ -n "${EXO_DSV4_MTP_EAGLE_K:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_EAGLE_K=$EXO_DSV4_MTP_EAGLE_K"
    [ -n "${EXO_DSV4_MTP_EAGLE_T:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_EAGLE_T=$EXO_DSV4_MTP_EAGLE_T"
    [ -n "${EXO_DSV4_MTP_LOG_INTERVAL:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_LOG_INTERVAL=$EXO_DSV4_MTP_LOG_INTERVAL"
    [ -n "${EXO_DSV4_MTP_PROFILE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_PROFILE=$EXO_DSV4_MTP_PROFILE"
    [ -n "${EXO_DSV4_MTP_REFCHECK_BATCH:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_REFCHECK_BATCH=$EXO_DSV4_MTP_REFCHECK_BATCH"
    [ -n "${EXO_DSV4_MTP_NO_BROADCAST:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_NO_BROADCAST=$EXO_DSV4_MTP_NO_BROADCAST"
    # W3 diagnostic: per-draft-step top-8 softmax dump. NOT a production
    # knob — diagnostic only. Cost when off: zero (single env.get hit).
    # Cost when on: ~1 sync per draft step (microseconds). See
    # mtp_module.py:740 dump block and /tmp/w3_eagle_audit.md.
    [ -n "${EXO_DSV4_MTP_DUMP_TOPK:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_DUMP_TOPK=$EXO_DSV4_MTP_DUMP_TOPK"
    # MTP verify-audit JSONL path (diagnostic: special-token draft/accept dumps).
    [ -n "${EXO_DSV4_MTP_VERIFY_AUDIT:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_VERIFY_AUDIT=$EXO_DSV4_MTP_VERIFY_AUDIT"
    # MTP reference-forward refcheck JSONL path (diagnostic: verify vs clean greedy).
    [ -n "${EXO_DSV4_MTP_REFCHECK:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_REFCHECK=$EXO_DSV4_MTP_REFCHECK"
    # MTP refcheck EVERY-CYCLE mode (1 = run ref forward every cycle, log divergences).
    [ -n "${EXO_DSV4_MTP_REFCHECK_ALL:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_REFCHECK_ALL=$EXO_DSV4_MTP_REFCHECK_ALL"
    # MTP tie-break losslessness fix (1 = recompute near-tie bonus via single-token forward).
    [ -n "${EXO_DSV4_MTP_TIEBREAK_FIX:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_TIEBREAK_FIX=$EXO_DSV4_MTP_TIEBREAK_FIX"
    [ -n "${EXO_DSV4_MTP_TIEBREAK_EPS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_TIEBREAK_EPS=$EXO_DSV4_MTP_TIEBREAK_EPS"
    # Greedy accept-rule alignment (see defaults block above).
    [ -n "${EXO_DSV4_MTP_ACCEPT_LOGPROBS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_ACCEPT_LOGPROBS=$EXO_DSV4_MTP_ACCEPT_LOGPROBS"
    [ -n "${EXO_DSV4_POOL_SNAPSHOT_BATCH:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_POOL_SNAPSHOT_BATCH=$EXO_DSV4_POOL_SNAPSHOT_BATCH"
    # Regime-b double-rollback fix (see defaults block above).
    [ -n "${EXO_DSV4_POOL_RESTORE_AFTER_TRIM:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_POOL_RESTORE_AFTER_TRIM=$EXO_DSV4_POOL_RESTORE_AFTER_TRIM"
    # Per-request MTP cycle statistics (diagnostic; one log line per stream).
    [ -n "${EXO_DSV4_MTP_CYCLE_STATS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_CYCLE_STATS=$EXO_DSV4_MTP_CYCLE_STATS"
    # Rowseq per-row REAL decode masks (batch-cache SDPA parity).
    [ -n "${EXO_DSV4_ROWSEQ_ROWMASK:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_ROWSEQ_ROWMASK=$EXO_DSV4_ROWSEQ_ROWMASK"
    # Unified bitwise-faithful spec rollback (ring+pool wholesale restore).
    [ -n "${EXO_DSV4_SPEC_STATE_RESTORE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SPEC_STATE_RESTORE=$EXO_DSV4_SPEC_STATE_RESTORE"
    [ -n "${EXO_DSV4_SPEC_CACHE_ROLLBACK:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SPEC_CACHE_ROLLBACK=$EXO_DSV4_SPEC_CACHE_ROLLBACK"
    [ -n "${EXO_DSV4_SPEC_CACHE_ROLLBACK_C2:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SPEC_CACHE_ROLLBACK_C2=$EXO_DSV4_SPEC_CACHE_ROLLBACK_C2"
    [ -n "${EXO_DSV4_ROWSEQ_FULLBLOCK:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_ROWSEQ_FULLBLOCK=$EXO_DSV4_ROWSEQ_FULLBLOCK"
    [ -n "${EXO_DSV4_ROWSEQ_FULLBLOCK_MOE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=$EXO_DSV4_ROWSEQ_FULLBLOCK_MOE"
    [ -n "${EXO_DSV4_MOE_PARTS_ROWSEQ:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MOE_PARTS_ROWSEQ=$EXO_DSV4_MOE_PARTS_ROWSEQ"
    [ -n "${EXO_DSV4_LAYER_HASH_DUMP:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_LAYER_HASH_DUMP=$EXO_DSV4_LAYER_HASH_DUMP"
    [ -n "${EXO_DSV4_LAYER_HASH_MAX_POS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_LAYER_HASH_MAX_POS=$EXO_DSV4_LAYER_HASH_MAX_POS"
    [ -n "${EXO_DSV4_LAYER_HASH_SUBOPS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_LAYER_HASH_SUBOPS=$EXO_DSV4_LAYER_HASH_SUBOPS"
    [ -n "${EXO_DSV4_SPEC_RB_LOG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SPEC_RB_LOG=$EXO_DSV4_SPEC_RB_LOG"
    # Long-ctx MTP gate + near-tie re-verify (see defaults block above).
    [ -n "${EXO_DSV4_MTP_MAX_CTX:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_MAX_CTX=$EXO_DSV4_MTP_MAX_CTX"
    [ -n "${EXO_DSV4_MTP_TIE_REVERIFY:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_TIE_REVERIFY=$EXO_DSV4_MTP_TIE_REVERIFY"
    [ -n "${EXO_DSV4_MTP_TIE_REVERIFY_EPS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_TIE_REVERIFY_EPS=$EXO_DSV4_MTP_TIE_REVERIFY_EPS"
    [ -n "${EXO_DSV4_MTP_TIE_REVERIFY_LOG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_TIE_REVERIFY_LOG=$EXO_DSV4_MTP_TIE_REVERIFY_LOG"
    # Row-sequential verify attention (the losslessness root fix).
    [ -n "${EXO_DSV4_VERIFY_ROWSEQ:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_VERIFY_ROWSEQ=$EXO_DSV4_VERIFY_ROWSEQ"
    [ -n "${EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=$EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX"
    [ -n "${EXO_DSV4_VERIFY_ROWSEQ_MAX_L:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_VERIFY_ROWSEQ_MAX_L=$EXO_DSV4_VERIFY_ROWSEQ_MAX_L"
    # min_p tail-clip for the temp>0 MTP correction/bonus sampling (default 0.05
    # in code = the DSv4 card value; set 0 to disable for A/B). Stops MTP
    # committing extreme-tail tokens that seed structured-output degeneration.
    [ -n "${EXO_DSV4_MTP_MIN_P:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_MIN_P=$EXO_DSV4_MTP_MIN_P"
    # γ=3 c=2 bistability tracer (see dsv4_mtp.py::_draft_tokens_batched).
    # When 1, writes /tmp/dsv4_c2_trace_pid<PID>.jsonl with per-step
    # timestamps + per-stream tokens. NOT a production knob — diagnostic
    # only. Inserts mx.eval() at every chain-step boundary, which acts
    # like a per-step fence, so do NOT validate fixes with this on.
    [ -n "${EXO_DSV4_C2_TRACE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_C2_TRACE=$EXO_DSV4_C2_TRACE"
    # Degeneration-hunt per-cycle spec trace (2026-05-29). When 1,
    # _speculative_next dumps committed tokens + cache offsets + n_accepted
    # per cycle to /tmp/dsv4_spec_trace_pid<PID>.jsonl on rank 0. Diagnostic
    # only — pairs with a plain-greedy capture to find first divergence.
    [ -n "${EXO_DSV4_SPEC_TRACE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SPEC_TRACE=$EXO_DSV4_SPEC_TRACE"
    # c>=2 degen-kill WEDGE tracer (2026-07-03). When 1, batch_generate.step
    # logs per-rank batch size + every generator eviction so a cross-rank diff
    # shows whether a mid-batch degen eviction (and its B-transition) is
    # symmetric. Diagnostic only.
    [ -n "${EXO_DSV4_WEDGE_TRACE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_WEDGE_TRACE=$EXO_DSV4_WEDGE_TRACE"
    [ -n "${EXO_DSV4_WEDGE_INJECT:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_WEDGE_INJECT=$EXO_DSV4_WEDGE_INJECT"
    # Runner hang-watchdog timeout (supervisor.py). Default 45s in-code; lower
    # only to validate the watchdog (a short value risks false kills under
    # legitimately slow steps).
    [ -n "${EXO_RUNNER_HANG_TIMEOUT_SECONDS:-}" ] && EXO_ENV="$EXO_ENV EXO_RUNNER_HANG_TIMEOUT_SECONDS=$EXO_RUNNER_HANG_TIMEOUT_SECONDS"
    # c=2 corruption ROOT-CAUSE fix (2026-07-03): fp32 activations make the
    # DSv4 forward batch-invariant (bf16's batch-size-dependent rounding flips
    # ~17% of argmaxes at B=2 vs B=1 -> temp>0 repetition degeneration). Weights
    # stay bf16/quantized so the weight-bandwidth-bound cost is ~unchanged.
    [ -n "${EXO_DSV4_FP32_ACT:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_FP32_ACT=$EXO_DSV4_FP32_ACT"
    [ -n "${EXO_DSV4_FP32_COLL_LOG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_FP32_COLL_LOG=$EXO_DSV4_FP32_COLL_LOG"
    [ -n "${EXO_DSV4_VERIFY_TRACE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_VERIFY_TRACE=$EXO_DSV4_VERIFY_TRACE"
    [ -n "${EXO_DSV4_VERIFY_DIAG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_VERIFY_DIAG=$EXO_DSV4_VERIFY_DIAG"
    # Phase 1.2 token-tree alpha distribution probe. When 1, draft_tokens
    # logs MTP top-5 IDs and _speculative_next joins them with verify-target
    # argmax to /tmp/dsv4_tree_alpha_probe_pid<PID>.jsonl on rank 0 only.
    [ -n "${EXO_DSV4_TREE_ALPHA_PROBE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_TREE_ALPHA_PROBE=$EXO_DSV4_TREE_ALPHA_PROBE"
    # Token-tree drafting (Phases 2-7 of the May-19 plan). When 1,
    # _speculative_next routes to _speculative_next_tree which uses K^gamma
    # top-K MTP expansion + tree-attention verify instead of the linear
    # gamma chain. K is set by EXO_DSV4_TREE_K (default 2). Greedy temp=0
    # only; temp>0 falls back to the linear path.
    [ -n "${EXO_DSV4_TREE_DRAFT:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_TREE_DRAFT=$EXO_DSV4_TREE_DRAFT"
    [ -n "${EXO_DSV4_TREE_K:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_TREE_K=$EXO_DSV4_TREE_K"
    # Greedy tree: only top-1 d1 expands d2 children. Cuts L_q=7 -> L_q=5
    # for K=2 gamma=2. Trades worse-case acceptance for cheaper verify.
    [ -n "${EXO_DSV4_TREE_GREEDY:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_TREE_GREEDY=$EXO_DSV4_TREE_GREEDY"
    # One-shot first-cycle diagnostic for the tree-verify side channel.
    # When 1, dsv4_mtp.py logs n_nodes/parent_idx/depth/mask.shape/positions
    # on the first cycle to ~/exo.log. Default off; off-state is bit-exact.
    [ -n "${EXO_DSV4_TREE_DEBUG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_TREE_DEBUG=$EXO_DSV4_TREE_DEBUG"
    [ -n "${EXO_DSV4_PSCACHE_DEBUG:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_PSCACHE_DEBUG=$EXO_DSV4_PSCACHE_DEBUG"
    [ -n "$EXO_DSV4_INDEXER_WINDOW" ]  && EXO_ENV="$EXO_ENV EXO_DSV4_INDEXER_WINDOW=$EXO_DSV4_INDEXER_WINDOW"
    [ -n "$EXO_DSV4_INDEXER_WINDOW_LATE" ] && EXO_ENV="$EXO_ENV EXO_DSV4_INDEXER_WINDOW_LATE=$EXO_DSV4_INDEXER_WINDOW_LATE"
    # Tiled-P indexer score block size (mlx-lm deepseek_v4 _indexer_score_tiled).
    # >0 caps the (B,64,L,P) indexer transient by processing pooled-P in blocks —
    # bounds the high-context prefill alloc spikes. Default OFF in the model.
    [ -n "${EXO_DSV4_INDEXER_PBLOCK:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_INDEXER_PBLOCK=$EXO_DSV4_INDEXER_PBLOCK"
    # MLX SDPA 2-pass blocks-heuristic override (Phase 2 exp 2 sweep).
    [ -n "$MLX_SDPA_BLOCKS" ]          && EXO_ENV="$EXO_ENV MLX_SDPA_BLOCKS=$MLX_SDPA_BLOCKS"
    # mlx-lm B>1/L>1 SDPA row-split kill switch (c=2 deep-degen A/B).
    [ -n "${MLX_LM_SDPA_ROWSPLIT:-}" ] && EXO_ENV="$EXO_ENV MLX_LM_SDPA_ROWSPLIT=$MLX_LM_SDPA_ROWSPLIT"
    # JACCL per-call trace (Phase 0 c=2 corruption diagnostic). When set
    # to 1, every collective entry writes one line to
    # /tmp/jaccl_trace_rank_${MLX_RANK}.log on each runner host. Use
    # for finding the first cross-rank divergence in the call sequence;
    # don't leave on permanently (fflush per call slows decode).
    [ -n "${JACCL_TRACE_CALLS:-}" ]    && EXO_ENV="$EXO_ENV JACCL_TRACE_CALLS=$JACCL_TRACE_CALLS"
    # Skip the cross-rank ack_sync_pre round-trip when the prior
    # collective on the same MeshGroup was its own ack_sync_post. At
    # gamma=2 100K decode this saves ~8 ms/forward (172 all_reduces ×
    # 50 us per skipped pre). Off by default; set to 1 to enable.
    # Implementation: mlx commit 0b8aca69 (mesh_impl.h fastskip).
    [ -n "${EXO_JACCL_ACK_PRE_FASTSKIP:-}" ] && EXO_ENV="$EXO_ENV EXO_JACCL_ACK_PRE_FASTSKIP=$EXO_JACCL_ACK_PRE_FASTSKIP"
    # Per-call output-hash diagnostic; orthogonal to JACCL_TRACE_CALLS
    # gating but uses the same trace file. Identifies transport
    # non-bit-exactness as a divergent hash at a specific call_id.
    [ -n "${JACCL_TRACE_HASH:-}" ]     && EXO_ENV="$EXO_ENV JACCL_TRACE_HASH=$JACCL_TRACE_HASH"
    # Per-stage RDMA progress logging (mesh_impl all_reduce +
    # drain_acks). Writes [jaccl-prog] lines to ~/exo.log stderr.
    # Localizes a wedge to a specific RDMA stage: ENTER /
    # PREFILL_DONE / POLL / CQE / DATA_DONE / ack POSTED / ack
    # DRAINED / DONE.
    [ -n "${JACCL_TRACE_PROGRESS:-}" ] && EXO_ENV="$EXO_ENV JACCL_TRACE_PROGRESS=$JACCL_TRACE_PROGRESS"
    # MLX_JACCL_ACK_SYNC_PRE: gate the ce5c64fd pre-lambda ack barrier.
    # Default OFF. Set to 1 to enable the start-of-lambda cross-rank
    # ACK round-trip (one extra ACK_SEND/RECV pair on the dedicated
    # ACK QP per collective). Intended to close the inter-lambda race
    # where peer SEND beats our recv-post on the data QP and UC
    # silently drops. Requires the bootstrap barrier in MeshGroup
    # ctors (mlx commit 3882458d on try/ack-qp-isolated). Off-by-
    # default preserves the post-Plan-A baseline; bench-time opt-in
    # via this env. See mlx/mlx/distributed/jaccl/mesh_impl.h
    # jaccl_ack_sync_pre_enabled() and the long doc block at the top.
    # Default flipped ON 2026-07-04: measured to keep c>=2 wedge failures
    # CLEANER — ACK_SYNC_PRE=1 self-heals with 0 IOConnectUnmapMemory GPU
    # faults; =0 saw the peer GPU-fault and the re-place stick in
    # RunnerConnecting. Pairs with the StallWatch UC-drop recovery
    # (mlx a5be4403). Set =0 to A/B the old off-by-default behavior.
    : "${MLX_JACCL_ACK_SYNC_PRE:=1}"
    [ -n "${MLX_JACCL_ACK_SYNC_PRE:-}" ] && EXO_ENV="$EXO_ENV MLX_JACCL_ACK_SYNC_PRE=$MLX_JACCL_ACK_SYNC_PRE"
    # MLX_JACCL_RECONNECT_FRESH: in-process device-context rebuild — the warmup
    # QP-flake fix (mlx e399ecfb); ~0.15s vs a 90s re-place. Validated prod default.
    : "${MLX_JACCL_RECONNECT_FRESH:=1}"
    [ -n "${MLX_JACCL_RECONNECT_FRESH:-}" ] && EXO_ENV="$EXO_ENV MLX_JACCL_RECONNECT_FRESH=$MLX_JACCL_RECONNECT_FRESH"
    # MLX_JACCL_RELIABLE_OPTIMISTIC: v2 small-collective path (no TCP barrier),
    # +29% decode @4K (mlx 57ffb39a). PP placements must keep this OFF —
    # PipelineLastLayer send/recv/all_gather ride the model group.
    : "${MLX_JACCL_RELIABLE_OPTIMISTIC:=1}"
    [ -n "${MLX_JACCL_RELIABLE_OPTIMISTIC:-}" ] && EXO_ENV="$EXO_ENV MLX_JACCL_RELIABLE_OPTIMISTIC=$MLX_JACCL_RELIABLE_OPTIMISTIC"
    # Pool-write donation threshold (session-4 pool fixes) — validated prod value.
    : "${EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES:=8388608}"
    [ -n "${EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=$EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES"
    # Stall sampler: cheap reboot-durable stack dumps when step() stops returning.
    : "${EXO_STALL_SAMPLER_SECONDS:=10}"
    [ -n "${EXO_STALL_SAMPLER_SECONDS:-}" ] && EXO_ENV="$EXO_ENV EXO_STALL_SAMPLER_SECONDS=$EXO_STALL_SAMPLER_SECONDS"
    # MLX_JACCL_CONFIRMED_BARRIER: reliable ack barrier over the TCP coordinator
    # instead of the UC ack exchange (which wedges on a lost completion).
    # Deterministic recv-side wedge PREVENTION for c>=2. Default off (adds a
    # coordinator round-trip per ack barrier); set =1 for c>=2 correctness.
    [ -n "${MLX_JACCL_CONFIRMED_BARRIER:-}" ] && EXO_ENV="$EXO_ENV MLX_JACCL_CONFIRMED_BARRIER=$MLX_JACCL_CONFIRMED_BARRIER"
    # Split gates to isolate pre vs post confirmed barrier (pre is entangled with
    # RDMA data-recv ordering; post runs after data drains).
    [ -n "${MLX_JACCL_CONFIRMED_BARRIER_PRE:-}" ]  && EXO_ENV="$EXO_ENV MLX_JACCL_CONFIRMED_BARRIER_PRE=$MLX_JACCL_CONFIRMED_BARRIER_PRE"
    [ -n "${MLX_JACCL_CONFIRMED_BARRIER_POST:-}" ] && EXO_ENV="$EXO_ENV MLX_JACCL_CONFIRMED_BARRIER_POST=$MLX_JACCL_CONFIRMED_BARRIER_POST"
    # MLX_JACCL_RELIABLE_DATA: reliable ARQ all_reduce data path (2-rank) — chunks
    # carry a seq header, receiver assembles + dedups + defers the reduce, and a
    # coordinator bitmask barrier retransmits missing chunks. Eliminates the
    # data-phase all_reduce STALLED wedge. Gated (perf cost + core-path change).
    : "${MLX_JACCL_RELIABLE_DATA:=1}"  # validated prod default (task #23/24 wedge fix)
    [ -n "${MLX_JACCL_RELIABLE_DATA:-}" ]         && EXO_ENV="$EXO_ENV MLX_JACCL_RELIABLE_DATA=$MLX_JACCL_RELIABLE_DATA"
    # MLX_JACCL_RELIABLE_MAX_SZ: cap reliable chunk size class (0=4KB..7=512KB).
    # Larger => fewer chunks => faster, but must still reliably COMPLETE on
    # librdma (large UC sends >=64KB/sz>=4 stick). Bisect for the sweet spot.
    : "${MLX_JACCL_RELIABLE_MAX_SZ:=2}"  # 16KB — MUST stay <=2 (>=sz4 UC sends stick)
    [ -n "${MLX_JACCL_RELIABLE_MAX_SZ:-}" ]        && EXO_ENV="$EXO_ENV MLX_JACCL_RELIABLE_MAX_SZ=$MLX_JACCL_RELIABLE_MAX_SZ"
    # MLX_JACCL_RELIABLE_IDLE_US: sleep per idle drain poll (anti-CPU-spin).
    [ -n "${MLX_JACCL_RELIABLE_IDLE_US:-}" ]       && EXO_ENV="$EXO_ENV MLX_JACCL_RELIABLE_IDLE_US=$MLX_JACCL_RELIABLE_IDLE_US"
    # MLX_EVENT_WAIT_*: interruptible GPU-event wait (mlx event.cpp). Event::wait
    # now POLLS MTL::SharedEvent::signaledValue() in userspace instead of Apple's
    # waitUntilSignaledValue (which traps into an UNINTERRUPTIBLE kernel GPU-wait
    # that ignores its timeout when a c>=2 collective is wedged). Polling lets a
    # wedged PEER self-abort (surface a captured stream exception, or hit the
    # total timeout) and reach group.reconnect() for in-place transport recovery
    # instead of hanging until the _check_hang SIGKILL. TIMEOUT_MS default 20000
    # (< the 45s watchdog and >> any healthy/warmup wait; keeps the primary's
    # StallWatch->reconnect->coordinator-barrier wait for the peer well under the
    # watchdog). POLL_US = sleep granularity (default 50), SPIN = spins before
    # sleeping (default 2000). POLL_US=0 restores the legacy blocking wait.
    : "${MLX_EVENT_WAIT_TIMEOUT_MS:=20000}"
    [ -n "${MLX_EVENT_WAIT_TIMEOUT_MS:-}" ] && EXO_ENV="$EXO_ENV MLX_EVENT_WAIT_TIMEOUT_MS=$MLX_EVENT_WAIT_TIMEOUT_MS"
    [ -n "${MLX_EVENT_WAIT_POLL_US:-}" ]    && EXO_ENV="$EXO_ENV MLX_EVENT_WAIT_POLL_US=$MLX_EVENT_WAIT_POLL_US"
    [ -n "${MLX_EVENT_WAIT_SPIN:-}" ]       && EXO_ENV="$EXO_ENV MLX_EVENT_WAIT_SPIN=$MLX_EVENT_WAIT_SPIN"
    # MLX_DIAG_HOLD_WEDGE: diagnostic only — hold a c>=2 wedge open (no
    # reconnect/re-place) so the peer can be sampled in its real stuck location.
    [ -n "${MLX_DIAG_HOLD_WEDGE:-}" ]       && EXO_ENV="$EXO_ENV MLX_DIAG_HOLD_WEDGE=$MLX_DIAG_HOLD_WEDGE"
    # MLX_STREAM_QOS: env-gated QoS pin for mlx stream worker threads
    # (see scheduler.h). user_initiated mitigates the rank-0 comm-stream
    # poll-stall under MTP load. Default off.
    [ -n "${MLX_STREAM_QOS:-}" ]        && EXO_ENV="$EXO_ENV MLX_STREAM_QOS=$MLX_STREAM_QOS"
    # MLX_STREAM_RT[+_COMPUTATION_US/_CONSTRAINT_US/_PERIOD_US]: Mach
    # real-time time-constraint policy on stream worker threads. Hard
    # contract with the scheduler — kernel will not preempt during the
    # computation window. Fix for the asymmetric JACCL poll-stall.
    [ -n "${MLX_STREAM_RT:-}" ]                  && EXO_ENV="$EXO_ENV MLX_STREAM_RT=$MLX_STREAM_RT"
    [ -n "${MLX_STREAM_RT_COMPUTATION_US:-}" ]   && EXO_ENV="$EXO_ENV MLX_STREAM_RT_COMPUTATION_US=$MLX_STREAM_RT_COMPUTATION_US"
    [ -n "${MLX_STREAM_RT_CONSTRAINT_US:-}" ]    && EXO_ENV="$EXO_ENV MLX_STREAM_RT_CONSTRAINT_US=$MLX_STREAM_RT_CONSTRAINT_US"
    [ -n "${MLX_STREAM_RT_PERIOD_US:-}" ]        && EXO_ENV="$EXO_ENV MLX_STREAM_RT_PERIOD_US=$MLX_STREAM_RT_PERIOD_US"
    # JACCL_POLL_INSTRUMENT: per-call wall + in-poll diagnostic for
    # all_reduce stalls (see mesh_impl.h). Emits one stderr line per
    # call whose total wall time exceeds the threshold (default 100ms).
    [ -n "${JACCL_POLL_INSTRUMENT:-}" ]               && EXO_ENV="$EXO_ENV JACCL_POLL_INSTRUMENT=$JACCL_POLL_INSTRUMENT"
    [ -n "${JACCL_POLL_INSTRUMENT_THRESHOLD_US:-}" ]  && EXO_ENV="$EXO_ENV JACCL_POLL_INSTRUMENT_THRESHOLD_US=$JACCL_POLL_INSTRUMENT_THRESHOLD_US"
    # MLX_SIGNAL_PROBE: per-Event::signal diagnostic on the GPU stream
    # (see mlx/backend/metal/event.cpp). Emits two stderr lines per
    # signal: SIGNAL_PROBE_ENC (ops at encode, t_enc_us) and
    # SIGNAL_PROBE_DONE (t_done_us, gap_us = SharedEvent completion
    # latency). Used to verify the γ=2 MTP bistable-stall hypothesis
    # (decode-time signal lands at the tail of a deep command buffer).
    # Diagnostic only; ZERO overhead when unset.
    [ -n "${MLX_SIGNAL_PROBE:-}" ]                    && EXO_ENV="$EXO_ENV MLX_SIGNAL_PROBE=$MLX_SIGNAL_PROBE"
    # MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE: when set to 1, mlx will
    # force-commit the producing GPU stream's pending command buffer
    # before a CPU primitive (e.g. AllReduce) waits on its event.
    # Targets the bistable peer-CQE-arrival-latency stall under γ=2
    # MTP-on. mlx commit 4d21baa2 / branch mtp-allreduce-eager-commit.
    [ -n "${MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE:-}" ] && EXO_ENV="$EXO_ENV MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE=$MLX_EAGER_COMMIT_BEFORE_CPU_COLLECTIVE"
    # Subgroup split init progress trace (logs per-rank to stderr at
    # each QP-exchange step). Use to localize a deadlock during
    # `MeshGroup::split` itself. Memory: dsv4_mtp_c2_split_attempt_2026_05_07.md.
    [ -n "${JACCL_TRACE_SPLIT:-}" ]    && EXO_ENV="$EXO_ENV JACCL_TRACE_SPLIT=$JACCL_TRACE_SPLIT"
    # When set, MeshGroup::split opens a fresh ibv_context per
    # subgroup instead of borrowing the parent's. Required on macOS
    # librdma to fully isolate QPs across subgroups; without it both
    # subgroups share the parent's context and post-init collectives
    # deadlock after a few calls.
    [ -n "${JACCL_SPLIT_FRESH_CTX:-}" ] && EXO_ENV="$EXO_ENV JACCL_SPLIT_FRESH_CTX=$JACCL_SPLIT_FRESH_CTX"
    # When set, MeshGroup::split makes the subgroup share the parent
    # group's CPU stream (cpu::CommandEncoder thread) instead of
    # allocating its own. Funnels master + coord lambdas onto one
    # FIFO encoder thread — needed on macOS where two distinct
    # encoder threads dispatching concurrently into separate QP sets
    # appears to deadlock at the librdma layer.
    [ -n "${JACCL_SPLIT_PARENT_STREAM:-}" ] && EXO_ENV="$EXO_ENV JACCL_SPLIT_PARENT_STREAM=$JACCL_SPLIT_PARENT_STREAM"
    # Per-step BatchGenerator state snapshot; writes JSONL to
    # /tmp/jaccl_step_rank_${rank}_pid${pid}.log. Diff across ranks
    # to find the first asymmetric Python state on the prefix-cache
    # share path. Memory: next_session_plan_jaccl_c2_prefix_cache.md.
    [ -n "${JACCL_TRACE_STEP:-}" ]     && EXO_ENV="$EXO_ENV JACCL_TRACE_STEP=$JACCL_TRACE_STEP"
    # Per-MTP-chain-step drift diagnostic; writes JSONL to
    # /tmp/mtp_drift_rank_${rank}_pid${pid}.log. Cross-rank diff
    # localises where in the unsharded MTP chain logits first
    # drift across ranks. Memory: jaccl_phase_f_outcome_2026_05_06.md.
    [ -n "${EXO_MTP_DRIFT_DUMP:-}" ]    && EXO_ENV="$EXO_ENV EXO_MTP_DRIFT_DUMP=$EXO_MTP_DRIFT_DUMP"
    # Per-cycle dsv4_mtp BS-transition trace. Writes JSONL to
    # /tmp/dsv4_mtp_trace_rank_${rank}_pid${pid}.log. Used to
    # localize cross-rank divergence at the c=2→c=1 transition.
    [ -n "${EXO_DSV4_MTP_TRANSITION_TRACE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_MTP_TRANSITION_TRACE=$EXO_DSV4_MTP_TRANSITION_TRACE"

    # Metal GPU timeout mitigations
    if [ "$EXO_DISABLE_METAL_TIMEOUT" == "1" ]; then
        EXO_ENV="$EXO_ENV MTL_DISABLE_TIMEOUT=1 MTL_COMMAND_BUFFER_TIMEOUT=0 EXO_DISABLE_METAL_TIMEOUT=1"
    fi


    # Rotate log: keep previous run as exo.log.prev, then append
    ssh "$NODE" "cp ~/exo.log ~/exo.log.prev 2>/dev/null; : > ~/exo.log"

    # caffeinate -s starts as a separate background process and uses -w to
    # keep the system awake for as long as the exo process exists.
    # IMPORTANT: don't `caffeinate -s python` because caffeinate is hardened
    # and dyld strips DYLD_INSERT_LIBRARIES (and other DYLD_*) when exec-ing
    # into a hardened binary, breaking our abort_tracer interposer.
    # Regenerate the node-local relaunch script from THIS script's EXO_ENV so a
    # quick node-side restart (~/relaunch_exo.sh) always uses the exact env that
    # start_cluster.sh (the single source of truth) would launch with. Session
    # 2026-07-07 lesson: the hand-edited relaunch scripts and this script had
    # drifted apart (transport stack, prefill step, log level, ...); deriving
    # one from the other closes that class of drift permanently.
    if [ "$NODE" == "macstudio-m4-1" ]; then
        NODE_PEERS="/ip4/$M4_2_TO_M4_1/tcp/52415/p2p/$M4_2_PEER_ID"
    elif [ "$NODE" == "macstudio-m4-2" ]; then
        NODE_PEERS="/ip4/$M4_1_TO_M4_2/tcp/52415/p2p/$M4_1_PEER_ID"
    else
        NODE_PEERS="/ip4/$M4_1_TO_MBP/tcp/52415/p2p/$M4_1_PEER_ID"
    fi
    # LAUNCH_TAIL is single-quote-assigned so $!/$EXO_PID stay LITERAL until the
    # node's zsh runs them (double-quoted \$ escapes expanded locally when this
    # string was interpolated into ssh args — caffeinate got an empty pid).
    LAUNCH_TAIL='& EXO_PID=$!; caffeinate -s -w $EXO_PID 2>/dev/null & wait $EXO_PID'
    LAUNCH_CMD="cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=$NODE_PEERS .venv/bin/python -m exo -v >> ~/exo.log 2>&1 $LAUNCH_TAIL"
    # Generate the file LOCALLY (printf does not expand $ in arguments) and scp
    # it — a remote unquoted heredoc would expand $!/$EXO_PID a second time.
    RELAUNCH_TMP=$(mktemp)
    {
        printf '#!/bin/zsh\n'
        printf '# GENERATED by start_cluster.sh %s — do not hand-edit; env changes belong\n' "$(date +%Y-%m-%d)"
        printf '# in start_cluster.sh (single source of truth). This script only restarts\n'
        printf '# the exo PROCESS; it does NOT deploy code or place models (run\n'
        printf '# start_cluster.sh for the full canonical bring-up incl. pinned DSv4).\n'
        # Quoted heredoc: nothing expands locally; the body is literal zsh
        # that runs on the node.
        cat <<'RELAUNCH_BODY'
# Graceful kill of any live exo FIRST. SIGTERM lets runners tear down their
# RDMA QPs cleanly AND release their Metal buffers before exit. Never use
# `screen -X quit` / `pkill -9` here: both skip the destructors, leaking QPs
# (TB-stack wedge) and orphaning ~60-80 GB of wired pages the OS then takes
# ~a minute (or a reboot) to reclaim.
pkill -TERM -f 'python.*exo' 2>/dev/null || true
for _i in {1..20}; do
  pgrep -f 'python.*exo' >/dev/null 2>&1 || break
  sleep 1
done
if pgrep -f 'python.*exo' >/dev/null 2>&1; then
  echo 'WARNING: exo did not exit on SIGTERM after 20s — escalating to SIGKILL (may leak RDMA QPs; reboot if TB wedges).'
  pkill -9 -f 'python.*exo' 2>/dev/null || true
  sleep 1
fi
screen -wipe >/dev/null 2>&1 || true
# Reclaim-curve check: wait for wired+compressor to drain before relaunching
# so the first placement does not refuse on transiently-low ram_available.
# Still stuck at the deadline = post-SIGKILL orphaned AGX pages (m4-2
# 2026-07-09: 61 GB in the compressor, no owning process) — reboot to clear.
_deadline=$(( $(date +%s) + 180 ))
while :; do
  _residual=$(vm_stat | awk '/page size of/{ps=$8} /Pages wired down:/{w=$4} /Pages occupied by compressor:/{c=$5} END{printf "%d",(w+c)*ps/1e9}')
  [ -n "$_residual" ] && [ "$_residual" -le 25 ] && break
  if [ "$(date +%s)" -ge "$_deadline" ]; then
    echo "WARNING: STUCK MEMORY — wired+compressor still ${_residual} GB 180s after the kill. Placement will refuse / JIT will 503 until it clears. ESCAPE HATCH: reboot this node."
    break
  fi
  echo "waiting for memory reclaim (${_residual} GB wired+compressor)..."
  sleep 5
done
RELAUNCH_BODY
        printf "screen -dmS exorun zsh -l -c '%s'\n" "$LAUNCH_CMD"
    } > "$RELAUNCH_TMP"
    scp -q "$RELAUNCH_TMP" "$NODE:relaunch_exo.sh"
    ssh "$NODE" "chmod +x ~/relaunch_exo.sh"
    rm -f "$RELAUNCH_TMP"

    if [ "$NODE" == "macstudio-m4-1" ] || [ "$NODE" == "macstudio-m4-2" ]; then
         ssh "$NODE" "screen -dmS exorun zsh -l -c '$LAUNCH_CMD'"
    else
         ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=$NODE_PEERS .venv/bin/python -m exo -v >> ~/exo.log 2>&1'"
    fi
done

# 4. Health Check / Topology Verification
# Wait for all 3 nodes AND their identities (friendlyName) to be populated.
API="http://$M4_1_IP:52415"

echo -n "Waiting for cluster to stabilize..."
CLUSTER_READY=false
for i in {1..90}; do
    response=$(curl -s "$API/state")
    node_count=$(echo "$response" | jq '.topology.nodes | length' 2>/dev/null)
    identity_count=$(echo "$response" | jq '.nodeIdentities | length' 2>/dev/null)

    # Handle null or empty counts
    if [ -z "$node_count" ] || [ "$node_count" == "null" ]; then node_count=0; fi
    if [ -z "$identity_count" ] || [ "$identity_count" == "null" ]; then identity_count=0; fi

    if [ "$node_count" -ge ${#NODES[@]} ] && [ "$identity_count" -ge ${#NODES[@]} ]; then
        echo " HEALTHY! (Nodes: $node_count, Identities: $identity_count)"
        CLUSTER_READY=true
        break
    fi
    echo -n "."
    sleep 2
done

if [ "$CLUSTER_READY" = false ]; then
    echo ""
    # Check for the specific pyo3 initialization panic that happens when uv.lock goes out of sync
    PYO3_PANIC=$(ssh macstudio-m4-1 "grep -i 'The Python interpreter is not initialized' ~/exo.log" 2>/dev/null || true)

    if [ -n "$PYO3_PANIC" ]; then
        echo "CRITICAL ERROR: Detected a corrupted Rust pyo3 binding state on the primary node!"
        echo "This usually happens when 'uv.lock' changes (e.g. from switching git branches) and the virtual environment gets out of sync."
        echo ""
        echo "AUTOMATIC FIX: Run the following command on ALL nodes to repair the bindings:"
        echo "  zsh -l -c 'cd ~/repos/exo && uv sync --reinstall-package exo_rs'"
        echo ""
        echo "Exiting."
        exit 1
    fi

    echo "TIMEOUT: Cluster did not stabilize."
    echo "Fetching logs from macstudio-m4-1:"
    ssh macstudio-m4-1 "tail -n 20 ~/exo.log"
    exit 1
fi


# 5. Create model instances on the Mac Studios

# Look up Mac Studio node IDs from cluster state (these differ from libp2p peer IDs).
echo "Looking up node IDs from cluster state..."
M4_1_NODE_ID=""
M4_2_NODE_ID=""
MBP_NODE_ID=""
for i in {1..15}; do
    NODE_STATE=$(curl -s "$API/state")
    M4_1_NODE_ID=$(echo "$NODE_STATE" | jq -r '.nodeIdentities | to_entries[] | select(.value.friendlyName | test("Studio.*M4-1")) | .key')
    M4_2_NODE_ID=$(echo "$NODE_STATE" | jq -r '.nodeIdentities | to_entries[] | select(.value.friendlyName | test("Studio.*M4-2")) | .key')
    # MBP_NODE_ID=$(echo "$NODE_STATE" | jq -r '.nodeIdentities | to_entries[] | select(.value.friendlyName | test("MacBook")) | .key')

    if [ -n "$M4_1_NODE_ID" ] && [ -n "$M4_2_NODE_ID" ]; then
        break
    fi
    echo "  Waiting for node identities to propagate..."
    sleep 2
done

echo "  Mac Studio M4-1: $M4_1_NODE_ID"
echo "  Mac Studio M4-2: $M4_2_NODE_ID"

if [ -z "$M4_1_NODE_ID" ] || [ -z "$M4_2_NODE_ID" ]; then
    echo "ERROR: Could not resolve all node IDs. Skipping instance creation."
    echo "Create instances manually from the dashboard."
    exit 1
fi

create_instance_with_retry() {
    # Two-step instance creation (same as dashboard):
    #   1. GET /instance/placement — computes shard assignments (retries until state is ready)
    #   2. POST /instance — creates the instance with the computed placement
    local label="$1"
    local model_id="$2"
    local sharding="${3:-Pipeline}"
    local instance_meta="${4:-MlxJaccl}"
    local min_nodes="${5:-2}"
    local def_temp="${6:-}"
    local def_top_p="${7:-}"
    local def_top_k="${8:-}"
    local def_min_p="${9:-}"
    local def_presence="${10:-}"
    local def_repetition="${11:-}"
    local max_kv_tokens="${12:-}"
    local max_prefix_sessions="${13:-}"
    local kv_cache_bits="${14:-}"
    local max_prefix_bytes="${15:-}"
    local prefill_step="${16:-}"
    local max_attempts=30

    for attempt in $(seq 1 $max_attempts); do
        # Check if instance already exists
        local existing
        existing=$(curl -s "$API/state" | jq -r --arg m "$model_id" \
            '[.. | objects | select(has("shardAssignments")) | select(.shardAssignments.modelId == $m)] | length' 2>/dev/null)
        if [ -n "$existing" ] && [ "$existing" != "null" ] && [ "$existing" -ge 1 ] 2>/dev/null; then
            echo "  Instance for $label already exists, skipping."
            return 0
        fi

        # Step 1: GET /instance/placement to compute shard assignments
        local placement_response placement_code
        placement_response=$(curl -s -w '\n%{http_code}' -G "$API/instance/placement" \
            --data-urlencode "model_id=$model_id" \
            --data-urlencode "sharding=$sharding" \
            --data-urlencode "instance_meta=$instance_meta" \
            --data-urlencode "min_nodes=$min_nodes")
        placement_code=$(echo "$placement_response" | tail -1)
        placement_response=$(echo "$placement_response" | sed '$d')

        if [ -z "$placement_code" ] || [ "$placement_code" -lt 200 ] 2>/dev/null || [ "$placement_code" -ge 300 ] 2>/dev/null; then
            local err_msg
            err_msg=$(echo "$placement_response" | jq -r '.detail // .error.message // empty' 2>/dev/null)
            if [ "$attempt" -lt "$max_attempts" ]; then
                echo "  Attempt $attempt/$max_attempts: placement not ready ($err_msg), retrying in 5s..."
                sleep 5
                continue
            else
                echo "  ERROR: Placement failed after $max_attempts attempts: $err_msg"
                return 1
            fi
        fi

        # Step 2: POST /instance with the placement result.
        # Inject per-instance sampling defaults into the inner tagged object
        # (TaggedModel wraps it as {"MlxJacclInstance": {...}}).
        local create_payload
        create_payload=$(echo "$placement_response" | jq -c \
            --argjson temp "${def_temp:-null}" \
            --argjson top_p "${def_top_p:-null}" \
            --argjson top_k "${def_top_k:-null}" \
            --argjson min_p "${def_min_p:-null}" \
            --argjson presence "${def_presence:-null}" \
            --argjson repetition "${def_repetition:-null}" \
            --argjson kv "${max_kv_tokens:-null}" \
            --argjson sessions "${max_prefix_sessions:-null}" \
            --argjson kv_bits "${kv_cache_bits:-null}" \
            --argjson bytes "${max_prefix_bytes:-null}" \
            --argjson prefill_step "${prefill_step:-null}" \
            '
            . as $i
            | ($i | keys[0]) as $tag
            | {instance: ($i | .[$tag] |= (
                (if $temp     != null then .defaultTemperature      = $temp       else . end)
                | (if $top_p    != null then .defaultTopP             = $top_p      else . end)
                | (if $top_k    != null then .defaultTopK             = $top_k      else . end)
                | (if $min_p    != null then .defaultMinP             = $min_p      else . end)
                | (if $presence != null then .defaultPresencePenalty  = $presence   else . end)
                | (if $repetition != null then .defaultRepetitionPenalty = $repetition else . end)
                | (if $kv       != null then .maxKvTokens             = $kv         else . end)
                | (if $sessions != null then .maxPrefixSessions       = $sessions   else . end)
                | (if $kv_bits  != null then .kvCacheBits             = $kv_bits    else . end)
                | (if $bytes    != null then .maxPrefixBytes          = $bytes      else . end)
                | (if $prefill_step != null then .prefillStepSize     = $prefill_step else . end)
              ) | {($tag): .[$tag]})}
        ')

        local create_response create_code
        create_response=$(curl -s -w '\n%{http_code}' -X POST "$API/instance" \
            -H "Content-Type: application/json" \
            -d "$create_payload")
        create_code=$(echo "$create_response" | tail -1)
        create_response=$(echo "$create_response" | sed '$d')

        if [ -n "$create_code" ] && [ "$create_code" -ge 200 ] 2>/dev/null && [ "$create_code" -lt 300 ] 2>/dev/null; then
            local msg
            msg=$(echo "$create_response" | jq -r '.message // empty' 2>/dev/null)
            echo "  ${msg:-Instance created.}"
            return 0
        else
            local err_msg
            err_msg=$(echo "$create_response" | jq -r '.detail // .error.message // empty' 2>/dev/null)
            echo "  ERROR creating instance (HTTP $create_code): $err_msg"
            return 1
        fi
    done
}

EXPECTED_RUNNERS=0

# ── Auto-place DeepSeek V4 Flash with RDMA ──
# Single 2-node Tensor + MlxJaccl instance spanning both Studios — the cluster's
# primary model. ~100 GB/rank at 6-bit leaves ~24 GB headroom for KV cache +
# activations on each 128 GB node (under the 124 GB iogpu.wired_limit_mb).
# Co-hosts with the much smaller Qwen3.6 aux model (see its block below); DSv4's
# footprint is why Qwen3.6's prefix/KV cache is hard-capped.
if [ "${DSV4_ENABLED:-0}" = "1" ]; then
    echo ""
    echo "Auto-placing DeepSeek V4 Flash ($DSV4_MODEL_ID) across both Studios via RDMA..."

    EXISTING_DSV4=$(curl -s "$API/state" | jq -r --arg m "$DSV4_MODEL_ID" \
        '[.. | objects | select(has("shardAssignments")) | select(.shardAssignments.modelId == $m)] | length' 2>/dev/null)
    if [ -z "$EXISTING_DSV4" ] || [ "$EXISTING_DSV4" = "null" ]; then
        EXISTING_DSV4=0
    fi

    if [ "$EXISTING_DSV4" -ge 1 ]; then
        echo "  DeepSeek V4 instance already running. Skipping."
    else
        create_instance_with_retry "DeepSeek V4 Flash" "$DSV4_MODEL_ID" "${DSV4_SHARDING:-Tensor}" "MlxJaccl" 2 \
            "$DSV4_TEMPERATURE" "$DSV4_TOP_P" "$DSV4_TOP_K" "$DSV4_MIN_P" \
            "$DSV4_PRESENCE_PENALTY" "$DSV4_REPETITION_PENALTY" \
            "$DSV4_MAX_KV_TOKENS" "$DSV4_MAX_PREFIX_SESSIONS" \
            "$DSV4_KV_CACHE_BITS" "$DSV4_MAX_PREFIX_BYTES" \
            "$DSV4_PREFILL_STEP_SIZE" || true

        echo -n "Waiting for 2 DeepSeek V4 runner(s) to become Ready..."
        READY=false
        READY_COUNT=0
        for i in {1..180}; do
            READY_COUNT=$(curl -s "$API/state" | jq -r --arg m "$DSV4_MODEL_ID" '
                . as $root
                | [ $root.instances | to_entries[]
                    | select(.value.MlxJacclInstance.shardAssignments.modelId == $m)
                    | .value.MlxJacclInstance.shardAssignments.runnerToShard | keys[] ] as $rids
                | [ $rids[] | $root.runners[.] | select(.RunnerReady? != null) ] | length
            ' 2>/dev/null)
            if [ -z "$READY_COUNT" ] || [ "$READY_COUNT" = "null" ]; then READY_COUNT=0; fi
            if [ "$READY_COUNT" -ge 2 ]; then
                echo " READY ($READY_COUNT/2)"
                READY=true
                break
            fi
            echo -n "."
            sleep 2
        done
        if [ "$READY" = false ]; then
            echo ""
            echo "  WARNING: DeepSeek V4 only $READY_COUNT/2 runners reached Ready."
            echo "  Check ~/exo.log on the Studios."
        fi
    fi
fi

# ── Auto-place Qwen3.6-35B-A3B with RDMA (co-hosted alongside DSv4) ──
# DEFAULT ON (2026-06-26). Co-hosts Qwen3.6 (~17.5GB/node) alongside
# DSv4-Flash (~78GB/node) — leaves ~21GB/node headroom (tight but workable
# for the aux/image-model workload Qwen3.6 serves). The fused-GDN guard
# (patches/__init__.py:86dad09f) now auto-skips the Qwen3.5 fused patches
# on Qwen3.6 (tagged qwen3_5_moe but architecturally different GDN) instead
# of crashing at warmup, so co-hosting loads clean on the vanilla MLX path.
# Set QWEN36_ENABLED=0 to keep DSv4 solo (full box for context/working set).
# Qwen3.6 ships MTP weights, so self-speculation is auto-enabled per-instance
# (independent of EXO_SPECULATIVE, which is the DSv4 knob).
if [ "${QWEN36_ENABLED:-1}" = "1" ]; then
    echo ""
    echo "Auto-placing Qwen3.6 ($QWEN36_MODEL_ID) across both Studios via RDMA..."

    EXISTING_QWEN36=$(curl -s "$API/state" | jq -r --arg m "$QWEN36_MODEL_ID" \
        '[.. | objects | select(has("shardAssignments")) | select(.shardAssignments.modelId == $m)] | length' 2>/dev/null)
    if [ -z "$EXISTING_QWEN36" ] || [ "$EXISTING_QWEN36" = "null" ]; then
        EXISTING_QWEN36=0
    fi

    if [ "$EXISTING_QWEN36" -ge 1 ]; then
        echo "  Qwen3.6 instance already running. Skipping."
    else
        create_instance_with_retry "Qwen3.6 35B-A3B" "$QWEN36_MODEL_ID" "Tensor" "MlxJaccl" 2 \
            "$QWEN36_TEMPERATURE" "$QWEN36_TOP_P" "$QWEN36_TOP_K" "$QWEN36_MIN_P" \
            "$QWEN36_PRESENCE_PENALTY" "$QWEN36_REPETITION_PENALTY" \
            "$QWEN36_MAX_KV_TOKENS" "$QWEN36_MAX_PREFIX_SESSIONS" \
            "$QWEN36_KV_CACHE_BITS" "$QWEN36_MAX_PREFIX_BYTES" \
            "$QWEN36_PREFILL_STEP_SIZE" || true

        echo -n "Waiting for 2 Qwen3.6 runner(s) to become Ready..."
        READY=false
        READY_COUNT=0
        for i in {1..180}; do
            READY_COUNT=$(curl -s "$API/state" | jq -r --arg m "$QWEN36_MODEL_ID" '
                . as $root
                | [ $root.instances | to_entries[]
                    | select(.value.MlxJacclInstance.shardAssignments.modelId == $m)
                    | .value.MlxJacclInstance.shardAssignments.runnerToShard | keys[] ] as $rids
                | [ $rids[] | $root.runners[.] | select(.RunnerReady? != null) ] | length
            ' 2>/dev/null)
            if [ -z "$READY_COUNT" ] || [ "$READY_COUNT" = "null" ]; then READY_COUNT=0; fi
            if [ "$READY_COUNT" -ge 2 ]; then
                echo " READY ($READY_COUNT/2)"
                READY=true
                break
            fi
            echo -n "."
            sleep 2
        done
        if [ "$READY" = false ]; then
            echo ""
            echo "  WARNING: Qwen3.6 only $READY_COUNT/2 runners reached Ready."
            echo "  Check ~/exo.log on the Studios."
        fi
    fi
fi

# Final environment export
export IBV_FORK_SAFE=${IBV_FORK_SAFE:-1}

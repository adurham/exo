# DSv4-Flash High-Context Prefill Cliff — Handoff (2026-06-21)

## THE PROBLEM
A single growing conversation (or a cold prefill) on DSv4-Flash sees prefill
throughput collapse as context grows:
  - ~270 t/s at 100K ctx, gentle decline to ~168 t/s at 340K (expected scaling)
  - SHARP CLIFF to ~40-48 t/s at ~340-380K (the unexplained part)
Manifests as periodic multi-second STALLS (8-32s) on individual 128-token
prefill chunks, alternating with fast chunks. The stalls — not steady slowdown —
are what drag the average down.

## CLUSTER / DEPLOY FACTS (verified this session)
- 2x Mac Studio M4 Max, 128GB each, RDMA over TB5. `start_cluster.sh` deploys.
- Parallelism: TENSOR (MoE-sharded, ATTENTION REPLICATED on both nodes — DSv4's
  LoRA-decomposed attention can't be cleanly head-sharded). NOT pipeline.
  Confirmed via /state TensorShardMetadata + DeepseekV4ShardingStrategy.
- seq-split (EXO_DSV4_SEQ_SPLIT) is ON by default (splits prefill query rows
  across nodes). Already active during all measurements — cliff persists with it.
- Model: 43 layers, compress_ratios alternating [4,128], index_topk=512,
  index_n_heads=64, sliding_window=128, max_position_embeddings=1048576.
  rope_scaling yarn factor=16 orig=65536 → 1M effective. NO input cap (the
  earlier "273163" was just a real request size, not truncation).
- Prefill path: TP setup uses mlx_lm `stream_generate` (the chunked-prefill
  per-chunk instrumentation in exo generate.py prefill() is the is_pipeline
  branch, which does NOT run here). prefill_step_size=128.
- DEPLOY MECHANISM (critical): runner imports mlx_lm from the VENV, which
  start_cluster force-reinstalls from the ./mlx-lm SUBMODULE (gitlink). Editing
  the submodule requires: commit+push submodule → bump gitlink in exo →
  commit+push exo → start_cluster resets+submodule update+reinstall. Local venv
  has a STALE pip copy until `uv pip install --no-deps --force-reinstall ./mlx-lm`.
- start_cluster only forwards EXPLICITLY-LISTED env vars into the runner EXO_ENV.
  A new EXO_* flag needs a passthrough line or it silently never reaches the runner.

## ROOT-CAUSE EVIDENCE (span profiler, EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1)
Two COMPLETING prefills, A=300K vs B=360K, baseline (no fix):
  - attn.indexer is the #1 context-SCALING cost: per-call avg 4532us → 5362us
    (+18%) A→B, while every non-scaling op is flat (moe.switch_mlp +0%,
    proj_qkv 0%, o_proj 0%, moe.gate 0%).
  - attn.indexer variance: avg ~5ms but MAX ~22ms (4x). Spikes across ~21
    indexer layers compound into the multi-second chunk stalls.
  - attn.sdpa also scales (+17%, 1432→1672us) but smaller absolute.
  - Mechanism of indexer cost: _indexer_score builds full (B,64,L,P) scores
    tensor (q @ pooled.T), transposes, collapses over heads; then caller
    argsorts over full P for top-k. P = pooled length, grows with context.
    Both the matmul (O(P)) and argsort (O(P log P)) scale with P.

## WHAT WAS TRIED AND ***FAILED*** (do not repeat)
TILED-P INDEXER (EXO_DSV4_INDEXER_PBLOCK, default 0=OFF):
  - Hypothesis: the cliff is memory-ALLOCATION pressure from the big (B,64,L,P)
    transient. Tile P into blocks, mx.eval per tile to free transients.
  - Local validation PASSED: bit-exact (max diff 0.0, top-k overlap 1.0 across
    P∈{25k,90k,250k}); peak memory per indexer call 4.36GB→0.40GB (91%).
  - CLUSTER A/B RESULT: throughput ~2% WORSE. B(360K) prefill 2140s→2177s.
    attn.indexer avg 5362→5900us (worse, per-tile eval-fence + launch overhead),
    max 22022→17350us (only the spikes eased slightly), total 316972→348795 (worse).
  - CONCLUSION: allocation was NOT the throughput bottleneck. Reducing peak
    memory did not speed prefill. The fix is a MEMORY tool (real, validated —
    useful for fitting higher context / 1M headroom) but NOT a throughput fix.
  - STATUS: committed, deployed, default OFF. Harms nothing. mlx-lm 59a5b9a,
    exo gitlink-bumped (24059598) + PBLOCK passthrough (57d02b66).
  - NOTE: A_300K_tiled span dump was CORRUPT (n=130, dump hit a reset boundary);
    only B_360K_tiled is valid. prefill_s is the reliable A/B metric, not that dump.

## STILL UNEXPLAINED (the actual open problem)
The SHARP cliff at ~340K is NOT yet mechanistically explained:
  - "compute scales with P" → would be GRADUAL, not a sharp cliff. ✗
  - "allocation pressure" → tiled-P fixed allocation, cliff REMAINED. ✗
  - So a DISCRETE event happens at ~340K context that step-changes per-chunk
    time. Not yet identified. This is the thing to find.
  - Reproduces in BOTH cold prefill (single 380K prompt) and growing session,
    at the same ~340K — so it's absolute-context-keyed, not session-path-keyed.
  - Ruled out: inter-node comms (distributed_callback spans ~100us); NOT swap
    (vm.swapusage ~0); NOT a runner crash (clean when not mis-signaled).

## NEXT ATTACK VECTORS (untried, in priority order)
1. FIND THE CLIFF MECHANISM FIRST (before any more fixes). Instrument per-chunk
   timing of the model-side spans (attn.indexer/attn.sdpa/attn.compressor/moe.*)
   ACROSS the 340K boundary in ONE prefill — does a specific span step-change at
   340K, or is it an allocator/kernel-cache/Metal-residency threshold? Candidate
   discrete triggers: MLX Metal compile-cache or buffer-pool threshold; the
   compressor/pool structure crossing a size; KV/pooled cache reallocation step;
   a power/thermal throttle at sustained load (check exo_gpu power/temp + macOS
   thermal level across the cliff — NOT yet checked).
2. EXO_DSV4_PREFILL_ARGPARTITION=1 (already in code, OFF): swaps prefill argsort
   O(P log P) → argpartition O(P), top-k SET identical (downstream gathered-KV
   attention is order-invariant). Attacks the argsort half of indexer compute.
   UNTESTED. Low-risk, quality-equivalent by construction.
3. If cliff is a Metal compile-cache/residency threshold: investigate
   mx.clear_cache cadence or buffer-pool limits at high P.

## VALIDATION DISCIPLINE (user rule — non-negotiable)
- Throughput is NOT a success metric without QUALITY validation. A "symmetric
  clean per-stream" win was once BOS-spam. ALWAYS run bench/quality_probe_dsv4.py
  + short-prompt curl ("capital of France"→"Paris", no <|begin_of_sentence|>
  spam) and show generated text before quoting t/s.
- Bit-exactness for any indexer/attention change: bench/indexer_tiled_p_bitexact.py
  pattern (max abs diff ≤1 bf16 ulp AND top-k SET overlap = 1.0).
- A/B with the span profiler: completing 300K + 360K probes, compare prefill_s
  (ground truth) and attn.indexer avg/max. Probes take ~28min/~36min each.

## ARTIFACTS / COMMANDS
- bench/indexer_tiled_p_bitexact.py — local bit-exact gate (no cluster).
- /tmp/probe.py (on node) — `python probe.py <TARGET_TOKENS>` builds an exact-
  token prompt (tokenizer-sliced) and sends ONE completing request.
- Span dump: EXO_PROFILER=spans; dumps per-prefill via the in-process
  profiler.get().dump() wired into prefill() start/end (generate.py). Signal-free
  (do NOT SIGUSR1 a live MLX runner — it crashed the runner once).
- Launch for diag: EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1 EXO_TRACING_ENABLED=true
  [EXO_DSV4_INDEXER_PBLOCK=16384] ./start_cluster.sh
- HARD RULE: never edit files directly on the studios; all mlx/mlx-lm changes go
  through git (local edit → submodule commit+push → gitlink bump → node fetch/reset
  → start_cluster reinstall). Local-only perf WIP preserved on mlx-lm branch
  `local-perf-wip-6eb7a6e` (seq-split v2, OPT-4 SDPA tiling — undeployed, unvalidated).

## DEPLOYED STATE RIGHT NOW
- exo origin/main = 57d02b66 (gitlink→mlx-lm 59a5b9a, tiled-P available default-OFF,
  PBLOCK passthrough, signal-free profiler dump, idle-reclaim, KV-evict fix,
  Qwen-off-by-default, DSML repair).
- Cluster currently running with EXO_DSV4_INDEXER_PBLOCK=16384 (tiled-P ON) —
  should be set back to default (unset/0) since it's a ~2% regression for
  throughput; only enable when memory headroom at very high context is the goal.
- Qwen unloaded; LXC gateway (hermes-gw-01) STOPPED+DISABLED (from earlier this
  session — needs restore if wanted). gabi session 96f50e was killed (resumable
  via `hermes --resume 20260620_201045_96f50e`).

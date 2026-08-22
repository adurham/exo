# exo Performance History — Fable's Independent Consolidation

**Compiled 2026-08-21 via Claude Fable 5, from the same raw extraction
data used for `docs/PERFORMANCE_HISTORY.md`.** This is a genuinely
independent synthesis — Fable was given only the raw per-file extracted
findings (file, date, outcome, numbers, why, lesson) and asked to build
its own structure and analysis from scratch, not to review or rubber-stamp
the other document. Treat these two docs as cross-checks on each other;
where they agree, trust it more; where they diverge or one caught
something the other missed, that's worth a closer look.

**Purpose:** Single reference for what has been tried, what the real
measured numbers were, why each attempt succeeded or failed, and what NOT
to re-attempt. Negative results are documented with the same rigor as
wins. Sources: 130 extracted findings from 197 repo markdown files.

**Hardware/model context:** Primarily DSv4-Flash on 2-node Apple Silicon
(M4 Max 128GB) clusters over Thunderbolt/jaccl RDMA; secondary:
Qwen3.5-397B-A17B, MiniMax-M2.7, Qwen3.6, M4 Ultra.

---

## 1. MoE `all_sum` Collective Cost (TP)

This was the dominant investigation thread of Aug 2026. Net verdict:
**the collective is genuinely expensive (~62% of TP prefill wall via
NOP-ablation), the fix is transport/chunking config and overlap — NOT
quantizing or ablating the collective, and NOT new GEMM kernels.**

- **WIN — NOP-ablation cost confirmation** (2026-08-19): baseline 162.5
  vs all_sum-off 427.5 tok/s = 2.63x, 62.0% share. NOP-ablation is
  diagnostic-only (output is garbage). *Lesson: NOP-ablation isolates a
  collective's true cost without relaunch, but is never shippable.*
- **WIN — 178ms/call was partly a measurement artifact** (2026-08-20):
  real collective cost ~12ms/call; the 178ms figure included queuing.
  *Lesson: verify whether a per-call cost is the op itself or includes
  queuing before designing around it.*
- **WIN — Bandwidth attribution** (2026-08-19): 16.8MB payload, 178ms
  flat with depth → effective ~92-96 MB/s vs realistic 6-10 GB/s; two
  independent arithmetic derivations converged, pointing at jaccl
  chunking config. *Lesson: two independent derivations converging
  promotes a hypothesis from story to mechanism.*
- **WIN — Hardware floor isolation** (2026-08-21): raw jaccl allreduce at
  8KB = 120.2µs vs in-model sync-span 4093.7µs (34.1x). Overhead is
  software, not wire. *Lesson: build the vendor's own isolated microbench
  at matching message size before blaming hardware.*
- **WIN — Stream-boundary decomposition** (2026-08-20): a non-collective
  CPU-op on the same stream reproduces 2.66x cost with **zero wire
  bytes**. Also: `MLX_METAL_FAST_SYNCH=1` is 1.5x SLOWER with 70x more
  variance — do not enable. *Lesson: test whether a same-stream
  non-collective op reproduces the cost before concluding the collective
  is expensive.*
- **WIN — Overlap is possible** (phase0b, 2026-08-20): all_sum runs on a
  CPU stream; overlap_ratio=0.995, ~33% of the 115ms serial budget
  recoverable. Overlap is destroyed only by same-GPU-stream FIFO
  ordering, not a device-wide barrier.
- **MIXED — Sequence-chunk pipelining** (lever2, 2026-08-20): real but
  modest 1.1-1.15x. Layer-pipelining is **invalid** (RMSNorm inter-layer
  dependency); sequence-chunk is the only valid overlap axis. Realistic
  ceiling from phase3 passive analysis: 5-7%, best case 10.5%.
- **NEG — Quantized all_sum: mathematically infeasible** (2026-08-19):
  moe.all_sum is a true partial-sum REDUCTION, not gather-semantics — you
  cannot quantize per-rank and sum. Live test also hung on first real
  request. **CLOSED. Do not reopen.**
- **NEG — Shared-scale int8 all_sum** (2026-08-19/20): correctness
  passed, but real-prefill measurement showed ~265ms/call vs 178ms
  baseline = **1.49x SLOWER**. The earlier "local_absmax costs
  400-420ms" motivation was a probe fence artifact (warm p50 0.426ms vs
  cold 56.8ms — `mx.eval()` flushes the ENTIRE pending lazy graph).
  **CLOSED.**
- **NEG — All-sum ablation as a shipping lever** (2026-08-21): unsafe,
  breaks correctness.
- **NEG — Comm/compute overlap "implementation"** (2026-08-21): overlap
  already exists in the current code. *Lesson: verify existing
  implementation before re-implementing an optimization.*
- **NEG — Expert co-location to reduce all_sum traffic** (section108 +
  2026-05-14 plan): TP all_sum volume is **independent of expert
  placement**. Bonus finding from the May plan: both ranks already held
  all 256 experts (per-rank read 1.67GB, not the assumed 3.35GB) — the
  sharding-axis assumption was wrong. *Lesson: read the actual sharding
  code before designing a bandwidth lever.*
- **NEG — MoE small-M GEMM kernel headroom** (2026-08-20): live kernel
  already **1.6x faster than the idealized ceiling** at L=512; MAXBE=64
  widening regressed -33% to -78%. *Lesson: benchmark existing kernel vs
  theoretical ceiling first — if at/above it, no kernel lever exists.*
- **NEG — MoE tile-geometry retune** (2026-08-18): bm=16 (current) 20.4%
  padding waste; bm=64 is 52.1% — monotonically worse on the skewed
  ragged-routing distribution. **CLOSED.**
- **INFO — Reconciliation**: 96-97% GPU utilization coexists with
  all_sum at 61-64% of wall because utilization reads "busy" while the
  submission thread blocks in an uninterruptible wait. OPT-7
  deferred-eval fence made B=2 prefill 23% SLOWER (111 vs 144).
- **INFO — Decode-side scale**: 43 real all_sum collectives per decode
  token (section101); jaccl overhead ~1.45ms of 6.6ms/token budget
  (~22%) at c=2.

---

## 2. Prefill Throughput & Context-Scaling Cliffs

- **WIN — The June breakthrough stack** (2026-06-24): five fixes → c=1
  500K prefill 167→251 t/s. Key item: `MLX_MAX_MB_PER_BUFFER` 50→200MB
  eliminated bimodal stalls (B=2: 144→317 t/s, +120%). Default 50MB
  triggers **non-deterministic mid-forward command-buffer flushes** on
  Apple Silicon. Indexer weight-fold: 64x compute reduction,
  bit-equivalent.
- **WIN — argpartition indexer fix** (MOE_KERNEL_HANDOFF, 2026-07-01):
  100K prefill 255→353 t/s (+38%); the 340K cliff (270→40 t/s)
  **eliminated** by argsort→argpartition. Chunk512/tile-sweep/fused-topk
  retunes were tried and conclusively falsified in the same campaign.
- **WIN — SEQ_SPLIT=1** (re-validated at 190-220K, 2026-08-18): 358.6 vs
  277.8 tok/s = +29%. *Lesson: re-validate at production-representative
  scale.*
- **NEG — STEP_SIZE=4096** (tested twice, 2026-08-18/19): 331.2 vs
  358.6 baseline = ~8% SLOWER end-to-end **despite** the isolated
  MoE-GEMM microbench being 15% more efficient; attn.sdpa is 1.78-2.03x
  worse at 4096. Note: on Qwen3.5/PP (request_lifecycle_trace),
  4096→8192 cut per-token MoE cost 31% — step-size tuning is
  model/topology-specific, don't transplant.
- **NEG — Tiled-P indexer memory fix** (PREFILL_CLIFF_HANDOFF,
  2026-06-21): -91% memory but ~2% WORSE throughput. *Lesson: a memory
  optimization doesn't fix throughput even when memory pressure seems
  plausible; gradual scaling can't explain a SHARP cliff.*
- **MIXED — CHUNK_OVERLAP lever** (2026-08-20): cross-stream ordering
  race fixed, but 1-of-2 live trials showed a correctness anomaly; lever
  declared **DEAD/CLOSED**.
- **MIXED — Two-level chunking OPT-4** (2026-06-12): initial claimed
  +35% (236→317.8) was a **buggy measurement**; corrected chunk256 was a
  ~140 tok/s REGRESSION, reverted.
- **INFO — Current profile is compute-bound and balanced**: 220K
  sync-span: attn 58.4%/ffn 41.6%; direct GPU telemetry 96.6-97.0%.
  attn.o_proj at 83.2% of roofline ceiling — don't chase it; the
  profitable anomaly is attn.proj_qkv span cost at 2.35x its isolated
  kernel cost.
- **INFO — Known-good baseline** (2026-08-21): 366.6/351.5/331.6 prefill
  and 17.48/18.60/17.26 decode tok/s at 100K/300K/500K.
- **INFO — Fixed-cost insight** (PREFILL_THROUGHPUT_PLAN): per-chunk
  cost near-flat 2.3-2.8s regardless of depth, and 147/200 logged
  prefills were <2K tokens — the common-case SMALL request bucket is the
  real-world lever.
- **INFO — Qwen3.5 differs**: real bottleneck is DeltaNet/SSM layers
  (22/30), not attention. Identify dominant layer TYPE per model before
  reusing DSv4 conclusions.

---

## 3. Decode Throughput & Dispatch Overhead

- **Verdict (roofline, 2026-08-21): decode is dispatch-bound, not
  bandwidth-bound.** Fixed per-layer dispatch-count reductions show up
  proportionally in decode, barely in prefill. Corroborated on M4
  Ultra/MiniMax (dispatch-scheduling-bound; MLX_SDPA_BLOCKS 512→88
  threadgroups: +6.5%).
- **WIN — clear_cache interval** (request_lifecycle_trace): 256→2048
  eliminated a 17ms stall per 256 tokens.
- **WIN — sparse SDPA matmul rewrite**: 415s→98s wall at 32K (4.2x),
  quality-neutral.
- **NEG — Indexer PBLOCK tiling** (2026-08-21): prefill-neutral but
  decode DEGRADED at small p_block (13.67→10.03 tok/s at 16384);
  p_block=262144 restored 17.39. The design's own "zero overhead"
  comment was wrong.
- **NEG — 550ms PP decode stall** (section110): six hypotheses refuted,
  confirmed NOT waiting on any distributed collective. Still unexplained
  — anyone resuming this: read section110 first.
- **NEG — gather_qmm M=1 "dispatch bug"**: no bug; M=1 correctly routes
  to gemv (vector_limit 10-32 by chip generation). *Lesson: read the
  eval_gpu dispatch ladder before hypothesizing a dispatch bug.*
- **INFO — Hardware wall exists**: on M4 Max quantized MoE, 32% GPU
  utilization ≈ 97% of the achievable forward-compute ceiling. That is a
  hardware wall, not a software bug — do not burn sessions trying to
  raise it.

---

## 4. Speculative Decoding (MTP / Eagle / Token-Tree / DSpark)

The single largest sink of effort in the repo. Structural facts first:

- **Verify-phase dominates the cycle**: draft ~4.5-4.9ms (7%) vs verify
  53-56ms (~91%). Verify cost = ~30ms floor (KV attention at depth) +
  ~5.3ms per L_q token. **Any speculative variant that widens verify
  loses at long context because the floor dominates.**
- **WIN — MTP self-spec with gamma=2**: 30.77 t/s vs ~27 MTP-off at
  100K. gamma=1 is -6%, gamma=3 is **-18%** — a sweet spot set by
  acceptance falloff, not monotonic.
- **WIN — losslessness fixes at zero cost** (2026-07-10): tie-break fix
  and gemv/gemm M-dispatch rounding drift fixed at source; c=1 27.4 t/s
  matches the old lossy baseline; row-seq 24.6 vs 19.1 sequential at
  122K.
- **WIN — Eagle K=8 no-renorm**: +0.83% decode, Welch t=6.19, p<0.001.
  Sub-1% wins are shippable when statistically proven.
- **WIN — rollback cost** (2026-07-12): assumed cost center (~32ms) was
  wrong; real cost was a ~72ms commit-forward fallback; final 0.79ms
  mean. Premise revision via sub-phase profiling.
- **NEG — Token-tree drafting: full arc is a dead end.** Phase 6:
  quality broken on cluster (needle FAIL) despite passing unit benches.
  Phase 6b: 3 bugs fixed, correctness restored, **no lift** (29.95 vs
  30.06). Phase 7: all 6 DFS/greedy/K configs in [29.7, 29.95] vs 30.06
  linear. Phase 8: tree's +44% verify-wall growth ate the +13% draft
  win. **Do not re-attempt tree drafting on DSv4-Flash.**
- **NEG — Second MTP head**: DSv4-Flash ships only ONE trained MTP head;
  can't add without training. (Its proposed alternative — token trees —
  also failed, above.)
- **NEG — DSpark FULLBLOCK context collapse** (2026-08-04): 27.56 tok/s
  at depth ~500 → **1.73 tok/s at depth ~14K** (15.9x). Cluster default
  is DSpark OFF. *Lesson: benchmark spec-decode across a depth range
  before shipping.*
- **NEG — MTP at c≥2 was structurally broken** (phase9): agg 3.5-5.8
  t/s; single-stream spec caches silently corrupt across concurrent
  streams. Phase 10 cache-lifecycle fix restored correctness but NOT
  throughput (still 5.7 agg). Full recovery required drain-elimination
  (phase 13: **34.5 agg t/s**, σ=0.04, ~50ms/token drain removed) plus
  rendezvous fix (EXO_BATCHED_PREFILL_RENDEZVOUS_MS ≥ 2000ms killed the
  σ~10 variance).
- **NEG — Eagle K=1 c=2 17x regression**: `broadcast_from_canonical(soft_emb)`
  — a 16KB collective placed immediately before dependent compute with
  zero work to hide behind = fully exposed serialized latency. Related:
  MLX per-rank logit drift at cycle 5+ flips argmax on near-tied logits;
  any rank-local tensor must be broadcast like tok_arr.
- **NEG — per-row SDPA verify variants** (2026-07-12): hypothesis DEAD.
- **INCONCLUSIVE — DSpark Native Head**: implemented, never live A/B'd.
  Standalone validation ≠ win.

---

## 5. Kernel Fusion Attempts

Scorecard: **2 wins, 5 failures, 1 catastrophic.** Fusion pays only when
dispatch count is the binding constraint AND the matmuls are large.

- **WIN — MoE gate+up fusion** (2026-08-21): decode +3.01% (18.879 vs
  18.328), prefill neutral. **Note the contradiction:** the same fusion
  measured **-3.8%** on 2026-06-26. Codebase changed in between; the
  2026-08-21 A/B is authoritative, but this shows fusion results don't
  survive codebase drift — re-measure, don't trust old numbers in either
  direction.
- **WIN — upstream fused gate/up (exo#1999)**: +1.2% c=1 / +1.1% c=2.
- **NEG — wq_a+wkv projection fusion** (2026-08-21): correct, -0.48%
  (noise). *A validated fusion pattern does not generalize to smaller
  matmuls.*
- **NEG — fused-topk**: +0.038 t/s (+0.13%), failed 2σ. Falsified twice
  (May plan + MOE_KERNEL_HANDOFF).
- **NEG — fused indexer Metal kernel** (2026-05-14): fused 215µs/call vs
  pipelined chain 117µs = **0.54x SLOWER**. Per-call dispatch-overhead
  analysis overestimates savings; only a PIPELINED microbench reveals
  truth.
- **NEG — fused softmax** (2026-08-21): real correctness break — after
  the flag sat default-off with "needs A/B validation" for **~5 weeks**.
  *Untested default-off flags hide real bugs.*
- **CATASTROPHIC — fusing inside mx.compile** (2026-05-18): post_attn+ffn_pre
  fusion: 30.06 → **7.2-10.5 t/s**. **Never nest `mx.fast.metal_kernel`
  inside an `mx.compile` boundary.** Also: mx.compile wrapping an
  already-compiled function can hurt; fused SDPA L≤8 fold also
  regressed (30.06→28.9).

---

## 6. Quantization

- **NEG — TurboQuant KV compression: all configs regressed.** Baseline
  40.9; 4-bit -10%, 3-bit+qmm -6%, 3-bit+rotate -11%. Theoretical
  bandwidth wins don't survive added compute overhead. **CLOSED.**
- **NEG — bf16→fp16 compute dtype**: ~7% faster quantized_matmul on
  qwen3_5_moe, but **REVERTED for DSv4 — 7x decode slowdown, JACCL lacks
  fp16 support.** Check transport dtype support before dtype switches.
- **MIXED — quantized SDPA** (MiniMax): +20-35% pure decode (modeled
  +47%); wins are bit-width- and head-dim-specific (~0% at 5-bit, 1.3-1.5x
  at 4/8-bit); one bad config hit a 3x regression (5 tok/s) on first
  cluster run.
- **INFO — KV bits policy**: 4-bit KV is +4% faster than bf16 at
  c=2/100K, but **bf16 kept as default** — 4-bit noise compounds at long
  context. Speed loses to quality here by policy.
- **NEG — TOPK=160**: 31.9-32.2 t/s but needle FAIL (BOS-only output).
  **Invalidated prior champion claims that ran at TOPK=160.** A
  throughput gain that trades away quality is not a win.

---

## 7. jaccl / RDMA / Transport

- **WIN — dual-cable split** (2026-08-21): RDMA data and TCP
  coordination on distinct physical cables; 9 transport bugs fixed
  alongside.
- **NEG — jaccl is NOT the decode bottleneck** (phase F, 2026-05-19):
  poll wall 8.15µs = 1.9% of verify wall; the mx.eval fence is 98% of
  the wait. This premise error killed an entire plan — instrument
  per-call before blaming transport.
- **NEG — MLX_JACCL_RELIABLE_MAX_SZ=3**: statistically identical to sz=2
  (166-169 vs 162-172). Clean null; don't retest without new evidence.
- **NEG — subgroup all_gather lever** (2026-08-21): faulted 0.4s into
  prefill, needle FAIL. A partial fix to a related transport issue
  doesn't cover all fault paths.
- **MIXED — timeout removal unmasking** (2026-08-09): removing a fatal
  retransmit timeout (p2p deadline 60s→300s) unmasked a scheduler
  deadlock the old timeout had been accidentally papering over.
- **Operational facts:** Thunderbolt RDMA degrades under repeated rapid
  teardown/reconnect — full reboot clears it. Chunked-prefill hard-stalled
  after chunk 0 with "recv() deadline in drain" and 8+ min zero
  activity. On clusters where RDMA interfaces are production-claimed,
  use passive instrumentation, not active probes.

---

## 8. Concurrency, PP vs TP, Batching

- **Structural tradeoff:** PP wins single-request decode (27-33 vs
  ~15-20 tok/s TP); TP wins concurrency. Sharding mode is **baked into
  weight tensors at load time** — no live re-dispatch seam (cold
  shard+load ~18.7s/rank).
- **NEG — hybrid/phase-swap designs are dead ends, twice.** 2026-08-04:
  honest prefill numbers (225/214/202 tok/s at 100K/300K/500K, after
  correcting inflated 1.42x figures from a wrong chars-per-token
  assumption) → final decision TP for both phases. 2026-08-16:
  sequential weight-swap design — 11.1% gain cold-only, 0% cached, 37.4s
  swap, 125.4/128GB memory; neither topology alone hits the 400 tok/s
  requirement, so swapping converts one miss into a smaller miss.
  **Rejected.**
- **WIN — N=2 admission deadlock fixed** (2026-08-05): 7 bugs, verified
  over 4 rounds/8 concurrent requests. EXO_PP_BATCHED_DECODE=1 works but
  stays OFF by default. Concurrent PP bugs usually trace to cross-rank
  state divergence.
- **MIXED — c=2 stability boundary**: c=2-from-start is stable (~20
  tok/s decode); **admitting into an already-running batch mid-decode is
  not.**
- **NEG — c=2 at 100K catastrophic slowdown** (phase J): 4.5 t/s agg
  warmup, cluster wedged. Docstring scaling claims were relative to a
  weak baseline.
- **WIN — B=2 quality root causes**: seq-split all_gather was
  batch-unsafe (not the suspected cache-merge); verified across
  100K-500K needle sweep, safety gate then removed. *Batch-unsafe
  collectives masquerade as cache/quality bugs.*
- **NEG — c=2 100K quality bug** (2026-05-23) invalidated the prior
  gamma=2 34.16 t/s champion. Champions must be re-verified for
  correctness at higher context/concurrency.
- **Cross-rank determinism hazard** (phase0c): MLX collectives are
  matched **positionally by EVAL order**, not program order —
  rank-dependent branching produces 100%-deterministic silent
  corruption.

---

## 9. Memory Leaks & Allocator Behavior

- **WIN — multi-turn leak** (2026-06-29): +29.5GB over 139 messages →
  flat 79.04GB over 11 turns. Cause: KVPrefixCache snapshot accumulators
  (+21 PoolingCache objects/turn); fix: explicit retention cap (4).
  Method: walk gc referrer chains on containers whose len() grows per
  turn.
- **WIN — four prefix-cache/prefill leak sites** (2026-06-27): verified
  by memory **plateau over repeated requests**, not a single run.
- **WIN — mlx#3596 allocator coalescing**: RSS growth 770→155 KB/token.
- **NEG — IOGPU residency abort**: MTLResidencySet::addAllocation
  silently kills the process on long sustained workloads, deterministic
  on M4 Max 128GB. Affects long benchmarks.
- Related allocator win: MLX_MAX_MB_PER_BUFFER (§2) — same theme:
  **Apple Silicon allocator defaults are tuned for small workloads and
  cause stalls/aborts at this scale.**

---

## 10. Measurement Methodology (read this before benchmarking anything)

This section exists because **measurement error caused more wasted
effort than any single bad optimization.**

- **Sync your spans.** Non-sync profiling showed a 3.15x regression
  that was really 2.03x under `EXO_PROFILER_SYNC_SPANS=1`. Sync-span
  overhead itself: ~15% prefill, ~77% decode — never compare
  instrumented vs uninstrumented numbers.
- **mx.eval() fences flush the ENTIRE pending lazy graph** — the "400ms
  local_absmax" was a 130x fence artifact (warm 0.426ms).
- **Per-op profiler attribution overstates critical-path cost** when the
  framework overlaps ops: probe attributed 46.5ms (92.7%) to FFN/MoE;
  NOP-removal saved only ~7.5ms.
- **Champion claims need ≥10 iterations at σ<0.3.** The 32.3 t/s acksync
  champion cratered to 4.3 t/s on redeploy, cause never found.
  Separately, a champion was invalidated by quality (TOPK=160) and
  another by a dual-mlx-pinning deploy failure (git submodule vs
  pyproject.toml — a fix to the wrong one silently doesn't deploy).
- **Cross-check log-parsed throughput against wall clock**: OPT-4's
  claimed +35% was a parsing bug hiding a regression.
- **A clean run is not evidence a code path executed** — grep for marker
  log lines.
- **Hypothesis-falsifying experiments are cheap and decisive**: FENCE=1
  predicted-faster came out WORSE (63.16 vs 57.10ms) — killed the
  chained-collective theory in one run.
- **Low-probability stalls need computed trial counts**: Event::wait
  stall rate ~0.17%/call — two blind repro runs were never going to
  catch it.

---

## 11. Recurring Patterns Across Entries

1. **Microbenchmark wins do not transfer end-to-end.** Occurred ≥4
   independent times: STEP_SIZE=4096 (isolated +15% → e2e -8%),
   fused-topk (+microbench → +0.13% e2e), fused indexer (dispatch
   analysis → 0.54x pipelined), TurboQuant (bandwidth math → -6 to
   -11%). **Rule: no lever ships on a component-level number alone.**
2. **"GPU is idle/busy" reasoning misleads both directions.** A 2935ms
   GPU-idle envelope was not capturable by fusion/fence tricks (three
   failed attempts, one -23% prefill); conversely 96-97% "utilization"
   coexisted with all_sum eating 62% of wall (blocked submission
   thread). Utilization telemetry is neither necessary nor sufficient
   evidence.
3. **Correctness fixes ≠ throughput fixes, and vice versa.** Token-tree
   (fixed, no lift), MTP c=2 cache fix (fixed, still 5.7 t/s),
   shared-scale int8 (correct, slower), TOPK=160 (fast, garbage). These
   are independent axes; every champion needs both a needle test and a
   σ-qualified throughput run.
4. **Optimizations rot; contradictions exist in the record.** gate+up
   fusion: -3.8% (June) vs +3.01% (August). STEP_SIZE tuning: opposite
   signs on Qwen3.5/PP vs DSv4/TP. bf16→fp16: win on one model, 7x loss
   on another. **Never transplant a result across model, topology, or a
   two-month code gap without re-measuring.**
5. **Untested default-off flags are a liability.** FUSED_SOFTMAX hid a
   correctness bug for ~5 weeks; PBLOCK's "zero overhead" comment was
   false; DSpark Native Head was never A/B'd. Either validate a flag or
   delete it.
6. **Speculative decoding's binding constraint at long context is the
   verify KV-attention floor (~30ms).** Every widening scheme (trees,
   higher gamma) loses to this floor; every win in this area came from
   correctness/overhead fixes (tie-break, rollback, drain-elimination),
   not cleverer drafting.
7. **Exposed collectives on the critical path are the recurring
   transport killer** — not bandwidth. Eagle K=1 (16KB broadcast → 17x),
   all_sum chunking config (92 MB/s effective), decode fence (98% of
   wait). Fix ordering/overlap/config, not the wire.
8. **Concurrency tier transitions break things silently.** c=1→c=2 broke
   MTP caches, quality (twice), admission, and champion claims. Any
   change validated at c=1 is unvalidated at c=2, and 100K-validated is
   unvalidated at 500K.

---

## Appendix: Closed Levers — Do Not Re-attempt Without New Evidence

| Lever | Verdict | Ref |
|---|---|---|
| Quantized/shared-scale int8 all_sum | Infeasible (true reduction) / 1.49x slower | 2026-08-19/20 |
| Token-tree drafting (all variants) | Never beat linear; verify-floor bound | phases 6-8 |
| Hybrid PP-prefill/TP-decode phase swap | Rejected twice | 2026-08-04, 08-16 |
| TurboQuant KV compression | All configs regressed | turboquant doc |
| MoE tile retune (bm>16), MAXBE widening, small-M kernel | At/above ceiling already | 2026-08-18/20 |
| Fused topk, fused indexer, wq_a+wkv fusion | Noise or slower | multiple |
| STEP_SIZE=4096 on DSv4/TP | -8% e2e, twice | 2026-08-18/19 |
| fp16 compute on DSv4/JACCL | 7x slowdown, no fp16 transport | lifecycle trace |
| JACCL_RELIABLE_MAX_SZ=3 | Clean null | 2026-08-20 |
| Expert co-location for TP traffic | Traffic independent of placement | section108 |
| PREFILL_CHUNK_OVERLAP | Dead/closed, correctness anomaly | 2026-08-20 |
| DSpark FULLBLOCK at depth | 15.9x collapse; default OFF | 2026-08-04 |
| MLX_METAL_FAST_SYNCH=1 | 1.5x slower, 70x variance | phase0a |

Still open: the 550ms PP decode stall (6 hypotheses refuted,
unexplained), the attn.proj_qkv 2.35x span-vs-kernel anomaly, and
sequence-chunk overlap (valid mechanism, 5-7% realistic ceiling —
marginal ROI).

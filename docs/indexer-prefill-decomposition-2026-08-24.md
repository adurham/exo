# attn.indexer prefill decomposition — sub-op breakdown at production shape, candidate-or-closed verdict (2026-08-24)

## Why this check

`docs/prefill-flops-roofline-aggregate-2026-08-22.md` T7 aggregate lists
`attn.indexer` at 4.0% of prefill wall time (`24,596.71 ms / 2310 calls
= 10.65 ms/call` at 220K,
`docs/dsv4-220k-prefill-span-profile-2026-08-18.md`) as one of the last
sub-spans in the "un-decomposed 28.8% non-GEMM remainder" — never
individually attacked with a fresh sub-op breakdown. This document is
that breakdown: sub-op FLOP/bandwidth arithmetic at production prefill
shape (B=1, TP=2, EXO_DSV4_SEQ_SPLIT=1, `L_band=1024`, 220K/100K/500K
sweeps), then a per-sub-op verdict against the archive of already-closed
indexer levers, then a candidate-or-closed conclusion.

**Scope**: code-reading + arithmetic. No live cluster contact.
No GPU microbenching. Where a definitive verdict genuinely requires a
runtime measurement, flagged as such.

## Prior art (all already closed — do not re-propose)

Absorbed before decomposition, so nothing below is a re-litigation:

- **`docs/PERFORMANCE_HISTORY.md` §4.6** (`docs/PERFORMANCE_HISTORY.md:1099-1105`):
  a whole-indexer fused Metal kernel (the "indexer fused kernel" plan
  `.hermes/plans/2026-05-14_185010-dsv4-indexer-fused-kernel.md`)
  was built, correctness-validated (159/160 topK overlap), and
  measured **0.54x = SLOWER once pipelined** in a 21-call microbench
  (215 µs/call vs the existing chain's 117 µs/call). Root cause: MLX's
  async graph already overlaps dispatch across the sub-op chain, so
  the per-call-in-isolation dispatch-overhead estimate that motivated
  the plan (~28.4→32.3 tok/s NOP-lift) badly overestimated what a
  single fused kernel could recover. **Do not re-attempt whole-chain
  fusion of the indexer.**
- **`docs/indexer-topk-fused-decode-only-2026-08-22.md`**:
  `EXO_DSV4_TOPK_FUSED` cannot engage at prefill — dispatch requires
  `scores.shape[1] == 1` (decode-only), and prefill has `scores.shape[1]
  = L_band = 1024`. Structural, cheap-to-verify, closed.
- **`docs/indexer-pblock-decode-regression-2026-08-21.md`**:
  `EXO_DSV4_INDEXER_PBLOCK` (tiled-P scoring for memory-bound stall
  spikes) was prefill-neutral at all tested depths and caused a real
  decode regression at deep context. Closed.
- **OPT-6 score-GEMM folding** (deepseek_v4.py:3675-3680,
  `PERFORMANCE_HISTORY.md:734`): the per-head weights are folded INTO
  `q` before the score GEMM, collapsing 64 heads to 1 and doing a
  SINGLE `(L,D)@(D,P)` matmul instead of 64 head GEMMs + a collapse.
  **64× compute reduction, already shipped**. The FLOP counts below
  use the folded form.
- **2026-06-21 profiler note**: indexer cost grows +18% with depth,
  driven by the P-scaling score GEMM. Consistent with the per-depth
  numbers below.
- **`docs/dsv4-4096-regression-root-cause-2026-08-19.md`**: every
  indexer sub-stage (score/pmask/topk) measured CHEAPER per token at
  larger chunk size. The indexer is not the 4096-chunk villain.

## Live code path

Prefill indexer entry: `SparseCompressedAttention.__call__`
(`mlx-lm/mlx_lm/models/deepseek_v4.py:4386-4841`), under
`with span("attn.indexer"):` (line 4563) →
`Indexer.__call__` (deepseek_v4.py:3782-3949).

Only fired for **sparse layers** (compress_ratio=4, 21 of 43 layers).
Compressed layers (ratio=128, 20 layers) do not call the indexer;
Local layers (ratio=0, 2 layers) don't either. This matches
the span profile: 2310 calls total = 21 sparse layers × 110 chunks.

## Sub-op sequence at production shape (per call, per rank)

Config: `hidden=4096`, `q_lora_rank=1024`, `index_n_heads=64`,
`index_head_dim=128`, `index_topk=512`, mxfp8 weights everywhere
in attention (`config.json` + `make_quantization_config`,
deepseek_v4.py:911). Prefill: `L_full=2048` (chunk),
`L_band = L_full / TP = 1024` under EXO_DSV4_SEQ_SPLIT=1. `TP=2`.
Compressed-pool length at ratio=4: `P = ceil(ctx/4)` = 25K/55K/125K
at 100K/220K/500K.

Sub-ops, in execution order (all inside `attn.indexer` span):

| # | sub-op | source | shape / cost formula |
|---|---|---|---|
| 1 | `indexer.compressor(x, pool_cache, offset)` — `wkv` + `wgate` mxfp8 GEMMs at FULL L (coherence) + pool ring update + rope on new pooled rows | deepseek_v4.py:3798, 3086-3266 | 2 × GEMM(L=2048, K=4096, N=256) = 8.59 GFLOP + pool ops |
| 2 | `pmask = _dispatch_pmask(pool_cache, L_full, offset)` | :3804, :637-682 | build bool (L_full, P) then slice to (L_band, P) |
| 3 | band-slice x, q_residual to `_seq_lo:_seq_hi` | :3806-3813 | array-view (free) |
| 4 | `wq_b(q_residual_band)` mxfp8 GEMM | :3818 | GEMM(L_band=1024, K=1024, N=8192) = **17.18 GFLOP** |
| 5 | `_rope_dispatch(q, q_off)` | :3820 | elementwise, ~free |
| 6 | `weights_proj(x_band)` mxfp8 GEMM | :3827/:3836 | GEMM(L_band, K=4096, N=64) = 0.54 GFLOP |
| 7 | `_indexer_score` (OPT-6 folded): sigmoid + scale, weight-fold reduce, single `(L_band, D=128) @ (D=128, P)` GEMM | :3675-3680 | ≈ 2·L_band·D·P FLOPs |
| 8 | pmask apply via `mx.where(pmask, scores, neg)`; OPT-12 tail-restricted to the `[vis_min, vis_max)` band (~L_band/ratio + 1 = ~257 columns) | :3840-3887 | bandwidth over (L_band, P) fp16 scores + bool mask |
| 9 | `mx.argsort(-scores, axis=-1)[..., :k]` (prefill argsort fallback, `EXO_DSV4_PREFILL_ARGPARTITION` default 0) | :3946-3949 | O(L_band · P log P) sort compute, but memory-bound: reads (L_band, P) fp16 scores, writes (L_band, k) int32 indices |
| 10 | `finalize(topk)` — implicit at line 4580 in the calling site — forces the whole lazy graph above to eval | :4580 | GPU sync barrier |

## FLOP breakdown, per-depth (per call, per rank)

Constants (no ctx dependence): `wq_b=17.18`, `compressor=8.59`,
`weights_proj=0.54` (GFLOP).

Score GEMM scales linearly with pool length P:

| context | P (pool@ratio=4) | score GFLOP | total GFLOP | score-share | wq_b-share | comp-share |
|---|---|---|---|---|---|---|
| 100K |  25000 |  6.55 | 32.86 | 19.9% | 52.3% | 26.1% |
| 220K |  55000 | 14.42 | 40.72 | 35.4% | 42.2% | 21.1% |
| 500K | 125000 | 32.77 | 59.07 | 55.5% | 29.1% | 14.5% |

**Compute-side headline: score GEMM overtakes wq_b as the largest sub-op
somewhere around 220K, and dominates at 500K.** All three GEMMs (score,
wq_b, compressor) are mxfp8-quantized. wq_b at `(1024, 1024, 8192)` is
a large-M dense GEMM in exactly the shape family MLX's mxfp8 quantized
GEMM handles well (attention's own `attn.proj_qkv` runs the same class
of shape at 84.7% of ceiling —
`docs/dsv4-attention-kernel-efficiency-2026-08-18.md`).

## Achieved-vs-ceiling attribution using existing span data

The `indexer.score` (8.34 µs/call) and `indexer.topk` (5.81 µs/call)
inner spans in the profile only measure **lazy-graph build time**, not
GPU compute — sanity: if the score GEMM at 220K (14.42 GFLOP) executed
in 8.34 µs it would be 1.7 PFLOPS, physically impossible on M4 Max's
11.66 TFLOPS dense ceiling. Real compute is charged to the outer
`attn.indexer` span's `finalize(topk)` sync barrier at line 4580.

Using the outer span (10.65 ms/call at 220K, 2310 calls) as the honest
per-call cost:

Achieved TFLOPS (attn.indexer, 220K) = 40.72 GFLOP / 10.65 ms
= **3.82 TFLOPS** ≈ **32.8% of the 11.66 TF dense-fp16 ceiling**.

This is far below the 61-85% figures seen for the four studied
attention spans. But the ceiling comparison is misleading because
the indexer is **not** a pure-GEMM span:
- Predicted GEMM wall (if all three GEMMs ran at attn.proj_qkv's 9.64
  TFLOPS): 40.72 GFLOP / 9.64 TFLOPS ≈ **4.22 ms**.
- Observed wall: **10.65 ms**.
- Gap: **~6.4 ms = 60% of the span** in NON-GEMM work: compressor pool
  ring update + `_dispatch_pmask` build + pmask `where` apply + argsort
  + eval barrier synchronization + lazy-dispatch overhead.

**So the indexer's 4.0% wall time splits roughly into ~1.6% GEMM (already
at ceiling) and ~2.4% non-GEMM overhead.** The non-GEMM half is where
any residual lever must live.

## Per-sub-op verdict

| # | sub-op | status | reason |
|---|---|---|---|
| 1 | indexer compressor GEMMs | **at ceiling** | 8.59 GFLOP mxfp8 GEMM, same class as attn.proj_qkv @ 84.7%. Predicted ~0.9 ms of the span. |
| 4 | wq_b GEMM | **at ceiling** | 17.18 GFLOP mxfp8 GEMM at (1024, 1024, 8192) — canonical dense large-M shape. Predicted ~1.8 ms. Attention's own wq_b in the same file at (1024, 1024, 16384) runs at 84.7%. |
| 6 | weights_proj | **at ceiling / negligible** | 0.54 GFLOP, ~0.06 ms. |
| 7 | score GEMM (folded) | **at ceiling** | OPT-6 shipped 64× reduction; the remaining single `(L, D=128, P)` GEMM is a plain mxfp8 GEMM. At 220K, 14.42 GFLOP / ~1.5-2 ms → 7-10 TFLOPS, ceiling-consistent. |
| 2 | pmask build | **already-closed (OPT-12 TAIL_PMASK)** | Row-causal pmask is monotone; OPT-12 restricts the O(L·P) `where` to a narrow ~257-column band. Deployed. |
| 8 | pmask where | **already-closed (OPT-12)** | Same. Predicted ~0.5-1 ms at 220K. |
| 9 | argsort → topk | **candidate territory, but bounded — see below** | `mx.argsort(-scores, ...)` over (L_band, P) fp16 at 220K = ~113 MB read; ~1-2 ms. |
| 10 | finalize/eval barriers | **already-closed (§4.6 dispatch-pattern)** | Sub-op fusion already tried, 0.54× pipelined. |

### The one plausible sub-op candidate: prefill top-K replacement

`EXO_DSV4_PREFILL_ARGPARTITION=1` already exists in the code
(deepseek_v4.py:3941-3945; env-gated, default OFF). It replaces
`argsort(-scores)[..., :k]` with `argpartition(-scores, kth=k-1)[..., :k]`
— identical top-k set for the downstream gathered attention (which is
softmax-order-invariant), O(P) instead of O(P log P).

**Predicted best-case saving**: if argpartition halves the top-k step
cost at P=55000 (a reasonable guess for asymptotic O(P) vs O(P log P) at
log2(55K)≈16), that removes at most ~1 ms of the ~1-2 ms argsort cost
= ~10% of the 10.65 ms span = **~0.4% e2e**.

- **Below the 1% threshold.**
- Additionally, the same lever's own code comment (`deepseek_v4.py:3937-3940`)
  notes argpartition is SLOWER than argsort at small P (Metal launch
  overhead), and there is an `EXO_DSV4_ARGPARTITION_MIN_P` guard. So
  even the 0.4% is conditional on tuning MIN_P — likely the reason
  this lever was left env-gated OFF rather than defaulted ON.

### The one plausible non-argsort candidate: fused Metal top-K kernel extended to L>1

`_fused_topk` already exists for decode (`scores.shape[1]==1`,
deepseek_v4.py:3888-3910). Extending it to arbitrary L would require
a per-row histogram/threshold Metal kernel that fits into the same
correctness envelope (masked scores map to lowest key, exact top-k
multiset).

**Predicted best-case saving** at 220K prefill: if the fused top-K
achieved the same ~5.5× speedup as the decode variant on the topk
substep, and topk is ~1-2 ms of the span, savings are ~0.8-1.6 ms/call
= 7.5-15% of `attn.indexer` = **0.3-0.6% e2e**.

- **Below the 1% threshold.**
- **This is the SAME class of design PERFORMANCE_HISTORY §4.6 already
  measured 0.54× pipelined** for the whole-indexer fused kernel. The
  narrower topk-only version might dodge the whole-chain fusion trap,
  but nothing about the L=1 decode microbench's 5.5× predicts L=1024
  will scale the same way — Metal grid launch overhead is amortized
  differently at L=1 (single row, launch-dominated) than at L=1024
  (many rows, memory-bandwidth-dominated on the score read anyway,
  the same bandwidth term argsort already touches).

## Final verdict — candidate or closed?

**CLOSED — no un-tried fused-kernel candidate with predicted e2e ≥ 1%
exists.**

The 4.0% wall time cap makes ≥1% e2e require **≥25% reduction of the
attn.indexer span**. Attributing the span:
- ~1.6% wall is GEMM (score + wq_b + compressor), already at ceiling.
- ~2.4% wall is non-GEMM (pmask, argsort, eval barriers, dispatch).

Even a *free* argsort (~1-2 ms) is at most ~15-20% span reduction =
0.6-0.8% e2e. Even a *free* pmask apply (already OPT-12-narrowed) is
~5-10% span reduction. Even a *free* argsort AND pmask apply combined
is ≤30% reduction = ~1.2% e2e best case — but only under the
counterfactual assumption both become genuinely free, which contradicts
the pipelined-dispatch reality PERFORMANCE_HISTORY §4.6 already
measured for a related whole-indexer fused kernel (0.54× SLOWER once
pipelined).

The rigorous ≥1% bar is not met by any single sub-op lever, and the
composite lever (argpartition + fused topk + tail-pmask, most of which
already exist gated) sums to ~0.7-1.0% best case, at the ragged edge
of the threshold, with high probability of the pipelined-dispatch trap
eating any per-call savings.

**Recommendation: close the last T7 remainder box for attn.indexer.**
Move any future indexer work to a real *runtime* verification of the
already-existing `EXO_DSV4_PREFILL_ARGPARTITION` lever at 220K/500K
prefill — a single live A/B, no new code — before considering any
kernel investment. If that A/B shows ≥1% e2e (unlikely per the ceiling
above), only then does the fused-topk-for-prefill investment become
justifiable.

## Follow-up experiment spec (if the argpartition A/B is later attempted)

- **Change**: relaunch cluster with `EXO_DSV4_PREFILL_ARGPARTITION=1
  EXO_DSV4_ARGPARTITION_MIN_P=8192` (guard against small-context
  Metal-launch regression per the code's own gate).
- **Baseline**: known-good `EXO_DSV4_SEQ_SPLIT=1 EXO_PREFILL_STEP_SIZE=2048`.
- **Measurement**: 100K, 220K, 500K prefill throughput ladder,
  ≥3 iterations per depth, σ<0.5 tok/s. Correctness: needle probe at
  100K.
- **Pre-registered decision**: if e2e ∆ < +1% at both 220K and 500K
  (σ-adjusted), keep default OFF and close indexer permanently.
- **Runtime cost**: one relaunch, ~20 min of cluster time.

## Honest uncertainty flags

1. **The ~1-2 ms argsort estimate is arithmetic, not measured**. MLX's
   `argsort` cost per row at P=55000 fp16 is bandwidth-bound with
   ~113 MB per call read, but the actual Metal kernel throughput
   (sort passes, memory reuse) is unknown from code reading alone.
   A microbench (`bench/indexer_score_microbench.py` already exists,
   could be extended with an argpartition A/B) would tighten this
   number — worth doing if the prod A/B above shows any signal.
2. **The "score GEMM at ceiling" claim is inferred**, not directly
   measured. `bench/indexer_score_microbench.py` already exists with
   a fp32 reference; adding a "score kernel measured TFLOPS vs matched-
   shape dense fp16 GEMM" cell (the same methodology as
   `attn_production_class_bench.py`) would confirm. Expected to
   confirm — the shape is dense GEMM-friendly and there is no known
   dispatch anomaly for this specific op.
3. **Cross-op eval-barrier attribution is estimated by subtraction**.
   The "6.4 ms non-GEMM residual" splits over pmask/argsort/finalize/
   dispatch by best-guess arithmetic, not by a real per-sub-op fenced
   probe. `EXO_PROFILER_SYNC_SPANS=1` would give per-sub-op wall time
   directly, but attn-indexer's inner spans (indexer.score,
   indexer.topk) are already emitted without an internal finalize — a
   sync-spans run would need matching code additions (a `finalize(scores)`
   after line 3839 and a `finalize(topk)` before line 3949). This is a
   real, cheap follow-up if the argpartition A/B shows signal and the
   sub-op attribution needs sharpening.
4. **`compress_ratio == 4` overlap pattern for the indexer's own
   Compressor**. The Compressor at ratio=4 uses `overlap=True`
   (deepseek_v4.py:3093), doubling `out_dim` for wkv/wgate. The FLOP
   count above uses the doubled `out_dim=256` — verified. The pool
   accumulator's per-chunk cost (accumulate_windows, deferred commit)
   is small on a memory basis but not exactly quantified here — folded
   into the "6.4 ms non-GEMM residual" bucket.

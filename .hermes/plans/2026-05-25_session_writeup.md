# 35 t/s plan — session writeup (2026-05-24 / 2026-05-25)

Source plan: `.hermes/plans/2026-05-23_35_tps_plan.md`
Session output docs:
- `.hermes/plans/2026-05-24_35_tps_plan_execution_results.md`
- `.hermes/plans/2026-05-24_w3_K8_norenorm_results.md`

## Champion progression this session

```
                                              mean      σ      Δ vs previous
baseline (K=0, with renorm)                  28.80    0.10    —
+ K=8 + no-renorm Eagle soft_emb             29.04    0.07    +0.83%   p<0.001
+ deferred PoolingCache update (default)     29.75    0.10    +2.45%   p<<0.001
─────────────────────────────────────────────────────────────────────────────
Session total Δ                                                +3.30%
Distance to 35 t/s target                                       -15.0% (-5.25 t/s)
```

**Live champion tag:** `champion-2026-05-25-pool-defer-29.8` at exo
commit `4a15cc7a`, mlx-lm commit `0e61738` (branch `eagle-soft-emb`).

## What's deployed in production defaults

- `EXO_DSV4_MTP_EAGLE_K=8` (start_cluster.sh, 2026-05-24)
- No-renorm K>1 soft_emb math in `mtp_module.py:715-729`
- Deferred `PoolingCache.update_and_fetch_deferred()` as the default
  path in `Compressor.__call__` (mlx-lm/mlx_lm/models/deepseek_v4.py,
  cache.py extended with `commit_pending` + `update_and_fetch_deferred`)
- All other knobs at start_cluster.sh defaults (γ=2, FENCE=4,
  INDEX_TOPK=512, KV bf16, MTP=1, SPECULATIVE=1)
- Escape hatch toggles in `/tmp/dsv4_nop_targets`:
  - `compressor_defer_off` — sync pool update (revert to pre-defer)
  - `compressor`, `compressor_proj`, `compressor_accum`,
    `compressor_compress`, `compressor_pool` — granular NOP probes
    (quality-destroying; bench-only)

## Workstreams explored

### W2 — fused-topk validation (NEGATIVE, no ship)

10-iter c=1 100K with vs without `EXO_DSV4_TOPK_FUSED=1` via live
file-toggle. Δ = +0.04 t/s (+0.13%), Welch t=0.98 (p~0.34), 2σ
separation FAIL. Below the measurement floor; not adopted as default.
Toggle still available for future experiments.

### W1 — c=2 100K spec-batched bug (DROPPED)

Instrumentation plan written (`/tmp/w1_c2_instrumentation_plan.md`).
User dropped c=2 entirely ("no more c=2"). c=2 path is now frozen;
fix lives in skill ref `2026-05-24-c2-deprioritized.md`. Don't menu
c=2 work going forward.

### W3 — Eagle implementation audit (LANDED, +0.83%)

Found and fixed renormalization deviation at `mtp_module.py:718` —
top-K probs were being normalized to sum-to-1 over the K-subset,
distorting the soft_emb mixture's L2 norm vs what the MTP head was
trained on. Empirical p_1 distribution from `EXO_DSV4_MTP_DUMP_TOPK=1`
showed slot-2 chained median p_1=0.74 (47% of steps below 0.70 — real
Eagle K>1 territory), so K>1 acceptance flatness was a code bug not
structural ceiling. Fix is in place at default K=8.

**Acceptance/decode puzzle** (unresolved, side note): mean_accept
dropped from 1.087/2 (K=0) to 0.894/2 (K=8 + no-renorm) — that's
-17.8% acceptance — but decode tps went UP +0.83%. Bench is ground
truth, but the mechanism is unexplained. Possibly the `_spec_accept_hist`
counter scope differs when Eagle is installed, or K>1 reduces some
verify-forward overhead that offsets the lower yield. Worth a future
investigation if anyone cares about acceptance histogram correctness.

### W4 — Compressor structural lever (PARTIAL +2.4% via defer)

**Profiler attribution** (EXO_PROFILER=spans + SIGUSR1 dump):
```
span                avg_us   total_ms    % of attn
attn (parent)        9823    265788     100.0%
  attn.compressor    4664    120091      45.2%  ← biggest single child
  attn.sdpa          2390     64664      24.3%
  attn.indexer       2942     38794      14.6%
  (unaccounted)        —      42239      16.0%  ← projections + RoPE + mask + concat
```

**Whole-compressor NOP probe (pre-defer):** 29.04 → 37.20 t/s (+28.1%)
**Whole-compressor NOP probe (post-defer):** 29.75 → 37.37 t/s (+25.6%)
**Sub-op attribution (pre-defer):**
- `compressor_proj` NOP: 27.45 t/s (-5.5% — negative, zero alloc costs > matmul)
- `compressor_compress` NOP: 37.25 t/s (= whole-compressor — the compress kernel chain IS the cost)
- `compressor_accum` / `compressor_pool` NOPs: not separately measured (cheap by deduction)

**Reorder experiment:** moved `self.compressor()` call BEFORE the
q/k/v projections in `CompressedAttention.__call__` and
`SparseCompressedAttention.__call__`. **Neutral** (28.97 vs 29.0
baseline). mlx already overlaps independent compute via its lazy
graph; statement order doesn't matter.

**Defer experiment (SHIPPED):** Extended `PoolingCache` API with:
- `update_and_fetch_deferred(px)` — writes `px` into pool storage NOW
  but defers the visible-offset bump to next step. Returns the
  PRE-WRITE prefix view so SDPA's lazy graph has no dependency on the
  compress kernel.
- `commit_pending()` — applies any staged offset bump. Called at the
  top of `Compressor.__call__` so all in-call reads see consistent
  state.
- `BatchPoolingCache` got stub versions that fall through to the sync
  path (c=2 unchanged per user policy).

Quality validation: 100K needle ✓ FALCON-MERCURY-7749, BOS=0,
bistability=0, short-prompt smoke ✓. Result: **+0.71-0.86 t/s
(+2.4-3.0%, Welch t=13.96, p<<0.001)**.

**Pipelined compress+norm+rope microbench:**
- `chain_simple` (R=128 D=128): per-call 0.319 ms, pipelined 0.087 ms = **3.69×**
- `chain_overlap` (R=4 D=256): per-call 0.271 ms, pipelined 0.107 ms = **2.54×**

Both exceed the 1.7× gate from `topk-fused-kernel-ship.md` — chain
genuinely serializes at per-call allocation/dispatch. **But the
cluster ceiling for fusing this chain is only ~0.5% throughput**
(math: 5 fires/token × 0.087 ms saved at most = 0.5/33.6 ms token wall
× 4× microbench-to-cluster divisor). Not worth multi-day Metal kernel
work for that ceiling.

**The remaining +25.6% NOP-probe gap is mostly SDPA shape**, not
compress dispatch. NOP replaces `pooled` with `mx.zeros((B, 0,
head_dim))`, so SDPA's `kv = concat(kv, pooled)` runs on local-KV
only at every layer. Narrower KV → less SDPA work. This is
structural and quality-defining — can't be optimized away without
changing the model's compressed memory mechanism.

## State left on cluster at session pause

- Cluster on `4a15cc7a` (defer-default champion)
- **m4-1 and m4-2 have `/tmp/dsv4_nop_targets = "compressor_compress"`**
  from the attribution bench that was running when laptop rebooted —
  CLEAR BEFORE next bench, otherwise quality breaks
- The interrupted bench was attempting to attribute the post-defer
  +25.6% NOP gap between compress-chain dispatch vs SDPA-shape. Result
  inferred from microbench math but not directly measured.

### Cleanup checklist for next session

```bash
ssh -i ~/.ssh/exo_cluster -o IdentitiesOnly=yes adam.durham@192.168.86.201 'rm -f /tmp/dsv4_nop_targets; pkill -f concurrent_bench 2>/dev/null'
ssh -i ~/.ssh/exo_cluster -o IdentitiesOnly=yes adam.durham@192.168.86.202 'rm -f /tmp/dsv4_nop_targets'
# Then verify quality with short curl + 100K needle
```

## What's NOT on the table (per user policy, do not re-attempt)

- c=2 work (frozen)
- `EXO_KV_CACHE_BITS != 0` (forbidden quality floor)
- `EXO_DSV4_INDEX_TOPK < 512` (forbidden quality floor)
- Custom Metal kernel for compress+norm+rope fusion — ceiling is
  only ~0.5% cluster throughput per the microbench math above. Skill
  ref `custom-metal-kernel-authorship.md` precedent (topk-fused) was
  +1.4% for similar effort; this is below that ROI.
- `@mx.compile` wrapping of compress chain — `_simple_compress_kv` /
  `_overlap_compress_kv` are ALREADY `@mx.compile`'d; wrapping further
  hits the "actively HURTS" anti-pattern (skill ref `mx-compile-patterns.md`).

## Next session — where to look for the next >1% lever

Honest assessment: the compressor is exhausted as a lever (~0.5%
ceiling left, multi-day cost). The remaining attn-wall slices are:

### Option A — Indexer (14.6% of attn wall, second-biggest child)

Already had work this year (`5518f17` bf16 fix, `ee68e44` SDPA
refactor, `topk-fused` kernel). But NOP-probe hasn't been re-run
recently with defer in place. **Quick win to scope: run
`compressor_compress` NOP and `indexer` NOP separately, measure each.**
If indexer NOP gives more than +1% cluster, there's room. If it gives
<+0.5%, it's tapped out.

Bench plan (~30 min wall):
1. baseline (no toggle): expected ~29.75
2. `indexer` NOP only: gives upper-bound for indexer optimization
3. Restore baseline, compare

### Option B — The 16% unaccounted attn wall

The profiler's `attn` parent shows 16% of total attn time NOT in any
named child span. That's the per-layer **projections + RoPE + masks +
mx.concatenate**. Currently no granular probe. Add span instrumentation
around each piece to see which sub-op dominates. If it's `mx.concatenate`
or the post-attn `wo_a` + `wo_b`, those might be fusable.

Cost: ~2 hours to add spans, run probe, restart cluster.

### Option C — Longer-context (160K, 200K) where compressor share grows

At 100K, compressor is 45% of attn wall. At 200K (the real production
target context), it might be 60%+ since the compressed-KV branch
scales sub-linearly while local KV scales linearly. **If you ever run
benchmarks at 200K**, re-attribute and the fusion lever might pay off
better.

### Option D — Cross-layer compress hoisting

The compressor runs INDEPENDENTLY per layer but with the SAME `x`
input shape and similar parameters. Could the compressor's pool be
SHARED across layers (per-layer-group instead of per-layer)? That's a
model architecture change — invasive — but would slash compressor wall
to 1/N of current. Unknown quality impact. **Only worth investigating
if user wants a multi-week structural project.**

## Recommended sequence for next session

1. **Cleanup** (cleanup checklist above).
2. **Run option A indexer NOP** (~30 min). If +>1%, dive into indexer.
   If not, move to (3).
3. **Option B sub-op profiler around the 16% unaccounted attn wall**
   (~2 hours). Decompose. If a >1% lever surfaces, work it.
4. If both A and B come up empty: discuss with user whether to do
   option C (longer-context retest) or option D (architecture change),
   or stop and accept 29.75 as champion.

## Commits + tags landed this session

```
exo (origin/main):
  ee2f2abc  (start of session) docs(fork-notes): May 23 2026
  928a390c  diag(mtp): EXO_DSV4_MTP_DUMP_TOPK=1
  3cca5896  diag(launcher): propagate dump env
  2d8d5efc  fix(mtp-eagle): no top-K renorm
  8eea311b  default(start_cluster): K=8 default     [champion-2026-05-24-K8-norenorm-29.0]
  b0df595a  chore: bump mlx-lm (compressor NOP)
  88fa70b8  Revert (BOS-spam false alarm)
  096466d4  Reapply (reproduced clean)
  1c08e125  chore: bump mlx-lm (granular NOPs)
  44a1fe19  chore: bump mlx-lm (compressor reorder, neutral)
  480834d6  chore: bump mlx-lm (defer toggle v1, crashed)
  1a7799e3  Revert (defer v1 crash)
  099d79d48 chore: bump mlx-lm (defer v2 with API extension)
  f3210e59  chore: bump mlx-lm (BatchPoolingCache stubs)
  4a15cc7a  chore: bump mlx-lm (defer default)        [champion-2026-05-25-pool-defer-29.8]

mlx-lm (eagle-soft-emb):
  c3690735  (start)
  2a341edc  diag: NOP toggle "compressor"
  9417a8c1  diag: granular sub-op NOPs
  3540ea22  perf: compressor before q/k/v (neutral)
  ee8a6086  diag: compressor_defer toggle v1 (crashed)
  5274ad1d  perf: PoolingCache deferred update API
  93e89182  fix: BatchPoolingCache stubs
  0e61738e  perf: defer pool update default

Tags:
  champion-2026-05-24-K8-norenorm-29.0  → exo 8eea311b
  champion-2026-05-25-pool-defer-29.8   → exo 4a15cc7a  ← CURRENT
```

## Skill refs updated / to update

Should be updated with this session's findings before next session:
- `references/perf-baselines.md` — add 29.75 entry with config
- `references/2026-05-23-c1-mtp-on-spec-sweep.md` — update K=8
  acceptance numbers (mean_accept 0.89/2 with no-renorm, not 1.07/2)
- New ref: `2026-05-25-compressor-defer-and-fusion-ceiling.md` —
  document the defer extraction + the +25% remaining NOP gap being
  SDPA-shape not dispatch
- `references/mx-compile-patterns.md` — add the compress chain to the
  "tried and failed" list (reorder was neutral)

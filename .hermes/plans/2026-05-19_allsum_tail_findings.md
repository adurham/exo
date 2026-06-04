# Phase B Findings — All_Sum Tail Investigation (DSv4, gamma=2, 100K c=1)

**Date:** 2026-05-19 ~10:30 CDT
**Plan:** `.hermes/plans/2026-05-18_1830-dsv4-verify-tail-investigation.md`
**Baseline anchor:** `baseline-2026-05-18-mtp-g2-topk512-30.06` (30.062 t/s sigma=0.059 10/10 clean)
**Probe:** EXO_DSV4_ALLSUM_PROBE (mlx-lm `79a6a72d`)

## TL;DR

**H1 (chained-collective peer-CQE arrival tail) is FALSIFIED.**

The verify-phase mean cost (57ms at FENCE=43) decomposes cleanly into
**~1.4ms per layer x 43 layers = 60ms drain time**. Per-layer eval is
TIGHT in steady-state (p50 ~ p99 within ~5%). Adding fences at every
layer (FENCE=1) does NOT clip the tail -- it MAKES IT WORSE (verify
mean 57 -> 63ms, max 70 -> 85ms). This is the opposite of what H1
predicted.

**The 35% verify spread (52->70ms) seen in the MTP-PROF baseline data
was inflated by iter-0 cold-compile-cache contributions.** In actual
steady-state (windows 800-840 of FENCE=43 probe data), verify is ~52ms
+/- 4ms (~8% spread, tight).

There is NO recoverable +1-2 t/s from a "fence placement" or
"chained-collective drain" fix. Phase C (the fix variants C1-C4) is
NOT actionable on this data.

## Methodology

Probe added in `DeepseekV4MoE.__call__` (deepseek_v4.py:1146-1163 fast
path, :1201-1224 span path) wrapping the existing `mx.eval(y)` fence
call. CPU-wall measured around the eval; per-layer p50/p99/max
aggregated over 20-cycle windows. Zero-cost when env var unset.

Two configurations benched against the same 75K-word prompt:
- **FENCE=43** (production default for gamma=2): only layer 42 fires
  the fence; one drain per forward.
- **FENCE=1** (diagnostic): every layer fires the fence; per-layer
  cost measurable directly.

Each config: 1 inference probe (max_tokens=30) + 1 100K bench
(iterations=2, warmup=1). Bench iter 1 = scored.

## Results

### FENCE=43 (production)

| Metric                 | Value                      |
|------------------------|----------------------------|
| Bench iter scored      | 29.9 t/s (vs 30.06 baseline; ~0.5% probe overhead) |
| MTP-PROF verify mean   | 57.10ms (matches baseline) |
| MTP-PROF verify range  | 47-70ms (n=60 cycles)      |
| ALLSUM steady-state p50 (windows 800-840) | **52ms** (one layer, full chain drain) |
| ALLSUM steady-state p99 (windows 800-840) | **56ms** |
| ALLSUM steady-state spread (warm only)    | **52-56ms = 8% spread** |

**Insight:** at steady-state, verify is much tighter than MTP-PROF's
average over the whole bench suggested. The 35% spread is the
iter-0 cold-cache contribution to MTP-PROF's running stats.

### FENCE=1 (diagnostic)

| Metric                 | Value                                            |
|------------------------|--------------------------------------------------|
| Bench iter scored      | 26.5 t/s (~12% slower than baseline)             |
| MTP-PROF verify mean   | 63.16ms                                          |
| MTP-PROF verify range  | 47.23-85.04ms (60% spread, WORSE than FENCE=43)  |
| ALLSUM per-layer p50   | 1.04-1.52ms (alternating, follows compress_ratio) |
| ALLSUM per-layer p99   | 1.07-1.74ms (TIGHT — p99/p50 ratio ~1.05-1.15)   |
| ALLSUM per-layer max_max | 9.78-15.61ms (rare outliers, no layer-localized) |
| Sum of per-layer p50s  | **59.68ms** (close to FENCE=43 verify mean 57ms) |

**Per-layer pattern (warm, decode):**

```
Layer  p50    p99    Pattern
0      1.09   1.22   start (special)
1      1.04   1.07   compress=0
2      1.52   1.74   compress=4
3      1.28   1.37   compress=128
4      1.50   1.66   compress=4
5      1.27   1.32   compress=128
...    [alternating ~1.50 / ~1.28 for layers 2-42]
42     1.52   1.66   end
43     0.86   0.95   MTP body (separate)
```

**Insight:** the layer-by-layer cost alternates with `compress_ratio`
(=4 layers slightly slower than =128 layers). NO localized hotspot at
layers 28-39 in steady-state. The "28-39 hotspot" I initially saw
was a cold-compile-cache artifact in the iter-0 cycles=20 window
(p99 up to 230ms for layers 28-39, then collapsed to ~1ms by
cycles=40).

### What FENCE=1 vs FENCE=43 tells us

| Quantity        | FENCE=43      | FENCE=1       | Delta             |
|-----------------|---------------|---------------|-------------------|
| Verify mean     | 57.10ms       | 63.16ms       | +6.06ms (+10.6%)  |
| Verify max      | 70.40ms       | 85.04ms       | +14.64ms (+20.8%) |
| Sum of per-layer drain | 57ms (one big) | 59.68ms (43 small) | +2.7ms |
| Per-fence overhead implied | -- | (63-57)/42 = 143us per extra fence | -- |

The extra 6ms at FENCE=1 = 42 extra fence events x ~143us per fence
(perf_counter call + mx.eval entry/exit + mlx queue drain handshake).

**The mlx queue drain handshake is the operative cost.** It's NOT a
chained-collective bug; it's just the steady-state cost of forcing a
GPU-CPU sync. FENCE=43 amortizes this -- you pay the handshake ONCE
per forward, draining 43 layers worth of work in one go.

### Cold-compile-cache observations

- At cycles=20 (iter-0 warmup), layers 28-39 showed p99 of 146-230ms
  per-layer eval, while layers 0-27 showed 10-33ms. By cycles=40, all
  layers were uniform at ~0.7-1ms p99.
- The MAX outlier ever observed across all windows: 2545ms (cycles=20,
  layer 42, FENCE=43 -- the entire 43-layer chain compile-cache miss
  on the very first forward).
- These are one-time costs amortized across the bench; in the 10-iter
  baseline run they contribute a small bias to the mean (probably ~5%
  of the 35% "spread" we attributed to chained-collective).

## Conclusion

The verify-phase cost is **dominated by uniform per-layer compute +
all_sum + drain**, not by a chained-collective tail or any
recoverable inefficiency. The fix variants C1-C4 in the original plan
all assumed H1; with H1 falsified, those fixes have no theoretical
basis.

### What this means for hitting 35 t/s

Per-cycle wall is bounded by `sum of per-layer all_sum drain time`.
At ~1.4ms x 43 layers = ~60ms verify. Plus 4.5ms draft, 0.8ms accept.
**The floor is ~65ms = ~31 t/s (assuming alpha=1)**, before any
per-layer compute reduction. To break that floor, options are:

1. **Reduce per-layer cost.** Each all_sum is ~0.7-1.5ms; this is the
   Thunderbolt 5 RDMA + Mac Studio M4 Max latency floor for the
   ~hidden_size * hc_mult * 2 bytes payload (~16KB per layer at our
   shape). To go lower we'd need to (a) reduce payload (different
   sharding scheme) or (b) reduce layer count (model architecture
   change, off limits).
2. **Increase tokens-per-cycle** (raise alpha). With alpha~0.52,
   tokens/cycle ~2. Raising alpha to 0.7 -> 2.4 tokens/cycle -> +20%
   throughput at fixed verify cost. This is Phase D4 (gamma=3 probe)
   territory and the most plausible remaining lever.
3. **Smaller payloads via tensor parallelism shape changes.** Currently
   we shard across hidden_dim. Could try sharding across heads
   instead. Risky.

## Recommended next steps

1. **Skip Phase C entirely.** Without H1 confirmation, the fix
   variants C1-C4 have no falsifiable mechanism. Don't try.
2. **Pivot to Phase D4 (gamma=3 acceptance probe).** This is now the
   highest-EV remaining lever. Plan:
   - Relaunch cluster with `EXO_SPECULATIVE_GAMMA=3`.
   - Inference probe (verify correctness).
   - 2-iter 100K bench, capture MTP_LOG mean_accept and verify
     timing.
   - If alpha_3 >= 0.4 (i.e. mean_accept >= 1.2), t/s should beat
     baseline; bench 10-iter for champion claim.
   - If alpha_3 < 0.3, abandon -- the verify cost growth from
     gamma=3 (3 extra tokens per forward) wins out.
3. **Document the probe.** Keep `EXO_DSV4_ALLSUM_PROBE` in the code
   (it's zero-cost when off, and useful for future debugging). The
   patch is already in mlx-lm `79a6a72d`.
4. **Update plan retro.** This document closes Phase A+B. Phase C is
   NOT actionable. Skip to Phase D4 OR accept the 30.06 t/s floor.

## Files / Artifacts

- Plan: `.hermes/plans/2026-05-18_1830-dsv4-verify-tail-investigation.md`
- Probe code: `mlx-lm/mlx_lm/models/deepseek_v4.py` (commit `79a6a72d`)
- Launcher forwarding: `start_cluster.sh` (commit `823d4589` on exo)
- Raw FENCE=43 log: `/tmp/exo_phaseB_fence43_archive.log` (210 ALLSUM-PROBE lines)
- Raw FENCE=1 log: `/tmp/exo_phaseB_fence1_archive.log` (2559 ALLSUM-PROBE lines)
- Analysis scripts: `/tmp/analyze.py`, `/tmp/analyze_fence1.py`, `/tmp/analyze_fence1_v3.py`
- Cluster state at writeup: baseline restored (mlx-lm `79a6a72d`, exo `823d4589`,
  FENCE=43, no probe env, READY 2/2).

---

## Phase D4 result: gamma=3 lever ALSO does not pay off

Quick 2-iter probe with `EXO_SPECULATIVE_GAMMA=3` (otherwise baseline config).

### Comparison

| Metric         | gamma=2 (baseline) | gamma=3              | Delta             |
|----------------|--------------------|----------------------|-------------------|
| draft (ms)     | 4.54               | 6.69                 | +47% (3 predicts) |
| verify (ms)    | 57.10              | 64.34                | +12.7% (L_q=4)    |
| accept (ms)    | 0.81               | 0.94                 | +16%              |
| total (ms)     | 62.65              | 72.17                | +15.2%            |
| mean_accept    | 1.04 / 2 = 0.52    | 1.12 / 3 = **0.37**  | acceptance DROPPED |
| tokens / cycle | 2.04               | 2.12                 | +3.9%             |
| t/s scored     | 30.06              | **26.1** (1 iter)    | **WORSE**         |

### Why

The model's `num_nextn_predict_layers=1` (only one MTP head) means chaining
3 predicts amplifies the prediction error nonlinearly. alpha_3 = 0.37 is
well below the 0.4 break-even threshold I set up-front (computed as: per-cycle
cost grew 15% so acceptance must grow >=15% to hold tokens/sec).

### Conclusion

gamma=3 is NOT a free win at current acceptance rates. To make gamma>=3 pay
off, would need (a) a better MTP head (train a 2-3 head model — far out of
scope), or (b) a richer draft mechanism (e.g. token-tree drafting) that
explores multiple branches per step. Neither is in-scope for a tonight-style
session.

## Final verdict

**The 30.06 t/s quality-correct baseline (FENCE=43, TOPK=512, gamma=2, MTP=1)
appears to be the ceiling for this hardware/model/architecture combo at
single-stream c=1.**

Available levers exhausted by today's two sessions (verify-forward plan +
this investigation plan):
- Compile-boundary collapse (Levers 1, 2) — both regressed and reverted
- Fence-density (FENCE=8 / FENCE=1) — no improvement, FENCE=1 worse
- Chained-collective tail clip — H1 falsified by Phase B data
- gamma=3 acceptance — alpha_3 too low

What's NOT exhausted (out of scope for tonight, worth future investigation):
- Different tensor-parallel sharding (heads vs hidden) — risky
- Better MTP head (architectural training change) — out of scope
- Token-tree drafting — out of scope
- mlx C++ allreduce kernel — would need to bisect the May-17 champion first
  (Phase D3 from the plan) to see if there's a known recoverable regression

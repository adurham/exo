# Phase F Findings: jaccl RDMA is NOT the bottleneck

**Date:** 2026-05-19 ~13:25 CDT
**Plan:** `.hermes/plans/2026-05-19_structural_to_35tps.md`
**Probes:** `EXO_DSV4_ALLSUM_PROBE` (CPU-wall around mx.eval fence) +
`JACCL_POLL_INSTRUMENT=1 THRESHOLD_US=0` (per-call RDMA poll-loop wall)

## TL;DR

**The structural plan's premise was WRONG.** I assumed the ~1.4ms/layer
cost was in RDMA collective barriers. Per-call jaccl wall is actually
**5-15us median (8us mean)** at steady state — accounts for **1.9% of
verify wall**.

The other 98% lives in **mlx's GPU/CPU stream management** —
encoder.dispatch, mx.eval graph evaluation, GPU kernel launch +
completion. This is what Phase G+H+I should target, not the
ACK barriers.

## Data

Combined-probe run (TOPK=512 FENCE=43 GAMMA=2, mlx@facbed9a, mlx-lm
6dcdd40a, decode steady-state after warmup):

### JACCL_POLL_INSTRUMENT (per all_reduce call's RDMA poll loop)

```
warm steady-state (call_id > 100, wall < 1ms):
  count:  3126
  mean:   8.15us
  median: 5.00us
  p90:    ~11us
  p99:    69us
  max:    547us

wall_us histogram:
  [   0,     5)us:  43.6% of calls
  [   5,    10)us: 40.1%
  [  10,    20)us: 14.3%
  [  20,    50)us:  0.9%
  [  50,   100)us:  0.4%
  [ 100,   500)us:  0.6%
  [ 500,  5000)us:  0.1%
```

### ALLSUM_PROBE (CPU wall around mx.eval at layer-42 fence)

```
warm decode-only windows (cycles 120-220, ~100 cycles):
  median p50: 37.4ms
  median p99: 37.9ms
  min:        35.4ms (best case)
```

### MTP-PROF (verify total)

Verify = ~57.10ms total per cycle from earlier baseline.

## Decomposition

Verify wall (one forward, L_q=3):
- ALLSUM probe at last fence: ~37ms ← CPU wait for GPU completion
- All jaccl calls combined (42 layers × ~8us × 2 sub-calls/layer): ~670us
- **Difference: ~36.3ms is GPU compute + CPU dispatch + sync**

## What this tells us

1. **Adding/removing ACK barriers (Phase G/H of structural plan) is
   optimizing 0.6% of cost.** Not worth pursuing.

2. **Phase F's premise (enable fastskip) was misguided.** Even if
   fastskip saved 100% of ack_sync_pre cost (which it can't), that's
   maybe 30us per call × 43 = 1.3ms verify saving. Marginal.

3. **The May-18 02:02 32.29 "champion" being on TOPK=160 (broken
   quality) was the ONLY way to get above the 30 t/s ceiling at this
   binary.** Without TOPK speedup, we're at the GPU+CPU sync floor.

4. **The 4.3 t/s regression on May-18 12:48 fix-branch deploy was
   probably ALSO not about ACK barriers** — it was a different bug
   in the fix branch's dedicated-ACK-QP code path. Re-investigation
   would have to bisect within the fix branch.

## New optimization targets (revised structural priorities)

### Priority 1: GPU stream coalescing / fewer mx.eval() syncs

Each `mx.eval(y)` blocks CPU on a synchronizer until GPU completion.
At FENCE=43 we have ONE such block per forward (currently 37ms wait).
The GPU is busy doing 43 layers of MoE+attn during that wait. Can we:
- Overlap CPU-side work (next-token prep, MTP head, sampling) with
  the GPU compute by pipelining?
- Issue the GPU work for the NEXT token in parallel with this token's
  generation (token-tree-style)?

### Priority 2: Reduce per-layer Python overhead

Each layer's Python call (`block.__call__`) builds a fresh lazy graph
that mlx then has to evaluate. At 43 layers × N microseconds = 
substantial Python overhead. The existing `_compiled_attn_pre`,
`_compiled_post_attn`, etc. compile pure chunks, but the layer's
`__call__` still has Python glue. Profiling needed.

### Priority 3: Faster MoE expert dispatch

The MoE call (`DeepseekV4MoE.__call__`) at TP=2 does:
- gate (compute expert routing)
- switch_mlp (gather_qmm for 6 of 256 experts)
- post_combine (weighted sum)
- all_sum (the 8us collective)

The gather_qmm + matmul is the dominant GPU work. Whether it's
already optimal needs MLX_BUILD_PROBE data or Metal System Trace.

### Priority 4: Concurrency (escape hatch — sidesteps the per-stream wall)

At c=2 MTP scales ~2.7x per the docstring. c=1 is the WORST case for
this architecture. If aggregate throughput at c=2 is >35 t/s with
acceptable per-stream latency, that may satisfy the user's goal.

## What I will NOT do

- Deploy the fix-branch mlx (deploy regression unexplained, fix targets
  the wrong layer)
- Phase G/H (ACK barrier merging) — optimizing 0.6% of cost
- Phase I (GPU-stream allreduce) — same, the collective is already fast

## What I recommend NEXT

Need user direction. The path to 35 t/s now looks like:
- **Option A**: Profile mlx-side: MLX_BUILD_PROBE to find which phase
  of the per-layer compute dominates. If it's gather_qmm or MoE expert
  dispatch, that's a mlx-internal optimization (writing a fused
  gather_qmm kernel). Real work, 1-2 days.
- **Option B**: CPU/GPU pipelining: overlap Python next-step prep with
  GPU compute. Requires rewriting parts of MTPBatchGenerator. 3-5 days.
- **Option C**: Accept that c=1 single-stream is at ~30 t/s, bench c=2
  for aggregate. If aggregate hits 35+, ship that.
- **Option D**: Accept current performance and invest in something
  else (cluster scaling? prefix caching for repeat-context workloads?).

# DSv4 Indexer fused kernel — collapse 5-7 dispatches/layer into one

## ABANDONED 2026-05-14 — phase-1 spike failed gate

Pipelined 21-call microbench at production shape on m4-1:
  current chain: 117 us/call pipelined
  fused (numerically correct, 159/160 top-K overlap): 215 us/call = 0.54x — SLOWER

Op-level breakdown explains why:
  matmul alone (q @ pool.T):              225 us/call
  argsort alone (pre-baked scores):       208 us/call
  other ops (ReLU/transpose/weights/sum):  37 us/call
  Total per-call current:                 287 us
  Total pipelined current:                117 us/call (Apple already overlaps)

Apple's matmul and argsort are heavily tuned. Hand-rolled Metal can't beat
them. Theoretical fusion ceiling = ~37 us/call of "other ops" -> 10-15 us
pipelined = ~0.25 ms/token = ~0.9% throughput. Below noise floor.

The "indexer is 95% dispatch overhead" theory was WRONG. The 3.8 ms NOP
gain is real compute (matmul + argsort) that LOOKS like overhead because
MLX pipelines so well. The pipelined microbench (117 us/call) is the true
ceiling, not the per-call (287 us) which over-attributes to overhead.

SAME LESSON as sparse_attn fused-kernel plan: pipelined microbench tells
truth, per-call analysis lies. The 5 plans now killed all fail on this
same trap — once MLX has pipelined the chain, the remaining bottleneck is
the actual compute, and Apple already runs it near hardware ceiling.


## Goal

Cut the indexer's per-token wall on the cluster from ~3.8 ms (~14% of total)
toward its theoretical floor (~80 µs total across 21 layers, per FLOP/BW
analysis), by fusing the per-call chain into one Metal kernel.

Target throughput: 28.4 → 31.0+ t/s (+9%+) at c=1 100K MTP-off.

## Why this is different from the killed plans

The earlier four plans (MoE kernel, sparse_attn kernel, expert co-location,
compress_ratios reshape) were all killed by data. None of them was
**both** dispatch-overhead bound AND able to be measured at the right scale.

The indexer IS both:

| Component  | Bottleneck nature              | Pipelined microbench
|------------|--------------------------------|----------------------
| MoE        | memory-BW (90% of BW floor)    | killed at +3% ceiling
| sparse_attn| compute (chain already pipelined)| killed at 1.23x
| **indexer**| **CPU dispatch overhead**      | **never measured at chain level**

The May 14 isolation probes show:

  BASELINE             28.4 t/s
  ALL_SUM NOP          28.9 t/s (+1.8%)   — fence is nearly free
  **INDEXER NOP**      **32.3 t/s (+13.7%)** — ~3.8 ms recoverable

Theoretical compute floor across 21 indexer calls is ~80 µs (FLOPs +
memory-BW budget). Measured cost is 3.8 ms ≈ **48x slower than the floor**.
Almost all of that is per-dispatch overhead times ~5-7 dispatches per call
times 21 layers per token = ~110 dispatches per token wired Python→MLX
→Metal launch cost.

Variant_d (May 13, applied) already optimized `_indexer_score` 2.1x at the
op-level microbench but produced zero cluster gain — because the issue is
NOT in any single op but in the dispatch BOUNDARIES between ops.

This is exactly the shape of problem where fused kernels actually work.

## What needs to be fused (per layer)

Per indexer call in `SparseCompressedAttention.__call__` (21 layers/token):

  1. `self.compressor(x, comp_cache, offset)`:
       - `_project_kv_gate(x)` → fused-quantized matmul (kv + gate combined)
       - At 1-of-4 decode tokens: `compress_func` + `norm` + `rope` to grow pool
       - At 3-of-4 tokens: pool_cache.update_and_fetch returns a slice (no copy)
  2. `self.wq_b(q_residual)` → Linear (q_lora_rank=1536 → 64*128=8192)
  3. `.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)` → no compute, just metadata
  4. `position_rope(q, offset)` → mx.fast.rope (already optimized)
  5. `_indexer_score(q, pooled, weights_x, scale, n_heads_inv_sqrt)`:
       - matmul q @ pooled.T → (B, H=64, L=1, P=25000)
       - ReLU
       - weights * (scale * n_heads_inv_sqrt) → (B, L, H, 1)
       - matmul (B, L, P, H) @ (B, L, H, 1) → (B, L, P, 1)
       - squeeze → (B, L, P)
  6. `make_mask(L, offset)` → (L, P) bool mask, often None at decode
  7. `mx.where(mask, scores, -inf)` (if mask)
  8. `mx.argsort(-scores, axis=-1)[..., :k]` → top-K=160 indices

That's roughly 5-7 distinct mx.* op dispatches per call. Plus several
implicit reshapes/transposes which may or may not become dispatches.

**Fusion target: ONE Metal kernel per indexer call** that:
- Takes: pooled (B, P, D), q (B, H, L, D), weights_x (B, L, H), pmask
- Produces: top-K=160 indices (int32)
- Computes the entire score → ReLU → H-reduce → mask → top-K pipeline inside

The compressor (input projection / pool growth / norm / rope) is more
complicated — it's stateful (pool_cache mutation) and uses quantized
matmul. **Leave compressor outside the kernel.** The kernel handles
the deterministic scoring + top-K path.

## Shape sanity (verify before writing kernel)

For DSv4-Flash-8bit on the cluster, per indexer call at c=1 100K MTP-off:

| Symbol | Value | Notes |
|---|---|---|
| B (batch) | 1 | c=1 |
| H (index_n_heads) | 64 | config.index_n_heads (NOT tensor-parallel sharded — compressor on each rank holds full HC) |
| L (q length) | 1 | MTP-off decode |
| D (index_head_dim) | 128 | config.index_head_dim |
| K (top-K) | 160 | EXO_DSV4_INDEX_TOPK runtime override |
| P (pool size) | ~25000 | at 100K context, compress_ratio=4 |

q tensor: (1, 64, 1, 128) bf16 = 16 KB per call
pool tensor: (1, 25000, 128) bf16 = ~6.4 MB
weights_x: (1, 1, 64) bf16 = ~128 B
scores intermediate: (1, 64, 1, 25000) bf16 = ~3.2 MB
output: (1, 1, 160) int32 = 640 B

Working set: ~10 MB per call. Fits in M4 Max L2 cache (~40 MB).
Memory-BW floor: 10 MB / 400 GB/s = 25 µs per call × 21 layers = 525 µs.
Compute floor: ~80 µs as noted.
Measured: ~3.8 ms / 21 = ~180 µs/call.

**Most of the 180 µs is overhead, not compute or BW.** Fusion can recover.

## Proposed kernel structure

`mx.fast.metal_kernel` taking inputs:
- `q`:        (1, 64, 1, 128) bf16
- `pooled`:   (1, P, 128) bf16
- `weights`:  (1, 1, 64) bf16
- `pmask`:    (1, P) bool, optional (passed as `None` → skip mask)
- `scale`:    float
- `head_inv_sqrt`: float

Output:
- `topk`:     (1, 1, 160) int32

Algorithm (one threadgroup per "pool position chunk" of size G, e.g. G=256):

  Phase 1 — Score per pool position
    For each thread (mapped to one pool index p):
      acc = 0
      for h in 0..64:
        for d in 0..128:
          acc += q[0, h, 0, d] * pooled[0, p, d]
      score = max(0, acc) * weights[0, 0, h] for each h, sum over h, apply scale * head_inv_sqrt
      if pmask is not None and !pmask[0, p]: score = -inf

  Phase 2 — top-K reduction (the hard part)
    Use a per-threadgroup running top-K heap, then merge across threadgroups.
    OR: bitonic-sort within threadgroup, scatter top-K to global memory,
    then a second small kernel pass to merge.
    OR: skip top-K in kernel and let MLX argsort the output scores —
    smaller kernel but loses 1 dispatch boundary savings.

The simplest **starting** version writes the full scores tensor to global
memory and lets MLX do `argsort(-scores)[..., :k]`. This already
eliminates 4-5 dispatches per call (matmul, ReLU, H-reduce-via-matmul,
weights mul, scaling) and keeps just argsort as the second dispatch.
That's a 5x dispatch reduction per call. **Start there.**

## Why the score-fusion alone should win

Per-call cost is ~180 µs measured. Per-dispatch overhead is roughly the
Python→MLX boundary (~20 µs each) + Metal launch (~20 µs each) = ~40 µs.

Per-call dispatches today (rough):
  - quantized_matmul (compressor) — 1
  - kv/gate split — 1 (or 2 if not fused)
  - compress + norm + rope (only 1-in-4 tokens) — 3
  - wq_b Linear — 1
  - reshape/transpose — 1
  - rope on q — 1
  - score matmul — 1
  - ReLU max — 1
  - weights mul + scale — 1
  - H-reduce matmul — 1
  - squeeze — 1 (free, view-only)
  - mask gen + apply — 1
  - argsort + slice — 1

That's ~12 dispatches/call × 21 calls = **252 dispatches/token** just for the indexer.

Fuse score computation (steps 6-12) into one kernel:
  - 6 dispatches collapse to 1 → save 5 dispatches/call × 21 calls = 105 dispatches/token.
  - At ~40 µs/dispatch overhead = 4.2 ms recoverable.

That matches the 3.8 ms NOP-measured headroom almost exactly. Numbers
line up — fusion CAN recover most of the indexer cost.

## Step-by-step plan

### Phase 1 — Microbench spike (1-2 days, validates kernel CAN win)

1. Write `bench/dsv4_indexer_pipelined_microbench.py`:
   - Build inputs at production shape (B=1, H=64, L=1, D=128, K=160, P=25000) bf16
   - Build the "current chain": replicate `_indexer_score` + argsort + slice
     for 21 sequential calls in a single eval (with different pool sizes
     to mimic real layer-by-layer variation: pool grows by 1 every
     compress_ratio=4 tokens, so 21 layers see 21 distinct pool sizes
     in the warmup window, then steady state)
   - Build a candidate fused kernel that does score → ReLU → H-reduce →
     scale → mask in one metal_kernel pass. Argsort stays outside as
     a second mx.argsort dispatch (or fused later in phase 2).
   - Verify numerical equivalence: at least the top-K=160 index set should
     overlap ≥ 158/160 with the current path (same character as variant_d
     vs baseline microbench).
   - Time both at p50 over 1000 iters.
   - CRITICAL: also run a **pipelined 21-call chain** in a single eval
     (the lesson from MoE/sparse_attn). Per-call alone can lie.

2. Decision gate: pipelined chain speedup ≥ 1.7x.
   - At 1.7x: saves (1 - 1/1.7) × 3.8 = 1.6 ms → 28.4 → 30.0 t/s = +5.6%
   - At 2.0x: saves 1.9 ms → 30.5 t/s = +7.4%
   - At 3.0x: saves 2.5 ms → 31.4 t/s = +10.6%
   - At 5.0x: saves 3.0 ms → 32.0 t/s = +12.7%

3. If gate met: proceed to phase 2.
   If 1.3-1.7x: marginal. Probably not worth integration cost; consider
   pushing further fusion (argsort + RoPE + wq_b).
   If <1.3x: abandon. Means the chain is already pipelined (same trap
   as sparse_attn). Document and accept ceiling.

### Phase 2 — Integration (2-3 days)

1. Add `_indexer_score_fused()` Metal kernel wrapper next to the existing
   `_indexer_score` Python function in `mlx-lm/mlx_lm/models/deepseek_v4.py`.
2. Modify `Indexer.__call__` to dispatch to the fused kernel when env
   `EXO_DSV4_INDEXER_FUSED=1` is set. Default off.
3. The `mx.argsort` step stays Python (Apple has a tuned argsort kernel).
4. Wire env through `start_cluster.sh`.

### Phase 3 — Cluster validation (1 day)

1. Commit + push to adurham/mlx-lm + adurham/exo, submodule bump.
2. Hot-sync the venv (per venv-trap recipe — `scp` to both nodes' .venv).
3. Bench progression:
   - 8K c=1 with `EXO_DSV4_INDEXER_FUSED=0`: confirm champion baseline.
   - 8K c=1 with `=1`: indexer fires only when pool size > index_topk=160,
     which doesn't happen at 8K — so this should be identical to baseline,
     confirming the env flag is dormant.
   - **100K c=1 MTP-off**: the real test. Target: 28.4 → 30.5+ t/s.
   - 100K c=2 MTP-off: confirm no regression.
4. Quality gate: byte-identical greedy decoding at temp=0 vs baseline at
   8K (sparse path dormant — should be trivially identical) AND at 100K
   (where the fused path fires). Allow up to 5-token divergence at 100K
   (bf16 ULP noise around top-K boundaries — same character as variant_d).
5. If +5% gate met: tag new champion.
   If marginal: ship default-off, doc.
   If regression: roll back.

## Files to change

mlx-lm submodule:
  - `mlx_lm/models/deepseek_v4.py`:
    - New `_indexer_score_fused()` function with metal_kernel source.
    - Modified `Indexer.__call__` to dispatch to fused path under env flag.
    - Existing `_indexer_score` untouched (fallback when env=0).

exo repo:
  - `src/exo/shared/constants.py`: `EXO_DSV4_INDEXER_FUSED` constant.
  - `start_cluster.sh`: propagate env var.
  - `bench/dsv4_indexer_pipelined_microbench.py`: phase-1 spike.

## Tests / validation

Numerical:
- `||scores_fused - scores_unfused|| / ||scores_unfused|| < 5e-2` over 50 seeds.
  (Same tolerance as variant_d which was accepted.)
- Top-K=160 index set overlap ≥ 158/160 over 50 seeds at production shape.
- Token-sequence match at temp=0 over 50 tokens at 100K: allow ≤5 token
  divergence (same accuracy bar as the existing variant_d transformation).

Perf gates:
- Phase 1: ≥ 1.7x PIPELINED microbench speedup vs current `_indexer_score` chain.
- Phase 3: ≥ +5% agg_tps at 100K c=1 MTP-off vs champion 28.4 t/s.

Regression:
- 8K c=1 (FUSED=0 or =1) identical: indexer dormant at 8K.
- c=2 100K MTP-off: no regression.
- MTP-on (`EXO_DSV4_MTP=1`): the comment in `_indexer_score` warns that
  this transformation reduces draft/verify acceptance under MTP-on. The
  fused kernel inherits the same bf16 cast removal, so it should have
  the same MTP-on penalty. NOT a regression vs current variant_d state,
  but worth flagging.

## Risks, tradeoffs, open questions

**Risk 1: The chain is already pipelined.** Same trap as sparse_attn —
MLX may already overlap the 12 indexer dispatches at the chain level via
its async graph executor. The pipelined microbench answers this in phase 1.
*Mitigation:* be ready to abandon if pipelined speedup < 1.3x. Don't
sink integration effort into a marginal microbench result.

**Risk 2: Argsort fusion is hard.** Top-K=160 of P=25000 inside a Metal
kernel requires bitonic sort or heap structures. Apple's argsort is
tuned and probably faster than anything we'd write. *Mitigation:* phase 2
leaves argsort OUTSIDE the kernel. Argsort is one dispatch; it's not the
problem (per the comment in code, argsort+slice is already 0.21 ms which
is small compared to the 3.8 ms total).

**Risk 3: pmask handling.** The mask is `(L, P)` and only fires when
L > 1 (per `make_mask`'s `L == 1: return None`). At decode L=1 the mask
is always None — so we can omit mask handling from the L=1 fast path
entirely. *Mitigation:* L=1-only kernel, fall back to current path for L > 1.

**Risk 4: Numerical drift accumulating across 21 layers.** Each layer's
top-K choice depends on `argsort(-scores)`. Small score perturbations can
flip near-tie indices. Over 21 layers per token, errors might compound.
The variant_d microbench showed 159-160/160 overlap on a single call —
across 21 layers that could mean 21 different tokens see slightly
different sparse-attn views. The 5-token tolerance at 100K should cover
this; if not we have a real quality problem.

**Risk 5: MTP-on penalty.** Already documented in `_indexer_score`. Fused
kernel inherits the same. NOT a regression but a known characteristic.

**Risk 6: pool size variability across layers.** Pool size varies between
21 layers within the warmup window. The kernel needs `shapeless` semantics
for `P` (or be parametrized at call time). `mx.fast.metal_kernel` supports
runtime shape params — straightforward.

**Open question 1:** Should we also fuse `wq_b(q_residual)` + RoPE +
indexer into a single mega-kernel? Probably yes if phase 3 hits the +5%
gate but not the +10% stretch. Argsort and compressor stay out.

**Open question 2:** Could we fuse the score computation with the next
layer's sparse-attn gather? They're sequential at the data-flow level
(topk indices flow from indexer to sparse_attn). A "produce-K-indices-
then-immediately-gather" kernel could save one round-trip to global
memory. Defer to phase 4 if phase 3 succeeds.

**Open question 3:** The compressor's pool growth (1-in-4 tokens) has
its own dispatch chain. Could that be fused? At 1-in-4 tokens that's
amortized to 1/4 of the per-step indexer cost. Smaller lever. Defer.

## Rollback path

Champion tag `champion-2026-05-13-29.47` on adurham/exo and adurham/mlx-lm.
Rollback: `~/.hermes/scripts/rollback_to_champion.sh`.
Flag defaults OFF (`EXO_DSV4_INDEXER_FUSED=0`).

## Why this is the right target NOW

After **five plan abandonments** (MoE kernel, sparse_attn kernel, expert
co-location, compress_ratios reshape — all data-driven), the May 14
isolation probes (ALL_SUM-NOP +1.8% only; INDEXER-NOP +13.7%) sharpened
the diagnosis. The indexer cost is overwhelmingly CPU dispatch overhead,
not GPU compute, not memory bandwidth, not cross-rank coordination.

CPU dispatch overhead IS the kind of problem fused Metal kernels reliably
solve. The MoE plan was killed because it was MEMORY-bound, not dispatch-
bound. The sparse_attn plan was killed because the chain was ALREADY
pipelined enough that fusion couldn't help. The indexer is **measurably
different from both**: it has theoretical floor ~80 µs vs measured 3.8 ms
(48x slower than floor), and the variant_d 2.1x op-level speedup that
produced ZERO cluster gain proves the bottleneck is BETWEEN ops, not
within them.

**Expected delivery:** 28.4 → 30-31 t/s at c=1 100K MTP-off (+5-10%),
~1 week of work. If phase 1 microbench falls short of 1.7x, accept it
as the genuine ceiling and stop chasing.

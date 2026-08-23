# P3 Worker C3 — IN-SITU test of the PoolingCache donation-failure hypothesis (2026-08-23)

**Task**: Worker C flagged PoolingCache buffer-donation failure as the most
testable candidate for the **~3.5–4.2 ms/token residual** (live +6.80 minus
attention-kernel +2.56–+3.34). Do not close it by citing C's +6.35 ms upper
bound — **TEST** it against the REAL production decode loop, and check
additivity so the pieces cannot double-count.

---

## 0. Headline

**Donation is NOT failing in production. But the hypothesis was pointing at a
real, and larger, mechanism sitting immediately next to it.**

The production decode loop's cache is **not** `PoolingCache` — it is
**`BatchPoolingCache`** (`_merge_caches` converts it at
`mlx-lm/mlx_lm/generate.py:1261` → `PoolingCache.merge` at
`mlx-lm/mlx_lm/models/cache.py:1822-1823`). Verified in-situ: the harness
printed `LIVE CACHE CLASS: BatchPoolingCache` on every run. Worker C's
microbench measured `PoolingCache`.

The two classes differ in exactly the way that decides this question:

| | `PoolingCache` (C's bench) | `BatchPoolingCache` (production) |
|---|---|---|
| storage growth | **`step`-chunked, 256 entries at a time** (`cache.py:1522-1528`) | **`mx.concatenate` to EXACTLY `max_pool`, i.e. +1 entry** (`cache.py:1899-1903`) |
| growth frequency at decode | 1 flush in 256 | **EVERY flush** |
| donation applicable to growth? | no, but it's rare | **no, and it's every time** |

So in production the dominant pool cost is **not** the donatable slice-assign
that the donation hypothesis is about. It is an **unconditional
`mx.concatenate` that reallocates and copies the whole O(P·D) pool on every
single pooled flush**, in all 62 live `BatchPoolingCache` objects. Donation
cannot help it; toggling donation barely moves it.

**Measured, in the real loop, on production silicon:**

| quantity | 100,026 | 352,599 | Δ |
|---|---|---|---|
| (a) production as-is — flush excess, amortized | **+0.557 ms/tok** | **+2.504 ms/tok** | **+1.947** |
| (c) donation maximally enabled (`mx.synchronize` per step) | +0.553 | +2.493 | +1.940 |
| (b) donation deliberately defeated | +1.850 | +3.437 | +1.587 |
| (d) **concat suppressed** (pool pre-padded), donation on | **+0.019** | **+0.055** | **+0.036** |
| (e) concat suppressed AND donation defeated | +1.002 | +1.066 | +0.064 |

**(a) ≈ (c) to within 0.011 ms/token at both depths** → the production loop does
**NOT** defeat donation. That is the clean negative result the task asked for,
stated plainly.

**(a) → (d) removes 98–99% of the pool cost** → the cost is the concat, not the
donation.

**Additivity verdict: the depth-scaling pool cost is ADDITIVE with, and
disjoint from, Worker C's attention-kernel delta — but the sum overshoots the
live anchor, so it cannot all be new.** See §5.

---

## 1. Q1a — code analysis: who holds references to cache buffers across steps?

### 1.1 The real call chain (cited)

Production runs `DSV4_SHARDING=Tensor`, `EXO_DSV4_MTP=0`, `EXO_DSV4_DSPARK=1`
(read live off the running rank-1 process command line on
`adams-mac-studio-m4-2`). In Tensor mode `self.group.size() > 1` is true but
`get_pipeline_info(self.model)` returns `None` (there is no pipeline split), so
`_pp_spec_active` stays False (`batch_generate.py:971-981`) and the PP/DSpark
loop is never entered. The bench route B1 traced therefore runs the plain path:

```
exo   ExoBatchGenerator.submit()      src/exo/worker/engines/mlx/generator/batch_generate.py:2678
        -> mlx_lm BatchGenerator.insert()                      mlx-lm/mlx_lm/generate.py:1915
exo   ExoBatchGenerator.step()        batch_generate.py:4131
        -> self._mlx_gen.next()       batch_generate.py:4228   (explicitly ONE _next() pass)
          -> BatchGenerator._next()                            mlx-lm/mlx_lm/generate.py:2097
            -> GenerationBatch.next()                          mlx-lm/mlx_lm/generate.py:1739
              -> GenerationBatch._step()                       mlx-lm/mlx_lm/generate.py:1564
                -> Model.__call__                              deepseek_v4.py:6889
                  -> DeepseekV4Model._forward_steps            deepseek_v4.py:6512
                    -> DeepseekV4Block.__call__                deepseek_v4.py:4869
                      -> SparseCompressedAttention / CompressedAttention
                        -> Compressor -> <pool>.update_and_fetch_deferred
```

`insert()` does **not** keep the caller's cache objects.
`PromptProcessingBatch.__init__` runs `self.prompt_cache = _merge_caches(caches)`
(`generate.py:1261`), and `_merge_caches` dispatches to `PoolingCache.merge`
(`cache.py:1822`), which returns `BatchPoolingCache.merge(caches)`
(`cache.py:2666-2700`). **Production decode therefore never touches
`PoolingCache.update_and_fetch_deferred` at all** — it runs
`BatchPoolingCache.update_and_fetch_deferred` (`cache.py:1841-1922`).

### 1.2 Cross-step reference holders in the real loop

Auditing `GenerationBatch._step()` (`generate.py:1564-1712`) for anything that
could keep a live view of a pool buffer alive into the next step:

| holder | file:line | holds a pool ref across steps? |
|---|---|---|
| `self._next_tokens` / `_next_logprobs` | generate.py:1628-1629 | **No.** These are `(B,)` / `(B,V)` logit-derived arrays. Their lazy graph goes back through `lm_head`, `norm`, and the block residual stream — **not** through the pool tensor, which is consumed by SDPA and never appears in the output residual as a *view*. |
| `mx.async_eval(self._next_tokens, self._next_logprobs, token_context)` | generate.py:1632 | **No.** This *submits* the graph; it does not retain a Python reference to intermediates. |
| `mx.eval(inputs, self._current_logprobs)` | generate.py:1639 | **No**, and it is a hard fence on the PREVIOUS step's outputs — it forces the previous graph to have already materialised before the current step's slice-update evaluates. This is the opposite of the reference-holding failure mode. |
| `eager_detach_caches(self.prompt_cache)` | generate.py:1650 | **Actively helps.** It walks the cache list and calls `mx.detach` on the `keys`/`values` leaves (`cache.py:116-130`), explicitly to break the cross-step SliceUpdate chain. `_CACHE_DETACH_ATTRS = ("keys","values")` (`cache.py:73-76`) — it does not reach `pooled`, so it neither helps nor hurts the pool specifically. |
| logits processors (`ban_token_ids`) | batch_generate.py:2658-2660 / generate.py:1621-1627 | **No.** `ban_token_ids` (generate.py:1718-1726, exo) does an in-place logit write on a `(1,V)` tensor. No cache dependency. |
| sampler | generate.py:1616-1624 | **No.** Consumes logprobs only. |
| `self.tokens` append / `inputs.tolist()` | generate.py:1652-1654 | **No.** Host-side Python ints. |
| `extract_cache(i)` on finish | generate.py:1714 | Fires only on the terminating step, not per-step. |
| MTP / DSpark cache handles | batch_generate.py:2759, :2695 | **Not on this path.** `EXO_DSV4_MTP=0` in production; DSpark is only reachable through `_pp_spec_active`, which is False in Tensor mode. |
| `mx.async_eval(self.pooled)` | cache.py:1918 | The fork's own donation-determinism guard. Present and firing on this path. |

**Code-analysis answer to Q1: the production loop does NOT hold references that
defeat donation.** Every candidate holder is either logit-side (no pool
dependency), host-side, or an explicit *anti*-holding measure. The one path
that could have re-introduced it — a speculative/MTP path stashing cache
snapshots across steps — is inactive in the live config.

The runtime measurement (§3) confirms this independently.

---

## 2. Harness design

`bench/p3_donation_insitu_harness.py` (new, additive; run from `/tmp` on the
studio, nothing under `~/repos/exo` on either studio was created/edited/deleted).

- **Real model object.** `dv4.Model(ModelArgs.from_dict(CFG))` at real
  production config (dims from Worker A's doc + the real `config.json`), so all
  43 real `v4_attention_factory` blocks, the real `DeepseekV4Block`, the real
  `DeepseekV4Model._forward_steps`, the real `Model.__call__` / `lm_head`.
  Production quantization rules replicated from
  `make_quantization_config` (`deepseek_v4.py:899-931`).
- **Real cache.** `model.make_cache()` (`deepseek_v4.py:6956-6979`), synthetically
  pre-filled to depth L (RotatingKVCache into steady-state rotation; pools to
  `P = L//ratio` with realistic remainder and overlap carry) — the same
  pre-fill discipline as Worker C's bench.
- **Real loop.** `mlx_lm.generate.BatchGenerator` driven via `insert()` +
  `next()`, mirroring `batch_generate.py:2678` and `:4228` including the
  bench-mode `ban_token_ids` processor and a real `make_sampler` object.
- **Real cache class verified at runtime**, not assumed: the harness prints
  `LIVE CACHE CLASS` from `gen._generation_batch.prompt_cache` and asserts the
  pool actually advanced (`POOL ADVANCE CHECK: 88,149 -> 88,193, +44 over 177
  steps, expect ~44` — the ratio-4 flush cadence, exactly as designed).
- **Production env**: `EXO_KV_CACHE_BITS=0`, `EXO_DSV4_INDEX_TOPK=512`,
  `EXO_DSV4_INDEXER_PBLOCK` unset, `EXO_DSV4_TOPK_FUSED=0`,
  `EXO_DSV4_ATTN_ALLSUM=0`, `EXO_DSV4_EXACT_TOPK=1`, `EXO_DSV4_FENCE_ASYNC=1`,
  `EXO_DSV4_FENCE_EVERY_N_LAYERS=4`, `EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=8388608`
  — all cross-checked against the *live rank-1 process command line*, not just
  `start_cluster.sh`.

### 2.1 What was simplified — and why the comparison survives

**`DeepseekV4MoE` is replaced by a depth-INDEPENDENT stub MLP.** A 256-expert
top-6 MoE at real width is the 397B parameter bulk of the model; it does not
fit alongside the guard, and MoE/collectives are explicitly out of scope.

Why the (a)/(b)/(c)/(d) comparison survives: the stub's cost is identical in
every configuration and at every depth, so it **cancels exactly** in every
delta and in every between-config difference reported here. What does NOT
survive: the **absolute** ms/token (39–42 ms here) is meaningless as a
production per-token figure and is never used as one.

**Explicitly NOT covered by this harness**: any residual component living in
MoE-at-depth, in the TP `all_sum` collective, in cross-rank arrival skew, or in
the 85 GB-resident allocator regime. Those remain open.

**Also not covered**: single-process, so no cross-rank interaction; random
weights; B=1 only.

### 2.2 The measurement that makes this readable: the flush-phase split

A ratio-4 `BatchPoolingCache` writes a pooled entry on **1 step in 4**. The raw
per-step series shows this as a perfectly clean mod-4 periodicity (see the raw
output in §3). Splitting the series by `index mod 4` and taking
`flush_median − nonflush_median` isolates **the entire pool-write cost of all
21 sparse layers in one step**; dividing by 4 amortizes it to ms/token. This is
a far sharper instrument than the overall median, and it is what makes the
(a)/(b)/(c)/(d) discrimination unambiguous.

---

## 3. Raw results (studio `adams-mac-studio-m4-2`, MLX `0.32.1.dev20260822+e40a416b2`)

128 timed steps after 48 warmup, one depth per process (a first run that swept
both depths in one process was discarded — the first depth measured pays Metal
pipeline JIT and came out *slower* than the deeper one, a pure ordering
artifact).

### 3.1 Config (a) — PRODUCTION AS-IS

```
########## CONFIG=a L=100026 PB=8388608 ##########
  pool storage total 695.3 MB  (P_comp4=25,006 P_idx=25,006 P_comp128=781)
  LIVE CACHE CLASS: BatchPoolingCache  (ratio-4 compressor pool)
  POOL ADVANCE CHECK: pool length 25,006 -> 25,050 (+44 over 177 steps, expect ~44)
  per-step ms: median 39.151  mean 39.771  p10 38.863  p90 41.301  min 38.738  max 54.405
  PHASE SPLIT (mod 4): {0: 41.26, 1: 39.151, 2: 38.951, 3: 39.015}
    flush phase=0  flush median 41.260  non-flush median 39.034
    FLUSH EXCESS = +2.226 ms/flush-step  = +0.557 ms/token amortized
  per-step ACTIVE-mem delta (MB): median 0.34  max 32.04

########## CONFIG=a L=352599 PB=8388608 ##########
  pool storage total 2431.7 MB  (P_comp4=88,149 P_idx=88,149 P_comp128=2,754)
  LIVE CACHE CLASS: BatchPoolingCache  (ratio-4 compressor pool)
  POOL ADVANCE CHECK: pool length 88,149 -> 88,193 (+44 over 177 steps, expect ~44)
  per-step ms: median 41.798  mean 44.439  p10 41.416  p90 51.715  min 41.281  max 71.832
  PHASE SPLIT (mod 4): {0: 41.83, 1: 41.648, 2: 41.563, 3: 51.684}
    flush phase=3  flush median 51.684  non-flush median 41.670
    FLUSH EXCESS = +10.014 ms/flush-step  = +2.504 ms/token amortized
  per-step ACTIVE-mem delta (MB): median 0.34  max 107.09
```

The mod-4 periodicity is visible in the raw series with no statistics at all:

```
raw first 24 step ms @352,599 (config a):
[41.73, 41.607, 41.612, 51.106, 42.059, 41.563, 41.74, 51.352,
 42.042, 41.699, 42.063, 51.221, 41.976, 41.919, 41.877, 51.143,
 42.198, 42.05, 41.854, 51.231, 42.315, 41.82, 41.785, 51.23]
```

Every 4th step costs ~+9.5 ms. That is the pool flush.

### 3.2 Config (b) — DONATION DELIBERATELY DEFEATED

Reached through a *production env var*, not a code hack:
`EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=1<<40` makes
`pool_bytes > _POOL_DEFER_COPY_MAX_BYTES` false at `cache.py:1904-1907`, so
`visible = self.pooled` is captured and **held** across the slice-assigns —
exactly the documented donation-blocking failure mode.

```
########## CONFIG=b L=100026 PB=1099511627776 ##########
  per-step ms: median 39.262  mean 41.158  p10 38.913  p90 46.646  min 38.731  max 62.014
  PHASE SPLIT (mod 4): {0: 46.557, 1: 39.328, 2: 39.085, 3: 39.092}
    flush phase=0  flush median 46.557  non-flush median 39.158
    FLUSH EXCESS = +7.399 ms/flush-step  = +1.850 ms/token amortized

########## CONFIG=b L=352599 PB=1099511627776 ##########
  per-step ms: median 42.039  mean 45.653  p10 41.668  p90 55.736  min 41.497  max 81.414
  PHASE SPLIT (mod 4): {0: 42.182, 1: 41.768, 2: 41.773, 3: 55.644}
    flush phase=3  flush median 55.644  non-flush median 41.896
    FLUSH EXCESS = +13.748 ms/flush-step  = +3.437 ms/token amortized
```

**Defeating donation is measurable and real** (+1.29 ms/tok at 100K, +0.93 at
352.6K vs config a) — the mechanism exists and the env var is a working live
A/B knob. It is simply **not what production is doing**.

### 3.3 Config (c) — DONATION MAXIMALLY ENABLED

Full `mx.synchronize()` after every step: the pipeline is drained, so no
in-flight prior-step graph can possibly hold a stale view of the pool buffer
when the slice-update evaluates. Donation is structurally guaranteed.

```
########## CONFIG=c L=100026 ##########
    flush phase=0  flush median 41.344  non-flush median 39.133
    FLUSH EXCESS = +2.212 ms/flush-step  = +0.553 ms/token amortized

########## CONFIG=c L=352599 ##########
    flush phase=3  flush median 51.700  non-flush median 41.730
    FLUSH EXCESS = +9.970 ms/flush-step  = +2.493 ms/token amortized
```

**(a) vs (c): +0.557 vs +0.553 @100K; +2.504 vs +2.493 @352.6K.** Differences of
0.004 and 0.011 ms/token — an order of magnitude below run-to-run spread.
**Production is already in the donation-succeeding regime.**

### 3.4 Config (d) — the per-flush `mx.concatenate` SUPPRESSED

Config (d) pre-pads all 62 live `BatchPoolingCache` pools by +192 entries after
they migrate into the generation batch, so `self.pooled.shape[1] < max_pool`
(`cache.py:1899`) never becomes true for the whole run. The concat never fires;
only the donatable slice-assign remains. (Verified: the harness prints
`pre-padded 62 BatchPoolingCache pools` and hard-fails if it finds zero — an
earlier attempt silently pre-padded 0 because the caches had not yet migrated
out of `PromptProcessingBatch`; that null run was discarded.)

```
########## CONFIG=d L=100026 ##########
  (d) pre-padded 62 BatchPoolingCache pools by +192 entries -> per-flush mx.concatenate suppressed
  POOL ADVANCE CHECK: pool length 25,006 -> 25,050 (+44 over 177 steps, expect ~44)
  per-step ms: median 39.245  mean 39.391  p10 39.019  p90 39.549  min 38.817  max 52.624
  PHASE SPLIT (mod 4): {0: 39.23, 1: 39.313, 2: 39.235, 3: 39.238}
    flush phase=1  flush median 39.313  non-flush median 39.238
    FLUSH EXCESS = +0.075 ms/flush-step  = +0.019 ms/token amortized
  per-step ACTIVE-mem delta (MB): median 0.18  max 0.49

########## CONFIG=d L=352599 ##########
  (d) pre-padded 62 BatchPoolingCache pools by +192 entries -> per-flush mx.concatenate suppressed
  POOL ADVANCE CHECK: pool length 88,149 -> 88,194 (+45 over 177 steps, expect ~44)
  per-step ms: median 41.831  mean 42.005  p10 41.564  p90 42.107  min 41.296  max 63.005
  PHASE SPLIT (mod 4): {0: 41.688, 1: 41.708, 2: 41.963, 3: 41.818}
    flush phase=2  flush median 41.963  non-flush median 41.741
    FLUSH EXCESS = +0.222 ms/flush-step  = +0.055 ms/token amortized
  per-step ACTIVE-mem delta (MB): median 0.34  max 10.06
```

**The mod-4 periodicity vanishes entirely.** p90 collapses from 51.72 → 42.11 ms
at 352.6K. The flush excess drops from +2.504 to **+0.055 ms/token** — a **98%
reduction** — and the depth scaling of the flush cost drops from +1.947 to
**+0.036 ms/token**, i.e. essentially to zero.

### 3.5 Config (e) — concat suppressed AND donation defeated

The control that separates the two mechanisms:

```
########## CONFIG=e(d+defeated) L=100026 ##########
    FLUSH EXCESS = +4.009 ms/flush-step  = +1.002 ms/token amortized
########## CONFIG=e(d+defeated) L=352599 ##########
    FLUSH EXCESS = +4.264 ms/flush-step  = +1.066 ms/token amortized
```

With the concat gone, the residual donation-sensitive cost is +1.00 → +1.07
ms/token — **essentially FLAT in depth (+0.064 over a 3.5× depth increase)**.
That is the signature of a fixed per-flush overhead, not an O(P·D) copy.

Whereas (b) minus (a) — donation defeated *while the concat is still there* —
was +1.29 at 100K falling to +0.93 at 352.6K. The apparent "donation cost"
shrinks with depth precisely because the concat has already reallocated the
buffer, so there is progressively less left for donation to save.

**Conclusion: the depth-scaling pool cost is 100% concat, 0% donation.**

### 3.6 Reproducibility — 3 independent runs per point

| config | depth | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| (a) production | 100,026 | +0.557 | +0.556 | +0.564 |
| (a) production | 352,599 | **+2.504** | **+2.515** | **+2.499** |
| (d) concat suppressed | 100,026 | +0.019 | +0.017 | +0.014 |
| (d) concat suppressed | 352,599 | **+0.055** | **+0.048** | **+0.045** |

Spread ≤ 0.016 ms/token on every point. This is a very stable measurement.

### 3.7 Allocator-stat evidence

Per-step `mx.get_active_memory()` deltas, max over the timed window:

| config | 100,026 | 352,599 |
|---|---|---|
| (a) production | 32.0 MB | **107.1 MB** |
| (b) donation defeated | 30.2 MB | 107.1 MB |
| (d) concat suppressed | **0.49 MB** | **10.1 MB** |

The 107 MB transient at 352.6K in config (a) is the concat: the ratio-4
compressor pool is 88,149 × 512 × 2 B = 90.3 MB, plus the indexer pool at
88,149 × 128 × 2 B = 22.6 MB → 112.9 MB, matching the 107 MB observed
(the allocator reuses part of the freed block within the step). **Suppressing
the concat drops it to 10 MB — a 10.6× reduction in transient allocation**,
confirming from the allocator side what the timing shows.

Note the *negative* result this also delivers: config (b) does NOT show a
larger transient than (a). A donation failure would allocate a fresh pool
buffer; it doesn't, because the concat had already allocated one. Consistent.

---

## 4. Q1 answered

**Does the REAL production decode loop's reference pattern defeat PoolingCache
donation at depth?**

**NO.** Both by code analysis (§1.2 — no cross-step pool reference holder
exists on the live path; `mx.eval(inputs, ...)` at generate.py:1639 and
`eager_detach_caches` at :1650 actively prevent it) and by measurement
(§3.3 — production (a) matches donation-maximally-enabled (c) to within
0.011 ms/token at both depths).

**Donation is RULED OUT as a source of the ~3.5–4.2 ms/token residual.** Worker
C's "+6.35 ms/token upper bound" is not being realised in production, and this
worker's job was to say so plainly rather than cite it. Said plainly.

**However, the probe that produced that upper bound was measuring the right
neighbourhood with the wrong class.** The real production class,
`BatchPoolingCache`, carries an *unconditional* O(P·D) pool copy per flush that
donation was never able to address, and that Worker C's `PoolingCache`-based
bench structurally could not see (256-entry chunked growth means the equivalent
concat fires 1 flush in 256 there, vs every flush in production).

---

## 5. Q2 — ADDITIVITY vs Worker C's kernel delta and B1's live anchor

### 5.1 Overlap analysis

Worker C's headline attention-kernel delta (**+2.56 ms/token**, range
+2.56–+3.34) was measured with `PoolingCache`, using per-step `mx.async_eval`
("donation able to work"). Question: does C's +2.56 already contain this
worker's +1.95?

**No — and here is why, from C's own numbers.** C's §4.3 pool-write probe was
explicitly excluded from the headline; C wrote that "in the whole-layer async
bench the same write costs far less, which is why the whole-layer delta is
+2.56 ms and not the +7 ms this probe would imply." C's whole-layer bench used
`PoolingCache`, whose growth is `step=256`-chunked: over C's 256-step window at
ratio 4 there are 64 flushes but only **~1 storage growth**, so the growth cost
is amortized 1/64 in C's number. Production pays it **64/64**.

Concretely: C's r=4 layer at 352.6K was 0.5403 ms/step, i.e. 11.35 ms/token
across 21 layers. This worker measures the flush excess alone at
+10.0 ms/flush-step (+2.50 amortized). If C's bench had been paying the
production concat every flush, C's r=4 number would have been ~+0.48 ms/step
higher — a ~+10 ms/token difference in the 43-layer total, which C did not
observe. **C's +2.56 does not contain the production concat cost.**

So the two are **structurally disjoint**:

- C measured: indexer GEMM + top-k + gather + SDPA growth, with a 1-in-256
  amortized pool growth.
- C3 measures: the 64-in-64 pool concat, with the kernels held constant across
  configs (they are identical in (a) and (d) — only the concat changes).

### 5.2 The arithmetic — and the overshoot

| component | Δ 100K → 352.6K (ms/token) |
|---|---|
| Worker C, attention kernels (headline, async) | **+2.56** (range +2.56 … +3.34) |
| Worker C3, `BatchPoolingCache` concat, production-measured, (a)−(d) at each depth | **+1.95** |
| **sum** | **+4.51** (range +4.51 … +5.29) |
| B1 live measured | **+6.80** |
| remaining residual | **+2.29** (range +1.51 … +2.29) |

Derivation of the +1.95: config (a) − config (d) is the pure concat cost.
At 100K: 0.557 − 0.019 = **+0.538 ms/tok**. At 352.6K: 2.504 − 0.055 =
**+2.449 ms/tok**. Depth delta = 2.449 − 0.538 = **+1.911 ms/tok**
(median-of-3-runs form: +1.947 − 0.036 = +1.911; both agree to 0.04).

**Does the sum overshoot the live +6.80? No — it lands at +4.51, comfortably
under.** There is no double-count signal. But this is a *qualified* no:

**The honest caveat that keeps this from being oversold**: C's bench used
`PoolingCache`, so it is not certain that C's number is entirely concat-free at
the margins (C ran 256 steps per point; a single growth event inside that
window contributes ~1/256 of a full-pool copy, which is ~0.04 ms/token at
352.6K — negligible, so the disjointness claim holds).

The larger qualification is in the other direction: **C's bench was ALSO
measuring the wrong cache class**, so C's absolute attention-path total is
itself understated relative to production by roughly this worker's +2.45
ms/token at 352.6K. Read together, C + C3 now explain **+4.51 of the live
+6.80 (66%)**, up from C's 38–49% alone, and the unexplained residual shrinks
from ~3.5–4.2 to **~1.5–2.3 ms/token**.

### 5.3 Where the (now smaller) residual can still live

Unchanged from C's list, minus the donation candidate which is now closed:

1. MoE all_sum arrival skew (43 collectives/token amplifying per-layer skew).
2. Inter-layer pipelining loss at depth — this harness DOES run all 43 blocks
   back-to-back, so it captures more of this than C's per-class bench did, but
   with a stub MoE the inter-layer working set is wrong.
3. Unified-memory / allocator pressure in the real 85 GB-resident regime — this
   harness peaked at 12.9 GB.
4. **MoE-at-depth itself** — completely uncovered here by construction.

---

## 6. Verdict

1. **Donation does not fail in production.** (a) ≈ (c) within 0.011 ms/token at
   both depths, across 3 runs. Negative result, stated plainly. The
   "+6.35 ms/token upper bound" is not realised.
2. **The donation hypothesis was adjacent to a real and bigger mechanism.**
   Production's `BatchPoolingCache` reallocates and copies the entire O(P·D)
   pool on EVERY flush via `mx.concatenate` (`cache.py:1899-1903`) — a cost
   donation structurally cannot address, and one that Worker C's
   `PoolingCache`-based bench could not see.
3. **Real per-token depth cost of it: +1.91 ms/token** over 100K → 352.6K
   (+0.54 → +2.45 ms/token absolute). Suppressing it removes 98% of the pool
   cost and 10.6× of the per-step transient allocation.
4. **Additive with C's kernel delta, not overlapping.** C (+2.56) + C3 (+1.91)
   = +4.47…+4.51, vs live +6.80. No overshoot, no double-count. Residual
   narrows to ~1.5–2.3 ms/token.
5. **This is a live, fixable lever** — see §8.

---

## 7. Honest limitations

- **MoE is a stub.** Absolute ms/token here (39–42) is NOT a production
  per-token figure and is never used as one. All conclusions rest on
  *differences* between configs and depths, in which the stub cancels exactly.
  **Residual components living in MoE-at-depth or in collective interplay are
  NOT covered by this harness** and are not claimed either way.
- **No live cluster A/B was possible.** The model instance was down
  (`instances:[]`) for the whole of this work and the user forbade relaunch. No
  runner was touched, no model loaded/unloaded, no `start_cluster.sh` run.
  Everything here is a harness result on production silicon, not a live
  measurement. **The +1.91 ms/token is a harness number and must be confirmed
  live** (§8).
- **Single-process, single-rank, B=1.** Attention is replicated per-rank in
  production (verified R1) so per-rank attention shapes are right, but nothing
  here sees cross-rank skew or the collective.
- **Random weights.** Irrelevant for a memcpy-shaped cost; possibly relevant
  for the data-dependent exact-topk kernel, which is held constant across
  configs anyway.
- **Idle-node, 12.9 GB peak.** Production decodes under ~85 GB residency; the
  allocator regime is different and could make the concat *worse*, not better.
- **The flush-phase split assumes a clean mod-4 cadence.** It is clean here
  (visible in the raw series). Under production's real prefill remainder the
  phase offset differs, but the amortization is unchanged.
- **B1's live anchors are n=1 per depth**, as B1 itself flags.
- **Config (d) is a measurement instrument, not a proposed patch.** Pre-padding
  a live pool by 192 entries is safe here because the harness controls the run
  length. A real fix must size growth properly (§8).
- **Nothing was committed to git.**

---

## 8. Next attack vector — the LIVE A/B to run when relaunch is authorized

### 8.1 The experiment

The clean live A/B is a **one-line code change** to `BatchPoolingCache`, making
its growth `step`-chunked exactly like `PoolingCache` already is.

**Change** (`mlx-lm/mlx_lm/models/cache.py:1899-1903`), from:

```python
            if self.pooled.shape[1] < max_pool:
                pad = mx.zeros(
                    (B, max_pool - self.pooled.shape[1], D), dtype=px.dtype
                )
                self.pooled = mx.concatenate([self.pooled, pad], axis=1)
```

to a step-chunked grow gated by a new env var so it is a true A/B:

```python
            if self.pooled.shape[1] < max_pool:
                _grow_step = int(os.environ.get("EXO_DSV4_POOL_GROW_STEP", "1"))
                _target = max_pool if _grow_step <= 1 else (
                    ((max_pool + _grow_step - 1) // _grow_step) * _grow_step
                )
                pad = mx.zeros(
                    (B, _target - self.pooled.shape[1], D), dtype=px.dtype
                )
                self.pooled = mx.concatenate([self.pooled, pad], axis=1)
```

`EXO_DSV4_POOL_GROW_STEP=1` (default) = **exactly today's behaviour, bit-for-bit**
— the A arm. `EXO_DSV4_POOL_GROW_STEP=256` = the B arm (matching
`PoolingCache.step = 256`).

Correctness note: growing the pool *larger* than `max_pool` is already safe by
construction — `make_mask` clamps the mask width to `self._visible_width`
(`cache.py:2174-2176`), and `_visible_width` is set from the returned tensor at
`cache.py:1920`, so extra trailing capacity is masked out exactly as the
existing deferred-slot slack is. This is the same invariant
`PoolingCache` relies on for its own 256-chunked storage.

### 8.2 Exact expected signature

Run `bench/p3_depth_anchor_probe.py` (B1's probe, EOS genuinely banned via
`/bench/chat/completions`) at both depths, both arms:

| depth | arm A (`GROW_STEP=1`) expected | arm B (`GROW_STEP=256`) expected |
|---|---|---|
| 100,026 | 35.79 ms/tok (B1 anchor) | **~35.25 ms/tok** (−0.54) |
| 352,599 | 42.59 ms/tok (B1 anchor) | **~40.14 ms/tok** (−2.45) |
| **depth delta** | **+6.80** | **~+4.89** (−1.91) |

Throughput form: **23.48 → ~24.91 tok/s at 352.6K (+6.1%)**; 27.94 → ~28.37
tok/s at 100K (+1.5%). The **asymmetry is the diagnostic** — a change that
helps the deep point ~4.5× more than the shallow one is the fingerprint of an
O(P·D) per-flush cost being removed. A uniform improvement at both depths would
mean something else changed.

**Secondary signature, cheaper to check first**: with `MLX_GPU_TIME=1`, the
per-step `[GPU_TIME]` line's `wall` should lose its mod-4 spike at 352.6K. Even
cheaper: B1's probe already dumps the full inter-token gap distribution — in
arm B the **p90 should collapse toward p50** at 352.6K (B1 measured p50 39.16 /
p90 61.94; expect p90 to drop by roughly 9–10 ms while p50 drops ~2.4).

**Falsification condition, stated up front**: if arm B shows **no** change at
352.6K, this mechanism is not on the live critical path (most likely because
the real MoE + all_sum cost hides it behind other work) and this worker's
+1.91 ms/token should be treated as a harness-only artifact of the stub-MoE
schedule. That is a real possible outcome and should be reported as such.

### 8.3 Cost

One code edit, one cluster relaunch, then 4 probe runs (2 depths × 2 arms).
The deep points cost ~19 minutes each (~17.6 of it prefill), so ~45 min of
cluster time total after relaunch.

---

## 9. Files

- `bench/p3_donation_insitu_harness.py` — new, additive. The harness. No
  existing bench script modified.
- `/tmp/p3c3_all2.log` (configs a/b/c), `/tmp/p3c3_d2.log` (configs d/e),
  `/tmp/p3c3_rep.log` (reps 2–3) on `adams-mac-studio-m4-2` — raw stdout,
  reproduced verbatim above.
- `/tmp/p3c3_{a,b,c,d,e}_{100026,352599}.json` — machine-readable per-step data
  including every step's ms and allocator deltas.
- Nothing under `~/repos/exo` on either studio was created, edited, or deleted.
  No runner process was touched. Nothing was committed to git.

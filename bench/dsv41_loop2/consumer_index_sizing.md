# Consumer-layer coarse-pass waste — sizing + go/no-go + correctness audit

**Scope:** the four CONSUMER index layers (24, 28, 32, 36) of the DSv4.1 M2
hierarchical indexer. They pay a **full-width** (`nb`-column) coarse score pass
and then throw away every non-candidate column. Read-only static analysis;
all line refs are to the laptop checkout `~/repos/exo/mlx-lm`.

**Verdict: GO.** Pessimistic (double-haircut) bound is **14.8% of per-chunk wall
at 750K**, **12.2% at 350K**, **3.9% at 100K** — all far above the 2% gate. The
restriction is **provably exactly equivalent** to the current score-all-then-mask,
given one condition that already holds (`HIER_BLOCK == candidate_block_size == 8`).

Config constants used (verified): `n=2048` (chunk, baked default), `b=1`,
`h=index_n_heads=32`, `d=index_head_dim=128`, `block=candidate_block_size=8`,
`candidate_topk_blocks=2048` → **C = 2048·8 = 16384 candidate columns**,
`k=index_topk=512`, `overfetch=16` → `(k+of)·block = 4224` columns re-scored per
layer. Ratios (`compress_ratios[2,8,14]=2`, `[20,24,28,32,36]=1`).
Refs: `config.py:63-84`, `indexer.py:123-133`, `indexer_hierarchical.py:145-148`,
`docs/dsv41-hierarchical-indexer-design-2026-10-04.md:35`,
`docs/benchmarks/.../p7c.out` (compress_ratios census), `PERFORMANCE_HISTORY.md:10886` (chunk 2048).

---

## 1. Code walkthrough — the consumer coarse pass

**Routing.** `Indexer.__call__` computes the consumer mask and dispatches:

* `indexer.py:545-546` — `cmask = shared.candidates if (self.uses_candidates and
  shared.candidates is not None) else None`. `uses_candidates` is
  `0 <= candidate_source_layer < layer_id` (`indexer.py:466`), i.e. layers 24–36.
* `indexer.py:553-558` — `hierarchical_topk_prod(..., cand_mask=cmask, cand_src=…)`.
  `cand_src` is `None` for consumers (only layer 20 publishes).

**The coarse sweep (where the waste is).** `indexer_hierarchical.py:455-457`
calls `coarse_block_scores(..., col_mask=cand_mask)`. Its body:

* `indexer_hierarchical.py:241` — `for c0 in range(0, nb, st)`: the loop bound is
  **`nb` — every compressed column**, unconditionally; `col_mask` never enters the
  loop control.
* `:245` — `s = _score_shared_columns(qc, keys, wc)`: `einsum("bshd,btd->bsht")`,
  i.e. `[b,n,h,t]`; dominant cost `2·b·n·h·t·d` MACs (`_score_shared_columns`,
  `:167-176`). Every column of the strip is scored.
* `:247-250` — visibility then masking, **after** the score:
  `vis = cols < lens`; `if col_mask is not None: vis = vis & col_mask[:,:,c0:c1]`;
  `s = mx.where(vis, s, -inf)`.
* `:253-260` — reduce each `block`-wide group to a per-block max and scatter into
  `block_maxima[:,:,b0:b0+nt]`.

So for a consumer layer at offset `E`, the coarse pass computes `b·n·nb·h·d`
MACs (all `nb` columns, bf16) and discards the `nb−C` non-candidate columns at
the `where` on `:250`. Only `C=16384` columns can survive the mask.

**Consumers of the coarse output** — only `block_maxima`:

* `indexer_hierarchical.py:473-474` — `top_blocks(bm, k+overfetch)`: `argpartition`
  the `[b,n,nb_blocks]` maxima for the best 528 blocks.
* `:475-477` — `exact_rescore_streaming(..., col_mask=cand_mask)`: fp32 re-score of
  **only those 528 blocks' columns** (4224 cols) with a running top-k.

**Why the mask is block-granular** (this is what makes the fix exact):
`candidate_mask_from_block_maxima` (`:287-320`) ends `mx.repeat(keep, block,
axis=-1)[..., :nb]` (`:320`) — the mask is **constant within each `block`-wide
group**. It is published by layer 20 as `shared.candidates = blk`
(`indexer.py:560`), width `[b,n,nb]` (`:466`). The coarse block and the candidate
block are asserted equal on the source layer (`indexer.py:539-544`).

**The waste, quantified statically:** the coarse pass evaluates `nb` columns per
consumer layer to produce `nb/block` block maxima, of which at most
`C/block = 2048` can be finite (a non-candidate block is `-inf` after `:250`).
At 350K that is `2048/43750 = 4.7%` of the block maxima being reachable.

---

## 2. Quantitative sizing

`n=2048, h=32, d=128, C=16384`, `k+of=528`, `block=8`, 4 consumer layers.

### 2a. Raw FLOPs (per chunk, `b=1`)

```
consumer coarse FLOPs = 4 · (2·n·nb·h·d) = 8·n·nb·h·d
restricted minimum    = 4 · (2·n·C ·h·d) = 8·n·C ·h·d      (constant)
```

| offset E | nb | consumer coarse FLOPs | restricted min | ratio | waste (nb−C)/nb |
|---|---|---|---|---|---|
| 100K | 100,000 | 6.71e12 | 1.10e12 | 6.1× | 83.6% |
| 350K | 350,000 | 2.35e13 | 1.10e12 | 21.4× | 95.3% |
| 750K | 750,000 | 5.03e13 | 1.10e12 | 45.8× | 97.8% |

### 2b. % of wall (FLOPs → time via a column model, anchored on the measured span)

Units = one fp32 full-width column-scoring. Coarse runs bf16, costing
`rbf ≈ 0.5` of an fp32 column on this hardware (central; 0.7 pessimistic).

Coarse columns per chunk, ratio-1-equivalent:
`3·(E/2) + 1·E + 4·E = 6.5·E` (layers 2/8/14 at ratio 2, layer 20, the four consumers).
Exact-pass columns per chunk: `8 · 4224 = 33,792` (fp32). So

```
f_cons = rbf·(4·E) / [ rbf·(6.5·E) + 33792 ]      (consumer-coarse share of the indexer span)
```

and the saving as a fraction of wall = `f_cons · (nb−C)/nb · S`, where `S` is the
measured indexer-score share of the per-chunk wall.

Measured `S` (pre-M2, from the campaign — see `PERFORMANCE_HISTORY.md`):
100K = **16.9% → 29.5%** mid→late (`:10945,10957`); 350K = **42.9% / 44.2%**
(`:10457`, both nodes); 750K = not directly measured — **extrapolated to
50–65%** (monotone depth trend 14.6%→44.2% over 350K, `:10453`).

| offset E | f_cons (rbf=.5) | S range | **% of wall, pessimistic → optimistic** | double-haircut (½·f_cons) |
|---|---|---|---|---|
| 100K | 0.557 | 0.169–0.295 | **7.9% → 13.7%** | 3.9% → 6.9% |
| 350K | 0.598 | 0.429–0.442 | **24.4% → 25.2%** | 12.2% → 12.6% |
| 750K | 0.607 | 0.55–0.65 | **29.7% → 38.6%** | 14.8% → 19.3% |

**Bounds.** Optimistic = high `S`, `rbf=0.5`. **Pessimistic = low `S` and, as a
stress test, halving `f_cons`** (crediting half the span to the memory-bound
exact gather / launch overhead / host syncs rather than to coarse FLOPs — the
conservative reading of a bf16 GEMM that may not hit 2× fp32 here). Even under
that double haircut the deep numbers are **14.8% (750K) and 12.2% (350K)**.

Cross-check vs the naive "pre-M2 wall" convention (`S·f_cons·(nb−C)/nb`):
7.9–13.7% / 24.4–25.2% / 29.7–38.6%. Consistent.

**Deep weighting (750K):** the two deep-columns of the table are 29.7% / 14.8%
(haircut) — the offset that dominates a long-context prefill and the one the
campaign is optimising for. Verdict unchanged.

### 2c. Gate

Threshold = 2% of wall. Pessimistic bound: **3.9% @100K, 12.2% @350K, 14.8% @750K**.
→ **GO**, all three offsets clear the gate; the deep offsets clear it by ~7×.

---

## 3. Correctness audit — is candidate-restricted coarse **exactly** equivalent?

**Claim: YES — elementwise-identical, given block-alignment.**

Let a consumer layer's coarse output be `m : [b,n,NB]`, `NB=ceil(nb/block)`.

*Current path* (`coarse_block_scores`, `:241-261`): for each block `g`,
`m[g] = max_{j∈block g} ( col_mask[j] ? score(j) : −inf )`, with `score(j)=−inf`
when `j ≥ lens`. A non-candidate block has every column masked → `m[g] = −inf`.

*Restricted path*: score only candidate blocks' columns, set every non-candidate
block to `−inf`, same reduction.

**Proof.** For a non-candidate block, current `m[g]=−inf` (all columns masked at
`:250`) and restricted `m[g]=−inf` by construction — equal. For a candidate block
`g`, the current max is taken over exactly its candidate, visible columns (the
`col_mask` AND at `:249` zeroes the rest to `−inf`, which cannot win a max); the
restricted path scores exactly those columns with the *same* expression
(`_score_shared_columns`, `:167-176`, invoked identically) and reduces with the
same visible test. Hence `m` is **elementwise identical**.

`top_blocks` (`:268-284`) is a pure function of `m`, so it selects the **same**
528 blocks. `exact_rescore_streaming` (`:326-384`) then re-scores those blocks
with the same `col_mask`, so `(top_v, top_i)` are **identical**. The only
degree of freedom — `argpartition` tie-breaking among equal block maxima — is
fed the identical input array in both paths, so it resolves identically. ∎

**The one condition that must hold:** `block_coarse == candidate_block_size`.
Because the mask is constant within each `block`-wide group (`:320`) and the
coarse reduces over `block`-wide groups, a coarse block is either fully candidate
or fully non-candidate. If `HIER_BLOCK ≠ candidate_block_size`, a coarse block
would straddle candidate and non-candidate 8-blocks and the equivalence breaks.

*Status:* this holds today — `HIER_BLOCK=8` (`indexer.py:123`),
`candidate_block_size=8` (release; `pB_indexer_test.py:42-43`), asserted on the
source layer (`indexer.py:539-544`). **Gap to flag:** there is **no assert for
consumer layers**; the implementation must either add one
(`assert hier_block == self.candidate_block_size` for `uses_candidates`) or read
the candidate block size off the mask. Nothing else is needed — the lens
visibility test must be kept (a candidate block with no visible column must stay
`−inf`), which the restricted pass does naturally.

**Second, independent subtlety — shared index space.** Consumers and the source
share the layer-20 key cache with ratio 1, so their `nb` is identical and the
`[b,n,nb]` mask indexes their own column space directly (no rescaling). Verified:
`attention.py:183` slices `shared.index_src_cache.index_k[:bsz,:compress_len]`
with `compress_len = end_pos//ratio`; layers 20/24/28/32/36 all have ratio 1.

---

## 4. Implementation sketch — GO

Seed the consumer coarse pass from the block indices of `shared.candidates` and
score only those blocks. Mirror the existing per-row gather in
`exact_rescore_streaming` (`:355-379`).

1. **Derive candidate block ids** from the block-aligned mask:
   `blk = col_mask[:, :, :NB*block].reshape(b, n, NB, block)[..., 0]` → `[b,n,NB]`
   bool (block-level keep). Gather the `≤2048` candidate block indices per row
   (`argwhere`/`argsort(~blk)`), padded to a fixed `Cblk=2048` with a count.
2. **New `coarse_block_scores_candidates(q, index_k, w, lens, cand_blocks, strip, dtype)`**:
   stream the candidate blocks in strips (`bp = strip//block` blocks), build the
   global column ids `bid*block + arange(block)`, gather keys via `_gather_rows`
   (`:189-201`), score bf16 with `_score_gathered_columns` (`:179-186`), apply
   `vis = (gcols < lens)`, reduce each block to a max, and **scatter** into
   `block_maxima` at the candidate block positions; leave the rest `−inf`.
3. **Wire it** in `hierarchical_topk_prod` (`:453-457`): when `cand_mask is not
   None`, take the restricted path; else the existing full sweep. `top_blocks` +
   `exact_rescore_streaming` (`:468-477`) are unchanged.

**Cost after:** consumer coarse drops from `nb` to `C=16384` columns per row
(with a per-row gather), i.e. `~6.1× / 21× / 46×` less work at 100K/350K/750K.

**Effort:** low. ~50–100 LOC, reusing `_gather_rows` / `_score_gathered_columns`
verbatim; one parity test (`restricted maxima == score-all-then-mask`,
elementwise, at random block-aligned masks) + recall. **≈0.5–1 day** incl. tests.
Risk: low — pure work reduction behind the existing `DSV41_INDEXER_HIER` gate,
and the equivalence is a proof (no recall loss by construction).

**Variant (better, not bit-equivalent):** for consumers, do a *single* fp32 pass
over the candidate columns and use it for **both** the block-maxima selection and
the top-k, dropping the separate bf16 coarse + fp32 re-score duplication.
Consumer score work → `C` fp32 columns (vs `nb` bf16 + `C` fp32). Strictly more
accurate (fp32 block maxima) but not bit-identical to today, so it rides the same
live-battery gate class. Consider only if the exact-equivalent path is already
in.

**No-go alternative:** n/a — the numbers clear the gate at every offset.

---

## References

* `mlx_lm/models/deepseek_v41/indexer_hierarchical.py:167-176, 189-201, 207-262,
  268-284, 287-320, 326-384, 414-478`
* `mlx_lm/models/deepseek_v41/indexer.py:114-133, 459-477, 526-562`
* `mlx_lm/docs/dsv41-hierarchical-indexer-design-2026-10-04.md:32-119`
* `mlx_lm/docs/dsv41-hierarchical-indexer-m2-integration-2026-10-05.md:23-43`
* `docs/PERFORMANCE_HISTORY.md:10448-10457, 10526, 10643, 10886, 10940-10963`
* `mlx-lm/benchmarks/dsv41/pB_indexer_test.py:38-43`

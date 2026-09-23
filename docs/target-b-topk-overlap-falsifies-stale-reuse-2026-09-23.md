# TARGET (b) DIAGNOSTIC: top-k overlap FALSIFIES stale-reuse at depth (2026-09-23)

## The measurement

`EXO_DSV4_TOPK_OVERLAP_LOG=1` enabled, 2 x ~275K requests, 1,323 decode-step
samples of consecutive-step top-k Jaccard overlap (ratio-4 indexer layers).

Split by **pool_size** (the decisive cut — see below):

| pool_size bucket | n | mean jaccard | frac(jaccard >= 0.90) |
|---|---|---|---|
| shallow (<512) | 1,113 | **0.9708** | 0.887 |
| deep (>=2000) | 210 | **0.4183** | **0.000** |

## Verdict: NOT viable

At the depth that matters for target (b) — pool_size >= 2000, i.e. 250K+ context —
consecutive decode steps share only **42%** of their selected top-k set, and
**not one** of 210 steps reached 0.90 overlap. Stale top-k reuse (rescore every N
steps, cheap incremental scoring between) would therefore be recomputing a
genuinely different selection most steps. **The highest-value remaining lever for
the 250K target is dead.**

## Two errors of mine this diagnostic exposed (recorded for honesty)

1. **I read the wrong aggregate first.** My probe sampled `tail -400` of the log
   and reported mean 0.684 / median 0.708, which I nearly published as the
   verdict. The full log (n=1323) gives a *bimodal* picture (p50=1.0000, 74.6%
   >= 0.90) — also misleading as a single number.
2. **I then guessed the wrong direction from the bimodality.** I hypothesised
   shallow=noisy, deep=stable, and was about to write that reuse "IS promising at
   depth". The pool_size split shows the exact opposite: shallow is the stable
   regime (0.97) and **deep is the unstable one (0.42)**. I caught this only
   because I went back and computed the split instead of trusting the guess.

The lesson matches the rest of this session: **the aggregate number is not the
answer; compute the cut that the mechanism actually depends on.** Here the
mechanism depends on pool_size (selectivity), and that cut reversed the
conclusion twice.

## Physical interpretation

- **Shallow (pool < 512):** `k = min(index_topk, pooled) = pooled` — every entry
  is selected, so the "set" is trivially stable and only rotation at the margin
  moves jaccard. High overlap is an artifact of triviality, not of stability.
- **Deep (pool >= 2000):** selection is genuinely selective (512 of 2000+), and
  the chosen subset genuinely reshuffles step to step. Selection is a real
  ranking over many near-tied candidates, so small hidden-state changes flip
  many picks at once.

That second property also explains why the per-cycle indexer cost is hard to
amortise: the selection is both expensive (O(context)) and genuinely
non-repetitive.

## Where target (b) stands after this

Decode at 250K measured again this boot: **28.99 and 27.70 t/s** at 276,427 and
273,626 depth (needle OK both). Still short of the 30 t/s bar.

Levers now closed (measured, not assumed):
- `EXO_DSV4_FENCE_ASYNC=1` — the biggest known decode lever (28.9 -> 37.0 t/s in
  its own A/B) is **already live**. No unclaimed win.
- stale top-k reuse — **falsified above** (0.42 overlap at depth).
- `EXO_DSV4_INDEX_TOPK < 512` — FORBIDDEN (skill #49, quality regression).
- `FENCE_EVERY_N_LAYERS` 4->8 — worth only ~+0.7 t/s and costs c=2 bistability.

So the honest position on (b): the gap is +22.9% ms/cycle, it is now *isolated*
to per-cycle depth-scaling work, every identified lever is either already active,
forbidden, tiny, or now falsified. The remaining unknown is which specific
per-cycle term (indexer scoring vs SDPA over the pooled set vs the compressor)
carries the +22.9% — that needs a span profile at depth, not another hypothesis.

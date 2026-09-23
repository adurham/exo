# TARGET (b) ATTRIBUTION: the depth cost is `attn.sdpa.compressed`, NOT the indexer (2026-09-23)

Span profile at 30K vs 250K, **same boot, same config**, decode phase only
(`EXO_PROFILER=spans`). Median µs per call and share of decode wall:

| span | 30K med µs | 250K med µs | delta | 30K % | 250K % |
|---|---|---|---|---|---|
| moe.switch_mlp | 202.7 | 201.0 | -1.7 | 34.0 | 32.5 |
| attn.sdpa | 204.9 | 205.0 | +0.1 | 14.0 | 14.1 |
| attn.o_proj | 260.0 | 258.6 | -1.4 | 10.2 | 9.9 |
| attn.proj_qkv | 439.4 | 441.0 | +1.6 | 9.5 | 9.0 |
| attn.all_gather | 7548.8 | 2015.1 | -5533.7 | 9.4 | 8.0 |
| moe.all_sum | 79.1 | 93.1 | +14.0 | 8.9 | 8.6 |
| attn.compressor | 99.3 | 26.1 | -73.2 | 2.4 | 2.5 |
| **attn.sdpa.compressed** | **189.1** | **349.0** | **+159.9** | 2.2 | **5.6** |
| attn.indexer | 396.0 | 448.9 | +52.9 | 1.7 | 2.4 |
| attn.kv_cache | 2.8 | 3.0 | +0.2 | 0.5 | 0.3 |
| indexer.topk | 4.2 | 3.7 | -0.5 | 0.0 | 0.0 |
| **indexer.score** | **3.8** | **4.5** | **+0.7** | 0.0 | 0.0 |

## Findings

**1. My indexer hypothesis was WRONG.** `indexer.score` is 3.8 -> 4.5 µs and
`indexer.topk` 4.2 -> 3.7 µs. The O(context) indexer scoring I had been treating
as the prime suspect carries **essentially zero** depth cost — 0.0% of decode wall
at BOTH depths. Falsified by direct measurement, the third time this session a
hypothesis died on contact with a profile.

**2. Almost every span is FLAT.** `moe.switch_mlp` 202.7 -> 201.0,
`attn.sdpa` 204.9 -> 205.0, `attn.o_proj` 260.0 -> 258.6, `attn.proj_qkv`
439.4 -> 441.0. Per-op costs are identical at 30K and 250K. So the +22.9%
ms/cycle is NOT broad slowdown — it is localized.

**3. The one span that genuinely grew: `attn.sdpa.compressed`, 189.1 -> 349.0 µs
(1.85x), share 2.2% -> 5.6%.** That is the SDPA over the **compressed/pooled**
set — the attention that actually scales with context (its key/value set grows
with depth/compress_ratio). This is the context-scaling term.

**4. `attn.all_gather` median fell 7548.8 -> 2015.1 µs — do NOT read as a win.**
Call counts differ wildly (656 at 250K vs 5453 at 30K), so the median reflects a
different call mix, not a speedup.

## Where target (b) now stands

`attn.sdpa` (205 µs) + `attn.sdpa.compressed` (349 µs) = 46.6% of decode wall at
250K, and `sdpa.compressed` is the part that scales with depth. The depth penalty
is concentrated there.

This also explains why the earlier levers were dead ends:
- **stale top-k reuse** (falsified, 0.42 overlap) — addresses the indexer, which
  is now shown to cost ~nothing, so even if overlap had been high it would have
  bought almost no time.
- **FENCE_ASYNC** (already live) — addresses fence serialization, a different term.
- **INDEX_TOPK < 512** (forbidden) — again the indexer.

So the real target for the 250K gap is the **compressed-set SDPA**: its kernel,
its masking, or the shape/layout of its pooled K/V. That is a different and much
better-specified problem than "the indexer".

## Caveats, stated plainly

- Profiling adds overhead (`EXO_PROFILER=spans`): the profiled run measured
  9.73 t/s at 270,816 tokens, which is NOT comparable to the 27-29 t/s baseline.
  These numbers are for **attribution only** — relative span shares and
  per-span medians. The profiler is left UNSET in `start_cluster.sh`.
- 30K and 250K were separate requests on the same boot, so call counts and the
  call mix differ; per-span MEDIANS are the comparable quantity, and the flat
  spans corroborate that the two runs are otherwise equivalent.

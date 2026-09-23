# TARGET (b) ATTRIBUTION — VALIDATED under sync-mode profiling (2026-09-23)

## Why this re-run

The prior attribution used `EXO_PROFILER=spans` **without** `SYNC_SPANS`, and a
prior investigation (`docs/dsv4-4096-regression-root-cause-2026-08-19.md`) warns
explicitly that non-sync span profiling lets MLX's lazy-eval work "leak" between
spans — the exact artifact that produced a bogus 3.15x SDPA ratio once before.
That doc's final word: only GPU-synced measurement is trustworthy for span
costs, and under sync mode SDPA is confirmed linear in query-row count.

So the attribution was re-measured with `EXO_PROFILER=spans` +
`EXO_PROFILER_SYNC_SPANS=1`, which forces a GPU sync at every span boundary.
(The 1800s hang timeout required for sync mode per that doc is now satisfied by
the Pipeline guard in `start_cluster.sh`.)

## Sync-mode result at ~279K decode

| span | median µs | share |
|---|---|---|
| moe.switch_mlp | 230.75 | 32.3% |
| attn.sdpa | 255.21 | 14.1% |
| attn.o_proj | 287.50 | 10.0% |
| attn.proj_qkv | 460.79 | 9.0% |
| moe.all_sum | 112.71 | 8.5% |
| attn.all_gather | 2782.71 | 8.0% |
| **attn.sdpa.compressed** | **369.96** | **5.8%** |
| attn.compressor | 117.58 | 2.5% |
| attn.indexer | 460.83 | 2.4% |
| attn.kv_cache | 16.96 | 0.3% |
| indexer.topk | 18.08 | 0.0% |
| **indexer.score** | **19.88** | **0.0%** |

## Verdict: the attribution HOLDS

Under artifact-resistant sync-mode measurement the conclusion is unchanged:

1. **`indexer.score` is 19.88 µs / 0.0% share** — the O(context) indexer scoring
   I originally suspected remains a non-factor. Confirmed, not a lazy-eval
   artifact.
2. **`attn.sdpa.compressed` is the 3rd-largest attention term at 369.96 µs /
   5.8%** — larger than the plain `attn.sdpa` per call (255.21 µs), and it is
   the SDPA whose key/value set scales with context depth. This is the
   context-scaling term.
3. The general picture is stable: MoE switch_mlp dominates at ~32%, attention
   splits across sdpa (14.1%) + sdpa.compressed (5.8%), and the indexer is
   negligible.

## What changed between the two profiles (honest note)

Absolute medians differ modestly between the non-sync and sync runs
(sdpa.compressed 349.0 -> 369.96 µs; sdpa 205.0 -> 255.21 µs) — expected, since
sync mode attributes lazy work to the span that forces the sync. The **relative
ordering and the 0.0% indexer share are identical**, which is the claim being
made. I am not comparing the non-sync and sync absolute numbers against each
other.

## Consequence for target (b)

Target (b) is now specified as: **the compressed-set SDPA path**
(`attn.sdpa.compressed` — its kernel, masking, or pooled K/V shape/layout),
NOT the indexer. All previously-tried levers are closed by measurement:
- stale top-k reuse — falsified (0.42 overlap at depth) AND now doubly moot
  (the indexer costs ~0.0% anyway)
- `EXO_DSV4_INDEX_TOPK < 512` — forbidden (#49), and moot for the same reason
- `EXO_DSV4_FENCE_ASYNC` — already live
- `EXO_DSV4_SPARSE_SDPA_TILE=128` — already set

Next concrete step for a future session: read the compressed-SDPA call site and
check for avoidable materialisation (mask construction, pooled-buffer copies,
dtype/shape layout) rather than sweeping more knobs. The prior doc already closed
SDPA *sub-tiling* as a lever (isolated microbench 0.998-1.047x tiling ratio), so
the remaining question is mask/layout/copy overhead, not kernel shape.

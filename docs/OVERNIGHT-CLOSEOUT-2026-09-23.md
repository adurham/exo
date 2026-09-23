# Overnight close-out 2026-09-23 — what shipped, what's verified, what's open

## SHIPPED (committed + pushed to origin/main)

1. **`89ebdbff0` — the real bug.** `_cache_object_nbytes()` (the leaf sizer behind
   `_total_bytes()`, which the byte-cap eviction branch tests) hand-rolled an
   attribute walk over `("keys", "values", "state")`. Every DSv4 sparse layer is
   a `CacheList`, which exposes no `.keys`/`.values` and whose `.state` returns a
   **list of tuples** — so the walk read `.nbytes` off tuples and returned 0 for
   every layer. Measured on real DSv4 shapes at 273,626 tokens: reported
   786,432 B vs actual 2,807,939,072 B — a **~3570x undercount**. The byte cap
   was therefore *structurally unreachable*, which is exactly what the eviction
   log showed (only ever "session cap N"). Fix: prefer the standard `.nbytes`
   property every mlx-lm cache class implements, falling back to the legacy walk.
2. **`1bda1fd62`** — set `DSV4_MAX_PREFIX_BYTES=12 GiB` (the cap value).
3. **`f14977029`, `bd0cd6710`, `c634d4919`** — three successive corrections to my
   own rationale as measurements landed (see "Corrections" below).
4. **`b3564c3f0`** — the 250K decomposition + fence status.

## VERIFIED ON LIVE HARDWARE

- Accounting fix agrees **exactly** with `.nbytes` per layer on real DSv4 shapes.
- A 5 x ~370K sweep accumulated 4 leaves (360,974 / 383,837 / 383,838 / 364,787
  tok) measured at 3.45 / 3.67 / 3.67 / 3.48 GiB.
- Byte cap correctly **SILENT** at 361K (3 leaves = 10.37 GiB < 12 GiB): the 5th
  add logged `session cap 4`, not `byte cap`. So normal depths — including the
  250K target — are operationally untouched. That was the design intent.
- Byte cap binds only above **~420K** (3 x 5.41 = 16.2 GiB at 565K > 12 GiB).
- GPU throttle that contaminated mid-session measurements is **RESOLVED**:
  14.85 TFLOPS sustained for 40 s at 1578 MHz, 100% residency, 55 W (earlier:
  2.27-4.0 TFLOPS at 338 MHz / 2 W). Reboot was NOT needed.
- Cluster left healthy: READY 2/2, pending 0, production env, wired limit 115000.

## TARGET STATUS

**(a) Stop the 500K collapse — root-caused and fixed; frequency reduced, not
eliminated.** Mechanism: at 565K the peak (117.24 GB) exceeds the 115.4 GB wired
limit, so macOS evicts the model's mmap-backed clean weight pages and re-faults
them from SSD (6,599 page-ins in one deep request vs 400 at 128K; pageouts=0 so
not swap; decompressions ~0). Retained prefix-cache leaves are a real
contributing term: they accumulate ratchet-style (measured before-prefill
"active": 87.8 -> 93.8 -> 97.4 -> 100.8 -> 106.1 GB across five deep requests,
one leaf's KV per step). The fix removes one ~5.4 GiB leaf at 565K.
**Not eliminated:** in the same sweep two runs came in at 16.43 and 16.83 t/s
while others sat ~25.5 — the stochastic slow mode still occurs.

**(b) 250K above 30 t/s — NOT MET.** Currently ~26-29 t/s at 250K. Root-caused
as **CYCLE COST, not acceptance**: 30K = 0.928 acc/cycle, 56.5 ms/cycle, 34.10
t/s; 273K = 0.896 acc/cycle (only -3.5%), 69.5 ms/cycle (**+22.9%**), 27.29 t/s.
To reach 30 t/s needs ms/cycle 69.5 -> 63.2 (-9%).

- The single biggest known decode lever (`EXO_DSV4_FENCE_ASYNC=1`: c=1 28.9 ->
  37.0 t/s per its own A/B, byte-identical output) is **already live**. No
  unclaimed win there.
- `FENCE_EVERY_N_LAYERS` 4->8 recovers ~+0.7 t/s (+3%) but costs c=2 bistability
  per the launcher's own note. Does not close the gap; surfaced as an option.
- **Next attack vector:** `EXO_DSV4_TOPK_OVERLAP_LOG=1` (read-only) measures
  consecutive-step top-k Jaccard overlap, testing whether the O(context) indexer
  rescoring every decode step is largely redundant. That is the one per-cycle
  term that grows with depth (~234 pooled entries at 30K vs ~2,135 at 273K).
  `EXO_DSV4_INDEX_TOPK<512` is FORBIDDEN (skill pitfall #49).

## CORRECTIONS I MADE TO MY OWN CLAIMS (the honest ledger)

Several intermediate conclusions were wrong and were corrected in-session by
measurement, each time with the code comment updated so nobody re-reads the bad
rationale:

1. **"Acceptance cliff at depth"** — WRONG. Built on a 46.30 t/s outlier; an n=4
   variance test showed 34.89 +- 0.64 is the real ~30K value and the decline is
   cycle cost.
2. **"Indexer top-k threshold causes a cliff at 65,536 tokens"** — FALSIFIED. A
   5-rung ladder (45K/60K/76K/98K/121K) showed a smooth ~30-34 t/s, no step at
   the predicted crossover. The source check (`k = min(index_topk, pooled)`) was
   real; the behavioural prediction was not.
3. **"The DSpark residual ~6.5-7 GB is shardable"** — WRONG. Sizing the head
   showed it is ~79.4 GB total / ~39.7 GB per rank, of which 97% is expert
   weights that are already sharded. Shardable remainder ≈ 0.12 GB. Lever dead.
4. **"A 275K leaf holds ~0.07 GB so the byte cap is inert"** — WRONG, and this
   one mattered. It was an artifact of my own incomplete ratio arithmetic AND of
   the accounting bug. Real size 2.62 GiB.
5. **"The fix frees ~10.8 GiB at 565K"** — over-stated 2x. Eviction order is
   (1) memory pressure, (2) session cap, (3) byte cap, and the check runs BEFORE
   insertion; the session cap already drops 5->3, so the byte cap removes only
   the 3rd leaf (~5.4 GiB).
6. **"The cap binds above ~300K"** — actually ~420K.

The recurring failure mode: **writing a fix against an assumed size before
measuring it.** Every one of these was caught by measuring the quantity directly.

## FILES

- `src/exo/worker/engines/mlx/cache.py` — the fix.
- `start_cluster.sh` — `DSV4_MAX_PREFIX_BYTES` + full rationale/corrections.
- `docs/ROOTCAUSE-bytcap-undercount-cachelist-2026-09-23.md` — the bug writeup.
- `docs/overnight-250k-cycle-cost-and-fence-status-2026-09-23.md` — target (b).
- Earlier tonight: `overnight-results-two-mechanisms-and-stochastic-collapse`,
  `overnight-final-n6-250k-bistability`, `overnight-ratchet-depth-specific-*`,
  `overnight-corrections-dspark-sizing-and-content-driven-tps`.

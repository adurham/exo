# OVERNIGHT CLOSE-OUT 2026-09-23 — final

Status: cluster healthy (READY 2/2, 0 pending, production env, wired limit
115000 unchanged). Tree committed and pushed on `origin/main` at `edc6b43c4`.

## SHIPPED (all pushed)

| commit | what |
|---|---|
| `89ebdbff0` | **THE BUG**: `_cache_object_nbytes` returned 0 for every `CacheList` layer -> `_total_bytes()` undercounted ~3570x -> the byte-cap eviction branch was structurally unreachable for DSv4. Fixed to use the standard `.nbytes`. |
| `1bda1fd62` | Set `DSV4_MAX_PREFIX_BYTES=12 GiB` (the cap value). |
| `6c3a68792` | Enabled context-adaptive prefill chunk shrink (128 past crossover). |
| `eb352e160` | Crossover 200000 -> 500000 (stop taxing the 250K target). |
| `4a2dad8ac` | Enabled read-only top-k overlap diagnostic. |
| `3fc587ae1` | Top-k overlap FALSIFIES stale-reuse at depth. |
| `97e6dd2f7` | Span attribution: depth cost is `attn.sdpa.compressed`, not the indexer. |
| `edc6b43c4` | Sync-mode profiling VALIDATES that attribution. |
| `d3b1cf001` | **Watchdog root-cause fix**: liveness probe now reads the STACK; flat footprint is not a hang. |
| `756574e6c` | Revert of my own `cb2958e07` (rejected timeout-bump approach). |
| plus 4 doc/rationale corrections | |

## TARGET (a) — stop the 500K collapse

**Root-caused; frequency reduced; NOT eliminated. Stated plainly.**

Mechanism: at 565-638K the peak (117.88-123.80 GB) exceeds the 115.0 GB wired
limit, so macOS evicts the model's mmap-backed clean weight pages and re-faults
them from SSD (39,930-66,003 page-ins vs 400 at 128K; pageouts=0; decompressions
~0). Retained prefix-cache leaves are a real accumulating contributor (ratchet:
before-prefill active 87.8 -> 93.8 -> 97.4 -> 100.8 -> 106.1 GB across five deep
requests, one leaf's KV per step).

Fixed the accounting bug that made the byte cap dead. Verified live: accounting
now agrees exactly with `.nbytes` per layer; 4 leaves measured 3.45/3.67/3.67/
3.48 GiB. The cap is correctly SILENT at 361K (5th add logged `session cap 4`,
not `byte cap`) so normal depths — including the 250K target — are untouched.

**Honest budget at 638K:** ~104 GB fixed + 16.63 GiB retained leaves + ~9 GB
prefill transient = ~130 GB vs 115.0 available. Removing ALL retained leaves
would still leave ~113 GB with near-zero margin. The prefix cache is a
contributor, not the cause.

**Best adaptive-prefill result:** peak 87.87 GB (under the limit by 27.1 GB),
17,010 page-ins, 25.75 t/s at 638K — but glm-5.3's decomposition shows ~19.5 GB
of that is boot state and ~10.5 GB is the chunk change, so ~2/3 boot / ~1/3
chunk. NOT yet isolated by a fresh-boot control run.

## TARGET (b) — 250K above 30 t/s

**NOT met. 28.99 / 27.70 t/s at 276,427 / 273,626.** But it is now precisely
specified rather than hypothetical.

- Decomposed to **CYCLE COST** (+22.9% ms/cycle, 56.5 -> 69.5) with acceptance
  essentially flat (-3.5%).
- Every candidate lever closed **by measurement**: `FENCE_ASYNC` already live;
  stale top-k reuse FALSIFIED (0.42 overlap at depth, and the indexer is 0.0% of
  wall anyway); `INDEX_TOPK<512` forbidden (#49); `FENCE_EVERY_N_LAYERS` 4->8
  worth only ~+0.7 t/s and costs c=2 bistability; `SPARSE_SDPA_TILE` already 128.
- **Attribution (sync-mode validated): the cost is `attn.sdpa.compressed`** —
  369.96 µs / 5.8% at 250K, larger per call than plain `attn.sdpa`, and it is the
  SDPA whose K/V set scales with depth. `indexer.score` is 19.88 µs / **0.0%**.
- **Next step for a future session:** read the compressed-SDPA call site for
  avoidable materialisation (mask construction, pooled-buffer copies, layout) —
  sub-tiling is already closed as a lever (0.998-1.047x tiling ratio).

## OPEN / NOT DONE (explicit)

1. **Fresh-boot control for the adaptive-prefill result** — the A/B glm-5.3
   specified (fresh boot, change OFF, same 638K depth) to isolate chunk size from
   boot state. Not run: each 638K run is ~35-40 min of prefill and the session
   ran out of runway.
2. **`attn.sdpa.compressed` investigation** — the specified next step for (b).
3. **Wired-limit experiment (115000 -> 124000)** — deliberately NOT taken: its
   root-cause commit states the wedge is "NOT recoverable by relaunch (wired
   memory pinned; only a full reboot clears it)", so a failed test could leave
   the cluster wedged until the user reboots. **Needs an explicit user decision.**
4. **Hang-watchdog deeper fix** — the probe now reads the stack for the native-
   setup class, but it still infers liveness from footprint growth for other
   phases. The general fix (distinguish "slow but progressing" from "stalled")
   remains open.
5. **Quality gate on the chunk-size change** — glm-5.3 recommended a
   greedy top-1 match / needle-straddling-crossover test. Not run.

## CORRECTION LEDGER (my own claims measurement killed)

1. "Acceptance cliff at depth" — WRONG (46.30 t/s outlier; real ~30K value 34.89).
2. "Indexer top-k threshold cliff at 65,536 tokens" — FALSIFIED (5-rung ladder).
3. "DSpark residual ~6.5-7 GB is shardable" — WRONG (only ~0.12 GB is).
4. "A 275K leaf holds ~0.07 GB so the byte cap is inert" — WRONG (2.62 GiB).
5. "The fix frees ~10.8 GiB at 565K" — 2x overstated (~5.4 GiB).
6. "The cap binds above ~300K" — actually ~420K.
7. "Stale top-k reuse is not viable (mean 0.684)" — right conclusion, wrong
   evidence (unrepresentative tail sample); then I guessed the bimodality meant
   "deep is stable" and the pool_size split proved the OPPOSITE (deep 0.42,
   shallow 0.97). Caught by computing the cut instead of trusting the guess.
8. "The indexer carries the depth cost" — FALSIFIED twice (non-sync + sync).
9. **Committed a reverted approach**: `cb2958e07` bumped the hang timeout for
   Tensor — an approach the skill reference documents as explicitly considered
   and rejected. Reverted in `756574e6c` after re-reading it.

The dominant failure mode across all nine: **acting on an assumed number before
measuring it**, and not re-reading the loaded skill's documented rejections
before making a change.

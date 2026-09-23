# Overnight close-out: all cheap levers measured and exhausted; the gap needs a decided trade (2026-09-23)

## Summary of the night

**Deliverable: a correct characterisation, and the elimination of every cheap
fix.** No config was changed; the cluster is healthy and on production default.

Both of the user's targets turned out to be **one problem**: an intermittent
"collapse mode" that appears from ~250K depth and worsens with depth.

## The measured picture

| depth | normal tps | collapse tps | N |
|---|---|---|---|
| 30,000 | 34.89 (stdev 0.64) | none seen | 4 |
| 44,930 | 31.30 | — | 1 |
| 59,879 | 33.09 | — | 1 |
| 76,175 | 33.93 | — | 1 |
| 97,911 | 32.63 | — | 1 |
| 121,097 | 30.53 | — | 1 |
| 127,929 | 29.72 | — | 1 |
| 250,000 | 26.2-28.7 (5 of 6) | **19.46** | 6 |
| 565,000 | 24.15 | **13.36** | 2 |

- **Stable and tight to ~120K.**
- **Bimodal from ~250K**: normal mode plus an intermittent collapse mode.
- Collapse frequency/severity worsens with depth (1-in-6 at 250K; present at
  565K in the one pair measured).

## Mechanism (established)

Memory pressure -> macOS evicts the model's **mmap-backed clean weight pages**
-> they are re-faulted from SSD.
- Direct evidence: 32,469 page-ins in a single request at 42K depth under
  tighter memory (vs 47 at 401 tokens).
- At 565K: page-ins 6,599 (vs 400 at 128K), peak 117.24 GB against a 120.6 GB
  wired limit (~97%).
- NOT swap (pageouts=0), NOT compressor (decompressions~0).
- Bistable because it depends on the margin at decode onset and is
  self-sustaining once started — matches the 352K doc's 4/16 base rate.

**Falsified tonight (do not re-litigate):**
- Indexer top-k acceptance cliff at 65,536 — no step; ladder flat 31-34 t/s
  from 45K to 122K across the predicted crossover. (The code mechanism
  `k = min(index_topk, pooled.shape[1])` is real; its throughput effect at
  these depths is not observable.)
- "tok/s is content-driven" — that was an artifact of comparing across scripts.
  Within one script, depth is the only variable.
- The 46.30 t/s "healthy 30K" figure — a single outlier. True value 34.89
  (n=4, stdev 0.64).

## Every cheap footprint lever, measured

| lever | GB/rank recovered | verdict |
|---|---|---|
| DSpark non-FFN sharding | **0.12** | measured (non-FFN = 0.243 GB of a 10.88 GB head) — 17x too small |
| Leaf snapshot retention 3→1 | **0.17** | measured from `[SNAPMEM]` log — 17x too small |
| `EXO_MLX_CLEAR_CACHE_INTERVAL=64` | — | already live (the 352K doc's own mitigation) |
| `EXO_DSV4_MTP_DRAFT_LMHEAD_BITS=4` | — | measured elsewhere: -8.7% at depth |
| `EXO_DSV4_DSPARK_CONF_TAU` | — | prior sweep: 0.5 already at plateau |
| `EXO_DSV4_DSPARK_TP_SHARD=1` | 3.0-3.5 | already live |

**Required: ~5 GB/rank** to restore the ~19 GB headroom seen at 115K.
**Available from cheap levers: 0.29 GB.** All are ~17x too small.

## The remaining options, with risk stated plainly

1. **Raise `DSV4_WIRED_LIMIT_MB`** (115000 → ~120000). Buys headroom directly.
   **Risk:** re-enters the documented `MetalAllocator` stuck-wedge regime
   (prefill ~5x drop, GPU idle, requires a full reboot to clear). The launcher
   comment is explicit that 115000 was chosen deliberately to stay out of it.
2. **Quantize the DSpark head below mxfp4.** Real GB, but quality-gated (the
   draft head drives acceptance) and unmeasured — needs its own A/B.
3. **Page/spill DSpark stages.** Large architectural change.

**None of these is safe to do unilaterally overnight on a sleeping owner's
production cluster.** Option 1 is a known-bad regime; options 2 and 3 are
research-grade and cannot be validated (quality-gated) in a few hours.

## What the user should decide

- **Is T2 actually still a target?** 250K delivers 26-29 typically. If the real
  workload is ~100-150K, the cluster is at 30-34 t/s and this is moot.
- **If 250K-500K matters**, the choice is between accepting the wired-limit
  risk (option 1, with a monitored, single-node, small-step test and a rollback
  plan), or funding option 2 properly.
- **Before any of that:** establish the collapse RATE on the current config
  (N>=10 at 250K and 565K), so a fix can be shown to move it. Tonight gives
  N=6 at 250K (1 collapse) and N=2 at 565K (1 collapse) — enough to prove the
  phenomenon, not enough to prove a fix.

## Cluster state (verified)
- Production defaults: `VERIFY_ROWSEQ_VEC=1`, `VERIFY_BATCH=1`,
  `DSPARK_TP_SHARD=1`, `ATTN_ALLSUM=0`, `MLX_CLEAR_CACHE_INTERVAL=64`.
- `iogpu.wired_limit_mb = 115000` (unchanged).
- 2/2 RunnerReady, 0 pending, end-to-end completion verified ("Paris", stop).
- Raw-GEMM canary at session start: 14.86 / 14.86 TFLOPS both nodes.

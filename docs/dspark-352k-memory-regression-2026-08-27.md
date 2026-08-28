# DSpark 352.6K memory regression — investigation, root cause, fix (2026-08-27/28)

Status: **root cause identified and code-verified; fix shipped (`2d85ccdcb`,
env-gated default OFF); deep-context protocol OFF phase + fix validation
in flight overnight** (driver: `/tmp/ab/protocol352/overnight_driver.py`,
artifacts under `/tmp/ab/protocol352/`).

## Symptom

Running the 352.6K-token paired decode protocol (the deep-context gate for
DSpark MTP production promotion — bar: >= +15% median vs spec-off), the
spec-ON arm stochastically collapsed:

- **16 ON runs, 4 collapsed to ~1 tok/s** (runs #4, #6, #8, #11); the rest
  16.9–31.1 tok/s.
- Collapse signature (from per-cycle `[MTP]` log timestamps on m4-1):
  median cycle gap **1.83–2.26 s** (vs 100 ms healthy), EVERY cycle stalled
  from decode onset to end — a persistent thrashing equilibrium, not
  sporadic spikes. Max single gap 10.4 s.
- Swap: 1.37–1.76 GB on BOTH nodes during collapsed runs; healthy runs sit
  at ~97 GB wired, 500–700 MB free, ~25 MB swap — razor-thin margin.
- MTP-PROF (GPU kernel time only) inflates during collapse (verify mean
  528→687 ms vs ~82 ms healthy): kernels slowed by memory contention, plus
  ~1.5 s/cycle invisible to the profiler = page-fault wait.
- Three distinct stable modes across runs: fast (~100 ms/cycle median),
  slow (~155–170 ms/cycle), collapsed (~1.8–2.3 s/cycle). Collapse is
  decided near decode onset (per-run initial-condition sensitivity), and
  the system fully recovers between runs.

Full 16-run ON table: `/tmp/ab/protocol352/summary_on.jsonl`; per-run
cycle-gap analysis in the session notes. ON runs #0–11 were one runner
instance; #12–15 a fresh instance (fresh instance: 27.3, 31.1, 19.2, 18.6
tok/s, zero collapses in 4).

## Why this is a regression, not a capacity limit

`docs/known-good-prefill-baseline-2026-08-21.md`: spec-OFF decode at
**500K** context ran 17.26 tok/s with ZERO swap on the same hardware.
500K > 352.6K — the KV footprint then was LARGER, and it fit comfortably.
Something spec-ON keeps resident eats the margin. (User explicitly rejected
a context-cap mitigation; root-cause fix required.)

## Root cause (code-verified)

**The DSpark draft head (~10.13 GB, mxfp4/mxfp8-quantized, 3 stages) is
loaded fully REPLICATED on both nodes under TP=2.**

- Load: `_overlay_dsv4_dspark_native` attaches the head as `inner.dspark`
  (`src/exo/worker/engines/mlx/utils_mlx.py:866`; head file
  10,876,789,654 B per the comment at `:385`).
- Sharding: `DeepseekV4ShardingStrategy.shard_model`
  (`src/exo/worker/engines/mlx/auto_parallel.py`) shards
  `model.model.layers` and the `mtp_blocks` — but **never
  `model.model.dspark`** (verified: zero shard-path references before the
  fix).
- Residency accounting (full byte math in
  `docs/dspark-352k-residency-analysis-2026-08-27.md`): spec-OFF ≈
  89.3 GB/node steady; spec-ON ≈ 99.4 GB/node. The delta is the head.
  Observed healthy spec-ON wired (~97 GB) matches within accounting slop.

**Thrash persistence mechanism:** the per-cycle TOUCHED working set at
352.6K is ~19–22 GB/node (MoE random expert routing ~13.4 GB + attention/
indexer/compressor weights ~3.7 GB + indexer pool full-scan ~450 MB +
pool gathers + DSpark head ~1.3 GB). MLX weights are mmap-backed clean
pages: under pressure macOS evicts them without swap accounting and
re-faults them from SSD every cycle — which matches the signature exactly
(huge stalls, inflated kernel times, only ~1.4–1.8 GB of anonymous swap).
Once eviction starts, every cycle re-faults → self-sustaining collapse;
runs that start under the ceiling stay healthy. Bistable equilibrium →
the observed stochastic 4/16 hit rate.

**Refuted along the way** (docs kept for the audit trail):

- Batched-vs-rowseq verify transients: only +4.73 MB/sparse-layer,
  bounded O(10 MB) by per-layer async fencing — NOT the driver
  (`docs/dspark-352k-batched-verify-transients-2026-08-27.md`).
- Metal allocator fragmentation: REFUTED — BufferCache size-class reuse
  works as designed
  (`docs/dspark-352k-allocator-pool-analysis-2026-08-27.md`).
- `EXO_DSV4_POOL_SNAPSHOT_BATCH=1`: INERT at c=1 (BatchPoolingCache only
  exists at c≥2). `EXO_DRAFT_KV_WINDOW`: PP-classic-draft only, unused by
  DSpark. DSpark ctx-KV rings: 384 KB total. Pool storage: 2.27 GB/node
  (corrected from 1.95 GB — prior docs missed the indexer's D=128 pool),
  identical both arms.
- Secondary churn feeder: `EXO_DSV4_SPEC_STATE_RESTORE=1` copies all 43
  rings + pool meta ≈ ~11 MB alloc+free per verify cycle into the MLX
  buffer-cache retention ratchet (`EXO_MLX_CLEAR_CACHE_INTERVAL=0` in
  production; the MTP loop's own `mx.clear_cache()` every 512 cycles
  bounds it coarsely). Plausible contributor to the fast→slow mode drift;
  addressed in validation via `EXO_MLX_CLEAR_CACHE_INTERVAL=64`.

## The fix (shipped)

`2d85ccdcb` — `perf(dspark): env-gated TP sharding for DSpark draft-head
FFN` (**`EXO_DSV4_DSPARK_TP_SHARD=1`, default OFF**).

Shards the 3 DSpark stages' MoE FFN weights (shared_experts + switch_mlp
gate/up/down) across the TP group via the same
`all_to_sharded/sharded_to_all` helpers the mtp_blocks loop uses, inside
`DeepseekV4ShardingStrategy.shard_model` (auto_parallel.py:1180+, after
the mtp loop — verified execution order: the head overlay attaches before
`tensor_auto_parallel` runs). Failure policy: try/except detaches the head
(rank-consistent) rather than crashing model load. Pre-verified: quantized
shard helpers preserve mxfp mode; draft forward executes on BOTH ranks so
collectives pair; 2048/2/32 divides cleanly.

- Expected recovery: **~3–3.5 GB/node** (FFN fraction of the head; the
  attention/main_proj/markov parts stay replicated).
- Expected cost: up to ~6 small RDMA collectives per draft cycle ≈
  +1–3 ms (~1–3%).
- **Honest framing:** this moves spec-ON from ~99.4 to ~96 GB/node —
  a plausible collapse-eliminator (bistability is sensitive to a few GB
  of initial margin), NOT a full margin-restorer. Robust headroom parity
  with spec-OFF would need the remaining ~6.5–7 GB (shard the non-expert
  projections where dims divide, quantize the head harder, or page
  stages). Zero collapses in 8 validation runs ≈ 90% confidence against
  the 25% base rate — the writeup must say so rather than declare the
  mode eliminated.

## Protocol completion plan (in flight)

Overnight driver phases (all artifacts `/tmp/ab/protocol352/`):

- **A. stripped-OFF ×9** (`run_off_*.json`): `EXO_SPECULATIVE=0
  EXO_DSV4_MTP=0 EXO_DSV4_DSPARK_FORCE_LOAD=0` — the TRUE production-OFF
  control (head-load gate confirmed live: "DSpark head load SKIPPED
  (~10 GB/node reclaimed)"). Paired verdict = 16 ON (current config,
  collapses included) vs this arm, +15% bar.
- **B. forced-OFF ×3** (`run_offheld_*.json`): original spec-off launch
  with `FORCE_LOAD=1` (head resident, no spec execution) — residency-only
  diagnostic probe. 3 runs is a probe, not proof (0.75³ ≈ 42% chance of
  zero collapses by luck); read swap/wired telemetry, not just tok/s.
- **C. verbon3 ×8** (`run_von3_*.json`): production ON env +
  `EXO_DSV4_DSPARK_TP_SHARD=1` + `EXO_MLX_CLEAR_CACHE_INTERVAL=64` +
  `EXO_MEMORY_PROFILE_PATH=/tmp/mem_verbon3.jsonl INTERVAL=16`.
  Smoke-gated (Paris probe + shard-engagement log check + detach-fallback
  check; aborts to stripped-OFF config on failure). Bundled fixes by
  design (both memory-side, non-interfering; ablate only if the bundle
  passes and a minimal production config is wanted). This arm is a
  SEPARATE candidate config — it must NOT be silently substituted as the
  numerator in the 16-ON-vs-OFF gate verdict.

Success criteria for promoting DSpark at 352.6K: verbon3 zero collapses
AND median > stripped-OFF + 15%. Any collapse in verbon3 → the honest
recommendation is spec-ON depth-gated OFF above some context (production
config keeps the batched-verify win at ≤100K; NOT a context cap on the
model — spec-off serving at max depth is proven).

## Verdict computation (when OFF lands)

```
python3 - <<'EOF'
import json, random, statistics as s
on  = [json.loads(l) for l in open('/tmp/ab/protocol352/summary_on.jsonl')]
off = [json.loads(l) for l in open('/tmp/ab/protocol352/summary_off.jsonl')]
a = [r['fixed_window_tok_s'] for r in on]; b = [r['fixed_window_tok_s'] for r in off]
ma, mb = s.median(a), s.median(b)
print(f"ON median {ma:.2f} (n={len(a)}, {sum(1 for x in a if x<5)} collapsed)  OFF median {mb:.2f} (n={len(b)})  delta {100*(ma-mb)/mb:+.1f}%")
random.seed(7)
ds = [100*(s.median(random.choices(a,k=len(a)))-s.median(random.choices(b,k=len(b))))/s.median(random.choices(b,k=len(b))) for _ in range(10000)]
ds.sort(); print(f"bootstrap CI [{ds[250]:+.1f} .. {ds[9750]:+.1f}] vs +15% bar")
EOF
```

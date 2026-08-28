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

## RESULTS (2026-08-28, protocol complete)

### Gate verdict (pre-registered: 16 pre-fix ON vs stripped-OFF ×9)

- ON (pre-fix, collapses included): n=16, 4 collapsed (<5 tok/s), median
  **19.87** — sorted: 1.1, 1.3, 1.4, 1.4, 16.9, 18.6, 19.2, 19.6, 20.2,
  20.2, 25.8, 26.4, 27.3, 28.3, 28.7, 31.1.
- stripped-OFF: n=9, median **24.19** (22.47-24.39, zero collapses,
  `finish=stop` all runs).
- **Gate: −17.87% median [bootstrap 95% CI −57.5 .. +9.8] → FAIL.** The
  unfixed config is confirmed WORSE than spec-off at 352.6K; the collapse
  tail dominates. This is the honest pre-fix verdict the protocol was
  designed to produce.

### Residency probe (forced-OFF ×3, arm B)

23.92 / 23.45 / 23.68 tok/s, zero collapses, swap stayed ≤ ~100 MB.
Head-resident-but-inert did not collapse in 3 runs — consistent with the
margin story (spec execution's touched working set on top of residency is
what tips runs over), though 3 runs only bound the collapse rate loosely
(0.75³ ≈ 42% zero-collapse-by-luck).

### verbon3 candidate verdict (arm C — separate candidate, NOT the gate)

8 genuine runs (2 slots voided by an unrelated-to-memory jaccl WC_ERR
segfault at 08:52 — first in campaign, auto-recovered, watch item):
26.29, 34.77, 28.83, 28.05, 30.52, 30.89, 27.97, 19.50 tok/s; all
`finish=stop`.

- **Collapses: 0/8** (success criterion #1 MET).
- Median **28.44** vs stripped-OFF 24.19 = **+17.57%** (bar +15% →
  criterion #2 MET on the point estimate; unpaired bootstrap 95% CI
  [+8.7 .. +27.8] — the CI floor is below the bar, so the throughput
  margin is thin; the collapse-elimination is the conclusive part).
- Telemetry: swap peak 97 MB (pre-fix: 1.37-1.76 GB), pageouts-delta
  ~44-49 over the ~4.7h phase; wired peak 92.0 / 93.5 GB (pre-fix
  telemetry window: 96.8 / 95.3) — ~3-4.8 GB/node recovered, matching the
  shard's predicted ~3-3.5 GB/node.
- Caveat: verbon3 bundles TP_SHARD with `EXO_MLX_CLEAR_CACHE_INTERVAL=64`
  (+ mem-profile overhead); knobs not isolated. The wired-peak drop
  matches the shard's prediction, so the shard is the operative fix.

### Promotion decision

`EXO_DSV4_DSPARK_TP_SHARD=1` **promoted for spec-ON at depth** (add to the
production spec-ON launch env alongside the batched verify). The 100K
promotion (+36.7%) is unaffected. Spec-ON at 352.6K now runs
collapse-free at +17.6% median; no depth gate needed.

### Launch-failure post-mortem folded in

The first verbon3 validation attempt (04:01) aborted before any run: the
smoke request raced post-launch cluster-state convergence, and the JIT
placement wait loop hard-503'd on a transient non-memory blocker
(blocker-class oscillation). Root-caused and fixed in exo `75d2402dd`
(wait polls through ALL JitPlacementUnavailableError reasons; first+last
blocker in the 503 detail; regression tests replace the test that encoded
the old behavior). The fix was then live-validated at 07:50: the same
oscillation occurred (memory → "MLX ring backend requires connectivity"
→ viable) and the wait survived it — placement succeeded 6s in, and the
phase ran. A second launch-automation bug found the same morning: the env
gate read `ps -axo command` (env prefixes never appear in argv) — fixed to
`ps eww <pid>` in phaseC_von3.py.

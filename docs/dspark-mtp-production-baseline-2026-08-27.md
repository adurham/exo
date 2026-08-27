# DSpark MTP Production Baseline — 2026-08-27

Tag: `dspark-mtp-prod-20260827-1745` (exo repo + mlx-lm submodule, tagged at the
exact commits this baseline was measured and promoted on).

**STATUS: PROMOTED — this is the new production default.**
`start_cluster.sh` now ships these defaults (changed 2026-08-27):
`EXO_DSV4_VERIFY_BATCH=1`, `EXO_SPECULATIVE_GAMMA=3`, `EXO_DSV4_MTP=1`,
`EXO_DSV4_MTP_DEDICATED=0`. A plain launch reproduces this baseline.

## The verdict that promoted it

24 paired runs (12 batched-ON vs 12 spec-OFF, time-adjacent pairs, same
100K-token prompt, temp=0, 256-token fixed window, `bench/golden_v1_probe.py`):

| metric | batched-ON | spec-OFF | delta |
|---|---:|---:|---:|
| median fixed-window tok/s @100K | **36.63** | 27.15 | **+36.71%** |
| 95% bootstrap CI (10K resamples) | | | **[+28.26%, +51.02%]** |
| pairs where ON > OFF | | | **12/12** |
| weakest / strongest pair | | | +14.17% / +56.17% |

Pre-registered PROMOTE bar (median >= +50% of wall-model prediction at
C_s=2.14, i.e. >= +13%, AND lower CI > 0): cleared on both counts.

Full 24-run table: `docs/PERFORMANCE_HISTORY.md` (2026-08-27 entry).

## Mechanism

- Depth-gated batched verify (`EXO_DSV4_VERIFY_BATCH=1`, `MIN_CTX=8192`):
  the pre-rowseq batched M=4 verify forward (reintroduced from git history
  after the indexer-stream-sharing variant FAILED: it saved 8% verify but
  killed acceptance -19% via stale pool snapshots). Batched path runs at
  ctx >= 8192; rowseq (bitwise) below — short-context byte-identity intact.
- Verify cost: 83.76 -> 60.60 ms mean (MTP-PROF n=1550), C_s 3.20 -> 2.14.
- Acceptance: 2.250 (parity with rowseq 2.118).
- Correctness gates: G0'' PASS (batched-vs-rowseq drift 74.7% <= base-vs-base
  run-to-run drift 99.3% at 100K — the batched path adds LESS noise than the
  base already has, since MLX Metal dispatch drift makes the base
  nondeterministic at depth); Tier-1 short-ctx byte-identity on the rowseq
  path; 10K-cycle soak clean; strict-argmax acceptance (Gate A) intact.
- Commits: submodule `mlx-lm` `dda9237` ("CORRECTED depth-gated batched
  verify"), parent `exo` `6eba31ff1` (pointer), docs `56be3ea18`.

## Promoted production config (exact)

```
EXO_SPECULATIVE=1
EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1
EXO_DSV4_DSPARK_FORCE_LOAD=1
EXO_DSV4_MTP_DEDICATED=0          # native checkpoint-bundled head
EXO_SPECULATIVE_GAMMA=3
EXO_SPECULATIVE_TEMP=0.0
EXO_SPECULATIVE_ALPHA=1.0
EXO_DSV4_VERIFY_BATCH=1
EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192
EXO_DSV4_VERIFY_ROWSEQ=1          # rowseq still the <8K path
EXO_DSV4_MTP_TIE_REVERIFY=0
EXO_DSV4_HC_COLLAPSE_KERNEL=1
EXO_DSV4_HC_EXPAND_KERNEL=1
EXO_DSV4_SPEC_EOS_BAN=0           # natural-end semantics (EOS ban caused degen)
EXO_DSV4_MTP_LOG_INTERVAL=1       # acceptance observability
EXO_DSV4_MTP_PROFILE=50           # per-phase timing (production-optional)
```

## Operational notes

- **Cold-start**: the first batched verify cycle can trip the jaccl
  GPU-event fence timeout IF a second model is concurrently resident and
  memory pressure slows the first cycle. exo's load warmup covers kernel
  JIT; the failure mode observed was placement-contention (the preview
  model still resident blocking 0731). Keep the cluster single-model
  during the first request after a launch. Warm kernels: zero issues
  across 12 consecutive runs.
- **352.6K depth: NOT YET MEASURED.** The +36.7% is @100K. The deep-context
  bar (>= +15% median vs spec-off @352.6K) is the open question; run the
  paired protocol at 352.6K before trusting max-depth production.

## Prior baselines superseded

- `known-good-prefill-baseline-2026-08-21.md` (decode 17.5-18.6 tok/s
  @100-500K, spec OFF entirely) — DSpark MTP now serves ~36.6 tok/s @100K
  on top of that prefill stack.
- `dspark-verdict-measurement-2026-08-26.md` (rowseq verdict +1.87%,
  REVERT) — superseded by the batched path.

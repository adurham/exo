# Overnight campaign 2026-09-22/23 — deep-context decode (REVISED after Phase 1 findings)

**Authorization:** user granted full free reign overnight, including relaunches.
User asleep. Leave cluster HEALTHY.

## What changed from the original plan

Two measured corrections (see
`docs/overnight-corrections-dspark-sizing-and-content-driven-tps-2026-09-23.md`):

- **Phase 4 (shard DSpark non-expert projections) is DEAD.** Measured: non-FFN
  components total 0.243 GB of a 10.88 GB quantized head; sharding them recovers
  ~0.12 GB/rank. The head is already ~50% sharded. Needs ~4.9 GB. Not the fix.
- **The depth ladder was confounded by CONTENT.** Acceptance varies 2.6x
  (0.810-2.077) vs cycle cost only 22% (54.6-66.5 ms), and acceptance tracks
  output content. The 27.8K rung measured 46.3 t/s where earlier runs gave 33.

## Revised targets

- **T1 (500K must not collapse):** mechanism CONFIRMED (mmap refault; 32,469
  page-ins at 42K depth). Fix must reduce per-rank footprint by GBs, not MBs.
- **T2 (250K >= 30 t/s):** may be CONTENT-dependent rather than a real compute
  shortfall. Settle with a controlled test before claiming anything.

## Sequence (revised, ordered by information value per minute)

### Step 1 — controlled content test at fixed depth  [NO relaunch]
`/tmp/content_control.py 9000 2` — easy vs freeform, alternating, same depth.
Decides whether T2 is real. Highest information value; cheap.

### Step 2 — complete the depth ladder  [NO relaunch]
Already running: 25K / 115K / 250K / 500K with pageins + acceptance + peak.
Gives the refault-vs-acceptance split at the two depths that matter.

### Step 3 — decide T1's lever with real numbers
Given the head is already 50% sharded and non-FFN is 0.24 GB, the remaining
footprint options are:
  (a) **head quantization below mxfp4** — quality-gated (draft head drives
      acceptance), needs its own A/B
  (b) **spill/page DSpark stages** — large change, risky, probably not a night
  (c) **reduce KV/snapshot retention** — `EXO_LEAF_SNAPSHOT_RETENTION=3` is
      live; snapshots hold state at depth. Cheap to test, memory-only effect.
Start with (c) as the one cheap, reversible, non-quality-affecting memory lever.

### Step 4 — health + writeup
Restore production defaults, verify end-to-end, commit, update memory.

## Bands (unchanged)
- PROMOTE: target met AND needle_hit AND acceptance parity
- Collapse is stochastic -> N>=6 for any collapse claim
- Raw-GEMM canary first: PASSED tonight (14.86 / 14.86 TFLOPS both nodes)
- Never compute tok/s from wall clock

## Hard rules
- Commit+push every turn.
- Never leave the cluster mid-relaunch.
- Root-cause fixes only.
- If unexplained, say so.

## Log
- Phase 0 canary: PASSED 14.86/14.86 TFLOPS
- Phase 1 rung 1 (27.8K): 46.30 t/s, acc 2.077 -> triggered Correction B
- DSpark sizing measured -> triggered Correction A, Phase 4 cancelled

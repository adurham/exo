# Overnight campaign 2026-09-22/23 — deep-context decode: fix 500K collapse, get 250K > 30 t/s

**Authorization:** user granted full free reign overnight ("do what is needed on the
cluster"), including relaunches. User is asleep — no clarifications possible. Make
decisions, document them, leave the cluster HEALTHY.

## Two targets (user: "both lol")

- **T1: 500K must not collapse.** Currently 13.36 t/s at 565K (vs 29-37 at <=120K).
  Mechanism CONFIRMED: mmap clean-page eviction + refault (32,469 page-ins/request
  at 42K; headroom down to 4.9 GB at 565K).
- **T2: 250K >= 30 t/s.** Currently 26.72 / 26.83 t/s (two independent runs). +12% needed.

## Pre-registered bands (BEFORE measuring — standing rule)

| outcome | criterion |
|---|---|
| PROMOTE | target met AND needle_hit=true AND acceptance parity (>= baseline -2%) |
| REJECT | within +/-2% of baseline |
| INCONCLUSIVE | between |

- Collapse is **stochastic** (4/16 in the 352K protocol). Single runs prove nothing;
  need N>=6 for a collapse claim, and a stated collapse definition
  (cycle gap > 500 ms sustained, or decode_tps < 15 at 500K).
- Always run the **raw-GEMM canary** first: ~15.2 TFLOPS healthy, single-digit =
  GPU power-degraded, reboot before trusting any number.
- **Never compute tok/s from wall clock** — use the probe's decode_tps/decode_s.

## Sequence

### Phase 0 — state + canary (must pass before any measurement)
- [ ] git clean, commit recorded
- [ ] raw fp16 GEMM canary both nodes; reboot via reboot-node.sh if degraded
- [ ] record current live env (full)

### Phase 1 — decompose the 250K shortfall (cheap, no relaunch)
- [ ] clean per-request acceptance + ms/cycle at 25K / 115K / 250K
- [ ] is T2's gap acceptance-driven or cycle-cost-driven?
- [ ] pageins at 250K (is refault already biting there, or only at 500K?)

### Phase 2 — 500K refault: is it dominant?
- [ ] pageins + fraction-of-wall at 500K
- [ ] if dominant -> the fix is footprint reduction, not compute

### Phase 3 — config levers (each = relaunch + measure, ~40 min/cycle)
- [ ] EXO_LEAF_SNAPSHOT_RETENTION=1 (memory; snapshots hold KV state at depth)
- [ ] any other live memory knob found in Phase 1/2
- [ ] measure at 500K: does headroom improve? does collapse go away?

### Phase 4 — structural lever: DSpark non-expert sharding
- [ ] extend `_shard_stage` (auto_parallel.py:1245) to remaining replicated parts
- [ ] expected: ~6.5-7 GB/node recovered vs 4.9 GB headroom at 565K
- [ ] QUALITY GATE MANDATORY (draft head feeds speculation; acceptance must hold)

### Phase 5 — leave cluster healthy
- [ ] restore production defaults
- [ ] final health check + end-to-end completion
- [ ] write up; commit+push; update warm memory

## Hard rules for the night
- Commit+push every turn (concurrent sessions can reset the tree).
- Never leave the cluster mid-relaunch. If a cycle fails, restore defaults first.
- Sample live faults BEFORE cleanup.
- Root-cause fixes only — no timeout bumps, no caps as "fixes".
- If something is unexplained, say so in the writeup rather than papering it.

## Log
(started 2026-09-22 ~22:40)

# MAJOR: a memory RATCHET, and it may be an artifact of cache-busting probes (2026-09-23)

## What the [MEM] logs show

`[MEM] before prefill` = the baseline memory the NEXT request starts from.
In strict time order tonight:

| ctx tokens | before prefill | after prefill | delta |
|---|---|---|---|
| 97,910 | 98.23 | 99.46 | +1.23 |
| 121,096 | 92.98 | 93.95 | +0.97 |
| 32,686 | 93.47 | 93.37 | **-0.10** |
| 30,488 | 93.53 | 92.81 | **-0.72** |
| 32,058 | 92.67 | 92.39 | **-0.28** |
| 30,173 | 91.76 | 91.53 | **-0.23** |
| 290,435 | 90.67 | 95.19 | +4.52 |
| 276,426 | 95.55 | 98.48 | +2.93 |
| 270,820 | 97.39 | 101.27 | +3.88 |
| 276,423 | 101.07 | 104.71 | +3.64 |
| 273,622 | 105.26 | 107.56 | +2.30 |
| 276,423 | 104.40 | 107.66 | +3.26 |

Two clear facts:

1. **Shallow runs (~30K) RECOVER** — every one returns slightly BELOW where it
   started (deltas -0.10 to -0.72 GB). No leak at shallow depth.
2. **Deep runs (~275K+) RATCHET UP** — baselines climb monotonically
   90.67 -> 95.55 -> 97.39 -> 101.07 -> 105.26 -> 104.40 GB across six
   consecutive deep requests. **+15 GB, never returning to baseline.**

This is exactly the behaviour that produces the observed collapse: each deep
request pushes the floor higher until a run starts too close to the ceiling
and cannot recover → the 19.46 / 13.36 runs.

**Notably: a *failed* deep run should LOWER the baseline, not raise it** — and
it did not. That points at the memory not being released at all.

## The critical caveat — and the discriminator

**Every probe request uses a fresh uuid in its prompt** (deliberate
cache-busting, so runs are not confounded by prefix-cache hits). That means
**each request creates a NEW prefix-cache LEAF**.

Prior investigations on this exact cluster documented:
- fact 772: *"a real streaming session creates NEW LEAVES per request... each
  new leaf = full DSv4 KV deepcopy... Eviction happens but GPU memory is not
  promptly reclaimed by the Metal allocator."*
- fact 775: multi-leaf accumulation fixed via `DSV4_MAX_PREFIX_SESSIONS` 4→1.
- fact 778: a SEPARATE "prefill working-set" leak on a growing leaf,
  documented as **real and NOT fixed**.
- fact 798: leaf cap re-sized to 2; parked leaves evictable by LRU.

**The user's real workload is a continuing conversation = ONE growing leaf**,
not a new leaf per turn. So tonight's ratchet may be substantially an artifact
of my own cache-busting probe rather than a property of production traffic.

## Discriminator test

`/tmp/leaf_vs_ratchet.py` — two arms at ~80K depth, watching the `[MEM]
before prefill` baseline:
- **MISS arm**: fresh uuid × 3 → new leaf each time
- **HIT arm**: identical prompt × 3 → prefix-cache hit → same leaf

If HIT stays flat while MISS climbs → the ratchet is leaf-accumulation, and
**the probe has been overstating the problem for real workloads**. That would
mean:
- the 250K/565K numbers measured tonight are a LOWER BOUND,
- the real continuing-conversation workload is in better shape than reported,
- and the correct fix targets leaf retention, not decode compute.

If HIT also climbs → the ratchet is intrinsic (the working-set leak of fact
778), which is a genuinely open, previously-flagged defect.

## Why this matters

If the ratchet is probe-induced, then **both user targets (T1 500K collapse,
T2 250K>30) may be substantially less severe in real use** — and the honest
recommendation is to re-measure with a CONTINUING-CONVERSATION workload before
any fix is scoped. That also matches the user's own instinct that 30 tok/s at
100K used to be routine.

## Status
- Ratchet: measured, in the logs, unambiguous.
- Attribution (leaf vs intrinsic): discriminator script written, NOT yet run.
- No cluster config changed.

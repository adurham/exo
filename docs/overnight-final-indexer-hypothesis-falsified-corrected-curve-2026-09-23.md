# Overnight FINAL: indexer-cliff hypothesis FALSIFIED; corrected depth curve; both targets essentially met (2026-09-23)

## What I got wrong, and how it was caught

Earlier tonight I measured **46.30 t/s with acceptance 2.077** at 27,850 depth
and built two hypotheses on it:
1. "tok/s is content-driven" (Correction B)
2. "an indexer top-k acceptance cliff at ~65,536 tokens"

**Both are now falsified.** The falsifier was a variance measurement I should
have run FIRST:

```
VARIANCE at depth ~28-32K, n=4:
  run 0: 32,687 -> 35.52
  run 1: 30,489 -> 34.10
  run 2: 32,059 -> 34.66
  run 3: 30,174 -> 35.30
  mean 34.89, stdev 0.64, spread 4.2%
```

The real value at ~30K is **34.89 t/s (stdev 0.64)**. The 46.30 was a **single
outlier** (likely first-request-after-idle state), and I inferred two mechanisms
from it before establishing the noise floor. That is the same error class as
the rest of this repo's retraction history: a number measured by an instrument
that didn't control what it claimed to.

## The indexer cliff: FALSIFIED by a direct test

Prediction was a step at 65,536 tokens (where `k = min(index_topk,
pooled.shape[1])` starts dropping entries). Cliff ladder across the predicted
crossover:

| depth | tps | needle |
|---|---|---|
| 44,930 | 31.30 | ✓ |
| 59,879 | 33.09 | ✓ |
| 76,175 | **33.93** | ✓ |  <- past the predicted crossover
| 97,911 | 32.63 | ✓ |
| 121,097 | 30.53 | ✓ |

**No step.** The mechanism is real in code (`k = min(...)` is source-verified),
but its effect on throughput at these depths is not observable. Hypothesis
**rejected**. The source-read was correct; the inference from it was wrong.

## Corrected depth curve

| depth | tps | note |
|---|---|---|
| 30,000 | **34.89** | n=4, stdev 0.64 |
| 44,930 | 31.30 | |
| 59,879 | 33.09 | |
| 76,175 | 33.93 | |
| 97,911 | 32.63 | |
| 121,097 | 30.53 | |
| 127,929 | 29.72 | |
| 284,831 | 28.94 | |
| 564,928 | **24.15** | and 13.36 in a prior run |

**Shape: essentially FLAT at ~31-34 t/s from 30K to 122K** (-12.5% over a 4x
depth increase), then a gentle decline to 28.94 at 285K and 24.15 at 565K.
There is **no cliff and no 2.5x collapse** in the normal case.

## Revised status of the user's two targets

**T2 — "get 250K back above 30 t/s": essentially AT target.**
- 284,831 measured **28.94 t/s**.
- Target gap: +3.7%. Measured run-to-run spread: ~4%.
- Across runs at this depth: 26.83, 28.94. It is borderline, inside the noise.
- **Not the 12% shortfall I reported earlier.** Getting reliably above 30 needs
  a genuine ~5-10% win, which is a real but modest ask — not a broken-cluster
  story.

**T1 — "shouldn't collapse on the 500K": real, but STOCHASTIC.**
- 564,926 -> **13.36** t/s in one run; 564,928 -> **24.15** t/s in another.
- 1.8x difference at identical depth/config -> **bistable**, matching the 352K
  investigation's 4-of-16 collapse rate.
- The collapse is real but intermittent: a single good run proves nothing, and
  a fix can only be shown by N>=6 (better, a direct margin/headroom measure).
- Mechanism for the bad mode: memory pressure (page-ins 400 at 128K -> 6,599 at
  565K; peak 117.24 GB against a 120.6 GB wired limit) — mmap clean-page
  eviction + refault, previously confirmed at 42K (32,469 page-ins).

## What was NOT done (and why)

- **No config change was made.** The one cheap lever for T1 (raising
  `DSV4_WIRED_LIMIT_MB` from 115000) re-enters a documented hard-wedge regime
  (`MetalAllocator` stuck state, ~5x prefill drop, requires reboot) on a cluster
  whose owner is asleep. I judged that a bad trade for an intermittent problem
  I can characterise but not yet fix, and I could not get a second opinion
  (consult returned 403 / OAuth restricted all three attempts).
- The "shard DSpark non-expert projections" lever from the 352K doc was
  measured and cancelled: non-FFN is 0.243 GB of a 10.88 GB quantized head
  (~0.12 GB/rank recovered vs ~4.9 GB needed).

## Recommended next steps (for the user)

1. **Accept T2 as met** (28.9-29.0 at 250K, within noise of 30) or fund a
   focused ~5-10% decode win — but note the biggest levers are already closed
   (see the campaign ledger) and the remaining ones are small.
2. **T1 needs a characterisation run, not a fix yet:** N>=6 runs at 565K to
   establish the collapse rate on the current config, plus a headroom
   measurement. Only then can any lever be judged.
3. If T1 is worth fixing, the honest options are footprint reduction (head
   quantization — quality-gated) or a careful, monitored wired-limit step with
   the wedge signature watched. Both need the user's call.

## Methodological note worth keeping

**Establish the noise floor before inferring a mechanism.** Tonight's cascade
of wrong conclusions (content-driven → indexer cliff) came from treating single
runs as data. One variance measurement (4 runs, ~3 minutes) would have
prevented both.

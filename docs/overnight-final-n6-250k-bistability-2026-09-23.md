# Overnight FINAL RESULTS (n=6 at 250K) — T2 measured, bistability confirmed at 250K (2026-09-23)

## The definitive T2 measurement: N=6 at ~250K depth

Same probe, same prompt, same config, 6 consecutive runs:

| run | depth | decode tps | needle |
|---|---|---|---|
| 0 | 290,436 | 27.43 | ✓ |
| 1 | 276,427 | 26.70 | ✓ |
| 2 | 270,821 | 26.25 | ✓ |
| 3 | 276,424 | **19.46** | ✓ |
| 4 | 273,623 | 28.74 | ✓ |
| 5 | 276,424 | 28.49 | ✓ |

**mean 26.18, stdev 3.43, min 19.46, max 28.74 — spread 47.7%.**

## What this settles

### T2 ("get 250K back above 30 t/s"): REAL shortfall, but not a stable number

- Mean is **26.18**, so the gap to 30 is ~13%.
- **But one run in six came in at 19.46** — a partial collapse, at 250K, not
  500K.
- The spread (19.46-28.74) is dominated by the collapse mode, not by ordinary
  jitter.

**So the honest statement is: 250K typically delivers ~26-29 t/s, with an
intermittent collapse mode that drops it to ~19.** It is not that "250K is 11%
slow"; it is that **the collapse mode is present at 250K too, just less often
or less severely than at 565K.**

This connects T1 and T2 into ONE problem rather than two:

| depth | normal tps | collapse tps | measured |
|---|---|---|---|
| 30K | 34.89 (n=4, sd 0.64) | — | tight, no collapse seen |
| 122K | 30.53 | — | tight |
| 250K | 26.2-28.7 (n=5 of 6) | **19.46** (1 of 6) | **bistable** |
| 565K | 24.15 | **13.36** | **bistable** |

- At 30-120K: stable, tight variance.
- From ~250K: a bimodal distribution appears (normal mode + collapse mode).
- The collapse mode gets more frequent / more severe with depth.

**This reframes BOTH user targets as one root cause: the collapse mode.**

### T1 ("shouldn't collapse at 500K"): confirmed, and it starts by 250K

13.36 at 565K and 19.46 at 276K are the same phenomenon. Fixing it fixes both.

## Mechanism (as established tonight)

- **Trigger:** memory pressure. At 565K peak hit 117.24 GB against a 120.6 GB
  wired limit (~97%), and page-ins went 400 (128K) -> 6,599 (565K).
- **Signature:** mmap clean-page eviction + refault of model weights
  (pageouts=0, decompressions~0 — not swap, not compressor). Confirmed
  directly at 42K depth when memory was tighter: 32,469 page-ins in one request.
- **Why bimodal:** whether eviction starts depends on the margin at decode
  onset. Once it starts it is self-sustaining (every cycle re-faults). Runs
  that start under the ceiling stay healthy. This is the bistable equilibrium
  the 352K doc described (4/16 collapses there).
- **Not** the indexer top-k threshold (falsified tonight: no step at the
  predicted 65,536 crossover; ladder was flat 31-34 t/s from 45K to 122K).
- **Not** acceptance alone (acceptance actually *recovers* in the pressure
  region: 0.814 at 128K -> 1.353 at 565K).

## Levers, honestly assessed

**Cancelled by measurement:**
- "Shard DSpark non-expert projections": non-FFN is 0.243 GB of a 10.88 GB
  quantized head; recovers ~0.12 GB/rank vs ~4.9 GB needed. 50x too small.
- `EXO_DSV4_MTP_DRAFT_LMHEAD_BITS=4`: -8.7% at depth (sign flips vs short ctx).
- `EXO_DSV4_DSPARK_CONF_TAU`: prior sweep showed 0.5 already at plateau.
- Indexer top-k: no measurable effect at these depths (falsified).

**Remaining, with risk stated:**
1. **Raise `DSV4_WIRED_LIMIT_MB`** (115000 -> e.g. 120000). Direct headroom.
   **Risk:** re-enters the documented `MetalAllocator` stuck-wedge regime
   (prefill ~5x drop, needs reboot). NOT done tonight — no second opinion
   available (consult 403) and the owner was asleep.
2. **Reduce per-rank footprint** — head quantization (quality-gated) or paging
   the DSpark head. Larger work.
3. **Characterise before fixing:** N>=10 at 565K and 250K on current config to
   establish the collapse RATE, plus a direct margin measurement (free GB at
   decode onset). Without a rate, no fix can be shown to work.

## Methodological lessons (this session's cascade of wrong conclusions)

1. **Establish the noise floor BEFORE inferring mechanisms.** I built two
   hypotheses (content-driven; indexer cliff) on single runs. One 4-run
   variance measurement would have prevented both.
2. **Never compute tok/s from wall clock** — use the probe's decode_tps.
3. **Depth ladders need N>=3 per rung** when the collapse is bistable, or the
   curve is unresolvable.
4. **A single good run proves nothing** for a bistable phenomenon; conversely a
   single bad run does not prove a regression.

## Cluster state at end
- Production config, all defaults (VERIFY_ROWSEQ_VEC=1, VERIFY_BATCH=1,
  DSPARK_TP_SHARD=1, ATTN_ALLSUM=0), wired limit 115000 unchanged.
- 2/2 RunnerReady, 0 pending, end-to-end completion verified.
- No config changes were made tonight.

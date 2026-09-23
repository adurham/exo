# PRE-REGISTRATION: adaptive prefill chunk shrink at deep context (2026-09-23)

Written BEFORE the measurement. Bands apply verbatim; no post-hoc adjustment.

## Change under test

Two env vars in `start_cluster.sh`, previously shipped EMPTY (feature disabled):
```
EXO_PREFILL_STEP_SIZE_HIGH_CTX=128     (was: unset -> stays at 2048)
EXO_PREFILL_STEP_SIZE_CROSSOVER=200000 (was: unset -> no crossover)
```
Effect: prefill chunks shrink from 2048 to 128 once context passes 200K.

## Hypothesis

The deep-context collapse is driven by total footprint exceeding the 115.4 GB
wired limit. Reconciled budget at 638K: ~104 GB fixed + 16.63 GiB retained
leaves + ~9 GB prefill transient. The (B,H=64,L,P) indexer score transient
scales with chunk size L. Shrinking L from 2048 to 128 should shrink the
transient, lowering PEAK, and should also improve prefill THROUGHPUT at depth
(the code's own measurement: 256-chunk is -30% vs 128 past 380K).

## Test

One run: `bench/long_decode_probe.py 565000 --max-tokens 250`, after a relaunch
with the change. Compare against the two same-config control runs measured
minutes earlier (fix live, no chunk change):

| metric | control run 0 | control run 1 |
|---|---|---|
| depth | 619,462 | 638,449 |
| decode tps | 13.12 | 12.04 |
| page-ins | 66,003 | 39,930 |
| peak | 117.88 GB | 123.80 GB |
| before active | 107.32 GB | 114.67 GB |

## PRE-REGISTERED BANDS (apply verbatim)

**SUCCESS (any one is a win):**
- `peak_gb` <= 115.4 (under the wired limit) — the primary bar, since the peak
  crossing the limit is the measured cause of the refault collapse; OR
- `pageins` < 20,000 (at least 2x better than the 39,930 best control); OR
- `decode_tps` >= 18.0 (clearly above the 12.04-13.12 collapse band, showing the
  request no longer collapses).

**PARTIAL:**
- `peak_gb` improves by >= 4 GB but stays above 115.4, AND `pageins` improves
  but stays >= 20,000, AND tps remains in the collapse band. -> transient is real
  but not the dominant term; keep the change (it is free at depth) and move to
  the fixed-footprint or wired-limit question.

**FAIL:**
- `peak_gb` within +-2 GB of control AND `pageins` >= 39,930 AND
  `decode_tps` <= 14 -> the transient is not the driver. REVERT the two env vars
  and record the negative result.

**Also required:** `needle_hit` must be True (output quality gate). If the
needle is missed, the run is INVALID regardless of the other numbers — chunk
size must not break correctness.

## Prefill throughput as a secondary read (not a pass/fail bar)

Control prefill degraded to ~214-227 tok/s at depth (from ~350 at low ctx). A
rise toward 300+ would confirm the -30% claim in this regime. Recorded, not gated.

# P6: Cross-rank all_sum rank0 tail-straggler — aggregate decode-time impact quantified — 2026-08-22

## Question

Follow-up to `docs/cross-rank-allsum-skew-2026-08-22.md` (T4): the rare
severe straggler events (>1ms cross-rank transport-time difference,
4.2x rank0-leaning) were left unquantified in aggregate. Exit
criterion: compute expected aggregate decode-time impact from
frequency × magnitude; if under ~0.5% of total decode time, log the
calculation and stop.

## Method — pure recomputation from the existing real traces, no cluster time

Reused the same real production traces
(`/tmp/jaccl_trace_rank0.log`/`rank1.log`, jaccl-internal
`steady_clock` timing of `reliable_all_reduce_v2`, 45,666 matched
8192-byte decode `moe.all_sum` call pairs). Calculation script:
`/tmp/p6_skew_calc.py`.

The effective per-call cost to the decode step is `max(rank0, rank1)`
(the collective completes only when both ranks arrive). The
skew-attributable excess per call is `max − min` — the time the fast
rank spends idle waiting. Summing:

| quantity | value |
|---|---|
| matched 8192B pairs | 45,666 |
| severe events (>1ms diff) | 93 (75 rank0-straggler, 18 rank1) |
| total effective all_sum time (Σ per-call max) | 4.167 s |
| severe-straggler excess (Σ max−min over severe calls) | 0.595 s |
| severe excess as % of all_sum effective time | 14.3% |
| implied decode wall for this traffic | ~35.5 s (cross-checked two ways: 45,666/43 layers ≈ 1062 tokens at ~30 tok/s → 35.4 s; and all_sum-as-11.7%-of-wall (§2 of PERFORMANCE_HISTORY) → 4.167/0.117 ≈ 35.6 s — the two independent estimates agree) |
| **severe-straggler excess as % of TOTAL decode time** | **~1.7% (upper bound)** |
| rank0-attributable portion (75/93 events) | ~1.3% (upper bound) |
| ALL skew (Σ max−min over every pair) as % of decode | ~7.4% (upper bound, mostly compute-arrival offset — NOT removable, see below) |

## Interpretation — honest bounds

**The exit-criterion threshold (~0.5%) is EXCEEDED at the upper
bound: severe straggler events represent up to ~1.7% of total decode
time (~1.3% attributable specifically to the rank0-leaning tail).**

However, this is an upper bound, not an expected recoverable win:

1. `max − min` counts the fast rank's idle wait as fully recoverable.
   That's only true if the straggler's lateness is caused by removable
   overhead (e.g. rank0 control-plane work stealing time). If it's a
   scheduling/arrival offset where the "late" rank was doing useful
   compute, the collective cost is unchanged by any fix — the 7.4%
   "all skew" row is the reductio: most of that is just per-step
   arrival jitter, obviously not a lever.
2. Severe events are 93/45,666 ≈ 0.2% of calls at ~6.4ms mean excess.
   The mechanism is unknown (never root-caused; candidates in the T4
   doc: rank0's master/control-plane role, hardware asymmetry,
   upstream scheduling).
3. The traces cover one workload window; tail rates this small have
   real sampling variance.

## Decision

Logged per the timebox. The realistic recoverable range is
0–1.3% of decode time depending on root cause, straddling the 0.5%
threshold. This does NOT justify blowing the P6 timebox now, but the
honest statement is: **this is not conclusively below-threshold** — a
future bounded investigation (correlate severe-straggler timestamps
against rank0 control-plane activity, e.g. master heartbeat/event
indexing) is justified if decode-side wins are ever needed again.
Outcome class: (c) genuine finding with real evidence, quantified,
no action taken within timebox by design.

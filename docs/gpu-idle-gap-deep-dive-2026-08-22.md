# GPU idle-gap deep dive: real gap-length distribution, occupancy/power/clock reconciliation, and cross-rank symmetry — 2026-08-22 (session 3, offline analysis of existing data + one live read-only powermetrics check)

## Why this investigation

Following a detailed Fable-provided investigation plan for attributing
decode's ~65-85% unexplained wall time (`docs/decode-time-budget-synthesis-2026-08-22.md`),
this document covers Steps 1 (partial, using existing data) and 6
(GPU frequency check) from that plan, plus a real discrepancy that
emerged between two measurements and required its own reconciliation.

## Part 1: reconciling an apparent GPU-idle contradiction

**Observed apparent contradiction**: earlier tonight's Instruments trace
found ~29-30% real GPU occupancy (§2.7 of `docs/PERFORMANCE_HISTORY.md`,
i.e. ~70% "idle" by duration-union accounting). A fresh, live
`sudo powermetrics --samplers gpu_power` check taken during a real
decode request (confirmed overlapping via background-process poll, not
assumed) showed **"GPU idle residency: 0.00%"** and **"GPU HW active
residency: 100.00%"** on every sample, with GPU frequency 819-1122 MHz
(well below the M4 Max's ~1.5+ GHz public peak spec) and power draw
4.6-7.1W.

**Reconciled via Fable consult**: these measure genuinely different
things and are not actually contradictory. `powermetrics`' "HW active
residency" is a power-STATE metric (is the GPU power-gated or not),
gated by a millisecond-scale hysteresis threshold — it will read 100%
active as long as gaps between GPU work never get long enough to
trigger power-gating, regardless of how much real work is happening in
that "active" window. The Instruments trace's occupancy figure is a
real work-duration accounting (union of actual command execution
intervals) — a fundamentally different, and for this question more
relevant, measurement.

**The power reading is the tiebreaker, and it sides with the Instruments
trace's "~30% real occupancy" figure, not the naive 0%-idle reading**:
4.6-7.1W is far below what a genuinely GPU-saturated bandwidth-bound
workload on this hardware should draw. A "100% active, but drawing only
~5-7W and running at little more than half peak clock" GPU is the
classic signature of bursty, low-average-load work — never power-gated,
but also never doing continuous real work.

**On the reduced clock frequency (819-1122 MHz)**: real, but assessed as
a downstream SYMPTOM of the same bursty-low-load pattern, not an
independent root cause. Apple's GPU DVFS ramps clock in response to
sustained queue pressure; a workload with frequent sub-millisecond-to-
multi-millisecond stalls never presents that sustained pressure, so the
governor never ramps. Fixing the underlying gap pattern should raise
clocks as a side effect, not require separate DVFS-specific work.
**However, while the gaps persist, reduced clock IS a real multiplicative
amplifier of the roofline gap**: kernels that do run, run at reduced
throughput. A rough decomposition check: 0.30 (real occupancy) × 0.65
(rough clock fraction, ~950MHz of ~1.5GHz peak) ≈ 0.20 — but the actual
measured roofline efficiency is ~0.12 (§4.3,
`docs/decode-roofline-dispatch-bound-2026-08-21.md`). **This gap (0.20
predicted vs 0.12 actual) points to a real, still-unidentified THIRD
factor** — likely per-kernel bandwidth efficiency below 100% even when
the GPU is actively executing (plausible causes: small GEMV/decode-shape
matmuls not saturating memory bandwidth per dispatch, or launch-tail
effects) — not yet investigated further this session.

## Part 2: real gap-length distribution (both ranks)

Computed from the already-captured Instruments trace data
(`docs/live-decode-two-rank-instruments-trace-2026-08-21.md`'s raw
`metal-gpu-intervals` XML, still on disk locally), using a proper UNION
of overlapping GPU command intervals (not a naive sum — verified this
produces the same ~30% occupancy figure as the original analysis,
confirming the methodology is sound):

| | rank0 (pid 53337) | rank1 (pid 61349) |
|---|---|---|
| Real merged (union) intervals | 7,217 | (not separately counted) |
| Union busy time | 5,362.83ms | 6,195.24ms |
| Trace window span | 17,629.45ms | 21,805.95ms |
| **Real occupancy** | **30.42%** | **28.41%** |
| Real gap count | 7,216 | 7,959 |
| Median gap | 528.33µs | 520.08µs |
| Mean gap | 1,699.92µs | 1,961.39µs |
| p95 gap | 7,417.96µs | (not computed) |
| Max gap | 74,717.38µs | (not computed) |

**Gap-length bucket distribution (both ranks, near-identical shape)**:

| Bucket | rank0 count (% of gaps) | rank0 time (% of gap-time) | rank1 count (% of gaps) |
|---|---|---|---|
| <50µs | 1,991 (27.6%) | 4.76ms (0.0%) | 2,247 (28.2%) |
| 50-500µs | 1,455 (20.2%) | 437.87ms (3.6%) | 1,618 (20.3%) |
| 0.5-5ms | 3,171 (43.9%) | 5,244.26ms (42.8%) | 3,299 (41.5%) |
| 5-20ms | 551 (7.6%) | 5,035.58ms (41.1%) | 709 (8.9%) |
| >20ms | 48 (0.7%) | 1,544.15ms (12.6%) | 86 (1.1%) |

**Real, notable finding: gap TIME is dominated by the 0.5-20ms buckets
combined (83.9% of all gap time), NOT the 50-500µs range** that would be
the classic signature of pure per-kernel CPU-dispatch latency. The
0.5-5ms and 5-20ms buckets are each roughly 10-100x longer than the
directly-measured real jaccl transport cost (36-66µs median/mean,
tonight's earlier finding) — ruling out the collective's wire cost
itself as the source of these specific gaps (already established, but
this independently corroborates it from a different angle).

**Cross-rank shape symmetry** (real, though clock-sync-limited, see
caveat below): both ranks show nearly identical bucket-count
percentages (e.g. 0.5-5ms bucket: 43.9% rank0 vs 41.5% rank1; 5-20ms:
7.6% vs 8.9%). This symmetry is itself a real, informative signal —
consistent with the gap pattern being a structural property of each
rank's own decode-loop code (same model, same code path, running
independently), rather than an asymmetric single-rank straggler effect.

## What this does NOT establish (honest limitations)

**Per-token gap-rate arithmetic was attempted but is inconclusive, not
a clean confirmation.** A naive per-token gap count (~20-22 gaps/token,
computed by dividing the trace window by the real 53.48ms/token decode
rate) does not cleanly match the real 43-collectives/token figure that
would strongly confirm "one gap = one layer's `moe.all_sum` fence." This
is NOT strong evidence against the per-layer-fence hypothesis — some
layers' fence-related stalls may be too short to register as a
separately-merged gap in this data, or the trace window's exact
prefill/decode boundary (estimated, not precisely known from this data
alone) may be distorting the count. This remains genuinely open,
not resolved either way.

**Cross-rank wall-clock correlation (Fable's most-recommended
disambiguating test — "does rank0's gap overlap rank1's BUSY interval,
indicating skew-wait, or are both idle simultaneously, indicating
shared CPU-side overhead") was NOT performed.** The two traces were
captured on separate physical machines (m4-1, m4-2) with independent,
non-synchronized system clocks — already flagged as a known limitation
in the earlier `docs/live-decode-two-rank-instruments-trace-2026-08-21.md`.
True timestamp-level cross-rank correlation requires clock
synchronization (e.g. an NTP-style ping-pong measurement, or a
simultaneous marker event visible in both traces) that was not set up
during tonight's capture. Only aggregate per-rank statistics (which
don't require synced clocks, and are reported above) can be honestly
compared — the bucket-shape symmetry is real and informative, but it is
NOT the same as a true overlap/skew determination.

## Disposition and next steps

Real progress: ruled out pure CPU-dispatch-latency as the dominant gap
cause (the 50-500µs bucket is small), reconciled an apparent
measurement contradiction with a real technical explanation rather than
picking one number and discarding the other, and established that both
ranks show a structurally similar gap pattern.

Still genuinely open, per Fable's plan: (1) attribute gaps by
surrounding kernel identity — bracketing each gap by the preceding/
following GPU event to determine if 0.5-5ms gaps specifically align
with per-layer boundaries (requires per-kernel labels, which tonight's
existing capture lacks — see the companion finding in this
investigation that `metal-gpu-intervals` provides only command-buffer-
level, not named-kernel-level, granularity); (2) a genuine clock-synced
cross-rank capture to test the skew-vs-shared-overhead hypothesis
cleanly; (3) the still-unidentified ~0.20-predicted-vs-0.12-actual
roofline-efficiency gap (Part 1) — a real, separate, smaller mystery
layered on top of the larger idle-time mystery.

# Step 7: interim synthesis — decode's unexplained wall time, real progress and honest remaining gaps — 2026-08-22 (session 3)

## Purpose

Closes out this session's execution of a detailed Fable-provided
investigation plan for attributing decode's large gap between real
wall-clock throughput and the theoretical bandwidth-bound roofline.
5 of 7 planned steps completed with real findings; 2 remain genuinely
open pending either a tool-install approval or a more involved live
capture. This is an honest INTERIM synthesis, not a final resolution —
written now rather than deferred, since real, useful progress was made
and shouldn't sit undocumented waiting for the remaining steps.

## What's ruled out (real, verified negative results)

1. **`moe.all_sum` (the RDMA collective) is NOT the cause.** Real
   jaccl-internal `steady_clock` timing on 45,666 live decode-time
   calls: median 36µs, mean 58-66µs — faster than the isolated wire-floor
   microbenchmark itself. Only 2.9-5.3% of real wall time.
   (`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`,
   `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`)
2. **Memory residency / expert-weight paging from disk is NOT the
   cause.** Real pageins delta across a full decode request: 1.0MB,
   zero swapins. Model shard confirmed fully resident (87.4GB, matching
   the expected ~83.5GB TP=2 shard) via `vmmap --summary`.
   (`docs/memory-residency-check-ruled-out-2026-08-22.md`)
3. **Pure CPU-dispatch-latency (the 50-500µs signature) is NOT the
   dominant cause.** Real gap-length distribution shows only 3.6% of
   gap TIME in that bucket; 83.9% is in the 0.5-20ms range — 10-100x
   longer than either the real transport cost or typical dispatch
   overhead. (`docs/gpu-idle-gap-deep-dive-2026-08-22.md`)
4. **The historic sync-span-profiler-derived "21.4% of decode wall
   time" figure for `moe.all_sum` was a methodology artifact**, not a
   real cost — confirmed via real arithmetic (43 layers × 4094µs would
   be 176ms/token, exceeding the real 53.48ms/token wall time by 3.29x,
   a mathematical impossibility).
   (`docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md`)

## What's confirmed real (positive findings, not yet fully explanatory)

1. **Real GPU occupancy is ~28-30% on both TP ranks** during live
   decode (Instruments `metal-gpu-intervals`, real hardware telemetry —
   not a Python-timer artifact, confirmed immune to the lazy-eval
   measurement trap that corrupted the sync-span figures).
2. **GPU clock frequency is genuinely reduced during decode** (819-1122
   MHz vs ~1.5GHz+ peak) — assessed as a real but likely DOWNSTREAM
   symptom of the same bursty-low-load pattern (DVFS doesn't ramp
   without sustained queue pressure), not an independent root cause,
   though it's a real multiplicative amplifier while the underlying
   pattern persists.
3. **A real, smaller sub-mystery remains even after crediting both
   occupancy and clock reduction**: a rough decomposition
   (0.30 occupancy × 0.65 clock-fraction ≈ 0.20) doesn't fully explain
   the measured 0.12 roofline efficiency — roughly 0.08 of efficiency
   is still unaccounted for, likely in per-kernel bandwidth efficiency
   (small-shape GEMV underutilization, launch-tail effects), not
   further investigated this session.
4. **The real per-token idle/unattributed time is 82.5-84.9%** of wall
   time — a sharper number than the session-opening "~65-85%" estimate,
   derived by stacking two independently-verified real measurements
   (real collective cost + theoretical compute floor) against real wall
   time rather than relying on a single instrument.
   (`docs/decode-time-budget-synthesis-2026-08-22.md`)

## What's still genuinely open (not resolved, not glossed over)

1. **True kernel-level attribution.** The existing Instruments trace
   data (already captured, analyzed extensively this session) provides
   only command-buffer-level granularity — no per-kernel MLX operation
   labels (`gather_qmm`, `switch_mlp`, `all_sum`, etc.). A fresh capture
   with a different Instruments template/config is needed to determine
   WHICH specific kernel(s) or code region(s) the dominant 0.5-20ms
   gaps correspond to. Not attempted this session (would need a live
   capture cycle; deferred rather than rushed).
2. **True cross-rank skew correlation.** Both ranks show similar
   gap-length DISTRIBUTION shapes (a real, if limited, signal), but
   true timestamp-level correlation (does rank0's gap overlap rank1's
   BUSY interval, indicating skew-wait, vs. both idle simultaneously,
   indicating shared CPU-side overhead) requires clock-synchronized
   captures on both nodes, which tonight's traces do not have.
3. **CPU-side Python/MLX profiling** (Fable's Step 2, py-spy) — not
   attempted; `py-spy` is not installed on the cluster nodes, and per
   the user's standing rule against installing software without
   explicit approval, this was deferred rather than installed
   unilaterally.
4. **The per-token gap-rate check was inconclusive**, not a clean
   confirmation OR refutation of the per-layer-fence hypothesis
   (~20-22 measured gaps/token vs. 43 real `moe.all_sum` calls/token —
   plausibly explained by some layers' stalls being too short to
   register as separately-merged gaps in the existing data, but this
   is a plausible explanation, not a confirmed one).

## Honest overall assessment

This session made real, verifiable progress narrowing the search space
for decode's dominant cost — ruling out three genuine candidates
(the collective, memory paging, pure dispatch latency) with hard
evidence, and reconciling one real apparent measurement contradiction
(GPU occupancy vs. powermetrics idle-residency) with a correct
technical explanation rather than picking a number and discarding the
other. The remaining mystery is now better-bounded: **something in the
0.5-20ms-per-gap range, occurring roughly 20+ times per token, on both
ranks with similar (but not yet timestamp-correlated) patterns, is
responsible for the bulk of decode's real cost.** The two most
promising next steps — a properly-labeled fresh Instruments capture,
and either a py-spy CPU profile or a clock-synced cross-rank capture —
were correctly identified but not completed this session, each for a
real, stated reason (tool-install approval, capture complexity) rather
than being silently skipped.

## Next steps for a future session

In priority order per the original Fable-provided plan and this
session's findings:
1. Fresh Instruments capture with a template/config that preserves
   per-kernel MLX operation labels (may require setting Metal debug
   labels on MLX's dispatch calls, or using a different xctrace
   template than "Metal System Trace" — needs investigation of what
   Instruments templates actually expose named kernel identity for
   this GPU stack).
2. Get explicit approval to install `py-spy`, then run it during a real
   decode request to see where CPU time (not GPU time) goes per layer —
   directly tests the "Python-level MoE routing/graph-construction
   overhead per layer" hypothesis that's consistent with the observed
   0.5-5ms gap population.
3. A clock-synchronized two-rank capture (e.g. an NTP-style ping-pong
   measurement between the nodes just before/during the trace, or a
   simultaneous marker event visible in both traces) to properly test
   the skew-vs-shared-overhead question — the single most discriminating
   test per Fable's original guidance, still not performed.

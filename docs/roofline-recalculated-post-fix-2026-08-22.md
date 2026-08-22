# Roofline recalculated post-fix: real headroom confirmed to persist — 2026-08-22 (Phase A, post-fence-fix investigation)

## Why this check

Per Fable's ranked recommendation following the async-fence fix
(`docs/async-fence-fix-validated-2026-08-22.md`): the original roofline
figure (~12% of theoretical bandwidth-bound peak,
`docs/decode-roofline-dispatch-bound-2026-08-21.md`) was computed
against the OLD, silently-broken-fence baseline (18.3 tok/s). Since
decode throughput changed by +23-67% depending on depth, that
percentage is stale and needs recalculation before any further
optimization-headroom claim is trusted. Pure arithmetic, zero cluster
risk, using the exact same methodology and inputs as the original
calculation (only the observed throughput changes).

## Method (unchanged from the original)

Bandwidth-bound roofline floor: 3.56 GB active-weight-bytes/token/node
÷ 546 GB/s (M4 Max public unified-memory-bandwidth spec) = **6.51
ms/token/node**. This figure is compute/bandwidth-side and is NOT
affected by the fence fix (the fix only changes decode-time
synchronization behavior, not how many bytes get read per token) — it
carries over unchanged from the original calculation.

## Real recalculated result

| Depth | Real decode tok/s (post-fix) | ms/token | % of theoretical peak | Slower than roofline by |
|---|---|---|---|---|
| 100K | 26.91 | 37.16 | **17.5%** | 5.71x |
| 300K | 24.44 | 40.92 | **15.9%** | 6.29x |
| 500K | 21.51 | 46.49 | **14.0%** | 7.14x |
| Short context (512-2000 tok prompt) | 29.2-31.1 | 32.15-34.25 | **19.0-20.2%** | 4.94-5.26x |

**Comparison to the pre-fix figure**: 18.3 tok/s → 54.64 ms/token →
11.9% of peak (matches the original doc's "~12%" figure, confirming
methodology consistency between the two calculations).

**Efficiency improved from ~11.9% to 14.0-20.2% depending on depth/context
shape** — a real, direct consequence of the async fence now genuinely
engaging (less time blocked in synchronous `mx.eval`, more time
available for actual useful work per unit wall-clock).

## Interpretation

**Real, substantial headroom still remains** — even the best post-fix
case (short-context, 20.2% of peak) is still running at less than 1/5
of the theoretical bandwidth-bound ceiling, a genuine ~5x gap. This
directly confirms Fable's prediction ("still likely ~15-20% of peak")
made before this calculation was run, and rules out "the fence fix
closed the gap, nothing left to find" as a premature conclusion.

The original roofline doc's dispatch-overhead hypothesis (the
48.13ms/token gap being consistent with ~320 dispatches/token ×
~150µs/dispatch) used the OLD 48.13ms gap figure. Recalculating that
same dispatch-count implication against the new, smaller gaps:

- 100K: gap = 37.16 - 6.51 = 30.65ms → ~204 dispatches/token at 150µs
- 300K: gap = 40.92 - 6.51 = 34.41ms → ~229 dispatches/token at 150µs
- 500K: gap = 46.49 - 6.51 = 39.98ms → ~267 dispatches/token at 150µs
- Short context: gap = 32.15-34.25 - 6.51 = 25.64-27.74ms → ~171-185
  dispatches/token at 150µs

All of these remain broadly consistent with the original doc's
independent per-layer op-count estimate (~430-650 dispatches/token for
the full 43-layer forward, though that count was for ALL layers'
combined op inventory, not literally comparable 1:1 to this back-of-envelope
implied-dispatch-count check — this is a rough consistency check, not a
precise re-derivation). The persisting gap is consistent with real
dispatch/synchronization overhead still being present, now smaller in
absolute terms post-fix but not eliminated.

## Disposition

Confirms real, actionable headroom remains post-fix. Proceeds to Phase
B (GPU clock symptom test) and Phase C (fresh dual CPU+GPU idle-gap
capture) per the planned investigation — the old Instruments trace data
and gap-length distribution are now stale (measured under the broken
fence) and need re-capturing before trusting any further attribution
claim.

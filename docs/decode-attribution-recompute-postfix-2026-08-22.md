# T1: Decode wall-time attribution recomputed against post-async-fence-fix baseline — 2026-08-22 (session 4)

## Why this check

Following the async-fence fix (§2.8/§13, decode 18.5 -> 29.2-30.9 tok/s),
the previously-published "82.5-84.9% of decode wall time unattributed"
figure (`docs/decode-time-budget-synthesis-2026-08-22.md`) is stale --
it was computed against the pre-fix ~18.7 tok/s baseline. Recomputing
against the new validated baseline before any further investigation,
per an independent Fable consult that flagged this as the mandatory
(but small, ~30min) first step.

## Recomputed table

Using the same two real, already-measured quantities as before
(roofline compute floor 6.51ms/token, real jaccl all_sum transport cost
43 calls/token x 36-66us median/mean = ~2.19ms/token combined estimate)
against the NEW real wall times:

| Context | tok/s | wall ms/tok | all_sum ms | roofline ms | unattributed ms | unattributed % |
|---|---|---|---|---|---|---|
| short ctx (low end) | 29.20 | 34.25 | 2.19 | 6.51 | 25.54 | 74.6% |
| short ctx (high end) | 31.10 | 32.15 | 2.19 | 6.51 | 23.45 | 72.9% |
| 100K ctx | 26.91 | 37.16 | 2.19 | 6.51 | 28.46 | 76.6% |
| 300K ctx | 24.44 | 40.92 | 2.19 | 6.51 | 32.21 | 78.7% |
| 500K ctx | 21.51 | 46.49 | 2.19 | 6.51 | 37.79 | 81.3% |

## Real conclusion

Headline changes from ~83-85% to **~73-81% depending on context depth**,
but the core conclusion survives unchanged: the large majority of
decode's real per-token wall time is still neither the collective nor
the theoretical compute floor. Per Fable's framing: the investigation
has now flipped from "why is the GPU idle" (the pre-fix question, since
occupancy jumped 29-30% -> 85.42% with the fence fix) to **"why is
GPU-busy time still ~4-5x the roofline floor"** (~28-40ms busy vs
6.51ms floor) -- this is now a kernel-efficiency / achieved-bandwidth
question, not a dispatch-gap question.

## Real methodology flaw identified, not yet resolved

The 6.51ms roofline floor used above is **active-MoE-expert-weight-bytes
only** (13B active params x 0.588 effective bytes/param / TP=2 / 546GB/s
peak bandwidth) -- it does NOT include KV-cache/attention read cost,
which is fixed per-token at short context but GROWS with context depth
under this cluster's sparse/compressed (MLA-style, numKeyValueHeads=1)
attention. This means the 500K-ctx unattributed % (81.3%) is likely
**overstated** -- the real achievable compute floor at deep context is
higher than 6.51ms once KV-read cost is properly counted, so the true
unattributed remainder there is smaller than shown. This directly
matches the context-scaling question Fable flagged as a missing
investigation axis (29.2 -> 21.5 tok/s, ~26% degradation, presumably
attention/KV-cost-driven) -- queued as T5 in the active task list, not
resolved here. Do not treat the 500K-ctx unattributed % above as final;
short-context and 100K figures are less affected since KV-read cost is
smaller there relative to the MoE weight-read floor.

## Next step

T2: fresh post-fix triple-purpose capture (per-kernel Metal capture +
xctrace timeline + concurrent powermetrics clock) at short context, to
directly attribute the real ~24-28ms of post-fix busy-plus-idle time
that current back-of-envelope arithmetic can't further decompose.

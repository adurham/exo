# Arithmetic reconciliation: the sync-span 4094us/call figure was a methodology artifact, not a real cost — 2026-08-22 (session 3, offline analysis)

## Why this check, and why it's zero-risk

Per an independent Fable consult after the jaccl-internal timing result
(median 36µs, real transport confirmed fast): before doing ANY further
live-cluster instrumentation (which would require a relaunch, and the
user has gone to bed with no one available to approve or notice a
degradation), do a pure arithmetic reconciliation using numbers already
in hand. This requires zero cluster access, zero risk, and — per
Fable's own framing — might close out the remaining open question
entirely on its own.

**It did.**

## The check

Real, validated production baseline (confirmed live moments before this
analysis, clean config, no diagnostic overhead):
- Decode throughput: 18.7 tok/s → **53.48ms real wall time per token**
- 43 layers, `EXO_DSV4_ATTN_ALLSUM=0` confirmed live (attention
  collectives disabled at decode) → exactly 43 `moe.all_sum` calls per
  token, one per layer.

**If the earlier sync-span-measured average of ~4094µs/call
(`docs/moe-allsum-collective-cost-confirmed-2026-08-21.md`) were a real
per-call cost that sums across layers:**

```
43 calls × 4094µs = 176.0 ms/token (predicted)
vs.
53.48 ms/token (real, measured wall time)

Ratio: 3.29x
```

**This is mathematically impossible as a real per-call cost.** It would
require `moe.all_sum` alone to consume 3.29x the ENTIRE real per-token
decode budget — leaving negative time for attention, the rest of the
MoE compute, sampling, everything else that demonstrably also happens
every token. A per-call cost cannot exceed the total wall-clock budget
it's a component of, let alone by 3.29x.

**Using tonight's real jaccl-internal measured cost instead:**

```
43 calls × 36.1µs (median) = 1.55 ms/token
43 calls × 66.3µs (mean)   = 2.85 ms/token

As a fraction of real 53.48ms/token wall time: 2.9% - 5.3%
```

This is a small, entirely plausible, non-dominant fraction — fully
self-consistent with everything else measured tonight, and with
`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`'s
conclusion that the real transport is fast.

## What this means: correcting two earlier claims

**The sync-span-measured "~4094µs average" and "moe.all_sum = 21.4% of
decode wall time" figures (`docs/moe-allsum-collective-cost-confirmed-2026-08-21.md`)
were methodology artifacts of forced per-span synchronization, not real
per-call costs.**

Per Fable's explanation, matching MLX's known lazy-evaluation model:
`mx.synchronize()` at a span boundary does not measure that span's own
cost — it drains the ENTIRE pending lazy graph accumulated since the
last sync point, which under MLX's async execution includes real GPU
work from the PRECEDING layers' compute (attention, MoE gate/up/down,
etc.) that hadn't been forced to materialize yet. A span that
synchronizes right after `all_sum` therefore misattributes a large
share of upstream, unrelated compute time to the `moe.all_sum` span
specifically. This is consistent with — and now explains — several
things that were flagged as odd earlier tonight and in the broader
`docs/PERFORMANCE_HISTORY.md` history without being fully resolved:

- Sync-span profiling's own documented overhead (~15% prefill / ~77%
  decode, noted repeatedly across this session's docs) is exactly the
  kind of cost this per-span forced-drain would produce.
- Per-layer forced synchronization additionally destroys the pipelining
  MLX's lazy graph would otherwise provide across layers — meaning the
  sync-span measurement isn't just misattributing cost between spans,
  it's measuring a SERIALIZED execution mode that production (running
  with `EXO_DSV4_FENCE_ASYNC=1`, non-blocking `mx.async_eval`) does not
  actually run in. The sync-span number is not just imprecise, it's
  measuring a different, slower execution regime than what's live.

**This retroactively changes how to read `docs/moe-allsum-collective-cost-confirmed-2026-08-21.md`'s
"21.4% of decode wall time" and "14.4% blended" figures**: these should
now be understood as upper-bound artifacts of the sync-span
methodology's forced-drain effect, not real, actionable per-collective
cost shares. The real, jaccl-internal-measured cost (2.9-5.3% of wall
time) is the trustworthy figure.

## What remains genuinely open

This does NOT fully explain where the rest of the real 53.48ms/token
budget goes (only ~2.9-5.3% is now confidently attributed to
`moe.all_sum`'s real transport cost). What it DOES establish: **the
`moe.all_sum` collective — and by extension the "34x software overhead
gap" framing from earlier tonight — was likely never the dominant
factor it appeared to be under sync-span measurement.** The real
bottleneck (if there is one beyond the confirmed real GPU compute + a
small real collective cost) more likely lives in genuine GPU compute
time across attention/MoE-compute spans, which the earlier roofline
finding (~12% of theoretical bandwidth-bound peak) and live Instruments
trace (~70% GPU idle, both ranks) already independently support as real
— just not specifically attributable to the collective anymore.

## Per Fable's explicit guidance: no live-cluster work done tonight after this point

Fable's review explicitly recommended NOT relaunching the cluster
further tonight for additional instrumentation (production uses
non-blocking `mx.async_eval`, so a naive `perf_counter`-around-async_eval
approach would measure near-nothing; a correct sampled/queue-depth probe
or GPU-side capture is a real, careful design task, not a
quick add — and the user is asleep with no one available to notice a
live-cluster degradation). This arithmetic check was explicitly
suggested as the safe, zero-risk alternative to do first — and it
turned out to resolve the open question well enough that further live
instrumentation is not obviously needed before the user is back.

## Cluster state

Untouched by this analysis — still in the clean validated baseline
config confirmed immediately before this check (`EXO_DSV4_MOE_FUSED_GATE_UP=1`
+ `EXO_DSV4_FENCE_ASYNC=1`, no diagnostic env vars, correctness and
throughput both confirmed at baseline moments before this document was
written).

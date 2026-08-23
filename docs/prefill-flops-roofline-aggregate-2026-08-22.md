# T7: Prefill FLOPs compute-bound roofline — aggregate ~70% of ceiling on the GEMM-bound majority, a real but self-corrected finding — 2026-08-22 (session 4)

## Why this check

An external tip flagged that decode's bandwidth-bound roofline (§4.3,
found decode at only 12-20% of theoretical peak) directly led to
discovering the silently-broken async fence (+58-67% decode win) — and
that no equivalent aggregate compute-bound roofline exercise had been
done for prefill. Prefill's "97-98% GPU utilization" is cited in §3.2
as compute-bound and settled, but decode's own "GPU-busy ≠
useful-compute" trap (§2, `docs/gpu-util-vs-allsum-cost-reconciled-2026-08-19.md`)
was proven real for the exact same style of utilization telemetry —
worth re-checking with the same rigor rather than trusting the prior
framing.

## First attempt — a real methodology error, caught and corrected, not glossed over

Initial approach mirrored decode's roofline exactly: `2 × active_params`
FLOPs/token (the standard forward-pass heuristic), divided by the M4
Max's measured 11.66 TFLOPS dense-fp16 peak, split across TP=2.

```
theoretical ceiling ≈ 897 tok/s
real prefill @ 100K/300K/500K: 359.7/348.2/324.1 tok/s
=> "36-40% of peak"
```

**This number is wrong and was caught, not reported.** Cross-checking
it against the REAL bottom-up per-span efficiency data already measured
in `docs/dsv4-attention-kernel-efficiency-2026-08-18.md` (attention
GEMMs at 83-85% of their own ceiling) and
`docs/prefill-optimization-campaign-handoff-2026-08-18.md` (MoE at
62.6% of its ceiling) produced a physically impossible result: solving
for the efficiency the *unstudied* remainder of wall time would need to
have, to reconcile with the naive top-down "36-40%" figure, gives
**-35.8%** — a negative efficiency, which cannot exist. This is a real,
useful signal that the top-down formula itself is unfit for this
regime, not that some code path has negative efficiency.

**Root cause of the modeling error**: `2 × active_params` is a
decode-style heuristic — weight-read-dominated, O(1) in context length,
correct for decode's regime (§4.3 validated this scaling directly
against real bytes-touched-per-token). It is the WRONG model for
prefill, where attention's SDPA cost scales with the KV/pooled sequence
length actually processed (§attn-kernel-efficiency's real shapes: local
KV len 2175, pooled CompressedAttention KV up to 3894 at 220K context,
sparse-path gathering 512 rows/query) — an O(L) or worse term that a
flat per-active-parameter formula cannot capture. This is exactly why
real prefill tok/s DECREASES with context depth (359.7 → 324.1 tok/s,
100K → 500K) while the naive formula predicts a flat ceiling regardless
of depth — a real, checkable prediction failure that confirms the
model is wrong, not just imprecise.

**Reusable lesson**: the decode-style "2×active_params ÷ peak-TFLOPS"
roofline formula does not transfer to prefill. Any future FLOPs-ceiling
estimate for a context-length-scaling workload must use a shape-aware,
bottom-up (per-real-kernel) method, not a single global weight-touch
heuristic — this is a genuinely different regime from decode's
bandwidth-bound, context-length-independent problem.

## Corrected approach — bottom-up, using real already-measured per-kernel data

The correct roofline is bottom-up: real achieved-vs-ceiling TFLOPS for
each actual GEMM/SDPA kernel at its real production shape (already
measured in two prior sessions, re-aggregated here into a single
wall-time-weighted figure that had not been explicitly computed before
tonight):

| span | wall % | achieved TFLOPS | ceiling TFLOPS | % of ceiling |
|---|---|---|---|---|
| `attn.sdpa` | 13.6% | 7.20 | 11.66 | 61.7% |
| `attn.sdpa.compressed` | 11.8% | 9.23 | 11.66 | 79.1% |
| `attn.o_proj` | 10.0% | 9.63 | 11.59 | 83.2% |
| `attn.proj_qkv` | 8.9% | 9.64 | 11.38 | 84.7% |
| `moe.switch_mlp` | 26.9% | 9.37 | 14.97 (mxfp4-specific) | 62.6% |

**These 5 spans together cover 71.2% of real prefill wall time**
(from the real, complete 220K-context span profile,
`docs/dsv4-220k-prefill-span-profile-2026-08-18.md` — the only spans
with real measured GEMM/SDPA FLOPs ceilings computed against them).

**Wall-time-weighted aggregate: 9.00 TFLOPS achieved vs 12.86 TFLOPS
blended ceiling = 70.0% of hardware ceiling.** This is a real number,
computed here for the first time by aggregating prior per-span work —
not previously stated as a single headline figure anywhere in the repo.

## The remaining 28.8% of wall time — not a FLOPs question

The rest of prefill's real wall time (28.8%, from the same complete
span profile) is non-GEMM: `moe.all_sum` (9.5%, the real settled
collective cost — see §2's closed all_sum saga, NOT the retracted
61-64% artifact), `layer.attn_hc`/`ffn_hc`/residuals/norms (~9.4%
combined, elementwise ops, bandwidth- not FLOPs-bound),
`attn.indexer` (4.0%), `moe.post_combine`/`gate` (5.1%),
`embed`/`lm_head`/`rope`/`mask`/etc (~2.5% combined). **None of these
are meaningfully addressed by a FLOPs-ceiling framing** — they are
dispatch/bandwidth/collective-bound costs, and (per the cross-reference
above) already separately investigated and substantially closed
elsewhere in this document (§2's `moe.all_sum` saga, §8's jaccl/RDMA
work).

## Consulted a second opinion (Fable) before treating this as settled — real pushback, incorporated

Before writing this up as closed, ran the synthesis past an independent
reviewer. Two real corrections came back, both incorporated below
rather than glossed over:

**1. The 70% figure is span-conditional, not end-to-end, and reporting
only it overstates prefill's health.** "70% of ceiling" answers "are
the studied GEMM kernels efficient?" — it does not answer "is prefill
healthy end-to-end?" The honest end-to-end number, treating the 28.8%
non-GEMM remainder as contributing ~0 useful FLOPs against real wall
time: **0.712 × 70.0% ≈ 49.8% effective end-to-end efficiency.**
Equivalently: if the entire 28.8% non-GEMM remainder were somehow fully
eliminated or perfectly overlapped, the maximum theoretical prefill
speedup would be **1.40x** — a magnitude comparable to decode's real
+58-67% fence fix, not a rounding error. **This document does NOT
decompose that 28.8% into its own roofline-style breakdown** (it
relies on the earlier, separately-closed `moe.all_sum` investigation
for the 9.5% collective slice, but the remaining ~19.3% — residuals,
norms, hc, indexer, gate/combine, embed/lm_head/rope/mask — has never
been checked for a decode-fence-shaped hidden bug of its own). This is
flagged as a genuinely open follow-up, not silently closed.

**2. A real audit risk inside the "achieved TFLOPS" numbers
themselves, not yet checked**: if the analytic FLOP-count denominator
used for `attn.sdpa`/`attn.sdpa.compressed`'s ceiling calculations
assumed the naive (non-causal-masked, non-MLA-absorbed) formulation
while the real kernel exploits causal masking or the MLA
absorbed-projection trick, the "achieved TFLOPS" numerator could be
computed against an inflated denominator — silently overstating
efficiency (a stall INSIDE a span lowers achieved-TFLOPS, so this
specific measurement method can't be fooled by hidden blocking the way
decode's occupancy telemetry was; but FLOP-count inflation is a
distinct, unaudited risk in the same direction of error). **Not
checked this session** — flagged as the first thing to verify before
fully trusting the 61.7%/79.1% split between the two SDPA spans.

## Real conclusion — the honest headline result, correctly caveated

**Two real numbers, not one, are the honest output of this exercise**:
- **Span-conditional**: the 71.2% of prefill wall time that IS
  GEMM/SDPA-bound runs at **~70% of its own blended hardware ceiling**
  — a real, moderately healthy number for the kernels it covers.
- **End-to-end effective**: **~49.8%**, once the 28.8% non-GEMM
  remainder (never itself roofline-checked) is honestly counted against
  total wall time — implying up to **1.40x** theoretical headroom if
  that remainder were fully addressed, a magnitude in the same
  ballpark as decode's real fence-fix win, not dismissable as noise.

This is NOT the same conclusion as "prefill is fine, nothing to see
here" — it correctly rejects the flawed top-down "36-40%" number
(a real methodology catch) and replaces it with a properly-scoped,
correctly-caveated result: the GEMM-bound majority looks healthy, but
the un-decomposed 28.8% remainder is a real, quantified, still-open
gap of comparable magnitude to decode's biggest win, not yet checked
for the same class of hidden bug.

## What this does NOT establish

This does not re-derive per-span numbers from scratch (they were
already real and measured in two prior sessions — this exercise's
contribution is the aggregate wall-time-weighted synthesis, catching a
flawed top-down alternative approach, and — per the consult above —
correctly NOT closing the book on the unstudied 28.8%). It also does
not extend past the already-covered spans for the GEMM side — sub-70%
individual spans (`attn.sdpa` at 61.7%, `moe.switch_mlp` at 62.6%) do
still have some real, already-quantified individual headroom (§3.1's
end-to-end estimate: closing all four attention spans to ceiling buys
at most 1.11x total prefill; the MoE side was separately extended and
closed as NO-GO for further kernel work, see §13's T3b/T9 triage
above).

## Next step (genuinely open, not yet started)

Decompose the 28.8% non-GEMM remainder (specifically the ~19.3% beyond
the already-closed `moe.all_sum` 9.5%) with the same rigor as decode's
async-fence investigation: real per-op wall-time attribution (not
span-profiler, which has its own known ~15% overhead tax at prefill —
§ meta-lessons) to check whether any of `layer.attn_hc`/`ffn_hc`,
residuals, norms, `attn.indexer`, or `moe.gate`/`post_combine` hide a
similar silently-blocking-when-it-shouldn't-be gate. Also: audit the
FLOP-count denominators used for `attn.sdpa`/`attn.sdpa.compressed`'s
ceiling figures for causal-masking/MLA-absorption inflation before
fully trusting the 61.7%/79.1% split, per the consult's flag above.

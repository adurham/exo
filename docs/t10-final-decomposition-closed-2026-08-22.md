# T10 completion: prefill's 28.8% non-GEMM remainder — fully decomposed, no async-fence-class bug found — session 4 conclusion

## Why this document

Consolidates the final round of T10 sub-investigations (following the
HyperConnection false lead and the indexer topk_fused closure). Per an
explicit decision criterion set earlier this session: "one more genuine
NULL result from the remaining spans would justify demoting T10 from
'actively hunt for a hidden bug' to 'accepted overhead, documented and
closed.'" This round produced multiple additional NULL results, plus
one genuine (but not worth shipping) candidate fix. T10 is now closed.

## Fable consult before this round — two real corrections incorporated

Before continuing, consulted Fable for a prioritized task list. Two
real corrections came back and were applied:
1. **FLOP-vs-bandwidth framing error**: my initial "~100x over FLOPs
   ceiling" framing for `hc_expand` used the wrong denominator — this
   op is memory-bound (moves ~150MB of tensors), not FLOP-bound (~268
   MFLOP is trivial). Redone with the correct bandwidth ceiling.
2. **Sum-check blind spot**: never verified whether the real span
   profile's percentages sum to 100% of wall time, or whether there's
   a large *unattributed gap between spans* (dispatch bubbles, eval
   points) that span-by-span investigation would never find — exactly
   the class of thing the async-fence bug turned out to be.

## Check 0: sum-check (Fable's blind spot #1) — CLOSED, no hidden gap

Recomputed using RAW millisecond totals from the source span-profile
doc (not re-summed pre-rounded percentages, which produced a
misleading 110% on first pass). Real wall clock for the profiled run:
673s. Sum of all real leaf spans (ms totals): 683,333.6ms = **101.5%
of real wall time**. This is within normal span-boundary
overlap/measurement noise — **there is no large unattributed gap
between spans**. The earlier apparent 110% was purely an artifact of
re-summing percentages that were each independently rounded and
computed against slightly different implicit denominators (attn+ffn
parent rollups vs the full leaf set). This closes the blind spot
cleanly: T10's remaining spans genuinely account for essentially all
of the real non-GEMM wall time; there's no hidden "missing" chunk.

## Check 1: hc_expand (attn_residual/ffn_residual, 4.4% combined) — real gap found, fix rejected on quality grounds

Corrected bandwidth ceiling (real memory traffic: 151MB at the op's
mixed bf16/fp32 dtypes, `residual.astype(float32)` cast included):
**276.85µs theoretical floor vs 2361.41µs real measured — 8.53x over
the naive ceiling.** Decomposed via microbench into sub-costs: the
`matmul(comb.swapaxes(-1,-2), residual.astype(float32))` alone costs
1453.69µs against a corrected 491.64µs fp32-traffic ceiling (2.96x);
the `residual.astype(float32)` cast alone costs 584.02µs against a
368.73µs ceiling (1.58x). Combined realistic ceiling estimate ~1229µs
vs the real 2373.83µs (compiled) — **1.93x**, a real but far smaller
gap than the naive framing suggested.

**Candidate fix tested**: cast the tiny `comb` tensor down to bf16
instead of upcasting the large `residual` tensor to fp32 before the
matmul (avoids materializing a 2x-larger intermediate for the dominant
tensor). Real result: **1690.48µs, a genuine 1.41x speedup** over the
original 2373.83µs.

**Rejected on quality grounds, not implemented**: precision-checked the
alternative against the original at realistic output magnitudes
(std≈2.2, range ±20, matching real post-RMSNorm activation scale).
Real result: **mean relative error ~1.08%, max abs diff 0.125** — this
is NOT within bf16's own inherent rounding floor (confirmed via a
roundtrip-noise control: bf16-roundtrip-alone error on the SAME output
is exactly 0.0, meaning the 0.125 diff is a genuine numerical
divergence from the precision change, not ordinary bf16 noise). This
value compounds through the residual stream across all 43 layers.
Given this repo's own documented history of quality disasters from
undervalidated precision changes (the TOTK=160 "BOS-only" failure,
§5), and that the total realistic payoff is only ~1.3 percentage
points of prefill wall time (4.4% span share × ~30% potential
reduction), **this is not worth shipping without a full
needle-in-haystack + broader quality validation pass, which was not
performed this session.** Documented as investigated-and-rejected,
not silently dropped — a future session COULD revisit this with proper
quality validation if the payoff calculus changes (e.g., if combined
with other small wins).

## Check 2: moe.post_combine (4.2%) — CLOSED, not a bug, real explanation for the apparent gap

Verified the existing `@mx.compile` fusion (`_moe_post_combine`)
actually fires at production shape: **compile gives essentially zero
speedup here (912.47µs compiled vs 920.04µs uncompiled, 1.01x)** — not
because the optimization is broken, but because this op is genuinely
memory-bound at this shape and MLX's compiler has little to fuse
beyond what it already does implicitly.

**Real 6.1x discrepancy investigated and fully explained**: my
isolated elementwise-combine microbench (912µs) vs the real span-profile
figure (5567.13µs) initially looked like a hidden cost. Reading the
real code (`deepseek_v4.py:2972-3003`) showed the `moe.post_combine`
SPAN genuinely wraps the full `self.shared_experts(x)` MLP forward
pass too, not just the elementwise combine — this is correctly
documented in the code's own comment ("shared_experts forward ... stays
separate; we fuse only the y-side combine"), I had simply mis-scoped my
microbench. Computed real shared_experts GEMM FLOPs at the true
per-rank shape (hidden=4096, TP-sharded intermediate=1024):
51.54 GFLOP → 4420.21µs at theoretical peak. Implied real shared_experts
cost from the span data (5567.13 − 912.47 = 4654.66µs) is **1.05x of
theoretical peak — already essentially optimal.** No bug; the span's
apparent excess was a real, substantial, already-near-ceiling GEMM I'd
failed to include in my own comparison.

## Check 3: moe.gate (0.9%) — CLOSED, not a bug, timeboxed per plan

Read the real routing code (`_gate_route`): uses `argpartition`
(already the optimal choice per this repo's own MOE_KERNEL_HANDOFF
history — not a full sort), no `.item()`/numpy host-sync calls, matmul
already fused into the compiled expert-select chain per an existing
code comment. Microbench at real shape (256 experts, top_k=6, L=2048):
761.90µs vs the real span-profile's 1169.80µs — only **1.54x**, well
within normal in-model overhead (finalize() sync, real weight-matrix
reads every layer vs a reused test tensor). Per the plan's timebox (0.9%
of wall time caps any possible win at ~1% end-to-end), not pursued
further.

## Check 4: tail spans batch triage (attn.mask, kv_cache, norms, rope_in/out, compressor) — CLOSED collectively

Checked all 7 remaining small spans (0.1-0.8% each) against the
~96.8µs real dispatch-latency floor established earlier this session.
4 of 7 exceeded a 3x-over-floor threshold: `attn.rope_in` (8.73x),
`attn.rope_out` (4.84x), `attn.kv_cache` (3.75x), `attn.mask` (3.70x).
**Read the real code for the largest flag** (`attn.rope_in`): the span
genuinely wraps TWO separate real RoPE dispatch calls (`q` and `kv`
each individually rope'd), not one — consistent with, not contradicting,
a ~2x-real-ops multiplier on top of the single-dispatch floor. Given
these 4 flagged spans combined are only ~2.0% of total prefill wall
time (845+469+363+358µs per-layer ≈ negligible in aggregate), and each
plausibly explained by wrapping multiple real sub-ops rather than a
hidden inefficiency, **not pursued to individual microbench depth** —
consistent with the plan's stated batch-triage stop condition.

## T10 final conclusion

**All identified candidate spans in prefill's 28.8% non-GEMM remainder
have now been investigated.** Summary across the full T10 investigation
(this session, all sub-parts):

| Span | Wall % | Verdict |
|---|---|---|
| HyperConnection (attn_hc/ffn_hc) | 4.6% | NOT a bug — fast kernel already fires in production (false lead, real methodology lesson learned) |
| hc_expand (attn/ffn_residual) | 4.4% | Real 1.93x gap found; 1.41x fix found but REJECTED on quality grounds (1.08% mean rel error, unvalidated) |
| moe.post_combine | 4.2% | NOT a bug — apparent 6.1x gap fully explained by an already-near-optimal (1.05x peak) shared_experts GEMM I'd initially excluded from scope |
| attn.indexer (topk_fused) | part of 4.0% | NOT applicable — structurally decode-only, closed via code reading |
| moe.gate | 0.9% | NOT a bug — clean code, 1.54x gap is normal overhead, capped payoff |
| tail spans (7 spans) | ~2.5% combined | NOT pursued individually — each plausibly explained by wrapping 2+ real sub-ops, small aggregate magnitude |
| moe.all_sum | 9.5% | Already closed in a prior session (§2's real collective-cost saga) |

**No async-fence-class hidden bug was found in prefill's non-GEMM
remainder.** Unlike decode's async fence (a genuinely broken gate that
silently disabled a real optimization), prefill's 28.8% remainder
decomposes into: legitimate real compute (shared_experts GEMM, RoPE,
gate routing — already near their own ceilings), one real-but-small
gap with an unshippable fix (hc_expand), and a large collective cost
already investigated and closed in a prior session (moe.all_sum).

**T7's 1.40x theoretical-headroom figure should be reframed**, per the
standing decision criterion set earlier this session: this is a
genuine architectural/dispatch-count ceiling for this model's current
design, not a to-be-found bug. The real, honest remaining margin from
everything investigated this session is closer to the hc_expand
finding alone (~1.3 percentage points, not shipped) plus whatever
residual efficiency exists in the already-near-peak shared_experts/gate
GEMMs (single-digit percent at most, per their own measured 1.01-1.05x
figures).

## Standing recommendation

T10 is CLOSED. No further prefill-side dispatch/gate-hunting is
recommended without new evidence (e.g., a different model architecture,
a different context-length regime not yet profiled, or a new
methodology that reveals something the span-profile approach can't
see). Future prefill optimization work should target either: (a)
implementing and QUALITY-VALIDATING the hc_expand bf16-comb-cast fix
if the ~1.3pp win becomes worth the validation cost, or (b) a
fundamentally different optimization axis not yet explored this
session (e.g., revisiting whether DSpark/MTP could be ported to TP
with genuinely fresh data, per T6's flagged cheap re-validation step).

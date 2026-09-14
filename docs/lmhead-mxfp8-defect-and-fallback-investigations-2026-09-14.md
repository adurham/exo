# lm_head mxfp8 defect: follow-up "smarter fix" investigations (2026-09-14)

**Status: ALL THREE INVESTIGATIONS BELOW ARE NEGATIVE RESULTS. No code
from this document is live in production. Production runs the blunt fix
(`EXO_DSV4_LMHEAD_MXFP8=0`, full BF16 lm_head) — see
`docs/incidents/lmhead-mxfp8-cross-lingual-glue-defect-2026-09-13.md` for
the defect itself and the blunt fix.**

This doc exists so nobody re-attempts these ideas without reading it first.
Each investigation below required real implementation and/or real
measurement effort (not just brainstorming) before being ruled out — the
value of this document is exactly in the specificity of *why* each one
fails, since "have we tried X" questions about lm_head quantization will
keep coming up as long as the ~2.5-4% BF16 throughput cost is visible in
benchmarks.

If you are reading this because you have a NEW idea for a faster
correct-or-near-correct lm_head: read §4 first (the structural MLX finding)
— it rules out an entire class of "conditional fallback" designs
regardless of the specific trigger heuristic, so check whether your idea
falls into that class before spending implementation time.

---

## Background: what's being optimized and why it's hard

DeepSeek-V4's lm_head is a single `129280 × 4096` linear projection,
1.059 GB in BF16, replicated (not sharded) on every TP rank. It is
byte-limited at the production call shapes (measured 92-94% of spec
bandwidth), so halving its bytes via mxfp8 quantization converts almost
directly into time — hence the original +6.0% decode win that motivated
shipping it ON by default (commit `80ec8ec03`, 2026-08-30). The problem
(see the incident doc) is that mxfp8's quantization noise is large enough
to flip top-1 argmax at low-margin decode positions, and that shows up as
a real, user-visible correctness defect in free-form prose — not just
benign rewording.

Every investigation below is implicitly asking the same question: **is
there a way to keep most of the mxfp8 speed win while eliminating (or
substantially reducing) the argmax-flip defect?** The honest answer, as of
this document, is no — not with any approach tried so far, for two
independent classes of reason (numerics: quantization noise doesn't go to
zero at feasible group sizes; and a framework/hardware-level compute-cost
fact about MLX's `mx.where` semantics, which rules out the conditional
class entirely).

---

## Investigation A: alternative fixed quantization schemes (numerics-only, no live test)

**Verdict: rejected — every alternative scheme MLX supports still has a
nonzero flip rate.**

Tested against the real lm_head weight (not synthetic), offline:

| Scheme | Result |
|---|---|
| Affine int8, g32 (finest group size supported) | Best of the alternatives tried. Still **1.44-1.76% top-1 flip rate** vs BF16 (range reflects different measurement passes on the same scheme) — not zero. All flips concentrated in the lowest margin band, same failure signature as mxfp8. |
| Affine int8, g64 | Worse than g32 (coarser grouping = more quantization error). Still nonzero. |
| Affine int8, g128 | Worse still. |
| mxfp4 | Far worse — 4-bit mantissa budget is nowhere near enough for a 129280-row lm_head at this precision requirement. |
| nvfp4 | Also far worse, same reason as mxfp4. |

None of `to_quantized`'s supported `(group_size, bits, mode)` combinations
reach a 0% flip rate. This rules out "just pick a gentler quantization
scheme" as a fix — the defect isn't specific to mxfp8's particular
encoding, it's a generic consequence of any lossy quantization applied to
a projection where argmax selection is sensitive to sub-1-nat margins.

A top-K logit re-rank scheme (compute the full projection in fast mxfp8,
then re-score only the top-K candidate rows in exact BF16) looked
near-lossless when evaluated on **synthetic** per-token inputs — the
obvious appeal being that it only pays the expensive BF16 cost for a small
K-row slice, not the full 129280-row output. But real residual-decode
testing (replaying actual hidden states from live generation, not
synthetic N(0,1) inputs) showed it **reproduces the same cross-script glue
defect at a nonzero rate** on real data. This was rejected specifically
because shipping it on the strength of the synthetic-only result would
have been exactly the same mistake that caused the original incident
(shipping mxfp8 on a high-margin eval that couldn't see the real failure
mode). The residual real-hidden-state distribution differs from synthetic
inputs precisely in the margin tail where flips live — synthetic testing
is structurally the wrong tool for validating this class of fix.

---

## Investigation B: conditional margin-based fallback (IMPLEMENTED, live-tested, FAILED both bars)

**Verdict: rejected. Built for real, tested live on production, failed
both correctness and throughput.**

### The idea

Compute the lm_head projection with the fast mxfp8 path first. For each
row (each token position in the batch), check the top1-vs-top2 margin of
the resulting mxfp8 logits. If the margin is below a threshold (candidate
for "this row is at flip risk"), redo *only that row* through the exact
BF16 weight and splice the corrected value in. High-margin rows (the
common case) never pay the BF16 cost at all.

### What was actually built

This was implemented, not just designed — the patch exists (currently
uncommitted working-tree state, see
`docs/documentation-inventory-2026-09-14.md` / the main session log for
its disposition):

- `mlx-lm/mlx_lm/utils.py` (+8 lines): stashes the un-quantized BF16
  weight as `mod._p05_bf16_weight` on the module *before* calling
  `to_quantized(...)`, so the fallback can redo rows without a second
  weight load. The weight is TP-replicated and already resident, so this
  is a reference assignment, not an extra copy.
- `mlx-lm/mlx_lm/models/deepseek_v4.py` (+68 lines): a
  `_lmhead_mxfp8_fallback()` function, gated by
  `EXO_DSV4_LMHEAD_MXFP8_FALLBACK` (default off) and
  `EXO_DSV4_LMHEAD_MXFP8_FALLBACK_MARGIN` (default `3.62`, the same
  margin-band boundary from the original P05 numerics). Spliced into
  `Model.__call__` immediately after the main `self.lm_head(h)` call.
- `start_cluster.sh` (+10 lines): env plumbing for the two new flags, both
  default OFF/unset, so the patch is inert unless explicitly enabled on
  top of `EXO_DSV4_LMHEAD_MXFP8=1` (which is itself no longer the
  default as of commit `1da54ee19`).

### Live validation result: FAILED, two independent ways

This was relaunched on production with both flags on, tested, then
relaunched back to normal — a real live A/B, not a simulation.

**Correctness: still 10/10 defect rate.** The splice sits in
`Model.__call__`, but DSpark's speculative draft head calls `lm_head`
**directly**, bypassing `Model.__call__` entirely:

```
mlx-lm/mlx_lm/models/deepseek_v4.py:7472 (inside DSparkStage.draft()):
    base_logits = lm_head(self.norm(x))     # (B, bs, V)
```

This is the call that seeds the draft tokens that DSpark's whole
accept/reject chain is built on. The `Model.__call__` splice only corrects
the **verify** path's logits, and by the time verify runs, a wrong draft
token may already be locked into the candidate set — the verify-side
correction cannot retroactively un-poison a bad draft choice. Patching one
call site (the "obvious main path") silently misses the other, and the
missed one is the one that matters most for output quality, since it's
upstream of everything else in the speculative chain.

**Throughput: slower, not faster.** 37.38 tok/s live-measured, vs a fresh
same-session bf16-only baseline of 39.06 tok/s. The offline cost model
this patch was built against — `mxfp8_time + fallback_frac × bf16_time`,
i.e. "you only pay the BF16 cost on the fraction of rows that trigger" —
is **wrong for production's actual batch shapes**, for two compounding
reasons:

1. The BF16 lm_head is weight-bound: a full 1.059 GB read dominates its
   cost regardless of how many rows are being projected (M=1 ≈ M=4 ≈
   ~2235-2238 µs, measured). A per-row conditional fallback in a
   *multi-row batch* pays that **entire** weight-read cost the moment
   **any single row** in the batch triggers — there's no partial-batch
   discount.
2. Production doesn't run single-row (M=1) decode at all — it runs
   DSpark's **batched verify shape** (M=4, `gamma+1` with
   `EXO_SPECULATIVE_GAMMA=3`). The live-measured margin distribution
   (65.6% of committed tokens fall below the 3.62 full-recall margin
   threshold — notably *worse* than the 53.6% assumed in the original
   offline estimate) means P(≥1 of 4 rows triggers) is close to 100% in
   practice. The batch pays the full BF16 weight-read on nearly every
   single call, stacked *on top of* the mxfp8 first-pass compute that
   still has to run to know which rows need the fallback in the first
   place. Net: worse than either pure mxfp8 or pure BF16 alone.

---

## Investigation C: extending the fallback to also cover the DSpark draft-head call site

**Verdict: rejected on sanity-check math + one decisive real
microbenchmark. NOT re-implemented live — the math was conclusive enough
that a live A/B would only have confirmed a foregone conclusion, and the
task's own instruction was to stop once decisive, not spend hours
re-confirming.**

The natural next question after Investigation B: if the correctness
failure is specifically that the draft-head call site (§B) is unpatched,
what if the splice is extended to cover *both* call sites (verify AND
draft)? This was investigated via math and one targeted microbenchmark,
not a full live re-implementation, for reasons that become clear below.

### Finding 1 — corrected batch-shape assumption

The live draft call (`_dspark.draft(..., width=gamma)`) passes
`width = min(EXO_SPECULATIVE_GAMMA, dspark_block_size)`. Production has
`EXO_SPECULATIVE_GAMMA=3` (confirmed live), and `dspark_block_size` is a
class-level ceiling of 5 (`self.block_size` in `DSparkStage.__init__`,
`dspark_block_size: int = 5` in `ModelArgs`) — so the real draft batch is
**M=3**, not M=5. (5 is only the architectural ceiling; the actual runtime
value is gated by the gamma setting.) At M=3, P(≥1 of 3 rows below the
3.62 margin) = 95.9%, vs 98.6% at verify's M=4 — still near-total, just
marginally less extreme than a naive M=5 assumption would suggest. This
doesn't change the verdict, but it's worth getting right: the near-100%
trigger rate is not an artifact of assuming the wrong batch size.

### Finding 2 — draft and verify cannot share one weight read

A real alternative-design idea worth checking (raised in review, not
dismissed out of hand): could draft and verify share a single unified
BF16 pass, amortizing the weight-read cost across both? No — they are
causally serialized by a full ~61-layer target-model forward pass: draft
produces candidate tokens → those tokens feed the verify forward pass →
verify's own hidden states feed its own, separate, later `lm_head` call.
There is no point in the execution graph where both row-sets co-exist, so
unifying them into one shared BF16 pass is **architecturally impossible**
here, not merely uneconomical.

### Finding 3 — the decisive result: `mx.where` computes both branches unconditionally

A real, isolated microbenchmark (synthetic BF16/mxfp8 weight matrices at
the exact production shape 129280×4096, run standalone on
macstudio-m4-1, zero production impact) of the actual deployed splice
pattern:

```python
mx.where(fallback_mask[..., None], bf16_logits, mxfp8_logits)
# where bf16_logits = h.astype(bf16) @ w_bf16.T   (computed over the FULL batch, every call)
#   and mxfp8_logits = the quantized-path projection
```

Measured at M=1/3/4/5: the candidate's cost tracked **95.9-97.8% of the
sum of the separate mxfp8 and bf16 matmul costs**, at every M tested,
**regardless of the margin/trigger-probability data**. Only ~2-4% was
saved from kernel scheduling overlap — nowhere near the savings a "skip
the BF16 branch when nothing triggers" mental model would predict.

This confirms `mx.where` over array-valued masks in MLX (true of most
lazy-array frameworks, including JAX) **computes both operand arrays
unconditionally before selecting elementwise** — there is no compute-level
skip, only a value-level one after both branches are already fully
materialized. This makes the entire mechanism's cost **invariant to
trigger probability**: even in a hypothetical world where the live margin
distribution were far more favorable than the measured 65.6%/95.9%/98.6%,
this specific implementation pattern would still unconditionally pay for
*both* matmuls on every single call, at every call site, regardless of
where in the code it's spliced in.

This one finding is what closes off the entire "per-row conditional
fallback via `mx.where`-style value-selection" design class — not just
this specific variant, not just this specific margin threshold, not just
this specific call site. **Do not re-attempt this mechanism a fourth time
without genuinely new information** — e.g. a different quantization scheme
entirely, or evidence that MLX has gained a true gather/scatter
conditional-compute primitive that can skip untouched rows' compute, not
just their output value.

### Cross-check against the live measurement (sanity, not a new experiment)

Projecting this microbenchmark's per-call-site numbers onto a full decode
step (draft M=3 + verify M=4, both extended) gives roughly +9% of total
per-token time added (lm_head is ~17% of a ~25.6ms/token budget at
39 tok/s, and its cost roughly doubles under the fully-extended
candidate) → a projected ~35-36 tok/s. That's consistent in sign and rough
magnitude with the actually-measured single-site regression from
Investigation B (37.38 vs 39.06 tok/s) — the fully-extended candidate
would plausibly be *worse* still, since it doubles down on the same
structural cost at a second call site. This cross-check is corroborating,
not the primary evidence; §Finding 3 above is what actually decides the
question.

### The one named, unmeasured, doubtful escape hatch

For completeness: a **true** conditional-compute implementation (force a
concrete host-side sync of the margin mask every decode step, branch in
Python on `mask.any()`, gather only the flagged rows, run a *smaller*
BF16 matmul on just that gathered subset, scatter the result back) could
in principle avoid paying for untouched rows' BF16 compute, since it
sidesteps `mx.where`'s unconditional-both-branches semantics entirely.
This was **not measured**, and is flagged as doubtful rather than
promising, for two structural reasons:

1. It requires an `mx.eval`-forcing host sync on every single decode
   step, stalling MLX's async CPU/GPU dispatch pipeline on a
   latency-bound hot path — itself a real, likely-comparable-or-worse
   cost, just a different one than the `mx.where` tax.
2. At P(trigger) ≈ 96-99% (Findings 1 and the original Investigation B
   numbers), the "gathered subset" is on average almost the *whole* batch
   anyway — there's essentially no subset left to exploit even if the
   gather/scatter itself were free.

Neither (1) nor (2) alone is fully conclusive, but together they argue
strongly that implementing this is not worth the effort without first
seeing evidence that either concern is wrong.

### One structurally-different idea not yet ruled out (low expectations, named for completeness)

A **per-BATCH** binary decision — compute the *whole* batch in BF16 or the
*whole* batch in mxfp8, chosen by a cheap upstream heuristic evaluated
*before* either matmul runs — would not trigger the "both operands always
computed" tax that kills the per-row design, since only one branch would
ever actually execute. This is structurally different from everything
above and has not been measured. However, it still inherits the
near-100%-trigger-rate problem from Finding 1: if the heuristic has to be
conservative enough to catch the correctness-critical rows, it will
almost always choose the BF16 branch for the whole batch anyway — so its
realistic ceiling is "roughly the same as staying on unconditional BF16,"
i.e. probably not worth building. Named here only so it isn't
independently rediscovered and treated as novel; it has a plausible but
unpromising ceiling, not a demonstrated one.

---

## Summary table (for fast future reference)

| Approach | Type of test | Flip rate / correctness | Throughput vs BF16-only | Verdict |
|---|---|---|---|---|
| mxfp8 (original) | Live production | ~11.5% *estimated* argmax flips at low margin | +6.0% (but this is the baseline being fixed) | **Shipped, then reverted** — see incident doc |
| Affine int8 g32 | Offline, real weights | 1.44-1.76% flip rate | Not measured (moot — nonzero flips) | **Rejected** — not zero |
| Affine int8 g64/g128 | Offline, real weights | Worse than g32 | Not measured | **Rejected** |
| mxfp4 / nvfp4 | Offline, real weights | Far worse | Not measured | **Rejected** |
| Top-K logit re-rank | Synthetic: looked lossless. Real residual-decode: reproduces defect | Nonzero on real hidden states | Not fully characterized (moot) | **Rejected** — synthetic-only result would repeat the original mistake |
| Conditional margin fallback, verify-site only | LIVE production A/B | 10/10 defect (draft-head call site unpatched) | 37.38 vs 39.06 tok/s — **slower** | **Rejected** — fails both bars |
| Conditional margin fallback, extended to draft+verify | Math + 1 real microbenchmark (not live) | Would fix the correctness gap in theory | Projected ~35-36 tok/s — **slower still** | **Rejected** — `mx.where` computes both branches unconditionally, invariant to trigger rate |
| True gather/scatter conditional compute | Unmeasured | Unknown | Unknown, likely offset by host-sync stall + near-100% trigger rate | **Not attempted** — named, doubtful, not worth building without new evidence |
| Per-batch binary BF16-or-mxfp8 heuristic | Unmeasured | Unknown | Ceiling ≈ same as unconditional BF16 (heuristic must be conservative) | **Not attempted** — named, low expectation |
| **BF16-only (current production default)** | LIVE production | 0/10 defect, confirmed | Baseline | **Shipped** — the price of correctness, confirmed durable |

## What would change this conclusion

Only genuinely new information: a fundamentally different quantization
scheme (not yet available in MLX) that gets the flip rate to zero at a
useful compute saving, or MLX gaining a true lazy conditional-compute
primitive (a gather/scatter that skips both the read *and* the compute for
untouched rows, not just the value-selection). Neither exists today.
Re-litigating the `mx.where`-based conditional-fallback mechanism itself,
at any margin threshold or call-site combination, without one of those two
things changing, will reproduce Investigation C's Finding 3.

## Cross-reference

- The defect and blunt fix: `docs/incidents/lmhead-mxfp8-cross-lingual-glue-defect-2026-09-13.md`
- Fix commit: `1da54ee192203194a35bddbb53625f7cb11799d9`
- Original P05 ship-decision numerics (predecessor investigation,
  2026-08-30): `tmp/p05-lmhead-mxfp8-20260830/`, `tmp/p05-review-20260830/`
- The uncommitted Investigation-B patch itself (disposition — commit as
  marked dead code vs. discard — is recorded in the main session log,
  `docs/PERFORMANCE_HISTORY.md`, 2026-09-12–14 entry)
- Structural fact this investigation depends on (DSpark's draft head calls
  `lm_head` directly, bypassing `Model.__call__`): also noted in
  `docs/PERFORMANCE_HISTORY.md` §5 (speculative decoding) as a durable
  architecture fact relevant to any future decode-path fix, not just this
  one.

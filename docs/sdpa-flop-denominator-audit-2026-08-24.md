# SDPA FLOP-count denominator audit — causal-masking / MLA-absorption inflation check for T7's 70.0%/49.8% figures (2026-08-24)

## Why this check

Per the second-opinion flag in
`docs/prefill-flops-roofline-aggregate-2026-08-22.md` §"consulted a second
opinion", the T7 aggregate (`9.00 TFLOPS achieved vs 12.86 TFLOPS ceiling
= 70.0% of ceiling; end-to-end 49.8%`) was allowed to close only
conditional on auditing the achieved-TFLOPS *numerators* used for
`attn.sdpa` and `attn.sdpa.compressed`. Specifically: if the analytic
FLOP-count formula assumed the naive (non-causal-masked,
non-MLA-absorbed) form while the live kernel exploits a cheaper form,
the "achieved TFLOPS" would be computed against inflated FLOPs and the
efficiency figures would silently *understate* the real headroom
(equivalently overstate current health). This document does that audit
by reading the bench formula, the live kernel dispatch, and computing
per-op wasted-FLOP fractions at production shapes.

**Scope**: code-reading + arithmetic. No live cluster contact. No GPU
microbenching. Uses the same shapes as
`bench/attn_production_class_bench.py` and the span profile in
`docs/dsv4-220k-prefill-span-profile-2026-08-18.md`.

## What the bench actually counts (FLOP formulas, verbatim)

`bench/attn_production_class_bench.py`:
- `attn.sdpa.compressed` (line 230): `flops = 2 * 2 * N_HEADS * L_BAND * CATTN_KV * HEAD_DIM`
  where at TP=2/seq_split=1/220K: `N_HEADS=32`, `L_BAND=1024`, `HEAD_DIM=512`,
  `CATTN_KV = LOCAL_KV + POOL_R128 = 2175 + 1719 = 3894`.
  Yields ~261.3 GFLOP per call.
- `attn.sdpa` (sparse) (line 283): `flops = 2 * 2 * N_HEADS * L_BAND * (LOCAL_KV + INDEX_TOPK) * HEAD_DIM`
  where `LOCAL_KV=2175`, `INDEX_TOPK=512`. Yields ~180.3 GFLOP per call.

Both use the standard "dense QK^T + PV" 4·H·L·KV·D formula. Neither
formula discounts causal-masked or sliding-window-masked FMAs.

## What the live kernel actually executes

### attn.sdpa.compressed (`CompressedAttention.__call__`, deepseek_v4.py:4188–4312)

```python
kv = mx.concatenate([kv, pooled[:, None]], axis=2)   # (B, 1, LOCAL_KV+POOL, D)
mask = _extend_mask(mask, pooled_mask, kv.shape[2])
out = scaled_dot_product_attention(q, kv, kv, ..., mask=mask, ...)
```

The `mask` is an **array mask** (per-row-causal for the local slice,
per-row-causal-with-ratio for the pool slice; assembled via
`_extend_mask` + `_dispatch_pmask`, deepseek_v4.py:4248, 4250). Not the
string `"causal"` — so `mx.fast.scaled_dot_product_attention` cannot
route to any causal fast-path. It executes the full dense QK^T over the
concatenated KV, applies the array mask (min-fill for booleans /
additive for float), and then dense PV.

The bench measured mask-on vs mask-off delta at only 5.7%/3.2%
(`dsv4-attention-kernel-efficiency-2026-08-18.md` "Measurement
uncertainty" section). That small delta = **the cost of the per-element
mask *application*** (indexing + where), not FMA savings from skipping
masked blocks. **The kernel does not block-skip masked FMAs.**

### attn.sdpa (`SparseCompressedAttention.__call__`, deepseek_v4.py:4386–4841)

For prefill (`L>1`), the sparse-path `_sparse_pooled_attention_inner`
(deepseek_v4.py:1475-1516) does:
```python
local_scores  = q_scaled @ local_kv.swapaxes(-1, -2)         # (B,H,L_q,LOCAL_KV)
local_scores  = _apply_score_mask(local_scores, local_mask)
pooled_scores = q_bl @ pooled_sq.swapaxes(-1,-2)             # via gathered rows
...  logsumexp / logaddexp / _split_softmax / weighted sum
```

Same story: full dense QK^T against `local_kv` (shape LOCAL_KV=2175) and
against the per-row gathered pooled rows (topk=512 per query), then
array-mask fill, then dense PV. The FLOP formula's `LOCAL_KV + INDEX_TOPK`
matches what the kernel touches. No block-skipping.

### MLA absorption?

DSv4-Flash uses `num_key_value_heads=1` — a *single shared MLA KV head*
that is broadcast across the 32 query heads on this rank. The bench's
`4·N_HEADS·L·KV·D` formula counts each Q head's QK^T and PV
independently against the shared KV. This matches what the fused SDPA
executes: `q` is shape `(B, H=32, L_BAND, D)` and `kv` is shape
`(B, 1, KV, D)`; mx.fast.sdpa broadcasts H over the shared KV, so each
head produces H·L·KV score entries and H·L·D output entries — exactly
the `4·H·L·KV·D` figure.

The alternative *"MLA absorption"* trick (fold wo·wkv into a latent-space
QK computation to avoid ever materializing full head-KV) is a **decode-time
optimization** that reduces FLOPs when L is small; for L=1024 prefill
rows the absorbed form costs MORE than the materialized form (extra
per-latent projection dominates the saving) and MLX / mlx-lm's DSv4
prefill path does not do it (grep for "absorb" / any latent-fold in
`SparseCompressedAttention` / `CompressedAttention` / `_sparse_pooled_attention_inner`
returns nothing). **The kernel materializes per-head KV; the bench
matches that; no MLA-absorption inflation.**

## Answering the flagged questions

### Q1: Does the FLOP count assume full dense (unmasked) attention while the real workload is causally masked?

**Yes** — but "yes" here means the FLOPs are counted for *work the kernel
actually performs*, not for work the kernel avoids. So the "achieved
TFLOPS" numerator equals `real dispatched FMAs / wall time` and is
**not inflated relative to executed work**. The 61.7% / 79.1% figures
are honest measurements of "GEMM ops per second at this KV shape."

### Q1 (continued): Depth-swept wasted-FLOP fractions if a hypothetical block-skipping kernel *could* skip masked FMAs

`attn.sdpa.compressed` at production `L_BAND=1024`, `sw=128`,
`LOCAL_KV=2175`:
- Per-query, local-side visible fraction = 128/2175 = 5.9%.
- Pool side (per-row causal with ratio=128): at deep context, `q_off ≫ ratio`
  so essentially all POOL rows are visible per query row (~99% avg).
- Wasted-FLOP fraction against the currently-counted `LOCAL_KV+POOL`:

| context | POOL_R128 | KV | wasted% |
|---|---|---|---|
| 100K | 782 | 2957 | **69.2%** |
| 220K | 1719 | 3894 | **52.6%** |
| 500K | 3907 | 6082 | **33.7%** |

(dominated by the local-side 94% waste; ratio shrinks as the pool grows
with depth and dilutes the local's share)

`attn.sdpa` (sparse) at `LOCAL_KV=2175`, `INDEX_TOPK=512`:
- Local visible 128/2175 = 5.9%; pool always exactly topk rows,
  100% visible (gather already limits to k).
- Wasted-FLOP fraction: (2175 - 128) / (2175 + 512) = **76.2%**,
  **depth-invariant** (both terms are chunk-shape constants).

### Q1 (continued): E2E opportunity of a causal+SW block-skipping kernel

`span_share × wasted_frac` at 220K (from `docs/dsv4-220k-prefill-span-profile-2026-08-18.md`):
- `attn.sdpa.compressed`: 11.8% × 52.6% = **~6.2% e2e best case**
- `attn.sdpa`:           13.6% × 76.2% = **~10.4% e2e best case**

**Big honest caveat, do NOT quote either of these as "extra headroom
found."** Two reasons:
1. That "best case" assumes the block-skipping kernel runs the surviving
   FMAs at the *same* per-FMA throughput as the current dense kernel.
   Real block-sparse SDPA kernels typically pay a per-block indexing/gating
   overhead (5-25% FMA-rate cost). A realistic ceiling is closer to
   **~3-5% e2e for `attn.sdpa`**, `~2-3% e2e for `attn.sdpa.compressed`**.
2. This is precisely the D=512 fused Metal SDPA design space that
   `exo-sdpa-fusion-analysis` recorded as **CLOSED (2026-07-14/16, "ALL
   LEVERS EXHAUSTED, 339 tok/s IS THE CEILING")**. The `bq=8` and `bq=16`
   variants of the D=512 fused kernel were built, correctness-validated,
   and measured NEUTRAL on real hardware. So the "opportunity" is not
   actually a new opportunity — it is the same one already-attempted and
   closed for structural MMA-tiling reasons on M4 Max. Do not resurrect
   without a genuinely new mechanism.

### Q2: MLA absorption?

**No inflation**. The formula matches the materialized-per-head KV
computation the live kernel executes. MLA absorption is not used at
prefill (would cost more FLOPs there). Verified by reading
`SparseCompressedAttention.__call__`,
`CompressedAttention.__call__`, `_sparse_pooled_attention_inner`,
`_sparse_pooled_attention` — no absorbed/latent QK path exists in any
prefill branch.

### Q3: Restated T7 aggregate

Since the bench numerators reflect real dispatched FMAs (not the
"useful" post-mask FMAs), **there is nothing to renumber in the T7
70.0%/49.8% aggregate**. It is not inflated by causal-masking or
MLA-absorption effects.

The only per-span line-item honest to add is the small mask-application
overhead (3.2%/5.7%), which trims the aggregate marginally:

| version | achieved (TF, weighted) | ceiling (TF, weighted) | %-of-ceiling | e2e effective |
|---|---|---|---|---|
| T7 published (2026-08-22) | 9.00 | 12.86 | **70.0%** | **49.8%** |
| Corrected for mask-apply overhead only | 8.87 | 12.86 | **69.0%** | **49.1%** |

**These are the corrected T7 figures. Delta = -1.0pp span-conditional, -0.7pp
end-to-end. Well within the doc's own stated measurement uncertainty.
Materially unchanged.** The campaign's headroom picture does not shift.

### Q3 (continued): Does the "hand-rolled split-softmax SDPA worth ~1.04x e2e" claim survive?

**Yes, unchanged.** That claim comes from the same bench measuring the
sparse-path split-softmax chain (`_sparse_pooled_attention_inner`) at
25.06 ms vs a same-FLOP fused `mx.fast.scaled_dot_product_attention` at
19.49 ms (25.06/19.49 = 1.286x isolated; scaled by the 13.6% wall share
and the sparse/local layer mix, ~1.04x total prefill). Nothing in this
audit changed either number: the achieved TFLOPS and the fused-equivalent
ceiling are both measured against the same (unblocked) FMA count, so the
*ratio* between them — which is what the ~1.04x claim depends on — is
unaffected by the mask-block-skipping question. The 1.04x claim
**survives**.

## Verdict (one paragraph)

The T7 SDPA achieved-TFLOPS denominators are **not** inflated by causal-
masking or MLA-absorption effects. The bench formula counts what the
kernel actually executes; the kernel does not block-skip masked FMAs
(measured mask-on vs mask-off delta is 3-6% mask-apply overhead, not
FMA savings); MLA absorption is not used at prefill. The T7
70.0%/49.8% figures move by at most **-1.0pp / -0.7pp** once the
mask-application overhead is honestly subtracted — well inside the
document's own stated uncertainty. The campaign's headroom picture is
**unchanged**. The one thing that could look like "hidden headroom"
found by this audit — a causal+SW *block-sparse* SDPA kernel that
skipped ~53-76% wasted FMAs, worth 6-10% e2e best-case — is exactly the
D=512 fused Metal SDPA design space `exo-sdpa-fusion-analysis` already
closed on structural MMA-tiling grounds. Do NOT re-open it. The
"hand-rolled split-softmax SDPA worth ~1.04x e2e" claim from
`dsv4-attention-kernel-efficiency-2026-08-18.md` survives unmodified,
because it is a ratio between two same-denominator measurements.

## Honest uncertainty flags (things code reading alone cannot fully settle)

1. **Does `mx.fast.scaled_dot_product_attention` on a bool array mask
   truly issue every masked FMA, or does its Metal implementation have a
   tile-level "all-masked-tile" short-circuit?** The mask-on vs mask-off
   3-6% delta is consistent with "no short-circuit" but a partial
   short-circuit (e.g. skip a rare all-masked tile only) would not show
   up as a big delta with a 95%-dense random mask. A definitive answer
   requires reading `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`
   (mlx submodule) — deferred as it does not move any T7 conclusion:
   even a full short-circuit would only *reduce* achieved-TFLOPS
   (making the current 61.7%/79.1% overstated, i.e. MORE headroom in
   the fused-kernel direction, not less), and that fused-kernel
   direction is already closed regardless.
2. **The wasted-FLOP percentages assume "every mask-blocked FMA is
   pure waste."** In the sparse path, the local-side split-softmax
   normalizer is computed over ALL LOCAL_KV=2175 scores (not just the
   sliding window) so that the split-softmax stays exact when combined
   with the pool-side normalizer. If a hypothetical block-skipping
   kernel had to preserve that normalizer property, the "achievable"
   savings shrink further. Not modeled here.
3. **The per-row visibility fraction on the pool side assumed ~99% at
   220K.** True average is slightly lower (rows near the top of the
   band see slightly fewer pool rows due to the per-row-causal rule
   with ratio=128). Recomputed exactly would move the "wasted%" for
   `attn.sdpa.compressed` by no more than 1-2pp — does not change any
   verdict above.

# Step-4 partition spec — PAPER ONLY (do NOT implement)

**Signal.** Uniform->skew is -14.7% (8.85 -> 7.55 TF, E=192/N=12288). It is
fully attributable to block-count inflation from partial tail blocks:
`pad_waste = nb_real*64 / N` is 1.00 (uniform, 192 full blocks) vs 1.70 (skew,
326 blocks), and a balanced-assignment construction (same skew expert weights,
but every expert's count rounded to a multiple of 64 -> 192 full blocks)
recovers to 9.04 TF (+19.6% vs skew, ~102% of uniform). So the loss is
M-packing waste, not an intrinsic kernel property.

**Why it matters at production scale.** At the real chunk the same effect is
worse: all 384 experts are touched, so 12288 pairs produce 397 blocks (mean
M=31, pad_waste 2.07) vs the N/64 lower bound of 192 — the production kernel
pays ~2x the per-block fixed cost (weight-tile decode + barriers + tail idle)
that the ideal packing would. This is the largest single driver of the real
chunk's 6.63 TF (44% of the 15.15 TF shape-roofline). Rows cannot be moved
across experts (weights differ), so the fix is a partition change, not a data
reshuffle.

**Spec (M-major / work-stealing segmented GEMM).** Replace the current
grid = `(out_e/64, nb_max)` — one threadgroup per (expert-block, 64-col tile),
so partial blocks waste most of their M — with grid = `(out_e/64, ceil(N/64))`,
exactly `N/64` threadgroups. Each threadgroup fills its 64-row M-tile by
consuming a static M-major schedule (or a device-side atomic work-queue) of
`(expert, row0, len)` items from `blk_tab`, concatenating consecutive items'
rows and gathering the x rows device-side through a per-row token map. Each
8-row mma fragment keeps its own expert index into `blk_tab` and decodes its
weight tile from that expert (numerics unchanged — every output row still uses
only its own expert's weights, bit-identical). Effect: every threadgroup does a
full 64 rows of useful work, removing the partial-block M waste.

**Expected bound.** Step-4 balanced = 9.04 TF vs skew 7.55 (E=192). At the real
chunk the same removal of the 2.07x block inflation projects 6.63 -> ~8.9 TF
(≈ uniform level), i.e. the MoE GEMM shortfall closes from ~44% to ~59% of the
shape-roofline. Cost/risk: a per-fragment expert index and an in-kernel x-row
gather across expert boundaries; weight-tile traffic per expert may rise
slightly if an expert's rows split across more threadgroups than ceil(len/64).
Shippable, battery-gated, ~1.5-2% e2e per the Loop-3 pricing — but do NOT build
it yet (the parent PM owns the gate verdict; the dequant arm below shows the
gap is not solely a partition/dequant problem either).

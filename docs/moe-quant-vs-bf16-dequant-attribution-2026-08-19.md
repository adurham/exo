# MoE 62.6%-of-ceiling gap is INTRINSIC to the ragged matmul, NOT dequantization (2026-08-19)

## Question (M2)

`bench/moe_production_class_bench.py` measures the real mlx-lm `SwitchGLU` at
DeepSeek-V4-Flash TP-rank prefill shapes running at **62.6% of a matched-shape
dense-GEMM ceiling** (L=2048 per-rank, i.e. `EXO_PREFILL_STEP_SIZE=4096`).
`docs/moe-per-stage-gpu-breakdown-2026-08-18.md` already proved the missing time
is *inside* the three expert GEMMs (gather_sort + scatter_unsort are only ~4-5%
combined), leaving two live hypotheses:

- **(A) Dequantization overhead** — the quantized kernel unpacks 4-bit/8-bit
  weights + scales to bf16 before the MAC, burning ALU/registers the dense fp16
  GEMM never pays.
- **(B) Intrinsic to the ragged gathered matmul** — per-expert run lengths are
  small and uneven, so tile occupancy / weight-stream reuse is poor regardless
  of weight dtype.

## Method

`bench/moe_quant_vs_bf16_ab.py` (new, committed) runs a **2x2 at identical shape
and identical routing indices**, all arms interleaved in one process so
clocks/thermals are fair:

|                      | bf16 weights   | quantized weights   |
|----------------------|----------------|---------------------|
| **gathered (ragged)**| `gather_mm`    | `gather_qmm` ← production |
| **dense (M rows)**   | plain `matmul` | `quantized_matmul`  |

- `dense_bf16` is the ceiling the 62.6% was measured against.
- `dense_quant / dense_bf16` isolates **pure dequant cost** (no raggedness).
- `gather_bf16 / dense_bf16` isolates **pure raggedness cost** (no dequant).
- `gather_quant / dense_bf16` is the production number, reproduced.

Routing uses the same skewed-prior top-6-of-256 generator as the original bench,
so the numbers are directly comparable. Two quant modes measured so the answer
is not recipe-vs-checkpoint dependent (per `MOE_KERNEL_HANDOFF.md`'s "PREMISE
CORRECTION"): **mxfp4 g32** (what `make_quantization_config()` specifies for
routed experts) and **affine 8-bit g64** (what the deployed checkpoint's
`config.json` actually says). Run on this laptop's M4 Max GPU.

## Results — L=2048 per-rank (production chunk), % of dense-bf16 ceiling

3 independent runs of the new bench, plus a same-session cross-check re-run of
the original bench which reproduced **62.6% exactly**:

| arm | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| `dense_bf16` (ceiling) | 100% | 100% | 100% | 100% |
| `dense_mxfp4_g32` (dequant only) | 88.6% | 80.3% | 84.9% | **~84.6%** |
| `dense_affine8_g64` (dequant only) | 85.9% | 82.1% | 81.2% | **~83.1%** |
| `gather_bf16` (raggedness only) | 58.7% | 55.0% | 57.8% | **~57.2%** |
| `gather_mxfp4_g32` (production) | 59.2% | 63.6% | 63.9% | **~62.2%** |
| `gather_affine8_g64` (production) | 60.6% | 62.7% | 59.2% | **~60.8%** |

Absolute (run 1): dense bf16 10.75 TFLOPS; gather bf16 6.31; gather mxfp4 6.36.

## Conclusion: hypothesis (B). The gap is intrinsic to the ragged matmul.

**Removing quantization entirely does not recover the gap.** Plain bf16 weights
in the gathered path run at ~57.2% of ceiling — *statistically identical to, and
if anything slightly worse than*, the quantized production path's ~62.2%. If
dequant were the cause, `gather_bf16` would have landed near 100%; it lands at
the production number.

Conversely, dequant *is* measurable but only in the dense regime: **~15-17%**
(`dense_quant` ≈ 83-85% of ceiling), and it is essentially **the same cost for
mxfp4 g32 and affine8 g64** — so the checkpoint-format question is irrelevant to
this gap.

**Quantization is net-neutral-to-BENEFICIAL in the production path.** In the
ragged gathered regime the kernel is weight-*bandwidth*-bound, not ALU-bound:
4-bit weights are 4x smaller to stream, and that saving fully offsets the
dequant ALU. The gathered quantized arm beat the gathered bf16 arm in all three
runs. The naive multiplicative model (`dequant% x raggedness%`, predicting
~45-52%) **over-predicts the loss** — the two effects are not independent
because they bottleneck on different resources.

### What this means for optimization

1. **A "dequant-free MoE" / bf16-expert-weights play is DEAD.** It would cost 4x
   the weight memory and 4x the weight bandwidth for *negative* speedup. Do not
   pursue. (Also kills any "pre-dequantize experts to bf16 and cache" variant for
   the same reason.)
2. The remaining ~40% is **tile occupancy / weight-stream reuse at small, uneven
   per-expert run lengths** — consistent with, and now independently confirming,
   the NO-GO conclusions in `docs/moe-tile-geometry-retune-dead-end-2026-08-18.md`
   and `bench/moe_run_length_tile_waste.py`.
3. The raggedness penalty is strongly **shape-dependent, and it is the dominant
   term at every size**: `gather_bf16` is 19.1% of ceiling at L=256, 40.2% at
   L=1024, 57.2% at L=2048, while `dense_quant` stays flat at ~83-89%
   throughout. Larger chunks amortize the raggedness; dequant cost never moves.
   This is the same mechanism behind the measured chunk-size efficiency ladder in
   `docs/dsv4-prefill-step-size-4096-retest-2026-08-18.md` (43% @ L=512, 62.6% @
   L=2048, 72.0% @ L=4096) — that ladder is a *raggedness-amortization* curve,
   not a dequant curve.

Any further attack on this gap must be **inside the gathered GEMM kernel's
handling of short/uneven expert runs** (Metal tile geometry, run-length-aware
dispatch, occupancy), and the dtype of the weights is not a lever.

## Files

- `bench/moe_quant_vs_bf16_ab.py` — new, committed, reusable.
  `--lengths` (default 1024 2048), `--iters`, `--json-out`.
  Full sweep runs in ~16s.

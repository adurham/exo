# Is the 62.6% MoE efficiency actually anomalous? (M3, 2026-08-19)

**Verdict: the gap is real and MoE-specific, but it is NOT ~37%.** The true
MoE-attributable gap is **~28% at production L=2048** (MoE runs at 72% of the
correct dense-quantized ceiling), and it is dominated by **routing raggedness**,
not by the mxfp4 quantization tax and not by gather/scatter plumbing.

Bench: `bench/moe_vs_dense_qmm_isolation.py` (laptop M4 Max, 32-core GPU,
isolated, MLX 0.32.0.dev20260804+ac73d0c9). Raw:
`bench/results/moe_vs_dense_qmm_isolation.json`.

## The problem with the headline number

`bench/moe_production_class_bench.py` divides the real SwitchGLU MoE block by a
matched-shape **dense fp16 GEMM**. That denominator conflates two unrelated
penalties: (a) mxfp4 quantized matmul is slower than fp16 GEMM at these shapes
even with no MoE at all, and (b) actual MoE overhead. You cannot attribute the
gap to MoE kernels until you subtract (a).

## Four-tier isolation, identical per-GEMM M/K/N

DSv4-Flash per-TP-rank shapes: hidden=4096, inter/rank=1024, experts=256,
top_k=6, mxfp4 g=32. FLOPs counted activated-only (gate+up+down), so all four
tiers do the same nominal arithmetic.

| Tier | What |
|---|---|
| T1 | dense fp16 GEMM — the current denominator |
| T2 | dense **mxfp4 qmm**, one weight, no routing/gather/scatter |
| T3 | gather_qmm via SwitchGLU, degenerate routing (all rows → expert 0) |
| T4 | real SwitchGLU, ragged top-6-of-256 (production) |

### Ratios (two independent runs; the deliverable)

| Ratio | Meaning | L=1024 | L=2048 |
|---|---|---|---|
| T4/T1 | **the headline "62.6%"** | 53.5 / 53.7% | **62.5 / 62.9%** |
| T2/T1 | quantized-matmul tax (no MoE involved) | 87.7 / 86.5% | 86.9 / 86.8% |
| T4/T2 | **TRUE MoE-specific efficiency** | 61.0 / 62.1% | **72.0 / 72.5%** |
| T3/T2 | gather/scatter plumbing alone | 87.9 / 89.8% | 88.0 / 88.8% |
| T4/T3 | raggedness / short per-expert runs | 69.4 / 69.2% | 81.8 / 81.7% |

L=2048 reproduces the reported 62.6% (62.5 / 62.9%), confirming this is the same
measurement, just decomposed.

Representative absolute timings (run 1, L=2048, M=12288): T1 26.810 ms /
11.53 TFLOPS, T2 30.854 ms / 10.02, T3 35.074 ms / 8.82, T4 42.865 ms / 7.21.

**Methodology note:** the four tiers are timed **round-robin within each
repetition**, not tier-by-tier. Sequential timing let GPU thermal drift alias
onto the tier comparison — the first version of this bench showed T4/T2 swinging
61.5% → 72.9% at L=1024 between back-to-back runs. With interleaving, absolute
TFLOPS still drift (run 2 was ~30% slower overall on a hot GPU) but every ratio
above reproduces within ~1 point. Only the ratios are trustworthy here.

## Findings

1. **The quantized-matmul tax is small and flat: ~13%.** Dense mxfp4 qmm at the
   exact production M/K/N sustains **86.5-87.7% of the fp16 GEMM ceiling** at
   both L values and across both runs. mxfp4 is NOT why MoE looks bad.

2. **The MoE-attributable gap is ~28%, not ~37%.** Against the correct
   dense-quantized denominator, production MoE runs at **72.0-72.5%** (L=2048) /
   61.0-62.1% (L=1024).

3. **Gather/scatter plumbing is cheap — ~12%.** With raggedness eliminated (every
   row → expert 0) the gather path still hits 88-90% of dense qmm, stable across
   both L and both runs. This independently corroborates
   `bench/moe_gpu_capture_profile.py`, which found gather_sort + scatter_unsort
   ≈ 4-5% of real per-call time. Optimizing those stages cannot recover more than
   ~12 points, realistically far less.

4. **Raggedness is the dominant MoE cost: T4/T3 = 69% (L=1024) / 82% (L=2048).**
   Holding the gather machinery constant and changing *only* the routing
   distribution costs 18-31%. The distribution is brutal: 169-181 of 256 experts
   used, **median M/expert 7 (L=1024) and 14 (L=2048), min 1, max 808-1653**.
   Nearly every expert run is a tiny GEMM with single-digit M against K=4096 —
   exactly the regime where a quantized GEMM kernel cannot fill its tiles.

5. **The gap closes as L grows, confirming the mechanism.** L=1024 → L=2048
   doubles median M/expert (7 → 14) and true MoE efficiency rises 61% → 72%,
   while the quant tax (87%) and plumbing cost (88%) stay flat. Efficiency here
   is a function of per-expert run length and essentially nothing else.

## Consequences for optimization work

- **Recompute the roofline.** The correct ceiling for gather_qmm is dense **mxfp4
  qmm (~87% of fp16)**, not fp16 itself. The "37% MoE gap" is overstated by ~10
  points and should be restated as **~28% at L=2048**.
- **Gather/scatter optimization is dead as a major lever** — ≤12 points available
  by this measurement, and the GPU-capture profile already put the real figure at
  4-5%.
- **Per-expert run length is the only lever that matters.** This is direct
  quantitative support for the M-batched quantized GEMV work flagged as "THE ONE
  BIG OPEN ITEM" in `MOE_KERNEL_HANDOFF.md`: median M/expert of 7-14 is precisely
  the M≤8 regime that design targets. It also explains why the tile-size sweep
  failed (`docs/moe-tile-geometry-retune-dead-end-2026-08-18.md`) — retuning tile
  geometry cannot help when the fundamental problem is that M is 7.
- **Cross-check against attention.**
  `docs/dsv4-attention-kernel-efficiency-2026-08-18.md` measured attention at
  83-85% of matched-shape dense fp16. Against that same fp16 denominator MoE is
  62.5%; against its own correct quantized denominator it is 72%. MoE is less
  anomalous relative to attention than the raw comparison implied, but still
  meaningfully behind.

## Caveats

- Laptop M4 Max (32-core GPU) vs the cluster's Mac Studio M4 Max (40-core).
  Ratios are the deliverable and transfer; absolute ms will be faster on the
  Studios.
- Single-rank, in-process. No TP collectives, no inter-node cost — deliberately,
  so the measurement isolates kernel efficiency.
- T3 (all rows → expert 0) is an idealized lower bound on routing cost: it keeps
  the gather/scatter code path but hands the GEMM one maximally long run. It is a
  control, not a reachable production state.
- Absolute TFLOPS in any single run are thermally contingent and should not be
  quoted standalone; see the methodology note above.

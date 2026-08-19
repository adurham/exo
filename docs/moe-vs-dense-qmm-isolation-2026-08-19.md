# Is the 62.6% MoE efficiency actually anomalous? (M3, 2026-08-19)

**Verdict: mostly yes — the gap is real and MoE-specific, but it is NOT ~37%.
The true MoE-attributable gap is ~27% at production L=2048, and it is dominated
by routing raggedness, not by the quantized-matmul tax and not by gather/scatter
plumbing.**

Bench: `bench/moe_vs_dense_qmm_isolation.py` (laptop M4 Max, 32-core GPU, isolated,
MLX 0.32.0.dev20260804+ac73d0c9). Raw: `bench/results/moe_vs_dense_qmm_isolation.json`.

## The problem with the headline number

`bench/moe_production_class_bench.py` divides the real SwitchGLU MoE block by a
matched-shape **dense fp16 GEMM**. That denominator conflates two unrelated
penalties: (a) mxfp4 quantized matmul is slower than fp16 GEMM at these shapes
even with no MoE at all, and (b) actual MoE overhead. You cannot attribute the
gap to MoE kernels until you subtract (a).

## Four-tier isolation, identical per-GEMM M/K/N

DSv4-Flash per-TP-rank shapes: hidden=4096, inter/rank=1024, experts=256, top_k=6,
mxfp4 g=32. FLOPs counted activated-only (gate+up+down), so all four tiers do the
same nominal arithmetic.

| Tier | What | L=1024 (M=6144) | L=2048 (M=12288) |
|---|---|---|---|
| T1 | dense fp16 GEMM (current denominator) | 13.445 ms / 11.50 TF | 26.611 ms / 11.62 TF |
| T2 | dense **mxfp4 qmm**, one weight, no routing | 15.353 ms / 10.07 TF | 30.698 ms / 10.07 TF |
| T3 | gather_qmm, degenerate routing (all rows → expert 0) | 17.156 ms / 9.01 TF | 34.716 ms / 8.91 TF |
| T4 | real SwitchGLU, ragged top-6-of-256 (production) | 24.946 ms / 6.20 TF | 42.248 ms / 7.32 TF |

Derived ratios:

| Ratio | Meaning | L=1024 | L=2048 |
|---|---|---|---|
| T4/T1 | **the headline "62.6%"** | 53.9% | **63.0%** |
| T2/T1 | quantized-matmul tax (no MoE involved) | 87.6% | 86.7% |
| T4/T2 | **TRUE MoE-specific efficiency** | 61.5% | **72.7%** |
| T3/T2 | gather/scatter plumbing alone | 89.5% | 88.4% |
| T4/T3 | raggedness / short per-expert runs | 68.8% | 82.2% |

L=2048 reproduces the reported 62.6% to within noise (63.0%), so this is the same
measurement, just decomposed.

## Findings

1. **The quantized-matmul tax is small.** Dense mxfp4 qmm at the exact production
   M/K/N runs at 10.07 TFLOPS — **86.7% of the fp16 GEMM ceiling**, flat across
   both L values. So mxfp4 is NOT the reason MoE looks bad. Only ~13 of the ~37
   headline points are "quantization", and even that is generous.

2. **The MoE-attributable gap is ~27%, not ~37%.** Against the correct
   dense-quantized denominator, production MoE runs at **72.7%** (L=2048) /
   61.5% (L=1024). Real, but a third smaller than the headline implies.

3. **Gather/scatter plumbing is cheap — ~11%.** With raggedness eliminated
   (every row routed to expert 0) the gather path still hits 88-90% of dense qmm.
   This independently corroborates `bench/moe_gpu_capture_profile.py`, which found
   gather_sort + scatter_unsort ≈ 4-5% of real per-call time. Optimizing the
   gather/scatter stages cannot recover more than ~11 points, and realistically
   much less.

4. **Raggedness is the dominant MoE cost: T4/T3 = 68.8% / 82.2%.** Holding the
   gather machinery constant and only changing routing from uniform to realistic
   skewed top-6, throughput drops by 18-31%. The routing distribution is brutal:
   169-181 of 256 experts used, **median M/expert 7 (L=1024) and 14 (L=2048),
   min 1, max 808-1653**. Nearly every expert run is a tiny GEMM with a
   single-digit M against K=4096. That is exactly the regime where a quantized
   GEMM kernel cannot fill its tiles.

5. **The gap closes as L grows, which confirms the mechanism.** L=1024 → L=2048
   doubles median M/expert (7 → 14) and true MoE efficiency jumps 61.5% → 72.7%,
   while the quant tax and the plumbing cost stay flat. Efficiency here is a
   function of per-expert run length, nothing else.

## Consequences for optimization work

- **Recompute the roofline.** The correct ceiling for gather_qmm is dense **mxfp4
  qmm at 10.07 TFLOPS**, not fp16 at 11.6. Any claim of a "37% MoE gap" is
  overstated by ~10 points and should be restated as ~27% at L=2048.
- **Gather/scatter optimization is dead as a major lever** — ≤11 points available,
  and the GPU-capture profile already put the real number at 4-5%.
- **The only lever that matters is per-expert run length.** This is direct
  quantitative support for the M-batched quantized GEMV work already flagged as
  "THE ONE BIG OPEN ITEM" in `MOE_KERNEL_HANDOFF.md`: median M/expert of 7-14 is
  precisely the M≤8 regime that design targets. It also explains why the tile-size
  sweep failed (`docs/moe-tile-geometry-retune-dead-end-2026-08-18.md`) — retuning
  tiles cannot help when the fundamental problem is that M is 7.
- **Cross-check against attention.** `docs/dsv4-attention-kernel-efficiency-2026-08-18.md`
  measured attention at 83-85% of matched-shape dense fp16. Against the same fp16
  denominator MoE is 63.0%; against its own correct quantized denominator it is
  72.7%. The MoE kernel is less anomalous relative to attention than the raw
  comparison suggested, but still meaningfully behind.

## Caveats

- Laptop M4 Max (32-core GPU) vs the cluster's Mac Studio M4 Max (40-core). Ratios
  are the deliverable and transfer; absolute ms will be faster on the Studios.
- Single-rank, in-process. No TP collectives, no inter-node cost — deliberately, so
  the measurement isolates kernel efficiency.
- T3 (all rows → expert 0) is an idealized lower bound on routing cost: it keeps
  the gather/scatter code path but hands the GEMM one maximally long run. It is not
  a reachable production state; it is a control.

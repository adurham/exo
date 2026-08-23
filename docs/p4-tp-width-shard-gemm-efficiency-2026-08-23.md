# P4: does TP=2 width-sharding create skinny GEMM shapes that hurt prefill efficiency? — NEGATIVE, and mildly the opposite — 2026-08-23

## Question

Under exo's `DeepseekV4ShardingStrategy` (MoE-only TP), the routed-expert
FFN is WIDTH-sharded: `gate_proj`/`up_proj` are `all_to_sharded` (N
halved, 2048 → 1024 per rank) and `down_proj` is `sharded_to_all` (K
halved, 2048 → 1024 per rank). The hypothesis under test: those halved
dimensions produce "skinny" GEMMs whose tile/simdgroup efficiency is
measurably worse than the unsharded width, meaning TP itself imposes a
per-rank efficiency tax on prefill that a different sharding layout
could recover.

Three sub-questions:
- **(a)** does halved N (gate/up) or halved K (down) degrade achieved
  efficiency vs the unsharded width?
- **(b)** is mxfp4 dequant fused into the actually-dispatched GEMM
  kernel, or is it a separate pass (which would change the roofline
  denominator)?
- **(c)** does any NOT-previously-closed lever fall out of this?

This is an **efficiency** question, not a wall-time question — the
per-rank GEMM legitimately does half the work, so every arm is scored on
achieved TFLOPS / GB/s (work ÷ time), never on raw ms.

## Method

Bench: `bench/p4_tp_width_shard_gemm_efficiency.py`, run with the repo
venv on an M4 Max (`applegpu_g16s`, MLX `0.32.0.dev20260804+ac73d0c9`).
Results: `bench/results/p4_tp_width_shard.json`.

Per `exo-perf-tuning` "Microbench Accuracy" and the 2026-08-22 P0
retraction (`docs/switch-mlp-bandwidth-artifact-retraction-2026-08-22.md`):

- **No `mx.eval()` inside the timed loop.** The P0 retraction showed a
  per-iteration eval charges ~172µs/call of host/dispatch overhead to
  the kernel and manufactured a fake 3.6x "headroom". Here the graph is
  dependency-CHAINED with ONE eval per 12 calls — what production's
  per-layer chain actually looks like.
- ≥3 warm-up iterations, `mx.eval` + `mx.synchronize` around the timed
  region, 5 trials.
- **Identical routing indices reused across every width arm**, so
  per-expert raggedness is held EXACTLY constant and width is the only
  variable.
- Roofline framing is `max(compute_time, memory_time)`, not the sum.

### What is actually dispatched (verified, not assumed)

`SwitchGLU.__call__` does `mx.expand_dims(x, (-2,-3))`, so
`GatherQMM::eval_gpu` always sees outer `M == 1`. At a 2048-token TP
prefill chunk: `B = 2048 × top_k 6 = 12288` rows, `E = 256` experts, so
`B/E = 48`.

| tier | kernel | gate | fires? |
|---|---|---|---|
| 1 | `gather_qmv_rhs` | needs `B/E <= gather_qmv_rhs_max_be()` == 6 | **NO** |
| 2 | `gather_qmm_rhs` | needs `M==1 && B>=16 && sorted && B/E>=4` | **YES** |

Confirmed **empirically**, not just by reading the gate: raising
`MLX_GATHER_QMV_RHS_MAXBE` to 64 (which puts `B/E=48` in tier-1 range)
produced a material timing change, proving tier 2 is what runs at the
default. Tile geometry is therefore `bm=16 bn=32 bk=32 wm=1 wn=2`; the
`bm=64` `_nax` variant is gated on `is_nax_available()`, which is **false**
on M4 Max (architecture gen 16 < the required 17).

Grid is `((N+bn-1)/bn, (M+bm-1)/bm, 1)`, so N-direction partial-tile
waste is `ceil(N/32)*32 / N`. Both per-rank (1024) and unsharded (2048)
widths are exact multiples of 32 → **zero partial-tile waste either way,
analytically**. The bench therefore tests the second-order concern tile
arithmetic cannot answer: whether the shorter K reduction or narrower N
costs achieved throughput via reduced arithmetic intensity or weaker
latency hiding.

## Real results

### (a) MoE arm — real `SwitchGLU`, mxfp4, top-6-of-256, L=2048

Production routing skew held constant across arms (`skew=power`:
12288 rows, 256 experts hit, median 26.5 rows/expert, max 1906, min 6).

| `moe_intermediate` | implied TP | ms/call | achieved TFLOPS | achieved GB/s | roofline eff |
|---|---|---|---|---|---|
| 512 | TP=4 | 22.57 | 6.851 | 47.9 | 11.31% |
| **1024** | **TP=2 (production)** | **43.87** | **7.049** | **44.7** | **10.55%** |
| 2048 | TP=1 (unsharded) | 91.78 | 6.739 | 40.6 | 9.57% |

**Achieved TFLOPS is FLAT across a 4x width range** — spread
6.739–7.049, max/min ratio **1.046**. Wall time scales essentially
linearly with width (actual÷linear-prediction = 1.000 / 0.972 / 1.017),
i.e. near-perfect weak scaling. If narrow widths were tile-inefficient,
the narrow arms would show *lower* TFLOPS and *super*-linear ms. Neither
appears.

### (a) Dense arm — isolated mxfp4 `qmm` at the exact production shapes

| shape | M | K | N | TFLOPS | % of compute peak |
|---|---|---|---|---|---|
| gate/up per-rank (TP=2) | 12288 | 4096 | 1024 | 9.939 | **85.2%** |
| gate/up unsharded | 12288 | 4096 | 2048 | 9.598 | 82.3% |
| gate/up TP=4 | 12288 | 4096 | 512 | 9.574 | 82.1% |
| down per-rank (TP=2) | 12288 | 1024 | 4096 | 9.510 | **81.6%** |
| down unsharded | 12288 | 2048 | 4096 | 9.267 | 79.5% |
| down TP=4 | 12288 | 512 | 4096 | 8.942 | 76.7% |

Derived ratios (per-rank ÷ unsharded):

- gate/up, N halved: **+3.6%** achieved TFLOPS (9.939 vs 9.598)
- down, K halved: **+2.6%** achieved TFLOPS (9.510 vs 9.267)

**The sharded shapes are slightly FASTER per unit work, not slower.**
The hypothesised penalty does not exist at TP=2; the sign is inverted.
Only at TP=4 does the trend finally bend the "expected" way for the
K-reduction case (down K=512: −3.5% vs unsharded) — i.e. a real skinny-K
penalty exists in principle, but it begins **beyond** the cluster's
actual TP=2 configuration, not at it.

Attention (replicated in exo, so it has no per-rank skinny dim at all)
was measured for completeness: a hypothetical head-sharded variant
(N=32768 → 16384) changes efficiency by **−0.55%** (8.551 → 8.504
TFLOPS) while halving wall time 1.99x — confirming attention sharding
would be a pure work-splitting win with no shape penalty, but that is a
*different* lever (and see §"Not a new lever" below).

### (b) mxfp4 dequant IS fused in-kernel — confirmed by code reading

Traced `mlx/backend/metal/kernels/fp_quantized.h`: the dispatched
`fp_gather_qmm_rhs` kernel loads weights through `QuantizedBlockLoader`,
which dequantizes **inside the threadgroup load path** as it streams
blocks into shared memory. There is no separate dequant pass and no
materialized fp16/bf16 weight tensor. The roofline denominator that
counts only the packed mxfp4 bytes is therefore **correct as used** —
this validates rather than corrects the existing prefill roofline work.

## Conclusion — NEGATIVE result

**(a) Refuted, with the sign inverted.** TP=2 width-sharding does not
create a skinny-GEMM efficiency penalty for DSv4-Flash's prefill-dominant
GEMMs. Both sharded shapes are 2.6–3.6% *more* efficient per unit work
than their unsharded counterparts, and the MoE arm's achieved TFLOPS is
flat (±4.6%) across a 4x width sweep. Analytically the widths are exact
multiples of the 32-wide tile, so zero partial-tile waste was expected —
the measurement confirms no second-order latency-hiding penalty either.

**(b) Confirmed fused.** No correction to the roofline denominator is
needed.

**(c) No new lever.** There is no recoverable efficiency sitting in the
sharded GEMM shapes, because there is no loss there to recover. The
kernels run at **76.7–85.2% of measured dense compute peak**, consistent
with the existing finding that these kernels are already at/near their
achievable ceiling (§3.4, and `docs/lever1-moe-smallm-headroom-2026-08-20.md`).

### Why this is worth having as a negative

It closes a plausible, cheap-sounding structural hypothesis ("TP is
taxing our GEMM shapes") that would otherwise keep resurfacing — and it
closes it with the sign measured, not assumed. It also independently
re-confirms, on a fresh bench with correct chained-eval methodology, that
the MoE GEMM path has no width-related headroom.

### Not a new lever: the attention row

The −0.55% head-sharding figure above is **not** a recommendation to
shard attention. It measures only the isolated GEMM's shape sensitivity.
Attention sharding's actual cost is the added cross-rank communication
and coherence risk, which is a separate, already-investigated question
(`EXO_DSV4_SEQ_SPLIT`, and the subgroup-`all_gather` wedge history in §8).
Nothing here changes that calculus.

## Caveats (stated honestly)

- Measured on a **laptop-class M4 Max** (32-core GPU, 38.7 GB), not the
  Studio nodes. Absolute TFLOPS will differ; the comparison is
  arm-vs-arm at identical shapes on identical hardware in the same
  process, which is what the efficiency question requires. Per the
  standing "microbench wins do not transfer end-to-end" lesson (§12.5),
  this result is used only to REJECT a lever, never to claim one — the
  direction of inference that failure mode does not threaten.
- `PEAK_TFLOPS = 11.66` is the measured dense fp16 square-GEMM peak on
  this same laptop, so "% of compute peak" is internally consistent but
  not comparable to Studio-node figures.
- Single routing-skew distribution (`power`, matched to production's
  ragged top-6-of-256 shape). Raggedness was held constant *by design*
  so width was the only variable; a different skew would change absolute
  TFLOPS but not the width comparison.

Outcome class: **negative result, cleanly measured, lever closed.**

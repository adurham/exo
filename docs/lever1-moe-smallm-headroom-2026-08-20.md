# LEVER 1: Is there real throughput on the table in MoE's small-M kernels?

**Verdict: NO. Close this lever.** At the per-expert token counts DSv4-Flash
actually lives at, the live `gather_qmv_rhs` / `gather_qmm_rhs` kernels are
**already at or slightly ABOVE a perfect dense grouped-GEMM equivalent** that
reads the same weight bytes with zero raggedness. There is no small-M kernel
gap to recover, because the regime is not small-M-inefficiency-bound — it is
**weight-streaming-bound with poor arithmetic intensity**, and the existing
kernels already stream at close to the achievable rate for that access pattern.

Bench: `bench/lever1_smallm_headroom.py`. Raw: `bench/results/lever1_prod_*.json`.
Laptop M4 Max (32-core GPU), MLX 0.32.0.dev20260804+ac73d0c9, isolated single
rank. Ratios are the deliverable; absolute ms will be faster on the 40-core
Studios.

## What the live path actually is (verified, not assumed)

| Fact | Source |
|---|---|
| MoE experts are **mxfp4 g=32**, not the checkpoint's top-level affine-8 | `mlx-lm/mlx_lm/utils.py:430-463` applies `make_quantization_config`'s per-layer mxfp4 override wherever on-disk scales are uint8 |
| hidden 4096, moe_intermediate 2048 → **INTER=1024 per TP rank**, 256 experts, top_k 6 | `config.json` of `deepseek-ai/DeepSeek-V4-Flash-0731` |
| **M always collapses to 1** at `GatherQMM::eval_gpu` | `SwitchGLU.__call__` does `expand_dims(x, (-2,-3))`; confirmed previously in `docs/mxfp4-gather-qmm-rhs-lhs-kernel-2026-08-19.md` |
| So every production call hits an M==1 gate: `gather_qmv_rhs` (B/E in [2,6]) or `gather_qmm_rhs` (B/E≥4) | `mlx/backend/metal/quantized.cpp:1888-1930` |
| Live per-rank MoE row count is **L≈1024** | TP's `prefill_batched` uses the full `EXO_PREFILL_STEP_SIZE=2048` (`generate.py:1384`, NO `//ranks` — that divisor is on the PP path only), and `EXO_DSV4_SEQ_SPLIT=1` halves query rows per rank |

Routing skew was **calibrated, not guessed**: `--skew 2.2` reproduces the
production run-length histogram documented in
`docs/moe-vs-dense-qmm-isolation-2026-08-19.md` (median M/expert 5-11, min 1,
max ~1571, 174-215 of 256 experts used).

## The measurement

Three tiers, identical expert-weight bytes, timed **round-robin interleaved**
and reported as **min-of-25** (not mean):

- `live` — `mx.gather_qmm` at the real SwitchGLU convention; whatever the
  dispatcher picks.
- `ceiling` — dense `mx.quantized_matmul` against all experts viewed as one
  `(E*INTER, HIDDEN)` matrix at M = mean rows/expert. Same bytes, same nominal
  FLOPs, **zero gather and zero raggedness**. A perfect grouped GEMM cannot beat
  this.
- `bwfloor` — same dense qmm at M=1: the pure weight-streaming rate.

## Results (skew 2.2 = production histogram, 2 runs each)

| L (rows/rank) | median M | live GB/s | ceiling GB/s | **headroom (live/ceiling time)** |
|---|---|---|---|---|
| 512 | 4 | 64-66 | 82-84 | **0.79x** |
| **1024 (production)** | **5** | **38-46** | **35-43** | **0.63x** |
| 2048 | 11 | 28 | 34-36 | **1.07x** |

**Headroom < 1.0x means the live kernel is FASTER than the idealized dense
grouped GEMM.** At the production shape it is ~1.6x faster. Reproducible across
runs to ±3%.

## The decisive A/B

`MLX_GATHER_QMV_RHS=0` (disables the M-batched qmv kernel, forcing the steel
`gather_qmm_rhs` tile path) changes the production shape by **under 4%**:

| L | qmv ON | qmv OFF | delta |
|---|---|---|---|
| 512 | 5.40 / 5.55 ms | 5.60 / 5.64 ms | +3% |
| 1024 | 8.41 / 10.20 ms | 9.63 / 9.74 ms | within noise |
| 2048 | 17.06 / 16.81 ms | 17.46 / 18.05 ms | +3% |

The two kernels are interchangeable at production shapes. Whatever
`gather_qmv_rhs` buys, it is not visible here — consistent with the
2026-07-02 handoff's own correction ("~1.01x at the 768-pair shape").

## The one real (negative) finding: MAXBE is correctly tuned

Widening the `gather_qmv_rhs` dispatch window with `MLX_GATHER_QMV_RHS_MAXBE=64`
is a **significant regression** at exactly the shapes it newly captures:

| L | default MAXBE=6 | MAXBE=64 | effect |
|---|---|---|---|
| 1024 | 11.6 ms | 15.5 ms | **-33%** |
| 2048 | 18.0 ms | 32.0 ms | **-78%** |

This independently re-confirms the `B/E <= 8` upper bound reasoning in
`quantized.cpp:1599-1620` on a shape and a routing distribution that bound was
never tested against. **The current default of 6 is correct — do not widen it.**

## Why there is no lever here

`bwfloor` (pure weight streaming, M=1) measures **~340 GB/s** — the machine's
real rate for this access pattern, well under the 546 GB/s spec peak. The live
MoE call runs at 28-46 GB/s of *unique* expert bytes. That looks like a 10x gap
but is not one: at median M=5, the kernel reads a full expert's weights to do
5 rows of work, so it is **arithmetic-intensity starved by construction**, and
the dense ceiling that shares that same starvation (`ceiling`) measures the
same 35-43 GB/s. The kernel is not leaving throughput on the table; the
*routing distribution* is.

This is the same mechanism `docs/moe-vs-dense-qmm-isolation-2026-08-19.md`
identified (raggedness, not tile geometry, not quant format) — but this
measurement adds the part that was missing: **the existing kernels already
capture essentially all of what a better small-M kernel could deliver.** The
M-batched GEMV design flagged as "THE ONE BIG OPEN ITEM" in
`MOE_KERNEL_HANDOFF.md` was built (`gather_qmv_rhs`, 2026-07-02) and this
confirms it is at ceiling — there is no second version of it worth writing.

## Consequences

- **Close LEVER 1.** No small-M kernel rewrite is justified. Do not build
  another M-gated kernel variant; do not retune tile geometry (already a
  documented dead end); do not widen MAXBE.
- The only lever that would move this number is **raising median M/expert**,
  i.e. changing the routing/batching shape, not the kernel. That is bounded:
  the earlier M-sweep found efficiency saturates by M≈32 and flatlines. Getting
  median M from 5 to 32 means ~6x more rows per MoE call, which means larger
  prefill chunks — and `EXO_PREFILL_STEP_SIZE=4096` was already measured as a
  net **~8% e2e regression** (SDPA's linear cost growth outweighs MoE's gain,
  fully root-caused in `docs/dsv4-4096-regression-root-cause-2026-08-19.md`).
  So that door is closed too, for a known reason.
- Net: **MoE GEMM kernel efficiency is not where the remaining prefill
  throughput is.** Redirect to a different part of the budget.

## ADDENDUM 2026-08-31 — re-checked. Verdict UPHELD, headline number RETRACTED.

Scoped local re-check (no cluster contact). **The conclusion below stands: there
is no small-M kernel lever. But the `0.63x` figure in the table above is an
M-alignment artifact and must not be cited.**

**Root cause of the artifact.** `gather_qmm` *and* dense `quantized_matmul` are
both tile-quantized in rows-per-expert at **~32 rows**. Cost per row is a
staircase, not a curve. Measured on the same build/machine as the original:

| dense qmm (the `ceiling` tier) | gather_qmm (the `live` tier) |
|---|---|
| M=32 → 6.75 ms, M=33 → **13.3 ms** (1.97x for one row) | m=32 → 0.954 us/row, m=31 → 1.390, m=33 → 1.359 |

The original run drew `mean_M=35` — just past a cliff — so its **denominator was
~2x inflated**. That, and only that, produced "headroom 0.63x = the live kernel
is 1.6x FASTER than an idealized dense GEMM". A kernel cannot beat an ideal GEMM
that reads the same bytes; that number should have been treated as a red flag at
the time. Comparing against any single M is invalid on a staircase.

**Corrected measurement.** Same kernel, same rows, same used-expert count, only
raggedness varying — real ragged histogram (6 independent draws) vs uniform swept
across a **full staircase period** (m=24..55, every alignment), in us/row:

| | us/row |
|---|---|
| live (ragged, real histogram), median of 6 draws | **1.383** (spread 1.371-1.413, ±1.5%) |
| balanced (uniform), median over full alignment period | **1.255** |
| **R = live / balanced** | **1.10x** |

Pre-registered gates were R≤1.15 holds / R≥1.30 reopen. **R=1.10x → holds.**
The live kernel pays no meaningful penalty for raggedness beyond the tile
quantization a balanced load pays too.

**The luckiest notch is not a lever.** Balanced at m=32/48 hits 0.94 us/row, a
~1.47x gap — but that is unreachable, and the reason is mechanistic, not
practical: the notch is a property of **uniformity, not of the mean**. Forcing
the real ragged histogram to `mean=32` leaves it at **1.415 us/row** (vs uniform
1.415/0.959 = **1.47x**), i.e. identical to the untouched real distribution.
Only **5.3%** of rows sit within ±2 of a 32-multiple; the histogram spans 1..413.
Reaching the notch requires making every expert's row count equal — a **router**
change (load balancing, which changes model outputs), not a kernel change. This
*strengthens* the conclusion below: the constraint is the routing distribution.

**MLX version caveat — resolved, no re-run needed.** Original: `ac73d0c9`;
campaign now `e40a416b2` (183 commits). The live path is **byte-identical**:
`affine_gather_qmv_rhs`, `affine_gather_qmv_rhs_chunk`, `affine_gather_qmm_rhs`
all SHA-match; `steel_gemm_gather.h` SHA-matches; the `gather_qmv_rhs` dispatch
block and the `MAXBE` bound are unchanged; `GatherQMM::eval_gpu` is untouched.
The delta adds a `qmv_wide` kernel, but it is reached only via `dispatch_qmv`,
which is called **only** from `QuantizedMatmul::eval_gpu` / `QQMatmul::eval_gpu`
— never from `GatherQMM`. (It would have perturbed the old `ceiling` tier, which
is exactly the tier now retracted.) `qmm_splitk`'s new `k_align` is a no-op here
(g=32: `K/32 == K/max(32,32)`).

**32-core vs 40-core caveat: still open, and NOT resolvable locally.** All of the
above is a laptop 32-core measurement, as was the original. This does not affect
the verdict — R=1.10x is a ratio between two runs of the same kernel on the same
device, and the artifact being corrected is arithmetic, not hardware-dependent.
But the *absolute* us/row and the exact tile-cliff positions are core-count
dependent (cf. PERFORMANCE_HISTORY's own 11.66→14.34 TFLOPS 32-vs-40-core
correction). Confirming the staircase geometry on a 40-core Studio requires
cluster access and is a **separate follow-up needing explicit relaunch approval**
— deliberately not taken here. It is not blocking: no decision changes on it.

Artifacts: `tmp/lever1-recheck-20260831/` (`ceiling_m_sweep`, `balanced_m_cliff`,
`alignment_fair`, `notch_reachability`, `.json` each). Everything below this line
is the original 2026-08-20 text, unmodified except this addendum.

## Caveats

- Laptop 32-core GPU vs cluster 40-core. Ratios transfer; absolute ms do not.
- Single-rank, in-process: no TP collectives, no `moe.all_sum`. Deliberate —
  this isolates kernel efficiency, which is the question asked.
- Synthetic weights and synthetic (but histogram-calibrated) routing. The
  run-length distribution is what drives the result, and it was matched to the
  documented production measurement.
- Timings are min-of-25 interleaved. Mean-based timing on this laptop was
  unusable — a concurrent load average of ~6 inflated means enough to flip
  tier orderings between runs. First two attempts at this bench produced
  non-reproducible numbers for exactly that reason before the fix.

# gather_qmm M=1 dispatch: confirmed already gemv-specialized — 2026-08-21 (session 2, part 11)

## Why this check

Per Fable review #1, flagged as a zero-risk, code-reading-only check:
verify MLX's `gather_qmm` (the MoE expert-matmul primitive underlying
`moe.switch_mlp`, ~30-45% of prefill/decode wall time per tonight's
kernel breakdowns) actually takes a gemv-specialized code path for
decode's M=1 shape, rather than a general gemm kernel wasting tiles on
a degenerate single-row case.

## Method

Read-only investigation of `mlx/backend/metal/quantized.cpp`'s
`GatherQMM::eval_gpu` dispatch logic (the vendored MLX fork at
`~/repos/exo/mlx`, not a live-hardware test — zero cluster risk).

## Findings

`GatherQMM::eval_gpu` has a multi-tier dispatch ladder based on `M`
(rows per matmul) and `B` (effective batch — total rows across all
expert-index pairs):

1. **`M==1 && B>=16 && sorted && ...`** → `gather_qmv_rhs` (a fast
   streaming-qmv kernel, added 2026-07-02, "M-batched qmv over sorted
   expert runs" — streams expert weights once per same-expert run).
   Gated `MLX_GATHER_QMV_RHS`.
2. **`M==1 && B>=16 && sorted && B/E>=4`** → `gather_qmm_rhs` (a
   different, non-broadcast M=1 kernel).
3. **`sorted && M>=16`** → `gather_qmm_rhs_lhs` (prefill-focused,
   run-length-encoded expert weight reads, added OPT-9 2026-06-24 to
   avoid an earlier "OPT-8" broadcast-allocation regression that was
   tested and reverted — see inline comment, real prior engineering
   history on this exact code path).
4. **`M >= vector_limit`** → `gather_qmm` (the general tiled steel-gemm
   kernel).
5. **Fallback (`transpose_=True`, small M)** → `gather_qmv` — **a
   dedicated gemv Metal kernel** (`_gather_qmv_fast_` variant when
   `N % 8 == 0 && K % 512 == 0`, else `_gather_qmv_`), with grid
   dimensions `(M, ceil(N/bn), B)` — genuinely gemv-shaped dispatch, not
   a degenerate 1-row case of a tiled gemm.

`vector_limit` (`get_qmv_batch_limit`) is a hardware-generation-aware
threshold (10-32 depending on Apple Silicon generation and
head/output-dim size) — well above 1, so **decode's M=1 case reliably
falls through tiers 1-3 (which require B>=16, a condition c=1 decode's
single-token dispatch doesn't meet — B is bounded by top-6-of-256
expert routing width per token, not tokens-in-flight) and lands on the
tier-5 `gather_qmv` gemv kernel**, not the general `gather_qmm` tiled
matmul.

## Conclusion

**No bug found — MLX's own dispatch already correctly routes decode's
M=1 shape to a dedicated gemv kernel.** This closes out the specific
check Fable flagged as worth verifying: it is NOT the case that DSv4
decode is wasting compute on a gemm kernel poorly suited to gemv-shaped
work. The code comments throughout this dispatch ladder (`OPT-8`
reverted for allocator stalls, `OPT-9`'s run-length-encoding fix, the
`B/E` tuning bounds with measured speedup ratios) show this exact
question has already received real prior engineering attention —
consistent with a mature, previously-optimized code path rather than an
overlooked gap.

This is a genuine negative result from a zero-risk, read-only
investigation — it does not identify a new lever, but it does close out
one of the open questions from tonight's Fable review cycle with
confidence, without requiring any further live-hardware risk.

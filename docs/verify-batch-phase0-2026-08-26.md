# Phase 0 — DSpark verify-path batching: batch-invariance micro-tests + design decision

**Date:** 2026-08-26
**Author:** PM agent (GLM-5.2 via Ollama Cloud), verify-path batching campaign
**Status:** Phase 0 COMPLETE. Design DECIDED: **indexer-stream-sharing** (per-row everything else; no custom Metal kernel).
**Commit:** (this doc + `bench/micro_batch_invariance.py`)

## TL;DR (plain language first)

The plan was to speed up DSpark speculative decoding by computing the verify
forward for all 4 draft rows in one batched pass instead of one row at a
time — the same fix that makes a 4-row matrix multiply cheaper than four
1-row multiplies. A pre-implementation test ("Phase 0") just proved this
**will not work** for the expensive parts: Apple's MLX/Metal computes a
4-row matrix multiply with a *different rounding order* than four separate
1-row multiplies, so the results differ by a few ULP. For speculative
decoding that is fatal — a 1-ULP difference can flip which token the model
picks at a near-tie, which is exactly the bug the current "one row at a
time" fix was shipped to prevent. The only batchable win that survives is
sharing the *single read* of the compressed KV cache across rows (while
still computing each row's scores separately), which targets the Indexer
search — the one piece that dominates per-row cost and scales with
context. We commit to that narrower design and measure it at the gates.

## Background

The prior DSpark/MTP enablement campaign (2026-08-25/26) closed with a
**REVERT** verdict: spec-ON measured +1.87% median (24-run protocol, CI
[-0.82, +9.45]%), short of the +10% PROMOTE bar. The cause was identified
as `C_s = 3.20` (verify costs 99ms/cycle = 20.2ms/row × 4.31 rows vs
34.5ms serial step) — speculation is at the break-even knife-edge, not
broken. The fix direction, agreed by three frontier-model consults
(Kimi/GLM/DSv4-Pro), is **verify-path batching**: compute the Indexer
top-k once per cycle and batch the verify forward, targeting `C_s ≈ 1.3`
→ predicted +55-80% (wall-clock model: ~+115%).

The current production path is **row-sequential verify**
(`EXO_DSV4_VERIFY_ROWSEQ=1` + `EXO_DSV4_ROWSEQ_FULLBLOCK=1`, shipped as
correctness fix `b9921962e`): the ENTIRE verify block (attn_hc, attn_norm,
attention, ffn_hc, ffn_norm) runs per-row, with only the MoE ffn batched
(quantized matmuls are bitwise batch-invariant M=1..8). This makes verify
bitwise-equivalent to L sequential single-token decodes, eliminating the
~1-ULP drift that flipped near-tie argmaxes across TP ranks and caused
the cross-rank wedge. Cost: verify scales with γ AND context (the Indexer
top-k over compressed KV — the same mechanism as the 2026-08-04 FULLBLOCK
context cliff).

## Phase 0 deliverables

1. **Batch-invariance micro-tests on real shapes** (`bench/micro_batch_invariance.py`)
   — GEMV M=1 vs GEMM M=4 row-wise 0-ulp for the big projections, fused
   fp32 SDPA L_q=1 vs L_q=4, the Indexer score GEMM (the one op the
   stream-share design would batch), and the Indexer cross-row top-k
   overlap. The 0-ulp table DECIDES the design.
2. **Per-row timing hooks** — design notes for `mx.eval()` fences at the
   op boundaries (Indexer → gather → attention core → proj), since Metal
   is async and unfenced timings measure dispatch not execution.
3. **This doc** — committed.

## The batch-invariance table (DECIDES the design)

Random weights at the **real DSv4-Flash shapes** (the 0-ulp property is a
kernel-dispatch property that depends on shapes/dtypes, not weight
values). M=4 = γ+1 (gamma=3, block_size=5 → verify M=4 in practice).
dtype=bf16 (production), quantized=8bit/group=64 (production).

| OP | MAX_ABS | MAX_ULP | 0-ULP |
|---|---:|---:|:---:|
| wq_a (QKV proj, dense bf16, K=4096) | 0.000e+00 | 0 | **PASS** |
| wq_a (QKV proj, quantized 8bit) | 0.000e+00 | 0 | **PASS** |
| wq_b (main q proj, dense bf16, 1024→32768) | 4.883e-04 | 131072 | **FAIL** |
| wo_b (O proj, dense bf16) | 4.883e-04 | 65536 | **FAIL** |
| MoE expert up (quantized 8bit) | 0.000e+00 | 0 | **PASS** |
| MoE expert down (quantized 8bit) | 0.000e+00 | 0 | **PASS** |
| lm_head (dense bf16, 4096→129280) | 7.812e-03 | 1900544 | **FAIL** |
| fused_fp32_sdpa (L_q=4 vs 4×L_q=1) | 1.526e-05 | 65536 | **FAIL** |
| indexer_score_gemm (the batched op) | 7.812e-03 | 65536 | **FAIL** |

### Indexer cross-row overlap (query-dependence)

| metric | value |
|---|---|
| query_dependent | **True** |
| all_rows_identical | False |
| mean_pairwise_jaccard | 0.0390 |
| n_rows_diff_vs_row0 | 3/3 |
| conclusion | **QUERY-DEPENDENT** → share the STREAM, not the SET |

**Code confirmation** (`deepseek_v4.py:3782-3881`, `Indexer.__call__`):
scores use `q = self.wq_b(q_residual)` (line 3818) AND
`self.weights_proj(x)` (line 3827), with a row-causal pmask (line 3804)
that differs per row. The Indexer is query-dependent by construction, not
just empirically.

## Critical findings

1. **The #1 implementation risk (Kimi) is CONFIRMED.** MLX/Metal
   dispatches M=4 dense bf16 GEMM with a different K-reduction order than
   M=1 GEMV — **not 0-ulp**. The big projections (wq_b 131072 ULP, wo_b
   65536 ULP, lm_head ~1.9M ULP) all fail. Only the **quantized** MoE ops
   and the small-K dense ops (wq_a, K=4096 fits one tile) pass 0-ulp.
   This matches the existing shipped code comment ("quantized matmuls are
   batch-invariant M=1..8") and explains why the current FULLBLOCK path
   batches only the MoE ffn.

2. **The fused fp32 SDPA also FAILs** (L_q=4 batched ≠ 4×L_q=1 per-row,
   65536 ULP). This is the **hard ceiling** on the design: the attention
   core CANNOT batch even with a custom rowseq-GEMM kernel for the dense
   projections, because the SDPA is a *fused* Metal kernel whose L_q>1
   path uses a different accumulation order than L_q=1. The full Kimi
   design ("batched M=4 forward + per-row attention on the decode-
   identical kernel") is **INFEASIBLE**: the attention must stay per-row
   L_q=1 (which is what rowseq already does).

3. **The Indexer score GEMM also FAILs** (65536 ULP). This refines the
   design: even "share the stream" cannot use a single batched L=4 score
   GEMM — it needs **per-row accumulators** (loop the L rows through the
   score GEMM one at a time, streaming `pooled` once). The win is sharing
   the pooled READ (the memory-bound compressed-KV stream), not sharing
   the GEMM dispatch.

4. **The Indexer is query-dependent** (Jaccard 0.039, confirmed in code).
   "Compute top-k once, share across rows" is NOT valid. Kimi's principle
   holds: share the STREAM, not the SET.

## DECIDED design: INDEXER-STREAM-SHARING (no custom kernel)

`EXO_DSV4_VERIFY_BATCH=1` (default OFF, rowseq path untouched for A/B):

At each of the 21 sparse layers, in the Indexer call path:
- **Stream the compressed KV (`pooled`) read ONCE** per cycle (the
  memory-bound part — the 1.2GB compressed-KV stream across 21 layers per
  the consults' bandwidth estimate).
- **Loop the L verify rows through the score GEMM one at a time** (per-row
  accumulators), because the batched L=4 score GEMM is not 0-ulp. Each
  row's scores are bitwise-identical to its rowseq per-row computation
  (same `pooled` tile order, same fp32 accumulation in the bf16 GEMM).
- **Per-row segmented top-k** (deterministic, tie-break by index) — sets
  differ per row (query-dependent).
- **Per-row gather, per-row attention core** (L_q=1 fused fp32,
  decode-identical — unchanged from rowseq).

Everything else (dense proj wq_b/wo_b, MoE, lm-head, block-level hc ops)
stays on the current rowseq/FULLBLOCK path. This is a **Python-level**
change to the Indexer call path, gated behind `EXO_DSV4_VERIFY_BATCH`,
with **no custom Metal kernel** required.

### Why this design (not the full Kimi design or custom kernel)

- **Full Kimi design (C_s~1.3) is infeasible**: the SDPA fail means the
  attention core cannot batch, and the dense proj fail means those need a
  custom rowseq-GEMM kernel too. Even with all custom kernels, the
  attention stays per-row, so the "batched M=4 forward" only helps the
  non-attention ops (MoE already batched; dense proj ~35% of per-row
  cost). Realistic ceiling without the attention: ~C_s 1.7-2.0, not 1.3.
- **Custom rowseq-GEMM Metal kernel**: significant engineering (match
  MLX's exact M=1 GEMV tile/reduction order bitwise for wq_b/wo_b/lm_head)
  for ops that are NOT the dominant per-row cost. The Indexer (the
  dominant cost, 50-85% of 20.2ms/row, and the context-scaling cliff
  source) is addressable WITHOUT a custom kernel — just stream the pooled
  read once and loop the score GEMM per row.
- **Indexer-stream-sharing (this design)**: the Indexer dominates per-row
  cost and scales with context. Streaming the compressed-KV read once
  (instead of γ+1 times) targets the dominant cost with a Python-level
  change. Projected C_s depends on the real Indexer share — the consults
  bracket it 12-85%; we measure at G2/G3.

### Per-row timing hooks (design notes)

Metal is async; unfenced timings measure dispatch, not execution. The
hook points (with `mx.synchronize()` + `perf_counter` fences before/after
each), gated behind `EXO_DSV4_VERIFY_PROFILE=1`:
1. `attn.compressor` → build pooled
2. `attn.proj_qkv` → wq_a/wq_b/wkv/q_norm
3. `attn.indexer` → score GEMM + top-k (the op this design targets)
4. `attn.gather` → pooled gather
5. `attn.sdpa` → sparse attention core (L_q=1 fused fp32)
6. `attn.o_proj` → wo_a/wo_b
7. `layer.ffn` → MoE (already batched)

The existing `_SECTION_TIME_ENABLED` / `_ATTN_SUB_ACC` span machinery in
`deepseek_v4.py` already provides these (see `_BUILD_PROBE_PERF` /
`_ATTN_SUB_ACC` at the attention `__call__`). The verify-batching
implementation reuses these spans; the per-row split is visible by
comparing the `attn.indexer` span time ON (streamed once) vs OFF (per-row
γ+1 times).

## Next steps

- **Step 1**: Implement `EXO_DSV4_VERIFY_BATCH=1` (indexer-stream-sharing,
  default OFF, rowseq untouched). Pre-commit gates (basedpyright, ruff,
  pytest scoped). Commit + push.
- **Step 2**: Gates G0-G3 (cluster): G0 cycle-level bitwise (500 @8K +
  200 @100K, 0-ulp on verify logits AND written KV); G1 Gate A (strict
  argmax temp=0); G2 Tier-1 (7-prompt byte-identity); G3 10K soak (zero
  uid divergence, rollback lengths match).
- **Step 3**: 4-run sanity A/B (ON/OFF/ON/OFF, 256-tok @100K): PASS iff
  C_s ≤ 1.6 and a ∈ [2.15, 2.36].
- **Step 4**: Full 24-run verdict with the wall-clock-derived bar.
- **Step 5**: Docs + commits after every milestone.

## Reproducing

```bash
cd ~/repos/exo
.venv/bin/python bench/micro_batch_invariance.py --json /tmp/phase0.json
```

No cluster needed — runs locally with random weights at real shapes.
Output: the 0-ulp table + the DECIDED design.
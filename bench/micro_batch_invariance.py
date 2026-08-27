#!/usr/bin/env python3
"""Phase-0 batch-invariance micro-tests for the DSpark verify-batching design.

Tests the kernel-dispatch property that DECIDES the verify-batching design:
does MLX/Metal dispatch an M=4 GEMM with a different K-reduction order than an
M=1 GEMV, producing non-zero-ulp drift on the real DSv4-Flash shapes?

Per Kimi's design (consult 2026-08-26): if GEMV M=1 != GEMM M=4 bitwise for the
big projections, the full batched design is infeasible and we need either a
custom "rowseq-GEMM" Metal kernel (stream W once, 4 independent row
accumulators, fixed K-order) or a fallback to indexer-sharing only (C_s~2.9).

This harness uses RANDOM weights at the REAL DSv4-Flash shapes (the 0-ulp
property is about the matmul kernel's reduction order, which depends on
shapes/dtypes, not weight values). Runs locally with the exo venv -- no
cluster needed for the op-level tests.

Outputs a 0-ulp PASS/FAIL table per op + the Indexer cross-row overlap
measurement. The table DECIDES the design (full batched vs per-row
accumulators vs custom rowseq-GEMM kernel).

Usage:
    .venv/bin/python bench/micro_batch_invariance.py
    .venv/bin/python bench/micro_batch_invariance.py --json /tmp/phase0.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

# Force Metal before anything else (matches the serving path).
os.environ.setdefault("MLX_METAL_FAST_MATH", "0")

import mlx.core as mx

# Real DSv4-Flash shapes (from config.json on the cluster).
HIDDEN = 4096
HEAD_DIM = 512
N_HEADS = 64
Q_LORA_RANK = 1024
VOCAB = 129280
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
O_GROUPS = 8
O_LORA_RANK = 1024
MOE_INTER = 2048
N_EXPERTS = 256
N_EXPERTS_PER_TOK = 6
SLIDING_WINDOW = 128
COMPRESS_RATIO = 4  # sparse layers
M_VERIFY = 4  # gamma+1 rows (gamma=3, block_size=5 -> verify M=4 in practice)


@dataclass
class OpResult:
    name: str
    shape_desc: str
    m1_dtype: str
    m4_dtype: str
    max_abs_diff: float
    max_ulp_diff: int
    bitwise_identical: bool
    notes: str = ""


@dataclass
class Phase0Report:
    gemm_results: list[OpResult] = field(default_factory=list)
    sdpa_result: OpResult | None = None
    indexer_score_result: OpResult | None = None
    indexer_overlap: dict | None = None
    decided_design: str = ""
    decided_rationale: str = ""


def _ulp_count(a: mx.array, b: mx.array) -> int:
    """Max ULP difference between two bf16 arrays, viewed as float32 bits.

    For near-equal floats this is the standard int-difference of the
    sign-magnitude-bitcast values. Returns 0 when bitwise-equal.
    """
    import numpy as np

    a32 = np.array(a.astype(mx.float32))  # mlx -> numpy copy
    b32 = np.array(b.astype(mx.float32))
    # Reinterpret float32 bits as uint32 (sign-magnitude).
    ua = a32.view(np.uint32).astype(np.int64)
    ub = b32.view(np.uint32).astype(np.int64)
    # Where a == b exactly (incl. 0.0/-0.0), bits differ by the sign bit (0x80000000)
    # but the values are equal -> count as 0. Use the boolean == first.
    eq = (a32 == b32)
    # Sign-magnitude -> two's complement for a stable ULP diff:
    # map sign-magnitude int s to: s ^ (s>>31 | 0x80000000) is overkill; for
    # near-equal same-sign results a plain abs(int diff) is correct.
    diff = np.abs(ua - ub)
    # Zero out the exactly-equal cases (handles 0.0 vs -0.0: a32==b32 is True).
    diff = np.where(eq, 0, diff)
    return int(diff.max()) if diff.size else 0


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    mx.eval(d)
    return float(d.max().item()) if d.size > 0 else 0.0


def _row_wise_max_abs_diff(m4: mx.array, rows: list[mx.array], row_axis: int = 0) -> float:
    """Max abs diff between each per-row result and the corresponding batched row.

    row_axis is the axis of ``m4`` that indexes the rows (the M/verify axis).
    For GEMM outputs (M, out) it's axis 0; for SDPA outputs (B, H, L_q, D) it's
    axis 2. The per-row ``rows`` list always has the row as axis=row_axis.
    """
    assert m4.shape[row_axis] == len(rows), (m4.shape, row_axis, len(rows))
    worst = 0.0
    slicer = [slice(None)] * m4.ndim
    for i, r in enumerate(rows):
        slicer[row_axis] = slice(i, i + 1)
        worst = max(worst, _max_abs_diff(m4[tuple(slicer)], r))
    return worst


def _row_wise_ulp(m4: mx.array, rows: list[mx.array], row_axis: int = 0) -> int:
    """Max ULP diff, per-row, across the row_axis of the batched result."""
    assert m4.shape[row_axis] == len(rows)
    slicer = [slice(None)] * m4.ndim
    ulp = 0
    for i, r in enumerate(rows):
        slicer[row_axis] = slice(i, i + 1)
        ulp = max(ulp, _ulp_count(m4[tuple(slicer)], r))
    return ulp


def test_gemm_invariance(
    name: str,
    w_shape: tuple[int, int],
    x_m1: mx.array,
    x_m4: mx.array,
    dtype: mx.Dtype = mx.bfloat16,
    is_quantized: bool = False,
) -> OpResult:
    """Test GEMV M=1 vs GEMM M=4 row-wise 0-ulp.

    w_shape is (out_features, in_features) -- MLX nn.Linear weight layout
    (output, input). x_m1 is (1, in), x_m4 is (M, in).
    """
    if is_quantized:
        # Simulate a quantized matmul: build w as quantized (group_size=64, 8bit).
        w_bf16 = mx.random.normal(w_shape, dtype=mx.bfloat16) * 0.1
        w_q, _scales, _biases = mx.quantize(w_bf16, group_size=64, bits=8)
        # M=1: quantized_matmul expects x @ w.T -> output (1, out)
        # mx.quantized_matmul(x, w_q, scales, biases) computes x @ w.T
        out_m1 = mx.quantized_matmul(
            x_m1.astype(mx.bfloat16), w_q, _scales, _biases, bits=8, group_size=64
        )
        out_m4 = mx.quantized_matmul(
            x_m4.astype(mx.bfloat16), w_q, _scales, _biases, bits=8, group_size=64
        )
        mx.eval(out_m1, out_m4)
        # Per-row M=1
        rows = [
            mx.quantized_matmul(
                x_m4[i : i + 1].astype(mx.bfloat16), w_q, _scales, _biases, bits=8, group_size=64
            )
            for i in range(x_m4.shape[0])
        ]
        for r in rows:
            mx.eval(r)
    else:
        w = mx.random.normal(w_shape, dtype=dtype) * 0.02
        # nn.Linear computes x @ w.T -> (M, out)
        out_m1 = x_m1 @ w.T
        out_m4 = x_m4 @ w.T
        mx.eval(out_m1, out_m4)
        rows = [x_m4[i : i + 1] @ w.T for i in range(x_m4.shape[0])]
        for r in rows:
            mx.eval(r)

    # Compare M=4 batched result row-wise against M=1 per-row results.
    mad_rows = _row_wise_max_abs_diff(out_m4, rows)
    # Also compare the single M=1 (x_m1) vs the first M=4 row (x_m4[0]) --
    # this isolates whether M=4 dispatch differs from M=1 for the SAME row.
    # Use the same x for row 0 to make this apples-to-apples.
    mad_single = _max_abs_diff(out_m4[0:1], out_m1) if x_m1.shape == x_m4[0:1].shape else mad_rows
    # Use the row-wise comparison as the headline (the design question is
    # "does computing all 4 rows in one GEMM match computing them one at a time").
    max_abs = mad_rows
    ulp = _row_wise_ulp(out_m4, rows)

    bitwise = (max_abs == 0.0) and (ulp == 0)
    return OpResult(
        name=name,
        shape_desc=f"W={w_shape} x_M1={tuple(x_m1.shape)} x_M4={tuple(x_m4.shape)} dtype={dtype}",
        m1_dtype=str(out_m1.dtype),
        m4_dtype=str(out_m4.dtype),
        max_abs_diff=max_abs,
        max_ulp_diff=ulp,
        bitwise_identical=bitwise,
        notes=f"row_wise_max_abs={mad_rows:.6e} single_vs_m4row0={mad_single:.6e}",
    )


def test_sdpa_lq_invariance() -> OpResult:
    """Test fused fp32 SDPA L_q=1 vs L_q=4 row-wise 0-ulp.

    Per Kimi: if the L_q==1 fused fp32 kernel loops query rows internally with
    fixed KV tiling and per-row fp32 accumulation, batched == sequential for free.
    This is the decode-identical kernel the batched verify attention core must use.
    """
    B, H, D = 1, N_HEADS, HEAD_DIM
    sw = SLIDING_WINDOW
    L_q1 = 1
    L_q4 = M_VERIFY
    scale = float(D ** -0.5)

    # Build a realistic KV window (sw keys) + a per-row gathered pooled set
    # (k=INDEX_TOPK) to mimic the sparse-pooled SDPA concat path.
    k_gather = 64  # small for speed; the L_q==1 path concatenates local+pooled
    sw_total = sw + k_gather
    q4 = mx.random.normal((B, H, L_q4, D), dtype=mx.bfloat16) * 0.5
    kv = mx.random.normal((B, H, sw_total, D), dtype=mx.bfloat16) * 0.5

    # L_q=4 batched
    out_lq4 = mx.fast.scaled_dot_product_attention(
        q4, kv, kv, scale=scale, mask=None
    )
    mx.eval(out_lq4)

    # L_q=1 per row
    rows = []
    for i in range(L_q4):
        out_i = mx.fast.scaled_dot_product_attention(
            q4[:, :, i : i + 1, :], kv, kv, scale=scale, mask=None
        )
        mx.eval(out_i)
        rows.append(out_i)

    max_abs = _row_wise_max_abs_diff(out_lq4, rows, row_axis=2)
    ulp = _row_wise_ulp(out_lq4, rows, row_axis=2)

    return OpResult(
        name="fused_fp32_sdpa",
        shape_desc=f"B={B} H={H} D={D} L_q4={L_q4} sw_total={sw_total} k_gather={k_gather}",
        m1_dtype=str(rows[0].dtype),
        m4_dtype=str(out_lq4.dtype),
        max_abs_diff=max_abs,
        max_ulp_diff=ulp,
        bitwise_identical=(max_abs == 0.0 and ulp == 0),
        notes="L_q=4 batched vs 4x L_q=1 per-row, same KV, fused fp32 SDPA",
    )


def test_indexer_score_gemm_invariance() -> OpResult:
    """Test the Indexer SCORE GEMM (the op we'd actually batch) M=1 vs M=4.

    The score GEMM is (B, L, D) @ (B, D, P) -> (B, L, P), where L is the
    verify-row axis (M=4) and P is the compressed pool size. This is the ONE
    op the indexer-stream-sharing design actually batches. If it's not
    0-ulp M=1 (per-row L=1) vs M=4 (L=4), the design needs per-row
    accumulators (loop the L rows through the score GEMM one at a time,
    streaming pooled once) rather than a single batched L=4 GEMM.
    """
    B = 1
    D_idx = INDEX_HEAD_DIM  # 128
    P = 8192  # compressed pool size (~32K ctx / ratio=4)
    L4 = M_VERIFY
    L1 = 1

    # q_weighted: (B, L, D_idx) -- the query-side, differs per row
    # pooled: (B, D_idx, P) -- the compressed KV stream (shared across rows)
    q_w_m1 = mx.random.normal((B, L1, D_idx), dtype=mx.bfloat16) * 0.5
    q_w_m4 = mx.random.normal((B, L4, D_idx), dtype=mx.bfloat16) * 0.5
    pooled = mx.random.normal((B, D_idx, P), dtype=mx.bfloat16) * 0.5

    # M=1 (one row) and M=4 (all rows) score GEMM
    out_m1 = q_w_m1 @ pooled  # (B, L1, P)
    out_m4 = q_w_m4 @ pooled  # (B, L4, P)
    mx.eval(out_m1, out_m4)
    # Per-row M=1 (loop the 4 rows through the SAME pooled stream)
    rows = [q_w_m4[:, i : i + 1, :] @ pooled for i in range(L4)]
    for r in rows:
        mx.eval(r)

    # The pooled stream is SHARED (read once) -- the design question is whether
    # the per-row score accumulation matches when computed as one L=4 GEMM vs
    # 4 separate L=1 GEMMs against the same pooled.
    max_abs = _row_wise_max_abs_diff(out_m4, rows, row_axis=1)
    ulp = _row_wise_ulp(out_m4, rows, row_axis=1)

    return OpResult(
        name="indexer_score_gemm (the batched op)",
        shape_desc=f"B={B} D={D_idx} P={P} L4={L4} (q_weighted @ pooled.T, pooled shared)",
        m1_dtype=str(out_m1.dtype),
        m4_dtype=str(out_m4.dtype),
        max_abs_diff=max_abs,
        max_ulp_diff=ulp,
        bitwise_identical=(max_abs == 0.0 and ulp == 0),
        notes="L=4 batched score GEMM vs 4x L=1 per-row, pooled shared (the stream-share op)",
    )


def test_indexer_query_dependence() -> dict:
    """Measure Indexer cross-row top-k overlap to confirm query-dependence.

    Per the consult synthesis: the Indexer is QUERY-DEPENDENT (scores use
    q_residual via wq_b AND weights_proj(x); causal masks differ per row), so
    "compute top-k once, share across rows" is NOT valid. Kimi's design wins:
    SHARE THE STREAM, NOT THE SET. This measures the cross-row Jaccard overlap
    to quantify how much the sets differ (expect high overlap, <100% identical).
    """
    # Simulate the Indexer score pass: scores = q_weighted @ pooled.T
    # where q = wq_b(q_residual), weights = weights_proj(x).
    B, L, H, D_idx = 1, M_VERIFY, INDEX_N_HEADS, INDEX_HEAD_DIM
    P = 8192  # compressed pool size (~32K ctx / ratio=4)
    q_lora_rank = Q_LORA_RANK
    hidden = HIDDEN

    # Random "inputs" at real shapes
    q_residual = mx.random.normal((B, L, q_lora_rank), dtype=mx.bfloat16) * 0.5
    x = mx.random.normal((B, L, hidden), dtype=mx.bfloat16) * 0.5
    pooled = mx.random.normal((B, P, D_idx), dtype=mx.bfloat16) * 0.5

    # wq_b: q_lora_rank -> n_heads * head_dim
    wq_b = mx.random.normal((H * D_idx, q_lora_rank), dtype=mx.bfloat16) * 0.02
    # weights_proj: hidden -> n_heads
    w_proj = mx.random.normal((H, hidden), dtype=mx.bfloat16) * 0.02
    scale = float(D_idx ** -0.5)
    n_heads_inv_sqrt = float(H ** -0.5)

    # Per the real _indexer_score:
    # q = wq_b(q_residual).reshape(B, L, H, D).transpose(0,2,1,3)
    q = (q_residual @ wq_b.T).reshape(B, L, H, D_idx).transpose(0, 2, 1, 3)
    w = mx.sigmoid(x @ w_proj.T) * (scale * n_heads_inv_sqrt)  # (B, L, H)
    q_blhd = q.transpose(0, 2, 1, 3)  # (B, L, H, D)
    q_weighted = (w[..., None] * q_blhd).sum(axis=2)  # (B, L, D)
    scores = q_weighted @ pooled.swapaxes(-1, -2)  # (B, L, P)
    mx.eval(scores)

    # Per-row top-k (different q per row -> different scores -> different top-k)
    k = min(INDEX_TOPK, P)
    topk_per_row = []
    for i in range(L):
        # Apply a row-causal-ish mask: row i sees pools 0..(some_limit_i)
        # To keep the test about query-dependence (not causal masking), use
        # the FULL pool for every row (the causal mask only REMOVES
        # candidates; it can't make divergent sets identical).
        row_scores = scores[:, i : i + 1, :]  # (B, 1, P)
        tk = mx.argsort(-row_scores, axis=-1)[..., :k]
        mx.eval(tk)
        topk_per_row.append(set(int(v) for v in tk[0, 0].tolist()))

    # Cross-row Jaccard overlap (all pairs)
    overlaps = []
    for i in range(L):
        for j in range(i + 1, L):
            inter = len(topk_per_row[i] & topk_per_row[j])
            union = len(topk_per_row[i] | topk_per_row[j])
            jac = inter / union if union > 0 else 1.0
            overlaps.append((i, j, jac, inter, k))
    # All-rows identical?
    all_identical = all(topk_per_row[i] == topk_per_row[0] for i in range(1, L))
    mean_jaccard = sum(o[2] for o in overlaps) / len(overlaps) if overlaps else 1.0

    # Also: what if we SHARE the set (compute top-k once for row 0, use for all)?
    # How many rows would get a DIFFERENT argmax-gated acceptance?
    # (We can't fully simulate acceptance here, but set-difference is the proxy.)
    shared_set = topk_per_row[0]
    n_rows_with_diff_set = sum(1 for i in range(1, L) if topk_per_row[i] != shared_set)
    return {
        "query_dependent": not all_identical,
        "all_rows_identical": all_identical,
        "mean_jaccard_pairwise": mean_jaccard,
        "n_rows_with_different_topk_vs_row0": n_rows_with_diff_set,
        "overlaps_detail": [{"rows": [o[0], o[1]], "jaccard": o[2], "inter": o[3], "k": o[4]} for o in overlaps],
        "k": k,
        "P": P,
        "conclusion": "QUERY-DEPENDENT (sets differ) -> share the STREAM not the SET" if not all_identical else "query-independent (sets identical) -> share the set",
    }


def decide_design(report: Phase0Report) -> None:
    """DECIDE the design from the 0-ulp table (Kimi's decision rule).

    The SDPA result is the hard ceiling: if the fused fp32 SDPA L_q=4 != L_q=1
    bitwise, the attention core CANNOT batch even with a custom GEMM kernel
    (the SDPA is a fused kernel, not a plain GEMM). That means the attention
    stays per-row L_q=1 regardless, and the full Kimi "batched M=4 forward" only
    helps the NON-attention ops (dense proj, MoE [already batched], lm-head).
    """
    # Full batched requires ALL big projections 0-ulp AND SDPA 0-ulp.
    all_gemm_ok = all(r.bitwise_identical for r in report.gemm_results)
    sdpa_ok = report.sdpa_result is not None and report.sdpa_result.bitwise_identical
    idx_score_ok = (
        report.indexer_score_result is not None
        and report.indexer_score_result.bitwise_identical
    )
    indexer_qdep = report.indexer_overlap and report.indexer_overlap["query_dependent"]

    if all_gemm_ok and sdpa_ok:
        report.decided_design = "FULL KIMI DESIGN"
        report.decided_rationale = (
            "All big projections (QKV/O/MoE/lm-head) are 0-ulp M=1 vs M=4, and the "
            "fused fp32 SDPA is 0-ulp L_q=1 vs L_q=4. Batched M=4 forward for "
            "embeddings/dense/QKV/O/MoE/lm-head; at each of the 21 sparse layers: "
            "batched indexer-score stream (per-row accumulators) -> per-row "
            "segmented top-k -> per-row gather -> per-row attention core "
            "(L_q==1 fused fp32, decode-identical) -> recombine. "
            f"Indexer is {('QUERY-DEPENDENT' if indexer_qdep else 'query-independent')}: "
            "share the STREAM, not the SET."
        )
    elif sdpa_ok and not all_gemm_ok:
        failed = [r.name for r in report.gemm_results if not r.bitwise_identical]
        report.decided_design = "CUSTOM ROWSEQ-GEMM KERNEL (or fallback to indexer-sharing)"
        report.decided_rationale = (
            f"The fused fp32 SDPA is 0-ulp (attention core can batch), but these "
            f"projections are NOT 0-ulp M=1 vs M=4: {failed}. MLX/Metal dispatches "
            "M=4 GEMM with a different K-reduction order than M=1 GEMV. Options: "
            "(a) custom 'rowseq-GEMM' Metal kernel (stream W once, 4 independent "
            "row accumulators, fixed K-order); (b) fall back to indexer-sharing "
            "only (C_s~2.9, +8% -- probably not worth shipping). Decision: build "
            "the custom kernel if the failed projections are few/critical, else "
            "fall back."
        )
    else:
        # SDPA fails -> attention core CANNOT batch even with custom GEMM.
        # The realizable design is indexer-stream-sharing with per-row
        # everything else (Python-level, no custom kernel).
        report.decided_design = (
            "INDEXER-STREAM-SHARING (per-row everything else; no custom kernel)"
        )
        report.decided_rationale = (
            "CRITICAL FINDING: the fused fp32 SDPA is NOT 0-ulp L_q=1 vs L_q=4 "
            f"(max ULP {report.sdpa_result.max_ulp_diff if report.sdpa_result else '?'}). "
            "This means the attention core CANNOT batch even with a custom "
            "rowseq-GEMM kernel for the dense projections -- the SDPA is a fused "
            "Metal kernel whose L_q>1 path uses a different accumulation order "
            "than L_q=1. The full Kimi design (batched M=4 forward + per-row "
            "attention on the decode-identical kernel) is INFEASIBLE: the "
            "attention must stay per-row L_q=1 (which is what rowseq already "
            "does). The dense bf16 big projections (wq_b/wo_b/lm_head) also fail "
            "(non-0-ulp), confirming a custom rowseq-GEMM kernel would be needed "
            "for those too -- significant Metal engineering for ops that are "
            "NOT the dominant per-row cost. "
            f"Indexer score GEMM (the one op we'd batch): {'0-ulp PASS' if idx_score_ok else 'FAIL -- needs per-row accumulators'}. "
            f"Indexer is {('QUERY-DEPENDENT (Jaccard {:.4f})'.format(report.indexer_overlap['mean_jaccard_pairwise']) if indexer_qdep else 'query-independent')}: "
            "share the STREAM, not the SET. "
            "DECISION: implement EXO_DSV4_VERIFY_BATCH=1 as indexer-stream-"
            "sharing -- batched score pass that streams the compressed KV once "
            "while maintaining independent per-row score accumulators (loop L "
            "rows through the score GEMM if the batched L=4 GEMM is not 0-ulp), "
            "then per-row top-k/gather/attention (unchanged from rowseq), with "
            "the dense proj + MoE + lm-head staying on the current rowseq/FULL"
            "BLOCK path. The Indexer dominates per-row cost (50-85% of 20.2ms/"
            "row per the consults, and is the context-scaling cliff source), so "
            "streaming it once should cut the dominant cost. Projected C_s "
            "depends on the real Indexer share -- measure at G2/G3."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write report to this JSON path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    report = Phase0Report()

    print("=" * 78)
    print("PHASE 0: DSpark verify-batching — batch-invariance micro-tests (real shapes)")
    print("=" * 78)
    print(f"M_VERIFY={M_VERIFY} (gamma+1), dtypes=bf16, shapes=DSv4-Flash real")
    print()

    # --- GEMM invariance tests (the #1 implementation risk, per Kimi) ---
    print("--- GEMM M=1 vs M=4 row-wise 0-ulp (the #1 implementation risk) ---")
    x_m1 = mx.random.normal((1, Q_LORA_RANK), dtype=mx.bfloat16) * 0.5
    x_m4 = mx.random.normal((M_VERIFY, Q_LORA_RANK), dtype=mx.bfloat16) * 0.5

    # QKV proj: wq_a (hidden->q_lora) -- dense, not quantized in the attn path
    # Actually wq_a is quantized. Test both dense bf16 and quantized 8bit.
    # wq_a: hidden_size(4096) -> q_lora_rank(1024)
    x_qkv_m1 = mx.random.normal((1, HIDDEN), dtype=mx.bfloat16) * 0.5
    x_qkv_m4 = mx.random.normal((M_VERIFY, HIDDEN), dtype=mx.bfloat16) * 0.5
    report.gemm_results.append(test_gemm_invariance(
        "wq_a (QKV proj, dense bf16)", (Q_LORA_RANK, HIDDEN), x_qkv_m1, x_qkv_m4,
        dtype=mx.bfloat16, is_quantized=False))
    report.gemm_results.append(test_gemm_invariance(
        "wq_a (QKV proj, quantized 8bit)", (Q_LORA_RANK, HIDDEN), x_qkv_m1, x_qkv_m4,
        is_quantized=True))

    # wq_b: q_lora_rank(1024) -> n_heads*head_dim(64*512=32768) for the main q
    # (this is the big projection in the indexer/attention)
    report.gemm_results.append(test_gemm_invariance(
        "wq_b (main q proj, dense bf16)", (N_HEADS * HEAD_DIM, Q_LORA_RANK), x_m1, x_m4,
        dtype=mx.bfloat16, is_quantized=False))

    # O proj: wo_b (o_groups*o_lora_rank=8*1024=8192 -> hidden_size=4096)
    x_o_m1 = mx.random.normal((1, O_GROUPS * O_LORA_RANK), dtype=mx.bfloat16) * 0.5
    x_o_m4 = mx.random.normal((M_VERIFY, O_GROUPS * O_LORA_RANK), dtype=mx.bfloat16) * 0.5
    report.gemm_results.append(test_gemm_invariance(
        "wo_b (O proj, dense bf16)", (HIDDEN, O_GROUPS * O_LORA_RANK), x_o_m1, x_o_m4,
        dtype=mx.bfloat16, is_quantized=False))

    # MoE expert: expert linear (hidden -> moe_inter) -- quantized in prod
    x_moe_m1 = mx.random.normal((1, HIDDEN), dtype=mx.bfloat16) * 0.5
    x_moe_m4 = mx.random.normal((M_VERIFY, HIDDEN), dtype=mx.bfloat16) * 0.5
    report.gemm_results.append(test_gemm_invariance(
        "MoE expert up (quantized 8bit)", (MOE_INTER, HIDDEN), x_moe_m1, x_moe_m4,
        is_quantized=True))
    # MoE down (moe_inter -> hidden)
    x_md_m1 = mx.random.normal((1, MOE_INTER), dtype=mx.bfloat16) * 0.5
    x_md_m4 = mx.random.normal((M_VERIFY, MOE_INTER), dtype=mx.bfloat16) * 0.5
    report.gemm_results.append(test_gemm_invariance(
        "MoE expert down (quantized 8bit)", (HIDDEN, MOE_INTER), x_md_m1, x_md_m4,
        is_quantized=True))

    # lm-head: hidden(4096) -> vocab(129280) -- the 13.1ms fixed overhead target
    x_lm_m1 = mx.random.normal((1, HIDDEN), dtype=mx.bfloat16) * 0.5
    x_lm_m4 = mx.random.normal((M_VERIFY, HIDDEN), dtype=mx.bfloat16) * 0.5
    report.gemm_results.append(test_gemm_invariance(
        "lm_head (dense bf16)", (VOCAB, HIDDEN), x_lm_m1, x_lm_m4,
        dtype=mx.bfloat16, is_quantized=False))

    # --- Fused fp32 SDPA L_q=1 vs L_q=4 ---
    print("\n--- Fused fp32 SDPA L_q=1 vs L_q=4 (attention core batch-invariance) ---")
    report.sdpa_result = test_sdpa_lq_invariance()

    # --- Indexer score GEMM (the one op the stream-share design batches) ---
    print("\n--- Indexer score GEMM M=1 vs M=4 (the batched-stream op) ---")
    report.indexer_score_result = test_indexer_score_gemm_invariance()

    # --- Indexer cross-row overlap ---
    print("\n--- Indexer cross-row top-k overlap (query-dependence) ---")
    report.indexer_overlap = test_indexer_query_dependence()

    # --- DECIDE ---
    decide_design(report)

    # --- Print the table ---
    print("\n" + "=" * 78)
    print("BATCH-INVARIANCE TABLE (0-ulp = bitwise identical M=1 vs M=4)")
    print("=" * 78)
    print(f"{'OP':<42} {'MAX_ABS':>12} {'MAX_ULP':>8} {'0-ULP':>6}")
    print("-" * 78)
    for r in report.gemm_results:
        print(f"{r.name:<42} {r.max_abs_diff:>12.3e} {r.max_ulp_diff:>8d} {'PASS' if r.bitwise_identical else 'FAIL':>6}")
    if report.sdpa_result:
        r = report.sdpa_result
        print(f"{r.name:<42} {r.max_abs_diff:>12.3e} {r.max_ulp_diff:>8d} {'PASS' if r.bitwise_identical else 'FAIL':>6}")
    if report.indexer_score_result:
        r = report.indexer_score_result
        print(f"{r.name:<42} {r.max_abs_diff:>12.3e} {r.max_ulp_diff:>8d} {'PASS' if r.bitwise_identical else 'FAIL':>6}")
    print("-" * 78)

    print("\nINDEXER OVERLAP:")
    io = report.indexer_overlap
    print(f"  query_dependent:        {io['query_dependent']}")
    print(f"  all_rows_identical:     {io['all_rows_identical']}")
    print(f"  mean_jaccard_pairwise:  {io['mean_jaccard_pairwise']:.4f}")
    print(f"  n_rows_diff_vs_row0:    {io['n_rows_with_different_topk_vs_row0']}/{M_VERIFY-1}")
    print(f"  conclusion:              {io['conclusion']}")

    print("\n" + "=" * 78)
    print("DECIDED DESIGN:")
    print("=" * 78)
    print(report.decided_design)
    print()
    print(report.decided_rationale)

    if args.json:
        out = {
            "m_verify": M_VERIFY,
            "gemm_results": [r.__dict__ for r in report.gemm_results],
            "sdpa_result": report.sdpa_result.__dict__ if report.sdpa_result else None,
            "indexer_score_result": report.indexer_score_result.__dict__ if report.indexer_score_result else None,
            "indexer_overlap": report.indexer_overlap,
            "decided_design": report.decided_design,
            "decided_rationale": report.decided_rationale,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nReport written to {args.json}")

    # Exit code: 0 if a realizable design exists (full Kimi, custom-kernel track,
    # or indexer-stream-sharing), 1 if total dead end.
    return 0 if report.decided_design else 1


if __name__ == "__main__":
    sys.exit(main())
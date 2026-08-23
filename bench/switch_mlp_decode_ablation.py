"""P0 (2026-08-22 session): decode-shape switch_mlp ablation matrix.

Why: T3 measured FusedSwitchGLU at B=1 decode shape achieving only 27.7% of
546 GB/s peak (151.4 GB/s, ~300us/call). Never attributed. This script splits
the 3.6x shortfall into named causes at the EXACT decode shape (B=1, L=1,
top_k=6-of-256, mxfp4 g=32, per-TP-rank inter=1024).

Tiers (all pipelined, 300 iters, mx.synchronize bracketing like T3):
  A  full FusedSwitchGLU path (T3 reproduction: gather_sort + fused gather_qmm
     + swiglu + down gather_qmm + scatter_unsort)
  B  core only: the two gather_qmm calls with pre-sorted indices (no
     gather_sort/scatter_unsort) -> bounds sort/scatter overhead
  C  dense equivalent: 6 stacked dense mxfp4 quantized_matmul calls at
     identical per-expert shapes (contiguous weights, no expert gather)
     -> bounds the sparse-gather access-pattern cost
  D  dense bf16 (non-quantized) matmul, same shapes -> bounds dequant cost
  E  8-bit affine quant variant of tier A -> dequant-format-specific cost
  F  batch sweep: tier A at B=1,2,4,8 synthetic tokens -> does efficiency
     scale with batch?

Bandwidth accounting: bytes touched = top_k * bytes_per_expert (weights
dominate; activations negligible at these shapes).
Run: .venv/bin/python bench/switch_mlp_decode_ablation.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx-lm"))

HIDDEN = 4096
INTER = 1024  # per-TP-rank
N_EXPERTS = 256
TOP_K = 6
GS = 32
PEAK = 546e9

WARMUP = 20
ITERS = 300


def timeit(fn):
    for _ in range(WARMUP):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = []
    for _ in range(ITERS):
        outs.append(fn())
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / ITERS


def quant_weights(mode, bits):
    kw = dict(group_size=GS, bits=bits)
    if mode:
        kw["mode"] = mode
    gate_up = mx.random.normal((N_EXPERTS, 2 * INTER, HIDDEN)).astype(mx.bfloat16)
    down = mx.random.normal((N_EXPERTS, HIDDEN, INTER)).astype(mx.bfloat16)
    gu_q = mx.quantize(gate_up, **kw)
    dn_q = mx.quantize(down, **kw)
    mx.eval(gu_q, dn_q)
    return gu_q, dn_q, kw


def bytes_per_expert(bits, mode):
    params = 2 * INTER * HIDDEN + HIDDEN * INTER  # gate+up+down
    w_bytes = params * bits / 8
    # scales: fp16 per group for mxfp4 uses e8m0 1 byte; affine has scale+bias
    if mode == "mxfp4":
        scale_bytes = params / GS * 1
    else:
        scale_bytes = params / GS * 4  # fp16 scale + fp16 bias
    return w_bytes + scale_bytes


def main():
    mx.random.seed(0)
    results = {}
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    def build_runner(gu_q, dn_q, kw, B, use_sort, core_only=False):
        x0 = mx.random.normal((B, 1, HIDDEN)).astype(mx.bfloat16)
        ind_pool = [mx.random.randint(0, N_EXPERTS, (B, 1, TOP_K)) for _ in range(64)]
        mx.eval(x0, *ind_pool)
        ctr = [0]

        def run():
            inds = ind_pool[ctr[0] % 64]
            ctr[0] += 1
            x = mx.expand_dims(x0, (-2, -3))
            if use_sort:
                xs, idx, inv = _gather_sort(x, inds)
            else:
                xs, idx, inv = x, inds, None
            g_u = mx.gather_qmm(xs, *gu_q, rhs_indices=idx, transpose=True,
                                sorted_indices=use_sort, **kw)
            gate, up = mx.split(g_u, 2, axis=-1)
            h = mx.multiply(up, mx.sigmoid(gate) * gate)
            out = mx.gather_qmm(h, *dn_q, rhs_indices=idx, transpose=True,
                                sorted_indices=use_sort, **kw)
            if use_sort:
                out = _scatter_unsort(out, inv, inds.shape)
            return out.squeeze(-2)

        return run

    # mxfp4 weights
    gu4, dn4, kw4 = quant_weights("mxfp4", 4)
    bpe4 = bytes_per_expert(4, "mxfp4")

    # Tier A: full path, B=1 (production decode: indices.size=6 < 64 so no sort!)
    for label, use_sort in [("A_nosort_prod", False), ("A_forced_sort", True)]:
        t = timeit(build_runner(gu4, dn4, kw4, 1, use_sort))
        bw = TOP_K * bpe4 / t
        results[label] = dict(us=t * 1e6, gbs=bw / 1e9, pct_peak=bw / PEAK * 100)
        print(label, results[label], flush=True)

    # Tier C: dense mxfp4 qmm, one expert weight, 6 sequential calls (same bytes)
    # dense tier: 256 separate expert weights, rotate which 6 are used per call
    gu_dense = [mx.quantize(mx.random.normal((2 * INTER, HIDDEN)).astype(mx.bfloat16),
                            group_size=GS, bits=4, mode="mxfp4") for _ in range(64)]
    dn_dense = [mx.quantize(mx.random.normal((HIDDEN, INTER)).astype(mx.bfloat16),
                            group_size=GS, bits=4, mode="mxfp4") for _ in range(64)]
    for w in gu_dense + dn_dense:
        mx.eval(w)
    xv = mx.random.normal((1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(xv)
    dctr = [0]

    def dense6():
        outs = []
        for j in range(TOP_K):
            k = (dctr[0] * TOP_K + j) % 64
            g_u = mx.quantized_matmul(xv, *gu_dense[k], transpose=True,
                                      group_size=GS, bits=4, mode="mxfp4")
            gate, up = mx.split(g_u, 2, axis=-1)
            h = mx.multiply(up, mx.sigmoid(gate) * gate)
            outs.append(mx.quantized_matmul(h, *dn_dense[k], transpose=True,
                                            group_size=GS, bits=4, mode="mxfp4"))
        dctr[0] += 1
        return mx.add(*outs[:2]) + outs[2] + outs[3] + outs[4] + outs[5]

    t = timeit(dense6)
    bw = TOP_K * bpe4 / t
    results["C_dense_qmm_x6"] = dict(us=t * 1e6, gbs=bw / 1e9, pct_peak=bw / PEAK * 100)
    print("C_dense_qmm_x6", results["C_dense_qmm_x6"], flush=True)

    # Tier D: dense bf16 matmul same shapes x6
    gu_bfs = [mx.random.normal((2 * INTER, HIDDEN)).astype(mx.bfloat16) for _ in range(32)]
    dn_bfs = [mx.random.normal((HIDDEN, INTER)).astype(mx.bfloat16) for _ in range(32)]
    mx.eval(*gu_bfs, *dn_bfs)
    bpe_bf = (2 * INTER * HIDDEN + HIDDEN * INTER) * 2
    bctr = [0]

    def densebf():
        outs = []
        for j in range(TOP_K):
            k = (bctr[0] * TOP_K + j) % 32
            g_u = xv @ gu_bfs[k].T
            gate, up = mx.split(g_u, 2, axis=-1)
            h = mx.multiply(up, mx.sigmoid(gate) * gate)
            outs.append(h @ dn_bfs[k].T)
        bctr[0] += 1
        return outs[0] + outs[1] + outs[2] + outs[3] + outs[4] + outs[5]

    t = timeit(densebf)
    bw = TOP_K * bpe_bf / t
    results["D_dense_bf16_x6"] = dict(us=t * 1e6, gbs=bw / 1e9, pct_peak=bw / PEAK * 100)
    print("D_dense_bf16_x6", results["D_dense_bf16_x6"], flush=True)

    # Tier E: 8-bit affine gather variant
    gu8, dn8, kw8 = quant_weights("affine", 8)
    bpe8 = bytes_per_expert(8, "affine")
    t = timeit(build_runner(gu8, dn8, kw8, 1, False))
    bw = TOP_K * bpe8 / t
    results["E_affine8_gather"] = dict(us=t * 1e6, gbs=bw / 1e9, pct_peak=bw / PEAK * 100)
    print("E_affine8_gather", results["E_affine8_gather"], flush=True)

    # Tier F: batch sweep on mxfp4 gather path (unique experts touched grows w/ B)
    for B in (1, 2, 4, 8, 16, 32):
        use_sort = B * TOP_K >= 64
        t = timeit(build_runner(gu4, dn4, kw4, B, use_sort))
        # bytes: expected unique experts touched (approx min(B*top_k, ...)); use
        # exact expectation of distinct draws: E = N*(1-(1-1/N)^(B*k))
        draws = B * TOP_K
        uniq = N_EXPERTS * (1 - (1 - 1 / N_EXPERTS) ** draws)
        bw = uniq * bpe4 / t
        results[f"F_B{B}"] = dict(us=t * 1e6, uniq_experts=uniq, gbs=bw / 1e9,
                                  pct_peak=bw / PEAK * 100, sort=use_sort)
        print(f"F_B{B}", results[f"F_B{B}"], flush=True)

    out = Path(__file__).parent / "results" / "switch_mlp_decode_ablation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()

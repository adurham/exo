"""M3: Is the 62.6% MoE efficiency actually anomalous?

`bench/moe_production_class_bench.py` compares the real mlx-lm SwitchGLU MoE
block (mxfp4, ragged top-6-of-256 routing, gather_qmm) against a matched-shape
DENSE fp16 GEMM and reports ~62.6%. That denominator conflates TWO different
penalties:

  (a) quantized-matmul inefficiency  -- mxfp4 qmm is simply slower than fp16
      GEMM at these shapes, with no MoE involved at all; and
  (b) MoE-specific overhead          -- routing, gather/sort, ragged per-expert
      run lengths, scatter/unsort.

This script separates them by measuring FOUR tiers at the EXACT same
per-GEMM M/K/N (DeepSeek-V4-Flash per-TP-rank production shapes):

  T1  dense fp16 GEMM                      (the current 62.6% denominator)
  T2  dense mxfp4 quantized_matmul         (same M/K/N, ONE weight, no routing,
                                            no gather, no scatter)
  T3  gather_qmm, degenerate routing       (all M rows -> expert 0: exercises the
                                            gather/scatter machinery with ZERO
                                            raggedness)
  T4  real SwitchGLU MoE, ragged routing   (production path)

Interpretation:
  T2/T1  = the quantized-matmul tax. If this is ~60-65%, then 62.6% is NOT
           anomalous -- the MoE kernel is already at the dense-quantized ceiling
           and there is no ~37% MoE-specific gap to chase.
  T4/T2  = the TRUE MoE-specific efficiency. This is the number any MoE kernel
           optimization can actually move.
  T3/T2  = cost of the gather/scatter plumbing alone (raggedness excluded).
  T4/T3  = cost of raggedness / short per-expert runs.

Run:  .venv/bin/python bench/moe_vs_dense_qmm_isolation.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mlx-lm"))
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

# --- DeepSeek-V4-Flash production config (per TP rank, worldSize=2) ---
HIDDEN = 4096
INTER_PER_RANK = 1024  # moe_intermediate_size 2048 // 2 TP ranks
N_EXPERTS = 256
TOP_K = 6
GROUP_SIZE = 32
BITS = 4
QMODE = "mxfp4"
DTYPE = mx.bfloat16

WARMUP = 5
ITERS = 20

# Real per-rank L values: 1024 = STEP_SIZE 2048, 2048 = STEP_SIZE 4096
# (SEQ_SPLIT halves the nominal chunk across the 2 TP ranks).
TOKEN_COUNTS = (1024, 2048)


def bench_call(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples


def make_routing(n_tokens: int, seed: int = 0) -> mx.array:
    """Realistic ragged top-k routing with a skewed (Zipf-ish) expert prior."""
    mx.random.seed(seed)
    prior = mx.random.normal((N_EXPERTS,)) * 1.0
    logits = mx.random.normal((n_tokens, N_EXPERTS)) + prior
    idx = mx.argpartition(-logits, TOP_K, axis=-1)[:, :TOP_K]
    return idx.astype(mx.uint32)


def quantize_weight(shape):
    """mxfp4-quantize a weight of the given (out, in) shape -> qmm-ready parts."""
    w = mx.random.normal(shape).astype(DTYPE)
    parts = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode=QMODE)
    wq, scales = parts[0], parts[1]
    biases = parts[2] if len(parts) > 2 else None
    mx.eval(wq, scales, *( [biases] if biases is not None else [] ))
    return wq, scales, biases


def qmm(x, packed):
    wq, scales, biases = packed
    return mx.quantized_matmul(
        x, wq, scales, biases, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode=QMODE,
    )


def flops_for(m: int) -> int:
    """gate + up (K=HIDDEN,N=INTER) + down (K=INTER,N=HIDDEN), 2 flops/MAC."""
    return 2 * m * (2 * HIDDEN * INTER_PER_RANK + INTER_PER_RANK * HIDDEN)


def main():
    print("MLX:", mx.__version__)
    print(
        f"config: hidden={HIDDEN} inter/rank={INTER_PER_RANK} experts={N_EXPERTS} "
        f"top_k={TOP_K} quant={QMODE} g={GROUP_SIZE} b={BITS} dtype={DTYPE}"
    )

    glu = SwitchGLU(HIDDEN, INTER_PER_RANK, N_EXPERTS, bias=False)
    glu.set_dtype(DTYPE)
    nn.quantize(glu, group_size=GROUP_SIZE, bits=BITS, mode=QMODE)
    mx.eval(glu.parameters())

    # Dense mxfp4 weights, same K/N as the expert GEMMs (nn.Linear layout: (out,in)).
    w_gate = quantize_weight((INTER_PER_RANK, HIDDEN))
    w_up = quantize_weight((INTER_PER_RANK, HIDDEN))
    w_down = quantize_weight((HIDDEN, INTER_PER_RANK))

    rows = []
    for n_tokens in TOKEN_COUNTS:
        M = n_tokens * TOP_K  # total activated rows across all three GEMMs
        fl = flops_for(M)
        rec = {"n_tokens": n_tokens, "M": M}

        # ---- T4: real MoE, ragged production routing ----
        idx = make_routing(n_tokens)
        x3 = mx.random.normal((1, n_tokens, HIDDEN)).astype(DTYPE)
        mx.eval(x3, idx)
        counts = mx.zeros((N_EXPERTS,), dtype=mx.uint32)
        counts = counts.at[idx.flatten()].add(mx.ones((idx.size,), dtype=mx.uint32))
        mx.eval(counts)
        nz = [v for v in counts.tolist() if v > 0]
        t4, _ = bench_call(lambda: glu(x3, idx))
        rec["moe_ragged_ms"] = t4 * 1e3
        rec["moe_ragged_tflops"] = fl / t4 / 1e12
        rec["experts_used"] = len(nz)
        rec["m_per_expert_median"] = statistics.median(nz)
        rec["m_per_expert_min"] = min(nz)
        rec["m_per_expert_max"] = max(nz)

        # ---- T3: gather path, degenerate routing (all rows -> expert 0) ----
        idx0 = mx.zeros((n_tokens, TOP_K), dtype=mx.uint32)
        mx.eval(idx0)
        t3, _ = bench_call(lambda: glu(x3, idx0))
        rec["moe_single_expert_ms"] = t3 * 1e3
        rec["moe_single_expert_tflops"] = fl / t3 / 1e12

        # ---- T2: dense mxfp4 quantized_matmul, same M/K/N, no routing ----
        xa = mx.random.normal((M, HIDDEN)).astype(DTYPE)
        xb = mx.random.normal((M, INTER_PER_RANK)).astype(DTYPE)
        mx.eval(xa, xb)

        def dense_q():
            return (qmm(xa, w_gate), qmm(xa, w_up), qmm(xb, w_down))

        t2, _ = bench_call(dense_q)
        rec["dense_mxfp4_ms"] = t2 * 1e3
        rec["dense_mxfp4_tflops"] = fl / t2 / 1e12

        # ---- T1: dense fp16 GEMM, same M/K/N ----
        a1 = mx.random.normal((M, HIDDEN)).astype(mx.float16)
        b1 = mx.random.normal((HIDDEN, INTER_PER_RANK)).astype(mx.float16)
        b1b = mx.random.normal((HIDDEN, INTER_PER_RANK)).astype(mx.float16)
        a2 = mx.random.normal((M, INTER_PER_RANK)).astype(mx.float16)
        b2 = mx.random.normal((INTER_PER_RANK, HIDDEN)).astype(mx.float16)
        mx.eval(a1, b1, b1b, a2, b2)
        t1, _ = bench_call(lambda: (a1 @ b1, a1 @ b1b, a2 @ b2))
        rec["dense_fp16_ms"] = t1 * 1e3
        rec["dense_fp16_tflops"] = fl / t1 / 1e12

        rec["T2_over_T1_quant_tax_pct"] = t1 / t2 * 100
        rec["T4_over_T1_pct"] = t1 / t4 * 100
        rec["T4_over_T2_true_moe_eff_pct"] = t2 / t4 * 100
        rec["T3_over_T2_gather_plumbing_pct"] = t2 / t3 * 100
        rec["T4_over_T3_raggedness_pct"] = t3 / t4 * 100
        rows.append(rec)

        print(
            f"\n=== L={n_tokens} (M={M} activated rows) "
            f"experts_used={rec['experts_used']}/{N_EXPERTS} "
            f"M/expert min={rec['m_per_expert_min']} "
            f"med={rec['m_per_expert_median']:.0f} max={rec['m_per_expert_max']} ==="
        )
        print(f"  T1 dense fp16 GEMM      {t1*1e3:8.3f} ms {rec['dense_fp16_tflops']:7.2f} TFLOPS")
        print(f"  T2 dense mxfp4 qmm      {t2*1e3:8.3f} ms {rec['dense_mxfp4_tflops']:7.2f} TFLOPS")
        print(f"  T3 gather, 1 expert     {t3*1e3:8.3f} ms {rec['moe_single_expert_tflops']:7.2f} TFLOPS")
        print(f"  T4 MoE, ragged (prod)   {t4*1e3:8.3f} ms {rec['moe_ragged_tflops']:7.2f} TFLOPS")
        print(f"  -- quant tax      T2/T1 = {rec['T2_over_T1_quant_tax_pct']:5.1f}%")
        print(f"  -- headline       T4/T1 = {rec['T4_over_T1_pct']:5.1f}%   (the '62.6%' number)")
        print(f"  -- TRUE MoE eff   T4/T2 = {rec['T4_over_T2_true_moe_eff_pct']:5.1f}%")
        print(f"  -- gather plumbing T3/T2 = {rec['T3_over_T2_gather_plumbing_pct']:5.1f}%")
        print(f"  -- raggedness      T4/T3 = {rec['T4_over_T3_raggedness_pct']:5.1f}%")

    out = Path(__file__).resolve().parent / "results" / "moe_vs_dense_qmm_isolation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"mlx": mx.__version__, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

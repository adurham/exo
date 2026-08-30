"""P05 Phase A (step 1b): corrected M-grid lm_head mxfp8 numerics.

Fixes vs lmhead_numerics.py v1:
  1. Batch calls use EXACTLY M rows for M in {1, 3, 4} — production's real
     lm_head call shapes (decode L=1, DSpark draft L=3, verify L=4). v1
     conflated "M rows/call" with M*8 rows.
  2. MLX_GEMV_BATCH_INVARIANT=1 + MLX_STEEL_BATCH_INVARIANT=1 set BEFORE
     mlx import — production env parity for the BF16 arm (production pins
     BF16 batch invariance with these flags).
  3. More input samples for a tighter flip-rate estimate (128 rows/case).
  4. Also measures: the quantized head's M-dependence at the REAL batch
     sizes (M=3, M=4 vs M=1 per-row) — the qmv vs qmv_wide kernel split.
  5. Reports flips stratified by margin — the quality question is whether
     flips concentrate on near-ties (benign-ish) or confident tokens (bad).
"""
import json
import os
import sys
from pathlib import Path

os.environ["MLX_GEMV_BATCH_INVARIANT"] = "1"
os.environ["MLX_STEEL_BATCH_INVARIANT"] = "1"

import numpy as np

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).parent
W_BF16 = HERE / "head_weight.bf16"
V, D = 129280, 4096
N_TOKENS = 128  # distinct hidden states per case


def load_head_bf16():
    raw = np.fromfile(W_BF16, dtype=np.uint16)
    assert raw.size == V * D
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(V, D)


def main():
    w_np = load_head_bf16()

    lm_bf16 = nn.Linear(D, V, bias=False)
    lm_bf16.weight = mx.array(w_np).astype(mx.bfloat16)
    lm_bf16.eval()
    mx.eval(lm_bf16.parameters())

    lm_q = nn.Linear(D, V, bias=False)
    lm_q.weight = mx.array(w_np).astype(mx.bfloat16)
    qmod = lm_q.to_quantized(group_size=32, bits=8, mode="mxfp8")
    qmod.eval()
    mx.eval(qmod.parameters())

    rng = np.random.default_rng(20260830)
    Hs = rng.standard_normal((N_TOKENS, D)).astype(np.float32)
    Hn = Hs / (np.sqrt((Hs ** 2).mean(axis=-1, keepdims=True)) + 1e-6)
    x_all = mx.array(Hn).astype(mx.bfloat16)  # (N, D)

    # ---- per-row M=1 references ----
    lb1_rows = []
    lq1_rows = []
    for i in range(N_TOKENS):
        lb1_rows.append(lm_bf16(x_all[i : i + 1]))
        lq1_rows.append(qmod(x_all[i : i + 1]))
    lb1 = mx.concatenate(lb1_rows, axis=0)
    lq1 = mx.concatenate(lq1_rows, axis=0)
    mx.eval(lb1, lq1)

    results: dict = {"n_tokens": N_TOKENS}

    # ---- quantization error, per-row (the core numerics) ----
    lb_np = np.asarray(lb1.astype(mx.float32))
    lq_np = np.asarray(lq1.astype(mx.float32))
    d = lq_np - lb_np
    margins = np.sort(lb_np, axis=-1)[:, ::-1]
    margin = margins[:, 0] - margins[:, 1]
    qerr = np.abs(d).max(axis=-1)
    flips = lb_np.argmax(-1) != lq_np.argmax(-1)
    print(f"=== quantized-vs-BF16 per-row (M=1), {N_TOKENS} real-weighted rows ===")
    print(f"logit err: max={np.abs(d).max():.3f} mean={np.abs(d).mean():.4f} "
          f"rms={np.sqrt((d**2).mean()):.4f}")
    print(f"logit scale: bf16 std={lb_np.std():.2f}")
    print(f"top-1 flips: {flips.sum()}/{N_TOKENS} = {100*flips.mean():.2f}%")
    print(f"greedy margin: p50={np.median(margin):.3f} p10={np.percentile(margin,10):.3f} "
          f"p90={np.percentile(margin,90):.3f}")
    # flips stratified by margin quartile
    qs = np.percentile(margin, [25, 50, 75])
    strat = []
    lo = 0.0
    for hi in list(qs) + [float("inf")]:
        hi_f = float(hi)
        if np.isfinite(hi_f):
            sel = (margin >= lo) & (margin < hi_f)
            label = f"[{lo:.2f}, {hi_f:.2f})"
        else:
            sel = margin >= lo
            label = f"[{lo:.2f}, inf)"
        if sel.sum() > 0:
            strat.append({
                "margin_range": label,
                "n": int(sel.sum()),
                "flips": int(flips[sel].sum()),
                "flip_rate": float(flips[sel].mean()),
            })
        lo = hi_f
    for s in strat:
        print(f"  margin {s['margin_range']}: {s['flips']}/{s['n']} "
              f"({100*s['flip_rate']:.1f}% flips)")
    # top-5 overlap
    k = 5
    top_b = np.argpartition(-lb_np, k, axis=-1)[:, :k]
    top_q = np.argpartition(-lq_np, k, axis=-1)[:, :k]
    ov = np.mean([len(set(top_b[i]) & set(top_q[i])) / k for i in range(N_TOKENS)])
    print(f"top-5 set overlap: {100*ov:.1f}%")

    results["quant_err"] = {
        "max": float(np.abs(d).max()), "mean": float(np.abs(d).mean()),
        "rms": float(np.sqrt((d ** 2).mean())),
    }
    results["flips_total"] = int(flips.sum())
    results["flip_rate"] = float(flips.mean())
    results["margin_strata"] = strat
    results["top5_overlap"] = float(ov)

    # ---- M-batch invariance at production batch sizes ----
    print("\n=== M-batch invariance (row i's output vs same row at M=1) ===")
    for M in (3, 4):
        # batched call with exactly M rows, repeated over the token set
        max_bf16_dep = 0.0
        max_q_dep = 0.0
        for start in range(0, N_TOKENS - M + 1, M):
            xb = x_all[start : start + M]
            lbb = lm_bf16(xb)
            lqb = qmod(xb)
            mx.eval(lbb, lqb)
            lbb_np = np.asarray(lbb.astype(mx.float32))
            lqb_np = np.asarray(lqb.astype(mx.float32))
            max_bf16_dep = max(max_bf16_dep,
                float(np.abs(lbb_np - lb_np[start : start + M]).max()))
            max_q_dep = max(max_q_dep,
                float(np.abs(lqb_np - lq_np[start : start + M]).max()))
        print(f"M={M}: BF16 head batch-vs-M1 max={max_bf16_dep:.2e} | "
              f"mxfp8 head batch-vs-M1 max={max_q_dep:.2e}")
        results[f"m_dep_M{M}"] = {"bf16": max_bf16_dep, "mxfp8": max_q_dep}

    # ---- throughput microbench (laptop GPU, direction only) ----
    import time

    def bench(fn, n=20, warmup=5):
        for _ in range(warmup):
            mx.eval(fn())
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2] * 1e6

    x1 = x_all[:1]
    x4 = x_all[:4]
    print("\n=== laptop-GPU timing (direction only; cluster A/B is the real test) ===")
    t_bf16_m1 = bench(lambda: lm_bf16(x1))
    t_bf16_m4 = bench(lambda: lm_bf16(x4))
    t_q_m1 = bench(lambda: qmod(x1))
    t_q_m4 = bench(lambda: qmod(x4))
    w_bytes_bf16 = V * D * 2
    w_bytes_q = (V * (D // 4) // 4) * 4 + V * (D // 32)  # packed u32 + u8 scales
    print(f"BF16 head M=1: {t_bf16_m1:8.1f} us  ({w_bytes_bf16/1e9:.3f} GB -> "
          f"{w_bytes_bf16/(t_bf16_m1*1e-6)/1e9:.0f} GB/s)")
    print(f"mxfp8 head M=1: {t_q_m1:8.1f} us  ({w_bytes_q/1e9:.3f} GB -> "
          f"{w_bytes_q/(t_q_m1*1e-6)/1e9:.0f} GB/s)")
    print(f"BF16 M=4: {t_bf16_m4:8.1f} us | mxfp8 M=4: {t_q_m4:8.1f} us")
    print(f"M=1 speedup: {t_bf16_m1/t_q_m1:.2f}x | M=4 speedup: {t_bf16_m4/t_q_m4:.2f}x")
    results["timing_us"] = {
        "bf16_m1": t_bf16_m1, "bf16_m4": t_bf16_m4,
        "mxfp8_m1": t_q_m1, "mxfp8_m4": t_q_m4,
    }

    out = HERE / "lmhead_numerics_v2.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
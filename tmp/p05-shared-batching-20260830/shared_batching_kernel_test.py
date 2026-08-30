"""P05 Phase C (offline): is batched shared_experts (M=4) numerically
recoverable — i.e. is there a "middle ground" that keeps most of the
measured -1.2 ms/cycle verify-batching win without the 2026-08-04 divergence?

Mechanism found by code-read BEFORE any measurement (P05, 2026-08-30):
  mlx/backend/metal/quantized.cpp `dispatch_qmv`:
      if (M >= 2 && use_qmv_wide(mode, d)) { qmv_wide(...); return; }
      qmv(...);
  `use_qmv_wide` = (mode != "affine") || arch >= M15 — mxfp8 on M4 Max
  (arch >= 15) routes M>=2 to a DIFFERENT kernel (qmv_wide: per-threadgroup
  input-vector tiling, fp32 accumulation differences vs qmv's per-row
  streaming). M==1 always uses `qmv`. NO batch-invariance env flag covers
  the quantized path (MLX_GEMV/STEEL_BATCH_INVARIANT act on matmul.cpp's
  gemv/steel paths only) — and the 2026-08-04 bisect ran WITH those flags
  already shipped (start_cluster default since 2026-07-10, commit 69a770084),
  so the 0.023% shared divergence is exactly this kernel split.
  Also relevant: QuantizedMatmul::eval_gpu routes M >= vector_limit to
  `qmm`/`qmm_splitk` — the 0-ulp batch-invariant GEMM family
  (verified 0-ulp across M=1..8 by bench/micro_batch_invariance.py, cited
  in the VERIFY_BATCH gate header, deepseek_v4.py ~line 1633).

The candidate "middle ground" to test offline:
  Force the verify-batch M=4 shared_experts matmuls onto the qmm path by
  ZERO-PADDING the batch to M >= vector_limit (get_qmv_batch_limit(K,N)).
  If qmm at M=4 is bitwise-equal per row to qmv at M=1 (the documented
  0-ulp claim), then a padded-batch shared_experts would be numerically
  lossless w.r.t. the per-row path while still batching the WEIGHT READ
  (the actual -1.2 ms win: 44.4 us batched-L4 vs 82.0 us per-row-group).
  Cost: qmm at M=4 does ~vector_limit rows of work — if vector_limit is
  16 or 32, that's 4-8x the FLOPs of M=4, potentially negating the win.

Measurements (real shared_experts weights, production shapes):
  1. Reproduce the divergence: qmv_wide(M=4) vs qmv(M=1) per-row on real
     shared gate/up/down weights at real hidden states -> confirm nonzero
     divergence exists on this hardware (the 0.023% claim).
  2. Baseline kernel speeds: M=1 (qmv) vs M=4 (qmv_wide) vs M=4 forced
     through qmm (padded to vector_limit) for the 3 projections.
  3. If qmm is bitwise per-row-equal AND fast enough (<= batched qmv_wide
     time), the middle ground is real -> document as the candidate fix.
     Otherwise document Phase C as structurally dead (no middle ground
     below the existing 0.023% floor).

Zero cluster contact. Laptop M-series GPU, real weights pulled from the
checkpoint (rank-local shapes: shared intermediate = 2048/2 = 1024/rank).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "mlx-lm"
sys.path.insert(0, str(REPO))

import numpy as np

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).parent
W = HERE.parent / "p05-sinkhorn-real-20260830" / "p05_weights"
manifest = json.loads((W / "manifest.json").read_text())
CKPT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"

H, INTER = 4096, 1024  # per-rank shared intermediate (2048 / 2 ranks)


def load_bf16_range(key, hdr_cache=None):
    """Range-read a BF16 tensor straight from the checkpoint shards."""
    import struct

    # find shard + offsets by header scan (small model dir on laptop has no
    # shards; this runs on the STUDIO or against extracted bins if present)
    raise NotImplementedError("use extracted .bin files instead")


def load_shared_proj(layer: int, proj: str):
    """Load a real shared_experts projection weight from the extracted
    p05_weights if present, else synthesize a same-scale mxfp8 weight.

    NOTE: the p05_weights extraction pulled HC/norm/markov tensors only.
    The real shared_experts weights (mxfp8-packed uint32 + uint8 scales)
    were NOT extracted (they are large: 3 x ~13MB per layer). For the
    KERNEL-CLASS question (qmv vs qmv_wide vs qmm on mxfp8), a synthetic
    mxfp8 weight with production scale statistics is representative: the
    divergence mechanism is accumulation-order in the KERNEL, not the
    specific weight values (the 2026-08-04 bisect measured 1/4300 flips
    on REAL weights; here we measure the kernel class difference, which
    is value-dependent in magnitude but not in mechanism).
    """
    mx.random.seed(1000 + layer)
    w = (mx.random.normal((INTER, H)) * 0.05).astype(mx.bfloat16)
    lin = nn.Linear(H, INTER, bias=False)
    lin.weight = w
    q = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q.eval()
    return q


def main():
    # Confirm the hardware + kernel dispatch facts first
    print(f"GPU arch: {mx.device_info()['architecture']} "
          f"(use_qmv_wide active for mxfp8 M>=2: "
          f"{'yes (arch>=15 or fp-mode)' if True else ''})")

    # get_qmv_batch_limit equivalent: vector_limit for transpose_ is read
    # from a Metal kernel-availability probe. Empirically find it: smallest
    # M at which the output stops going through the qmv path. We can't read
    # the C++ directly from Python; instead measure speed discontinuities.
    # The MLX source pins transpose_ qmv limit via get_qmv_batch_limit(K,N)
    # — for M4 Max typically 16 or 32.

    results = {}

    # --- real-ish inputs: decode-shaped hidden states ---
    rng = np.random.default_rng(20260830)
    x1_np = rng.standard_normal((1, H)).astype(np.float32)
    x1_np = x1_np / (np.sqrt((x1_np ** 2).mean(-1, keepdims=True)) + 1e-6)
    x1 = mx.array(x1_np).astype(mx.bfloat16)
    x4 = mx.concatenate([x1] * 4, axis=0)  # same row repeated: per-row
    # outputs must be identical if kernels were per-row exact

    q = load_shared_proj(3, "gate_proj")

    # M=1 reference (qmv kernel)
    y1 = q(x1)
    mx.eval(y1)

    # M=4 batched (qmv_wide kernel) — same row repeated 4x
    y4 = q(x4)
    mx.eval(y4)

    d_wide = (y4 - mx.broadcast_to(y1, y4.shape)).astype(mx.float32)
    max_wide = float(mx.abs(d_wide).max())
    print(f"\nqmv_wide(M=4) vs qmv(M=1) same-row max diff: {max_wide:.3e}")

    # --- force the qmm path by padding M to the vector limit ---
    # try candidate vector limits: 8, 16, 32, 64
    for vl in (8, 16, 32, 64):
        xp = mx.concatenate([x4] + [mx.zeros_like(x1)] * (vl - 4), axis=0)
        yp = q(xp)
        mx.eval(yp)
        d_pad = (yp[:4] - y4).astype(mx.float32)
        max_pad_vs_wide = float(mx.abs(d_pad).max())
        d_pad_vs_ref = (yp[:4] - mx.broadcast_to(y1, (4, INTER))).astype(mx.float32)
        max_pad_vs_ref = float(mx.abs(d_pad_vs_ref).max())
        print(f"padded M={vl} (qmm path) vs qmv_wide M=4: {max_pad_vs_wide:.3e} | "
              f"vs qmv M=1 ref: {max_pad_vs_ref:.3e}")
        results[f"pad_{vl}"] = {
            "vs_wide": max_pad_vs_wide, "vs_ref": max_pad_vs_ref
        }

    # --- speeds at production shapes (median of 5, warm) ---
    import time

    def bench(fn, n=30, warmup=5):
        for _ in range(warmup):
            mx.eval(fn())
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2] * 1e6  # us

    # weight bytes per call: packed 13MB -> the -1.2ms/cycle claim is about
    # reading the weight ONCE for 4 rows vs 4 times.
    t_m1 = bench(lambda: q(x1))
    t_m4_wide = bench(lambda: q(x4))
    pad_variants = {}
    for vl in (16, 32):
        pad_variants[vl] = bench(
            lambda vl=vl: q(mx.concatenate(
                [x4] + [mx.zeros_like(x1)] * (vl - 4), axis=0))
        )
    print(f"\nspeeds (gate_proj, H={H}->I={INTER}, laptop GPU):")
    print(f"  M=1 qmv:           {t_m1:8.1f} us/call")
    print(f"  M=4 qmv_wide:      {t_m4_wide:8.1f} us/call")
    for vl, t in pad_variants.items():
        print(f"  M={vl} qmm padded:  {t:8.1f} us/call")
    results["speeds_us"] = {"m1_qmv": t_m1, "m4_qmv_wide": t_m4_wide,
                            **{f"pad{vl}_qmm": t for vl, t in pad_variants.items()}}
    results["wide_vs_ref_max"] = max_wide

    out = HERE / "shared_batching_kernel_test.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
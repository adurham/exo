# P05 Phase C: REAL shared_experts weights offline M-invariance test.
# Runs standalone on m4-1 beside production (no relaunch).
#
# Question (P03 target #3): is there a numerically-safer batched
# shared_experts formulation that keeps most of the measured -1.2 ms/cycle
# verify-batching win? The 2026-08-04 bisect found batched shared was the
# isolated divergence source (0.023% vs 0%). P05 code-read found the
# mechanism: mlx quantized.cpp dispatch_qmv routes M>=2 mxfp8 to qmv_wide
# (a different kernel with different accumulation tiling) vs M==1's qmv,
# and NO batch-invariance flag covers the quantized path (GEMV/STEEL flags
# act only on matmul.cpp paths).
#
# Test with the REAL layer-3 shared_experts weights (native FP8 e4m3 +
# e8m0 scales, repacked to MLX mxfp8 like production's sanitize does):
#   A. reproduce the divergence: batched M=4 vs per-row M=1 (x4 same rows)
#   B. candidate middle grounds, measured for numerics AND speed:
#      1. per-row on the BATCHED weight (today's production fix = baseline)
#      2. qmv_wide batched M=4 (the blocked fast path)
#      3. forced per-row qmm via zero-padding M to the qmm threshold
#   C. The specific question from the task: does a "numerically safer
#      batched formulation" exist? Test the fp32-accumulate wrapper option
#      too: cast the mxfp8 dequant through a small fp32 epilogue won't
#      help (accumulation happens INSIDE the kernel), so the only true
#      middle grounds are (a) qmm-padded batching (if 0-ulp) or
#      (b) accept qmv_wide's per-row divergence profile on REAL weights.
import os
import sys

sys.path.insert(0, os.path.expanduser("~/repos/exo/mlx-lm"))

import json
import struct
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

CKPT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"
OUT = Path.home() / "repos/exo/tmp/p05-shared-batching-20260830"
OUT.mkdir(parents=True, exist_ok=True)
H, INTER = 4096, 1024  # per-rank shapes (2048/2)


def load_f8(shard, key):
    """Load native F8_E4M3 weight + F8_E8M0 scale from the checkpoint."""
    with open(CKPT / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        info = hdr[key]
        o0, o1 = info["data_offsets"]
        fh.seek(n + 8 + o0)
        raw = fh.read(o1 - o0)
    import numpy as np
    if info["dtype"] == "F8_E4M3":
        # reinterpret the 1-byte e4m3 as the uint8 payload MLX mxfp8 packs
        return np.frombuffer(raw, dtype=np.uint8).reshape(info["shape"])
    if info["dtype"] == "F8_E8M0":
        return np.frombuffer(raw, dtype=np.uint8).reshape(info["shape"])
    raise ValueError(info["dtype"])


def build_quantized_linear(shard, wkey, skey, out_dim, in_dim):
    """Repack the checkpoint's native fp8 into an MLX mxfp8 QuantizedLinear
    exactly the way production's sanitize() does (weight.view(uint32) +
    scale repeated to group granularity). This mirrors the on-disk -> MLX
    conversion the runner performs."""
    import numpy as np
    w_u8 = load_f8(shard, wkey)
    s_u8 = load_f8(shard, skey)  # (out_dim//32?, groups) — shape from header

    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.weight = mx.zeros((out_dim, in_dim))
    # MLX mxfp8 packing: 4 x e4m3 per uint32, scales uint8 per 32-group
    w5 = w_u8.reshape(out_dim, in_dim // 4, 4)
    packed = np.zeros((out_dim, in_dim // 4), dtype=np.uint32)
    for j in range(4):
        packed |= w5[:, :, j].astype(np.uint32) << (8 * j)
    # scales: checkpoint stores (out//group_rows?) — inspect: [16, 32] for
    # w1 means 16 row-groups x 32 col-groups; MLX wants (out_dim, in_dim/32)
    # e8m0 per 32-element group along the input dim. Expand rows.
    s_exp = np.repeat(np.repeat(s_u8, 32, axis=0), 1, axis=1)  # (512, 32)?
    q = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q.weight = mx.array(packed)
    q.scales = mx.array(s_exp).reshape(out_dim, in_dim // 32)
    q.eval()
    mx.eval(q.parameters())
    return q


def main():
    import numpy as np

    shard = "model-00005-of-00048.safetensors"
    print("loading real layer-3 shared_experts gate_proj (w1)...")
    w1_u8 = load_f8(shard, "layers.3.ffn.shared_experts.w1.weight")
    s1_u8 = load_f8(shard, "layers.3.ffn.shared_experts.w1.scale")
    print(f"w1: {w1_u8.shape} e4m3, scales: {s1_u8.shape} e8m0")

    # Build MLX mxfp8 quantized linear EXACTLY the production way
    # (sanitize(): uint8 weight -> .view(uint32); scale (16,32) e8m0 ->
    #  repeat(v, 4, -1) -> (16,128) -> repeat(...,128, 0) -> (2048,128))
    out_dim, in_dim = w1_u8.shape  # (2048, 4096)
    w5 = w1_u8.reshape(out_dim, in_dim // 4, 4)
    packed = np.zeros((out_dim, in_dim // 4), dtype=np.uint32)
    for j in range(4):
        packed |= w5[:, :, j].astype(np.uint32) << (8 * j)
    s_rep = np.repeat(np.repeat(s1_u8, 4, -1), 128, 0)  # (2048, 128)
    assert s_rep.shape == (out_dim, in_dim // 32), s_rep.shape

    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.weight = mx.zeros((out_dim, in_dim))
    q = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q.weight = mx.array(packed)
    q.scales = mx.array(s_rep)
    q.eval()
    mx.eval(q.parameters())
    print("built mxfp8 QuantizedLinear:", type(q).__name__,
          q.weight.shape, q.scales.shape)

    # realistic inputs: decode-shaped
    rng = np.random.default_rng(20260830)
    Hs = rng.standard_normal((64, H)).astype(np.float32)
    Hn = Hs / (np.sqrt((Hs ** 2).mean(-1, keepdims=True)) + 1e-6)
    x_all = mx.array(Hn).astype(mx.bfloat16)

    results = {"host": os.uname().nodename, "gpu": mx.device_info()["architecture"]}

    # ---- A. reproduce the batched-vs-per-row divergence (M=4) ----
    y1_list = [q(x_all[i:i+1]) for i in range(64)]
    y1 = mx.concatenate(y1_list, axis=0)
    mx.eval(y1)
    y4_list = [q(x_all[i:i+4]) for i in range(0, 64, 4)]
    y4 = mx.concatenate(y4_list, axis=0)
    mx.eval(y4)
    d = (y4 - y1).astype(mx.float32)
    max_div = float(mx.abs(d).max())
    nz = int((mx.abs(d) > 0).sum())
    tot = int(d.size)
    print(f"\nA. qmv_wide(M=4) vs qmv(M=1) on REAL layer-3 w1: "
          f"max={max_div:.3e} nonzero={nz}/{tot} ({100*nz/tot:.4f}%)")

    # ---- B. middle grounds ----
    # B3: zero-pad to force qmm (vector_limit probe)
    print("\nB. middle-ground candidates:")
    x4_0 = x_all[:4]
    ref_rows = []
    for i in range(4):
        r = q(x4_0[i:i+1])
        mx.eval(r)
        ref_rows.append(r)
    ref = mx.concatenate(ref_rows, axis=0)

    for vl in (8, 16, 32):
        pad = mx.concatenate([x4_0] + [mx.zeros((1, H), dtype=mx.bfloat16)] * (vl - 4), axis=0)
        yp = q(pad)
        mx.eval(yp)
        dp = (yp[:4] - ref).astype(mx.float32)
        print(f"  padded-M={vl} (qmm path) vs per-row ref: "
              f"max={float(mx.abs(dp).max()):.3e}")

    # ---- C. speeds (production shapes, per-rank I=1024 slice) ----
    def bench(fn, n=30, warmup=8):
        for _ in range(warmup):
            mx.eval(fn())
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts)//2] * 1e6

    # per-rank slice: rows 0:1024 of the weight/scales (rank-0 half).
    # NOTE: q is (2048 out, 4096 in) FULL-width; production shards
    # all-to-sharded on axis max(ndim-2,0)=1 for packed QuantizedLinear —
    # the packed weight (out, in/4) shards to (out, in/8) per rank...
    # actually QuantizedAllToShardedLinear slices the INPUT dim. Emulate:
    # keep out full? No — production gate_proj is all-to-sharded on the
    # OUTPUT (intermediate) dim for gate/up. For this kernel-class test the
    # rank slice shape only affects SIZE, not the kernel class. Slice rows
    # (out dim) to 1024 AND pack consistently: slice packed weight rows.
    q_r = nn.Linear(in_dim, 1024, bias=False)
    q_r = q_r.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q_r.weight = q.weight[:1024]
    q_r.scales = q.scales[:1024]
    q_r.eval()
    mx.eval(q_r.parameters())

    t_m1 = bench(lambda: mx.concatenate([q_r(x_all[i:i+1]) for i in range(4)], axis=0))
    t_m4 = bench(lambda: q_r(x_all[:4]))
    pads = {}
    for vl in (16, 32):
        pad = mx.concatenate([x_all[:4]] + [mx.zeros((1, H), dtype=mx.bfloat16)] * (vl - 4), axis=0)
        pads[vl] = bench(lambda vl=vl, pad=pad: q_r(pad))
    print(f"\nC. speeds (per-rank w1 slice, I=1024):")
    print(f"  per-row M=1 x4 (production fix): {t_m1:8.1f} us")
    print(f"  batched M=4 qmv_wide (blocked):  {t_m4:8.1f} us")
    for vl, t in pads.items():
        print(f"  padded M={vl} qmm:              {t:8.1f} us")

    results.update({
        "divergence_m4_vs_m1": {"max_abs": max_div, "nonzero": nz, "total": tot,
                                 "pct": 100*nz/tot},
        "speeds_us": {"per_row_m1_x4": t_m1, "batched_m4_wide": t_m4,
                      **{f"pad{vl}_qmm": t for vl, t in pads.items()}},
    })
    out = OUT / "real_shared_m_invariance.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
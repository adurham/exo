# P05 Phase C follow-up #2: same pad8 speed+numerics measurement on the SECOND
# real tensor — layers.3.ffn.shared_experts.w2 (down_proj, 4096x2048). Confirms
# the w1 pad8 result (lossless AND faster than per-row) is not single-tensor luck.
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/repos/exo/mlx-lm"))

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"
OUT = Path.home() / "repos/exo/tmp/p05-shared-batching-20260830"

SHARD = "model-00005-of-00048.safetensors"
WKEY = "layers.3.ffn.shared_experts.w2.weight"
SKEY = "layers.3.ffn.shared_experts.w2.scale"


def load_f8(shard, key):
    with open(CKPT / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        info = hdr[key]
        o0, o1 = info["data_offsets"]
        fh.seek(n + 8 + o0)
        raw = fh.read(o1 - o0)
    if info["dtype"] in ("F8_E4M3", "F8_E8M0"):
        return np.frombuffer(raw, dtype=np.uint8).reshape(info["shape"])
    raise ValueError(info["dtype"])


def main():
    w_u8 = load_f8(SHARD, WKEY)
    s_u8 = load_f8(SHARD, SKEY)
    out_dim, in_dim = w_u8.shape  # (4096, 2048)
    print(f"w2: {w_u8.shape} scales {s_u8.shape}")
    w5 = w_u8.reshape(out_dim, in_dim // 4, 4)
    packed = np.zeros((out_dim, in_dim // 4), dtype=np.uint32)
    for j in range(4):
        packed |= w5[:, :, j].astype(np.uint32) << (8 * j)
    # scale layout: checkpoint (out/128?, groups) — same repeat pattern as w1 runs:
    # repeat 4x along cols then 128x along rows -> (out_dim, in_dim/32)
    s_rep = np.repeat(np.repeat(s_u8, 4, -1), 128, 0)
    assert s_rep.shape == (out_dim, in_dim // 32), s_rep.shape

    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.weight = mx.zeros((out_dim, in_dim))
    q = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q.weight = mx.array(packed)
    q.scales = mx.array(s_rep)
    q.eval()
    mx.eval(q.parameters())

    # per-rank out slice 1024 rows
    q_r = nn.Linear(in_dim, 1024, bias=False)
    q_r = q_r.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q_r.weight = q.weight[:1024]
    q_r.scales = q.scales[:1024]
    q_r.eval()
    mx.eval(q_r.parameters())

    rng = np.random.default_rng(20260830)
    Xs = rng.standard_normal((64, in_dim)).astype(np.float32)
    Xn = Xs / (np.sqrt((Xs ** 2).mean(-1, keepdims=True)) + 1e-6)
    x_all = mx.array(Xn).astype(mx.bfloat16)

    def bench(fn, n=30, warmup=8):
        for _ in range(warmup):
            mx.eval(fn())
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2] * 1e6

    t_m1 = bench(lambda: mx.concatenate([q_r(x_all[i:i + 1]) for i in range(4)], axis=0))
    t_m4 = bench(lambda: q_r(x_all[:4]))
    pad8 = mx.concatenate([x_all[:4]] + [mx.zeros((1, in_dim), dtype=mx.bfloat16)] * 4, axis=0)
    t_pad8 = bench(lambda: q_r(pad8))

    # numerics: M=4 rows via pad8 vs per-row ref, and vs batched m4 (divergence expected)
    ref = mx.concatenate([q_r(x_all[i:i + 1]) for i in range(4)], axis=0)
    yp = q_r(pad8)
    y4 = q_r(x_all[:4])
    mx.eval(ref, yp, y4)
    d_pad8 = (yp[:4] - ref).astype(mx.float32)
    d_m4 = (y4 - ref).astype(mx.float32)
    print(f"per-row M=1 x4:        {t_m1:8.1f} us")
    print(f"batched M=4 qmv_wide:  {t_m4:8.1f} us")
    print(f"padded M=8 qmm:        {t_pad8:8.1f} us")
    print(f"pad8 vs per-row ref:   max={float(mx.abs(d_pad8).max()):.3e} nz={int((mx.abs(d_pad8) > 0).sum())}/{d_pad8.size}")
    print(f"m4_wide vs per-row ref: max={float(mx.abs(d_m4).max()):.3e} nz={int((mx.abs(d_m4) > 0).sum())}/{d_m4.size}")

    results = {
        "host": os.uname().nodename,
        "gpu": mx.device_info()["architecture"],
        "tensor": WKEY + " (per-rank out-slice 1024)",
        "speeds_us": {"per_row_m1_x4": t_m1, "batched_m4_wide": t_m4, "pad8_qmm": t_pad8},
        "numerics": {
            "pad8_vs_per_row": {"max_abs": float(mx.abs(d_pad8).max()),
                                 "nonzero": int((mx.abs(d_pad8) > 0).sum()), "total": int(d_pad8.size)},
            "m4_wide_vs_per_row": {"max_abs": float(mx.abs(d_m4).max()),
                                    "nonzero": int((mx.abs(d_m4) > 0).sum()), "total": int(d_m4.size)},
        },
    }
    (OUT / "real_shared_pad8_speed_w2.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'real_shared_pad8_speed_w2.json'}")


if __name__ == "__main__":
    main()
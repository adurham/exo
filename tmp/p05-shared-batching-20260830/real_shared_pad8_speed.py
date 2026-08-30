# P05 Phase C follow-up: real-weight padded-qmm SPEED at M=8 (the lossless pad level).
# The numerics for padded-qmm M=8/16/32/64 on real weights are already captured
# (real_shared_padded_qmm.json: M=8 is bitwise-lossless on BOTH w1 and w2; M>=16
# diverges 1-ULP-class). What was NEVER measured is the SPEED of the M=8 pad on
# real weights — only pad16/pad32 were timed (257us vs per-row 268.5us). This
# script measures per_row_m1_x4 vs batched_m4_wide vs pad8_qmm vs pad16_qmm on
# the real layer-3 shared_experts w1 slice, same protocol as the original runs.
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
H = 4096


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
    shard = "model-00005-of-00048.safetensors"
    print("loading real layer-3 shared_experts w1 (per-rank slice, I=1024)...")
    w1_u8 = load_f8(shard, "layers.3.ffn.shared_experts.w1.weight")
    s1_u8 = load_f8(shard, "layers.3.ffn.shared_experts.w1.scale")
    out_dim, in_dim = w1_u8.shape  # (2048, 4096)
    w5 = w1_u8.reshape(out_dim, in_dim // 4, 4)
    packed = np.zeros((out_dim, in_dim // 4), dtype=np.uint32)
    for j in range(4):
        packed |= w5[:, :, j].astype(np.uint32) << (8 * j)
    s_rep = np.repeat(np.repeat(s1_u8, 4, -1), 128, 0)  # (2048, 128)
    assert s_rep.shape == (out_dim, in_dim // 32)

    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.weight = mx.zeros((out_dim, in_dim))
    q = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q.weight = mx.array(packed)
    q.scales = mx.array(s_rep)
    q.eval()
    mx.eval(q.parameters())

    # per-rank slice (out rows 0:1024), same as the original speed section
    q_r = nn.Linear(in_dim, 1024, bias=False)
    q_r = q_r.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q_r.weight = q.weight[:1024]
    q_r.scales = q.scales[:1024]
    q_r.eval()
    mx.eval(q_r.parameters())

    rng = np.random.default_rng(20260830)
    Hs = rng.standard_normal((64, H)).astype(np.float32)
    Hn = Hs / (np.sqrt((Hs ** 2).mean(-1, keepdims=True)) + 1e-6)
    x_all = mx.array(Hn).astype(mx.bfloat16)

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
    pad8 = mx.concatenate([x_all[:4]] + [mx.zeros((1, H), dtype=mx.bfloat16)] * 4, axis=0)
    t_pad8 = bench(lambda: q_r(pad8))
    pad16 = mx.concatenate([x_all[:4]] + [mx.zeros((1, H), dtype=mx.bfloat16)] * 12, axis=0)
    t_pad16 = bench(lambda: q_r(pad16))
    pad32 = mx.concatenate([x_all[:4]] + [mx.zeros((1, H), dtype=mx.bfloat16)] * 28, axis=0)
    t_pad32 = bench(lambda: q_r(pad32))

    print(f"per-row M=1 x4 (production fix): {t_m1:8.1f} us")
    print(f"batched M=4 qmv_wide (divergent): {t_m4:8.1f} us")
    print(f"padded M=8  qmm (LOSSLESS):      {t_pad8:8.1f} us")
    print(f"padded M=16 qmm (1-ULP):        {t_pad16:8.1f} us")
    print(f"padded M=32 qmm (1-ULP):        {t_pad32:8.1f} us")

    # quick numerics re-confirmation at M=8 on this slice (sanity, matches prior)
    ref = mx.concatenate([q_r(x_all[i:i + 1]) for i in range(4)], axis=0)
    yp = q_r(pad8)
    mx.eval(ref, yp)
    d = (yp[:4] - ref).astype(mx.float32)
    print(f"pad8 numerics recheck: max={float(mx.abs(d).max()):.3e} nz={int((mx.abs(d) > 0).sum())}")

    results = {
        "host": os.uname().nodename,
        "gpu": mx.device_info()["architecture"],
        "tensor": "layers.3.ffn.shared_experts.w1.weight (per-rank out-slice 1024)",
        "speeds_us": {
            "per_row_m1_x4": t_m1,
            "batched_m4_wide": t_m4,
            "pad8_qmm": t_pad8,
            "pad16_qmm": t_pad16,
            "pad32_qmm": t_pad32,
        },
        "pad8_numerics_recheck": {"max_abs": float(mx.abs(d).max()),
                                   "nonzero": int((mx.abs(d) > 0).sum())},
        "note": "pad8 is the only bitwise-lossless padded level on real weights (see real_shared_padded_qmm.json); this file adds its SPEED.",
    }
    (OUT / "real_shared_pad8_speed.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT / 'real_shared_pad8_speed.json'}")


if __name__ == "__main__":
    main()
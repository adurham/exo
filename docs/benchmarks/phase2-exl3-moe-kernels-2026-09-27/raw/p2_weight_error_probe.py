#!/usr/bin/env python3
"""Weight-level quantization error probe (cheap sanity check).

Question: is the 14% output error in the affine baseline REAL re-quantization
loss, or an artifact/bug in the baseline construction?

Measures, for a single expert's w1/w2/w3:
  rel error of affine-4bit/gs64 requantization vs the fp16 EXL3-dequantized W
  rel error of affine-3bit/gs64
  rel error of EXL3 k=3 reconstruction vs ... itself (0 by construction)
Also reports where the weight mass sits (max|w| / outlier structure) since
requantizing an already-rotated/outlier-spread matrix is the suspected cause.
"""
import json
import os
import numpy as np
import mlx.core as mx

from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.mlx.reconstruct import reconstruct_public_mlx

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = 1
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
from safetensors import safe_open

for proj in ("w1", "w2", "w3"):
    p = f"layers.{LAYER}.ffn.experts.0.{proj}."
    with safe_open(os.path.join(MODEL, idx[p + "trellis"]), framework="np") as f:
        t = f.get_tensor(p + "trellis")
        suh = f.get_tensor(p + "suh")
        svh = f.get_tensor(p + "svh")
    k = t.shape[2] * 16 // 256
    lay = EXL3Layer(key=p, in_features=len(suh), out_features=len(svh),
                    k=k, trellis=t, suh=suh, svh=svh, mul1=True)
    W = np.array(reconstruct_public_mlx(lay), dtype=np.float32)   # (in, out)
    Wt = np.array(W.T, dtype=np.float16)                          # (out, in)
    wmax = float(np.abs(W).max())
    rms = float(np.sqrt((W ** 2).mean()))
    # kurtosis-ish outlier indicator
    p999 = float(np.percentile(np.abs(W), 99.9))

    line = (f"{proj}: shape={W.shape} max|w|={wmax:.3f} rms={rms:.4f} "
            f"p99.9={p999:.4f} max/rms={wmax / rms:.1f}")
    for bits in (4, 3, 8):
        for gs in (64, 32):
            q = mx.quantize(mx.array(Wt), group_size=gs, bits=bits, mode="affine")
            deq = mx.dequantize(*q, group_size=gs, bits=bits, mode="affine")
            mx.eval(deq)
            d = np.array(deq, dtype=np.float32)
            err = float(np.sqrt(((d - Wt.astype(np.float32)) ** 2).mean())
                        / (np.sqrt((Wt.astype(np.float32) ** 2).mean()) + 1e-12))
            line += f" | aff{bits}/gs{gs}={err * 100:.3f}%"
    print(line, flush=True)

# Also: what is the weight-level error of the ORIGINAL EXL3 quantization?
# (EXL3 vs the fp8 source is long gone, but we can at least show that the
#  fp16-reference round trip is exact.)
print("\ninterpretation:", flush=True)
print("  If aff4/gs64 weight error is ~1-4%, then a 14% MoE OUTPUT error means", flush=True)
print("  amplification through silu + summation of 6 experts -- investigate.", flush=True)
print("  If aff4/gs64 weight error is itself ~10%+, the 14% is genuine loss from", flush=True)
print("  requantizing EXL3-derived weights, and NOT indicative of Plan B quality", flush=True)
print("  (Plan B would quantize from the fp8 source, not from EXL3 weights).", flush=True)

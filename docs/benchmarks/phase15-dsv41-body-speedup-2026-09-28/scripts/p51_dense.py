#!/usr/bin/env python3
"""p51 -- one layer's dense projections at M=1: EXL3Linear vs MLX affine qmv.

Reconstructs each EXL3 dense group of a layer to fp16 (the exact weights the
EXL3 kernel computes with), then times the whole set at M=1 as EXL3Linear,
as affine 8-bit / 6-bit / 4-bit (group 64), and as fp16. Error is reported
vs the fp16 reconstruction (i.e. extra error ON TOP of EXL3's own).
"""
import os, sys, time
import numpy as np
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer
from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx

ck = Exl3Checkpoint(HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
L = int(os.environ.get("P51_LAYER", "2"))
pre = f"layers.{L}."
groups = sorted(k[:-8] for k in ck.index if k.startswith(pre) and k.endswith(".trellis")
                and ".experts." not in k)
lins, Ws = [], []
for g in groups:
    lay = load_dense_layer(ck, g)
    lins.append(EXL3Linear(lay).release_source())
    Ws.append(mx.contiguous(reconstruct_public_mlx(load_dense_layer(ck, g)).T))   # [out, in] fp16
mx.eval(Ws)
n_params = sum(w.size for w in Ws)
exl3_bytes = sum(ck.header(g + ".trellis")["shape"][0] * ck.header(g + ".trellis")["shape"][1]
                 * ck.header(g + ".trellis")["shape"][2] * 2 for g in groups)
print(f"layer {L}: {len(groups)} groups, {n_params/1e6:.1f}M params, EXL3 trellis {exl3_bytes/1e6:.1f} MB", flush=True)
xs = [mx.random.normal((1, w.shape[1])).astype(mx.float16) for w in Ws]
mx.eval(xs)
ref = [x @ w.T for x, w in zip(xs, Ws)]
mx.eval(ref)


def bench(name, fns, nbytes):
    for _ in range(5):
        mx.eval([f(x) for f, x in zip(fns, xs)])
    ts = []
    for _ in range(30):
        s = time.perf_counter(); mx.eval([f(x) for f, x in zip(fns, xs)]); ts.append(time.perf_counter() - s)
    ys = [f(x).astype(mx.float32) for f, x in zip(fns, xs)]
    cos = min(((y * r.astype(mx.float32)).sum() / (mx.linalg.norm(y) * mx.linalg.norm(r.astype(mx.float32)))).item()
              for y, r in zip(ys, ref))
    med = np.median(ts) * 1e3
    print(f"{name:22s} {med:7.3f} ms/layer  x40={med*40:6.1f} ms  {nbytes/1e6:7.1f} MB/layer "
          f"({nbytes*40/1e9:5.2f} GB/40L)  eff {nbytes/(med/1e3)/1e9:6.1f} GB/s  min_cos {cos:.6f}", flush=True)


bench("EXL3Linear", lins, exl3_bytes)
bench("fp16 matmul", [lambda x, w=w: x @ w.T for w in Ws], n_params * 2)
for bits in (8, 6, 4):
    qs = [mx.quantize(w, group_size=64, bits=bits) for w in Ws]
    mx.eval(qs)
    nb = sum(q[0].nbytes + q[1].nbytes + q[2].nbytes for q in qs)
    bench(f"affine {bits}-bit g64",
          [lambda x, q=q, b=bits: mx.quantized_matmul(x, q[0], q[1], q[2], transpose=True, group_size=64, bits=b)
           for q in qs], nb)

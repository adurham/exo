"""Forensics 11: isolate the fused-kernel vs eager gap at D=512.

At a small dense-mask shape, compare the fused output against a numpy-fp64
reference. If the fused kernel itself is the one that deviates by ~0.3% p50
from exact, the eager-vs-kernel gap is entirely the kernel's internal
precision, and no call restructuring can do better than the kernel's own
error bar.
"""
import mlx.core as mx
import numpy as np

mx.random.seed(0)
SCALE = 512 ** -0.5
q = mx.random.normal((1, 4, 128, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 3894, 512)).astype(mx.bfloat16)
DENSE = mx.ones((1, 1, 128, 3894), dtype=mx.bool_)

o_fast = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=SCALE, mask=DENSE)

# eager fp32 in MLX
s32 = (q.astype(mx.float32) @ mx.transpose(kv.astype(mx.float32), (0, 1, 3, 2))) * SCALE
smax = mx.max(s32, axis=-1, keepdims=True)
p32 = mx.exp(s32 - smax)
Z32 = mx.sum(p32, axis=-1, keepdims=True)
o_eager = (p32 @ kv.astype(mx.float32)) / Z32

# numpy fp64 exact reference on the same bf16 inputs
qn = np.array(mx.transpose(q[0], (1, 0, 2)).astype(mx.float32), dtype=np.float64)
kn = np.array(kv[0, 0].astype(mx.float32), dtype=np.float64)
vn = np.array(kv[0, 0].astype(mx.float32), dtype=np.float64)
sn = (qn @ kn.T) * SCALE
pn = np.exp(sn - sn.max(-1, keepdims=True))
o_np = (pn @ vn) / pn.sum(-1, keepdims=True)
o_np4 = mx.array(o_ref4 := o_np[None].transpose(0, 2, 1, 3))


def pct(a, b, tag):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf)
    rel = d / (mx.abs(bf) + 1e-6)
    sr = mx.sort(mx.reshape(rel, (-1,)))
    n = sr.shape[0]
    rms = mx.sqrt(mx.mean(d * d))
    orms = mx.sqrt(mx.mean(af * af))
    print("%-42s p50 %.5f p90 %.4f max %.3f rms/out %.5f" % (
        tag, float(sr[int(n * 0.5)].item()), float(sr[int(n * 0.9)].item()),
        float(sr[n - 1].item()), float((rms / orms).item())))


pct(o_np4, o_eager, "np64 vs eager-fp32 (expect ~1e-3 bf16 out)")
pct(o_np4, o_fast, "np64 vs fast kernel")
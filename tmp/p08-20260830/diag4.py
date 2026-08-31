"""Forensics: is explicit-vs-fast error an fp32-precision difference?

Production shape, DENSE mask (isolates formula from mask handling).
  E1: fp32 QK^T + fp32 softmax + fp32 PV
  E2: bf16 QK^T (as the harness writes it) + fp32 softmax + fp32 PV
Both compared against the fast kernel output.
"""
import mlx.core as mx

mx.random.seed(0)
SCALE = 512 ** -0.5
q = mx.random.normal((1, 32, 1024, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 3894, 512)).astype(mx.bfloat16)
DENSE = mx.ones((1, 1, 1024, 3894), dtype=mx.bool_)

o_fast = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=SCALE, mask=DENSE)

# E1: everything fp32
s32 = (q.astype(mx.float32) @ mx.transpose(kv.astype(mx.float32), (0, 1, 3, 2))) * SCALE
mx32 = mx.max(s32, axis=-1, keepdims=True)
p32 = mx.exp(s32 - mx32)
p32n = p32 / mx.sum(p32, axis=-1, keepdims=True)
o_E1 = p32n @ kv.astype(mx.float32)

# E2: bf16 QK^T, then fp32 softmax and fp32 PV
s16 = (q @ kv.transpose(0, 1, 3, 2)) * SCALE
s16f = s16.astype(mx.float32)
mx16 = mx.max(s16f, axis=-1, keepdims=True)
p16 = mx.exp(s16f - mx16)
p16n = p16 / mx.sum(p16, axis=-1, keepdims=True)
o_E2 = p16n @ kv.astype(mx.float32)


def relm(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    r = mx.abs(af - bf) / (mx.abs(bf) + 1e-6)
    return float(mx.mean(r).item()), float(mx.max(r).item())


print("E1 (fp32 QK)   vs fast: mean %.8f max %.6f" % relm(o_E1, o_fast))
print("E2 (bf16 QK)   vs fast: mean %.8f max %.6f" % relm(o_E2, o_fast))
print("E1 vs E2             : mean %.8f max %.6f" % relm(o_E1, o_E2))
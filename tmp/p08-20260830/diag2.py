"""Forensics: why does explicit softmax attention differ from mx.fast.sdpa?

Variants at a small production-like shape (dense mask, no sinks):
  D: fully-fp32 explicit softmax attention
  E: probabilities cast to bf16 before the PV matmul
  F: bf16 QK^T + bf16 PV (naive eager pipeline)
Compared against the fast kernel and an fp64 numpy reference.
"""
import mlx.core as mx
import numpy as np

mx.random.seed(0)
q = mx.random.normal((1, 4, 64, 128)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 512, 128)).astype(mx.bfloat16)
SCALE = 128 ** -0.5
DENSE = mx.ones((1, 1, 64, 512), dtype=mx.bool_)

o_fast = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=SCALE, mask=DENSE)

v32 = kv.astype(mx.float32)
s32 = (q.astype(mx.float32) @ kv.astype(mx.float32).transpose(0, 1, 3, 2)) * SCALE
smax32 = mx.max(s32, axis=-1, keepdims=True)
p32 = mx.exp(s32 - smax32)
Z32 = mx.sum(p32, axis=-1, keepdims=True)
o_D = (p32 @ v32) / Z32                    # fp32 throughout
pE = (p32 / Z32).astype(mx.bfloat16)       # probabilities in bf16
o_E = (pE @ kv)                            # bf16 PV on bf16 V

# F: bf16 scores end to end
s16 = (q @ kv.transpose(0, 1, 3, 2)) * SCALE
s16f = s16.astype(mx.float32)
smax16 = mx.max(s16f, axis=-1, keepdims=True)
p16 = mx.exp(s16f - smax16)
p16n = (p16 / mx.sum(p16, axis=-1, keepdims=True)).astype(mx.bfloat16)
o_F = (p16n @ kv).astype(mx.float32)

# numpy fp64 ground truth (heads loop, small)
q_np = np.array(mx.transpose(q[0], (1, 0, 2)).astype(mx.float32))    # (H,64,D)
k_np = np.array(kv[0, 0].astype(mx.float32))                         # (512,D)
v_np = np.array(kv[0, 0].astype(mx.float32))
o_ref = np.zeros_like(q_np)
for h in range(q_np.shape[0]):
    sc = (q_np[h] @ k_np.T) * SCALE
    ps = np.exp(sc - sc.max(-1, keepdims=True))
    o_ref[h] = (ps @ v_np) / ps.sum(-1, keepdims=True)
o_ref4 = mx.array(o_ref)[None].transpose(0, 2, 1, 3)                 # (1,4,64,D)


def relm(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    r = mx.abs(af - bf) / (mx.abs(bf) + 1e-6)
    return float(mx.mean(r).item()), float(mx.max(r).item())


print("D fp32-explicit   vs fast : mean %.8f max %.6f" % relm(o_D, o_fast))
print("D fp32-explicit   vs np64 : mean %.8f max %.6f" % relm(o_D, o_ref4))
print("fast kernel       vs np64 : mean %.8f max %.6f" % relm(o_fast, o_ref4))
print("E bf16-probs PV   vs np64 : mean %.8f max %.6f" % relm(o_E, o_ref4))
print("F bf16-everything vs np64 : mean %.8f max %.6f" % relm(o_F, o_ref4))
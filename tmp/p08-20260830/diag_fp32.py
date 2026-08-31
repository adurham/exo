import mlx.core as mx

mx.random.seed(0)
q = mx.random.normal((1, 32, 128, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 3894, 512)).astype(mx.bfloat16)
SCALE = 512 ** -0.5
DENSE = mx.ones((1, 1, 1, 3894), dtype=mx.bool_)

# fast kernel WITHOUT sinks (dense mask = no masked-out entries at all)
o_fast = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=SCALE, mask=DENSE)

v32 = kv.astype(mx.float32)
s32 = (q.astype(mx.float32) @ kv.astype(mx.float32).transpose(0, 1, 3, 2)) * SCALE
smax = mx.max(s32, axis=-1, keepdims=True)
p32 = mx.exp(s32 - smax)
Z32 = mx.sum(p32, axis=-1, keepdims=True)

# D: fully fp32 explicit softmax attention
o_D = (p32 @ v32) / Z32

# E: probabilities cast to bf16 before PV (the classic error source)
pE = p32 / Z32
pE16 = pE.astype(mx.bfloat16)
o_E = (pE16 @ v32).astype(mx.bfloat16)
o_E = (mx.transpose(pE16, (0, 1, 2, 3)) @ v32)


def relm(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf) / (bf * bf + 1.0)


print("checkpoint")
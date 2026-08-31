import mlx.core as mx

mx.random.seed(0)
q = mx.random.normal((1, 32, 128, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 3894, 512)).astype(mx.bfloat16)
SCALE = 512 ** -0.5
DENSE = mx.ones((1, 1, 1, 3894), dtype=mx.bool_)

# fast kernel WITHOUT sinks
o_fast = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=DENSE)

s32 = (q.astype(mx.float32) @ kv.astype(mx.float32).transpose(0, 1, 3, 2)) * SCALE
smax = mx.max(s32, axis=-1, keepdims=True)
p = mx.exp(s32 - smax)
v32 = kv.astype(mx.float32)

# D: fp32 explicit, no sink, softmax fp32, PV fp32, output stays fp32
pD = p / mx.sum(p, axis=-1, keepdims=True)
o_D = pD @ v32  # fp32 output, no bf16 cast

# E: bf16 P (as production harness does) but fp32 V
pE = p.astype(mx.bfloat16)
o_E = (pE @ v32).astype(mx.bfloat16)

# F: everything bf16 exactly like the harness sink-check block
s16 = ((q @ kv.transpose(0, 1, 3, 2)) * SCALE).astype(mx.float32)
s16m = mx.where(DENSE, s16, mx.full((1, 1, 1, 3894), -1e30, mx.float32))
smax16 = mx.max(s16m, axis=-1, keepdims=True)
p16 = mx.exp(s16m - smax16 := mx.max(s16m := s16m, -1, keepdims=True) if False else mx.max(s16m, -1, keepdims=True))
"""Check the exact sink semantics of mx.fast.sdpa: does sinks=0 add +1 to Z?

Small shape, DENSE mask (mask content can't interact with sinks here).
Compare fast output (with/without sinks) against explicit variants:
  A: Z += exp(sinks - smax) i.e. +1 when smax keys-only and sinks=0
  A2: Z += exp(sinks - smax_including_sink) = 1.0 exactly here
  B: no sink at all
If fast-with-sinks ~= fast-without-sinks and both ~= B(no sink), the sink
adds exp(-m)<1 mass — NOT +1 — and my harness formula was wrong.
"""
import mlx.core as mx

mx.random.seed(0)
q = mx.random.normal((1, 4, 64, 128)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 512, 128)).astype(mx.bfloat16)
SCALE = 128 ** -0.5
DENSE = mx.ones((1, 1, 64, 512), dtype=mx.bool_)
s0 = mx.zeros((4,), dtype=mx.bfloat16)
s5 = mx.full((4,), 5.0, dtype=mx.bfloat16)

o_fast_s0 = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=DENSE, sinks=s0)
o_fast_None = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=SCALE, mask=DENSE)
o_fast_s5 = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=DENSE, sinks=s5)

v32 = kv.astype(mx.float32)
s32 = (q.astype(mx.float32) @ v32.transpose(0, 1, 3, 2)) * SCALE
smax = mx.max(s32, axis=-1, keepdims=True)
p = mx.exp(s32 - smax)
ZK = mx.sum(p, axis=-1, keepdims=True)

# A: +1 in Z (previous harness assumption)
o_A = ((p @ v32) / (ZK + 1.0)).astype(mx.bfloat16)
# B: no sink
o_B = ((p @ v32) / ZK).astype(mx.bfloat16)
# A5: sink value 5: Z += exp(5 - smax)
o_A5 = ((p @ v32) / (ZK + mx.exp(5.0 - smax)).astype(mx.float32)).astype(mx.bfloat16)


def relm(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    r = mx.abs(af - bf) / (mx.abs(bf) + 1e-6)
    return float(mx.mean(r).item()), float(mx.max(r).item())


print("fast(sinks=0) vs fast(None)     : mean %.8f max %.6f" % relm(o_fast_s0, o_fast_None))
print("fast(sinks=0) vs A(+1 in Z)     : mean %.8f max %.6f" % relm(o_fast_s0, o_A))
print("fast(sinks=0) vs B(no sink)     : mean %.8f max %.6f" % relm(o_fast_s0, o_B))
print("fast(sinks=5) vs A5(exp(5-m))   : mean %.8f max %.6f" % relm(o_fast_s5, o_A5))
print("fast(sinks=5) vs B(no sink)     : mean %.8f max %.6f" % relm(o_fast_s5, o_B))
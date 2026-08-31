"""Forensics 10: CONFIRMATION -- round fp32 scores to bf16 before softmax.

If mx.fast.sdpa materializes the score tile in bf16, then
softmax(bf16(fp32_scores)) should reproduce the fused output closely.
Production shape, windowed mask, sinks=0.
"""
import mlx.core as mx

mx.random.seed(0)
SCALE = 512 ** -0.5
q = mx.random.normal((1, 32, 1024, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 3894, 512)).astype(mx.bfloat16)
sinks = mx.zeros((32,), dtype=mx.bfloat16)

q_pos = mx.arange(2048) + 127
k_idx = mx.arange(2175)
m_local = (q_pos[:, None] >= k_idx[None, :]) & (q_pos[:, None] < k_idx[None, :] + 128)
pool_idx = mx.arange(1719)[None, :]
query_pos = 218880 + mx.arange(1, 2049)[:, None]
m_pool = pool_idx < (query_pos // 128)
mask = mx.concatenate([m_local[None, None], m_pool[None, None]], axis=-1)
mask = mask[..., 0:1024, :]
mx.eval(mask)

o_fast = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=mask, sinks=sinks)

v32 = kv.astype(mx.float32)
s_fp32 = (q.astype(mx.float32) @ mx.transpose(
    kv.astype(mx.float32), (0, 1, 3, 2))) * SCALE

NEG_FILL = mx.full((1, 1, 1, 3894), -1e30, mx.float32)


def attend_from(score_f32):
    sm = mx.where(mask, score_f32, mx.full((1, 1, 1, 3894), -1e30, mx.float32))
    smax = mx.max(sm, axis=-1, keepdims=True)
    p = mx.exp(sm - smax)
    Z = mx.sum(p, axis=-1, keepdims=True)
    return (p @ v32) / Z


# H1: pure fp32 scores
o_H1 = attend_from(s_fp32)

# H2: bf16-ROUNDED scores (kernel-emulation)
s16 = s_fp32.astype(mx.bfloat16).astype(mx.float32)
o_H2 = attend_from(s16)

# H3: fp32 scores but probabilities cast to bf16 before PV
s32m = mx.where(mask, s_fp32, NEG_FILL)
smax = mx.max(s32m, axis=-1, keepdims=True)
p32 = mx.exp(s32m - smax)
p16 = (p32 / mx.sum(p32, axis=-1, keepdims=True)).astype(mx.bfloat16)
o_H3 = (p16 @ kv).astype(mx.float32)


def pct(a, b, tag):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf)
    rel = d / (mx.abs(bf) + 1e-6)
    sr = mx.sort(mx.reshape(rel, (-1,)))
    n = sr.shape[0]
    rms = mx.sqrt(mx.mean(d * d))
    orms = mx.sqrt(mx.mean(af * af))
    print("%-42s p50 %.5f p90 %.4f p99.9 %.3f max %.2f rms/out %.5f" % (
        tag, float(sr[int(n * 0.5)].item()), float(sr[int(n * 0.9)].item()),
        float(sr[int(n * 0.999)].item()), float(sr[n - 1].item()),
        float((rms / orms).item())))


pct(o_H1, o_fast, "H1 pure fp32 scores vs fast")
pct(o_H2, o_fast, "H2 bf16-rounded scores vs fast")
pct(o_H3, o_fast, "H3 bf16 probs+PV vs fast")
pct(o_H2, o_H1, "H2 vs H1 (score rounding effect)")
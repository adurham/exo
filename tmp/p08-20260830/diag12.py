"""C'/D' exact numerics test: merge partial SDPA outputs via LSE.

Split 3894 keys into pooled [0:1719) and local [1719:3894). Each half gets a
fused mx.fast.sdpa call (kernel precision for O_i) plus an explicit LSE_i
computed from the kernel-precision scores (bf16 score representation).
Merge with exact LSE weights. Compare to the fused single-call output.
"""
import mlx.core as mx

mx.random.seed(0)
SCALE = 512 ** -0.5
L_BAND = 1024
LOCAL_KV = 2175
POOL = 1719
CATTN = LOCAL_KV + POOL
q = mx.random.normal((1, 32, L_BAND, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, CATTN, 512)).astype(mx.bfloat16)
sinks = mx.zeros((32,), dtype=mx.bfloat16)

q_pos = mx.arange(2048) + 127
k_idx = mx.arange(LOCAL_KV)
m_local_full = (q_pos[:, None] >= k_idx[None, :]) & (q_pos[:, None] < k_idx[None, :] + 128)
pool_idx = mx.arange(POOL)[None, :]
query_pos = 218880 + mx.arange(1, 2049)[:, None]
m_pool = pool_idx < (query_pos // 128)
m_local = m_local_full
mask_full = mx.concatenate([m_local[None, None], m_pool[None, None]], axis=-1)
mask_band = mask_full[..., 0:L_BAND, :]
mx.eval(mask_band)

o_full = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=mask_band, sinks=sinks)


def merged_two_way():
    o1 = mx.fast.scaled_dot_product_attention(
        q, kv[..., :POOL, :], kv[..., :POOL, :],
        scale=SCALE, mask=mask_band[..., :POOL], sinks=sinks)
    o2 = mx.fast.scaled_dot_product_attention(
        q, kv[..., POOL:, :], kv[..., POOL:, :],
        scale=SCALE, mask=mask_band[..., POOL:], sinks=sinks)
    return o1, o2


o1, o2 = merged_two_way()

# LSE for each half, from bf16-rounded scores (kernel-representable precision)
NEG = -1e30
s16 = ((q @ kv.transpose(0, 1, 3, 2)) * SCALE).astype(mx.bfloat16).astype(mx.float32)


def lse_of(mask_part, key_lo, key_hi):
    sp = s16[..., key_lo:key_hi]
    NEG_P = mx.full(sp.shape, NEG, mx.float32)
    sm = mx.where(mask_part, sp, NEG_P)
    s_max = mx.max(sm, axis=-1, keepdims=True)
    p = mx.exp(sm - s_max)
    return mx.log(mx.sum(p, axis=-1, keepdims=True)) + s_max


lse1 = lse_of(mask_band[..., :POOL], 0, POOL)
lse2 = lse_of(mask_band[..., POOL:], POOL, CATTN)
lmax = mx.maximum(lse1, lse2)
w1 = mx.exp(lse1 - lmax)
w2 = mx.exp(lse2 - lmax)
o_merged = ((o1.astype(mx.float32) * w1 + o2.astype(mx.float32) * w2)).astype(mx.bfloat16)


def pct(a, b, tag):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf)
    rel = d / (mx.abs(bf) + 1e-6)
    sr = mx.sort(mx.reshape(rel, (-1,)))
    n = sr.shape[0]
    rms = mx.sqrt(mx.mean(d * d))
    orms = mx.sqrt(mx.mean(af * af))
    print("%-44s p50 %.5f p90 %.4f p99.9 %.3f max %.2f rms/out %.5f" % (
        tag, float(sr[int(n * 0.5)].item()), float(sr[int(n * 0.9)].item()),
        float(sr[int(n * 0.999)].item()), float(sr[n - 1].item()),
        float((rms / orms).item())))


pct(o_merged, o_full, "merged-2way (SDPA partials, LSE merge)")
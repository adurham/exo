"""Forensics 3: at FULL production shape, is the mean-rel-err gap a metric
artifact or a real numerics difference?

E2m mirrors the harness path exactly (bf16 QK^T via matmul, fp32 softmax with
NEG=-1e30 mask fill, fp32 PV) vs the fast kernel with the REAL causal mask.
Reports the rel-err percentiles and RMS-normalized error.
"""
import mlx.core as mx

mx.random.seed(0)
SCALE = 512 ** -0.5
L_BAND, LOCAL_KV, POOL, CATTN_KV = 1024, 2175, 1719, 3894
SLIDING, N_CHUNK_L = 128, 2048

q = mx.random.normal((1, 32, L_BAND, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, CATTN_KV, 512)).astype(mx.bfloat16)
sinks = mx.zeros((32,), dtype=mx.bfloat16)

q_pos = mx.arange(N_CHUNK_L) + 127
k_idx = mx.arange(LOCAL_KV)
m_local = (q_pos[:, None] >= k_idx[None, :]) & (
    q_pos[:, None] < k_idx[None, :] + 128)
pool_idx = mx.arange(POOL)[None, :]
query_pos = 218880 + mx.arange(1, N_CHUNK_L + 1)[:, None]
m_pool = pool_idx < (query_pos // 128)
mask = mx.concatenate([m_local[None, None], m_pool[None, None]], axis=-1)
mask_band = mask[..., 0:L_BAND, :]
mx.eval(mask_band)

o_fast = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=mask_band, sinks=sinks)

NEG = -1e30
s16 = (q @ kv.transpose(0, 1, 3, 2)) * SCALE
s32m = mx.where(mask_band, s16.astype(mx.float32),
                mx.full((1, 32, L_BAND, CATTN_KV), NEG, mx.float32))
smax = mx.max(s32m, axis=-1, keepdims=True)
p32 = mx.exp(s32m - smax)
Z32 = mx.sum(p32, axis=-1, keepdims=True)
o_E2 = (p32 @ kv.astype(mx.float32)) / Z32


def stats(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf)
    rel = d / (mx.abs(bf) + 1e-6)
    sr = mx.sort(mx.reshape(rel, (-1,)))
    n = sr.shape[0]
    rms = mx.sqrt(mx.mean(d * d))
    orms = mx.sqrt(mx.mean(af * af))
    print("p50 %.6f p90 %.6f p99 %.6f p99.9 %.6f max %.4f rms/out %.6f" % (
        float(sr[int(n * 0.5)].item()), float(sr[int(n * 0.9)].item()),
        float(sr[int(n * 0.99)].item()), float(sr[int(n * 0.999)].item()),
        float(sr[n - 1].item()), float((rms / orms).item())))


print("E2m (harness path) vs fast, rel-err percentiles:")
stats(o_E2, o_fast)
"""Forensics 8, clean: does sinks=zeros change the fast SDPA output at the
production shape (vs sinks=None)? And does the explicit masked-softmax with a
NEG fill reproduce it?"""
import mlx.core as mx

mx.random.seed(0)
SCALE = 512 ** -0.5
N_HEADS, HEAD_DIM, L_BAND = 32, 512, 1024
LOCAL_KV, POOL_128, CATTN_KV = 2175, 1719, 3894
SLIDING, N_CHUNK_L, OFF_LAST = 128, 2048, 218880

q = mx.random.normal((1, N_HEADS, L_BAND, HEAD_DIM)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, CATTN_KV, HEAD_DIM)).astype(mx.bfloat16)
sinks = mx.zeros((N_HEADS,), dtype=mx.bfloat16)

q_pos = mx.arange(N_CHUNK_L) + 127
k_idx = mx.arange(LOCAL_KV)
m_local = (q_pos[:, None] >= k_idx[None, :]) & (q_pos[:, None] < k_idx[None, :] + SLIDING)
pool_idx = mx.arange(POOL_128)[None, :]
query_pos = OFF_LAST + mx.arange(1, N_CHUNK_L + 1)[:, None]
m_pool = pool_idx < (query_pos // 128)
mask_band = mx.concatenate([m_local[None, None], m_pool[None, None]], axis=-1)
mask_band = mask_band[..., 0:L_BAND, :]
mx.eval(mask_band)

o_sinks0 = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=mask_band, sinks=sinks)
o_none = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=mask_band)

s16 = (q @ kv.transpose(0, 1, 3, 2)) * SCALE
NEG_FULL = mx.full((1, N_HEADS, L_BAND, CATTN_KV), -1e30, mx.float32)
s32m = mx.where(mask_band, s16.astype(mx.float32), NEG_FULL)
smax = mx.max(s32m, axis=-1, keepdims=True)
p32 = mx.exp(s32m - smax)
Z32 = mx.sum(p32, axis=-1, keepdims=True)
o_explicit = (p32 @ kv.astype(mx.float32)) / Z32


def pct(a, b, tag):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf)
    rel = d / (mx.abs(bf) + 1e-6)
    sr = mx.sort(mx.reshape(rel, (-1,)))
    n = sr.shape[0]
    print("%-40s p50 %.5f p90 %.5f p99.9 %.4f max %.2f" % (
        tag, float(sr[int(n * 0.5)].item()), float(sr[int(n * 0.9)].item()),
        float(sr[int(n * 0.999)].item()), float(sr[n - 1].item())))


pct(o_explicit_vs_fast := o_explicit, o_sinks0, "explicit(keys-only Z) vs fast")
pct(o_sinks0, o_none, "fast(sinks=0) vs fast(None)")
pct(o_explicit, o_none, "explicit(keys-only Z) vs fast(None)")
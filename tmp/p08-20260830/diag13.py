"""Forensics 13: WHY is the 2-way merged output 7% off when each piece is
computed by the same fused kernel? Two suspects:
  S1: the LSE is wrong (bf16-rounded scores don't match what the kernel
      actually uses), so merge weights are wrong by ~1/lse error.
  S2: the kernel's own per-call rounding differs enough between call shapes
      (3894 keys vs 1719/2175 keys) that merged-vs-full differs by more than
      kernel-vs-exact.
Test S1 first: compare explicit-LSE(z) against the true Z implied by the
fused output. Derive: for each head/row, o_fused * Z_true = sum_j p_j v_j.
With o_i from a half-call, o_half_i * Z_half_i = sum_j∈half p_j v_j.
Check consistency: does (o1*Z1 + o2*Z2) /(Z1+Z2) == o_full when
Z_half computed explicitly in fp32 with bf16-rounded scores?
The clean check: compare explicit Z (bf16 scores) against a probe Z extracted
via a trick: run the fused kernel with V = ones; then o = sum_j p_j * 1 / Z =
1.0 exactly IF sinks add exp(-m) mass... no, o = sum_j p_j / Z which is 1 for
any Z. Not usable. Instead use V = arange marker: o_j = sum p_j v_j / Z, and
with v_j = j (all same value c in a column), o = c * sum p / Z = c. Also 1.
Hmm: make V = ones * c for column 0 and -c elsewhere: o = c*(Z_v/Z - ...) no.

Better S1 test: compute merged output O_merge using LSE from fp32 scores vs
from bf16 scores. If fp32-LSE merge is much closer, the kernel's LSE differs
from my bf16-scores LSE (i.e. kernel uses higher precision scores).
"""
import mlx.core as mx

mx.random.seed(0)
SCALE = 512 ** -0.5
L_BAND, LOCAL_KV, POOL, CATTN = 1024, 2175, 1719, 3894
q = mx.random.normal((1, 4, 128, 512)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, CATTN, 512)).astype(mx.bfloat16)
sinks = mx.zeros((4,), dtype=mx.bfloat16)

q_pos = mx.arange(2048) + 127
k_idx = mx.arange(LOCAL_KV)
m_local = (q_pos[:, None] >= k_idx[None, :]) & (q_pos[:, None] < k_idx[None, :] + 128)
pool_idx = mx.arange(POOL)[None, :]
query_pos = 218880 + mx.arange(1, 2049)[:, None]
m_pool = pool_idx < (query_pos // 128)
mask_full = mx.concatenate([m_local[None, None], m_pool[None, None]], axis=-1)
mask_band = mask_full[..., 0:128, :]
mx.eval(mask_band)

o_full = mx.fast.scaled_dot_product_attention(
    q, kv, kv, scale=SCALE, mask=mask_band, sinks=sinks)

mask_p = mask_band[..., :POOL]
mask_l = mask_band[..., POOL:]
kv_p = kv[:, :, :POOL, :]
kv_l = kv[:, :, POOL:, :]
mx.eval(mask_p, mask_l, kv_p, kv_l)
o1 = mx.fast.scaled_dot_product_attention(
    q, kv_p, kv_p, scale=SCALE, mask=mask_p, sinks=sinks)
o2 = mx.fast.scaled_dot_product_attention(
    q, kv_l, kv_l, scale=SCALE, mask=mask_l, sinks=sinks)

# LSE variants
s32 = (q.astype(mx.float32) @ mx.transpose(kv.astype(mx.float32), (0, 1, 3, 2))) * SCALE
s16 = ((q @ kv.transpose(0, 1, 3, 2)) * SCALE).astype(mx.bfloat16).astype(mx.float32)


def lse_from(scores, mask_part, lo, hi):
    sp = scores[:, :, :, lo:hi]          # scores are (1,H,L,KV); slice KEYS
    NEG_P = mx.full(sp.shape, -1e30, mx.float32)
    sm = mx.where(mask_part, sp, NEG_P)
    smax = mx.max(sm, axis=-1, keepdims=True)
    p = mx.exp(sm - smax)
    return mx.log(mx.sum(p, axis=-1, keepdims=True)) + smax


def merge(o1, o2, l1, l2):
    lmax = mx.maximum(l1, l2)
    w1 = mx.exp(l1 - lmax)
    w2 = mx.exp(l2 - lmax)
    return (o1.astype(mx.float32) * w1 + o2.astype(mx.float32) * w2).astype(mx.bfloat16)


def pct(a, b, tag):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    d = mx.abs(af - bf)
    rel = d / (mx.abs(bf) + 1e-6)
    sr = mx.sort(mx.reshape(rel, (-1,)))
    n = sr.shape[0]
    rms = mx.sqrt(mx.mean(d * d))
    orms = mx.sqrt(mx.mean(af * af))
    print("%-46s p50 %.5f p90 %.4f p99.9 %.3f max %.2f rms/out %.5f" % (
        tag, float(sr[int(n * 0.5)].item()), float(sr[int(n * 0.9)].item()),
        float(sr[int(n * 0.999)].item()), float(sr[n - 1].item()),
        float((rms / orms).item())))


# A: merge with fp32-score LSE
m_fp32 = merge(o1, o2, lse_from(s32, mask_p, 0, POOL),
               lse_from(s32, mask_l, POOL, CATTN))
# B: merge with bf16-score LSE
m_bf16 = merge(o1, o2, lse_from(s16, mask_p, 0, POOL),
               lse_from(s16, mask_l, POOL, CATTN))
pct(m_fp32, o_full, "merge, LSE from fp32 eager scores")
pct(m_bf16, o_full, "merge, LSE from bf16-rounded scores")
pct(o1, o_full, "pooled-half SDPA alone vs full (sanity: garbage expected)")
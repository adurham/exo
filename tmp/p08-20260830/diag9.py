"""Forensics 9 (consultant-directed isolation).

Same p50/p90/tail metric while varying: head_dim (128 vs 512), mask kind
(dense vs real production windowed+causal), sinks (on vs None).
Whichever switch moves p50 by >5x identifies the divergence source.
"""
import mlx.core as mx

mx.random.seed(0)


def build_mask(l_band, kv_len):
    N_chunk = 2048
    q_pos = mx.arange(N_chunk) + 127
    local_kv = N_chunk + 127   # ring len = 127 + 2048 for offset=127
    k_idx = mx.arange(local_kv)
    m_local = (q_pos[:, None] >= k_idx[None, :]) & (q_pos[:, None] < k_idx[None, :] + 128)
    pool_idx = mx.arange(kv_len - local_kv)[None, :]
    query_pos = 218880 + mx.arange(1, N_chunk + 1)[:, None]
    m_pool = pool_idx < (query_pos // 128)
    mask = mx.concatenate([m_local[None, None], m_pool[None, None]], axis=-1)
    return mask[..., 0:l_band, :]


def run(head_dim, mask_kind, with_sinks):
    SCALE = head_dim ** -0.5
    q = mx.random.normal((1, 32, 1024, head_dim)).astype(mx.bfloat16)
    kv = mx.random.normal((1, 1, 3894, head_dim)).astype(mx.bfloat16)
    sinks = mx.zeros((32,), dtype=mx.bfloat16)
    if mask_kind == "dense":
        mask = mx.ones((1, 1, 1024, 3894), dtype=mx.bool_)
    else:
        mask = build_mask(1024, 3894)

    args = [q, kv, kv]
    kw = {"scale": SCALE, "mask": mask}
    if with_sinks:
        kw["sinks"] = sinks
    o_fast = mx.fast.scaled_dot_product_attention(*args, **kw)

    s32 = (q.astype(mx.float32) @ mx.transpose(kv.astype(mx.float32), (0, 1, 3, 2))) * SCALE
    s32m = mx.where(mask, s32, mx.full(s32.shape, -1e30, mx.float32))
    smax = mx.max(s32m, axis=-1, keepdims=True)
    p32 = mx.exp(s32m - smax)
    o_ref = (p32 @ kv.astype(mx.float32)) / mx.sum(p32, axis=-1, keepdims=True)

    af = o_ref.astype(mx.float32)
    bf = o_fast.astype(mx.float32)
    d = mx.abs(af - bf)
    rel = d / (mx.abs(bf) + 1e-6)
    sr = mx.sort(mx.reshape(rel, (-1,)))
    n = sr.shape[0]
    rms = mx.sqrt(mx.mean(d * d))
    orms = mx.sqrt(mx.mean(af * af))
    return (float(sr[int(n * 0.5)].item()), float(sr[int(n * 0.9)].item()),
            float(sr[n - 1].item()), float((rms / orms).item()))


for hd in (128, 512):
    for mk in ("dense", "windowed"):
        for ws in (False, True):
            p50, p90, mxr, rms = run(hd, mk, ws)
            print("hd=%3d mask=%-8s sinks=%-5s : p50 %.5f p90 %.5f max %.3f rms/out %.5f"
                  % (hd, mk, ws, p50, p90, mxr, rms))
import mlx.core as mx

mask = mx.ones((1, 1, 128, 3894), dtype=mx.bool_)
m_p = mask[..., :1719]
print(m_p.shape)
kv = mx.random.normal((1, 1, 3894, 512)).astype(mx.bfloat16)
q = mx.random.normal((1, 4, 128, 512)).astype(mx.bfloat16)
o = mx.fast.scaled_dot_product_attention(
    q, kv[..., :1719], kv[..., :1719], scale=0.044, mask=m_p)
print("ok", o.shape)
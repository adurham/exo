"""Forensics 2: the mean-rel-err metric vs near-zero outputs.

Hypothesis: mean rel err over ALL elements is dominated by elements near zero
(relative error unbounded); the distributions are actually identical. Compute:
  - elementwise |a-b| / (|b|+eps) distribution (percentiles)
  - same with a scale-aware denominator: |a-b| / max(|b|, 1e-2 * global_scale)
  - max-abs-diff and RMS-diff normalized by output RMS
"""
import mlx.core as mx

mx.random.seed(0)
SCALE = 512 ** -0.5
q = mx.random.normal((1, 4, 64, 128)).astype(mx.bfloat16)
kv = mx.random.normal((1, 1, 512, 128)).astype(mx.bfloat16)
DENSE = mx.ones((1, 1, 64, 512), dtype=mx.bool_)

o_fast = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=SCALE, mask=DENSE)

s32 = (q.astype(mx.float32) @ mx.transpose(kv.astype(mx.float32), (0, 1, 3, 2))) * SCALE
mx32 = mx.max(s32, axis=-1, keepdims=True)
p32n = mx.exp(s32 - mx32) / mx.sum(mx.exp(s32 - mx32), axis=-1, keepdims=True)
o_E1 = p32n @ kv.astype(mx.float32)

a = o_E1.astype(mx.float32)
b = o_fast.astype(mx.float32)
d = mx.abs(a - b)
rel = d / (mx.abs(b) + 1e-6)
flat_rel = mx.reshape(rel, (-1,))
flat_a = mx.reshape(a, (-1,))
sorted_rel = mx.sort(flat_rel)
n = sorted_rel.shape[0]
for pct in (50, 90, 99, 99.9):
    idx = int(n * pct / 100)
    print("rel-err p%-5s: %.6f" % (pct, float(sorted_rel[idx].item())))
print("rel-err max : %.4f" % float(sorted_rel[n - 1].item()))
print("abs diff max: %.6e (out std %.3e)" % (
    float(mx.max(d).item()), float(mx.std(a).item())))
rms = mx.sqrt(mx.mean(d * d))
out_rms = mx.sqrt(mx.mean(a * a))
print("RMS diff / RMS out: %.6f" % float((rms / out_rms).item()))
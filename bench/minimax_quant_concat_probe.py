"""Safety probe: does mx.concatenate along output-dim on packed-quantised
weights produce a merged QuantizedLinear-equivalent buffer whose
``mx.quantized_matmul`` output matches three separate per-row matmuls?

Answers the "is Week 2 feasible" question before I extend the fused-QKV
installer. Tests bits ∈ {4, 5, 8}, group_size=64, mode='affine' — the
shapes the production 5-bit MiniMax-M2.7 actually uses.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_DISPATCH_COUNT", "1")

import mlx.core as mx


def _make_projections(hidden: int, q_out: int, kv_out: int, bits: int, group_size: int):
    """Pretend there are three separate Linears, quantise them, then also
    build a merged-weight variant that concatenates along output-dim."""
    wq = mx.random.normal((q_out, hidden)).astype(mx.bfloat16)
    wk = mx.random.normal((kv_out, hidden)).astype(mx.bfloat16)
    wv = mx.random.normal((kv_out, hidden)).astype(mx.bfloat16)

    qq, sq, bq = mx.quantize(wq, group_size=group_size, bits=bits, mode="affine")
    qk, sk, bk = mx.quantize(wk, group_size=group_size, bits=bits, mode="affine")
    qv, sv, bv = mx.quantize(wv, group_size=group_size, bits=bits, mode="affine")
    mx.eval(qq, sq, bq, qk, sk, bk, qv, sv, bv)

    qm = mx.concatenate([qq, qk, qv], axis=0)
    sm = mx.concatenate([sq, sk, sv], axis=0)
    bm = mx.concatenate([bq, bk, bv], axis=0)
    mx.eval(qm, sm, bm)

    return {
        "sep": (qq, sq, bq, qk, sk, bk, qv, sv, bv),
        "mrg": (qm, sm, bm),
    }


def _sep_matmul(x, qq, sq, bq, qk, sk, bk, qv, sv, bv, group_size, bits):
    kw = dict(transpose=True, group_size=group_size, bits=bits, mode="affine")
    q = mx.quantized_matmul(x, qq, scales=sq, biases=bq, **kw)
    k = mx.quantized_matmul(x, qk, scales=sk, biases=bk, **kw)
    v = mx.quantized_matmul(x, qv, scales=sv, biases=bv, **kw)
    return q, k, v


def _mrg_matmul(x, qm, sm, bm, q_out, kv_out, group_size, bits):
    qkv = mx.quantized_matmul(
        x, qm, scales=sm, biases=bm,
        transpose=True, group_size=group_size, bits=bits, mode="affine",
    )
    q, k, v = mx.split(qkv, [q_out, q_out + kv_out], axis=-1)
    return q, k, v


def probe(bits: int, group_size: int = 64):
    hidden = 3072
    q_out = 3072
    kv_out = 512

    wts = _make_projections(hidden, q_out, kv_out, bits, group_size)
    x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    mx.eval(x)

    q_s, k_s, v_s = _sep_matmul(x, *wts["sep"], group_size=group_size, bits=bits)
    q_m, k_m, v_m = _mrg_matmul(x, *wts["mrg"], q_out=q_out, kv_out=kv_out, group_size=group_size, bits=bits)
    mx.eval(q_s, k_s, v_s, q_m, k_m, v_m)

    def maxabs(a, b):
        return float(mx.max(mx.abs(a - b)).item())

    print(f"bits={bits} group_size={group_size}:")
    print(f"  Q max|sep - mrg|: {maxabs(q_s, q_m):.2e}")
    print(f"  K max|sep - mrg|: {maxabs(k_s, k_m):.2e}")
    print(f"  V max|sep - mrg|: {maxabs(v_s, v_m):.2e}")

    # Dispatch counts
    mx.metal.reset_dispatch_count()
    _ = _sep_matmul(x, *wts["sep"], group_size=group_size, bits=bits)
    mx.eval(_)
    sep_dc = mx.metal.dispatch_count()

    mx.metal.reset_dispatch_count()
    _ = _mrg_matmul(x, *wts["mrg"], q_out=q_out, kv_out=kv_out, group_size=group_size, bits=bits)
    mx.eval(_)
    mrg_dc = mx.metal.dispatch_count()
    print(f"  dispatch count: separate={sep_dc}  merged={mrg_dc}")
    print()


def main():
    for bits in (4, 5, 8):
        probe(bits)


if __name__ == "__main__":
    main()

"""Quick dispatch-count probe: does merging Wq/Wk/Wv weights into one
matmul at load time reduce dispatches vs three separate q_proj/k_proj/
v_proj calls? If yes, that's a Python-only stepping stone toward the
full fused kernel. If no, we need a custom Metal kernel from the start.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("MLX_DISPATCH_COUNT", "1")

import mlx.core as mx

HIDDEN = 3072          # per-rank
N_Q = 24
N_KV = 4
HEAD_DIM = 128
Q_OUT = N_Q * HEAD_DIM       # 3072
KV_OUT = N_KV * HEAD_DIM     # 512


def separate(x, wq, wk, wv):
    q = x @ wq.T
    k = x @ wk.T
    v = x @ wv.T
    return q, k, v


def merged(x, w_qkv):
    """Single matmul against concatenated weights."""
    qkv = x @ w_qkv.T
    q, k, v = mx.split(qkv, [Q_OUT, Q_OUT + KV_OUT], axis=-1)
    return q, k, v


def bench_fn(name, fn, args, repeats=5, warmup=3):
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)
    samples = []
    for _ in range(repeats):
        mx.metal.reset_dispatch_count()
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        dt = time.perf_counter() - t0
        samples.append((mx.metal.dispatch_count(), dt))
    dc = sum(d for d, _ in samples) / repeats
    dt = sum(t for _, t in samples) / repeats * 1000
    print(f"  {name:<24s}  {dc:4.1f} dispatches   {dt:6.3f} ms")
    return samples


def main():
    wq = mx.random.normal((Q_OUT, HIDDEN)).astype(mx.bfloat16)
    wk = mx.random.normal((KV_OUT, HIDDEN)).astype(mx.bfloat16)
    wv = mx.random.normal((KV_OUT, HIDDEN)).astype(mx.bfloat16)
    w_qkv = mx.concatenate([wq, wk, wv], axis=0)
    mx.eval(wq, wk, wv, w_qkv)

    for seq_len, label in [(1, "decode L=1"), (1024, "prefill L=1024")]:
        x = mx.random.normal((1, seq_len, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)
        print(f"\n{label}:")
        bench_fn("three separate matmuls", separate, (x, wq, wk, wv))
        bench_fn("one merged matmul+split", merged, (x, w_qkv))

    # Numerical equivalence spot-check
    x = mx.random.normal((1, 1, HIDDEN)).astype(mx.bfloat16)
    q_s, k_s, v_s = separate(x, wq, wk, wv)
    q_m, k_m, v_m = merged(x, w_qkv)
    mx.eval(q_s, k_s, v_s, q_m, k_m, v_m)
    def maxrel(a, b):
        return float(mx.max(mx.abs(a - b)).item())
    print("\nNumerical delta (should be ~0 for same math):")
    print(f"  Q: {maxrel(q_s, q_m):.2e}")
    print(f"  K: {maxrel(k_s, k_m):.2e}")
    print(f"  V: {maxrel(v_s, v_m):.2e}")


if __name__ == "__main__":
    main()

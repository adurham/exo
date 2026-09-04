#!/usr/bin/env python3
"""Instrument build_arm's real peak memory, chunk by chunk."""
import gc
import sys

import mlx.core as mx

HIDDEN = 4096
PER_RANK_INTER = 1024
E = 256


def mem():
    return (mx.get_active_memory() / 1e9, mx.get_cache_memory() / 1e9)


def quantize_chunked(out_dims, in_dims, n_experts, group_size, bits, chunk):
    mx.random.seed(0)
    scale = (1.0 / in_dims) ** 0.5
    w_parts, s_parts, b_parts = [], [], []
    has_b = None
    for start in range(0, n_experts, chunk):
        n = min(chunk, n_experts - start)
        src = mx.random.uniform(low=-scale, high=scale, shape=(n, out_dims, in_dims)).astype(mx.bfloat16)
        mx.eval(src)
        packed = mx.quantize(src, group_size=group_size, bits=bits, mode="affine")
        mx.eval(packed)
        w, s, *rest = packed
        has_b = bool(rest)
        w_parts.append(w)
        s_parts.append(s)
        if rest:
            b_parts.append(rest[0])
        del src, packed, w, s, rest
        a, c = mem()
        print(f"    chunk@{start}: active={a:.2f}GB cache={c:.2f}GB", flush=True)
    print("    concatenating...", flush=True)
    weight = mx.concatenate(w_parts, axis=0)
    scales = mx.concatenate(s_parts, axis=0)
    biases = mx.concatenate(b_parts, axis=0) if has_b else None
    mx.eval(weight, scales)
    if biases is not None:
        mx.eval(biases)
    a, c = mem()
    print(f"    after concat: active={a:.2f}GB cache={c:.2f}GB", flush=True)
    del w_parts, s_parts, b_parts
    gc.collect()
    mx.clear_cache()
    a, c = mem()
    print(f"    after free:   active={a:.2f}GB cache={c:.2f}GB", flush=True)
    return weight, scales, biases


bits = int(sys.argv[1]) if len(sys.argv) > 1 else 6
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 32
print(f"bits={bits} chunk={chunk}")
print("gate:")
g = quantize_chunked(PER_RANK_INTER, HIDDEN, E, 32, bits, chunk)
print("up:")
u = quantize_chunked(PER_RANK_INTER, HIDDEN, E, 32, bits, chunk)
print("down:")
d = quantize_chunked(HIDDEN, PER_RANK_INTER, E, 32, bits, chunk)
tot = sum(x.nbytes for arm in (g, u, d) for x in arm if x is not None)
a, c = mem()
print(f"ARM TOTAL nbytes={tot / 1e9:.2f}GB  active={a:.2f}GB cache={c:.2f}GB")

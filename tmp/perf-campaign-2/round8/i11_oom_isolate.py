#!/usr/bin/env python3
"""Isolate WHY the bench OOMs where the instrument does not.

Hypothesis: running the empty-graph baseline chain BEFORE allocating the
arm weights is what triggers the Metal "Insufficient Memory" failure,
even though mx.get_active_memory()/get_cache_memory() both report ~0
afterwards.

Arm A: quantize immediately (expected OK, matches the instrument).
Arm B: run the baseline chain first, then quantize (expected FAIL).
"""
import gc
import sys

import mlx.core as mx
import numpy as np

HIDDEN, INTER, E = 4096, 1024, 256


def quantize_chunked(out_dims, in_dims, n_experts, group_size, bits, chunk):
    mx.random.seed(0)
    scale = (1.0 / in_dims) ** 0.5
    w, s, b = [], [], []
    for start in range(0, n_experts, chunk):
        n = min(chunk, n_experts - start)
        src = mx.random.uniform(low=-scale, high=scale, shape=(n, out_dims, in_dims)).astype(mx.bfloat16)
        mx.eval(src)
        packed = mx.quantize(src, group_size=group_size, bits=bits, mode="affine")
        mx.eval(packed)
        w.append(packed[0]); s.append(packed[1])
        if len(packed) > 2:
            b.append(packed[2])
        del src, packed
    weight = mx.concatenate(w, axis=0)
    scales = mx.concatenate(s, axis=0)
    biases = mx.concatenate(b, axis=0) if b else None
    mx.eval(weight, scales)
    if biases is not None:
        mx.eval(biases)
    del w, s, b
    gc.collect(); mx.clear_cache()
    return weight, scales, biases


def baseline_chain(chain_len=300):
    x = mx.random.normal(shape=(4, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x)
    rng = np.random.default_rng(0)
    idx = [mx.array(rng.integers(0, E, size=(4, 6)).astype(np.uint32)) for _ in range(16)]
    mx.eval(idx)
    carry, last = x, None
    for i in range(chain_len):
        fake = mx.broadcast_to(mx.expand_dims(carry, -2), (4, 6, HIDDEN))
        last = fake
        carry = x + 1e-9 * mx.mean(fake, axis=-2).astype(x.dtype)
    mx.eval(last, carry)
    mx.synchronize()


arm = sys.argv[1]
print("arm:", arm)
if arm == "B":
    for _ in range(6):
        baseline_chain(300)
    gc.collect(); mx.clear_cache()
    print(f"after baseline: active={mx.get_active_memory()/1e9:.2f}GB cache={mx.get_cache_memory()/1e9:.2f}GB")
try:
    g = quantize_chunked(INTER, HIDDEN, E, 32, 6, 32)
    print(f"quantize OK: active={mx.get_active_memory()/1e9:.2f}GB")
except Exception as e:
    print(f"quantize FAILED: {type(e).__name__}: {e}")

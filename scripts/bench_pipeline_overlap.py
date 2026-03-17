#!/usr/bin/env python3
"""Test: can CPU draft and GPU verify run truly in parallel on Apple Silicon?

If total time ≈ max(cpu, gpu) → they overlap (pipeline viable)
If total time ≈ cpu + gpu → they don't overlap (pipeline useless)
"""
import ctypes
import os
import time
import threading
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
F32_LIB = os.path.join(SCRIPT_DIR, "..", "src", "exo", "worker", "engines", "mlx", "cpu_draft.dylib")

# Reuse the float32 CPU draft (16.6ms on M4 Max)
_lib = ctypes.CDLL(F32_LIB)

# Import the structs from bench_cpu_draft_c.py
class LayerWeights(ctypes.Structure):
    _fields_ = [
        ("in_norm", ctypes.c_void_p), ("post_norm", ctypes.c_void_p),
        ("q", ctypes.c_void_p), ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p), ("o", ctypes.c_void_p),
        ("gate", ctypes.c_void_p), ("up", ctypes.c_void_p),
        ("down", ctypes.c_void_p),
        ("q_norm", ctypes.c_void_p), ("k_norm", ctypes.c_void_p),
        ("n_heads", ctypes.c_int), ("n_kv", ctypes.c_int),
        ("head_dim", ctypes.c_int), ("hidden", ctypes.c_int),
        ("inter", ctypes.c_int), ("scale", ctypes.c_float),
    ]

class KVCache(ctypes.Structure):
    _fields_ = [
        ("k", ctypes.c_void_p), ("v", ctypes.c_void_p),
        ("seq_len", ctypes.c_int), ("max_seq", ctypes.c_int),
    ]

_lib.cpu_draft_forward.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(LayerWeights),
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(KVCache),
    ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
]
_lib.cpu_draft_forward.restype = None


def to_np32(arr):
    return np.array(arr.astype(mx.float32))

def ptr(arr):
    return arr.ctypes.data if arr is not None else 0


def main():
    # Load draft model weights (float32)
    print("Loading Qwen3-0.6B for CPU draft...")
    model, tok = load('mlx-community/Qwen3-0.6B-8bit')
    mx.eval(model.parameters())
    args = model.args
    inner = model.model

    # Dequantize all weights to float32
    emb = inner.embed_tokens
    embed_np = np.ascontiguousarray(to_np32(
        mx.dequantize(emb.weight, emb.scales, emb.biases, emb.group_size, emb.bits)))

    n_layers = len(inner.layers)
    hidden = args.hidden_size
    vocab = args.vocab_size
    hd = args.head_dim
    theta = args.rope_theta

    layer_arrays = []
    c_layers = (LayerWeights * n_layers)()

    for i, layer in enumerate(inner.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        arrays = {}

        def dq(proj):
            return np.ascontiguousarray(to_np32(mx.dequantize(
                proj.weight, proj.scales, getattr(proj, 'biases', None),
                proj.group_size, proj.bits)))

        arrays['in_norm'] = np.ascontiguousarray(to_np32(layer.input_layernorm.weight))
        arrays['post_norm'] = np.ascontiguousarray(to_np32(layer.post_attention_layernorm.weight))
        arrays['q'] = dq(attn.q_proj)
        arrays['k'] = dq(attn.k_proj)
        arrays['v'] = dq(attn.v_proj)
        arrays['o'] = dq(attn.o_proj)
        arrays['gate'] = dq(mlp.gate_proj)
        arrays['up'] = dq(mlp.up_proj)
        arrays['down'] = dq(mlp.down_proj)
        arrays['q_norm'] = np.ascontiguousarray(to_np32(attn.q_norm.weight)) if hasattr(attn, 'q_norm') else None
        arrays['k_norm'] = np.ascontiguousarray(to_np32(attn.k_norm.weight)) if hasattr(attn, 'k_norm') else None

        layer_arrays.append(arrays)
        c_layers[i].in_norm = ptr(arrays['in_norm'])
        c_layers[i].post_norm = ptr(arrays['post_norm'])
        c_layers[i].q = ptr(arrays['q'])
        c_layers[i].k = ptr(arrays['k'])
        c_layers[i].v = ptr(arrays['v'])
        c_layers[i].o = ptr(arrays['o'])
        c_layers[i].gate = ptr(arrays['gate'])
        c_layers[i].up = ptr(arrays['up'])
        c_layers[i].down = ptr(arrays['down'])
        c_layers[i].q_norm = ptr(arrays['q_norm']) if arrays['q_norm'] is not None else 0
        c_layers[i].k_norm = ptr(arrays['k_norm']) if arrays['k_norm'] is not None else 0
        c_layers[i].n_heads = attn.n_heads
        c_layers[i].n_kv = attn.n_kv_heads
        c_layers[i].head_dim = hd
        c_layers[i].hidden = hidden
        c_layers[i].inter = args.intermediate_size
        c_layers[i].scale = attn.scale

    final_norm = np.ascontiguousarray(to_np32(inner.norm.weight))
    lm_head = embed_np  # tied

    del model, inner
    mx.clear_cache()

    # Create CPU KV caches
    MAX_SEQ = 64
    c_caches = (KVCache * n_layers)()
    kv_arrays = []
    for i in range(n_layers):
        nkv = c_layers[i].n_kv
        k_buf = np.zeros((nkv, MAX_SEQ, hd), dtype=np.float32)
        v_buf = np.zeros((nkv, MAX_SEQ, hd), dtype=np.float32)
        kv_arrays.append((k_buf, v_buf))
        c_caches[i].k = ptr(k_buf)
        c_caches[i].v = ptr(v_buf)
        c_caches[i].seq_len = 0
        c_caches[i].max_seq = MAX_SEQ

    cpu_logits = np.zeros(vocab, dtype=np.float32)

    # Create GPU SDPA tensors (simulating verify step)
    N_Q, N_KV = 32, 2
    CTX = 8192
    q_gpu = mx.random.normal((1, N_Q, 4, hd)).astype(mx.bfloat16)  # q_seq=4 for verify
    k_gpu = mx.random.normal((1, N_KV, CTX, hd)).astype(mx.bfloat16)
    v_gpu = mx.random.normal((1, N_KV, CTX, hd)).astype(mx.bfloat16)
    mx.eval(q_gpu, k_gpu, v_gpu)
    SCALE = 1.0 / (hd ** 0.5)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _lib.cpu_draft_forward(1, ptr(embed_np), c_layers, n_layers,
            ptr(final_norm), ptr(lm_head), vocab, hidden, c_caches, 0,
            ctypes.c_float(theta), ptr(cpu_logits))
    for i in range(n_layers):
        c_caches[i].seq_len = 0

    for _ in range(3):
        out = mx.fast.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu, scale=SCALE)
        mx.eval(out)

    ITERS = 10

    # Measure CPU-only
    times_cpu = []
    for _ in range(ITERS):
        for i in range(n_layers):
            c_caches[i].seq_len = 0
        t0 = time.perf_counter_ns()
        for d in range(2):  # 2 draft tokens
            _lib.cpu_draft_forward(1, ptr(embed_np), c_layers, n_layers,
                ptr(final_norm), ptr(lm_head), vocab, hidden, c_caches, d,
                ctypes.c_float(theta), ptr(cpu_logits))
        times_cpu.append((time.perf_counter_ns() - t0) / 1e6)

    # Measure GPU-only (verify with q_seq=4)
    times_gpu = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = mx.fast.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu, scale=SCALE)
        mx.eval(out)
        mx.synchronize()
        times_gpu.append((time.perf_counter_ns() - t0) / 1e6)

    # Measure OVERLAPPED (CPU + GPU in parallel)
    times_both = []
    for _ in range(ITERS):
        for i in range(n_layers):
            c_caches[i].seq_len = 0

        mx.synchronize()
        t0 = time.perf_counter_ns()

        def cpu_work():
            for d in range(2):
                _lib.cpu_draft_forward(1, ptr(embed_np), c_layers, n_layers,
                    ptr(final_norm), ptr(lm_head), vocab, hidden, c_caches, d,
                    ctypes.c_float(theta), ptr(cpu_logits))

        cpu_thread = threading.Thread(target=cpu_work)
        cpu_thread.start()

        # GPU verify simultaneously
        out = mx.fast.scaled_dot_product_attention(q_gpu, k_gpu, v_gpu, scale=SCALE)
        mx.eval(out)
        mx.synchronize()

        cpu_thread.join()
        times_both.append((time.perf_counter_ns() - t0) / 1e6)

    avg_cpu = sum(times_cpu) / len(times_cpu)
    avg_gpu = sum(times_gpu) / len(times_gpu)
    avg_both = sum(times_both) / len(times_both)
    theoretical_max = max(avg_cpu, avg_gpu)
    theoretical_sum = avg_cpu + avg_gpu
    overlap_pct = (1 - (avg_both - theoretical_max) / (theoretical_sum - theoretical_max)) * 100

    print(f"\nResults:")
    print(f"  CPU draft (2 tokens):    {avg_cpu:.1f}ms")
    print(f"  GPU verify (q_seq=4):    {avg_gpu:.1f}ms")
    print(f"  CPU + GPU overlapped:    {avg_both:.1f}ms")
    print(f"  Theoretical max(c,g):    {theoretical_max:.1f}ms")
    print(f"  Theoretical c+g:         {theoretical_sum:.1f}ms")
    print(f"  Overlap: {overlap_pct:.0f}%")
    print()
    if avg_both < theoretical_sum * 0.7:
        print("  ✓ TRUE PARALLELISM — CPU and GPU overlap! Pipeline viable.")
    elif avg_both < theoretical_sum * 0.9:
        print("  ~ PARTIAL OVERLAP — some benefit from pipelining.")
    else:
        print("  ✗ NO OVERLAP — CPU and GPU contend for bandwidth. Pipeline won't help.")


if __name__ == "__main__":
    main()

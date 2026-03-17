#!/usr/bin/env python3
"""Benchmark CPU draft model using C/Accelerate implementation."""
import ctypes
import os
import time
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "..", "src", "exo", "worker", "engines", "mlx", "cpu_draft.dylib")

# Load C library
_lib = ctypes.CDLL(LIB_PATH)

# LayerWeights struct
class LayerWeights(ctypes.Structure):
    _fields_ = [
        ("in_norm", ctypes.c_void_p),
        ("post_norm", ctypes.c_void_p),
        ("q", ctypes.c_void_p),
        ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("o", ctypes.c_void_p),
        ("gate", ctypes.c_void_p),
        ("up", ctypes.c_void_p),
        ("down", ctypes.c_void_p),
        ("q_norm", ctypes.c_void_p),
        ("k_norm", ctypes.c_void_p),
        ("n_heads", ctypes.c_int),
        ("n_kv", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("hidden", ctypes.c_int),
        ("inter", ctypes.c_int),
        ("scale", ctypes.c_float),
    ]

class KVCache(ctypes.Structure):
    _fields_ = [
        ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("seq_len", ctypes.c_int),
        ("max_seq", ctypes.c_int),
    ]

_lib.cpu_draft_forward.argtypes = [
    ctypes.c_int,          # token_id
    ctypes.c_void_p,       # embed
    ctypes.POINTER(LayerWeights),  # layers
    ctypes.c_int,          # n_layers
    ctypes.c_void_p,       # final_norm
    ctypes.c_void_p,       # lm_head
    ctypes.c_int,          # vocab_size
    ctypes.c_int,          # hidden
    ctypes.POINTER(KVCache),  # caches
    ctypes.c_int,          # offset
    ctypes.c_float,        # rope_theta
    ctypes.c_void_p,       # logits_out
]
_lib.cpu_draft_forward.restype = None


def to_np32(arr):
    return np.array(arr.astype(mx.float32))


def ptr(arr):
    return arr.ctypes.data


def main():
    print("Loading Qwen3-0.6B and dequantizing...")
    t0 = time.perf_counter()
    model, tok = load('mlx-community/Qwen3-0.6B-8bit')
    mx.eval(model.parameters())
    args = model.args

    inner = model.model
    emb = inner.embed_tokens
    embed_np = np.ascontiguousarray(to_np32(
        mx.dequantize(emb.weight, emb.scales, emb.biases, emb.group_size, emb.bits)))

    n_layers = len(inner.layers)
    hidden = args.hidden_size
    vocab = args.vocab_size
    hd = args.head_dim
    theta = args.rope_theta

    # Extract layer weights
    layer_arrays = []  # keep references alive
    c_layers = (LayerWeights * n_layers)()

    for i, layer in enumerate(inner.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        in_norm = np.ascontiguousarray(to_np32(layer.input_layernorm.weight))
        post_norm = np.ascontiguousarray(to_np32(layer.post_attention_layernorm.weight))

        def dq(proj):
            return np.ascontiguousarray(to_np32(mx.dequantize(
                proj.weight, proj.scales, getattr(proj, 'biases', None),
                proj.group_size, proj.bits)))

        q_w = dq(attn.q_proj)
        k_w = dq(attn.k_proj)
        v_w = dq(attn.v_proj)
        o_w = dq(attn.o_proj)
        gate_w = dq(mlp.gate_proj)
        up_w = dq(mlp.up_proj)
        down_w = dq(mlp.down_proj)

        q_norm_np = np.ascontiguousarray(to_np32(attn.q_norm.weight)) if hasattr(attn, 'q_norm') else None
        k_norm_np = np.ascontiguousarray(to_np32(attn.k_norm.weight)) if hasattr(attn, 'k_norm') else None

        arrays = [in_norm, post_norm, q_w, k_w, v_w, o_w, gate_w, up_w, down_w, q_norm_np, k_norm_np]
        layer_arrays.append(arrays)

        c_layers[i].in_norm = ptr(in_norm)
        c_layers[i].post_norm = ptr(post_norm)
        c_layers[i].q = ptr(q_w)
        c_layers[i].k = ptr(k_w)
        c_layers[i].v = ptr(v_w)
        c_layers[i].o = ptr(o_w)
        c_layers[i].gate = ptr(gate_w)
        c_layers[i].up = ptr(up_w)
        c_layers[i].down = ptr(down_w)
        c_layers[i].q_norm = ptr(q_norm_np) if q_norm_np is not None else 0
        c_layers[i].k_norm = ptr(k_norm_np) if k_norm_np is not None else 0
        c_layers[i].n_heads = attn.n_heads
        c_layers[i].n_kv = attn.n_kv_heads
        c_layers[i].head_dim = hd
        c_layers[i].hidden = hidden
        c_layers[i].inter = args.intermediate_size
        c_layers[i].scale = attn.scale

    final_norm = np.ascontiguousarray(to_np32(inner.norm.weight))

    if args.tie_word_embeddings:
        lm_head = embed_np
    else:
        lm = model.lm_head
        lm_head = np.ascontiguousarray(to_np32(mx.dequantize(
            lm.weight, lm.scales, getattr(lm, 'biases', None),
            lm.group_size, lm.bits)))

    del model, inner
    mx.clear_cache()

    print(f"Ready in {time.perf_counter()-t0:.1f}s")
    print(f"Layers: {n_layers}, Hidden: {hidden}, Vocab: {vocab}, Head dim: {hd}")
    weight_mb = sum(a.nbytes for arrays in layer_arrays for a in arrays if a is not None)
    weight_mb += embed_np.nbytes + final_norm.nbytes + lm_head.nbytes
    print(f"Total weight memory: {weight_mb/1e6:.0f} MB (float32)")

    # Create KV caches
    MAX_SEQ = 256
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

    logits = np.zeros(vocab, dtype=np.float32)

    # Warmup
    print("\nWarming up...")
    for i in range(3):
        _lib.cpu_draft_forward(
            1, ptr(embed_np), c_layers, n_layers,
            ptr(final_norm), ptr(lm_head),
            vocab, hidden, c_caches, i,
            ctypes.c_float(theta), ptr(logits))

    # Reset caches
    for i in range(n_layers):
        c_caches[i].seq_len = 0
        kv_arrays[i][0].fill(0)
        kv_arrays[i][1].fill(0)

    # Benchmark
    print("Benchmarking...")
    times = []
    for i in range(10):
        t0 = time.perf_counter_ns()
        _lib.cpu_draft_forward(
            1, ptr(embed_np), c_layers, n_layers,
            ptr(final_norm), ptr(lm_head),
            vocab, hidden, c_caches, i,
            ctypes.c_float(theta), ptr(logits))
        times.append((time.perf_counter_ns() - t0) / 1e6)

    avg = sum(times) / len(times)
    predicted = logits.argmax()
    print(f"\nC/Accelerate CPU forward pass: {avg:.1f}ms per token")
    print(f"Predicted token: {predicted}")
    print(f"3 draft tokens: {avg*3:.1f}ms")

    verify = 40
    acceptance = 0.63
    tok_per_step = 1 + acceptance + acceptance**2 + acceptance**3
    draft_3 = avg * 3
    total = draft_3 + verify
    effective = total / tok_per_step
    print(f"\nWith 63% acceptance, 3 draft tokens:")
    print(f"  Draft: {draft_3:.0f}ms + Verify: {verify}ms = {total:.0f}ms")
    print(f"  Tokens/step: {tok_per_step:.2f}")
    print(f"  Effective: {effective:.1f}ms/tok")
    print(f"  Baseline: 35ms/tok")
    print(f"  Speedup: {35/effective:.2f}×")


if __name__ == "__main__":
    main()

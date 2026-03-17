#!/usr/bin/env python3
"""Benchmark INT8 quantized CPU draft model."""
import ctypes
import os
import time
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "..", "src", "exo", "worker", "engines", "mlx", "cpu_draft_q8.dylib")
_lib = ctypes.CDLL(LIB_PATH)


class Q8LayerWeights(ctypes.Structure):
    _fields_ = [
        ("in_norm", ctypes.c_void_p), ("post_norm", ctypes.c_void_p),
        ("q_w", ctypes.c_void_p), ("q_s", ctypes.c_void_p), ("q_b", ctypes.c_void_p),
        ("k_w", ctypes.c_void_p), ("k_s", ctypes.c_void_p), ("k_b", ctypes.c_void_p),
        ("v_w", ctypes.c_void_p), ("v_s", ctypes.c_void_p), ("v_b", ctypes.c_void_p),
        ("o_w", ctypes.c_void_p), ("o_s", ctypes.c_void_p), ("o_b", ctypes.c_void_p),
        ("gate_w", ctypes.c_void_p), ("gate_s", ctypes.c_void_p), ("gate_b", ctypes.c_void_p),
        ("up_w", ctypes.c_void_p), ("up_s", ctypes.c_void_p), ("up_b", ctypes.c_void_p),
        ("down_w", ctypes.c_void_p), ("down_s", ctypes.c_void_p), ("down_b", ctypes.c_void_p),
        ("q_norm", ctypes.c_void_p), ("k_norm", ctypes.c_void_p),
        ("n_heads", ctypes.c_int), ("n_kv", ctypes.c_int), ("head_dim", ctypes.c_int),
        ("hidden", ctypes.c_int), ("inter", ctypes.c_int), ("group_size", ctypes.c_int),
        ("scale", ctypes.c_float),
    ]


class KVCache(ctypes.Structure):
    _fields_ = [
        ("k", ctypes.c_void_p), ("v", ctypes.c_void_p),
        ("seq_len", ctypes.c_int), ("max_seq", ctypes.c_int),
    ]


_lib.cpu_draft_q8_forward.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Q8LayerWeights),
    ctypes.c_int, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(KVCache),
    ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
]
_lib.cpu_draft_q8_forward.restype = None


def to_np32(arr):
    return np.array(arr.astype(mx.float32))

def to_np_raw(arr):
    """Get raw uint32 array without conversion."""
    return np.array(arr)

def ptr(arr):
    return arr.ctypes.data if arr is not None else 0


def main():
    print("Loading Qwen3-0.6B (keeping weights quantized)...")
    t0 = time.perf_counter()
    model, tok = load('mlx-community/Qwen3-0.6B-8bit')
    mx.eval(model.parameters())
    args = model.args
    inner = model.model

    # Dequantize embedding only (needed for lookup)
    emb = inner.embed_tokens
    embed_np = np.ascontiguousarray(to_np32(
        mx.dequantize(emb.weight, emb.scales, emb.biases, emb.group_size, emb.bits)))

    n_layers = len(inner.layers)
    hidden = args.hidden_size
    vocab = args.vocab_size
    hd = args.head_dim
    theta = args.rope_theta

    # Extract quantized weights (keep as uint32 + f32 scales/biases)
    layer_arrays = []
    c_layers = (Q8LayerWeights * n_layers)()

    total_bytes = embed_np.nbytes

    for i, layer in enumerate(inner.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        arrays = {}

        arrays['in_norm'] = np.ascontiguousarray(to_np32(layer.input_layernorm.weight))
        arrays['post_norm'] = np.ascontiguousarray(to_np32(layer.post_attention_layernorm.weight))

        gs = attn.q_proj.group_size

        for name, proj in [('q', attn.q_proj), ('k', attn.k_proj),
                           ('v', attn.v_proj), ('o', attn.o_proj),
                           ('gate', mlp.gate_proj), ('up', mlp.up_proj),
                           ('down', mlp.down_proj)]:
            arrays[f'{name}_w'] = np.ascontiguousarray(to_np_raw(proj.weight))
            arrays[f'{name}_s'] = np.ascontiguousarray(to_np32(proj.scales))
            arrays[f'{name}_b'] = np.ascontiguousarray(to_np32(proj.biases)) if hasattr(proj, 'biases') and proj.biases is not None else None

        if hasattr(attn, 'q_norm'):
            arrays['q_norm'] = np.ascontiguousarray(to_np32(attn.q_norm.weight))
            arrays['k_norm'] = np.ascontiguousarray(to_np32(attn.k_norm.weight))

        layer_arrays.append(arrays)

        c_layers[i].in_norm = ptr(arrays['in_norm'])
        c_layers[i].post_norm = ptr(arrays['post_norm'])
        for name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
            setattr(c_layers[i], f'{name}_w', ptr(arrays[f'{name}_w']))
            setattr(c_layers[i], f'{name}_s', ptr(arrays[f'{name}_s']))
            setattr(c_layers[i], f'{name}_b', ptr(arrays.get(f'{name}_b')))
        c_layers[i].q_norm = ptr(arrays.get('q_norm'))
        c_layers[i].k_norm = ptr(arrays.get('k_norm'))
        c_layers[i].n_heads = attn.n_heads
        c_layers[i].n_kv = attn.n_kv_heads
        c_layers[i].head_dim = hd
        c_layers[i].hidden = hidden
        c_layers[i].inter = args.intermediate_size
        c_layers[i].group_size = gs
        c_layers[i].scale = attn.scale

        for a in arrays.values():
            if a is not None:
                total_bytes += a.nbytes

    final_norm = np.ascontiguousarray(to_np32(inner.norm.weight))
    total_bytes += final_norm.nbytes

    # LM head — use float32 embedding for tied weights (skip dequant of 155M elements)
    if args.tie_word_embeddings:
        lm_w = embed_np  # already float32
        lm_s = None  # signal to C code: use float32 path
        lm_b = None
        lm_gs = 0
    else:
        lm = model.lm_head
        lm_w = np.ascontiguousarray(to_np_raw(lm.weight))
        lm_s = np.ascontiguousarray(to_np32(lm.scales))
        lm_b = np.ascontiguousarray(to_np32(lm.biases)) if hasattr(lm, 'biases') and lm.biases is not None else None
        lm_gs = lm.group_size

    del model, inner
    mx.clear_cache()

    print(f"Ready in {time.perf_counter()-t0:.1f}s")
    print(f"Layers: {n_layers}, Hidden: {hidden}, Vocab: {vocab}")
    print(f"Total weight memory: {total_bytes/1e6:.0f} MB (quantized + norms)")

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

    print("\nWarming up...")
    for i in range(3):
        _lib.cpu_draft_q8_forward(
            1, ptr(embed_np), c_layers, n_layers,
            ptr(final_norm), ptr(lm_w), ptr(lm_s), ptr(lm_b), lm_gs,
            vocab, hidden, c_caches, i,
            ctypes.c_float(theta), ptr(logits))

    # Reset
    for i in range(n_layers):
        c_caches[i].seq_len = 0

    print("Benchmarking...")
    times = []
    for i in range(10):
        t0 = time.perf_counter_ns()
        _lib.cpu_draft_q8_forward(
            1, ptr(embed_np), c_layers, n_layers,
            ptr(final_norm), ptr(lm_w), ptr(lm_s), ptr(lm_b), lm_gs,
            vocab, hidden, c_caches, i,
            ctypes.c_float(theta), ptr(logits))
        times.append((time.perf_counter_ns() - t0) / 1e6)

    avg = sum(times) / len(times)
    print(f"\nINT8 C/Accelerate CPU forward pass: {avg:.1f}ms per token")
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
    print(f"\n  Pipelined (draft during verify): {max(draft_3, verify)/tok_per_step:.1f}ms/tok")
    print(f"  Pipelined speedup: {35/(max(draft_3, verify)/tok_per_step):.2f}×")


if __name__ == "__main__":
    main()

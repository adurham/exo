#!/usr/bin/env python3
"""Microbench for DSv4 Compressor variants.

At c=1 100K decode, the Compressor's __call__ does:
  1. _project_kv_gate (quantized_matmul, already fused via Phase H)
  2. accumulate_windows from PoolingCache
  3. _simple_compress_kv OR _overlap_compress_kv on accumulated chunks
  4. norm + rope

Profile shows attn.compressor = 23.2 ms / decode step (was warmup-prefill;
decode steady-state value is much lower since only L_chunks=0 or 1 per step).

Bench BOTH compress functions at realistic decode-step shapes:
  - simple (compress_ratio=128, half of layers): kv,gate [B,L,R=128,D]  ape [R=128,D]
  - overlap (compress_ratio=4, half of layers): kv,gate [B,L,R=4,2*D] ape [R=4,2*D]

At decode, L_chunks=1 typically (one new chunk just ready). At prefill chunk
size 128, L_chunks=N/R per call.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx


# === SIMPLE COMPRESSOR (compress_ratio=128, half the layers) ===
def baseline_simple(kv, gate, ape, head_dim):
    """Current implementation: fp32 cast around gate+ape softmax."""
    weights = mx.softmax(gate.astype(mx.float32) + ape, axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


def variant_simple_precise(kv, gate, ape, head_dim):
    """Drop explicit fp32 cast, use precise=True (promotes internally)."""
    gate_plus_ape = gate + ape.astype(gate.dtype)
    weights = mx.softmax(gate_plus_ape, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


def variant_simple_bf16(kv, gate, ape, head_dim):
    """Pure bf16 softmax — fastest but lossiest."""
    gate_plus_ape = gate + ape.astype(gate.dtype)
    weights = mx.softmax(gate_plus_ape, axis=-2)
    return (kv * weights).sum(axis=-2)


# === OVERLAP COMPRESSOR (compress_ratio=4, half the layers) ===
def baseline_overlap(kv, gate, ape, head_dim):
    """Current implementation: precise=True softmax (no explicit fp32 cast)."""
    B, L, R, D = kv.shape
    gate = gate + ape.astype(gate.dtype)
    kv_0 = mx.zeros((B, 1, R, D // 2), dtype=kv.dtype)
    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([kv_0, kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)
    gate_0 = mx.full((B, 1, R, D // 2), -mx.inf, dtype=kv.dtype)
    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([gate_0, gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)
    weights = mx.softmax(gate, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


def variant_overlap_no_precise(kv, gate, ape, head_dim):
    """Drop precise=True — pure bf16 softmax."""
    B, L, R, D = kv.shape
    gate = gate + ape.astype(gate.dtype)
    kv_0 = mx.zeros((B, 1, R, D // 2), dtype=kv.dtype)
    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([kv_0, kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)
    gate_0 = mx.full((B, 1, R, D // 2), -mx.inf, dtype=kv.dtype)
    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([gate_0, gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)
    weights = mx.softmax(gate, axis=-2)  # no precise
    return (kv * weights).sum(axis=-2)


def time_variant(fn, args_tuple, n_trials=100, n_warmup=10):
    for _ in range(n_warmup):
        out = fn(*args_tuple)
        mx.eval(out)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(*args_tuple)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    return out, times


def diff_stats(ref, alt):
    diff = (ref.astype(mx.float32) - alt.astype(mx.float32)).abs()
    return float(diff.max()), float(diff.mean())


def bench_set(name, baseline_fn, variant_fns, args_tuple, n_trials):
    print(f"\n=== {name} ===")
    ref_out, ref_times = time_variant(baseline_fn, args_tuple, n_trials)
    bm = sum(ref_times) / len(ref_times)
    print(f"  {'baseline':28s} mean={bm:6.3f}ms speedup=1.00x")
    for vname, vfn in variant_fns:
        out, times = time_variant(vfn, args_tuple, n_trials)
        m = sum(times) / len(times)
        sp = bm / m if m > 0 else 0
        max_d, mean_d = diff_stats(ref_out, out)
        print(f"  {vname:28s} mean={m:6.3f}ms speedup={sp:.2f}x  max_diff={max_d:.4g} mean_diff={mean_d:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--simple-L", type=int, default=1, help="L_chunks for simple compressor")
    ap.add_argument("--overlap-L", type=int, default=1, help="L_chunks for overlap compressor")
    args = ap.parse_args()

    B = 1
    head_dim = 128

    # SIMPLE: compress_ratio=128
    R_s = 128
    D_s = head_dim
    L_s = args.simple_L
    kv_s = mx.random.normal(shape=(B, L_s, R_s, D_s)).astype(mx.bfloat16)
    gate_s = mx.random.normal(shape=(B, L_s, R_s, D_s)).astype(mx.bfloat16)
    ape_s = mx.random.normal(shape=(R_s, D_s)).astype(mx.float32)
    bench_set(
        f"_simple_compress_kv  (R={R_s} L={L_s} D={D_s})",
        baseline_simple,
        [
            ("precise=True (no cast)", variant_simple_precise),
            ("bf16 softmax (no precise)", variant_simple_bf16),
        ],
        (kv_s, gate_s, ape_s, head_dim),
        args.n_trials,
    )

    # OVERLAP: compress_ratio=4, D doubled
    R_o = 4
    D_o = head_dim * 2
    L_o = args.overlap_L
    kv_o = mx.random.normal(shape=(B, L_o, R_o, D_o)).astype(mx.bfloat16)
    gate_o = mx.random.normal(shape=(B, L_o, R_o, D_o)).astype(mx.bfloat16)
    ape_o = mx.random.normal(shape=(R_o, D_o)).astype(mx.float32)
    bench_set(
        f"_overlap_compress_kv (R={R_o} L={L_o} D={D_o})",
        baseline_overlap,
        [
            ("no precise=True", variant_overlap_no_precise),
        ],
        (kv_o, gate_o, ape_o, head_dim),
        args.n_trials,
    )


if __name__ == "__main__":
    main()

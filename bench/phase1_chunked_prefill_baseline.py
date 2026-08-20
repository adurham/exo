#!/usr/bin/env python3
"""PHASE 1: Chunked prefill for DSv4-Flash's full forward pass.

Splits a prompt into K sequence chunks (K=2 to start) and runs them through
the REAL DeepseekV4 `Model.__call__` (mlx-lm) SEQUENTIALLY (no overlap --
that's Phase 2), advancing a real KV cache chunk-by-chunk exactly like
production prefill (`_prefill` in mlx_lm/generate.py) does with
`prefill_step_size`.

Goal: establish a CORRECTNESS baseline -- chunked prefill must produce
bit-identical (or numerically negligible-diff) final-position logits and
final KV-cache state vs a single monolithic (K=1) forward pass over the
same prompt. This is the necessary precondition before Phase 2 (chunk
overlap / pipelining) can be trusted.

Uses a SMALL SYNTHETIC DeepseekV4 config (real `Model`/`DeepseekV4Model`/
attention/MoE code, not a mock) because the real ~600B-param DSv4-Flash
checkpoint is not present on this laptop (only tokenizer/config were
downloaded, no safetensors). This mirrors this repo's established
precedent for testing DSv4 forward-pass mechanics without the full
checkpoint (see test_batch_generate_chunked_prefill_live_wiring.py).

Run: .venv/bin/python bench/phase1_chunked_prefill_baseline.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v4 import Model, ModelArgs


def make_tiny_dsv4() -> Model:
    """Small-but-real DeepseekV4 config: exercises MLA attention, MoE
    routing, compressed/sparse attention layers, and MTP head plumbing
    at a size that eval()s in seconds on a laptop GPU."""
    args = ModelArgs(
        model_type="deepseek_v4",
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.5,
        q_lora_rank=128,
        qk_rope_head_dim=32,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        head_dim=64,
        scoring_func="sqrtsoftplus",
        sliding_window=128,
        o_groups=2,
        o_lora_rank=128,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=64,
        num_nextn_predict_layers=0,
        tie_word_embeddings=False,
        topk_method="noaux_tc",
        n_mtp_layers=0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    model.eval()
    return model, args


def sequential_chunked_prefill(model, tokens: mx.array, k: int):
    """Split `tokens` (shape (1, T)) into K sequential chunks, run each
    through the FULL model with a real advancing KV cache -- exactly the
    production `_prefill` shape (mlx_lm/generate.py: process n_to_process
    tokens per step, cache.offset advances each call). Returns
    (final_logits_over_full_seq_concat, cache) for correctness comparison.
    """
    cache = model.make_cache()
    T = tokens.shape[1]
    chunk_size = (T + k - 1) // k
    boundaries = list(range(0, T, chunk_size)) + [T]

    all_logits = []
    t_per_chunk = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        chunk = tokens[:, lo:hi]
        t0 = time.perf_counter()
        logits = model(chunk, cache=cache)
        mx.eval(logits)
        dt = time.perf_counter() - t0
        t_per_chunk.append(dt)
        print(f"  chunk {i}: tokens[{lo}:{hi}] (len={hi - lo})  "
              f"logits.shape={logits.shape}  {dt * 1000:.1f} ms  "
              f"cache[0].offset={cache[0].offset if hasattr(cache[0], 'offset') else '?'}")
        all_logits.append(logits)

    full_logits = mx.concatenate(all_logits, axis=1)
    return full_logits, cache, t_per_chunk


def monolithic_prefill(model, tokens: mx.array):
    cache = model.make_cache()
    t0 = time.perf_counter()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    dt = time.perf_counter() - t0
    return logits, cache, dt


def cache_state_equal(cache_a, cache_b, atol=0.0) -> bool:
    """Compare KV cache contents layer-by-layer where possible."""
    ok = True
    for i, (ca, cb) in enumerate(zip(cache_a, cache_b)):
        for attr in ("keys", "values"):
            ka = getattr(ca, attr, None)
            kb = getattr(cb, attr, None)
            if ka is None or kb is None:
                continue
            if ka.size == 0 or kb.size == 0:
                if ka.shape != kb.shape:
                    print(f"  [cache mismatch] layer {i} {attr} shape {ka.shape} vs {kb.shape}")
                    ok = False
                continue
            if ka.shape != kb.shape:
                print(f"  [cache mismatch] layer {i} {attr} shape {ka.shape} vs {kb.shape}")
                ok = False
                continue
            diff = float(mx.max(mx.abs(ka.astype(mx.float32) - kb.astype(mx.float32))).item())
            if diff > atol:
                print(f"  [cache diff] layer {i} {attr} max_abs_diff={diff:.6g}")
                if diff > 1e-3:
                    ok = False
    return ok


def main():
    mx.random.seed(0)
    model, args = make_tiny_dsv4()

    T = 96  # prompt length -- multiple K values divide evenly-ish
    tokens = mx.random.randint(0, args.vocab_size, (1, T))
    mx.eval(tokens)

    print(f"=== PHASE 1: chunked prefill baseline (T={T} tokens, "
          f"vocab={args.vocab_size}, layers={args.num_hidden_layers}) ===\n")

    print("[monolithic K=1]")
    mono_logits, mono_cache, mono_dt = monolithic_prefill(model, tokens)
    print(f"  full forward: {mono_dt * 1000:.1f} ms  logits.shape={mono_logits.shape}\n")

    results = {}
    for k in [2, 3, 4]:
        print(f"[chunked K={k}]")
        chunk_logits, chunk_cache, t_per_chunk = sequential_chunked_prefill(model, tokens, k)
        total_dt = sum(t_per_chunk)

        # Correctness: last-token logits must match (this is what decode
        # actually consumes) AND full per-position logits should match
        # (validates chunk boundaries don't corrupt earlier positions'
        # causal masking/attention).
        last_diff = float(mx.max(mx.abs(
            chunk_logits[:, -1, :].astype(mx.float32)
            - mono_logits[:, -1, :].astype(mx.float32)
        )).item())
        full_diff = float(mx.max(mx.abs(
            chunk_logits.astype(mx.float32) - mono_logits.astype(mx.float32)
        )).item())
        last_argmax_match = bool(
            chunk_logits[:, -1, :].argmax(-1).item() == mono_logits[:, -1, :].argmax(-1).item()
        )
        cache_ok = cache_state_equal(chunk_cache, mono_cache)

        print(f"  total chunked time: {total_dt * 1000:.1f} ms "
              f"({total_dt / mono_dt:.2f}x vs monolithic)")
        print(f"  last-position logits max_abs_diff = {last_diff:.6g}")
        print(f"  full-sequence logits max_abs_diff = {full_diff:.6g}")
        print(f"  last-position argmax match: {last_argmax_match}")
        print(f"  KV cache state match: {cache_ok}")
        print(f"  PASS: {last_diff < 1e-3 and last_argmax_match and cache_ok}\n")

        results[k] = dict(
            total_dt=total_dt, last_diff=last_diff, full_diff=full_diff,
            argmax_match=last_argmax_match, cache_ok=cache_ok,
        )

    print("=== SUMMARY ===")
    all_pass = True
    for k, r in results.items():
        p = r["last_diff"] < 1e-3 and r["argmax_match"] and r["cache_ok"]
        all_pass &= p
        print(f"  K={k}: {'PASS' if p else 'FAIL'}  "
              f"last_diff={r['last_diff']:.3g}  slowdown={r['total_dt'] / mono_dt:.2f}x")
    print(f"\nOverall Phase 1 gate: {'PASS' if all_pass else 'FAIL'}")
    print("(NOTE: laptop timings are noisy/single-GPU-shared -- ratios, not "
          "absolute ms, are the signal here. Correctness match is the real gate.)")


if __name__ == "__main__":
    main()

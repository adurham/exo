#!/usr/bin/env python3
"""BATCHED twin of bench/phase1_chunked_prefill_baseline.py.

Verifies the chunk-boundary overlap-pooling carry fix on the BATCHED path
(BatchPoolingCache), which is exo's production serving path
(prefill_batched -> batch_generate).

Two gates:

  GATE A (no crash + chunk-boundary correctness): batched chunked prefill
  (K chunks) vs batched monolithic (K=1) prefill through the REAL
  DeepseekV4 forward with BatchPoolingCache. Before the fix this raised
  AttributeError (BatchPoolingCache had no _overlap_kv_carry). After the
  carry mechanism is added, divergence must be at fp-rounding level, NOT
  the ~0.8-0.9 max-abs-diff bug signature.

  GATE B (per-stream carry independence): the batch is run with B=2
  streams of DIFFERENT lengths AND different content, then each stream is
  ALSO run alone (B=1, single-stream PoolingCache path) with the same
  chunking. Per-stream batched output must match its own single-stream
  output. A shared/broadcast carry (one stream's carry leaking onto
  another) shows up here and nowhere else.

Run: .venv/bin/python bench/phase1_batched_chunked_prefill.py
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.generate import _make_cache
from mlx_lm.models.deepseek_v4 import Model, ModelArgs


def make_tiny_dsv4():
    """Same small-but-real config as the non-batched baseline. NOTE the
    explicit compress_ratios: layers 2 and 4 are ratio=4 (the `overlap`
    pooling layers this fix is about), matching the production
    DSv4-Flash pattern [0, 0, 4, 128, 4, 128, ...]."""
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
        compress_ratios=[0, 0, 4, 128, 4, 0],
    )
    model = Model(args)
    mx.eval(model.parameters())
    model.eval()
    return model, args


def batched_prefill(model, tokens, lengths, k):
    """Run a batched prefill of `tokens` (B, T) split into K sequential
    chunks with a real BatchPoolingCache-backed cache, calling prepare()
    with the per-chunk valid lengths exactly like the batched serving
    path does. Returns concatenated logits (B, T, V)."""
    batch_size = tokens.shape[0]
    cache = _make_cache(model, [0] * batch_size, None)
    total = tokens.shape[1]
    chunk_size = (total + k - 1) // k
    bounds = list(range(0, total, chunk_size)) + [total]

    out = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        chunk = tokens[:, lo:hi]
        chunk_lengths = [max(min(le, hi) - lo, 0) for le in lengths]
        for c in cache:
            _prepare_cache(c, chunk_lengths)
        logits = model(chunk, cache=cache)
        mx.eval(logits)
        out.append(logits)
    return mx.concatenate(out, axis=1), cache


def _prepare_cache(c, lengths):
    if hasattr(c, "caches"):
        for sub in c.caches:
            _prepare_cache(sub, lengths)
    elif hasattr(c, "prepare"):
        c.prepare(lengths=lengths)


def max_abs_diff(a, b):
    return float(
        mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
    )


def main():
    mx.random.seed(0)
    model, args = make_tiny_dsv4()
    print(f"compress_ratios = {args.compress_ratios}  "
          f"(overlap/ratio-4 layers: "
          f"{[i for i, r in enumerate(args.compress_ratios) if r == 4]})\n")

    # B=2, DIFFERENT lengths and DIFFERENT content per stream.
    total = 96
    len_a, len_b = 96, 68
    stream_a = mx.random.randint(0, args.vocab_size, (1, total))
    stream_b = mx.random.randint(0, args.vocab_size, (1, total))
    tokens = mx.concatenate([stream_a, stream_b], axis=0)
    lengths = [len_a, len_b]
    mx.eval(tokens)
    print(f"=== BATCHED chunked prefill (B=2, lengths={lengths}) ===\n")

    print("[batched monolithic K=1]")
    mono, _ = batched_prefill(model, tokens, lengths, 1)
    print(f"  logits.shape={mono.shape}\n")

    all_pass = True
    for k in (2, 3, 4):
        print(f"[batched chunked K={k}]")
        chunked, _ = batched_prefill(model, tokens, lengths, k)
        d_all = max_abs_diff(chunked, mono)
        d_a = max_abs_diff(chunked[0:1, :len_a], mono[0:1, :len_a])
        d_b = max_abs_diff(chunked[1:2, :len_b], mono[1:2, :len_b])
        ok = d_all < 1e-3
        all_pass &= ok
        print(f"  full-batch logits max_abs_diff  = {d_all:.6g}")
        print(f"  stream0 (len={len_a}) max_abs_diff = {d_a:.6g}")
        print(f"  stream1 (len={len_b}) max_abs_diff = {d_b:.6g}")
        print(f"  GATE A PASS: {ok}\n")

    # GATE B: per-stream carry independence. Each stream run ALONE through
    # the single-stream chunked path must match its row of the B=2 batch.
    print("=== GATE B: per-stream carry independence (B=2 vs B=1) ===")
    k = 3
    batched, _ = batched_prefill(model, tokens, lengths, k)
    for idx, (row, le) in enumerate(
        zip((stream_a, stream_b), (len_a, len_b), strict=True)
    ):
        solo, _ = batched_prefill(model, row[:, :le], [le], k)
        d = max_abs_diff(batched[idx : idx + 1, :le], solo[:, :le])
        ok = d < 1e-3
        all_pass &= ok
        print(f"  stream{idx} (len={le}) batched-vs-solo max_abs_diff = "
              f"{d:.6g}   PASS: {ok}")

    print(f"\nOverall batched gate: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()

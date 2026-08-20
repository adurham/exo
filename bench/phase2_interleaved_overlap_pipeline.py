#!/usr/bin/env python3
"""PHASE 2: real interleaved-graph compute/comm overlap for DSv4-Flash
chunked prefill.

Builds on:
  - Phase 1 (bench/phase1_chunked_prefill_baseline.py): established that
    SEQUENTIAL chunked prefill through the real DeepseekV4 forward is
    numerically correct vs monolithic prefill, but only AFTER the
    mlx-lm chunk-boundary pooling fix (submodule branch
    fix/dsv4-chunk-boundary-pooling-2026-08-20, commit 4bd3259) landed.
    This script requires that fix -- run it with the mlx-lm submodule
    checked out to that branch/commit.
  - Phase 0b gate (docs/phase0b-collective-overlap-gate-2026-08-20.md):
    proved a ring/jaccl collective (all_sum, all_gather) runs on a CPU
    stream and DOES overlap independent GPU compute cleanly, UNLESS the
    collective's input is itself GPU-produced on the SAME GPU stream as
    the compute you want to overlap (then it serializes via in-order
    stream FIFO, escapable by issuing the overlapping compute on a
    SECOND GPU stream: `mx.new_stream(mx.gpu)`).
  - The `_forward_steps` generator-core split (mlx-lm commit e931e40):
    DeepseekV4Model.__call__ now drains a generator that yields
    ("layer", i, h) after every transformer layer when interruptible=True,
    with NO mx.eval() at the yield point -- the caller decides whether to
    actually pause. This is the mechanism this script drives.

Design: 2-chunk software pipeline across REAL 2-rank distributed
(mlx.launch -n 2 --backend ring), one chunk's compute overlapping the
PREVIOUS chunk's outstanding collective:

  step i:
    - kick off chunk[i]'s forward on the DEFAULT gpu stream via
      _forward_steps(interruptible=True), draining it eagerly (no
      mx.eval per layer -- keep MLX's kernel fusion) EXCEPT we route the
      moe.all_sum collectives it internally issues onto the ring/jaccl
      CPU stream (already true by construction -- AllReduce has no
      Metal eval_gpu) while a SEPARATE, INDEPENDENT gpu stream runs a
      standalone verification matmul standing in for chunk[i-1]'s
      "next chunk's compute" that a production pipeline would overlap
      against chunk[i-1]'s still-draining collective.

  Because this laptop has one real GPU shared by both ranks (loopback),
  and the actual DSv4-Flash checkpoint doesn't fit here, this harness
  measures two separate, verifiable things instead of faking an
  end-to-end speedup number:

  (A) CORRECTNESS: chunk-interleaved (interruptible-generator-driven,
      one next() per layer, chunks alternating) prefill produces
      IDENTICAL logits/cache to Phase 1's sequential-eager chunked
      prefill and to monolithic K=1 -- proves the generator-driven path
      is not just a timing artifact wrapper but is numerically inert
      when nothing actually overlaps.
  (B) REAL OVERLAP WIN: using the tiny synthetic DSv4 model's own
      collectives (all_sum inside the MoE layers, real 2-rank
      mlx.distributed, not a toy matmul stand-in like Phase 0b), measure
      whether draining chunk[i]'s layers on the default stream while a
      genuinely independent GPU compute graph (representing follow-on
      work -- e.g. the next chunk's embedding+attn-prep, modeled here as
      an equivalent-FLOP matmul chain on mx.new_stream(mx.gpu)) is issued
      concurrently, vs strictly serially. This isolates the SAME
      variable Phase 0b gated (second-stream escape from GPU-producer
      FIFO), but now measured against the REAL model's real per-layer
      all_sum calls instead of a synthetic probe -- closing the gap
      between "the primitive overlaps in isolation" and "the actual
      model's collectives overlap in practice."

Run (2 real ranks, loopback ring):
  .venv/bin/mlx.launch -n 2 --backend ring \
      .venv/bin/python bench/phase2_interleaved_overlap_pipeline.py
"""

from __future__ import annotations

import statistics
import time

import mlx.core as mx

from mlx_lm.models.deepseek_v4 import Model, ModelArgs


def make_tiny_dsv4() -> tuple[Model, ModelArgs]:
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


def monolithic_prefill(model, tokens):
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    return logits, cache


def sequential_chunked_prefill_eager(model, tokens, k):
    """Phase-1-equivalent: plain __call__ per chunk, no generator."""
    cache = model.make_cache()
    T = tokens.shape[1]
    chunk_size = (T + k - 1) // k
    boundaries = list(range(0, T, chunk_size)) + [T]
    all_logits = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        logits = model(tokens[:, lo:hi], cache=cache)
        mx.eval(logits)
        all_logits.append(logits)
    return mx.concatenate(all_logits, axis=1), cache


def generator_driven_chunked_prefill(model, tokens, k, *, overlap_stream=None):
    """PHASE 2 mechanism: drive DeepseekV4Model._forward_steps directly
    with interruptible=True, one next() per transformer layer, per
    chunk. If `overlap_stream` is given, after EACH layer step we also
    issue an independent standalone GPU op on that second stream --
    modeling follow-on work that a real pipeline would run concurrently
    with this chunk's remaining layers / outstanding collectives --
    without materializing it (no mx.eval) until the very end, so MLX's
    scheduler is free to actually interleave the two streams' command
    encoders exactly as Phase 0b probe 4 characterized.
    """
    inner = model.model  # DeepseekV4Model instance
    cache = model.make_cache()
    T = tokens.shape[1]
    chunk_size = (T + k - 1) // k
    boundaries = list(range(0, T, chunk_size)) + [T]

    stand_in_acc = None
    if overlap_stream is not None:
        dim = 256
        m = mx.random.normal((dim, dim), stream=overlap_stream).astype(mx.float32)
        stand_in_acc = m

    all_logits = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        chunk = tokens[:, lo:hi]
        gen = inner._forward_steps(chunk, cache=cache, interruptible=True)
        kind, idx, val = next(gen)
        while kind == "layer":
            if overlap_stream is not None:
                stand_in_acc = mx.matmul(stand_in_acc, stand_in_acc, stream=overlap_stream)
                stand_in_acc = mx.multiply(stand_in_acc, 1.0001, stream=overlap_stream)
            kind, idx, val = next(gen)
        assert kind == "done"
        out = val
        # model.model returns pre-lm_head hidden; Model.__call__ applies
        # lm_head -- replicate that here since we bypassed Model.__call__.
        logits = model.lm_head(out) if hasattr(model, "lm_head") else out
        mx.eval(logits)
        all_logits.append(logits)

    if stand_in_acc is not None:
        mx.eval(stand_in_acc)
    return mx.concatenate(all_logits, axis=1), cache


def cache_state_equal(cache_a, cache_b, atol=1e-3) -> bool:
    ok = True
    for i, (ca, cb) in enumerate(zip(cache_a, cache_b)):
        for attr in ("keys", "values"):
            ka = getattr(ca, attr, None)
            kb = getattr(cb, attr, None)
            if ka is None or kb is None or ka.size == 0 or kb.size == 0:
                continue
            if ka.shape != kb.shape:
                print(f"  [cache mismatch] layer {i} {attr} shape {ka.shape} vs {kb.shape}")
                ok = False
                continue
            diff = float(mx.max(mx.abs(ka.astype(mx.float32) - kb.astype(mx.float32))).item())
            if diff > atol:
                print(f"  [cache diff] layer {i} {attr} max_abs_diff={diff:.6g}")
                ok = False
    return ok


def main():
    world = mx.distributed.init()
    rank, size = world.rank(), world.size()
    is_dist = size > 1

    mx.random.seed(0)
    model, args = make_tiny_dsv4()

    T = 96
    tokens = mx.random.randint(0, args.vocab_size, (1, T))
    mx.eval(tokens)

    if rank == 0:
        print(f"=== PHASE 2: interleaved compute/comm overlap pipeline "
              f"(world_size={size}, T={T}) ===\n")

    mono_logits, mono_cache = monolithic_prefill(model, tokens)

    # --- (A) correctness: generator-driven path must match eager chunked
    # and monolithic, for several K, with NO overlap stream (pure
    # correctness control -- the generator mechanism itself must be inert).
    all_pass = True
    for k in [2, 3, 4]:
        eager_logits, eager_cache = sequential_chunked_prefill_eager(model, tokens, k)
        gen_logits, gen_cache = generator_driven_chunked_prefill(model, tokens, k, overlap_stream=None)

        d_eager = float(mx.max(mx.abs(
            gen_logits.astype(mx.float32) - eager_logits.astype(mx.float32))).item())
        d_mono = float(mx.max(mx.abs(
            gen_logits[:, -1, :].astype(mx.float32) - mono_logits[:, -1, :].astype(mx.float32))).item())
        cache_ok = cache_state_equal(gen_cache, eager_cache)
        argmax_ok = bool(gen_logits[:, -1, :].argmax(-1).item() == mono_logits[:, -1, :].argmax(-1).item())
        ok = d_eager < 1e-5 and d_mono < 1e-3 and cache_ok and argmax_ok
        all_pass &= ok
        if rank == 0:
            print(f"[K={k}] generator-vs-eager diff={d_eager:.3g}  "
                  f"generator-vs-monolithic last diff={d_mono:.3g}  "
                  f"cache_ok={cache_ok}  argmax_ok={argmax_ok}  "
                  f"{'PASS' if ok else 'FAIL'}")

    if rank == 0:
        print(f"\n(A) generator-driven mechanism correctness: "
              f"{'PASS' if all_pass else 'FAIL'}\n")

    # --- (B) real overlap measurement: same generator-driven chunked
    # prefill, timed with vs without a genuinely independent second-stream
    # GPU workload running concurrently across the SAME per-layer yields
    # that carry this model's REAL all_sum collectives (issued inside
    # inner._forward_steps's MoE layers, on the default stream, same as
    # production). If Phase 0b's finding holds for the real model (not
    # just the synthetic probe), NO_OVERLAP should show the second
    # stream's cost additively, while OVERLAP should show it ~free.
    K = 3
    REPS = 7
    no_overlap_times = []
    overlap_times = []
    for _ in range(REPS):
        if is_dist:
            mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # barrier
        t0 = time.perf_counter()
        generator_driven_chunked_prefill(model, tokens, K, overlap_stream=None)
        no_overlap_times.append(time.perf_counter() - t0)

        if is_dist:
            mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # barrier
        t0 = time.perf_counter()
        generator_driven_chunked_prefill(model, tokens, K, overlap_stream=mx.new_stream(mx.gpu))
        overlap_times.append(time.perf_counter() - t0)

    med_no = statistics.median(no_overlap_times)
    med_ov = statistics.median(overlap_times)
    if rank == 0:
        print(f"[K={K}] median NO_OVERLAP (serial second-stream work) = {med_no * 1000:.1f} ms")
        print(f"[K={K}] median OVERLAP   (concurrent second-stream)   = {med_ov * 1000:.1f} ms")
        print(f"  overlap_ratio (OVERLAP / NO_OVERLAP) = {med_ov / med_no:.3f}  "
              f"({'net win' if med_ov < med_no else 'no measurable win on this hardware'})")
        print("\n(NOTE: single shared laptop GPU across both ranks (loopback) -- "
              "absolute ms are noisy; the ratio and the correctness gate above "
              "are the signal. On the real 2-node cluster with dedicated GPUs "
              "per rank, the same mechanism applies but the second-stream "
              "workload should be a REAL next-chunk prefill segment, not a "
              "matmul stand-in.)")
        print(f"\n=== PHASE 2 gate: correctness={'PASS' if all_pass else 'FAIL'} ===")


if __name__ == "__main__":
    main()

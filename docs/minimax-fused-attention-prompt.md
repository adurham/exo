# Session prompt — Fused MiniMax attention Metal kernel (Phase 3)

Drop-in prompt to start a new session focused on the fused-attention
kernel project. Self-contained so the agent can resume without this
transcript.

> **Status as of 2026-04-24 (post-Week-3):** Phase 3 attempted three
> Python-level dispatch-fusion levers under `EXO_MINIMAX_FUSED_ATTN=1`
> and netted **+1.5 % cluster decode** (Week 2 fused QKV) plus a Week-3
> joined-RoPE wash (≈0 %). Local microbench confirmed the dispatch
> count drops to 17/layer with both fusions active, but per-dispatch
> wall-time savings on M4 Ultra are non-linear (single-dispatch saves
> register as 0 %). The **C++ MLX-patch path described below is now
> the only remaining +3–5 % lever.** See `docs/minimax-decode-
> optimization.md` § "Phase 3 outcomes" for the full scoreboard and
> what's already been ruled out.
>
> Things you DON'T need to redo:
> - dispatch-count instrumentation (mlx@22ef1101 ships
>   `mx.metal.dispatch_count()` behind `MLX_DISPATCH_COUNT=1`)
> - sub-op profiling (`bench/minimax_dispatch_count.py` +
>   `bench/minimax_rope_cache_breakdown.py`)
> - cluster A/B harness (`bench/minimax_cluster_ab.py`)
> - fused QKV scaffold (`src/exo/worker/engines/mlx/patches/minimax/`)
> - `MLX_SDPA_MAX_TG` analysis (resolved as identical to `BLOCKS=88` at
>   the cluster's tg_per_block; see memory)
> - The "0/62 layers installed" sharded-class trap — already fixed in
>   exo@116ad61f. `nn.QuantizedLinear` and
>   `mlx.nn.distributed.QuantizedAllToShardedLinear` are now both
>   recognised.

---

```
Starting Phase 3 of MiniMax decode optimization. Read these first:

- docs/minimax-decode-optimization.md — overall plan + all prior
  session outcomes. Section "Phase 3 — fused attention kernel" at the
  bottom is the scope for this session.
- docs/minimax-quantized-sdpa-design.md — Phase 2 quant-SDPA design;
  complete and cluster-neutral. Useful reference for how the existing
  quant SDPA kernel works (you'll be fusing around it).
- docs/fork-notes.md — carried mlx / mlx-lm fork state.
- docs/kv-cache-architecture.md — how QuantizedKVCache stores state
  (you'll be directly writing to this during the fused kernel).

Git state to resume from:

- adurham/exo@main = 6ae331fe (MLX_SDPA_BLOCKS passthrough + sub-span
  instrumentation landed this session)
- adurham/mlx@main = 1f6eb6bd (Q/V hoist + uint64 pack + contiguous
  access ports — cluster-neutral, local +26-29%)
- adurham/mlx-lm@main = 77ed380 (scaled_dot_product_attention_quant
  routing — unchanged since Phase 2)

Problem statement

Two sessions of SDPA kernel-compute optimizations produced 0% cluster
decode gains on 2-rank M4 Ultra TP + MiniMax-M2.7 at 66K context. The
NOOP sweep said SDPA = 66% of decode and the sub-span instrumentation
confirmed SDPA per-call = 3.8 ms (at trace-on) dominates attention.
But kernel-compute micro-optimizations never translate. Why:

An MLX_SDPA_BLOCKS sweep landed a +6.5% cluster decode win by cutting
dispatched threadgroup count 512 → 88. Sharp cliff at 92+ confirms the
cluster is dispatch-scheduling-bound, not compute-bound. bf16 KV vs
8-bit quant KV gives identical cluster tok/s (27.8 both ways), further
proof: 3× kernel-compute difference is invisible when dispatch cost
dominates.

The remaining cluster lever is dispatch-count reduction. Current
attention block issues ≥5 separate dispatches per layer:

  1. Q = q_proj(x)
  2. K = k_proj(x)
  3. V = v_proj(x)
  4. rope(Q), rope(K), cache.update_and_fetch(K, V) — 2-3 dispatches
  5. scaled_dot_product_attention_quant(Q, K_packed, V_packed, ...)
  6. o_proj(output)

× 62 layers = 310-434 dispatches per decode token just for attention.
On M4 Ultra with our measured dispatch-scheduling overhead, that's
where the cluster cost actually lives.

Goal

Write a single Metal kernel that replaces the attention block end-to-
end for MiniMax decode (q_seq_len=1). Takes x (layer input), does
RMSNorm + QKV projections + QK-norm (cross-rank all_sum is still an
MLX op, stays outside) + RoPE + KV-cache write + SDPA + o_proj, writes
residual output. One dispatch per layer instead of 5-7.

Scope constraints

- Decode only (q_seq_len = 1). Prefill keeps the existing multi-
  dispatch path; fused kernel's use_fallback gates it off when
  q_seq_len > 1.
- Quantized KV only at first (8-bit bits, group_size=64, head_dim=128,
  matches the cluster config). bits ∈ {4, 5, 8} × head_dim ∈ {64, 128}
  coverage is v2; v1 ships bits=8 × head_dim=128 and v2 fills in the
  grid later if v1 wins on cluster.
- Per-rank, single-batch. MiniMax's cross-rank collectives (QK-norm
  all_sum, MoE all_sum) stay outside the fused kernel. Fusing them in
  would require GPU-resident collectives, which MLX doesn't have.
- 2-rank tensor parallelism — each rank processes its own shard of
  heads. n_q_heads_per_rank = 24, n_kv_heads_per_rank = 4,
  head_dim = 128.

Prior art to study

These exist in the codebase and are partial templates (none are
directly portable — MiniMax has different head layout / no shared
expert / different QK-norm shape — but the kernel scaffolding
patterns are reusable):

- src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_fused_gqa_attention.py
  The full fused GQA attention scaffold for Qwen3.5. Shows the
  pattern for Q/K/V projection + SDPA + o_proj in one custom-
  primitive wrapper. Expects pre-sharded Q/K (post Phase 1). Read
  this to understand the host-side plumbing.
- src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_qk_rmsnorm.py
  Template for the QK RMSNorm kernel. Per-head, not joined. MiniMax
  needs joined-heads variant.
- mlx/backend/metal/kernels/sdpa_vector_quant.h
  The current quant SDPA kernel (what you're fusing around). Has Q
  pre-scale + V hoist + uint64 half-pack + contiguous-chunk access
  already in it.
- adurham/mlx@snapshot-2026-03-22-final has several experimental
  fused kernels that were tried and reverted. Read their commit
  messages and reverts (902eed46, 8572dee3) before going down those
  paths — "zero effect on M4 Max bandwidth, bottleneck is serial
  softmax chain" was the finding. We're not attacking compute, we're
  attacking dispatch count.

Pipeline

1. Week 1 — kernel skeleton. Start with existing `sdpa_vector_2pass_1_quant`
   and extend:
     - Accept input x (before Q/K/V proj) instead of Q/K/V arrays.
     - Accept Q/K/V weight matrices as kernel buffers.
     - Do the fused FMA for Q/K/V inside the same kernel before SDPA.
     - Accept o_proj weight, fuse the o_proj matmul after SDPA into the
       same kernel's output write.
     - Pass the cache's packed K/V arrays and write the new K/V tile
       inline (avoid the separate cache.update_and_fetch dispatch).
2. Week 2 — correctness. Unit tests against the reference
   (QKV proj → RoPE → cache update → SDPA → o_proj done as separate MLX
   ops). Tolerance: 2e-2 abs/rel for bf16 outputs.
3. Deploy to cluster. Bench at 50K context against the current
   blocks=88 baseline (27.86 tok/s). Expected win from dispatch-count
   reduction alone: +10-20% decode (33-34 tok/s at 50K). If <10%,
   dispatch overhead isn't ACTUALLY the fixed per-call cost we think
   it is — need to rethink.

Exit criteria

- Ship v1 if: cluster decode ≥ 30 tok/s at 50K context (+8% over
  current 27.86) AND token-level output matches the existing path
  within bf16 divergence budget AND no regression to prefill or any
  other workload.
- Abort if: 3 full cluster deploys show <5% cluster gain. The fixed-
  overhead analysis was wrong and we need diagnostic work instead.

Things NOT to try

- Kernel-compute micro-optimizations on the SDPA math itself. Two
  sessions of this pattern produced 0% cluster gain. (See memory
  `feedback_unquant_sdpa_prior_knowledge.md`.)
- mx.compile on the decoder. Validated 0.1% cluster-effective (see
  Exp 1 in minimax-decode-optimization.md).
- Further blocks-tuning beyond 88. Curve is sharp; 88 is the sweet
  spot and the +6.5% is already shipped.
- Speculative decoding variants for MiniMax — user has explicitly
  ruled this out (see memory `feedback_no_minimax_speculation.md`).
- CPU-assisted SDPA port. Cluster is dispatch-bound, not compute-
  bound, so Accelerate BLAS running in parallel doesn't help.

Cluster-deploy guardrails

- ASK before running start_cluster.sh. Each deploy disrupts Huihui
  scouts for 3-5 min and the SSH agent has been flaky.
- Local microbench first. Trust it only for ruling OUT kernel bugs,
  not for predicting cluster gains.
- After any cluster deploy, bench at 50K with Huihui ON (production
  config) + 5 warm runs. Blocks=88 is the live config.

Infrastructure that's in place

- Sub-span tracer: EXO_PROFILER=spans now produces per-op breakdown
  (attn.qkv_proj, attn.qk_norm, attn.reshape_rope_cache, attn.sdpa,
  attn.o_proj + moe.*). SIGUSR1 to the runner process dumps span
  stats. Use this to verify the fused kernel collapses multiple sub-
  span entries into one (or none).
- MLX_SDPA_BLOCKS env override: lets you A/B block counts per deploy
  without rebuilding mlx. Used for the sweep that landed blocks=88.
- NOOP_{ATTN,SDPA,MOE,ALLSUM} env gates in auto_parallel.py for
  bisection.
```

---

## Why this might still not work

The NOOP sweep conclusion that "kernel compute doesn't matter,
dispatch count does" is the strongest hypothesis we have — but it's
a hypothesis. It's possible the blocks=88 win comes from something
other than dispatch-count reduction (GPU-scheduler quantization, L2
cache behavior, something else). Before a multi-week kernel-fusion
project, consider a cheaper validation:

- Instrument the Python dispatch count per layer during a decode run
  (count `mx.eval` calls, graph-build calls, Metal command-buffer
  submits). If layer has e.g. 15 dispatches instead of 7, fusing
  them to 1 should give ~14/15 ≈ 93% speedup on that portion.
- Or port `0fe800bf` (`MLX_SDPA_MAX_TG`) properly and see if it's the
  same lever as `MLX_SDPA_BLOCKS` or a different one. If different,
  another cheap knob exists.

These are half-day experiments that could save the 1-2 weeks of
kernel work if the answer is different than we think.

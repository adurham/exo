# MiniMax-M2.7 decode optimization plan

Engineering plan for closing the MiniMax decode throughput gap. Built
from a span-level profile of the live cluster plus a deep-research sweep
of the MLX / Apple Silicon attention + MoE kernel landscape.

## Baseline (profiled 2026-04-22)

`mlx-community/MiniMax-M2.7-5bit`, 2-node tensor-parallel ring over
Thunderbolt-5 RDMA (jaccl), 66K context, 200-token decode.

Per-layer wall-time breakdown (via `EXO_MINIMAX_TRACE=1`, instrumentation
at `mlx-lm/mlx_lm/models/minimax_trace.py` + `ShardedMoE` /
`WrappedMiniMaxAttention` span points in
`src/exo/worker/engines/mlx/auto_parallel.py`):

| Span                | avg / call | % of wall-time |
| ------------------- | ---------- | -------------- |
| **attn**            | **12.2 ms**| **69.7 %**     |
| moe.switch_mlp      | 3.5 ms     | 20.1 %         |
| moe.all_sum         | 780 µs     | 4.5 %          |
| moe.router_topk     | 640 µs     | 3.7 %          |
| moe.weighted_reduce | 367 µs     | 2.1 %          |

Server-reported decode tps ≈ **17 tok/s**. Spans account for 299.5 s of
304 s server total ≈ 100 % — instrumentation is not leaking time.

The prior "62 RDMA all-reduces per token dominate" diagnosis was wrong;
see `memory/minimax_moe_decode_bottleneck.md` for the corrected story.

## Ranked levers

| #   | Lever                                                                                            | Effort    | Attn win     | Overall decode | Risk | Notes |
| --- | ------------------------------------------------------------------------------------------------ | --------- | ------------ | -------------- | ---- | ----- |
| **1** | **Kill Q/K `all_gather` via sharded-RMSNorm on concatenated heads**                           | **1 day** | **~15–25 %** | **~10–15 %**   | low  | Unblocks #3 |
| **2** | **Quantized SDPA kernel** (K/V dequantize in-register per 32-key tile)                         | **5–8 d** | **~30–45 %** | **~20–30 %**   | med  | Biggest single lever |
| 3   | Port the full 4-dispatch `batched_oproj_moe` pattern from `patches/qwen3_5_moe` (drop shared-expert paths) | ~1 week   | —            | +4–7 %         | med  | Gated on #1 |
| 4   | 5-bit-specific MoE kernel variants (fuse 5-bit unpack with FMA in SIMD registers)                 | 2–3 d     | —            | +1–2 %         | low  | Stacks on #3 |
| 5   | Fuse partial-RoPE + QK-norm + reshape into one kernel                                             | 2 d       | ~3–5 %       | ~2–3 %         | low  | Mostly subsumed by #1 |
| 6   | Tune `sdpa_vector_2pass` blocks heuristic (current 4096 threadgroups oversubscribed on 40-core M4) | 0.5 d     | ~3–8 %       | ~2–5 %         | low  | Trivial bench sweep |
| 7   | Fuse router: sigmoid + `e_score_correction_bias` + argpartition + renorm → one kernel            | 1–2 d     | —            | ~1–2 %         | low  | Cheap; paves way for #3 |
| 8   | `mx.compile` the decoder forward (shapeless for `cache.offset`)                                  | 1 d       | ~2–5 %       | ~1–3 %         | low  | Bounded win |

**Combined ceiling if #1 + #2 + #3 + #4 land: ~35–45 % decode speedup.**
17 → 23–25 tok/s at 66K context.

**Revised after Phase 1 landed (2026-04-23):** Lever #1's decode
contribution was 0 % (payload-shrinkage doesn't help at decode's L=1
payload — see phase-1-actual section below). The combined decode
ceiling drops to **~20–30 %** from levers #2 + #3 + #4 alone:
17 → 20.5–22 tok/s at 66K context. TTFT separately improved ~16 %
from Phase 1 and is banked. **Phase 2 (lever #2, quantized SDPA) is
now the sole remaining lever for decode throughput.**

**Revised again after Phase 2 shipped (2026-04-23, session 5):** Lever
#2's decode contribution at the baseline 5-bit KV config was ~0 %
(cluster measured 16.2 tok/s vs 17.0 tok/s baseline). The v1 kernel
wins the expected 1.3–1.5× at bits ∈ {4, 8} but loses ~10 % at
bits=5 × head_dim=128 because the MLX 5-bit pack factor forces
`qk_per_thread=8`, which halves active simd lanes. The bandwidth
analysis was correct in isolation but missed thread-occupancy loss at
the specific (bits, head_dim) the baseline profile actually runs.

A branch-free half-pack rewrite of the 5-bit path (mlx commit
`f784d2c3`) restored simd occupancy — microbench went from 0.96× to
1.19× vs dequantize+SDPA on the 66K production shape. **It still
landed at 0 % on the cluster.**

### Why kernel-level wins aren't translating: the NOOP sweep (2026-04-24)

After three consecutive 0 % cluster gains from kernel / dispatch-level
optimizations, we pivoted from "make kernels faster" to "figure out
where the budget actually lives". Added shape-preserving noop gates
(env-var-controlled) that bypass each major decoder section one at a
time, benched each against the locked baseline with Huihui scouts and
prediction-bot paused (MiniMax-only cluster, 5-bit KV, 5 warm runs per
config with ±0.4 s wall variance).

| Config         | Wall (s) | Decode tok/s | Share of decode       |
| -------------- | -------- | ------------ | --------------------- |
| Baseline       | 258.3    | **18.60**    | —                     |
| NOOP_ALLSUM    | 253.8    | 19.29        | RDMA = **3.7 %**      |
| NOOP_MOE       | 202.6    | 23.35        | MoE = **21 %**        |
| NOOP_ATTN      | 87.0     | 55.04        | attention = **66 %**  |
| NOOP_SDPA      | 87.1     | 54.87        | SDPA = **100 % of attention** |

**Per-token decode budget:**

| Section                                     | ms/token | % of 54 ms |
| ------------------------------------------- | -------- | ---------- |
| SDPA kernel (all of attention)              | 35.8     | **66 %**   |
| MoE (router + switch_mlp + reduce)          | 11.2     | 21 %       |
| Other (projections, norms, RoPE, KV update) | ~5       | 9 %        |
| RDMA collectives (2× per layer × 62)        | ~2       | 4 %        |

**The critical finding**: NOOP_ATTN ≈ NOOP_SDPA (55 vs 55 tok/s).
Q/K/V projections, RoPE, KV cache update, RMSNorm all contribute
**zero measurable cost** — the entire 35.8 ms/token attention budget
is the single SDPA kernel call per layer.

**Per-SDPA-call on cluster: 577 µs** (35.8 ms ÷ 62 layers).
Standalone microbench on M4 base: 1000 µs. Cluster is M4 Ultra (4×
more GPU cores), so pure kernel compute ≈ 250 µs/call. That leaves
**~327 µs/call of fixed per-dispatch overhead** that no kernel
compute optimization can touch. This is why Phase 2's 1.19×
microbench kernel win gave 0 % on cluster: 19 % of 250 µs = 47 µs
saved per call = 2.9 ms/token = ~5 % — entirely within the ±0.4 s
wall-time noise floor of a 5-run bench.

### Revised levers (post-NOOP sweep)

**Kernel-compute wins are capped at ~13 % decode** if we hit
bandwidth/compute floor on the 250 µs compute portion. Even a 2×
kernel (TurboQuant's claimed ceiling) is ≤ +25 % decode because the
327 µs fixed overhead doesn't move.

**What actually breaks 20 tok/s** requires attacking the 327 µs fixed
per-call overhead:

| Lever | Expected decode | Effort | Notes |
| ----- | --------------- | ------ | ----- |
| **`mx.compile` the decoder forward** | **up to +55 %** | 1–2 d | If the 327 µs is MLX graph-build + Metal dispatch overhead, compilation caches it away. Highest-expected-value move. |
| Batch multiple layers' SDPA into one kernel call | +10–20 % | 5–7 d | Cuts call count 62→N. Custom Metal kernel. Uncertain. |
| Reduce `blocks=1024` heuristic at 66K | +2–5 % | 0.5 d | 1024 intermediate arrays per call = allocator pressure. Tune down. |
| TurboQuant native SDPA (PR #3328) | ≤+25 % | 3–5 d | Only touches the 250 µs compute portion. Hard-capped by fixed overhead. |

**Phase 2 is complete and banked** — the kernel is correct, unit-
tested, fork-upstreamable. It's just not the decode lever on this
hardware. Next push should target `mx.compile` (the fixed-overhead
attack). Phase 3 (MoE fusion) is deferred: +4–7 % on paper but the
same fixed-overhead ceiling likely applies.

### Diagnostic session 2026-04-24: unquant-port + blocks sweep + Phase 3 pivot

Took one more swing at kernel-compute wins before accepting Phase 2's
conclusion: ported four optimizations from adurham/mlx upstream-fork's
unquant SDPA kernel (Q hoist, V hoist, bits=5 half-pack uint64 load,
contiguous-chunk access) into the quant kernel — all shipped in mlx
`1f6eb6bd`. Local microbench wins were real:

| bits | before | after | delta |
| ---- | ------ | ----- | ----- |
| 4    | 1363 µs | 940 µs  | −31 % |
| 5    | 1828 µs | 1300 µs | −29 % |
| 8    | 1278 µs | 950 µs  | −26 % |

**Cluster: 0 %** — same pattern as every prior kernel-compute
optimization.

Decomposed `attn` span into 5 sub-spans (qkv_proj, qk_norm,
reshape_rope_cache, sdpa, o_proj) via sub-span instrumentation in
`src/exo/worker/engines/mlx/auto_parallel.py:1041-1125` (exo commit
`6590fc46`). Confirms SDPA dominates attention at ~62 % of attn time
on cluster, other sub-spans in noise. NOOP-sweep analysis holds up.

**The one cluster win that did materialize: MLX_SDPA_BLOCKS=88.**

Added env-var override for the 2-pass `blocks` heuristic in mlx
`1f6eb6bd`; swept values on cluster:

| MLX_SDPA_BLOCKS | cluster decode tok/s |
| --------------- | -------------------- |
| 40              | 21.0 |
| 64              | 24.5 |
| 72              | 26.8 |
| 80              | 27.0 |
| **88**          | **27.84** |
| 92              | 23.5  (cliff) |
| 96              | 23.2 |
| 160             | 23.2 |
| 512 (default)   | 26.14 |

Sharp peak at 88, sharp cliff at 92. Explanation: M4 Ultra has ~80 GPU
cores × 4 simdgroup slots = ~320 concurrent TG slots. blocks=88 × 4
kv_heads = 352 TGs ≈ 1.1 dispatch rounds. blocks=92 × 4 = 368 TGs
crosses a GPU-scheduler quantization boundary (exact cause unclear)
and regresses sharply. Matches the unshipped mlx commit `21129617`
("cap SDPA 2-pass blocks to limit threadgroup scheduling on M3/M4").

Shipped as opt-in env var (exo commit `6ae331fe`, set
`MLX_SDPA_BLOCKS=88` before invoking `start_cluster.sh` for MiniMax
workloads). Not baked in as default because optimum is workload-
specific (different head counts / context lengths / hardware tiers
need different values).

**Also validated: bf16 KV cache (no quant) gives identical cluster
tok/s to 8-bit quant KV** — 27.79 vs 27.84 at blocks=88. The 3×
kernel-compute difference between bf16 and quant SDPA shown by local
microbench is invisible on cluster. This is the cleanest proof that
cluster is dispatch-scheduling-bound, not compute-bound. Ruled out
CPU-assisted SDPA port (snapshot-branch commit `a0580f93`) as an
option — it attacks GPU compute, which isn't the bottleneck.

**Post-session baseline:** 27.86 tok/s at 50K context with
`MLX_SDPA_BLOCKS=88` + Huihui scouts + MiniMax bits=8 KV. +6.5 % over
26.14 fresh-deploy baseline.

## Phase 3 — fused attention Metal kernel

**The only remaining cluster lever is dispatch-count reduction.** The
diagnostic work above proves kernel compute doesn't matter; only
dispatch scheduling does. Each layer currently issues 5–7 separate
Metal dispatches for attention (Q proj, K proj, V proj, RoPE + cache
update, SDPA, o_proj). × 62 layers = 310–434 dispatches per decode
token just for attention.

**Target:** one Metal kernel per layer that fuses {RMSNorm + QKV proj
+ QK-norm partial-SS + RoPE + KV cache write + SDPA + o_proj} into a
single dispatch. Cross-rank collectives stay outside — MLX has no
GPU-resident collective primitive.

**Expected win:** +10–20 % cluster decode if dispatch overhead really
is ~200 µs/call and we collapse 5 calls to 1. If <10 %, the fixed-
overhead model is wrong and we need deeper diagnostic work
(Instruments / Metal GPU capture) before any more kernel work.

**Effort:** 1–2 weeks, dedicated session. Significant Metal kernel
authoring. Unit tests and bench cycles against live cluster.

**Scope v1:** decode only (q_seq_len=1), 8-bit KV (matches cluster),
head_dim=128 (matches MiniMax M2.7). v2 fills in the bits × head_dim
grid once v1 ships a cluster gain.

**Full prompt for starting this as a new session:** see
[`minimax-fused-attention-prompt.md`](./minimax-fused-attention-prompt.md).

## Lever detail

### #1 — Kill the Q/K `all_gather`

`WrappedMiniMaxAttention.__call__` at
`src/exo/worker/engines/mlx/auto_parallel.py:876-901` concatenates Q and
K, `all_gather`s across ranks, applies `q_norm` / `k_norm`
(`RMSNorm(head_dim × num_attention_heads)` — over the joined 6144-wide
vector, not per-head), then splits back. The collective payload is
2 × (q_dim + k_dim) × bf16 bytes per token per layer.

MiniMax's qk-norm is mathematically a single RMSNorm over the joined
head-stack, but each rank owns contiguous heads. The partial
sum-of-squares is the only cross-rank quantity needed; every other step
is local. Swap the `all_gather` for an `all_sum` of a single fp32 scalar,
wrapped in a `ShardedRMSNorm`-style helper. The pattern exists in
`mlx-lm/mlx_lm/models/minimax.py:62` (`ShardedRMSNorm` /
`sharded_rms_norm`) — adapt it to Q/K.

Collective payload drops ~10,000×. Attention per-layer wall-time drops
~15–25 % because the `all_gather` is on the critical path every decode
step.

**Also unblocks lever #3**: the fused GQA kernel at
`src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_fused_gqa_attention.py`
expects pre-sharded Q/K (its `fused_qk_rmsnorm.py` does per-head norm,
not joined). Once #1 lands, MiniMax's norm reduces to "per-rank RMS over
joined local heads" and the kernel becomes portable with a different
reduction length (48/2 × 64 = 1536 vs Qwen3.5's 2048).

### #2 — Quantized SDPA kernel

**See [`minimax-quantized-sdpa-design.md`](./minimax-quantized-sdpa-design.md)
for the full Phase 2 design.**


Current hot loop is `_dequantize_then_sdpa` at
`mlx-lm/mlx_lm/models/base.py:117-132`: `mx.dequantize(*keys)` and
`mx.dequantize(*values)` on every SDPA call, every decode token.
At 66K context that re-materializes ~67 MB of K and ~67 MB of V per
layer per token. MLX has **zero quantized-SDPA Metal kernel** — `grep
quantized mlx/backend/metal/kernels` returns nothing.

The existing 2-pass decode kernel (`sdpa_vector_2pass` at
`mlx/backend/metal/scaled_dot_product_attention.cpp:711`) shards K along
the sequence into blocks; the template for 5-bit register decode already
exists in exo at
`src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_fused_gqa_projections_8bit.py`.

Target: a Metal kernel that accepts `(Q bf16, K_q uint8 group_size=64
bits=5, K_scales bf16, V_q uint8, V_scales bf16)` and dequantizes K/V
in-register per 32-key tile inside the 2-pass loop. Risk: 5-bit packing
layout must match `QuantizedKVCache`'s storage; head_dim=64 is a
first-class size in MLX so no template surprises there
(`sdpa_vector.h:43-46`).

Scales with context. Biggest single lever. Multi-day work but has a
direct code template in the fork.

### #3–#4 — MoE kernel work

`patches/qwen3_5_moe/batched_moe.py` already has the full 4-dispatch
fused pattern (o_proj + residual + RMSNorm + gate GEMV + softmax + topk
+ SwiGLU + down_proj + epilogue). Porting to MiniMax removes the
shared-expert register-sharing tricks (M2 has no shared expert,
`shared_intermediate_size=0`), changes gather indexing for 256/top-8,
and needs 5-bit kernel variants of `batched_softmax_topk_swiglu_Nbit`
and `batched_merged_down_proj_Nbit`.

Honest expected win: **+4–7 % overall decode** on top of the
gate+up-only fusion baseline (Qwen3.5 got +6 % with shared-expert
sharing; MiniMax has no shared expert to amortize against, but has 62
per-layer dispatches so kernel fusion savings scale linearly).
**+6 % is not the ceiling, but +15 % is not on the table either** —
~40 % of the 20 % `switch_mlp` slice is DRAM bandwidth on 5-bit weight
reads, which is physical and not recoverable via kernel fusion.

### #5–#8 — Smaller levers, evaluate after phase 2

All in the 1–5 % decode range. Defer decisions until the profile after
phase 2 shows what's left.

## Fork bugs to fix first

Forensic read of `mlx-lm/mlx_lm/models/minimax.py` surfaced two issues
worth fixing before any kernel work (~2 hours total):

1. **`ModelArgs.head_dim` defaults to `None`** → falls back to
   `hidden_size // num_heads = 64`. Released MiniMax-M2 has
   `head_dim=128`. The mlx-community config sets it explicitly so
   current loads work, but any sanitizer that strips fields would
   silently corrupt the model. Add an assertion at load time that
   `args.head_dim * args.num_attention_heads == q_proj.out_features`.
2. **`shared_intermediate_size` is a dead field** in `ModelArgs`. M2 has
   no shared expert; field is never referenced in the model body. Safe
   to delete. Leaving it creates an upstream-merge hazard where
   shared-expert logic could silently re-attach to a module with no
   weights.

## Dead ends — do NOT pursue

Validated against upstream + research:

- **Upstream MLX flash-attention #2955** — closed April 2026, no impl,
  no maintainer reply.
- **MLX PagedAttention #2228** — open since May 2025, zero Apple
  movement; single-request workload of size 1 doesn't need paging.
- **fp8 / int8 matmul attention compute** — M4 has no native fp8;
  simdgroup MMA is fp16/bf16 only.
- **NAX kernels / M5 "Neural Accelerators"** — wrong hardware (M4 only
  hits `sdpa_vector`, bypasses NAX entirely).
- **vllm-metal as a drop-in** — different runtime, doesn't speak exo's
  TP-over-jaccl; cherry-picking its kernel = lever #2 dressed
  differently.
- **turboquant-mlx direct adoption** — 3-bit compressed-domain is
  overkill when we're already at 5-bit KV.
- **MXFP4 / NVFP4 paths** — currently 60× slower than expected for MoE
  distributed per MLX issue #3402.
- **Megablocks / ScatterMoE / SonicMoE** — designed for batch ≥ 128 or
  Hopper-only; zero applicability to batch=1 decode on M4.
- **ANE offload (Orion)** — no MoE routing support, no quantization,
  GPT-2 scale only.
- **Persistent single-kernel MoE** — Metal threadgroup-memory budget
  (32 KB) can't hold 256 expert weight tiles.
- **JACCL PIPELINE / NUM_BUFFERS / BUFFER_SIZES tuning** — already
  A/B'd across 8+ commits Feb-Mar 2026 and reverted ("stable, RDMA not
  the bottleneck"). Shipped values are the measured winners.
- **Ring vs mesh at N=2** — `RingGroup` degenerates to same 1-hop
  exchange as mesh with extra chunking bookkeeping. Zero topological
  advantage.
- **Expert-parallel for 2-rank MoE** — JACCL has no all-to-all-v
  primitive; you'd be writing the collective + MLX binding + router
  reshard + placement logic to save <45 ms/token. Wrong project.
- **Async all-reduce overlap** — requires either GPU-resident
  `all_reduce` (not shipped; no Apple Silicon NCCL equivalent) or
  speculative QKV issue before collective resolves (changes model
  semantics). Research project, not a knob.
- **jaccl refactor PR #3412 cherry-picks** — #3412 is code-organization,
  not perf. Nothing to harvest.

## Recommended sequence

### Phase 1 — quick win + profile reset (~1.5 days)

1. Fix fork bugs (head_dim assertion, drop `shared_intermediate_size`) — ~2 h
2. Implement lever #1 (kill Q/K `all_gather`) — ~1 d
3. Re-profile. Confirm attn drops from ~70 % to ~55–60 %.

Exit criterion: decode tok/s at 66K context improves 10–15 % over
baseline. If no improvement, rethink — the profile is telling us
something we don't yet understand.

#### Phase 1 actual (landed 2026-04-23, commit `d47688b9`)

First attempt (two separate ShardedRMSNorm calls) regressed decode ~14 %
because it doubled the collective count per layer. Fused follow-up
(single `all_sum` of a `(B, L, 2)` combined partial-SS tensor) restored
one collective per layer and delivered:

- **Attn total wall-time: 208.7 s → 180.5 s (−14 %)**
- **TTFT: 255 s → 216 s (−16 %)** at 66K context
- Pure decode wall-time: unchanged within noise (48.8 s → 49.8 s)

The predicted decode win did not materialize. Cause: at L=1 decode,
both the original `all_gather` and the replacement `all_sum` have
negligible payload; per-call cost is ~800 µs of Metal↔CPU fencing
regardless of payload size. Only prefill, where the payload scales
with L, benefits. Lesson: do not model Apple-Silicon MLX collective
wins from payload shrinkage alone — fence count is the dominant
variable. All decode wins now depend on phase 2.

### Phase 2 — the big lever (~5–8 days)

4. Quantized SDPA kernel (lever #2).

Exit criterion: attn drops from ~55–60 % to ~30–40 % of wall-time and
decode tok/s clears the phase-1 ceiling by another 20–30 %.

### Phase 3 — MoE fusion (~1 week), only if #2 lands

5. Port 4-dispatch `batched_oproj_moe` to MiniMax (lever #3).
6. 5-bit kernel variants (lever #4).

Exit criterion: `moe.switch_mlp` drops from ~20 % toward ~10 % of
wall-time.

### Skip phases

Levers #5–#8 unless re-profiling after phase 2 reveals something
unexpected.

## Budget summary

Original (pre-Phase-1-measurement) estimates:

- **Phase 1**: 1.5 days → +10–15 % decode.
- **Phase 1 + 2**: ~10 days → +30–45 % decode.
- **Phase 1 + 2 + 3**: ~3 weeks → +35–50 % decode.

**Revised after Phase 1 landed:**

- **Phase 1 actual**: ~0 % decode (TTFT −16 %, banked but orthogonal).
- **Phase 2 alone**: 5–8 days → +20–30 % decode (unchanged estimate).
- **Phase 2 + 3**: ~2–3 weeks → +25–35 % decode.

17 tok/s → 21–23 tok/s at 66K context is the honest ceiling. 17 → 25+
requires lever #2 to land at the upper end of its range AND Phase 3
to stack cleanly on top, which we won't know until Phase 2 ships.

## Key file pointers

- `mlx-lm/mlx_lm/models/minimax.py:62` — `sharded_rms_norm` (adapt for lever #1)
- `mlx-lm/mlx_lm/models/minimax.py:90-166` — unsharded `MiniMaxAttention`
- `mlx-lm/mlx_lm/models/base.py:117-132` — **the dequantize-before-SDPA hot loop (lever #2 target)**
- `src/exo/worker/engines/mlx/auto_parallel.py:856-932` — `WrappedMiniMaxAttention` (**lever #1 target**)
- `src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_moe.py:99-187` — 4-dispatch pattern (**lever #3 template**)
- `src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_fused_gqa_attention.py` — existing fused-GQA scaffold (portable to MiniMax after #1)
- `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_qk_rmsnorm.py` — template for MiniMax norm kernel
- `mlx/backend/metal/scaled_dot_product_attention.cpp:711` — `sdpa_vector_2pass` (lever #2 base)
- `mlx/backend/metal/kernels/sdpa_vector.h` — the decode Metal kernel (394 lines)

## Research sources

### MLX / Apple Silicon ecosystem

- [Proposal: FlashAttention / PagedAttention in MLX #2955](https://github.com/ml-explore/mlx/issues/2955) — closed, no impl
- [PagedAttention integration in MLX #2228](https://github.com/ml-explore/mlx/issues/2228) — stale
- [MLX PR #2078 `gather_qmm_rhs`](https://github.com/ml-explore/mlx/pull/2078) — prefill-only MoE speedup
- [MLX issue #3402 MXFP4 MoE 60× regression](https://github.com/ml-explore/mlx/issues/3402)
- [Apple ML Research — Exploring LLMs with MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Hmbown/ZMLX MoE kernel toolkit](https://github.com/Hmbown/ZMLX)
- [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx)
- [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)

### MiniMax-M2 specifics

- [Upstream config](https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/config.json)
- [Upstream modeling code](https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/modeling_minimax_m2.py)
- ["Why did M2 end up as a full-attention model?"](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) — confirms no Lightning Attention hybrid
- [MTP weights missing issue #47](https://github.com/MiniMax-AI/MiniMax-M2/issues/47)

### Papers referenced (for context, not direct ports)

- Apple Silicon EP for MoE inference: [arxiv 2506.23635](https://arxiv.org/html/2506.23635v1)
- Open-TQ-Metal compressed-domain attention: [arxiv 2604.16957](https://arxiv.org/html/2604.16957)
- SonicMoE (Hopper, not portable): [paper](https://hf.co/papers/2512.14080)

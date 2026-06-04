# DSv4 MoE: fused routed-experts Metal kernel (gate+up+SwiGLU+down at top-K)

> **NO-GO (May 14 2026, phase-1 spike result):** killed by the dispatch-
> overhead microbench BEFORE any Metal was written. Pipelined 43-layer chain
> measured at 207 us/layer vs 187 us memory-bandwidth floor -- only 20 us/layer
> of dispatch-overhead headroom = 860 us total = 3% throughput. Below the 1.5x
> decision gate. The +52% MoE-NOP win is memory-bandwidth-bound, not dispatch-
> bound. See references/distributed-bottleneck-attribution.md for the full
> postmortem. **Do not re-attempt without re-running the pipelined microbench
> first.** Real next levers: c=2 amortization, INDEX_TOPK reduction, attention-
> path NOPs at 100K to find the other 4-6ms of token wall.

---


**Goal:** Cut DSv4 per-token MoE wall on the 2-Studio cluster from the current
~3 ms-equivalent contribution (the gap between baseline 35 t/s and MoE-NOP
53.2 t/s at 8K, 29.47 -> 41.2 t/s at 100K) by fusing the routed-experts hot
path into a single Metal dispatch instead of 3 separate gather_qmm calls plus
sort/unsort/post-combine overhead.

Demonstrated headroom (May 14 2026, venv-synced NOP probes):
- 8K c=1: 35.0 -> 53.2 t/s (+52%) when entire MoE is bypassed
- 100K c=1: 29.47 (champion) -> 41.2 t/s (+40%) when MoE bypassed

We won't recover ALL of that (we still need correct numerical output), but
even half -- getting from 29.5 -> ~35 t/s at 100K -- would be a major win.

---

## Current state (what the kernel must replace)

DeepseekV4MoE._raw_local (mlx-lm/mlx_lm/models/deepseek_v4.py:976-988):

    def _raw_local(self, x, input_ids):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, scores = self.gate(x, input_ids)         # (1) routing
        y = self.switch_mlp(x, inds)                   # (2) routed experts
        shared_out = self.shared_experts(x)            # (3) shared experts (dense)
        return _moe_post_combine(y, scores, shared_out)  # (4) weighted sum

The expensive call is (2) self.switch_mlp(x, inds). At c=1, decode token:
- x: shape (1, 1, 4096) bf16
- inds: shape (1, 1, 6) int32 -- 6 active expert indices
- 256 routed experts, each is gate_proj (4096 -> 2048) + up_proj (4096 -> 2048)
  + down_proj (2048 -> 4096), all mxfp4 g32 packed
- Each rank holds HALF the experts after TP sharding, so 128 experts/rank,
  output_dim halved to 1024 per matmul
- After _install_dsv4_fused_gate_up (default-on via EXO_DSV4_FUSED_MOE=1),
  gate_proj and up_proj weights are pre-concatenated as _fused_w_gu so we do
  ONE gather_qmm for both (output shape (..., 2*hidden_dim/N_rank)), then
  down_proj is still a separate gather_qmm call. 2 dispatches per layer
  instead of 3 -- already shipped as the May 13 baseline.

Today's dispatch sequence per token per layer (rank 0 or rank 1, c=1):
1. gate(x, input_ids) -> top-6 indices + scores  (few microseconds)
2. mx.gather_qmm(x, _fused_w_gu, ...) -> (1, 1, 6, 2*1024) fused up||gate
3. activation(x_up, x_gate) -> SwiGLU (limited) -> (1, 1, 6, 1024)
4. mx.gather_qmm(...) -> down_proj -> (1, 1, 6, 4096)
5. _scatter_unsort -> reorder if sorted (skipped at c=1, top-6 < threshold)
6. .squeeze(-2) -> (1, 1, 6, 4096) retained for post-combine
7. shared_experts(x) -> dense FFN (1 hidden, hidden*n_shared=2048*1)
8. _moe_post_combine(y, scores, shared_out) -> sum(y * scores[..., None]) + shared

For one decode token across 43 layers * 2 dispatches (gate||up, down) +
shared_experts (3 dispatches) + sort/scatter overhead = ~250 Metal kernel
launches per token just for MoE. At ~20-30 us Metal launch overhead per
dispatch, that's ~5-7 ms of dispatch-only overhead per token -- consistent
with the ~3 ms wall savings we see when MoE NOP fires.

---

## Proposed approach

One Metal kernel call per layer that does ALL of:
- Read the 6 active expert indices from inds
- Gather 6 expert weight tiles from _fused_w_gu (gate||up) -- mxfp4 packed
- Compute fused (gate*up) matmul -> SwiGLU -> down -> weighted-sum-with-scores
- Add shared_experts output (also computed in-kernel or fused in via a
  separate small dispatch)
- Output: (1, 1, hidden_dim) ready for the layer's all_sum

Kernel name: dsv4_routed_experts_topk (or similar). Lives in
mlx-lm/mlx/backend/metal/kernels/dsv4_routed_experts.metal and is bound via
mx.fast.metal_kernel from Python -- same pattern as mlx-lm/mlx/mlx/fast/qmm.cpp's
gather_qmm.

### Why fusion wins

1. Eliminate 5 of 6 dispatch boundaries. Today: 2 gather_qmm + 1 activation
   + 1 down_proj + 1 sort/scatter + 1 post_combine. After: 1 dispatch. At ~25 us
   launch overhead per dispatch (Metal command-buffer commit), that's
   ~125 us/layer * 43 layers = 5.4 ms/token saved on launch overhead alone.

2. Keep intermediates in registers / threadgroup memory. Current path
   materializes (1, 1, 6, 2*1024) then (1, 1, 6, 1024) then (1, 1, 6, 4096) in
   GPU global memory. Fused kernel keeps these in shared memory across the
   pipeline stages -- saves ~50 KB of HBM round-trip per layer.

3. Co-issue gate weight reads. Gate is a tiny (1, 4096) @ (4096, 256) matmul
   that costs <50 us but blocks all subsequent work. Embedding the gate inside
   the same kernel lets it issue against the same threadgroup that then proceeds
   to the routed-expert gather.

### What stays outside the fused kernel

- The cross-rank mx.distributed.all_sum -- MUST stay outside so the
  post-allreduce fence (mx.eval) can fire (per dsv4_v4block_compile_2026_05_08.md
  -- collapse to 7-8 t/s if all_sum lives inside the compile boundary)
- Layer norm before / residual add after -- those are in DeepseekV4Block,
  outside the MoE call
- Shared experts: probably keep outside as a separate mx.compile'd block,
  unless we want to also fuse them into the same kernel. Decide based on
  whether they're on the critical path (they're tiny so probably not).

### What's hard about it

1. mxfp4 unpacking inside Metal. Weights are stored as 4-bit packed with bf16
   scales per group of 32. Each kernel-side dequant is ~10 SIMD instructions.
   There's reference code in mlx's quantized.metal -- qmm_n and qmm_t kernels
   already do this for non-gathered case; gather_qmm.metal does it for gathered
   case. We can copy the dequant helpers verbatim.

2. Top-K is data-dependent. The kernel needs to read 6 expert indices then
   conditionally gather from 128 experts. Can't unroll. But 6 is small enough
   that a per-thread for k in 0..6 loop is fine.

3. SwiGLU with swiglu_limit. DSv4 uses a LIMITED SwiGLU:
   min(swiglu_limit, silu(gate)) * up. Cheap arithmetic but needs the limit
   constant passed in.

4. Scores broadcast. The post-combine step is sum(y * scores[..., None], axis=-2).
   scores has shape (1, 1, 6) (per-expert weight). Inside the kernel, each
   thread for a hidden_dim element multiplies the 6 partial-down outputs by
   the 6 scores and sums.

---

## Step-by-step plan

### Phase 1 -- Spike (1-2 days, validate the kernel idea works)

Goal: prove that a single fused Metal kernel beats the current 2-dispatch
SwitchGLU on a representative shape, single-node, no cluster.

1. Write bench/dsv4_routed_experts_microbench.py that:
   - Builds _fused_w_gu and down_w mxfp4-packed weights at production shapes
     (hidden=4096, inter=2048, n_experts=128 per rank, bits=4, group_size=32,
     top-K=6, batch=1)
   - Runs the current path: 2x gather_qmm + activation + scatter + post-combine
   - Runs an mx.fast.metal_kernel candidate kernel that does the fused path
   - Verifies numerical equivalence (||a - b|| < 1e-3 in bf16) and reports wall
     time (median of 1000 iters) for each
   - Run on m4-1 via SSH: ssh adam.durham@192.168.86.201 'cd ~/repos/exo &&
     uv run python3 bench/dsv4_routed_experts_microbench.py'

2. Iterate on the kernel until it's >= 2x faster than the current 2-dispatch
   path on the microbench. Target: <50 us/call vs current ~150 us/call.

3. Decision gate: if the spike shows < 1.5x microbench speedup, abandon the
   kernel and pivot to a different lever (kernel-level perf is high-cost,
   low-EV if the dispatch overhead model is wrong). If >= 1.5x, continue to
   phase 2.

### Phase 2 -- Integration into DSv4 (2-3 days)

1. Add DSv4FusedRoutedExperts class in
   src/exo/worker/engines/mlx/auto_parallel.py (alongside
   FusedDeepseekV4SwitchGLU). Its __call__(x, indices, scores) returns the
   post-combined output, replacing both switch_mlp(x, inds) AND
   _moe_post_combine(y, scores, shared_out) calls.

2. Modify _install_dsv4_fused_gate_up (auto_parallel.py:1305+) to ALSO
   pre-flatten down_proj weights and rebind switch_mlp.__class__ to
   DSv4FusedRoutedExperts. Gated by EXO_DSV4_FUSED_ROUTED=1 (off by default
   so we can A/B without rolling back).

3. Update DeepseekV4MoE._raw_local in mlx-lm/mlx_lm/models/deepseek_v4.py to
   detect the fused-routed class and skip the explicit _moe_post_combine step
   when it's in use.

4. Make sure the mx.compile boundary in install_compiled_forward
   (deepseek_v4.py:956+) re-traces correctly with the new class.

### Phase 3 -- Cluster validation (1 day)

1. Push commits to adurham/mlx-lm + adurham/exo. Bump submodule.

2. Force uv sync + scp the venv copy on both nodes (per the verification
   recipe in references/distributed-bottleneck-attribution.md).

3. Bench progression:
   - 8K c=1: target 35 -> 45+ t/s. Bench: bench/concurrent_bench.py
     --concurrency 1 --iterations 3 --warmup 1 --max-tokens 128
     --prompt-words 100. With EXO_DSV4_FUSED_ROUTED=0 to confirm baseline,
     then =1 to measure delta.
   - 100K c=1 MTP-off: target 29.47 -> 33-35 t/s. Bench:
     bench/concurrent_bench.py --concurrency 1 --iterations 1 --warmup 0
     --max-tokens 256 --prompt-words 75000 --timeout 900. ~7 min wall.
     Compare to champion champion-2026-05-13-29.47.

4. Quality gate: at 8K c=1 with EXO_DSV4_FUSED_ROUTED=1, verify model output
   is coherent (not garbage like the NOP probes produce). Use a 3-shot fixed-
   prompt comparison vs baseline output at temp=0 -- diff should be empty
   (numerical-equivalence with mxfp4 packing). If the fused gather_qmm matches
   numerically, the SwiGLU+post-combine fusion will too provided we don't
   introduce a different reduction order.

5. If 100K wins: tag champion-<date>-<tps> on both repos; update
   ~/.hermes/scripts/rollback_to_champion.sh to point at new tag; update
   memory with new champion bookmark.

6. If 100K doesn't win (e.g. 100K stays at 29.47 even though 8K improves):
   investigate -- most likely culprit is that at 100K context the bottleneck
   shifts from MoE compute to sparse-attention/indexer, and MoE savings get
   masked. Run sparse_attn NOP at 100K to test that hypothesis before
   declaring defeat.

---

## Files likely to change

mlx-lm submodule (/Users/adam.durham/repos/exo/mlx-lm/):
- mlx_lm/models/deepseek_v4.py -- _raw_local updated to use the fused class
  when installed. ~10 line delta.

exo repo (/Users/adam.durham/repos/exo/):
- src/exo/worker/engines/mlx/auto_parallel.py -- add DSv4FusedRoutedExperts
  class (~80 lines) next to existing FusedDeepseekV4SwitchGLU. Update
  _install_dsv4_fused_gate_up to also flatten down_proj + rebind to the new
  class when EXO_DSV4_FUSED_ROUTED=1. ~30 line delta.
- src/exo/shared/constants.py -- add EXO_DSV4_FUSED_ROUTED env constant
  (default 0).
- start_cluster.sh -- propagate EXO_DSV4_FUSED_ROUTED env to workers.

The Metal kernel itself -- likely options:
- Option A (preferred): inline mx.fast.metal_kernel(...) Python literal in
  auto_parallel.py, defining the .metal source as a string constant. No build-
  system changes; mlx JIT-compiles at first call. Same pattern as
  mlx-lm/mlx_lm/models/deepseek_v4.py:_indexer_score did before refactor.
- Option B: new file
  mlx/mlx/backend/metal/kernels/dsv4_routed_experts.metal + corresponding
  mx.fast.gather_swiglu_routed C++ binding in mlx/mlx/fast.cpp. More work,
  but cleaner and lets us use mlx's existing metal-kernels compilation
  pipeline. Defer to phase 2 if Option A wins spike.

New file:
- bench/dsv4_routed_experts_microbench.py (~150 lines) -- phase-1 spike.

---

## Tests / validation

Numerical equivalence:
- bf16 element-wise diff ||a_fused - a_unfused|| / ||a_unfused|| < 1e-3 on
  random inputs across 100 seeds. mxfp4 packing introduces ~1e-3 noise
  inherently so this matches existing op tolerance.
- Greedy-decoding token sequence match: prompt "Count: 1, 2, 3," at temp=0 for
  50 tokens, fused vs unfused output must be byte-identical. This is the same
  character-of-test as the May 13 indexer-refactor validation that caught the
  bf16 boundary tie issue.

Perf gates (the success criteria):
- Phase 1 spike: >= 1.5x microbench speedup on single-node m4-1 vs the current
  2-dispatch path. <50 us/call vs current ~150 us/call.
- Phase 3 cluster: >= +10% agg_tps at 100K c=1 MTP-off vs champion 29.47.
  Below +10% is not worth the maintenance cost of a custom kernel.

Regression tests:
- The existing bench/full_moe_microbench.py and bench/gather_qmv_microbench.py
  continue to pass at champion-equivalent numbers when
  EXO_DSV4_FUSED_ROUTED=0.
- The cluster bench at 8K c=1 with EXO_DSV4_FUSED_ROUTED=0 shows the same
  35.0 t/s as champion. (Off-by-default means existing path is untouched.)

---

## Risks, tradeoffs, open questions

Risks:

1. Metal-kernel JIT compile overhead at first call. mlx caches compiled
   kernels at process scope, so this only matters at warmup. Should add <=1
   second to cluster startup. Mitigated by exo's existing warmup_inference
   call that runs a dummy forward before bench starts.

2. The dispatch-overhead model could be wrong. I'm estimating ~25 us per
   Metal command-buffer commit based on prior cluster traces, but Apple
   Silicon's command-buffer scheduling has been opaque historically. If
   actual launch overhead is ~5 us not 25 us, the dispatch-fusion savings
   would only be ~1 ms instead of ~5 ms per token. Phase 1 spike validates
   the model before committing to phase 2.

3. Numerical drift accumulation across 43 layers. Even a 1e-4 per-layer delta
   could compound to 4e-3 by the lm_head and shift greedy-decoded tokens.
   Mitigation: match the reduction order (sum across top-K with scores, NOT
   sum within each expert's down output first). This is the character-of-bug
   that bit me on the May 13 indexer-refactor -- the bench passed unit tests
   but produced different tokens at temp=0 boundary cases.

4. Cluster c=2 regression. This optimization is c=1-focused. At c=2 the per-
   token wall is lower because the same MoE invocation amortizes across 2
   tokens. A fused kernel might not fuse as effectively when handling 2 tokens
   (the top-K indices differ between tokens, breaking some of the gather
   coalescing). Test phase 3 at c=2 100K and verify no regression below the
   existing c=2 champion.

5. MTP-on regression. The MTP self-spec path also calls MoE. Verify
   EXO_DSV4_MTP=1 benches don't regress.

Tradeoffs:

- Custom kernel = maintenance burden. mlx upgrades can break the kernel's
  assumed mxfp4 packing layout. Need to keep the kernel narrowly scoped to
  DSv4-Flash-8bit specifically and version-pin mlx in pyproject.toml. Or:
  guard the kernel behind EXO_DSV4_FUSED_ROUTED=0 default so users can opt
  out if it breaks.

- Option A (inline literal) vs Option B (mlx/fast binding). Option A ships
  faster but is harder to test in isolation and uglier in the codebase. Start
  with A for the spike; if it wins, optionally promote to B in a phase-4
  cleanup.

Open questions (to answer during the spike):

1. Is gather_qmm's sorted_indices=True fast path actually faster at top-K=6?
   Upstream sets sort_threshold=64 (skipped at K=6), exo's
   FusedDeepseekV4SwitchGLU sets it to 8 (still skipped at K=6). Does forcing
   sort at K=6 help? Probably not (sort overhead > 6-element gain), but worth
   a quick A/B in the microbench.

2. Can we batch the gate matmul into the same kernel? Gate is (1, 4096) @
   (4096, 256) = 1 MB of weights. Fitting all 256 expert-routing weights in
   threadgroup memory is tight but plausible. Defer to phase 2 unless spike
   shows gate is on the critical path.

3. Co-issue shared_experts? Shared experts are dense (not routed), 1 expert
   at (4096, 2048). Could be fused into the kernel's "free" SIMD time while
   it waits on memory. Defer to phase 2; benchmark spike with and without
   first.

4. Should we co-locate experts to avoid TP sharding entirely? From the
   earlier session: routed experts at mxfp4 = ~147 GB, exceeds single-node
   128 GB RAM. So no -- but we could co-locate the gate (small) and only TP
   the experts themselves.

---

## Rollback path

Champion tag still valid: champion-2026-05-13-29.47 on both adurham/exo and
adurham/mlx-lm repos. Rollback script at
~/.hermes/scripts/rollback_to_champion.sh. The new flag defaults OFF
(EXO_DSV4_FUSED_ROUTED=0) so even if the code lands and breaks, the default
behavior is unchanged from champion.

---

## Why this is the right next move

The May 13 + May 14 instrumentation work proved three things:

1. The cluster is GPU-compute bound, not collective-bound. all_sum is ~1 ms
   / 33.9 ms = 3% of decode wall at c=1 100K. The collective levers (overlap,
   quantize, fewer all_sums, expert co-location) all have <1 ms recoverable
   upside.

2. MoE is the dominant per-layer compute. MoE NOP wins +40% at 100K (29.47
   -> 41.2 t/s), confirming MoE compute is the single biggest shrinkable item
   on the per-token critical path.

3. Python-level fusion (FusedDeepseekV4SwitchGLU, EXO_DSV4_COMPILE_FFN,
   compiled forward) is already deployed and we're still at 35/29.47. The
   remaining MoE wall is in the dispatch-overhead and intermediate-
   materialization terms that ONLY a kernel-level fusion can eliminate.

This is exactly the lever-2 ("custom Metal kernels") from the original
optimization framing. The data now justifies the investment.

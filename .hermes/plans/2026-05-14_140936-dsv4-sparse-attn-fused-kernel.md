# DSv4 sparse_attn fused Metal kernel - gather + SDPA in one dispatch

## ABANDONED 2026-05-14 — phase-1 spike failed gate

Phase 1 microbench (bench/dsv4_sparse_pooled_microbench.py) on m4-1
at production shape (B=1, H=32, L=1, D=512, K=160, sw=128, P=25000)
showed pipelined 21-call speedup of:

  - gather-fused + Apple SDPA: 1.07x (path B)
  - full-fused metal_kernel:    1.23x (path C, numerically correct)

Gate was >= 1.7x. Both below. Path C is even below the 1.3x marginal floor.

Numerical equivalence of Path C is fine: norm-relative error 4.4e-3
(bf16-ULP level). The kernel works correctly. It is just not fast
enough to matter at the cluster level.

WHY:
The pipelined sparse_attn wall on single-node m4-1 is only 1.12 ms total
for all 21 calls. Even collapsing it to zero would save 1.12 ms/token =
~4% throughput. The +31% NOP gain (~3 ms/token) does NOT live in local
kernel dispatch overhead - same lesson as the MoE plan: MLX already
pipelines Metal dispatches at the chain level, so the 5-dispatches-x-21-
layers-x-25-us theory was wrong.

The +31% NOP win lives in CLUSTER pipeline fence behavior:
  - skipping the sparse path skips KV cache writes (memory BW)
  - removes a sync point that gates next-layer compute
  - the inter-rank all_sum traffic compounds across 21 layers
A local kernel cannot recover that because it was not the bottleneck.

PIVOT to one of:
  - c=2 amortization study (higher batch amortizes KV read pressure)
  - structural compress_ratios reshape (halve number of sparse layers)
  - lever 4: reduce memory bandwidth (expert co-location, quant downgrade)

## Goal

Cut per-token sparse-attention wall on the 2-Studio cluster at long context (100K) by fusing the multiple dispatches inside `_sparse_pooled_attention()` L=1 fast path into one Metal kernel.

Demonstrated headroom (May 14 2026, venv-synced NOP probes at 100K c=1 MTP-off, INDEX_TOPK=160, EXO_SPECULATIVE=0):

- BASELINE: 29.2 t/s
- sparse_attn NOP: 38.3 t/s (+31.2%)
- indexer NOP: 33.4 t/s (+14.4%)
- indexer + sparse_attn NOP: 39.5 t/s (+35.3%)

This is a DISPATCH-OVERHEAD-BOUND lever (NOT memory-bandwidth-bound like MoE). The compute per layer is small (~25 KB gather + ~288-position SDPA per head). There are 21 layers per token going through this path at 100K context.

Unlike the MoE plan (killed by phase-1 spike showing memory-bound ceiling), this kernel ISNT fighting against memory bandwidth - the working set per call fits in L2.

## Current state (what the kernel must replace)

SparseCompressedAttention.__call__ at 100K c=1 L=1 MTP-off runs this sequence per layer (mlx-lm/mlx_lm/models/deepseek_v4.py:1612+):

1. q projection + norm + RoPE -> q shape (B=1, H=64, L=1, D=512)
2. kv projection + norm + RoPE -> kv shape (B=1, 1, L=1, D=512), appended to local_cache
3. pooled = self.compressor(x, comp_cache, offset) -> (B=1, P, D=512) where P ~= 25000 at 100K
4. pmask = comp_cache.make_mask(L, offset)
5. topk = self.indexer(...) -> (B=1, L=1, k=160) int32
6. Branch (sparse path):
   sparse_mask = mx.take_along_axis(pmask, topk, axis=2)
   out = _sparse_pooled_attention(q, kv, pooled, topk, mask, sparse_mask, scale, sinks)
7. Inside _sparse_pooled_attention() at L=1 (line 715+):
   - GATHER: pooled_gathered = mx.take_along_axis(pool, topk, axis=3)  -> (B, 1, k, D)
   - CONCAT: combined_kv = mx.concatenate([local_kv, pooled_kv], axis=2)  -> (B, 1, sw+k, D)
   - MASK: combined_mask = mx.concatenate([lm, pm], axis=-1)
   - SDPA: mx.fast.scaled_dot_product_attention(q, combined_kv, combined_kv, ...)

Thats 4 Metal dispatches per layer + 1 upstream sparse_mask gather = 5 dispatches.
5 dispatches x 21 layers = ~105 dispatches per token from this path alone.
At ~25 us per Metal dispatch boundary, that is ~2.6 ms/token of pure launch overhead.
The wall we save (29.2 -> 38.3 t/s) implies ~2.4 ms of recoverable per-token wall.
Numbers line up - this IS the dispatch-overhead bottleneck.

## Shape sanity (verify before writing kernel)

For DSv4-Flash-8bit on the cluster (TP-sharded across 2 ranks):

- B (batch): 1
- H (q heads, per rank): 32  (num_attention_heads=64 / 2 ranks)
- L (q sequence): 1
- D (head_dim): 512  (MLA latent - q and kv share this)
- K (top-K): 160  (INDEX_TOPK)
- sw (sliding window): 128
- P (pool size): ~25000 at 100K context
- sw+K (combined SDPA length): 288

The combined SDPA is q(1, 32, 1, 512) @ K^T(1, 1, 288, 512) per rank - small enough that the kernel is compute-light. The expensive part is the per-layer launch overhead summed over 21 layers.

## Proposed approach

ONE mx.fast.metal_kernel per layer that does:

Inputs:
- q: (B, H, L, D) bf16
- local_kv: (B, 1, sw, D) bf16
- pooled: (B, P, D) bf16 (the full pool, NOT pre-gathered)
- topk: (B, L, K) int32
- local_mask: (B, H, L, sw) bool or bf16
- pmask: (B, L, P) bool (NOT pre-gathered to sparse_mask)
- sinks: (H,) bf16
- scale: float

Output:
- out: (B, H, L, D) bf16

Computation per (b, h, l) threadgroup:

1. Load q[b, h, l, :] into registers across D//tg_size threads.
2. Pre-load topk[b, l, :] into threadgroup memory.
3. Phase A - local SDPA: compute scores against local_kv[b, 0, 0..sw, :].
4. Phase B - sparse SDPA: for k in 0..K: gather pooled[b, topk[k], :], score, apply pmask[b, l, topk[k]].
5. Online softmax with sinks (FlashAttention pattern, tracks max + sum-exp across both phases).
6. Weighted V-sum across both phases (V == K in MLA, same buffer).
7. Write out[b, h, l, :].

The gather happens INSIDE the kernel. No mx.take_along_axis pre-materialization, no concat. Saves 4-5 dispatches per layer.

### Why this works where the MoE kernel didnt

- MoE was MEMORY-bound at the chain level (3.2 GB read / 400 GB/s = 8.0 ms floor; pipelined chain was 8.93 ms = 90% of floor). Dispatch fusion saves ~20 us/layer max = 3% throughput.
- sparse_attn is DISPATCH-OVERHEAD-bound. Per-call data movement is ~50 KB; per-call wall is ~115 us. ~115 us / call - 0.125 us memory = ~114 us is dispatch + launch + sync. Fusing 5 dispatches into 1 has direct computable upside.
- The SDPA compute itself is already optimal (Apples mx.fast.SDPA). We dont replace it - we just stop paying dispatch boundaries between gather, concat, mask-concat, and SDPA.

### What stays outside the kernel

- mx.distributed.all_sum (cluster fence)
- q/kv projections, RoPE, layer norms
- local_cache.update_and_fetch
- self.compressor(x, comp_cache, offset)
- self.indexer(...) [defer to phase-4]

### Whats tricky

1. MLA shared K/V (D=512, K==V). Gather once, reuse for both score and value matmul.
2. Mask formats - local_mask may be bool or fp; pmask is bool. Kernel template param.
3. Attention sinks - learned per-head bias added to softmax denominator. Reference math at deepseek_v4.py:629+ (_split_softmax).
4. Online softmax across two phases (FlashAttention pattern). At sw+K=288 should fit in tg memory without 2-pass.
5. TP sharding: H=32 per rank, kernel runs symmetric.

## Step-by-step plan

### Phase 1 - Microbench spike (1-2 days, validates kernel CAN win)

1. Write bench/dsv4_sparse_pooled_microbench.py:
   - Build inputs at production shape (B=1, H=32, L=1, D=512, K=160, sw=128, P=25000) bf16
   - Run current path (the L=1 fast-path branch from _sparse_pooled_attention line 715-770)
   - Implement candidate mx.fast.metal_kernel fused version
   - Verify numerical equivalence (||diff|| < 2e-2 over 50 seeds)
   - Time both at p50 over 1000 trials
   - Run on m4-1: ssh adam.durham@192.168.86.201 'cd ~/repos/exo && uv run python3 bench/dsv4_sparse_pooled_microbench.py'

2. CRITICAL: PIPELINED MICROBENCH (the lesson from the MoE plan):
   - Chain 21 calls in a single eval (different topk per call, same KV pool).
   - Compare TOTAL pipelined wall for unfused vs fused.
   - This is the number that matters for the cluster - per-call microbench can lie if mlx is already pipelining the dispatches.

3. Iterate kernel until >= 1.7x faster than current chain on PIPELINED microbench. Target: total chain wall <40 us/call vs current ~115 us/call.

4. Decision gate:
   - >= 1.7x pipelined speedup: proceed to phase 2. Cluster math: 21 layers x ~50 us savings = 1.05 ms/token = ~3.5 t/s at 100K = 29.2 -> ~32.5 t/s. Above 10% gate.
   - 1.3x-1.7x: marginal. Consider fusing indexer too (phase 4) and re-microbench before phase 2.
   - <1.3x: abandon. Pivot to (a) c=2 amortization, (b) compressed_attn similar treatment (+13% smaller upside, reuse infra), or (c) structural change (reduce 21 sparse layers via compress_ratios reshape).

### Phase 2 - Integration (2-3 days)

1. Add _sparse_pooled_attention_fused() in mlx-lm/mlx_lm/models/deepseek_v4.py next to _sparse_pooled_attention_inner. Inline .metal source as Python string constant (Option A from MoE plan).

2. Modify _sparse_pooled_attention() L=1 branch:
   if L == 1 and bool(os.environ.get("EXO_DSV4_SPARSE_FUSED", "")):
       return _sparse_pooled_attention_fused(q, local_kv, pooled, topk, local_mask, pmask, scale, sinks)
   # else existing concat+SDPA path

3. Wire EXO_DSV4_SPARSE_FUSED through start_cluster.sh.

4. Kernel takes pmask directly (NOT pre-gathered sparse_mask), eliminating the upstream take_along_axis dispatch too.

### Phase 3 - Cluster validation (1 day)

1. Commit + push adurham/mlx-lm + adurham/exo. Submodule bump.

2. Hot-sync the venv (per venv-trap recipe in distributed-bottleneck-attribution.md skill ref) - both nodes. CRITICAL: every iteration loop on the kernel requires either git commit + push + uv sync, or scp into both venvs. Skipping this step burns whole sessions on misleading results.

3. Bench progression:
   - 8K c=1 with EXO_DSV4_SPARSE_FUSED=0 and =1: both should be ~35 t/s (sparse path dormant at 8K).
   - 100K c=1 MTP-off (the real test): target 29.2 -> 33+ t/s. Bench command: concurrent_bench.py --concurrency 1 --iterations 1 --warmup 0 --max-tokens 256 --prompt-words 75000 --timeout 900. ~7 min wall.
   - 100K c=2 MTP-off: verify no regression vs c=2 champion.

4. Quality gate: byte-identical greedy decoding at temp=0 vs baseline at 8K (sparse dormant); allow <=5 token divergence over 50 tokens at 100K.

5. Decisions:
   - +10% or better: tag new champion. Update rollback_to_champion.sh + memory.
   - 0% to +10%: keep code, flag off by default.
   - Regression: roll back submodule, leave kernel on experiment/sparse-fused branch.

## Files likely to change

mlx-lm submodule:
- mlx_lm/models/deepseek_v4.py - new _sparse_pooled_attention_fused (~150 lines: Python wrapper + .metal source string), ~10-line branch in _sparse_pooled_attention. Existing path untouched.

exo repo:
- src/exo/shared/constants.py - EXO_DSV4_SPARSE_FUSED env constant (default 0).
- start_cluster.sh - propagate to workers.

New file:
- bench/dsv4_sparse_pooled_microbench.py (~200 lines).

## Tests / validation

Numerical:
- ||a_fused - a_unfused|| < 2e-2 over 50 seeds at production shape.
- Token-sequence match at temp=0 over 50 tokens: byte-identical at 8K (sparse dormant), <=5 divergence at 100K.

Perf gates:
- Phase 1: >= 1.7x PIPELINED microbench speedup.
- Phase 3: >= +10% agg_tps at 100K c=1 MTP-off vs champion 29.47.

Regression:
- 8K c=1 (FUSED=0 or =1) = champion 35 t/s.
- c=2 100K MTP-off no regression.
- MTP-on unaffected (L=1 fast path only).

## Risks, tradeoffs, open questions

Risks:

1. Apples mx.fast.SDPA may already be faster than any hand-rolled kernel for this shape. Phase-1 spike answers. Mitigation: keep gather inline, fuse around SDPA not replace.

2. Attention sinks numerics - cannot bit-match Apples kernel easily. Compare token IDs across seeds in spike.

3. Bool vs additive mask handling - template param or wrapper coercion.

4. Same-character-as-MoE failure mode: per-call microbench can lie. Phase-1 MUST run the 21-call PIPELINED microbench. Per-call alone is insufficient.

5. Indexer integration deferred to phase-4.

Tradeoffs:

- Custom kernel maintenance burden. Gated behind off-by-default flag.
- L=1-only. MTP-on / batched prefill miss the win. Acceptable (cluster runs MTP-off canonically).
- Per-rank H=32 might be too few threads for single-threadgroup approach. Validate in phase 1.

Open questions answered during spike:

1. How much does gather alone cost vs SDPA?
2. Can we eliminate sparse_mask gather by handling pmask inside kernel? (Almost certainly yes; include phase 2.)
3. Fuse indexer into same kernel? Defer to phase-4.
4. compressed_attn similar treatment? Smaller +13% upside; reuse infra if phase 3 wins.

## Rollback path

- Champion tag champion-2026-05-13-29.47 on adurham/exo + adurham/mlx-lm.
- Rollback script ~/.hermes/scripts/rollback_to_champion.sh.
- Flag defaults OFF so default behavior unchanged from champion.

## Why this is the right target NOW

May 14 attention-NOP sweep proved:

1. sparse_attn at 100K is the #1 non-MoE lever (+31.2%) and #2 overall after MoE (+41% but bandwidth-bound). MoE kernel was correctly killed by pipelined microbench.

2. sparse_attn is DISPATCH-OVERHEAD-bound, not memory-bandwidth-bound. Per-call ~50 KB data vs ~115 us wall = 99% is launch/sync overhead. Kernels are the right tool.

3. The path has a clean fast-path branch (L=1) already isolated at _sparse_pooled_attention line 715-770. No model refactor - just a new branch under an env-gated flag.

Expected delivery: 29.2 -> 33+ t/s at c=1 100K MTP-off, +10% champion bump, ~1 week of work.

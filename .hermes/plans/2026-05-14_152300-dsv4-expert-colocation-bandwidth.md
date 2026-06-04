# DSv4 expert co-location for memory-bandwidth reduction (ABANDONED at phase-0)

## TL;DR

The +41% MoE-NOP gain at 100K c=1 (29.2 -> 41.2 t/s) is memory-bandwidth-bound,
not dispatch-overhead-bound (the MoE kernel plan proved that). Real lever is to
reduce per-token expert-weight read traffic. Route histogram (89K prefill tokens,
2026-05-14) shows PER-LAYER expert concentration is strong: top-32 experts per
layer captures 56.4% of routings on average. Co-locating those experts on both
ranks (avoiding cross-rank fetch) saves a meaningful fraction of the 3.35 GB/token
of expert reads.

Expected delivery: 29.2 -> 33-36 t/s at c=1 100K MTP-off, depending on what
"co-location" actually means architecturally in exo's MoE sharding code.


## ✗ PIVOT REQUIRED — phase-0 read showed Lever A doesn't apply

Reading auto_parallel.py + mlx-lm switch_layers.py revealed:

- SwitchLinear weights are shaped `(num_experts=256, output_dims, input_dims)`.
- `all_to_sharded_linear_in_place` uses `_all_to_sharded` which returns
  `max(weight.ndim - 2, 0)` — for a 3-D weight that is **axis 1 (output_dims)**,
  NOT axis 0 (num_experts).
- Both ranks hold ALL 256 experts. What is split is hidden_dims=2048: each rank
  computes half the intermediate width of every routed expert.
- `down_proj` is sharded sharded-to-all (axis = -1, input_dim), so it ALSO
  spans both ranks. After `down_proj`, an `all_sum` reconciles the partial
  output activations across ranks.

This means:

1. **No cross-rank expert FETCH happens**. Both ranks already have all
   weights locally. Per-token expert reads are paid ONCE per rank, not
   reduced by replicating top-K experts. Lever A is a no-op architecturally.

2. **Per-token expert read bytes per rank** are actually about half of what
   the plan assumed:
     - per expert: 12.98 MB
     - per layer: 6 experts × 12.98 MB = 77.88 MB **÷ 2 (output-dim shard) = 38.94 MB**
     - per token: 43 layers × 38.94 MB = **1.67 GB/rank/token**
     - at 400 GB/s memory BW: 1.67 / 400 = 4.2 ms theoretical floor
     - actual fwd_build at 100K is 28.5 ms with `eval_block` 0.01 ms
     - GPU% reported as 87-88%

3. **The +41% MoE-NOP gain is NOT explained by per-rank read traffic** alone.
   At 4.2 ms floor vs 28.5 ms wall, MoE compute + the all_sum fence is
   dominating, not pure memory bandwidth. The NOP also skips a Metal
   dispatch barrier that the all_sum-with-eval-fence imposes.

## What this means for available levers

Re-cast against the actual architecture:

### Lever A' (revised): NOT applicable
Expert co-location for cross-rank fetch reduction makes no sense — there is
no cross-rank fetch. Skip.

### Lever B: Mixed-precision per-layer cold-expert quant — STILL VIABLE
The routing data DOES show 60-116 cold experts per layer in layers 14-42.
Each cold expert contributes ~38.94 MB / 256 = 152 KB to the per-token,
per-rank read amortization. If we drop cold experts from mxfp4 (0.5156 B/elem)
to mxfp3 (0.40 B/elem ≈ 22% smaller) or zero them out entirely with a
fallback to shared_experts:
  - Per-layer save: 60-116 cold experts × 152 KB × 0.22 ≈ 2-3 MB/layer
  - Per-token save: ~85-130 MB read traffic (5-8% of current 1.67 GB)
  - Expected throughput gain: ~3-5%
  - Quality risk: NON-trivial. Cold experts being cold doesn't mean unused —
    they fire 10-30% of mean. Lowering their precision will perturb decode.

### Lever C (NEW): exo cluster the WHOLE all_sum out — replicate experts
**This is the real lever the data justifies.** If we accept extra memory
cost to REPLICATE (un-shard) the gate_proj/up_proj/down_proj of EACH
expert across both ranks, then:
  - Each rank runs the full expert locally (no output_dim split).
  - The all_sum after MoE becomes UNNECESSARY (output is already complete).
  - At 12.98 MB × 256 experts × 43 layers × 2 (replicate across ranks) =
    **285 GB extra**. NOT FEASIBLE — exceeds 128 GB DRAM per rank.

  But: replicating ONLY the top-K experts per layer:
  - top-32 per layer × 43 layers × 12.98 MB × 2 (replicate) = **35.7 GB extra
    per rank** for the replicated copies. Each rank ALSO keeps the existing
    sharded versions of all 256.
  - When a token routes to a top-32 expert: use local replicated copy, SKIP
    the per-output-dim shard + all_sum cycle for that expert.
  - When a token routes to a non-top-32 expert (43.6% of routings): fall back
    to the existing sharded + all_sum path.
  - **Critical**: the MoE all_sum is ONE collective for the whole layer's
    routed output. If even ONE of the 6 routed experts goes through the
    sharded path, the all_sum still has to happen. So we save the all_sum
    ONLY when ALL 6 routed experts for the token are in the local top-32.
  - Per the histogram, P(all 6 routed experts in top-32 per layer):
    rough upper bound = (0.564)^6 ≈ 3.3% (assuming independence, which is
    optimistic). LOW.

### Lever D (NEW): reduce all_sum frequency
EXO_DSV4_FENCE_EVERY_N_LAYERS is already set to 43 (max). This is
already optimized. Not a new lever.

### Lever E (NEW): kv_lora_rank-aware sharding
DSv4 has kv_lora_rank=512 and q_lora_rank=1536. The MLA latent KV is
relatively small. Worth seeing if the bandwidth from KV reads is actually
the bigger lever than MoE. Per the May 14 NOPs:
  - MoE NOP: +41% (memory-bound MoE)
  - sparse_attn NOP: +31% (cluster fence behavior, not bandwidth-bound)
  - compressed_attn NOP: +13%
  - indexer NOP: +14%
  Total non-MoE attention savings: ~58%
  Indicates BOTH MoE and attention are large contributors. Not just MoE.

## Recommended pivot

**Drop expert co-location plan. Go after a different lever.**

The phase-0 finding invalidates the bandwidth-reduction premise. Options:

1. **Pivot to c=2 amortization bench** (queued task #5). Higher batch
   amortizes expert reads naturally. Simple to measure, no code changes.

2. **Pivot to compress_ratios structural reshape** (task #6). Reduce 21
   sparse-attn layers to 10 via the compress_ratios pattern.

3. **Accept the diminishing returns**. Champion is 29.47 t/s, MoE-NOP
   ceiling is ~41 t/s, sparse_attn ceiling adds another ~10%. Combined
   theoretical ceiling ~45 t/s at current architecture. The remaining
   gap may require fundamental changes (MTP-on, different sharding).

4. **Lever B (mixed-quant cold experts)** if 3-5% is worth the quality
   risk. Could work but small and risky.

5. **Re-probe with proper decode-only sample**. Current data is dominated
   by prefill which may have different routing than steady-state decode.
   Want to confirm before any major change.

## Probe results (the data this plan rests on)

`tools/dsv4_route_hist_summary.py` on 100K prefill (rank 0, m4-1, run on
2026-05-14, EXO_DSV4_ROUTE_HIST=1):

  Per-expert weight size: 12.98 MB (mxfp4 g32, 3 matrices per expert)
  Per-token routed-expert reads: 6 experts/layer x 43 layers = 3.35 GB/token

  GLOBAL concentration (top-K hottest experts across all layers):
    top-8:   5.0% of routings
    top-32:  18.3%
    top-64:  33.6%
    top-128: 60.0%
    --> Global hot set is WEAK. Co-locating "the K hottest experts overall"
        across all layers does not help much.

  PER-LAYER concentration (top-K hottest IN EACH LAYER):
    top-8:  23.8% avg
    top-16: 37.8% avg
    top-24: 48.2% avg
    top-32: 56.4% avg
    top-48: 68.6% avg
    --> Per-layer hot set is STRONG. Each layer has its own hot subset.

  Cross-layer reuse (which experts are hot in MULTIPLE layers):
    Experts in top-32 of  2+ layers: 250 of 256
    Experts in top-32 of  5+ layers: 157 of 256
    Experts in top-32 of 10+ layers:   6 of 256
    Experts in top-32 of 20+ layers:   0 of 256
    --> Most experts are useful SOMEWHERE but few are universally hot.
        ~5 experts are hot in ~25% of layers (E153, E250, E171, E83, E141).

  Late-layer dead weight:
    Layers 14-42 have 60-116 COLD experts (<10% of mean) each.
    Layer 40 has 116 cold experts of 256. Almost half the layer is dead.
    --> Some experts are essentially never used in some layers.

## Architectural context (what "co-location" can mean here)

exo currently runs MoE with experts sharded across the 2 ranks (TP-style sharding
over the n_routed_experts axis). The setup at start_cluster.sh + auto_parallel.py
uses `SwitchGLU` with experts distributed (need to verify exactly how — that's
phase-0 of this plan). The all_sum after MoE reconciles per-rank partial outputs.

Possible co-location levers (ordered by impact):

  LEVER A: **REPLICATE the per-layer top-K experts on BOTH ranks.**
    For each layer, the K most-used experts are stored on both nodes.
    When a token routes to a top-K expert, BOTH ranks have it locally —
    no cross-rank fetch needed. Cold experts stay sharded as today.
    Cost: K * 43 * 12.98 MB = K * 558 MB extra memory per rank.
    At K=32: +17.85 GB per rank. Currently each rank uses ~75 GB DRAM
    (model fits comfortably in 128 GB).

  LEVER B: **EVICT cold experts entirely (mixed precision per layer).**
    Quantize the K hottest experts at current mxfp4 (full quality), and
    the cold experts at mxfp3 or below. Halves the read bandwidth for
    cold-expert tokens. Independent of co-location — could stack.

  LEVER C: **RANK-AFFINITY ROUTING (pre-permute the layout per-rank).**
    Rather than replicating, RENAME expert IDs per-rank so that each
    rank holds a different optimal subset. The gate output is per-token
    so we can't dynamically split. But we CAN pre-compute a per-layer
    permutation that puts the "rank-0-preferred" experts physically
    contiguous on rank-0's shard. Reduces remote fetches via locality
    rather than replication.

LEVER A is the simplest, biggest-impact starting point. LEVERs B and C are
follow-ons if A delivers but doesn't close the gap.

## Goal of phase-1 (this plan)

Implement and validate LEVER A: per-layer top-32 expert replication on both ranks.
Quantify the actual bandwidth and throughput savings against a reproducible
microbench AND on the cluster at 100K c=1 MTP-off.

## Step-by-step plan

### Phase 0 — Understand current MoE sharding (0.5 day)

Required reading before any code change:
  1. `src/exo/worker/engines/mlx/auto_parallel.py:_install_dsv4_fused_gate_up`
     and surrounding code. Find where SwitchGLU's `experts` parameter is split.
  2. `mlx-lm/mlx_lm/models/deepseek_v4.py:DeepseekV4MoE.__init__` — note the
     `self.switch_mlp = SwitchGLU(...)` line and `self.sharding_group`.
  3. mlx-lm's `mlx_lm/models/switch_layers.py` — see how SwitchGLU dispatches
     tokens to experts. Specifically: is the all_sum on the OUTPUT of all
     experts (sharded then reduced), or is there cross-rank gather BEFORE
     each expert call?

If sharding is "each rank owns half the experts and runs its half locally,
then all_sum at end" — then the bandwidth IS recoverable by ensuring both
ranks have the hot ones. The all_sum still happens but the per-rank work
becomes proportional to the hot-expert hit rate.

If sharding is "every rank computes every expert it gets" (replicated weights,
no sharding) — then we're already paying for the full 3.35 GB/token on each
rank and the lever is different (would be: skip cold experts entirely or
lower-quant them, which is Lever B).

Output of phase 0: a 1-paragraph note in this plan stating WHICH sharding
mode is active. This determines whether to proceed with Lever A as written,
or pivot to Lever B.

### Phase 1 — Microbench (1 day)

`bench/dsv4_moe_colocation_microbench.py`:
- Build a synthetic SwitchGLU at production shape (256 experts, hidden=4096,
  mid=2048, mxfp4 g32).
- Generate a synthetic routing pattern matching the captured histogram (e.g.,
  use the actual L20 distribution — strong top-32 = 67% concentration).
- Time three configurations:
  (a) baseline: 256 experts, 6 routed per token, all on one rank
      (single-node measurement of per-token expert read cost)
  (b) sharded: 128 experts per rank, cross-rank all_sum at end (today's
      cluster baseline simulated on a single node — gives us the
      shared-bandwidth budget number)
  (c) co-located: 128 experts per rank PLUS top-32 of each layer replicated
      on both ranks (simulate the lever).

The decision gate: **pipelined 21-layer chain throughput** (matching MoE
microbench from May 14 lesson — per-call alone lies).

Decision gate: >= 10% pipelined chain speedup over (b). Below that, Lever A
doesn't recover enough bandwidth to be worth integrating.

### Phase 2 — Cluster integration (2-3 days)

If phase-1 wins:

1. Modify `src/exo/worker/engines/mlx/auto_parallel.py` to support a new
   sharding mode "shard-plus-replicate":
   - For each layer, identify the top-32 experts from a saved histogram
     (load from /tmp/dsv4_route_hist/ at cluster startup).
   - Place those 32 on BOTH ranks; place the remaining 224 sharded
     112-per-rank as today.
   - At MoE forward time: for each token, check if any of its 6 routed
     experts are in the local top-32 (per layer). If yes: compute locally,
     skip cross-rank fetch for that expert. If no: fall back to today's
     all-sum path.

   This will require code that knows expert IDs per-rank, which the
   current shard-by-axis code probably doesn't handle cleanly. Phase-0
   reading tells us how invasive this is.

2. Env-gated behind `EXO_DSV4_COLOCATE_TOPK=N` (default 0 = off). Set to
   32 for the new behavior. Existing path untouched when N=0.

3. Histogram source: ship `tools/dsv4_route_hist_summary.py` to write a
   sidecar JSON of per-layer top-K expert IDs into the model dir at
   profile-collection time. Worker reads it at startup.

### Phase 3 — Cluster validation (1 day)

1. Commit, push, scp to both venvs (per venv-trap recipe in the skill).
2. Bench progression:
   - 8K c=1 with `EXO_DSV4_COLOCATE_TOPK=0`: champion baseline = 35 t/s.
   - 8K c=1 with `EXO_DSV4_COLOCATE_TOPK=32`: should be identical or
     slightly better (8K context fits fewer total expert calls so the
     lever is dormant).
   - 100K c=1 MTP-off with `=32`: target 29.2 -> 33+ t/s (+10% gate).
     Stretch: 35+ t/s.
   - 100K c=2 MTP-off: verify no regression vs c=2 champion.
3. Quality gate: byte-identical greedy decode at temp=0 with `=0` vs `=32`
   on 8K (lever dormant). Up to 5 token divergence at 100K is acceptable.

4. If +10% gate met: tag new champion. If marginal (0-10%): keep code,
   default off, doc the win. If regression: roll back, debug.

## Files likely to change

- `tools/dsv4_route_hist_summary.py` — ADD `--out-topk-json model_dir/topk.json` mode
- `src/exo/worker/engines/mlx/auto_parallel.py` — new "shard-plus-replicate"
  sharding mode for routed experts
- `mlx-lm/mlx_lm/models/deepseek_v4.py` — minimal: thread the per-rank
  expert-ID map to SwitchGLU
- `mlx-lm/mlx_lm/models/switch_layers.py` — possibly: a fast-path that skips
  the cross-rank all_sum for tokens whose 6 routed experts are all local
- `src/exo/shared/constants.py` — `EXO_DSV4_COLOCATE_TOPK` constant
- `start_cluster.sh` — propagate env var
- `bench/dsv4_moe_colocation_microbench.py` — phase-1 spike

## Risks, tradeoffs, open questions

**Risk 1: Memory.** K=32 = 17.85 GB extra per rank. Each rank currently uses
~75 GB DRAM. Headroom is OK (128 GB total, ~50 GB free) but tight. A K=48
config would push to 26.78 GB extra = ~102 GB used, leaves only 26 GB for
KV cache. At 100K c=1 KV cache size is ~10 GB (kv_lora_rank=512, head_dim=128,
fp8 KV). At c=2 it doubles. Plan stays at K=32; may want K=24 for safety.

**Risk 2: Routing-pattern drift.** The histogram captured today is for ONE
specific prompt (the needle-in-haystack 75K-word probe). Other workloads
(code, math, dialogue) may route differently. Mitigation: collect
histograms for 3-5 diverse 100K prompts, take the UNION of top-K per layer.
This grows the replicated set somewhat but stays bounded.

**Risk 3: Bench-vs-real-traffic skew.** The bench-probe uses one prompt
class; production may be different. Mitigation: make the topk-json file
hot-swappable so production traffic can re-tune over time.

**Risk 4: Same character-of-failure as MoE / sparse_attn kernel plans.** The
local per-rank wall reduction may not translate to cluster-level throughput
because the slowest rank still gates the next layer (FIFO comm queue).
Mitigation: profile per-rank GPU% before/after; only ship if BOTH ranks see
the time reduction. This is the same trap that killed the prior two plans.

**Risk 5: The bandwidth model may be wrong.** The plan assumes per-token
expert read traffic is 3.35 GB and that the lever halves the cross-rank
component. But if the actual bottleneck is NOT the per-rank expert read
(if it's the cross-rank fetch for non-local experts via an in-network
gather), the savings calculation changes. Phase-1 microbench resolves this.

**Open question 1:** How does the current SwitchGLU shard experts? Phase-0.

**Open question 2:** Does the all_sum at MoE output ALREADY include
cross-rank expert weight fetches, or only the per-expert OUTPUT activations?
If it's just outputs, the cross-rank bandwidth is small (4096 floats per
token = 8 KB) and our entire lever is wrong. Phase-0 also answers this.

**Open question 3:** Should we also pursue LEVER B (mixed-precision quant)?
Reading the data: layers 14-42 have 60+ cold experts. Quantizing those at
mxfp3 (~25% smaller) saves ~3.3 GB per rank of memory and ~25% of per-token
read bandwidth on cold-expert tokens. May stack with Lever A.

## Why this is the right target NOW

- May 14 attention-NOP sweep + MoE-NOP sweep proved MoE at 100K is the
  biggest non-attention lever (+41%) and that it's MEMORY-bandwidth-bound,
  not dispatch-overhead-bound (MoE kernel NO-GO confirmed this).
- May 14 sparse_attn fused kernel ABANDONED at phase-1 spike (1.23x pipelined,
  needed >=1.7x). Same character-of-failure: local kernel savings can't
  recover cluster-level cost.
- ROUTE_HIST probe on 2026-05-14 produced concrete, quantitative routing
  data. Per-layer concentration is strong (top-32 = 56% of routings) —
  unlike global concentration which is weak.
- The architectural change is small: a new sharding mode in auto_parallel.py
  + a config knob. No new kernels needed. No model surgery. Reversible by
  env var default-off.

## Rollback path

- Champion tag: `champion-2026-05-13-29.47` on `adurham/exo` and
  `adurham/mlx-lm`. Rollback script `~/.hermes/scripts/rollback_to_champion.sh`.
- Lever defaults OFF (`EXO_DSV4_COLOCATE_TOPK=0`). Even with code merged,
  unchanged default behavior equals champion behavior.

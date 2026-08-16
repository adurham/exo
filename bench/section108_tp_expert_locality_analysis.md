# Section 108: Can expert co-location reduce TP cross-node traffic for DSv4?

## Verdict (Q2, the crux)

**Expert co-location CANNOT reduce TP's cross-node all_sum traffic for DSv4, because the
all_sum is a fixed-size reduction of the full (tokens × hidden) activation tensor that
happens on every layer regardless of which experts fired or where they live.** DSv4's TP
sharding is **not** an expert-parallel (EP) scheme with token routing to expert-owning
ranks — it's classic Megatron-style tensor-parallel sharding of each expert's weight
matrices along the intermediate dimension. There is no "which rank owns which expert"
concept in the compute path at all; every rank computes a **partial FFN result for every
active expert** and the ranks are summed together. Locality of an expert can't reduce
network traffic because there is no per-expert dispatch/gather over the network — the
communication volume is a function of `(batch × seq_len × hidden_size)` only, never a
function of routing decisions.

## Q1: How are the 256 experts split across the 2 ranks? — CONFIRMED

They are **not split at all** — every rank holds all 256 experts. What's sharded is each
expert's **weight matrix**, along the intermediate (FFN hidden) dimension, the standard
Megatron column/row-parallel MLP pattern applied per-expert:

- `DeepseekV4ShardingStrategy.shard_model` (`src/exo/worker/engines/mlx/auto_parallel.py:1074-1080`):
  ```
  self.all_to_sharded_linear_in_place(layer.ffn.switch_mlp.gate_proj)
  self.sharded_to_all_linear_in_place(layer.ffn.switch_mlp.down_proj)
  self.all_to_sharded_linear_in_place(layer.ffn.switch_mlp.up_proj)
  ```
  (same for `shared_experts`). This applies to **all 256 routed experts' weights
  identically on both ranks** — there is no per-expert routing of which weights live on
  which rank.
- The generic sharding predicates (`auto_parallel.py:718-736`, forwarded into MLX's
  `shard_inplace`/`_shard` in `.venv/.../mlx/nn/layers/distributed.py:38-73`) slice
  `gate_proj`/`up_proj` on `max(weight.ndim - 2, 0)` (the output/intermediate axis) and
  `down_proj` on `-1` (the input/intermediate axis). For `SwitchLinear`/`QuantizedSwitchLinear`
  weight shape `(num_experts=256, output_dims, input_dims)` (confirmed via
  `mlx-lm/mlx_lm/models/switch_layers.py:42-46,100-103`), slicing on `ndim-2` /
  `-1` slices the **output_dims / input_dims axis, not the num_experts axis (axis 0)**.
  So each rank ends up with **all 256 experts, each at half intermediate width**
  (e.g. gate/up_proj: `(256, inter/2, hidden)`; down_proj: `(256, hidden, inter/2)`).
- This is confirmed structurally by the MTP block sharding loop
  (`auto_parallel.py:1121-1126`), which applies the **identical** `all_to_sharded`/
  `sharded_to_all` calls to `mtp.ffn.switch_mlp` — no per-expert-index logic anywhere.
  (A stray code comment at `auto_parallel.py:1113-1117`, "each rank holds half the
  experts," is **stale/inaccurate** — it describes a mental model the sharding code does
  not implement. Flag for cleanup, but do not treat it as evidence.)
- Contrast: `DeepSeekShardingStrategy` (V3/V3.2/KimiK2) uses the same
  `all_to_sharded`/`sharded_to_all` primitives for its MoE — same width-sharding pattern,
  not expert-axis partitioning. This is architecturally consistent across all
  Megatron-TP-sharded MoE models in this codebase; DSv4 is not special-cased toward
  expert-per-rank assignment.

## Q2: Is the all_sum data-dependent on expert placement? — CONFIRMED: NO, fixed-size

Traced `DeepseekV4MoE.__call__` (`mlx-lm/mlx_lm/models/deepseek_v4.py:2604-2833`):

1. `sum_gradients(sharding_group)(x)` on the **input** (`:2718-2719`) — VJP-only hook, no
   forward-pass traffic.
2. `inds, scores = self.gate(x, ...)` (`:2740-2751`) — routing decision (top-6-of-256),
   computed **identically and redundantly on every rank** (the gate weight isn't sharded).
   Both ranks pick the exact same experts for the exact same tokens; no rank ever needs to
   learn "which rank has expert e" because every rank has every expert.
3. `y = self.switch_mlp(x, inds)` (`:2790`) → `SwitchGLU`/`gather_qmm` dispatch
   (`switch_layers.py:76-90`) — each rank runs the **full gather_qmm over its own
   half-width weight slice** for all selected experts. This is pure local compute, zero
   network traffic, and its cost is **independent of which rank holds what** since both
   ranks hold everything.
4. `y = mx.distributed.all_sum(y, group=self.sharding_group)` (`:2836`) — the **only**
   cross-node collective in the MoE forward. `y` has shape `(B, L, hidden_size)` — the
   **model's fixed hidden dimension**, not a function of `num_experts_per_tok`,
   `n_routed_experts`, or routing skew. Sums the two ranks' half-width-computed partial
   FFN outputs into the full-width result. Traffic volume = `B × L × hidden_size ×
   dtype_bytes`, **constant per token regardless of which of the 256 experts fired.**

So the answer is unambiguous: **fixed-size reduction of a (tokens × hidden) tensor**,
happening on every layer (43 times) regardless of routing. There is no gather/scatter of
tokens to "expert-owning" ranks anywhere in this path — the width-sharding model makes
"expert ownership" a non-concept. Expert placement literally cannot change what crosses
the wire.

## Q3: Is there a load-balance win from expert placement instead? — INFERRED, likely small/moot

The premise (skewed routing making one rank's FFN compute slower, stalling the other rank
at the all_sum/barrier) doesn't apply here either, because **both ranks compute all 256
experts' partial outputs on every token** — there's no rank-specific expert subset to be
imbalanced by routing skew. Each rank's per-token FFN compute cost is `top_k=6 × (half
intermediate width)`, identical on both ranks by construction, independent of *which* 6
of 256 experts were chosen. Routing skew (e.g. one expert getting hammered) would slow
**both ranks equally** (they both compute that expert), not create cross-rank imbalance.

- Diagnostic check: `/tmp/dsv4_route_hist` does **not exist** on this host — the
  `EXO_MOE_EXPERT_HIST_DIAG` diagnostic has never been run/collected. No skew data
  available to quantify, but per the mechanism above, skew data wouldn't identify a
  rank-imbalance opportunity under this sharding scheme even if collected — it would only
  be useful for a genuinely different sharding scheme (true EP/token-to-expert-rank
  dispatch), which DSv4 does not implement.
- Conclusion: **no real win available from expert placement under the current
  architecture**, for either the traffic-volume reason (Q2) or the load-balance reason
  (Q3). Both windows the user's idea might have worked through are closed by the same
  underlying fact: width-sharding replicates every expert everywhere.

## Q4: What would it take to actually do expert-axis (EP) sharding? — INFERRED

To make locality meaningful at all, DSv4 would need genuine expert-parallel sharding
(assign expert *indices* to ranks, not weight-axis slices), which is a materially
different mechanism than what exists:

- **Weight layout**: would need per-expert-index slicing on axis 0 of the `(256, out,
  in)` `SwitchLinear`/`QuantizedSwitchLinear` weight tensors instead of axis
  `ndim-2`/`-1`. Straightforward at the tensor-slicing level (`_split`+`mx.split` already
  generalize to any axis, `distributed.py:29-35`), but is a **new sharding predicate**,
  not a drop-in — `shard_inplace`'s `sharding_predicate` callables would need to switch
  from `(axis, segments)` on the width dims to axis-0 slicing keyed by an explicit
  expert->rank permutation table.
- **Quantization group boundaries**: not a blocker for EP sharding — quantization groups
  run along `input_dims`, unaffected by which axis-0 (expert) rows are kept per rank.
  Slicing axis 0 doesn't cross group boundaries. Width-sharding (axis 1/2, current
  scheme) is the one that has to be group_size-aligned; that constraint disappears under EP.
- **The real blocker is the forward-pass gather/dispatch math**: `gather_qmm`'s
  `rhs_indices=inds` (`switch_layers.py:76-84`) currently indexes into the **full local
  256-expert weight tensor** — every rank can serve every routed index because every rank
  has every expert. Under EP, `inds` would need to be **filtered per rank** (only dispatch
  tokens routed to locally-owned experts) and the **output must be scattered back and
  combined across ranks** — i.e., a genuine all-to-all or gather/scatter of *token
  activations*, not weights. This is a full MoE-dispatch rewrite (routing table,
  cross-rank token shuffle, unsort/combine), not a sharding-strategy tweak. It also
  reintroduces exactly the traffic-volume-scales-with-tokens problem TP was trying to
  avoid, except now shaped by routing skew — i.e. it trades a *fixed* per-token cost
  (current all_sum) for a *variable* one (token counts sent to remote experts), which is
  not obviously better and needs real modeling before committing to it.
- **Given no `n_group`/`topk_group` in this DSv4 config** (unconstrained top-6-of-256
  routing), an EP scheme also has no free load-balance guarantee the way DSv3's grouped
  routing provides — arbitrary/measured-load rank assignment would need to be re-tuned
  continuously as the model/traffic shifts, adding operational complexity for a benefit
  that (per Q2/Q3) doesn't even exist at the current TP granularity.

**Recommendation: do not pursue expert-locality placement.** It requires a full EP
rewrite with real cross-node token traffic and no established payoff model, to fix a
mechanism (all_sum volume) that provably doesn't respond to expert placement under the
current architecture.

## Q5: Ranked TP prefill overheads (real ones worth attention)

Ranking by expected value — highest-leverage / cheapest-to-test first:

1. **(b) Sparse-indexer all_gather under `EXO_DSV4_SEQ_SPLIT`** — HIGHEST PRIORITY,
   cheap experiment already available.
   - What: splits attention query rows across ranks for prefill (`SparseCompressedAttention`
     / `CompressedAttention` only — `auto_parallel.py:1067-1073`), then `all_gather`s the
     halves back (comment at `deepseek_v4.py:169-171,180-188` — a subgroup all_gather that
     has a documented stuck-send/wedge failure mode, with a kill switch already built in).
     One all_gather **per prefill chunk** on top of the 43 per-layer all_sums.
   - Overlap/eliminate: it's a **live env toggle, default 1, already plumbed through
     start_cluster.sh** — directly A/B-able (`EXO_DSV4_SEQ_SPLIT=0` vs `1`) with **one
     relaunch, no rebuild**. This is the single cheapest experiment on the list and
     should be run before investing in anything else.
   - Cheap experiment: flip the env var, compare prefill tok/s, no code changes.

2. **(a) Per-chunk `mx_barrier(group)`** — likely real, moderate cost, but load-bearing.
   - What: explicit rank-sync barrier after every prefill chunk's forward
     (`generate.py:1412-1417` per task context; confirmed call site pattern at
     `src/exo/worker/engines/mlx/generator/generate.py` `mx_barrier(group)` after
     `model(...)` — comment there and at `utils_mlx.py:589` calls it the mechanism
     "documented as the source of PP's prefill edge" i.e. this is exactly the
     serialization cost that makes PP prefill outperform TP prefill for this workload.
   - Overlap/eliminate: risky — it exists specifically to guard against rank drift
     before the next chunk's all_sum collectives fire (comment above the barrier call).
     Removing/relaxing it without care reintroduces the cross-rank graph-position drift
     bugs the `_fence_every_n`/Phase-H lever comments (`deepseek_v4.py:2657-2669`)
     describe fighting. Any change here should follow the same eval-every-N-layers style
     lever already built for the all_sum fence, not blanket removal.
   - Cheap experiment: none free — this needs careful staged testing, not a flag flip.

3. **(d) 43 per-layer all_sums** — largest aggregate volume but least tunable per-op;
   only tunable via batching frequency.
   - What: one `all_sum` of `(B, L, hidden)` per layer per chunk (`deepseek_v4.py:2836`).
     `EXO_DSV4_FENCE_EVERY_N_LAYERS` already exists to batch the *eval fence* every N
     layers instead of every layer (`deepseek_v4.py:2650-2660`), but the all_sum
     collective itself still fires every layer — only the blocking `mx.eval(y)` after it
     is what N-layers batching defers. Note the code comments this was tried at the
     all_sum-inside-compile level (effectively N=∞) and **collapsed throughput to 7.7
     tok/s** at c=2 100K context (`deepseek_v4.py:2660-2662`) — so this lever is known to
     have a bad failure mode at the extreme and needs care.
   - Overlap/eliminate: partially explored already (Lever 6/Phase H); further headroom
     unclear without new data — this is the highest-effort, most already-tuned item on
     the list. Lower priority than (a)/(b) for new investigation.

4. **(c) Replicated attention compute** — lowest priority; not a network overhead at all.
   - What: attention (LoRA-decomposed Q/output projections) runs full, un-sharded,
     identically on both ranks by design (`DeepseekV4ShardingStrategy` docstring,
     `auto_parallel.py:1031-1041`) — a deliberate memory/compute tradeoff, not a
     communication cost. It wastes GPU cycles (2x redundant attention compute across
     ranks) but generates **zero cross-node traffic**, so it doesn't compete with (a)/(b)
     as a "TP overhead" in the network sense the user is worried about. Only relevant if
     attention compute time itself (not comms) turns out to bound prefill wall-clock —
     no evidence collected here to support or refute that; would need a profiling pass,
     not a network fix.

## Summary table

| Item | Mechanism | Cross-node traffic scales with...? | Actionable? |
|---|---|---|---|
| Q2 all_sum | fixed (B,L,hidden) reduction | nothing routing-related — constant per token | No — locality can't help |
| Expert placement (user's idea) | N/A — no per-expert dispatch exists | N/A | Not viable as stated; would require full EP rewrite (Q4), unproven payoff |
| (b) seq-split all_gather | per-chunk indexer gather | chunk count / seq length | **Yes — free A/B via env var** |
| (a) mx_barrier | per-chunk rank sync | chunk count | Maybe, but load-bearing; careful staged test only |
| (d) 43 all_sums | per-layer reduction | layer count (fixed 43) | Partially tuned already; diminishing/risky returns |
| (c) replicated attention | redundant compute, no network | N/A (compute, not comms) | Not a comms overhead; lowest priority here |

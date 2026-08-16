# Section 101 — TP Decode Capability Audit

Read-only audit of `~/repos/exo` (adurham/exo, commit `f3573fc17` at time of
writing). Scope per task: (1) what TP decode actually shards/collectives,
(2) whether PP+MTP is structurally impossible or merely gated, (3) how/when
sharding mode is chosen, (4) weight-residency mechanics at `shard_model()`
time, (5) cold-load cost + whether a cache-over-wire primitive already exists
that could survive a restart. Every claim below is marked CONFIRMED (read the
code directly) or INFERRED (reasoned from adjacent code/comments, not
directly executed/traced end-to-end).

---

## 1. TP decode path — what `DeepseekV4ShardingStrategy` actually shards

**CONFIRMED.** `DeepseekV4ShardingStrategy` (`src/exo/worker/engines/mlx/auto_parallel.py:1031-1145`)
is MoE-only sharding:

- Attention is **replicated on every rank**, unsharded — the class docstring
  states this explicitly and gives the reason: DSv4's LoRA-decomposed
  Q/output projection (`wo_a`/`_grouped_output_projection`) breaks under a
  head-split (`auto_parallel.py:1032-1042`).
- Only the MoE block is sharded, per layer: `shared_experts.{gate,down,up}_proj`
  and `switch_mlp.{gate,down,up}_proj` (`auto_parallel.py:1082-1087`), plus the
  same six ops applied to each MTP block if present (`auto_parallel.py:1118-1127`).
- `layer.ffn.sharding_group = self.group` is the sole per-layer state
  set on the MoE module (`auto_parallel.py:1065`); no `ShardedMoE` wrapper is
  used — the DSv4 `DeepseekV4MoE.__call__` itself checks `self.sharding_group is
  not None` and does the reduction inline (`auto_parallel.py:1044-1046`,
  confirmed in `mlx-lm/mlx_lm/models/deepseek_v4.py:2712-2714` and
  `:2830-2833`).

**Per-token collectives (decode, L==1), CONFIRMED from
`mlx-lm/mlx_lm/models/deepseek_v4.py`:**

- `DeepseekV4MoE.__call__` (`deepseek_v4.py:2688-2833`):
  - **Input side:** `x = sum_gradients(self.sharding_group)(x)` (`:2712-2714`).
    `sum_gradients` (`mlx/python/mlx/nn/layers/distributed.py:14-27`) is a
    `custom_function` whose **forward pass is the identity** (`return x`,
    line 21) — the `all_sum` it wraps only fires in the **backward/vjp**
    (line 25), which decode never triggers. So this call contributes **zero
    collectives at inference time**, despite the name suggesting otherwise —
    it exists purely to make gradient sync correct for training-adjacent
    code paths. **This is a fact worth flagging: `sum_gradients` on the
    input side is a training-only hook, not a decode-time collective.**
  - **Output side:** `y = mx.distributed.all_sum(y, group=self.sharding_group)`
    (`:2830-2833`) — this is the **real** per-layer decode collective, one
    genuine RDMA `all_sum` per MoE forward.
- Attention: `_fused_sharded_qk_norm` in `auto_parallel.py:1210-1250` is only
  invoked when `attn.sharding_group` is set, which for DSv4 only happens
  under the prefill-only, opt-in `EXO_DSV4_SEQ_SPLIT` gate
  (`auto_parallel.py:1066-1081`, "Prefill L>1 only; decode is length-gated
  inside the attention. Default off"). **At decode, attention is unsharded
  and replicated — zero attention-side collectives.**
- No separate indexer `all_gather` was found gated for decode; the seq-split
  `all_gather` path referenced in `deepseek_v4.py` comments (lines 166-188)
  is explicitly prefill-only (`L>1`) per the same comments.

**Per-token collective count for 43 layers, CONFIRMED arithmetic from the
above:** **1 real collective (`all_sum`) per MoE layer per decode step** ×
43 main layers = **43 `all_sum` calls per decode token** when no MTP head is
active. If MTP block(s) are present and sharded identically
(`auto_parallel.py:1118-1127`), each additional MTP block/verify step adds
one more `all_sum`. The `sum_gradients` input-side call adds **zero**
additional wire traffic at inference time (forward = identity). Fence
batching (`EXO_DSV4_FENCE_EVERY_N_LAYERS`, `deepseek_v4.py:2660-2669`)
changes how often results are *forced to materialize* (`mx.eval` fences),
not the collective *count* — the `all_sum` op itself still gets issued once
per layer into the lazy graph regardless of fence cadence.

---

## 2. MTP under TP vs PP — structural conflict or gate?

**Verdict: NOT structurally impossible under PP. It is a real engineering
conflict in the *default* transport path, but this fork already ships a
SEPARATE, parallel PP-native speculative-decode mechanism that coexists with
`PipelineLastLayer` by construction — it just isn't the same code path as
TP's `DSv4MTPBatchGenerator`.**

Three distinct facts, all CONFIRMED by reading the code:

1. **TP's MTP path (`DSv4MTPBatchGenerator`, `speculative/dsv4_mtp.py`) is
   selected in `batch_generate.py:__post_init__` (`:812-926`) purely on
   `EXO_SPECULATIVE=1` + "is the loaded model a `DeepseekV4Model` with an
   `mtp` submodule" (`:828-832`). Nothing in that branch inspects
   `self.group.size()`, sharding mode, or `PipelineShardMetadata` — it will
   attempt to build `DSv4MTPBatchGenerator` under PP just as readily as
   under TP.** This branch is NOT the thing that "excludes" PP.

2. **The actual PP-specific conflict is documented and fixed, not a
   structural dead end.** `pp_speculation.py:896-913` (comment block)
   describes a *real, once-shipped* bug: PP+MTP crashed because a collective
   (`mx.distributed.all_gather`) was mixed onto the *same* jaccl transport
   that `PipelineFirstLayer`/`PipelineLastLayer` use for their own automatic
   p2p send/recv handoff, starving jaccl's ack bookkeeping
   (`"[jaccl] drain_acks STALLED ... UC completion lost"`, forcing a runner
   crash). **This was fixed** by replacing the collective with a plain
   send/recv matching the rest of the file's discipline
   (`pp_speculation.py:906-913`) — i.e. the conflict was a *transport
   sharing* bug, already root-caused and patched, not an unfixable
   architectural incompatibility.

3. **PP has its own dedicated speculative-decode machinery that
   subclasses (not fights) `PipelineLastLayer`.** `SpecPipelineFirstLayer`/
   `SpecPipelineLastLayer` (`pp_speculation.py:453-585`) literally
   *subclass* `PipelineFirstLayer`/`PipelineLastLayer` and are installed via
   `_install_spec_layers` (`generate.py:2139-2143`) — activated whenever
   `EXO_PP_DRAFT_MODEL` is set and `group.size() > 1` under PP
   (`generate.py:2123-2131`). DSv4's native MTP head is explicitly wired
   into this PP path too: `batch_generate.py:1003-1046` constructs
   `DSv4MTPPredictor(self.model, mtp_idx=0)` for the **PP** `_pp_spec_active`
   branch (guarded by `self.group is not None and self.group.size() > 1`,
   `:972-975`) — this is PP+MTP running today, using the model's native MTP
   head, through the PP-specific `pp_dspark_decode_loop`/
   `pp_speculative_decode_loop` machinery, NOT `DSv4MTPBatchGenerator`.

**So the "verified claim that MTP works only under TP, conflicting with
`PipelineLastLayer`" is only true for one specific code path**
(`DSv4MTPBatchGenerator`, the *TP-side* batch-generation MTP implementation)
— it is genuinely never invoked with a `PipelineShardMetadata`-sharded model
because `batch_generate.py`'s `_pp_spec_active` branch intercepts and
diverts to the PP-native speculative loop first (`:2662-2675`, checked
before falling through to the `self._mlx_gen.insert(...)` TP/non-PP path at
`:2678`). **This is a routing/dispatch difference, not a hardware or
math-level incompatibility** — PP's own parallel MTP implementation exists,
is wired to the same native `mtp[0]` weights, and was specifically
engineered around the exact `PipelineLastLayer` handoff conflict the user's
framing refers to. The skill notes (`exo-sharding-mode-tradeoffs`,
2026-08-04 update) independently confirm this split: *"a completely
different sharding scheme's completely different speculative-decode
implementation (`dsv4_mtp.py` vs PP's own `pp_dspark_decode_loop`)"* — this
is corroborating, not contradicting, evidence (INFERRED cross-reference,
not re-verified line-by-line here beyond what's cited above).

**Practical implication for the phase-swap goal:** if the plan is
"prefill on PP, decode on TP, with MTP active during decode," there is
**no need to reach TP for MTP to work** — PP+MTP is already a real, shipped
code path (`SpecPipelineLastLayer` + `DSv4MTPPredictor`). The open question
is whether *that* PP-MTP path's measured throughput meets the target, not
whether it exists.

---

## 3. Model load path — where sharding mode is chosen, and whether it can change without reload

**CONFIRMED, full trace:**

1. **API request time** — `sharding: Sharding = Sharding.Pipeline` is a
   field on the instance-placement request type (`src/exo/api/types/api.py:305`,
   `:316`). `src/exo/api/main.py:568` defaults `sharding=Sharding.Pipeline`
   for a new instance; `:624` and `:1417-1422` show the API iterating both
   `Sharding.Pipeline`/`Sharding.Tensor` as placement *candidates* when
   choosing where/how to admit a model — i.e. the decision is made once, at
   instance-creation/placement time, by the master.
2. **Master placement** — `src/exo/master/placement.py:170-286` and
   `src/exo/master/placement_utils.py:531-548` (`get_shard_assignments`)
   `match` on `sharding` and call either
   `get_shard_assignments_for_pipeline_parallel` (`:462-484`) or
   `get_shard_assignments_for_tensor_parallel` (`:490-518`), producing either
   `PipelineShardMetadata(...)` (`start_layer`/`end_layer`/`device_rank`) or
   `TensorShardMetadata(...)` (`device_rank`/`world_size`, full layer range
   on every rank) objects — one immutable metadata object per runner,
   assigned once.
3. **Runner start** — this `ShardMetadata` is handed to the worker as
   `bound_instance.bound_shard` and passed into `shard_and_load`
   (`utils_mlx.py:245-250`).
4. **`shard_and_load` dispatch** (`utils_mlx.py:437-447`) — a `match
   shard_metadata:` on `TensorShardMetadata()` → `tensor_auto_parallel(...)`
   or `PipelineShardMetadata()` → `pipeline_auto_parallel(...)`. This is a
   **one-shot dispatch at process start** — the branch executes exactly once
   per runner process lifetime, during `load_mlx_items` (called from
   runner bootstrap).
5. **`tensor_auto_parallel`** (`auto_parallel.py:704-849`) picks a
   model-family-specific strategy object (`DeepseekV4ShardingStrategy` for
   DSv4) and calls `.shard_model(model)`, which permanently mutates the live
   `nn.Module` tree in place (`shard_inplace`/`shard_linear`, sets
   `sharding_group` attributes) and returns it.

**No seam exists today to change sharding mode without a full process
restart / weight reload.** CONFIRMED reasoning:

- The dispatch (`match shard_metadata: ...`) happens exactly once, at the
  top of `shard_and_load`, before any request is served. There is no
  re-dispatch call anywhere else in `utils_mlx.py`, `auto_parallel.py`, or
  `generator/*.py` — `shard_model()` is a **generator that runs to
  completion and returns a new module tree once**, not a function that can
  be re-invoked on a live model.
- Sharding mutates weight *tensors themselves* in place
  (`shard_inplace`/`shard_linear` in `auto_parallel.py` call into
  `mlx/python/mlx/nn/layers/distributed.py`'s `_shard`, which literally
  slices and reassigns `.weight`/`.scales`/`.biases` arrays) — for TP this
  discards the non-owned half of every MoE projection's weight
  irreversibly (no full-precision copy retained anywhere in the runner
  process once `mx.clear_cache()` runs, `auto_parallel.py:1110`). Going
  from TP → PP (or vice versa) would need the FULL, unsharded weights back,
  which no longer exist in that process's memory.
- `bound_instance.bound_shard` (the `ShardMetadata`) is read exactly once at
  `load_mlx_items` call time (`utils_mlx.py:245`); nothing subscribes to
  changes on it afterward — INFERRED (did not trace the full instance /
  event-sourcing model, but no second reader of `bound_shard` was found in
  `utils_mlx.py`, `auto_parallel.py`, or `generator/*.py`).

**Conclusion for #3: the sharding mode is fixed at model-load time, chosen
once by the master at placement, dispatched once in `shard_and_load`, and
irreversibly baked into the live weight tensors by `shard_model()`. There is
no existing seam — env var, model-card field, or instance-metadata
re-read — that changes it without a full process restart and fresh weight
load.**

---

## 4. Weight residency mechanics at `shard_model()`

**CONFIRMED (partially) / INFERRED (partially):**

- **CONFIRMED:** `load_model(model_path, lazy=True, strict=False)`
  (`utils_mlx.py:349`) loads the model with **MLX lazy arrays** — the whole
  unsharded checkpoint graph is constructed but not materialized. This is
  "(b) load everything then discard [the non-owned shard]" in spirit, not
  "(a) load only shard from disk" — but because MLX arrays are lazy, "load
  everything" doesn't mean the full weights ever exist in dense GPU/unified
  memory simultaneously; it means the *plan* to load them exists until
  something forces evaluation.
- **CONFIRMED:** inside `DeepseekV4ShardingStrategy.shard_model`
  (`auto_parallel.py:1062-1111`), each layer is processed **one at a time**:
  `mx.eval(layer.parameters())` (materializes that one layer's full,
  unsharded weights) → shard in place (discard non-owned half) → `mx.eval(layer)`
  → `mx.clear_cache()` (`:1109-1110`). This is a genuine
  streaming/progressive materialize-then-discard pattern, not
  "materialize the whole 300GB+ model, then discard half" — peak memory
  during load is bounded by one layer's full weights, not the whole model's.
- **CONFIRMED:** `mx.clear_cache()` after each layer's shard step
  (`auto_parallel.py:1110`, `:1142`) explicitly returns MLX's internal
  buffer-cache memory to the system allocator promptly, per-layer — this is
  the mechanism that keeps peak memory low during a 43-layer sequential
  shard pass.
- **INFERRED (not verified this session):** whether MLX exposes a
  documented public API to "release specific weight arrays and load
  replacements in place" for an *already-running, already-serving* model
  (as opposed to during the one-shot load sequence above) — no such call
  site was found anywhere in `auto_parallel.py`, `utils_mlx.py`, or
  `generator/*.py`; the pattern above (`mx.eval` + slice-assign + `clear_cache`)
  is only ever used during the initial, single `shard_model()` pass, never
  invoked a second time against a live serving model. Given finding #3 (no
  re-dispatch seam) and the fact that TP sharding permanently discards the
  non-owned weight half with no retained backup, **a sequential in-place
  layout swap (TP→PP or PP→TP) on a live process is not supported by any
  existing code path** — it would need to be built from scratch (re-fetch
  discarded weights from disk, likely re-triggering nearly the full 18s
  load cost measured in §5), not merely "unwired."

---

## 5. Cost of a full restart, and existing cache-survival primitives

**CONFIRMED — cold load timing, read live from the running cluster's log
files (not measured fresh this session, read-only per constraints):**

```
adams-mac-studio-m4-1.local: "Time taken to shard and load model: 18.75s"
adams-mac-studio-m4-2.local: "Time taken to shard and load model: 18.68s"
```
(`~/exo.log` on each node, `exo.worker.engines.mlx.utils_mlx:load_mlx_items:250`.)
**~18.7s per rank for a full model shard+load**, both ranks essentially in
lockstep. This is a real, recent, in-context measurement, not synthetic.

**Existing cache-over-wire primitive — CONFIRMED, and it is exactly the
right shape for a restart-based fallback, with one real limitation:**

- `run_prefill_for_request` (`src/exo/worker/engines/mlx/disaggregated/serve.py:20-92`)
  is a genuine, already-built primitive: given a `PrefillRequest`
  (token ids + request id), it builds a fresh KV cache, runs `mlx_prefill`
  against a loaded model, and returns the populated `KVCacheType` — this is
  literally "run prefill in one process, get a cache object back," already
  wired into `exo.worker.disaggregated.server`.
- `write_cache_to_wire` (`src/exo/worker/engines/mlx/disaggregated/adapter.py:217-241`)
  serializes that cache (header + per-layer KV chunks, `send_mlx_kv_cache`)
  onto a `BinaryIO` stream — this is the actual "survive the reload" wire
  format: a prefill server can hand a populated cache to a *separate*
  process (a fresh decode process, post-restart) over the wire, with no
  disk round-trip needed.
- `remote_prefill()` (`src/exo/worker/engines/mlx/generator/remote_prefill.py:19-76`)
  is the **client-side receiver** of that same wire protocol
  (`remote_prefill_fetch` + `ingest_into_mlx_cache`), already logging
  transfer/inject timing and effective tok/s (`:66-71`) — i.e. the
  prefill-here / decode-there split this fork's own disaggregated-prefill
  feature already performs in production is architecturally identical to
  "prefill under PP, restart into TP, decode there."

**The real limitation, CONFIRMED, is a coverage gap, not a design flaw:**
`adapter.py:105-111` — the wire protocol's per-layer-cache-type dispatch
explicitly `raise NotImplementedError` for `QuantizedKVCache() | CacheList() |
PoolingCache()`. The module's own header comment (`adapter.py:13-18`)
states DSv4's *actual* per-layer cache shape (post Blaizzy PR #1192) is a
`CacheList` composed of `RotatingKVCache` + two `PoolingCache` instances —
**exactly the composite type this protocol currently refuses to serialize.**
So today, `write_cache_to_wire`/`remote_prefill` cannot carry a real DSv4
KV cache across a process boundary; the primitive exists, is proven for
whatever cache types it *does* support, but needs the `CacheList`/
`PoolingCache`/`QuantizedKVCache` branches implemented before it could
actually be pointed at a PP-prefill → TP-restart-for-decode handoff for
this model.

---

## Summary table

| Question | Verdict | Confidence |
|---|---|---|
| What does TP decode shard? | MoE only; attention fully replicated | CONFIRMED |
| Collectives/token @ 43 layers | 43 real `all_sum` (MoE output); `sum_gradients` input hook is a no-op forward pass | CONFIRMED |
| Is PP+MTP structurally impossible? | **No.** PP has its own native MTP path (`SpecPipelineLastLayer` + `DSv4MTPPredictor`) that already coexists with `PipelineLastLayer`'s handoff; the once-real conflict (mixed collective on jaccl transport) was found and fixed. TP's `DSv4MTPBatchGenerator` is simply routed around under PP, not blocked by it. | CONFIRMED |
| Can sharding mode change without reload? | **No.** Chosen once at master placement, dispatched once in `shard_and_load`, and irreversibly baked into weight tensors by in-place sharding. No re-dispatch seam exists. | CONFIRMED |
| Weight residency at shard time | Lazy full-checkpoint graph, materialized+discarded one layer at a time, `mx.clear_cache()`'d per layer — bounded peak memory, but no retained full-weight backup for a later un-shard | CONFIRMED (mechanism) / INFERRED (no live-swap API exists) |
| Cold load cost | ~18.7s/rank, measured from live cluster logs | CONFIRMED |
| Cache-survives-restart primitive | `run_prefill_for_request` + `write_cache_to_wire` + `remote_prefill` already exist and are exactly the right shape, but the wire protocol doesn't yet serialize DSv4's actual `CacheList`/`PoolingCache` cache type — a real, scoped gap, not a redesign | CONFIRMED |

## Bottom line for the phase-swap decision

A phase-swap-without-restart mechanism is **not buildable on top of existing
seams** — sharding is baked into weights at one-shot load time with no live
re-dispatch path (§3, §4). The **restart fallback is real and closer to
working than expected**: PP+MTP already works today without needing TP at
all (§2), cold load is cheap (~18.7s/rank, §5), and this fork already has a
working prefill-here/decode-there wire primitive (§5) — it just needs
`CacheList`/`PoolingCache` support added to `write_cache_to_wire`'s dispatch
before it can carry a real DSv4 cache across the restart boundary.

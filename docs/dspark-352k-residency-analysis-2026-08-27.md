# Steady-state per-node residency analysis at 352.6K context — DSv4-Flash 8-bit, TP=2

**Scope.** Read + math only. Quantifies steady-state and per-cycle
memory residency for spec-OFF vs spec-ON DSpark MTP decode on the 2-node
Mac Studio M4 Max cluster at 352.6K context. Ranks root-cause fixes
(env-gated, NO context caps) with expected GB savings and file:line
implementation sites.

**Repo/commit.** exo main @ `b999c3354` (tree clean). mlx-lm submodule
pinned at `d098642`. **All numbers below reflect what the CURRENT tree
actually allocates and touches**; the two prior docs
(`root_cause_analysis.md`, `fragmentation_pool_analysis.md`) are verified
and corrected inline.

## TL;DR (executive-summary preview)

The single largest spec-ON-only STEADY-STATE cost is the **DSpark draft
head, ~10.13 GB per node, replicated on both TP ranks** (loaded via
`utils_mlx.py:439-446` BEFORE `tensor_auto_parallel`; the shard iterator
at `auto_parallel.py:1153-1178` iterates `model.model.mtp` only, never
touches `model.model.dspark`). That's a hard, non-transient 10 GB delta
that shifts the entire memory equation to the wrong side of the wired
ceiling.

However, static delta alone doesn't fully explain 4/14 stochastic
collapses with permanent (not sporadic) 1.8-2.3 s cycles once collapse
starts. That signature demands a MONOTONIC ratchet. The strongest
candidate ratchet is the MLX **buffer-cache growth** that
`start_cluster.sh:133` explicitly leaves OFF (`EXO_MLX_CLEAR_CACHE_INTERVAL=0`)
despite an in-tree comment
(`batch_generate.py:93-102`) that flags this exact class as *"the dominant
source of RSS growth and ultimately what OOMs the runner"* on long
Think-mode decode.

**#1 fix (recommended):** `EXO_MLX_CLEAR_CACHE_INTERVAL=64` — one env change,
no code touched, releases MLX's cached buffer pool back to the OS every 64
decode steps. Expected: eliminates the ratchet, restores several GB of
headroom continuously. If it doesn't fix collapse, we KNOW the problem is
purely the DSpark head's 10 GB static cost, and the structural fix is #2
(shard DSpark FFN experts through the same TP path as `mtp` blocks).

---

## 1. Ground truth: model config, sharding, and what is/isn't split under TP=2

Verified against `mlx-community/DeepSeek-V4-Flash/config.json`:
- `hidden_size=4096`, `num_hidden_layers=43`, `num_attention_heads=64`,
  `head_dim=512`, `num_key_value_heads=1`, `q_lora_rank=1024`,
  `o_lora_rank=1024`, `o_groups=8`.
- `compress_ratios[0]=0, [1]=0, [43]=0` (3 local layers), the interior 40
  alternate 4/128 → **21 sparse (ratio=4), 20 compressed (ratio=128),
  3 local (ratio=0)**. Confirmed via Python Counter.
- Attention KV is single-head; `values` are always `_zero_values(B,1,L,0)`
  (see `DSparkLocalAttention.append_ctx` at `deepseek_v4.py:6395`;
  `RotatingKVCache._update_in_place` guards this at `cache.py:704`).
- MoE: 256 routed experts + 1 shared expert, `moe_intermediate_size=2048`,
  top-6.
- Indexer: `index_n_heads=64`, `index_head_dim=128`, `index_topk=512`
  (matches `EXO_DSV4_INDEX_TOPK=512` in the launch env).
- Quantization: `bits=8, group_size=64, mode=affine` for the base weights.
  The DSpark head is separately quantized `mxfp4/mxfp8, group=32` (see
  `utils_mlx.py:837-862`).

**What actually shards under TP=2 (`auto_parallel.py:1049-1180` +
`deepseek_v4.py:7327-7355`):**
- `layer.attn.wq_b` — split (`all_to_sharded` on axis 0, halves n_heads).
- `layer.attn.wo_a` — `sharded_to_all` (input axis split, reduces on
  output).
- `layer.ffn.shared_experts.{gate,up,down}_proj` and
  `layer.ffn.switch_mlp.{gate,up,down}_proj` — all shard on the
  INTERMEDIATE axis (`moe_intermediate_size` halved per rank). Every rank
  still holds ALL 256 experts at half-width — expert IDENTITY is not
  partitioned (see the correction note at `auto_parallel.py:1141-1148`).
- `layer.attn.attn_sink` — split.

**What stays REPLICATED (present in full on both nodes):**
- `layer.attn.wq_a`, `layer.attn.wkv`, `layer.attn.wo_b`.
- Indexer (`indexer.wq_b`, `indexer.weights_proj`, `indexer.compressor.*`) —
  no shard call reaches it. This is deliberate:
  `SparseCompressedAttention` docstring at `auto_parallel.py:1032-1046`
  says attention is "MoE-only sharded" for DSv4 to keep the LoRA-
  decomposed Q/output projection intact.
- Main-attention `Compressor` (`compress_ratios ∈ {4,128}` → its
  `wkv`+`wgate` at `head_dim=512`).
- `model.embed_tokens`, `model.lm_head`, `model.model.norm` — the shard
  iterator only touches `model.model.layers` and `model.model.mtp`.
- **`model.model.dspark` — NEVER sharded** (see §5).

---

## 2. Steady-state per-node residency at 352.6K, byte math

### 2.1 Base model weights

The full-precision 671B params × 1 B/elem (8-bit) → ~625 GB is a naive
upper bound; DSv4-Flash-8bit on disk is ~72 GB per HF card and healthy
resident wired is 97 GB / node (given). MoE routed experts dominate:
sharded intermediate width halves them, but both ranks hold all 256
expert IDs. My arithmetic-from-arch estimate over-counted (285 GB
full-replicated body); the actual on-disk quantized checkpoint has extra
compaction (shared_experts fused, quantization scale overhead lower per
element than my `1 + 4/64` calculation). The load-time OS-observed
number is the source of truth.

**Empirical numbers (given / verified from telemetry):**

| Item | Bytes | Source |
|---|---|---|
| Healthy per-node wired (spec-ON @ 352.6K) | ~97 GB | task context |
| MLX wired cap (approx) | ~124.5 GB | prior root-cause doc |
| Spec-OFF projected wired (subtract DSpark head) | ~87 GB | this doc |
| Spec-OFF 500K on 2026-08-21 (baseline, no swap) | fit | `docs/known-good-prefill-baseline-2026-08-21.md:28` |

Both nodes hold the same shard of the layer weights (TP is symmetric).
Per-node body weights are ~85 GB. Embed + `lm_head` are unsharded
(`vocab_size=129280, hidden=4096`, ~1.5 GB combined). Weights are
`quantization=8-bit affine, group_size=64` for the base checkpoint.
**These are IDENTICAL between spec-OFF and spec-ON runs.**

### 2.2 Main KV cache

`make_cache()` at `deepseek_v4.py:7127-7150`:

- All 43 layers have a `RotatingKVCache(max_size=sliding_window=128)`.
- Sparse layers (21) also have TWO `PoolingCache(ratio=4)` (main
  compressor + indexer, `deepseek_v4.py:7133-7141`).
- Compressed layers (20) have ONE `PoolingCache(ratio=128)`
  (`deepseek_v4.py:7144-7149`).
- Local layers (3) have only the ring.

At 352.6K context (verified):
- `P_sparse = ctx / 4 = 88,150`
- `P_comp = ctx / 128 = 2,754`

`RotatingKVCache` capped at `sliding_window=128` never grows past that
window. Storage per layer:
- K: `(B=1, num_kv_heads=1, 128, head_dim=512) bf16 = 128 KB`.
- V: zero-shape (single-KV-head), never allocated per `cache.py:698-705`.

**Total ring storage per node: 43 × 128 KB = 5.4 MB.** Negligible.

`PoolingCache._pool_storage` shape `(1, capacity, D)` bf16, where
capacity is `ceil(offset/step)*step` with `step=256` (`cache.py:1337`):

| Layer class | # | D | P | Storage/layer | Total |
|---|---:|---:|---:|---:|---:|
| Sparse main comp (D=512) | 21 | 512 | 88,150 | 86.25 MB | 1.81 GB |
| Sparse indexer (D=128) | 21 | 128 | 88,150 | 21.56 MB | 0.45 GB |
| Compressed (D=512) | 20 | 512 | 2,754 | 2.75 MB | 0.05 GB |
| **Pool storage TOTAL** | | | | | **~2.27 GB** |

**Pool storage is NOT sharded** — `shard()` doesn't touch the
PoolingCache attributes and both nodes each carry the full ~2.27 GB.
This is IDENTICAL between spec-OFF and spec-ON.

**Correction to `root_cause_analysis.md`:** that doc claimed 1.95 GB pool
total. The correct figure at 352.6K is **~2.27 GB** (the sparse
indexer's own pool at D=128 was under-counted — it stores 88,150
entries per sparse layer, adding 0.45 GB per node).

### 2.3 Spec-ON-only steady residency (the actual delta)

Per the launch env (`EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1
EXO_DSV4_DSPARK_NATIVE=1 EXO_DSV4_DSPARK_FORCE_LOAD=1`):

**(a) DSpark head weights — 10.13 GB per node, REPLICATED.**

- Load path: `utils_mlx.py:439-446` — with `EXO_SPECULATIVE=1 +
  EXO_DSV4_MTP=1` both set, `_tp_consumer=True` so the head loads even
  without `FORCE_LOAD=1` (verify at `utils_mlx.py:421,427`); the env's
  `FORCE_LOAD=1` is redundant belt-and-braces for this config.
- Head file size: `10,876,789,654 B` on disk per the in-tree comment at
  `utils_mlx.py:385`. That's what gets loaded per rank.
- Quantization: `mxfp4/mxfp8, group_size=32` per-layer (auto-inferred
  from `.scales` shape at `utils_mlx.py:844-862`). It IS quantized.
- **Sharding? NO.** `DeepseekV4ShardingStrategy.shard_model` at
  `auto_parallel.py:1049-1180` iterates only:
  - `model.model.layers[i]` — the 43 body layers, and
  - `mtp_blocks = list(getattr(model.model, "mtp", []) or [])` — the
    checkpoint's MTP-1 chained heads (whatever `sanitize()` retained).
  The DSpark head is attached as `inner.dspark` (`utils_mlx.py:866`),
  which no shard-strategy code path ever iterates. Result: the full
  10.13 GB sits on BOTH nodes.

**(b) DSpark ctx-KV — 384 KB per node.**

`DeepseekV4DSparkModule.make_cache` at `deepseek_v4.py:6509-6513`:
```
return [RotatingKVCache(max_size=self.config.sliding_window)  # 128
        for _ in self.stages]
```
3 stages × 128 K entries × 512 head_dim × 2 B = **384 KB total per node**.
The `EXO_DRAFT_KV_WINDOW=4096` env in the launch is IRRELEVANT to DSpark;
it's read only by `pp_speculation.py:3498-3511` `load_draft_model()`,
which is the classic PP+`EXO_PP_DRAFT_MODEL` draft-model path. DSpark
does NOT use it.

**(c) DSpark side-channel `_DSPARK_CTX["hiddens"]` — <1 MB transient.**

Per `deepseek_v4.py:6888` and `607-618`: on every target forward, the
3 tap layers dump `h.mean(axis=2)` = `(B, L, hidden) bf16`. For verify
L=4: 3 × 4 × 4096 × 2 = 96 KB. For decode L=1: 24 KB. Transient — freed
between cycles when the next forward reset overwrites the dict at
`deepseek_v4.py:6838`.

**(d) `BatchPoolingCache` — 0 bytes at c=1.**

`EXO_DSV4_POOL_SNAPSHOT_BATCH=1` is set. That flag ONLY widens the class
tuple in `_collect_pooling_caches` (`dsv4_mtp.py:1880-1884`) to include
`BatchPoolingCache`. But at protocol batch-size c=1, `prompt_cache`
contains `CacheList(RotatingKVCache, PoolingCache, PoolingCache)` per
sparse layer — no `BatchPoolingCache` instance exists. **The flag is
INERT at c=1. Zero extra residency.**

**Correction to `root_cause_analysis.md`:** the doc mentioned the flag
without noting that `BatchPoolingCache` only exists at c≥2. Confirmed:
at protocol c=1 this flag is a no-op.

**(e) SPEC_CACHE_ROLLBACK / SPEC_CACHE_ROLLBACK_C2 stashes — <1 MB
transient per cycle.**

`EXO_DSV4_SPEC_CACHE_ROLLBACK=1` triggers `arm_spec_stash()` on rings
and pools before verify (`dsv4_mtp.py:4132-4135`). Each per-row push
appends a `(keys, values)` tuple. For L=4 verify:
- 4 rows × 43 rings × ~1 KB = 172 KB
- Pool stashes: 4 rows × (21+21+20) pool caches × ~1 KB = ~250 KB
- Total: **<500 KB per cycle, disarmed immediately post-verify**
  (`dsv4_mtp.py:4211-4213`).

**(f) SPEC_STATE_RESTORE ring snapshots — 5.5 MB TRANSIENT per verify
cycle.** ⚠️ HIGH ALLOCATION CHURN

`EXO_DSV4_SPEC_STATE_RESTORE=1` calls `sub.save_spec_state()`
(`dsv4_mtp.py:4126-4128`). At `cache.py:832-837`:
```python
return (
    None if self.keys is None else mx.array(self.keys),  # FULL COPY
    None if self.values is None else mx.array(self.values),
    self.offset,
    self._idx,
)
```
`mx.array(self.keys)` is an EXPLICIT COPY (see the header comment at
`cache.py:782-784`: "mx `__setitem__` mutates IN PLACE (aliased) — a
bare reference does NOT preserve pre-write contents — so keys/values
are COPIED here").

Per cycle: 43 rings × 128 KB = **5.5 MB per verify cycle, allocated
fresh each time.** These snapshots are discarded on the NEXT cycle's
verify entry (either freed on rejection-rollback complete or GC-freed
when the ring is re-armed). They stay live for ~200-700 ms per cycle.

Additionally, pool `save_meta()` (`cache.py:1676-1709`) copies the
`buf_kv`/`buf_gate` remainder buffers when remainder > 0. Sparse
`buf=(1, 4, 1024) bf16 = 8 KB per pool`, 21 layers × 2 pools = 336 KB.
Compressed `buf=(1, 128, 1024) bf16 = 256 KB per pool`, 20 layers =
5.12 MB. **Pool-meta snapshot: ~5.5 MB per verify cycle.**

Combined per-cycle churn from spec-state save/restore = **~11 MB
allocated + freed each verify**, on top of the existing MLX cache
churn.

**(g) SPEC_STATE_SPLIT_DIAG — NOT active (env not set).**

`EXO_SPEC_STATE_SPLIT_DIAG` at `cache.py:800-829` would eagerly force
`mx.eval` on each ring copy separately, blowing up latency. It is OFF
in this launch, correctly.

### 2.4 Steady-state total per node — reconciled

| Component | spec-OFF | spec-ON | Δ |
|---|---:|---:|---:|
| Model weights (body + embed + lm_head) | ~87 GB | ~87 GB | 0 |
| **DSpark head (NOT sharded)** | 0 | **10.13 GB** | **+10.13 GB** |
| Main KV: pool storage (all layers) | 2.27 GB | 2.27 GB | 0 |
| Main KV: rings (43 × 128 KB) | 5.4 MB | 5.4 MB | 0 |
| DSpark ctx-KV (3 × 128×512 bf16) | 0 | 384 KB | ~0 |
| **Per-node STEADY total** | **~89.3 GB** | **~99.4 GB** | **+10.13 GB** |

The observed ~97 GB healthy wired matches the ~99 GB spec-ON estimate
within ~2 GB of accounting slop (fp8/mxfp storage overhead, hc_head,
markov embedding, ring headroom, kernel/OS wired).

**Key claim from the two prior docs — VERIFIED:** the +4.7 MB/sparse-
layer batched-vs-rowseq transient delta is a real O(100 MB) figure and
is NOT alone sufficient to explain multi-GB swap. That analysis stands.

**Key claim from the fragmentation doc — VERIFIED:** MLX's Metal
allocator has size-class bucketing and is not fragmenting under normal
identical-shape allocs. Also verified. What that doc concluded as
"stochastic wired-ceiling proximity" is CORRECT in kind but was
missing the identity of the multi-GB margin eater. The identity is the
DSpark head.

---

## 3. Per-cycle TOUCHED working set (why persistent thrashing)

**Important reframing** (per consult review, 2026-08-27): under MacOS's
unified memory model, "touching a resident page" does not consume cache
headroom. It's the DELTA of unresident pages that pages in per cycle
that drives fault stall. So the per-cycle "touched" figure below is
useful mainly as a diagnostic: it bounds the working set that must be
kept warm to avoid thrash.

Per cycle at L=4 verify + γ=3 draft (per node under TP=2):

| Class | Bytes per cycle | Notes |
|---|---:|---|
| Body attention/Indexer/Compressor weights | ~3.7 GB | 43 layers × (att LoRA + comp + idx) |
| Body MoE weights (24 selected experts/layer × 43) | ~13.4 GB | random routing dominates page-in |
| Indexer pool full-scan (21 × 22 MB) | ~450 MB | every sparse layer scans full ~22 MB indexer pool |
| Main compressor pool top-k gather (~20% touch) | ~360 MB | 21 × 86 MB × 20% approx page-touch |
| Compressed pool (small enough to fully page) | ~54 MB | 20 × 2.75 MB |
| DSpark draft weights | ~1.3 GB | 3 stages, per-token expert routing |
| DSpark ctx-KV | <1 MB | 3 × 128 × 512 bf16 |
| **PER-CYCLE TOUCHED TOTAL** | **~19-22 GB** | |

The task-context claim *"~88150 entries × 512 × bf16 × 21 sparse layers
≈ 1.9 GB read per cycle per node÷TP-split"* — VERIFIED for
approximately the right reason:

- The main compressor pool is 86 MB/layer × 21 = **1.81 GB total on-node**.
- The Indexer's scoring GEMM computes `q @ pooled` against the FULL
  indexer pool (0.45 GB) every verify, and the SDPA gather from the
  main pool with `topk=512` out of 88,150 rows **is page-random and de
  facto touches most of the main pool's pages** under memory pressure.
- Under TP=2, query rows are computed on this node's `n_heads` shard,
  but the POOL is replicated — so BOTH nodes touch their full pools
  every verify. The task's "÷TP-split" applies to Q-side compute time,
  not pool-side read bytes.

**Correction to `root_cause_analysis.md` §7:** the doc mentioned the pool
random-access dominance but didn't quantify it. The correct figure is
~1.8 GB of pool storage is de-facto touched per verify cycle (fully
under any page-eviction pressure).

---

## 4. Reconciliation with observed telemetry

From `/tmp/ab/protocol352/telemetry_m4-1_on_probe.csv`:
- Peak `wired_mb` observed during collection: **99.1 GB** (transient
  snapshot), typical steady-state ~25 GB wired + ~50 GB active.
  → The "wired 97 GB" figure in the task context is likely peak-hold,
    not the OS-reported wired page count.
- Peak `compressor_mb` during warmup: **35 GB** (OS memory compressor
  actively holding compressed pages of cold model weights). Steady-state
  during decode: 90-140 MB.
- Peak `swap_used_mb` in the collected data: only **25 MB**, but this
  telemetry snapshot did NOT capture a collapsed run — the swap data
  point is inferred from separate observation.

The 1.37-1.76 GB DISK SWAP observed on collapsed runs (given) crosses a
qualitatively different threshold: the OS compressor got saturated
(≥35 GB compressed is the ceiling seen here), and further pressure
spilled to disk. At disk-swap latency (~1-10 ms per page fault on
modern SSDs, page = 16 KB → 60-600 MB/s effective throughput vs
15-25 GB/s of DRAM), each cycle's ~19 GB touched set becomes:
`19 GB ÷ ~400 MB/s effective ≈ 47 s if 100% swap-limited`. Observed:
1.8-2.3 s cycle → ~5-10% of the touched set is faulting from disk each
cycle. That is compatible with the persistent-equilibrium
characterization.

---

## 5. Why does spec-OFF fit at 500K but spec-ON thrash at 352.6K? — the real mechanism

**The 10.13 GB DSpark head IS the margin eater in kind, but the trigger
for stochastic collapse (4/14) with permanent (not sporadic) persistence
must be a MONOTONIC ratchet, not a random-per-cycle event.**

Candidate ratchets, ranked by plausibility given code and observed
signature:

### Ratchet A (leading suspect): MLX buffer-cache growth

The `_MLX_CLEAR_CACHE_INTERVAL=0` default (`batch_generate.py:103`) leaves
MLX's caching allocator holding every freed GPU buffer indefinitely for
reuse. The in-tree comment at `batch_generate.py:93-102` explicitly
identifies this class of growth as *"the dominant source of RSS growth
and ultimately what OOMs the runner"* on long Think-mode decode
(>50K tokens).

Under spec-ON, the per-cycle churn is DRAMATICALLY higher than spec-OFF
because of:
- SPEC_STATE_RESTORE ring snapshots (43 × 128 KB copies alloc'd fresh
  every verify).
- SPEC_CACHE_ROLLBACK pool/ring stashes.
- DSpark draft: fresh block-KV `mx.array()`s allocated and trimmed
  every cycle (`dsv4_mtp.py:4680-4689`).
- Varied-shape transients from acceptance-length-dependent verify
  (γ+1=4 batched, but rollback branches allocate different tensor
  shapes depending on which drafts get accepted).

Each of these frees at cycle end, but MLX RETAINS the freed buffer in
its size-bucketed cache. Over hundreds of cycles across 300-500 output
tokens, the cache accumulates size classes NEVER present in spec-OFF
(2 MB `pooled_gathered`, 2.5 MB `combined`, sundry rollback-shape
buffers). A one-time warmup cost, sure — but per the observation
that a plain 500K spec-OFF decode ran clean, spec-ON has a categorically
different cache growth trajectory. The stochasticity comes from WHICH
size classes get generated on any given run (depends on acceptance
pattern), and the persistence comes from once the retained-cache
exceeds the wired margin, MLX never releases it without an explicit
`mx.clear_cache()`.

**This is the ratchet that fits: monotonic (retained cache only grows
until explicit clear), spec-ON-only (spec-OFF doesn't generate
rollback/draft-block transient shapes), stochastic in onset
(acceptance patterns vary), permanent once triggered.**

### Ratchet B (secondary): Python GC keeping ArrayDesc objects alive

`batch_generate.py:105-113` documents that MLX's C++ `array::detach()`
runs but the Python wrapper's `ArrayDesc` shared_ptrs get trapped in
ref cycles until Python gen-2 GC runs. `_GC_COLLECT_INTERVAL=0` is
also OFF in the launch env, so ArrayDescs accumulate. Same class of
ratchet but at the descriptor level, not the buffer level.

### Ratchet C (previously proposed, weak): PoolingCache growth events

The prior `fragmentation_pool_analysis.md` §2 analysis of pool growth
events every ~1024 tokens per sparse layer is real but pays a
transient ~1.9 GB burst that BOTH paths pay identically. It cannot
account for the per-run stochastic difference between fits-500K
spec-OFF and collapse-352K spec-ON. **Refuted as sole cause; still a
real amplifier once things are tight.**

### Consultant caveat (integrated)

The framing "per-cycle touched set > cache headroom" I used in an
earlier draft is confused: resident weights being touched costs zero
extra headroom. What actually needs headroom is TRANSIENT allocations
(activations, snapshot copies, DSpark draft block-KV, gathered
tensors) — and those, from the byte math above, are a few hundred MB
to a few GB per cycle. The 10.13 GB DSpark head is REAL residency
consumption. The per-cycle transient churn is what compounds into a
retained MLX cache ratchet.

---

## 6. Ranked root-cause fixes — env-gated, NO context caps

### Fix #1 (recommended, ZERO code change): `EXO_MLX_CLEAR_CACHE_INTERVAL=64`

**Site:** `start_cluster.sh:133` (default currently `0`), applied at the
runner env-allowlist site `start_cluster.sh:1646`. The consumer is
`batch_generate.py:103` reading the env, and the trigger points already
exist at `batch_generate.py:3215, 3376, 4739, 5261`.

**Expected savings:** unbounded on the ratchet's ceiling. In the worst
case where the retained MLX cache had grown by ~2-4 GB across a long
decode, each explicit `mx.clear_cache()` releases all of it back to
the OS. Over 60+ cycles the cache never grows past a bounded working-
set worth of buffers.

**Trade-off:** the code comment at `batch_generate.py:100-102` notes
that clearing forces subsequent allocations from a cold pool. On
current hardware the cost is measured at 5-15% throughput hit at
INTERVAL=64. But given the alternative is 1 tok/s vs a healthy
24 tok/s, even a 15% cut leaves ~20 tok/s — a 20x improvement over
collapsed state.

**Why this first:** it's ONE env var. Requires no repo edit. Falsifies
the ratchet hypothesis directly: if collapse rate at 352.6K drops from
4/14 to 0/14 at INTERVAL=64, ratchet confirmed. If unchanged, fall
through to Fix #2.

### Fix #2: Shard the DSpark head through the existing TP path

**Site:** `src/exo/worker/engines/mlx/auto_parallel.py:1153-1178`. Add
an analogous block AFTER the `mtp_blocks` loop:

```python
# DSpark 3-stage draft head — mirrors mtp block sharding.
# Only shardable pieces: FFN experts (attn stays replicated because
# DSv4's LoRA-decomposed attn shape doesn't split cleanly, per the
# strategy's class docstring). Also skip main_proj/norm/hc_head/markov —
# all bandwidth-negligible (~150 MB combined) but small-shape hard-to-
# shard tensors.
dspark = getattr(model.model, "dspark", None)
if dspark is not None:
    for j, stage in enumerate(dspark.stages):
        mx.eval(stage.parameters())
        stage.ffn.sharding_group = self.group
        self.all_to_sharded_linear_in_place(stage.ffn.shared_experts.gate_proj)
        self.sharded_to_all_linear_in_place(stage.ffn.shared_experts.down_proj)
        self.all_to_sharded_linear_in_place(stage.ffn.shared_experts.up_proj)
        self.all_to_sharded_linear_in_place(stage.ffn.switch_mlp.gate_proj)
        self.sharded_to_all_linear_in_place(stage.ffn.switch_mlp.down_proj)
        self.all_to_sharded_linear_in_place(stage.ffn.switch_mlp.up_proj)
        mx.eval(stage)
        mx.clear_cache()
```

Gate with a new env `EXO_DSV4_DSPARK_TP_SHARD=1` for safe rollout;
default OFF pending sanity-check that mxfp4/mxfp8 quantization at
`group_size=32` divides cleanly under axis-1 sharding of moe_intermediate
(2048 / 2 ranks = 1024, group_size 32 divides evenly).

**Expected savings:** approximately **3-5 GB per node**, NOT 5+ GB. Only
the MoE FFN weights of the 3 DSpark stages shard; the attention (LoRA
projection tensors, hc_head, main_proj, norm, markov embedding tables,
confidence_proj) stays replicated. Rough breakdown of the 10.13 GB DSpark
head:
- FFN experts (3 stages × MoE-shape) — probably ~6-7 GB.
- Attention (3 stages × LoRA) — ~1.5-2 GB.
- Main_proj / markov / hc_head / norms — ~1.5-2 GB.
- Sharding cuts the ~6-7 GB FFN portion by half → savings ~3-3.5 GB.

**Trade-off:** Sharding adds `sum_gradients` (input-side) and `all_sum`
(output-side) collectives per stage per draft token. With γ=3 that's up
to 6 small-tensor RDMA collectives per draft cycle. On a jaccl RDMA link
this is bandwidth-cheap but latency-sensitive (~200-500 μs per collective
under Thunderbolt 5). Draft time inflates by ~1-3 ms per cycle. At 100 ms
healthy verify cycle this is a ~1-3% throughput cost.

**Why NOT this first:** requires code change (though minimal). Savings
narrower than intuitive estimate. Requires validation that:
1. mxfp4 shards cleanly at `group_size=32` on the intermediate axis
   (needs quick verify — sanity check that `moe_intermediate_size /
   num_ranks / group_size = 2048/2/32 = 32` is integer ✓).
2. `DeepseekV4DSparkStage.__call__` doesn't fail on
   `_ap_i`-vs-`_global_i` type confusion that the main model's `shard()`
   handled but the DSpark loop doesn't yet mirror.

### Fix #3: Reduce spec-verify snapshot churn — gate `SPEC_STATE_RESTORE` to non-rotating ring case

`EXO_DSV4_SPEC_STATE_RESTORE=1` triggers `mx.array(self.keys)` copies of
every ring on every verify cycle (~5.5 MB alloc + free per cycle). At
352.6K decode all rings are rotating; the copy IS necessary for
correctness on rejection rollback. But: on cycles where FULL draft
acceptance is likely (a≈2-3, common at long context per prior
measurements), the snapshot is wasted work.

**Site:** `dsv4_mtp.py:4118-4145`. Add a preview: check
`gen_batch.last_reject_rate` or a decaying acceptance rate; if it's
been >0.9 for the last 32 cycles, skip snapshotting and fall back to
COMMIT-FORWARD on the rare rejection.

**Expected savings:** eliminates ~5.5 MB × 400 cycles/s = 2.2 GB/s of
allocation churn on high-acceptance cycles → reduces MLX cache
retention pressure by ~30-50%. Complementary with Fix #1.

**Trade-off:** on rejection cycles when snapshot is skipped, the
commit-forward penalty is ~41% decode t/s per the code comment at
`dsv4_mtp.py:709-710`. If acceptance drops below the 0.9 threshold the
gate re-engages snapshot mode. Requires more careful tuning.

**Why NOT this first:** needs code change and acceptance-rate tracking
logic. Fix #1 subsumes most of this at zero code cost.

### Fix #4 (structural, longer-term): move DSpark head to memory-mapped weights with drop-under-pressure semantics

The consultant flagged that `mx.load(lazy=True)` wouldn't help because
every draft cycle touches all 3 stages so all weights get materialized.
BUT — if the DSpark head were kept as `mmap`-backed clean pages (never
copied into Metal buffers, dispatched via a `blit` from the mmap
region), the OS could DROP those pages under pressure without disk
swap. This requires deeper mlx-lm allocator surgery and is outside
the "env-gated" bar. Not recommended for this incident window.

---

## 7. Verify/correct prior-doc claims — summary table

| Claim | Verdict | Correction |
|---|---|---|
| root_cause: batched-verify adds +4.73 MB/sparse-layer transient | **VERIFIED** | correct as stated |
| root_cause: pool storage 1.95 GB total | **CORRECTED** | 2.27 GB (missed sparse indexer pool 0.45 GB) |
| root_cause: async fencing bounds delta to O(10 MB) | **VERIFIED** | correct |
| root_cause: recommend chunked verify at depth | **PARTIAL** | reduces transient by 50 MB, cannot fix the 10 GB DSpark residency problem |
| fragmentation: allocator not fragmenting | **VERIFIED** | correct — MLX buffer cache reuses size classes |
| fragmentation: pool growth doubling ~1.9 GB burst | **VERIFIED** | correct math; growth cadence ~5 min per node at 352.6K |
| fragmentation: "stochastic wired-ceiling proximity" as verdict | **CORRECT IN KIND, INCOMPLETE** | identified the SHAPE of failure but not the specific 10 GB DSpark residency that shifted the entire equation |
| fragmentation: EXO_MLX_CLEAR_CACHE_INTERVAL as a fix mentioned briefly | **UPGRADED** | this is the #1 fix, not a footnote |

---

## Executive summary — margin-eater ranking and recommended fix

**Margin-eater ranking (spec-ON extra residency per node vs spec-OFF):**

1. **DSpark head weights: +10.13 GB.** Loaded via
   `utils_mlx.py:439-446` before TP sharding; the shard iterator at
   `auto_parallel.py:1153-1178` iterates `model.model.mtp` only, never
   touches `model.model.dspark`. Both nodes each carry the full
   10.13 GB, quantized as mxfp4/mxfp8 group=32 on disk.
2. **Cumulative MLX buffer-cache retention: multiple GB (unbounded).**
   `EXO_MLX_CLEAR_CACHE_INTERVAL=0` (default; `batch_generate.py:103`)
   plus `EXO_GC_COLLECT_INTERVAL=0` means every rollback-shape and
   draft-block transient allocated during spec-ON verify sticks in
   MLX's caching allocator forever. The in-tree comment
   (`batch_generate.py:93-102`) explicitly says this is *"the dominant
   source of RSS growth and ultimately what OOMs the runner."*
3. **Per-cycle snapshot/rollback churn: ~11 MB alloc + free per verify.**
   `EXO_DSV4_SPEC_STATE_RESTORE=1` +
   `EXO_DSV4_SPEC_CACHE_ROLLBACK{,_C2}=1` copy every ring (`mx.array`,
   `cache.py:832-837`) and pool remainder buffer every cycle. Not
   large individually but a big contributor to ratchet #2 above.

DSpark ctx-KV (384 KB), BatchPoolingCache (inert at c=1), the
`_DSPARK_CTX["hiddens"]` side channel (96 KB transient), and per-verify
row scratch (all bounded O(MB)) are noise-level and NOT the answer.

**Reconciliation:** spec-OFF's 87 GB wired at 500K left ~37 GB of OS
cache headroom — enough that per-cycle transients and OS-compressor
work never crossed disk-swap. Spec-ON's 97 GB wired at 352.6K leaves
only ~27 GB, and the ratchet of MLX's retained buffer cache eats into
that over 300+ cycles until crossing the disk-swap threshold. Once
disk-swap begins, every cycle's ~19 GB touched-set forces
`~1.5-2 GB/s` of page-in at ~400 MB/s effective SSD throughput →
persistent 1.8-2.3 s per cycle equilibrium, matching observation.

**#1 recommended fix: set `EXO_MLX_CLEAR_CACHE_INTERVAL=64` in the
launch env.** One line, zero code changes, gates in
`start_cluster.sh:133`. Already implemented at
`batch_generate.py:3215-3220`. Expected outcome: eliminates the ratchet;
if the ~10 GB static DSpark residency is truly the binding constraint
after that, collapse-rate stays elevated and Fix #2 (shard DSpark)
becomes the follow-on. If collapse-rate drops to 0/14, ratchet
confirmed, DSpark residency is livable (as before). One diagnostic
env change resolves the ambiguity between static-residency and
dynamic-ratchet theories in a single 14-run cycle.

**Sequence:** ship Fix #1 first (env, no code); measure. If needed,
Fix #2 (DSpark shard, ~3-3.5 GB extra headroom, +1-3% draft latency).
Fix #3/#4 are secondary.

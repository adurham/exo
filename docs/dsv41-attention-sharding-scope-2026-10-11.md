# DSv4.1 — Scope: shard the replicated attention work across the existing 2 nodes (2026-10-11)

**Author:** p20 campaign (delegated, offline). **Branch** `p20/attention-sharding-scope`
(worktree `/private/tmp/p20-attn-shard-wt`, off `origin/main` `beda06114`).
**Live tree:** exo `ade481853` (`deploy/next22-rankfix`) on both nodes; model
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, 2-rank tensor-parallel
(`worldSize=2`, jaccl/RDMA over one TB5 cable pair).
**Every number is labelled `[MEASURED]`, `[DERIVED]`, `[MODELED]`, `[BOUND]` or `[ESTIMATE]`.**
Owner question (verbatim intent): *the replicated attention projections that currently
refuse to split — can that be done? if it's done now would it get us some benefits now?*

---

## VERDICT (read this first)

1. **The premise is largely mis-attributed.** The *big* attention projections are **already
   head/group-split** at `world=2` today: `attn.wq_b` (32 of 64 heads), `attn.wo_b` (4 of 8
   groups, input-sliced) and `attn.wo_a` (rank's 4 groups) — `exl3_build.py:702-718`. So
   "sharding the replicated attention projections" is **mostly already done**; there is no
   large replicated *projection* left to split.
2. **The real replicated term is the INDEXER COMPUTE, not a projection.** The whole `Indexer`
   (score + top-k + emit) is **100 % duplicated** on both ranks (`indexer.py` has no collective;
   its weights are built pre-head-slice and never sliced — `exl3_build.py:713-724` slice list
   touches only `attn.wq_b`/`attn.wo_b`). It runs on **8 of 40** layers
   (`index_source_layer_ids = [2,8,14,20,24,28,32,36]`, config-verified) and is **O(context)**.
   Measured class share **31.1 % @100K** (`[MEASURED]`, *unreconciled*) and `attn.indexer.score`
   **~43 % of deep prefill wall @~340K** (`[MEASURED]`). **This duplicates the single largest
   term in the owner's pain band.**
3. **It can be sharded NOW, and it is worth a bounded build — gated on one reconciliation.**
   A query-row band of the indexer's *query side* is the old V4-Flash seq-split pattern
   (still in-tree, `deepseek_v4.py:5448-5462,5777-5806`); it halves the per-rank indexer work
   and is reconstructed by a small `cidx` exchange. Modeled **`+15.4 %` per-chunk wall @100K
   (`1.18×`)** `[MODELED]`; **`+6.9 % @25K → +16.9 % @130K`**.
4. **The literal lever the owner asked about — splitting the residual replicated *weights*
   (`wq_a`, `wkv`) — is NOT worth building.** They are small (`~9-14 M MAC/token/layer`, ~15 %
   of attention dense work, <1 % of the MoE block) and are blocked by *structure* (shared q
   latent / single MQA KV head) plus the 128-block `_block_bounds` uniform-`world` constraint.
   Cost > gain.
5. **Do NOT build the full attention row-split** (each rank takes half the query rows × full
   heads): it is **like-for-like** on the already-split classes and needs a `~1.7 GB/rank`
   full-head weight reload `[DERIVED]`. The correct "phase 2" is **Megatron sequence
   parallelism** (reduce-scatter the residual, run the row-local replicated ops on the band),
   which captures the same gain at **no extra weight memory** but is a much larger change.

**Ranked recommendation:** (0) reconcile the indexer share `[gate]` → (1) build the narrow
indexer query-band (~`3-6` eng-days `[ESTIMATE]`) → (2) only then consider SP-hybrid
(~`12-25` eng-days `[ESTIMATE]`). Owner decision points at §7.

---

## 1. Corrections to the owner's premise (with evidence)

| # | Premise | Reality | Evidence |
|---|---|---|---|
| C1 | "the replicated attention **projections**" | The big projections are **split** today at w=2 → not replicated | `exl3_build.py:702-718` |
| C2 | they "**refuse to split**" | At w=2 nothing raises; the *hard `raise`s* are w=3 only | `exl3_build.py:324-325` (`n % (8·world)`); `load.py:80-85` (`world not in (1,2)`) |
| C3 | the head is replicated | The head is **vocab-split** (`ShardedHead`, `_SHARD_HEAD=1`) | `exl3_build.py:808-812` |
| C4 | — | The genuine replicated O(context) term is the **indexer compute** | `indexer.py` (no collective); `config` `index_source_layer_ids` |

---

## 2. Replicated vs SPLIT @ `world=2` — enumeration with file:line

Shard decision point: `exl3_build.py:697-724`. `attn_tp = world>1 and _SHARD_ATTN and tp_on and
group is not None` (`:698`) → **True** at the live defaults; `_SHARD_HEAD` (`:808`) → **True**.
The dense loop (`:713-724`) slices **only** three tails; everything else falls to the `else` at
`:724` (FULL).

| tensor / op | status @w=2 | evidence |
|---|---|---|
| `attn.wq_a` `[1280,5120]` (LoRA q down-proj) | **REPLICATED** (shared q latent, no head axis) | `:724` else; `attention.py:150` |
| `q_norm` | **REPLICATED** | `:724`; `attention.py:150` |
| `attn.wq_b` `[32768,1280]` | **SPLIT** axis=out → 32 heads/rank | `exl3_build.py:715-718`; `attention.py:151` |
| `attn.wo_a` (8×`[1024,4096]`) | **SPLIT** by group → 4 groups/rank | `exl3_build.py:707-709` |
| `attn.wo_b` `[5120,8192]` | **SPLIT** axis=in | `exl3_build.py:715-718` |
| `attn.wkv` `[512,5120]` (single MQA KV head) | **REPLICATED** | `:724`; `attention.py:155` |
| KV rope + fp8/fp4 fake-quant, window/comp cache | **REPLICATED** (identical both ranks; keys are shared) | `attention.py:155-172` |
| `attn.attn_sink` | **SPLIT** per-head | `exl3_build.py:757-760` |
| `sparse_attn` (sdpa, `attn.sdpa` 13.6 %) | **SPLIT** by head → 32 heads × all 2048 rows | `attention.py:227-228` |
| `attn.all_sum` (the **only** attention collective) | head-shard partial `[2048,5120]` bf16 = **20.97 MB/layer** `[DERIVED]` | `attention.py:241-243` |
| **`Indexer`** (score + top-k + emit, 31.1 %@100K) | **REPLICATED** (full heads, full rows, full `nb`) | `indexer.py` (no collective); not in slice list `:713-724`; `attention.py:197` |
| `compressor` (kv-source layers 2/8/14/20) | **REPLICATED** | `:724`; `compressor.py:48-95` no collective |
| routed experts (384) | **SPLIT** intermediate-width | `exl3_build.py:684`; `exl3/loader.py:286-299` |
| shared experts w1/w2/w3 | **SPLIT** | `exl3_build.py:719-722` |
| RMSNorm (attn/ffn/final/q/kv) | **REPLICATED** (plain fp32) | `:752-756`; 1.7 %@100K |
| hyper-connections `hc_*` (5.2 %@100K) | **REPLICATED** (plain fp32, row-wise residual mix) | `model.py:100-105`; `:752` |
| `ffn.gate.weight/bias(+bias_vl)` | **REPLICATED** | `:752-756` |
| `embed.weight` `[129280,5120]` | **REPLICATED** (**1.318 GB** bf16 `[DERIVED]`, zero compute) | `exl3_build.py:815` |
| `head` | **SPLIT** vocab (`ShardedHead`) | `exl3_build.py:808-812` |
| MoE `moe.all_sum` (separate collective) | head/… split tail | `moe.py:180-189` |

**Cross-check vs the measured @100K shares** (`docs/dsv41-prefill-measure-pack-2026-10-11.md`
Task B, weighted 40-layer fold): indexer 31.1 %, MoE down 19.1, sdpa 13.6, dense-o 7.2,
MoE up 7.2, HC 5.2, dense-qkv 4.3, MoE glue 3.3, shared 3.2, MoE gate 3.0, norms 1.7,
dense-other 0.7, MoE-router 0.5. The **replicated** set on the per-rank wall is
≈ **indexer 31.1 + HC 5.2 + norms 1.7 + (wq_a≈0.55 + wkv≈0.22, from dense-qkv 4.3) + router 0.5
≈ 39.3 %** `[DERIVED]`. The split set (MoE experts ≈29.3 + sdpa 13.6 + dense-o 7.2 + shared 3.2
+ glue 3.3 ≈ 56.6 %) is what the head/expert shard already covers. **The enumeration is
consistent with the fold** — the divergence flagged in the measure pack is the *magnitude* of
the indexer share, not its replicated/split status.

---

## 3. Per-class shardability for the CURRENT pair

### 3a. The LoRA q/o projections — why they cannot head-split (and what a re-layout would need)

- **Format on the LIVE path is EXL3 trellis, not affine** (`DENSE_MODE` default `exl3`,
  `exl3_build.py:550,630-632`; the affine path shards identically, `:691-696`). Trellis
  `(in_tiles, out_tiles, packed)` uint16, 16-wide tiles; `suh` (in signs) / `svh` (out signs)
  (`exl3/ref/layer.py:24`, `exl3/loader.py:16-21`).
- **Exactness comes from blockwise rotations** (`HAD_DIM=128`, `exl3/hadamard.py:13,56-82`;
  `reconstruct.py:54-59`): an output slice on a **128-block boundary** commutes with
  reconstruction. That is *why* the head-split works at all (`head_dim=512=4·128`).
- **The wall is `_block_bounds` (`exl3_build.py:316-326`):** `if n % (8·world): raise` and only
  the **uniform** `rank·n//world` tiling is expressible. The axis is a raw feature axis, **not
  a head axis** — a head cut is a side-effect valid only because `head_dim` is a 128-multiple
  and `n_heads` divides by `world`. Nothing finer/custom (e.g. 48/16) is expressible.
- **Why `wq_a` (and only `wq_a`) cannot head-split:** `wq_a: Linear(dim → q_lora_rank)` produces
  the **shared** q latent `[.,1280]` — **no head axis** (`attention.py:102,150-151`). Every head
  *and* the indexer consume all 1280 features (`attention.py:197`). Sharding its output requires
  an `all_gather` of the latent — a collective the live path does **not** have (its single
  attention collective is the head-shard `all_sum`, `attention.py:241-243`). `wkv` is a **single
  shared KV head** (`head_dim=512` is the whole KV vector) → not head-splittable at all.
- **A head-aligned re-layout / small bf16 shard:** a head-index trellis boundary is *legal*
  (512=4·128) but needs new code + a **re-quant**; it does **not** unblock `wq_a`. A bf16 `wq_a`
  shard is **13.11 MB/rank** `[DERIVED]` (6.55 M params × 2 B) but `wq_b` still needs all 1280
  latent features on each rank → it just **moves the cost to a new per-layer collective**
  (20.97 MB/2048-row chunk, same class as today's). **No free win; not worth building.**

### 3b. sdpa + indexer COMPUTE — can the seq-split be ported?

- **The old machinery is still in-tree and live for v4.** Knob `EXO_DSV4_SEQ_SPLIT`
  (`auto_parallel.py:121-126`, default ON); gating sets `sharding_group` **only** on
  `SparseCompressedAttention`/`CompressedAttention` (`:1122-1126`); mechanism in
  `deepseek_v4.py:5448-5462` (contiguous query band `_band=L//N`, `_seq_lo=rank·band`),
  `:5444-5447,5464-5480` (**KV/compressor/pool kept FULL** on every rank), `:5537-5557`
  (indexer banded internally via `seq_band`), `:5777-5796` (**zero-pad the band to full `L` +
  `all_sum`** — the default, ARQ-reliable, bit-exact path) or `:5801-5806` (fallback
  `all_gather`). It is **not wired to v41** (the v41 loader bypasses `auto_parallel`).
- **A query-row split partitions attention cleanly.** Softmax normalization in `sparse_attn` is
  over **keys** (the gather axis), never queries (`sparse_attention.py:334-356,485`); query
  tiles concatenate independently (`:494`). Each row's output depends only on its own `q` and
  its own gathered key set → row partition is exact **modulo kernel tiling** (see §6). The
  design note states the same (`deepseek_v4.py:272-273` "Quality is exact").
- **Composition with the existing head-split (the crux).** Per-rank projection work ∝
  `rows × heads`. Head-split today = `2048×32`; a full row-split = `1024×64` = **identical** →
  `wq_b`/`sdpa`/`wo_a`/`wo_b` are **like-for-like**, *not* saved. A row-split therefore only
  helps the **row-local replicated** classes. Moreover the head-split and a row-split cannot
  both be halved without a **2-D exchange** (quarter-blocks need 4 ranks, not 2). **The correct
  phase-2 shape is Megatron sequence parallelism**, not a plain attention row-split (§7).
- **The indexer is the class that a row band genuinely halves** (its score/top-k is O(n·nb)
  *per query row*): banding its query side gives each rank `1024 rows × all 32 index heads`,
  and its **key side stays full** (coherence). Output `cidx` is `[b,n,index_topk=512]` int32
  (`config:137`), deterministic per row → the band's `cidx` equals the full run's slice, and the
  exchange is tiny (**4.19 MB/layer** `[DERIVED]`, **8 source layers** → **34 MB/chunk**).
- **Non-source layers reuse** `shared.topk_idxs` (`attention.py:200-201`) — so **one gather per
  source layer (8), not per layer (40)**, and the source/consumer band must be **consistent on
  the same rank**.

### 3c. `head` / norms / HC / embeddings — worth splitting at all?

| class | size `[DERIVED]` | compute @100K | split-now? |
|---|---|---|---|
| `head` | already vocab-split | — | **already done** (`:808-812`) |
| `embed.weight` | **1.318 GB bf16** | 0 (gather) | **memory only** — a shard saves RAM, **no time** (row-indexed lookup, trivially partitionable) |
| RMSNorm | ≤20 KB/layer | 1.7 % | **row-local** → folds into phase 2 (SP), not worth alone |
| hyper-connections `hc_*` | ~1.97 MB/layer fp32 | 5.2 % | **row-local** → folds into phase 2; the largest non-indexer replicated term |
| `wq_a` / `wkv` | 6.55 M / 2.62 M params | ~0.55 % / ~0.22 % | **No** (structural wall, §3a) |
| `ffn.gate` (router score+topk) | ~3.93 MB bf16 | 0.5 % | **No** (tiny; row-local → phase 2) |

---

## 4. MODELED band — per-chunk wall delta at the four owner depths

Inputs: measured class shares @100K (`[MEASURED]`, measure pack Task B); a saturating indexer
share fitted to the measured anchors **31.1 % @100K** and **~43 % @340K** (`[MODELED]`,
`Smax=51.6 %, x0=68K`); the non-indexer replicated row-local set held constant at
**8.17 %** `[DERIVED]` (HC 5.2 + norms 1.7 + `wq_a` 0.55 + `wkv` 0.22 + router 0.5). *Removing
X % of wall is a `1/(1−X)` speedup.*

| depth | indexer share `[MODELED]` | **P1** (indexer band) save / speedup | **P2** (SP-hybrid) save / speedup |
|---:|---:|---:|---:|
| **25K** | 13.9 % | **6.9 %** / **1.07×** | 11.0 % / 1.12× |
| **50K** | 21.9 % | **10.9 %** / **1.12×** | 15.0 % / 1.18× |
| **100K** | 30.7 % | **15.4 %** / **1.18×** | 19.5 % / 1.24× |
| **130K** | 33.9 % | **16.9 %** / **1.20×** | 21.0 % / 1.27× |

**Added/changed collective payloads** `[DERIVED]` (chunk = 2048 rows, hidden 5120 bf16):
P1 adds `cidx all_gather [1,2048,512] int32` **4.19 MB/layer × 8 = 34 MB/chunk** — negligible
beside the existing per-chunk sync (two `[2048,5120]` bf16 all_sums ≈ **1.68 GB/chunk**). P2
**replaces** the attention `all_sum` (20.97 MB/layer) with a `reduce_scatter` half (10.49 MB/layer)
plus small latent gathers (`q` 5.24 MB + `kv` 2.10 MB + `cidx` 4.19 MB per layer) — **net ≈ a wash
(+~2 % wire)**. **`sdpa` (13.6 %) is a caveat [BOUND]:** its **head-split** work is not saved by a
row band, but the per-row KV gather traffic (`rows×k×d`, MQA, shared indices) is *duplicated* today
— if `sparse_attn` is gather/bandwidth-bound at depth, part of the 13.6 % is effectively replicated
and P2's number is a **floor**; profile before quoting the sdpa term.

**Decode (M=1): unchanged, small.** The mechanism is length-gated (V4 used `_SEQ_SPLIT_MIN_L=16`,
`deepseek_v4.py:278`); decode stays replicated, so decode sees **no gain and no expected
regression** beyond a possible sub-1 % extra-sync. (Decode is deprioritized per the brief.)

---

## 5. Effort `[ESTIMATE]` and risks

| piece | effort `[ESTIMATE]` | scope |
|---|---|---|
| **(0) Reconcile the indexer share** | **0.5-1 d** | re-run the Task B fold with `DSV41_INDEXER_HIER` pinned + the phase-22 method on the same shapes; the gain is gated on this |
| **(1) Indexer query-band + `cidx` all_gather** | **3-6 d** | plumb a `(lo,hi)` band into the v41 `Indexer` call (`attention.py:197`), keep key side full, add the `cidx` `all_gather` (or a padded `all_sum`), keep the 8 source/consumer layers band-consistent, leave decode replicated |
| **(2) Megatron sequence parallelism (SP-hybrid)** | **12-25 d** | reduce-scatter the residual to bands; run row-local ops (norms, HC, gate, `wq_a`, `wkv`, indexer) on the band; all_gather small latents; keep head-split `wq_b`/`sdpa`/`wo` as today |

**Correctness / memory / regression risks.**
- **Indexer share is UNRECONCILED** — the measure pack flagged a **3.7× divergence** from
  phase-22 (3.6 %@16K vs 13.5 %@16K). **The entire P1/P2 gain is gated on this.** *(Risk: high.)*
- **"Bit-exact" is a claim, not a fact** — `M=1024` vs `M=2048` can pick a different Metal
  matmul tiling/split-K, perturbing fp scores so **near-tied top-k can flip**. Validate with
  **top-k overlap + final-logit diffs**, do not assert exactness. (The v4 "bit-exact" claim
  carries the same caveat.) *(Risk: medium.)*
- **Source/consumer band coupling** — layer 20 is the candidate source; consumers 24/28/32/36
  must band **consistently** (the mask must be *produced* banded, not full-then-sliced).
  *(Risk: medium.)*
- **New collective reliability** — the live `all_gather` comment (`exl3_build.py:390-392`,
  "JACCL mesh all_gather rejects this size") argues for the **padded `all_sum`** variant (v4's
  default) rather than a bare `all_gather` for P1. Verify int32 `all_gather` support if used.
  *(Risk: low-med.)*
- **Memory** — P1: **none**. P2: must re-verify the banded residual + gathered latents fit the
  served-context envelope (net likely neutral-to-slightly-up), and the HC path carries
  **multiple residual streams** — reduce-scatter/gather the **collapsed 5120-wide** tensor.
  *(Risk: medium.)*
- **Regression surface** — P1 must not move decode or the (already-shipped) M2 hierarchical
  pass; P2 touches **every layer's residual layout** and needs the full live quality battery.
  *(Risk: med-high for P2.)*
- **Numeric gates** — P1 changes *which rank* computes the indexer (no new numerics if the
  banded top-k matches) → a **top-k-overlap RED test** + a final-logit check; P2 → the standard
  two-phase live battery.

---

## 6. Recommended ordered path + owner decision points

1. **Reconcile the indexer share** (offline attribution re-run). *Gate: if the reconciled @100K
   share is ≳20 %, proceed; if it collapses toward the phase-22 3.6 %, STOP — the lever is
   small.*
2. **Phase 1 — indexer query-band + `cidx` exchange** (`3-6 d`). Bit-exact-modulo-tiling; no
   memory change; captures ~half the indexer share. RED-prove the banded top-k equals the full
   top-k on real shapes before any live A/B.
3. **Phase 2 — SP-hybrid (Megatron sequence parallelism)** (`12-25 d`) *only if* phase 1 lands
   AND a fresh deep attribution still shows the replicated set ≳30 % of wall. This is the shape
   that captures the **full** replicated set (indexer + HC + norms + `wq_a` + `wkv` + gate) at
   **no extra weight memory** — unlike a plain attention row-split (rejected: like-for-like +
   ~1.7 GB/rank reload).

**Owner decision points:** (a) fund the indexer-share reconciliation; (b) fund phase 1;
(c) whether phase 2 (a residual-layout change across all 40 layers) is in scope now, or parked
behind a third node / a fresh deep attribution.

**Not recommended:** splitting the residual replicated **weights** (`wq_a`, `wkv`) — small and
structurally blocked (§3a); embeddings — memory, not time (§3c); a plain full attention
row-split — like-for-like on the head-split classes + weight reload (§3b).

---

## Method, provenance, non-disturbance

- **OFFLINE. 0 boots, 0 config changes, 0 generation requests, 0 node-side file edits.** Node
  contact was **read-only over ssh** (one `cat config.json` on `macstudio-m4-2` to read
  `compress_ratios` / `*_source_layer_ids` / candidate settings); the studio pair was serving
  live traffic and was not touched otherwise. No build was started.
- **Own worktree** `/private/tmp/p20-attn-shard-wt` (branch `p20/attention-sharding-scope`, off
  `origin/main` `beda06114`); the shared checkout at `ade481853` and the two live sibling
  sessions were left untouched.
- Two independent offline code-reads (this session): dense/LoRA sharding + the quant-layout wall
  (`scratch`-side report), and the attention/indexer compute path + seq-split portability
  (report). Every file:line above was **re-verified against the live tree** by this author;
  the v4 seq-split mechanism (`deepseek_v4.py:5440-5559,5775-5815`), the v41 shard decision
  (`exl3_build.py:316-326,697-724,808-812`), the attention collective (`attention.py:140-244`)
  and the node config were read directly.
- A second-opinion consult (reference model) confirmed the composition claims and corrected the
  "bit-exact" wording to "validate via top-k overlap" and named the phase-2 shape as Megatron
  sequence parallelism.
- Prior art used: `docs/dsv41-prefill-measure-pack-2026-10-11.md` (per-op shares),
  `docs/dsv41-tp3-scaling-readiness-2026-10-11.md` (the w3 raise sites / live-path correction),
  `docs/benchmarks/phase20-throughput/ROUND-PRICING.md` (dense-slice pricing), and the in-tree
  `EXO_DSV4_SEQ_SPLIT` mechanism.

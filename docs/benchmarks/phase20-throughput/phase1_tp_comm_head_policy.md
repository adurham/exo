# Phase 20 — Phase-1 design input: TP communication, DSpark head weights/memory, gamma policy, rank plumbing (BRIEF R2, Q1–Q6)

Forensics pass R2. READ-ONLY: source + live `ssh studioN` + `GET /state`. No generation
requests, no process control. Sources of record:

* exo `deploy/next16-instr` (= next14-gamma incl. per-request `spec_gamma`, template commit
  `01c416b10`), path `/private/tmp/next16-instr/src/exo/`.
* mlx-lm fork pinned `6cc9c1e`, path `/Users/adam.durham/repos/exo/mlx-lm/mlx_lm/models/deepseek_v41/`.
* Checkpoint metadata read live from `studio1` `~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw/`
  (39-shard `model.safetensors.index.json` + per-shard headers).
* Live facts: `ps eww` of the current runner on both nodes; `/state` via
  `curl http://192.168.86.48:52415/state`; `docs/` + `PERFORMANCE_HISTORY.md`.

Every claim is tagged `[src]` (read from the file cited) or **INFERRED** (derived, not
directly stated at source). Line numbers are to the pinned checkouts above.

---

## Q1 — TP collective inventory of one decode forward

### 1.1 How the model is sharded

DSv4.1's tensor parallelism is **engine-owned**, built *inside* the loader, not by exo's
generic placement — `mlx-lm/docs`:

> `load.py:3-11` — "its tensor parallelism is built INTO `exl3_build.build_model` (routed
> experts take one rank's intermediate-width slice and are `all_sum`ed; dense projections
> are sharded on 128-wide Hadamard blocks; the head is vocab-sharded)."

The only thing taken from exo is the geometry: `device_rank` / `world_size` off a
`TensorShardMetadata` (`load.py:68-96`, rejects world ∉ {1,2} and any multi-rank Pipeline
shard). `world=2` on this cluster (`/state`: both runners are `TensorShardMetadata` with
`"worldSize": 2`, `deviceRank` 0 and 1, `startLayer 0 / endLayer 40`).

Gates that decide what is sharded (all default ON in `exl3_build.py:354-357`):

```
_SHARD_SHARED = os.environ.get("DSV41_TP_SHARED", "1") == "1"
_SHARD_ATTN   = os.environ.get("DSV41_TP_ATTN",   "1") == "1"
_DRAFT_SHARD  = os.environ.get("DSV41_DRAFT_SHARD","1") == "1"
_SHARD_HEAD   = os.environ.get("DSV41_TP_HEAD",   "1") == "1"
```

### 1.2 Per-layer collective sites (body, 40 layers)

| op | site | collective | active in production? |
|---|---|---|---|
| attention out-proj | `attention.py:229-231` | `attn.all_sum` (heads sharded, partial sum) | **disabled** — see 1.4 |
| MoE tail | `moe.py:180-189` | `moe.all_sum` (routed-expert width slice) | **YES**, 1× per layer |
| shared expert | `exl3_build.py:529,545-550` | *no* collective (in-width slice only) | n/a |
| lm_head | `exl3_build.py:349-351` / `:317-327` | `head` all_sum (vocab-sharded) | 1× per forward |
| indexer / compressor | `indexer.py`, `compressor.py` | none (grep for `all_sum`/`distributed` = empty) | n/a |
| engram | `engram.py` | none (host prefetch) | n/a |

Verbatim, attention:
> `attention.py:229-231` — `if self.group is not None:  # heads sharded: sum partials` /
> `with span("attn.all_sum"): out = _coll.all_sum(out, group=self.group)`

`self.group` is only set when `attn_tp` is true (`exl3_build.py:515-545`,
`a.group = group`).

Verbatim, MoE tail (two operand orders, one collective either way):
> `moe.py:180-189` — `if self.group is not None and self.shared_sharded:` … `y = y +
> self.shared_experts(xf)…` then `with span("moe.all_sum"): y = _all_sum_tail(y,
> self.group)`; `elif self.group is not None:` all_sum first, then shared_expert add.

`_all_sum_tail` payload dtype is env-gated, **default bf16** (`moe.py:51,56-63`):
> `_MOE_ALLSUM_BF16 = os.environ.get("DSV41_MOE_ALLSUM_BF16", "1") == "1"` … `return
> _coll.all_sum(y.astype(mx.bfloat16), group=group).astype(mx.float32)`

The body head is `ShardedHead` (`exl3_build.py:299`), one `all_sum` per forward:
> `exl3_build.py:349-351` — `w = y.shape[-1]; pad = […]; return _coll.all_sum(mx.pad(y,
> pad), group=self._group)` (docstring `:299-305`: "one all_sum rebuilds the full row …
> all_sum is used instead of all_gather because the JACCL mesh all_gather rejects this size").

Body forward head argmax is called **once** (`model.py:335`, gated by `argmax=True`):

```
model.py:335   logits = self.head.argmax(h.astype(mx.float32))   # token ids [b, n]
```

`ShardedHead.combine_argmax` (`exl3_build.py:317-327`) does exactly one `all_sum` over the
tiny `[world, rows, 2]` max/index buffer.

### 1.3 Draft (DSpark) head collectives — **per round, not per forward**

The draft runs **sharded only in its routed experts and markov head**; everything else is
replicated (`exl3_build.py:667-672` docstring: "Routed draft experts take the same
intermediate-width rank slice as the body (one all_sum per stage); everything else is
replicated, so the draft runs identically on every rank").

* `DraftMoE.__call__` — **one `all_sum` per stage**, ×3 stages:
  > `mtp.py:193-194` — `if self.group is not None:  # experts hold one rank's width slice`
  > / `y = _coll.all_sum(y, group=self.group)`
  > set at `exl3_build.py:681` — `stage.ffn.group = group`.
* `head.combine_argmax` — **one `all_sum` PER BLOCK POSITION**, i.e. `gamma` of them per
  draft round, because it is called inside the sequential `for k in range(bs)` loop:

  ```
  mtp.py:323   for k in range(bs):
  mtp.py:326-327   if sharded: nxt = head.combine_argmax(step_logits.astype(mx.float32))
  mtp.py:329       else:       nxt = mx.argmax(step_logits, axis=-1)
  ```
  `sharded = getattr(self, "vocab_sharded", False) and hasattr(head, "combine_argmax")`
  (`mtp.py:317`), and `head.vocab_sharded = True` is set at `exl3_build.py:718`.

> **CORRECTION to BRIEF-R1's starting arithmetic.** R1 stated "one all_sum in
> head.combine_argmax (vocab-sharded) = 4 cross-rank all_sums per draft". The source shows
> `combine_argmax` is invoked **once per drafted position** (the `for k in range(bs)` loop,
> `mtp.py:323-332`), so a draft of width `gamma` issues **`3 + gamma`** collectives, not 4:
> 3 (one per stage, `DraftMoE`) + `gamma` (one per position, `combine_argmax`). At width 3
> that is 6, not 4.

Note the draft does **not** issue the 40 body MoE all_sums: the draft block is 3 stages,
not 40 layers. The draft attn (`DraftAttention.draft_block`, `mtp.py:136-170`) and
`main_proj` (replicated, `exl3_build.py:710-711`) issue **no** collective.

### 1.4 Is `attn.all_sum` actually firing? (body)

The build's `DSV41_TP_ATTN` default is `"1"` (`exl3_build.py:355`) and production does **not**
forward it (`grep DSV41_TP_ATTN start_cluster.sh` = empty), so the builder's default applies.
That is a **different switch** from the legacy `EXO_DSV4_ATTN_ALLSUM=0` seen in live env
(both nodes) and in `PERFORMANCE_HISTORY.md:3054-3055` ("both attention-tail all_sums are
dead because production sets `EXO_DSV4_ATTN_ALLSUM:=0`"). **`EXO_DSV4_ATTN_ALLSUM` is read
only by the legacy `deepseek_v4.py`** (`mlx_lm/models/deepseek_v4.py:2330`) — the v41 model
does **not** read it (grep of `deepseek_v41/*.py` for that name = empty). So for the *served*
v41 engine the only relevant switch is `DSV41_TP_ATTN`, which is unset → sharded. **INFERRED:
`attn.all_sum` DOES fire for this engine, 40× per body forward**, despite the live
`EXO_DSV4_ATTN_ALLSUM=0`. This is a material uncertainty flagged for Phase 0 to settle with a
PROF=1 span dump on the actual build (see Q1.6); if a live v41 span dump shows no
`attn.all_sum` line, the count is 41, not 81.

### 1.5 Collectives per decode round and payload bytes (M = 4 rows)

One speculative round = **draft forward** (width γ) + **one body verify forward** on
`1 + gamma` rows (`rounds.py:425-427`, `spec.py:318`).

| source | collective | count / round | payload @ M=4 (hidden 5120) |
|---|---|---|---|
| body MoE tail ×40 layers | `moe.all_sum` | 40 | `4×5120×4 B` fp32 = 81,920 B, **bf16 (default) 40,960 B** |
| body head | `head` all_sum | 1 | `[world, rows, 2]` fp32 ≈ 32 B |
| body attn (if active, §1.4) | `attn.all_sum` | 40 | `4×5120×4 B` = 81,920 B |
| draft `DraftMoE` ×3 stages | `all_sum` | 3 | `4×5120×4 B` = 81,920 B |
| draft `combine_argmax` | `all_sum` | **γ** | `[world, 1, 2]` fp32 ≈ 16 B |

**Totals:** **44 + γ** collectives/round if `attn.all_sum` is off; **84 + γ** if on. At the
default γ = 3 that is **47** or **87** per round.

**M is `gamma + 1`, not 4.** The verify forward feeds `[anchor, draft×gamma]`
(`rounds.py:425-427`); at γ = 3 that is 4 rows, at γ = 5 it is 6. So "M=4" ≈ the γ=3 case.
Per-round body payload ≈ `40 × (γ+1) × 5120 × 2 B` (bf16) = **2.15 MB/round at γ=3**, 3.0 MB at
γ=5 (the head/all_sums are metadata-scale, negligible).

### 1.6 Stream and GPU→CPU→GPU drain

Collectives run on a **CPU stream** and each forces a host hand-off (this is the whole
reason `collective.py` exists):

> `collective.py:3-12` — "JACCL runs `all_sum` on a CPU stream, and any GPU work that
> consumes its result is encoded behind a wait on that event *inside a Metal command
> buffer*."
> `collective.py:14-17` — "`sync_collectives()` (or `DSV41_SYNC_COLLECTIVES=1`) evaluates
> every collective's result on the host before any GPU work is encoded behind it … Use it
> for warmup / the first forward; **steady-state decode leaves it off**."

Cross-confirmed in `PERFORMANCE_HISTORY.md` (Campaign-2 round 2/3):
> "(`mlx/distributed/jaccl/jaccl.cpp:88` — the collective stream is `new_stream(Device::cpu)`)"
> "jaccl is a HOST BOUNCE BY CONSTRUCTION — its own posix_memalign + ibv_reg_mr buffers with a
> CPU memcpy per collective (mesh_impl.h:789-801)."

So every `all_sum` is a genuine GPU→CPU→memcpy→RDMA→CPU→GPU round-trip. The steady-state
path does **not** add an explicit `mx.eval` per collective (that is the sync-collectives /
warm-guard path only); the drain is inherent to the CPU-stream collective consuming GPU
tensors.

### 1.7 Measured jaccl latency (from `docs/`)

* **Wire transport, jaccl-internal `steady_clock`, 45,666 real decode calls, 8 KB payload:**
  **median 36.1 µs (rank0) / 36.0 µs (rank1), mean 66.3 / 58.9 µs**
  (`docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md`; `PERFORMANCE_HISTORY.md:543-560`).
* **Isolated `allreduce_bench` floor at the 8 KB decode message size: ~120 µs** (`PERFORMANCE_HISTORY.md:500`).
* **Sync-span (per-op, artifact-inflated):** `moe.all_sum` 43,983 µs avg late-feed at deep
  prefill; 15–107 µs median at shallow decode; **~1400 µs/layer** "local drain" (round 1),
  `PERFORMANCE_HISTORY.md:8510+`.
* **Live `runner_log/stderr.log` on studio1 (this boot's predecessors):** `moe.all_sum`
  129 calls × median 107.25 µs; `attn.all_sum` (v4 build) 946 calls × median 14.4 µs.

> **INFERRED per-collective fixed cost.**
> (a) With `attn.all_sum` **on**: the dominant term is the drain — historically
> ~1400 µs per layer at 43 layers ≈ **60 ms per verify cycle** (`PERFORMANCE_HISTORY.md`:
> "~1400us is per VERIFY CYCLE, not per token — x43 = 60.2ms"). That brackets the whole
> verify, so it cannot be a per-token figure. Per round, drain ≈ 1400 µs × (M/16?) is not
> cleanly separable from compute; treat the collective *drain* as the 40-call serial
> bubble, not a fixed µs/call.
> (b) Wire-only floor: ~**36–120 µs** per all_sum (measured range). At ~45 collectives/round
> that is **~1.6–5.4 ms/round** of pure RDMA transport.
> The two are not additive; the drain dominates when `sync_collectives`/warm-guard is active
> (warmup + prefill chunks) and the wire floor dominates in steady decode.

---

## Q2 — DSpark head weights & replication feasibility

### 2.1 The head class and its checkpoint tensors

Class `DSparkHead` (`mtp.py:255-282`). Structure read off the index
(`mtp.py:5-12` docstring; verified live below):

```
mtp.{0,1,2}.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b,attn_sink}
mtp.{0,1,2}.ffn.{gate,shared_experts,experts.{0..127}}
mtp.{0,1,2}.{attn,ffn}_norm, hc_{attn,ffn}_{fn,base,scale}
mtp.0.main_proj, mtp.0.main_norm
mtp.2.norm, mtp.2.markov_head.{embed,head}, mtp.2.confidence_head.proj
```

Measured live from the checkpoint's own safetensors headers on `studio1` (all `mtp.*`
tensors live in shards `model-00038` + `model-00039`):

| group | count | bytes |
|---|---|---|
| `mtp.*` TOTAL | 4,836 tensors | **7,243,204,928 B = 7.243 GB** |
| stage 0 experts (128×) | 1,536 | 2,270.6 MB |
| stage 0 attn | 51 | 63.5 MB |
| stage 0 other | 27 | 62.4 MB |
| stage 1 experts | 1,536 | 2,270.6 MB |
| stage 1 attn | 51 | 63.5 MB |
| stage 1 other | 22 | 23.0 MB |
| stage 2 experts | 1,536 | 2,270.6 MB |
| stage 2 attn | 51 | 63.5 MB |
| stage 2 other | 26 | 155.4 MB |

dtype/quant: experts are EXL3 (`trellis` I16, `mul1`, `suh`/`svh`); the `mtp_bits = 4` field in
`config.json`, and the sample expert head `mtp.0.ffn.experts.0.w1.trellis [320,144,64] I16`.
The draft head's two big dense tensors are fp16:
`mtp.2.markov_head.head.weight [129280, 256] F16 = 66.191 MB`,
`mtp.2.markov_head.embed.weight [129280, 256] F16 = 66.191 MB`;
`mtp.0.main_proj.trellis [960,320,64] I16 = 39.322 MB`.

(For contrast, the **body** head `head.trellis [320,8080,96] I16 + suh + svh = 496.7 MB` — this
is the `ShardedHead` of Q1, not the draft head.)

### 2.2 How the head is sharded today

`build_mtp(ck, args, *, rank, world, group)` (`exl3_build.py:667-720`):

* **Routed draft experts — width-sliced across ranks** (`world>1` → each rank keeps 1/2 of
  the intermediate width):
  > `exl3_build.py:669-671` — `stage.ffn.experts = Exl3Experts(load_experts(ck, 0, prefix=pre
  > + "ffn.experts.", rank=rank, world=world))` and `stage.ffn.group = group`.
* **`markov_head.weight` — vocab-sharded** (last `exl3_build.py:713-718`):
  > `v = args.vocab_size // world; w = head.markov_head.weight[rank * v:(rank + 1) * v]` …
  > `head.vocab_sharded = True`
* **Everything else — replicated per rank**: draft attention (`wq_a/q_norm/wq_b/wkv/kv_norm/
  wo_a/wo_b/attn_sink`), `ffn.gate`, `ffn.shared_experts`, all `*_norm`, `hc_*`, and — at the
  top level — `main_proj`, `main_norm`, `markov_embed`, `confidence_proj`. The docstring says
  so explicitly (`exl3_build.py:671-672`: "everything else is replicated, so the draft runs
  identically on every rank").

### 2.3 Per-rank bytes today vs fully replicated (arithmetic)

* experts total = 2,270.6 × 3 = **6,811.8 MB**; per rank = **3,405.9 MB**.
* `markov_head.weight` total = 66.191 MB; per rank = **33.1 MB**.
* replicated remainder = 7,243.2 − 6,811.8 − 66.191 = **365.2 MB per rank** (all replicated;
  includes `markov_embed` 66.191 MB, the 3×63.5 MB attn blocks, projections, norms).
* **Per-rank resident today = 3,405.9 + 33.1 + 365.2 ≈ 3,804.2 MB ≈ 3.80 GB.**
* **Full head replicated on both ranks = 7,243.2 MB ≈ 7.24 GB per rank.**
* **Incremental memory to go fully replicated = 7,243.2 − 3,804.2 ≈ 3,439 MB ≈ 3.44 GB per rank**
  (the 3,405.9 MB of the peer's expert half + 33.1 MB of `markov_head`).

### 2.4 Falsifier outcome (memory)

Budget facts: runner footprint **~107 GB steady / 113.7 GB peak on a 128 GB node**;
`nodeMemory.ramAvailable` **~17 GB**; launcher wired guardrail **`DSV4_WIRED_LIMIT_MB` default 115000**
(≈ 112.3 GiB / 115 GB).

* Full head (7.24 GB) vs free RAM (17 GB): **fits** — the falsifier "head weights exceed free
  memory per node" does **not** trigger on raw free memory.
* But the **increment is added on top of the live footprint**: 113.7 GB peak + 3.44 GB =
  **117.1 GB > 115 GB wired limit** → **busts the guardrail by ~2 GB.**
* **INFERRED verdict:** replication is *memory-tight, not memory-impossible*. It needs one of
  (a) raising `DSV4_WIRED_LIMIT_MB` (risks the 128 GB node), (b) reclaiming ≥3.5 GB elsewhere
  (`EXO_JIT_MEMORY_RESERVE_GB` is 18.0 today, `EXO_DRAFT_KV_WINDOW=4096`), or (c) replicating
  only the **experts** (the 3,405.9 MB half) and leaving `markov_head` vocab-sharded — the
  markov-head all_sums then remain (still `gamma` per round), but the 3 per-round `DraftMoE`
  all_sums vanish. **Recommended Phase-3 experiment shape: replicate experts only, keep
  markov sharded; that is ~3.41 GB, still just over the wired headroom.**

### 2.5 Exact code seam for a "replicated head, no all_sum" path

All in `exl3_build.build_mtp`:

1. `exl3_build.py:669-671` — call `load_experts(ck, 0, prefix=pre+"ffn.experts.", rank=0,
   world=1)` instead of `rank=rank, world=world` (full 128-expert, full-width slice).
2. `exl3_build.py:681` — omit `stage.ffn.group = group` (so `DraftMoE.__call__`'s
   `if self.group is not None` all_sum, `mtp.py:193`, is skipped).
3. `exl3_build.py:713-718` — skip the `markov_head` vocab slice and do not set
   `head.vocab_sharded` (so `mtp.py:326-327` takes the `mx.argmax` branch, no all_sum).
4. Caller `load.build_draft_head` (`load.py:207-243`) passes `group=group`; it would need to
   pass `group=None` (or a new `replicate=True`) for the no-all_sum path. Nothing else.

**Draft-state cache is already per-rank** — `head.make_cache(1)` (`mtp.py:280-282`,
`rounds.py:398`) builds an independent `DraftWindow` ring on each rank; replication does not
touch it. `collective.warm_guard(("draft", width))` in `mtp.draft` (`mtp.py:292`) becomes a
no-op when `group is None`. **No cross-rank draft state exists to reconcile.**

---

## Q3 — Gamma policy facts (and a material correctness finding)

### 3.1 What is constructed

`GammaPolicy` (`mlx-lm spec.py:80-125`):

```
spec.py:87   def __init__(self, gammas=(1, 2, 3, 4), start=3, draft_ms=(8.5, 0.9),
                          overhead_ms=4.0, verify_ms=None, warmup=4):
spec.py:96   def update(self, gamma, n_acc): ... (accumulates self.tried[k]/self.acc[k])
spec.py:112  def next(self): ... (argmax over gammas of e/t; returns self.g)
spec.py:77   VERIFY_MS = {1: 58.5, 2: 74.9, 3: 87.9, 4: 97.7, 5: 111.7, 6: 120.1}
```

* **`start`** = the gamma returned for the first `warmup` (=4) rounds; set to the request's
  gamma: `_spec_policy(gamma)` → `GammaPolicy(start=gamma)` (`rounds.py:297-301`).
* **candidate set** `[src]` = `gammas=(1,2,3,4)` **only** (γ=5 is *not* a candidate by default);
  the `t` model is `dms[0] + dms[1]*g + v[g+1] + oh` = `8.5 + 0.9g + VERIFY_MS[g+1] + 4.0`
  (`spec.py:121`) — i.e. the cost table is `VERIFY_MS` **shifted by +1** (verify runs `g+1`
  rows). `start`/`gammas`/`warmup` are **NOT** env- or API-settable: `_spec_policy` hardcodes
  the rest of the defaults. This is a real design limitation for Phase 2 (no way to add 5 to
  the candidate set without a code change).

The engine fields (`engine.py:280-282`): `gamma: int = 3`, `adaptive_gamma: bool = True`.

### 3.2 Where `next()` / `update()` are (and are not) called — **`update()` is never called in the engine**

* `policy.next()` **is** called, once per round: `rounds.py:408` — `gamma = int(policy.next())
  if policy is not None else 1`.
* `policy.update(...)` is called **only** in the standalone `spec.generate()` harness
  (`spec.py:335`, `pol.update(g, n)`). Grep of the whole exo engine (`src/exo/worker/engines/
  mlx/dsv41/`) for `.update(` finds **only** `agreement.py` cancel-set updates — the
  `GammaPolicy.update` method is **dead code on the serving path**.

> **FINDING (material):** in the exo engine the adaptive policy never observes acceptance
> (`update` is never called), so `self.tried`/`self.acc` stay 0 and `_q(k)` returns the
> default 0.7 for every `k`. With all `q=0.7`, `next()` (after `warmup=4` rounds) picks
> `argmax_g (1+0.7+0.7²…)/t_g`, which for the default `VERIFY_MS`/`draft_ms`/`oh` computes to
> **g=3** (rates: g1 0.0193, g2 0.0214, g3 0.0224, g4 0.0217). So `adaptive_gamma=True`
> today is **functionally identical to fixed γ=3** — it is inert, not adaptive. (INFERRED: the
> arithmetic above; the code facts — `update` uncalled, defaults — are `[src]`.) This is why the
> `spec_gamma` wire field (commit `01c416b10`) is the only way to select an arm, and why a
> real Phase-2 adaptive policy needs an `update(...)` call wired into `_rounds` (see Q3.4).

### 3.3 Per-request `spec_gamma` (commit `01c416b10`, "default behavior unchanged")

Path: API clamp → task params → engine `_rounds` → policy start gamma.

* `api/types/api.py` (added by `01c416b10`): `spec_gamma: int | None = None` on
  `ChatCompletionRequest`, with `clamp_spec_gamma` → `max(1, min(6, v))`; non-int/bool → `None`.
* `shared/types/text_generation.py`: `spec_gamma: int | None = None` on `TextGenerationTaskParams`.
* `engine.py:803-805`: `self._rounds(session, anchor, max_tokens - 1, logprobs=lp_k,
  spec_gamma=params.spec_gamma)`.
* `engine.py:1004-1007`:
  ```
  gamma_for_request = spec_gamma if spec_gamma is not None else self.gamma
  policy = (_spec_policy(gamma_for_request)
            if (head is not None and self.adaptive_gamma) else None)
  ```
  The policy is built **once per request**, so `spec_gamma` sets the start gamma; per-round
  changes then come only from `policy.next()`.

### 3.4 What per-ROUND gamma switching (3↔5) would require

| component | site | requirement / finding |
|---|---|---|
| `_one_round` | `rounds.py:408` | already reads `gamma = policy.next()` **every round**; per-round switching is structurally already there |
| candidate set | `spec.py:87` | default `gammas=(1,2,3,4)` → **must add 5**; not settable today |
| `head.draft(width=γ)` | `rounds.py:414-420`, `mtp.py:290-302` | width is a plain param; `bs = width or block_size` — **no state sized by γ** |
| `head.append_ctx` | `rounds.py:460`, `mtp.py:284-288` | appends `accepted+1` rows — **independent of γ** |
| draft-state capacity | `mtp.py:280-282` (DraftWindow ring) | ring is `window_size` (see `EXO_DRAFT_KV_WINDOW=4096`); **width-independent** |
| `_ensure_capacity_for_round` | `rounds.py:304-331, 412` | grows the **body** context cache to `offset+1+γ`; a no-op unless the cache is near its cap |
| tap feed | `rounds.py:460` | feeds `tapcat(taps)[:, :accepted+1]` — γ-independent |

**No state is sized by the max gamma at request start.** γ is a per-round Python int; the
draft window and body cache are sized by `window_size`/`max_kv_tokens`, not γ.

### 3.5 Does width 5 vs 3 change the draft cost? Yes

`_draft` processes `bs = width` rows in one parallel forward (`mtp.py:302-306`,
`mx.concatenate([anchor, noise × (bs-1)])`) and loops `for k in range(bs)` for Markov sampling
(`mtp.py:323-332`), issuing **`bs` `combine_argmax` collectives** (Q1.3). So a γ=5 draft does
more block-attention work *and* 2 more collectives than γ=3. `[src]`.

### 3.6 Gamma-dependent compiled kernels and the non-monotonic matrix

Two `mx.compile`/shape warm guards key on the active width/rows:

* draft: `warm_guard(("draft", int(width or self.block_size)))` (`mtp.py:292`)
* body: `warm_guard(("body", rows, last_logit_only, return_taps, argmax, logprobs))`
  (`model.py:201-211`), and any forward with `>16` rows uses `sync_collectives()` (always
  host-sync) — prefill only, not decode.

So each new γ is a **new shape**: the first `DSV41_SYNC_WARM_CALLS` (=2) calls at that width
host-sync every collective (`collective.py:52-62`). A per-round 3↔5 switch therefore re-enters
the warm path (≈2 slow rounds) **every** time it flips — a real, named cost of adaptive
switching, on top of the steady-state path being sync-free.

**Non-monotonic matrix (g3 24.76 > g4 24.49 < g5 28.07 benign; VERIFY row marginals
R3→R4 9.8 / R4→R5 14.0 / R5→R6 8.4 ms):** the source supports a **shape-specialization**
explanation, not a kernel bug. `VERIFY_MS` (`spec.py:77`) is the t/g table with exactly those
marginals, and the prior note `phase19-latency/raw/phase3-kernel-sourceread.md` §3b pins the
M-step: `simd_mt` steps 4→**8** at `batch=5` (`gemv_metal.py:1422` `simd_mt = 1 if batch==1
else (2 if batch==2 else (4 if batch<=4 else 8))`), the trellis bytes re-read **increase**
when the group grows, and x is **padded** to `MT*groups` for M∈5..7 (`gemv_metal.py:1452-1459`).
The note's own conclusion: "M=5 is *more*, not less, expensive per group — the step cannot
create a *saving*." So R4→R5 (+14.0 ms) is a genuine discrete step, and the **throughput**
non-monotonicity (g4 < g3 < g5) is the interaction of that step with the higher
accepted-tokens yield at g5 — **INFERRED** from the two source facts; the source does *not*
claim g5 < g3, only that the M=5 step is real.

> **Consequence for PREREG D6:** the D6 up-switch signal (`p4 = P(acc≥4|acc≥3)`) is
> **unobservable while running at γ=3** (positions 4/5 are never drafted), exactly as D6 says
> — and the engine's own `update()` gap (Q3.2) means even the *observed* `_q(k)` are never fed
> back. Phase 2 must implement both the `update()` call and a γ=5 candidate.

---

## Q4 — Acceptance accounting

### 4.1 How the counters are filled

`_rounds` accumulates after each `_one_round` returns `(tokens, ms, accepted, gamma)`
(`rounds.py:334-346`; `engine.py:1026-1036`):

```
engine.py:1030-1036
  self._spec_rounds += 1
  self._spec_accepted += int(_accepted)
  self._spec_drafted += int(_gamma)
  self._spec_accept_hist[min(int(_accepted), len(self._spec_accept_hist) - 1)] += 1
```

Fields: `_spec_rounds` / `_spec_accepted` / `_spec_drafted` are `init=False` ints
(`engine.py:321-323`); `_spec_accept_hist` is `list[int]` of **length 7** (`engine.py:329-331`,
`default_factory=lambda: [0]*7`). `[src]`

### 4.2 Exposure as `mtp_accepted_histogram_cumulative`

The histogram is threaded out through `_final_response` (`rounds.py:198,223,236`) and lands in
`GenerationStats.mtp_accepted_histogram_cumulative` (`api/types/api.py`, added by `01c416b10`):

> `api/types/api.py` — "Per-position acceptance histogram — index ``k`` counts rounds
> (cumulative since the worker process started) that accepted exactly ``k`` drafts,
> ``k = 0..gamma``. Length is always ``>= 7`` so it covers the engine's max gamma (6)."

The API emits it inside the existing SSE comment frame:
`api/adapters/chat_completions.py:352,361,388` — `yield f": generation_stats
{stats_to_emit.model_dump_json()}\n\n"`. `[src]`

**Array length for γ up to 6:** fixed **7** (`0..6`); the write is index-clamped at
`len-1 = 6` (`engine.py:1034`). `[src]`

### 4.3 Is the per-round accepted count available inside `_rounds`?

**Yes.** `_one_round` returns `accepted` as its 3rd tuple element (`rounds.py:345,461-466`:
`return (committed, (time.perf_counter()-started)*1e3, accepted, gamma)`), and `_rounds`
already destructures it as `_accepted` (`engine.py:1016`). A per-round JSONL carrying
`(round_ms, accepted, gamma, position, ctx_offset)` is a **pure local addition inside the
`while` loop** (`engine.py:1013-1048`); `_round_ms` and `_gamma` are already in hand, and
`int(cache.offset)` (the round's start position) is available via `session.cache.cache.offset`.
No new plumbing, no extra collective. `[src]`

### 4.4 Reset semantics

Cumulative **per worker process, not per request**: the fields are `init=False` dataclass
fields on the single long-lived `Dsv41Engine` instance (`engine.py:321-331`), never zeroed in
`_generate`/`_end_turn`; the docstring says "cumulative since the worker process started"
(`api/types/api.py`). Deltas across successive requests give the live per-round
`p1..pk` survival curve. `[src]`

---

## Q5 — Rank plumbing & failure containment

### 5.1 `spec_gamma`: HTTP → engine on each rank

1. **API (master node, studio1):** `api/main.py:1088,1097` build `TextGeneration(task_params=…)`;
   the adapter already copied the clamped `spec_gamma` onto `TextGenerationTaskParams`
   (`chat_completions.py`, `01c416b10`).
2. **Master:** `master/main.py:261-276` — `params = command.task_params.model_copy(update={"prefill_endpoint":
   …})`, then a single `TaskCreated(task=TextGenerationTask(task_id, command_id,
   instance_id=decode_instance_id, task_status=Pending, task_params=params))`. `TextGeneration`
   carries `task_params: TextGenerationTaskParams` (`shared/types/tasks.py:60-62`).
3. **Node fan-out:** the node's `Worker` handles the task and calls `self._start_runner_task(task)`
   for the instance's runner(s) (`worker/main.py:478,623-627`), i.e. the **same immutable
   `task_params` object is broadcast to both TP ranks** (a `TensorShardMetadata` instance has
   one runner per rank; `/state` shows runner `fb0cbe9f…` = rank 0 and `ee56fffb…` = rank 1 in
   one instance `4d02870e…`). `TextGenerationTaskParams` is `frozen=True`
   (`shared/types/text_generation.py:176`).
4. **Runner → engine:** `runner.py:431,813` dispatch `TextGeneration()` to the generator;
   `Dsv41Engine._generate(params, …)` (`engine.py:697`) reads `params.spec_gamma`
   (`engine.py:803-805`).

### 5.2 Can a per-request field differ between ranks?

**No — by construction.** `spec_gamma` is a field **inside** the single frozen
`TextGenerationTaskParams` broadcast to both ranks; there is no per-rank parameter channel.
This is exactly what PREREG-D3/D6 need: anything that must not diverge (the Mode-2 extra
`mx.eval`s, the adaptive gamma) is safe *only* through this object. A separate env var or a
per-rank side channel **cannot** be per-request, because env is fixed at boot (D3) and no
per-rank request field exists.

### 5.3 Failure containment

* **Hang watchdog:** `EXO_RUNNER_HANG_TIMEOUT_SECONDS` default **45.0 s**
  (`runner/supervisor.py:83`); if a runner has in-progress work but emits no event for that
  window it is SIGKILLed, `is_alive()` flips false → `RunnerFailed` → re-place.
* **One rank raising inside a round:** the MLX forward is a lazy graph; the single sync is
  `mx.eval(logits)` at `rounds.py:435`. An exception on rank A before that leaves rank B
  blocked in `all_sum`; the 45 s supervisor watchdog is the backstop
  (documented as the real recovery path in `PERFORMANCE_HISTORY.md`: rank 1 "died seconds
  later on the peer EOF mid-JACCL collective"). `[src]`
* **Instrumentation must never kill a runner.** The model's own fence hook is the template:
  > `model.py:180-183, 261-273` — hook "must not take locks, touch model state, or call
  > `mx.*`"; a raise is caught, logged once, and the hook disabled (`self._fence_hook = None`).
  Any per-round JSONL writer / timer must be wrapped the same way — catch `Exception`, log
  once, disable — so a full disk or a bad path degrades to no instrumentation, never a
  `RunnerFailed`.

### 5.4 Which rank hosts API/master vs jaccl rank 0

**CONFIRMED live** (`GET /state`, read-only):
`instances.4d02870e…shardAssignments.runnerToShard` has two `TensorShardMetadata`s; runner
`fb0cbe9f-0f7c-43ca-b984-8e4b121d72f4` carries `"deviceRank": 0, "worldSize": 2`, runner
`ee56fffb-…` carries `"deviceRank": 1`. The API/master is `studio1` = m4-1 = 192.168.86.48.
The brief's live claim — "runnerToShard deviceRank 0 on runner fb0cbe9f = node 5114… =
studio2" — is consistent: **rank 0 = studio2 (m4-2, 192.168.86.47, jaccl coordinator), rank 1
= studio1 (m4-1, API/master)**. This matches PREREG-D7. `[src]`

---

## Q6 — Launcher allow-list (`start_cluster.sh`)

Mechanism (`start_cluster.sh:1879-2730`): the launcher accumulates a single
`EXO_ENV="… VAR=$VAR"` string, forwarding each variable only when it is set in the *launcher's*
environment:

```
[ -n "${VAR:-}" ] && EXO_ENV="$EXO_ENV VAR=$VAR"
```

### 6.1 Are the round-prof / adaptive-gamma vars forwarded today? — **NO**

```
$ grep -c "ROUND_PROF\|round_prof\|ADAPTIVE_GAMMA\|adaptive_gamma\|SPEC_GAMMA\|spec_gamma" start_cluster.sh
0
```

There is **no** forwarding line for `EXO_DSV41_ROUND_PROF`, `EXO_DSV41_ROUND_PROF_PATH`, or
`EXO_DSV41_ADAPTIVE_GAMMA`. (They also do not appear anywhere in `src/exo/` — they are
vars the next16-instr build is *expected* to introduce; nothing reads them yet in the
`deploy/next16-instr` tree I read.) `[src]`

### 6.2 Nearest existing `EXO_DSV41_*` forwarding lines (the template to copy)

```
start_cluster.sh:2728  [ -n "${EXO_DSV41_CHECKPOINT_SPACING_ROWS:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV41_CHECKPOINT_SPACING_ROWS=$EXO_DSV41_CHECKPOINT_SPACING_ROWS"
start_cluster.sh:2729  [ -n "${EXO_DSV41_CHECKPOINT_MARGIN_ROWS:-}" ]  && EXO_ENV="$EXO_ENV EXO_DSV41_CHECKPOINT_MARGIN_ROWS=$EXO_DSV41_CHECKPOINT_MARGIN_ROWS"
start_cluster.sh:2730  [ -n "${EXO_DSV41_CHECKPOINT_KEEP:-}" ]        && EXO_ENV="$EXO_ENV EXO_DSV41_CHECKPOINT_KEEP=$EXO_DSV41_CHECKPOINT_KEEP"
```

The bare `DSV41_*` (no `EXO_` prefix) forwards live in the `2666-2713` block, e.g.:
```
start_cluster.sh:2696  [ -n "${DSV41_MOE_ALLSUM_BF16:-}" ] && EXO_ENV="$EXO_ENV DSV41_MOE_ALLSUM_BF16=$DSV41_MOE_ALLSUM_BF16"
start_cluster.sh:2713  [ -n "${DSV41_ASYNC_EVAL:-}" ]      && EXO_ENV="$EXO_ENV DSV41_ASYNC_EVAL=$DSV41_ASYNC_EVAL"
```
(Prefix matters: the model/engine reads `DSV41_*`; exo reads `EXO_DSV41_*`. A new
`EXO_DSV41_ROUND_PROF` would need a line here **and** a reader under `src/exo/`.)

### 6.3 Other process env needed for a runner-written `/tmp` JSONL

The runners are launched by the launcher's ssh wrapper via the node venv python; the
per-rank trace-file precedent is `JACCL_TRACE_CALLS` writing
`/tmp/jaccl_trace_rank_${MLX_RANK}.log` **on each runner host** (`start_cluster.sh:2736-2745`
comment). Practical notes for a `/tmp` JSONL path:

* The runner runs as the same user (`adam.durham` on both nodes), so `/tmp` and `~/.exo/`
  are both writable; a per-rank filename must include the rank (`MLX_RANK`) or the two
  runners overwrite each other. There is **no** container/sandbox namespace to worry about.
* **Do not use `/private/tmp` on macOS for a file a long-lived process opens once** — the
  documented failure is unlink-while-open (`PERFORMANCE_HISTORY.md:543-560`: a `rm -f` of the
  trace path after startup silently orphaned 1.68 MB of writes with no error). If the writer
  opens once at engine construction, never delete the path mid-run.
* The launcher's allow-list is **the only** way these reach the runner (D3): env is fixed at
  boot, so a JSONL *path* that must vary per chunk has to be appended-to by the runner, not
  re-exported. `EXO_DSV41_ROUND_PROF_PATH` (if introduced) has to be forwarded here.

---

## Digest — the 10 findings that most change the Phase 1–3 plan

1. **Draft `combine_argmax` is per-position, not once.** BRIEF-R1's "4 all_sums per draft"
   is wrong: a width-γ draft issues **3 + γ** (3 `DraftMoE` stage all_sums + γ `combine_argmax`)
   = **6 at γ=3** (`mtp.py:193,326-327,323`). Phase-3 "remove draft syncs" must target both.
2. **Per round total collectives = 44 + γ** (body: 40 `moe.all_sum` + 1 head + draft) **if
   `attn.all_sum` is off**, or **84 + γ if on** (= 87 at γ=3). `attn.all_sum` activity is the
   one genuinely uncertain item (§1.4).
3. **Head replication is memory-tight, not impossible: +3.44 GB/rank** (experts half 3,406 MB
   + `markov_head` 33 MB); full head is 7.24 GB/rank vs 3.80 GB/rank today. That **busts the
   115 GB wired limit** given 113.7 GB peak → needs `DSV4_WIRED_LIMIT_MB` raised, ≥3.5 GB
   reclaimed, or **experts-only replication**.
4. **Draft-state cache is already per-rank** (`make_cache` builds an independent ring on each
   rank); replication needs no cross-rank draft-state reconciliation.
5. **`GammaPolicy.update()` is never called on the serving path** — `adaptive_gamma=True` is
   inert and today evaluates to γ=3 (`rounds.py:408` calls `next()`; only `spec.generate`
   calls `update()`). Phase 2 must add the `update()` call.
6. **The candidate set is hardcoded `(1,2,3,4)` and the VERIFY table starts at 5** — a γ=5 arm
   is neither a policy candidate nor a `VERIFY_MS` key; both need code changes for Phase 2/3.
7. **Per-round γ switching is zero-copy / zero-realloc**: γ is a plain Python int consumed by
   `head.draft(width=γ)`; no state (draft window, body cache) is sized by max γ. The only
   cost is re-entering the `warm_guard` slow path (~2 sync rounds) on every flip.
8. **`spec_gamma` is a field in the single frozen `TextGenerationTaskParams` broadcast to both
   ranks** — ranks cannot diverge on it. Any per-rank control field would deadlock the
   collectives; this is the only safe per-request channel.
9. **Steady decode runs the CPU-stream collectives unsynced**; only warmup (first 2 calls per
   shape) and >16-row forwards host-sync (`collective.py:14-17,52-62`; `model.py:210-211`).
   Per-round mode fields are therefore safe to add without changing the steady path.
10. **Instrumentation safety template = the model's fence hook** (`model.py:261-273`):
    catch `Exception`, log once, self-disable — so a JSONL/timer bug can never trip the 45 s
    supervisor hang-watchdog and SIGKILL a runner.

## NOT verified

* **`attn.all_sum` firing.** Read from source (`DSV41_TP_ATTN` default "1", unset at launch);
  NOT confirmed by a live v41 span dump. The live `EXO_DSV4_ATTN_ALLSUM=0` is read only by the
  legacy `deepseek_v4.py`, so it does not settle this. **Phase 0 must settle it** (the count
  is 44+γ vs 84+γ).
* **Per-collective fixed latency.** The µs/call figures are from *prior* builds'
  `docs/`/PERFORMANCE_HISTORY; no live PROF=1 dump on `f4bb14746` was taken (read-only mandate;
  PM owns live runs). 36–120 µs (wire) and ~1400 µs/layer (drain) are historical.
* **Head-replication memory fit.** The +3.44 GB arithmetic is against the *reported*
  107/113.7 GB footprint and 115000 MB guardrail; no live wired-limit headroom probe was run.
* **`EXO_DSV41_ROUND_PROF` / `_PATH` / `ADAPTIVE_GAMMA` semantics.** Confirmed absent from
  `start_cluster.sh` (grep=0) and absent from `src/exo/` in `deploy/next16-instr`; their
  intended behavior is the next16-instr build's own (not in the tree I read).
* **The g4<g3<g5 throughput non-monotonicity mechanism.** INFERRED from the M=5 `simd_mt`
  step + yield interaction; source confirms the step, not the throughput ranking.
* **Draft `DraftWindow` bytes.** `EXO_DRAFT_KV_WINDOW=4096` is live env but I did not confirm
  the code path that reads it in the pinned mlx-lm tree; the ring is `window_size` in
  `mtp.py:63-67`.


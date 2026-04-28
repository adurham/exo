# Fork Notes

Tracks the divergences between `adurham/mlx` / `adurham/mlx-lm` `main` and
their ml-explore upstream, plus the exo-side workarounds. Update this file
whenever the fork gains/drops a patch against upstream.

File map:
| Topic | Where |
|---|---|
| Dependency pins | `uv.lock` (`mlx` and `mlx-lm` have `git = "…?branch=main#<sha>"`) |
| Fork sources | `pyproject.toml` `[tool.uv.sources]` |
| KV cache config → runtime | [kv-cache-architecture.md](./kv-cache-architecture.md) |
| Upstream PR/issue tracker | [upstream-prs.md](./upstream-prs.md) |

## adurham/mlx-lm main

Pinned (as of exo `31ac5da4`, 2026-04-27): `main@2a1dcf653e3e95181c381d4d6329c2c0a92ce032`. `:main` is now the canonical fork branch — the rltakashige graft has been reverted, the cluster's Blaizzy-derived `deepseek_v4.py` is consolidated onto `:main`, and `:dsv4-perf-bisect` is retired (still exists on `origin` for history but unused).

Patches carried on `:main` against `ml-explore/mlx-lm:main`:

1. `_quantize` predicate fix — `5aa2d89` (filed upstream as PR #1216).
2. Cluster-working `deepseek_v4.py` (Blaizzy variant w/ DSv4-Flash sharding-friendly compressor + indexer) — `e98e75c`.
3. `EXO_PROFILER` spans in `V4Attention.__call__` + `DeepseekV4MoE.__call__` — `98ebfde`. 14 named spans; runtime no-ops without `EXO_PROFILER` set.
4. Sparse SDPA matmul rewrite — `87f4625` (see below).
5. Indexer fp32→bf16 — `f4dd9e7` (see below).
6. MoEGate fp32→bf16 + Compressor softmax cleanup — `2a1dcf6` (see below).

### `_quantize` predicate fix (commit `5aa2d89`)

`load_model._quantize`'s `class_predicate` returned the per-layer override dict before checking `to_quantized`, so models that pre-quantize specific layers in `__init__` (e.g. DSv4's `DeepseekV4MoE` calling `SwitchLinear.to_quantized(..., mode="mxfp4")` on its experts) trip with `ValueError: Unable to quantize model of type QuantizedSwitchLinear`. One-line reorder; works for any pre-quantizing model.

Filed upstream as **mlx-lm PR [#1216](https://github.com/ml-explore/mlx-lm/pull/1216)**. Drop the fork patch when that lands.

### DSv4 profiler spans (commit `98ebfde`)

`V4Attention.__call__` and `DeepseekV4MoE.__call__` are wrapped with 14 named spans via `mlx_lm.profiler` (the same hook the MiniMax instrumentation uses):

- V4Attention: `attn_q_lora`, `attn_kv_proj`, `attn_rope`, `attn_cache_update`, `attn_compressor`, `attn_indexer`, `attn_gather`, `attn_sdpa_sparse`, `attn_sdpa_dense`, `attn_o_proj`
- DeepseekV4MoE: `moe_gate`, `moe_switch_mlp`, `moe_shared`, `moe_all_sum`

Names use underscores (no dots) so each is a top-level entry in `SpanStats.dump()` — the % column denominator becomes the sum of all non-overlapping spans, giving correct attribution. Runner registers the `SpanProfilerHook` only when `EXO_PROFILER=spans` is set; with no hook registered, `_span()` returns the null context manager and `_finalize()` is a no-op, so production deploys add zero kernel-time overhead.

Fork-only — intrusive instrumentation, unlikely to be upstreamed in this form. `start_cluster.sh` already forwards `EXO_PROFILER` to runners.

### Sparse SDPA matmul rewrite (commit `87f4625`) — **4.2× wall-time speedup at 32K**

`_sparse_pooled_attention` had two `(broadcast)·(broadcast).sum()` patterns that materialized a `(B, H, L, k, D)` intermediate before reducing — at 32K-prefill shapes (H=64, L=512, k=512, D=128) that's ~4 GB of bf16 traffic per layer per chunk for *each* of the score path and the output path. Profiling on the cluster (2× M4 Max, `mlx-community/DeepSeek-V4-Flash-6bit`) showed `attn_sdpa_sparse` averaging 298 ms/call and consuming **73-78%** of total wall time at 32K context.

Both expressions are exactly a batched matmul over `L`:
- `pooled_scores[b,h,l,k] = sum_d q_scaled[b,h,l,d] * pooled[b,0,l,k,d]` → `(q.transpose(0,2,1,3) @ pooled[:,0].swapaxes(-1,-2)).transpose(0,2,1,3)`
- `out_pooled[b,h,l,d] = sum_k pooled_weights[b,h,l,k] * pooled[b,0,l,k,d]` → mirror form

Rewriting as transpose-then-matmul lets MLX's GEMM kernel tile through without ever materializing the 5-D tensor. Equivalence verified numerically (max abs diff ≈ 1e-6, FP-accumulation order noise).

Live bench, 32K context, `concurrent_bench --concurrency 1 --max-tokens 200`, no profiler:

| | Wall | Decode |
|---|---|---|
| Pre-opt (lock pinned to `98ebfde`) | 415.06s | 28.6 tok/s |
| Post-opt (lock pinned to `87f4625`) | **98.19s** | 28.8 tok/s |
| Δ | **−76% / 4.2×** | unchanged ✓ |

Decode rate identical (sparse path fires only at L>1 prefill; L=1 decode goes through `attn_sdpa_dense`), so the win is entirely on prefill. Generic improvement to upstream's `deepseek_v4.py` — strong candidate for an upstream PR to `ml-explore/mlx-lm`.

### Indexer fp32→bf16 (commit `f4dd9e7`) — **+3.4% decode at 100K, −10% wall at 100K**

`Indexer.__call__` had three explicit `.astype(mx.float32)` casts in the score path: `q.astype(fp32) @ pooled.swapaxes(-1, -2).astype(fp32)` and `weights = self.weights_proj(x).astype(fp32) * (n_heads**-0.5)`. The fp32 GEMM was 2× the bandwidth/FLOPs vs bf16 on a hot decode-time op whose cost scales with pooled-cache size. Live profile (`EXO_PROFILER=spans`, 32K bench) had `attn_indexer` at 6.6% of decode at 32K, extrapolating to ~19% at 100K.

Drop the casts. MLX's bf16 GEMM accumulates in fp32 internally so the matmul is preserved up to the final downcast; `mx.argpartition` top-k selection is robust to small score perturbations because top-k routing decisions are inherently fuzzy at the boundary.

Live bench, c=1, max-tokens=200, no profiler:

| | 32K decode | 32K wall | 100K decode | 100K wall |
|---|---|---|---|---|
| Pre (lock @ `87f4625`) | 29.0 tok/s | 97.93s | 26.4 tok/s | 480.81s |
| Post (lock @ `f4dd9e7`) | 29.3 tok/s | 94.55s | 27.3 tok/s | **432.79s** |
| Δ | +1.0% (noise) | −3.5% | **+3.4%** | **−10.0%** (saved 48s prefill) |

Quality smoke test: standard chat-completion prompts produce coherent technically-accurate output — no regression. Generic improvement to upstream's `deepseek_v4.py`; clean upstream candidate.

### MoEGate fp32→bf16 + Compressor softmax cleanup (commit `2a1dcf6`)

Same playbook applied to two more sites:

1. `MoEGate.__call__` had `flat.astype(fp32) @ self.weight.T.astype(fp32)` for the routing matmul. `moe_gate` was the **largest** decode contributor at 14.9% (32K). Drop both casts and rename the weight cache from `_weight_T_f32` → `_weight_T`. Logits become bf16; downstream `_expert_select` pipes through sigmoid/softmax(precise=True) which preserves precision for the top-k routing decision.
2. `Compressor.__call__` cast gate to fp32 before `mx.softmax(precise=True)`. The `precise=True` flag already promotes internally regardless of input dtype — the cast was vestigial.

Live bench, c=1, max-tokens=200, no profiler (deltas vs `f4dd9e7` baseline; cumulative deltas vs original `87f4625` baseline):

| | Pre (`f4dd9e7`) | Post (`2a1dcf6`) | Δ vs prior | Cumulative Δ |
|---|---|---|---|---|
| 32K decode | 29.3 tok/s | 29.6 tok/s | +1.0% (noise) | **+2.1%** vs 29.0 |
| 32K wall | 94.55s | 94.37s | unchanged | **−3.6%** vs 97.93s |
| 100K decode | 27.3 tok/s | **27.7 tok/s** | **+1.5%** | **+4.9%** vs 26.4 |
| 100K wall | 432.79s | 431.98s | unchanged | **−10.2%** vs 480.81s |

Quality clean. Generic improvement to upstream's `deepseek_v4.py`; clean upstream candidate.

**fp32 cast audit exhausted.** V4Attention's q_lora/kv_proj/o_proj are already bf16 via `nn.Linear`. Remaining fp32 sites in the file are HC code (weights stored fp32, math depends on it), tiny scalars, training-only paths, and RoPE position indices — none worth touching.

### Why we're on main, not a snapshot

Main carries upstream ml-explore's BatchGenerator rewrite (`PromptProcessingBatch`
+ `GenerationBatch`, `Response` moved out of `BatchGenerator`, `next()` returns
`(prompt_responses, generation_responses)`, `next_generated()` returns only
the generation list). Exo has been ported to this API — see the migration
commit chain `519024e4`, `2b1b4bf9`, `97bd2126`.

### What works

- `KVCache`, `QuantizedKVCache`, `RotatingKVCache`, `ArraysCache` all behave
  as before.
- `QuantizedKVCache.merge` (upstream added) lets 4-bit KV flow through
  `_merge_caches` without the stale
  `does not yet support batching with history` error.

### What was ported — **MTP speculative decoding** ✓

`MTPBatchGenerator` ported to the new mlx-lm API (commit `5abedbf`). Now
subclasses `BatchGenerator` correctly, uses `_generation_batch` /
`_unprocessed_sequences` / `SequenceStateMachine`, and `_next()` returns
`(prompt_resps, gen_resps)`. Deque token buffer drains one per call; GDN
cache rolled back for rejected drafts via `c.offset` / `c.rollback()`.

**QWEN35_ENABLED still 0** — flip to 1 in `start_cluster.sh` after confirming
the live smoke test shows "MTP speculative decoding enabled" in the log and no
regressions on Huihui.

### What needs a port — **opt_batch_gen fast-next patch**

`src/exo/worker/engines/mlx/patches/opt_batch_gen.py` monkey-patches
`BatchGenerator.next` with a hand-rolled fast path that overlaps GPU
forward, async-eval, and detokenisation. It uses `self.active_batch` / the
old `Response` class. The patch is already gated off when
`EXO_SPECULATIVE=1` (start_cluster.sh sets that), so on the live cluster
it's not loaded. If it's ever re-enabled, the internal references need the
same rename (`active_batch → _generation_batch`, Response → GenerationBatch.Response).

### DeepSeek-V4 prefill chunk cap

DSv4's `compress_ratio == 4` attention layers (`mlx_lm/models/deepseek_v4.py:1439-1448`)
flatten the per-query top-`k` indexer gather to a dense KV of length `L × k`,
then pass it to `scaled_dot_product_attention`. The score tensor is therefore
`(B, n_heads, L, L × k)` — cubic in `L` until `k` saturates at
`index_topk=512` (i.e. `L ≥ 2048`), then quadratic. Single allocation must fit
under Apple Silicon's per-buffer cap of ~80 GiB (`86,586,540,032` bytes on
M4 Max).

Hard ceiling on prefill chunk size — observed and projected:

| L (chunk) | k = min(512, L/4) | per-layer score buf (bf16) |
|---|---|---|
| 1024 | 256 | 34 GB |
| 1280 | 320 | 67 GB |
| 1500 | 375 | **108 GB** ← crashes |
| 1904 | 476 | **221 GB** ← observed 2026-04-26 (227 GB Metal rejection) |
| 2048 | 512 | 275 GB |

**Set `EXO_PREFILL_STEP_SIZE ≤ 1024` (we use 512) for DSv4.** The default 2048
or `start_cluster.sh`'s 4096 will OOM on any multi-turn chat where the prior
assistant turn pushes total prompt past ~1.2K tokens. Reported on Blaizzy
[#1192](https://github.com/ml-explore/mlx-lm/pull/1192#issuecomment-4323191525);
real fix is upstream (sparse query-grouped SDPA that never materializes the
dense `L × k` view).

## adurham/mlx main

Pinned (as of exo `65655ced`): `1cfcb5b6fb769d9cbe312860646f13fb788090b3`.

### What we're carrying against upstream

Two revert commits on top of `e64e280d` (the upstream merge):

1. `1a176363 Revert "Fix jaccl init bug (#3418)"`
2. `1cfcb5b6 Revert "Jaccl refactor (#3412)"`

### Why

Upstream's #3412 (jaccl refactor, author: Angelos Katharopoulos, April 14)
splits the jaccl implementation into a `lib/jaccl/` tree with new Config
objects and a ring/mesh dispatch. On our 2-rank RDMA setup it crashes at
`mx.distributed.init(backend="jaccl", strict=True)` with:

```
ValueError: vector
```

thrown from inside the nanobind layer (libc++'s `std::vector::at` default
`what()` on out-of-range). The symptom on rank 1 is a cascading
`RuntimeError: [jaccl] Send failed with errno=22` when the rank-0 coordinator
dies mid-handshake.

We haven't rooted the underlying bug yet. Reverting #3412 + its follow-up
#3418 restores the pre-refactor jaccl (single-file `jaccl.cpp` + top-level
`mesh.cpp`/`ring.cpp`/`utils.cpp`) which has been shipping on our cluster
for months. The reverts are purely mechanical — no conflicts — so pulling
future upstream main merges shouldn't re-apply them automatically.

### Work to re-enable the refactor

The pre-refactor jaccl worked fine on our 2-rank Thunderbolt RDMA, so
`#3412` is strictly a regression for this topology — not a pre-existing
bug. Upstream is unaware because the intersection of (exo users × jaccl
backend × 2 ranks × Apple-Silicon-over-Thunderbolt-Bridge × post-#3412
release) is effectively just this cluster. A proper fix + upstream PR
is the right long-term path rather than carrying reverts indefinitely.

**Phase 1 — minimal repro (no exo in the loop).** ~20-line Python
harness that calls `mx.distributed.init(backend="jaccl", strict=True)`
with the same `MLX_IBV_DEVICES` + `MLX_JACCL_COORDINATOR` env that
`start_cluster.sh` sets. Run against **upstream mlx main** (NOT our
reverted fork) on both Studios. Capture the rank-0 traceback and the
rank-1 `RuntimeError: [jaccl] Send failed with errno=22` cascade.
~30 minutes. Temporarily takes MiniMax down during the test window —
run in a maintenance window, not while the prediction-bot needs it.

**Phase 2 — root cause.** Three candidates in probability order:

1. *Config parsing* — devices parsed as `string` instead of `[string]`
   in the new 3D tensor. Quickest check: log the parsed `Config`
   object at the top of `jaccl::init` before `MeshGroup::MeshGroup`.
2. *Empty-diagonal short-circuit in `create_connections`* — at N=2
   the 3D mesh is degenerate; some `std::vector::at` call likely
   indexes a missing row. `grep -n '\.at(' lib/jaccl/` — probably
   only a handful of sites.
3. *Bind/listen ordering in `SideChannel::SideChannel`* — timing log
   on both ranks. Less likely (both ranks fail symmetrically, which
   suggests a config/shape bug rather than a race).

Variance here: config-parsing = ~1 hour, QP ordering = ~1 day.

**Phase 3 — patch + local verify.** Fork upstream mlx into a working
copy, apply the fix, `uv sync` against a local editable path
(`pyproject.toml` has the commented-out lines for this at `[tool.uv.
sources]` already). Run `start_cluster.sh` *without* the two revert
commits. MiniMax READY at 2-rank = fix confirmed. ~1 hour + rebuild
time.

**Phase 4 — upstream PR.** Ship against `ml-explore/mlx` with a
regression test (ideally a pure config-parsing unit test that doesn't
need RDMA hardware). Reference `#3412` + `#3418`. Once merged, drop
`1a176363` + `1cfcb5b6` from this fork and bump the pin.

> **Upstream tracking:** classified "won't upstream the revert" in
> [upstream-prs.md](./upstream-prs.md) — an issue + reproducer is the
> right artifact, not a revert PR.

## adurham/mlx main — DSB barrier in `Fence::update`

Pinned (as of exo `6ae331fe`): `1f6eb6bd` (carries the patch as part of the
broader pin).

### What we're carrying

In `mlx/backend/metal/fence.cpp::Fence::update` (CPU fast-path, ~6 lines):

```cpp
// ARM64 std::atomic seq_cst → STLR + DMB ISH (Inner Shareable, CPU-only).
// GPU/DMA sit outside that domain. Use DSB SY (Full System) to force the
// store to coherence visible to all observers.
f.cpu_value()->store(count, std::memory_order_seq_cst);
__dsb(0xF);
```

Plus the GPU-side `fence.metal` "nuclear" rewrite (outer-30 × inner-1M spin
with a system-scope `__metal_atomic_load_explicit` fallback) and a
`MAX_ACTIVE_TASKS` 10→5 throttle in `transforms.cpp`. All from
@rltakashige's `address-rdma-gpu-locks` branch (Feb 16-25, 2026).

**Read-side companion patch (added 2026-04-27, commit `d6ecdaa9`):**

```cpp
// Symmetric to Fence::update's seq_cst store + DSB SY: without a
// read-side full-system barrier the CPU can sit on a stale cache
// line that the GPU has already updated.
while (f.cpu_value()->load(std::memory_order_seq_cst) < count) {
  __dsb(0xF);
}
```

Mirrors the write side. Closes Known Gap #1 below. Added in response to
the 2026-04-27 partial-deadlock observation (see "Reproduction status").

### Why

Issue **[ml-explore/mlx#3142](https://github.com/ml-explore/mlx/issues/3142)** — `[BUG] GPU locking using METAL_FAST_SYNCH=1 and the JACCL backend`,
opened by @rltakashige Feb 18, 2026. Still **OPEN**.

Symptom (per #3142 and vskiwi's #3141 writeup): on multi-rank jaccl/RDMA
inference with `MLX_METAL_FAST_SYNCH=1`, the GPU's `fence_wait` Metal
kernel can spin on a stale cached value indefinitely after the CPU has
already written the updated counter. The CPU-side fast-fence store
compiles to `STLR + DMB ISH` (Inner Shareable scope — CPU cores only); the
GPU and DMA engines sit in the Full System domain and may not observe the
update. `DSB SY` after the store forces it to a point of coherence visible
to all observers.

### History — read this before changing anything in this area

- **Feb 16, 2026** — @rltakashige's first attempt landed on his
  `address-rdma-gpu-locks` branch (commit `dccc5415`, originally
  `__dsb(0xE)` / DSB ST).
- **Feb 17** — same branch switched to `__dsb(0xF)` / DSB SY ("Use DSB SY
  and not ST.") after empirical testing showed ST insufficient on Apple
  Silicon. Then explicit `std::atomic::store(seq_cst)` for clarity.
- **Feb 18-25** — same branch added the GPU-side spin-loop changes and
  the `MAX_ACTIVE_TASKS` throttle ("Sorry, going to use a nuclear
  solution for now." → "make nuclear solution even more nuclear 2").
- **Feb 18** — @vskiwi opened **PR [#3141](https://github.com/ml-explore/mlx/pull/3141)** with a polished writeup of the same
  fix (CPU-side DSB SY + GPU-side system-scope atomic load fallback),
  tested on 4× M3 Ultra 512 GB / GLM-5 754B tensor parallel.
- **Feb 19** — @awni merged **#3144** (cross-CB fence sync — adjacent
  but does *not* address coherence).
- **Feb 19** — vskiwi reported being unable to reproduce the deadlock on
  a clean cluster after #3144 landed. **PR #3141 closed Feb 24** by mutual
  agreement (@angeloskath, @vskiwi). Issue #3142 stayed open.
- **Mar 8** — vskiwi posted a [follow-up in #3142](https://github.com/ml-explore/mlx/issues/3142#issuecomment-4018576460) identifying a **read-side**
  coherence gap: `Fence::wait`'s spin loop also needs `__dsb(0xF)`. Their
  evidence: `sample` of a deadlocked process showed 271/271 samples stuck
  in the read loop. **Our fork does NOT carry this read-side fix.**
- **Apr 26** — opened our own PR [#3456](https://github.com/ml-explore/mlx/pull/3456) (just the write-side patch).
  Description rewritten same day with the full prior-art context above
  after @zcbenz pushed back ("you need to prove this is not AI
  hallucinations first") — that pushback makes sense given #3141's history.
- **Apr 28** — **PR #3456 closed without merge.** zcbenz consulted the
  Apple Metal team and confirmed there is no supported CPU↔GPU atomic
  memory coherence primitive in Metal and no public way to implement a
  guaranteed-working GPU spinlock. Their position: the DSB barrier "may
  appear to work but only triggers implementation quirks, not because
  Apple Silicon follows ARM specs" — unlikely to hold across all
  hardware/macOS versions. zcbenz offered to merge flag-gated as an
  explicitly documented hack; declined, to avoid landing unsupported
  behavior in upstream under our name. **The patch stays fork-only.**
  If the underlying hang ever needs an upstream fix, the right path is
  Apple shipping it directly (or a successor to #3141 from
  @rltakashige / @vskiwi with their larger-cluster repro).
- **Apr 27, evening** — **first 2-node-cluster repro.** DSv4-Flash-6bit
  on 2× M4 Max RDMA, multi-turn chat workload (~7K-token context,
  prefill chunks at 4096 env / 512 actual). Decode-step max times (PROF
  log lines, normally ~30ms steady-state) spiked to 5070, 6295, 9276,
  and **34097 ms**, paired across both ranks (5070/5018, 6295/6331,
  9276/9356, 34097/34097 — consistent with one rank's GPU stall blocking
  the other in cross-rank `all_sum`). Self-recovered every time rather
  than full-deadlocking — consistent with this being the read-side
  variant of the coherence gap, where the existing nuclear GPU spin
  eventually nudges things forward. No RDMA/jaccl error in either log;
  pure GPU stall, application-silent. Recovery: instance delete →
  supervisors clean up runners → `start_cluster.sh` restart. **Used to
  require a node reboot; the existing nuclear patches cap blast radius
  enough that we can recover without rebooting now.** Triggered the
  read-side fix (commit `d6ecdaa9` above).

### Reproduction status — A/B is hard

The bug reproduces under specific conditions (per #3141 / #3142):
- 4 nodes preferred over 2; repro frequency increases with rank count
- Large tensor-parallel models (GLM-5 754B observed)
- Per-layer `mx.eval` during exo's TP weight sharding (~78 rapid fence
  cycles immediately after JACCL init); `mlx_lm.sharded_load`'s bulk-eval
  pattern does *not* consistently trigger
- macOS 26.2 → silent data corruption; 26.3+ → SIGABRT or hang

**Our 2× M4 Max cluster was previously thought to sit below the threshold.**
Attempted A/B test 2026-04-26: built four binary-distinct mlx variants
(no-fix / DSB ST / DSB SY / DSB SY + nuclear), ran each on real
jaccl/RDMA over Thunderbolt with `MLX_METAL_FAST_SYNCH=1`, payloads
4 KB → 16 MB, sustained 60-110 s. **All four variants pass.** Synthetic
2-rank loads do not reliably reproduce the hang.

**That premise was invalidated 2026-04-27**: real DSv4 inference on
this same cluster produced the partial-deadlock pattern (paired
multi-second decode-step spikes, see History above). Not the synthetic
A/B harness — the actual MoE-heavy long-context multi-turn workload.

**Repro signal going forward: PROF `max=` decode-step time** in
`exo.worker.runner.llm_inference.runner:handle_generation_tasks` log
lines. Steady-state ~30ms at decode; anything >200ms is interesting,
>1000ms is a partial-deadlock event, paired with same-magnitude spike
on the other rank. No need to construct a synthetic load — DSv4
production traffic exercises it.

Caveat carried forward: the bug is non-deterministic even on configs
that hit it (rlt: "fairly difficult to reproduce"). A run that doesn't
spike doesn't prove a fix works; need bench windows long enough for
spikes to accumulate.

### Known gaps in our fork's fix

1. ~~**Read-side**: vskiwi's Mar 8 #3142 comment proposes adding~~
   ~~`__dsb(0xF)` inside `Fence::wait`'s spin loop.~~ **Closed
   2026-04-27 (commit `d6ecdaa9`)** in response to the partial-deadlock
   observation above. See "What we're carrying" for the patch.
2. **Hazard tracking**: vskiwi's #3142 Feb 18 comment also recommends
   `HazardTrackingModeUntracked → HazardTrackingModeDefault` in
   `allocator.cpp:15` to address a separate buffer-lifetime crash they
   saw at long contexts. Our fork doesn't carry this either; we haven't
   observed the crash. Don't add on speculation alone — different
   symptom from the fence hang.

### Coordination

If the upstream PR gets serious traction, **coordinate with @rltakashige
and @vskiwi** — they originated the work, have the larger-cluster repro
environment, and the right move is for one of them (not adurham) to be
the upstream author of any successor to #3141.

### Disposition — keep, earning its keep

Decision 2026-04-27 (revised, post-partial-deadlock observation):
**keep the patch chain on adurham/mlx:main; do NOT drop on next
rebase.** The 2026-04-28 "expendable, drop on next rebase" decision is
superseded.

Rationale:
- The "we can't reproduce on 2× M4 Max" premise is invalidated by the
  2026-04-27 partial-deadlock observation. DSv4 production traffic
  exercises the symptom.
- The fact that the existing nuclear patches turned what used to be
  full-deadlocks-requiring-reboot into self-recovering 5-34s spikes is
  positive evidence they're doing real work, even under the Apple Metal
  team's "implementation quirk" framing.
- Read-side fix added 2026-04-27 (`d6ecdaa9`) — needs cluster bench to
  validate against the PROF max-step regression signal.

Forward plan:
- **Keep all four pieces** (write-side DSB, GPU-side nuclear spin,
  `MAX_ACTIVE_TASKS=5`, read-side DSB).
- **Validate the read-side fix** by running DSv4 production-shaped
  workload long enough to catch the partial-deadlock signal, comparing
  PROF max-step distributions before/after.
- **Restore the "drop on rebase" disposition only if** the read-side
  fix demonstrably eliminates the spikes AND we can show the
  write-side+nuclear patches alone are sufficient — which we currently
  can't.
- **Don't reopen #3456.** Apple's position is settled; any upstream
  successor needs to come from rlt/vskiwi at their cluster scale, or
  from Apple themselves. The 2-node repro changes the evidence picture
  but not the upstream policy answer.
- **Hazard-tracking change (Known Gap #2) stays unimplemented** — it
  addresses a different symptom (buffer-lifetime crash at long
  contexts) we haven't observed.

> **Upstream tracking:** PR #3456 closed without merge 2026-04-28 (see
> [upstream-prs.md](./upstream-prs.md) "Recently closed without merge").
> Apple Metal team confirmed no supported CPU↔GPU coherence primitive;
> the patch chain stays fork-only.

## adurham/mlx main — quantized SDPA kernel + dispatch knob

Pinned (as of exo `6ae331fe`): `1f6eb6bd` (`sdpa: port unquant 2-pass
perf improvements to quant kernel`).

### Custom primitives carried on main

- `mx.fast.scaled_dot_product_attention_quant(q, k_packed, k_scales,
  k_biases, v_packed, v_scales, v_biases, *, scale, group_size, bits,
  do_causal, sinks)` — Metal kernel that fuses the quant dequantize
  into the 2-pass FlashDecoding inner loop. Decode-only gate
  (`q.shape(2) == 1`); prefill / unsupported configs fall through to
  `dequantize() + scaled_dot_product_attention()`. Used by MiniMax
  decode via `mlx_lm/models/base.py:117`.
- Kernel body lives in `mlx/backend/metal/kernels/sdpa_vector_quant.h`.
  Carries 4 optimizations ported from the unquant `sdpa_vector_2pass_1`
  path (Q hoist + pre-scale, V memory-read hoist, bits=5 half-pack
  uint64 load, contiguous-chunk access). Local M4 Max microbench
  −26–29 % vs unoptimized; **cluster M4 Ultra neutral** (see
  `docs/minimax-quantized-sdpa-design.md` + `memory/phase2_real_goal.md`).

### MLX_SDPA_BLOCKS env var

> **Upstream tracking:** opened as PR #3455 (see
> [upstream-prs.md](./upstream-prs.md)) — applies to upstream's existing
> 2-pass kernel without needing the broader Phase-2 quant work.


`scaled_dot_product_attention.cpp` carries an env-var override for the
2-pass blocks heuristic: `sdpa_2pass_blocks_override()` reads
`MLX_SDPA_BLOCKS`, and if set to a positive int, overrides both the
bf16 and quantized dispatches. No-op when unset.

**Empirical sweet-spot on the 2-rank M4 Ultra cluster for MiniMax-
M2.7 at 50K context: `MLX_SDPA_BLOCKS=88`** gives +6.5 % decode
(26.14 → 27.86 tok/s). Sharp cliff at `blocks=92`. Reason: matches
~320 concurrent simdgroup slots on M4 Ultra (4 kv_heads × 88 = 352
TGs ≈ 1.1 dispatch rounds). See `memory/minimax_sdpa_blocks88.md` for
the full sweep + cluster config context.

Forwarded through `start_cluster.sh` as of exo `6ae331fe`. Not baked
in as default because optimum is workload-specific.

## Open optimization projects

### DeepSeek V4 sharding + fused MoE (in upstream review)

Today's work on adurham/exo:main, filed as two stacked upstream PRs:

- **exo PR [#1996](https://github.com/exo-explore/exo/pull/1996)** — drops upstream's full-attention DSv4 sharding (`_shard_v4_attention_heads` + `_AllSumLinear`) for a MoE-only strategy. Replicates attention on every rank, shards only the MoE block. Fixes the saturated-softmax gibberish observed running upstream PR #1978 + `mlx-community/DeepSeek-V4-Flash-6bit`.
- **exo PR [#1999](https://github.com/exo-explore/exo/pull/1999)** — stacks on #1996; adds an opt-in `_FusedSwitchGLU` that fuses `gate_proj + up_proj` into a single `gather_qmm` dispatch. **+1.2% c=1 / +1.1% c=2** on `mlx-community/DeepSeek-V4-Flash-6bit` (2× M4 Max RDMA), gated behind `EXO_DSV4_FUSED_MOE=1`. Default-on for DSv4 in `start_cluster.sh`.

Bench baseline at 34.6 tok/s c=1; 75.2 tok/s aggregate at c=8 (sub-linear scaling, no errors, tail ratio 1.00). Long-context degradation `-17%` decode at 32K vs 50w was originally suspected to be the DSv4 compressor pool — **profiling 2026-04-28 refuted that** (compressor is only ~2% of decode). Real picture: at 32K the `_sparse_pooled_attention` prefill path was 73% of total wall time; the matmul rewrite (mlx-lm `87f4625`, see above) cut wall by 4.2×. Decode regression is now spread across many small ops; `attn_indexer` (5% of decode, scales with pooled cache size) is the next-most-likely context-scaling lever.

### MiniMax fused MoE kernels

`src/exo/worker/engines/mlx/patches/` only contains `qwen3_5/` and
`qwen3_5_moe/` fused batched-MoE kernels. `mlx-community/MiniMax-M2.7-4bit-mxfp4`
(256 experts, top-8, 62 layers) runs the vanilla MLX MoE path: unfused
router → Python-level top-k → per-expert kernel launches → gather → TP
all-reduce, repeated for every layer for every decode token.

Symptom: at ~40K context under TP=2, MiniMax decode collapses to ~16 tok/s
even though memory, KV cache, and attention all look healthy. The per-token
MoE dispatch + 62 cross-rank all-reduces per step are the ceiling — context
length makes it feel worse only because attention's absolute cost is also
growing in parallel.

What to build: a `patches/minimax/` module paralleling `patches/qwen3_5_moe/`
— batched MoE dispatch (256-expert scatter/gather), shared-expert fusion if
MiniMax has one, and whatever Q/K/V + RMSNorm hot block MiniMax attention
has. Re-tune the Qwen3.5 kernel shapes for MiniMax's `hidden_size=3072`,
`intermediate_size=1536`, `num_attention_heads=48`, `num_key_value_heads=8`.

Reference:
- `src/exo/worker/engines/mlx/patches/qwen3_5_moe/apply.py` — entry point
- `src/exo/worker/engines/mlx/patches/__init__.py::maybe_apply_patches` — dispatch on `model_type`
- `~/.claude/projects/-Users-adam-durham-repos-exo/memory/minimax_moe_decode_bottleneck.md` — the full diagnosis + numbers

### Jaccl refactor debug (see above)

## Updating these pins

```bash
# Bump mlx-lm alone
uv lock --upgrade-package mlx-lm

# Bump mlx alone
uv lock --upgrade-package mlx

# Both
uv lock --upgrade-package mlx --upgrade-package mlx-lm
```

After bumping, smoke-test with:

```bash
uv run pytest src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py tests/test_vision_cache.py
bash ./start_cluster.sh   # redeploys both Studios
python3 /tmp/stream_probe.py   # quick TTFT + streaming check (see docs/)
```

If MiniMax fails to come up, check the first `RunnerFailed` traceback in
`~/.exo/exo_log/exo.log` — most API-drift failures manifest there with a
clear `AttributeError`. Record the delta in this file when resolving.

# Upstream PR Inventory

Cross-repo tracker for what `adurham/{exo,mlx,mlx-lm}` carries on top of upstream
and what's been pushed forward to upstream review. Companion to
[fork-notes.md](./fork-notes.md), which tracks dependency pins, and to
[fork-vs-upstream-inventory.md](./fork-vs-upstream-inventory.md) — the complete
file-by-file audit of every change vs upstream (273 files, 100% of core source
mapped to a doc or rationale).

Last refresh: 2026-06-04 (fork lineage reconciliation + branch push sweep; see "Fork lineage & push state" below). Prior: 2026-05-27 (PM — after zcbenz reviewed that day's mlx batch).

---

## Fork lineage & push state (2026-06-04)

Triggered by the question "is any mlx/mlx-lm work upstreamable?" — audited divergence against the **canonical** source of truth, not Blaizzy's fork.

**Lineage (confirmed via GitHub API):**
- `adurham/mlx-lm` parent + source root = **`ml-explore/mlx-lm`** (the canonical repo). `Blaizzy/mlx-lm` is itself just a *sibling* fork of `ml-explore/mlx-lm`, not above us in the chain.
- Action taken: **dropped the redundant `blaizzy` remote** from the local `mlx-lm` clone. Remotes are now `origin` (adurham), `upstream` (ml-explore), `rlt` (Takashige). The `upstream` remote already pointed at `ml-explore/mlx-lm`.
- `adurham/mlx` parent = **`ml-explore/mlx`** (`upstream` remote present).

**Upstreamability verdict — nothing clean in mlx or mlx-lm:**
- **mlx-lm**: 241 commits ahead of `ml-explore/mlx-lm` main, 175 of them adurham's. *All* adurham work is DSv4-specific, and **`deepseek_v4.py` does not exist on `ml-explore/mlx-lm` (nor Blaizzy)** — canonical tops out at deepseek_v3/v32. No upstream target exists for the DSv4 stack, including the cache work that sounds generic (`PoolingCache.save_meta/restore_meta`, deferred-pool update, pool-freeze) — all coupled to the DSv4 sparse-pooled path. (The genuinely-generic DSv4 perf rewrites — `87f4625` matmul, `f4dd9e7`/`2a1dcf6` bf16 — remain "candidate, not yet filed", gated on a DSv4 PR landing upstream; see mlx-lm group table below.)
- **mlx**: 116 commits ahead of `ml-explore/mlx` main, but only 2 are adurham's and **both are merge commits** (zero original code). The 116 are the exo-team JACCL / RDMA-over-Thunderbolt collective subsystem (Exo Bot 93, rltakashige/Takashige 13, Thump604 10) — exo-cluster-specific, not ours to upstream.
- **The only thing with a real upstream home remains the thinking-parser fused-delimiter fix → exo#2149** (already filed). Everything else sits on top of either fork-only `deepseek_v4.py` or the exo-only JACCL layer.

**Branch push sweep (origin = adurham forks):**
- `mlx-lm`: pushed `merge-backup-1776362512`; repo now 0 commits unpushed across all local branches. Champion work (`f8b277f`) already on origin.
- `mlx`: pushed 4 previously-local-only branches → `feature/nuclear-gpu-fix`, `merge-backup-1776362512`, `pr-3293` (head_dim 192/256 fused SDPA), `pr-3307` (chunked SDPA). Champion JACCL HEAD (`db757dcb0`) already on origin as `fix/jaccl-ack-qp-top-level`.
- **Not pushed:** `fix-dsb-barrier` local `80c634ac5` is the *same* DSB-barrier change already on origin as `de6102230` (recommit, different SHA). Pushing would require a force-overwrite for zero content gain — left as-is.

---

## Status board

### Open PRs (11)

Sorted by most recent activity. Staleness column = days since last update.

| Repo | PR | Title | Stale | Activity |
|---|---|---|---|---|
| `exo-explore/exo` | [#2149](https://github.com/exo-explore/exo/pull/2149) | fix(runner): handle fused/embedded think delimiters in parse_thinking_models | **0d** (today) | Opened 2026-06-04. `parse_thinking_models` flipped `is_thinking` only on exact `accumulated == think_end`; mlx-lm's streaming detokenizer emits multi-token `last_segment` deltas so `</think>` arrives **fused** (`"done.</think>def x():"`) or **spanning** chunks — both miss the boundary and leak reasoning into `content`. Substring/boundary split added; clean fast-path + prefix-buffering preserved. +5 tests (31/31), pure-Python parser verified against upstream base (09f9ea31) in isolation. Generic to any thinking model. Cherry-pick of fork `08eac031`, clean on main, 2 files +193/-0. Companion fork-only work: MTP tie-break losslessness fix (`docs/mtp-tiebreak-losslessness-fix.md`). |
| `ml-explore/mlx` | [#3596](https://github.com/ml-explore/mlx/pull/3596) | perf(metal): coalesce sub-small_size_ allocations to single cache bucket | 8d | Opened 2026-05-27. **zcbenz reviewed within hours, asked "How did you profile?"** Replied 2026-05-27 with full methodology: `MLX_LOG_NEW_BUFFER_PATH` cache-miss histogram (~70 cache-miss allocs/step, dominated by size-4 fp32 scalars), vmmap IOAccelerator region count (before: 14K-224K climbing; after: ~2,400 flat), throughput A/B (32.1 t/s both, RSS rate 770→155 KB/tok). **Awaiting zcbenz follow-up.** 12-line patch; the strongest of today's batch. |
| `exo-explore/exo` | [#2121](https://github.com/exo-explore/exo/pull/2121) | fix(runner): only rank 0 emits ChunkGenerated under tensor-parallel execution | **0d** (today) | Opened 2026-05-27. **Regression** introduced by PR #2000 (engine abstraction) which removed the pre-existing `if device_rank == 0` guard. On any 2-rank TP topology, every accepted token is emitted twice. Reproducer: `"Repeat exactly: FALCON-MERCURY-7749"` returns `"FALCONFALCON-MERCURY-MERCURY-7749-7749"`. 1-line fix + comment. No engagement yet (exo-explore backlog). |
| `ml-explore/mlx-lm` | [#1216](https://github.com/ml-explore/mlx-lm/pull/1216) | fix(utils): skip already-quantized layers in load_model._quantize predicate | 12d | Opened 2026-04-27. **Third-party validation 2026-05-15**: `DevOpsBenjamin` ("Please accept this two line fix that made me lose 3 hours from MLX Studio"). No maintainer engagement yet despite community signal. |
| `exo-explore/exo` | [#1996](https://github.com/exo-explore/exo/pull/1996) | fix(deepseek_v4): drop full-attention sharding for MoE-only strategy | 30d | Opened 2026-04-27. **Contested.** rltakashige commented 2026-04-28: `mlx-community/DeepSeek-V4-Flash-6bit` is sanitized for Blaizzy's variant, not theirs; claims their implementation is "considerably better in performance and stability." Their only published bench (PR #1195) is 30.1 tok/s single-node, vs our 34.6 tok/s on 2-node M4 Max RDMA TP — claim doesn't survive contact. **Holding ground; no maintainer triage.** Position: PR is scoped to compatibility with the de facto public checkpoint. Revisit if any DSv4 PR merges upstream. |
| `exo-explore/exo` | [#1999](https://github.com/exo-explore/exo/pull/1999) | perf(deepseek_v4): fuse switch_mlp gate_proj + up_proj into single gather_qmm | 30d | Opened 2026-04-27. Stacks on #1996, +1.2% c=1 / +1.1% c=2 bench-validated. Blocked on #1996 resolution. |
| `exo-explore/exo` | [#1992](https://github.com/exo-explore/exo/pull/1992) | feat: peer-to-peer model distribution | 31d | Opened 2026-04-26. **Biggest PR in the queue** (+2023/-74 across 26 files). Zero engagement. |
| `ml-explore/mlx-lm` | [#1204](https://github.com/ml-explore/mlx-lm/pull/1204) | minimax: validate head_dim against checkpoint, drop unused shared_intermediate_size | 31d | Opened 2026-04-26. Zero engagement, CI not run yet. |
| `exo-explore/exo` | [#1985](https://github.com/exo-explore/exo/pull/1985) | feat: Prometheus `/metrics` endpoint | 31d | Opened 2026-04-26. Zero engagement. |
| `exo-explore/exo` | [#1988](https://github.com/exo-explore/exo/pull/1988) | feat: `EXO_KV_CACHE_BITS` env var + step=16384 for QuantizedKVCache | 31d | Opened 2026-04-26. Zero engagement. |
| `exo-explore/exo` | [#1990](https://github.com/exo-explore/exo/pull/1990) | fix: skip KV cache quantization in single-node BatchGenerator mode | 31d | Opened 2026-04-26. Zero engagement. |

**Pattern note (refined 2026-05-27).** Two very different review cultures:
- **mlx-explore** (zcbenz, angeloskath): fast + critical. All 3 mlx PRs opened today got a zcbenz review **within hours**. He closes weak motivations immediately (#3594), challenges unproven mechanisms (#3595), and asks for hard evidence on perf claims (#3596). Submit to mlx-explore *only* with airtight repro/data — he will catch hand-waving. Net: mlx PRs move fast in *both* directions. **This is where to spend effort.**
- **exo-explore**: NOT a dead queue — merged 10 PRs in the 2 weeks before 2026-05-27 (#2087–#2114). But every recent merge is by a **core/frequent contributor** (AndreiCravtov, Evanev7, Heidar-An). Zero external-fork PRs merged in that window. The accurate read is **"merges insider work fast, ignores external PRs"** — not "backlogged." Consequence: waiting does nothing for our 6 untouched exo PRs (#1985/#1988/#1990/#1992/#1204/#2121); they need a maintainer to *actively adopt* them. The one reaction we got (#1996) was from a third-party contributor disputing, not a maintainer triaging. **Low ROI to chase exo-explore** unless a maintainer signals interest first; keep these as fork-context.
- **mlx-lm**: went quiet ~3 weeks before 2026-05-27 (big merge cluster 36–38 days ago by angeloskath + many external contributors, then slowed). 30 open PRs. Not hostile to external work — historically merges lots of it — just paused. mlx-lm#1216 has a community +1 which improves its odds when review resumes; #1204 is a quiet wait.

### Recently merged (3)

| Repo | PR | Title | Merged |
|---|---|---|---|
| `ml-explore/mlx` | [#3455](https://github.com/ml-explore/mlx/pull/3455) | Add `MLX_SDPA_BLOCKS` env var for 2-pass vector kernel block-count override | **2026-05-11** by angeloskath (was approved by zcbenz 2026-04-27, sat ~2 weeks before merge) |
| `exo-explore/exo` | [#1989](https://github.com/exo-explore/exo/pull/1989) | fix: route by in-flight tasks only — completed tasks were skewing load balance | (pre-2026-04-28) |
| `exo-explore/exo` | [#1991](https://github.com/exo-explore/exo/pull/1991) | fix: map presence_penalty and frequency_penalty from ChatCompletionRequest | (pre-2026-04-28) |

### Recently closed without merge (4)

| Repo | PR | Title | Resolution |
|---|---|---|---|
| `ml-explore/mlx` | [#3594](https://github.com/ml-explore/mlx/pull/3594) | feat: enable logsumexp output in fused SDPA dispatch | Closed 2026-05-27 by zcbenz, same day opened. "Outputting logsumexp is mostly used by backward in training, so the motivation does not make sense for me. For VJP there is **#3241** which includes the logsumexp output implementation." Apple's [#3241](https://github.com/ml-explore/mlx/pull/3241) (Fused Flash Attention backward/VJP kernels, open since 2026-03-11) already lands LSE output as part of the backward pass. Our forward-chunking motivation didn't stand on its own. **Acknowledged**; the held chunked-SDPA PR (`adurham:pr/sdpa-chunked-dispatch`) should be rebuilt to consume #3241's LSE output once that merges, rather than adding a separate LSE path. |
| `ml-explore/mlx` | [#3595](https://github.com/ml-explore/mlx/pull/3595) | perf(scheduler): MLX_STREAM_QOS env var to pin StreamThread QoS class | Closed 2026-05-27 by adurham, same day opened. zcbenz: "The thread doing busy 17M+ iterations means it is not being descheduled?" — correct. The 17M poll-iters prove the thread *was* scheduled, so "descheduled under QoS pressure" was the wrong diagnosis. The QoS theory was carried over from an unrelated cross-rank bistability debug session and never independently confirmed. `MLX_STREAM_QOS` "helped" on our cluster but we can't prove the mechanism, so shipping it upstream on an unverified theory is wrong. **Self-closed.** Reopen only with a clean Instruments/os_signpost trace showing the worker actually losing its core to a competing thread. Lesson: don't upstream a mitigation whose mechanism we can't demonstrate. |
| `ml-explore/mlx` | [#3456](https://github.com/ml-explore/mlx/pull/3456) | fix: address GPU locks by synchronizing GPU and CPU memory with DSB barrier | Closed 2026-04-28 by zcbenz. Apple Metal team confirmed there is no supported CPU↔GPU atomic memory coherence primitive and no public way to implement a guaranteed-working GPU spinlock; the DSB barrier "may appear to work but only triggers implementation quirks, not because Apple Silicon follows ARM specs." zcbenz offered to merge it flag-gated as a documented hack — declined to avoid landing unsupported behavior in upstream. **Patch stays fork-only** (see fork-notes.md). If the hang ever needs an upstream fix, the right path is Apple shipping it directly. |
| `ml-explore/mlx` | [#3454](https://github.com/ml-explore/mlx/pull/3454) | Add `mx.metal.dispatch_count()` for kernel-dispatch diagnostics | Closed 2026-04-28 by adurham. zcbenz pointed out that `mx.export_to_dot` already covers the fusion-validation use case: counting primitive nodes in the exported DOT graph distinguishes fused (1 `ScaledDotProductAttention` node) from unfused (5-node `Transpose → Matmul → Multiply → Softmax → Matmul`) at the same granularity needed for our Phase-3 fused-QKV / joined-RoPE validation. Patch is **not** carried on `adurham/mlx:main` either — Phase-3 is shipped, no current need. Branch's git history preserves the patch if a kernel-internal fusion case ever needs runtime dispatch counting. |

### Open issues / design questions (3)

| Repo | Issue | Topic |
|---|---|---|
| `ml-explore/mlx-lm` | [#1203](https://github.com/ml-explore/mlx-lm/issues/1203) | Skip `lm_head` on non-final PP ranks? — design ack on return-type asymmetry |
| `exo-explore/exo` | [#1986](https://github.com/exo-explore/exo/issues/1986) | Radix-trie KV prefix cache — design ack + PR shape preference |
| `exo-explore/exo` | [#1987](https://github.com/exo-explore/exo/issues/1987) | Sampling defaults — would you accept per-instance + cluster-env tiers on top of #1947? |

### Comments left (1)

- `ml-explore/mlx-lm` [#941](https://github.com/ml-explore/mlx-lm/pull/941) — +1 on `QuantizedKVCache.merge()`; offered to drive the `BatchQuantizedKVCache` redesign maintainer asked for.

### Held / drafted but not yet opened (2)

Branches built + tested clean on M4 Max. Description drafts at [`./upstream-pr-drafts/`](./upstream-pr-drafts/). Held pending the outcome of related open PRs or further investigation of known limitations.

| Repo | Branch | Title | Reason held |
|---|---|---|---|
| `ml-explore/mlx` | `adurham:pr/sdpa-chunked-dispatch` | feat: chunked SDPA dispatch for long key sequences (>65K) | **Prerequisite #3594 was CLOSED by zcbenz** (LSE output rejected as standalone — Apple's #3241 covers it via the backward path). So this branch as-built (which stacks its own LSE-output path) is **no longer the right shape**. Rework needed: once Apple's [#3241](https://github.com/ml-explore/mlx/pull/3241) merges, rebuild chunked-dispatch to consume *its* LSE output instead of carrying our own. Until #3241 lands, **do not open** — there's no LSE-output path on upstream main to depend on. Branch kept for the chunk/merge logic + tests, but the LSE plumbing must be re-pointed. Was a revival of @Thump604's #3307. |
| `ml-explore/mlx` | `adurham:pr/sdpa-head-dim-192-256` | feat: head_dim=192 + 256 fused SDPA support (with kL-aware routing) | **bf16/fp16 paths verified clean**, but float32 + bq=32 + bd=192 hits the same 32KB threadgroup memory limit as the existing `head_dim=256 + float32` skip in `test_sdpa_chunked.py`. Need to decide whether to: (a) submit as-is and document the float32 limitation in the PR body (acceptable, matches the existing precedent), or (b) parameterize bq per-dtype to fit float32 first (cleaner story, more work). Also overlaps with @Thump604's [#3293](https://github.com/ml-explore/mlx/pull/3293) ("fix: add head_dim=256 to fused SDPA full attention kernel") — closed unmerged 2026-04-04 with the same vllm-mlx rationale. If we open we should credit them like we did on #3594. |

### Fork sync state (2026-05-29)

All three forks now build the cluster from their `main` branch. exo's
`pyproject.toml` pins `adurham/mlx@main` and `adurham/mlx-lm@main` directly
(no more WIP branch names) — see "Cluster pins track main" below.

| Fork | vs upstream/main | Notes |
|---|---|---|
| `adurham/mlx-lm` | 0 behind, 243 ahead | Fully current; cluster pins `main` (= former eagle-soft-emb perf work + upstream sync incl. Gemma 4 fix #1240) |
| `adurham/mlx` | 0 behind, 132 ahead | Fully current; cluster pins `main` (ported lib/jaccl layout; main == former wip/try-ack-qp-isolated-ported, same SHA) |
| `adurham/exo` | 0 behind, 794 ahead | Fully current after 2026-05-29 sync of upstream `051a64e3` (per-phase energy telemetry) + pin-track-main flip |

#### Cluster pins track main (2026-05-29)

Flipped exo's `[tool.uv.sources]` pins from the WIP branch names to `main`:
- `mlx`: `wip/try-ack-qp-isolated-ported` → `main` (same SHA `f17b9fd4`, pure relabel — zero rebuild risk)
- `mlx-lm`: `eagle-soft-emb` → `main` (SHA `1028276a` → `2ec0de96`, adds only the DSv4-irrelevant Gemma 4 KV-projection fix #1240)

Rationale: fork mains already reflected what the cluster ran, so future
experimentation flows in via `git push adurham main` instead of maintaining
divergent WIP branch names. Validated end-to-end before merge to exo main
(commit `723f19a8`): cluster rebuilt clean on both Studios, c=1 100K MTP-off
bench = 27.3 t/s median (σ≈0.13, 0 errors, tail_ratio 1.00), quality probe =
3/3 needles found, 0 special-token leaks, 0 bistability at 100K context. The
27.3 vs ~29.3 historical is pre-existing c=1 drift, not the pin flip (mlx is
byte-identical; mlx-lm delta cannot touch DSv4). Rollback tags:
`pre-main-pin-exo-2026-05-29`, `pre-main-pin-mlx-lm-2026-05-29`.

#### Completed: exo upstream sync (24 commits, 2026-05-26 → 2026-05-27)

Synced cleanly via merge-resolve + 8 follow-up fix commits. The 4-commit deferral from 2026-04-28 above (P/D #1993, engine abstraction #2000) had already been absorbed into the cluster's working branch by the time the sync was attempted, so the structural conflict was much smaller than predicted. The actual conflicts were:

- `pyproject.toml` — upstream restructured base deps into `[project.optional-dependencies] mlx` extra. Took upstream's structure wholesale, re-layered our deltas (mlx/mlx-lm sources, prometheus-client).
- `src/exo/master/main.py` — upstream renamed `_send_event → _send_indexed_event` + inlined what we'd extracted into our `_index_apply_broadcast` helper. Kept our helper, applied the rename to inner call sites.
- `src/exo/worker/runner/bootstrap.py` — additive merge (kept QoS pinning + mlx-lm profiler hooks, added upstream's `RunnerTerminationError` dataclass).
- `src/exo/worker/runner/llm_inference/batch_generator.py` — adopted upstream PR #2048 fix (`extend(agreed)` instead of our filter-via-_maybe_queue) while keeping our coord-subgroup fast-path optimization.
- `uv.lock` — regenerated from merged pyproject.

8 follow-up fixes needed post-merge:
1. Restore `RunnerTerminationError` class (lost in conflict resolution)
2. Use `card_cache.get` instead of removed `get_card` function
3. Drop our redundant `RunnerStatusUpdated` send (supervisor now uses `RunnerTerminationError` channel)
4. `uv sync --extra mlx --all-packages` in start_cluster.sh (post-restructure)
5. EXO_TARGET_BRANCH env var in start_cluster.sh (cherry-pick from port-test)
6. `exo_tools.client` import in bench scripts (post-restructure)
7. agree_on_tasks correctly applying #2048 fix
8. **send_chunk rank-0 guard** (was the c=1 double-emit blocker — submitted as PR [#2121](https://github.com/exo-explore/exo/pull/2121))

Champion config validated post-sync: 30.7 t/s c=1 100K MTP+γ=2 (= prior baseline within noise), 3/3 quality probe needles, 0 errors. Full procedure documented in skill `exo/exo-upstream-sync` for next time.

---

## By repo

### exo (`exo-explore/exo`)

324 commits ahead of upstream/main, 23 behind. 12 logical change-groups.

| Group | Status | Notes |
|---|---|---|
| Prometheus `/metrics` endpoint | **PR #1985** | Single commit cleanup; dropped Grafana JSON, reframed docs as Prometheus-generic. 6/6 tests pass. |
| Load Balancer In-Flight Task Fix | **PR #1989** | Fixes routing by only counting in-flight tasks (Pending/Running) instead of all historical tasks. |
| Quantized KV cache (`kv_cache_bits`) | **PR #1988** | Env-var override + step=16384 frag fix. Most of the infra was already upstream — quotes the maintainer's TODO comment in description. 12/12 prefix-cache tests pass. |
| BatchGenerator KV Cache Crash Fix | **PR #1990** | Fixes single-node crash when `EXO_KV_CACHE_BITS` is set by skipping quantization in single-node mode. Fixes upstream issue #1875. |
| Radix-trie KV prefix cache | **Issue #1986** | 1200+ LoC; depends on two prereq features (caps, `pin()`) that aren't upstream either. Asked maintainer for PR-shape preference. PR #1929 was self-closed for paperwork (wrong base fork). |
| Instance sampling resolver | **Issue #1987** | Partially superseded by upstream #1947 (card tier merged 2026-04-21). Asked about per-instance + cluster-env tiers. |
| Presence/Frequency API mapping | **PR #1991** | Wires up the presence_penalty and frequency_penalty from the API layer to the upstream sampling resolver. |
| DeepSeek-V4 cluster integration | wait on whichever DSv4 PR lands | Three competing mlx-lm DSv4 PRs as of 2026-04-26, all OPEN/BLOCKED awaiting review: Blaizzy [#1192](https://github.com/ml-explore/mlx-lm/pull/1192) (most recently active), rlt [#1195](https://github.com/ml-explore/mlx-lm/pull/1195), machiabeli [#1189](https://github.com/ml-explore/mlx-lm/pull/1189). Fork's tokenizer fallback should drop once one lands. Sharding strategy is generic. **Cluster currently runs Blaizzy variant via mlx-lm pin to `adurham/mlx-lm:main`** (which carries Blaizzy's `deepseek_v4.py` + `_quantize` predicate fix on top); 34.6 tok/s c=1 on 2× M4 Max RDMA. **Don't pivot to rlt's variant unless their PR merges upstream** — rlt's only published bench (PR #1195) is 30.1 tok/s single-node which doesn't beat ours, and switching would cost a 155 GB redownload + quality revalidation + rebench from scratch. |
| MTP speculative decoding | **complete, pending smoke test** | `MTPBatchGenerator` ported to new mlx-lm API (`PromptProcessingBatch`/`GenerationBatch`/`SequenceStateMachine`). Deque token buffer, correct GDN rollback, per-token state machine. Flip `QWEN35_ENABLED=0 → 1` in `start_cluster.sh` after live smoke test confirms no regressions. |
| P2P model distribution | **PR #1992** | Reauthored from the original 16-commit chain (3 redundant after upstream PR #1829 landed; 5 dropped as cancel/no-op pairs). 2 clean commits: P2P core + cancel→pause/`DownloadPaused` refactor. Adds X-File-SHA256 header + receiver-side hash verification, EXO_FILE_SERVER_MAX_CONCURRENCY cap with 503/Retry-After, real path-traversal defense (the original fork's `is_relative_to(model_dir)` check let `..` escape laterally into a sibling model — caught and fixed). 61 new tests; security-relevant cases use raw sockets to bypass aiohttp client URL normalization. `docs/p2p-model-distribution.md` covers the security stance. |
| Phase-3 fused QKV / joined RoPE | depends on mlx#3454 + Phase-2 | Python-level kernel fusion. Off by default (`EXO_MINIMAX_FUSED_ATTN`). Wants `dispatch_count()` to land first. |
| Fused MoE dispatches | not started | Qwen3.5-specific kernel optimizations. ~6% prefill win. Pattern is generic. |
| Pipeline-parallel speculation | not started | Long chain, generic over draft model. Off by default (`EXO_PP_DRAFT_MODEL`). |
| TurboQuant KV cache | not started | Shipped disabled-by-default after overhead analysis. Could be upstreamed as optional KV strategy. |
| Cluster glue (`start_cluster.sh`, model cards, etc.) | won't upstream | adurham-cluster-specific config. |

### mlx (`ml-explore/mlx`)

38 ahead, 1 behind. 6 change-groups (1 active PR, 2 closed-without-merge fork-evaluated).

| Group | Status | Notes |
|---|---|---|
| `mx.metal.dispatch_count()` | **PR #3454 closed 2026-04-28** | zcbenz pointed at `mx.export_to_dot` as covering the use case (graph node count distinguishes fused vs unfused at the same granularity we used the runtime counter for). Not carried on `adurham/mlx:main` either — Phase-3 is shipped, no current need. Closed branch's git history preserves the patch if a kernel-internal fusion case (where the graph stays the same but the kernel does more work) ever needs runtime dispatch counting. |
| Chunked SDPA + LogSumExp | needs Thump604 coordination | Thump604's work; their PRs #3293 (head_dim=256) and #3307 (chunked) **both closed without merge** 2026-04-04 after zcbenz asked for perf-regression data and a separate issue. Author hasn't returned. |
| `MLX_SDPA_FUSED_THRESHOLD` env var | folded into Thump604 work | The gate this parameterizes (`sdpa_full_large_hd_ok`, head_dim 192/256, `key_sequence_length > 16384`) doesn't exist upstream — it would have to land via re-revival of #3293/#3307. |
| `MLX_SDPA_BLOCKS` env var | **PR #3455** | Bundled into Phase-2 commit `1f6eb6bd` but conceptually independent. Tunes the 2-pass blocks heuristic for both bf16 and quant dispatches. Empirical sweet-spot 88 at 50K/M4 Ultra/MiniMax (+6.5%). Could land standalone since the override applies to upstream's existing 2-pass kernel. ~10 lines. **Round-1 review (zcbenz, 2026-04-26)**: dropped the test, switched to `env::get_var`, renamed away from `override` keyword. **Round-2 (zcbenz, 2026-04-27 UTC)**: helper function dropped — env read inlined directly at the call site (`a7a77ab6`). |
| Quantized SDPA kernel (Phase-2) | NEEDS-CLEANUP, large surface | Tightly coupled to `quantized.h` packing; head_dim limited to 64/128; needs upstream-facing docs of the bits/group contract; depends on upstream wanting QuantizedKVCache. Multi-PR series if pursued. |
| RDMA GPU lock fix (DSB barriers + nuclear spin) | **fork-only — PR #3456 closed 2026-04-28** | Write-side `__dsb` patch in `Fence::update` + GPU-side nuclear spin + `MAX_ACTIVE_TASKS=5` + **read-side `__dsb` in `Fence::wait` (added 2026-04-27, commit `d6ecdaa9`)**. zcbenz consulted the Apple Metal team and confirmed there is **no supported CPU↔GPU atomic memory coherence primitive** in Metal and no public way to implement a guaranteed-working GPU spinlock; the DSB barriers "may appear to work but only triggers implementation quirks, not because Apple Silicon follows ARM specs." Patch chain stays fork-only. **Disposition revised 2026-04-27**: previously "drop on next rebase absent symptoms"; now "keep, earning its keep" after a 2-node DSv4-Flash-6bit production run produced the partial-deadlock pattern (paired multi-second decode-step spikes capping at 34s). Original prior art: mlx#3141 (vskiwi, closed Feb 24 2026) + still-open issue mlx#3142 (rltakashige). The 2-node repro changes the evidence picture but not the upstream policy answer — don't reopen #3456. See `docs/fork-notes.md` for the full patch chain, history including the 2026-04-27 observation, and PROF `max=` decode-step time as the regression signal. |
| Jaccl refactor revert | won't upstream | Reverts upstream #3412 + #3418; should be a bug report on ml-explore/mlx for the 2-rank RDMA init failure, not a revert PR. |

### mlx-lm (`ml-explore/mlx-lm`)

52 ahead, 1 behind. 7 active groups.

| Group | Status | Notes |
|---|---|---|
| MiniMax config validation + dead-field cleanup | **PR #1204** | Validates `head_dim × num_heads` against `q_proj` shape; drops unused `shared_intermediate_size`. `test_all_models` passes. |
| DSv4 sparse SDPA matmul rewrite | **candidate, not yet filed** | mlx-lm `87f4625`. Replaces two `(broadcast)·(broadcast).sum()` patterns in `_sparse_pooled_attention` with batched matmul over L. Eliminates the `(B, H, L, k, D)` intermediate (~4 GB at 32K shapes per layer per chunk). **Bench-validated 4.2× wall-time speedup at 32K** on `mlx-community/DeepSeek-V4-Flash-6bit` (415s → 98s, decode unchanged at 28.8 tok/s). Pure semantic rewrite — equivalent up to FP-accumulation noise (`max abs diff ≈ 1e-6`). Generic; applies to any DSv4 deployment. Worth filing as a clean PR once the multi-DSv4-PR situation upstream settles (Blaizzy #1192 / machiabeli #1189 / rlt #1195 still in flight). See `fork-notes.md` for details. |
| DSv4 Indexer fp32→bf16 | **candidate, not yet filed** | mlx-lm `f4dd9e7`. Drops three explicit `.astype(mx.float32)` casts in `Indexer.__call__` score path (q, pooled, weights). MLX's bf16 GEMM accumulates fp32 internally; argpartition top-k is robust to small score perturbations. Bench-validated **+3.4% decode at 100K, −10% wall at 100K** on DSv4-Flash-6bit (saved 48s of prefill). Generic; same gating as the matmul rewrite (wait for one DSv4 PR upstream to settle). |
| DSv4 MoEGate fp32→bf16 + Compressor softmax cleanup | **candidate, not yet filed** | mlx-lm `2a1dcf6`. (a) `MoEGate.__call__` routing matmul was running fp32 GEMM though both operands were bf16 — moe_gate was the largest decode contributor (14.9% at 32K). Drop the casts; downstream sigmoid/softmax(precise=True) handle precision for top-k routing. (b) `Compressor.__call__` cast gate to fp32 before `mx.softmax(precise=True)` — the precise flag already promotes internally, the cast was vestigial. Bench-validated **+1.5% decode at 100K**, cumulative **+4.9% decode / −10.2% wall at 100K** when stacked with `f4dd9e7`. Generic; same gating as above. |
| DSv4 `EXO_PROFILER` spans | **fork-only, not for upstream** | mlx-lm `98ebfde`. 14 named spans wrapping `V4Attention.__call__` + `DeepseekV4MoE.__call__`. No-op when `EXO_PROFILER` unset. Intrusive instrumentation — upstream wouldn't accept inline `with span(...)` blocks; their preference is hookable abstractions which we already use here. Fork-carried indefinitely as a debugging tool. |
| `_skip_lm_head` guard | **Issue #1203** | Asked maintainer about return-type asymmetry / compile-graph homogeneity before PR'ing 5-model change. Upstream's PipelineMixin inverts our rank convention (rank 0 = last layers). |
| `QuantizedKVCache.merge()` | **superseded by #941** | ochafik PR open since 2026-03-02, stalled on maintainer (angeloskath) wanting `BatchQuantizedKVCache` redesign. Comment left offering to help drive that. |
| GDN float32 precision fixes | not ours to upstream | Multi-author cluster: dmcc73 (precision), rlt (lightning indexer), Apple/angeloskath (padding eval). Competing PR #1066 (kernelpool, Kahan-compensation approach). Needs DM coordination, not unilateral PR. |
| DeepSeek-V4 model | rides Blaizzy #1192 | Adurham fixes (`223604e`, `6ee9898`, `15de79d`) on top of Blaizzy's PR. Push fixes to that PR rather than open new. **Also: rlt PR #1195 ("Implement DSV4") opened 2026-04-25** — three competing DSv4 PRs (Blaizzy #1192, machiabeli #1189, rlt #1195). Don't add a fourth. |
| DeepSeek-V4 sparse-index attention quadratic blowup | reported on #1192 ([comment](https://github.com/ml-explore/mlx-lm/pull/1192#issuecomment-4323191525)) | `V4Attention.__call__` for `compress_ratio == 4` layers (`mlx_lm/models/deepseek_v4.py:1439-1448`) flattens the per-query top-`k` indexer gather into a *dense* KV of length `L × k` then feeds it to `scaled_dot_product_attention`, materializing a `(B, n_heads, L, L × k)` score tensor — cubic in `L` until `k` saturates at `index_topk=512` (i.e. `L ≥ 2048`). Crashes on Mac at any single prefill chunk above ~1.2K tokens (single-buffer cap ~80 GiB). Observed: 1904-token turn-2 prefill → 227 GB single-allocation rejection. Same pattern in `deepseek_v32.py` and the other competing DSv4 PRs (#1195, #1189). Workaround: cap `prefill_step_size ≤ 1024` for DSv4 (we use 512). Real fix is query-grouped sparse SDPA — offered upstream. |
| MiniMax per-span tracer + `EXO_PROFILE_LAYERS` | **complete** | Extracted to generic `ProfilerHook` protocol in `mlx_lm/profiler.py`. `EXO_PROFILER=spans,layer_memory` replaces both env vars. `minimax_trace.py` is now a thin shim. qwen3_5.py calls `profiler.on_layer_start/end`. Bootstrap registers hook from `EXO_PROFILER`. |
| Batched expert-routing dispatch (`BatchedSwitchGLU`) | **complete** | Extracted from inline monkey-patch to `BatchedSwitchGLU(SwitchGLU)` class in `mlx_lm/models/switch_layers.py`. `fuse_weights()` concatenates quantised gate+up proj for a single `gather_qmm` dispatch. |
| `mx.fast.scaled_dot_product_attention_quant` integration | depends on mlx Phase-2 | Routes quant-KV attention through fused MLX kernel. Blocked on Phase-2 quant SDPA kernel landing in mlx upstream first. |
| GDN chunkwise prefill experiments | abandoned | 2.1× microbench, no real-world win on M4 Max. Cleanly reverted with measurement data preserved in commit chain. |

---

## Coordination — third-party authors

| Author | What they own | How to engage |
|---|---|---|
| **rlt** (Ryuichi Leo Takashige) | DSB barrier (mlx), lightning-indexer batch>1 fix (mlx-lm), DSv4 PR #1195, GDN precision branch where dmcc73's work is sitting | Active upstream contributor; reach out via Discord or directly |
| **dmcc73** (David Correia) | The actual GDN fp32 precision fixes carried in `rlt/fix/float32-logprobs` | No upstream PRs from them yet; they'd be the natural author |
| **Thump604** | Chunked SDPA + LogSumExp + head_dim 192/256 + MLX_SDPA_FUSED_THRESHOLD territory in mlx | PRs #3293/#3306/#3307 closed unmerged 2026-04-04 ("narrowing active upstream work to vllm-mlx [...] If I need this again, I'll reintroduce it from a fresh branch"). Author explicitly opened the door for someone else to revive. We revived #3306 → ours [#3594](https://github.com/ml-explore/mlx/pull/3594) with full credit. Chunked-SDPA (#3307) revival held at `adurham:pr/sdpa-chunked-dispatch` pending #3594 review. head_dim=192/256 (#3293) revival held at `adurham:pr/sdpa-head-dim-192-256` pending decision on float32 limitation. |
| **Blaizzy** | DeepSeek-V4 PR #1192 in mlx-lm | Push our sanitize fixes to their PR, not a new one |
| **ochafik** (Olivier Chafik) | `QuantizedKVCache.merge()` PR #941 | Stalled; comment posted offering to drive the maintainer-requested redesign |
| **angeloskath** (Apple) | Authored `b0e1769` "Ensure padding and offset are evaluated" in `rlt/fix/float32-logprobs` | Apple maintainer aware of GDN precision territory |

---

## Won't upstream

- mlx jaccl revert — should be a bug report, not a revert PR
- exo `start_cluster.sh` and model-card additions — adurham-cluster-specific
- adurham fork-only patches that undo earlier fork-only patches (e.g. `55505bd` removing `log_g` clamp added by an earlier fork commit)

---

## Recommended next actions

1. **Don't expect quick review feedback.** Status snapshot 2026-05-27: 6 of the 7 stale-≥29-days PRs are on `exo-explore/exo` with zero engagement. The one signal: 3rd-party user `DevOpsBenjamin` `+1`'d mlx-lm#1216 ("lost 3 hours from MLX Studio") with no maintainer follow-up. The mlx-explore side does move (mlx#3455 took APPROVED → merged in ~2 weeks), but exo-explore looks heavily backlogged. **Action**: don't open a 13th PR until at least one of the 4 stale exo-explore PRs (#1985, #1988, #1990, #1992) draws a comment or merge. Four new PRs (mlx#3594, mlx#3595, mlx#3596, exo#2121) opened today — fresh, no engagement yet.
2. **If mlx PR #3594 (logsumexp) gets reviewed favorably** → open the held chunked-SDPA PR (`adurham:pr/sdpa-chunked-dispatch`) as a stacked follow-up, with @Thump604 credit chain intact.
3. **For the held head_dim=192/256 PR** (`adurham:pr/sdpa-head-dim-192-256`) → decide on approach (submit as-is with float32 limitation documented vs. parameterize bq per-dtype first). Worth checking what zcbenz's appetite is for documented limitations after seeing the response to #3594.
4. **If radix-trie issue #1986 gets a green light** → split into the 3-PR sequence (caps → pin → trie+extend-in-place).
5. **If sampling-tier issue #1987 gets a green light** → defer per-instance + cluster-env per maintainer guidance. (The uncontroversial API mapping fixes have already been submitted as PR #1991).
6. **DSv4 cluster sharding (exo)** waits on whichever DSv4 PR (Blaizzy #1192 / machiabeli #1189 / rlt #1195) lands.
7. **MTP** port to the new BatchGenerator API landed 2026-04-26. Needs live cluster smoke test (Huihui + Qwen3.5-397B) before flipping `QWEN35_ENABLED=0 → 1` in `start_cluster.sh`.

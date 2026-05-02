# Upstream PR Inventory

Cross-repo tracker for what `adurham/{exo,mlx,mlx-lm}` carries on top of upstream
and what's been pushed forward to upstream review. Companion to
[fork-notes.md](./fork-notes.md), which tracks dependency pins.

Last refresh: 2026-04-28.

---

## Status board

### Open PRs (9)

| Repo | PR | Title | Status |
|---|---|---|---|
| `ml-explore/mlx-lm` | [#1204](https://github.com/ml-explore/mlx-lm/pull/1204) | minimax: validate head_dim against checkpoint, drop unused shared_intermediate_size | review required, CI not run yet |
| `ml-explore/mlx-lm` | [#1216](https://github.com/ml-explore/mlx-lm/pull/1216) | fix(utils): skip already-quantized layers in load_model._quantize predicate | review required, opened 2026-04-27 |
| `ml-explore/mlx` | [#3455](https://github.com/ml-explore/mlx/pull/3455) | Add `MLX_SDPA_BLOCKS` env var for 2-pass vector kernel block-count override | **APPROVED** by zcbenz 2026-04-27, awaiting merge |
| `exo-explore/exo` | [#1985](https://github.com/exo-explore/exo/pull/1985) | feat: Prometheus `/metrics` endpoint | review required |
| `exo-explore/exo` | [#1988](https://github.com/exo-explore/exo/pull/1988) | feat: `EXO_KV_CACHE_BITS` env var + step=16384 for QuantizedKVCache | review required |
| `exo-explore/exo` | [#1990](https://github.com/exo-explore/exo/pull/1990) | fix: skip KV cache quantization in single-node BatchGenerator mode | review required |
| `exo-explore/exo` | [#1992](https://github.com/exo-explore/exo/pull/1992) | feat: peer-to-peer model distribution | review required |
| `exo-explore/exo` | [#1996](https://github.com/exo-explore/exo/pull/1996) | fix(deepseek_v4): drop full-attention sharding for MoE-only strategy | rltakashige commented 2026-04-28 — `mlx-community/DeepSeek-V4-Flash-6bit` is sanitized for Blaizzy's variant, not theirs; claims their implementation is "considerably better in performance and stability." Their only published bench (PR #1195) is 30.1 tok/s single-node, vs our 34.6 tok/s on 2-node M4 Max RDMA TP — claim doesn't survive contact. **Holding ground; no reply yet.** Position: PR is scoped to compatibility with the de facto public checkpoint, not a claim about which implementation is better. Revisit if any DSv4 PR merges upstream. |
| `exo-explore/exo` | [#1999](https://github.com/exo-explore/exo/pull/1999) | perf(deepseek_v4): fuse switch_mlp gate_proj + up_proj into single gather_qmm | review required, stacks on #1996, +1.2% c=1 / +1.1% c=2 bench-validated |

### Recently merged (2)

| Repo | PR | Title |
|---|---|---|
| `exo-explore/exo` | [#1989](https://github.com/exo-explore/exo/pull/1989) | fix: route by in-flight tasks only — completed tasks were skewing load balance |
| `exo-explore/exo` | [#1991](https://github.com/exo-explore/exo/pull/1991) | fix: map presence_penalty and frequency_penalty from ChatCompletionRequest |

### Recently closed without merge (2)

| Repo | PR | Title | Resolution |
|---|---|---|---|
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

### Fork sync state (2026-04-28)

| Fork | vs upstream/main | Notes |
|---|---|---|
| `adurham/mlx-lm` | 0 behind, 57 ahead | Fully current |
| `adurham/mlx` | 0 behind, 40 ahead | Fully current |
| `adurham/exo` | **4 behind**, 359 ahead | **Merge deferred** — see below |

#### Deferred: exo upstream merge (4 commits)

Attempted 2026-04-28; aborted due to **7 conflicts** in cluster-critical files. Dominant blocker is `f0d1371d` MLX P/D (#1993, +4000/-84, 53 files) — major prefill/decode disaggregation feature. Secondary: `c80b10c0` engine abstraction (#2000) renames `runner/llm_inference/runner.py → runner/runner.py` and deletes `runner/image_models/runner.py`, both of which our fork has commits on. Two trivial commits (#1997, #1998) merge cleanly.

Conflict severity by fork-commit count touching each file:

| File | Fork commits | Caused by |
|---|---|---|
| `engines/mlx/generator/generate.py` | 47 | #1993 |
| `engines/mlx/generator/batch_generate.py` | 44 | #2000 + #1993 |
| `engines/mlx/cache.py` | 24 | #1993 |
| `engines/mlx/auto_parallel.py` | 22 | #1993 |
| `runner/llm_inference/runner.py` (renamed → `runner/runner.py`) | 18 | #2000 |
| `runner/llm_inference/batch_generator.py` | 12 | #2000 + #1993 |
| `master/main.py` | 5 | #1993 |
| `runner/image_models/runner.py` (deleted) | 2 | #2000 |

Pickup plan + scope estimate (1-2 days) in memory `exo_upstream_merge_2026_04_28_deferred.md`. Phase 1 is a comprehension pass on the new `worker/disaggregated/` module to determine whether it applies to our 2-node TP cluster — that question changes whether conflict resolution is mostly textual or architectural.

**Don't attempt the merge while perf work is in flight** (DSv4 long-context profile session is currently running in another session against `engines/mlx/generator/generate.py`).

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
| **Thump604** | Chunked SDPA + LogSumExp + head_dim 192/256 + MLX_SDPA_FUSED_THRESHOLD territory in mlx | PRs #3293/#3307 closed unmerged; need to ask if they want to revive |
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

1. **Open mlx PR for `MetalAllocator` heap+small_size leak fix** (fork commits `2c6e86db` + `4d543e19`). Two-line constant change. Verified 3.7× context-ceiling lift on DSv4-Flash long decode. Real upstream regression — any model with many scalar `mx.array` values per step on long workloads will leak the same way. Bring `MLX_LOG_NEW_BUFFER_PATH` cache-miss histogram + `vmmap` region-count diff as evidence. See `fork-notes.md` for the full reproduction. **High-priority new PR; clear win for upstream users.**
2. **Wait for review feedback** on the 9 open PRs and 3 issues before drafting more. All but P2P (#1992) were opened 2026-04-26; #1992 went up the same day after a re-author. Give a few days. (mlx#3454 + mlx#3456 both closed 2026-04-28 — see "Recently closed without merge".)
2. **If radix-trie issue #1986 gets a green light** → split into the 3-PR sequence (caps → pin → trie+extend-in-place).
3. **If sampling-tier issue #1987 gets a green light** → defer per-instance + cluster-env per maintainer guidance. (The uncontroversial API mapping fixes have already been submitted as PR #1991).
4. **DSv4 cluster sharding (exo)** waits on whichever DSv4 PR (Blaizzy #1192 / machiabeli #1189 / rlt #1195) lands.
5. **If Thump604 returns** → revive #3293/#3307 with their authorship, fold our `MLX_SDPA_FUSED_THRESHOLD` env var into that conversation.
6. **MTP** port to the new BatchGenerator API landed 2026-04-26. Needs live cluster smoke test (Huihui + Qwen3.5-397B) before flipping `QWEN35_ENABLED=0 → 1` in `start_cluster.sh`.

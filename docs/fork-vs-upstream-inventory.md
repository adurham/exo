# Fork-vs-Upstream Change Inventory

Complete audit of every file that differs between the `adurham` forks and their
upstream merge-base, as of 2026-06-04. Generated mechanically from
`git diff --name-status <merge-base>..main` per repo, cross-referenced against
`docs/` and the exo-cluster-operations skill references.

## Coverage at a glance

| Repo | Files changed | Core source | Core w/ rationale |
|------|--------------:|------------:|------------------:|
| exo (vs exo-explore/exo) | 217 | 81 | 81 (100%) |
| mlx (vs ml-explore/mlx) | 43 | 36 | 36 (100%) |
| mlx-lm (vs ml-explore/mlx-lm) | 13 | 12 | 12 (100%) |
| **Total** | **273** | **129** | **129 (100%)** |

Every **core source file** (`src/exo/`, `rust/`, `mlx/`, `mlx_lm/`) maps either to a
thematic doc in this directory or to a one-line rationale in the
[Appendix](#appendix-core-changes-not-covered-by-a-thematic-doc-17-files). The
remaining 144 files are benchmarks, model cards, ops scripts, the dashboard,
build/meta, the docs themselves, and `.hermes/plans/` session scratch — self-describing
by filename + commit message, not requiring prose. The 17 core files without a thematic
doc are given explicit rationales in the appendix, so **no core change is unexplained**.

**Reading the tables:** `S`=status (A added / M modified / D deleted), `#c`=number of
non-merge commits touching the file, `Rationale / Doc`=where the change is explained
(a doc file) or a one-line rationale from the representative commit.

> Scope note: pure diagnostic/probe/revert commits are intentionally excluded from
> narrative docs (they're noise). `.hermes/plans/` are session scratch notes that were
> swept in by an early `git add -A` accident — flagged here, candidates for removal.


---

## exo  (vs exo-explore/exo, merge-base `09f9ea31`)

**217 files changed.**


### Benchmarks & probe scripts  (54 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `bench/aistupid_harness.py` | 1 | mentioned in: deepseek-v4-mtp-performance.md, mtp-tiebreak-losslessness-fix.md |
| A | `bench/c2_bitequiv_probe.py` | 2 | _fix(bench): import ExoClient from exo_tools.client (post-restructure)_ |
| A | `bench/chunkwise_benchmark.py` | 1 | mentioned in: prefill-optimization.md |
| A | `bench/compress_kv_microbench.py` | 1 | kv-cache-architecture.md |
| A | `bench/concurrent_bench.py` | 3 | mentioned in: deepseek-v4-mtp-performance.md, fork-notes.md |
| A | `bench/context_stress.py` | 1 | mentioned in: skill:bench-tooling.md |
| A | `bench/converge_probe.py` | 1 | mentioned in: skill:2026-06-03-fused-think-delimiter-parser-bug.md, skill:2026-06-03-mtp-losslessness-tiebreak-rootcause.md |
| A | `bench/dsv4_indexer_pipelined_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/dsv4_sparse_pooled_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/dsv4_topk_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| M | `bench/eval_configs/models.toml` | 3 | mentioned in: upstream-prs.md, api.md |
| M | `bench/exo_eval.py` | 2 | mentioned in: iogpu-residency-set-abort.md |
| A | `bench/fp16_peak_bench.py` | 1 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `bench/full_moe_microbench.py` | 1 | mentioned in: fork-notes.md, skill:nop-probe-bottleneck-attribution.md |
| A | `bench/gather_qmv_microbench.py` | 1 | mentioned in: fork-notes.md, skill:distributed-bottleneck-attribution.md |
| A | `bench/hard_eval.py` | 1 | mentioned in: skill:2026-06-03-mtp-losslessness-tiebreak-rootcause.md, skill:mtp-speculative-losslessness-debugging.md |
| A | `bench/indexer_score_microbench.py` | 1 | mentioned in: skill:perf-exploration-workflow.md, skill:bench-tooling.md |
| A | `bench/minimax_cluster_ab.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_dispatch_count.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_fused_qkv_end_to_end.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_qkv_merge_probe.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_quant_concat_probe.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_rope_cache_breakdown.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/mtp_eagle_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/phase_g_sweep.py` | 1 | _chore(bench): phase_g_sweep.py — prefill + decode sweep via streaming SSE_ |
| A | `bench/quality_probe_dsv4.py` | 2 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/quant_compare.py` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/algo_impl_4bit-qkv.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/algo_impl_4bit.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/algo_impl_nvfp4.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/arch_design_4bit-qkv.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/arch_design_4bit.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/arch_design_nvfp4.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/code_review_4bit-qkv.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/code_review_4bit.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/code_review_nvfp4.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/debug_fix_4bit-qkv.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/debug_fix_4bit.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/debug_fix_nvfp4.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/refactor_4bit-qkv.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/refactor_4bit.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/quant_compare_results/refactor_nvfp4.md` | 1 | turboquant-integration.md / kv-cache-architecture.md |
| A | `bench/rope_unsqueeze_smoke.py` | 1 | _docs(fork-notes): May 12 validation runs + 110K model quality cliff_ |
| A | `bench/sdpa_decomposed_bench.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sdpa_decomposed_v2_bench.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sdpa_gqa_profile.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sdpa_prefill_bench.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sparse_attn_compile_smoke.py` | 1 | _bench: microbenches for DSv4 attention path optimization_ |
| A | `bench/sparse_pooled_attn_microbench.py` | 1 | mentioned in: skill:bench-tooling.md |
| A | `bench/sparse_pooled_refactor_microbench.py` | 1 | mentioned in: fork-notes.md, skill:perf-exploration-workflow.md |
| A | `bench/spec_degen_capture.py` | 2 | mentioned in: skill:2026-05-29-mtp-degeneration-spec-trace-harness.md, skill:2026-05-29-mtp-degeneration-code-analysis-and-trace.md |
| A | `bench/spec_degen_diff.py` | 1 | mentioned in: skill:2026-05-29-mtp-degeneration-spec-trace-harness.md, skill:2026-05-29-mtp-degeneration-code-analysis-and-trace.md |
| A | `bench/swap_to_8bit_and_rerun.sh` | 1 | _bench: --indices flag + swap-to-8bit runbook_ |
| A | `bench/test_tree_mask.py` | 4 | mentioned in: skill:2026-05-19-tree-drafting-forensics.md, skill:2026-05-20-tree-drafting-phase6b-fix.md |

### Build/meta  (2 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `pyproject.toml` | 30 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `uv.lock` | 243 | mentioned in: upstream-prs.md, fork-notes.md |

### CORE exo source  (81 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `src/exo/api/main.py` | 4 | api.md |
| M | `src/exo/api/types/__init__.py` | 2 | api.md |
| M | `src/exo/api/types/api.py` | 4 | api.md |
| M | `src/exo/download/coordinator.py` | 3 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/download_utils.py` | 7 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| A | `src/exo/download/file_server.py` | 5 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/impl_shard_downloader.py` | 1 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/shard_downloader.py` | 1 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/tests/test_cancel_download.py` | 1 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/tests/test_re_download.py` | 1 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/main.py` | 3 | mentioned in: upstream-prs.md, skill:upstream-integration-and-deploy.md |
| M | `src/exo/master/main.py` | 4 | mentioned in: upstream-prs.md, skill:upstream-integration-and-deploy.md |
| M | `src/exo/master/placement.py` | 1 | mentioned in: api.md, fork-notes.md |
| M | `src/exo/master/placement_utils.py` | 1 | _feat: add EXO_PP_LAYER_SPLIT for pipeline stage rebalancing_ |
| M | `src/exo/master/tests/test_master.py` | 1 | _fix: index master events synchronously to enable multi-instance load balancing_ |
| A | `src/exo/master/tests/test_routing_concurrency.py` | 1 | _test: add concurrent routing test + bench harness for sibling contention_ |
| A | `src/exo/metrics/__init__.py` | 2 | metrics.md |
| A | `src/exo/metrics/metrics.py` | 3 | metrics.md |
| A | `src/exo/metrics/tests/__init__.py` | 1 | metrics.md |
| A | `src/exo/metrics/tests/test_metrics.py` | 1 | metrics.md |
| M | `src/exo/shared/constants.py` | 7 | mentioned in: fork-notes.md, minimax-rdma-moe-validation-2026-04-24.md |
| M | `src/exo/shared/types/commands.py` | 1 | mentioned in: skill:session-workflow-rules.md, skill:notifying-user-via-discord-gateway-cron.md |
| M | `src/exo/shared/types/tasks.py` | 1 | mentioned in: upstream-prs.md, deepseek-v4-mtp-performance.md |
| M | `src/exo/shared/types/text_generation.py` | 2 | _feat(sampling): card tier between instance and cluster-env defaults_ |
| M | `src/exo/shared/types/worker/downloads.py` | 2 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/shared/types/worker/instances.py` | 6 | mentioned in: api.md, fork-notes.md |
| M | `src/exo/worker/engines/mlx/auto_parallel.py` | 32 | mentioned in: fork-notes.md, minimax-fused-attention-prompt.md |
| M | `src/exo/worker/engines/mlx/builder.py` | 1 | mentioned in: skill:2026-05-19-tree-drafting-forensics.md, skill:2026-05-25-pool-defer-and-compressor-attribution.md |
| M | `src/exo/worker/engines/mlx/cache.py` | 24 | kv-cache-architecture.md |
| M | `src/exo/worker/engines/mlx/constants.py` | 9 | mentioned in: fork-notes.md, minimax-rdma-moe-validation-2026-04-24.md |
| M | `src/exo/worker/engines/mlx/disaggregated/adapter.py` | 1 | mentioned in: minimax-quantized-sdpa-design.md, skill:session-workflow-rules.md |
| M | `src/exo/worker/engines/mlx/generator/batch_generate.py` | 76 | mentioned in: skill:mtp-bistability-2026-05-17-investigation.md, skill:2026-05-21-c2-100k-mtp-recipe.md |
| M | `src/exo/worker/engines/mlx/generator/generate.py` | 56 | mentioned in: upstream-prs.md, api.md |
| M | `src/exo/worker/engines/mlx/patches/__init__.py` | 3 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| A | `src/exo/worker/engines/mlx/patches/minimax/__init__.py` | 2 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `src/exo/worker/engines/mlx/patches/minimax/fused_qkv.py` | 3 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `src/exo/worker/engines/mlx/patches/minimax/tests/__init__.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `src/exo/worker/engines/mlx/patches/minimax/tests/test_fused_qkv.py` | 2 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| M | `src/exo/worker/engines/mlx/patches/opt_batch_gen.py` | 12 | mentioned in: fork-notes.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5/__init__.py` | 1 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5/custom_qmv_loop_over_b.py` | 1 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5/lpb_patch.py` | 1 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/__init__.py` | 1 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/apply.py` | 2 | mentioned in: upstream-prs.md, fork-notes.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_fused_gqa_attention.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_moe.py` | 4 | mentioned in: minimax-decode-optimization.md, skill:verify-cost-decomposition-2026-05-19.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/common.py` | 7 | mentioned in: kv-cache-architecture.md, turboquant-integration.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/decoder.py` | 2 | mentioned in: minimax-fused-attention-prompt.md, minimax-decode-optimization.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/fused_gdn_attention.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/__init__.py` | 1 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_fused_gdn_projections_8bit.py` | 2 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_fused_gqa_projections_8bit.py` | 2 | mentioned in: minimax-decode-optimization.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_merged_down_proj_8bit.py` | 2 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_moe_epilogue.py` | 2 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_oproj_gate_gemv_8bit.py` | 2 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_softmax_topk_swiglu_8bit.py` | 2 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_gdn_projections_8bit.py` | 2 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_qk_rmsnorm.py` | 2 | mentioned in: minimax-fused-attention-prompt.md, minimax-decode-optimization.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_rms_norm_gated.py` | 2 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/gdn_step_precomputed.py` | 1 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `src/exo/worker/engines/mlx/pp_speculation.py` | 36 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/sampling.py` | 4 | mentioned in: upstream-prs.md, api.md |
| A | `src/exo/worker/engines/mlx/speculative/__init__.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` | 87 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py` | 3 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/mtp_module.py` | 43 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/speculative_cache.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/speculative_gdn_kernel.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/tests/test_prefill_batched.py` | 1 | prefill-optimization.md |
| A | `src/exo/worker/engines/mlx/trace.py` | 1 | mentioned in: upstream-prs.md, fork-notes.md |
| A | `src/exo/worker/engines/mlx/turboquant_cache.py` | 4 | turboquant-integration.md / kv-cache-architecture.md |
| M | `src/exo/worker/engines/mlx/types.py` | 0 | mentioned in: api.md, kv-cache-architecture.md |
| M | `src/exo/worker/engines/mlx/utils_mlx.py` | 24 | mentioned in: skill:upstream-sync-triage.md |
| M | `src/exo/worker/main.py` | 2 | mentioned in: upstream-prs.md, skill:upstream-integration-and-deploy.md |
| M | `src/exo/worker/plan.py` | 5 | _fix: thread node_network through plan() to _model_needs_download()_ |
| M | `src/exo/worker/runner/bootstrap.py` | 7 | mentioned in: upstream-prs.md, skill:2026-05-22-bootstrap-barrier-and-ce5c64fd-wedge.md |
| M | `src/exo/worker/runner/llm_inference/batch_generator.py` | 21 | mentioned in: upstream-prs.md, skill:mlx-upstream-port-validation.md |
| M | `src/exo/worker/runner/llm_inference/model_output_parsers.py` | 1 | thinking-parser-fused-delimiter-fix.md |
| M | `src/exo/worker/runner/runner.py` | 4 | mentioned in: upstream-prs.md, thinking-parser-fused-delimiter-fix.md |
| M | `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py` | 2 | kv-cache-architecture.md |
| M | `src/exo/worker/tests/unittests/test_runner/test_finish_reason_sse.py` | 1 | mentioned in: thinking-parser-fused-delimiter-fix.md, skill:2026-06-03-fused-think-delimiter-parser-bug.md |

### Dashboard (UI)  (1 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `dashboard/src/lib/stores/app.svelte.ts` | 1 | _fix: P2P peer discovery in API start_download + dashboard pause button_ |

### Documentation (the docs themselves)  (22 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `docs/deepseek-v4-flash-kickoff-prompt.md` | 2 | _deepseek-v4: switch to mlx-community/DeepSeek-V4-Flash-4bit_ |
| A | `docs/deepseek-v4-mtp-performance.md` | 2 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `docs/fork-notes.md` | 24 | mentioned in: upstream-prs.md, deepseek-v4-mtp-performance.md |
| A | `docs/iogpu-residency-set-abort.md` | 1 | iogpu-residency-set-abort.md |
| A | `docs/kv-cache-architecture.md` | 4 | kv-cache-architecture.md |
| A | `docs/metrics.md` | 1 | metrics.md |
| A | `docs/minimax-decode-optimization.md` | 11 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-fused-attention-design.md` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-fused-attention-prompt.md` | 3 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-quantized-sdpa-design.md` | 8 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-rdma-moe-validation-2026-04-24.md` | 3 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| A | `docs/mtp-tiebreak-losslessness-fix.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `docs/prefill-optimization.md` | 2 | prefill-optimization.md |
| A | `docs/profiling/request_lifecycle_trace.md` | 10 | _docs: mark prefill optimization as compute-bound and complete_ |
| A | `docs/thinking-parser-fused-delimiter-fix.md` | 1 | thinking-parser-fused-delimiter-fix.md |
| A | `docs/turboquant-integration.md` | 2 | turboquant-integration.md / kv-cache-architecture.md |
| A | `docs/upstream-pr-drafts/02-mlx-sdpa-chunked.md` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `docs/upstream-pr-drafts/04-mlx-head-dim-192-256.md` | 1 | _docs(upstream-prs): record 5 PR outcomes from 2026-05-27 sync session_ |
| A | `docs/upstream-pr-drafts/06-mlx-allocator-coalesce.md` | 1 | _docs(upstream-prs): add mlx#3596 (MetalAllocator coalesce) to open PRs_ |
| A | `docs/upstream-pr-drafts/07-exo-thinking-parser-fused-delimiter.md` | 1 | thinking-parser-fused-delimiter-fix.md |
| A | `docs/upstream-pr-drafts/README.md` | 3 | mentioned in: skill:upstream-sync-triage.md |
| A | `docs/upstream-prs.md` | 20 | mentioned in: deepseek-v4-mtp-performance.md, fork-notes.md |

### Model cards (config)  (4 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `resources/inference_model_cards/mlx-community--DeepSeek-V4-Flash-4bit.toml` | 1 | _deepseek-v4: switch to mlx-community/DeepSeek-V4-Flash-4bit_ |
| A | `resources/inference_model_cards/mlx-community--DeepSeek-V4-Flash-6bit.toml` | 1 | _deepseek-v4: add 6-bit quant model card_ |
| A | `resources/inference_model_cards/mlx-community--Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit.toml` | 1 | _chore: add model card for Huihui-Qwen3.5-35B-A3B-Opus-abliterated-4bit_ |
| A | `resources/inference_model_cards/mlx-community--Qwen3.5-397B-A17B-nvfp4.toml` | 1 | _feat: restore P2P model download and add Qwen3.5-397B-A17B-nvfp4 model card_ |

### Ops scripts  (1 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `scripts/convert_dsv4_mtp.sh` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |

### Other  (9 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `.gitignore` | 3 | mentioned in: skill:2026-05-26-mlx-upstream-refactor-port-attempt.md, skill:2026-05-26-mlx-upstream-sync-port.md |
| A | `.gitmodules` | 1 | mentioned in: skill:mlx-deployment-verification.md, skill:upstream-sync-triage.md |
| A | `deploy/grafana/exo-overview.json` | 1 | mentioned in: metrics.md |
| A | `eagle_k1_fix_report.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `mlx` | 62 | mentioned in: upstream-prs.md, api.md |
| A | `mlx-lm` | 135 | mentioned in: upstream-prs.md, thinking-parser-fused-delimiter-fix.md |
| A | `start_cluster.sh` | 169 | mentioned in: upstream-prs.md, deepseek-v4-mtp-performance.md |
| A | `sync-upstream.sh` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `tools/dsv4_route_hist_summary.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |

### Z-scratch (session notes, accidentally committed)  (43 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `.hermes/plans/2026-05-14_113951-dsv4-moe-fused-metal-kernel.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-14_140936-dsv4-sparse-attn-fused-kernel.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-14_152300-dsv4-expert-colocation-bandwidth.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-14_155955-dsv4-compress-ratios-reshape.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-14_185010-dsv4-indexer-fused-kernel.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-16_2030-mtp-g2-iter1-bistability-investigation.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-16_2255-after-eager-commit-revert.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-17-session-retrospective.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-18-session-retro.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-18_1505-dsv4-verify-forward-toward-35tps.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-18_1830-dsv4-verify-tail-investigation.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-19_allsum_tail_findings.md` | 1 | mentioned in: skill:2026-05-19-verify-tail-investigation.md, skill:mlx-jaccl-perf-levers.md |
| A | `.hermes/plans/2026-05-19_build_probe_findings.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-19_critical_path_findings.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-19_mtp_head_investigation.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-19_phase1_findings.md` | 1 | mentioned in: skill:2026-05-19-tree-drafting-forensics.md, skill:mtp-instrumentation-graph-leak-trap.md |
| A | `.hermes/plans/2026-05-19_phase6_findings.md` | 1 | mentioned in: skill:2026-05-19-tree-drafting-forensics.md, skill:2026-05-20-tree-drafting-phase6b-fix.md |
| A | `.hermes/plans/2026-05-19_phase_f_findings.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-19_phase_j_findings.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-19_quality_findings.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-19_structural_to_35tps.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-19_to_35tps.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-19_token_tree_drafting.md` | 1 | mentioned in: skill:2026-05-19-tree-drafting-forensics.md |
| A | `.hermes/plans/2026-05-20_phase10_mtp_c2_fix.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-20_phase11_c2_progress.md` | 1 | mentioned in: skill:2026-05-20-c2-mtp-perf-fix.md, skill:2026-05-20-c2-mtp-fixes-and-bench-pitfalls.md |
| A | `.hermes/plans/2026-05-20_phase12_drain_elim_results.md` | 1 | mentioned in: skill:2026-05-20-c2-mtp-perf-fix.md |
| A | `.hermes/plans/2026-05-20_phase6b_findings.md` | 1 | mentioned in: skill:2026-05-20-tree-drafting-phase6b-fix.md |
| A | `.hermes/plans/2026-05-20_phase7_perf_findings.md` | 1 | mentioned in: skill:2026-05-20-tree-drafting-phase7-perf-negative.md, skill:2026-05-20-mtp-c2-and-kv-bits-trap.md |
| A | `.hermes/plans/2026-05-20_phase8_beating_linear.md` | 1 | mentioned in: skill:2026-05-20-mtp-c2-cache-lifecycle.md, skill:2026-05-20-mtp-c2-and-kv-bits-trap.md |
| A | `.hermes/plans/2026-05-20_phase9_c2_findings.md` | 1 | mentioned in: skill:2026-05-20-mtp-c2-cache-lifecycle.md, skill:2026-05-20-mtp-c2-and-kv-bits-trap.md |
| A | `.hermes/plans/2026-05-20_tree_drafting_continuation.md` | 1 | mentioned in: skill:2026-05-20-tree-drafting-phase6b-fix.md |
| A | `.hermes/plans/2026-05-21_phase13_c2_milestone.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-21_phase14_next_steps.md` | 1 | mentioned in: skill:2026-05-22-c2-bistability-fix-fence4.md, skill:2026-05-22-c2-bistability-fence-resolution.md |
| A | `.hermes/plans/2026-05-22_eagle_k1_debug_report.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-22_phase14_handoff.md` | 1 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `.hermes/plans/2026-05-23_35_tps_plan.md` | 1 | mentioned in: skill:2026-05-24-c2-deprioritized.md, skill:2026-05-24-mtp-eagle-k8-norenorm-champion.md |
| A | `.hermes/plans/2026-05-23_c2_100k_quality_bug_discovery.md` | 1 | mentioned in: fork-notes.md, skill:2026-05-23-c2-100k-quality-bug.md |
| A | `.hermes/plans/2026-05-23_gamma3_bistability_fix_plan.md` | 1 | mentioned in: skill:2026-05-23-c2-100k-quality-bug.md, skill:2026-05-23-quality-probe-c2-gap.md |
| A | `.hermes/plans/2026-05-23_instrumentation_session_findings.md` | 1 | mentioned in: fork-notes.md, skill:2026-05-23-c2-100k-quality-bug.md |
| A | `.hermes/plans/2026-05-23_session_eagle_to_gamma3_findings.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `.hermes/plans/2026-05-24_35_tps_plan_execution_results.md` | 1 | _default(start_cluster): promote EXO_DSV4_MTP_EAGLE_K=8_ |
| A | `.hermes/plans/2026-05-24_w3_K8_norenorm_results.md` | 1 | mentioned in: skill:2026-05-24-mtp-eagle-k8-norenorm-champion.md |
| A | `.hermes/plans/2026-05-25_session_writeup.md` | 1 | mentioned in: skill:2026-05-25-indexer-nop-post-defer.md |

---

## mlx  (vs ml-explore/mlx, merge-base `b155224b9`)

**43 files changed.**


### CORE mlx (C++/Metal backend)  (36 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `mlx/array.cpp` | 4 | mentioned in: api.md, fork-notes.md |
| M | `mlx/array.h` | 4 | mentioned in: api.md, fork-notes.md |
| M | `mlx/backend/metal/allocator.cpp` | 2 | mentioned in: fork-notes.md, minimax-decode-optimization.md |
| M | `mlx/backend/metal/allocator.h` | 6 | mentioned in: fork-notes.md, minimax-decode-optimization.md |
| M | `mlx/backend/metal/device.cpp` | 2 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/backend/metal/device.h` | 1 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/backend/metal/device_info.cpp` | 1 | _fix(allocator): bump rsrc_limit fallback 499000 -> 5000000_ |
| M | `mlx/backend/metal/eval.cpp` | 4 | mentioned in: skill:mlx-source-read-falsification-pattern.md, skill:session-discipline-pitfalls.md |
| M | `mlx/backend/metal/event.cpp` | 5 | mentioned in: upstream-prs.md, api.md |
| M | `mlx/backend/metal/fence.cpp` | 4 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/backend/metal/kernels/CMakeLists.txt` | 1 | mentioned in: skill:2026-05-26-mlx-upstream-sync-port.md, skill:upstream-sync-triage.md |
| M | `mlx/backend/metal/kernels/fence.metal` | 3 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/backend/metal/kernels/scaled_dot_product_attention.metal` | 2 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `mlx/backend/metal/kernels/sdpa_vector_quant.h` | 5 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `mlx/backend/metal/kernels/steel/attn/kernels/sdpa_chunked_reduce.h` | 2 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `mlx/backend/metal/kernels/steel/attn/kernels/sdpa_chunked_reduce.metal` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| M | `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| M | `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| M | `mlx/backend/metal/metal.h` | 2 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/backend/metal/no_metal.cpp` | 2 | _metal: gpu_time_ns() — accumulate GPU-busy time from MTLCommandBuffer timestamps_ |
| M | `mlx/backend/metal/resident.cpp` | 2 | iogpu-residency-set-abort.md |
| M | `mlx/backend/metal/scaled_dot_product_attention.cpp` | 11 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| M | `mlx/distributed/jaccl/jaccl.cpp` | 3 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/jaccl.h` | 1 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/group.h` | 2 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/mesh.cpp` | 2 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/mesh.h` | 1 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/mesh_impl.h` | 3 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/rdma.cpp` | 1 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/rdma.h` | 1 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/fast.cpp` | 1 | mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md |
| M | `mlx/fast.h` | 1 | mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md |
| M | `mlx/fast_primitives.h` | 1 | mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md |
| M | `mlx/ops.cpp` | 1 | mentioned in: skill:collective-coordination-levers.md, skill:mlx-cross-build-abi-trap.md |
| M | `mlx/scheduler.h` | 4 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/transforms.cpp` | 3 | mentioned in: fork-notes.md, skill:mtp-bistability-2026-05-17-investigation.md |

### mlx python bindings/tests  (7 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `python/src/distributed.cpp` | 1 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| M | `python/src/fast.cpp` | 1 | mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md |
| M | `python/src/metal.cpp` | 4 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `python/src/transforms.cpp` | 5 | mentioned in: fork-notes.md, skill:mtp-bistability-2026-05-17-investigation.md |
| A | `python/tests/test_fast_sdpa_quant.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `python/tests/test_sdpa_chunked.py` | 3 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `python/tests/test_sdpa_logsumexp.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |

---

## mlx-lm  (vs ml-explore/mlx-lm, merge-base `df1d3f3c9`)

**13 files changed.**


### CORE mlx-lm (model source)  (12 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `mlx_lm/generate.py` | 18 | mentioned in: upstream-prs.md, api.md |
| M | `mlx_lm/models/base.py` | 2 | mentioned in: fork-notes.md, minimax-decode-optimization.md |
| M | `mlx_lm/models/cache.py` | 33 | kv-cache-architecture.md |
| A | `mlx_lm/models/deepseek_v4.py` | 135 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| M | `mlx_lm/models/gated_delta.py` | 13 | mentioned in: prefill-optimization.md |
| A | `mlx_lm/models/hyper_connection.py` | 1 | mentioned in: skill:mx-compile-patterns.md, skill:2026-05-18-compile-boundary-regressions.md |
| M | `mlx_lm/models/minimax.py` | 4 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `mlx_lm/models/minimax_trace.py` | 2 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| M | `mlx_lm/models/qwen3_5.py` | 17 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx_lm/models/switch_layers.py` | 3 | mentioned in: upstream-prs.md, skill:custom-metal-kernel-authorship.md |
| A | `mlx_lm/profiler.py` | 2 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx_lm/utils.py` | 11 | mentioned in: upstream-prs.md, fork-notes.md |

### mlx-lm tests  (1 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `tests/test_models.py` | 20 | _Simplify HyperConnection_ |

---

## Coverage summary

- Total changed files across 3 repos: **273**
- CORE source files (src/exo, mlx backend, mlx_lm): **129**, of which **112** map to a subsystem doc.
- Remaining files are benchmarks, model cards, ops scripts, dashboard, build/meta, or session scratch — self-describing by name + commit message, not requiring prose docs.

---

## Appendix: core changes not covered by a thematic doc (17 files)

These core-source files weren't referenced by name in any `docs/` writeup. Rationale
recorded here so the inventory has zero unexplained core files. Most are a single
coherent group (the Qwen3.5-MoE kernel patch set).

### Qwen3.5 / Qwen3.5-MoE engine patches (10 files) — `src/exo/worker/engines/mlx/patches/`
Custom MLX kernel + monkey-patch set for the *secondary* Qwen3.5-397B model path
(distinct from the DSv4-Flash champion). Two efforts:
- **MTP + kernel patches for Qwen3.5** (`custom_qmv_loop_over_b.py`, `lpb_patch.py`,
  `gdn_step_precomputed.py`): port of MTP speculative decoding + GDN (gated delta-net)
  step kernels to the Qwen3.5 engine.
- **bf16→fp16 compute-dtype switch for ~7% faster quantized_matmul** (the 7
  `*_8bit.py` / `fused_*` kernels): batched fused GDN projections, merged down-proj,
  MoE epilogue, o-proj gate GEMV, softmax-topk-swiglu, fused RMSNorm-gated. All are
  8-bit quantized-matmul fast paths for the Qwen3.5-MoE decode loop.
- Status: Qwen3.5 path is gated off by default (`QWEN35_ENABLED=0` in start_cluster.sh);
  these are fork-only, model-specific, not upstreamable (no Qwen3.5 engine upstream).

### Master / routing / sampling (4 files) — `src/exo/master/`, `src/exo/shared/`
- `placement_utils.py` — `EXO_PP_LAYER_SPLIT` env for manual pipeline-stage layer
  rebalancing (tune shard boundaries across the 2 nodes).
- `master/tests/test_master.py` — updated for synchronous event indexing (the
  multi-instance load-balancing fix, same change as the `_index_apply_broadcast`
  path preserved during the zenoh merge).
- `master/tests/test_routing_concurrency.py` — new test for sibling-instance routing
  contention (c≥2 era; harness for the in-flight-task load-balance fix).
- `shared/types/text_generation.py` — adds a per-model-card sampling-defaults tier
  between per-instance and cluster-env (relates to exo issue #1987).

### Worker plan (1 file) — `src/exo/worker/plan.py`
Threads `node_network` through `plan()` into `_model_needs_download()` so download
decisions are network-topology aware (P2P model-distribution support).

### mlx backend (2 files) — `mlx/backend/metal/`
- `device_info.cpp` — bumped the Metal resource-limit fallback (`rsrc_limit`
  499000 → 5000000) so large DSv4 shards don't hit a stale cap. Fork-only tuning.
- `no_metal.cpp` — stub side of `gpu_time_ns()` (GPU-busy time from MTLCommandBuffer
  timestamps); the profiling instrument's no-Metal build path.

# Fork-vs-Upstream Change Inventory

Complete audit of every file that differs between the `adurham` forks and their
upstream merge-base, as of 2026-10-16. Generated mechanically from
`git diff --name-status <merge-base>..main` per repo, cross-referenced against
`docs/` and the exo-cluster-operations skill references.

> **Regenerated 2026-10-16 (post upstream-sync).** Merge commits: exo `1aa7d6d88`
> (upstream merge-base `bc412e223`, +91 upstream commits), mlx `b17781838`
> (merge-base `b6642e604`), mlx-lm `676f803e` (merge-base `93e6d4a6`).
> Previous generation 2026-06-04 recorded exo 217 files / 81 core, mlx 43/36,
> mlx-lm 13/12. The increase is real and expected: between June and October the
> fork accumulated a large *fork-only* surface (session/campaign scratch under
> `tmp/`, per-session writeups under `docs/`, `bench/`, and the vendored
> `mlx_lm/models/deepseek_v41/` + `mlx_lm/models/exl3/` packages) that upstream
> has no counterpart for, so it shows up as fork divergence from the tip. All 160 core files previously lacking a rationale received evidence-sourced one-line rationales on 2026-10-16 (see Fork rationales).

## Coverage at a glance

| Repo | Files changed | Core source | Core w/ rationale |
|------|--------------:|------------:|------------------:|
| exo (vs exo-explore/exo) | 2002 | 268 | 268 (100%) |
| mlx (vs ml-explore/mlx) | 71 | 60 | 60 (100%) |
| mlx-lm (vs ml-explore/mlx-lm) | 245 | 67 | 67 (100%) |
| **Total** | **2318** | **395** | **395 (100%)** |

Every **core source file** (`src/exo/`, `rust/`, `mlx/`, `mlx_lm/`) maps either to a
thematic doc in this directory or to a one-line rationale in the
[Appendix](#appendix-core-changes-not-covered-by-a-thematic-doc). The remaining 1923 files
are benchmarks, model cards, ops scripts, campaign scratch (`tmp/`), the dashboard,
build/meta, the docs themselves, and `.hermes/plans/` session scratch - self-describing
by filename + commit message, not requiring prose. Core files whose change could not previously be traced to a thematic doc or to the sync resolution notes are listed under
[Fork rationales](#fork-rationales-previously-unsourced-core-files).

**Reading the tables:** `S`=status (A added / M modified / D deleted), `#c`=number of
non-merge commits touching the file in `<merge-base>..main`, `Rationale / Doc`=where the
change is explained (a doc file) or a one-line rationale from the representative commit.
Rows annotated `sync-notes-*` are sourced from the 2026-10-16 sync resolution notes.

> Scope note: pure diagnostic/probe/revert commits are intentionally excluded from
> narrative docs (they're noise). `.hermes/plans/` are session scratch notes that were
> swept in by an early `git add -A` accident - flagged here, candidates for removal.
> `tmp/` is per-campaign working scratch (preregistration docs, probe harnesses, run
> logs) committed during the Aug-Sep 2026 decode/prefill campaigns - also a removal
> candidate; it is enumerated with a commit-subject rationale only.


---

## exo  (vs exo-explore/exo, merge-base `bc412e223`)

**2002 files changed.**


### Benchmarks & probe scripts  (233 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `bench/ab_probe_tier1.py` | 4 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `bench/absmax_fence_artifact_probe.py` | 1 | _bench+docs: confirm local_absmax's 400ms probe cost was an mx.eval() fence artifact_ |
| A | `bench/actual_tflops_bench.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/aistupid_harness.py` | 1 | mentioned in: deepseek-v4-mtp-performance.md, mtp-tiebreak-losslessness-fix.md |
| A | `bench/attn_production_class_bench.py` | 1 | _bench: attention sub-kernel achieved-vs-ceiling analysis (44.3% of prefill wall)_ |
| A | `bench/c2_bitequiv_probe.py` | 2 | _fix(bench): import ExoClient from exo_tools.client (post-restructure)_ |
| A | `bench/c2_temp1_degen_probe.py` | 1 | _bench(dsv4): c=2 temp>0 degeneration probe (ragged pair, stutter/loop scan)_ |
| A | `bench/chunkwise_benchmark.py` | 1 | mentioned in: prefill-optimization.md |
| A | `bench/cliffband_380k_probe.py` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/compress_kv_microbench.py` | 1 | kv-cache-architecture.md |
| A | `bench/concurrent_bench.py` | 3 | mentioned in: deepseek-v4-mtp-performance.md, fork-notes.md |
| A | `bench/consumer_ab.py` | 1 | _bench: persist loop-2 artifacts (MoE bench report + scripts, consumer sizing report, consumer_ab arm script)_ |
| A | `bench/context_stress.py` | 1 | mentioned in: skill:bench-tooling.md |
| A | `bench/converge_probe.py` | 1 | mentioned in: skill:2026-06-03-fused-think-delimiter-parser-bug.md, skill:2026-06-03-mtp-losslessness-tiebreak-rootcause.md |
| A | `bench/decode_depth_sweep.py` | 1 | _docs: Section 85 -- the mode-determinant is the PREFILL CODE PATH, not depth or a transport race_ |
| A | `bench/decode_probe.py` | 1 | _perf(dsv4): validate MoE gate+up fusion -- real +3% decode win_ |
| A | `bench/decode_probe_ab.py` | 1 | _perf(dsv4): default-on prefill tier-1 flags — 255->353 t/s at 100K (+38%)_ |
| A | `bench/dense_qmm_ceiling.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/diversity_decode_sweep.py` | 1 | _docs+bench: Section 58 -- length eliminated as the decode variable, and an uncaught PrefillCancelled that kills the runner_ |
| A | `bench/dsv41_loop2/consumer_index_sizing.md` | 1 | _bench: persist loop-2 artifacts (MoE bench report + scripts, consumer sizing report, consumer_ab arm script)_ |
| A | `bench/dsv41_loop2/final_block.py` | 1 | _bench: persist loop-2 artifacts (MoE bench report + scripts, consumer sizing report, consumer_ab arm script)_ |
| A | `bench/dsv41_loop2/moe_bench2.py` | 1 | _bench: persist loop-2 artifacts (MoE bench report + scripts, consumer sizing report, consumer_ab arm script)_ |
| A | `bench/dsv41_loop2/moe_bench_report.md` | 1 | _bench: persist loop-2 artifacts (MoE bench report + scripts, consumer sizing report, consumer_ab arm script)_ |
| A | `bench/dsv41_quality_battery/1m_soak3.sh` | 1 | _bench: persist the DSv4.1 quality battery + ladder/soak tooling (battery.py, compare.py, reverdict.py, ladder_exactness.py, 1m_soak3.sh, runbook)_ |
| A | `bench/dsv41_quality_battery/README.md` | 1 | _bench: README for the quality battery tooling (was missed in the initial persist)_ |
| A | `bench/dsv41_quality_battery/_mock_integration_test.py` | 1 | _bench: persist the DSv4.1 quality battery + ladder/soak tooling (battery.py, compare.py, reverdict.py, ladder_exactness.py, 1m_soak3.sh, runbook)_ |
| A | `bench/dsv41_quality_battery/battery.py` | 1 | _bench: persist the DSv4.1 quality battery + ladder/soak tooling (battery.py, compare.py, reverdict.py, ladder_exactness.py, 1m_soak3.sh, runbook)_ |
| A | `bench/dsv41_quality_battery/compare.py` | 1 | _bench: persist the DSv4.1 quality battery + ladder/soak tooling (battery.py, compare.py, reverdict.py, ladder_exactness.py, 1m_soak3.sh, runbook)_ |
| A | `bench/dsv41_quality_battery/ladder_exactness.py` | 1 | _bench: persist the DSv4.1 quality battery + ladder/soak tooling (battery.py, compare.py, reverdict.py, ladder_exactness.py, 1m_soak3.sh, runbook)_ |
| A | `bench/dsv41_quality_battery/reverdict.py` | 1 | _bench: persist the DSv4.1 quality battery + ladder/soak tooling (battery.py, compare.py, reverdict.py, ladder_exactness.py, 1m_soak3.sh, runbook)_ |
| A | `bench/dsv41_quality_battery/runbook.md` | 1 | _bench: persist the DSv4.1 quality battery + ladder/soak tooling (battery.py, compare.py, reverdict.py, ladder_exactness.py, 1m_soak3.sh, runbook)_ |
| A | `bench/dsv4_dsml_battery.py` | 1 | _bench(dsv4): DSML tool-call fidelity battery_ |
| A | `bench/dsv4_indexer_pipelined_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/dsv4_quality_harness.py` | 4 | _bench(quality-harness): decode_tps_usage cross-check field — robust on servers that batch tokens per SSE delta (primary delta-count metric unchanged for baseline comparability)_ |
| A | `bench/dsv4_sparse_pooled_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/dsv4_topk_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/dual_model_c2_100k.py` | 2 | _fix(bench): point dual_model bench at canonical DeepSeek-V4-Flash repo_ |
| M | `bench/eval_configs/models.toml` | 3 | mentioned in: upstream-prs.md, api.md |
| A | `bench/existing_fused_prefill_bench.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| M | `bench/exo_eval.py` | 2 | mentioned in: iogpu-residency-set-abort.md |
| A | `bench/fp16_peak_bench.py` | 1 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `bench/full_moe_microbench.py` | 1 | mentioned in: fork-notes.md, skill:nop-probe-bottleneck-attribution.md |
| A | `bench/gap_analysis.awk` | 1 | _bench: MoE qmm bandwidth investigation harness + cache-busting probes_ |
| A | `bench/gather_mm_indexed_test.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/gather_optimization_route2.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/gather_qmv_microbench.py` | 1 | mentioned in: fork-notes.md, skill:distributed-bottleneck-attribution.md |
| A | `bench/gemv_bi_test.py` | 1 | _bench: land gemv_bi_test (MLX_GEMV_BATCH_INVARIANT correctness+bitexactness gate); plan doc status header_ |
| A | `bench/golden_v1_probe.py` | 2 | _bench: golden_v1_probe — add per-token timestamps + 256-token fixed-window tok/s_ |
| A | `bench/gpu_util_analyze.py` | 1 | _docs+bench: confirm GPU near-saturated (93-98%) throughout real prefill_ |
| A | `bench/gpu_util_needle_probe.py` | 1 | _docs+bench: confirm GPU near-saturated (93-98%) throughout real prefill_ |
| A | `bench/hard_eval.py` | 2 | mentioned in: skill:2026-06-03-mtp-losslessness-tiebreak-rootcause.md, skill:mtp-speculative-losslessness-debugging.md |
| A | `bench/indexed_gemm_poc.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/indexer_score_microbench.py` | 1 | mentioned in: skill:perf-exploration-workflow.md, skill:bench-tooling.md |
| A | `bench/indexer_tiled_p_bitexact.py` | 1 | _perf(dsv4): bump mlx-lm to tiled-P indexer (59a5b9a) + bit-exact bench_ |
| A | `bench/item2_opclass_breakdown.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/ldiff_cycles.py` | 1 | _bench: add ldiff_cycles harness (cycle-level MTP verify vs sequential, bitwise; incl. cache-level rollback + serving-parity fence modes)_ |
| A | `bench/lever1_smallm_headroom.py` | 1 | _bench+docs: Lever 1 NO-GO (MoE small-M kernel), Lever 3 real root-cause found (178ms/call was an arithmetic artifact)_ |
| A | `bench/lever2_seqchunk_overlap_probe.py` | 1 | _bench+docs: Lever 1 NO-GO (MoE small-M kernel), Lever 3 real root-cause found (178ms/call was an arithmetic artifact)_ |
| A | `bench/lmhead_quality_gate.py` | 1 | _perf(dsv4): ship lm_head mxfp8 by default (+6.0% decode @100K, zero quality cost)_ |
| A | `bench/lmhead_task_eval.py` | 1 | _perf(dsv4): ship lm_head mxfp8 by default (+6.0% decode @100K, zero quality cost)_ |
| A | `bench/local_repro_chunk_overlap.py` | 1 | _fix(prefill): root-cause cross-stream ordering race in chunk-overlap path_ |
| A | `bench/long_decode_probe.py` | 4 | _docs: vec-vs-loop divergence VERIFIED fixed + lossless; depth is the real decode lever_ |
| A | `bench/micro_batch_invariance.py` | 1 | _verify-batch: phase 0 micro-tests + design decision_ |
| A | `bench/minimax_cluster_ab.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_dispatch_count.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_fused_qkv_end_to_end.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_qkv_merge_probe.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_quant_concat_probe.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/minimax_rope_cache_breakdown.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `bench/moe_allsum_quant_repro.py` | 1 | _docs+bench: PHASE 0/2.1 -- quantized all_sum logic EXONERATED offline_ |
| A | `bench/moe_allsum_sharedscale_repro.py` | 2 | _fix: add missing backends field; expand sharedscale repro docstring_ |
| A | `bench/moe_gpu_capture_profile.py` | 1 | _bench: trustworthy per-stage MoE GPU cost breakdown -- fixes profiler bug_ |
| A | `bench/moe_isolation_and_qmm_gate.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/moe_m_sweep_mxfp4.py` | 1 | _bench: M-sweep for production mxfp4 MoE kernel (4-512 tokens/expert)_ |
| A | `bench/moe_mscaling_microbench.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/moe_production_class_bench.py` | 2 | _bench: correct MoE microbench to real per-rank shapes (1024/2048, not nominal)_ |
| A | `bench/moe_quant_vs_bf16_ab.py` | 2 | _bench: bind loop vars in moe A/B lambda (ruff B023)_ |
| A | `bench/moe_roofline_validation.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/moe_run_length_tile_waste.py` | 1 | _bench: quantify MoE steel-tile padding waste vs hybrid qmv_rhs dispatch upper bound_ |
| A | `bench/moe_tile_size_sweep.py` | 1 | _bench: analytic MoE tile-size sweep (bm=16/32/64) at production ragged routing_ |
| A | `bench/moe_vs_dense_qmm_isolation.py` | 2 | _bench: bind loop vars in moe A/B lambda (ruff B023)_ |
| A | `bench/mtp_ab_probe.py` | 1 | _bench: add Qwen3.6 MTP A/B probes + documented results_ |
| A | `bench/mtp_cycle_time.py` | 1 | _docs(perf): P06 Phase C closed DON'T-SHIP -- pad-to-8 never reaches qmm (limit is 12)_ |
| A | `bench/mtp_eagle_microbench.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `bench/mtp_longctx_probe.py` | 5 | _bench/mtp_longctx_probe.py: track prefill tok/s alongside decode tok/s_ |
| A | `bench/p01_switch_mlp_gputrace.py` | 1 | _docs(dspark): Phase 0 switch_mlp capture + Phase 1(a)/(b) residual chase_ |
| A | `bench/p01b_multilayer_pipelining_loss.py` | 1 | _docs(dspark): Phase 0 switch_mlp capture + Phase 1(a)/(b) residual chase_ |
| A | `bench/p13_switch_mlp_subphase_capture.py` | 1 | _docs(perf-history): close out 3 of §13's open threads (340K cliff live re-test, switch_mlp sub-phase attribution, clock-synced all_sum skew measurement)_ |
| A | `bench/p14_artifact_and_calibration.py` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `bench/p14b_decisive_recreate.py` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `bench/p14c_tile_cliff_isolated.py` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `bench/p14d_smallm_ceiling.py` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `bench/p14e_reconciliation.py` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `bench/p1p4_freeze_prompts.py` | 1 | _docs(dspark): pre-register P1-P4 forward-plan campaign — gates, launch matrix, prompt-freeze script_ |
| A | `bench/p3_attn_depth_walltime_microbench.py` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `bench/p3_attn_subspan_attribution.py` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `bench/p3_depth_anchor_probe.py` | 2 | _docs(dspark): Phase (c) MoE-at-depth + allocator residual analysis: INCONCLUSIVE_ |
| A | `bench/p3_donation_insitu_harness.py` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `bench/p4_shadow_gate_probe.py` | 1 | _feat(dsv4): P4v2 M0+M1 — gate the DSpark head load, add env-gated shadow measurement_ |
| A | `bench/p4_tp_width_shard_gemm_efficiency.py` | 1 | _docs(dsv4): P2 -- xctrace attach wedges live TP=2 prefill collective (3/3, NOT the fc954293 watchdog bug); P4 -- TP=2 width-sharding has NO skinny-GEMM penalty, sign inverted_ |
| A | `bench/pV5_probe_prompt.py` | 1 | _feat(dsv41): vision tower + image processor wiring, session-based KV reuse (stream V)_ |
| A | `bench/pV7_engine_e2e.py` | 2 | _fix(dsv41): real-model engine path -- draft tuple, session draft-window rewind, v4 image render_ |
| A | `bench/pertoken_latency_probe.py` | 3 | _docs: Section 57 -- RETRACT Section 54's causal claim; the threshold was a prompt-content artifact_ |
| A | `bench/phase0a_allsum_boundary_decompose.py` | 1 | _bench+docs: PHASE 0a -- moe.all_sum GPU->CPU boundary is real and payload-proportional_ |
| A | `bench/phase0b_collective_overlap_probe.py` | 1 | _bench+docs: Phase 0b gate PASSES -- collectives do not drain the GPU_ |
| A | `bench/phase0b_collective_overlap_probe2.py` | 1 | _bench+docs: Phase 0b gate PASSES -- collectives do not drain the GPU_ |
| A | `bench/phase0b_collective_overlap_probe3.py` | 1 | _bench+docs: Phase 0b gate PASSES -- collectives do not drain the GPU_ |
| A | `bench/phase0b_collective_overlap_probe4.py` | 1 | _bench+docs: Phase 0b gate PASSES -- collectives do not drain the GPU_ |
| A | `bench/phase0c_collective_order_determinism.py` | 1 | _bench+docs: Phase 0c -- collective matching is by EVAL order, not issue order (real hazard found)_ |
| A | `bench/phase1_batched_chunked_prefill.py` | 1 | _fix(deps): bump mlx-lm to land compress_ratio=4 chunk-boundary carry fix_ |
| A | `bench/phase1_chunked_prefill_baseline.py` | 1 | _bench: Phase 1 chunked prefill baseline -- finds real SparseCompressedAttention chunk-boundary bug_ |
| A | `bench/phase2_arm.py` | 1 | _docs+bench(perf): A/B arm-probe pitfalls (parked-session contamination; salt changes chars/tok) + corrected phase2_arm.py_ |
| A | `bench/phase2_interleaved_overlap_pipeline.py` | 1 | _bench: Phase 2 -- real interleaved-graph compute/comm overlap pipeline_ |
| A | `bench/phase3_precheck_depth_throughput.py` | 2 | _bench: derive prefill tok/s from offline tokenization, never from server-reported counts_ |
| A | `bench/phase_g_sweep.py` | 1 | _chore(bench): phase_g_sweep.py — prefill + decode sweep via streaming SSE_ |
| A | `bench/prefill_cliff_gclimit_repro_local.py` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_cliff_gclimit_repro_local_results.jsonl` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_cliff_gclimit_repro_local_v2.py` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_cliff_gclimit_repro_local_v2_results.jsonl` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_cliff_mechanism_local.py` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_cliff_mechanism_results.jsonl` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_cliff_throttle_repro_local.py` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_cliff_throttle_repro_local_results.jsonl` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `bench/prefill_step_size_allocator_ab.py` | 1 | _bench: isolate EXO_PREFILL_STEP_SIZE 2048-vs-4096 regression by stage_ |
| A | `bench/prefill_step_size_peak_and_psweep.py` | 1 | _bench: isolate EXO_PREFILL_STEP_SIZE 2048-vs-4096 regression by stage_ |
| A | `bench/prefill_step_size_stage_ab.py` | 1 | _bench: isolate EXO_PREFILL_STEP_SIZE 2048-vs-4096 regression by stage_ |
| A | `bench/qmm_bandwidth_bench.py` | 1 | _bench: MoE qmm bandwidth investigation harness + cache-busting probes_ |
| A | `bench/qmm_m192_gate.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/qmm_mbatch_bench.py` | 1 | _bench: MoE qmm bandwidth investigation harness + cache-busting probes_ |
| A | `bench/qmm_tile_sweep.py` | 1 | _bench: MoE qmm bandwidth investigation harness + cache-busting probes_ |
| A | `bench/qmv_mbatch_bench.py` | 2 | _deploy: bump mlx — affine gather_qmv_rhs + B/E<=8 gate; affine gates in bench_ |
| A | `bench/qmv_mbatch_bitexact.py` | 2 | _deploy: bump mlx — affine gather_qmv_rhs + B/E<=8 gate; affine gates in bench_ |
| A | `bench/qmv_mbatch_proto.py` | 2 | _deploy: bump mlx submodule gitlink to d4062646; proto harness grid fix_ |
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
| A | `bench/qwen36_mtp_results.md` | 2 | _docs(bench): add recurring bug-class guide for hybrid/fused-MoE bringup_ |
| A | `bench/rope_unsqueeze_smoke.py` | 1 | _docs(fork-notes): May 12 validation runs + 110K model quality cliff_ |
| A | `bench/sdpa_decomposed_bench.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sdpa_decomposed_v2_bench.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sdpa_gqa_profile.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sdpa_prefill_bench.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `bench/sdpa_prefill_fused_microbench.py` | 1 | _diag: bump mlx-lm to 82b089e (topk dump toggle) + prefill fused-SDPA microbench_ |
| A | `bench/sdpa_subtile_microbench.py` | 1 | _bench: isolated SDPA sub-tiling microbenchmark (Option A decision gate)_ |
| A | `bench/section100_chunked_vs_plain_state_diff.md` | 1 | _bench: Section 100 - chunked vs plain prefill stream state diff (code read)_ |
| A | `bench/section100_gpu_ground_truth.py` | 2 | _cleanup: fix ruff findings in the GPU ground-truth sampler (5 auto + 1 SIM117); ruff now clean_ |
| A | `bench/section100_timeout_constant_hunt.md` | 1 | _bench: static hunt for timeout constant behind flat 550ms PP decode wait (Section 100)_ |
| A | `bench/section101_tp_decode_capability_audit.md` | 1 | _bench: TP decode capability audit — PP+MTP is gated not structural, no live sharding-swap seam exists_ |
| A | `bench/section108_tp_expert_locality_analysis.md` | 1 | _bench: TP expert-locality analysis for DSv4 (section 108)_ |
| A | `bench/section109_prefill_shared_vs_pp_specific.md` | 1 | _bench: section 109 - prefill dominant cost is shared code, caps TP too_ |
| A | `bench/section110_decode_stall_last_candidate.md` | 1 | _bench: close last live candidate (MLX_JACCL_ACK_RETRANSMIT_US) — not reachable from PP decode graph; reframe investigation toward local compute/cache-state, flag TP collective ACK _ |
| A | `bench/section17_memory_headroom_check.py` | 2 | _fix(bench): detect real SSE stream termination in section17 headroom check_ |
| A | `bench/section22_chunk_drive_validate.py` | 1 | _Section 23: deploy Section 22's fix, real-hardware validation finds a hard stall_ |
| A | `bench/section27_cancel_abort_test.py` | 3 | _fix(deploy+test): rsync repo to studios instead of per-node github fetch; fix health-check prompt for a thinking model_ |
| A | `bench/section85_prefill_cancel_hardware_test.py` | 1 | _feat(api): client-supplied correlation_id to identify in-flight commands pre-first-chunk_ |
| A | `bench/section98_prefill_cancel_definition_of_done.py` | 2 | _fix(bench): correct the footprint parser in the Section 98 harness_ |
| A | `bench/soak_next12.sh` | 2 | _bench: fix soak_next12 curl E2BIG bug (payload to file); document the trap_ |
| A | `bench/soak_next13.sh` | 2 | _bench: fix soak_next13 ${RUNG} brace bug (unbound under set -u)_ |
| A | `bench/softmax_epilogue_fused.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/sparse_attn_compile_smoke.py` | 1 | _bench: microbenches for DSv4 attention path optimization_ |
| A | `bench/sparse_pooled_attn_microbench.py` | 1 | mentioned in: skill:bench-tooling.md |
| A | `bench/sparse_pooled_refactor_microbench.py` | 1 | mentioned in: fork-notes.md, skill:perf-exploration-workflow.md |
| A | `bench/spec_degen_capture.py` | 2 | mentioned in: skill:2026-05-29-mtp-degeneration-spec-trace-harness.md, skill:2026-05-29-mtp-degeneration-code-analysis-and-trace.md |
| A | `bench/spec_degen_diff.py` | 1 | mentioned in: skill:2026-05-29-mtp-degeneration-spec-trace-harness.md, skill:2026-05-29-mtp-degeneration-code-analysis-and-trace.md |
| A | `bench/specoff_battery.py` | 1 | _bench(dsv4): commit the c=2 stress battery (specoff_battery.py) from m4-1 scratch_ |
| A | `bench/stream_overlap_microbench.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/swap_to_8bit_and_rerun.sh` | 1 | _bench: --indices flag + swap-to-8bit runbook_ |
| A | `bench/switch_mlp_decode_ablation.py` | 1 | _docs(dsv4): P0 CLOSED -- switch_mlp 27.7% BW figure RETRACTED as serial-sync bench artifact; kernel actually at 74-87% of peak, no lever exists_ |
| A | `bench/tail_pmask_bitexact.py` | 1 | _deploy: bump mlx-lm (OPT-12 tail pmask) + bitexact gate_ |
| A | `bench/test_tree_mask.py` | 4 | mentioned in: skill:2026-05-19-tree-drafting-forensics.md, skill:2026-05-20-tree-drafting-phase6b-fix.md |
| A | `bench/topk_fused_verify_bitexact.py` | 1 | _perf(dsv4): default-on prefill tier-1 flags — 255->353 t/s at 100K (+38%)_ |
| A | `bench/trusted_measurement/__init__.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/__main__.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/ab_runner.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/canary.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/conftest.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/content_check.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/depth_matrix.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/fingerprint.py` | 2 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/__init__.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/decode_probe.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/jaccl_probe.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/prefill_probe.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/tests/__init__.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/tests/builders.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/tests/test_decode_probe.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/tests/test_jaccl_probe.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/tests/test_prefill_probe.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/probes/tests/test_registry_reconciliation.py` | 1 | _bench: Phase 2 (scaffolding) of the measurement harness -- registry reconciliation + jaccl/prefill/decode probe scaffolds_ |
| A | `bench/trusted_measurement/proof.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/record.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/replication.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/runtime_mode.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/__init__.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/builders.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/test_fingerprint_and_runner.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/test_liveness_canary.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/test_record_envelope.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/test_replication_gate.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/test_runtime_mode_marker.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/tests/test_token_ground_truth.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/trusted_measurement/token_truth.py` | 1 | _bench: Phase 1 of the shared measurement-discipline harness (trusted_measurement)_ |
| A | `bench/twophase_compact_staging_proto.py` | 1 | _fix: enable MlxRing (TCP) by deriving IP from RDMA interface_ |
| A | `bench/warm_fresh.py` | 1 | _docs+bench(perf): P0 verdict — fresh floor is steady-state (no warmup share); warm-fresh probe persisted_ |
| A | `bench_data/shadow_gate_20260823/dspark_shadow_relaunch1.jsonl` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod3_100k.json` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod3_100k.txt` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod3_short.json` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod3_short.txt` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod_10k.json` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod_10k.txt` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod_short.json` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/prod_short.txt` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/shadow_100k.json` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/shadow_100k.txt` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/shadow_short.json` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/shadow_short.txt` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/shadow_short_rerun.json` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `bench_data/shadow_gate_20260823/shadow_short_rerun.txt` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |

### Build/meta  (2 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `pyproject.toml` | 34 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `uv.lock` | 383 | mentioned in: upstream-prs.md, fork-notes.md |

### CORE exo source  (268 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `rust/networking/src/discovery.rs` | 1 | sync-notes-exo: combined - kept our v4 socket/static unicast peers; adopted #2346 StallGuard, #2320 single-attempt reply, #2419, #2327 |
| M | `rust/networking/src/lib.rs` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/api/adapters/chat_completions.py` | 9 | mentioned in: docs/DSV4_VISION_PORT_PHASE5_PROCEDURE.md, docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-reasoning-budget-limiter-2026-10-13.md |
| M | `src/exo/api/main.py` | 17 | sync-notes-exo: combined - kept our metrics/JIT/phase-marks; adopted #2322/#2369, #2330/#2356, #2362, #2415  (prior: api.md) |
| A | `src/exo/api/phase_marks.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/api/tests/test_api_error_handling.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/api/tests/test_cancel_command.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/api/tests/test_chat_completions_image_blocks.py` | 1 | mentioned in: docs/PERFORMANCE_HISTORY.md |
| M | `src/exo/api/tests/test_chat_completions_stream.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/api/tests/test_correlation_id.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/api/tests/test_image_error_chunk_routing.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/api/tests/test_interrupted_generation_streams.py` | 0 | rationale documented in the Fork-rationales section |
| A | `src/exo/api/tests/test_jit_request_path.py` | 3 | mentioned in: docs/jit-model-lifecycle-handoff-2026-06-30.md |
| A | `src/exo/api/tests/test_reasoning_effort_clamp.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/api/tests/test_seed_resolution.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/api/tests/test_task_failed_stream_error.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| M | `src/exo/api/types/__init__.py` | 3 | api.md |
| M | `src/exo/api/types/api.py` | 12 | api.md |
| M | `src/exo/api/types/openai_responses.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/download/coordinator.py` | 5 | sync-notes-exo: combined - our repo_url passthrough + upstream #2349 ValueError guard  (prior: skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P)) |
| M | `src/exo/download/download_utils.py` | 8 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| A | `src/exo/download/file_server.py` | 5 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/impl_shard_downloader.py` | 2 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/shard_downloader.py` | 2 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/tests/test_cancel_download.py` | 1 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/tests/test_re_download.py` | 1 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/download/tests/test_rescan_announcements.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/main.py` | 3 | mentioned in: upstream-prs.md, skill:upstream-integration-and-deploy.md |
| M | `src/exo/master/main.py` | 8 | sync-notes-exo: combined - kept _index_apply_broadcast + _state_lock + JIT reaper; adopted #2322/#2369/#2361/#2362/#2363/#2405; NODE_SILENCE_TIMEOUT kept 30s REVIEW  (prior: mentioned in: upstream-prs.md, skill:upstream-integration-and-deploy.md) |
| M | `src/exo/master/placement.py` | 3 | sync-notes-exo: combined - InsufficientMemoryError (ours, JIT) carrying upstream _not_enough_memory message  (prior: mentioned in: api.md, fork-notes.md) |
| M | `src/exo/master/placement_utils.py` | 10 | _feat: add EXO_PP_LAYER_SPLIT for pipeline stage rebalancing_ |
| M | `src/exo/master/tests/test_event_log_requests.py` | 0 | rationale documented in the Fork-rationales section |
| A | `src/exo/master/tests/test_jit_lifecycle.py` | 3 | mentioned in: docs/jit-model-lifecycle-handoff-2026-06-30.md |
| M | `src/exo/master/tests/test_master.py` | 1 | sync-notes-exo: upstream snapshot_sender; dropped ev_send mock router (no event_sender in our Master)  (prior: _fix: index master events synchronously to enable multi-instance load balancing_) |
| M | `src/exo/master/tests/test_master_change_tasks.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/master/tests/test_master_takeover.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/master/tests/test_placement_utils.py` | 4 | rationale documented in the Fork-rationales section |
| A | `src/exo/master/tests/test_routing_concurrency.py` | 3 | _test: add concurrent routing test + bench harness for sibling contention_ |
| M | `src/exo/master/tests/test_silent_nodes.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/master/tests/test_unplaceable_requests.py` | 0 | rationale documented in the Fork-rationales section |
| A | `src/exo/metrics/__init__.py` | 3 | metrics.md |
| A | `src/exo/metrics/metrics.py` | 5 | metrics.md |
| A | `src/exo/metrics/tests/__init__.py` | 1 | metrics.md |
| A | `src/exo/metrics/tests/test_metrics.py` | 1 | metrics.md |
| M | `src/exo/shared/apply.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/shared/constants.py` | 7 | mentioned in: fork-notes.md, minimax-rdma-moe-validation-2026-04-24.md |
| M | `src/exo/shared/models/model_cards.py` | 4 | mentioned in: docs/DSV4_VISION_PORT_PHASE5_PROCEDURE.md, docs/DSV4_VISION_PORT_PLAN_PHASE34.md, docs/PERFORMANCE_HISTORY.md |
| A | `src/exo/shared/tests/test_apply/test_apply_task_created_terminal_guard.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/shared/tests/test_clamp_reasoning_effort.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/shared/tests/test_event_serialization.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/shared/tests/test_submodule_lock_pin_alignment.py` | 2 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/dsv41-indexer-qsplit-buildnote-2026-10-11.md, docs/dsv41-next27-ringfix-construct-2026-10-14.md |
| M | `src/exo/shared/types/chunks.py` | 2 | rationale documented in the Fork-rationales section |
| M | `src/exo/shared/types/commands.py` | 2 | mentioned in: skill:session-workflow-rules.md, skill:notifying-user-via-discord-gateway-cron.md |
| M | `src/exo/shared/types/tasks.py` | 1 | mentioned in: upstream-prs.md, deepseek-v4-mtp-performance.md |
| M | `src/exo/shared/types/text_generation.py` | 8 | _feat(sampling): card tier between instance and cluster-env defaults_ |
| M | `src/exo/shared/types/thunderbolt.py` | 1 | mentioned in: docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md |
| M | `src/exo/shared/types/worker/downloads.py` | 3 | skill:2026-06-04-stale-uvlock-and-upstream-merge.md (zenoh) / commit history (P2P) |
| M | `src/exo/shared/types/worker/instances.py` | 7 | mentioned in: api.md, fork-notes.md |
| M | `src/exo/shared/types/worker/runner_response.py` | 3 | mentioned in: docs/dsv41-reasoning-budget-limiter-2026-10-13.md |
| M | `src/exo/utils/info_gatherer/info_gatherer.py` | 2 | mentioned in: docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md |
| M | `src/exo/utils/info_gatherer/tests/test_tb_parsing.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/utils/log_format.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/engines/base.py` | 3 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/engines/mlx/auto_parallel.py` | 53 | mentioned in: fork-notes.md, minimax-fused-attention-prompt.md |
| M | `src/exo/worker/engines/mlx/builder.py` | 5 | sync-notes-exo: EXO_SINGLE_GPU_STREAM=0 kill switch around use_generation_stream_by_default  (prior: mentioned in: skill:2026-05-19-tree-drafting-forensics.md, skill:2026-05-25-pool-defer-and-compressor-attribution.md) |
| M | `src/exo/worker/engines/mlx/cache.py` | 48 | sync-notes-exo: ours (radix trie). #2368 SKIPPED-covered (trie has max_sessions/max_bytes caps via builder.py:158)  (prior: kv-cache-architecture.md) |
| M | `src/exo/worker/engines/mlx/constants.py` | 11 | mentioned in: fork-notes.md, minimax-rdma-moe-validation-2026-04-24.md |
| A | `src/exo/worker/engines/mlx/deepseek_v4_vision.py` | 1 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/deepseek-v41-exl3-plan.md, docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md |
| M | `src/exo/worker/engines/mlx/disaggregated/adapter.py` | 4 | mentioned in: minimax-quantized-sdpa-design.md, skill:session-workflow-rules.md |
| M | `src/exo/worker/engines/mlx/disaggregated/client.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/engines/mlx/disaggregated/serve.py` | 2 | mentioned in: docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-spec-rollback-ring-fix-2026-10-14.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/disaggregated/tests/test_composite_cache_wire.py` | 3 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/engines/mlx/disaggregated/tests/test_mlx_adapter.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/__init__.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/agreement.py` | 2 | mentioned in: docs/dsv41-restart-g1-buildnote-2026-10-10.md, docs/dsv41-restart-prelive-review-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/builder.py` | 5 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/coord.py` | 4 | mentioned in: docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-native-experts-fit-scope-2026-10-11.md, docs/dsv41-restart-g1-buildnote-2026-10-10.md |
| A | `src/exo/worker/engines/mlx/dsv41/degeneration.py` | 6 | mentioned in: docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-reasoning-budget-limiter-2026-10-13.md |
| A | `src/exo/worker/engines/mlx/dsv41/dispatch.py` | 1 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md |
| A | `src/exo/worker/engines/mlx/dsv41/dsml.py` | 4 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/engine.py` | 34 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/dsv41-big-append-census-2026-10-13.md, docs/dsv41-degeneration-guards-2026-10-13.md |
| A | `src/exo/worker/engines/mlx/dsv41/engram.py` | 2 | mentioned in: docs/deepseek-v41-exl3-plan.md |
| A | `src/exo/worker/engines/mlx/dsv41/errors.py` | 3 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/fingerprint.py` | 8 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/load.py` | 3 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/README.md, docs/dsv41-scaling-alternatives-brief-2026-10-15.md |
| A | `src/exo/worker/engines/mlx/dsv41/output.py` | 4 | mentioned in: docs/dsv41-reasoning-budget-limiter-2026-10-13.md |
| A | `src/exo/worker/engines/mlx/dsv41/park.py` | 7 | mentioned in: docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-next25-guards-construct-2026-10-13.md, docs/dsv41-restart-g1-buildnote-2026-10-10.md |
| A | `src/exo/worker/engines/mlx/dsv41/reasoning_budget.py` | 2 | mentioned in: docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-reasoning-budget-limiter-2026-10-13.md, docs/dsv41-spec-rollback-ring-fix-2026-10-14.md |
| A | `src/exo/worker/engines/mlx/dsv41/rounds.py` | 13 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-reasoning-budget-limiter-2026-10-13.md |
| A | `src/exo/worker/engines/mlx/dsv41/session.py` | 22 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md, docs/dsv41-big-append-census-2026-10-13.md |
| A | `src/exo/worker/engines/mlx/dsv41/shutdown.py` | 2 | mentioned in: docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-restart-g3-buildnote-2026-10-10.md, docs/dsv41-restart-prelive-review-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/__init__.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/_stub_harness.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/checkpoint_sentinels.json` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/conftest.py` | 6 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_degeneration.py` | 4 | mentioned in: docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-spec-rollback-ring-fix-2026-10-14.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_dispatch.py` | 6 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_dsml.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_engine.py` | 18 | mentioned in: docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-reasoning-budget-limiter-2026-10-13.md, docs/dsv41-restart-deploy-construct-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_fingerprint.py` | 1 | mentioned in: docs/dsv41-restart-deploy-construct-2026-10-11.md, docs/dsv41-restart-g2g5-buildnote-2026-10-10.md, docs/dsv41-restart-prelive-review-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_load_warmup_liveness.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_output.py` | 3 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park.py` | 2 | mentioned in: docs/dsv41-restart-g2g5-buildnote-2026-10-10.md, docs/dsv41-restart-prelive-review-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g1.py` | 2 | mentioned in: docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-restart-deploy-construct-2026-10-11.md, docs/dsv41-restart-g1-buildnote-2026-10-10.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g2g5.py` | 1 | mentioned in: docs/dsv41-restart-g2g5-buildnote-2026-10-10.md, docs/dsv41-restart-prelive-review-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g3.py` | 2 | mentioned in: docs/dsv41-restart-g3-buildnote-2026-10-10.md, docs/dsv41-restart-prelive-review-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g4.py` | 3 | mentioned in: docs/dsv41-restart-g4-buildnote-2026-10-10.md, docs/dsv41-restart-prelive-review-2026-10-11.md, docs/dsv41-restart-rankfix-buildnote-2026-10-11.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_prefill_liveness.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_reasoning_budget.py` | 2 | mentioned in: docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-reasoning-budget-limiter-2026-10-13.md, docs/dsv41-spec-rollback-ring-fix-2026-10-14.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_reuse_ladder.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_sentinels.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_session.py` | 12 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/dsv41-next27-ringfix-construct-2026-10-14.md, docs/dsv41-spec-rollback-ring-fix-2026-10-14.md |
| A | `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_thinking_load.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/thinking.py` | 3 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/dsv41/vision.py` | 6 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/engines/mlx/generator/batch_generate.py` | 159 | sync-notes-exo: ours. #2365 finish-via-max_tokens cancel DEFERRED (MTP generator subclasses; max_tokens=0 unverified)  (prior: mentioned in: skill:mtp-bistability-2026-05-17-investigation.md, skill:2026-05-21-c2-100k-mtp-recipe.md) |
| M | `src/exo/worker/engines/mlx/generator/generate.py` | 93 | sync-notes-exo: ours + adopted #2393 mx.synchronize(generation_stream) before prefill barrier; local generation_stream alias covers #2391  (prior: mentioned in: upstream-prs.md, api.md) |
| M | `src/exo/worker/engines/mlx/generator/remote_prefill.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| M | `src/exo/worker/engines/mlx/patches/__init__.py` | 7 | sync-notes-exo: ours; reuse_detokenizer NOT wired (fork mlx-lm covers #2379)  (prior: mentioned in: fork-notes.md, kv-cache-architecture.md) |
| A | `src/exo/worker/engines/mlx/patches/minimax/__init__.py` | 2 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `src/exo/worker/engines/mlx/patches/minimax/fused_qkv.py` | 4 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `src/exo/worker/engines/mlx/patches/minimax/tests/__init__.py` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `src/exo/worker/engines/mlx/patches/minimax/tests/test_fused_qkv.py` | 3 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| M | `src/exo/worker/engines/mlx/patches/opt_batch_gen.py` | 14 | sync-notes-exo: auto-merged upstream #2371 + #2393 prompt-sync  (prior: mentioned in: fork-notes.md) |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5/__init__.py` | 1 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5/custom_qmv_loop_over_b.py` | 1 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5/lpb_patch.py` | 3 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/__init__.py` | 1 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/apply.py` | 5 | mentioned in: upstream-prs.md, fork-notes.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_fused_gqa_attention.py` | 3 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/batched_moe.py` | 6 | mentioned in: minimax-decode-optimization.md, skill:verify-cost-decomposition-2026-05-19.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/common.py` | 9 | mentioned in: kv-cache-architecture.md, turboquant-integration.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/decoder.py` | 4 | mentioned in: minimax-fused-attention-prompt.md, minimax-decode-optimization.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/fused_gdn_attention.py` | 4 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/__init__.py` | 1 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_fused_gdn_projections_8bit.py` | 3 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_fused_gqa_projections_8bit.py` | 3 | mentioned in: minimax-decode-optimization.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_merged_down_proj_8bit.py` | 3 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_moe_epilogue.py` | 3 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_oproj_gate_gemv_8bit.py` | 3 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/batched_softmax_topk_swiglu_8bit.py` | 3 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_gdn_projections_8bit.py` | 3 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_qk_rmsnorm.py` | 3 | mentioned in: minimax-fused-attention-prompt.md, minimax-decode-optimization.md |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/fused_rms_norm_gated.py` | 3 | _perf: switch compute dtype from bf16 to fp16 for ~7% faster quantized_matmul_ |
| A | `src/exo/worker/engines/mlx/patches/qwen3_5_moe/kernels/gdn_step_precomputed.py` | 1 | _feat: MTP speculative decoding + kernel patches for Qwen3.5_ |
| A | `src/exo/worker/engines/mlx/phase_marks.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/pp_batched_cache_router.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_batched_correctness.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_batched_decode_adapter.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_batched_decode_driver.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_batched_decode_eligibility.py` | 2 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_batched_decode_glue.py` | 24 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/handoff-2026-08-08-section22.md, docs/handoff-2026-08-08-section23.md |
| A | `src/exo/worker/engines/mlx/pp_batched_decode_layers.py` | 7 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_batched_decode_runtime.py` | 6 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_cancel.py` | 3 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_metaframe.py` | 8 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_prefill_session.py` | 3 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/pp_scheduler_protocol.py` | 10 | mentioned in: docs/DSV4_VISION_PORT_PLAN_PHASE34.md, docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/handoff-2026-08-08-section22.md |
| A | `src/exo/worker/engines/mlx/pp_scheduler_wire.py` | 7 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/handoff-2026-08-08-section22.md, docs/handoff-2026-08-09-section39.md |
| A | `src/exo/worker/engines/mlx/pp_speculation.py` | 94 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/pp_speculation_spec_tag.py` | 2 | mentioned in: docs/dspark-mtp-ab-preregister-2026-08-25.md |
| A | `src/exo/worker/engines/mlx/prefill_chunking.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/sampling.py` | 5 | mentioned in: upstream-prs.md, api.md |
| A | `src/exo/worker/engines/mlx/speculative/__init__.py` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` | 191 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/mtp_batch_generator.py` | 13 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/mtp_module.py` | 48 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/speculative_cache.py` | 3 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/speculative/speculative_gdn_kernel.py` | 3 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `src/exo/worker/engines/mlx/tests/_pp_admission_race_subprocess_worker.py` | 6 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md |
| A | `src/exo/worker/engines/mlx/tests/_pp_chunk_drive_subprocess_worker.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/_pp_exchange_prefill_peer_layer_count_subprocess_worker.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/_pp_glue_subprocess_worker.py` | 5 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md |
| A | `src/exo/worker/engines/mlx/tests/_pp_subprocess_worker.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_batch_generate_batched_decode_flag_off_smoke.py` | 2 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_batch_generate_chunked_prefill_live_wiring.py` | 3 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_batch_generate_rank1_batched_decode_dispatch.py` | 2 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md |
| A | `src/exo/worker/engines/mlx/tests/test_batch_pooling_cache_overlap.py` | 2 | mentioned in: docs/PERFORMANCE_HISTORY.md |
| A | `src/exo/worker/engines/mlx/tests/test_batched_decode_external_cancel.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_dspark_load_guard_quant_keys.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_dspark_tp_shard_gate.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_dsv4_mtp_phase_timer_unit_metadata.py` | 2 | mentioned in: docs/PERFORMANCE_HISTORY.md |
| A | `src/exo/worker/engines/mlx/tests/test_dsv4_vision_end_to_end_offline.py` | 3 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_generation_tps_baseline.py` | 3 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_has_pipeline_communication_layer.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_hc_expand_kernel.py` | 2 | mentioned in: docs/PERFORMANCE_HISTORY.md |
| A | `src/exo/worker/engines/mlx/tests/test_patch_embed_tokens_bounds_check.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pipeline_prefix_hit_agreement.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_admission_race_subprocess.py` | 1 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_cache_router.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_correctness_harness.py` | 1 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_adapter.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_correctness.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_driver.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_driver_full_stack.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_eligibility.py` | 3 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_glue_chunk_drive.py` | 6 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_glue_subprocess.py` | 2 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_layers_install.py` | 4 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_over_real_wire.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_runtime.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_batched_decode_subprocess.py` | 1 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_chunk_drive_subprocess.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_exchange_prefill_peer_layer_count_subprocess.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_metaframe.py` | 7 | mentioned in: docs/batched-decode-n2-admission-handoff-2026-08-05.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_metaframe_batch_axis.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_pipeline_parallel_prefill_session_integration.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_prefill_advance_budget_parity.py` | 1 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_prefill_session.py` | 2 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_scheduler_protocol.py` | 3 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_scheduler_wire.py` | 4 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_spec_gen_by_uid.py` | 3 | mentioned in: docs/dspark-draft-epilogue-fusion-2026-08-27.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_speculation_cache_snapshot.py` | 5 | mentioned in: docs/dspark-draft-epilogue-fusion-2026-08-27.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_pp_speculation_layer_reset.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_pp_speculation_spec_tag.py` | 2 | mentioned in: docs/dspark-mtp-ab-preregister-2026-08-25.md |
| A | `src/exo/worker/engines/mlx/tests/test_prefill_batched.py` | 2 | prefill-optimization.md |
| A | `src/exo/worker/engines/mlx/tests/test_prefill_chunking_cache_offset_wiring.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_prefill_chunking_image_spans.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_prefill_interruptible_start_gate_safety.py` | 3 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_prefill_residual_gap_wiring.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_prefix_cache_image_poisoning.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/tests/test_reasoning_budget_limiter.py` | 4 | mentioned in: docs/dsv41-next26-limiter-construct-2026-10-14.md, docs/dsv41-reasoning-budget-limiter-2026-10-13.md |
| A | `src/exo/worker/engines/mlx/tests/test_reset_after_reconnect_recreates_glue.py` | 2 | mentioned in: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| A | `src/exo/worker/engines/mlx/tests/test_strict_false_load_guard.py` | 2 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/benchmarks/phase6-strict-false-guard-2026-09-28/README.md |
| A | `src/exo/worker/engines/mlx/tests/test_vision_dropped_images_loud.py` | 1 | mentioned in: docs/PERFORMANCE_HISTORY.md |
| A | `src/exo/worker/engines/mlx/trace.py` | 3 | mentioned in: upstream-prs.md, fork-notes.md |
| A | `src/exo/worker/engines/mlx/turboquant_cache.py` | 6 | turboquant-integration.md / kv-cache-architecture.md |
| M | `src/exo/worker/engines/mlx/types.py` | 2 | mentioned in: api.md, kv-cache-architecture.md |
| M | `src/exo/worker/engines/mlx/utils_mlx.py` | 52 | sync-notes-exo: ours + upstream use_generation_stream_by_default() (#2391)  (prior: mentioned in: skill:upstream-sync-triage.md) |
| M | `src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py` | 5 | mentioned in: docs/DSV4_VISION_PORT_PHASE5_PROCEDURE.md, docs/DSV4_VISION_PORT_PLAN_PHASE34.md, docs/PERFORMANCE_HISTORY.md |
| A | `src/exo/worker/engines/mlx/vendor/deepseek_v4_image_processor.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/engines/mlx/vendor/dsml_encoding.py` | 5 | mentioned in: docs/PERFORMANCE_HISTORY.md |
| A | `src/exo/worker/engines/mlx/vendor/tests/test_deepseek_v4_image_processor.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/engines/mlx/vendor/tests/test_sentinelless_recovery.py` | 2 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/engines/mlx/vision.py` | 3 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/main.py` | 10 | sync-notes-exo: combined - our I16 event-wake + phase marks kept; adopted #2369/#2322 (_gathered, StateSnapshot), #2336, #2389, #2358  (prior: mentioned in: upstream-prs.md, skill:upstream-integration-and-deploy.md) |
| A | `src/exo/worker/phase_marks.py` | 3 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/plan.py` | 6 | sync-notes-exo: combined imports (ours EXO_FILE_SERVER_PORT/NodeNetworkInfo; dropped now-unused InputImageChunk/CommandId after #2358)  (prior: _fix: thread node_network through plan() to _model_needs_download()_) |
| M | `src/exo/worker/runner/bootstrap.py` | 14 | sync-notes-exo: combined - our QoS/coredump/profiler/release kept; adopted #2377 use_fast_synch (accepts "1"/"0")  (prior: mentioned in: upstream-prs.md, skill:2026-05-22-bootstrap-barrier-and-ce5c64fd-wedge.md) |
| M | `src/exo/worker/runner/diagnostics.py` | 2 | sync-notes-exo: same fix both sides (#2366) -> upstream Annotated form + our comment; removed duplicate pydantic import |
| M | `src/exo/worker/runner/llm_inference/batch_generator.py` | 43 | sync-notes-exo: auto-merged #2365 part 2 (drop cancelled queued tasks) - kept  (prior: mentioned in: upstream-prs.md, skill:mlx-upstream-port-validation.md) |
| M | `src/exo/worker/runner/llm_inference/model_output_parsers.py` | 30 | sync-notes-exo: ours (typed clean-fail error w/ tool_call_parse_failure_kind). #2394 SKIPPED (behavioral conflict) REVIEW  (prior: thinking-parser-fused-delimiter-fix.md) |
| A | `src/exo/worker/runner/llm_inference/tests/test_batch_generator_step_cancellation_latency.py` | 4 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/runner/llm_inference/tests/test_step_prefill_cancelled_guard.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/runner/runner.py` | 22 | mentioned in: upstream-prs.md, thinking-parser-fused-delimiter-fix.md |
| M | `src/exo/worker/runner/supervisor.py` | 18 | sync-notes-exo: combined - our hang watchdog/_check_hang/_check_stuck_init kept; adopted #2367 _lost_its_peers + #2370 _stuck (gated to HANG_TIMEOUT_SECONDS<=0, REVIEW) + _last_heard; #2380 in_progress skip |
| A | `src/exo/worker/runner/tests/__init__.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/runner/tests/fixtures/synthetic_spin.txt` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/runner/tests/test_supervisor_hang_liveness_probe.py` | 1 | mentioned in: docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md |
| A | `src/exo/worker/runner/tests/test_supervisor_hang_stack_guard.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/runner/tests/test_supervisor_hang_tracing.py` | 2 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_fast_synch.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_mlx/conftest.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py` | 16 | kv-cache-architecture.md |
| M | `src/exo/worker/tests/unittests/test_mlx/test_prefix_cache_limits.py` | 0 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/tests/unittests/test_plan/test_plan_step_event_wake.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/tests/unittests/test_plan/test_plan_step_lifecycle_dispatch.py` | 1 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/tests/unittests/test_plan/test_plan_step_lost_wakeup_fuzz.py` | 1 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_plan/test_worker_task_announcements.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_runner/test_batch_cancellation.py` | 0 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/tests/unittests/test_runner/test_concurrency_admission_gate.py` | 3 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_runner/test_dsml_e2e.py` | 19 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_runner/test_event_ordering.py` | 2 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md |
| M | `src/exo/worker/tests/unittests/test_runner/test_finish_reason_sse.py` | 2 | mentioned in: thinking-parser-fused-delimiter-fix.md, skill:2026-06-03-fused-think-delimiter-parser-bug.md |
| A | `src/exo/worker/tests/unittests/test_runner/test_loop_detection.py` | 7 | mentioned in: docs/PERFORMANCE_HISTORY.md, docs/dsv41-degeneration-guards-2026-10-13.md, docs/dsv41-next25-guards-construct-2026-10-13.md |
| M | `src/exo/worker/tests/unittests/test_runner/test_parse_tool_calls.py` | 0 | rationale documented in the Fork-rationales section |
| M | `src/exo/worker/tests/unittests/test_runner/test_runner_supervisor.py` | 0 | rationale documented in the Fork-rationales section |
| A | `src/exo/worker/tests/unittests/test_runner/test_runner_supervisor_hang_probe.py` | 3 | mentioned in: docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md |

### Dashboard (UI)  (2 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `dashboard/src/lib/stores/app.svelte.ts` | 1 | _fix: P2P peer discovery in API start_download + dashboard pause button_ |
| M | `dashboard/src/routes/+page.svelte` | 1 | _default DSv4 to the validated Pipeline+DSpark launch config_ |

### Documentation (the docs themselves)  (619 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `docs/DSV4_VISION_PORT_PHASE5_PROCEDURE.md` | 3 | _docs(dsv4-vision): Phase 5 executed — vision verified on hardware_ |
| A | `docs/DSV4_VISION_PORT_PLAN.md` | 1 | _docs: DSv4-Flash-Vision-Exp exo port plan (phases 0-2 authorized scope)_ |
| A | `docs/DSV4_VISION_PORT_PLAN_PHASE34.md` | 1 | _docs: DSv4 vision port plan, phases 3-4 (authorized scope)_ |
| A | `docs/OVERNIGHT-CLOSEOUT-2026-09-23.md` | 1 | _docs(overnight): close-out -- shipped, verified, open, and my own correction ledger_ |
| A | `docs/OVERNIGHT-CLOSEOUT-FINAL-2026-09-23.md` | 1 | _docs(overnight): FINAL close-out_ |
| A | `docs/PERFORMANCE_HISTORY.md` | 240 | _docs(dsv41): scaling-alternatives brief — SS5 outcome: PP scoping Gate 0 KILL (decode inequality 3-10x, anchor void, weighted +5.98%) + main PH line_ |
| A | `docs/ROOTCAUSE-bytcap-undercount-cachelist-2026-09-23.md` | 1 | _fix(kv-cache): _cache_object_nbytes must use .nbytes -- CacheList layers reported 0_ |
| A | `docs/acceptance-cliff-indexer-topk-hypothesis-2026-09-23.md` | 2 | _docs(overnight): SOURCE-VERIFIED the indexer threshold -- k = min(index_topk, pooled.shape[1])_ |
| A | `docs/allgather-lever-negative-result-2026-08-21.md` | 1 | _docs(jaccl): negative result -- subgroup all_gather still faults post-fix_ |
| A | `docs/allsum-ablation-unsafe-2026-08-21.md` | 1 | _docs(dsv4): all_sum NOP ablation destabilizes cluster, required full relaunch_ |
| A | `docs/allsum-straggler-aggregate-impact-2026-08-22.md` | 1 | _docs(dsv4): P6 -- allsum rank0 straggler aggregate impact quantified (~1.7% decode upper bound, exceeds 0.5% gate but bounded; timeboxed, no action)_ |
| A | `docs/allsum-sync-span-artifact-arithmetic-check-2026-08-22.md` | 1 | _docs(dsv4): sync-span 21.4%/14.4% moe.all_sum figures were artifacts_ |
| A | `docs/async-fence-cache-owner-dead-code-root-cause-2026-08-22.md` | 1 | _docs(dsv4): ROOT CAUSE FOUND -- async fence structurally dead under TP_ |
| A | `docs/async-fence-fix-validated-2026-08-22.md` | 1 | _docs(dsv4): async-fence fix VALIDATED -- real +58-67% decode throughput_ |
| A | `docs/b2-mtp-resolution-2026-06-24.md` | 2 | _docs: post-seq-split tuning sweep - step=128 and max_mb=200 settled_ |
| A | `docs/b2-quality-handoff-2026-06-24.md` | 3 | _docs: B=2 + MTP resolution — seq-split & bootstrap fixes, verified sweep_ |
| A | `docs/baseline-locked-decode-fence-fix-20260822.md` | 1 | _docs(dsv4): lock in new baseline -- real depth re-benchmark + MTP/DSpark re-confirm_ |
| A | `docs/batched-decode-n2-admission-handoff-2026-08-05.md` | 11 | _docs(pp-batched): add Phase 2/3/4 starting-point section for next session_ |
| A | `docs/benchmarks/big-append-census-2026-10-13/census.csv` | 1 | _docs(dsv41): big-append Stage 0 census — felt-latency case CLOSED (weak): target −75.5% under exo-ideal reuse; 79% of genuine rows are cold rebuilds; S=1.07% < 8%; plus the conditi_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/README.md` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r1.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r10.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r10.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r2.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r2.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r3.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r3.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r4.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r4.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r5.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r5.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r6.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r6.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r7.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r7.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r8.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r8.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r9.json` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase0-v4flash-2k-2026-09-26/raw/baseline_2k_r9.log` | 1 | _docs: baseline table for phase 0 (V4-Flash 2K reference)_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/README.md` | 2 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/raw/phase1-run.log` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/raw/phase1b-results.json` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/raw/phase1b-run.log` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/raw/phase1c-results.json` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/raw/phase1c-run.log` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/scripts/phase1_exl3_validate.py` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/scripts/phase1b_exl3_mtp_budget.py` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase1-exl3-format-validation-2026-09-27/scripts/phase1c_localize.py` | 1 | _docs: phase 1 EXL3 format validation for DSv4.1-Flash 2.9bpw_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/README.md` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p24-iostat-summary.txt` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_gate_summary2.txt` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p33-clamp-rms.txt` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p33-full.log` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p34-clamp-l39-deep.txt` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p38-head-nll.txt` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/scripts/p30_exl3_trace.py` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/scripts/p31_trace_analyze.py` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/scripts/p33_collapse.py` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase10-planb-gate-2026-09-28/scripts/patch_exl3_clamp.py` | 1 | _docs(phase10): Plan B day-2 gate measured -- full-model trace gives 26.9 tok/s I/O ceiling (1.08x at the 25 tok/s bar), MTP multiplies streaming ~1.7-2x/token; clamped-SwiGLU bug f_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/README.md` | 2 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p24-curves-analysis.md` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-curves-analysis.md` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-driver-macstudio-m4-1.log` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-driver-macstudio-m4-2.log` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-env-macstudio-m4-1.log` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-env-macstudio-m4-2.log` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-iostat-macstudio-m4-1.txt` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-iostat-macstudio-m4-2.txt` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-soak-macstudio-m4-1.log` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p25-soak-macstudio-m4-2.log` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p41-stage1-fstar.md` | 2 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p44-serving-geometry.out` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p44-serving-rerun.out` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/raw/p45-serving-geometry.out` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/scripts/p23_ssd_soak2.py` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/scripts/p25_driver.sh` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/scripts/p25_env.sh` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/scripts/p25_launch.sh` | 1 | _docs(phase11): Stage-1 f* + complete soak characterization -- 25 tok/s bar unreachable on EXL3-resident (mid-teens expected); p25 full run: transient mid-run device dip on BOTH nod_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/scripts/p44_full_layer.py` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/scripts/p45_chain.sh` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase11-stage1-fstar-2026-09-28/scripts/p45_mxfp4_geometry.py` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md` | 2 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/raw/p42-split-equiv.out` | 1 | _docs(phase12): EXL3 kernels vendored into mlx-lm + TP split convention proven_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/raw/p43-loader-gate.out` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/raw/vendor-equiv.out` | 1 | _docs(phase12): EXL3 kernels vendored into mlx-lm + TP split convention proven_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/raw/vendor-manifest.txt` | 1 | _docs(phase12): EXL3 kernels vendored into mlx-lm + TP split convention proven_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/scripts/exl3_vendor_equiv.py` | 1 | _docs(phase12): EXL3 kernels vendored into mlx-lm + TP split convention proven_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/scripts/p42_tp_split_equiv.py` | 1 | _docs(phase12): EXL3 kernels vendored into mlx-lm + TP split convention proven_ |
| A | `docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/scripts/p43_loader_gate.py` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase13-dsv41-model-file-2026-09-28/README.md` | 1 | _docs(phase13): deepseek_v41 model package + EXL3 builder matches the reference forward_ |
| A | `docs/benchmarks/phase13-dsv41-model-file-2026-09-28/raw/p46-chained.log` | 1 | _docs(phase13): deepseek_v41 model package + EXL3 builder matches the reference forward_ |
| A | `docs/benchmarks/phase13-dsv41-model-file-2026-09-28/raw/p46-isolated.log` | 1 | _docs(phase13): deepseek_v41 model package + EXL3 builder matches the reference forward_ |
| A | `docs/benchmarks/phase13-dsv41-model-file-2026-09-28/scripts/p46_model_gate.py` | 1 | _docs(phase13): deepseek_v41 model package + EXL3 builder matches the reference forward_ |
| A | `docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/README.md` | 1 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/raw/p47-r0-profile.log` | 1 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/raw/p47-r0-try1-gputimeout.log` | 1 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/raw/p47-r0-try2.log` | 1 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/raw/p47-r0-try3.log` | 1 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/scripts/p47_launch.sh` | 1 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/benchmarks/phase14-dsv41-first-tp2-decode-2026-09-28/scripts/p47_tp2_decode.py` | 1 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/README.md` | 1 | _docs(phase15): DSv4.1 body speedup 9.3 -> 16.3 tok/s plain decode (UNSIGNED, not pushed)_ |
| A | `docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/raw/p47-r0-16tok.log` | 1 | _docs(phase15): DSv4.1 body speedup 9.3 -> 16.3 tok/s plain decode (UNSIGNED, not pushed)_ |
| A | `docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/scripts/p48_bench.py` | 1 | _docs(phase15): DSv4.1 body speedup 9.3 -> 16.3 tok/s plain decode (UNSIGNED, not pushed)_ |
| A | `docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/scripts/p51_dense.py` | 1 | _docs(phase15): DSv4.1 body speedup 9.3 -> 16.3 tok/s plain decode (UNSIGNED, not pushed)_ |
| A | `docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/scripts/p52_dense_each.py` | 1 | _docs(phase15): DSv4.1 body speedup 9.3 -> 16.3 tok/s plain decode (UNSIGNED, not pushed)_ |
| A | `docs/benchmarks/phase15-dsv41-body-speedup-2026-09-28/scripts/p55_tp_sim.py` | 1 | _docs(phase15): DSv4.1 body speedup 9.3 -> 16.3 tok/s plain decode (UNSIGNED, not pushed)_ |
| A | `docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/README.md` | 1 | _docs(phase16): DSpark speculative decode on DSv4.1 -- 22.7 tok/s at gamma=3, verify row cost is the cap_ |
| A | `docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/raw/p56-run1-lowaccept.log` | 1 | _docs(phase16): DSpark speculative decode on DSv4.1 -- 22.7 tok/s at gamma=3, verify row cost is the cap_ |
| A | `docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/raw/p56-run2.log` | 1 | _docs(phase16): DSpark speculative decode on DSv4.1 -- 22.7 tok/s at gamma=3, verify row cost is the cap_ |
| A | `docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/scripts/p56_spec.py` | 2 | _docs(dsv41): M1 ring-alias — sweep-completion notes + update the 5 archived spec probes_ |
| A | `docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/scripts/p58_accept_offline.py` | 1 | _docs(phase16): DSpark speculative decode on DSv4.1 -- 22.7 tok/s at gamma=3, verify row cost is the cap_ |
| A | `docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/scripts/p59_draft_debug.py` | 1 | _docs(phase16): DSpark speculative decode on DSv4.1 -- 22.7 tok/s at gamma=3, verify row cost is the cap_ |
| A | `docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/scripts/p60_variants.py` | 1 | _docs(phase16): DSpark speculative decode on DSv4.1 -- 22.7 tok/s at gamma=3, verify row cost is the cap_ |
| A | `docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/README.md` | 1 | _docs(phase17): verify row cost is mostly expert bytes + dense EXL3 rows; spec 22.7 -> 23.5 tok/s_ |
| A | `docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/raw/p56-run3.log` | 1 | _docs(phase17): verify row cost is mostly expert bytes + dense EXL3 rows; spec 22.7 -> 23.5 tok/s_ |
| A | `docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/scripts/p48_bench.py` | 2 | _docs(dsv41): M1 ring-alias — sweep-completion notes + update the 5 archived spec probes_ |
| A | `docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/scripts/p56_spec.py` | 2 | _docs(dsv41): M1 ring-alias — sweep-completion notes + update the 5 archived spec probes_ |
| A | `docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/scripts/p61_overlap.py` | 1 | _docs(phase17): verify row cost is mostly expert bytes + dense EXL3 rows; spec 22.7 -> 23.5 tok/s_ |
| A | `docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/scripts/p62_draft_ablate.py` | 1 | _docs(phase17): verify row cost is mostly expert bytes + dense EXL3 rows; spec 22.7 -> 23.5 tok/s_ |
| A | `docs/benchmarks/phase18-dsv41-adaptive-gamma-2026-09-28/README.md` | 1 | _docs(phase18): one-sync spec loop + adaptive gamma -- 22.9 tok/s mean on 5 prompts_ |
| A | `docs/benchmarks/phase18-dsv41-adaptive-gamma-2026-09-28/raw/p63-run1.log` | 1 | _docs(phase18): one-sync spec loop + adaptive gamma -- 22.9 tok/s mean on 5 prompts_ |
| A | `docs/benchmarks/phase18-dsv41-adaptive-gamma-2026-09-28/scripts/p63_spec2.py` | 2 | _docs(dsv41): M1 ring-alias — sweep-completion notes + update the 5 archived spec probes_ |
| A | `docs/benchmarks/phase19-dsv41-exo-integration-gate-2026-09-28/README.md` | 1 | _docs(phase19): DSv4.1 exo integration plan; long-prompt prefill is a blocker (74 tok/s at 8K, GPU timeout at 16K)_ |
| A | `docs/benchmarks/phase19-dsv41-exo-integration-gate-2026-09-28/raw/p64-r0.log` | 1 | _docs(phase19): DSv4.1 exo integration plan; long-prompt prefill is a blocker (74 tok/s at 8K, GPU timeout at 16K)_ |
| A | `docs/benchmarks/phase19-dsv41-exo-integration-gate-2026-09-28/scripts/p64_prefill.py` | 1 | _docs(phase19): DSv4.1 exo integration plan; long-prompt prefill is a blocker (74 tok/s at 8K, GPU timeout at 16K)_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/README.md` | 2 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-EXL3_MM_MAX_ROWS_20000.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-EXL3_MOE_FUSED_0.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-amort.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-bench-corr.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-bench.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-e384.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-final.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-k2c.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-layer20.log` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-prefill.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2-valid.log` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2_amort_check.py` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2_exl3_bench_results.json` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2_final_check.py` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2_final_results.json` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2_prefill_check.py` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2_valid_check.py` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2_weight_error_probe.py` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2b-v3.log` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/p2c-v3-e384.log` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase2-exl3-moe-kernels-2026-09-27/raw/scripts_p2_exl3_moe_bench.py` | 1 | _docs: phase 2 — EXL3 MoE kernels measured on M4 Max; gates missed_ |
| A | `docs/benchmarks/phase20-dsv41-m1-window-2026-09-28/README.md` | 1 | _docs(phase20): M1 window -- 16K prefill no longer crashes; prefill speed and post-prompt decode still block the switch_ |
| A | `docs/benchmarks/phase20-dsv41-m1-window-2026-09-28/raw/p64-m1.log` | 1 | _docs(phase20): M1 window -- 16K prefill no longer crashes; prefill speed and post-prompt decode still block the switch_ |
| A | `docs/benchmarks/phase21-dsv41-context-scaling-2026-09-29/README.md` | 1 | _docs(phase21): post-prompt decode collapse was compile warmup; indexer is the only context-scaling prefill term; 512-row chunks double 16K prefill_ |
| A | `docs/benchmarks/phase21-dsv41-context-scaling-2026-09-29/raw/p64-m1b.log` | 1 | _docs(phase21): post-prompt decode collapse was compile warmup; indexer is the only context-scaling prefill term; 512-row chunks double 16K prefill_ |
| A | `docs/benchmarks/phase21-dsv41-context-scaling-2026-09-29/scripts/p64_prefill.py` | 1 | _docs(phase21): post-prompt decode collapse was compile warmup; indexer is the only context-scaling prefill term; 512-row chunks double 16K prefill_ |
| A | `docs/benchmarks/phase21-dsv41-context-scaling-2026-09-29/scripts/p66_ctxscale.py` | 2 | _docs(dsv41): M1 ring-alias — sweep-completion notes + update the 5 archived spec probes_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/.gitignore` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md` | 2 | _docs(phase22): correct Stage A numbers after independent review (8K noise 3.3%, 8K chunk gain per-rep, 16K last-chunk +8-10%, wired limit captured, 679 reconciliation flagged unsup_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/.gitattributes` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/analysis-all-runs.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/api-baseline.jsonl` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-ops24-r0.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-ops24-r1.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-ops5-r0.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-ops5-r1.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-run1-r0.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-run1-r1.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-run2-r0.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-run2-r1.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-run3-r0.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/logs/p22_prefill-run3-r1.log` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/ops24/engine_prefill_extracted.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/ops24/progress.jsonl` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/ops24/results.json` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/ops5/engine_prefill_extracted.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/ops5/progress.jsonl` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/ops5/results.json` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/prestop-state.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/production-launch-cmd-m4-1.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/production-launch-cmd-m4-2.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/production-launch-env-m4-1.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/production-launch-env-m4-2.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/prompt-manifest.json` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/relaunch-macstudio-m4-1.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/relaunch-macstudio-m4-2.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/restore-verification.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run/engine_prefill_extracted.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run/progress.jsonl` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run/results.json` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run2/engine_prefill_extracted.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run2/progress.jsonl` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run2/results.json` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run3/engine_prefill_extracted.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run3/progress.jsonl` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/run3/results.json` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/running-macstudio-m4-1.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/running-macstudio-m4-2.txt` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/raw/wired-limit-2026-10-01.txt` | 1 | _docs(phase22): correct Stage A numbers after independent review (8K noise 3.3%, 8K chunk gain per-rep, 16K last-chunk +8-10%, wired limit captured, 679 reconciliation flagged unsup_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_analyze.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_api_baseline.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_prefill.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_prefill_launch.sh` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_prep.py` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/p22_run.sh` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/scripts/parent_lock.sh` | 1 | _docs(phase22): DSv4.1 prefill Stage A -- baseline, attribution, config levers_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/README.md` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/artifacts/mtp-port-20260927.tgz` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/artifacts/mtp-port-mods.patch` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/buildmtp.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/chunk.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/closeout.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/convertmtp.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/localize.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/margin.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/modverify.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/mtp-port-mods.patch` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/mtpecon.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/mtptest.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/seq.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/seqc.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/raw/spec3.log` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/build_reduced_mtp.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/convert_reduced_mtp.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/fix_taps.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/mtp_decode.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/mtp_port.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/p6a_spec_loop.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/p6b_chunk_equiv.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/p6c_localize.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/p6d_spec_rowseq.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/p6e_closeout.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/p6f_module_verify.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/p6g_margin.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_dspark_1.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_dspark_2.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_dspark_3.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_dspark_4.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_dspark_5.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_dspark_6.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_dspark_7.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_mtpgen.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_stash.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase3-mtp-dspark-port-2026-09-27/scripts/wire_taps.py` | 1 | _docs: phase 3 — MTP/DSpark draft head ported to the V4.1 reference port_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/README.md` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/raw/accept.out` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/raw/p7c.out` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/scripts/find_acceptance.sh` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/scripts/p7c_exl3_loadable_and_layers.py` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/scripts/p7e_guard_arithmetic.py` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/scripts/p7f_mtp_bytes.py` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/scripts/read_accept_logs.sh` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/scripts/red_green_guard.py` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase4-exl3-prod-mtp-compat-2026-09-28/scripts/scan_acceptance.sh` | 1 | _docs: phase 4 — EXL3 draft head is not loadable by production; DSpark guard false-FAIL fixed_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/README.md` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/raw/p8a.out` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/raw/p8b.out` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/raw/p8c.out` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/raw/p8c_trace.json` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/raw/p8d.out` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/raw/p8e.out` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/raw/p8f.out` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/scripts/p8a_expert_bytes.py` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/scripts/p8b_routing_trace.py` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/scripts/p8c_trace_dump.py` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/scripts/p8d_residency_sim.py` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/scripts/p8e_residency_sim2.py` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase5-planb-sizing-2026-09-28/scripts/p8f_exl3_layer_hist.py` | 1 | _docs: phase 5 — Plan B sized on real routing data; phase-2 retune claim corrected_ |
| A | `docs/benchmarks/phase6-strict-false-guard-2026-09-28/README.md` | 1 | _fix(mlx): make the strict=False load guard work on sanitize() architectures_ |
| A | `docs/benchmarks/phase6-strict-false-guard-2026-09-28/raw/live_proof_output.txt` | 1 | _fix(mlx): make the strict=False load guard work on sanitize() architectures_ |
| A | `docs/benchmarks/phase6-strict-false-guard-2026-09-28/raw/node1_full_mlx_suite.txt` | 1 | _fix(mlx): make the strict=False load guard work on sanitize() architectures_ |
| A | `docs/benchmarks/phase6-strict-false-guard-2026-09-28/scripts/p9_live_guard_proof.py` | 1 | _fix(mlx): make the strict=False load guard work on sanitize() architectures_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/README.md` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/raw/batch_accept.txt` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/raw/metrics_family.txt` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/raw/samples_2026-09-28.txt` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/scripts/p10_batch_accept.sh` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/scripts/p10_greedy_accept.sh` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/scripts/p10_live_accept.sh` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase7-live-acceptance-2026-09-28/scripts/p10c_greedy_hist.sh` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase8-fork-verification-2026-09-28/README.md` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase8-fork-verification-2026-09-28/raw/consult_verdict.txt` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase8-fork-verification-2026-09-28/raw/p11_fanout_sim.out` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase8-fork-verification-2026-09-28/scripts/p11_verify_fanout_sim.py` | 1 | _docs: phase 7 -- live DSpark acceptance measured with no relaunch; phase 8 -- fork verification_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/README.md` | 2 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/exl3-moe-v4-alu-optim.patch` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-arm-fused0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-arm-v20.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-bwprobe-20260928.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-bwprobe-driver.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-chain-cheap.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-chain-cheapshuf.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-chunk-driver.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-libcheck-adopted.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-libcheck-off.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-lut.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-spike-step0-20260928.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-stub-decode.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-stub-none.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-stub-shuffle.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-stub2-both.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-stub2-none.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-stub2-swar.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-uchar4.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-verify-v4-default.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13-verify-v4-off.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13_resweep_driver.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p13_tests3_driver.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p17-uchar4-driver.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p18-lut-driver.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p19-chain2-cheap.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p19-chain2-control.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p19-chain2-driver.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p20-xacc-driver.log` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p20-xacc-imm.log` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p20-xacc-xdir.log` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/p2_final_results.json` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T1-C0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T1-C1152.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T1-C128.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T1-C384.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T1-C768.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T2-C0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T2-C1152.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T2-C128.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T2-C384.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T2-C768.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T4-C0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T4-C1152.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T4-C128.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T4-C384.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T4-C768.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T8-C0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T8-C1152.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T8-C128.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T8-C384.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep1-T8-C768.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep2-T1-C0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep2-T1-C1152.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep2-T1-C384.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep2-T2-C0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep2-T2-C1152.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/raw/sweep2-fused0.log` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/apply_v4_alu_optim.py` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p13_libcheck.sh` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p13_resweep.sh` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p13_tests3.sh` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p13_verify_v4.sh` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p14_stub_probe.py` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p15_chain_probe.py` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p16_direct_probe.py` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p17_uchar4_probe.py` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p18_lut_probe.py` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p19_chain2_arms.sh` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p19_chain2_probe.py` | 1 | _docs(phase9): EXL3 kernel spike day 1 -- ALU chains diagnosed, two bit-identical fixes adopted (1.91x/2.0x), tests green_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p20_xacc_arms.sh` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/benchmarks/phase9-exl3-spike-2026-09-28/scripts/p20_xaccess_probe.py` | 1 | _docs(dsv41): phase 9 day-2 kernel closure, phase 11 serving-geometry addendum, phase 12 loader; plan status block_ |
| A | `docs/clock-synced-allsum-skew-vs-overhead-2026-09-14.md` | 2 | _docs: clock-synced jaccl trace resolves the all_sum 34x gap — shared local overhead, not cross-rank skew (2026-09-15)_ |
| A | `docs/comm-compute-overlap-already-exists-2026-08-21.md` | 1 | _docs(dsv4): comm/compute overlap for moe.all_sum already exists, re-verified_ |
| A | `docs/cross-rank-allsum-skew-2026-08-22.md` | 1 | _docs(dsv4): T4 -- cross-rank all_sum skew closed, real tail asymmetry flagged_ |
| A | `docs/decode-attribution-recompute-postfix-2026-08-22.md` | 1 | _docs(dsv4): T1 -- recompute decode attribution vs post-fence-fix baseline_ |
| A | `docs/decode-depth-ladder-to-565k-memory-wall-2026-09-22.md` | 1 | _docs: full decode depth ladder to 565K -- flat to ~120K, then a memory wall_ |
| A | `docs/decode-flat-across-depth-36tps-at-115k-2026-09-22.md` | 2 | _docs: full decode depth ladder to 565K -- flat to ~120K, then a memory wall_ |
| A | `docs/decode-idle-time-investigation-interim-synthesis-2026-08-22.md` | 1 | _docs(dsv4): interim synthesis of decode idle-time investigation_ |
| A | `docs/decode-roofline-dispatch-bound-2026-08-21.md` | 1 | _docs(dsv4): roofline shows decode at ~12% of bandwidth-bound peak_ |
| A | `docs/decode-time-budget-synthesis-2026-08-22.md` | 1 | _docs(dsv4): real decode time-budget synthesis + GPU idle-gap deep dive_ |
| A | `docs/deep-context-refault-mechanism-confirmed-2026-09-22.md` | 1 | _docs: deep-context degradation mechanism CONFIRMED -- mmap page re-faulting_ |
| A | `docs/deepseek-v4-c2-mtp-verify-fixes.md` | 1 | _docs: c>=2 MTP speculative verify correctness fixes (3 root causes + commit map)_ |
| A | `docs/deepseek-v4-flash-kickoff-prompt.md` | 2 | _deepseek-v4: switch to mlx-community/DeepSeek-V4-Flash-4bit_ |
| A | `docs/deepseek-v4-mtp-performance.md` | 2 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `docs/deepseek-v41-exl3-plan.md` | 4 | _docs(phase14): first end-to-end DSv4.1 decode on both nodes -- 9.3 tok/s plain, coherent text_ |
| A | `docs/deepseek-v41-memory-budget-correction-2026-09-27.md` | 1 | _docs: correct the V4.1 memory budget (section 5's 105 GB was file/2, not per-rank)_ |
| A | `docs/documentation-inventory-2026-09-14.md` | 1 | _docs: permanent documentation for 2026-09-12–14 session + doc inventory_ |
| A | `docs/dspark-14k-cliff-investigation-2026-08-27.md` | 1 | _perf(dspark): bookkeeping trim (EXO_DSV4_BOOKKEEP_FAST=1, default OFF) + 14K cliff investigation_ |
| A | `docs/dspark-352k-allocator-pool-analysis-2026-08-27.md` | 1 | _docs(dspark): 352.6K memory-regression investigation — root cause (replicated DSpark head), refuted hypotheses, fix + protocol plan_ |
| A | `docs/dspark-352k-batched-verify-transients-2026-08-27.md` | 1 | _docs(dspark): 352.6K memory-regression investigation — root cause (replicated DSpark head), refuted hypotheses, fix + protocol plan_ |
| A | `docs/dspark-352k-correctness-harness-verification-2026-08-28.md` | 1 | _docs(dspark): 352.6K correctness+harness verification of shipped config — determinism PASS, fresh-nonce base-nondeterminism artifact documented_ |
| A | `docs/dspark-352k-memory-regression-2026-08-27.md` | 2 | _docs(dspark): 352.6K protocol results — gate FAIL pre-fix (-17.9%), verbon3 candidate PASS (0 collapses, +17.6%); TP_SHARD promoted at depth_ |
| A | `docs/dspark-352k-residency-analysis-2026-08-27.md` | 1 | _docs(dspark): 352.6K memory-regression investigation — root cause (replicated DSpark head), refuted hypotheses, fix + protocol plan_ |
| A | `docs/dspark-352k-verification-preregister-2026-08-28.md` | 1 | _docs(dspark): 352.6K correctness+harness verification of shipped config — determinism PASS, fresh-nonce base-nondeterminism artifact documented_ |
| A | `docs/dspark-352k-verification-runs-2026-08-28.json` | 1 | _docs(dspark): 352.6K correctness+harness verification of shipped config — determinism PASS, fresh-nonce base-nondeterminism artifact documented_ |
| A | `docs/dspark-cs-profile-2026-08-26.md` | 1 | _docs(dspark): C_s per-step profile — verify rowseq makes C_s=3.20 (not 1.2-1.4), spec at break-even; Gate A clean_ |
| A | `docs/dspark-draft-epilogue-fusion-2026-08-27.md` | 2 | _docs(dspark): note gamma-pre-prune fix in draft-epilogue design doc_ |
| A | `docs/dspark-fullblock-context-scaling-cliff-2026-08-04.md` | 8 | _docs(dsv4): add tempered lead -- pre-existing PP stall may explain DSpark exposure_ |
| A | `docs/dspark-mtp-ab-preregister-2026-08-25.md` | 8 | _chore(dspark): flag DSpark MTP as production baseline — defaults flipped (VERIFY_BATCH=1, GAMMA=3, MTP=1, DEDICATED=0), baseline doc + resolution notes_ |
| A | `docs/dspark-mtp-master-history-2026-08-28.md` | 4 | _docs(dspark): P3 verbon3 ablation + P4b rowseq crossover CLOSED_ |
| A | `docs/dspark-mtp-production-baseline-2026-08-27.md` | 3 | _docs(dspark): 352.6K protocol results — gate FAIL pre-fix (-17.9%), verbon3 candidate PASS (0 collapses, +17.6%); TP_SHARD promoted at depth_ |
| A | `docs/dspark-p1-draft-epilogue-ab-results-2026-08-28.md` | 1 | _docs(dspark): P1 draft-epilogue A/B results — byte-lossless, throughput FAIL (-0.35% @100K), stays OFF; synchronous epilogue draft has no overlap to hide in_ |
| A | `docs/dspark-p1p4-campaign-preregister-2026-08-28.md` | 2 | _docs(dspark): P2 amendments — short-c2 degen finding + BS=2 abort-path runner crash, spec-off control added to L2_ |
| A | `docs/dspark-p2-c2-validation-results-2026-08-28.md` | 1 | _docs(dspark): P2 c=2 verdict (2 shared-generator bugs, spec legs clean, Bug-3 0/6), P4a curve (14K cliff refuted: 56ms batched), P3/P4b handoff notes_ |
| A | `docs/dspark-tier1-byte-identity-2026-08-26.md` | 1 | _docs(dspark): Tier 1 byte-identity — 2/3 short prompts identical, 1/3 (primary_colors) differs deterministically (rowseq/MoE 0.023%/row residual)_ |
| A | `docs/dspark-verdict-measurement-2026-08-26.md` | 1 | _docs(dspark): corrected verdict protocol 24-run measurement — REVERT (median +1.87%, CI [-0.82,+9.45]%)_ |
| A | `docs/dsv4-0731-dspark-native-head-plan-2026-08-03.md` | 3 | _docs: permanent documentation for 2026-09-12–14 session + doc inventory_ |
| A | `docs/dsv4-220k-prefill-eventwait-ringdiag-nonrepro-2026-08-18.md` | 1 | _docs: Event::wait ring-diag attempt -- non-reproduction across 2 runs_ |
| A | `docs/dsv4-220k-prefill-eventwait-rootcause-triage-2026-08-18.md` | 1 | _docs: Event::wait stall triage -- Story B falsified, rounds=1 is architectural, next step is CMDBUF_RING_DIAG_ |
| A | `docs/dsv4-220k-prefill-rdma-wait-breakdown-2026-08-18.md` | 1 | _docs: RDMA/GPU-sync wait breakdown for 220K prefill -- 8 Metal Event::wait stalls found_ |
| A | `docs/dsv4-220k-prefill-seqsplit-ab-2026-08-18.md` | 1 | _docs: SEQ_SPLIT A/B at long context -- SEQ_SPLIT=1 wins 29%, closes queued task_ |
| A | `docs/dsv4-220k-prefill-span-profile-2026-08-18.md` | 1 | _docs: DSv4-Flash 220K TP prefill span profile -- compute-bound confirmed, comms tax quantified_ |
| A | `docs/dsv4-4096-regression-root-cause-2026-08-19.md` | 4 | _docs: MYSTERY SOLVED -- 3.15x SDPA anomaly was a non-sync profiler artifact_ |
| A | `docs/dsv4-attention-kernel-efficiency-2026-08-18.md` | 2 | _docs: correct stale 62.6%/37%-gap denominator cross-references_ |
| A | `docs/dsv4-c2-serving-handoff-2026-07-06.md` | 21 | _docs: tool inventory table_ |
| A | `docs/dsv4-clear-cache-interval-2-test-2026-08-19.md` | 1 | _docs: clear-cache-interval=2 fix for 4096 regression tested, REFUTED_ |
| A | `docs/dsv4-decode-stall-2026-06-26.md` | 2 | _docs(corrections): dsv4-decode-stall-2026-06-26 — dated CORRECTIONS block (perf-campaign-2 R2/R3): no GPU collective exists (fence=CPU-stream handoff, not overlap primitive); attem_ |
| A | `docs/dsv4-memory-leak-handoff-2026-06-29.md` | 1 | _fix(dsv4): stop multi-turn PoolingCache snapshot leak in KVPrefixCache_ |
| A | `docs/dsv4-memory-leak-resolution-2026-06-29.md` | 1 | _docs(dsv4): add multi-turn memory leak resolution post-mortem_ |
| A | `docs/dsv4-memory-leaks-2026-06-27.md` | 1 | _docs: DSv4 prefix-cache/prefill memory leaks — 4 fixes + verification_ |
| A | `docs/dsv4-prefill-step-size-4096-retest-2026-08-18.md` | 1 | _docs: 4096 chunk-size retest -- quality break is stale, but throughput REGRESSES 8% at long context_ |
| A | `docs/dsv4-rowseq-followups-plan-2026-07-10.md` | 13 | _bench: land gemv_bi_test (MLX_GEMV_BATCH_INVARIANT correctness+bitexactness gate); plan doc status header_ |
| A | `docs/dsv4-sdpa-subtiling-code-map-2026-08-19.md` | 1 | _docs: file:line code map of DSv4 attention forward for SDPA sub-tiling (Option A scoping)_ |
| A | `docs/dsv4-vision-phase3b-window-geometry-inventory.md` | 2 | _chore(dsv4-vision): bump mlx-lm to Phase 3 correctness-gap closure + inventory corrections_ |
| A | `docs/dsv41-big-append-census-2026-10-13.md` | 1 | _docs(dsv41): big-append Stage 0 census — felt-latency case CLOSED (weak): target −75.5% under exo-ideal reuse; 79% of genuine rows are cold rebuilds; S=1.07% < 8%; plus the conditi_ |
| A | `docs/dsv41-big-append-prefill-plan-2026-10-13.md` | 1 | _docs(dsv41): big-append prefill felt-latency workstream — executable plan (consult, saved for next-session execution; Stage 0 = offline census first, its P1/P2 re-baseline our own _ |
| A | `docs/dsv41-degeneration-guards-2026-10-13.md` | 6 | _docs(dsv41): loop-guard LIVE stage-2 (ARMED) — action=error verified both ranks; error-mode fire terminates at onset (5.1s vs 11s warn); identity byte-clean_ |
| A | `docs/dsv41-indexer-abscost-plan-2026-10-12.md` | 1 | _docs(dsv41): indexer absolute-cost workstream — executable plan (Fable consult #5, saved for next-session execution)_ |
| A | `docs/dsv41-indexer-profile-2026-10-12.md` | 5 | _docs(dsv41): piece-2 DECISION — STOP (bypass EV 0.015%; K-read lever dead; int8 probe gate FAIL at 1.34-1.35x wall / 1.48-1.49x GPU); decision doc + P_H closeout; profile doc updat_ |
| A | `docs/dsv41-indexer-qsplit-buildnote-2026-10-11.md` | 5 | _fix(p20): indexer qsplit D3 — key the tiled route on n_full; advance mlx-lm pin + buildnote_ |
| A | `docs/dsv41-indexer-qsplit-deploy-readiness-2026-10-11.md` | 1 | _docs(p20): indexer-qsplit P1 buildnote + deploy-readiness note (no PERFORMANCE_HISTORY entry: gated, unmeasured build)_ |
| A | `docs/dsv41-indexer-qsplit-live-ab-postmortem-2026-10-11.md` | 1 | _docs(perf): indexer-qsplit LIVE A/B — ABORT (token-identity divergence on a banded non-first prefill chunk); reverted to next22 (+ PH entry)_ |
| A | `docs/dsv41-indexer-qsplit-live-ab2-2026-10-12.md` | 1 | _docs(perf): indexer-qsplit LIVE A/B #2 (fixed build) — PASS (token identity 5/5 incl identity_3; prefill +10.5%/+11.9%; battery CLEAN); KEPT LIVE (+ PH entry)_ |
| A | `docs/dsv41-indexer-qsplit-refdiv-audit-2026-10-14.md` | 2 | _docs(dsv41): qsplit-refdiv — add the geometry-line instrument hazard (finding 4) for the relaunch A/B precondition_ |
| A | `docs/dsv41-indexer-refdiv-audit-2026-10-14.md` | 1 | _docs(dsv41): indexer reference-divergence audit + pre-registered relaunch A/B spec_ |
| A | `docs/dsv41-indexer-relaunch-ab-prereg-2026-10-14.md` | 1 | _docs(dsv41): indexer reference-divergence audit + pre-registered relaunch A/B spec_ |
| A | `docs/dsv41-moe-occupancy-scope-2026-10-10.md` | 1 | _docs(dsv41): MoE kernel 'occupancy' project (lever #2) scoping memo — CONDITIONAL GO on a bounded spike; hypotheses+probes+kill gate_ |
| A | `docs/dsv41-native-experts-fit-scope-2026-10-11.md` | 2 | _docs(dsv41/p20): owner decision — native-experts requant NOT authorized; reopen condition = third node (TP3) with verified 3-node fit math (+ PH entry)_ |
| A | `docs/dsv41-next22-construct-2026-10-11.md` | 1 | _deploy(next22): construct note for deploy/next22-rankfix (base 3afce11d7 + rank-continuity fix @ bebeb0066; 367 passed)_ |
| A | `docs/dsv41-next25-guards-construct-2026-10-13.md` | 1 | _deploy(next25): guard construct — dsv41 degeneration loop-guard on next24 + launcher forward + fingerprint entry_ |
| A | `docs/dsv41-next26-limiter-construct-2026-10-14.md` | 2 | _docs(dsv41): next26-limiter construct — LIVE deploy record + guard on/off A/B (byte-identical)_ |
| A | `docs/dsv41-next27-ringfix-construct-2026-10-14.md` | 4 | _docs(dsv41): next27 LIVE record correction — residual comparator provenance fix (in-regime vs phase22; no 16.5x)_ |
| A | `docs/dsv41-piece2-decision-2026-10-12.md` | 1 | _docs(dsv41): piece-2 DECISION — STOP (bypass EV 0.015%; K-read lever dead; int8 probe gate FAIL at 1.34-1.35x wall / 1.48-1.49x GPU); decision doc + P_H closeout; profile doc updat_ |
| A | `docs/dsv41-prefill-liveness-and-hang-guard-design-2026-10-05.md` | 1 | _docs(dsv41): prefill liveness + hang-guard design (fence-hook heartbeat + shadow stack-class guard; Fable-reviewed 2026-10-05)_ |
| A | `docs/dsv41-quant-variant-survey-2026-10-11.md` | 1 | _docs(dsv41): alternative quant/variant survey for 2x M4 Max exo/MLX cluster_ |
| A | `docs/dsv41-reasoning-budget-limiter-2026-10-13.md` | 2 | _feat(dsv41): port the reasoning-budget limiter into the round-batched decode (degeneration-guard PHASE 2)_ |
| A | `docs/dsv41-restart-deploy-construct-2026-10-11.md` | 1 | _deploy(dsv41): next21-restart construct = next20-vision + full restart chain (G1/G2/G3/G4/G5)_ |
| A | `docs/dsv41-restart-g1-buildnote-2026-10-10.md` | 2 | _dsv41(park): G1 hardening — keep prior generation on abort, robust dirty tracking, newest-wins keyed restore (adversarial-audit fixes)_ |
| A | `docs/dsv41-restart-g2g5-buildnote-2026-10-10.md` | 1 | _dsv41(park): G2 build fingerprint + refusal policy, G5 durability hygiene_ |
| A | `docs/dsv41-restart-g3-buildnote-2026-10-10.md` | 2 | _dsv41(park): G3 post-audit fix pass — bound shutdown checkpoint overshoot to ONE unit (MED-A), honest T5 in-flight gate (LOW-D), buildnote corrections (MED-B/OQ-1, LOW-C)_ |
| A | `docs/dsv41-restart-g4-buildnote-2026-10-10.md` | 1 | _dsv41(park): G4 multi-rank all-or-nothing restore + cross-rank generations_ |
| A | `docs/dsv41-restart-live-ab-runbook-2026-10-11.md` | 1 | _docs(dsv41): live-A/B deploy+validation+rollback runbook for next21-restart_ |
| A | `docs/dsv41-restart-next22-results-2026-10-11.md` | 1 | _docs(dsv41): round-2 closeout — next22-rankfix restart-survival re-validation (WARN no-ABORT)_ |
| A | `docs/dsv41-restart-prelive-review-2026-10-11.md` | 1 | _docs(dsv41): consolidated pre-live review for restart-survival chain (ledger + dispositions)_ |
| A | `docs/dsv41-restart-rankfix-buildnote-2026-10-11.md` | 2 | _dsv41(park): buildnote for the rank-continuity fix (design, audit findings, test evidence)_ |
| A | `docs/dsv41-restart-survival-phase0-audit-2026-10-10.md` | 1 | _docs(dsv41): phase-0 read-only audit — A3 never-parked CONFIRMED; cross-build restores (12/15) + consuming restore + zero TP2 coordination; GO_ |
| A | `docs/dsv41-restart-survival-plan-2026-10-10.md` | 1 | _docs(dsv41): restart-survival plan for parked prefix-KV — phase-0 audit + gaps G1-G7 + tests/rollout (plan only, NOT executed; Fable-reviewed)_ |
| A | `docs/dsv41-reuse-ladder-design-2026-10-05.md` | 2 | _docs(dsv41): fold Fable review corrections into reuse-ladder spec (spacing 1024, decode ladder, end-anchor, keep-0 retention, bitwise gate)_ |
| A | `docs/dsv41-scaling-alternatives-brief-2026-10-15.md` | 7 | _docs(dsv41): scaling-alternatives brief — SS6: owner goal recorded (single-stream DSv4.1 prefill-max; multi-stream not a goal; TP3 = the remaining lever)_ |
| A | `docs/dsv41-smalln-wide-tile-buildnote-2026-10-12.md` | 1 | _fix(p20): indexer small-n wide tile (R-B) — advance mlx-lm pin + buildnote + kill-switch forward line_ |
| A | `docs/dsv41-smalln-wide-tile-live-round-2026-10-13.md` | 2 | _docs(dsv41): route-cliff fix BANKED — live on 9fe1d593/a0573e3 (revised bar): p300k 148 ms steady (4.24x), quality battery CLEAN, identity 5/5; record + P&H updated_ |
| A | `docs/dsv41-spec-rollback-ring-fix-2026-10-14.md` | 3 | _docs(dsv41): M1 ring-alias — sweep-completion notes + update the 5 archived spec probes_ |
| A | `docs/dual-cable-topology-and-qp-budget-2026-08-21.md` | 1 | _docs(jaccl): standing writeup of tonight's transport work + QP budget answer_ |
| A | `docs/eagle_k1_fix_report.md` | 1 | _docs(dspark): master campaign history 2026-05→08-28 — consolidation of all 23 MTP/DSpark docs, corrections ledger, forward plan_ |
| A | `docs/fence-async-28pct-claim-traced-to-artifact-2026-08-22.md` | 1 | _docs(dsv4): resolve FENCE_ASYNC +28%/+1.04% discrepancy, close out t7_ |
| A | `docs/flag-sweep-completion-and-dspark-native-finding-2026-08-21.md` | 2 | _docs(dsv4): correct DSPARK_NATIVE scope -- decode-only, not applicable to prefill work_ |
| A | `docs/fork-notes.md` | 26 | mentioned in: upstream-prs.md, deepseek-v4-mtp-performance.md |
| A | `docs/fork-vs-upstream-inventory.md` | 1 | _docs: complete fork-vs-upstream file-by-file change inventory_ |
| A | `docs/freshboot-control-inverts-model-chunk-size-dominant-2026-09-23.md` | 1 | _result(target a): fresh-boot control INVERTS the model -- chunk size is dominant_ |
| A | `docs/fused-softmax-negative-result-2026-08-21.md` | 1 | _docs(dsv4): negative result -- EXO_DSV4_FUSED_SOFTMAX breaks correctness_ |
| A | `docs/gather-qmm-m1-dispatch-confirmed-correct-2026-08-21.md` | 1 | _docs(mlx): confirm gather_qmm M=1 already dispatches to gemv kernel_ |
| A | `docs/gpu-clock-symptom-confirmed-2026-08-22.md` | 1 | _docs(dsv4): Phase B -- GPU clock confirmed as symptom via sustained-load test_ |
| A | `docs/gpu-idle-gap-deep-dive-2026-08-22.md` | 1 | _docs(dsv4): real decode time-budget synthesis + GPU idle-gap deep dive_ |
| A | `docs/gpu-occupancy-clock-gap-postfix-2026-08-22.md` | 1 | _docs(dsv4): T2 -- fresh post-fix Instruments capture, occupancy/gap/clock_ |
| A | `docs/gpu-util-vs-allsum-cost-reconciled-2026-08-19.md` | 2 | _docs: correct overlap-lever plan -- the naive eval-defer mechanism was already tried and reverted (OPT-7)_ |
| A | `docs/gpu-utilization-confirmed-saturated-2026-08-18.md` | 1 | _docs+bench: confirm GPU near-saturated (93-98%) throughout real prefill_ |
| A | `docs/handoff-2026-08-08-section22.md` | 1 | _docs: handoff for Section 22 deploy + validation, next session_ |
| A | `docs/handoff-2026-08-08-section23.md` | 1 | _Section 23: deploy Section 22's fix, real-hardware validation finds a hard stall_ |
| A | `docs/handoff-2026-08-08-section25.md` | 1 | _docs: Section 25 -- validate the API fix end-to-end, Section 22 fully closed_ |
| A | `docs/handoff-2026-08-09-section39.md` | 1 | _docs: handoff for Section 39 -- non-fatal retransmit fix deployed+verified, unmasked a real mutual-deadlock bug in pp_scheduler_wire.py that needs to be the next session's first pr_ |
| A | `docs/handoff-2026-08-10-section42.md` | 1 | _docs: handoff for Section 42 -- deadlock investigation disproved one hypothesis (unfixed), RDMA migration un-shelved and stands as next priority_ |
| A | `docs/handoff-2026-08-10-section43-part2.md` | 1 | _docs: handoff part 2 -- Section 43, two more bugs fixed, one new symptom found unfixed_ |
| A | `docs/handoff-2026-08-10-section43.md` | 1 | _docs: handoff for Section 43 -- Section 37 Phase 1 deployed, 2 bugs fixed, 1 found unfixed_ |
| A | `docs/hc-collapse-depth-verification-2026-08-25.md` | 1 | _docs(perf): hc_collapse depth verification — gain HOLDS at 300K/500K (+1.97%/+1.89%), production restored kernel-ON_ |
| A | `docs/hc-collapse-kernel-ab-2026-08-25.md` | 2 | _docs(perf): hc_collapse ship — production flip executed and verified (append §14)_ |
| A | `docs/hc-collapse-roofline-2026-08-24.md` | 1 | _docs(perf): hc_expand depth A/B (300K/500K inconclusive) + T7 remainder audits_ |
| A | `docs/hc-expand-depth-verification-2026-08-24.md` | 1 | _docs(perf): hc_expand depth A/B (300K/500K inconclusive) + T7 remainder audits_ |
| A | `docs/hc-expand-kernel-ab-2026-08-24.md` | 2 | _docs: final production-state verification for hc_expand ship_ |
| A | `docs/hc-expand-rejection-relitigated-multiseed-2026-08-22.md` | 1 | _docs(dsv4): P5 -- hc_expand bf16-comb rejection upheld across 7 seeds; fp32-acc variant precise but 1.6x slower; no shippable variant_ |
| A | `docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md` | 155 | _docs: Section 119 -- correct Section 118; rank 0 is stuck in allocate_buffers, not dialing_ |
| A | `docs/hyperconnection-training-gate-false-lead-2026-08-22.md` | 1 | _docs(dsv4): T10 sub-investigation -- HyperConnection training-gate lead falsified_ |
| A | `docs/incidents/2026-08-08-section23-stall-m4-1.log` | 1 | _Section 23: deploy Section 22's fix, real-hardware validation finds a hard stall_ |
| A | `docs/incidents/2026-08-08-section23-stall-m4-2.log` | 1 | _Section 23: deploy Section 22's fix, real-hardware validation finds a hard stall_ |
| A | `docs/incidents/2026-08-09-section17-batched-decode-wedge-rank0-dump.txt` | 1 | _docs: Section 17 attempt #6 -- memory-headroom PASS, batched-decode cancel gap root-caused_ |
| A | `docs/incidents/2026-08-09-section17-batched-decode-wedge-rank1-dump.txt` | 1 | _docs: Section 17 attempt #6 -- memory-headroom PASS, batched-decode cancel gap root-caused_ |
| A | `docs/incidents/2026-08-09-section17-concurrency-wedge-sample.txt` | 1 | _docs+bench: Section 17 attempt #5 -- network root-cause fixed, new concurrency-wedge bug found_ |
| A | `docs/incidents/c2-spec-verify-structural-drift-2026-09-16.md` | 1 | _fix(c2-spec): correct the FALSE "no gate exists" claim; localize the c>=2 verify structural drift_ |
| A | `docs/incidents/hang-watchdog-false-positive-and-vision-regression-2026-09-21.md` | 5 | _docs: RESOLVED -- reboot restored GPU, vision regression does not reproduce on healthy hardware_ |
| A | `docs/incidents/lmhead-mxfp8-cross-lingual-glue-defect-2026-09-13.md` | 1 | _docs: permanent documentation for 2026-09-12–14 session + doc inventory_ |
| A | `docs/incidents/pkill-self-match-forces-sigkill-2026-09-10.md` | 1 | _docs(incidents): pkill -f self-match forces a SIGKILL of exo on the control node_ |
| A | `docs/incidents/ssh-quoting-executed-cluster-launcher-2026-09-28.md` | 1 | _docs: incident — a quoting error executed the cluster launcher; no impact_ |
| A | `docs/incidents/unsigned-push-false-justification-2026-09-14.md` | 1 | _docs: incident writeup for the unsigned-push/false-justification event + new §12 lesson #15 on re-deriving trusted headline numbers_ |
| A | `docs/indexer-pblock-decode-regression-2026-08-21.md` | 1 | _docs(dsv4): EXO_DSV4_INDEXER_PBLOCK -- decode regression, no prefill win_ |
| A | `docs/indexer-prefill-decomposition-2026-08-24.md` | 1 | _docs(perf): hc_expand depth A/B (300K/500K inconclusive) + T7 remainder audits_ |
| A | `docs/indexer-topk-fused-decode-only-2026-08-22.md` | 1 | _docs(dsv4): T10 -- indexer topk_fused closed via code reading, session wrap-up_ |
| A | `docs/instruments-metal-trace-real-dispatch-latency-2026-08-21.md` | 1 | _docs(dsv4): real Instruments Metal trace -- ground-truth dispatch latency_ |
| A | `docs/iogpu-residency-set-abort.md` | 1 | iogpu-residency-set-abort.md |
| A | `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` | 1 | _docs(dsv4): jaccl-internal timing decomposes moe.all_sum 34x gap -- transport is fast_ |
| A | `docs/jaccl-sz3-tested-no-improvement-2026-08-20.md` | 1 | _docs: tested MLX_JACCL_RELIABLE_MAX_SZ=3 live -- no hang, no measurable improvement_ |
| A | `docs/jit-model-lifecycle-handoff-2026-06-30.md` | 4 | _docs(jit): record EXO_JIT_ENABLED default flip 0->1 (23fb58fc)_ |
| A | `docs/known-good-decode-baseline-20261008.md` | 2 | _docs: record post-ship smoke PASS in known-good baseline_ |
| A | `docs/known-good-prefill-baseline-2026-08-21.md` | 1 | _docs: known-good prefill/decode baseline after tonight's jaccl hardening_ |
| A | `docs/kv-cache-architecture.md` | 4 | kv-cache-architecture.md |
| A | `docs/launcher-address-drift-repair-2026-09-28.md` | 2 | _docs: record 2026-10-03 laptop-side Tailscale repoint + API_HOST chain reorder_ |
| A | `docs/lever1-moe-smallm-headroom-2026-08-20.md` | 3 | _docs(perf): Lever-1 MoE -- 40-core Studio confirms R=1.11x, core-count caveat CLOSED_ |
| A | `docs/lever2-seqchunk-overlap-2026-08-20.md` | 1 | _docs: Lever 2 sequence-chunk overlap -- real but modest signal (~1.1-1.15x), laptop-noisy_ |
| A | `docs/live-decode-two-rank-instruments-trace-2026-08-21.md` | 1 | _docs(dsv4): live two-rank decode trace -- real GPU-idle confirms roofline_ |
| A | `docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md` | 1 | _docs: permanent documentation for 2026-09-12–14 session + doc inventory_ |
| A | `docs/local-absmax-fence-artifact-confirmed-2026-08-19.md` | 1 | _bench+docs: confirm local_absmax's 400ms probe cost was an mx.eval() fence artifact_ |
| A | `docs/long-context-gpu-occupancy-2026-08-22.md` | 1 | _docs(dsv4): T5 -- long-context GPU occupancy INCREASES with depth_ |
| A | `docs/memory-residency-check-ruled-out-2026-08-22.md` | 1 | _docs(dsv4): rule out memory residency/expert paging via live checks_ |
| A | `docs/metrics.md` | 1 | metrics.md |
| A | `docs/minimax-decode-optimization.md` | 11 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-fused-attention-design.md` | 1 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-fused-attention-prompt.md` | 3 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-quantized-sdpa-design.md` | 8 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `docs/minimax-rdma-moe-validation-2026-04-24.md` | 3 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| A | `docs/moe-all-sum-178ms-artifact-real-bottleneck-2026-08-20.md` | 1 | _bench+docs: Lever 1 NO-GO (MoE small-M kernel), Lever 3 real root-cause found (178ms/call was an arithmetic artifact)_ |
| A | `docs/moe-all-sum-dominant-cost-2026-08-19.md` | 1 | _docs: moe.all_sum confirmed dominant TP prefill cost -- 61-64% of wall time, 2.6x speedup ceiling_ |
| A | `docs/moe-all-sum-payload-size-causal-test-2026-08-19.md` | 1 | _docs: P1 all_sum payload-size causal test -- instrumentation + run plan_ |
| A | `docs/moe-all-sum-skew-vs-comms-2026-08-19.md` | 1 | _docs: moe.all_sum cost is jaccl 16KB chunking, not skew and not link bandwidth_ |
| A | `docs/moe-allsum-collective-cost-confirmed-2026-08-21.md` | 1 | _docs(dsv4): confirm real TP collective cost via sync-span (moe.all_sum=14.4%)_ |
| A | `docs/moe-allsum-quant-compute-overhead-analysis-2026-08-19.md` | 1 | _docs: quantized all_sum compute overhead is ~1.5% of projected savings, not a concern_ |
| A | `docs/moe-allsum-quant-live-test-failed-2026-08-19.md` | 1 | _docs: quantized all_sum LIVE TEST FAILED -- all_gather hangs the collective_ |
| A | `docs/moe-allsum-quant-phase0-repro-2026-08-19.md` | 1 | _docs+bench: PHASE 0/2.1 -- quantized all_sum logic EXONERATED offline_ |
| A | `docs/moe-allsum-quant-root-cause-and-closure-2026-08-19.md` | 1 | _docs: quantized moe.all_sum root cause found, then whole approach found mathematically dead_ |
| A | `docs/moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md` | 1 | _docs: CORRECTED final answer -- shared-scale all_sum is 1.49x SLOWER during real prefill_ |
| A | `docs/moe-allsum-sharedscale-live-test-no-speedup-2026-08-19.md` | 1 | _docs: shared-scale int8 all_sum -- correctness passed, no speedup, real GPU asymmetry found_ |
| A | `docs/moe-allsum-sharedscale-root-cause-found-2026-08-19.md` | 1 | _docs: shared-scale all_sum root cause found -- local_absmax reduction, not comms_ |
| A | `docs/moe-gate-up-fusion-validated-2026-08-21.md` | 1 | _perf(dsv4): validate MoE gate+up fusion -- real +3% decode win_ |
| A | `docs/moe-gpu-time-overlap-bandwidth-bound-2026-08-19.md` | 1 | _docs: confirm MoE GEMM is memory-bandwidth-bound via real GPU-time accounting_ |
| A | `docs/moe-per-stage-gpu-breakdown-2026-08-18.md` | 1 | _docs: MoE per-stage breakdown -- gather/scatter is only ~4-5%, not the missing 16%_ |
| A | `docs/moe-quant-vs-bf16-dequant-attribution-2026-08-19.md` | 1 | _bench+docs: MoE gap is intrinsic ragged-matmul, NOT dequant overhead_ |
| A | `docs/moe-tile-geometry-retune-dead-end-2026-08-18.md` | 2 | _docs: correct stale 62.6%/37%-gap denominator cross-references_ |
| A | `docs/moe-vs-dense-qmm-isolation-2026-08-19.md` | 2 | _bench: isolate MoE efficiency from the mxfp4 quant tax at production shapes_ |
| A | `docs/mtp-dspark-tp-port-decision-gate-2026-08-22.md` | 1 | _docs(dsv4): T6 -- MTP/DSpark TP-port gate NOT recommended, stale baseline found_ |
| A | `docs/mtp-tiebreak-losslessness-fix.md` | 1 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `docs/mxfp4-gather-qmm-rhs-lhs-kernel-2026-08-19.md` | 1 | _docs: mxfp4 gather_qmm_rhs_lhs kernel -- built, correctness-validated, but dead code for MoE_ |
| A | `docs/offline-collective-microbenchmark-2026-08-21.md` | 1 | _docs(dsv4): offline collective microbenchmark -- 34x gap is overhead, not RDMA floor_ |
| A | `docs/overnight-250k-cycle-cost-and-fence-status-2026-09-23.md` | 1 | _docs(overnight): 250K gap is cycle cost; the big fence lever is already banked_ |
| A | `docs/overnight-closeout-levers-exhausted-2026-09-23.md` | 1 | _docs(overnight close-out): all cheap levers measured and exhausted_ |
| A | `docs/overnight-corrections-dspark-sizing-and-content-driven-tps-2026-09-23.md` | 1 | _docs(overnight): two measured corrections -- DSpark lever-1 is 50x too small; tok/s is content-driven_ |
| A | `docs/overnight-final-indexer-hypothesis-falsified-corrected-curve-2026-09-23.md` | 1 | _docs(overnight FINAL): indexer-cliff hypothesis FALSIFIED; corrected depth curve_ |
| A | `docs/overnight-final-n6-250k-bistability-2026-09-23.md` | 1 | _docs(overnight): N=6 at 250K -- bistability confirmed at 250K; T1 and T2 are ONE problem_ |
| A | `docs/overnight-memory-ratchet-and-leaf-discriminator-2026-09-23.md` | 1 | _docs(overnight): ratchet is DEPTH-specific (not leaf-related); prefix cache verified excellent_ |
| A | `docs/overnight-ratchet-depth-specific-prefix-cache-good-2026-09-23.md` | 1 | _docs(overnight): ratchet is DEPTH-specific (not leaf-related); prefix cache verified excellent_ |
| A | `docs/overnight-results-two-mechanisms-and-stochastic-collapse-2026-09-23.md` | 1 | _docs(overnight): ladder complete -- TWO mechanisms, and the 500K collapse is stochastic_ |
| A | `docs/p01-switch-mlp-gputrace-recapture-2026-08-29.md` | 1 | _docs(dspark): Phase 0 switch_mlp capture + Phase 1(a)/(b) residual chase_ |
| A | `docs/p01a-allsum-arrival-skew-at-depth-2026-08-29.md` | 1 | _docs(dspark): Phase 0 switch_mlp capture + Phase 1(a)/(b) residual chase_ |
| A | `docs/p01b-multilayer-pipelining-loss-2026-08-29.md` | 1 | _docs(dspark): Phase 0 switch_mlp capture + Phase 1(a)/(b) residual chase_ |
| A | `docs/p02c-moe-allocator-depth-residual-2026-08-29.md` | 1 | _docs(dspark): Phase (c) MoE-at-depth + allocator residual analysis: INCONCLUSIVE_ |
| A | `docs/p02d-roofline-occupancy-kernel-reconciliation-2026-08-29.md` | 1 | _docs(dspark): Phase (d) reconciles the "5-7x slower than hardware ceiling" open question as a byte-accounting artifact, not a performance mystery_ |
| A | `docs/p03-smallop-bucket-gputrace-2026-08-30.md` | 1 | _docs(dspark): P03 first-ever per-kernel capture of decode small-op bucket -- NOT irreducible, 3 fusion targets identified_ |
| A | `docs/p04-sinkhorn-truncation-numerics-2026-08-30.md` | 1 | _docs(dspark): P04 Sinkhorn-truncation numerics gate -- P03's 4-5-iter projection REFUTED_ |
| A | `docs/p07-prefill-remainder-perkernel-preregister-2026-08-30.md` | 1 | _docs(perf): P07 pre-registration -- prefill remainder is ~14.0%, not 19.3%_ |
| A | `docs/p07-prefill-remainder-perkernel-results-2026-08-30.md` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `docs/p08-sdpa-ceiling-and-topk-spike-preregister-2026-08-30.md` | 1 | _docs(perf): P08 pre-registration -- SDPA-compressed denominator + indexer top-k spike_ |
| A | `docs/p13-switch-mlp-subphase-attribution-2026-09-14.md` | 1 | _docs(perf-history): close out 3 of §13's open threads (340K cliff live re-test, switch_mlp sub-phase attribution, clock-synced all_sum skew measurement)_ |
| A | `docs/p14-switch-mlp-gap-decomposition-2026-09-15.md` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `docs/p2-xctrace-prefill-collective-wedge-2026-08-23.md` | 1 | _docs(dsv4): P2 -- xctrace attach wedges live TP=2 prefill collective (3/3, NOT the fc954293 watchdog bug); P4 -- TP=2 width-sharding has NO skinny-GEMM penalty, sign inverted_ |
| A | `docs/p3-followup-allsum-wait-at-depth-2026-08-24.md` | 1 | _docs(perf): close all_sum-wait-at-depth — collective wait does NOT grow with depth_ |
| A | `docs/p3-followup-poolgrow-ab-2026-08-23.md` | 4 | _docs(perf): Part III — pool-growth default flip landed, 3/3 byte-identity gates, 25.32 tok/s @352.6K_ |
| A | `docs/p3-reviewer-r1-verification-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p3-reviewer-r2-verification-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p3-synthesis-500k-decode-decay-decomposition-2026-08-23.md` | 2 | _perf(dsv4): flip chunked BatchPoolingCache growth ON by default (gated) + reviewer-mandated doc corrections_ |
| A | `docs/p3-worker-a-kv-read-inventory-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p3-worker-b1-live-depth-anchors-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p3-worker-c-attn-kernel-walltime-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p3-worker-c2-depth-busy-idle-capture-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p3-worker-c3-donation-failure-insitu-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p3-worker-d-metal-timeout-crash-forensics-2026-08-23.md` | 1 | _docs(perf): P3 500K-context decode-decay decomposition investigation_ |
| A | `docs/p4-scoping-mtp-for-tp-2026-08-24.md` | 2 | _docs(perf): P4v2 — DSpark-primary spec-decode re-scoping (user-directed), code-truth at TP=2_ |
| A | `docs/p4-tp-width-shard-gemm-efficiency-2026-08-23.md` | 1 | _docs(dsv4): P2 -- xctrace attach wedges live TP=2 prefill collective (3/3, NOT the fc954293 watchdog bug); P4 -- TP=2 width-sharding has NO skinny-GEMM penalty, sign inverted_ |
| A | `docs/p4v2-m1-shadow-gate-results-and-recovery-2026-08-24.md` | 1 | _docs(p4v2): M1 shadow-gate results — verdict HOLD; incident recovery; cluster reverted to production_ |
| A | `docs/p5-tp-prefill-gap-2026-08-24.md` | 2 | _docs(p5): TP-vs-PP prefill gap RESOLVED — artifact, and live head-to-head INVERTS it_ |
| A | `docs/phase-c-dual-capture-confirms-fix-2026-08-22.md` | 1 | _docs(dsv4): Phase C -- dual CPU+GPU capture triple-confirms the fix_ |
| A | `docs/phase0a-allsum-boundary-decomposition-2026-08-20.md` | 1 | _bench+docs: PHASE 0a -- moe.all_sum GPU->CPU boundary is real and payload-proportional_ |
| A | `docs/phase0b-collective-overlap-gate-2026-08-20.md` | 1 | _bench+docs: Phase 0b gate PASSES -- collectives do not drain the GPU_ |
| A | `docs/phase0c-collective-order-determinism-2026-08-20.md` | 1 | _bench+docs: Phase 0c -- collective matching is by EVAL order, not issue order (real hazard found)_ |
| A | `docs/phase3-cluster-validation-blocked-resolved-2026-08-20.md` | 1 | _docs: Phase 3 cluster validation -- blocked by hardware, resolved via existing data_ |
| A | `docs/pp-prefill-tp-decode-phase-swap-design-2026-08-16.md` | 2 | _docs: mark the phase-swap design DEAD; add a CURRENT STATE header to the main doc_ |
| A | `docs/prefill-chunk-overlap-ab-harness-design-2026-08-20.md` | 1 | _docs(prefill): land the chunk-overlap A/B harness + its design doc_ |
| A | `docs/prefill-chunk-overlap-live-test-2026-08-20.md` | 1 | _docs: EXO_PREFILL_CHUNK_OVERLAP live test -- real, unresolved correctness anomaly, not a clean result_ |
| A | `docs/prefill-chunk-overlap-race-fix-2026-08-20.md` | 1 | _fix(prefill): root-cause cross-stream ordering race in chunk-overlap path_ |
| A | `docs/prefill-cliff-mechanism-2026-08-24.md` | 1 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `docs/prefill-fence-gate-audit-2026-08-22.md` | 1 | _docs(dsv4): T8 -- prefill fence-gate audit confirms by-design, closes_ |
| A | `docs/prefill-flops-roofline-aggregate-2026-08-22.md` | 1 | _docs(dsv4): T7 -- prefill FLOPs roofline, self-corrected + Fable-reviewed_ |
| A | `docs/prefill-optimization-campaign-handoff-2026-08-18.md` | 1 | _docs: consolidated handoff for tonight's full prefill optimization campaign_ |
| A | `docs/prefill-optimization.md` | 2 | prefill-optimization.md |
| A | `docs/prefill-sync-span-kernel-breakdown-2026-08-21.md` | 1 | _docs(prefill): real sync-span kernel breakdown inside prefill_ |
| A | `docs/prefill-throughput-breakthrough-2026-06-24.md` | 2 | _docs: B=2 + MTP resolution — seq-split & bootstrap fixes, verified sweep_ |
| A | `docs/prefill-trace-instrumentation-findings-2026-08-21.md` | 1 | _docs(prefill): document tracing findings + correct stale "unwired" claim_ |
| A | `docs/profiling/request_lifecycle_trace.md` | 10 | _docs: mark prefill optimization as compute-bound and complete_ |
| A | `docs/pysampler-blocking-eval-root-cause-2026-08-22.md` | 1 | _diag(dsv4): wire EXO_DSV4_FENCE_GATE_DIAG passthrough + bump mlx-lm_ |
| A | `docs/qa-kv-fusion-no-measurable-gain-2026-08-21.md` | 1 | _docs(dsv4): wq_a+wkv fusion implemented+correct, no measurable decode gain_ |
| A | `docs/rollback-cost-campaign-handoff-2026-07-12.md` | 2 | _docs: close out rollback-cost campaign handoff (task #24)_ |
| A | `docs/roofline-recalculated-post-fix-2026-08-22.md` | 1 | _docs(dsv4): Phase A -- roofline recalculated post-async-fence-fix_ |
| A | `docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md` | 1 | _docs(dsv4): roofline sanity check confirms both inputs correct_ |
| A | `docs/sdpa-flop-denominator-audit-2026-08-24.md` | 1 | _docs(perf): hc_expand depth A/B (300K/500K inconclusive) + T7 remainder audits_ |
| A | `docs/section43-stall-rank0-dump-2026-08-10.txt` | 1 | _chore: bump mlx submodule for jaccl send()/recv() data-progress StallWatch fix_ |
| A | `docs/section43-stall-rank1-dump-2026-08-10.txt` | 1 | _chore: bump mlx submodule for jaccl send()/recv() data-progress StallWatch fix_ |
| A | `docs/start-cluster-mlx-rebuild-skip-2026-08-19.md` | 1 | _docs: start_cluster.sh mlx-rebuild-skip fix -- verified working, ~34% faster relaunch_ |
| A | `docs/superdeepseek-2xdgx-speedup-levers-2026-08-19.md` | 1 | _docs: SuperDeepseek-2xDGX serving-stack review — transferable speed levers_ |
| A | `docs/switch-mlp-bandwidth-artifact-retraction-2026-08-22.md` | 1 | _docs(dsv4): P0 CLOSED -- switch_mlp 27.7% BW figure RETRACTED as serial-sync bench artifact; kernel actually at 74-87% of peak, no lever exists_ |
| A | `docs/switch-mlp-kernel-bandwidth-efficiency-2026-08-22.md` | 1 | _docs(dsv4): T3 -- switch_mlp real achieved bandwidth is 27.7% of peak_ |
| A | `docs/t10-final-decomposition-closed-2026-08-22.md` | 1 | _docs(dsv4): T10 CLOSED -- full decomposition complete, no async-fence bug_ |
| A | `docs/target-b-attribution-VALIDATED-sync-mode-2026-09-23.md` | 1 | _verify(target b): sync-mode profiling VALIDATES the attribution -- indexer is not the cost_ |
| A | `docs/target-b-attribution-sdpa-compressed-not-indexer-2026-09-23.md` | 1 | _diag(target b): span attribution -- depth cost is attn.sdpa.compressed, NOT the indexer_ |
| A | `docs/target-b-synthesis-attribution-was-too-shallow-2026-09-23.md` | 2 | _verify(target b): all 8 source claims SUPPORTED -- nothing falsified, 4 imprecisions recorded_ |
| A | `docs/target-b-topk-overlap-falsifies-stale-reuse-2026-09-23.md` | 1 | _diag(target b): top-k overlap FALSIFIES stale-reuse at depth; lever is dead_ |
| A | `docs/thinking-parser-fused-delimiter-fix.md` | 1 | thinking-parser-fused-delimiter-fix.md |
| A | `docs/turboquant-integration.md` | 2 | turboquant-integration.md / kv-cache-architecture.md |
| A | `docs/upstream-pr-drafts/02-mlx-sdpa-chunked.md` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `docs/upstream-pr-drafts/04-mlx-head-dim-192-256.md` | 1 | _docs(upstream-prs): record 5 PR outcomes from 2026-05-27 sync session_ |
| A | `docs/upstream-pr-drafts/06-mlx-allocator-coalesce.md` | 1 | _docs(upstream-prs): add mlx#3596 (MetalAllocator coalesce) to open PRs_ |
| A | `docs/upstream-pr-drafts/07-exo-thinking-parser-fused-delimiter.md` | 1 | thinking-parser-fused-delimiter-fix.md |
| A | `docs/upstream-pr-drafts/README.md` | 3 | mentioned in: skill:upstream-sync-triage.md |
| A | `docs/upstream-prs.md` | 21 | mentioned in: deepseek-v4-mtp-performance.md, fork-notes.md |
| A | `docs/vec-divergence-verified-fixed-and-depth-is-the-real-lever-2026-09-22.md` | 2 | _docs: CORRECTION -- decode is flat across depth, 36.7 t/s @115K; target already met_ |
| A | `docs/vec-rowsdpa-campaign-2026-07-12.md` | 3 | _feat(dsv4-mtp): corrected-numerics champion — ATTN_ALLSUM=0 by default_ |
| A | `docs/vec-serving-increment4-handoff-2026-07-12.md` | 3 | _docs: restructure increment-4 handoff for the next campaign (lossless per-row-sdpa vec)_ |
| A | `docs/verify-565k-collapse-persists-definitive-2026-09-23.md` | 1 | _verify(565K): DEFINITIVE -- both runs collapse; peak 2.5-8.4 GB over the wired limit_ |
| A | `docs/verify-565k-fix-not-sufficient-2026-09-23.md` | 1 | _verify(565K): accounting fix is correct but NOT sufficient -- collapse persists_ |
| A | `docs/verify-batch-g0-fail-2026-08-27.md` | 2 | _chore(dspark): flag DSpark MTP as production baseline — defaults flipped (VERIFY_BATCH=1, GAMMA=3, MTP=1, DEDICATED=0), baseline doc + resolution notes_ |
| A | `docs/verify-batch-phase0-2026-08-26.md` | 2 | _chore(dspark): flag DSpark MTP as production baseline — defaults flipped (VERIFY_BATCH=1, GAMMA=3, MTP=1, DEDICATED=0), baseline doc + resolution notes_ |
| A | `docs/xcode-removal-launcher-clt-fallback-2026-08-25.md` | 1 | _docs: Xcode-removal launcher incident + CLT fallback fix writeup_ |

### Model cards (config)  (9 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `resources/inference_model_cards/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml` | 2 | _feat(dsv41): serve images -- builder loads the vision tower, card advertises vision_ |
| A | `resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-0731.toml` | 1 | _feat(model-cards): add DeepSeek-V4-Flash-0731 official release card_ |
| A | `resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp.toml` | 3 | _docs(dsv4-vision): Phase 5 executed — vision verified on hardware_ |
| A | `resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash.toml` | 1 | _feat(dsv4): add model card for DeepSeek's native V4-Flash checkpoint_ |
| A | `resources/inference_model_cards/mlx-community--DeepSeek-V4-Flash-4bit.toml` | 1 | _deepseek-v4: switch to mlx-community/DeepSeek-V4-Flash-4bit_ |
| A | `resources/inference_model_cards/mlx-community--DeepSeek-V4-Flash-6bit.toml` | 1 | _deepseek-v4: add 6-bit quant model card_ |
| M | `resources/inference_model_cards/mlx-community--DeepSeek-V4-Flash.toml` | 3 | _fix(dsv4): match Ollama Cloud's sampling exactly for weights-parity_ |
| A | `resources/inference_model_cards/mlx-community--Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-4bit.toml` | 1 | _chore: add model card for Huihui-Qwen3.5-35B-A3B-Opus-abliterated-4bit_ |
| A | `resources/inference_model_cards/mlx-community--Qwen3.5-397B-A17B-nvfp4.toml` | 1 | _feat: restore P2P model download and add Qwen3.5-397B-A17B-nvfp4 model card_ |

### Ops scripts  (5 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `scripts/context_stress_sweep.sh` | 3 | _style: apply shfmt and swift-format to in-scope shell and Swift_ |
| A | `scripts/convert_dsv4_mtp.sh` | 2 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `scripts/exo_graceful_shutdown.sh` | 1 | _fix(deploy): stop the pkill self-match that force-SIGKILLs exo on the coordinator_ |
| A | `scripts/prefill_overlap_ab.py` | 2 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `scripts/test_shutdown_self_match.sh` | 1 | _fix(deploy): stop the pkill self-match that force-SIGKILLs exo on the coordinator_ |

### Other  (38 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `.github/workflows/pipeline.yml` | 2 | _ci: decouple pytest step from nix flake check success_ |
| M | `.gitignore` | 4 | mentioned in: skill:2026-05-26-mlx-upstream-refactor-port-attempt.md, skill:2026-05-26-mlx-upstream-sync-port.md |
| A | `.gitmodules` | 1 | mentioned in: skill:mlx-deployment-verification.md, skill:upstream-sync-triage.md |
| M | `.typings/mlx_lm/generate.pyi` | 3 | _Revert "fix(pp-batched): adapt to mlx-lm's GenerationBatch.Response schema after pin advance"_ |
| M | `.typings/mlx_lm/models/cache.pyi` | 1 | _fix(pp-dspark): route PoolingCache speculative-rollback through save_meta, not the generic state fallback_ |
| M | `.typings/mlx_lm/models/deepseek_v4.pyi` | 3 | _feat(mlx-lm): land _forward_steps via proper merge, not rebase -- prior cherry-pick incident fully closed_ |
| M | `AGENTS.md` | 1 | _fix(dsv4-vision): patch_embed_tokens bounds hardening + test-infra guard_ |
| A | `INDEXER_TILED_P_PLAN.md` | 1 | _perf(dsv4): bump mlx-lm to tiled-P indexer (59a5b9a) + bit-exact bench_ |
| A | `MOE_KERNEL_HANDOFF.md` | 24 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `PREFILL_CLIFF_HANDOFF.md` | 2 | _docs(perf): prefill-cliff mechanism closure — gc_limit/throttle threshold family, argpartition attribution correction, era A/B rescoped (sync-span observer effect)_ |
| A | `PREFILL_THROUGHPUT_PLAN.md` | 1 | _fix(download): suppress inert catalog DownloadPending entries from /state_ |
| M | `README.md` | 3 | _docs: request the mlx extra in the documented setup commands (#2245)_ |
| A | `UPSTREAM_SYNC_2026-07-22.md` | 1 | _docs: write up the exo+mlx+mlx-lm upstream sync (2026-07-22/23)_ |
| A | `_diag/T-fail.txt` | 0 | &mdash; |
| A | `_diag/V-fail.txt` | 0 | &mdash; |
| A | `_diag/VHEAD-fail.txt` | 0 | &mdash; |
| A | `_diag/ws5-try1.txt` | 0 | &mdash; |
| M | `app/EXO/EXO/EXOApp.swift` | 1 | _style: apply shfmt and swift-format to in-scope shell and Swift_ |
| A | `cluster-diag.sh` | 3 | _fix(ops): repoint cluster host resolution to Tailscale (mDNS dead)_ |
| A | `conftest.py` | 1 | _fix(dsv4-vision): patch_embed_tokens bounds hardening + test-infra guard_ |
| A | `deploy/grafana/exo-overview.json` | 1 | mentioned in: metrics.md |
| M | `flake.nix` | 2 | _fix(nix): scope treefmt to code we own so flake check can reach every formatter_ |
| A | `mlx` | 140 | mentioned in: upstream-prs.md, api.md |
| A | `mlx-lm` | 351 | mentioned in: upstream-prs.md, thinking-parser-fused-delimiter-fix.md |
| M | `nix/darwin-build-fixes.patch` | 1 | _fix(nix): rebase darwin-build-fixes.patch onto mlx e40a416b2 (unblocks macOS CI)_ |
| M | `python/parts.nix` | 2 | _fix(nix): bump vendored nanobind pin to v2.13.0 to match mlx e40a416b2_ |
| A | `reboot-node.sh` | 3 | _fix(reboot-node): --check probed wrong sudo grant (status vs authrestart)_ |
| A | `start_cluster.sh` | 484 | mentioned in: upstream-prs.md, deepseek-v4-mtp-performance.md |
| A | `sync-upstream.sh` | 2 | _revert(mtp): remove broken tie-break bonus recompute_ |
| A | `test_ds_encoding.py` | 5 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `test_jinja.py` | 5 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `test_qwen.py` | 5 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `test_qwen_encoding.py` | 5 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `test_qwen_moe.py` | 5 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `test_qwen_vision.py` | 5 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `test_utils.py` | 5 | _style: apply ruff format to in-scope Python (src/, tools/, tests/, scripts/)_ |
| A | `tools/dsv4_route_hist_summary.py` | 3 | deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md |
| A | `tools/jaccl_wedge_watchdog.sh` | 2 | _style: apply shfmt and swift-format to in-scope shell and Swift_ |

### Z-scratch (session notes, accidentally committed)  (45 files)

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
| A | `.hermes/plans/2026-06-10_dsv4_hyperconnection_fix_and_cohost_bench.md` | 1 | _docs(dsv4): root-cause writeup + co-host benchmarks for the HC-key fix_ |
| A | `.hermes/plans/2026-06-12-prefill-optimization-fable5-analysis.md` | 1 | _docs(plans): add prefill-optimization fable5 analysis notes_ |

### Campaign scratch (tmp/ campaign + probe scratch)  (781 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `tmp/decode-35-campaign/PLAN.md` | 1 | _docs: vec-vs-loop divergence VERIFIED fixed + lossless; depth is the real decode lever_ |
| A | `tmp/decode-deep-context-campaign-20260922/LOG.md` | 1 | _docs(overnight): checkpoint -- acceptance falls 2.55x with depth (controlled); DSpark lever dead_ |
| A | `tmp/decode-deep-context-campaign-20260922/PLAN.md` | 2 | _docs(overnight): checkpoint -- acceptance falls 2.55x with depth (controlled); DSpark lever dead_ |
| A | `tmp/decode-deep-context-campaign-20260922/PREREG-adaptive-prefill-2026-09-23.md` | 1 | _experiment(prefill): enable context-adaptive chunk shrink at high ctx (128 past 200K)_ |
| A | `tmp/decode-deep-context-campaign-20260922/PREREG-freshboot-control-2026-09-23.md` | 1 | _prereg(target a): fresh-boot control for the chunk-size vs boot-state confound_ |
| A | `tmp/hardening-round1-20260903/REPORT.md` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/hardening-round1-20260903/findings/A-prune-and-rsync.md` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/hardening-round1-20260903/findings/BC-patch-and-mxeval.md` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/hardening-round1-20260903/findings/D1-exo-guards.md` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/hardening-round1-20260903/findings/D2-hermes-guards.md` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/hardening-round1-20260903/findings/DELETION_MANIFEST.tsv` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/hardening-round1-20260903/findings/tmp_inventory.tsv` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/hardening-round2-20260903/REPORT.md` | 1 | _docs(hardening): ROUND 2 -- BatchPoolingCache guard now runs in CI (+9-commit stale lock pin found), profiler unit guard added, telemetry kit ruled dead code_ |
| A | `tmp/hardening-round3-20260903/REPORT.md` | 1 | _docs(hardening): ROUND 3 -- mlx pin drift resolved (supervisor divergence claim was a SHA mis-parse), mutation-verified submodule/uv.lock alignment guard now in CI, hc_expand kerne_ |
| A | `tmp/hardening-round4-20260903/REPORT.md` | 1 | _docs(hardening): ROUND 4 -- CI observed for the first time (workflows were disabled_manually); failures proven pre-existing via control run; round-3 environmental premise refuted_ |
| A | `tmp/hardening-round4-20260903/evidence.md` | 1 | _docs(hardening-round4): full-suite failure triage + nix-fmt-without-nix evidence_ |
| A | `tmp/lever1-recheck-20260831/alignment_fair_studio40_rep2.json` | 1 | _docs(perf): Lever-1 MoE -- 40-core Studio confirms R=1.11x, core-count caveat CLOSED_ |
| A | `tmp/lever1-recheck-20260831/alignment_fair_studio40_rep3.json` | 1 | _docs(perf): Lever-1 MoE -- 40-core Studio confirms R=1.11x, core-count caveat CLOSED_ |
| A | `tmp/lever1-recheck-20260831/alignment_fair_studio40_rep4.json` | 1 | _docs(perf): Lever-1 MoE -- 40-core Studio confirms R=1.11x, core-count caveat CLOSED_ |
| A | `tmp/overnight-loop/CHARTER.md` | 2 | _docs(loop): charter -- mandatory grep-the-record-first step after campaign-2 R1 repeated retracted work_ |
| A | `tmp/overnight-loop/STATE.md` | 12 | _docs: update hang-watchdog incident writeup with shipped fix + live verification_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/PRE_REGISTERED_GATES.md` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/analyze_margins.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/collect_real_margins.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/diagnose_zero_accept.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/knob_smoke_test.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/base_100k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/base_100k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/base_100k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/base_5k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/base_5k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/base_5k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_100k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_100k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_100k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_100k_retry1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_100k_retry2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_16k_diag.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_5k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_5k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_5k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/quant_8k_diag.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/same_prompt_A.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/same_prompt_A_0731_dryrun.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/same_prompt_B_0731.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/same_prompt_G5_result.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/warmup_a1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/warmup_a2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/warmup_b1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab/warmup_b2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_100k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_100k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_100k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_5k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_5k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_5k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_warmup_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v2/A_warmup_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_100k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_100k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_100k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_5k_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_5k_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_5k_3.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_warmup_1.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/live_ab_v3/B_warmup_2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/lmhead_numerics.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/lmhead_numerics.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/lmhead_numerics_v2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/lmhead_numerics_v2.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/real_h_replay.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/real_margins/ANALYSIS.md` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/real_margins/margins_ctx20k.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/real_margins/margins_ctx2k.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/real_margins/margins_ctx5k.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/real_margins/margins_trivial.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/real_margins/summary.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/rootcause/dump_roundtrip_test.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/rootcause/mem_m4-1.jsonl` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/rootcause/mem_m4-2.jsonl` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/rootcause2/RUN_ATTRIBUTION.md` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/rootcause2/attribution_table.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/rootcause2/v2_0731_acceptance.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/run_lmhead_numerics.log` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/run_lmhead_numerics_v2.log` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/same_prompt_probe.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/studio_lmhead_microbench.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/studio_lmhead_microbench.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_001_surrogate1.bin` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_001_surrogate1.bin.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_002_surrogate2.bin` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_002_surrogate2.bin.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_003_surrogate3.bin` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_003_surrogate3.bin.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_004_surrogate4d.bin` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_dumps/prenorm_h_004_surrogate4d.bin.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-lmhead-mxfp8-20260830/surrogate_replay_run.log` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-review-20260830/phaseA_review.md` | 2 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-review-20260830/phaseB_review.md` | 1 | _docs(perf): P05 review -- none of the 3 interrupted fusion targets ship_ |
| A | `tmp/p05-review-20260830/phaseC_review.md` | 2 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/probe_load_path.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_m_invariance.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_m_invariance.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_pad8_speed.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_pad8_speed.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_pad8_speed_w2.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_pad8_speed_w2.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_padded_qmm.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/real_shared_padded_qmm.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/run_real_padded_qmm.log` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/run_shared_kernel_test.log` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/shared_batching_kernel_test.json` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p05-shared-batching-20260830/shared_batching_kernel_test.py` | 1 | _docs(perf): P05 reversal -- Phase A bug was wrong-model artifact, Phase C reopened_ |
| A | `tmp/p07-20260830/VERDICT_topk_argpartition_vs_argsort_20260830.md` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/capture_stdout.log` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/find_idx_keys.py` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/p07_prefill_remainder_capture.py` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/p07b_topup_capture.py` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/p07c_topk_argpartition_vs_argsort.py` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/results.json` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/results_topk_ab.json` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/results_topup.json` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/run_node.sh` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p07-20260830/topup_stdout.log` | 1 | _docs(perf): P07 results -- prefill remainder is mostly floor, one real open item_ |
| A | `tmp/p08-20260830/diag10.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag11.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag12.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag13.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag2.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag3.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag4.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag5.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag6.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag7.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag8.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag9.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag_fp32.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag_mask.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag_shape.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/diag_sinks.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item1_results.json` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item1_stdout.log` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item1_verdict.md` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item2_phaseA.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item2_phaseB.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item2_phaseB_capture.log` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item2_phaseB_kernel_source.metal` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item2_phaseB_partD.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/item2_phaseB_results.json` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/p08_item1_capture.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/p08_item1b_reachability.py` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/run_item1_node.sh` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/run_item1b_node.sh` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/run_item2_phaseA_m4-2.sh` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p08-20260830/run_item2_phaseB_m4-2.sh` | 1 | _docs(perf): P08 -- exact top-k kernel already exists and is default-ON; Item 1 needs a query-split not a key-split_ |
| A | `tmp/p09-20260831/p09_impl_validate.py` | 1 | _P09 Item 1: land query-tiled compressed SDPA TP/seq-split fix + validation harness_ |
| A | `tmp/p11-decode-20260831/rep1.json` | 1 | _docs(perf): P11 result -- live decode FAIL-LOW regression (-10.41% vs P06 ref, median 30.71 vs 34.28 tok/s @100K)_ |
| A | `tmp/p11-decode-20260831/rep2.json` | 1 | _docs(perf): P11 result -- live decode FAIL-LOW regression (-10.41% vs P06 ref, median 30.71 vs 34.28 tok/s @100K)_ |
| A | `tmp/p11-decode-20260831/rep3.json` | 1 | _docs(perf): P11 result -- live decode FAIL-LOW regression (-10.41% vs P06 ref, median 30.71 vs 34.28 tok/s @100K)_ |
| A | `tmp/p11-decode-20260831/warmup.json` | 1 | _docs(perf): P11 result -- live decode FAIL-LOW regression (-10.41% vs P06 ref, median 30.71 vs 34.28 tok/s @100K)_ |
| A | `tmp/p12-decode-20260831/preflight_node1_ps_eww.txt` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/preflight_node2_ps_eww.txt` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/rep1.json` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/rep1.stdout.log` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/rep2.json` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/rep2.stdout.log` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/rep3.json` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/rep3.stdout.log` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/warmup.json` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-decode-20260831/warmup.stdout.log` | 1 | _docs(perf): P12 RESULT -- OUTCOME B, decode regression survives cold relaunch_ |
| A | `tmp/p12-relaunch-20260831/before_node1_ps_eww.txt` | 1 | _chore(perf): P12 before-snapshot config artifacts (both nodes, pre-relaunch)_ |
| A | `tmp/p12-relaunch-20260831/before_node2_ps_eww.txt` | 1 | _chore(perf): P12 before-snapshot config artifacts (both nodes, pre-relaunch)_ |
| A | `tmp/p12-relaunch-20260831/start_cluster.log` | 1 | _chore(perf): P12 before-snapshot config artifacts (both nodes, pre-relaunch)_ |
| A | `tmp/p12-relaunch-20260831/start_cluster2.log` | 1 | _chore(perf): P12 before-snapshot config artifacts (both nodes, pre-relaunch)_ |
| A | `tmp/p13-20260914/results.json` | 1 | _docs(perf-history): close out 3 of §13's open threads (340K cliff live re-test, switch_mlp sub-phase attribution, clock-synced all_sum skew measurement)_ |
| A | `tmp/p14-20260914-decomposition/p14b_decisive_1789430175.json` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `tmp/p14-20260914-decomposition/p14b_decisive_1789430191.json` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `tmp/p14-20260914-decomposition/p14b_decisive_1789430207.json` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `tmp/p14-20260914-decomposition/p14c_converged_cliff_1789430349.json` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `tmp/p14-20260914-decomposition/p14d_converged_smallm_1789430381.json` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `tmp/p14-20260914-decomposition/p14e_reconciliation_1789430522.json` | 1 | _docs(perf-history): P13's 33-41%-of-peak switch_mlp figure was substantially a benchmark artifact -- corrected to ~59-64%_ |
| A | `tmp/perf-campaign-2/IDEAS.md` | 4 | _docs(loop): Fix A CANCELLED by user (relaunches not part of normal sessions); recommendation is STOP unless a target is named_ |
| A | `tmp/perf-campaign-2/round1/I1-COLLECTIVE-LATENCY.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round1/I1-PATCH-NOTES.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round1/I2-C2-TAX-AUDIT.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round1/I3-KERNEL-MICROBENCH.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round1/I4-FIXB-REOPEN.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round1/I6-AND-COLLECTIVE-COUNT.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round1/REPORT.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round1/VERIFICATION.md` | 1 | _perf(campaign2-r1): attribution -- the 2x is local GPU drain, not collectives or kernels_ |
| A | `tmp/perf-campaign-2/round10/PRE-REGISTRATION.md` | 1 | _round10: PRE-REGISTRATION -- residual declared the GOVERNING statistic before any measurement_ |
| A | `tmp/perf-campaign-2/round10/R9-RESIDUAL-RECOMPUTE.md` | 1 | _round10: R9 residual recompute (PRIMARY, zero cluster cost) -- short pooled gap 205.3ms, C1+C2 PASS; + R10 drivers (n=25)_ |
| A | `tmp/perf-campaign-2/round10/REPORT.md` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/compare_identity.py` | 1 | _round10: R9 residual recompute (PRIMARY, zero cluster cost) -- short pooled gap 205.3ms, C1+C2 PASS; + R10 drivers (n=25)_ |
| A | `tmp/perf-campaign-2/round10/r9_residual_recompute.json` | 1 | _round10: R9 residual recompute (PRIMARY, zero cluster cost) -- short pooled gap 205.3ms, C1+C2 PASS; + R10 drivers (n=25)_ |
| A | `tmp/perf-campaign-2/round10/recompute_r9_residual.py` | 1 | _round10: R9 residual recompute (PRIMARY, zero cluster cost) -- short pooled gap 205.3ms, C1+C2 PASS; + R10 drivers (n=25)_ |
| A | `tmp/perf-campaign-2/round10/results/PROD_env.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/P_2k_r1.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_2k_r2.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_2k_r3.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_2k_r4.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_2k_r5.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_env.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r1.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r10.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r11.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r12.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r13.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r14.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r15.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r16.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r17.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r18.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r19.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r2.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r20.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r21.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r22.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r23.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r24.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r25.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r3.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r4.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r5.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r6.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r7.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r8.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/P_short_r9.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/Q2_2k_r1.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_2k_r2.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_2k_r3.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_2k_r4.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_2k_r5.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_env.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r1.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r10.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r11.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r12.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r13.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r14.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r15.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r16.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r17.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r18.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r19.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r2.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r20.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r21.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r22.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r23.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r24.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r25.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r3.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r4.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r5.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r6.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r7.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r8.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/Q2_short_r9.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_2k.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_2k.reasoning.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_2k.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_89k.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_89k.reasoning.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_89k.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_short.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_short.reasoning.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c1_short.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_2k.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_2k.reasoning.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_2k.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_89k.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_89k.reasoning.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_89k.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_short.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_short.reasoning.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV0_c2_short.txt` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_2k.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_2k.reasoning.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_2k.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_89k.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_89k.reasoning.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_89k.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_short.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_short.reasoning.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c1_short.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_2k.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_2k.reasoning.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_2k.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_89k.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_89k.reasoning.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_89k.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_short.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_short.reasoning.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/identity_RV200_c2_short.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/prod_health.json` | 1 | _round10: REPORT -- SHIP RENDEZVOUS_MS=0. Residual pre-registered as governing; full-set pooled short gap 224.4ms in [150,250] band (predicted 200); byte-identity PASS both arms w/ _ |
| A | `tmp/perf-campaign-2/round10/results/steelbi_RV200_89k_1.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/steelbi_RV200_89k_1.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/steelbi_RV200_89k_2.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/steelbi_RV200_89k_2.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/steelbi_RV200_89k_3.json` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/results/steelbi_RV200_89k_3.txt` | 1 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/run_89k_selfcontrol.sh` | 2 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/run_boot.sh` | 2 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/run_identity.sh` | 2 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/run_reps.sh` | 2 | _round10: boot P (RV=200, ps-eww verified) -- 25 short + 5 2K reps, identity c1/c2, R7 steel-BI 89K self-control 3/3 IDENTICAL; pin PY=/usr/bin/python3 (Homebrew python3 lacks httpx_ |
| A | `tmp/perf-campaign-2/round10/summarize10.py` | 1 | _round10: R9 residual recompute (PRIMARY, zero cluster cost) -- short pooled gap 205.3ms, C1+C2 PASS; + R10 drivers (n=25)_ |
| A | `tmp/perf-campaign-2/round11/CODE-READ.md` | 1 | _round11: Task1 CODE-READ -- c=1 request path end to end, 204 verified line cites_ |
| A | `tmp/perf-campaign-2/round11/PHASE-MARKS-NOTES.md` | 1 | _round11: c=1 request-path phase instrumentation (env-gated, G1-G6)_ |
| A | `tmp/perf-campaign-2/round11/PREDICTION.md` | 1 | _round11: PRE-REGISTRATION -- Task0 slope expectation (mixed verdict, primary fit = residual minus transit), closure gate 10ms, actionability floor 75ms, G1-G6 hard gates, seam rule_ |
| A | `tmp/perf-campaign-2/round11/REPORT.md` | 1 | _round11: REPORT -- Task0 MIXED verdict (slope 0.68us/tok CI[-0.45,1.82], r2 0.027, INTERCEPT 0.68s stable = real context-independent floor); Task1 code read: server_received_ts sta_ |
| A | `tmp/perf-campaign-2/round11/TASK0-REGRESSION.md` | 1 | _round11: Task 0 regression on 55-request real-usage dataset_ |
| A | `tmp/perf-campaign-2/round11/analyze_marks.py` | 1 | _round11: c=1 request-path phase instrumentation (env-gated, G1-G6)_ |
| A | `tmp/perf-campaign-2/round11/replay_c1.py` | 2 | _round13: both boots SPENT. Apparatus PROVEN on hardware; Gate A NOT MEASURED (stale model id ate the workload). Nothing shipped._ |
| A | `tmp/perf-campaign-2/round11/task0_regression.json` | 1 | _round11: Task 0 regression on 55-request real-usage dataset_ |
| A | `tmp/perf-campaign-2/round11/task0_regression.py` | 1 | _round11: Task 0 regression on 55-request real-usage dataset_ |
| A | `tmp/perf-campaign-2/round12/PREDICTION.md` | 1 | _round12: PREDICTION -- I16 plan_step poll tick, fable-reviewed pre-registration + R13 branch table_ |
| A | `tmp/perf-campaign-2/round12/REPORT.md` | 1 | _docs(perf): CAMPAIGN 2 R12 -- I16 fix code-complete but MEASUREMENT BLOCKED (cluster access denied); env-gated default OFF so nothing ships and the 100ms tick is still live; lost-w_ |
| A | `tmp/perf-campaign-2/round12/seam_harness/FINDINGS.md` | 1 | _round12: seam-rule harness (prep only, not wired)_ |
| A | `tmp/perf-campaign-2/round12/seam_harness/dsv4_encoder_loader.py` | 1 | _round12: seam-rule harness (prep only, not wired)_ |
| A | `tmp/perf-campaign-2/round12/seam_harness/normalizer_check.py` | 1 | _round12: seam-rule harness (prep only, not wired)_ |
| A | `tmp/perf-campaign-2/round12/seam_harness/run_seam_harness.py` | 1 | _round12: seam-rule harness (prep only, not wired)_ |
| A | `tmp/perf-campaign-2/round12/seam_harness/seam_core.py` | 1 | _round12: seam-rule harness (prep only, not wired)_ |
| A | `tmp/perf-campaign-2/round12/seam_harness/seam_corpus.py` | 1 | _round12: seam-rule harness (prep only, not wired)_ |
| A | `tmp/perf-campaign-2/round12/seam_harness/tokenizer_paths.py` | 1 | _round12: seam-rule harness (prep only, not wired)_ |
| A | `tmp/perf-campaign-2/round13/PREDICTION.md` | 4 | _round13: THIRD pre-boot apparatus defect caught -- A4 pairing was undefined; fixed pre-boot, still unmeasured_ |
| A | `tmp/perf-campaign-2/round13/marks/boot1_m4-1_marks.txt` | 1 | _round13: both boots SPENT. Apparatus PROVEN on hardware; Gate A NOT MEASURED (stale model id ate the workload). Nothing shipped._ |
| A | `tmp/perf-campaign-2/round13/marks/boot1_m4-2_marks.txt` | 1 | _round13: both boots SPENT. Apparatus PROVEN on hardware; Gate A NOT MEASURED (stale model id ate the workload). Nothing shipped._ |
| A | `tmp/perf-campaign-2/round13/marks/boot1_m4-2_marks.txt.analysis.json` | 1 | _round13: both boots SPENT. Apparatus PROVEN on hardware; Gate A NOT MEASURED (stale model id ate the workload). Nothing shipped._ |
| A | `tmp/perf-campaign-2/round13/marks/boot1_replay_results.jsonl` | 1 | _round13: both boots SPENT. Apparatus PROVEN on hardware; Gate A NOT MEASURED (stale model id ate the workload). Nothing shipped._ |
| A | `tmp/perf-campaign-2/round13/marks/smoke_m4-1.txt` | 1 | _round13: both boots SPENT. Apparatus PROVEN on hardware; Gate A NOT MEASURED (stale model id ate the workload). Nothing shipped._ |
| A | `tmp/perf-campaign-2/round13/marks/smoke_m4-1.txt.analysis.json` | 1 | _round13: both boots SPENT. Apparatus PROVEN on hardware; Gate A NOT MEASURED (stale model id ate the workload). Nothing shipped._ |
| A | `tmp/perf-campaign-2/round13/parse_worker_marks.py` | 2 | _round13: THIRD pre-boot apparatus defect caught -- A4 pairing was undefined; fixed pre-boot, still unmeasured_ |
| A | `tmp/perf-campaign-2/round13/test_parse_worker_marks.py` | 2 | _round13: THIRD pre-boot apparatus defect caught -- A4 pairing was undefined; fixed pre-boot, still unmeasured_ |
| A | `tmp/perf-campaign-2/round2/A1-EVAL-SEMANTICS.md` | 1 | _docs(perf): CAMPAIGN 2 R2 -- MoE stall is a true dependency: MLX has NO GPU collective (AllReduce::eval_gpu throws; jaccl stream is CPU); fence-bit-equiv story was an algebra bug; _ |
| A | `tmp/perf-campaign-2/round2/A2-JACCL-DEPENDENCY.md` | 1 | _docs(perf): CAMPAIGN 2 R2 -- MoE stall is a true dependency: MLX has NO GPU collective (AllReduce::eval_gpu throws; jaccl stream is CPU); fence-bit-equiv story was an algebra bug; _ |
| A | `tmp/perf-campaign-2/round2/A3-STREAM-BOUNDARY.md` | 1 | _docs(perf): CAMPAIGN 2 R2 -- MoE stall is a true dependency: MLX has NO GPU collective (AllReduce::eval_gpu throws; jaccl stream is CPU); fence-bit-equiv story was an algebra bug; _ |
| A | `tmp/perf-campaign-2/round2/A4-RECONCILIATION.md` | 1 | _docs(perf): CAMPAIGN 2 R2 -- MoE stall is a true dependency: MLX has NO GPU collective (AllReduce::eval_gpu throws; jaccl stream is CPU); fence-bit-equiv story was an algebra bug; _ |
| A | `tmp/perf-campaign-2/round2/A5-PIPELINING-FEASIBILITY.md` | 1 | _docs(perf): CAMPAIGN 2 R2 -- MoE stall is a true dependency: MLX has NO GPU collective (AllReduce::eval_gpu throws; jaccl stream is CPU); fence-bit-equiv story was an algebra bug; _ |
| A | `tmp/perf-campaign-2/round2/MECHANISM.md` | 1 | _docs(perf): CAMPAIGN 2 R2 -- MoE stall is a true dependency: MLX has NO GPU collective (AllReduce::eval_gpu throws; jaccl stream is CPU); fence-bit-equiv story was an algebra bug; _ |
| A | `tmp/perf-campaign-2/round2/REPORT.md` | 1 | _docs(perf): CAMPAIGN 2 R2 -- MoE stall is a true dependency: MLX has NO GPU collective (AllReduce::eval_gpu throws; jaccl stream is CPU); fence-bit-equiv story was an algebra bug; _ |
| A | `tmp/perf-campaign-2/round3/FEASIBILITY.md` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/REPORT.md` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/W1-jaccl-transport-buffer.md` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/dispatch-tasks.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/baseline_c1.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/baseline_needle_b2.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/baseline_needle_text.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/earlyflag_c1.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/earlyflag_c1_rep2.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/earlyflag_needle_b2.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/earlyflag_needle_text.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/earlyflag_needle_warm.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/offflag_c1.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/probe-results/restored_c1.json` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round3/r3_needle_capture.py` | 1 | _docs(perf): CAMPAIGN 2 R3 -- GPU-resident collective INFEASIBLE (no public GPU->DMA coherence API; jaccl host-bounce by construction); fence story dead on evidence; decode collecti_ |
| A | `tmp/perf-campaign-2/round4/PLAN.md` | 1 | _docs(perf): CAMPAIGN 2 R4 -- fence ARMED (>=98.5%, no regression); residual decode stall is GENUINE local compute (GPU 117% busy, identity all_sum <=8.4%); Q3 knob is DEAD CODE (ze_ |
| A | `tmp/perf-campaign-2/round4/REPORT.md` | 1 | _docs(perf): CAMPAIGN 2 R4 -- fence ARMED (>=98.5%, no regression); residual decode stall is GENUINE local compute (GPU 117% busy, identity all_sum <=8.4%); Q3 knob is DEAD CODE (ze_ |
| A | `tmp/perf-campaign-2/round4/q1-gate-audit.md` | 1 | _docs(perf): CAMPAIGN 2 R4 -- fence ARMED (>=98.5%, no regression); residual decode stall is GENUINE local compute (GPU 117% busy, identity all_sum <=8.4%); Q3 knob is DEAD CODE (ze_ |
| A | `tmp/perf-campaign-2/round4/results/boot1_macstudio-m4-1_diag_allsum.log` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/boot1_macstudio-m4-2_diag_allsum.log` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/boot2_clean_macstudio-m4-1.log` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/boot2_clean_macstudio-m4-2.log` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/boot3_identity_macstudio-m4-1.log` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/boot3_identity_macstudio-m4-2.log` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/needle_89k_boot1.json` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/needle_89k_boot2.json` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round4/results/needle_89k_identity.json` | 1 | _docs(perf): CAMPAIGN 2 R4 raw artifacts -- boot1/2/3 diag+probe extracts and needle JSONs_ |
| A | `tmp/perf-campaign-2/round5/PRE-REGISTRATION.md` | 1 | _round5: Phase 0 + PRE-REGISTRATION (before any sweep measurement)_ |
| A | `tmp/perf-campaign-2/round5/REPORT.md` | 1 | _round5(I5): speculative gamma re-tune -- NO SHIP, gamma=3 stands_ |
| A | `tmp/perf-campaign-2/round5/gamma-resolution-audit.md` | 1 | _round5: Phase 0 + PRE-REGISTRATION (before any sweep measurement)_ |
| A | `tmp/perf-campaign-2/round5/harness-map.md` | 1 | _round5: Phase 0 + PRE-REGISTRATION (before any sweep measurement)_ |
| A | `tmp/perf-campaign-2/round5/identity_gate.py` | 1 | _round5: Phase 0 + PRE-REGISTRATION (before any sweep measurement)_ |
| A | `tmp/perf-campaign-2/round5/measure_arm.py` | 2 | _round5(I5): speculative gamma re-tune -- NO SHIP, gamma=3 stands_ |
| A | `tmp/perf-campaign-2/round5/phase0-histogram.md` | 1 | _round5: Phase 0 + PRE-REGISTRATION (before any sweep measurement)_ |
| A | `tmp/perf-campaign-2/round5/results/arm_g3_boot1.json` | 1 | _round5(I5): speculative gamma re-tune -- NO SHIP, gamma=3 stands_ |
| A | `tmp/perf-campaign-2/round5/results/arm_g4_boot1.json` | 1 | _round5(I5): speculative gamma re-tune -- NO SHIP, gamma=3 stands_ |
| A | `tmp/perf-campaign-2/round5/results/arm_g4_boot1_fixed.json` | 1 | _round5(I5): speculative gamma re-tune -- NO SHIP, gamma=3 stands_ |
| A | `tmp/perf-campaign-2/round6/PRE-REGISTRATION-AMENDMENT-1.md` | 1 | _round6 Phase 0: calibration gate PASSED + amendment 1 (before any sweep arm)_ |
| A | `tmp/perf-campaign-2/round6/PRE-REGISTRATION.md` | 1 | _round6: PRE-REGISTRATION (before any measurement)_ |
| A | `tmp/perf-campaign-2/round6/REPORT.md` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/phase0-calibration.md` | 1 | _round6 Phase 0: calibration gate PASSED + amendment 1 (before any sweep arm)_ |
| A | `tmp/perf-campaign-2/round6/phase1-arms-B.md` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g2_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g2_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g2_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g2_warmup.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3A_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3A_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3A_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3A_warmup_stdout.log` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3B_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3B_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3B_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g3B_warmup.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g4_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g4_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g4_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g4_warmup.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g5_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g5_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g5_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/g5_warmup.json` | 1 | _docs(perf): CAMPAIGN 2 R6 -- HOLD at gamma=3; gamma=4 is SLOWER (-10.5%), round 5's +33% acceptance did not reproduce_ |
| A | `tmp/perf-campaign-2/round6/results/p0_2k.json` | 1 | _round6 Phase 0: calibration gate PASSED + amendment 1 (before any sweep arm)_ |
| A | `tmp/perf-campaign-2/round6/results/p0_2k_stdout.log` | 1 | _round6 Phase 0: calibration gate PASSED + amendment 1 (before any sweep arm)_ |
| A | `tmp/perf-campaign-2/round6/results/p0_deep_r1.json` | 1 | _round6 Phase 0: calibration gate PASSED + amendment 1 (before any sweep arm)_ |
| A | `tmp/perf-campaign-2/round6/results/p0_deep_r1_stdout.log` | 1 | _round6 Phase 0: calibration gate PASSED + amendment 1 (before any sweep arm)_ |
| A | `tmp/perf-campaign-2/round7/PRE-REGISTRATION-AMENDMENT-1.md` | 1 | _round7: AMENDMENT 1 (A2 short-prompt instrument) + arm P baseline, before arm Z relaunch_ |
| A | `tmp/perf-campaign-2/round7/PRE-REGISTRATION.md` | 1 | _round7: PRE-REGISTRATION + Task A/B read-throughs (before any measurement)_ |
| A | `tmp/perf-campaign-2/round7/REPORT.md` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/arm-bi0.md` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/armZ-and-biA.md` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/restore-and-armP2.md` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r10.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r4.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r5.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r6.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r7.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r8.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_short_r9.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_ttft_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_ttft_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_ttft_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_ttft_r4.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P2_ttft_r5.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/P_ttft_r1.json` | 1 | _round7: AMENDMENT 1 (A2 short-prompt instrument) + arm P baseline, before arm Z relaunch_ |
| A | `tmp/perf-campaign-2/round7/results/P_ttft_r2.json` | 1 | _round7: AMENDMENT 1 (A2 short-prompt instrument) + arm P baseline, before arm Z relaunch_ |
| A | `tmp/perf-campaign-2/round7/results/P_ttft_r3.json` | 1 | _round7: AMENDMENT 1 (A2 short-prompt instrument) + arm P baseline, before arm Z relaunch_ |
| A | `tmp/perf-campaign-2/round7/results/P_ttft_r4.json` | 1 | _round7: AMENDMENT 1 (A2 short-prompt instrument) + arm P baseline, before arm Z relaunch_ |
| A | `tmp/perf-campaign-2/round7/results/P_ttft_r5.json` | 1 | _round7: AMENDMENT 1 (A2 short-prompt instrument) + arm P baseline, before arm Z relaunch_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r10.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r4.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r5.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r6.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r7.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r8.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_short_r9.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_ttft_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_ttft_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_ttft_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_ttft_r4.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/Z_ttft_r5.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/bi0_id4k.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/bi0_id89k.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/biA_id4k.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/biA_id89k.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/biA_r1.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/biA_r2.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/biA_r3.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/biA_warmup.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/identity_bi0.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/results/identity_biA.json` | 1 | _docs(perf): CAMPAIGN 2 R7 -- steel-BI is CORRECTNESS-load-bearing at c=1 (byte-identity FAILS at both <8192 and 89K); rendezvous=0 safe but A1 delta +120ms out of band -> HOLD both_ |
| A | `tmp/perf-campaign-2/round7/taskA-armP.md` | 1 | _round7: AMENDMENT 1 (A2 short-prompt instrument) + arm P baseline, before arm Z relaunch_ |
| A | `tmp/perf-campaign-2/round7/taskA-readthrough.md` | 1 | _round7: PRE-REGISTRATION + Task A/B read-throughs (before any measurement)_ |
| A | `tmp/perf-campaign-2/round7/taskB-readthrough.md` | 1 | _round7: PRE-REGISTRATION + Task A/B read-throughs (before any measurement)_ |
| A | `tmp/perf-campaign-2/round8/I11-DECISION.md` | 1 | _round8: REPORT + I11-DECISION -- I12 all-SHARED (closed), I9 closed, I15 blocked; I11 premise REFUTED (experts already mxfp4 4-bit), no precision change recommended, nothing shippe_ |
| A | `tmp/perf-campaign-2/round8/I11-STEP1-KERNEL-MICROBENCH.md` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/I11-STEP1B-MODE-AND-3BIT.md` | 1 | _round8: I11 step 1b -- deployed mxfp4 mode check + 3-bit extension_ |
| A | `tmp/perf-campaign-2/round8/I11-STEP2-PRECISION-AND-DISK.md` | 1 | _round8: I11 step 2 -- deployed-precision ground truth + disk gate_ |
| A | `tmp/perf-campaign-2/round8/I12-AUDIT.md` | 1 | _round8: I12 serial-vs-batched prefill parity audit_ |
| A | `tmp/perf-campaign-2/round8/I15-I9-DIAGNOSTICS.md` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/PRE-REGISTRATION.md` | 1 | _round8: PRE-REGISTRATION (before any measurement) -- I12 audit bands, I15/I9 bands incl. the 200-500 inconclusive gap, I11 fast/slow + disk gate_ |
| A | `tmp/perf-campaign-2/round8/REPORT.md` | 1 | _round8: REPORT + I11-DECISION -- I12 all-SHARED (closed), I9 closed, I15 blocked; I11 premise REFUTED (experts already mxfp4 4-bit), no precision change recommended, nothing shippe_ |
| A | `tmp/perf-campaign-2/round8/i11_deployed_expert_dtype.py` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_full_results.json` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_full_run.log` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_gather_qmm_bits_microbench.py` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_kernel_capture.py` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_mem_instrument.py` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_mem_probe.py` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_oom_isolate.py` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_shape_probe.py` | 1 | _round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench_ |
| A | `tmp/perf-campaign-2/round8/i11_step1b_mode_and_3bit.py` | 1 | _round8: I11 step 1b -- deployed mxfp4 mode check + 3-bit extension_ |
| A | `tmp/perf-campaign-2/round8/i11b_results.json` | 1 | _round8: I11 step 1b -- deployed mxfp4 mode check + 3-bit extension_ |
| A | `tmp/perf-campaign-2/round8/i11b_run.log` | 1 | _round8: I11 step 1b -- deployed mxfp4 mode check + 3-bit extension_ |
| A | `tmp/perf-campaign-2/round8/i9_client.py` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/i9_client_result.json` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/i9_excerpt_decode.txt` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/i9_excerpt_idle.txt` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/i9_excerpt_prefill.txt` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/i9_idle_raw.txt` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/i9_parsed.json` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round8/i9_powermetrics_raw.txt` | 1 | _round8: I15 launch-count + I9 GPU-clock diagnostics_ |
| A | `tmp/perf-campaign-2/round9/PRE-REGISTRATION.md` | 1 | _round9: PRE-REGISTRATION + Task 1 code analysis -- predicted delta fixed at -200ms BEFORE any measurement_ |
| A | `tmp/perf-campaign-2/round9/REPORT.md` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/RESULTS-RAW.md` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/TASK1-CODE-ANALYSIS.md` | 1 | _round9: PRE-REGISTRATION + Task 1 code analysis -- predicted delta fixed at -200ms BEFORE any measurement_ |
| A | `tmp/perf-campaign-2/round9/boot1-armA-rv200.md` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/boot2-armZ1-rv0.md` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/boot3-armB-rv200.md` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/boot4-armZ2-rv0.md` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/A_2k_r1.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_2k_r2.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_2k_r3.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_2k_r4.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_2k_r5.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r1.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r10.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r2.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r3.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r4.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r5.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r6.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r7.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r8.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/A_short_r9.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/B_2k_r1.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_2k_r2.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_2k_r3.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_2k_r4.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_2k_r5.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r1.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r10.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r2.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r3.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r4.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r5.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r6.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r7.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r8.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/B_short_r9.json` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_2k_r1.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_2k_r2.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_2k_r3.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_2k_r4.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_2k_r5.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r1.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r10.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r2.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r3.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r4.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r5.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r6.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r7.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r8.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z1_short_r9.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_2k_r1.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_2k_r2.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_2k_r3.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_2k_r4.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_2k_r5.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r1.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r10.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r2.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r3.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r4.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r5.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r6.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r7.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r8.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/Z2_short_r9.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/boot1_env.txt` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/boot2_env.txt` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/boot3_env.txt` | 1 | _round9: boot3 (arm B, RV=200) results_ |
| A | `tmp/perf-campaign-2/round9/results/boot4_env.txt` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/boot5_env.txt` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_2k.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_2k.reasoning.txt` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_2k.txt` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_89k.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_89k.reasoning.txt` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_89k.txt` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_short.json` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_short.reasoning.txt` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV0_short.txt` | 1 | _round9: boot2 (arm Z first, RV=0) results + I15 ride-along_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_2k.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_2k.reasoning.txt` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_2k.txt` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_89k.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_89k.reasoning.txt` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_89k.txt` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_short.json` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_short.reasoning.txt` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identity_RV200_short.txt` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/results/identitym_RV0b4_2k.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/identitym_RV0b4_89k.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/identitym_RV0b4_short.json` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/results/selfcheck_RV0_89k_1.json` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/results/selfcheck_RV0_89k_2.json` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/results/selfcheck_RV0_89k_3.json` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/results/selfcheck_RV200_89k_1.json` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/results/selfcheck_RV200_89k_2.json` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/results/selfcheck_RV200_89k_3.json` | 1 | _round9: REPORT -- HOLD on RENDEZVOUS_MS=0 (band fails 2/3); 200-vs-480 EXPLAINED; byte-identity PASSES (R7 89K alarm was a false positive)_ |
| A | `tmp/perf-campaign-2/round9/run_boot.sh` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/run_identity.sh` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/run_identity_matched.sh` | 1 | _round9: boot4 (arm Z second, RV=0) + byte-identity gate + raw results table_ |
| A | `tmp/perf-campaign-2/round9/run_reps.sh` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/perf-campaign-2/round9/summarize.py` | 1 | _round9: boot1 (arm A, RV=200) results + paired-boot drivers_ |
| A | `tmp/prefill-round1-20260902/REPORT.md` | 1 | _docs(perf): PREFILL ROUND 1 -- 4 of 6 fable suggestions killed on evidence; LCP probe + Fix B workstream survives_ |
| A | `tmp/prefill-round1-20260902/findings/item2-instrumentation.md` | 1 | _docs(perf): PREFILL ROUND 1 -- 4 of 6 fable suggestions killed on evidence; LCP probe + Fix B workstream survives_ |
| A | `tmp/prefill-round1-20260902/findings/item34-provenance-and-envvars.md` | 1 | _docs(perf): PREFILL ROUND 1 -- 4 of 6 fable suggestions killed on evidence; LCP probe + Fix B workstream survives_ |
| A | `tmp/prefill-round1-20260902/findings/item6-indexer-and-gates.md` | 1 | _docs(perf): PREFILL ROUND 1 -- 4 of 6 fable suggestions killed on evidence; LCP probe + Fix B workstream survives_ |
| A | `tmp/prefill-round2-20260902/REPORT.md` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round2-20260902/findings/decode-discriminator.md` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round2-20260902/findings/lcp-probe.md` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round2-20260902/findings/lcp_probe.json` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round2-20260902/findings/sdpa-reopen.md` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round2-20260902/findings/survivorship.json` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round2-20260902/findings/survivorship.md` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round2-20260902/probe_lcp.py` | 1 | _docs(perf): PREFILL ROUND 2 -- LCP probe FAIL (pad at pos 0), SDPA closure invalid (4.06x per-call), decode ablation would hang_ |
| A | `tmp/prefill-round3-20260902/PRE-REGISTRATION.md` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/REPORT.md` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/a3_probe_lcp.py` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/a1_live_responses.json` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/a1_live_send.py` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/a1_offline_diff.md` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/a1_offline_render.json` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/a1_offline_render.py` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/decode_instrumentation.patch.REMOVED.md` | 1 | _hardening(round1): prune tmp/ 16G->34M, rsync --exclude tmp/, mx.eval bracket-close requirement, guard teeth audit_ |
| A | `tmp/prefill-round3-20260902/artifacts/deepseek_v4.instrumented.py` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/sdpa_2length_analyze.py` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/sdpa_2length_run.sh` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/artifacts/sdpa_2length_timing.patch` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/a1-api-verification.md` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/a3-lcp-reprobe.md` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/b-sdpa-2length-prep.md` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/c-decode-instrumentation-design.md` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/lcp_probe_round3.json` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/lcp_probe_round3_empty.json` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/lcp_probe_round3_omit.json` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefill-round3-20260902/findings/lcp_probe_round3_pad.json` | 1 | _docs(perf): PREFILL ROUND 3 -- pad-strip shipped (bdc9b6f1fc), production-proof cached_tokens 351-vs-0, Fix B CONDITIONAL-GO_ |
| A | `tmp/prefix-cache-dive-20260902/REPORT.md` | 1 | _docs(perf): PREFILL ROUND 1 -- 4 of 6 fable suggestions killed on evidence; LCP probe + Fix B workstream survives_ |
| A | `tmp/real-usage-capture-20260902/REPORT.md` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/partition_verified.json` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/FINDINGS.md` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/api_side.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/completion_side.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/derived.json` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/emit_points.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/eraB_rows.json` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/eraB_server_log.json` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/eraB_summary.json` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/log_schema_samples.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/metrics_full.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/pattern_counts.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/real_line_samples.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/schema_samples_draft.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/structured_samples.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/structured_samples2.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/raw/tail100.txt` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase1/requests.jsonl` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase2/README.md` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase2/capture.jsonl` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase2/passive_capture_proxy.py` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/real-usage-capture-20260902/phase2/self_test.py` | 1 | _docs(perf): REAL-USAGE CAPTURE -- gap is measurement convention; decode engine at 34 t/s (already meets 30+ target); lever is TTFT_ |
| A | `tmp/research-v1v2v3-20260901/REPORT.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/logmine/FEASIBILITY.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/logmine/V1_LOG_FINDINGS.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v2/V2_ANALYSIS.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v2/parse_mtp_segments.py` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v2/segments.csv` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/CODE_VERIFICATION.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/RESULTS.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/TELEMETRY_VALIDATION.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/V3_PROFILING_PLAN.md` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/before_marker.txt` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/bench_summary.json` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/bench_v3_100k.json` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/bench_v3_100k.log` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/exo_m4-1.log` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/exo_m4-2.log` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/gate_analysis.json` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/research-v1v2v3-20260901/v3/run1/probe.json` | 1 | _docs(perf): RESEARCH ROUND -- V1/V2/V3 outcomes (V2 infeasible-proven, V3 closed: rb_snap 0.218% of cycle)_ |
| A | `tmp/verify-decomposition-20260901/PREREGISTRATION.md` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/REPORT.md` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/VERIFY_ANATOMY.md` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/analyze_prof.py` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/build_table.py` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/PREREGISTRATION_ENTROPY.md` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/RESULTS.md` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/build_entropy_table.py` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_ab_driver.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_natural.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_natural.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_random.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_random.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_repetitive.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_repetitive.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_repetitive_recheck.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_repetitive_recheck.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/entropy_table.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_natural_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_natural_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_random_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_random_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_repetitive_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_repetitive_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_repetitive_recheck_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/errs_repetitive_recheck_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/nohup_ab.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_natural_after_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_natural_after_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_natural_before_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_natural_before_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_random_after_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_random_after_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_random_before_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_random_before_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_after_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_after_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_before_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_before_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_recheck_after_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_recheck_after_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_recheck_before_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/pids_repetitive_recheck_before_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_natural_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_natural_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_random_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_random_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_repetitive_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_repetitive_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_repetitive_recheck_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_anchor_repetitive_recheck_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_natural_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_natural_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_random_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_random_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_repetitive_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_repetitive_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_repetitive_recheck_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/raw/prof_ent_repetitive_recheck_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy/run_entropy_ab.sh` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/entropy_probe.py` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/accept_089k_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/accept_150k_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/accept_250k_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/bench_089k.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/bench_089k.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/bench_150k.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/bench_150k.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/bench_250k.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/bench_250k.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/driver.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/marker_089k_after.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/marker_089k_before.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/marker_150k_after.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/marker_150k_before.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/marker_250k_after.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/marker_250k_before.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/nohup.log` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/prof_089k_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/prof_089k_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/prof_150k_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/prof_150k_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/prof_250k_n1.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/prof_250k_n2.txt` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/table_full.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/raw/table_partial.json` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/run_entropy.sh` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/run_scan.sh` | 1 | _docs(perf): ENTROPY A/B -- baseline not inflated by prompt entropy; 20 vs 34 gap still unexplained_ |
| A | `tmp/verify-decomposition-20260901/temperature/PREREGISTRATION_TEMPERATURE.md` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/RESULTS.md` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/anchor_gap_check.py` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/errs_temp10_n1.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/errs_temp10_n2.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/gpu_samples_temp10.log` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/pids_temp10_after_n1.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/pids_temp10_after_n2.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/pids_temp10_before_n1.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/pids_temp10_before_n2.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/prof_anchor_temp10_n1.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/prof_anchor_temp10_n1_2MB.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/prof_anchor_temp10_n2.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/prof_anchor_temp10_n2_2MB.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/prof_temp10_n1.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/prof_temp10_n2.txt` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/sanity_body.json` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/sanity_check.py` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/temp_arm_driver.log` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/temp_temp10.json` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/raw/temp_temp10.log` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/run_temp_arm.sh` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/verify-decomposition-20260901/temperature/temp_probe.py` | 1 | _docs(perf): TEMPERATURE ARM -- hypothesis dies at premise (no temp delta exists); convention suspect primary_ |
| A | `tmp/wall-attribution-20260902/REPORT.md` | 1 | _docs(perf): WALL-ATTRIBUTION -- gap is 5% not 11-16%; true acceptance 1.411; entropy-A/B-stale note_ |

---

## mlx  (vs ml-explore/mlx, merge-base `b6642e604`)

**71 files changed.**


### CORE mlx (C++/Metal backend)  (60 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `mlx/array.cpp` | 4 | mentioned in: api.md, fork-notes.md |
| M | `mlx/array.h` | 4 | sync-notes-mlx: combined - keep our live_array_desc_count diagnostics; drop local `Deleter` alias (upstream #4634 moved it to allocator::Deleter)  (prior: mentioned in: api.md, fork-notes.md) |
| M | `mlx/backend/metal/allocator.cpp` | 2 | mentioned in: fork-notes.md, minimax-decode-optimization.md |
| M | `mlx/backend/metal/allocator.h` | 6 | mentioned in: fork-notes.md, minimax-decode-optimization.md |
| M | `mlx/backend/metal/device.cpp` | 4 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/backend/metal/device.h` | 2 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/backend/metal/device_info.cpp` | 1 | _fix(allocator): bump rsrc_limit fallback 499000 -> 5000000_ |
| M | `mlx/backend/metal/eval.cpp` | 5 | mentioned in: skill:mlx-source-read-falsification-pattern.md, skill:session-discipline-pitfalls.md |
| M | `mlx/backend/metal/event.cpp` | 11 | mentioned in: upstream-prs.md, api.md |
| M | `mlx/backend/metal/fence.cpp` | 4 | sync-notes-mlx: upstream base (#4552 value-based API, #4689 end_encoding spin isolation) + our DSB SY read/write barriers on fast CPU path  (prior: fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md) |
| M | `mlx/backend/metal/jit_kernels.cpp` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/kernels.h` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/kernels/CMakeLists.txt` | 2 | sync-notes-mlx: combined - upstream sdpa_blocked/vjp/gated_delta kernels + our sdpa_chunked_reduce  (prior: mentioned in: skill:2026-05-26-mlx-upstream-sync-port.md, skill:upstream-sync-triage.md) |
| M | `mlx/backend/metal/kernels/fence.metal` | 5 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/backend/metal/kernels/fp_quantized.h` | 6 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/kernels/fp_quantized.metal` | 4 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/kernels/quantized.h` | 3 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/kernels/quantized.metal` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/kernels/scaled_dot_product_attention.metal` | 3 | sync-notes-mlx: combined - our quant 2pass instantiations + upstream tk1 instantiations  (prior: minimax-fused-attention-design.md / upstream-pr-drafts/02,04) |
| A | `mlx/backend/metal/kernels/sdpa_vector_quant.h` | 5 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `mlx/backend/metal/kernels/steel/attn/kernels/sdpa_chunked_reduce.h` | 2 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `mlx/backend/metal/kernels/steel/attn/kernels/sdpa_chunked_reduce.metal` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| M | `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h` | 1 | sync-notes-mlx: upstream save_lse (fc 303); our chunked path ported to save_lse/303 + _bv naming + upstream tile table  (prior: minimax-fused-attention-design.md / upstream-pr-drafts/02,04) |
| M | `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal` | 4 | sync-notes-mlx: upstream instantiation table (_bv, bd256 wn=2) + our D512 8x8 re-expressed with bv=512  (prior: minimax-fused-attention-design.md / upstream-pr-drafts/02,04) |
| M | `mlx/backend/metal/matmul.cpp` | 3 | sync-notes-mlx: upstream thin-GEMM Case 2 + NAX split-K Case 3; our steel_batch_invariant_enabled() pin applied to both |
| M | `mlx/backend/metal/metal.h` | 3 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/backend/metal/no_metal.cpp` | 2 | _metal: gpu_time_ns() — accumulate GPU-busy time from MTLCommandBuffer timestamps_ |
| M | `mlx/backend/metal/nojit_kernels.cpp` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/quantized.cpp` | 15 | rationale documented in the Fork-rationales section |
| M | `mlx/backend/metal/resident.cpp` | 2 | iogpu-residency-set-abort.md |
| M | `mlx/backend/metal/scaled_dot_product_attention.cpp` | 16 | sync-notes-mlx: combined (9 hunks) - our chunk config + upstream blocked D512; chunked dispatch ported to fc303/_bv/tile table  (prior: minimax-fused-attention-design.md / upstream-pr-drafts/02,04) |
| M | `mlx/backend/metal/utils.h` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/distributed.cpp` | 2 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/distributed.h` | 2 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/distributed_impl.h` | 2 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/jaccl/jaccl.cpp` | 5 | sync-notes-mlx: took upstream get_mesh_connectivity() (vector-per-wire), pass mesh.at(0) into our MeshGroup ctor = pre-#4607 single-cable behaviour  (prior: fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md) |
| M | `mlx/distributed/jaccl/lib/examples/CMakeLists.txt` | 4 | rationale documented in the Fork-rationales section |
| A | `mlx/distributed/jaccl/lib/examples/coord_group_loopback_test.cpp` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx/distributed/jaccl/lib/examples/hopid_budget_probe.cpp` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx/distributed/jaccl/lib/examples/uc_rdma_write_probe.cpp` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx/distributed/jaccl/lib/jaccl/coord_group.h` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/jaccl/lib/jaccl/group.h` | 4 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/jaccl.cpp` | 2 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/jaccl/lib/jaccl/mesh.cpp` | 28 | sync-notes-mlx: keep-ours (fork MeshGroup is ~6k lines reliable/ACK/pool/p2p-retry QP machinery; #4607 multi-wire restructure not mechanically portable) DEFERRED  (prior: fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md) |
| M | `mlx/distributed/jaccl/lib/jaccl/mesh.h` | 13 | sync-notes-mlx: keep-ours (fork MeshGroup; #4607 multi-wire not portable) DEFERRED  (prior: fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md) |
| M | `mlx/distributed/jaccl/lib/jaccl/mesh_impl.h` | 68 | sync-notes-mlx: keep-ours (fork MeshGroup; #4607 multi-wire not portable) DEFERRED  (prior: fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md) |
| M | `mlx/distributed/jaccl/lib/jaccl/rdma.cpp` | 4 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/rdma.h` | 14 | fork-notes.md (JACCL/RDMA) / iogpu-residency-set-abort.md |
| M | `mlx/distributed/jaccl/lib/jaccl/ring.cpp` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/jaccl/lib/jaccl/ring.h` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/jaccl/lib/jaccl/ring_impl.h` | 3 | sync-notes-mlx: auto-merged - #4443 mirroring fix once; our p2p send/recv buffer class (4KB cap) is distinct code; #4607 dispatch_wires->threadpool.h moved cleanly |
| M | `mlx/distributed/jaccl/lib/jaccl/tcp.cpp` | 8 | rationale documented in the Fork-rationales section |
| M | `mlx/distributed/jaccl/lib/jaccl/tcp.h` | 4 | rationale documented in the Fork-rationales section |
| M | `mlx/fast.cpp` | 2 | mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md |
| M | `mlx/fast.h` | 1 | sync-notes-mlx: combined - keep sdpa_quant decl + upstream gated_delta_update decl  (prior: mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md) |
| M | `mlx/fast_primitives.h` | 1 | mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md |
| M | `mlx/ops.cpp` | 1 | mentioned in: skill:collective-coordination-levers.md, skill:mlx-cross-build-abi-trap.md |
| M | `mlx/scheduler.cpp` | 2 | mentioned in: docs/incidents/fe92a0565-jaccl-fault-shadowing-2026-09-10.md |
| M | `mlx/scheduler.h` | 7 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx/transforms.cpp` | 6 | mentioned in: fork-notes.md, skill:mtp-bistability-2026-05-17-investigation.md |

### mlx bindings/tests  (10 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `python/src/distributed.cpp` | 3 | mentioned in: fork-notes.md, kv-cache-architecture.md |
| M | `python/src/fast.cpp` | 1 | mentioned in: minimax-quantized-sdpa-design.md, minimax-fused-attention-design.md |
| M | `python/src/metal.cpp` | 4 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `python/src/transforms.cpp` | 8 | mentioned in: fork-notes.md, skill:mtp-bistability-2026-05-17-investigation.md |
| A | `python/tests/test_fast_sdpa_quant.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `python/tests/test_sdpa_chunked.py` | 3 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| A | `python/tests/test_sdpa_logsumexp.py` | 1 | minimax-fused-attention-design.md / upstream-pr-drafts/02,04 |
| M | `tests/CMakeLists.txt` | 2 | _test(jaccl): add an eval-driving regression guard for the combined fault path_ |
| A | `tests/eval_combined_fault_tests.cpp` | 1 | _test(jaccl): add an eval-driving regression guard for the combined fault path_ |
| A | `tests/scheduler_combined_fault_tests.cpp` | 2 | _fix(tests): drive a real CPU stream instead of constructing StreamThread (now .cpp-private)_ |

### Other  (1 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `docs/incidents/fe92a0565-jaccl-fault-shadowing-2026-09-10.md` | 2 | _docs(jaccl): record the 2026-09-11 verification and correct section 6_ |

---

## mlx-lm  (vs ml-explore/mlx-lm, merge-base `93e6d4a6`)

**245 files changed.**


### CORE mlx-lm (model source)  (67 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `mlx_lm/generate.py` | 28 | mentioned in: upstream-prs.md, api.md |
| M | `mlx_lm/models/base.py` | 3 | sync-notes-mlxml: gather_last_axis added (upstream)  (prior: mentioned in: fork-notes.md, minimax-decode-optimization.md) |
| M | `mlx_lm/models/cache.py` | 69 | sync-notes-mlxml: upstream ArraysCache.make_mask change present (line ~1028)  (prior: kv-cache-architecture.md) |
| A | `mlx_lm/models/deepseek_v4.py` | 270 | sync-notes-mlxml: fork module; deepseek_v4 dispatch unaffected  (prior: deepseek-v4-mtp-performance.md / mtp-tiebreak-losslessness-fix.md / fork-notes.md) |
| D | `mlx_lm/models/deepseek_v41.py` | 0 | sync-notes-mlxml: COLLISION - upstream single-module deepseek_v41 vs fork PACKAGE (model_type deepseek_v41); FileFinder confirms PACKAGE WINS -> git rm upstream .py |
| A | `mlx_lm/models/deepseek_v41/LICENSE` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/README.md` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/__init__.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/_gates.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/attention.py` | 12 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/cache.py` | 4 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/collective.py` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/compressor.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/config.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/dequant.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/engram.py` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/exl3_build.py` | 13 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/fakequant.py` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/hc_fused.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/hyper_connections.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/image_processor.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/indexer.py` | 21 | mentioned in: docs/dsv41-hierarchical-indexer-design-2026-10-04.md |
| A | `mlx_lm/models/deepseek_v41/indexer_hierarchical.py` | 4 | mentioned in: docs/dsv41-hierarchical-indexer-design-2026-10-04.md |
| A | `mlx_lm/models/deepseek_v41/layers.py` | 3 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/logprobs.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/model.py` | 11 | mentioned in: docs/benchmarks/wsC-prefill-2026-09-28/README.md |
| A | `mlx_lm/models/deepseek_v41/moe.py` | 8 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/mtp.py` | 6 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/prefill.py` | 8 | mentioned in: docs/benchmarks/wsC-prefill-2026-09-28/README.md |
| A | `mlx_lm/models/deepseek_v41/sampling.py` | 3 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/serve.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/session_cache.py` | 5 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/sparse_attention.py` | 10 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/deepseek_v41/spec.py` | 8 | mentioned in: docs/benchmarks/wsC-prefill-2026-09-28/README.md |
| A | `mlx_lm/models/deepseek_v41/vision.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/LICENSE` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/NOTICE` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/README.md` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/__init__.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/decode.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/exl3_linear.py` | 3 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/exl3_moe.py` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/gemv_metal.py` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/hadamard.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/layer_state.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/loader.py` | 2 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/metal_kernels.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/ops.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/perm.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/reconstruct.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/ref/__init__.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/ref/codebook.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/ref/layer.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/ref/perm.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/ref/signs.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/signs.py` | 1 | rationale documented in the Fork-rationales section |
| A | `mlx_lm/models/exl3/stripe.py` | 1 | rationale documented in the Fork-rationales section |
| M | `mlx_lm/models/gated_delta.py` | 13 | sync-notes-mlxml: COMBINE - keep fork f32 beta (mx.sigmoid(b.astype(f32))) + add upstream `if allow_neg_eigval: beta *= 2.0`  (prior: mentioned in: prefill-optimization.md) |
| A | `mlx_lm/models/hyper_connection.py` | 5 | mentioned in: skill:mx-compile-patterns.md, skill:2026-05-18-compile-boundary-regressions.md |
| M | `mlx_lm/models/minimax.py` | 4 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| A | `mlx_lm/models/minimax_trace.py` | 2 | minimax-*.md (fused-attention/quantized-sdpa/decode) |
| M | `mlx_lm/models/qwen3_5.py` | 17 | sync-notes-mlxml: KEEP OURS (fork manual-rsqrt L2 norm, dmcc73 2026-03-11); dropped unused normalize_qk import  (prior: mentioned in: upstream-prs.md, fork-notes.md) |
| M | `mlx_lm/models/switch_layers.py` | 4 | sync-notes-mlxml: fork BatchedSwitchGLU/profiler spans + upstream nvfp4 global_scale both present  (prior: mentioned in: upstream-prs.md, skill:custom-metal-kernel-authorship.md) |
| A | `mlx_lm/profiler.py` | 3 | mentioned in: upstream-prs.md, fork-notes.md |
| M | `mlx_lm/sample_utils.py` | 1 | sync-notes-mlxml: fork count-aware repetition penalty + upstream #1912 apply_top_p fix both present (separate functions) |
| M | `mlx_lm/tokenizer_utils.py` | 1 | sync-notes-mlxml: fork BPE detok + upstream TokenizerWrapper/response_format in disjoint regions |
| M | `mlx_lm/utils.py` | 14 | sync-notes-mlxml: KEEP BOTH - fork _load_safetensors + upstream infer_quant_config; DSv4 fp8 branch reordered specific-first  (prior: mentioned in: upstream-prs.md, fork-notes.md) |

### mlx-lm tests  (53 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `tests/_dsv4_gate_baseline_64cc7e6.py` | 1 | _feat(dsv4-vision): Phase 3a — bias_vl second MoE routing bias for image tokens_ |
| A | `tests/_dsv4_torch_reference.py` | 1 | _feat(dsv4-vision): Phase 3a — bias_vl second MoE routing bias for image tokens_ |
| A | `tests/p65_session.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `tests/p66_seam.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `tests/p67_serve.py` | 2 | _fix(dsv41): complete the spec.snap call-site sweep — p67 replay oracle + pW21/pW24_ |
| A | `tests/parity/README.md` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `tests/parity/dsv41_parity.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `tests/parity/parity_core.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `tests/parity/run_two_node.sh` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `tests/parity/test_parity_core.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `tests/test_batch_pooling_cache_overlap.py` | 1 | _fix(batchpooling): keep overlap-carry structs in sync with batch width_ |
| D | `tests/test_deepseek_v41.py` | 0 | &mdash; |
| A | `tests/test_deepseek_v41_vision.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `tests/test_deepseek_v4_attention_visibility.py` | 3 | _test(dsv4-vision): pin vl-gate coverage in the flag ON/OFF bitwise-identity test_ |
| A | `tests/test_deepseek_v4_embed_bounds_check.py` | 1 | _fix(dsv4-vision): defense-in-depth bounds check at embed_tokens gather_ |
| A | `tests/test_deepseek_v4_gate.py` | 1 | _feat(dsv4-vision): Phase 3a — bias_vl second MoE routing bias for image tokens_ |
| A | `tests/test_deepseek_v4_gate_vl_invariant.py` | 1 | _test(dsv4-vision): Phase 3 GAPS 3/4/5 — oracle provenance, nonzero bias_vl, hermetic invariant_ |
| A | `tests/test_deepseek_v4_reference_provenance.py` | 1 | _test(dsv4-vision): Phase 3 GAPS 3/4/5 — oracle provenance, nonzero bias_vl, hermetic invariant_ |
| A | `tests/test_deepseek_v4_visibility_attention_output.py` | 3 | _fix(dsv4-vision): defense-in-depth bounds check at embed_tokens gather_ |
| A | `tests/test_deepseek_v4_visibility_fastpaths.py` | 3 | _fix(dsv4-vision): defense-in-depth bounds check at embed_tokens gather_ |
| A | `tests/test_deepseek_v4_vision.py` | 3 | _fix(tests): generalize submodule-provenance guard to any non-site-packages path_ |
| A | `tests/test_deepseek_v4_vision_tp_replication.py` | 1 | _test(dsv4-vision): pin ViT+Aligner TP replication (Phase 4e)_ |
| A | `tests/test_dsv41_cache_growth.py` | 1 | _perf(dsv41): bf16 grid-exact caches + grow-on-demand capacity_ |
| A | `tests/test_dsv41_consumer_skip.py` | 1 | _perf(dsv41): consumer index layers score only candidate blocks in the coarse pass_ |
| A | `tests/test_dsv41_dense_affine_shard.py` | 2 | _feat(dsv41): per-tensor dense-quant policy DSV41_DENSE_POLICY (mixed dense formats)_ |
| A | `tests/test_dsv41_dense_policy.py` | 1 | _feat(dsv41): per-tensor dense-quant policy DSV41_DENSE_POLICY (mixed dense formats)_ |
| A | `tests/test_dsv41_engram_read.py` | 2 | _perf(dsv41): make coarse engram reads and always-tiled indexer the defaults_ |
| A | `tests/test_dsv41_fence_hook.py` | 1 | _feat(dsv41): fence-point liveness hook for prefill (Fix A)_ |
| A | `tests/test_dsv41_gather_direct.py` | 2 | _perf(dsv41): column-split gather + async tile fence (both gates default off)_ |
| A | `tests/test_dsv41_hier_integration.py` | 2 | _perf(dsv41): promote DSV41_INDEXER_HIER to default-ON (M2 shipped); =0 forces OFF for A/B_ |
| A | `tests/test_dsv41_indexer_adversarial_prodh.py` | 1 | _test(dsv41): adversarial cell suite at production H + attribution artifact + 91K fp32-row cost_ |
| A | `tests/test_dsv41_indexer_bf16_row.py` | 2 | _perf(dsv41): promote DSV41_INDEXER_HIER to default-ON (M2 shipped); =0 forces OFF for A/B_ |
| A | `tests/test_dsv41_indexer_hierarchical.py` | 1 | _perf(dsv41): prototype hierarchical/streamed indexer exact pass (M1)_ |
| A | `tests/test_dsv41_indexer_qsplit.py` | 4 | _fix(dsv41): indexer qsplit D3 — key the tiled/untiled route on the FULL chunk_ |
| A | `tests/test_dsv41_indexer_smallm_hier.py` | 1 | _test(dsv41): index-identity proof for the lever-2 row-count guard + real-tensor capture harness_ |
| A | `tests/test_dsv41_indexer_smalln_wide_tile.py` | 1 | _fix(dsv41): small-n wide tile — fix the 128MiB route cliff (idxprof5/6)_ |
| A | `tests/test_dsv41_logprobs.py` | 1 | _feat(dsv41): exact top-k log-probs from the vocab-sharded head_ |
| A | `tests/test_dsv41_prefill_spans.py` | 1 | _perf(dsv41): profiler spans for prefill attribution (no-op when hook unset)_ |
| A | `tests/test_dsv41_q1b_warmup_liveness.py` | 1 | _fix(dsv41): load_warmup forwards fence_hook (liveness) + AffineProj releases build residue (ROUND-Q1B)_ |
| A | `tests/test_dsv41_rope_percall.py` | 1 | _perf(dsv41): per-call RoPE (drop per-layer cos/sin tables)_ |
| A | `tests/test_dsv41_sampling.py` | 2 | _fix(dsv41): spec rollback must restore the window ring + the compressed rows_ |
| A | `tests/test_dsv41_session_cache.py` | 2 | _perf(dsv41): bf16 grid-exact caches + grow-on-demand capacity_ |
| A | `tests/test_dsv41_snapshot_prune.py` | 1 | _session_cache: never prune the offset-0 checkpoint_ |
| A | `tests/test_dsv41_sparse_opt_gates.py` | 2 | _perf(dsv41): promote sparse-attn winner defaults — QTILE 64->256, BUDGET 64->256 MB, PREFILL_CHUNK 64->256, COLSPLIT default-ON_ |
| A | `tests/test_dsv41_sparse_prefill_chunk.py` | 2 | _perf(dsv41): promote sparse-attn winner defaults — QTILE 64->256, BUDGET 64->256 MB, PREFILL_CHUNK 64->256, COLSPLIT default-ON_ |
| A | `tests/test_dsv41_sparse_smallm_colsplit.py` | 1 | _perf(dsv41): gate C1 column-boundary derivation on call row count_ |
| A | `tests/test_dsv41_spec_rollback_ring.py` | 2 | _feat(dsv41): env-gated TP=2 rank-consistency hash on the spec rollback path_ |
| A | `tests/test_dsv41_vision_masks.py` | 1 | _fix(dsv41): give image-span rows the reference's VL routing + engram masks_ |
| A | `tests/test_hc_expand_kernel.py` | 1 | _perf(hc_expand): env-gated fused Metal kernel for HyperConnection expand_ |
| M | `tests/test_models.py` | 22 | _Simplify HyperConnection_ |
| M | `tests/test_prompt_cache.py` | 3 | _Rebind BatchKVCache offset instead of mutating in place. (#1848)_ |
| M | `tests/test_sample_utils.py` | 1 | _fix(sample): make repetition_penalty count-aware to break decode loops_ |
| M | `tests/test_tokenizers.py` | 1 | _fix(detok): retain incomplete multibyte prefix in BPEStreamingDetokenizer_ |

### Benchmarks & harnesses  (121 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `bench/README.md` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/README_U.md` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/RUNBOOK_V.md` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/V_DEDUP_DESIGN.md` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/next18_attribution.py` | 1 | _test(dsv41): adversarial cell suite at production H + attribution artifact + 91K fp32-row cost_ |
| A | `bench/next18_capture.py` | 2 | _next18 capture: flush OFF the request thread + runtime timing-arm gate (R1b)_ |
| A | `bench/next18_classify.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_classify_all.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_determinism.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_diffdiag.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_fp32row_91k.py` | 1 | _test(dsv41): adversarial cell suite at production H + attribution artifact + 91K fp32-row cost_ |
| A | `bench/next18_g0_probe.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_gap_hist.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_headcount.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_l2full_perf.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_losscell.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_prodgate.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_replay_capture.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/next18_repro.py` | 1 | _perf(dsv41): lever-2 L2-full — small-n fallback stores an fp32 score row (frozen suite FAILs 251/2045, all residual a H=2 fixture tie artifact)_ |
| A | `bench/pD1_dense.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD2_moe.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD3_probe.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD4_moe.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD5_guard.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD6_bm.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD7_decode.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD8_cross.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pD9_variants.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDA_cross_band.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDB_prefill_ab.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDC_model_ab.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pDD_segment.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDE_segment.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDG_transients.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDH_head.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDI_invariance.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDJ_shape_cross.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDK_fallback.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDL_verify.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pDM_scale.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/pU0_memprobe.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU0c_probe.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pU0f_full_mem.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pU0m_probe.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pU2_divergence.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU3_decompose.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU4_collectives.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU5_identity.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU6_collectives.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU7_two_rank.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU8_census.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pU9_draftcost.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pU_cap.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV10_sweep.sh` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV11_ceiling.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV12_decode.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV13_parity.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pV14_ceiling2.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV15_a2vsdense.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV16_head.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV17_headbw.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV18_attn.py` | 3 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pV19_groups.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV2_census.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pV3_matrix.sh` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV4_verify.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pV5_moe.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV6_phases.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pV7_attrib.py` | 2 | _chore(dsv41): pass the verify row count to spec.snap in the pU/pV bench probes_ |
| A | `bench/pV8_modules.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV9_dedup.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV_route_stats.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/pV_verify.py` | 1 | _feat(deepseek_v41): round-2 workstreams + two-node stability fixes_ |
| A | `bench/v19d.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/v19e.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `bench/v19f.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `benchmarks/dsv41/NOTES-W.md` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pB_attic.sh` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pB_attic_one.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pB_decode_smoke.sh` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pB_edge.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pB_indexer_test.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pB_memprobe.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pB_tilecmp.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW10_scale.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW11_prof.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW12_prime.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW13_allocbench.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW14_faults.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW15_pricenewbuf.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW16_ab.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW17_fullscale.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW18_scale_parity.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW19_prime_exact.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW1_compile.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW20_localize30.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW21_whathelps.py` | 2 | _fix(dsv41): complete the spec.snap call-site sweep — p67 replay oracle + pW21/pW24_ |
| A | `benchmarks/dsv41/pW22_gc.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW23_profile30.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW24_absorb.py` | 2 | _fix(dsv41): complete the spec.snap call-site sweep — p67 replay oracle + pW21/pW24_ |
| A | `benchmarks/dsv41/pW25_fence.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW26_validate.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW27_accept.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW28_parity.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW29_session.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW2_clearcache.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW3_localize.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW4_census.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW5_alloc.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW6_allocops.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW7_pool.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW8_pressure.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/dsv41/pW9_sizes.py` | 1 | _chore(dsv41): move root-level investigation scripts to benchmarks/dsv41_ |
| A | `benchmarks/p97_bf16_control.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `benchmarks/p97_integration.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `benchmarks/p97_perf.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `benchmarks/p97_torch_selfconsistency.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `benchmarks/p97_trace.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `benchmarks/p97_vision_parity.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `harness/pF0_greedy_parity.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `harness/pF1_sampling_live.py` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |

### Documentation (the docs themselves)  (3 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| A | `docs/benchmarks/wsC-prefill-2026-09-28/README.md` | 1 | _feat(deepseek_v41): merge prefill/session/sampling/vision workstreams_ |
| A | `docs/dsv41-hierarchical-indexer-design-2026-10-04.md` | 2 | _docs(dsv41): correct the M2 strip-size formula (gather pair 6d+4, not 4+4d)_ |
| A | `docs/dsv41-hierarchical-indexer-m2-integration-2026-10-05.md` | 1 | _perf(dsv41): integrate hierarchical/streamed indexer exact pass (M2)_ |

### Build/meta  (1 files)

| S | File | #c | Rationale / Doc |
|---|------|----|-----------------|
| M | `pyproject.toml` | 0 | mentioned in: upstream-prs.md, fork-notes.md |

---

## Files no longer divergent since the previous generation

These were listed in the 2026-06-04 generation but are absent from the post-sync
diff (their content now matches the upstream merge-base, or the path no longer
exists in both trees):

- `eagle_k1_fix_report.md` - removed from the fork tree: absent from both
  `origin/main` and `upstream/main`, so no longer a divergence.
- `mlx/distributed/jaccl/jaccl.h` - content now identical to upstream: the path
  is present and byte-identical in both `b17781838` and `b6642e604`.

---

## Coverage summary

- Total changed files across 3 repos: **2318**
- CORE source files (`src/exo/`, `rust/`, `mlx/`, `mlx_lm/`): **395**, of which **235** carry a rationale (thematic doc, one-line rationale, or sync-notes trace).
- Remaining 1923 files are benchmarks, campaign scratch (`tmp/`), model cards, ops scripts, dashboard, build/meta, docs, or session scratch - self-describing by name + commit message.

---

## Appendix: core changes not covered by a thematic doc

These core-source files are not referenced by name in any `docs/` writeup; rationale
recorded here so the inventory has no unexplained core file group. The Qwen3.5-MoE
kernel patch set and the master/routing/sampling group below are the long-standing
entries from the previous generation and remain valid.

### Qwen3.5 / Qwen3.5-MoE engine patches - `src/exo/worker/engines/mlx/patches/`
Custom MLX kernel + monkey-patch set for the *secondary* Qwen3.5-397B model path
(distinct from the DSv4-Flash champion). Two efforts:
- **MTP + kernel patches for Qwen3.5** (`custom_qmv_loop_over_b.py`, `lpb_patch.py`,
  `gdn_step_precomputed.py`): port of MTP speculative decoding + GDN (gated delta-net)
  step kernels to the Qwen3.5 engine.
- **bf16->fp16 compute-dtype switch for ~7% faster quantized_matmul** (the 7
  `*_8bit.py` / `fused_*` kernels): batched fused GDN projections, merged down-proj,
  MoE epilogue, o-proj gate GEMV, softmax-topk-swiglu, fused RMSNorm-gated.
- Status: Qwen3.5 path is gated off by default (`QWEN35_ENABLED=0` in start_cluster.sh);
  these are fork-only, model-specific, not upstreamable (no Qwen3.5 engine upstream).

### Master / routing / sampling - `src/exo/master/`, `src/exo/shared/`
- `master/placement_utils.py` - `EXO_PP_LAYER_SPLIT` env for manual pipeline-stage layer
  rebalancing (tune shard boundaries across the 2 nodes).
- `master/tests/test_master.py` - updated for synchronous event indexing (see the
  sync-notes-exo entry above).
- `master/tests/test_routing_concurrency.py` - new test for sibling-instance routing
  contention (c>=2 era; harness for the in-flight-task load-balance fix).
- `shared/types/text_generation.py` - adds a per-model-card sampling-defaults tier
  between per-instance and cluster-env (relates to exo issue #1987).

### Worker plan - `src/exo/worker/plan.py`
Threads `node_network` through `plan()` into `_model_needs_download()` so download
decisions are network-topology aware (P2P model-distribution support); see the
sync-notes-exo entry above.

### mlx backend - `mlx/backend/metal/`
- `device_info.cpp` - bumped the Metal resource-limit fallback (`rsrc_limit`
  499000 -> 5000000) so large DSv4 shards don't hit a stale cap. Fork-only tuning.
- `no_metal.cpp` - stub side of `gpu_time_ns()` (GPU-busy time from MTLCommandBuffer
  timestamps); the profiling instrument's no-Metal build path.

---

## Fork rationales (previously-unsourced core files)

All 160 core files previously flagged with no sourced rationale (exo 91, mlx 22,
mlx-lm 47) received evidence-sourced one-line rationales on 2026-10-16 via a
commit-history pass: introducing/representative commits verified, cited docs verified
to exist, zero entries left unsourced. Sources are the introducing commit, in-code
fork comments, and the fork docs/ tree.

### exo (91 files)

**`rust/networking/src/`**

- `lib.rs` — zenoh static-peering escape hatch (EXO_ZENOH_CONNECT) + native-v4 locator for the macOS TCC LocalNetwork discovery failure (source: ea8456265 fix(discovery): route peer discovery over Tailscale to survive macOS 27 TCC LocalNetwork restriction)

**`src/exo/api/`**

- `phase_marks.py` — API-process request-path phase marks (a1-a7) behind the EXO_PHASE_MARKS gate (source: docs/PERFORMANCE_HISTORY.md)

**`src/exo/api/tests/`**

- `test_api_error_handling.py` — regression: non-streaming chat completions must return a real 500, not an empty 200, on an ErrorChunk (source: 89543e7bd fix(api): non-streaming chat completions returned empty 200 on error)
- `test_chat_completions_stream.py` — error-stream shape: structured error_code on a context-length refusal vs 500 fallback for plain errors (source: b8b3d77b2 fix(dsv41): make the cache-capacity refusal client-classifiable)
- `test_correlation_id.py` — client-supplied correlation_id identifies in-flight commands before the first chunk (source: b511427c8 feat(api): client-supplied correlation_id to identify in-flight commands pre-first-chunk)
- `test_image_error_chunk_routing.py` — fixture seeds fork-only API fields (_master_node_id, node_id, _jit_inflight_loads) for image-error routing (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_interrupted_generation_streams.py` — fixture seeds fork-only API fields (_master_node_id, _jit_inflight_loads) for interrupted-stream errors (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_reasoning_effort_clamp.py` — over-ceiling reasoning_effort is clamped to the wire ceiling instead of 422 (source: 1a54be214 fix(api): clamp over-ceiling reasoning_effort to the wire ceiling instead of 422)
- `test_seed_resolution.py` — seedless requests get a random seed resolved at API admission (they were fully deterministic) (source: 66506259e fix(sampling): seedless requests were fully deterministic - resolve a random seed at API admission)

**`src/exo/api/types/`**

- `openai_responses.py` — Responses-API types carry the clamped reasoning_effort wire value (source: 1a54be214 fix(api): clamp over-ceiling reasoning_effort to the wire ceiling instead of 422)

**`src/exo/download/tests/`**

- `test_rescan_announcements.py` — three zero-byte-catalog announcement tests skipped: fork 375a456a6 suppresses inert DownloadPending entries (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)

**`src/exo/master/tests/`**

- `test_event_log_requests.py` — Master fixture drops event_sender: the fork master commits its own events via _index_apply_broadcast (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_master_change_tasks.py` — master-sent events are read off the global stream (fork master has no event_sender channel) (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_master_takeover.py` — Master constructed without event_sender; fork master applies/broadcasts its own events (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_placement_utils.py` — jaccl TCP placement: same physical cable on both ends, non-RDMA cable preference, link-local demotion (source: 792138e15 fix(placement): demote link-local IPs below any routable path for jaccl TCP)
- `test_silent_nodes.py` — captures master events by stubbing _index_apply_broadcast (fork master has no event_sender) (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_unplaceable_requests.py` — unplaceable request's ErrorChunk read from the global event stream (fork master self-commits events) (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)

**`src/exo/shared/`**

- `apply.py` — apply_task_created refuses to resurrect an already-terminal task (I16 defense in depth) (source: de6d5369b fix(state): apply_task_created must not resurrect a terminal task)

**`src/exo/shared/tests/`**

- `test_clamp_reasoning_effort.py` — shared clamp_reasoning_effort unit tests behind the API wire-ceiling clamp (source: 1a54be214 fix(api): clamp over-ceiling reasoning_effort to the wire ceiling instead of 422)
- `test_event_serialization.py` — diagnostics evidence tuple: lax-field event round-trip that killed receiving nodes (source: c0dc84561 fix(events): diagnostics evidence tuple killed receiving nodes (lax field))

**`src/exo/shared/tests/test_apply/`**

- `test_apply_task_created_terminal_guard.py` — guards apply_task_created against resurrecting a terminal task (I16 follow-up) (source: de6d5369b fix(state): apply_task_created must not resurrect a terminal task)

**`src/exo/shared/types/`**

- `chunks.py` — ErrorChunk gains error_code + tool_call_parse_failure_kind so failures are client-classifiable and counted by kind (source: b8b3d77b2 fix(dsv41): make the cache-capacity refusal client-classifiable)

**`src/exo/utils/`**

- `log_format.py` — truncate_for_log: caps huge reprs (tool-schema payloads) so INFO logs stay readable (source: 7b941f4f4 fix(observability): surface runner crash cause + stop log flooding)

**`src/exo/utils/info_gatherer/tests/`**

- `test_tb_parsing.py` — Thunderbolt ident() hardening regression tests from the 2026-08-22 incident (empty/partial iface maps) (source: fc9542938 fix: harden TB monitor + tracing-aware runner hang watchdog (2026-08-22 incident))

**`src/exo/worker/`**

- `phase_marks.py` — worker-process mark stream (mark_state_applied / mark_plan_step_observed) for I16 Gate A, EXO_PHASE_MARKS-gated (source: docs/PERFORMANCE_HISTORY.md)

**`src/exo/worker/engines/`**

- `base.py` — Engine park hooks (idle_tick/idle_tick_seconds, begin_shutdown/shutdown_park) with no-op defaults (source: docs/dsv41-restart-g1-buildnote-2026-10-10.md)

**`src/exo/worker/engines/mlx/`**

- `phase_marks.py` — runner-subprocess phase marks (b1-b11) over the batch_generate/runner request lifecycle (source: docs/PERFORMANCE_HISTORY.md)
- `prefill_chunking.py` — image-span-aware prefill chunk planner (Phase 4d), spans re-based into the local prefill frame (source: 01b66315e fix(dsv4-vision): re-base image spans into the local prefill frame (4d))
- `vision.py` — DSv4-native branch in exo's vision layer (Phase 4b) + fail-closed on unidentified cached image regions (source: docs/DSV4_VISION_PORT_PLAN_PHASE34.md)

**`src/exo/worker/engines/mlx/disaggregated/`**

- `client.py` — CacheList/PoolingCache composites ingested over the KV-cache wire, offset from the restored cache (source: e3b6a0bed feat(disaggregated): CacheList + PoolingCache over the KV-cache wire)

**`src/exo/worker/engines/mlx/disaggregated/tests/`**

- `test_composite_cache_wire.py` — wire round-trip for DSv4-Flash composite caches (CacheList/PoolingCache): tensor + meta_state equality (source: e3b6a0bed feat(disaggregated): CacheList + PoolingCache over the KV-cache wire)
- `test_mlx_adapter.py` — adapter tests migrated to mlx-lm's .cache (was .state) semantic change (upstream ee19be4) (source: 9b3a03611 fix(mlx-lm-sync): adapt exo to mlx-lm .state semantic change (upstream ee19be4))

**`src/exo/worker/engines/mlx/dsv41/`**

- `__init__.py` — DSv4.1 engine package docstring/__all__; mlx_lm.models.deepseek_v41 only imported lazily (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)
- `builder.py` — DSv4.1 Builder: engine dispatch, vision-tower load, load-warmup liveness, 1M-context memory bound (source: a62a001c6 fix(dsv41): load warmup emits fence-backed liveness beats (ROUND-Q1B))
- `dsml.py` — DSv4.1 DSML dialect (calls/invoke/parameter sentinels) on exo's DSML machinery, no MLX import at runtime (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)
- `errors.py` — DSv4.1 engine error types; invalid request input fails alone instead of crashing the runner (source: 5a3bbe74a fix(dsv41): invalid request input fails alone instead of crashing the runner)
- `fingerprint.py` — park/guard build fingerprint tables (KV allow/neutral): rank-invariant layout, limiter and guard entries (source: docs/dsv41-restart-deploy-construct-2026-10-11.md)
- `thinking.py` — DSv4.1 thinking-marker resolution for this tokenizer (single added tokens) (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)
- `vision.py` — DSv4.1 vision path: image-row mask for image prefill, v4 image render, per-request isolation (source: 9bd473798 fix(dsv41): install the image-row mask for the image prefill (VL routing + engram masks))

**`src/exo/worker/engines/mlx/dsv41/tests/`**

- `__init__.py` — package docstring for the DSv4.1 text-plumbing test package (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)
- `_stub_harness.py` — stub harness for the park-idle-sessions-to-SSD tests (Option E) (source: 0626478d2 feat(dsv41): park idle sessions to SSD (Option E))
- `checkpoint_sentinels.json` — checkpoint-derived sentinel table (DSML + thinking marker strings/ids) with provenance (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)
- `conftest.py` — DSv4.1 test fixtures: FakeTokenizer wrapper parity + degeneration-guard/limiter harness (source: docs/dsv41-reasoning-budget-limiter-2026-10-13.md)
- `test_dsv41_dispatch.py` — engine dispatch + model-card tests for the DSv4.1 engine (ROUND-Q1B warmup liveness) (source: a62a001c6 fix(dsv41): load warmup emits fence-backed liveness beats (ROUND-Q1B))
- `test_dsv41_dsml.py` — DSv4.1 DSML dialect parsing tests (text plumbing, CPU-pinned) (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)
- `test_dsv41_load_warmup_liveness.py` — load warmup must emit fence-backed liveness beats (ROUND-Q1B) (source: a62a001c6 fix(dsv41): load warmup emits fence-backed liveness beats (ROUND-Q1B))
- `test_dsv41_output.py` — output pipeline: EOS turn no longer hangs, parser drained at end of turn (source: 24eab0fc9 fix(dsv41): engine EOS turn no longer hangs; mock answers from a real position table; drain parser at end of turn)
- `test_dsv41_prefill_liveness.py` — prefill liveness: fence-hook wiring plus checkpoint-ladder rungs (source: 614a40bbe DSv4.1: periodic checkpoint ladder + end-anchored margin rung for reuse)
- `test_dsv41_reuse_ladder.py` — checkpoint-ladder reuse tests (periodic rungs + end-anchored margin) (source: 614a40bbe DSv4.1: periodic checkpoint ladder + end-anchored margin rung for reuse)
- `test_dsv41_sentinels.py` — sentinel-string tests against the checkpoint sentinel table (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)
- `test_dsv41_thinking_load.py` — thinking-marker load tests for the DSv4.1 tokenizer (source: e1695d803 feat(dsv41): DSv4.1 text plumbing -- DSML dialect, thinking markers, model card, engine dispatch, round helpers, tests)

**`src/exo/worker/engines/mlx/tests/`**

- `_pp_exchange_prefill_peer_layer_count_subprocess_worker.py` — subprocess worker reproducing the exchange_prefill_peer_layer_count send/send deadlock (source: c7766f0ab fix(pp-batched-decode): exchange_prefill_peer_layer_count send/send deadlock -- real production incident)
- `test_batched_decode_external_cancel.py` — external cancellation routed through the real eviction protocol (source: 717523cb6 fix(pp-batched-decode): route external cancellation through the real eviction protocol)
- `test_dspark_load_guard_quant_keys.py` — DSpark load guard must not false-FAIL on quantized heads (source: 72c678eee fix(mlx): stop the DSpark load guard reporting a false FAIL on quantized heads)
- `test_dspark_tp_shard_gate.py` — env-gated TP sharding for the DSpark draft-head FFN (source: 2d85ccdcb perf(dspark): env-gated TP sharding for DSpark draft-head FFN)
- `test_dsv4_vision_end_to_end_offline.py` — offline end-to-end vision pipeline acceptance (Phase 4), multi-image + image_token_id guard (source: 82a448d55 test(dsv4-vision): offline end-to-end vision pipeline (Phase 4 acceptance))
- `test_generation_tps_baseline.py` — generation_tps metric + model-id baseline (uptime was being reported as generation time) (source: b843ca300 fix(metrics+deploy): generation_tps reported machine uptime as generation time; DSV4_MODEL_ID defaulted to the stale preview checkpoint)
- `test_patch_embed_tokens_bounds_check.py` — patch_embed_tokens bounds hardening for out-of-range token ids (source: dab4a432a fix(dsv4-vision): patch_embed_tokens bounds hardening + test-infra guard)
- `test_pipeline_prefix_hit_agreement.py` — PP KV-prefix-cache reuse via cross-rank hit-length agreement (source: 5e496ddb4 Re-enable PP KV-prefix-cache reuse via cross-rank hit-length agreement)
- `test_pp_batched_decode_adapter.py` — thin Response-classification adapter for the batched-decode session (source: ced9db437 feat(pp-batched): thin Response-classification adapter for the batched-decode session)
- `test_pp_batched_decode_driver.py` — Phase-1 scheduler/cache-router driver glue for batched decode (source: 3a5a096a3 feat(pp-batched): Phase 1 scheduler/cache-router driver glue)
- `test_pp_batched_decode_layers_install.py` — load-time install of batched-decode pipeline layers (subclass fix outside batch_step_scope) (source: f5199cae9 feat(pp-batched): load-time install for batched-decode pipeline layers)
- `test_pp_metaframe_batch_axis.py` — metaframe protocol v3 batch-axis stacking for decode batching (source: 089e0c731 feat(pp-batched): metaframe protocol v3 -- batch-axis stacking for decode batching)
- `test_pp_prefill_session.py` — ResumablePrefillSession: chunked-prefill interruption and mid-chunk-drive cancel (source: 44b5c645c feat(pp-prefill): ResumablePrefillSession -- the real driver for chunked-prefill interruption)
- `test_pp_scheduler_wire.py` — scheduler control-message wire encoding + two-phase chunk-drive redesign (source: 51603ad8b fix(pp-batched): two-phase chunk-drive redesign closing real cross-rank desync + freeze bugs)
- `test_pp_speculation_layer_reset.py` — reset stale spec-layer send/recv flags before a request's first-token call (source: 0d351baf6 fix(pp-dspark): reset stale spec-layer send/recv flags before each request's first-token call)
- `test_prefill_chunking_cache_offset_wiring.py` — covers the Phase-4d guard wiring: cache offsets, frames, both prefill loops (source: 9c080bdb8 test(dsv4-vision): cover the 4d guard wiring, frames, and both loops)
- `test_prefill_chunking_image_spans.py` — image-span-aware prefill chunk planner tests (Phase 4d) (source: 3993859b8 feat(dsv4-vision): image-span-aware prefill chunk planner (Phase 4d))
- `test_prefill_residual_gap_wiring.py` — covers both Phase-4d residual gaps in the prefill chunk plan (source: 10013d004 test(dsv4-vision): cover both 4d residual gaps)
- `test_prefix_cache_image_poisoning.py` — fail closed on unidentified cached image regions (prefix-cache poisoning) (source: d16a1039d fix(dsv4-vision): fail closed on unidentified cached image regions)

**`src/exo/worker/engines/mlx/vendor/`**

- `deepseek_v4_image_processor.py` — vendored, torch-free port of the DeepSeek-V4 image processor (source: f7712949a feat(vision): torch-free port of DeepSeek-V4 image processor)

**`src/exo/worker/engines/mlx/vendor/tests/`**

- `test_deepseek_v4_image_processor.py` — parity tests for the vendored torch-free image-processor port (source: f7712949a feat(vision): torch-free port of DeepSeek-V4 image processor)
- `test_sentinelless_recovery.py` — sentinel-less tool calls are recovered instead of failing the turn (source: af67755e0 fix(dsv4): recover sentinel-less tool calls instead of failing the turn)

**`src/exo/worker/runner/llm_inference/tests/`**

- `test_batch_generator_step_cancellation_latency.py` — cancellation observation bounded to ~1 decode step (reverted, then reapplied) (source: 12a84b077 fix(pp-batched-decode): bound cancellation-observation latency to ~1 decode step)
- `test_step_prefill_cancelled_guard.py` — PrefillCancelled escaping step() used to kill the runner process (Section 58) (source: 14af95e3f fix(cancel): PrefillCancelled escaping step() killed the runner process (Section 58))

**`src/exo/worker/runner/tests/`**

- `__init__.py` — package marker for the runner hang-watchdog test dir (source: fc9542938 fix: harden TB monitor + tracing-aware runner hang watchdog (2026-08-22 incident))
- `test_supervisor_hang_stack_guard.py` — stack-class hang guard (Fix B, shadow default) for the runner supervisor (source: ea858d8b4 fix(supervisor): stack-class hang guard (Fix B, shadow default))
- `test_supervisor_hang_tracing.py` — tracing-aware runner hang watchdog (2026-08-22 incident) (source: fc9542938 fix: harden TB monitor + tracing-aware runner hang watchdog (2026-08-22 incident))

**`src/exo/worker/runner/tests/fixtures/`**

- `synthetic_spin.txt` — synthetic spinning-stack fixture for the stack-class hang guard (source: ea858d8b4 fix(supervisor): stack-class hang guard (Fix B, shadow default))

**`src/exo/worker/tests/unittests/`**

- `test_fast_synch.py` — fork accepts EXO_FAST_SYNCH="1"/"0" because start_cluster.sh exports 1 (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)

**`src/exo/worker/tests/unittests/test_mlx/`**

- `conftest.py` — shared mlx test fixtures, incl. host-pressure-independent KV eviction setup (source: 5751a0fac test(kv-cache): make eviction tests host-pressure-independent (7 -> 0 failures))
- `test_prefix_cache_limits.py` — skipped: upstream #2368 limits target its flat-list cache, the fork uses the bounded radix trie (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)

**`src/exo/worker/tests/unittests/test_plan/`**

- `test_plan_step_event_wake.py` — I16 event-wake for plan_step (env-gated, default OFF) (source: 84bdcd756 round12: I16 event-wake for plan_step (env-gated, default OFF))
- `test_plan_step_lifecycle_dispatch.py` — lifecycle tasks must never be dispatched non-blocking (I16 dup-ConnectToGroup crash) (source: 3f06afdde fix(worker): never dispatch lifecycle tasks non-blocking (I16 dup-ConnectToGroup crash))
- `test_plan_step_lost_wakeup_fuzz.py` — I16 Gate-A lost-wakeup fuzz harness (pre-registration apparatus) (source: e6cde09eb round13: I16 Gate-A apparatus built; pre-registration found LATENTLY INVALID pre-boot; still unmeasured, nothing shipped)
- `test_worker_task_announcements.py` — runs plan_step on the worker's own task group (worker._tg) (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)

**`src/exo/worker/tests/unittests/test_runner/`**

- `test_batch_cancellation.py` — two #2365 cancel tests skipped: fork cancel() keeps glue/PP-spec/batched-decode teardown (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_concurrency_admission_gate.py` — admission gate for concurrent generation requests (PP stream-never-closed hang) (source: 88f3ed971 fix(runner): admission gate for concurrent generation requests -- root cause of the PP stream-never-closed hang)
- `test_dsml_e2e.py` — fork DSML tool-call parsing end-to-end (orphan-tail recovery, multi-call correctness) (source: 84cc57bcb fix(dsml): orphan-tail detector missed tails followed by the model's own closing fence)
- `test_parse_tool_calls.py` — malformed tool calls kept as errors (fork clean-fail); upstream #2394 text-fallback tests reverted (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)
- `test_runner_supervisor.py` — flat stall kill is the fallback when the fork hang watchdog is off (HANG_TIMEOUT_SECONDS=0) (source: 9fe0719a9 merge: upstream/main (bc412e223, 91 commits) into deploy/next27-ringfix lineage)

### mlx (22 files)

## Rationale fragment — mlx core files (fork-divergent vs upstream b6642e604)

**`mlx/backend/metal/`**
- `jit_kernels.cpp` — OPT-9 gather_qmm_rhs_lhs JIT instantiation: no-broadcast sorted prefill kernel (source: docs/prefill-throughput-breakthrough-2026-06-24.md; 980ac156a "perf(gather_qmm): OPT-9 sorted rhs_lhs kernel")
- `kernels.h` — declares get_gather_qmm_rhs_lhs_kernel for the OPT-9 no-broadcast sorted-prefill kernel (source: docs/prefill-throughput-breakthrough-2026-06-24.md; 980ac156a "perf(gather_qmm): OPT-9 sorted rhs_lhs kernel")
- `nojit_kernels.cpp` — non-JIT metallib stub for the same OPT-9 gather_qmm_rhs_lhs kernel (source: docs/prefill-throughput-breakthrough-2026-06-24.md; 980ac156a "perf(gather_qmm): OPT-9 sorted rhs_lhs kernel")
- `quantized.cpp` — gather_qmm_rhs_lhs dispatch (M>=16 sorted) + env-tunable gather_qmv_rhs tile config (source: docs/moe-tile-geometry-retune-dead-end-2026-08-18.md; 980ac156a, 089161722)
- `utils.h` — EXO_CMDBUF_RING_DIAG always-on command-buffer label, to ID the last op before a live GPU stall (source: docs/dsv4-220k-prefill-eventwait-rootcause-triage-2026-08-18.md; 55d56e527)

**`mlx/backend/metal/kernels/`**
- `fp_quantized.h` — M-batched sorted-run fp gather_qmv_rhs (register-held x rows, dequant-once weights) + fp_gather_qmm_rhs_lhs (source: 6f0ed1dd7 "perf(gather_qmv_rhs): v5"; docs/mxfp4-gather-qmm-rhs-lhs-kernel-2026-08-19.md)
- `fp_quantized.metal` — instantiates the fp_gather_qmv_rhs (M_TILE,RPS) kernel variants so the M-batched sorted-run path compiles (source: 6f0ed1dd7 "perf(gather_qmv_rhs): v5"; docs/mxfp4-gather-qmm-rhs-lhs-kernel-2026-08-19.md)
- `quantized.h` — affine M-batched gather_qmv_rhs (DSv4 is affine 8-bit g64, not mxfp4) + affine_gather_qmm_rhs_lhs OPT-9 kernel (source: 5033173b2 "perf(gather_qmv_rhs): affine variant"; docs/moe-tile-geometry-retune-dead-end-2026-08-18.md)
- `quantized.metal` — instantiates affine_gather_qmv_rhs across (M_TILE,RPS) and gs{32,64}/bits{4,8} for the dispatch's selected tiles (source: 5033173b2 "perf(gather_qmv_rhs): affine variant"; docs/mxfp4-gather-qmm-rhs-lhs-kernel-2026-08-19.md)

**`mlx/distributed/`**
- `distributed.cpp` — Group::split_tcp_coord + Group::reconnect forwarding to the QP-less TCP coord group and jaccl recovery (source: docs/dual-cable-topology-and-qp-budget-2026-08-21.md; fa69bafd1, 8212f670e)
- `distributed.h` — public API for split_tcp_coord (QP-less control-plane-only group; Thunderbolt max_qp=3) and reconnect() (source: docs/dual-cable-topology-and-qp-budget-2026-08-21.md; fa69bafd1, 8212f670e)
- `distributed_impl.h` — GroupImpl virtual defaults: split_tcp_coord throws unless the backend supports it; reconnect() is a no-op (source: docs/dual-cable-topology-and-qp-budget-2026-08-21.md; fa69bafd1, 8212f670e)

**`mlx/distributed/jaccl/lib/examples/`**
- `CMakeLists.txt` — registers the fork's three probes/tests in the examples target (source: 23fa410ef, 5831d3b06 "examples(jaccl): register hopid_budget_probe in CMake build")
- `coord_group_loopback_test.cpp` — RDMA-free forked loopback test of the QP-less TCP CoordGroup incl. desync tripwire (source: docs/dual-cable-topology-and-qp-budget-2026-08-21.md; fa69bafd1)
- `hopid_budget_probe.cpp` — measures how many UC QPs reach RTS on one Thunderbolt device before the NHI HopID budget runs out (source: docs/prefill-cliff-mechanism-2026-08-24.md; 23fa410ef)
- `uc_rdma_write_probe.cpp` — proves a UC QP can complete a one-sided IBV_WR_RDMA_WRITE on macOS librdma, the ACK-coalesce premise (source: 23fa410ef "examples(jaccl): add UC RDMA-write and HopID-budget feasibility probes")

**`mlx/distributed/jaccl/lib/jaccl/`**
- `coord_group.h` — QP-less TCP-only CoordGroup: full collective interface on one framed socket, own call_id namespace, loud desync throw (source: docs/dual-cable-topology-and-qp-budget-2026-08-21.md; fa69bafd1)
- `jaccl.cpp` — post-#4607 it feeds mesh.at(0) to the single-wire fork MeshGroup + our coordinator; multi-wire mesh DEFERRED (source: notes-mlx.md resolution table; b17781838 merge)
- `ring.cpp` — RingGroup::all_sum JACCL_TRACE_TIMING steady_clock instrumentation (RingGroup, not MeshGroup, is this 2-node topology's group) (source: docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md; e40a416b2)
- `ring.h` — declares the lazy-init timing flag/fields backing that env-gated JACCL_TRACE_TIMING trace (source: docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md; e40a416b2)
- `tcp.cpp` — retry-on-EAGAIN recv (transient jitter != dead peer), per-channel retry deadlines (p2p 300s), per-subgroup TCP coordinator (source: 0b0e2ad75, f8b77fe5a; docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md)
- `tcp.h` — SO_RCVTIMEO send+recv bounding and retry-deadline overrides so coord/recovery side channels outlast the data path (source: 444452be9; docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md)

### mlx-lm (47 files)

**`mlx_lm/models/deepseek_v41`**

- `LICENSE` — Apache-2.0 license copied with the vendored PipeNetwork/deepseek-v41-mlx model code (source: bd1bfd1e feat(mlx-lm): deepseek_v41 model package + EXL3 builder (exo phase 13); docs/benchmarks/phase13-dsv41-model-file-2026-09-28/README.md)
- `README.md` — Documents vendor source (PipeNetwork a9567ae, Apache-2.0), local MoE/attention/compressor edits and vision/exl3_build additions (source: bd1bfd1e feat(mlx-lm): deepseek_v41 model package + EXL3 builder; 6e54b90d feat(deepseek_v41): merge prefill/session/sampling/vision workstreams)
- `__init__.py` — Package entry re-exporting ModelArgs/Model; docstring records the vendor source and the exl3_build path (source: bd1bfd1e feat(mlx-lm): deepseek_v41 model package + EXL3 builder (exo phase 13))
- `_gates.py` — LOCAL ADDITION: shared row-count gates (_FENCE_MIN_ROWS) for the indexer/sparse-attention C1 levers, replacing twin defaults (source: 938b8112 perf(dsv41): lever-2 row-count guard (ABORTED, not shipped) + identity proof suite; docs/PERFORMANCE_HISTORY.md)
- `attention.py` — Vendored V4.1 MLA (sliding window + shared compressed positions); fork adds grouped wo_a and indexer qsplit/row-count fences (source: bd1bfd1e vendor commit; dc0e79c2 dsv41: indexer query-band split across TP ranks (DSV41_INDEXER_QSPLIT, default OFF))
- `cache.py` — Vendored incremental decode/chunked-prefill state; fork adds spec-rollback window-ring restore and bf16 grow-on-demand capacity (source: bd1bfd1e vendor commit; c4d19826 fix(dsv41): spec rollback must restore the window ring + the compressed rows)
- `collective.py` — LOCAL ADDITION: TP collectives plus a GPU-watchdog guard (JACCL all_sum on a CPU stream trips the Metal timeout past ~5s peer skew) (source: 4091b5ff feat(deepseek_v41): round-2 workstreams + two-node stability fixes)
- `compressor.py` — Vendored learned KV compression; fork allows wkv/wgate to be quantized modules, fp32 nn.Linear path unchanged (source: bd1bfd1e vendor commit; mlx_lm/models/deepseek_v41/README.md (vendoring note) local change 3)
- `config.py` — Vendored V4.1 config accepting both shipped layouts: flat inference config.json and the wrapped training layout (source: bd1bfd1e feat(mlx-lm): deepseek_v41 model package + EXL3 builder (exo phase 13))
- `dequant.py` — Vendored dequantizer turning the natively quantized release (fp8 + ue8m0 scales, fp4 experts) into bf16 (source: bd1bfd1e feat(mlx-lm): deepseek_v41 model package + EXL3 builder (exo phase 13))
- `engram.py` — Vendored n-gram hash-lookup engram tables (layers 1/14, ~100GB each); exl3_build reads rows on demand (source: bd1bfd1e vendor commit; docs/deepseek-v41-exl3-plan.md)
- `exl3_build.py` — LOCAL ADDITION: builds Model from an exllamav3 EXL3 checkpoint (EXL3Linear per trellis group, stacked EXL3SwitchGLU experts, rank slice) (source: bd1bfd1e vendor commit; docs/benchmarks/phase13-dsv41-model-file-2026-09-28/README.md)
- `fakequant.py` — Vendored activation fake-quantization (inference half of V4.1 QAT): fp8/fp4 round-trips the reference expects (source: bd1bfd1e feat(mlx-lm): deepseek_v41 model package + EXL3 builder (exo phase 13))
- `hc_fused.py` — LOCAL ADDITION: fused hyper-connection kernels (hc_mult==4) for the staggered coefficient path; 9.3 -> 16.3 tok/s TP=2 decode (source: 14d1c74d perf(deepseek_v41): 9.3 -> 16.3 tok/s plain TP=2 decode, same output)
- `hyper_connections.py` — Vendored Hyper-Connections residual stream (hc_mult=4) with V4.1's staggered, one-sub-layer-later coefficient schedule (source: bd1bfd1e vendor commit; docs/deepseek-v41-exl3-plan.md)
- `image_processor.py` — Torch-free port of the DSv4.1 preprocessing: FOUR sentinel types, no compress padding (not the older V4-Flash 5-type revision) (source: 6e54b90d merge vision workstreams; mlx_lm/models/deepseek_v41/README.md (vendoring note) local change 5; docs/DSV4_VISION_PORT_PLAN.md)
- `layers.py` — Vendored core layers: RMSNorm (eps 1e-20 kept exact), YaRN rope with inverse path, clamped SwiGLU (source: bd1bfd1e feat(mlx-lm): deepseek_v41 model package + EXL3 builder (exo phase 13))
- `logprobs.py` — LOCAL ADDITION: exact top-k logprobs over a vocab-sharded head via one all_sum of a per-row summary (no full vocab row on the wire) (source: 6950c793 feat(dsv41): exact top-k log-probs from the vocab-sharded head)
- `moe.py` — Vendored V4.1 MoE (384 experts, top-6, sqrt-softplus scoring); fork adds the group hook all_summing routed partials before the shared expert (source: bd1bfd1e vendor commit; mlx_lm/models/deepseek_v41/README.md (vendoring note) local change 1)
- `mtp.py` — Vendored DSpark 3-stage semi-autoregressive MTP draft head, with the node-local MTP/DSpark port edits on top (source: bd1bfd1e vendor commit; docs/benchmarks/phase3-mtp-dspark-port-2026-09-27)
- `sampling.py` — LOCAL ADDITION: seeded temperature/top-p/top-k sampling for the DSpark loop with proper speculative accept + residual resampling (source: 6e54b90d feat(deepseek_v41): merge prefill/session/sampling/vision workstreams)
- `serve.py` — LOCAL ADDITION: one Session per conversation -- prefix reuse, delta prefill, speculative decode, all state kept across turns (source: 4091b5ff feat(deepseek_v41): round-2 workstreams + two-node stability fixes)
- `session_cache.py` — LOCAL ADDITION: keeps a finished turn's ModelCache alive so the next turn prefills only the delta (source: 6e54b90d feat(deepseek_v41): merge prefill/session/sampling/vision workstreams)
- `sparse_attention.py` — Vendored gather-based sparse attention (MLX stand-in for the TileLang kernel); fork adds C1 column-boundary + row-count gating (source: bd1bfd1e vendor commit; 3bf8316f perf(dsv41): gate C1 column-boundary derivation on call row count)
- `vision.py` — LOCAL ADDITION (workstream H): native V4.1 vision tower -- 32-layer ViT + 3x3 aligner loaded bf16, fp32-verified vs torch reference (source: 6e54b90d merge vision workstreams; mlx_lm/models/deepseek_v41/README.md (vendoring note) local change 5; docs/DSV4_VISION_PORT_PLAN.md)

**`mlx_lm/models/exl3`**

- `LICENSE` — Apache-2.0 license copied with the vendored PonyExl3 kernel tree (source: 9ea86f91 feat(mlx-lm): vendor EXL3 trellis kernels as mlx_lm.models.exl3 (phase-2 integration))
- `NOTICE` — PonyExl3 attribution notice copied alongside the vendored kernels (third-party components listed) (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `README.md` — Records provenance (PonyExl3 8e7fa6b + patched node tree), the vendored minimal closure, local import/header changes and the equivalence gate (source: 9ea86f91 vendor commit; 6391efca loader follow-up; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `__init__.py` — Vendored init reduced to the two-class surface (EXL3Linear, EXL3SwitchGLU); upstream forward/linear/generate modules not vendored (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md local change 3)
- `decode.py` — Vendored MLX+Metal tile decode (no Python loop over codewords); absolute ponyexl3.* imports rewritten relative (source: 9ea86f91 feat(mlx-lm): vendor EXL3 trellis kernels as mlx_lm.models.exl3 (phase-2 integration))
- `exl3_linear.py` — Vendored EXL3 quantized linear as an nn.Module (activations stay on device); used for the DSv4.1 dense projections (source: 9ea86f91 vendor commit; 6e54b90d feat(deepseek_v41): merge prefill/session/sampling/vision workstreams)
- `exl3_moe.py` — Vendored EXL3 SwitchGLU experts + node-tree kernel work: v3 chunked prologue, v4 ALU opts ON, additive silu_clamp activation (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `gemv_metal.py` — Vendored fused Metal GEMV/GEMM (on-the-fly trellis decode + dot); v4 ALU opts (SWAR, direct-read shuffle) defaulted ON (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `hadamard.py` — Vendored MLX Hadamard helpers (parity with exl3.ref.hadamard); kept -- part of the kernels' import closure (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `layer_state.py` — Vendored per-layer MLX runtime state (device-resident trellis, signs, decoded weights) in the vendored closure (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `loader.py` — LOCAL ADDITION (not vendored): exllamav3-style EXL3 safetensors loader incl. the proven 2-way TP intermediate slice (source: 6391efca feat(mlx-lm): EXL3 checkpoint loader with proven TP slice (phase-12 follow-up); docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `metal_kernels.py` — Vendored custom Metal kernels for EXL3 trellis unpack and codebook decode (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `ops.py` — Vendored MLX-native fused linear ops (matmul and mx.compile prefill blocks) used by the vendored forward paths (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `perm.py` — Vendored tensor-core tile permutation (MLX take path) required by the trellis layout (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `reconstruct.py` — Vendored MLX weight reconstruction (Metal decode path), gated bit-identical to upstream on real expert tensors (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md; docs/benchmarks/phase1-exl3-format-validation-2026-09-27/README.md)
- `signs.py` — Vendored MLX sign unpacking for EXL3 suh/svh packed sign bitfields (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `stripe.py` — Vendored striped EXL3 forward (cached stripe decode + MLX matmul per chunk) (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)

**`mlx_lm/models/exl3/ref`**

- `__init__.py` — Vendored numpy reference package init, reduced correspondingly; kept because the Metal code builds its codebooks through ref/ (source: 9ea86f91 vendor commit; mlx_lm/models/exl3/README.md)
- `codebook.py` — Vendored procedural EXL3 codebook (CPU port of exllamav3_ext quant/codebook.cuh) needed by the Metal decode path (source: 9ea86f91 vendor commit; docs/deepseek-v41-exl3-plan.md)
- `layer.py` — Vendored EXL3 linear layer container (numpy reference counterpart of exl3_linear.py) (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `perm.py` — Vendored numpy tile-element permutation from exllamav3 modules/quant/exl3_lib/quantize.py (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)
- `signs.py` — Vendored numpy packed sign-bitfield unpacker (suh/svh) mirroring LinearEXL3.unpack_bf (source: 9ea86f91 vendor commit; docs/benchmarks/phase12-exl3-vendor-tp-split-2026-09-28/README.md)

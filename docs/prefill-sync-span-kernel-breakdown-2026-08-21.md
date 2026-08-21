# Prefill sync-span kernel breakdown — 2026-08-21 (session 2, part 2)

## Context

Follow-on to `docs/prefill-trace-instrumentation-findings-2026-08-21.md`,
which confirmed prefill is compute-bound at the request-trace (T()/
`request_trace`) granularity — framework/orchestration spans (barrier,
mem_checkpoint, cache_trim) total <0.02% of prefill wall time, and the
remaining ~100% sits inside one opaque `prefill.stream_generate` span that
our own instrumentation can't see inside.

This doc captures the real per-kernel breakdown INSIDE that opaque span,
using mlx-lm's separate model-side `SpanProfilerHook` (`EXO_PROFILER=spans`)
run in sync mode (`EXO_PROFILER_SYNC_SPANS=1`, forces `mx.synchronize()` at
every span boundary so each span measures its own real GPU kernel time
instead of MLX's lazy/async graph-build time — see
`mlx-lm/mlx_lm/profiler.py::SpanProfilerHook` docstring).

## Method

1. Cluster relaunched (config-only, same commit `c67b93225`, same
   `known-good-prefill-20260821-165048`-adjacent state — no code changes)
   with `EXO_TRACING_ENABLED=true EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1`.
   Verified all three env vars live in both nodes' actual process env via
   `ps aux` before trusting output.
2. Ran a single 100K-context prefill via
   `bench/phase3_precheck_depth_throughput.py --targets 100000` (sync-span
   mode necessarily serializes the pipeline, so full 300K/500K runs were
   skipped — one clean depth is enough to get the relative kernel-time
   breakdown, which is the useful output, not absolute tok/s from this
   run).
3. Pulled the `[PROFILER pid=...] span breakdown:` dump that immediately
   follows the real prefill's `Prefill complete: 70659 tokens in 225.09s`
   log line (confirmed by exact log adjacency, not just timestamp
   proximity) from both nodes' `~/exo.log`, and cross-checked the two
   ranks agree.

## Result: throughput under sync-span profiling

Sync-per-span serializes MLX's normally-async/lazy execution graph, so
absolute throughput drops hard under this instrumentation — expected
overhead from the profiler itself, not a regression:

- Prefill: 313.9 tok/s wall (vs 368.7 tok/s un-instrumented baseline —
  ~15% slower, i.e. relatively low overhead for full kernel-level sync
  attribution)
- Decode: 4.07 tok/s (vs 17.95 tok/s un-instrumented — ~77% slower; decode
  is far more sensitive to serialization since each token is a much
  smaller, more numerous set of ops)

Only the RELATIVE per-kernel percentages below are meaningful; do not use
this run's absolute tok/s for any throughput comparison.

## Real per-kernel prefill breakdown (100K context, rank 0 / macstudio-m4-1)

Both ranks agree almost exactly (max 0.5 percentage-point difference on
any span, confirming this is a stable, deterministic breakdown, not noise):

| Span | n (calls) | total ms | % of prefill wall |
|---|---|---|---|
| **attn** (top-level bucket) | 1591 | 113,306.58 | **56.7%** |
| **ffn** (top-level bucket) | 1591 | 86,557.49 | **43.3%** |
| moe.switch_mlp | 1591 | 59,908.65 | 30.0% |
| attn.sdpa | 851 | 26,601.87 | 13.3% |
| attn.o_proj | 1591 | 19,978.12 | 10.0% |
| attn.sdpa.compressed | 740 | 19,122.91 | 9.6% |
| attn.proj_qkv | 1591 | 17,829.70 | 8.9% |
| attn.all_gather | 1435 | 17,043.09 | 8.5% |
| moe.all_sum | 1591 | 16,440.11 | 8.2% |
| moe.post_combine | 1591 | 8,138.30 | 4.1% |
| layer.attn_hc | 1591 | 4,845.08 | 2.4% |
| layer.ffn_hc | 1591 | 4,786.13 | 2.4% |
| attn.indexer | 777 | 4,737.36 | 2.4% |
| attn.compressor | 1517 | 4,514.22 | 2.3% |
| layer.attn_residual | 1591 | 4,315.77 | 2.2% |
| layer.ffn_residual | 1591 | 4,313.04 | 2.2% |
| moe.gate | 1591 | 1,880.52 | 0.9% |
| attn.rope_in | 1591 | 1,335.69 | 0.7% |
| attn.rope_out | 1591 | 718.16 | 0.4% |
| attn.kv_cache | 1591 | 652.50 | 0.3% |
| model.embed | 37 | 646.55 | 0.3% |
| model.lm_head | 37 | 474.11 | 0.2% |
| everything else (mask, norms, switch.* sub-ops, indexer sub-ops, model.final_norm/attn_mask) | — | — | <0.2% each, ~1% combined |

`attn` + `ffn` = exactly 100% of instrumented wall time (they are the two
top-level buckets everything else nests under) — confirms the profiler's
own accounting is internally consistent, not just plausible-looking.

## Interpretation

**Prefill at 100K context is attention-dominated (56.7%), not MoE-dominated,
which is the OPPOSITE of decode's known profile.** For comparison, the
2026-06-26 decode-phase sync-span profile (warm memory facts 743/751) found
decode to be **83% FFN/MoE-bound** (`moe.switch_mlp` alone = 73.2% of
decode wall time), with attention at only 17%. Prefill inverts that
almost exactly: attention buckets (`attn.sdpa` + `attn.sdpa.compressed` +
`attn.o_proj` + `attn.proj_qkv` + `attn.all_gather` + `attn.indexer` +
`attn.compressor` ≈ 56.4%) dominate over FFN/MoE (`moe.switch_mlp` +
`moe.all_sum` + `moe.post_combine` + `moe.gate` ≈ 43.2%).

This makes sense structurally: prefill processes the FULL prompt length in
one pass through attention (quadratic-ish cost in sequence length even with
DSv4's sparse indexer/compressor path), while decode processes one token at
a time through attention (cheap per-step) but still runs the full dense
MoE FFN every step. At 100K context, prefill's attention cost has caught up
to and surpassed its MoE cost; the exact crossover point as a function of
context depth was not measured this session (would need the same sync-span
breakdown at 300K/500K — not run tonight due to the ~4x wall-time cost of
sync-span serialization at deeper context).

Two sub-spans worth flagging for future kernel-level optimization work,
since they are large AND currently un-decomposed further:

- `attn.sdpa` (13.3%) + `attn.sdpa.compressed` (9.6%) = 22.9% combined —
  the two scaled-dot-product-attention variants (dense vs DSv4's sparse/
  compressed indexer path) together are the single largest attention cost,
  larger than `moe.switch_mlp` alone would need to be optimized against.
- `attn.all_gather` (8.5%) — a real TP collective cost inside attention,
  distinct from `moe.all_sum` (8.2%, the FFN-side collective). Combined,
  the two all-to-all-style collectives cost ~16.7% of prefill wall time.
  This is NOT the same thing tonight's earlier transport-hardening session
  fixed (that was reconnect/teardown/QP-budget correctness bugs, not
  steady-state collective cost) — it's worth asking whether either
  collective's algorithm/chunking could be tightened, but that's a new,
  unexplored lever, not a re-litigation of tonight's transport fixes.

## Honest conclusion

This is real, new, actionable information the earlier work never had:
prefill's actual kernel-time profile, broken down to sub-operation
granularity, confirmed consistent across both TP ranks. It does NOT by
itself prove a specific optimization will land more throughput — every
listed span is presumably necessary compute for a correct forward pass —
but it identifies where the wall-time actually goes (`attn.sdpa*`
variants and TP collectives inside attention, more than MoE, unlike
decode) as the concrete target for any future prefill-side optimization
attempt, rather than guessing.

**Not yet done:** the same breakdown at 300K/500K context, to see whether
the attn/ffn split shifts with depth (plausible, since attention cost
scales with sequence length while MoE cost per token is roughly constant).
That's the natural next increment if this line of investigation continues.

## Cluster state after this session

Restarted with `EXO_TRACING_ENABLED=true EXO_PROFILER=spans
EXO_PROFILER_SYNC_SPANS=1` — this is diagnostic-mode overhead, NOT the
production/benchmarking configuration. Before trusting any future
throughput number, confirm the cluster has been restarted WITHOUT these
env vars (or check `ps aux` on both nodes for `EXO_PROFILER=` presence),
since sync-span mode alone cuts prefill throughput ~15% and decode ~77%.

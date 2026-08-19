# MoE per-stage GPU cost breakdown: gather/scatter is NOT the missing 16% (2026-08-18)

## Context

Tonight's MoE-efficiency investigation left one open question: the
production-class SwitchGLU benchmark measured 62.6% of matched-shape
dense-GEMM ceiling at the standing 2048-token prefill chunk size (see
`bench/moe_production_class_bench.py` and its companion doc). Where does
the other ~37.4% go? The existing `EXO_PROFILER=spans` breakdown of the
MoE block's six internal sub-spans (`switch.gather_sort`, `switch.up_proj`,
`switch.gate_proj`, `switch.activation`, `switch.down_proj`,
`switch.scatter_unsort` -- see `docs/dsv4-220k-prefill-span-profile-2026-08-18.md`)
reported implausible microsecond-scale per-call times for all six, summing
to ~89ms against the real (correctly measured) parent `moe.switch_mlp`
span's 167,314ms total -- an ~1900x gap, physically impossible for real
GEMM/argsort/scatter work at this scale.

## Root cause of the misleading numbers (not a bug -- a known, documented mode)

`mlx-lm/mlx_lm/profiler.py`'s `SpanProfilerHook` is lazy by design: span
boundaries only call `mx.eval`/sync when `EXO_PROFILER_SYNC_SPANS=1` is
set (the class docstring says so explicitly: "By default spans only
bracket graph-build time... Set `EXO_PROFILER_SYNC_SPANS=1` to
`mx.synchronize()` at BOTH span boundaries... at the cost of serializing
the pipeline"). Tonight's earlier `EXO_PROFILER=spans` run that produced
the 220K-context profile did NOT set `EXO_PROFILER_SYNC_SPANS=1`, so the
six `switch.*` sub-spans (which never call `finalize()`/`mx.eval()`
internally themselves) measured pure Python graph-construction time, not
real GPU kernel time. The outer `moe.switch_mlp` span's own number is
still fully trustworthy -- it does call `finalize()` at its close, so its
167,314ms/26.9% figures are real. This is expected, documented profiler
behavior encountered without the flag needed for sub-span attribution --
not a code defect to fix, just a measurement-methodology gap in that one
prior run.

## Method

Rather than re-running the full 220K-context cluster profile with
`EXO_PROFILER_SYNC_SPANS=1` (which would serialize the whole prefill
pipeline and distort absolute throughput, per the class's own docstring
warning), built a standalone isolated benchmark
(`bench/moe_gpu_capture_profile.py`) that constructs one `SwitchGLU` at
exact production shapes (hidden_size=4096, moe_intermediate_size/TP-rank
=1024, n_routed_experts=256, top_k=6, mxfp4 group_size=32 bits=4) with
realistic skewed top-6-of-256 routing (Gumbel-argmax over per-expert
popularity skew, not uniform), and times each of the six stages in
ISOLATION: pre-materialize each stage's inputs via `mx.eval`, then time
the stage's own operation + a single trailing `mx.eval` over 80 iterations,
subtracting a measured empty-eval baseline overhead (~155-166us).

This gives a serialized-upper-bound per stage. To check whether that
upper bound is actually close to the real number (i.e. whether meaningful
inter-stage GPU scheduling overlap exists that the isolated numbers would
be missing), compared `sum(isolated stages)` against the real single-eval
whole-block time (the same sync pattern production actually uses) --
this ratio is the "overlap factor."

Run on this laptop's own Apple GPU (M4 Max, 32-core) -- same chip
architecture/generation as the two Mac Studio cluster nodes (M4 Max,
40-core), so relative stage proportions transfer directly to the cluster;
absolute ms would run somewhat faster on the Studios' larger core count.

## Results (3 independent runs, reproducible)

| stage | isolated ms | % of whole-block |
|---|---|---|
| gather_sort | ~0.89-0.93 | ~1.9% |
| up_proj | ~14.95-15.06 | ~31.5-32.1% |
| gate_proj | ~15.07-15.51 | ~32.1-32.6% |
| activation | ~0.40-0.41 | ~0.8-0.9% |
| down_proj | ~15.88-16.61 | ~33.4-34.9% |
| scatter_unsort | ~1.21-1.26 | ~2.5-2.6% |

Whole-block real time: 47.47-47.75 ms/call (3 runs). Sum of isolated
stages: 48.61-49.63 ms/call. **Overlap factor: 1.024-1.044x across all
three runs** -- very close to 1.0, meaning there is negligible hidden
inter-stage GPU scheduling overlap in this block. The isolated per-stage
numbers ARE the real breakdown, not just a loose upper bound.

## Conclusion

**The three GEMMs (up_proj + gate_proj + down_proj) account for ~97% of
real per-call MoE time. `gather_sort` + `scatter_unsort` -- the actual
data-movement/reorder stages -- are only ~4-5% combined.**

This directly resolves the question a plausible-sounding hypothesis raised
earlier tonight (that gather/scatter might be a material 15-25% of MoE
cost, materially changing where the "remaining 16% gap" analysis should
look): **that hypothesis is wrong.** The real proportions are much closer
to what the broken profiler numbers implied by coincidence (~1% each for
gather/scatter), just not exactly that, and confirmed via a trustworthy
method this time rather than a measurement artifact.

**The unexplained ~16% gap between the measured 62.6%-of-dense-ceiling MoE
efficiency and 100% is INSIDE the GEMM kernels themselves** -- it is not
hiding in data movement, sorting, or the SwiGLU activation. This is
consistent with, and further confirms, tonight's earlier NO-GO conclusion
on the hybrid short-run/long-run dispatch idea (`bench/moe_run_length_tile_waste.py`
and its companion analysis): the remaining gap is a property of the GEMM
kernel's own execution efficiency at this ragged shape (tile occupancy,
weight-streaming efficiency, dequant overhead inside the kernel), not
something a dispatch-level fix (routing to a different existing kernel
based on run length) can meaningfully close. Any further work on this
specific gap would need to be inside the GEMM kernel itself (e.g. Metal
occupancy/tile-geometry tuning for this shape), a genuinely new,
unscoped investigation -- not attempted tonight.

## Files

- `bench/moe_gpu_capture_profile.py` -- committed, reusable. Run with
  `--skip-capture` for a fast isolated-stage-only result (recommended,
  ~15s), or without it to also produce a `.gputrace` bundle for manual
  Xcode GUI inspection (no scriptable/non-interactive way to extract
  per-kernel timing from Xcode's internal `.gputrace` binary format was
  found this session -- the isolated-timing method above is the
  trustworthy result, the GPU capture is supplementary corroboration
  only, not required).

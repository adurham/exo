Shared-scale int8 all_sum: local_absmax's ~400-420ms is a probe fence artifact, not a real cost (2026-08-19)
=============================================================================================================

Summary
-------

Built an isolated (single-process, non-distributed) benchmark,
`bench/absmax_fence_artifact_probe.py`, to test the caveat already flagged
in `moe-allsum-sharedscale-root-cause-found-2026-08-19.md`: that the
probe's `mx.eval()` fence at `local_absmax` might be forcing a synchronous
GPU round-trip that pays for the ENTIRE preceding lazy graph backlog (the
rest of the MoE layer's matmuls), not the abs-max reduction itself.

**Confirmed: it is a fence-placement artifact, not a real per-call cost.**
`mx.max(mx.abs(y))` itself costs ~0.4ms when timed in isolation (warm `y`,
or `y` pre-fenced immediately before). The ~400-420ms only appears when the
timed region is the FIRST `mx.eval()` hit after a backlog of unevaluated
upstream ops (e.g. a matmul chain standing in for the rest of the MoE
layer) -- in that case the timed region silently absorbs the cost of
materializing everything queued before it, because `mx.eval()` schedules
and waits for the ENTIRE pending lazy graph, not just the requested output.

Method
------

Ran 4 conditions, shape (1,2048,4096) matching the live probe data,
30 trials each after 5 warmup trials, on GPU:

- **A (warm reduction)**: `y` fully evaluated before the timed region;
  time only `mx.eval(mx.max(mx.abs(y)))`.
- **B (cold/lazy reduction)**: `y` is the unevaluated tail of an 8-deep
  matmul chain (same shape); time `mx.eval(mx.max(mx.abs(y)))` -- this
  mirrors the real call site, where `y` arrives as still-lazy MoE output.
- **C (pre-fenced reduction)**: same chain as B, but `mx.eval(y)` runs
  BEFORE the timed region (simulating "a fence already ran right before
  this phase"); only the reduction itself is timed.
- **D (chain-eval-alone)**: cold chain, time `mx.eval(y)` with NO
  reduction at all -- isolates the backlog-flush cost by itself.

Results
-------

```
A: warm y, timed reduction                        p50=  0.426ms
B: cold/lazy y, timed reduction+chain              p50= 56.796ms
C: pre-fenced y, timed reduction                   p50=  0.427ms
D: cold y, timed chain eval only (no reduction)    p50= 60.320ms
```

A and C (true reduction-only cost) are statistically identical at
~0.4ms. B and D (backlog-flush cost, with or without the reduction
tacked on) are both ~57-60ms and dominate almost entirely -- the
reduction itself (B - D delta, within noise) contributes essentially
nothing beyond the backlog flush.

Note the absolute backlog-flush number here (~57-60ms, from an 8-matmul
synthetic chain on a single unloaded local GPU) is NOT directly comparable
to the live cluster's ~400-420ms (real MoE layer compute, contended
2-node distributed run, different chain depth/shape mix) -- this bench
isn't reproducing the live magnitude, only the MECHANISM. The key result
is the RATIO: A/C (true op cost) vs B/D (backlog-flush cost) differs by
~130x in this synthetic setup, which is the same qualitative signature
(reduction cost negligible vs. fence-placement cost) the live probe data
showed (local_absmax ~98% of per-call time called into question by the
probe's own module comment).

Conclusion
----------

The live probe's `local_absmax` phase does NOT reflect a real, fixed
~400ms cost of `mx.max(mx.abs(y))`. It reflects the cost of being the
first `mx.eval()` fence encountered after an unevaluated MoE-layer graph
-- i.e. an artifact of the probe's own instrumentation design, exactly as
the caveat in the prior doc predicted. The real per-call cost of the
abs-max reduction itself is on the order of sub-millisecond, consistent
with it fusing into or overlapping with surrounding lazy-graph compute
under normal (unfenced) scheduling.

Implication for the shared-scale design: this REOPENS the "no speedup"
mystery from `moe-allsum-sharedscale-live-test-no-speedup-2026-08-19.md`.
The previously identified "culprit" (local_absmax) is not the real cost
center under normal (unfenced) execution -- the actual live-cluster
no-speedup cause is still unexplained and needs re-investigation without
relying on the fenced probe's absolute numbers. The probe's RELATIVE
phase breakdown should also be treated with caution beyond just the
absolute-magnitude caveat already noted, since it's now shown that fence
placement, not real per-phase cost, can dominate which phase looks
"expensive."

Next steps (not attempted here)
--------------------------------

1. Re-measure the live cluster's shared-scale path with the mx.eval()
   fences at `local_absmax`/`scale_allsum` REMOVED (keep only
   entry/exit timestamps around the whole `_sharedscale_compute_scale`
   + payload path, single fence at the very end) to see real per-call
   cost without a fence landing mid-graph.
2. If a genuine no-speedup signal remains under that measurement, look
   at whether the two-phase collective's own synchronization (not any
   individual phase's "own" cost) is the real, distributed-specific
   culprit -- something this single-process bench cannot probe, since
   it has no `mx.distributed.all_sum` cross-rank barrier.

Files
-----

- Bench script: `bench/absmax_fence_artifact_probe.py` (isolated,
  single-process, reusable -- run with `uv run python
  bench/absmax_fence_artifact_probe.py`).

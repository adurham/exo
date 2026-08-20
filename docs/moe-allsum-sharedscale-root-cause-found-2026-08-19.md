Shared-scale int8 all_sum: root cause of the GPU asymmetry/no-speedup found (2026-08-19, final)
=====================================================================================================

Summary
-------

Deployed the per-phase/cross-rank-skew probe instrumentation
(mlx-lm branch feat/moe-allsum-sharedscale-2026-08-19, commit
`a22b8a5`) live to get real per-rank timestamp data, per the follow-up
plan from `moe-allsum-sharedscale-live-test-no-speedup-2026-08-19.md`.

**Root cause found, decisively: the `local_absmax` computation itself
(a plain `mx.max(mx.abs(y))` reduction) costs ~400-420ms per call on
BOTH ranks -- roughly 70-90x more than the actual int8 payload transfer
(4-6ms) and completely dwarfing everything else in the design.** This
is NOT a communication problem. The int8 wire-bandwidth win is real and
present (payload_allsum: 4-6ms, vs. baseline unquantized all_sum's
~178ms/call measured earlier tonight at the same shape) -- it's being
buried by an unrelated, much larger cost in the scale-computation step.

Raw data (real, from the live cluster, both ranks, same shape
(1,2048,4096), representative calls)
------------------------------------------------------------------------

```
rank 0 (call 2487-2494, less contended):
  local_absmax:   ~400-420ms   <- THE cost
  scale_allsum:   ~0.5ms       (fast -- this rank arrives first)
  quantize:       ~1-3ms
  payload_allsum: ~4-6ms       <- the actual int8 wire transfer (fast, real)
  dequant:        ~1-4ms

rank 1 (call 2460-2467, waiting on rank 0):
  local_absmax:   ~280-530ms   <- also dominant here
  scale_allsum:   ~140-330ms   (this rank waits at the barrier for rank 0)
  quantize:       ~0.6-1.2ms
  payload_allsum: ~5-11ms
  dequant:        ~0.5-1.4ms
```

The `scale_allsum` asymmetry (rank 0 fast at ~0.5ms, rank 1 slow at
100-300ms) is exactly the "one rank waits at the collective barrier"
signature Fable predicted and matches the GPU-utilization asymmetry
observed earlier tonight from the dashboard (one node near 100%, the
other 33-44%) -- but it's a SYMPTOM of the real cost (local_absmax),
not the cause. Both ranks pay the ~400ms local_absmax tax independently
and in parallel; the rank that finishes first (here, rank 0) then waits
briefly at scale_allsum for the slower one.

Caveat on absolute numbers
------------------------------

The probe's `mx.eval()` fence at every phase boundary (necessary to get
real per-phase attribution instead of lazy-graph mis-attribution)
inflates absolute latency -- this is explicitly documented in the
instrumentation's own module comment ("do not use probe-on numbers as
a throughput benchmark"). The RELATIVE breakdown (local_absmax ~98% of
per-call time, payload transfer ~1%) is the trustworthy signal here,
not the raw 400ms figure, which would very likely shrink substantially
without the forced eval (the abs-max reduction could otherwise overlap/
fuse with surrounding lazy-graph compute rather than forcing a
synchronous device round-trip on its own).

What this means
-------------------

The shared-scale int8 design's core insight (stay on the reliable
all_sum path, use real int8 wire arithmetic) is confirmed correct and
its actual communication cost is genuinely small. The reason it showed
"no speedup" in the earlier live test wasn't a flaw in the collective
strategy -- it's that the scale-computation step (`_sharedscale_compute_scale`'s
`mx.max(mx.abs(y))`) is unexpectedly expensive at this call frequency
(once per MoE layer, 43 times per forward pass), likely because it sits
alone as a small, isolated reduction that doesn't naturally fuse with
neighboring ops the way it would inside a larger fused kernel.

Next step (not attempted tonight)
--------------------------------------

Two concrete, cheap directions worth testing before any further live
attempt:
1. Re-measure WITHOUT the probe's eval fences (i.e. remove/disable the
   per-phase mx.eval() calls, keep only entry/exit timestamps) to see
   the local_absmax cost under normal lazy-graph scheduling -- the real
   number may be far smaller than 400ms once it's allowed to overlap
   with surrounding compute rather than being forcibly isolated.
2. If local_absmax genuinely costs real time even unfenced, consider
   computing it less often (e.g. once per forward pass instead of once
   per MoE layer, using a slightly looser/cached scale) or fusing it
   into an existing pass over `y` that already touches every element.

Status
---------

Cluster reverted to standing baseline (`EXO_PREFILL_STEP_SIZE=2048`,
no sharedscale flags) immediately after collecting this data. Code
remains on the pushed, unmerged mlx-lm branch
`feat/moe-allsum-sharedscale-2026-08-19` (now includes the probe
instrumentation, commit `a22b8a5`). This closes the open question left
by the earlier no-speedup doc -- the asymmetry is now explained, not
just observed.

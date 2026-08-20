Lever 2: sequence-chunk pipelining shows real but modest overlap (~1.1-1.15x), laptop-noisy (2026-08-20)
==============================================================================================================

Context
----------

Follow-up to Fable's corrected Lever 2 design: NOT layer-pipelining
(wrong -- layer N+1's attention depends nonlinearly on layer N's
all_sum via RMSNorm), but SEQUENCE-CHUNK pipelining (split the prompt,
overlap chunk A's all_sum with chunk B's independent same-layer
compute). Also directly informed by the same-day Lever 3 finding: the
real all_sum cost is a GPU->CPU->GPU stream-boundary drain (jaccl's
collectives are pinned to a CPU-only stream, `AllReduce::eval_gpu`
throws), not wire bandwidth -- so the actual question this probe
answers is "can MLX's async_eval schedule GPU compute for chunk B
concurrently with chunk A's collective sitting on the CPU stream."

IMPORTANT CAVEAT: laptop, not cluster
------------------------------------------

This ran on the user's MacBook Pro M4 Max via `mlx.launch -n 2 --backend
ring` (loopback, both ranks share ONE GPU) -- this machine is NOT an
idle dedicated test box; the user runs other things on it concurrently.
Absolute latency numbers here are noisy and not directly comparable to
the cluster's real RDMA numbers. The PAIRED (interleaved arms, same
time window) comparison is the trustworthy number -- it cancels out
contention that affects both arms equally. Do not quote the raw
per-arm medians as clean absolute costs.

Result (production-like comm:compute ratio, P_REPS=2, interleaved,
n=10, bit-identical correctness confirmed)
------------------------------------------------------------------------

```
PAIRED serial/pipelined speedup:      med=1.013  min=0.883  max=1.227
PAIRED serial/pipelined_deep speedup: med=1.148
```

- `pipelined` (shallow prefetch, overlap depth 1): essentially NEUTRAL,
  high variance (0.88x-1.23x) -- likely swamped by laptop contention
  noise at this depth.
- `pipelined_deep` (deeper async prefetch): a real, modest **median
  1.148x** speedup -- smaller than the "perfect overlap" theoretical
  ceiling (which would be `sum(compute+comm)/max(compute,comm)` =~
  1.41x at this measured ratio) but a genuine, reproducible signal in
  the right direction, not noise-only.
- `lazy_noeval` (just removing eval fences, no real async scheduling)
  measured WORSE than serial in every run -- confirms the gain requires
  genuine `async_eval`-based overlap, not merely OPT-7's already-tried-
  and-reverted "defer the fence" idea. This is the correct, different
  mechanism Fable specified.

Interpretation
------------------

Real, positive, structurally-sound signal for sequence-chunk pipelining
-- genuine overlap is achievable and distinct from the already-dead
fence-removal approach. The magnitude (~1.1-1.15x) is smaller than the
naive "eliminate all comm cost" ceiling would suggest, consistent with:
(a) imperfect prefetch depth in this quick probe, (b) real scheduling/
dependency overhead MLX's lazy graph pays even when the two collectives
ARE logically independent, (c) laptop noise likely still not fully
eliminated at n=10.

This is NOT yet a cluster-ready result -- it's a laptop-loopback
structural proof that overlap is possible and worth pursuing, not a
production speedup number. Real production DSv4 forward-pass
integration (splitting the prompt into 2 sequence chunks, staggering
per-layer async_eval, auditing the KV-cache path for hidden blocking
mx.eval calls per Fable's step 3) is real engineering work not
attempted tonight -- this probe only validates the underlying MLX
scheduling mechanism works and gives a rough magnitude estimate.

Files
--------

`bench/lever2_seqchunk_overlap_probe.py` (built by the dispatched
subagent, completed by parent after the subagent's dispatch hit a
600s timeout mid-sweep -- the subagent's partial results before
timeout were consistent with this final run, same qualitative pattern).

Next step (not attempted)
-----------------------------

Real DSv4 forward-loop integration per Fable's 5-step build order:
(1) confirm jaccl overlap holds with the REAL all_sum call (this probe
used a synthetic all_sum, not jaccl's actual collective -- worth
re-verifying the mechanism transfers), (2) chunked forward with
staggered async_eval per layer, (3) audit KV-cache path for blocking
mx.eval, (4) cross-chunk causal masking, (5) shared-expert overlap on
top. Real cluster validation only after single-node/loopback
correctness is airtight.

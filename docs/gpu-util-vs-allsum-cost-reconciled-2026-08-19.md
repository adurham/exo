GPU-utilization vs all_sum-cost contradiction resolved: the wait is real, disguised as "busy" (2026-08-19)
================================================================================================================

Context
-------

Two independently-measured, both-real findings from tonight appeared to
directly contradict each other:

- FINDING A (`docs/gpu-utilization-confirmed-saturated-2026-08-18.md`):
  `exo_gpu_usage_ratio` telemetry showed 96-97% GPU utilization,
  continuously, throughout real long-context prefill. No idle bubbles
  found across 16 samples.
- FINDING B (`docs/moe-all-sum-dominant-cost-2026-08-19.md`): NOP-ablating
  `moe.all_sum` alone (leaving everything else unchanged) gave a
  confirmed 2.6x speedup -- meaning all_sum accounts for 61-64% of
  prefill wall time.

If the GPU is genuinely 97% busy doing arithmetic, there's no room for
all_sum to independently account for 61-64% of wall time -- these can't
both be describing "GPU spends most of its time computing."

Resolution
-------------

Both are real; they're measuring different things. `exo_gpu_usage_ratio`
almost certainly measures device OCCUPANCY (was anything resident/
scheduled on the GPU this sample interval), not achieved compute
throughput. A GPU spinning on a wait/sync kernel, or with the submission
thread merely alive-but-blocked, reads as "busy" on this kind of metric
even when doing zero useful arithmetic.

**Confirmed directly in jaccl's own source** (`mlx/distributed/jaccl/lib/jaccl/mesh_impl.h`,
lines ~1341-1345): the reliable-path drain loop explicitly yields the
CPU core specifically to avoid "starv[ing] the Metal/GPU submission
threads under sustained c>=2 load (which parks the peer's main thread in
an uninterruptible GPU wait -> _check_hang)." This is the engineers'
OWN documented acknowledgment that the collective's wait can put the GPU
submission thread into an "uninterruptible GPU wait" state -- exactly
the mechanism that would read as "busy" on an occupancy-style metric
while contributing zero useful compute.

Mechanistically: when the MLX graph hits `mx.distributed.all_sum(y)`
(deepseek_v4.py:2837), the calling thread (which owns GPU command-buffer
submission) blocks until the CPU-driven RDMA/ARQ exchange completes on a
SEPARATE thread. The GPU submission thread staying resident/scheduled
during that block is plausibly what the utilization counter is picking
up as "97% busy" -- not 97% of cycles doing real matmul/attention work.

What this means for the CPU-offload idea
--------------------------------------------

This reframes the user's original idea precisely. Not "move arithmetic
off the GPU onto idle CPU" (an earlier consult correctly ruled that out
-- there's no meaningful non-arithmetic GPU work to relocate, and
cross-stream sync would cost more than it saves). Instead: **decouple
GPU work submission from the collective's completion wait**, so the GPU
can keep issuing/executing the NEXT layer's independent compute (the
parts of the graph that don't depend on the reduced `y`) while the
CPU-side RDMA exchange for the current layer's all_sum runs in the
background on its own thread -- true compute/comm overlap, not
relocation.

This is a real, structurally different lever from the quantized-all_sum
work (which reduces HOW MUCH data moves) -- this reduces HOW MUCH THE
CRITICAL PATH WAITS for that data to move, by finding independent work
to overlap with the wait. The two are complementary, not competing:
quantization shrinks the wait itself; overlap hides whatever wait
remains behind useful work.

Next step (UPDATE: already tried, reverted -- do not re-attempt this specific mechanism)
------------------------------------------------------------------------------------------

Checked the exact code: gating the forced `mx.eval()` after `moe.all_sum`
by `_fence_every_n` (i.e. deferring the sync to let more layers' work
queue up before flushing) was **already tried and reverted** as "OPT-7"
(comment at `deepseek_v4.py` ~2967-2972): it made B=2 prefill 23%
SLOWER (111 vs 144 tok/s). Root cause per the comment: without the
per-layer eval, MLX builds a LARGER lazy graph that's more expensive to
evaluate at the eventual fence point than doing incremental per-layer
evals -- the anticipated overlap benefit didn't materialize, and graph-
accumulation cost dominated instead. This is real, already-measured
evidence against the "just remove the sync" framing of the overlap idea.

**This does not kill the underlying idea, but it does kill the specific
mechanism scoped above.** The reconciliation (GPU-busy-metric ≠
GPU-doing-useful-work, the collective's wait is real and disguised)
still stands and is real. What's now known NOT to work: naively
deferring MLX's own lazy-eval fence. What remains untested: whether a
genuinely different overlap mechanism -- e.g. restructuring the graph so
independent NEXT-layer compute is explicitly issued/scheduled before the
current layer's `mx.eval()` fence blocks (rather than relying on lazy-
eval's own graph accumulation to find that overlap implicitly) -- could
succeed where the implicit approach failed. This is a materially harder,
more invasive change (real graph restructuring, not a one-line env-gate
change) and should be scoped carefully, with the OPT-7 failure mode
explicitly designed around, before attempting it.

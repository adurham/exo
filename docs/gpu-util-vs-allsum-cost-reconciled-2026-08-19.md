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

Next step
------------

Scope a concrete plan for compute/comm overlap in the DSv4 prefill
layer loop -- specifically: is there independent GPU work available
to overlap with a given layer's all_sum (e.g. the NEXT layer's
attention/indexer computation, which doesn't depend on the current
layer's post-all_sum residual until the very end), and can MLX's lazy
graph + async command buffer model actually express that overlap, or
does the current code structure force a hard sync at each all_sum call
site (the "Phase H Lever 1" comment at deepseek_v4.py:2838-2840
suggests a DELIBERATE forced-eval immediately after all_sum -- worth
checking whether that forced sync is itself the thing preventing
overlap that would otherwise be possible).

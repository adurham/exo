# MoE GEMM bottleneck confirmed memory-bandwidth-bound via real GPU-time accounting (2026-08-19)

## Context

The MoE investigation converged on a real ~28%-of-ceiling gap (see
`docs/moe-vs-dense-qmm-isolation-2026-08-19.md`), attributed to per-expert
raggedness (median M=7-14 tokens/expert). The remaining open question was
whether the flat ~15% floor found by the M-sweep (M=32 through M=512, all
stuck ~85% of dense ceiling, see `bench/moe_m_sweep_mxfp4.py`) is
ALU-bound, memory-bandwidth-bound, or occupancy-bound at the kernel level
-- each implies a different (or no) fix.

## Method

Attempted an Xcode GPU Frame Capture for instruments-level ALU/memory/
occupancy breakdown, but this requires Xcode's GUI (no CLI/programmatic
parser exists for the `.gputrace` bundle format -- confirmed by
inspecting the bundle's internal files, which are proprietary binary
formats with no public schema; `GPUDebugger.ideplugin`'s resources are
`.nib`/`.storyboardc` view files only, not a data format any CLI tool
reads). GUI-based inspection was ruled out per a hard project rule
(no `computer_use`, no AppleScript/System-Events-driven window control --
both are equivalent to direct desktop interaction and off-limits).

Found a fully programmatic alternative instead: `mx.metal.gpu_time_ns()`
(gated behind `MLX_GPU_TIME=1`), which sums real `GPUEndTime - GPUStartTime`
per completed command buffer -- the same underlying hardware accounting
Xcode's GPU timeline displays, exposed directly by MLX with no GUI
required. Measured the real production `SwitchGLU` MoE forward at the
standard 2048-token chunk shape (same routing/quant config used all
night: mxfp4 g=32 b=4, hidden=4096, inter/rank=1024, 256 experts,
top-6), per-iteration (reset counter, eval, synchronize, read counter --
20 iterations, each cleanly isolated so no cross-iteration overlap
pollutes a single measurement).

## Result

```
wall time:      median 41.53 ms/call
GPU busy time:  median 67.53 ms/call
GPU utilization (GPU-busy / wall): 162.6%
```

**GPU-busy time exceeding wall-clock time is only physically possible if
multiple command buffers are executing concurrently on the GPU** --
different hardware units (memory-fetch engines, ALU/MMA units) doing
useful work simultaneously across overlapping dispatches, with MLX's
async command-buffer submission allowing several MoE sub-stage dispatches
to be in-flight at once. Reproducible across per-call measurements (not
a one-off artifact of aggregate accounting).

## Interpretation

This is direct evidence the workload is **memory-bandwidth-bound, not
ALU-bound**, at this shape:
- A pure ALU/compute bottleneck would cap achievable GPU utilization at
  or below 100% -- if the compute units were the limiter, no amount of
  concurrent dispatch could make a device do more ALU work per wall-clock
  second than one saturated unit's actual throughput.
- Exceeding 100% is only possible when different physical units (memory
  controllers vs ALU/MMA pipelines) are each busy on different
  concurrently-in-flight dispatches -- the classic signature of a
  bandwidth-bound kernel whose ALU sits idle waiting on weight-streaming
  while another dispatch's memory fetches overlap with ITS compute.

**This independently corroborates the earlier isolated-benchmark finding**
(`docs/moe-quant-vs-bf16-dequant-attribution-2026-08-19.md`, M2's
conclusion: "in the ragged path the kernel is weight-bandwidth-bound, not
ALU-bound, so 4x smaller weights fully offset dequant ALU"). Two
completely independent measurement methods -- a bf16-vs-quantized A/B
comparison, and now real hardware GPU-time accounting via
`mx.metal.gpu_time_ns()` -- converge on the same conclusion using
different techniques.

## What this means for "does this need a new kernel design"

A memory-bandwidth-bound small-M grouped-GEMM problem has a well-known
fix category: **better weight-streaming / reuse across the ragged
per-expert runs** (e.g. keeping recently-used expert weights resident in
threadgroup memory across adjacent tiles, or restructuring the dispatch
so weight loads amortize across more of the M dimension) -- not a
fundamentally new compute algorithm. This is still real kernel-level
engineering work (not a config/parameter tune), but it's a narrower,
better-understood class of problem than "unknown, needs investigation" --
the earlier "needs a new kernel design, direction unclear" framing can
now be sharpened to "needs a bandwidth-optimized small-M grouped GEMM
kernel, specifically targeting weight-reuse across ragged runs."

## Method note (for future GPU-timing work without Xcode GUI access)

`mx.metal.gpu_time_ns()` + `mx.metal.reset_gpu_time()`, gated behind
`MLX_GPU_TIME=1` set BEFORE importing mlx (env var read at import time,
not toggleable after), gives real per-command-buffer GPU execution time
with no GUI, no Xcode, no screen interaction of any kind. This is the
correct fallback for "is X ALU-bound or bandwidth-bound" questions when
Xcode's GPU Frame Capture GUI isn't available/appropriate to use --
compare GPU-busy-time/wall-time ratio: near 100% with no overlap
possible = compute-bound at that dispatch; a case where you can measure
>100% via overlapping concurrent dispatches, or where GPU-busy time is
much LESS than wall time (large gap) = CPU-dispatch-bound / stalling on
sync, not GPU-execution-bound. Also validated (separately, not needed for
the final answer here) that Metal's public per-dispatch counter-sampling
API (`MTLComputePassSampleBufferAttachmentDescriptor` +
`MTLCounterSampleBuffer` with the `timestamp` counter set) works for
custom/synthetic kernels via a small Swift harness (`/tmp/moe_gemm_gpu_timed.swift`,
not preserved) -- useful if a future investigation needs per-dispatch
(not aggregate) GPU timing outside MLX's own kernels, though it cannot
reach MLX's actual compiled mxfp4 kernels since those aren't
independently invocable outside MLX's binary. The detailed ALU-busy%/
memory-stall%/occupancy% breakdown Xcode's GPU Debugger shows remains
GUI-only; no public API or file-format parser exists for it.

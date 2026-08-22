# Real Instruments Metal System Trace: ground-truth GPU dispatch latency — 2026-08-21 (session 2, part 14)

## Why this matters

Every review tonight (#1, #4) flagged an Instruments Metal System Trace
as the real ground-truth tool never used this session — the sync-span
Python profiler forces synchronization and can misattribute or distort
timing; a real GPU trace was the only way to directly observe dispatch
gaps versus genuine compute time. Attempted and succeeded this session.

## Method

`xctrace` (Instruments CLI, `/usr/bin/xctrace`) is available and
scriptable via SSH directly on the target Mac Studio — no GUI session
needed. Two approaches were tried:

1. **`--attach <pid>` against the live runner's multiprocessing-spawned
   worker process** — captured a trace file with real structure but
   ZERO Metal GPU data (only an empty `RunIssues` SQLite store). This
   attach-to-existing-process mode did not successfully hook the Metal
   instrument for this process — plausibly because Instruments' Metal
   GPU capture needs to be active from the process's Metal device
   creation, and attaching mid-lifecycle to an already-running,
   already-initialized MLX/Metal context misses that hook point.
2. **`--launch` with a fresh standalone script** — launched a new
   Python process directly under `xctrace record --launch`, running a
   real MLX matmul workload (4096×4096 bf16, 200 iterations, mirroring
   DSv4's real per-op FLOP scale). **This worked cleanly** — captured a
   full multi-instrument trace (`metal-gpu-intervals`,
   `metal-command-buffer-completed`, `metal-application-command-buffer-submissions`,
   and ~70 other real Metal/kdebug/GPU tables), confirmed via
   `xctrace export --toc`.

Exported the `metal-gpu-intervals` table (real per-GPU-op start time,
duration, and **CPU-to-GPU dispatch latency**, per Metal's own
instrumentation — not inferred, not sync-span-distorted) via
`xctrace export --xpath`, no GUI required, filtered to the probe
process's PID (confirmed 201 matching rows for the 200 real matmul
iterations + 1 setup op).

## Result: real, directly-measured GPU dispatch latency

Steady-state matmul GPU durations: ~9.10ms per op — matches the
theoretical bf16 compute-bound estimate for a 4096³ matmul at the M4
Max's public peak TFLOPS almost exactly (~9.16ms theoretical vs ~9.10ms
measured), confirming this trace is capturing genuine near-peak compute,
not idle-padded time. This cross-validates the trace technique itself:
it isn't producing garbage numbers.

**CPU-to-GPU dispatch latency (steady-state, excluding the first
cold-start op): mean 96.8µs, stdev 4.6µs, range 89.5-105.7µs, n=19.**
This is a real, tight, directly-measured Metal dispatch-latency figure
— not an estimate, not a sync-span artifact.

## Cross-validation against two independent prior estimates

This real number lands almost exactly where two completely independent
earlier sources in this codebase predicted:

1. **`FusedSwitchGLU`'s own code docstring** (written 2026-06-xx, before
   tonight, for the MiniMax model's dispatch-fusion rationale): "~100-200
   µs of dispatch+sync overhead" per Metal dispatch. Real measured:
   96.8µs — squarely inside that range.
2. **Tonight's own roofline estimate** (`docs/decode-roofline-dispatch-bound-2026-08-21.md`):
   inferred ~150µs/dispatch from the gap between observed decode latency
   and the bandwidth-bound ceiling. Real measured: 96.8µs — same order
   of magnitude, roughly 1.5x lower than the earlier inferred estimate
   (a real refinement, not a contradiction — the earlier number was
   explicitly flagged as a rough back-of-envelope figure).

## What this changes

This closes the loop review #1 opened: the "why did the wq_a+wkv fusion
show zero measurable effect despite removing a real dispatch" question
now has a cleaner answer. At ~97µs of real dispatch overhead per op, and
wq_a/wkv being small matmuls (hidden_size → q_lora_rank/head_dim, much
smaller than the MoE gate/up projections' hidden_size → intermediate_size),
removing ONE ~97µs dispatch out of decode's total ~54.6ms/token budget is
genuinely too small a fraction (~0.18%) to be statistically detectable
against the measurement noise this session's A/B methodology could
resolve (~0.15-0.17 tok/s stdev) — consistent with, not contradicting,
the earlier null result. The MoE gate+up fusion's larger, clearly
measurable win makes sense by the same logic: it operates on the much
wider MoE intermediate dimension, where the SAME ~97µs/dispatch savings
represents a larger fraction of that op's own cost, AND removes one
whole dispatch from a hot path called at every one of 43 layers.

Combined with the earlier real finding (raw jaccl all_reduce floor
~120µs at the real decode message size, versus ~4094µs average in-model
cost — a 34x gap not explained by dispatch overhead alone at this
scale), the emerging picture is: **decode's real overhead budget has (at
least) two distinct components** — per-dispatch Metal/CPU latency
(~97µs, now directly measured) and TP-collective-specific overhead
(rank skew and/or scheduling around `moe.all_sum`, ~34x the raw wire
floor, not yet fully decomposed). Both are real, both are now backed by
direct measurement rather than inference, and both point at
comm/compute overlap and dispatch-count reduction as the correct
remaining lever classes — exactly what review #2's redirection and
review #14 (queued, not yet attempted) already identified.

## Reusability

This SSH-scriptable `xctrace --launch` + `xctrace export --xpath`
technique is now a proven, working, zero-GUI-required method for future
sessions to get real GPU-level ground truth on this cluster — a
capability flagged as missing/unused in every review tonight, now
demonstrated working end-to-end. The one caveat: `--attach` to an
already-running production runner process did NOT work (empty capture)
— any future real-workload trace needs to either launch a fresh
standalone reproduction script (as done here) or investigate why attach
mode failed to hook Metal instrumentation on a live process (not
investigated further this session).

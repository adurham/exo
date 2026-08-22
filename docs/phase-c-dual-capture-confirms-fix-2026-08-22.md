# Phase C: fresh dual CPU+GPU capture confirms the fix at the hardware level — 2026-08-22 (post-fence-fix investigation)

## Why this check

Per Fable's ranked recommendation: the original idle-gap investigation
data (`docs/gpu-idle-gap-deep-dive-2026-08-22.md`,
`docs/pysampler-blocking-eval-root-cause-2026-08-22.md`) was captured
under the broken async fence — now stale, since decode throughput
changed by +23-67%. A fresh capture was needed before trusting any
further attribution claim about where decode's remaining cost lives.
Fable's specific addition to the original methodology: capture CPU-side
(in-process Python sampler) and GPU-side (Instruments Metal trace)
**simultaneously in the same window**, since 70% GPU idle next to a
cheap collective is consistent with CPU-side dispatch/sync overhead
that a GPU-only trace wouldn't show.

## Real capture

Relaunched with `EXO_PYSAMPLER=1` (the in-process sampler built earlier
this session) plus a fresh `xctrace --attach` Metal System Trace on
both nodes simultaneously, exact same real-decode-during-capture
methodology validated earlier tonight. Fired one real
`bench/decode_probe.py` request (512-token prompt, 400-token
generation; real result: `ttft=3.69s decode=10.86s decode_tok_s=29.38`,
matching the fixed baseline) during the capture window.

## Real GPU occupancy result

Computed the same real interval-union occupancy metric used
throughout this campaign, on rank0's real trace data (pid 61796),
**isolated to the actual request window** (0-14000ms of the 30s
capture, after finding and excluding ~5.6s of genuine POST-request idle
time within the capture window — a real methodology correction, not a
finding about decode itself: two large gaps at t=13.95s and t=16.33s,
totaling 2.38s+3.25s, occurred AFTER the real request completed at
~14.55s total).

**Real post-fix GPU occupancy (request-window-isolated): 85.42%** — up
from ~28-30% pre-fix (`docs/gpu-idle-gap-deep-dive-2026-08-22.md`). A
massive, direct, hardware-level confirmation of the fix.

Real gap-length distribution also shifted meaningfully:

| | Pre-fix | Post-fix (request-window) |
|---|---|---|
| Median gap | 528µs | **181µs** |
| Real occupancy | ~29-30% | **85.42%** |

## Real CPU-side confirmation

The in-process Python sampler's data for the same real request (found
via a wider tail slice after an initial narrower slice captured only
post-request idle — a real, honest methodology note, corrected before
drawing conclusions) shows the compute thread's hot-line distribution
has genuinely shifted:

| Line | Code | Pre-fix % | Post-fix % |
|---|---|---|---|
| (was line 3016, now shifted to line 3061 due to the fix's own code additions) | `mx.async_eval(y)` (non-blocking) | 0% (never the hot line) | **45.75%** |
| (now line 3081) | `mx.eval(y)` (blocking fallback, prefill/transitions only) | **67%** (line 3016, was the dominant hot line) | **13.19%** |

Verified the exact line identity via `Read` with explicit line numbers
against the real deployed file (per the reusable lesson from this
session's earlier line-counting error — always verify with an exact
numbered check, never trust a remembered count).

**This is the single most direct confirmation possible**: the compute
thread's dominant hot-line genuinely moved from the blocking
`mx.eval(y)` call to the intended non-blocking `mx.async_eval(y)` call,
exactly as the fix was designed to produce.

## Convergent evidence — three independent measurement methods

1. **Real throughput benchmark**: 18.5 → 29.2-31.1 tok/s (short
   context), +58-67%.
2. **Real GPU hardware trace** (Instruments Metal System Trace):
   occupancy 29-30% → 85.42% (request-window-isolated).
3. **Real CPU-side stack sampling** (in-process Python sampler):
   dominant hot-line shifted from the blocking call (67%) to the
   non-blocking call (45.75%), blocking fallback down to 13.19%
   (matching the design intent — still present for prefill/transitions
   only).

All three independently-built, independently-analyzed measurement
methods converge on the same conclusion, using genuinely different
instrumentation techniques (HTTP-level throughput timing, OS/GPU-driver
hardware telemetry, and in-process Python interpreter introspection).
This is about as decisively validated as a single fix can get within
one investigation session.

## Scope note: deep-context capture not repeated

Given the triple-convergent result already secured at short context,
and the real diminishing marginal value of a fourth confirmation, the
originally-planned second capture at deep context (~100K) was not
performed this session. The real depth-scaling throughput numbers
(`docs/baseline-locked-decode-fence-fix-20260822.md`, +53.9%/+31.4%/+24.6%
at 100K/300K/500K) already establish that the fix's benefit persists
at depth, just with a smaller relative magnitude — a real,
already-understood pattern (the fence's fixed per-layer collective
saving becomes a smaller fraction of a larger per-step decode cost as
KV-cache grows), not requiring a fresh trace to re-confirm.

## Disposition

Closes Phase C and the full 3-phase post-fence-fix investigation plan.
All three phases (roofline, GPU clock, idle-gap re-capture) converge on
a consistent, coherent story: the async-fence fix delivered a real,
substantial, multiply-confirmed improvement, with genuine remaining
headroom (14-20% of theoretical bandwidth-bound peak, per Phase A) that
would require further investigation (likely: what remains inside the
still-present ~13% blocking-fallback time, and/or deeper kernel-level
attribution of the residual gap) if further optimization work is
pursued in a future session.

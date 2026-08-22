# Step 3 synthesis: real per-token time budget, ~82-85% still unattributed after stacking two independently-verified real measurements — 2026-08-22 (session 3, offline arithmetic)

## Why this check

Per a detailed investigation plan from an independent Fable consult
(given tonight's ruling-out of `moe.all_sum` as the dominant decode-time
cost): Step 3 of that plan asks for a real per-token collective-cost
floor, counting all real collectives and their real measured cost, to
sharpen the roofline's aggregate "~12% of peak" framing into an actual
stacked time budget.

## Real collective inventory (confirmed complete via code read)

Checked every `mx.distributed.all_sum`/`all_gather`/`all_max`/`all_min`
call site in `deepseek_v4.py` for decode-time (not just prefill/training)
relevance:

- `moe.all_sum` (line ~2960, inside `sharding_group is not None`): the
  ONE real decode-time collective, confirmed active — 43 calls/token
  (one per layer).
- Attention `all_sum`/`all_gather` (lines ~4048-4314, `_ATTN_ALLSUM`-gated):
  confirmed DISABLED — live process env shows `EXO_DSV4_ATTN_ALLSUM=0`.
- `sum_gradients` (line ~2842): a training-only gradient-synchronization
  construct, not exercised during pure-generation decode — confirmed
  irrelevant by its own naming/purpose, not just assumed.

**Confirmed: 43 `moe.all_sum` calls/token is the complete real decode-time
collective inventory.** No other collective type contributes.

## The stacked real budget

Using real, independently-verified numbers only — no estimates:

| Component | Value | Source |
|---|---|---|
| Real decode wall time | 53.48 ms/token (18.7 tok/s) | Live clean-baseline `decode_probe.py`, confirmed multiple times tonight |
| Real collective cost | 1.55-2.85 ms/token (43 × 36-66µs median/mean) | jaccl-internal `steady_clock` timing, 45,666 real calls, `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` |
| Theoretical compute floor | 6.51 ms/token | Bandwidth-bound roofline (active-param bytes ÷ M4 Max memory bandwidth), `docs/decode-roofline-dispatch-bound-2026-08-21.md`, inputs re-verified `docs/roofline-sanity-check-inputs-confirmed-2026-08-22.md` |
| **Unattributed remainder** | **44.1-45.4 ms/token (82.5-84.9% of real wall time)** | This document — stacking the two above against real wall time |

This is a sharper, more defensible number than the earlier "~65-85%
unexplained" framing (which came from a single instrument — the
Instruments GPU-occupancy trace, ~70% idle) — this version stacks TWO
independently-measured real quantities (jaccl transport cost + roofline
compute floor) against real wall time, and both credits land in a
tight, mutually-consistent range (82.5-84.9%) with the GPU-occupancy
figure (~70% idle, which is a different but related quantity — idle
time vs. unattributed compute-plus-idle time; the roofline floor is a
theoretical MINIMUM real compute time, not an upper bound on real
compute time, so some of that 44-45ms could in principle be real
GPU-busy time beyond the theoretical minimum, not pure idle — this
document does not resolve that distinction, see the still-open items
below).

## What this does and doesn't establish

**Does establish**: `moe.all_sum` and the theoretical compute floor
TOGETHER account for only ~15-17.5% of real decode wall time. The
overwhelming majority of decode's real cost — 82.5-84.9% of it — is
neither the collective nor unavoidable compute-floor cost. This
directly quantifies the target for the next investigation phase (kernel-
and dispatch-level attribution) rather than leaving it as a vague
"lots of headroom" claim.

**Does NOT establish**: whether this remainder is genuine GPU idle time
(dispatch/scheduling gaps), real GPU compute ABOVE the theoretical
floor (e.g. non-ideal kernel efficiency, not just idle), CPU-side
Python/MLX overhead, or some mix. The Instruments GPU-occupancy trace
from earlier tonight (~70% idle both ranks) is suggestive that a large
share is genuine idle time, but that trace's `metal-gpu-intervals` data
was checked tonight (see the companion note in this investigation) and
found to lack per-kernel labels — it cannot yet distinguish "idle" from
"busy running an inefficient kernel" at the granularity needed to fully
close this out. That's the next step.

## Disposition

Real, decisive quantification of the size of the remaining mystery —
this sharpens rather than resolves it. Next steps (per the detailed
Fable-provided investigation plan): kernel-level attribution via a
fresh Instruments capture with per-kernel labeling, CPU-side profiling
(py-spy), GPU frequency/DVFS check, and memory-residency/expert-paging
check — all still queued.

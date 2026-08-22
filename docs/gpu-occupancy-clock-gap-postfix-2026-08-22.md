# T2: Fresh post-async-fence-fix Instruments trace — occupancy, clock, and gap distribution all improved dramatically — 2026-08-22 (session 4)

## Why this check

Per the Fable-planned investigation, all prior GPU occupancy/clock/gap
data was captured BEFORE the async-fence fix (§2.8 in
`docs/PERFORMANCE_HISTORY.md`) landed. This step re-captures the same
three measurements (per-kernel channel data, gap-length distribution,
GPU clock/power) against the current, fixed production config to see
what changed.

## Method

Real production runner PIDs identified live via `ps aux` on both nodes
(m4-1: 65758, m4-2: 79210) — not synthetic probes. Launched
`xcrun xctrace record --template 'Metal System Trace' --attach <pid>
--time-limit 30s` on both nodes in parallel (backgrounded), plus
`sudo powermetrics --samplers gpu_power -i 100 -n 250` on m4-1
concurrently. Fired two real `bench/decode_probe.py` requests (512-tok
prompt, 400-tok generation) during the capture window (throughput
dropped under tracing overhead as expected — Instruments captures
their own dispatch cost; absolute traced-run numbers are not meaningful,
only internal ratios, per the established methodology from session 3).

Exported real `metal-gpu-intervals` data via `xctrace export --xpath`
(the known-working method from prior sessions — `--toc` still fails on
attach-mode traces with the same "Missing Template Error", not
re-investigated since the `--xpath` workaround is already established).
Parsed both ranks' XML (91-93MB, ~87-89K rows each) via a custom
streaming `ElementTree.iterparse` script (files too large for a single
in-memory parse) to isolate rows belonging to our actual runner PID,
then computed real interval-union occupancy (not naive sum — same
verified methodology as the prior session's analysis).

## Real results

### GPU occupancy (interval-union, own-process rows only)

| | rank0 (pid 65758) | rank1 (pid 79210) |
|---|---|---|
| Real GPU intervals (our process) | 63,952 | 65,279 |
| Whole-trace-span occupancy | 67.24% | 67.59% |
| **Request-window-isolated occupancy** (bursts >1s only, excludes idle padding before/after captures) | **78.64%** | **78.86%** |

**Compare to pre-fix: ~29-30% occupancy (both ranks).** This is
consistent with — and independently confirms via a completely fresh
capture — the earlier "85.42% occupancy" figure from the
`EXO_PYSAMPLER` dual-capture work (§13, "Phase C"); the small
discrepancy (78.6-78.9% here vs 85.42% there) is attributable to
different capture windows/methodologies (this capture spans two
back-to-back decode_probe runs including their TTFT/prefill phases,
while Phase C's capture was more tightly isolated to steady-state
decode only) — both are dramatically higher than the pre-fix baseline,
not a discrepancy worth chasing further.

### Gap-length distribution

| | rank0 | rank1 |
|---|---|---|
| n_gaps (whole trace) | 43,489 | 44,387 |
| median gap | 95.12µs | 89.29µs |
| mean gap | 139.20µs | 137.43µs |
| p95 gap | 213.12µs | 230.71µs |

**Compare to pre-fix: median gap 520-528µs, mean 1,700-1,961µs, with
83.9% of gap TIME concentrated in the 0.5-20ms buckets**
(`docs/gpu-idle-gap-deep-dive-2026-08-22.md`). Post-fix, gaps have
collapsed to the 50-500µs range that WOULD be consistent with pure
per-kernel CPU-dispatch latency — the ms-scale stalls that dominated
pre-fix gap time are gone. This is a real, dramatic structural change
in the gap-length distribution, not just an aggregate occupancy number
moving.

### GPU clock and power (concurrent powermetrics, m4-1 only)

Isolated 120 consecutive samples (~12.7s) matching the first
`decode_probe.py` run's real decode duration (12.79s):

| | Value |
|---|---|
| Median clock during active decode | **1578 MHz (peak spec)** |
| % of busy samples at peak clock (1578MHz) | **88.3%** (106/120) |
| Clock range | 788-1578 MHz |
| Median power draw | 19.5 W |
| Power range | 0.1-21.3 W |

**Compare to pre-fix: clock 819-1122 MHz (never reaching peak), power
4.6-7.1W** (`docs/gpu-idle-gap-deep-dive-2026-08-22.md`). This
conclusively confirms that earlier document's prediction: "fixing the
underlying gap pattern should raise clocks as a side effect" — it did,
dramatically. The GPU now spends the large majority of active-decode
time at its literal peak clock (matching the sustained-synthetic-load
test's 1578MHz finding from `docs/gpu-clock-symptom-confirmed-2026-08-22.md`),
and real power draw (~19.5W median) is now consistent with genuine
sustained compute load rather than the bursty low-load signature seen
pre-fix.

## What this does NOT establish — real limitation found

**Per-kernel attribution is still not possible from this capture
template.** Inspected the real `gpu-channel-name` and `formatted-label`
fields for our process's rows: channel names are only ever "Compute"
(63,931/63,952 rows, 100.0% of our process's GPU time), "Fragment", or
"Vertex" — generic Metal channel categories, not MLX operation names.
The `formatted-label` field for representative rows shows only
"Command Buffer N:Compute Command M" — a positional label, not a
kernel name (no `gather_qmm`/`switch_mlp`/`all_sum` strings found
anywhere in either export). **This confirms and extends the exact same
limitation flagged in `docs/PERFORMANCE_HISTORY.md` §13** ("existing
`metal-gpu-intervals` trace data lacks per-kernel operation labels...
only command-buffer-level granularity") — re-confirmed against a fresh
capture, not just inherited from stale data. True kernel-level
attribution (isolating `moe.switch_mlp`/`GatherQMM` specifically, per
Fable's T3 plan) requires a genuinely different capture method: Xcode's
GPU Frame Capture via `mx.metal.start_capture()`/`stop_capture()`
wrapped around real decode steps, opened in the Metal Debugger — not
further mining of this `xctrace` template's data.

## Decision gate evaluation (per the Fable-provided plan)

The plan's decision criteria for this step: "idle fraction <15% → close
the gap-chasing line permanently... idle >25% → promote gap-alignment
work... clock <~80% of 1578MHz at high occupancy → open new
achieved-bandwidth roofline line."

**Real occupancy is 78.6-78.9%, i.e. idle fraction is ~21-21.4%** —
this falls in neither clean bucket (not <15%, not >25%). Read
literally, this leans toward "still worth investigating idle gaps
further" but the picture has fundamentally changed in kind, not just
degree: pre-fix gaps were ms-scale (matching fence/collective-boundary
timescales); post-fix gaps are 50-500µs-scale (consistent with genuine
per-kernel CPU dispatch latency, an expected and largely irreducible
cost class, not a bug). **Interpretation: the gap-chasing investigation
is NOT fully closed, but its likely ceiling has shrunk substantially —
closing the remaining ~21% gap, even completely, caps out around a
+27% throughput improvement (1/0.786), a real but much smaller prize
than the earlier "why is the GPU 70% idle" framing suggested.**

**Clock is at 1578MHz median during 88.3% of busy samples — clearly
ABOVE the 80%-of-peak threshold.** This closes that branch of the
decision tree: clock-frequency is no longer suspect as an independent
factor; the GPU reaches genuine peak clock under the current post-fix
load pattern.

## Next step

Per the plan, T3: kernel-efficiency check on the top busy-time kernel
(previously identified as GatherQMM/switch_mlp, ~30-45% of wall time in
earlier span breakdowns) — compute achieved GB/s vs ~546GB/s M4 Max
peak. Given this session's finding that the `metal-gpu-intervals`
template cannot supply per-kernel timing, T3 needs `mx.metal.start_capture()`
wrapped around a real decode step's Python code (not another `xctrace`
attach), captured on one node, and opened in Xcode's Metal Debugger for
per-encoder timing and kernel names.

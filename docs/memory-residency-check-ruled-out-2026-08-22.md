# Step 5: memory residency check — expert-weight paging conclusively ruled out — 2026-08-22 (session 3, live read-only checks)

## Why this check

Fable's investigation plan flagged memory residency / expert-weight
paging as its own top suspect for a decode-time Mac Studio MoE
workload: "total params likely exceed what's wired; per-token expert
selection touches cold pages → GPU stalls on faults that look like
idle." Real check, read-only, no relaunch — `vm_stat`, `vmmap`, and a
before/after pageins delta around a real decode request.

## Real check #1: pageins delta across a real decode request

Snapshotted `vm_stat`'s cumulative `Pageins` counter before and after a
real decode request (`bench/decode_probe.py`, 512-token prompt +
600-token generation, confirmed 495 real tokens generated, 1.97s TTFT +
26.73s decode, 18.48 tok/s — matching clean baseline throughput, so this
was a representative real request, not a degraded one).

**Result: only 66 real pageins (1.0 MB) across the entire request.
Swapins delta: exactly 0** (unchanged before/after — confirms no
swap-to-disk memory pressure). At ~0.002 MB/token, this is not
consistent with continuous per-token expert-weight paging from disk —
if MoE routing were regularly touching cold, non-resident expert
weights, this delta would be orders of magnitude larger (each expert
shard is real MB-to-GB scale, not KB scale).

## Real check #2: is the model shard actually fully resident?

Initial check via `ps aux`'s RSS column showed only 16.46 GB for the
real runner process — a real, initially confusing ~5x mismatch against
the expected TP=2 per-node shard size (166,878,536,440 bytes total
model ÷ 2 ≈ 83.5 GB). This looked like it might indicate weights were
NOT fully resident, motivating a deeper check rather than assuming
either the ps result or the size math was simply wrong.

**Resolved via `vmmap --summary <pid>`**: real breakdown showed
`IOAccelerator (graphics)`: 86.7GB VIRTUAL / 86.7GB RESIDENT / 86.6GB
DIRTY, with the process TOTAL at 88.0GB resident / 87.4GB dirty.
**`ps aux`'s standard RSS column excludes GPU/Metal-owned unified
memory** — on Apple Silicon's unified memory architecture, GPU buffers
are tracked through a separate `IOAccelerator` VM region, not counted
in the conventional per-process RSS metric most tools (including `ps`)
report. This was a real measurement-tool gap, not a genuine memory
shortfall — resolved by using the right tool (`vmmap`) rather than
trusting the first (wrong) number.

**Real resident memory (87.4GB) closely matches the expected shard size
(~83.5GB)** — confirming the model shard genuinely IS fully resident in
unified memory on this node, not partially paged or evicted.

## Conclusion

**Memory residency / expert-weight paging is conclusively ruled out as
a cause of decode's idle-gap pattern.** Both real checks — near-zero
pageins during a real decode request, and confirmed full shard
residency via the correct memory-accounting tool — independently
support this. This closes Step 5 of the investigation plan with a
clean, real negative result, and also surfaces a real, reusable
lesson: `ps aux`'s RSS column is NOT a reliable memory-footprint metric
for MLX/Metal processes on Apple Silicon — GPU-resident unified memory
requires `vmmap --summary` (specifically the `IOAccelerator (graphics)`
line) to see accurately. Any future investigation checking this
cluster's memory usage should use `vmmap`, not `ps`, to avoid
re-encountering this same false alarm.

## Disposition

Closes Step 5. Remaining open items from the investigation plan: Step 2
(CPU profiling, deferred — py-spy not installed, no install approval
given within the session), Step 4 (cross-rank skew correlation, needs
clock-synced capture), and full kernel-level attribution (needs a fresh
Instruments capture with per-kernel labels, per
`docs/gpu-idle-gap-deep-dive-2026-08-22.md`).

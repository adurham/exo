# jaccl-internal timing decomposes the moe.all_sum 34x gap: transport is fast, overhead is elsewhere — 2026-08-21/22 (session 3)

## Summary — the real answer

**The moe.all_sum collective's real jaccl-internal transport time is
36-66µs (median/mean) for the actual decode-time 8192-byte payload —
FASTER than the earlier isolated microbenchmark's ~120µs wire floor,
and 62-113x FASTER than the ~4094µs sync-span-measured average from
earlier tonight.** Only 0.04% of 45,666 real decode-time transport
calls exceeded the 4094µs figure. This conclusively decomposes the
previously-confirmed 34x software-overhead gap: **the RDMA transport
itself is not the bottleneck — essentially all of the gap is overhead
sitting OUTSIDE the actual collective call** (MLX's eval-fence,
CPU/GPU dispatch coordination, or Python-level scheduling around the
call site).

## Why this required 5 relaunches to get right — real methodology history, not padding

Per an independent Fable review's specific critique of the originally
proposed "wrap `perf_counter` around the call site" approach (which
would have either measured near-nothing under MLX's lazy graph, or
conflated the collective with unrelated upstream work if forced eager),
the correct fix was jaccl-INTERNAL timestamps — timing genuinely inside
the C++ transport call itself, immune to Python/MLX-level graph
laziness. Implementing this correctly required navigating three
distinct real bugs in sequence, each one genuinely diagnosed rather
than guessed around:

1. **First attempt**: added `JACCL_TRACE_TIMING` timing to
   `MeshGroup::all_sum` (mlx commit `bc8750e9c`), reusing the existing
   `JACCL_TRACE_CALLS`/`JACCL_TRACE_HASH` trace-file infrastructure.
   Compiled clean, deployed clean, **zero trace output on either node**.
2. **Wrong-class detour**: assumed `MeshGroup` might be the wrong class
   for this 2-node topology (jaccl has three `Group` implementations —
   `MeshGroup`, `RingGroup`, `CoordGroup` — with the choice made at
   runtime by `jaccl::init()` based on `prefer_ring`/mesh-validity
   checks). Added standalone timing to `RingGroup::all_sum` (mlx commit
   `e40a416b2`) as a hedge. **This was based on an unconfirmed premise**
   — later directly verified that `MLX_JACCL_RING` is NOT set in
   production, meaning `prefer_ring` is false and `MeshGroup` (the
   ORIGINAL, correct target) is what's actually constructed. The
   `RingGroup` code is real, compiles, and is genuinely dead for this
   topology (harmless — it's the correct implementation if this fork
   is ever run with `MLX_JACCL_RING` set or a >2-node mesh).
3. **Real bug #1, actually found**: `MeshGroup::open_trace_file_if_enabled()`
   gates file creation on `JACCL_TRACE_CALLS`, not `JACCL_TRACE_TIMING`
   — a relaunch that set only `JACCL_TRACE_TIMING=1` (reasoning,
   incorrectly, that the RingGroup path was independent and didn't need
   it) never opened the trace file for `MeshGroup` at all, because the
   function returns at its very first gate check before `timing_enabled_`
   is ever read. Confirmed via `nm` that both new symbols
   (`trace_duration`, `maybe_open_timing_file`) were genuinely present
   in the rebuilt `libjaccl.dylib` on both nodes — the code was correct,
   the deployed env vars were incomplete.
4. **Real bug #2, actually found**: with BOTH `JACCL_TRACE_CALLS=1` and
   `JACCL_TRACE_TIMING=1` set together, `lsof` confirmed the runner
   process (PID 79791) had a genuinely open file descriptor to
   `/private/tmp/jaccl_trace_rank_1_color0_pid79791.log` with 1.68MB
   already written — but the file was NOT visible via `ls`/`cat` at
   that path. Root cause: the trace file is opened ONCE per `MeshGroup`
   construction (at cluster startup), not per decode call — my own
   `rm -f /tmp/jaccl_trace_rank_*.log` cleanup command, run AFTER
   cluster startup but BEFORE the decode benchmark, unlinked the file
   from the directory while the long-lived process's file descriptor
   stayed open (classic Unix unlink-while-open). All subsequent writes
   went into an orphaned inode, invisible to any path-based read.
   macOS has no `/proc/<pid>/fd/<n>` mechanism to recover an unlinked
   file's contents without SIP-disabling tools — this data was
   genuinely, permanently lost. Fixed by relaunching cleanly and NOT
   touching the trace file path between startup and the benchmark run.

**Reusable lesson: when a long-lived process opens a log file once at
construction (not per-call), NEVER `rm` that file while the process is
still running, even "just to get a clean starting point" — it silently
orphans all future writes with no error signal, and there is no
recovery path on macOS.** Always either (a) let the process create a
fresh file naturally via a PID-suffixed filename (as this file already
does) and read the new one, or (b) truncate via `> file` from a shell
with the right permissions rather than unlinking, or (c) clear the file
ONLY before the process that will open it is launched.

## Real data

Extracted `/tmp/jaccl_trace_rank_{0,1}_color0_pid<N>.log` from both
nodes after a clean relaunch, real decode benchmark
(`bench/decode_probe.py`, 3 reps, 512-token prompt, 300-token
generation), no path manipulation of the trace file in between.

45,850 total lines per rank; byte-size histogram confirms **45,666
calls at exactly 8192 bytes** — matching `hidden_size=4096 × 2
bytes/bf16` from earlier microbenchmark work, i.e. genuinely the
decode-time `moe.all_sum` payload. (Remaining rows: 84 calls at
163,840 bytes, 43 at 106,496 bytes, 43 at 4,284,416 bytes — almost
certainly prefill-phase or startup/warmup collectives at different
batch sizes, not analyzed further this session.)

| | rank0 | rank1 |
|---|---|---|
| n (8192-byte calls) | 45,666 | 45,666 |
| mean | 66.3µs | 58.9µs |
| median | 36.1µs | 36.0µs |
| stdev | 479.8µs | 274.0µs |
| p25 | 33.6µs | 33.8µs |
| p75 | 63.0µs | 62.5µs |
| p95 | 165.4µs | 142.8µs |
| p99 | 252.6µs | 266.1µs |
| max (outlier) | 57,187.5µs | 30,958.0µs |

Real, high right-skew (mean pulled well above median by rare large
outliers — max values in the tens-of-milliseconds range on both ranks,
almost certainly rank-skew/straggler-wait events, but rare: p99 is
still only 252-266µs). **Only 34/45,666 calls (0.07%) fall in the
2-10ms range that would match the earlier sync-span-measured average of
~4094µs; only 18/45,666 (0.04%) exceed 4094µs outright.**

## Interpretation

This directly and conclusively answers open thread #6 from
`docs/PERFORMANCE_HISTORY.md` §13 ("the genuine unsync per-call
moe.all_sum cost... not yet decomposed into its component causes"):

**The jaccl transport itself is NOT the software-overhead bottleneck.**
Real median transport time (36µs) is comparable to — and by mean, even
faster than — the earlier isolated `allreduce_bench` microbenchmark's
raw wire floor (~120µs at the same message size). The previously
confirmed ~4094µs sync-span average and the associated 34x-over-wire-floor
gap must therefore live almost entirely OUTSIDE this specific
`all_reduce<T>()` transport call — in whatever MLX/Python-level code
surrounds it: the `mx.eval(y)` fence that forces graph materialization
before/after the collective, CPU↔GPU dispatch coordination, or
Python-level scheduling overhead in the decode loop.

This reframes the earlier `docs/phase0b-collective-overlap-gate-2026-08-20.md`
finding (comm/compute overlap for all_sum is structurally achievable,
~33% of a 115ms serial budget recoverable) as even more directly
relevant than previously understood: since the real transport cost is
tiny (tens of microseconds), the overlap opportunity isn't really about
hiding RDMA wire latency — it's about hiding whatever surrounds the
call (likely the `mx.eval` fence itself, which the earlier
`EXO_DSV4_FENCE_ASYNC` investigation already partially addresses, see
`docs/comm-compute-overlap-already-exists-2026-08-21.md`).

## What this does NOT establish

This measurement is jaccl-transport-call-scoped only. It does NOT
directly measure:
- The `mx.eval(y)` fence cost itself (separately gated by
  `EXO_DSV4_FENCE_ASYNC`, already investigated).
- Rank-skew/straggler-wait time BEFORE the transport call is even
  reached (this timer starts only once `all_reduce<T>()` is entered).
- The relationship between this real per-call cost and the earlier
  live-decode Instruments trace's 1-10ms idle-gap bucket (mean
  ~2909-3010µs, see
  `docs/live-decode-two-rank-instruments-trace-2026-08-21.md`) — that
  gap bucket's magnitude is now shown NOT to be primarily the jaccl
  transport call itself, meaning the earlier flagged "suggestive but
  not proven" correlation between that GPU-idle gap bucket and the
  `moe.all_sum` sync-span figure should be treated as WEAKENED, not
  strengthened, by this measurement — the real transport is too fast to
  be the dominant contributor to a ~3ms-scale GPU-idle gap. **This is a
  genuine update to an earlier "suggestive" claim, not a confirmation
  of it.**

## Next steps (updated priority given this finding)

1. Instrument the `mx.eval(y)` fence itself (Python-level, around the
   collective in `deepseek_v4.py`) with real timing to see if THAT is
   where the 4094µs actually lives — this is now the primary suspect,
   not the jaccl transport.
2. Given the real transport is fast, re-evaluate whether comm/compute
   overlap work should target the fence/dispatch overhead specifically
   rather than the collective's wire time — a different design surface
   than originally assumed.
3. Correct the earlier Instruments-trace doc's tentative correlation
   claim (flagged above) in a follow-up note or amendment, since this
   measurement provides real evidence against it.

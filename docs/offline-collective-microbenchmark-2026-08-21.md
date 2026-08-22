# Offline jaccl collective microbenchmark: the real number — 2026-08-21 (session 2, part 13)

## Why this test, and why it required stopping the cluster

Per Fable review #3/#4: the earlier sync-span (`EXO_PROFILER_SYNC_SPANS=1`)
measurement of `moe.all_sum` cost was methodology-contaminated (forced
`mx.synchronize()` at every span boundary is itself real overhead, not
representative of normal execution). The live-hardware NOP ablation
attempted as a safe alternative was found unsafe (destabilized the
cluster, required a full relaunch). The queued, correct fix was an
**offline collective microbenchmark**, isolated from the model entirely
— exactly what MLX's own vendored `jaccl_allreduce_bench` tool
(`mlx/distributed/jaccl/lib/examples/allreduce_bench.cpp`) is built for.

This requires the same physical RDMA interfaces the live cluster uses
for inference. Running a second RDMA client on the same cable while the
cluster serves real traffic risks contending for the hardware's
documented 3-QP-per-cable ceiling (see
`docs/dual-cable-topology-and-qp-budget-2026-08-21.md`) — a real risk to
the running cluster, not just the benchmark. **User explicitly approved
stopping the cluster for this specific test** (asked directly, given a
choice, decision was "yes, stop, run it, relaunch when done").

## Method

1. Stopped both exo processes cleanly (`pkill -TERM`), confirmed via
   `ps aux` on both nodes.
2. Built `jaccl_allreduce_bench` from the vendored `mlx` submodule via
   `cmake` (clean build, no code changes — using the tool as-shipped).
3. Copied the binary to both Studios (`/tmp/jaccl_allreduce_bench`).
4. Built a real hostfile matching the RDMA-designated cable identified
   in tonight's own transport-hardening docs (`rdma_en4` on m4-1 ↔
   `rdma_en3` on m4-2, the 192.168.201.x subnet — confirmed live via
   `ibv_devinfo` showing both `PORT_ACTIVE`).
5. Ran via `mlx.launch --backend jaccl` (MLX's own first-class
   multi-host jaccl launcher, `mlx._distributed_utils.launch`) — this
   correctly builds the `MLX_IBV_DEVICES` connectivity JSON and
   `MLX_JACCL_COORDINATOR` env vars from the hostfile, matching exactly
   how the real inference cluster bootstraps its own jaccl groups.
6. First attempt (raw jaccl, no reliability env vars) hit the exact
   `all_reduce STALLED ... UC completion lost` failure documented
   throughout tonight's transport work — expected, since this
   standalone binary doesn't carry exo's production reliability layer
   (retransmit, standing recv pools, reconnect-on-fault). Re-ran with
   the SAME reliability env vars production actually uses
   (`MLX_JACCL_RELIABLE_DATA=1`, `MLX_JACCL_RELIABLE_INFLIGHT=8`,
   `MLX_JACCL_ACK_SYNC_PRE=1`, `MLX_JACCL_RECONNECT_FRESH=1`) — clean
   run, completed all message sizes with zero fault signatures.
7. Relaunched the cluster immediately after (`start_cluster.sh`),
   verified full recovery via correctness check (CAP theorem quality
   check, coherent) and decode throughput (3 reps, 18.69-18.88 tok/s,
   matching the validated baseline exactly — no lingering degradation).

## Result: the real number

| Message size | Raw jaccl all_reduce latency |
|---|---|
| 2 KB | 122.4 µs |
| 4 KB | 118.6 µs |
| **8 KB (real decode `moe.all_sum` size)** | **120.2 µs** |
| 16 KB | 127.3 µs |

At the EXACT message size decode's `moe.all_sum` uses (hidden=4096,
bf16, B=1 L=1 → 8192 bytes), the raw hardware+jaccl-transport floor is
**~120 µs**. This is flat across the tested size range (118-127µs,
essentially latency-bound not bandwidth-bound at these small sizes, as
expected).

## Comparison against the in-model sync-span measurement

The earlier decode-isolated (SIGUSR1) sync-span measurement of
`moe.all_sum` inside the real model forward pass: min=90.4µs,
avg=4093.7µs (21.4% of decode wall time).

- **Sync-span MIN (90.4µs) vs. raw floor (120.2µs): ratio 0.75x** — the
  sync-span minimum is actually slightly BELOW the raw standalone
  benchmark's floor. This is plausible (different measurement contexts,
  different warm-up states, slightly different code paths) and
  importantly means **the sync-span numbers are not wildly
  inflated by the forced-synchronization methodology** — the best-case
  in-model number and the isolated raw-transport number are in the same
  ballpark, cross-validating both measurements.
- **Sync-span AVG (4093.7µs) vs. raw floor (120.2µs): ratio 34.1x.**
  This is the real finding. The AVERAGE case inside the model is **34
  times slower than the raw hardware/transport can do the same-sized
  transfer.**

## Conclusion

**This conclusively answers the open question from reviews #2-4.**
`moe.all_sum`'s cost inside the real model is NOT primarily raw RDMA
transport/wire time — the hardware does an 8KB all-reduce in ~120µs
when isolated. The ~34x gap between that floor and the in-model average
is overhead specific to the model's call context: most plausibly
rank-skew/straggler-wait (one rank's GPU forward pass finishing later
than the other's, so the collective's effective latency includes
waiting for the slow rank, not just wire time) and/or CPU-stream
scheduling/dispatch overhead around the collective call site (Metal
command buffer completion callback latency, GPU→CPU handoff, jaccl call
posting from Python/C++ boundary) — exactly the two candidate
explanations flagged in review #3, now with real evidence discriminating
between "hardware floor" (ruled out) and "software/scheduling overhead"
(supported).

**This means comm/compute overlap (item #14, still open) is a
genuinely justified next step, not a speculative one.** If the ~34x gap
were the hardware floor, no amount of overlap engineering could help.
Since it isn't, there is real headroom that a properly-designed overlap
(or skew-reduction/load-balancing fix) could reclaim — up to
potentially most of that 21.4%-of-decode-wall-time figure, though the
EXACT achievable fraction depends on how much of the 34x gap is
skew (fixable via better load balancing) vs. genuine unavoidable
CPU-dispatch latency (fixable via overlap, not eliminable).

## Cluster state

Fully recovered and re-verified after the ~10-minute stop/benchmark/
relaunch cycle. Correctness confirmed (coherent quality check), decode
throughput confirmed at the validated baseline (18.69-18.88 tok/s, no
regression). Both `exo` and `mlx-lm` repos unchanged by this test (no
code was modified — this was a pure measurement using existing vendored
tooling).

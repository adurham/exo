# Instrumentation Session Findings — 2026-05-23 PM (16:00-17:00 CDT)

## TL;DR

1. **Built a per-step JSONL tracer** in `dsv4_mtp.py::_draft_tokens_batched`
   gated by `EXO_DSV4_C2_TRACE=1`. Captures per-chain-step timestamps, per-
   stream tokens (pre/post broadcast), Metal memory deltas, eagle install
   wall, predict wall, argmax+broadcast wall. Cycle summary record has
   bistability_flag (any step > 2× cycle median).
2. Shipped at commit `a1caaeb3` on origin/main + start_cluster.sh forwarder
   for the env gate.
3. **At γ=3 K=1 FENCE=4 with TRACE=1 on a 160-token prompt: 5/5 iters
   symmetric, 1158 cycles 0 bistable, agg 26.82 t/s σ=0.02.**
4. **At γ=3 K=1 FENCE=4 with TRACE=0 (control): 5/5 symmetric, agg 27.34
   t/s σ=0.07.** Tracer overhead is only ~2% throughput, not the 33% I had
   estimated.
5. **Bistability is NOT reproducing in either configuration on the current
   code state at 160-token prompts.** The prior session's "iter 1 = 40.57,
   iter 2 = 21.16 collapsed" observation does not reproduce in this
   environment.
6. **Per-step trace confirms the chain is rock-stable.** wall_step_ms
   p50=2.93, p99=3.13, max 3.90 across 3474 step records. predict() is
   86% of step wall (median 2.57ms); eagle install + argmax+broadcast are
   small.
7. **Cross-rank tok_post equality: 3474/3474.** The broadcast is producing
   bit-identical outputs across ranks on every step. **Within-rank
   tok_pre == tok_post: 3474/3474.** Both ranks compute the same argmax
   locally, so the broadcast is a no-op confirmation, not a divergence-
   masking event. (At this prompt length / iter count, no rank-drift is
   observed.)
8. **100K-context bench is still running.** That's the user's real
   workload; the 160-tok results may not generalize.

## What was instrumented

`dsv4_mtp.py::_draft_tokens_batched` now contains:

- Module-level `_C2_TRACE_ENABLED` gate from `EXO_DSV4_C2_TRACE=1`
- `_c2_trace_write()` JSONL writer (modeled on existing
  `_tree_alpha_probe_write`)
- `_c2_trace_metal_mb()` helper for `mx.metal.get_active_memory()` /
  `get_peak_memory()` capture
- Per-step record contains:
  - `ts_step_start_ns`, `ts_after_eagle_install_ns`, `ts_after_predict_ns`,
    `ts_step_end_ns` (perf_counter_ns)
  - `wall_step_ms`, `wall_eagle_install_ms`, `wall_predict_ms`,
    `wall_argmax_broadcast_ms`
  - `tok_post_broadcast_per_stream` (after coord-group broadcast)
  - `tok_pre_broadcast_per_stream` (rank-local argmax)
  - `metal_active_mb_start/end`, `metal_peak_mb_start/end`
  - `eagle_installed`, `temp_zero`, `cycle`, `step`, `B`, `gamma`, `pid`
- Per-cycle summary record:
  - `cycle_wall_ms`, `per_step_wall_ms`, `median_step_wall_ms`,
    `max_step_wall_ms`, `bistability_flag`

**Important caveat: tracer inserts `mx.eval()` at every step boundary so the
timestamps reflect real wall, not lazy graph fill.** Those evals act like
the proposed per-step fence fix — a trace run cannot VALIDATE a fix, only
LOCATE the mechanism.

## Environment used for this session

- `EXO_SPECULATIVE_GAMMA=3` (the config we want to make stable)
- `EXO_DSV4_MTP_EAGLE_K=1` (matches the prior session's config)
- `EXO_DSV4_FENCE_EVERY_N_LAYERS=4` (production default since 2026-05-22)
- `EXO_DSV4_INDEX_TOPK=512`, `EXO_KV_CACHE_BITS=0`, `EXO_DSV4_MTP=1`
- `LOG_LEVEL=WARNING` (kill 30KB/s router debug spam)
- `EXO_TRACING_ENABLED=false`
- `EXO_PROFILER_LEVEL=0`
- `EXO_LAYER_EVAL_INTERVAL=0`
- `EXO_DSV4_C2_TRACE=1` (or 0 for control)

## Cluster state checks before benching

Both nodes had stale runners from the prior FENCE=2 failed experiment.
Killed cleanly (`lsof ... | xargs kill`, `pkill -f exo.main`) + screen -wipe.
GPU verified actually-idle via `sudo powermetrics --samplers gpu_power`:
m4-1 GPU 93% idle 12-31 mW, m4-2 GPU 90% idle 18-31 mW. The dashboard's
"100% GPU 67°C 93W" was stale cached telemetry from the prior experiment.

This is a documented pitfall now: **dashboard polling cadence outruns the
underlying GPU-state sampler when the cluster is just sitting on a
resident model with no decode active.** When the dashboard shows ~95W on
a non-bench cluster, sanity-check with `powermetrics` before doing any
"the cluster is stuck" debugging.

## Bench results (160-token prompt — NOT the production workload)

### Run 1: γ=3 K=1 FENCE=4 + TRACE=1

```
warmup 0: per-stream=[13.40, 13.40] agg=26.80 sym=1.000
iter 0:   per-stream=[13.41, 13.41] agg=26.81 sym=1.000 bistab=False
iter 1:   per-stream=[13.41, 13.41] agg=26.82 sym=1.000 bistab=False
iter 2:   per-stream=[13.43, 13.43] agg=26.85 sym=1.000 bistab=False
iter 3:   per-stream=[13.41, 13.41] agg=26.82 sym=1.000 bistab=False
iter 4:   per-stream=[13.40, 13.40] agg=26.80 sym=1.000 bistab=False

agg_mean=26.82 σ=0.021 min=26.80 max=26.85
symmetry_mean=1.000 symmetry_min=1.000 bistab_iters=0/5
```

Tracer captured 1158 cycles (= 5 iters × ~230 cycles/iter from 256
max_tokens decoded per stream). 0 cycles flagged bistable.

### Run 2: γ=3 K=1 FENCE=4 + TRACE=0 (control)

```
warmup 0: per-stream=[13.66, 13.66] agg=27.32 sym=1.000
iter 0:   per-stream=[13.63, 13.63] agg=27.27 sym=1.000 bistab=False
iter 1:   per-stream=[13.71, 13.70] agg=27.41 sym=1.000 bistab=False
iter 2:   per-stream=[13.69, 13.69] agg=27.38 sym=1.000 bistab=False
iter 3:   per-stream=[13.63, 13.63] agg=27.26 sym=1.000 bistab=False
iter 4:   per-stream=[13.70, 13.70] agg=27.39 sym=1.000 bistab=False

agg_mean=27.34 σ=0.073 min=27.26 max=27.41
symmetry_mean=1.000 symmetry_min=1.000 bistab_iters=0/5
```

**Bistability is not reproducing on 160-tok prompts in either config.**

## Trace data analysis (γ=3 K=1 FENCE=4 TRACE=1, 1158 cycles)

### Per-step wall distribution (n=3474)

```
wall_eagle_install_ms:    p50=0.156  p90=0.168  p99=0.234  max=0.308
wall_predict_ms:          p50=2.572  p90=2.666  p99=2.756  max=3.600
wall_argmax_broadcast_ms: p50=0.248  p90=0.289  p99=0.386  max=0.495
wall_step_ms (total):     p50=2.930  p90=3.026  p99=3.129  max=3.903
```

### Per-step distribution by step index (γ=3, B=2)

```
step=0: n=1158 p50=2.858 p90=2.948 max=3.903
step=1: n=1158 p50=2.984 p90=3.060 max=3.212
step=2: n=1158 p50=2.933 p90=3.016 max=3.234
```

**No step-index has a systematic slow tail.** No "step 2 always slow" or
"iter-N+1 step 0 always slow" pattern.

### Cross-rank divergence (m4-1 vs m4-2 step wall)

```
|m4-1.wall - m4-2.wall|: p50=0.009  p90=0.022  p99=0.116  max=0.175 (ms)
```

Cross-rank step walls are within 116µs at p99. The two ranks execute the
chain in tight lockstep.

### Cross-rank token equality (post-broadcast)

3474/3474 (cycle, step) pairs have identical `tok_post_broadcast_per_stream`
between m4-1 and m4-2. Broadcast is functioning correctly.

### Within-rank pre vs post broadcast

3474/3474 records have `tok_pre_broadcast_per_stream == tok_post_broadcast_per_stream`. The broadcast is never overriding a rank's local argmax in this run — meaning at 160-tok prompts both ranks produce bit-identical
argmax locally, so MLX rank-drift is not manifesting.

### Metal memory growth

```
first cycle (18):    active=79023 MB peak=79484 MB
middle cycle (1176): active=79023 MB peak=79490 MB  (+6 peak over 1158 cycles)
last cycle (2332):   active=79340 MB peak=79490 MB  (+317 active, +6 peak)
```

The +317 MB active is the prefill-cache extension across the 5 iters (each
iter has its own KV cache). No allocator leak.

## Code paths read this session

### `dsv4_mtp.py::_draft_tokens_batched` (lines 1486-1791)

The c=2 batched draft chain. Loops `γ` times calling `self.mtp.predict()`.
Each step:
1. Optional Eagle soft-emb install (K≥1, only at i≥1)
2. `predict()` → (logits, h)
3. Eagle clear in finally
4. temp=0: `argmax(logits)` → `broadcast_from_canonical(tok_arr)`
   temp>0: `categorical(logits)` → `broadcast_from_canonical(tok_arr)`
5. (No per-step fence here — that's the prior plan's intended fix point.)

### `mtp_module.py::draft_tokens` (lines 660-790)

The c=1 chain. Same structure but with a per-step `mx.eval(tok_arr)` fence
at line 786. Comment at lines 669-681 explains the rationale: chained
predicts queue γ lazy `all_sum`s, peer-CQE arrival tail accumulates, fence
forces drain.

### `mlx/mlx/distributed/jaccl/mesh_impl.h`

Key constants:
- `NUM_BUFFERS = 2` (per size class per peer, used by both send & recv)
- `PIPELINE = 2` (also 2)
- `ACK_RECV_POOL = 64` (only for dedicated ACK QP, subgroups)
- `FRAME_SIZE = 4096`, `BUFFER_SIZES = 8` (8 size classes)

The buffer pool indexes by `(sz, buff, peer)` and reuses slots across
consecutive collectives. Recv buffer is zeroed before each post (lines
975-989) as a defensive measure against partial DMA fill in the cross-call
reuse case.

### `mesh.cpp::all_reduce` / `all_gather` / `send` / `recv`

Each collective dispatch acquires `collective_mutex_` (lines 576, 590,
606, 631). **Per-MeshGroup, all collectives serialize through this
mutex.** Chained collectives on the same MeshGroup cannot overlap their
data-buffer use.

### `MLX_JACCL_ACK_SYNC_PRE` env gate (mesh_impl.h:48-57)

Default OFF. The pre-lambda ack barrier added in `ce5c64fd` was found to
make c=2 WORSE alone (33.93 → 15.30 t/s) and was env-gated. So all
collectives currently rely on `ack_sync_post` (post-lambda barrier) for
cross-rank sync. No pre-barrier is in place — the inter-lambda recv-FIFO
race window described in the doc comment is genuinely open at runtime.

## Hypotheses about γ=3 c=2 100K bistability mechanism

(Updated after this session — the prior plan's hypothesis is no longer
supported by the data on 160-tok prompts; we're waiting on 100K bench to
see if it reproduces there.)

### H1: Bistability is c=2 + 100K + γ-depth dependent (not just γ-depth)

The 160-tok bench reproduces neither bistability nor the 40.57 t/s peak.
At 100K context, the verify-forward dominates wall (~150ms+ per token from
sparse-attention indexer), and the chain-collective queue depth interacts
with that timing differently. Test: 100K c=2 γ=3 K=1 FENCE=4 bench is
running now. If THAT shows bistability, the mechanism is context-length
dependent. If not, the prior session's measurement was transient.

### H2: Bistability needs cluster warmup state we don't have

The prior session ran multiple back-to-back experiments before hitting the
40.57/iter-2-collapse pattern. There may be a thermal / KV-cache-state /
jaccl-QP-state precondition for the failure mode that a fresh cluster
doesn't have. Pitfall #9 ("prefill rate degrades on long-uptime clusters")
suggests this is a real class of bug.

### H3: The 40.57 was the same "iter-0 warmup transient" pattern we've
seen before

From session_search of 2026-05-22 phase14 handoff Q3: "the FIRST decode
step may have a different code path (e.g. first _next() call goes through
_first_step_and_capture_batch which runs a non-spec forward — different
completion timing)." The 40.57 might just be the asymmetric-warmup
phenomenon where one stream gets ahead and the agg appears inflated, and
then "iter 2 collapsed" is actually steady-state. If so, the real γ=3 c=2
steady-state IS ~27 t/s (what we're measuring now), NOT 40+.

If H3 is correct, we have a SERIOUS PROBLEM: γ=3 at 27.3 t/s is LOWER than
γ=2 at 34.16 t/s. The extra chain depth costs more than it saves.
Acceptance rate would need to be very high to overcome this.

### H4: 100K-context attention bandwidth dominates and decouples γ from
throughput

At 100K context, the per-token verify-forward through the sparse-attention
indexer is doing massive (compress + index + select-K) work. The MTP
draft cost is small relative to that. The acceptance rate per chain step
determines whether γ pays off. If at 100K context the acceptance rate is
high (e.g. 2+ tokens accepted per γ=3 cycle on average), then γ=3 wins;
if it's flat at γ=2's level (also 2/3 accepted), γ=3 has the same yield
at higher wall.

## Open questions for the 100K bench

1. Does the 100K c=2 γ=3 K=1 FENCE=4 result land near 27 t/s (consistent
   with the 160-tok finding) or near 40 t/s (matching the prior session)?
2. Does bistability reproduce at 100K?
3. What's the MTP acceptance rate? (We don't have telemetry on this yet
   in our instrumentation.)
4. If 100K shows 27 t/s and stable: **γ=3 is a loss vs γ=2 and the prior
   session's 40.57 was a measurement artifact.** Revert plan, look for a
   different path to 35 t/s.
5. If 100K shows 40+ t/s with bistability: **the per-step fence becomes
   the validated mitigation** (we have evidence it stabilizes); root cause
   work continues on the residual JACCL FIFO race window described in the
   `MLX_JACCL_ACK_SYNC_PRE` doc comment.

## Code added this session (committed, on origin/main)

Commit `a1caaeb3`:
- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`: +278 lines
  (tracer module-level state, per-step record emit, per-cycle summary)
- `start_cluster.sh`: +6 lines (EXO_DSV4_C2_TRACE forwarder)

Behavior when `EXO_DSV4_C2_TRACE=1` is NOT set: no records written, no
extra mx.eval, no overhead.

## Scripts left for next session

On laptop:
- `/tmp/launch_g3k1_traced.sh` — launches cluster with TRACE=1 + γ=3 K=1
- `/tmp/launch_g3k1_notrace.sh` — control (TRACE=0)
- `/tmp/bench_g3k1_traced.py` — 5-iter c=2 bench (short prompt) with
  per-stream gen_tps + symmetry diagnostics
- `/tmp/bench_g3k1_100k.py` — same bench at 100K context (the right
  workload)
- `/tmp/dsv4_c2_trace_m4-1.jsonl` — full trace from run 1 (4633 records,
  2.6 MB)
- `/tmp/dsv4_c2_trace_m4-2.jsonl` — same from m4-2

On m4-1:
- `/tmp/bench_g3k1_traced.py`, `/tmp/bench_g3k1_100k.py`
- `/tmp/g3k1_traced_5iter.{json,log}` — traced run results
- `/tmp/g3k1_notrace_5iter.{json,log}` — control run results
- `/tmp/g3k1_100k_3iter.{json,log}` — 100K bench results (in progress)
- `/tmp/dsv4_c2_trace_pid*.jsonl` — local trace files

## Next-session resumption

1. Read `/tmp/g3k1_100k_3iter.log` on m4-1 first.
2. If 100K bench shows steady-state ~27 t/s (no spike, no collapse):
   the γ=3 path is dead, look for a different 35-t/s lever.
3. If 100K bench shows 40+ t/s steady-state stable: declare victory on
   γ=3 (no fix needed) and update production default to γ=3.
4. If 100K bench shows the 40.57/21.16 bistability shape: the prior plan's
   per-step fence approach is needed. Compare against the trace data we
   now have at 160-tok (which IS rock-stable with the fence) to confirm
   the mechanism transfers.
5. Eventually if needed, dig into the JACCL FIFO race window
   (mesh_impl.h:32-57 + ack_sync_post path) which is the actual structural
   risk acknowledged in the code comments.

# Prefill request-trace instrumentation: findings — 2026-08-21 (session 2)

## Context

Continuing from the same-day jaccl transport hardening session (see
`docs/dual-cable-topology-and-qp-budget-2026-08-21.md` and
`docs/known-good-prefill-baseline-2026-08-21.md`). That session's handoff,
based on a Fable consult, claimed:

> `request_trace.dump()`/`.reset()` are documented in the module docstring
> as the intended usage pattern but are NEVER actually called anywhere in
> the real production code paths — the spans are being recorded into an
> in-memory list but nothing ever logs/dumps them.

**This claim was WRONG for the serial prefill path**, and the error
propagated from a stale grep (searching for the literal string
`request_trace.dump`) that missed an import alias. This doc corrects the
record and captures what real trace data from tonight's rerun actually
shows.

## Correction: the wiring already exists

`src/exo/worker/engines/mlx/generator/generate.py`'s `prefill()` function
(the serial, single-stream prefill path — used by every concurrency=1
benchmark, which is what all of last night's and tonight's throughput
numbers come from) does:

```python
from exo.worker.engines.mlx.trace import request_trace as _rt
_rt.reset()      # line 779, prefill() entry
...
_rt.dump()        # line 960, prefill() exit
```

`git blame` confirms this was added by commit `8a207e9969` on **2026-06-21**
— nearly two months before tonight's session incorrectly declared it
missing. A stale grep for the unaliased `request_trace.dump` string (used
both in this session and in an even older warm-memory fact, #658, also
now stale) missed the `as _rt` import alias and produced a false negative.

**What actually IS still unwired** (this part of the original claim is
correct): `prefill_batched()` — the ≥2-concurrent-task batched-prefill path
used by `submit_batched()`/`_batched_start_task()` — has `T(...)` spans
throughout but no `reset()`/`dump()` call anywhere in that path or its
caller. Those spans really do accumulate and vanish. This does not affect
single-stream benchmarking (which is 100% of the throughput numbers
gathered so far) but would need the same two-line fix if batched-prefill
tracing is ever needed.

**Corollary — nothing needed building.** The originally planned "wire up
the tracing" step reduced to: restart the cluster with
`EXO_TRACING_ENABLED=1` (a module-level env read at import time, can't be
toggled on a live process) and pull the already-working trace output from
`~/exo.log`.

## What was done tonight

1. Relaunched the cluster via `EXO_TRACING_ENABLED=true ./start_cluster.sh`
   (backgrounded per the standing Bash-background rule — this is a ~9 min
   full node-prep + build + placement cycle, not a quick command).
   Verified `EXO_TRACING_ENABLED=true` in both nodes' live process env via
   `ps aux` before trusting any subsequent trace output.
2. Reran the exact same depth ladder as last night's baseline
   (`bench/phase3_precheck_depth_throughput.py`, targets
   100000,300000,500000, tokenizer-ground-truth token counts, needle-in-
   haystack correctness check on every run) to (a) reconfirm parity with
   the known-good baseline post-restart and (b) generate fresh trace data.
3. Pulled the real `[TRACE] Request timeline:` blocks from both nodes'
   `~/exo.log` (8 dumps total: 1 warmup + 3 real depths × 2 ranks) and
   parsed the per-span offsets/durations.

## Reconfirmed baseline (post tracing-enabled restart)

| Context | Prefill throughput | Decode throughput | Needle check |
|---|---|---|---|
| 100,000 tokens | 368.7 tok/s | 17.95 tok/s | PASS |
| 300,000 tokens | 351.9 tok/s | 18.50 tok/s | PASS |
| 500,000 tokens | 333.2 tok/s | 17.16 tok/s | PASS |

Matches last night's known-good numbers (366.6 / 351.5 / 331.6 tok/s
prefill) within run-to-run noise. `EXO_TRACING_ENABLED=1` adds no
measurable overhead at these depths, confirming the trace.py docstring's
"near-zero overhead when off / negligible when on for T() spans" claim in
practice, not just by inspection.

## Real trace-data findings

Named framework spans instrumented via `T(...)` — `prefill.clear_cache`,
`prefill.barrier`, `prefill.mem_checkpoint`, `prefill.cache_trim_and_rollback`
— total single-digit milliseconds (worst case ~35ms) against prefill wall
times of 190s (100K), 600s (300K), and 1057s (500K). That's **under 0.02%**
of prefill wall time. Cluster-level orchestration/barrier/memory-checkpoint
overhead is definitively NOT the prefill bottleneck — this is now measured,
not assumed.

The remaining ~99.98% of prefill wall time lands inside a single opaque
`prefill.stream_generate` span — this wraps mlx-lm's own internal chunked
generator loop, which is not instrumented at the sub-chunk level from our
side. However, `distributed_prompt_progress_callback` — which mlx-lm calls
once per internal prefill chunk — IS captured via `T("prefill.distributed_callback")`,
and its firing cadence is a reliable proxy for real per-chunk compute time.

Chunk-to-chunk delta analysis (time between consecutive
`distributed_callback` firings) at all three depths:

- **100K** (20 callbacks): min 5.13s, max 5.51s, mean 5.41s, spread 6.9%
- **300K** (21 callbacks): min 5.12s, max 5.61s, mean 5.43s, spread 9.1%
- **500K** (21 callbacks): min 5.11s, max 5.57s, mean 5.44s, spread 8.5%

Flat, low-variance (~7-9% spread, no growth trend, no stall outliers) at
every depth tested. This is the signature of steady, real compute+collective
work — not a hidden queueing bug, network stall, or memory-pressure cliff
that would show up as a widening or drifting cadence at deeper context.

## Conclusion

**Prefill is confirmed compute-bound at the instrumented granularity.**
This directly answers the question the tracing work was meant to test
(per Fable's framing: microsecond tracing tests whether prefill is
compute-bound, it doesn't act on the assumption that it is). The answer is
yes — there is no orchestration/barrier/collective-control-plane overhead
worth chasing at this level. The earlier 12-lever prefill campaign's
"~1-3% ceiling, no further orchestration ideas" conclusion holds; tonight's
jaccl transport hardening did not uncover a hidden non-compute lever,
because there wasn't one to find at this instrumentation depth.

## Open item / real next step

The model-side `SpanProfilerHook` (`EXO_PROFILER=spans`, distinct from
`EXO_TRACING_ENABLED`) can see *inside* the opaque `prefill.stream_generate`
span down to per-kernel granularity (`moe.switch_mlp`, `attn.indexer`,
`attn.sdpa`, `attn.compressor`, `attn.all_sum`, etc.) — but only when run
with `EXO_PROFILER_SYNC_SPANS=1`, which forces `mx.synchronize()` at every
span boundary so each span measures its own real GPU kernel time instead of
MLX's lazy/async graph-build time (see class docstring in
`mlx-lm/mlx_lm/profiler.py::SpanProfilerHook` — without sync mode, spans
only bracket graph submission, and lazy compute piles onto whichever span
forces the next eval, typically an `all_sum` collective, which would
falsely look like ~100% of wall time). This run had `EXO_PROFILER` unset,
so no model-side span data exists yet for prefill.

Decode was already profiled this way on 2026-06-26 (see warm memory facts
743/751) and found MoE-bound (`moe.switch_mlp` = 73% of decode wall time).
Prefill's equivalent breakdown has never been captured — it goes through a
different, non-`@mx.compile`'d code path in this config
(`EXO_DSV4_COMPILE_FFN=0`, `EXO_DSV4_COMPILE_LAYER=0`, confirmed via live
`ps aux` on both nodes), which removes the June 2026 "compiled paths hide
Python-level spans from the profiler" limitation (see warm memory fact
743) — sync-span profiling should see real prefill kernel attribution in
this config, unlike the compiled-forward runs where it previously
couldn't.

**Next step:** relaunch with `EXO_PROFILER=spans EXO_PROFILER_SYNC_SPANS=1`
and rerun a prefill pass to get the real per-kernel time-budget breakdown
inside prefill (the equivalent of the 2026-06-26 decode breakdown). Expect
measurably reduced absolute throughput during this run (sync-per-span
serializes the pipeline) — that's expected overhead from the profiler
itself, not a regression; only the *relative* per-kernel percentages are
the useful output, not absolute tok/s from a sync-spans run.

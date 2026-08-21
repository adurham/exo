# all_sum bypass ablation: unsafe, cluster required a fresh relaunch — 2026-08-21 (session 2, part 10)

## Why this test

Follow-on to Fable review #3, which flagged that the sync-span RTT
comparison (min=90µs, avg=4094µs per `moe.all_sum` call, decode-only
isolated via SIGUSR1, 21.4% of decode wall time) is methodology-
contaminated — `EXO_PROFILER_SYNC_SPANS=1` forces `mx.synchronize()` at
every span boundary, which is itself real overhead not present in normal
execution, so the absolute microsecond numbers can't be trusted as the
"real" unsync cost. Fable's recommended safe alternative: a differential
ablation using the codebase's own built-in `all_sum` NOP file-toggle
(`/tmp/dsv4_nop_targets`, monkey-patches `mx.distributed.all_sum` to
pass through unreduced when the toggle file contains `"all_sum"`) —
measure end-to-end decode throughput with and without the collective,
no profiler needed, garbage output is fine since only tok/s matters.

## What happened

1. Established a clean baseline (no profiler, `EXO_DSV4_MOE_FUSED_GATE_UP=1`
   only) via `bench/decode_probe.py`, 5 reps: ~18.6-19.0 tok/s, consistent
   with all prior gate-up-only measurements this session.
2. Enabled the NOP toggle on both nodes (`echo -n "all_sum" >
   /tmp/dsv4_nop_targets`) — this is a live file-toggle, no restart
   needed, code already exists in the deployed `deepseek_v4.py`.
3. Reran the same probe. Result was NOT a clean "faster but garbage
   output" measurement as hoped — it was **unstable and partially
   failing**: run 1 at 3.48 tok/s (18x slower, not faster), runs 3-4
   returned 0 tokens (outright failures), runs 2/5 at ~22.4 tok/s (the
   only runs that looked like a plausible "faster" result). TTFT on the
   failing/slow runs spiked to 44-47s (vs the normal ~1-4s).

## Why this happened (diagnosis, not fully investigated)

Skipping the cross-rank reduce means each TP rank computes on its OWN
unreduced partial sum from that point forward — the two ranks' hidden
states diverge immediately and irreversibly for the rest of that
forward pass (and, since decode reuses KV cache state across steps,
likely corrupts cache state for subsequent tokens too). This is much
more disruptive than the ablation appears at first glance: it's not
"same computation, wrong final answer" (which would just look like
garbage output at normal speed) — it's "the two ranks' subsequent
computation trajectories diverge," which plausibly triggers all sorts of
downstream inconsistency (mismatched cache shapes/positions between
ranks, degeneration kill-switch false-triggers from repetitive/corrupt
logits, task-agreement collectives seeing inconsistent state). The wide
variance (18x slower on some runs, faster on others, outright failures
on others) is consistent with hitting different failure modes on
different runs, not a clean single effect.

## Recovery

Removing the NOP toggle file did NOT restore normal service — a
follow-up correctness check still failed with HTTP 500 after the toggle
was reverted. The runner had self-healed at the jaccl transport layer
(`jaccl reconnect complete... resumed serving with model resident`
visible in the log) but remained unable to serve real requests. A full
clean cluster relaunch (`./start_cluster.sh`, no toggle, standard
config) was required to fully recover. Post-relaunch, both a short
correctness check and a 100K-context needle-in-haystack check passed
cleanly, confirming full recovery.

## Conclusion

**This specific ablation approach (`all_sum` NOP toggle) is NOT a safe
way to measure the collective's true cost** — despite existing in the
codebase as an apparently-designed diagnostic tool, it destabilizes the
cluster badly enough to require a full relaunch, not just a quick
correctness recheck. This is now empirically confirmed, not assumed.
This is a genuinely more disruptive test than the earlier `EXO_PROFILER`
sync-span or `EXO_DSV4_SEQSPLIT_GATHER_VIA_ALLSUM` tests tonight, both of
which recovered cleanly via the runner's own reconnect/retry logic
without needing a full relaunch.

**Does not change the underlying finding**: `moe.all_sum` is confirmed
real and substantial (21.4% of decode wall time under sync-span
isolation, the best available measurement even with its acknowledged
methodology caveat). What remains genuinely unmeasured is the PRECISE
unsync magnitude — this session did not find a safe way to get that
number. The qualitative conclusion (TP collective cost is large enough
to be the primary remaining decode-side lever, comm/compute overlap is
the real next project) stands on the sync-span evidence alone; the
ablation attempt neither strengthens nor weakens it, it just failed to
provide a cleaner confirming number.

## Recommendation

**Do not use the `all_sum` NOP toggle for throughput measurement again.**
If a cleaner unsync measurement is wanted in a future session, the safer
path (not attempted tonight, higher engineering cost) would be adding a
NEW dedicated timing instrumentation point that measures the collective
call's wall time directly at its call site using plain
`time.perf_counter()` around just that call (not `mx.synchronize()`
before AND after every span, which is what makes sync-span mode
expensive/distorting) — closer to how `attn.all_gather`'s cost was
already captured via `T(...)` spans in the request-trace path earlier
tonight, which did NOT require forced synchronization and did not
destabilize anything.

Cluster confirmed fully healthy and back to the validated-good
production config (`EXO_DSV4_MOE_FUSED_GATE_UP=1` only) after this
test, via full relaunch + real correctness verification at both short
and 100K-context depth.

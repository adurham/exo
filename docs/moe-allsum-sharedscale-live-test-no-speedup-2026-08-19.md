Shared-scale int8 all_sum: correctness passed, but NO speedup and real GPU-utilization asymmetry (2026-08-19)
==================================================================================================================

Summary
-------

Deployed the shared-scale int8 all_sum design (100% on jaccl's reliable
all_sum path, no all_gather, overflow-fixed qmax -- see
`moe-allsum-quant-root-cause-and-closure-2026-08-19.md` and the
`feat/moe-allsum-sharedscale-2026-08-19` mlx-lm branch, commit
`187d0f0`) live to the 2-node cluster for the first real end-to-end
test.

**Correctness: PASSED.** 38,067-token fresh prefill, secret-code
needle recall correct (truncated by max_tokens=50, not a quality
failure), `cached_tokens: 0` confirmed fresh. No crash, no HTTP 500 --
a real, clean improvement over the earlier all_gather-based attempt.

**Speed: NO measurable improvement.** ~165-168 tok/s cumulative,
statistically indistinguishable from the unquantized baseline (~162-172
tok/s measured repeatedly all night at the same 2048-token step size).
The projected ~1.44x end-to-end speedup did not materialize.

**New, real anomaly: sustained GPU-utilization asymmetry during the
request.** User caught this directly from the exo dashboard (visual:
one node ~34% GPU / 31W, the other 100% / 68W) and it was confirmed via
direct `/metrics` polling -- 4 samples during the active request showed
m4-1 consistently 98-100% while m4-2 sat at 32-44%, a real and sustained
imbalance, not a one-frame snapshot artifact. Both nodes returned to
symmetric ~3% idle immediately before and after the request -- the
asymmetry is specific to this code path being active, not a pre-existing
hardware difference (standing baseline runs both nodes at comparable,
high, roughly-symmetric utilization).

Diagnosis (not yet confirmed, reverted before further live debugging)
---------------------------------------------------------------------------

Consulted on the likely cause. Ranked hypotheses:

1. **Most likely**: the two-phase design (Phase 1: tiny scalar all_sum
   to agree on a shared scale; Phase 2: the real int8 payload all_sum)
   introduces an extra synchronization point per layer that isn't
   overlapping cleanly across ranks. One rank may be genuinely computing
   (the low-utilization one) while the other spins/polls waiting on a
   completion (reads as "100% busy" per the same GPU-utilization-vs-
   real-work distinction found earlier tonight in
   `docs/gpu-util-vs-allsum-cost-reconciled-2026-08-19.md` -- a GPU
   submission thread parked in an uninterruptible wait can read as fully
   utilized while doing zero useful arithmetic).
2. Checked the code directly for an explicit `if rank == 0` /
   rank-privileged hot path in `_sharedscale_compute_scale` /
   `_quantized_moe_all_sum_sharedscale` -- **none found**, both
   functions are symmetric across ranks by construction. This rules out
   the simplest explanation (an accidental single-rank bottleneck coded
   directly into the reduction).
3. Not yet checked: whether prefill's per-layer `all_sum` payload size
   at this cluster's real shape sits in a latency-bound regime rather
   than a bandwidth-bound one -- if so, the extra Phase-1 round trip's
   FIXED latency cost could be eating the Phase-2 bandwidth savings
   almost entirely, explaining both "no speedup" and consistent with
   (but not proof of) the asymmetry.

Why this wasn't debugged further live tonight
---------------------------------------------------

Per the standing "don't iterate against production hardware on
half-understood distributed bugs" discipline (same reasoning as the
all_gather thread's closure) -- got a real, correct, but not-yet-
beneficial result, plus a genuine new anomaly that needs proper
attribution (per-rank timestamped tracing around the all_sum calls,
not guessing from utilization telemetry alone) before any further live
attempt. Reverted the mlx-lm submodule pointer to the standing pin and
relaunched the cluster at baseline config immediately.

Status
---------

Code remains correct, committed, and available on
`feat/moe-allsum-sharedscale-2026-08-19` (mlx-lm submodule), pushed,
NOT merged. **Do not redeploy without first**: (a) adding per-rank
wall-clock timestamps around each phase of the collective to identify
which rank is actually the straggler and by how much (Fable's suggested
discriminating test), (b) checking the real per-layer payload size
against jaccl's latency/bandwidth crossover point to see if this
collective is even in the bandwidth-bound regime the whole design
assumes.

Cluster restored to standing baseline (`EXO_PREFILL_STEP_SIZE=2048`, no
sharedscale flag) and confirmed healthy after this test.

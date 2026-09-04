# OVERNIGHT LOOP — STATE

**Mode:** AUTONOMOUS (authorized 2026-09-02, runs until user says STOP). **Currently PAUSED
awaiting user direction** (2026-09-04 ~11:30 CDT) — no PM in flight.
**Charter:** `/Users/adam.durham/.hermes/cache/OVERNIGHT-LOOP-CHARTER.md` (read first on context loss;
includes the mandatory GREP-THE-RECORD-FIRST step).
**Supervisor model:** claude-opus-5

---

## WHERE THINGS STAND

**Campaign 1 (2026-08-30 → 09-03): CLOSED.** The 20-vs-34 tok/s "gap" was a measurement
convention (94.5% TTFT). Shipped: pad-strip (server cached_tokens 0→351), canonical serializer +
golden bytes, BatchPoolingCache fix, profiler unit metadata, CI re-enabled + guards live.

**Hardening (09-03): DONE.** CI had been `disabled_manually` for 8 months; now runs pytest green
(1107 passed). Alignment guard, kernel guard, golden-token nondeterminism root-caused (OOB embedding
read into recycled Metal pages). tmp/ 16GB→34MB + rsync exclude.

**Campaign 2 (09-03 → 09-04, 8 rounds): THROUGHPUT PHASE CLOSED. ZERO SHIPPED.**
Every lever closed with source/GPU evidence. Decode is at a physics floor for this model on this
hardware; prefill at c=1 is exhausted and confirmed on the production path. Full ledger below.
Five supervisor errors in briefs were caught by PMs and are in the record.

---

## CAMPAIGN 2 LEDGER (per-item, with the round that closed it)

| # | lever | verdict | round |
|---|---|---|---|
| I1 | TP all_sum latency | CLOSED — jaccl 2.6% of budget; MLX has NO GPU collective (`AllReduce::eval_gpu` throws, jaccl stream is CPU); GPU-resident collective INFEASIBLE (no public GPU→DMA coherence API; jaccl host-bounce by construction). Stack-level floor. | R1-R3 |
| I2 | c=2 tax knobs | CLOSED — 3 knobs dead code; FENCE cadence live but moot with async fence armed; `MLX_STEEL_BATCH_INVARIANT` is CORRECTNESS-load-bearing at c=1 (byte-identity fails both regimes); RENDEZVOUS=0 safe but unresolvable on the TTFT instrument → folds into TTFT pivot | R1/R4/R7 |
| I3 | kernel bandwidth | CLOSED — 83.9% of peak (chained-graph). The "53%" was the retracted 08-22 serial-sync artifact reproduced | R1 |
| I4 | Fix B re-open | CLOSED — round-4 test was invalid (sub-chunk) but conclusion held on a valid re-test | R1 |
| I5 | γ re-tune | CLOSED — all reachable γ measured on a calibrated ruler; γ=4 is −4.5 t/s vs 1.3 boot spread; cycle 61→74ms monotonic in γ. HOLD γ=3 | R5/R6 |
| I6 | per-row expert reads | CLOSED — 1.42× shared-vs-distinct, ~4-8ms of 56ms | R1 |
| I7 | lm_head vocab-sharding | CANCELLED — ~0.45 t/s, below measurement floor | R7 review |
| I8 | per-draft cost | RESOLVED — ~3-6ms per extra draft row; acceptance doesn't cover it | R6 |
| I9 | GPU P-state | CLOSED — decode clock 0.2% ABOVE prefill | R8 |
| I10 | Fix A trie persistence | CANCELLED by user — "I don't normally re-launch in the middle of sessions." The 49% figure measured CAMPAIGN relaunches, not normal usage; ~0 value in steady state | user, 09-04 |
| I11 | expert precision | **PREMISE FALSE** — experts are ALREADY mxfp4 4-bit at load (`deepseek_v4.py:952`). "6-bit" was a supervisor error in 3 briefs. 3-bit = +4%, inside boot variance. No change | R8 |
| I12 | serial-vs-batched prefill parity | CLOSED — all six optimizations reachable from the serial driver by construction | R8 |
| I13 | idle-gap warmup | CANCELLED — only matters if warmup is in the shipping loop | R7 review |
| I14 | within-boot drift | CANCELLED — measurement artifact | R7 review |
| I15 | kernel-launch count | BLOCKED — probe vars boot-gated; rides free on the next boot | R8 |

**Key facts established this campaign** (each source- or GPU-verified): async fence IS armed
(≥98.5%, the 08-22 +58-67% fix holds under MTP-on); verify GPU is 117% busy on real compute —
no idle gap; removing the entire collective moves wall ≤8.4%; the 06-26 "fence is load-bearing
for bit-equiv" story was an algebra bug; `FENCE_EVERY_N_LAYERS` HAS a live reader
(deepseek_v4.py:2959 — R4's "dead" claim was wrong); `EXO_PROFILER=spans` silently re-blocks the
fence; `ab_probe_tier1.py` is NOT a pass/fail gate.

**Supervisor errors caught by PMs (all in the record):** R1 "never attacked" framing → reproduced
a retracted artifact; R3 SHA mis-parse of `git submodule status` (hardening); R4 "FENCE is dead"
(wrong); R7 cited a nonexistent gate; R8 "experts are 6-bit" (wrong, 3 briefs). Process fix:
charter now mandates grep-the-record before any brief.

**Measurement reality:** ~6 t/s between-boot decode variance (P13-P15), 1.3 t/s within bracketed
boots this week; prefill boot-stable to 0.02%. Server `stats.generation_tps` via
`bench/long_decode_probe.py` (calibrated 29.1@2K, 32.8@89K) is the only trusted decode ruler.
Never build a new harness — R5 lost a round to one.

---

## NEXT (needs user direction — loop is paused)

Fix A is DEAD (user: relaunches are not part of normal sessions). With it gone, the TTFT pivot's
only remaining item is RENDEZVOUS_MS 200→0 (safe, proven; -480ms on the short-prompt instrument
but unresolved against the 200ms claim; needs a paired-boot design) — a one-knob round, and the
user should decide whether a ~200-480ms TTFT trim per turn is worth a boot cycle.

Everything else is closed. Recommendation: STOP the loop unless the user names a target.

**Cluster:** healthy on shipped production config (γ=3, BI=1, RV=200, mxfp4 experts, async
fence armed), both nodes READY, verified 2026-09-04. No probe/diag env leftover. Tree clean,
everything pushed (exo main @ 5e9717fe7 + this file).

# OVERNIGHT LOOP — STATE

**Mode:** AUTONOMOUS (authorized 2026-09-02, runs until user says STOP)
**Charter:** `/Users/adam.durham/.hermes/cache/OVERNIGHT-LOOP-CHARTER.md` (read this first on context loss)
**Supervisor model:** claude-opus-5

---

## ROUND IN FLIGHT

- **deleg_f38de5cd — Ask B: decode-wall instrumentation** (dispatched 20:31)
  - G1 control run (bracket env ON, decode-profiling OFF, patch unapplied) then instrumented
    S1-S4, attribution of the ~5.0% out-of-bracket wall to candidate A (fenced coord collectives,
    `dsv4_mtp.py:2259-2310`) vs candidate B (`agree_on_tasks`/`agree_on_cancellations_fast`,
    `batch_generator.py:678-720`), then revert.
  - Pre-flight findings already banked: A-fence EXECUTES every decode cycle at c=1 (gate verified
    verbatim) so the experiment can satisfy its own gate — unlike Ask A's SDPA arm.
    `decode.step.mlx_next` spans are recorded but never dumped, so the G2 coverage denominator
    was substituted (documented in `PRE-REGISTRATION-ASKB.md`).
  - Artifacts: `tmp/prefill-round4-exec-askb-20260902/`

## NEXT ACTION WHEN IT COMPLETES

1. Append findings to `docs/PERFORMANCE_HISTORY.md`, commit + push (same turn).
2. `mcp__consult` fable with the results -> get round-N+1 direction.
3. Dispatch the next PM (`agent_type='pm'`, `role='orchestrator'`) with a task file in
   `~/.hermes/cache/`.
4. Update this file.

---

## CAMPAIGN LEDGER (what is closed, so no round re-opens it)

| Thread | Status |
|---|---|
| BatchPoolingCache overlap-carry defect | FIXED, deployed, verified live (37260bb) |
| Decode "regression" P12-P15 | Retired as boot variance |
| V2 acceptance-counter mining | Infeasible-proven (data never written) |
| V3 SPEC_STATE_RESTORE snapshot cost | CLOSED (rb_snap 0.150ms = 0.218% of cycle) |
| 11-16% unaccounted decode wall | Corrected to ~5.0% flat (accounting artifact) |
| Entropy hypothesis | Refuted (natural prose = repetitive) |
| Temperature hypothesis | Refuted at the premise (no delta exists) |
| Real-usage 20-vs-34 gap | Measurement convention: 94.5% TTFT, no decode loss |
| Prefix-cache hit rate | Near-perfect (97.6% session-wide); "0 full hits" correct-by-design |
| Pad-strip | SHIPPED (bdc9b6f1fc) — live proof cached_tokens 0 -> 351 |
| Serialization contract | SHIPPED (fb394a378) — canonical serializer + golden byte tests |
| Fix B (decode-KV retention) | DEAD — 6/9 serialization variants zero the cache |
| SDPA 4.06x per-call anomaly | CLOSED — batched-path only, production is c=1 serial |
| Prefill c=1 | Effectively exhausted (chunk overhead <0.02%, indexer 4% closed, clear-cache dead code) |
| P16 boot-variance characterization | PARKED by user |
| V4 c=2 concurrency | DROPPED by user (workload shape) |

**Reopen triggers:** prefix-keyed cache redesign + SDPA anomaly only if concurrency returns.

## OPEN QUESTIONS FEEDING FUTURE ROUNDS

- The ~5.0% out-of-bracket decode wall (Ask B answers this).
- Mild prefill depth degradation 426.0 -> 418.6 -> 406.6 t/s (fable: <1% ROI, low priority).
- Whether anything else in the client serialization path can cost cache (contract now guards it).

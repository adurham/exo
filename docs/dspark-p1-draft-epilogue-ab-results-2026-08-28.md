# P1 results: draft-epilogue fusion A/B — GATE FAIL (no throughput win), flag stays OFF (2026-08-28/29)

**Pre-registration:** `dspark-p1p4-campaign-preregister-2026-08-28.md` (committed
`3e23282dc` before any run). Stack: exo `75d2402dd` + mlx-lm `d098642`, verbon3
env ± `EXO_DSV4_DRAFT_EPILOGUE`. Artifacts: `/tmp/ab/p1p4/` (runs, log offsets,
`p1_verdict.json`).

## Verdict

| Gate | Bar | Result | Pass |
|------|-----|--------|------|
| 1 Tier-1 | 7/7 byte-identical (content+reasoning) to Aug-28 EPI-OFF captures | 7/7 identical | **PASS** |
| 2 Cross-arm identity | ON stream == OFF stream at both depths | byte-identical @100K (5768 ch) AND @352.6K (2170 chunks, identical) | **PASS** |
| 3 Within-arm determinism | all runs per (arm,depth) identical | 6/6 ON @100K one hash; 6/6 OFF; 3/3 ON @352.6K; 3/3 OFF | **PASS** |
| 4 Throughput @100K | median ≥ +8% AND min(ON) > max(OFF) | **−0.35%** median (37.62 vs 37.76); min(ON)>max(OFF) false; boot CI [−0.56, +0.51] | **FAIL** |
| 5 352.6K health | no collapses, swap <500MB, no faults | 0 collapses (30.25–30.42 all runs), swap 50/65 MB, 0 fault lines either node | **PASS** |
| 6 Mechanism | consume-cycle draft ≈ 0 ms | draft 8.20→0.55 ms — mechanism ENGAGED | PASS (see below) |

**Decision per pre-registered PROMOTE bar: NOT promoted.**
`EXO_DSV4_DRAFT_EPILOGUE` stays default-OFF. Correctness is fully clean — the
optimization simply does not deliver at the promoted config.

## Why the theoretical +16% did not materialize (mechanism, MTP-PROF windowed)

Per-cycle phase means @100K (m4-1, windowed per-run, n=400-450 cycles):

| Phase | OFF (EPI=0) | ON (EPI=1) | delta |
|-------|------------:|-----------:|------:|
| draft | 8.20 ms | 0.55 ms | −7.65 |
| verify | 64.18 | 64.08 | −0.1 |
| accept | 1.55 | 9.47 | **+7.92** |
| rollback | 2.03 | 0.50 | −1.5 |
| **total** | **75.98** | **74.61** | **−1.37 (−1.8%)** |

The consume path works exactly as designed (draft ≈ free at cycle start), but the
epilogue draft's cost re-appears in the accept-phase window (+7.9 ms): the
epilogue `_dspark.draft()` + `mx.eval` is **synchronous inside the cycle
epilogue** — it does not overlap with anything. The design doc's premise was that
the draft would overlap with the accept/rollback/bookkeeping tail (~10.3 ms); in
reality the tail work it was supposed to hide behind is itself mostly CPU-side
bookkeeping that the draft compute cannot run under (single Metal stream +
synchronous eval), so the fusion just MOVES the draft, minus small wins
(rollback −1.5 ms, draft-dispatch overhead −0.6 ms). Net −1.37 ms/cycle ≈ −1.8%
cycle time — and end-to-end fixed-window tok/s measured −0.35% (noise-level,
sign negative; min(ON) 37.56 < max(OFF) 37.86, arms overlap).

End-to-end run rates @100K (chunk-event rate over full 1500-token window,
byte-identical output both arms):

- OFF: 37.70/37.72/37.74/37.77/37.81/37.86 — median 37.76
- ON: 37.56/37.59/37.62/37.63/37.66/38.21 — median 37.62 (−0.35%)

@352.6K (informational, n=3/arm, shared window 2170 chunks): ON 30.25/30.27/30.30
vs OFF 30.35/30.35/30.42 — median −0.26% (boot CI [−0.57, −0.17]). Same mechanism
at depth (m4-1 windowed, n=700 cycles): draft 8.18→0.57 ms, accept 1.55→9.52 ms,
verify 84.8 (unchanged), total 96.5→95.3 ms (−1.2%) — the small cycle win again
fails to survive to end-to-end tok/s (dominated by the depth-scaled verify wall +
inter-cycle overhead).

## What this buys the campaign anyway

1. **The 4-stacked-changes static audit is now live-verified for EPI:** the flag
   ON is byte-identical to OFF at short ctx, 100K, and 352.6K — the consume path
   produces the identical draft (design's determinism claim confirmed on
   hardware). Any future revival (e.g. a genuinely async epilogue draft on a
   second Metal stream) starts from a proven-lossless base.
2. **Draft cost is confirmed second-order at the promoted config:** 8.2 ms of
   76 ms (10.8%) — even a PERFECT overlap would cap at ~+12%, and the measured
   verify wall (64 ms) is where the cycle lives. Future effort goes to verify,
   not draft.
3. Negative result recorded per pre-registration; no post-hoc bar adjustment.

## Files

- Runs: `/tmp/ab/p1p4/run_{on,off}{100k,352k}_*.json` (chunks + timestamps)
- Tier-1: `/tmp/ab/p1p4/tier1_epi_on.json` vs `/tmp/ab/g0_352/tier1_live.json`
- Phase extraction: `/tmp/ab/p1p4/p4_phase_extract.json` + `extract_p4.py`
- Verdict JSON: `/tmp/ab/p1p4/p1_verdict.json`

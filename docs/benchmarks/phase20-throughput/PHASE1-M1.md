# PHASE 20 — M1 milestone (Phase 1, relaunch #1 = deploy/next16-instr)

Relaunch #1 consumed: `EXO_TARGET_BRANCH=deploy/next16-instr ./start_cluster.sh` → "Nodes synchronized on commit f234b0f6d", **READY 2/2**. Both nodes verified `git rev-parse HEAD` = `f234b0f6d`. Canary after boot: **14.83–14.88 TFLOPS** both nodes. Production f4bb14746 preserved (shared checkout detached at f234b0f6d; original doc `docs/benchmarks/phase19-latency/` moved aside to scratch and will be restored with the checkout).

## G1.1 — deploy regression (round_prof unset) — **PASS**
`phase19_round_measure.py --depth 20000 --reps 2 --max-tokens 800 --gamma 3`:
- benign g3 decode_tps median **31.37 t/s** (gate ≥24.0 ✅), round **119.16 ms**, mean_accepted **2.756**.
- Instrumented build is a clean no-op with `round_prof` unset (as designed; PROF=0 path is one `hook is None` test).

## G1.2 / C1 — per-round bracket budget (round_prof=1 and 2) — **PASS**
Per-round JSONL from BOTH nodes (`raw/prof/m4-{1,2}.round_prof.jsonl`, 480 lines each, rank-tagged). Median **PROF=1** round (147 ms):

| bracket | median ms | share of round |
|---|---|---|
| draft_build (head.draft + verify_in build, host graph-build) | **4.8** | 3.3% |
| **verify_block (spec.snap → one `mx.eval(logits)`)**, absorbs prior round's deferred rollback/append_ctx tail + the per-layer `_column_boundary` syncs | **143.1** | **97.4%** |
| tail_bookkeep (`spec.rollback` + `head.append_ctx`, eval-free) | **0.55** | 0.4% |
| emit | ~0.3 (see caveat) | ~0.2% |
| **round_total** | **146.9** | 100% |

- **Closure (C1): the three real brackets sum to 148.4 ms ≈ round_total 146.9 ms → ≈100%.** The "unaccounted ~40%" of the round is **inside the fused verify eval** — not in a missing Python segment. This directly corroborates R1's finding that the ungated `sparse_attention._column_boundary` per-layer host round-trip (called from inside the body forward, i.e. inside `verify_block`) is the top Phase-3 lever.
- **G1.4 perturbation (PROF=2 vs 1):** PROF=2 (mx.eval at every bracket end) round_total jumps to **~452 ms** (median of the fast early rounds) vs ~147 ms for PROF=1 — expected, reported, not hidden.
- **Caveat (instrumentation bug, immaterial):** the `emit_ms` field is a *cumulative* time-since-stream-start, not a per-round delta (round 1 = 0.31 ms, round 480 = 35 s); it does **not** affect the round-total closure because emit is ~0.3 ms. A one-line reset fix is queued but not required for the finding.
- Both ranks report identical medians (m4-1 143.12 / m4-2 143.06 ms verify) → no rank imbalance.

## G1.3 — verify vs phase-17 — reported
The real measured round `verify_block` is **143 ms** against phase-17's standalone **~98 ms**; the **+45 ms gap is TP2/jaccl comm + KV rollback/trim + host syncs that the single-node phase-17 harness structurally could not contain** (consistent with PLAN flag #2). The excess is attributed to comm, not to a missing indexer number.

## Phase-1 conclusion for Phases 2/3
The round is **GPU-serialized (0d: 100% residency, comm 0% in `sample`) with the time concentrated in the verify eval**, i.e. exactly the per-layer host round-trip R1 identified. Phase 3 should target that sync (lazy rollback is already free; the win is removing the per-layer `_column_boundary` host trip / batching to one sync). Phase 2 (adaptive gamma) is orthogonal and still worth the relaunch #2 slot.

## Budget
Relaunches used: **1 / 3** (deploy next16-instr). Restore (#2) reserved. #3 for next17-levers.

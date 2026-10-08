# PHASE 20 — Phase-3 milestone (M3): decode round-time levers

Author: Phase-3 PM. Status: **measurements taken, byte-identity running**. See §6 for the
RESTORED line, filled after the final restore relaunch.

## 1. What was deployed, and how the relaunch was spent

Single remaining experimental relaunch (budget: next16-instr #1, restore #2 — ledger settled in
`PHASE3-DECISION.md` §0) was spent on plan **(i)+**: the **already-built, already-unit-tested**
`deploy/next16-instr` (f234b0f6d) with **BOTH lever gates disabled via env**:

```
DSV41_SPARSE_COLSPLIT=0   DSV41_INDEXER_HIER=0
```

`EXO_TARGET_BRANCH=deploy/next16-instr ./start_cluster.sh` → "Nodes synchronized on commit
f234b0f6d" → READY (2/2). Both nodes `git rev-parse HEAD` = `f234b0f6d`. Post-boot canary
14.83–14.87 TFLOPS both nodes (healthy). **Both gates confirmed present =0 in the LIVE runner
environ on both nodes** (ps eww) — the env took.

Rationale (from `PHASE3-DECISION.md`): the *code* fix is provable off-cluster, but the *value* of
a bundled code deploy (plan ii) rests on per-request A/B plumbing that is unprovable new code
whose failure mode is catastrophic on the last launch. The env-only deploy answers the load-bearing
question — *does removing these host-syncs actually convert to wall time, and how much* — at
near-zero risk. A second opinion concurred. The code fix was still built+proven (see §4) and is
one step from shipping.

## 2. Measured result (levers OFF vs the default-ON baselines)

All numbers from the `next16-instr` build, `round_prof=1` per round (per-request field), same
harness protocol as the Phase-1 baselines. OFF = both gates `0`; ON = build defaults (`COLSPLIT=1`,
`HIER=1`).

| workload | metric | **OFF (this run)** | ON baseline | Δ |
|---|---|---|---|---|
| benign 20K g3, 800 tok | round_total median | **94.9 ms** (5 reps, 94.87–95.04) | 119.16 ms (M1 G1.1, same build) | **−24.3 ms (−20.4%)** |
| benign 20K g3 | decode t/s | **39.0–41.4** (median 38.99) | 31.37 | **+25%** |
| agentic 91K g3, 800 tok | round_total median | **101.3 ms** (4 reps, 101.18–101.72) | 153.4 ms (M1) | **−52 ms (−34%)** |
| agentic 91K g3 | decode t/s | **29.4–31.2** (median 31.17) | 20.61 | **+35–51%** |
| either | mean_accepted | 2.60 benign / 2.07 agentic | 2.756 / ~2.0 | **unchanged** (not a gamma effect) |

Per-round PROF brackets, levers OFF (pooled, both ranks identical to <0.1 ms):

| bracket | median ms | share |
|---|---|---|
| draft_build | 0.55 | 0.6% |
| **verify_block** | **92.4** | **98.5%** |
| tail_bookkeep | 0.09 | 0.1% |
| round_total | 93.8 | 100% |

Per-request round time from the client: benign 94.9 ms (211–216 rounds), agentic 101.2–101.7 ms
(252–269 rounds). Rank0==rank1 (no rank imbalance).

## 2b. Same-session default-ON control (production f4bb14746 restored boot)

To remove cross-boot thermal drift from the ON reference, the identical protocol was re-run on
the **restored production boot** (f4bb14746, gates at their defaults; `round_prof` is a no-op
there, so only client-side round time is available — the PROF bracket files on that boot are
stale).

| workload | ON control (this session, prod boot) | **OFF** | Δ (same-session) |
|---|---|---|---|
| benign 20K g3 | 144.5 ms / 26.3 t/s (4 reps, 144.5–145.2) | 94.9 ms / 39.0 t/s | **−49.6 ms (−34%)** |
| agentic 91K g3 | 156.6 ms / ~18 t/s (3 reps, 156.55–156.61) | 101.3 ms / 29.4–31.2 t/s | **−55.3 ms (−35%)** |

mean_accepted held (benign ON 2.535–2.883 vs OFF 2.71–2.95; agentic ON 1.74–2.07 vs OFF 1.97–2.13),
so the gain is **not** a token-count artefact. Caveat: this ON control is a **cross-build AND
cross-boot** comparison; today's production boot is slower (144.5 ms) than M1's next16-instr run
(119.16 ms) — thermal/GPU-state drift between boots, the known confound. The same-build ON-vs-OFF
(§2) is therefore the primary number, and the same-session control **corroborates** a ≥30% effect
that is robust to the drift direction.

## 3. Attribution (which gate did it)

Both gates were flipped together in this one boot, so boot-level attribution is by the mechanism
and the shape-dependence of the delta, not by a second boot:
- the delta is **larger at deeper context** (24 ms @20K benign vs 52 ms @91K agentic) and the
  verify_block bracket is 98.5% of the round — consistent with **per-compressing-layer** host
  round-trips (Lever 1), whose count grows with the number of ratio>0 layers carrying `kv2` in
  each verify forward;
- Lever 2 (indexer hierarchy) touches only the 8 index-source layers and, at decode's large strip
  sizes, already merges to ~1 eval/layer — a small, context-flat contribution.
A single-gate split (COLSPLIT-only vs HIER-only) would cost the *restore* relaunch, which the
budget forbids; the combined magnitude and its ctx-scaling are the evidence, and this is stated
as a limitation (§7).

## 4. Code fix deliverable (verified, ready-but-unspent)

`deploy/next17-levers` (exo `576e9d279`, mlx-lm `3bf8316`, based on prod `f4bb14746`, **not** on
next16-instr): gates the C1 boundary derivation+check on the call's row count
(`if kv2 is not None and _COLSPLIT and m > _FENCE_MIN_ROWS:`) so decode (m=1) and verify (m=4)
skip the two host round-trips and fall back to the value-identical `_gather_split`; prefill (large
m) keeps C1 unchanged. Both gates carry a comment recording WHY their defaults disagreed.
New `test_dsv41_sparse_smallm_colsplit.py` = **17 tests, PASS** (independently re-run by the PM).
This makes the env flip unnecessary and is the shippable form of Lever 1.

## 5. Remaining gap to 30 t/s (honest)

- benign: 30 t/s at ~2.6 tok/round ⇒ need ~120 ms/round; OFF is already 94.9 ms ⇒ **benign >30 t/s
  is met at 20K**. At deeper/agentic shapes it is not.
- agentic: 30 t/s at ~3.07 tok/round (mean_acc 2.07) ⇒ need ~102 ms/round; OFF is 101.3 ms ⇒
  **agentic is right at 30 t/s** (29.4–31.2).
- The lever therefore lands the **PREREG Phase-3 gate (agentic ≥22.5 t/s = win; ≥30 t/s not
  expected)** — and, measured, it *does* reach ~30 t/s at 91K agentic. But this is **with gamma
  fixed at 3**; sustained deep-context production (128K+ turns) will dilute both the acceptance
  and the per-round saving, and the remaining ms-gap is shape-dependent, not a fixed number.

## 5b. R8a byte/quality gate (battery, levers OFF @ depth 40000)

`battery.py --label p3off --depth 40000 all` on the OFF build → **PHASE VERDICT: CLEAN
(needles 6/6, tools 10/10, prose DIRTY 0, REVIEW 0)** — element-wise identical to the frozen
default-ON `results/g3` baseline (same depth, needles 6/6, tools 10/10, prose 0 DIRTY).
`same_script_glue` ADVISORY hits 6 vs the baseline's 4 (both are the known noisy heuristic,
demoted to advisory; 0 cross-lingual glue fragments → byte-identity on the deterministic
subset (needles/tools) holds per PREREG D4, with prose detectors clean).
Artifacts: `bench/dsv41_quality_battery/results/p3off/`, `raw/p3/battery_p3off.stdout`.

Same-session default-ON control is taken from the RESTORE boot (f4bb14746, gates default) after
the restore, using the identical protocol — this removes cross-boot thermal drift from the
ON-vs-OFF comparison (§2 uses the M1 baselines as the primary ON reference; the restore-boot
control is the confirmatory same-session cross-check).

## 6. RESTORED line

`RESTORED f4bb14746 READY 2/2 canary 14.70/14.77 TFLOPS parity decode=20.7 (100K, reasoning-only) / 24.4–26.8 (20K benign) t/s prefill=269.5 rows/s`

### Affirmative parity evidence (production, deploy/next13 @ f4bb14746)
- **Both nodes** `git rev-parse HEAD` = `f4bb14746c68deea005f41f590e27e6b182b6384`, branch
  `deploy/next13`. Launcher: "Nodes synchronized on commit f4bb14746" → **READY (2/2)** @ 10:30:11.
- **Canary** after boot: studio1 14.81/14.70/14.70, studio2 14.77/14.76/14.79 → healthy.
- **Env parity**: 102/102 EXO/DSV41/MLX/MTL/IBV/AGX/PYTHON* var names identical to the pre-campaign
  snapshot `raw/prod-env-pre-relaunch1-m4-1-ps-eww.txt`; the two Phase-3 lever gates are **absent
  (unset)** → build defaults, i.e. production behaviour identical to before the campaign.
- **Decode parity**: benign 20K g3 = 24.4–26.8 t/s, agentic 91K = 17.5–19.5 t/s on this boot;
  fresh 100K-class prefill = **269.5 rows/s** (≥260 gate ✅).
- Production f4bb14746 was never modified: the shared checkout was only `git checkout`ed to
  f234b0f6d for the experiment and back; the untracked `docs/benchmarks/phase19-latency/` was moved
  aside and restored.

## 7. Limitations (stated, not hidden)

1. **Both levers were flipped together** in one boot; a per-gate split would cost the restore
   relaunch, which the budget forbids. Attribution (§3) is by mechanism + ctx-scaling, not a
   second boot. Lever 1 (the `_column_boundary` host sync) is the dominant, ctx-scaling term;
   Lever 2's residual is small.
2. **Single-arm-vs-baseline, not interleaved** — the PREREG `§1` interleaving rule is infeasible
   with one env-fixed boot. Amendment A-P3-1 in `PHASE3-DECISION.md` records this.
3. **Cross-boot thermal drift** between the same-build ON baseline (M1, 119.16 ms) and the OFF run
   (94.9 ms) inflates the apparent Δ; the same-session control (§2b) is ~30–35% and is the
   drift-robust estimate.
4. **The env-only deploy leaves the foot-gun** (two disagreeing module defaults). The code branch
   `deploy/next17-levers` (§4) removes it and is the recommended shippable form; it was **not**
   deployed because its per-request A/B plumbing carried unacceptable last-launch risk.


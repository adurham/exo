# PHASE 20 — Phase-3 decision record (written BEFORE the single remaining relaunch)

Author: Phase-3 PM. Written 2026-10-08 ~08:1x CDT, **before any Phase-3 deploy or measurement**.
This precedes the data it judges. Any later change is an amendment with a reason.

## 0. Relaunch ledger (settled from the shared-checkout git reflog + node log rotations)

| # | deploy | evidence | counts against budget? |
|---|---|---|---|
| 1 | `deploy/next16-instr` f234b0f6d (Phase 1) | HEAD reflog checkout `f4bb14746 → f234b0f6d` @ 2026-10-08 00:20:39 | **YES** (consumed) |
| 2 | restore `deploy/next13` f4bb14746 | HEAD reflog checkout `f234b0f6d → deploy/next13` @ 2026-10-08 00:37:27; node exo.log rotation `00:28:01` + current proc start 00:44:08 | **YES** (consumed) |
| 3 | THIS experiment | to be spent now | **the last one** |

Confirmed: the `PHASE20-FINAL-MF.md` line "Relaunch budget 2/3 used: #1 = deploy/next16-instr; #2 = restore"
is the **doc-side ledger only** and omitted the actual next16→next13 relaunch that the reflog records.
Authoritative fact for this phase: **exactly ONE experimental launch remains.** After it, only the
final restore launch (a separate, dedicated spend) is required to return production.
No earlier Phase-20 relaunch is recorded between 00:20 and 00:37 (single next16 deploy, single restore).

## 1. Decision: (i)+ — env-only relaunch of the already-built `next16-instr`, BOTH levers disabled

**Deploy target:** `EXO_TARGET_BRANCH=deploy/next16-instr` (unchanged, already built + 242 dsv41 tests green),
with `DSV41_SPARSE_COLSPLIT=0` AND `DSV41_INDEXER_HIER=0` exported on the laptop before `./start_cluster.sh`.
Both vars are confirmed forwarded by `start_cluster.sh:2708` and `:2674` (only-when-set).

### Why (i)+ and not (ii) [the bundled code fix]
The brief's own rule: prefer (ii) iff the code change is **proven correct + numerically neutral before deploying**.
The gate *fixes* are provable locally (byte-neutrality of the gather paths is stated in the C1 docstring and
testable with mlx on this laptop). But (ii)'s **value** rests on a per-request on/off A/B plumbing that is
**new, unproven code whose failure mode is catastrophic** (a TP-rank-divergent `mx.eval` count can corrupt or
hang the run on the last launch). Moreover, under the cluster's 5-concurrent-task model and possible user
arrival, a per-request *module-global* gate is unsound (request B's flip contaminates request A mid-decode;
mixed batches make "per-request arm" meaningless). A second opinion (auxiliary reference model) confirmed:
"(ii) fails its antecedent — take (i)." **Interleaving is therefore infeasible under a one-launch budget.**

### PREREG amendment (A-P3-1), recorded now, before data
`PREREG §1` requires ">=5 reps for any A/B claim, medians + IQR, arms interleaved". With ONE launch and
boot-time env, the two arms CANNOT be interleaved. Adopted rule:
- **Protocol:** same harness + identical prompts, same session, boot A (both levers OFF) then boot B —
  a **single-arm-vs-frozen-baseline design with within-boot g3 control arms** interleaved (control =
  instrumented build with the levers ON, i.e. production-default behaviour; treat = both OFF).
- **Boot B is `next16-instr` with both gates OFF.** **Boot A control = today's fresh production f4bb14746
  baseline** (same harness, same session, captured minutes apart) **PLUS** a within-boot `next16-instr`
  default-arm smoke to separate build-vs-env effects.
- **Primary evidence is boot-invariant:** the firing-rate of the two candidate syncs, measured directly
  (Lever-1 `_column_boundary` call count; Lever-2 per-strip eval count). These cannot drift.
- **Secondary:** `round_ms` median + IQR from `round_prof=1` JSONL (>=5 reps), and wall-clock ms/round from
  100K g3 interleaved reps. Because cross-boot loses thermal control, the delta is only claimed as a win
  if the firing counter confirms the mechanism AND the round delta clears the pre-registered +3 ms.

## 2. Pre-registered gates & falsifiers (frozen)

- **G-P3.1 (Lever-1 mechanism):** the Lever-1 firing counter must read ~0 `_column_boundary` derivations
  at decode with `DSV41_SPARSE_COLSPLIT=0`, vs a measured production firing rate >0. **No counter exists in
  next16-instr** (the C1 comment cites an adjacent measurement; no runtime count). => the primary Lever-1
  evidence is the **round ms delta + eval-count delta** only. If the round delta is within ±1%, Lever-1 is
  **FALSIFIED as inferred** (decode not entering the branch) and the finding is reported as a negative.
- **G-P3.2 (Lever-1 win):** `round_ms` median improvement >= **5 ms/round** on agentic replay AND no benign
  regression AND R8a clean. (PREREG Phase-3 gate text: >=3 ms/round ships.)
- **G-P3.3 (Lever-2):** `DSV41_INDEXER_HIER=0` changes which top-k path runs. Gate = chosen ids IDENTICAL
  (or quality-exact via battery) BEFORE attributing any win. Delta >= 3 ms/round required.
  **Prior risk (noted):** `DSV41_INDEXER_HIER=0` is a *different algorithm*, not a host-sync removal — the
  geometry log shows decode already merges to ~1 eval/layer, so the residual win is expected small.
- **G-P3.4 (R8a):** battery output byte-identical to a baseline captured ON THE CURRENT CLUSTER at gamma 3,
  conditional on a same-build A1-vs-A2 determinism replicate (PREREG D4). If not byte-identical, degrade to
  "identical on the deterministic subset (needles/tools) + detectors clean on prose".
- **Gate: agentic >= 22.5 t/s = win; >= 30 t/s not expected. Remaining ms-gap to 30 t/s reported honestly.**

## 3. Hard rules in force
Idle-guard before every chunk and relaunch; abort-on-arrival watcher every chunk (<=15 min); RAW-GPU canary
after boot and after READY, before any measurement; reboot only via `./reboot-node.sh studio1 studio2`
(max 2 cycles, never raw shutdown); FORBIDDEN `EXO_KV_CACHE_BITS!=0`, `EXO_DSV4_INDEX_TOPK<512`,
`repetition_penalty!=1.0`; artifacts force-added under gitignored bench/**.

## 4. Deliverables also in scope (ready-but-unspent if the launch is not spent on them)
- `deploy/next17-levers` (code fix, both levers) + `deploy/next16-levers` (code fix + per-request A/B
  plumbing) staged and **laptop-verified** (unit tests + byte-neutrality of the gather paths). NOT deployed.
- Phase-1 doc errata: the `PHASE1-M1.md`/`PHASE20-FINAL-MF.md` relaunch-ledger line corrected (see §0).

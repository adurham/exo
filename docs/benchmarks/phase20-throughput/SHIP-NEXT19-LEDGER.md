# SHIP-NEXT19-DENSE — ceremony ledger (resumable)

Owner directive (verbatim): "Promote to production — full ship ceremony (production branch, known-good tags, PH shipped line)".
Date: 2026-10-09 (CDT). PM: delegation PM. Operator session: depth-1 subagent.

## DECLARED BUDGET
1 ship boot + <=6 same-boot restarts + 1 reserve boot (rollback). Idle-guard every chunk; canary after boot AND after READY.

## TARGET
Union-merge production branch `deploy/next19-dense` = fb4f9290b (next18 production lineage) UNION cfd74d49f (dense eval lineage) + mlx-lm gitlink bump -> 689e4ea.
Frozen perf (vs exl3 control): agentic 87.24 ms (control 101.07, delta -13.83); benign 80.97 ms (control 94.56, delta -13.59). 15 ms floor = owner-accepted MISS.
Fallback (pre-authorized rollback): exo deploy/q1-dense-qn @ cfd74d49f + mlx-lm 689e4ea; instant in-place = DSV41_DENSE=exl3.

## LEDGER
| # | step | status |
|---|---|---|
| 0 | recon (lineage, guard, doc conventions, leases .48/.47 current) | DONE |
| 1 | build union merge + bump (mid-coder) | DONE: tip 99e2966e (merge e82023e7b + bump 99e2966e); pushed both forks |
| 2 | union verification (PM) | DONE: gitlink 689e4ea; engine bootstrap+builder liveness present; all 8 tokens present; 0 residual eval strings; uv.lock unchanged (245cef7b) |
| 3 | deploy boot + canary + installed-module verify | DONE: synced 99e2966ee, READY 2/2, canary 14.77/14.85 healthy; installed exl3_build.py has _parse_dense_policy+resolve_dense_mode; runner env exactly DSV41_DENSE=affine6 + policy (no strays) |
| 4 | battery depth 40000 + parity smoke + t2 probe | DONE — ALL GREEN: battery CLEAN (needles 6/6, tools 10/10 incl t2, prose 0 DIRTY/0 REVIEW, park PASS); smoke benign 81.30 ms (ctl 94.56) / agentic 86.03 ms (ctl 101.07), both within +/-2ms; t2 probe 10/10 structured (no XML leak). FALSIFIER DID NOT FIRE |
| 5 | tags both forks + PH line (main) + ROUND-Q1B-FIX sec-PROMOTION | DONE — tag known-good-dense-q68-20261009-231842 (exo 99e2966ee, mlx-lm 689e4ea) pushed both forks; PH d1714990d on main; ROUND section-PROMOTION dbfaa41e2 on deploy/phase20-campaign; raw artifacts committed |
| 6 | end-state + budget accounting | DONE — shared checkout detached @99e2966ee clean; cluster LIVE (READY 2/2, canary 14.86/14.86 healthy, baked defaults); no strays; temp main worktree removed |

## END STATE (verified)
- Shared checkout `~/repos/exo`: HEAD `99e2966ee` (detached), mlx-lm WT `689e4ea`, working tree CLEAN.
- Cluster LIVE on the production build: `Nodes synchronized on commit 99e2966ee.`; HEALTHY; **READY (2/2)** (both runners RunnerReady); canary **14.86 / 14.86 healthy**; runner env exactly `DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64` (no strays); installed `exl3_build.py` carries the policy code on both nodes.
- Tags (both forks): `known-good-dense-q68-20261009-231842` (exo `99e2966ee`, mlx-lm `689e4ea`) — pushed, deref-verified.
- Docs: PH shipped line on `main` (`d1714990d`); ROUND-Q1B-FIX section-PROMOTION + raw ship artifacts on `deploy/phase20-campaign` (`dbfaa41e2`).
- No stray processes (local or nodes). Temp main worktree removed.
- Pre-registered falsifier: DID NOT FIRE.
- **Budget: 1 ship boot + 0 arm restarts + 1 reserve (held/unspent).**

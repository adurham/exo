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
| 1 | build union merge + bump (mid-coder) | IN PROGRESS |
| 2 | union verification (PM) | PENDING |
| 3 | deploy boot from shared checkout + canary + installed-module verify | PENDING |
| 4 | battery depth 40000 + parity smoke + t2 probe | PENDING |
| 5 | tags both forks + PH line (main) + ROUND-Q1B-FIX sectionPROMOTION | PENDING |
| 6 | end-state + budget accounting | PENDING |

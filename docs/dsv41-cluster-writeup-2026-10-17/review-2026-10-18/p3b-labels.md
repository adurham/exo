[FIX] evidence.md:L638 "`iogpu.wired_limit_mb = 115000` -> 120.6 GB"
SRC: evidence.md:L634 "| `sysctl iogpu.wired_limit_mb` | **120000** (120 GB) | **120000** |"
FIX: One conversion rule: 115000 MiB = 120.6 GB, but 120000 MB = 120 GB decimal (125.8 GB if MiB). State which.

[FIX] evidence.md:L552 "**210.58 GB** (\"196.2 GiB = 210.7 GB\" in `docs/deepseek-v41-exl3-plan.md:116`)"
SRC: evidence.md:L551 "= **210.68 GB** (196.22 GiB readable)"
FIX: Not rounding: 210.58 GB = 196.12 GiB; 196.2 GiB = 210.67 GB, matching measured 210.68 GB. Relabel GB/GiB conversion.

[FIX] evidence.md:L91 "CONVENTION-AMBIGUOUS (nominal depth; B1 re-measured real 352.6K at 23.48, +9.2%, methodological gap)"
SRC: evidence.md:L43 "CONFLICT vs p3 doc (23.48 @352,599)"
FIX: Prepend CONFLICT: same 21.51-vs-23.48 pair carries CONFLICT at L43; CONVENTION-AMBIGUOUS alone loses the conflict.

[FIX] evidence.md:L347 "none (production V4 comparison \"~430 @100K\" stated in :27-28; see CONFLICT note vs old-stack 359-366)"
SRC: evidence.md:L38 "| 8 | prefill | 359.7 tok/s | 100K |"
FIX: Replace leading "none" with CONFLICT (~430 @100K vs 359.7 tok/s); a prose conflict inside a none-flag is unfindable.

[FIX] evidence.md:L36 "| 6 | decode | 17.26 tok/s | 500K (nominal) | 2026-08-21 |"
SRC: evidence.md:L33 "500K (nominal; real ~352.6K)"
FIX: Depth cell should read 500K (nominal; real ~352.6K) with CONFLICT depth-label, as rows 3 and 13 already do.

[FIX] evidence.md:L366 "PROVENANCE-WEAK + CONFLICT — artifact numbers 110/75/56 vs corrected framefix numbers below"
SRC: evidence.md:L368 "| prefill rows/s (framefix, delta) | r500 delta completes at 103.6 rows/s"
FIX: 110 tok/s cannot be paired with 103.6 rows/s. Convert to one unit or mark CONVENTION-AMBIGUOUS, and flag L368 "none".

[NOTE] evidence.md:L597 "CONFLICT / CONVENTION: 3.2 GiB is reproduced exactly"
SRC: evidence.md:L945 "convention difference, not a wrong number; state the counting rule"
FIX: Tag CONVENTION-AMBIGUOUS, not CONFLICT: 3.2 vs 4.13 GiB is one quantity under two counting rules, not two measurements.

[NOTE] evidence.md:L950 "| 11 | on-disk checkpoint size | 210.68 GB measured (`du -sk`) | 210.58 GB in docs/deepseek-v41-exl3-plan.md:116 |"
SRC: evidence-live.md:L22 "storageSize=210,713,013,432 B"
FIX: Register omits a third value: live /state storageSize 210,713,013,432 B (210.71 GB logical) vs 210.68 GB du disk usage.

[NOTE] evidence.md:L267 "none (565K value CONFLICT-qualified below)"
SRC: evidence.md:L271 "CONFLICT/bimodal — 565K measured 13.36 and 24.15 within one night"
FIX: Bare "none" should carry a CONFLICT-pointer token so this row is machine-findable as half of the 565K pair.

[NOTE] evidence.md:L594 "| **total, one 1M session** | | | **4,434,427,904 B = 4.13 GiB** |"
SRC: evidence.md:L356 "PROVENANCE-WEAK + CONVENTION-AMBIGUOUS (documented as DERIVED anchor; no raw run preserved in repo)"
FIX: 4.13 GiB is arithmetic from config dims, no run behind it: tag PROVENANCE-WEAK/DERIVED, as L356 does.

MORE_BEYOND_CAP

CHECKED: 44

END P3B

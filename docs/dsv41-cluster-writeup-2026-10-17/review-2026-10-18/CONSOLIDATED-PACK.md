# Consolidated PACK findings — pre-verified, awaiting Fable ruling

Source: 2 DSv4.1-Flash subagents (deleg_68e66da2), 17 raw findings.
Every item re-verified by the parent against the pack source before listing. Nothing applied.

These are defects in the EVIDENCE PACKS (evidence.md, evidence-live.md), not the writeup.
The writeup inherits pack figures, so where a writeup line quotes a corrected pack value it
must follow. Those inheritance points are named per item.

## CONFIRMED — BLOCK (pack contradicts its own source)

Q01 [P3A-1] evidence-live.md:939 — "Attempt 1, which had only ~98 K cached, is the closer proxy: it reached 802,816 of 1,000,003 tokens"
SRC: evidence-live.md:701 "prefill reached **802,816 / 1,000,003 tokens** (80.3%)".
With ~98K cached the denominator is ~902K, so 802,816 is 89% and ~99K remain — not 80.3%/~197K.
FIX: drop "~98 K cached", or restate as 802,816/~902 K (89%) and ~99 K remaining.
INHERITS: WRITEUP.md §9.4 (the ~197K remainder and the 2.9 h extrapolation).

Q02 [P3A-2] evidence-live.md:704 — "matches a resident conversation on 299946 rows"
SRC: evidence-live.md:859 "`prompt_tokens_details.cached_tokens` | **299,008**".
FIX: 299946 -> 299,008.
INHERITS: WRITEUP.md §9.4 (the R12 fix cites evidence-live.md:704 with 299946).

Q03 [P3A-3] evidence.md:242 — "36.63 vs spec-OFF 27.15 (+36.71%)"
SRC: evidence.md:48 same pair; 36.63/27.15 = +34.92%.
FIX: show +34.9% and label +36.71% as the baseline doc's paired statistic.
INHERITS: WRITEUP.md §6.1 already distinguishes both — verify it still reads correctly.

## CONFIRMED — FIX (label / unit / register)

Q04 [P3B-1] evidence.md:638 — "`iogpu.wired_limit_mb = 115000` -> 120.6 GB"
SRC: evidence.md:634 "**120000** (120 GB)".
Two conversion rules in one section: 115000 MiB = 120.6 GB but 120000 MB = 120 GB decimal.
FIX: one rule; state it.

Q05 [P3B-2] evidence.md:552 — "210.58 GB" flagged "minor rounding CONFLICT, 0.1 GB"
SRC: evidence.md:551 "210.68 GB (196.22 GiB readable)".
Not rounding: 210.58 GB = 196.12 GiB; 196.2 GiB = 210.67 GB, matching the measured 210.68 GB.
FIX: relabel as a GB/GiB conversion error, not rounding.

Q06 [P3B-3] evidence.md:91 — "CONVENTION-AMBIGUOUS (nominal depth; B1 re-measured real 352.6K at 23.48, +9.2%)"
SRC: evidence.md:43 same 21.51-vs-23.48 pair carries "CONFLICT".
FIX: prepend CONFLICT so the pair is machine-findable from both ends.

Q07 [P3B-4] evidence.md:347 — flag cell "none (production V4 comparison ... see CONFLICT note vs old-stack 359-366)"
SRC: evidence.md:38 "359.7 tok/s | 100K".
FIX: replace the leading "none" with CONFLICT (~430 @100K vs 359.7).

Q08 [P3B-5] evidence.md:36 — depth cell "500K (nominal)"
SRC: evidence.md:33 "500K (nominal; real ~352.6K)".
FIX: match row 3's depth cell wording.

Q09 [P3B-6] evidence.md:366 — "110 tok/s at 500K" paired as CONFLICT with
SRC: evidence.md:368 "103.6 rows/s".
Different units cannot be a conflicting pair.
FIX: convert to one unit or mark CONVENTION-AMBIGUOUS; flag L368.

## CONFIRMED — NOTE

Q10 [P3A-4] evidence-live.md:133 — "6.54 s for 200 tokens = 30.4 tok/s"
SRC: evidence-live.md:168 d1k row 34.5 = 81/2.35 (n-1 convention).
FIX: state the n-1 convention once; 200/6.54 = 30.6.
INHERITS: WRITEUP.md §9.1 already states the convention — verify consistency.

Q11 [P3A-5] evidence.md:243 — "28.30 vs 27.49 (+1.87%)"
SRC: evidence.md:248 same section uses ratio-of-medians; 28.30/27.49 = +2.95%.
FIX: correct the percentage or name the paired statistic.

Q12 [P3A-6] evidence.md:292 — "28.66 (-12.6%)"
SRC: evidence.md:292 "33.58"; 28.66/33.58 = -14.7%.
FIX: correct the delta.

Q13 [P3A-7] evidence.md:185 — "PP hides 55.8 s drain outside the loop"
SRC: evidence.md:183 "PP 277.0 (TTFT 255.06 s) | 70,656 real"; 255.06 - 70,656/372 = ~65 s.
FIX: use ~65 s (or the 62.6 s TTFT gap).

Q14 [P3B-7] evidence.md:597 — "CONFLICT / CONVENTION: 3.2 GiB is reproduced exactly"
SRC: evidence.md:945 "convention difference, not a wrong number".
FIX: tag CONVENTION-AMBIGUOUS, not CONFLICT.

Q15 [P3B-8] evidence.md:950 — register row omits a third value
SRC: evidence-live.md:22 "storageSize=210,713,013,432 B".
FIX: add the live /state storageSize as a third value.

Q16 [P3B-9] evidence.md:267 — flag cell "none (565K value CONFLICT-qualified below)"
SRC: evidence.md:271 "CONFLICT/bimodal — 565K measured 13.36 and 24.15".
FIX: carry a CONFLICT token so the row is findable as half of the pair.

Q17 [P3B-10] evidence.md:594 — "4.13 GiB" untagged
SRC: evidence.md:356 same quantity tagged "PROVENANCE-WEAK + CONVENTION-AMBIGUOUS".
FIX: tag PROVENANCE-WEAK/DERIVED.

## REFUTED

None. All 17 survived re-verification.

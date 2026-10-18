[BLOCK] evidence.md:L115 "~166-169 vs baseline ~162-172 (statistically identical)"
SRC: docs/moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md:26 -> "clean secret-code recall, HTTP 200). Cumulative throughput: ~168-169"
FIX: "~168-169 vs baseline ~162-172 (statistically identical)"; "166" appears nowhere in the cited doc.

[FIX] evidence.md:L81 "331.6 (repeats 333.9/334.0/331.6 across 3 runs)"
SRC: docs/known-good-prefill-baseline-2026-08-21.md:28 -> "| 500,000 tokens | 331.6 tok/s | 17.26 tok/s | PASS |"
FIX: append :33, the line carrying the three-run 333.9/334.0/331.6 spread.

[FIX] evidence.md:L125 "362.8 / 348.5 / 331.1 vs same-night baseline 368.7 / 351.9 / 333.2 (parity)"
SRC: docs/moe-gate-up-fusion-validated-2026-08-21.md:25 -> "348.5/331.1 tok/s prefill vs baseline 368.7/351.9/333.2 — within"
FIX: cite :24-25; 362.8 sits on :24, outside the cited single line.

[FIX] evidence.md:L156 "SEQ_SPLIT=0: 191,330 tok/688.775s = 277.8"
SRC: docs/dsv4-220k-prefill-seqsplit-ab-2026-08-18.md:36 -> "**277.8 tok/s** wall-clock, matching the server's own live-computed"
FIX: cite :35-36; token count and 688.775s are on :35.

[FIX] evidence.md:L174 "PP single-request 24.63 vs 25.30 @ short ctx"
SRC: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:9344 -> "2,078 tok : 0.49 -> 24.09 tok/s"
FIX: add :8937-8938, which hold the 25.30/24.63 p50-gap pair; :9344-9346 has only the 2,078-tok row.

[FIX] evidence.md:L245 "later full result 27.57"
SRC: docs/dspark-mtp-ab-preregister-2026-08-25.md:844 -> "| tok/s @100K | ≥ +15% vs 28.46 → ≥ 32.73 | 28.46 | 27.57 (−3.1%) |"
FIX: add :814,844; cited :893-897 and :1126-1134 carry no 27.57.

[FIX] evidence.md:L257 "γ_mean=3.31 @100K: 99 ms (20.2 ms/row); T6 baseline 34.25 ms/token"
SRC: docs/dspark-mtp-master-history-2026-08-28.md:152 -> "| Verify cost @100K (γ_mean=3.31) | 99 ms (20.2 ms/row, +22% above cost model) |"
FIX: cite docs/dspark-mtp-master-history-2026-08-28.md:152 by path; p4-scoping and mtp-dspark docs contain no 3.31/20.2.

[FIX] evidence.md:L831 'HyperConnection training-gate 4.32x "async-fence-shaped bug"'
SRC: docs/hyperconnection-training-gate-false-lead-2026-08-22.md:87 -> "production. This is NOT a bug.**"
FIX: cite :35 (4.32x speedup) or :91 (retraction); :87 asserts the opposite of the finding label.

[FIX] evidence.md:L865 "first-send loss on 16380-byte UC chunks"
SRC: docs/hybrid-pp-prefill-tp-decode-design-2026-08-04.md:4444 -> "**NOT YET FIXED.** This is a genuine jaccl C++ transport-layer bug,"
FIX: add :10307,10751; "16380" occurs nowhere in the cited range.

[FIX] evidence.md:L366 'r500 "110 tok/s at 500K"; r750 "75 tok/s at 750K"; r1m ≈56 tok/s'
SRC: docs/PERFORMANCE_HISTORY.md:10173 -> "r750 = full 749,983-row prefill in 9,960 s → **75 tok/s at 750K**"
FIX: add :10173; cited :10300-10301,10303,10374-10390 hold r160 but not r500/r750/r1m.

[FIX] evidence.md:L326 "pools-correct 15.3"
SRC: docs/dsv4-rowseq-followups-plan-2026-07-10.md:7 -> "> old lossy baseline), c=2 pools-correct 15.3 t/s/stream. Cornerstones:"
FIX: add docs/dsv4-rowseq-followups-plan-2026-07-10.md:7; "15.3" absent from handoff doc :27,112,415.

[FIX] evidence.md:L400 "prefill end-of-loop 243.6-243.8 @89,407"
SRC: docs/fork-notes.md:678 -> "prefill: 243.6-243.8 tok/s (end-of-prefill, ~89,407 prompt tokens)"
FIX: add :678; cited :588 carries only the 15.4 → 29.47 headline.

CHECKED: 432

END P1

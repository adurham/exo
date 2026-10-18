# Consolidated PACK findings, round 2 — pre-verified, awaiting Fable ruling

Source: 3 DSv4.1-Flash subagents (deleg_3cf17139 tasks 1/2/4). 38 raw findings, deduped to 26
(p2 and p4 overlap heavily on the GONE-artifact set). Every item re-verified by the parent.
Nothing applied.

NOTE: agent line numbers were taken pre-fix and have shifted; current lines are given below.

## CONFIRMED — BLOCK

R01 [P1-1] evidence.md:130 — "~166-169 vs baseline ~162-172"
SRC: docs/moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md:26 "~168-169". "166" appears nowhere.
FIX: "~168-169 vs baseline ~162-172".

R02 [P2-1] evidence-live.md:150 — "Quoted verbatim from `~/.exo/exo_log/exo.log` on node0"
SRC: `ssh macstudio-m4-1 grep -c "160004 prompt tokens" exo.log` -> 0; same over all 5 *.zst -> 0; log starts 2026-10-18 13:56.
Every on-node source for this line is rotation-destroyed.
FIX: mark 661.18 s / 242.0 tok/s UNVERIFIED; cite a preserved copy.

R03 [P4-1] evidence-live.md:87 — calibration tokenizer MISMATCH
SRC: controller `~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json` sha 8f9f37ca / 6,367,146 B vs node served checkpoint c90dfa01 / 6,367,257 B.
All local counts (159,999 / 999,999 / 558,995 B) come from a non-identical vocabulary.
FIX: state the mismatch; label the local counts approximate.

## CONFIRMED — MISSING-RAW (artifact gone; re-point or mark unrecoverable)

R04 [P2-2/P4-2] evidence-live.md:51 — cited archive `exo.2026-10-16_18-38-04_871911.log.zst`
SRC: `ssh macstudio-m4-1 ls` -> No such file; earliest survivor 2026-10-17_20-35-30.
FIX: re-point to `exo.2026-10-17_20-40-39_097308.log.zst` (same digest=633711af, ACTION=error).

R05 [P2-3/P4-3] evidence-live.md:103 — `prompts/prompts_160k.json` (558,995 B)
SRC: `ls ~/.hermes/cache/scratch/writeup/prompts` -> No such file; whole writeup/ tree absent.
FIX: mark the byte/token counts unrecoverable.

R06 [P2-4/P4-6] evidence-live.md:296 — `prompts/guard_banana.json` (and guard_loop, limiter_96, limiter_default)
SRC: `find ~ -name guard_banana.json` -> 0 hits.
FIX: mark all four deliverable-3 bodies gone.

R07 [P2-5/P4-4] evidence-live.md:126 — `run_160k.sse.timed.txt`
SRC: `find ~ -maxdepth 7 -name '*.sse.timed.txt'` -> empty.
FIX: mark the verbatim response block unrecoverable.

R08 [P2-6/P4-5] evidence-live.md:244 — obs first-token line (and all depth/300K/1M obs lines)
SRC: `grep -c "1003 prompt tokens"` -> 0; "300003" -> 0; "1000003" -> 0.
FIX: label the depth/300K/1M tables UNVERIFIED.

R09 [P4-7] evidence-live.md:734 — "391 `: prefill_progress` frames"
SRC: no capture filename is ever cited for 4b/4c; `mdfind -name '*.sse'` -> 0.
FIX: name the artifact or mark the frame series uncheckable.

R10 [P4-8] evidence-live.md:156 — `prompts/depth_<tag>.json` (all five)
SRC: `mdfind -name depth_d100k.json` -> 0 on both hosts.
FIX: mark the depth table's client windows unrecomputable.

R11 [P4-9] evidence-live.md:687 — `prompts/prompts_1m.json` (3,391,475 B)
SRC: `mdfind -name prompts_1m.json` -> 0 on both hosts.
FIX: mark the 1M body gone; the headline claim rests on quoted server lines only.

## CONFIRMED — FIX (citation path/line corrections)

R12 [P1-2] evidence.md:96 — three-run spread cited to :28 only
SRC: docs/known-good-prefill-baseline-2026-08-21.md:33 carries 333.9/334.0/331.6.
FIX: add :33.

R13 [P1-3] evidence.md:140 — "362.8 / 348.5 / 331.1" cited to :25
SRC: doc:24 carries 362.8.
FIX: cite :24-25.

R14 [P1-4] evidence.md:171 — token count and 688.775 s cited to :36
SRC: doc:35 carries both.
FIX: cite :35-36.

R15 [P1-5] evidence.md:189 — "24.63 vs 25.30" cited to :9344-9346
SRC: doc:8937-8938 carries the pair.
FIX: add :8937-8938.

R16 [P1-6] evidence.md:261 — "later full result 27.57" cited to :893-897,1126-1134
SRC: doc:844 carries 27.57.
FIX: add :844.

R17 [P1-7] evidence.md:273 — γ_mean=3.31 / 99 ms cited to two docs that lack it
SRC: docs/dspark-mtp-master-history-2026-08-28.md:152 carries it.
FIX: cite that path.

R18 [P1-8] evidence.md:848 — 4.32x cited to doc:87
SRC: doc:87 says "production. This is NOT a bug."; 4.32x is at :35, retraction at :91.
FIX: cite :35 (and :91).

R19 [P1-9] evidence.md:882 — "16380-byte UC chunks" cited to :4444-4446
SRC: :4444 carries the NOT-YET-FIXED text, not 16380.
FIX: add :10307,10751.

R20 [P1-10] evidence.md:382 — r500/r750/r1m cited to :10300-10301,10303,10374-10390
SRC: docs/PERFORMANCE_HISTORY.md:10173 carries the r750 line.
FIX: add :10173.

R21 [P1-11] evidence.md:342 — "pools-correct 15.3" cited to the handoff doc
SRC: docs/dsv4-rowseq-followups-plan-2026-07-10.md:7 carries 15.3.
FIX: add that path.

R22 [P2-7] evidence-live.md:893/894 — "2.92 tok/s"
SRC: 100 / 33.87 = 2.952; :878 itself states 2.95.
FIX: 2.92 -> 2.95 in both cells.

R23 [P2-8] evidence-live.md:238 — "`deep-context-refault-mechanism` note"
SRC: no file by that name; it is `docs/deep-context-refault-mechanism-confirmed-2026-09-22.md`.
FIX: write the full path.

## CONFIRMED — NOTE

R24 [P2-9] evidence-live.md:172 — doc named without an anchor
SRC: the figures live at docs/moe-all-sum-dominant-cost-2026-08-19.md:18 (2.63x/62.0%), :19 (2.57x/61.1%).
FIX: anchor both.

R25 [P4-10] evidence-live.md:29,128,150 — DRIFT: cited log lines are build-specific and no longer byte-checkable; a bare turn-id grep on the current log returns 46 false hits (prompt text echoed by later clients quoting the writeup).
FIX: note the false-provenance trap.

R26 [P2-10] evidence-live.md:345 — the guard dump survives on m4-2
SRC: `ssh macstudio-m4-2` -> file present, task_id a5f8395c, generated 268, period 1, action error.
FIX: cite that dump as the on-disk source for this fire.

## REFUTED

R27 [P1-12] evidence.md:416 — agent said add :678, but the citation already reads `:588,677-678,816,894`. NO-CHANGE-NEEDED.

## GOOD NEWS (recorded, no change)

P4 EXISTS-OK x2: the Oct-13 warn dump and Oct-14 error dump both survive and decode to their
quoted loop tails verbatim, corroborating the pack's metadata.

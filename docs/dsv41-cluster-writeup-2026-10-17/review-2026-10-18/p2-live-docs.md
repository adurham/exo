[BLOCK] evidence-live.md:L124 "Quoted verbatim from `~/.exo/exo_log/exo.log` on node0"
SRC: ssh macstudio-m4-1 'grep -c "160004 prompt tokens" ~/.exo/exo_log/exo.log' -> 0; same over all 5 *.zst -> 0; log starts 2026-10-18 13:56
FIX: Mark 661.18 s / 242.0 tok/s UNVERIFIED and cite a preserved copy: rotation has destroyed every on-node source for this line.
[MISSING-RAW] evidence-live.md:L25 "`~/.exo/exo_log/exo.2026-10-16_18-38-04_871911.log.zst`"
SRC: ssh macstudio-m4-1 'ls ~/.exo/exo_log/' -> No such file; 5 zst remain, earliest exo.2026-10-17_20-35-30
FIX: Re-point to exo.2026-10-17_20-40-39_097308.log.zst, whose loop-guard line carries the same digest=633711af and ACTION=error.
[MISSING-RAW] evidence-live.md:L77 "Prompt body written to `prompts/prompts_160k.json` (558,995 bytes"
SRC: ls /Users/adam.durham/.hermes/cache/scratch/writeup/prompts -> No such file or directory; writeup/ tree absent
FIX: Recover the bodies or drop the byte/token counts; the entire prompts/ directory, incl. prompts_1m.json (L687), is gone.
[MISSING-RAW] evidence-live.md:L269 "Request body `prompts/guard_banana.json`"
SRC: find /Users/adam.durham -name 'guard_banana.json' -> 0 hits; writeup/prompts absent
FIX: All six deliverable-3 bodies are missing (guard_loop, limiter_default, limiter_96 too); cite the SSE captures or re-state as non-reproducible.
[MISSING-RAW] evidence-live.md:L100 "(`run_160k.sse.timed.txt`, epoch-prefixed)"
SRC: find /Users/adam.durham -maxdepth 7 -name '*.sse.timed.txt' -> empty
FIX: The 160K SSE capture exists nowhere on this machine; quote the response from evidence.md or mark the verbatim block unrecoverable.
[MISSING-RAW] evidence-live.md:L217 "obs first-token turn=fec83031-7137-46c0-b381-ba466bb19979 latency=4.39s"
SRC: ssh macstudio-m4-1 grep -c "1003 prompt tokens" exo.log -> 0, *.zst -> 0; "300003 prompt tokens" -> 0; "1000003 prompt tokens" -> 0
FIX: All depth, 300K and 1M obs lines (L217-L911) are rotation-lost; cite a preserved log copy or label the table UNVERIFIED.
[FIX] evidence-live.md:L318 "The runner's own log for the same turn (task_id `a5f8395c-cccc-41c4-91c4-6da384d0fa6b`)"
SRC: ssh macstudio-m4-2 dsv41-degeneration-guard-token-20261016-235228-35168.json -> task_id a5f8395c…, generated 268, returned 269, period 1, digest 633711af
FIX: Cite that surviving dump on m4-2 as the on-disk source for this fire; the quoted log lines themselves are gone.
[FIX] evidence-live.md:L866 "**server decode rate** | **2.92 tok/s** (turn-end 6760.2 s - first-token 6726.3 s)"
SRC: python3 100/33.87 -> 2.952; evidence-live.md:878 itself states 2.95 tok/s
FIX: Change 2.92 to 2.95 in both table cells (L865 and L866); the stated window and token count give 2.95, not 2.92.
[FIX] evidence-live.md:L211 "consistent with the repo's own `deep-context-refault-mechanism` note"
SRC: ls docs | grep refault -> docs/deep-context-refault-mechanism-confirmed-2026-09-22.md (no file named deep-context-refault-mechanism)
FIX: Write the full path: `docs/deep-context-refault-mechanism-confirmed-2026-09-22.md`; the shorthand names no existing file.
[NOTE] evidence-live.md:L145 "`docs/moe-all-sum-dominant-cost-2026-08-19.md` is the document that quantifies the `moe.all_sum` cost"
SRC: grep -nE 'docs/[A-Za-z0-9_.-]+\.md:[0-9]' evidence-live.md -> 0 matches; the figures live at moe-all-sum-dominant-cost-2026-08-19.md:18
FIX: Anchor doc claims: `docs/moe-all-sum-dominant-cost-2026-08-19.md:18` for the 2.63x / 62.0% row (2.57x / 61.1% at :19).
CHECKED: 12
END P2

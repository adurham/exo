[GONE] evidence-live.md:L25 "in `~/.exo/exo_log/exo.2026-10-16_18-38-04_871911.log.zst`, retrieved from the"
SRC: ssh macstudio-m4-1 'ls ~/.exo/exo_log/exo.2026-10-16_18-38-04_871911.log.zst' -> ls: No such file or directory
NOTE: Earliest surviving archive is exo.2026-10-17_20-35-30_611784.log.zst (first line 2026-10-17 20:35:30), ~16 h after the last capture ended.

[DRIFT] evidence-live.md:L29 "[DSV41] loop-guard env (rank 1/2, digest=633711af): EXO_LOOP_DETECT=1"
SRC: zstdcat exo.2026-10-17_20-35-30_611784.log.zst | grep 'loop-guard env' -> '...engine:__post_init__:737 ] ... (rank 0/2, digest=633711af)'
NOTE: Same knob string and digest survive in a later boot, but source tag 535->737 and rank 1/2->0/2 shift: the cited line is build-specific and no longer byte-checkable.

[GONE] evidence-live.md:L100 "The full response, verbatim, exactly as the capture holds it (`run_160k.sse.timed.txt`, epoch-prefixed):"
SRC: mdfind -name run_160k.sse.timed.txt -> 0; ls ~/.hermes/cache/scratch/writeup -> No such file or directory
NOTE: prompts_160k.json is gone too (mdfind 0/0), so the 558,995-byte body, its 159,999 token count and the epoch-stamped response all lack raw records.

[DRIFT] evidence-live.md:L128 "[DSV41] obs first-token turn=3b9d617a-7499-4671-952b-58bf73676c06 latency=661.36s"
SRC: ssh grep -c 'obs first-token turn=3b9d617a' exo.log -> 0; wc -l -> 158855; head -1 -> 2026-10-18 13:56:18.789
NOTE: Current exo.log holds only 10-18 13:56-14:28 (152 obs turn-end lines, none of the pack's turns); no absolute exo.log line numbers are cited, only build tags, so drift is total rather than an offset.

[DRIFT] evidence-live.md:L124 "Quoted verbatim from `~/.exo/exo_log/exo.log` on node0"
SRC: ssh grep -c 'turn=3b9d617a' exo.log -> 46; grep -c '########## S1-1 L236 vs L682' -> 92; od -c of first hit -> '<|tool_calls|><|end_of_sentence|>'
NOTE: Those 46 hits are prompt text echoed inside logged chat bodies by later clients quoting the writeup, not [DSV41] obs records, so a bare turn-id grep on the current log yields false provenance.

[GONE] evidence-live.md:L156 "# one row (repeated per depth); bodies live in prompts/depth_<tag>.json"
SRC: mdfind -name depth_d100k.json -> 0 (controller), 0 (macstudio-m4-1)
NOTE: All five depth bodies are gone, so the table's client decode windows and 1003/10004/50004/100003/300003 counts cannot be recomputed from raw bytes.

[GONE] evidence-live.md:L269 "Request body `prompts/guard_banana.json`"
SRC: mdfind -name guard_banana.json -> 0 / 0; guard_loop.json, limiter_96.json, limiter_default.json -> 0,0,0
NOTE: All four deliverable-3 bodies are gone: the loop deltas, the repetition_loop error event, the 3b negative control and the 96-vs-600 limiter pair survive only as prose.

[GONE] evidence-live.md:L687 "Body `prompts/prompts_1m.json`, 3,391,475 bytes, 999,999 tokens by the checkpoint-family tokenizer"
SRC: mdfind -name prompts_1m.json -> 0 (controller), 0 (macstudio-m4-1)
NOTE: The 3,391,475-byte 1M body is gone, so the headline 1M claim now rests entirely on quoted server lines; neither its content nor its token length is re-checkable.

[MISMATCH] evidence-live.md:L61 "(`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash/tokenizer.json`) and are labelled"
SRC: shasum -a 256 -> controller 8f9f37ca/6367146 B vs node dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw c90dfa01/6367257 B
NOTE: The cited calibration tokenizer exists but is not the served checkpoint's tokenizer, so all local counts (159,999 / 999,999 / 558,995 B) come from a non-identical vocabulary file.

[EXISTS-OK] evidence-live.md:L454 "`dsv41-degeneration-guard-token-20261014-135110-69384.json` — detector `token`, period 12, repeats >=6, action `error`, 1613 tokens generated"
SRC: python3 json.load -> period 12, repeats 6, action error, generated 1613, digest 633711af; decode(token_ids[-44:]) -> '...Let me write.\n\nGo.\n\nNow.\n\nWriting'
NOTE: Real bytes match the pack's metadata and its decoded loop tail verbatim, so this dump genuinely corroborates the Oct-14 action=error knob set.

[EXISTS-OK] evidence-live.md:L444 "`dsv41-degeneration-guard-token-20261013-224154-5634.json` — detector `token`, period 1, repeats >=6, action `warn`, 143 tokens generated"
SRC: python3 json.load -> period 1, repeats 6, action warn, generated 143, digest fb0de5b4; decode(tail) -> '...banana banana banana banana banana banana'
NOTE: The older warn-mode sample survives and decodes to the quoted banana-count tail; its digest fb0de5b4 differs from the live 633711af exactly as the pack states.

[GONE] evidence-live.md:L734 "The real evidence for a prefill that never reaches generation is the **391 `: prefill_progress` frames**."
SRC: grep -nE '\.sse' evidence-live.md -> 1 hit (L100 only); mdfind -name '*.sse' -> 0 on both hosts; archives 0 hits for 519cd399/f8104803
NOTE: No capture filename is ever cited for 4b/4c, so the 391- and 342-frame series behind every attempt-1/attempt-2 band rate has no artifact to check at all.

MORE_BEYOND_CAP

END P4

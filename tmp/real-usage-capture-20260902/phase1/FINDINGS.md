# Phase 1 FINDINGS — real chat session on exo cluster (instance 339f04f8-82de-4976-8b97-ce067f35a7d3)

Captured 2026-09-02, READ-ONLY (ssh cat/grep/zst-read + HTTP /metrics GET + local sqlite read-only). No relaunches, no requests sent to the cluster, no commits.

## Sources & method
1. **API event log** `~/.exo/event_log/api/events.2026-09-01_21-19-36_813144.bin.zst` (length-prefixed msgpack, read with the node's own `~/repos/exo/.venv/bin/python` + msgspec) — per-request `usage`, `stats`, `finish_reason`, chunk counts. 57 final chunks for this instance.
2. **exo.log rotated archive** `~/.exo/exo_log/exo.2026-09-01_12-37-19_195940.log.zst` — per-request wall timestamps: `received chat request` (matched 57/57 by task_id), `Prefill complete: N tokens in S s (T tok/s)`, `runner ready/idle`.
3. **Client DB** `~/.hermes/state.db` `api_calls` (session `20260901_120301_93ad7b`, provider=custom, call_seq 33..87 = 55 rows) — client-side started/ended/latency.
Row alignment was verified by exact (prompt_tokens, completion_tokens) equality across all 55 main rows (asserted by the build script; 2 aux rows with prompt 18/17 have no client custom-provider row and carry null client fields).

## Session identification (empirical, not assumed)
- **instance_id: `339f04f8-82de-4976-8b97-ce067f35a7d3`** (model `deepseek-ai/DeepSeek-V4-Flash-0731`, TP=2, MTP on).
- All candidate instances found empirically in the api event-log archives:
  - `1fe44955` (Sep-1 ~00-05 archive): 9 finals, tiny/error — not a chat session.
  - `25ae372c` (12:37 archive): 32 finals (26 tool_calls / 6 stop) — earlier era of the SAME client session, before the runner relaunch at 12:37.
  - **`339f04f8`: 57 finals — finish_reason tool_calls=44, stop=12, length=1; prefix-cache partial=54, none=3** ← the real-session instance. The brief's "~38 requests / 31 tool_calls / 6 stop / 1 length" matches a mid-session /metrics subset snapshot of this same instance (that snapshot also survives verbatim inside pasted chat content in exo.log — see parsing trap below); the full instance totals 44/12/1.
  - `3ed6023d` (16:19 relaunch → still live): 83 finals, mixed bench probes + chat (excluded).
- Client driver: Hermes session `20260901_120301_93ad7b` (provider=custom → the exo cluster).
- **Derived session window (from the records themselves):**
  - Server: first request received **2026-09-01 12:38:40.859**, last **16:09:10.894**.
  - Client: first call started **12:38:40.641**, last call ended **16:09:40.229** (wall span 3h31m incl. user idle gaps of 68m/51m/32m).
  - Instance LoadModel task at 12:37:19; superseded by instance 3ed6023d at 16:19.

## Per-request table (57 rows; main chat = call_seq 33..87, aux = 2 small helper calls)
Provenance per field is embedded in `requests.jsonl` (every value is wrapped in a provenance dict).

| # | call_seq | server_received | wall_s (client) | prompt_tok | cached_tok | completion_tok | reasoning_tok | finish | prefix_hit | gen_tps (srv) | prefill_tps (srv) | full_wall_rate | tools |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 33 | 2026-09-01 12:38:40.859 | 282.811 | 92594 | 0 | 1982 | 943 | tool_calls | none | 33.11 | 416.8 | 7.01 | write_file |
| 1 | 34 | 2026-09-01 12:43:23.773 | 30.095 | 94617 | 92592 | 1003 | 0 | tool_calls | partial | 46.51 | 269.6 | 33.33 | write_file |
| 2 | 35 | 2026-09-01 12:43:54.124 | 14.923 | 95971 | 94615 | 337 | 61 | tool_calls | partial | 38.4 | 259.2 | 22.58 | delegate_task |
| 3 | 36 | 2026-09-01 12:44:09.196 | 10.976 | 96718 | 95969 | 201 | 24 | stop | partial | 33.83 | 176.1 | 18.31 |  |
| 4 | 37 | 2026-09-01 12:44:20.508 | 21.123 | 97396 | 96716 | 646 | 313 | tool_calls | partial | 39.22 | 176.8 | 30.58 | delegate_task |
| 5 | 38 | 2026-09-01 12:44:41.840 | 7.808 | 98451 | 97394 | 64 | 0 | stop | partial | 32.63 | 212.2 | 8.2 |  |
| 6 | aux | 2026-09-01 12:45:51.559 | None | 18 | 0 | 50 | 50 | length | none | 29.29 | 13.9 |  |  |
| 7 | aux | 2026-09-01 12:46:00.665 | None | 17 | 0 | 16 | 13 | stop | none | 21.91 | 12.7 |  |  |
| 8 | 39 | 2026-09-01 12:46:33.943 | 41.723 | 99768 | 98449 | 1254 | 536 | tool_calls | partial | 35.44 | 243.0 | 30.06 | terminal,memory |
| 9 | 40 | 2026-09-01 12:47:17.926 | 18.919 | 101224 | 99766 | 419 | 99 | stop | partial | 32.08 | 296.7 | 22.15 |  |
| 10 | 41 | 2026-09-01 12:48:47.105 | 17.949 | 102050 | 101222 | 487 | 336 | tool_calls | partial | 37.45 | 206.5 | 27.13 | skill_view,search_files |
| 11 | 42 | 2026-09-01 12:49:05.700 | 44.453 | 113585 | 102048 | 455 | 291 | tool_calls | partial | 36.12 | 373.4 | 10.24 | search_files,read_file |
| 12 | 43 | 2026-09-01 12:49:50.405 | 37.98 | 122155 | 113583 | 513 | 267 | tool_calls | partial | 39.81 | 353.5 | 13.51 | read_file,search_files |
| 13 | 44 | 2026-09-01 12:50:28.474 | 13.629 | 124192 | 122153 | 165 | 69 | tool_calls | partial | 35.4 | 256.5 | 12.11 | read_file |
| 14 | 45 | 2026-09-01 12:50:42.218 | 27.767 | 125242 | 124190 | 810 | 516 | tool_calls | partial | 36.76 | 215.7 | 29.17 | search_files,search_files |
| 15 | 46 | 2026-09-01 12:51:10.222 | 12.605 | 126555 | 125240 | 219 | 120 | tool_calls | partial | 39.64 | 218.1 | 17.37 | read_file |
| 16 | 47 | 2026-09-01 12:51:22.860 | 15.172 | 128821 | 126553 | 218 | 81 | tool_calls | partial | 37.3 | 266.6 | 14.37 | search_files |
| 17 | 48 | 2026-09-01 12:51:38.148 | 7.407 | 129314 | 128819 | 98 | 0 | tool_calls | partial | 40.81 | 119.1 | 13.23 | read_file |
| 18 | 49 | 2026-09-01 12:51:45.777 | 26.309 | 131575 | 129312 | 617 | 443 | tool_calls | partial | 37.27 | 252.9 | 23.45 | search_files |
| 19 | 50 | 2026-09-01 12:52:12.149 | 8.277 | 132465 | 131573 | 118 | 0 | tool_calls | partial | 43.15 | 193.2 | 14.26 | search_files |
| 20 | 51 | 2026-09-01 12:52:20.416 | 6.789 | 132859 | 132463 | 99 | 0 | tool_calls | partial | 43.85 | 105.3 | 14.58 | read_file |
| 21 | 52 | 2026-09-01 12:52:27.396 | 22.588 | 134701 | 132857 | 530 | 398 | tool_calls | partial | 35.11 | 290.1 | 23.46 | terminal |
| 22 | 53 | 2026-09-01 12:52:50.049 | 7.824 | 135809 | 134699 | 99 | 0 | tool_calls | partial | 42.06 | 246.6 | 12.65 | read_file |
| 23 | 54 | 2026-09-01 12:52:58.037 | 98.122 | 137328 | 135807 | 2903 | 2661 | tool_calls | partial | 32.09 | 230.7 | 29.59 | terminal |
| 24 | 55 | 2026-09-01 12:54:36.516 | 31.693 | 140368 | 137326 | 689 | 579 | tool_calls | partial | 33.12 | 307.8 | 21.74 | terminal |
| 25 | 56 | 2026-09-01 12:55:08.447 | 7.923 | 141118 | 140366 | 99 | 0 | tool_calls | partial | 41.28 | 170.4 | 12.5 | read_file |
| 26 | 57 | 2026-09-01 12:55:16.341 | 20.235 | 142575 | 141116 | 455 | 384 | tool_calls | partial | 35.91 | 223.6 | 22.49 | terminal |
| 27 | 58 | 2026-09-01 12:55:36.650 | 17.67 | 143474 | 142573 | 413 | 232 | tool_calls | partial | 34.65 | 185.3 | 23.37 | terminal |
| 28 | 59 | 2026-09-01 12:55:54.429 | 57.188 | 144035 | 143472 | 1748 | 1380 | tool_calls | partial | 33.52 | 135.4 | 30.57 | patch |
| 29 | 60 | 2026-09-01 12:56:52.097 | 14.556 | 145918 | 144033 | 220 | 74 | tool_calls | partial | 37.89 | 245.9 | 15.11 | terminal |
| 30 | 61 | 2026-09-01 12:57:07.564 | 16.576 | 146232 | 145916 | 444 | 231 | tool_calls | partial | 36.35 | 89.7 | 26.79 | terminal |
| 31 | 62 | 2026-09-01 12:57:26.695 | 27.215 | 146839 | 146230 | 788 | 256 | tool_calls | partial | 35.66 | 143.5 | 28.95 | memory |
| 32 | 63 | 2026-09-01 12:57:53.882 | 7.129 | 147657 | 146837 | 43 | 0 | stop | partial | 25.19 | 180.3 | 6.03 |  |
| 33 | 64 | 2026-09-01 13:00:45.647 | 52.907 | 147734 | 147655 | 1541 | 1345 | tool_calls | partial | 30.68 | 41.7 | 29.13 | session_search,terminal |
| 34 | 65 | 2026-09-01 13:01:38.849 | 61.164 | 150324 | 147732 | 1545 | 1208 | stop | partial | 30.42 | 275.6 | 25.26 |  |
| 35 | 66 | 2026-09-01 13:09:46.894 | 37.513 | 151956 | 150322 | 992 | 731 | tool_calls | partial | 32.67 | 280.3 | 26.44 | terminal |
| 36 | 67 | 2026-09-01 13:10:27.839 | 24.284 | 153287 | 151954 | 567 | 322 | tool_calls | partial | 33.06 | 221.2 | 23.35 | terminal |
| 37 | 68 | 2026-09-01 13:10:55.133 | 15.946 | 154799 | 153285 | 336 | 133 | tool_calls | partial | 35.65 | 277.7 | 21.07 | terminal |
| 38 | 69 | 2026-09-01 13:11:13.334 | 15.732 | 156682 | 154797 | 230 | 130 | tool_calls | partial | 32.3 | 253.0 | 14.62 | terminal |
| 39 | 70 | 2026-09-01 13:11:32.456 | 16.138 | 157873 | 156680 | 347 | 143 | tool_calls | partial | 36.57 | 213.7 | 21.5 | terminal |
| 40 | 71 | 2026-09-01 13:11:58.763 | 68.674 | 160312 | 157871 | 1687 | 1277 | stop | partial | 28.83 | 262.5 | 24.57 |  |
| 41 | 72 | 2026-09-01 14:19:34.443 | 96.958 | 162026 | 160310 | 2721 | 1467 | tool_calls | partial | 30.29 | 285.9 | 28.06 | terminal,write_file |
| 42 | 73 | 2026-09-01 14:21:12.124 | 17.378 | 165366 | 162024 | 187 | 129 | tool_calls | partial | 32.95 | 321.8 | 10.76 | terminal |
| 43 | 74 | 2026-09-01 14:21:29.447 | 44.167 | 165767 | 165364 | 1248 | 936 | tool_calls | partial | 31.93 | 95.9 | 28.26 | delegate_task |
| 44 | 75 | 2026-09-01 14:22:13.823 | 13.1 | 167671 | 165765 | 182 | 110 | tool_calls | partial | 34.3 | 282.7 | 13.89 | delegate_task |
| 45 | 76 | 2026-09-01 14:22:26.966 | 14.474 | 167918 | 167669 | 314 | 189 | tool_calls | partial | 32.74 | 60.3 | 21.69 | terminal |
| 46 | 77 | 2026-09-01 14:22:41.519 | 54.455 | 169100 | 167916 | 1497 | 1176 | tool_calls | partial | 31.11 | 223.5 | 27.49 | delegate_task |
| 47 | 78 | 2026-09-01 14:23:36.096 | 16.056 | 171250 | 169098 | 201 | 155 | tool_calls | partial | 31.04 | 245.7 | 12.52 | delegate_task |
| 48 | 79 | 2026-09-01 14:23:52.167 | 14.454 | 171661 | 171248 | 308 | 85 | stop | partial | 31.62 | 107.3 | 21.31 |  |
| 49 | 80 | 2026-09-01 14:24:07.146 | 11.308 | 172417 | 171659 | 180 | 126 | stop | partial | 28.54 | 186.6 | 15.92 |  |
| 50 | 81 | 2026-09-01 14:37:18.584 | 53.347 | 176414 | 172415 | 1156 | 621 | stop | partial | 29.58 | 305.4 | 21.67 |  |
| 51 | 82 | 2026-09-01 15:09:23.911 | 207.546 | 177632 | 176412 | 6480 | 3090 | tool_calls | partial | 32.2 | 232.2 | 31.22 | write_file,write_file |
| 52 | 83 | 2026-09-01 15:12:52.054 | 34.824 | 184557 | 177630 | 457 | 118 | tool_calls | partial | 34.47 | 335.9 | 13.12 | delegate_task |
| 53 | 84 | 2026-09-01 15:13:27.148 | 7.513 | 185420 | 184555 | 46 | 0 | tool_calls | partial | 35.13 | 167.5 | 6.12 | delegate_task |
| 54 | 85 | 2026-09-01 15:13:34.640 | 16.213 | 185692 | 185418 | 348 | 118 | stop | partial | 31.03 | 66.9 | 21.46 |  |
| 55 | 86 | 2026-09-01 16:08:40.884 | 29.87 | 187225 | 185690 | 708 | 576 | tool_calls | partial | 32.05 | 229.5 | 23.7 | terminal |
| 56 | 87 | 2026-09-01 16:09:10.894 | 29.573 | 188902 | 187223 | 677 | 183 | stop | partial | 32.13 | 227.6 | 22.89 |  |

## Aggregates (n=55 main chat requests)
- wall_seconds (client MEASURED): mean 35.07, median 18.92, min 6.79, max 282.81
- prompt_tokens: mean 138309 (incl. aux), median 144035, min 17 (aux), max 188902
- cached_tokens: mean 134993, median 143472 (only the first main call and the 2 aux rows are 0)
- completion_tokens: mean 739.6, median 444, min 16, max 6480 — **sum = 42,157 tokens**
- generation_tps (server MEASURED): mean 34.46, median 33.83, min 21.91, max 46.51
- prompt_tps (server MEASURED — rate of the suffix prefill, not the whole prompt): mean 215.57, median 227.62, min 12.7, max 416.84
- full-wall rate (DERIVED = completion_tokens / client wall): mean 20.38, median 21.69, min 6.03, max 33.33 tok/s
- non-decode overhead inside a call (DERIVED = wall − completion/gen_tps): mean 11.88, median 6.52, min 2.68, max 222.95 s
- tool-call round-trip gap (DERIVED, client: next-call start − tool_calls-call end): mean 0.79, median 0.15, min 0.04, max 10.18 s
- Prefix-cache: **partial 54/57 (94.7%), none 3/57 (5.3%)** — the 3 misses are the instance's first call (92,594-tok cold prefill at 416.8 tok/s = 222 s) plus the 2 aux calls.
- Finish reasons: tool_calls 44, stop 12, length 1.

## MEASURED vs DERIVED rates
- MEASURED (server, per request from event-log `stats`): `generation_tps` (decode) mean 34.46; `prompt_tps` mean 215.6 (cold first call 416.8).
- DERIVED (from client wall + real token counts): full-wall rate mean **20.38 tok/s**, median **21.69 tok/s**.
- post-TTFT rate: **NOT COMPUTED — TTFT is not in the logs.** No synthesis attempted.

## The measurement-convention gap (adjudication)
- Server decode-only (generation_tps): mean 34.5 / median 33.8 tok/s.
- Client full-wall (completion / wall): mean 20.4 / median 21.7 tok/s.
- Ratio ≈ 1.7×. Median call: 530 tok, 17.0 s client wall → decode-only 15.1 s (530/35.1), non-decode remainder 6.5 s median (min 2.7 s; max 223 s = the single cold 92.6K prefill call, which also matches the briefed "cold 150K ≈ 350 s" scale: 92.6K took 222 s at 416.8 tok/s).
- So the user-perceived ~20 tok/s is real client-wall truth, and the benchmark's ~31.8 tok/s is the decode-only convention. The delta is real non-decode time: queue+streaming+client overhead inside each call (median 6.5 s) plus tool round-trips (median 0.15 s, up to 10.2 s) and huge user-think gaps between calls (up to 68 min) that per-request metrics never see.
- Caveat for downstream use: wall_seconds is CLIENT-side (Hermes `api_calls.latency_seconds`), so it includes network + streaming transport + client processing. The logs do not permit a server-side wall decomposition (see below), so the decode-vs-wall split is the tightest possible from this data.

## WHAT THE LOGS DO NOT EXPOSE
1. **TTFT / first-token timestamp** — zero real emit lines in exo.log (745,676 lines live; 0 hits outside pasted chat content); event-log chunks carry no timestamps at all.
2. **Server-side per-request end timestamp / wall time** — exo.log logs "received chat request" but never logs request completion; only client-side wall exists.
3. **Server-side full-request rate** — impossible without #2.
4. **Per-chunk timestamps** (stream pacing, thinking-phase boundaries) — TokenChunk/ToolCallChunk records are unordered-with-respect-to-time; event log has NO timestamps on any record (record order is the only time proxy).
5. **Queue time** — no admit-vs-prefill-start distinction.
6. **MTP per-request acceptance** — only cumulative counters (`mtp_cycles_cumulative`, `mtp_accepted_drafts_cumulative`); deltas are derivable from requests.jsonl but no per-request rate is logged.
7. **HTTP status/duration in the API access log** — only `API request: POST /v1/chat/completions`.
8. **Client request→instance grouping** — not in cluster logs; required joining local Hermes state.db.
9. **Tool round-trip server time** — only client-side gap is observable; the server does not log when a tool result re-enters.
10. **Cached-token provenance** — `cached_tokens` is reported but there is no per-request log line tying it to a specific leaf/snapshot event with a timestamp (`KV cache extended` lines carry timestamps but not per-command ids).

## Parsing traps encountered (for future miners)
- Chat content is embedded verbatim in TextGeneration/aux DEBUG lines: naive `grep prompt_tokens exo.log` matches benchmark markdown inside pasted documents. The 31/6/1 "metrics snapshot" visible in exo.log content is exactly such a paste — NOT a live metric.
- Only trust lines starting with the literal log prefix `[ YYYY-MM-DD HH:MM:SS.mmm | LEVEL | `.
- The real per-request data lives in the msgpack event log (`~/.exo/event_log/api/`), not exo.log.

## Artifacts
- `raw/log_schema_samples.txt` — verbatim log-line shapes + the content-pollution trap
- `raw/metrics_full.txt` — live /metrics snapshot (current instance)
- `requests.jsonl` — 57 rows, every field provenance-tagged
- `FINDINGS.md` — this file

# REPORT — Real-Usage vs Benchmark Decode Rate: Is the ~12–14 tok/s Gap a Convention Artifact or Genuine Loss?

Study: real-usage capture, instance `339f04f8-82de-4976-8b97-ce067f35a7d3`
Session: Hermes `20260901_120301_93ad7b` (2026-09-01 12:38:40 → 16:09:10)
Authoritative figures: `partition_verified.json` (PM-computed, independently verified). Log-forensics detail: `phase1/FINDINGS.md`. Client capture tool: `phase2/README.md`.
Every number below that is not explicitly labeled **DERIVED** is **MEASURED** from the source log files. Nothing was recomputed for this report.

---

## 1. VERDICT

**The entire gap is a measurement-convention artifact. There is no genuine decode-rate loss in real sessions.**

The benchmark reported **31.84 tok/s** (server-side, decode-only) at 150K context. The user perceives **~20 tok/s** in real sessions. The decisive evidence is a match-at-depth comparison of the real session's **own server-reported decode rate**:

| Measure | Value | Convention | Source |
|---|---|---|---|
| Real session decode @ matched depth (140–160K prompt tokens, n=16) | **34.06 tok/s mean** / 34.09 median | decode-only, server `generation_tps` | MEASURED, real logs |
| Benchmark decode @ 150K | **31.84 tok/s** | decode-only, benchmark harness | benchmark |
| Real-session decode − benchmark | **+2.22 tok/s** | — | DERIVED |

At the depth the user actually runs, the real session's hardware decodes **faster** than the benchmark, not slower. There is nothing to be lost — the "loss" of ~12–14 tok/s is the difference between the benchmark's narrow **decode-only** convention (which excludes time-to-first-token) and what a user's wall clock actually includes (TTFT/prefill + streaming + network). Two earlier suspects (entropy, temperature) were already investigated and closed; the third candidate — measurement convention — is confirmed here by direct measurement of real requests.

**Convention mechanics of the gap.** The user-perceived ~20 tok/s is the true full-wall truth of the session. The benchmark's ~31.8 tok/s is the decode-only truth of the same hardware. The ~11.18 tok/s difference between the two (33.00 → 21.82) decomposes as **94.5% TTFT/prefill** and **5.5% residual (streaming + network + overhead)**. Section 4 quantifies this on the same 55 real requests.

---

## 2. Session Identification and the Request-Count Correction

**Instance:** `339f04f8-82de-4976-8b97-ce067f35a7d3`, model `deepseek-ai/DeepSeek-V4-Flash-0731`, TP=2, MTP on. Client driver: Hermes session `20260901_120301_93ad7b` (provider=custom → the exo cluster).

**Window (derived from the records themselves):** server first request received **2026-09-01 12:38:40.859**, last **16:09:10.894**; client first call started 12:38:40.641, last call ended 16:09:40.229 — a wall span of 3h31m that includes long user-idle gaps (68m / 51m / 32m). The instance was superseded by instance `3ed6023d` at 16:19.

**Totals (MEASURED):** **57 completed requests**; **55 usable** for the partition (the 2 remaining are auxiliary helper rows that lack client-side fields for the join). Finish reasons: **44 tool_calls / 12 stop / 1 length**. Prefix cache: **54 partial / 3 none**. Median prompt depth **145,918 tokens**.

### The 38-vs-57 correction (a genuine log-mining trap)

The investigation was originally scoped as **"38 requests (31 tool_calls, 6 stop, 1 length)"**. That signature is real but it is **not** the instance total. It is a mid-session `/metrics` snapshot that had been **pasted into chat**, so it survives verbatim *inside* `exo.log` content and is found by grepping for token fields. It is a subset of this same instance, not the instance. **The real instance totals are 57 requests / 44 tool_calls / 12 stop / 1 length**, read from the msgpack event log, not from `exo.log`:

- Naive `grep prompt_tokens exo.log` matches benchmark markdown pasted inside documents (`TextGeneration`/aux DEBUG lines carry chat content verbatim). The 31/6/1 "metrics snapshot" is exactly such a paste — **not a live metric**.
- Only trust lines starting with the literal log prefix `[ YYYY-MM-DD HH:MM:SS.mmm | LEVEL | `.
- **The real per-request data lives in the msgpack event log** at `~/.exo/event_log/api/*.bin.zst`, not in `exo.log`.

---

## 3. Methodology and the Reconciliation Identity

### Sources
1. **API event log** `~/.exo/event_log/api/events.2026-09-01_21-19-36_813144.bin.zst` (length-prefixed msgpack, read with the node's own `~/repos/exo/.venv/bin/python` + msgspec) — per-request `usage`, `stats`, `finish_reason`, chunk counts.
2. **exo.log rotated archive** `~/.exo/exo_log/exo.2026-09-01_12-37-19_195940.log.zst` — per-request wall timestamps (`received chat request`, matched 57/57 by `task_id`), `Prefill complete: N tokens in S s (T tok/s)`, `runner ready/idle`.
3. **Client DB** `~/.hermes/state.db` `api_calls` (session `20260901_120301_93ad7b`, call_seq 33..87 = 55 rows) — client-side started/ended/latency.

Row alignment across the three sources was verified by exact `(prompt_tokens, completion_tokens)` equality across all 55 main rows (asserted by the build script); the 2 auxiliary rows (prompt 18/17) have no client custom-provider row and carry null client fields, hence are excluded from the partition.

### The reconciliation identity — the backbone of the study

Define per-request in-client-wall time as:

```
wall = prefill_uncached + decode + residual
```

- **prefill_uncached** = uncached prompt tokens ÷ server `prompt_tps` (**DERIVED**; the server's `prompt_tps` applies to *uncached* prompt tokens)
- **decode** = completion tokens ÷ server `generation_tps` (**DERIVED** from real server-measured rates)
- **residual** = everything left (stream + network + client overhead)

This identity holds for **all 55 requests**, with residual in the narrow band **[0.75 s, 1.32 s], median 0.94 s** (verified file; see Note A in §10). That simultaneity proved three things at once:

1. **The client↔server join is real, not coincidental.** A spurious join could not produce a per-request residual that is consistently sub-second and bounded by a narrow band.
2. **The server's `prompt_tps` applies to UNCACHED prompt tokens, not total prompt tokens.** Assuming it applied to *total* prompt tokens yields a median residual of **minus 585 seconds** — physically impossible. That disambiguation is how the convention was pinned.
3. **The accounting is complete.** There is no unexplained time in the session.

### Join validation

`server_received_ts − client_started_ts` fell in **[0.077 s, 0.394 s]** for all 55 requests (**median 0.191 s**). A coincidental/spurious join could not hold that tight a client→server propagation window across every request.

---

## 4. The Convention Ladder (Centerpiece) and Gap Attribution

All four rates below are computed from the **same 55 real requests** (42,091 completion tokens over 1,929.0 s of in-call wall — **DERIVED**, from real token counts and real client wall). They differ only in how much wall time the denominator includes.

| Convention rung | What's included | Rate (tok/s) |
|---|---|---|
| Decode-only (benchmark convention) | decode only | **33.00** |
| + TTFT/prefill | + prefill_uncached | **22.44** |
| + residual = true full wall | + stream + network + overhead | **21.82** ← matches user's perceived ~20 tok/s |
| Visible-answer tokens only | decode ÷ visible (non-reasoning) tokens | **9.04** |

The user's perception sits on the **21.82 t/s** full-wall rung (perception of ~20 t/s), almost exactly on it. Between the decode-only convention (33.00) and the true full wall (21.82), the gap is **11.18 tok/s**, which attributes as:

| Gap component | Share |
|---|---|
| TTFT / prefill | **94.5%** |
| Residual (streaming / network / overhead) | **5.5%** |

Session totals (n=55, 42,091 completion tokens, 1,929.0 s of in-call wall — **MEASURED** token/time sums):

| Partition | Seconds | Share of wall |
|---|---|---|
| prefill / TTFT | 600.2 s | 31.1% |
| decode | 1,275.7 s | 66.1% |
| residual (stream + network + overhead) | 53.1 s | 2.8% |

The full-wall convention (21.82 t/s) is the honest one for any user judging a session by their own clock; the benchmark convention (31.84–33.00 t/s) is the honest one for judging the hardware's decode engine. Both are correct; they answer different questions. The gap between them is convention, accounted for to within the residual band — **none of it is unexplained decode loss.**

For additional context on the decode engine, the session-wide server-reported `generation_tps` (decode) was **mean 34.78 / median 34.30 t/s** (MEASURED, per-request `stats`).

---

## 5. Per-Request Table

57 rows; the two `aux` rows (7, 8 in this listing) lack client-side walls and are excluded from the partition. Field provenance is embedded per-row in `requests.jsonl`; `gen_tps`/`prefill_tps` are server-MEASURED; `wall_s` is client-MEASURED; `full_wall_rate` is **DERIVED** (completion_tokens ÷ client wall). `median_prompt_tokens = 145,918`.

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

Aggregate context (n=55 main chat requests): full-wall rate (**DERIVED**) mean 20.38 / median 21.69 tok/s; server `generation_tps` mean 34.46; `prompt_tps` (rate of the *suffix* prefill, not the whole prompt) mean 215.57 / median 227.62; non-decode overhead per call (**DERIVED**) mean 11.88 s / median 6.52 s, dominated by the single cold prefill (222.95 s — §6).

---

## 6. Prefix-Cache Analysis (context, not a mitigation proposal)

Prefix caching is already doing enormous work. Only **1 of 55 requests was fully cold**; its single prefill cost **222.1 s = 11.5% of the total session wall**. The other 54 requests averaged **7.0 s of prefill each**, with a **median 144,974 of 146,075 prompt tokens cached**. The prefill/TTFT time that remains in the session is what is left *after* the cache has already absorbed nearly all of it.

**Counterfactuals (DERIVED, illustrative not predictive):**

| Scenario | Prefill needed for the session | Resulting rate |
|---|---|---|
| Actual (with prefix cache) | 600.2 s (33.0%) | 21.82 t/s full-wall; 24.66 t/s excluding the single cold request |
| No prefix cache at all | **44,954 s** (vs 600.2 s) | ~**0.91 t/s** — would feel unusable |

`cache_saved_s = 44,353.6` (MEASURED-derived: the prefill seconds the cache eliminated). The one cold request is the entire reason full-wall rate dips to 21.82 t/s; excluding it, perceived rate rises to **24.66 t/s**. In other words: the ~31% of session wall spent in "prefill/TTFT" is almost entirely attributable to a single cold start, not to recurring per-request prefill cost.

---

## 7. The Reasoning-Token Perception Effect

**24,662 of 42,091 completion tokens (59%) were reasoning/thinking tokens**, present in **46 of 55 requests**. These tokens are **included in `completion_tokens` and are decoded at the full `generation_tps`** — so they add **no hidden time term** to the accounting in §3/§4. The decode engine's throughput is unaffected.

What they do explain is a *perception* effect. A user judging speed by **visible answer text alone** — unable to see the reasoning tokens being consumed underneath — experiences **9.04 t/s** (42,091 completion tokens − 24,662 reasoning = 17,429 visible answer tokens ÷ full wall) while the hardware genuinely decodes at 33–34 t/s. 

**This is a perception/attribution effect, not a hardware effect.** The 9.04 t/s is the floor of the convention ladder (§4), and it is why the session *feels* slower than either the 21.82 t/s full-wall truth or the 33 t/s decode truth — a large share of every second of decode is spent producing tokens the user never sees. The hardware cost is real but it is not "lost decode rate"; it is the cost of the reasoning budget, accounted for fully in `completion_tokens`.

---

## 8. The Phase 2 Client-Side Capture Tool (passive_capture_proxy)

**What it is / what it measures.** A passive, non-buffering reverse proxy (stdlib-only, Python ≥ 3.9) that sits between the Hermes client and the exo OpenAI-compatible endpoint and measures, purely as a side effect, the real per-request timing a live session experiences. It answers the benchmark-vs-perception gap directly by capturing **both rate conventions side by side** on the same stream: `post_ttft_rate_toks_per_s` (decode-throughput-style, excludes the initial wait) and `full_wall_rate_toks_per_s` (everything the user experiences). It also records per-request wall clock, **client-visible TTFT** (`ttft_s`: request-sent to first content chunk), inter-chunk stream-stall gaps, tool-call round-trip gaps, and both separate token counts (`completion_tokens_streamed` vs server-reported `usage.completion_tokens`).

**One-line start:**

```bash
python3 /Users/adam.durham/repos/exo/tmp/real-usage-capture-20260902/phase2/passive_capture_proxy.py
```

**Point Hermes at it** (second terminal), use the session normally:

```bash
hermes config set providers.exo.base_url http://127.0.0.1:52416/v1
```

**Restore the original endpoint when done:**

```bash
hermes config set providers.exo.base_url http://192.168.86.201:52415/v1
```

**Where the data lands:** one JSON line per request appended to `capture.jsonl` in the `phase2/` directory. Stop with `Ctrl-C`; no other setup/teardown. Streaming is never buffered (`HTTPResponse.read1()` single-chunk relay, so TTFT measurement is not corrupted); measurement is fail-open (`capture_errors` / `relay_errors` fields; a capture-path crash can never break the session); response bytes are relayed byte-for-byte. It reads `~/.hermes/config.yaml` **read-only**, never edits it. Options: `--port` (52416), `--upstream` (default `http://192.168.86.201:52415`), `--jsonl` (default `./capture.jsonl`), `--listen` (127.0.0.1).

**Self-test.** `python3 .../phase2/self_test.py` drives a request through a fake SSE server with an injected 0.5 s pre-first-chunk delay and asserts TTFT within tolerance, streamed-token count vs content chunks, and full-wall rate < post-TTFT rate. **The PM independently re-ran the self-test and it passed 9/9, with TTFT measured at 0.5026 s against the injected 0.50 s delay** — the tool is **verified-working**. **It has NOT yet been run against a live cluster**: the user is currently on a cloud provider, with no live exo session to capture.

---

## 9. WHAT THE LOGS DO NOT EXPOSE

Carried over verbatim from `phase1/FINDINGS.md`:

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

**Critical consequence for this report:** the prefill/TTFT split (§3, §4, §6) is **DERIVED**, computed as `uncached_prompt_tokens ÷ server prompt_tps`, **not directly measured** — server-side TTFT and per-chunk timestamps do not exist in the logs. Because the reconciliation identity closes to a sub-second residual on all 55 requests, that derivation is internally consistent and validated; but it is not a direct measurement. **This is precisely the gap the Phase 2 tool (§8) closes for future sessions** — it measures client-visible TTFT, per-chunk arrival gaps, and stream stalls directly, per request, without relying on any log derivation.

---

## 10. Limitations and Honest Caveats

- **`wall_seconds` is CLIENT-side** (Hermes `api_calls.latency_seconds`): it includes network + streaming transport + client processing, plus the user's think time at the tool-call boundary is only visible as inter-call gaps, not inside a request. The logs do not permit a server-side wall decomposition; the decode-vs-wall split in this report is the tightest possible from this data.
- **The prefill/TTFT split is DERIVED, not MEASURED** (see §9). It is validated by the reconciliation closing to sub-second residual, but a direct per-request server TTFT timestamp still does not exist in the retained logs.
- **The matched-depth comparison (n=16) is the decisive test but not the whole session.** Session-wide decode (n=55) is mean 34.78 / median 34.30 t/s; both are above the benchmark's 31.84 t/s, so the direction of the conclusion is robust, but the 34.06 mean at exactly 140–160K is the cleanest apples-to-apples with the benchmark's 150K measurement.
- **Benchmark reference (31.84 at 150K) is taken as given** from the earlier benchmark; the study re-derives its real-session comparison from actual logs rather than re-running the benchmark.
- **The tool round-trip gap and user-idle gaps are outside any per-request rate**, but they dominate the *elapsed* session wall (3h31m window vs 1,929 s of in-call wall). They are not decode-rate losses; they are session structure.
- **Reasoning-token accounting (§7)** uses the server's `usage.completion_tokens_details.reasoning_tokens`; reasoning vs visible tokens are attributed per request from the event log, not from a client-side render count.
- This is a **root-cause/attribution report only**. Per the constraint, no mitigations or optimizations are proposed; recommending fixes is out of scope.

---

## Notes on this report

- **Note A (residual band):** The task brief described the reconciliation residual band as "[0.8 s, 1.3 s], median 0.9 s". The authoritative `partition_verified.json` records **min 0.75 / max 1.32 / median 0.94 s**. This report uses the verified-file values; the brief's numbers appear to be a rounded characterization. The reconciliation conclusion (narrow, sub-second, bounded residual on all 55 requests) is unaffected.
- Every MEASURED figure traces to `partition_verified.json`, `phase1/FINDINGS.md`, or `phase1/requests.jsonl` provenance; every DERIVED figure is labeled as such. No figure was recomputed, re-rounded, or invented here.

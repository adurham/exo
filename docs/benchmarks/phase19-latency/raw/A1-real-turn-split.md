# A1 — REAL per-call prefill-vs-decode decomposition inputs

**Session** `20261007_092009_9a2ed7` · provider `custom` (exo / dsv41 engine) · 2026-10-07, first call 09:20:21 CDT → last call end 09:50:35 CDT. All 42 ledger rows are `provider='custom'`.

## Derivation of every number (sources)

**Ledger** — `/Users/adam.durham/.hermes/state.db`, table `api_calls`, opened read-only (`file:...?mode=ro`, `uri=True`):
- `WHERE session_id='20261007_092009_9a2ed7' ORDER BY call_seq` → 42 rows, all `provider='custom'`.
- `started`(CDT) = `datetime(started_at,'unixepoch','-5 hours')`; `latency_s`=`latency_seconds`; `out`=`output_tokens`; `reas`=`reasoning_tokens`; `prompt`=`prompt_tokens_total`; `cread`=`cache_read_tokens`.

**Exo log (authoritative)** — `macstudio-m4-1:~/exo.log`, absolute `/Users/adam.durham/exo.log` (ssh-confirmed).
- `turn reuse:` line is emitted by `dsv41/engine.py:_start_turn` **immediately after `session.prefill(tokens)` returns** → the timestamp is the **prefill-completion** instant, NOT request start (verified in source, lines 950-968).
- `session reuse:` and `prefill controls: (rows=M)` lines are emitted at `session.get` / prefill entry → **near request start** (+0.2–0.4 s).
- `delta_rows = prefill=` (rows actually re-fed); `reuse_rows = reuse=` (rows served from live cache). Both come from the same `turn` object.
- **Match rule**: `prompt=` in the log equals ledger `prompt_tokens_total` (exact) **and** time order. Independent check: ledger `cache_read_tokens == log reuse=` for **all 41** matched calls (0 mismatches).
- Cross-checked on `macstudio-m4-2:~/exo.log` (rank 1/2 shard): identical values, timestamps +~1 ms. m4-1 used as authoritative.
- `reuse undershoot` warning is emitted only when `prefill_tokens > 256` (source line 960) — so its absence on a small-delta call is expected, not a signal.

**Time window searched:** `2026-10-07 09:20:00`–`09:51:00` CDT. `turn reuse:` lines found in window: **41**. Leftover/unmatched: **0**. The single call without a reuse line is **call_seq 1 → COLD / FULL prefill (`delta_rows = FULL prompt = 24095`, `prefill==prompt`); stated explicitly, not guessed.**

> NOTE (`prefill_s` / `decode_s`): left **BLANK by design** — the delta-prefill ROWS/S rate is being measured separately by the parent. As supporting forensics only, the `turn reuse` line's offset from call start equals the prefill-completion latency (e.g. call 7: +56.4 s of a 136.1 s call); the actual prefill rate must come from the parent's ROWS/S measurement, so no prefill_s is asserted here.

## Per-call table

| call_seq | started CDT | latency_s | output_tok | reasoning_tok | prompt_tok | cache_read | delta_rows (prefill=) | reuse_rows (reuse=) | prefill_s | decode_s | prefill-dominated? | undershoot warn? |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 1 | 09:20:21 | 98.80 | 175 | 122 | 24095 | 0 | 24095 (FULL=cold) | — | _ | _ | — (cold full prefill) | — |
| 2 | 09:22:00 | 6.80 | 90 | 5 | 24340 | 24095 | 245 | 24095 | _ | _ | **YES** (Δ/g=2.58x) | no (Δ≤256, below warn threshold) |
| 3 | 09:22:51 | 14.79 | 233 | 93 | 25300 | 24431 | 869 | 24431 | _ | _ | **YES** (Δ/g=2.67x) | **YES** |
| 4 | 09:23:06 | 25.28 | 284 | 37 | 28743 | 25300 | 3443 | 25300 | _ | _ | **YES** (Δ/g=10.73x) | **YES** |
| 5 | 09:23:32 | 15.13 | 231 | 54 | 30178 | 28743 | 1435 | 28743 | _ | _ | **YES** (Δ/g=5.04x) | **YES** |
| 6 | 09:23:47 | 42.88 | 133 | 77 | 39293 | 30178 | 9115 | 30178 | _ | _ | **YES** (Δ/g=43.40x) | **YES** |
| 7 | 09:24:30 | 136.14 | 1572 | 1228 | 53564 | 39293 | 14271 | 39293 | _ | _ | **YES** (Δ/g=5.10x) | **YES** |
| 8 | 09:26:48 | 44.70 | 830 | 300 | 56313 | 54588 | 1725 | 54588 | _ | _ | **YES** (Δ/g=1.53x) | **YES** |
| 9 | 09:27:34 | 41.32 | 743 | 356 | 57541 | 56313 | 1228 | 56313 | _ | _ | **YES** (Δ/g=1.12x) | **YES** |
| 10 | 09:28:17 | 85.00 | 1432 | 945 | 59602 | 57541 | 2061 | 57541 | _ | _ | no (Δ/g=0.87x) | **YES** |
| 11 | 09:29:46 | 19.04 | 245 | 92 | 61169 | 59602 | 1567 | 59602 | _ | _ | **YES** (Δ/g=4.65x) | **YES** |
| 12 | 09:30:06 | 42.73 | 721 | 181 | 63904 | 61169 | 2735 | 61169 | _ | _ | **YES** (Δ/g=3.03x) | **YES** |
| 13 | 09:30:56 | 37.78 | 606 | 495 | 64853 | 63904 | 949 | 63904 | _ | _ | no (Δ/g=0.86x) | **YES** |
| 14 | 09:31:33 | 12.26 | 109 | 19 | 66620 | 64853 | 1767 | 64853 | _ | _ | **YES** (Δ/g=13.80x) | **YES** |
| 15 | 09:31:46 | 51.42 | 828 | 661 | 68147 | 66620 | 1527 | 66620 | _ | _ | **YES** (Δ/g=1.03x) | **YES** |
| 16 | 09:32:38 | 15.94 | 199 | 40 | 69862 | 68147 | 1715 | 68147 | _ | _ | **YES** (Δ/g=7.18x) | **YES** |
| 17 | 09:32:55 | 6.33 | 92 | 0 | 70159 | 69862 | 297 | 69862 | _ | _ | **YES** (Δ/g=3.23x) | **YES** |
| 18 | 09:33:02 | 9.17 | 159 | 0 | 70622 | 70159 | 463 | 70159 | _ | _ | **YES** (Δ/g=2.91x) | **YES** |
| 19 | 09:33:12 | 82.59 | 1542 | 1023 | 71044 | 70622 | 422 | 70622 | _ | _ | no (Δ/g=0.16x) | **YES** |
| 20 | 09:34:37 | 52.64 | 885 | 427 | 72678 | 72070 | 608 | 72070 | _ | _ | no (Δ/g=0.46x) | **YES** |
| 21 | 09:38:53 | 42.46 | 804 | 650 | 73568 | 73564 | 4 | 73564 | _ | _ | no (Δ/g=0.00x) | no (Δ≤256, below warn threshold) |
| 22 | 09:39:35 | 19.28 | 299 | 67 | 74893 | 73568 | 1325 | 73568 | _ | _ | **YES** (Δ/g=3.62x) | **YES** |
| 23 | 09:39:55 | 14.38 | 265 | 81 | 75338 | 74893 | 445 | 74893 | _ | _ | **YES** (Δ/g=1.29x) | **YES** |
| 24 | 09:40:11 | 46.99 | 251 | 143 | 83663 | 75338 | 8325 | 75338 | _ | _ | **YES** (Δ/g=21.13x) | **YES** |
| 25 | 09:40:58 | 11.21 | 123 | 14 | 84980 | 83663 | 1317 | 83663 | _ | _ | **YES** (Δ/g=9.61x) | **YES** |
| 26 | 09:41:09 | 21.99 | 375 | 216 | 86015 | 84980 | 1035 | 84980 | _ | _ | **YES** (Δ/g=1.75x) | **YES** |
| 27 | 09:41:32 | 17.75 | 353 | 22 | 86529 | 86015 | 514 | 86015 | _ | _ | **YES** (Δ/g=1.37x) | **YES** |
| 28 | 09:41:51 | 69.18 | 1110 | 786 | 87509 | 86529 | 980 | 86529 | _ | _ | no (Δ/g=0.52x) | **YES** |
| 29 | 09:43:01 | 45.21 | 720 | 513 | 88853 | 87509 | 1344 | 87509 | _ | _ | **YES** (Δ/g=1.09x) | **YES** |
| 30 | 09:43:47 | 24.70 | 373 | 165 | 90793 | 88853 | 1940 | 88853 | _ | _ | **YES** (Δ/g=3.61x) | **YES** |
| 31 | 09:44:11 | 19.82 | 293 | 95 | 92236 | 90793 | 1443 | 90793 | _ | _ | **YES** (Δ/g=3.72x) | **YES** |
| 32 | 09:44:32 | 39.34 | 597 | 486 | 94402 | 92236 | 2166 | 92236 | _ | _ | **YES** (Δ/g=2.00x) | **YES** |
| 33 | 09:45:12 | 10.76 | 147 | 0 | 95238 | 94402 | 836 | 94402 | _ | _ | **YES** (Δ/g=5.69x) | **YES** |
| 34 | 09:45:24 | 21.65 | 373 | 250 | 95611 | 95238 | 373 | 95238 | _ | _ | no (Δ/g=0.60x) | **YES** |
| 35 | 09:45:45 | 7.47 | 103 | 0 | 96008 | 95611 | 397 | 95611 | _ | _ | **YES** (Δ/g=3.85x) | **YES** |
| 36 | 09:45:53 | 22.36 | 362 | 258 | 97134 | 96008 | 1126 | 96008 | _ | _ | **YES** (Δ/g=1.82x) | **YES** |
| 37 | 09:46:16 | 12.90 | 157 | 52 | 98382 | 97134 | 1248 | 97134 | _ | _ | **YES** (Δ/g=5.97x) | **YES** |
| 38 | 09:46:30 | 18.26 | 271 | 160 | 99585 | 98382 | 1203 | 98382 | _ | _ | **YES** (Δ/g=2.79x) | **YES** |
| 39 | 09:46:48 | 59.54 | 995 | 725 | 100945 | 99585 | 1360 | 99585 | _ | _ | no (Δ/g=0.79x) | **YES** |
| 40 | 09:47:49 | 110.62 | 1906 | 1411 | 102063 | 100945 | 1118 | 100945 | _ | _ | no (Δ/g=0.34x) | **YES** |
| 41 | 09:49:41 | 30.57 | 452 | 150 | 104396 | 103088 | 1308 | 103088 | _ | _ | **YES** (Δ/g=2.17x) | **YES** |
| 42 | 09:50:12 | 23.42 | 361 | 0 | 104880 | 104396 | 484 | 104396 | _ | _ | **YES** (Δ/g=1.34x) | **YES** |

## Flags

- **Prefill-dominated calls (delta_rows > output+reasoning tokens), 32 of 42:** 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 41, 42
  - Worst offenders: call 6 Δ=9115 vs 210 gen (43x), call 24 Δ=8325 vs 394 gen (21x), call 14 Δ=1767 vs 128 gen (14x), call 4 Δ=3443 vs 321 gen (11x), call 25 Δ=1317 vs 137 (10x).
- **`reuse undershoot` warnings** (refed>256, i.e. prefix collapsed to older checkpoint): fired on **39 of 42** calls (calls 2 and 21 have Δ≤256 so no warning, correctly per source threshold). Undershoot is effectively universal in this turn → the resident-session reuse is working but the prefix match consistently falls 1–N rows short of the newest boundary, forcing extra refeeds.
- **Cold call:** call_seq 1 (no resident session; full 24095-row prefill, latency 98.8 s).
- **Reuse undershoot warnings IN WINDOW: 39.** No other warning types flagged.

## Raw matched log lines used (macstudio-m4-1:~/exo.log)

All lines below are anchored to the 2026-10-07 09:20–09:51 CDT window. Format: `<logline_no>: [ <timestamp> | LEVEL | module:fn:line ] [DSV41] ...`.

### call_seq 1 — 09:20:21 — COLD (no `turn reuse:` line)
- none found (first request of session). delta_rows = FULL prompt = 24095 (prefill==prompt).
### call_seq 2 — 09:22:00 — delta=245 reuse=24095
- `10881:[ 2026-10-07 09:22:00.257 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 24340-token prompt matches a resident conversation on 24221 rows[0m`
- `10899:[ 2026-10-07 09:22:01.933 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=24340 prefill=245 reuse=24095 cache=24340 rewind=24270 UNCOMMITTED[0m`

### call_seq 3 — 09:22:51 — delta=869 reuse=24431
- `11997:[ 2026-10-07 09:22:52.074 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 25300-token prompt matches a resident conversation on 24431 rows[0m`
- `12128:[ 2026-10-07 09:22:56.053 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=25300 prefill=869 reuse=24431 cache=25300 UNCOMMITTED[0m`
- `12129:[ 2026-10-07 09:22:56.053 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=869 rows (reused=24431)[0m`

### call_seq 4 — 09:23:06 — delta=3443 reuse=25300
- `13040:[ 2026-10-07 09:23:07.067 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 28743-token prompt matches a resident conversation on 25397 rows[0m`
- `13122:[ 2026-10-07 09:23:20.512 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=28743 prefill=3443 reuse=25300 cache=28743 rewind=25533 UNCOMMITTED[0m`
- `13123:[ 2026-10-07 09:23:20.512 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=3443 rows (reused=25300)[0m`

### call_seq 5 — 09:23:32 — delta=1435 reuse=28743
- `13974:[ 2026-10-07 09:23:32.409 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 30178-token prompt matches a resident conversation on 28784 rows[0m`
- `14011:[ 2026-10-07 09:23:38.254 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=30178 prefill=1435 reuse=28743 cache=30178 rewind=29027 UNCOMMITTED[0m`
- `14012:[ 2026-10-07 09:23:38.254 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1435 rows (reused=28743)[0m`

### call_seq 6 — 09:23:47 — delta=9115 reuse=30178
- `15740:[ 2026-10-07 09:23:47.794 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 39293-token prompt matches a resident conversation on 30236 rows[0m`
- `16074:[ 2026-10-07 09:24:23.975 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=39293 prefill=9115 reuse=30178 cache=39293 rewind=30409 UNCOMMITTED[0m`
- `16075:[ 2026-10-07 09:24:23.975 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=9115 rows (reused=30178)[0m`

### call_seq 7 — 09:24:30 — delta=14271 reuse=39293
- `18124:[ 2026-10-07 09:24:30.715 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 53564-token prompt matches a resident conversation on 39374 rows[0m`
- `18480:[ 2026-10-07 09:25:26.861 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=53564 prefill=14271 reuse=39293 cache=53564 rewind=39427 UNCOMMITTED[0m`
- `18481:[ 2026-10-07 09:25:26.861 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=14271 rows (reused=39293)[0m`

### call_seq 8 — 09:26:48 — delta=1725 reuse=54588
- `23647:[ 2026-10-07 09:26:48.432 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 56313-token prompt matches a resident conversation on 54796 rows[0m`
- `23696:[ 2026-10-07 09:26:55.518 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=56313 prefill=1725 reuse=54588 cache=56313 rewind=55137 UNCOMMITTED[0m`
- `23697:[ 2026-10-07 09:26:55.518 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1725 rows (reused=54588)[0m`

### call_seq 9 — 09:27:34 — delta=1228 reuse=56313
- `26618:[ 2026-10-07 09:27:34.679 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 57541-token prompt matches a resident conversation on 56617 rows[0m`
- `26653:[ 2026-10-07 09:27:40.042 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=57541 prefill=1228 reuse=56313 cache=57541 rewind=57143 UNCOMMITTED[0m`
- `26654:[ 2026-10-07 09:27:40.042 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1228 rows (reused=56313)[0m`

### call_seq 10 — 09:28:17 — delta=2061 reuse=57541
- `29879:[ 2026-10-07 09:28:17.543 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 59602-token prompt matches a resident conversation on 57901 rows[0m`
- `29931:[ 2026-10-07 09:28:26.128 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=59602 prefill=2061 reuse=57541 cache=59602 rewind=58284 UNCOMMITTED[0m`
- `29932:[ 2026-10-07 09:28:26.128 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=2061 rows (reused=57541)[0m`

### call_seq 11 — 09:29:46 — delta=1567 reuse=59602
- `34795:[ 2026-10-07 09:29:46.583 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 61169-token prompt matches a resident conversation on 60551 rows[0m`
- `34827:[ 2026-10-07 09:29:53.071 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=61169 prefill=1567 reuse=59602 cache=61169 rewind=61035 UNCOMMITTED[0m`
- `34828:[ 2026-10-07 09:29:53.071 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1567 rows (reused=59602)[0m`

### call_seq 12 — 09:30:06 — delta=2735 reuse=61169
- `37443:[ 2026-10-07 09:30:06.782 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 63904-token prompt matches a resident conversation on 61265 rows[0m`
- `37510:[ 2026-10-07 09:30:18.019 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=63904 prefill=2735 reuse=61169 cache=63904 rewind=61414 UNCOMMITTED[0m`
- `37511:[ 2026-10-07 09:30:18.019 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=2735 rows (reused=61169)[0m`

### call_seq 13 — 09:30:56 — delta=949 reuse=63904
- `40863:[ 2026-10-07 09:30:56.494 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 64853-token prompt matches a resident conversation on 64089 rows[0m`
- `40891:[ 2026-10-07 09:31:00.855 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=64853 prefill=949 reuse=63904 cache=64853 rewind=64626 UNCOMMITTED[0m`
- `40892:[ 2026-10-07 09:31:00.856 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=949 rows (reused=63904)[0m`

### call_seq 14 — 09:31:33 — delta=1767 reuse=64853
- `44555:[ 2026-10-07 09:31:34.266 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 66620-token prompt matches a resident conversation on 65352 rows[0m`
- `44608:[ 2026-10-07 09:31:41.547 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=66620 prefill=1767 reuse=64853 cache=66620 rewind=65460 UNCOMMITTED[0m`
- `44609:[ 2026-10-07 09:31:41.547 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1767 rows (reused=64853)[0m`

### call_seq 15 — 09:31:46 — delta=1527 reuse=66620
- `47185:[ 2026-10-07 09:31:46.480 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 68147-token prompt matches a resident conversation on 66643 rows[0m`
- `47221:[ 2026-10-07 09:31:52.883 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=68147 prefill=1527 reuse=66620 cache=68147 rewind=66729 UNCOMMITTED[0m`
- `47222:[ 2026-10-07 09:31:52.883 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1527 rows (reused=66620)[0m`

### call_seq 16 — 09:32:38 — delta=1715 reuse=68147
- `51506:[ 2026-10-07 09:32:39.006 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 69862-token prompt matches a resident conversation on 68812 rows[0m`
- `51546:[ 2026-10-07 09:32:46.085 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=69862 prefill=1715 reuse=68147 cache=69862 rewind=68975 UNCOMMITTED[0m`
- `51547:[ 2026-10-07 09:32:46.085 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1715 rows (reused=68147)[0m`

### call_seq 17 — 09:32:55 — delta=297 reuse=69862
- `54305:[ 2026-10-07 09:32:55.850 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 70159-token prompt matches a resident conversation on 69906 rows[0m`
- `54328:[ 2026-10-07 09:32:57.832 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=70159 prefill=297 reuse=69862 cache=70159 rewind=70062 UNCOMMITTED[0m`
- `54329:[ 2026-10-07 09:32:57.832 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=297 rows (reused=69862)[0m`

### call_seq 18 — 09:33:02 — delta=463 reuse=70159
- `57021:[ 2026-10-07 09:33:02.727 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 70622-token prompt matches a resident conversation on 70159 rows[0m`
- `57034:[ 2026-10-07 09:33:05.242 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=70622 prefill=463 reuse=70159 cache=70622 rewind=70252 UNCOMMITTED[0m`
- `57035:[ 2026-10-07 09:33:05.242 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=463 rows (reused=70159)[0m`

### call_seq 19 — 09:33:12 — delta=422 reuse=70622
- `59752:[ 2026-10-07 09:33:12.726 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 71044-token prompt matches a resident conversation on 70622 rows[0m`
- `59767:[ 2026-10-07 09:33:15.165 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=71044 prefill=422 reuse=70622 cache=71044 rewind=70782 UNCOMMITTED[0m`
- `59768:[ 2026-10-07 09:33:15.165 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=422 rows (reused=70622)[0m`

### call_seq 20 — 09:34:37 — delta=608 reuse=72070
- `65300:[ 2026-10-07 09:34:37.654 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 72678-token prompt matches a resident conversation on 72071 rows[0m`
- `65323:[ 2026-10-07 09:34:40.794 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=72678 prefill=608 reuse=72070 cache=72678 rewind=72587 UNCOMMITTED[0m`
- `65324:[ 2026-10-07 09:34:40.795 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=608 rows (reused=72070)[0m`

### call_seq 21 — 09:38:53 — delta=4 reuse=73564
- `71691:[ 2026-10-07 09:38:53.486 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 73568-token prompt matches a resident conversation on 73564 rows[0m`
- `71696:[ 2026-10-07 09:38:53.698 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=73568 prefill=4 reuse=73564 cache=73568 UNCOMMITTED[0m`

### call_seq 22 — 09:39:35 — delta=1325 reuse=73568
- `76251:[ 2026-10-07 09:39:36.042 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 74893-token prompt matches a resident conversation on 74218 rows[0m`
- `76285:[ 2026-10-07 09:39:41.804 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=74893 prefill=1325 reuse=73568 cache=74893 rewind=74372 UNCOMMITTED[0m`
- `76286:[ 2026-10-07 09:39:41.805 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1325 rows (reused=73568)[0m`

### call_seq 23 — 09:39:55 — delta=445 reuse=74893
- `79643:[ 2026-10-07 09:39:56.153 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 75338-token prompt matches a resident conversation on 74964 rows[0m`
- `79663:[ 2026-10-07 09:39:58.676 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=75338 prefill=445 reuse=74893 cache=75338 rewind=75193 UNCOMMITTED[0m`
- `79664:[ 2026-10-07 09:39:58.676 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=445 rows (reused=74893)[0m`

### call_seq 24 — 09:40:11 — delta=8325 reuse=75338
- `83571:[ 2026-10-07 09:40:11.502 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 83663-token prompt matches a resident conversation on 75419 rows[0m`
- `83783:[ 2026-10-07 09:40:46.847 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=83663 prefill=8325 reuse=75338 cache=83663 rewind=75604 UNCOMMITTED[0m`
- `83784:[ 2026-10-07 09:40:46.848 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=8325 rows (reused=75338)[0m`

### call_seq 25 — 09:40:58 — delta=1317 reuse=83663
- `87840:[ 2026-10-07 09:40:58.566 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 84980-token prompt matches a resident conversation on 83810 rows[0m`
- `87987:[ 2026-10-07 09:41:04.397 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=84980 prefill=1317 reuse=83663 cache=84980 rewind=83915 UNCOMMITTED[0m`
- `87988:[ 2026-10-07 09:41:04.397 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1317 rows (reused=83663)[0m`

### call_seq 26 — 09:41:09 — delta=1035 reuse=84980
- `91762:[ 2026-10-07 09:41:09.822 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 86015-token prompt matches a resident conversation on 84998 rows[0m`
- `91792:[ 2026-10-07 09:41:14.664 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=86015 prefill=1035 reuse=84980 cache=86015 rewind=85104 UNCOMMITTED[0m`
- `91793:[ 2026-10-07 09:41:14.664 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1035 rows (reused=84980)[0m`

### call_seq 27 — 09:41:32 — delta=514 reuse=86015
- `96089:[ 2026-10-07 09:41:32.953 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 86529-token prompt matches a resident conversation on 86235 rows[0m`
- `96109:[ 2026-10-07 09:41:35.717 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=86529 prefill=514 reuse=86015 cache=86529 rewind=86390 UNCOMMITTED[0m`
- `96110:[ 2026-10-07 09:41:35.717 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=514 rows (reused=86015)[0m`

### call_seq 28 — 09:41:51 — delta=980 reuse=86529
- `100075:[ 2026-10-07 09:41:56.206 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 87509-token prompt matches a resident conversation on 86555 rows[0m`
- `100106:[ 2026-10-07 09:42:00.748 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=87509 prefill=980 reuse=86529 cache=87509 rewind=86883 UNCOMMITTED[0m`
- `100107:[ 2026-10-07 09:42:00.748 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=980 rows (reused=86529)[0m`

### call_seq 29 — 09:43:01 — delta=1344 reuse=87509
- `106051:[ 2026-10-07 09:43:02.634 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 88853-token prompt matches a resident conversation on 88299 rows[0m`
- `106087:[ 2026-10-07 09:43:08.499 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=88853 prefill=1344 reuse=87509 cache=88853 rewind=88620 UNCOMMITTED[0m`
- `106088:[ 2026-10-07 09:43:08.499 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1344 rows (reused=87509)[0m`

### call_seq 30 — 09:43:47 — delta=1940 reuse=88853
- `111441:[ 2026-10-07 09:43:47.577 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 90793-token prompt matches a resident conversation on 89370 rows[0m`
- `111488:[ 2026-10-07 09:43:55.576 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=90793 prefill=1940 reuse=88853 cache=90793 rewind=89573 UNCOMMITTED[0m`
- `111489:[ 2026-10-07 09:43:55.576 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1940 rows (reused=88853)[0m`

### call_seq 31 — 09:44:11 — delta=1443 reuse=90793
- `115958:[ 2026-10-07 09:44:12.226 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 92236-token prompt matches a resident conversation on 90962 rows[0m`
- `115999:[ 2026-10-07 09:44:18.484 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=92236 prefill=1443 reuse=90793 cache=92236 rewind=91166 UNCOMMITTED[0m`
- `116000:[ 2026-10-07 09:44:18.484 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1443 rows (reused=90793)[0m`

### call_seq 32 — 09:44:32 — delta=2166 reuse=92236
- `120352:[ 2026-10-07 09:44:32.635 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 94402-token prompt matches a resident conversation on 92335 rows[0m`
- `120427:[ 2026-10-07 09:44:42.104 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=94402 prefill=2166 reuse=92236 cache=94402 rewind=92530 UNCOMMITTED[0m`
- `120428:[ 2026-10-07 09:44:42.104 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=2166 rows (reused=92236)[0m`

### call_seq 33 — 09:45:12 — delta=836 reuse=94402
- `125832:[ 2026-10-07 09:45:12.964 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 95238-token prompt matches a resident conversation on 94892 rows[0m`
- `125857:[ 2026-10-07 09:45:17.011 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=95238 prefill=836 reuse=94402 cache=95238 rewind=95000 UNCOMMITTED[0m`
- `125858:[ 2026-10-07 09:45:17.011 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=836 rows (reused=94402)[0m`

### call_seq 34 — 09:45:24 — delta=373 reuse=95238
- `130079:[ 2026-10-07 09:45:24.575 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 95611-token prompt matches a resident conversation on 95238 rows[0m`
- `130091:[ 2026-10-07 09:45:26.793 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=95611 prefill=373 reuse=95238 cache=95611 rewind=95386 UNCOMMITTED[0m`
- `130092:[ 2026-10-07 09:45:26.793 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=373 rows (reused=95238)[0m`

### call_seq 35 — 09:45:45 — delta=397 reuse=95611
- `134915:[ 2026-10-07 09:45:46.750 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 96008-token prompt matches a resident conversation on 95865 rows[0m`
- `134936:[ 2026-10-07 09:45:49.123 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=96008 prefill=397 reuse=95611 cache=96008 rewind=95984 UNCOMMITTED[0m`
- `134937:[ 2026-10-07 09:45:49.123 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=397 rows (reused=95611)[0m`

### call_seq 36 — 09:45:53 — delta=1126 reuse=96008
- `139193:[ 2026-10-07 09:45:54.027 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 97134-token prompt matches a resident conversation on 96008 rows[0m`
- `139237:[ 2026-10-07 09:45:59.208 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=97134 prefill=1126 reuse=96008 cache=97134 rewind=96112 UNCOMMITTED[0m`
- `139238:[ 2026-10-07 09:45:59.208 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1126 rows (reused=96008)[0m`

### call_seq 37 — 09:46:16 — delta=1248 reuse=97134
- `144236:[ 2026-10-07 09:46:16.959 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 98382-token prompt matches a resident conversation on 97396 rows[0m`
- `144267:[ 2026-10-07 09:46:22.612 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=98382 prefill=1248 reuse=97134 cache=98382 rewind=97496 UNCOMMITTED[0m`
- `144268:[ 2026-10-07 09:46:22.612 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1248 rows (reused=97134)[0m`

### call_seq 38 — 09:46:30 — delta=1203 reuse=98382
- `148716:[ 2026-10-07 09:46:30.395 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 99585-token prompt matches a resident conversation on 98438 rows[0m`
- `148756:[ 2026-10-07 09:46:35.821 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=99585 prefill=1203 reuse=98382 cache=99585 rewind=98540 UNCOMMITTED[0m`
- `148757:[ 2026-10-07 09:46:35.821 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1203 rows (reused=98382)[0m`

### call_seq 39 — 09:46:48 — delta=1360 reuse=99585
- `153467:[ 2026-10-07 09:46:48.678 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 100945-token prompt matches a resident conversation on 99749 rows[0m`
- `153509:[ 2026-10-07 09:46:54.741 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=100945 prefill=1360 reuse=99585 cache=100945 rewind=99857 UNCOMMITTED[0m`
- `153510:[ 2026-10-07 09:46:54.741 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1360 rows (reused=99585)[0m`

### call_seq 40 — 09:47:49 — delta=1118 reuse=100945
- `159806:[ 2026-10-07 09:47:49.657 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 102063-token prompt matches a resident conversation on 101674 rows[0m`
- `159841:[ 2026-10-07 09:47:54.807 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=102063 prefill=1118 reuse=100945 cache=102063 rewind=101940 UNCOMMITTED[0m`
- `159842:[ 2026-10-07 09:47:54.807 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1118 rows (reused=100945)[0m`

### call_seq 41 — 09:49:41 — delta=1308 reuse=103088
- `167977:[ 2026-10-07 09:49:41.747 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 104396-token prompt matches a resident conversation on 103478 rows[0m`
- `168017:[ 2026-10-07 09:49:47.665 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=104396 prefill=1308 reuse=103088 cache=104396 rewind=103969 UNCOMMITTED[0m`
- `168018:[ 2026-10-07 09:49:47.665 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=1308 rows (reused=103088)[0m`

### call_seq 42 — 09:50:12 — delta=484 reuse=104396
- `173230:[ 2026-10-07 09:50:12.430 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.session:get:1004 ] [1m[DSV41] session reuse: this 104880-token prompt matches a resident conversation on 104550 rows[0m`
- `173252:[ 2026-10-07 09:50:15.143 | [1mINFO    [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:957 ] [1m[DSV41] turn reuse: prompt=104880 prefill=484 reuse=104396 cache=104880 rewind=104849 UNCOMMITTED[0m`
- `173253:[ 2026-10-07 09:50:15.143 | [33m[1mWARNING [0m | exo.worker.engines.mlx.dsv41.engine:_start_turn:964 ] [33m[1m[DSV41] reuse undershoot: refed=484 rows (reused=104396)[0m`

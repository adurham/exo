# ROUND 9 — BOOT 3 (arm B, RV=200)

Launch: `DSV4_KV_CACHE_BITS=0 EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200 ./start_cluster.sh`
(production defaults otherwise). Launch log `/tmp/r9_boot3_launch.log`.

- READY (2/2) at **2026-09-04T18:42:47Z**.
- Mandatory idle: **300 s slept** (18:42:47Z → 18:47:47Z). First rep 18:48:20Z.

## Gate — `ps eww` on the REAL runner PIDs, BOTH nodes (`results/boot3_env.txt`)

| node | runner PID | `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` |
|---|---|---|
| macstudio-m4-1 | 37706 | **200** |
| macstudio-m4-2 | 49659 | **200** |

`EXO_DECODE_PROBE` / `MLX_GPU_TIME` absent (correct — boot-2 only). **GATE PASS.**

## Set 1 — 5x 2K reps

reps (ms): 7230, 7810, 8270, 8470, 8180
**median 8180 ms, range [7230, 8470].** prompt_tokens 2239–2331. prompt_tps median 316.63.
residual median 674.8 ms, range [560.4, 821.2].

## Set 2 — 10x short reps (decision instrument)

reps (ms): 2180, 1960, 2130, 1730, 1900, 1760, 1820, 1850, 1640, 1730
**median 1835 ms, range [1640, 2180].** prompt_tokens 222–238. prompt_tps median 192.04.
**residual median 634.2 ms, range [558.6, 818.4]** (DIAGNOSTIC ONLY).

All 15 reps `prefix_cache_hit = none`.

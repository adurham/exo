# ROUND 9 — BOOT 1 (arm A, RV=200)

Launch: `DSV4_KV_CACHE_BITS=0 EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200 ./start_cluster.sh`
(all other settings at production defaults = R7 end state). Launch log `/tmp/r9_boot1_launch.log`.

- Teardown verified clean on both nodes before launch (0 `exo -v` procs).
- READY (2/2) at **2026-09-04T17:55:10Z**.
- Mandatory idle: **300 s slept** (17:55:10Z → 18:00:10Z). First rep at 18:00:45Z
  (= 5 m 35 s after READY).

## Gate — RV read off `ps eww` on the REAL runner PIDs, BOTH nodes

Raw capture: `results/boot1_env.txt`.

| node | runner PID (`.venv/bin/python -m exo -v`) | `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` as read |
|---|---|---|
| macstudio-m4-1 | 7507 | **200** |
| macstudio-m4-2 | 18268 | **200** |

Also confirmed identical on both: `EXO_DSV4_BATCHED_PREFILL=1`, `EXO_SPECULATIVE_GAMMA=3`,
`EXO_DSV4_MTP=1`, `MLX_STEEL_BATCH_INVARIANT=1`. `EXO_DECODE_PROBE` / `MLX_GPU_TIME` absent
(correct — those are boot-2 only). **GATE PASS.**

## Set 1 — 5x 2K reps (run FIRST, warm-state matching only, not the decision statistic)

| tag | TTFT (ms) | prompt_tokens | prompt_tps | residual (ms) | prefix_cache_hit |
|---|---|---|---|---|---|
| A_2k_r1 | 8560 | 2331 | 296.94 | 713.3 | none |
| A_2k_r2 | 8340 | 2239 | 289.09 | 598.5 | none |
| A_2k_r3 | 8370 | 2308 | 300.67 | 697.0 | none |
| A_2k_r4 | 7100 | 2285 | 350.66 | 586.6 | none |
| A_2k_r5 | 8980 | 2308 | 289.74 | 1017.6 | none |

**median 8370 ms, range [7100, 8980].** prompt_tokens 2239–2331.

## Set 2 — 10x short (~20-token) reps (the decision instrument)

| tag | TTFT (ms) | prompt_tokens | prompt_tps | residual (ms) | prefix_cache_hit |
|---|---|---|---|---|---|
| A_short_r1 | 2060 | 226 | 169.55 | 733.0 | none |
| A_short_r2 | 1690 | 226 | 201.72 | 574.6 | none |
| A_short_r3 | 1990 | 224 | 174.65 | 713.2 | none |
| A_short_r4 | 1850 | 226 | 188.49 | 656.3 | none |
| A_short_r5 | 2060 | 228 | 164.74 | 682.1 | none |
| A_short_r6 | 2070 | 224 | 165.45 | 722.2 | none |
| A_short_r7 | 2260 | 224 | 162.98 | 891.8 | none |
| A_short_r8 | 1930 | 226 | 178.74 | 671.2 | none |
| A_short_r9 | 1850 | 222 | 188.60 | 678.2 | none |
| A_short_r10 | 1850 | 226 | 193.94 | 689.8 | none |

**median 1960 ms, range [1690, 2260].** prompt_tokens 222–228.
prompt_tps median 176.70. **residual median 686.0 ms, range [574.6, 891.8]** (DIAGNOSTIC ONLY).

All 15 reps `prefix_cache_hit = none`.

## Byte-identity capture (RV=200 side), `--run-id r9id`, temp=0

`results/identity_RV200_{short,2k,89k}.json` + `.txt` (content) + `.reasoning.txt`.
prompt_tokens: short 194, 2K 1917, 89K 81867. NOTE: DSv4 is a thinking model and spent the
whole budget in `reasoning_content` (`finish_reason=length`, `content_chars=0` on all three),
so the diff is taken on `reasoning_content`, which is the actual generated token stream.

## I12 (runtime confirmation, taken on this boot's ~89K request)

Both nodes' runner logs, `~/.exo/exo_log/exo.log`:
- `Starting prefill` (SERIAL driver, `generate.py:899`): **19 occurrences**
- `Starting batched prefill` (`generate.py:1428`): **0 occurrences**
- 89K request, both nodes verbatim:
  `Prefill complete: 81866 tokens in 203.39s (402.5 tok/s)` (m4-1) /
  `... 203.37s (402.5 tok/s)` (m4-2)

**SERIAL DRIVER CONFIRMED at runtime on the 89K request, on both ranks.**
Tiled-SDPA / exact-topk markers: see boot-note discussion — no such log markers exist in the
deployed code (both branches are silent); reported in the final summary.

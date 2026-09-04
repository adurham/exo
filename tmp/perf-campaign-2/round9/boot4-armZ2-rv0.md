# ROUND 9 — BOOT 4 (arm Z second, RV=0) — FINAL BOOT, cluster left UP on this config

Launch: `DSV4_KV_CACHE_BITS=0 EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0 ./start_cluster.sh`
(production defaults otherwise; probe vars deliberately NOT set — boot-2 only).
Launch log `/tmp/r9_boot4_launch.log`.

- READY (2/2) at **2026-09-04T18:54:42Z**.
- Mandatory idle: **300 s slept** (18:54:42Z → 18:59:42Z). First rep 19:00:05Z.

## Gate — `ps eww` on the REAL runner PIDs, BOTH nodes (`results/boot4_env.txt`)

| node | runner PID | `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` |
|---|---|---|
| macstudio-m4-1 | 46161 | **0** |
| macstudio-m4-2 | 58339 | **0** |

**GATE PASS.**

## Set 1 — 5x 2K reps

reps (ms): 7940, 8130, 8100, 7330, 7620
**median 7940 ms, range [7330, 8130].** prompt_tokens 2285–2354. prompt_tps median 304.91.
residual median 400.4 ms, range [336.8, 493.9].

## Set 2 — 10x short reps (decision instrument)

reps (ms): 1450, 1520, 1640, 1440, 1740, 1430, 1720, 1680, 1690, 1490
**median 1580 ms, range [1430, 1740].** prompt_tokens 222–230. prompt_tps median 208.36.
**residual median 469.4 ms, range [405.7, 566.5]** (DIAGNOSTIC ONLY).

All 15 reps `prefix_cache_hit = none`.

## Byte-identity gate (Task 3) — matched-budget capture on this boot

Boot-1's RV=200 capture used `--max-tokens` 64/64/200 (short/2K/89K); the first RV=0 capture
on boot 2 used 400/400/200, so short and 2K were NOT comparable. This boot re-ran the RV=0
side at the SAME budgets as RV=200 (`results/identitym_RV0b4_*.json`). All use
`--run-id r9id`, temp=0, and identical `prompt_tokens` per size.

DSv4 is a thinking model: at these budgets `finish_reason=length` and the whole budget lands
in `reasoning_content` (`content` empty). The diff is therefore taken on the full generated
stream = `reasoning_content` + `content`.

| prompt | prompt_tokens (RV200 / RV0) | ctok | BYTE-IDENTICAL |
|---|---|---|---|
| short (20) | 194 / 194 | 64 / 64 | **YES** (275/275 chars, exact) |
| 2K | 1917 / 1917 | 64 / 64 | **YES** (283/283 chars, exact) |
| ~89K | 81867 / 81867 | 200 / 200 | **NO** — diverges at char 330; lens 905 vs 926 |

### Control that isolates the 89K divergence — RV=0 boot 2 vs RV=0 boot 4

Same prompt, same budget, same run-id, **same arm**, different boots:
`identity_RV0_89k.json` vs `identitym_RV0b4_89k.json` → **BYTE-IDENTICAL (926/926 chars,
exact)**. So RV=0 reproduces itself exactly across boots at 89K; the 89K difference is
between the *arms*, not boot noise.

First divergence (char 330), verbatim context:
```
RV=200: ... document contains repeated text and secret inserted. Need not reveal? ...
RV=0  : ... document contains repeated text and secret code inserted. Need not reveal? ...
```
Reported as measured. **The 89K byte-identity check FAILS; short and 2K PASS.**

## Clean-logs veto — RV=0 boot #2 (this boot)

Both nodes' `~/.exo/exo_log/exo.log` + `/tmp/r9_boot4_launch.log`:
- rank-disagreement / task-set-mismatch / "out of sync" / "closed communication": **0 on both nodes.**
- Launch log: **0** error/traceback/critical/fail hits (excluding the expected
  `error.svelte.js` build-artifact filename, as in R7).
- Every WARNING/ERROR-class line on both nodes falls in one of four pre-existing background
  classes, none inference-related:
  - `fetch_file_list_with_cache` HF catalog poll for `mlx-community/GLM-4.7-8bit-gs32`
    (an unrelated model; 10 on m4-1, 11 on m4-2)
  - `failed to validate model card` for 4 cards in `resources/inference_model_cards` (x4 both nodes)
  - `mx.metal.get_{peak,cache,active}_memory is deprecated` (x1 each)
  - `[transformers] Unrecognized keys in rope_parameters` (x1)
  - plus normal `[jaccl-v2] ENTER/EXIT rank=... call_id=...` collective trace lines
- Serial-prefill confirmation on this boot: `Starting prefill` **19x**,
  `Starting batched prefill` **0x**, identical on both nodes.

**VETO PASSES: zero errors, zero tracebacks attributable to inference, zero
rank-disagreement / task-set-mismatch evidence.**
(The same veto on RV=0 boot 2 gave the identical picture — see `boot2-armZ1-rv0.md`.)

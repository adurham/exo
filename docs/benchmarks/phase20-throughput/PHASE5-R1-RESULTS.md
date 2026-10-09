# PHASE 5 — R1 VALIDATION SESSION: RESULTS (relaunch #1 of ≤3)

Session: R1 (Phase-5 P3), 2026-10-08 CDT. Operator: R1 subagent.
Worktree: `/private/tmp/phase20-campaign` (branch `deploy/phase20-campaign`).
Status: **R1 ABORTED MID-SESSION — cluster RESTORED to production.** No ships.

## 0. TL;DR / verdict

**R1 FAILED at the LEVER arm: the lever build could not be measured cleanly.** The 91K
real-tensor capture hook, when enabled, flushes a ~197 MB `.npz` via
`np.savez_compressed` **on the server's main event loop**; the multi-minute zlib deflate
starves the exo runner's event channel and trips the 45 s runner hang-watchdog, which
**SIGKILLs the runner mid-prefill**. Both nodes were killed at 21:36. No lever-arm timing,
no capture for offline replay, no battery, no token-identity diff vs the lever.

Production is **RESTORED** to `deploy/next13 @ 576e9d279` + mlx-lm `3bf8316`, gates unset,
canary healthy, token-identical to the pre-R1 production.

- G2 (agentic Δ≥15 ms disjoint): **NOT TESTED** (no lever data).
- G3 (battery CLEAN on lever): **NOT TESTED**.
- Leg A (greedy token-identity): prod reference + prod-vs-prod control **PASS**; lever diff **NOT TESTED**.
- Leg C (91K replay 0-diff, HARD R2 precondition): **NOT SATISFIED** (no capture).
- Trigger telemetry (leg G): **NOT COLLECTED**.
- **R2 stays HARD-BLOCKED** (also on the separate req-3 re-ratification).

## 1. Ledger / budget (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| R1 | P3 | exo `deploy/next18-identity @ ff676b3ca` + mlx-lm `deploy/next18-lever2 @ 17bbd98`, capture env ON | fixed-replay A/B; 91K capture replay; token-identity; battery | **SPENT (relaunch #1)** |
| — | P3 | restore `deploy/next13 @ 576e9d279` + mlx-lm `3bf8316` | return to production after the R1 abort | **SPENT (counts as relaunch #2 per kit §I)** |
| R2 | P3 | next18 (L2-full default-on) | ship | **CONTINGENT + HARD-BLOCKED** (req-3 re-ratification) |
| R3 | P3 | restore best SHIPPED | reserve-only | HELD |

**Budget after R1: 2/3. One relaunch remains.**

Relaunch #1 was SPENT at the `./start_cluster.sh` invocation (it completed READY and was
verified live) — not "attempted". The restore that followed is relaunch #2.

## 2. A — PRE-FLIGHT (PASS)

- Both nodes: `HEAD=576e9d279 MLX=3bf8316` (`Adams-Mac-Studio-M4-1`, `…-M4-2`).
- idle guard: `ok=True` (no active TextGeneration, no non-own POST in window, state.db clean).
- canary: **studio1 median 14.84, studio2 14.84 TFLOPS — healthy**.

## 3. B — BASELINE ARM (live production boot, 0 budget) (PASS)

Harness: `r1kit/r1_driver.py`, fixed salt `r1fix-a`, benign 20K ×4 + agentic 91K ×6.

| arm | reps (ms/round) | median | decode t/s | mean_accepted |
|---|---|---|---|---|
| agentic 91K | 130.23 / 130.28 / 130.63 / 130.38 / **130.42** | **130.38** | 24.09 | 2.1581 |
| benign 20K | 119.05 / 118.86 / 118.81 | **118.86** | 31.22 | 2.7736 |

Live production reproduced the **frozen next17-defaults anchor** (benign 118.56 / agentic
130.07, `PHASE4-CAMPAIGN.md:13-15`) within noise — NOT the older prod 145.75/157.10, which
predates the next17 gate. Content-fixed, same harness as the frozen runs. Drift check: PASS.

## 3b. LEG A — greedy token-identity (prod reference + prod-vs-prod CONTROL) (PASS)

The served chat API does **not** expose raw token ids (`TokenChunk.token_id` is dropped by
the OpenAI adapter; SSE emits text deltas only). Deltas are also **not 1:1 with tokens**
(observed 30 deltas for 29 tokens on a tiny prompt). So identity is established on the
strongest available end-to-end signal: full streamed text (thinking+content) SHA-256, the
re-encoded canonical token-id sequence (deterministic `f(text)` with the **exact DSv4
tokenizer**, `deepseek-ai--DeepSeek-V4-Flash-0731`), and `generation_tokens`.

- Fixed prompt: `RM.build_prompt(20000, 'r1fix-tokenid', 'count')` = 20,073 tokens, sha `deec4f8d1a3c71fa`.
- **prod-vs-prod control** (2 runs on the live prod boot): `all_text_identical=true`,
  `all_ids_identical=true`, `first_id_divergence=null`; sha_think `e65639ce0dd6ef41`, 300/300 gen tokens. **DETERMINISTIC.**
- **prod reference** (`legA_base`): sha_think `e65639ce0dd6ef41` — same.
- Lever-arm diff: **NOT OBTAINED** (see §5).

## 4. C — DEPLOY LEVER (relaunch #1) (PASS at deploy, then aborted)

- Working tree: exo `ff676b3ca` + mlx-lm WT `17bbd98`; guard checks OK
  (lever-2 `if _HIER and n > _FENCE_MIN_ROWS`=1, `_L2_FULL`=3, lever-1 `and m > _FENCE_MIN_ROWS`=1,
  `DSV41_NEXT18_CAPTURE` forward present ×9).
- Launch env: `EXO_TARGET_BRANCH=deploy/next18-identity`,
  `DSV41_NEXT18_CAPTURE=/tmp/next18_91k.npz`, `DSV41_NEXT18_CAPTURE_NS=1,4`; all A/B gate env UNSET (DEFAULTS).
- READY (2/2); post-boot canary **14.86 / 14.85 (healthy)**.
- **Installed-module verify (BOTH nodes):** `exo=ff676b3ca`, `mlx-lm-wt=17bbd98`;
  installed `indexer.py`: lever-2 guard ×1, `_L2_FULL` ×3; `_FENCE_MIN_ROWS` default 16;
  installed `sparse_attention.py`: lever-1 guard ×1.
- **Runner env (BOTH nodes):** `DSV41_NEXT18_CAPTURE=/tmp/next18_91k.npz`,
  `DSV41_NEXT18_CAPTURE_NS=1,4`; **no** `DSV41_INDEXER_L2_FULL` / `_HIER` / `_SMALLN_ROW_BF16` /
  `SPARSE_FENCE_MIN_ROWS` (DEFAULTS confirmed).
- Capture hook confirmed LIVE in the runners: log `[DSV41] next18 capture hook ACTIVE -> /tmp/next18_91k.npz`
  + stderr `[next18_capture] installed: path=… ns=[1, 4] max_calls=128 ab=True store_k=True flush_every=128`.

## 5. D — LEVER ARM (FAIL — runner SIGKILL)

First agentic rep (91K cold prompt) never completed. Timeline (studio1 exo.log):

```
21:35:37 WARN  _check_hang: Runner 64142fbf silent for 45s; liveness probe baseline footprint=106.10GB,
                extending 20s for a growth check (1/20).
21:36:00 WARN  [HANG_STACK] mode=shadow runner 64142fbf silent for 67s verdict=kill stack_class=gpu
                spin=True footprint_gb=106.10 growth_gb=+0.00 cpu_delta_s=21.0 spin_fraction=0.5 extensions_used=1/20
21:36:00 CRIT  Runner … hung: 1 task(s) in progress, no event for 67s (>45s). SIGKILLing … re-placement.
21:36:03 CRIT  [HANG_STACK] wrote thread dump to /tmp/exo_hang_89484.txt
21:36:08 ERROR Runner terminated with signal=9 (Killed: 9) … RuntimeError: Runner found to be dead
```

**ROOT CAUSE (from the hang dump, both nodes).** `/tmp/exo_hang_89484.txt` main-thread stack
ends in `dispatcher_vectorcall → zlib_Compress_compress → deflate (libz)`, sampling in
`deflate` for the entire 67 s window; ~2061 sample ticks, main thread busy the whole time
(spin=True). The capture hook's flush path — `np.savez_compressed` of a ring holding the
**full `index_k`** buffer per record ([1, ~91k, 128] bf16 ≈ 23 MB/record → ~197 MB `.npz` at the
ring flush) — runs **synchronously on the server's main event loop / request thread**. During
the multi-minute deflate no events are emitted, so the watchdog's event-silence heuristic
fires. Both nodes show the identical dump (`/tmp/exo_hang_94473.txt` on studio2).

**What landed:** `/tmp/next18_91k.npz.tmp.<pid>.npz` on both nodes (197,848,133 B) — the temp
file was mid-write (or the `os.replace` never completed because the process was SIGKILLed),
so **there is no valid capture** to replay and **no valid lever timing**. All lever reps are
null (`ttft` 0.73 s — the request was aborted, not served).

**Secondary confound (would invalidate a timing A/B even without the kill):** the flush and
the per-call A/B diff run inside the runner process. With `flush_every=128` the flush lands
**at the ~128th small-n call** — i.e. inside the decode window of ~250–260 rounds — so a
long serial deflate would block decode for the rest of that rep and distort its ms/round.
`DSV41_NEXT18_CAPTURE` must not be ON during a timing arm.

## 6. E/F/G — offline replay, battery, trigger telemetry (NOT RUN)

- **Leg C (91K replay):** no capture → not run. **HARD R2 precondition NOT satisfied.**
- **Battery (G3):** not run (lever arm never produced a clean measurement; running the
  battery under capture-flush instability would itself risk another watchdog kill).
- **Trigger telemetry:** not collected. (Note for the record: the L2-full guard's "trigger"
  is 100 % of small-n calls by design — it *is* the small-n path; a separate trigger-rate leg
  would belong to the L2-*guard* variant, not L2-full.)

## 7. I — RESTORE (PASS)

`git checkout --detach 576e9d279` + `git -C mlx-lm checkout 3bf8316`, all gate/capture env unset,
`EXO_TARGET_BRANCH=deploy/next13`, `./start_cluster.sh`.

- Nodes synchronized on commit **576e9d279**; READY (2/2); **canary 14.85 / 14.86 (healthy)**.
- Post-restore assertions (BOTH nodes): `exo=576e9d279`, `mlx-lm-wt=3bf8316`;
  installed `indexer.py` **`_L2_FULL`=0** (absent, expected); lever-1 guard present;
  runner env shows **no** `DSV41_*` keys. Capture temp files removed from both `/tmp`.
- **Parity smoke** vs the pre-R1 baseline: benign **118.21 ms** vs 118.86 baseline
  (**Δ 0.65 ms**, within noise; ship-day precedent reproduced to 0.04 ms); `mean_accepted`
  identical (2.7163).
- **Restored-prod token identity:** `sha_think=e65639ce0dd6ef41` == pre-R1 production
  reference. Gate env unset; no stray bench processes.

## 8. Why this needs a second relaunch (blocked by the R1 1-relaunch cap)

A capture that does not block the event loop needs either (a) an env change (e.g.
`DSV41_NEXT18_CAPTURE_STORE_K=0` — records a `index_k` SHA instead of the buffer, order
smaller; and/or a much larger `flush_every` so the flush lands after decode), **and/or**
(b) a code fix that moves the flush off the request thread. Both are **launch-time changes
requiring a relaunch**, which the R1 = 1-relaunch cap forbids. **HALT and report** is the
correct disposition (task rule: "If a step needs a second relaunch, HALT and report
instead."). Recommend the next round re-scope the capture as a **dedicated, bench-only,
short-prompt capture** (small n, tiny nb) off the timing path, or patch the hook to flush in
a worker thread / on a timer **between** requests.

## 9. Recommended next-step (for the plan owner; NOT executed)

1. Fix/limit the capture: set `DSV41_NEXT18_CAPTURE_STORE_K=0` (+ large `flush_every`) **or**
   move the flush off the request thread; keep capture **OFF** during any timing arm.
2. Re-run R1 as **R1b** (charge the remaining relaunch) with: lever timing arms (capture OFF)
   → then the capture leg as a separate, bounded activity.
3. G2/G3/leg-A/leg-C all still pending. **R2 remains HARD-BLOCKED** on req-3 re-ratification.

## 10. Artifacts

Laptop (`~/.hermes/cache/scratch/p5/r1/`): `baseline_agentic.json`, `baseline_benign.json`,
`restore_smoke_benign.json`, `LEDGER.md`, `PH-R1.md`, `legA_capture.py`, `run_arm.sh`,
`deploy_lever.sh`, `restore_production.sh`, `measure_baseline.log`, `measure_lever.log`.
`/tmp/p5r1/`: `legA_prod_control.json`, `legA_base.json`, `legA_restore.json`,
`legA_prompt.json`, `deploy_lever.log`, `restore_production.log`, `state_*.json`.
Nodes: `/tmp/exo_hang_89484.txt` (studio1), `/tmp/exo_hang_94473.txt` (studio2) — the root-cause dumps.

## 11. Evidence IDs

| ID | artifact |
|---|---|
| E-R1-BASE-AGENTIC | `baseline_agentic.json` — median 130.38 ms, reps [130.23,130.28,130.63,130.38,130.42] |
| E-R1-BASE-BENIGN | `baseline_benign.json` — median 118.86 ms, reps [119.05,118.86,118.81] |
| E-R1-LGACTRL | `/tmp/p5r1/legA_prod_control.json` — prod-vs-prod 0 diff |
| E-R1-LGABASE | `/tmp/p5r1/legA_base.json` — prod reference sha `e65639ce…` |
| E-R1-INSTALL | installed-module + runner-env read (both nodes), §4 |
| E-R1-KILL | exo.log HANG_STACK verdict=kill; `/tmp/exo_hang_89484.txt` (zlib flush) |
| E-R1-RESTORE | §7 restore assertions + parity smoke + token identity |
| E-R1-LEDGER | `LEDGER.md` |

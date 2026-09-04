# ROUND 9 — BOOT 2 (arm Z first, RV=0) + I15 ride-along

Launch: `DSV4_KV_CACHE_BITS=0 EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0 EXO_DECODE_PROBE=1
EXO_DECODE_PROBE_EVERY=16 MLX_GPU_TIME=1 MLX_GPU_TIME_LOG_EVERY=16 ./start_cluster.sh`
Everything else at production defaults. Launch log `/tmp/r9_boot2_launch.log`.

## VOID attempt (recorded for honesty, not measured)
A first boot-2 attempt at 18:08:26Z aborted at the launcher's own
`Verifying commit consistency between nodes` gate: `macstudio-m4-1: 82e168eba` vs
`macstudio-m4-2: 70013cab9`. Cause was mine — I committed the boot-1 note while the
launcher was mid-rsync, so the two nodes rsynced different working trees. **No reps were
run on that attempt.** Relaunched cleanly at 18:17:11Z with the tree stable; no commits
were made during any subsequent launch.

- READY (2/2) at **2026-09-04T18:21:23Z**.
- Mandatory idle: **300 s slept** (18:21:23Z → 18:26:23Z). First rep 18:26:40Z.

## Gate — `ps eww` on the REAL runner PIDs, BOTH nodes (`results/boot2_env.txt`)

| node | runner PID | `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` | probe vars |
|---|---|---|---|
| macstudio-m4-1 | 25044 | **0** | `EXO_DECODE_PROBE=1`, `MLX_GPU_TIME=1` |
| macstudio-m4-2 | 36536 | **0** | `EXO_DECODE_PROBE=1`, `MLX_GPU_TIME=1` |

**GATE PASS.**

## Set 1 — 5x 2K reps

| tag | TTFT (ms) | prompt_tokens | residual (ms) | prefix_cache_hit |
|---|---|---|---|---|
| Z1_2k_r1 | 8370 | — | 579.7 | none |
| Z1_2k_r2 | 7060 | — | 397.4 | none |
| Z1_2k_r3 | 7780 | — | 431.3 | none |
| Z1_2k_r4 | 7000 | — | 338.3 | none |
| Z1_2k_r5 | 7460 | — | 448.5 | none |

**median 7460 ms, range [7000, 8370].** prompt_tokens 2215–2377. prompt_tps median 325.28.
residual median 431.3 ms, range [338.3, 579.7].

## Set 2 — 10x short reps (decision instrument)

reps (ms): 1570, 1460, 1630, 1470, 1660, 1550, 1570, 1940, 1540, 1630

**median 1570 ms, range [1460, 1940].** prompt_tokens 220–236. prompt_tps median 201.84.
**residual median 484.7 ms, range [417.1, 725.0]** (DIAGNOSTIC ONLY).

All 15 reps `prefix_cache_hit = none`.

## I15 ride-along — kernel LAUNCHES per decode step: **STILL NOT OBTAINABLE (no count exists)**

The probe env vars were successfully set at process start this time (R8's blocker is gone —
confirmed on `ps eww`), and all three probes fired live in the runner logs:

```
[EXO_DECODE_PROBE pid=25183] tokens=16 wall_ms=54.24 gpu_ms=34.52 gpu_pct=63.6
[EXO_DECODE_PROBE pid=25183] tokens=32 wall_ms=45.12 gpu_ms=29.32 gpu_pct=65.0
[EXO_DECODE_PROBE pid=25183] tokens=48 wall_ms=45.86 gpu_ms=29.83 gpu_pct=65.1
[GPU_TIME pid=25183] steps=16 B=1 wall=29.57 gpu=24.24 pct=82.0 pre_fwd=0.002
                     fwd_build=27.19 sample=2.295 async=0.050 eval=0.00 post=0.03
[BG_DECODE_PROBE pid=25183] step=496 wall_ms=75.19 gpu_ms=84.36 gpu_pct=112.2
```
(31 `BG_DECODE_PROBE`, 3 `EXO_DECODE_PROBE`, 3 `STREAM_GEN_PROBE`, 1 `[GPU_TIME]` line on
node 1; node 2 identical shape under pid 36696.)

**But none of these emit a kernel-launch COUNT.** Read of the emitting code —
`src/exo/worker/engines/mlx/generator/generate.py:2416-2450`,
`batch_generate.py:4120-4149`, `mlx-lm/mlx_lm/generate.py:585-600` and `:1690-1712` —
shows every probe reports only wall-ms / gpu-ms / gpu-pct. `mx.metal.dispatch_count()` is
never called on any decode path in the deployed code (its only call sites are
`bench/minimax_*.py` and a minimax unit test, none of which run in the runner). So R8's
blocker is only *half* removed: the env vars now take effect, but there is no counter
wired to print. **No launches/step integer is reported. The pre-registered >500 / <200
bands cannot be applied.** Unblocking would need a one-line `dispatch_count()` delta added
to the probe — a code change, explicitly out of scope for this round.

Raw GPU-busy numbers, reported as-is: single-stream decode `gpu_pct` 63.6–65.1 %
(EXO_DECODE_PROBE, 2K-context), `[GPU_TIME]` `pct=82.0` at B=1, and batched-generator
`BG_DECODE_PROBE` `gpu_pct` 92.1–117.4 % on the long 89K decode.

## Clean-logs veto (RV=0 boot #1) — see final summary; PASSES

# ROUND 9 — RAW MEASUREMENT RESULTS (no decision made)

Four paired boots, executed in the pre-registered order, 2026-09-04.
This file is DATA ONLY. The ship/hold call is the PM's; the pre-registered band is not
applied here. `REPORT.md` deliberately not written.

## Per-boot table — short-prompt (~20-token) instrument

| boot | arm | RV as read from `ps eww`, m4-1 (pid) | RV, m4-2 (pid) | short median (ms) | short full RANGE (ms) | prompt_tokens range | 2K median (ms) |
|---|---|---|---|---|---|---|---|
| 1 | A (RV=200) | **200** (7507) | **200** (18268) | **1960** | [1690, 2260] | 222–228 | 8370 |
| 2 | Z first (RV=0) | **0** (25044) | **0** (36536) | **1570** | [1460, 1940] | 220–236 | 7460 |
| 3 | B (RV=200) | **200** (37706) | **200** (49659) | **1835** | [1640, 2180] | 222–238 | 8180 |
| 4 | Z second (RV=0) | **0** (46161) | **0** (58339) | **1580** | [1430, 1740] | 222–230 | 7940 |

All 60 reps (4 boots x [5x 2K + 10x short]) reported `prefix_cache_hit = none`.
Every boot: 300 s idle actually slept after READY before the first rep; boot order was
A → Z → B → Z as pre-registered; 2K reps always ran before short reps.

### Boot-variance bar
- **RV=200 A-vs-B spread (the pre-registered bar) = |1960 − 1835| = 125 ms.**
- RV=0 Z1-vs-Z2 spread (for reference) = |1570 − 1580| = **10 ms**.

### Gaps (raw, no interpretation)
- Median of RV=0 medians (1575) − median of RV=200 medians (1897.5) = **−322.5 ms**.
- Pairwise: Z1−A = −390, Z1−B = −265, Z2−A = −380, Z2−B = −255 ms.
- Range overlap: RV=0 max = 1940 (Z1_short_r8), RV=200 min = 1640 (B_short_r9) →
  **ranges are NOT entirely disjoint** (the single Z1 1940 rep overlaps the RV=200 band;
  the other 19 RV=0 reps all sit below both RV=200 minima).

## Secondary diagnostic — residual (DIAGNOSTIC ONLY, governs nothing)

`residual = prefill_s*1000 − ((prompt_tokens−1)/prompt_tps)*1000`, using server-side
`prompt_tps` from `server_stats`.

| boot | arm | short residual median (ms) | short residual range (ms) | short prompt_tps median | 2K residual median (ms) |
|---|---|---|---|---|---|
| 1 | A (200) | 686.0 | [574.6, 891.8] | 176.70 | 697.0 |
| 2 | Z1 (0) | 484.7 | [417.1, 725.0] | 201.84 | 431.3 |
| 3 | B (200) | 634.2 | [558.6, 818.4] | 192.04 | 674.8 |
| 4 | Z2 (0) | 469.4 | [405.7, 566.5] | 208.36 | 400.4 |

## Byte-identity gate (Task 3), `--run-id r9id`, temp=0

RV=200 side = boot 1; RV=0 side = boot 4, budgets matched exactly (64/64/200 tokens).
Diff taken on the full generated stream (`reasoning_content` + `content`) because DSv4
spends these budgets entirely in reasoning (`finish_reason=length`, `content` empty).

| prompt | prompt_tokens both arms | result |
|---|---|---|
| short (20) | 194 | **BYTE-IDENTICAL** |
| 2K | 1917 | **BYTE-IDENTICAL** |
| ~89K | 81867 | **NOT identical** — diverges at char 330 (905 vs 926 chars) |

**Control (isolates boot noise from the arm):** RV=0 boot 2 vs RV=0 boot 4, same 89K
prompt/budget/run-id → **BYTE-IDENTICAL, 926/926 chars.** RV=0 reproduces itself exactly
across boots at 89K, so the 89K difference tracks the arm, not the boot.

Divergence, verbatim:
```
RV=200: ... repeated text and secret inserted. Need not reveal? ...
RV=0  : ... repeated text and secret code inserted. Need not reveal? ...
```

## Clean-logs veto — run on BOTH RV=0 boots (2 and 4): **PASSES on both**

- rank-disagreement / task-set-mismatch / "out of sync" / "closed communication": **0** on
  both nodes, both boots.
- Launch logs: **0** real hits (only R7's expected `error.svelte.js` build-artifact filename).
- All WARNING-class lines on both nodes are pre-existing background, none inference-related:
  HF catalog poll for an unrelated model (`mlx-community/GLM-4.7-8bit-gs32`), 4 invalid model
  cards, `mx.metal.get_*_memory` deprecations, a transformers `rope_parameters` notice, plus
  normal `[jaccl-v2] ENTER/EXIT` collective trace lines.

## I15 ride-along (boot 2 only) — **NO COUNT OBTAINABLE; band cannot be applied**

R8's blocker (env vars not set at process start) is **removed**: `EXO_DECODE_PROBE=1` and
`MLX_GPU_TIME=1` were verified present on both runner PIDs and all probes fired live.

But **none of the deployed probes emit a kernel-launch count.** Every probe
(`generate.py:2416-2450`, `batch_generate.py:4120-4149`, `mlx-lm/generate.py:585-600`,
`:1690-1712`) prints only wall-ms / gpu-ms / gpu-pct. `mx.metal.dispatch_count()` is never
called on any decode path in deployed code (only in `bench/minimax_*.py` and a minimax unit
test). Getting launches/step needs a one-line code change — out of scope. **No integer is
reported; the >500 / <200 bands are not applied.**

Raw probe output actually captured (reported as-is, not a substitute):
```
[EXO_DECODE_PROBE pid=25183] tokens=16 wall_ms=54.24 gpu_ms=34.52 gpu_pct=63.6
[EXO_DECODE_PROBE pid=25183] tokens=32 wall_ms=45.12 gpu_ms=29.32 gpu_pct=65.0
[EXO_DECODE_PROBE pid=25183] tokens=48 wall_ms=45.86 gpu_ms=29.83 gpu_pct=65.1
[GPU_TIME pid=25183] steps=16 B=1 wall=29.57 gpu=24.24 pct=82.0 pre_fwd=0.002 fwd_build=27.19 sample=2.295 async=0.050 eval=0.00 post=0.03
[BG_DECODE_PROBE pid=25183] step=496 wall_ms=75.19 gpu_ms=84.36 gpu_pct=112.2
```

## I12 ride-along — SERIAL DRIVER **CONFIRMED at runtime**; SDPA/topk markers **DID NOT FIRE**

On the ~89K requests (run on boots 1, 2 and 4), both nodes' runner logs:
- `Starting prefill` (serial driver, `generate.py:899`): **19 occurrences per boot, per node**
- `Starting batched prefill` (`generate.py:1428`): **0 occurrences**
- 89K completion, verbatim (boot 1): `Prefill complete: 81866 tokens in 203.39s (402.5 tok/s)`
  on m4-1, `... 203.37s (402.5 tok/s)` on m4-2. Boot 2: `... 195.21s (419.4 tok/s)`.

**The serial driver ran, on both ranks, on the 89K request. Confirmed.**

**Tiled-SDPA / exact-topk markers: NOT FOUND — stated plainly, they did not fire, because no
such marker exists to fire.** Grep of both nodes' logs for `SPARSE_FUSED|QUERY_TILED|TILED|
exact_topk|EXACT_TOPK|TOPK` returned zero lines on every boot. Source read confirms why: the
tiled-SDPA branch (`mlx-lm/mlx_lm/models/deepseek_v4.py:4540`) and the exact-topk branch
(`:4186`) contain **no logging, print, or counter of any kind** — they are silent by
construction. The only nearby instrumentation is file-toggle-gated and targets different
branches (`/tmp/dsv4_fused_dispatch` counts the fused-softmax path at `:2850`,
`/tmp/dsv4_topk_dump` dumps indices at `:2806`), and `[SPARSE_FUSED]` at `:2007` reports the
fused-softmax epilogue, not either I12 branch. So the runtime confirmation I12 asked for is
**not obtainable without adding a counter** — that would be a code change, out of scope. The
serial-driver half of I12 IS confirmed; the marker half is not, for the stated reason.

## Artifacts

- Per-boot notes: `boot1-armA-rv200.md`, `boot2-armZ1-rv0.md`, `boot3-armB-rv200.md`,
  `boot4-armZ2-rv0.md`
- Raw JSON (all 60 reps + identity captures): `results/`
- `ps eww` captures: `results/boot{1,2,3,4}_env.txt`
- Drivers (new, this round): `run_boot.sh`, `run_reps.sh`, `run_identity.sh`,
  `run_identity_matched.sh`, `summarize.py`. `bench/long_decode_probe.py` was **not modified**.

## Execution notes / deviations

1. **One VOID boot-2 attempt** (18:08Z): the launcher's own node-commit-consistency gate
   aborted it (`m4-1: 82e168eba` vs `m4-2: 70013cab9`) because I committed the boot-1 note
   while the launcher was mid-rsync. No reps were run on it; boot 2 was relaunched cleanly at
   18:17Z. No commits were made during any subsequent launch.
2. **Launcher push-gate**: `start_cluster.sh:1132` `read -p` cannot be fed via a stdin pipe —
   the un-`-n`'d `ssh` calls at lines 944/984/1000/1033 drain the pipe first and the gate hits
   EOF → "Aborted." Answered via a tmux pty + `send-keys` instead (as R7 did). Recorded in
   `run_boot.sh`.
3. Identity budgets: the first RV=0 identity capture (boot 2) used 400-token budgets while the
   RV=200 side used 64 — not comparable. Re-run at matched 64/64/200 on boot 4; only the
   matched pair is reported above.
4. `start_cluster.sh` was **not edited**. Nothing under `src/` or `bench/` was modified.
   No pushes.

## End state

Cluster left **UP on boot 4 (RV=0)**, runner PIDs m4-1 **46161** / m4-2 **58339**, both
verified `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=0` on `ps eww`.

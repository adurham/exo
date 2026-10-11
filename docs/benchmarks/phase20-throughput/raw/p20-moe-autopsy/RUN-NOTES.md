# Phase-20 MoE autopsy — RUN NOTES (raw facts / commands / file:line)

Offline single-node bench on `studio2` (Mac Studio M4 Max, 128 GB), node venv
`~/repos/exo/.venv/bin/python` (mlx 0.32.3.dev20260918+603f16eb7). Production
was never touched: the instrumented mlx_lm lived only in the scratch dir
`/tmp/p20moe_pkg/` (md5 `ed54ff…`); the node's site-packages **and** src-tree
`gemv_metal.py` both stayed at `649d2d6a0a68c8d6eab2a8a5394b7ab9` (verified
before and after). Cluster idle-guarded + canary-checked before and after each
round (`bench/phase20_guard.py idle` rc=0, `canary` healthy 14.85 t/s both
nodes, `state_active_tasks=0`). 0 boots, 0 serving requests, read-only loads.

## Code facts (file:line)

- Instrumented log line: `mlx-lm/mlx_lm/models/exl3/gemv_metal.py:2011`
  (`if _MM_SEG_LOG: _mm_seg_log_capture(...)`, inside `inner_mm_seg_mlx`
  at `:1963`). Helper `_mm_seg_log_capture` at `:1901`; flag/list at
  `:1585-1586`. Env-gated `EXL3_MM_SEG_LOG=1`, default off — production path
  byte-identical when unset. The single Metal dispatch exposes no per-segment
  GPU cycles, so the log records the **host-side per-segment M/N/K
  decomposition** (from `blk_tab`/`nb_real`); the **launch wall** is measured
  host-side (`perf_counter`) around the drained `mx.eval` in the bench.
- Kernel grid: `((out_e//_MM_BN)*_GEM_THREADS, nb_max, 1)` at `:2059`;
  `_MM_BM=_MM_BN=64` at `:1567-1568`; `blk_tab` (3, nb_max) u32
  `[expert,row0,len]` at `:1898`; `nb_real` device scalar.
- Call sites: `exl3_moe.py:906` (gate), `:910` (up), `:935` (down) inside
  `EXL3SwitchGLU._prefill` (`:851`).

## Commands run

```
# scratch package (production untouched) + overlay instrumented file
ssh studio2 'rm -rf /tmp/p20moe_pkg && mkdir -p /tmp/p20moe_pkg /tmp/p20moe \
  && cp -R ~/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm /tmp/p20moe_pkg/mlx_lm'
scp mlx-lm/mlx_lm/models/exl3/gemv_metal.py studio2:/tmp/p20moe_pkg/mlx_lm/models/exl3/gemv_metal.py
scp bench/p20_moe_autopsy.py studio2:/tmp/p20moe/
scp docs/benchmarks/phase10-planb-gate-2026-09-28/raw/p30_exl3_trace_clamp.json studio2:/tmp/p20moe/
# per-stage run (idle-guarded; node, detached)
python3 bench/phase20_guard.py idle && python3 bench/phase20_guard.py canary
ssh studio2 'cd /tmp/p20moe && nohup bash run_stage.sh <STAGE> > stage<STAGE>.log 2>&1 &'
#   run_stage.sh exports P20_PKG=/tmp/p20moe_pkg EXL3_MM_SEG=v19d STAGE=<n> MB_REPS=7
#   and runs: ~/repos/exo/.venv/bin/python /tmp/p20moe/p20_moe_autopsy.py
# step-2 analysis (laptop, reads raw JSON only)
python3 bench/p20_moe_roofline.py
```

Bench source: `bench/p20_moe_autopsy.py` (STAGE=1 capture+timing, 3 dequant arm,
4 balanced test). Real routing chunk built from the phase-10 production trace
`p30_exl3_trace_clamp.json` (531 tok x 40 layers, clamped silu); 2048 tokens
drawn with replacement from layer-0 records x top-6 = 12288 pairs/rank.
Calibrated in-session ceilings: dense bf16 `mx.matmul` 15.15 TF, triad BW 431 GB/s.
Timed regions all end in `mx.eval` drain; warmups drained; medians over >=7 reps.

## Raw artifact inventory (this dir)

- `p20_moe_autopsy_stage1.json` — ceilings + Step-1 capture (real/uniform/skew)
- `p20_moe_autopsy_stage3.json` — Step-3 dequant-excluded arm
- `p20_moe_autopsy_stage4.json` — Step-4 uniform/skew/skew-balanced
- `p20_moe_autopsy_analysis.json` — Step-2 roofline + achieved-BW (bench/p20_moe_roofline.py)
- `stage1.log`, `stage34.log` — verbatim node stdout (incl. `[mm_seg]` log lines)
- `p20_moe_instrumentation.patch` — submodule diff for the log line

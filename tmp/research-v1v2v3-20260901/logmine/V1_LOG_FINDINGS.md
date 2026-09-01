# V1 LOG MINING — GPU / memory / restart forensics for the P13/P14/P15 boot windows (Aug 31 2026)

**Scope:** Aug 31 2026 rotated logs on both nodes, covering the P13 (20:42), P14 Arm A (21:25), P14 Arm B (22:04), and P15 (22:54) boots. All counts are from the decompressed rotated `.zst` logs (the boot-attributable, timestamped logs). The live `exo.log` covers Sep 1 only and is excluded.

**Boot-to-log mapping** (from `docs/PERFORMANCE_HISTORY.md` pre-flight PIDs):
- P13 → `exo.2026-08-31_20-42-00` (node1) / `...20-42-02` (node2)
- P14 Arm A → `...21-25-50` / `...21-25-51`
- P14 Arm B → `...22-04-39` / `...22-04-40`
- P15 → `...22-54-04` / `...22-54-05`

---

## a. Metal / IOGPU warnings or errors

**count = 0 in ALL 8 Aug 31 rotated logs (both nodes, all 4 boots).**

Command: `grep -cE "Metal|IOGPU|IOGPUDevice|MTL|Metal API|GPU" <log>` → **0** for every Aug 31 file.

The only Metal/GPU hits anywhere in the retained logs are in the **Sep 1** log (`...00-09-25` / `...00-09-26`), where 184 hits are all **Hermes campaign prompt text** (the word "GPU"/"Metal" in task descriptions), not real log lines — verified 0 of them are timestamped log lines. **No Metal/IOGPU driver warnings or errors occurred during the P13/P14/P15 boots.**

---

## b. Jetsam events, memory-pressure events, low-memory warnings

**count = 0 for all genuine Jetsam/memory-pressure keywords in ALL 8 Aug 31 rotated logs.**

Command: `grep -cE "jetsam|Jetsam|memory.?pressure|low.?memory|memory warning|Memory pressure|killed by|OOM|out of memory" <log>` → **0** for every Aug 31 file.

The only memory-related lines in the Aug 31 logs are **wired-limit INFO lines** (not pressure events), e.g. verbatim from node1 P15 window:
```
[ 2026-08-31 22:54:19.375 | INFO     | exo.worker.engines.mlx.utils_mlx:set_wired_limit_for_model:1742 ] Wired limit set to 112.30 GiB GiB.
[ 2026-08-31 23:23:53.508 | INFO     | exo.worker.engines.mlx.utils_mlx:set_wired_limit_for_model:1742 ] Wired limit set to 112.30 GiB GiB.
```
(2 per boot at model load, plus 2 more in the P15 window at the 23:23:53 restart — the second pair is the post-restart reload.) **No Jetsam kills, no memory-pressure events, no low-memory warnings occurred during P13/P14/P15.**

---

## c. Runner-restart events

### c1. The P15 BatchPoolingCache crash + auto-restart — CONFIRMED on both nodes

**Ground truth from the campaign doc is reproduced exactly.** The crash is `ValueError: [reshape] Cannot reshape array of size 1 into shape (2,1,1,1)` at `mlx_lm/models/cache.py:2050` in `fetch_overlap_carry`, at **23:23:46** on both nodes.

**node1** (`exo.2026-08-31_22-54-04_689174.log`), verbatim:
```
[ 2026-08-31 23:23:46.632 | WARNING  | exo.worker.runner.bootstrap:entrypoint:368 ] Runner 83960858-499e-430b-a395-2bbac718dbcd crashed with critical exception [reshape] Cannot reshape array of size 1
...
  File "/Users/adam.durham/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/cache.py", line 2050, in fetch_overlap_carry
    valid = mx.array(self._overlap_carry_valid).reshape(batch_size, 1, 1, 1)
ValueError: [reshape] Cannot reshape array of size 1 into shape (2,1,1,1).
[ 2026-08-31 23:23:47.664 | ERROR    | exo.worker.runner.supervisor:_check_runner:632 ] Runner terminated with exitcode=0
Runner error: ValueError: [reshape] Cannot reshape array of size 1 into shape (2,1,1,1).
```

**node2** (`exo.2026-08-31_22-54-05_862579.log`), verbatim:
```
[ 2026-08-31 23:23:46.634 | WARNING  | exo.worker.runner.bootstrap:entrypoint:368 ] Runner 72bb5761-5b08-45e2-ab64-3f3953d14e68 crashed with critical exception [reshape] Cannot reshape array of size 1
...
[ 2026-08-31 23:23:47.328 | ERROR    | exo.worker.runner.supervisor:_check_runner:632 ] Runner terminated with exitcode=0
Runner error: ValueError: [reshape] Cannot reshape array of size 1 into shape (2,1,1,1).
```

### c2. Auto-restart confirmed by runner lifecycle (PID+lstart-equivalent forensics)

The runner process identity change is visible in the runner lifecycle lines. **node1** (verbatim):
```
[ 2026-08-31 22:54:19.231 | INFO     | exo.worker.runner.runner:__init__:127 ] hello from the runner      <- P15 boot runner (original)
[ 2026-08-31 23:23:46.699 | INFO     | exo.worker.runner.bootstrap:entrypoint:390 ] bye from the runner     <- crash exit
[ 2026-08-31 23:23:47.708 | INFO     | exo.worker.main:plan_step:226 ] Worker plan: CreateRunner           <- supervisor schedules restart
[ 2026-08-31 23:23:50.814 | INFO     | exo.worker.runner.runner:__init__:127 ] hello from the runner      <- restarted runner (attempt 1)
[ 2026-08-31 23:23:50.929 | INFO     | exo.worker.runner.bootstrap:entrypoint:390 ] bye from the runner     <- attempt 1 exits
[ 2026-08-31 23:23:50.977 | INFO     | exo.worker.main:plan_step:226 ] Worker plan: CreateRunner
[ 2026-08-31 23:23:53.178 | INFO     | exo.worker.runner.runner:__init__:127 ] hello from the runner      <- restarted runner (final)
[ 2026-08-31 23:23:53.507 | INFO     | exo.worker.runner.supervisor:start_task:370 ] Starting task LoadModel(...)
```
**node2** (verbatim):
```
[ 2026-08-31 22:54:19.240 | INFO     | exo.worker.runner.runner:__init__:127 ] hello from the runner
[ 2026-08-31 23:23:46.701 | INFO     | exo.worker.runner.bootstrap:entrypoint:390 ] bye from the runner
[ 2026-08-31 23:23:47.470 | INFO     | exo.worker.main:plan_step:226 ] Worker plan: CreateRunner
[ 2026-08-31 23:23:50.574 | INFO     | exo.worker.runner.runner:__init__:127 ] hello from the runner
[ 2026-08-31 23:23:50.717 | INFO     | exo.worker.runner.bootstrap:entrypoint:390 ] bye from the runner
[ 2026-08-31 23:23:50.849 | INFO     | exo.worker.main:plan_step:226 ] Worker plan: CreateRunner
[ 2026-08-31 23:23:53.030 | INFO     | exo.worker.runner.runner:__init__:127 ] hello from the runner
[ 2026-08-31 23:23:53.483 | INFO     | exo.worker.runner.supervisor:start_task:370 ] Starting task LoadModel(...)
```

**Restart timeline (both nodes):** crash 23:23:46 → supervisor `CreateRunner` 23:23:47 → first restart attempt 23:23:50 (exits immediately) → final restart 23:23:53 → `LoadModel` 23:23:53. This matches the campaign doc's "runner children started 23:23:50, restart sequence completed 23:24:32.7" (the 23:23:50 line is the first attempt; the surviving runner is the 23:23:53 one). The runner_id (83960858... / 72bb5761...) is the **model-instance id and is stable across the restart** — the restart is a new *process* with the same instance id, so runner_id alone does not reveal the restart; the lifecycle lines do.

### c3. No other runner crashes in P13/P14A/P14B

**count = 0** for genuine crash markers (`RunnerFailed|runner terminated|Runner terminated|crashed with critical|Runner error`) in the P13, P14A, and P14B logs (both nodes). The only non-zero counts are the P15 window (3 each, all the BatchPoolingCache crash).

The high raw `Traceback` counts in the Aug 31 logs (31–68 per file) are **NOT runner crashes** — they decompose into:
- **`ModelCard` pydantic validation errors** at startup (routine, non-fatal)
- **`Exception: Failed to fetch file list: 404`** — the documented HF-404 loop (`DownloadCoordinator._emit_existing_download_progress`), which the campaign doc already root-caused as unrelated log noise that never interrupted a probe.

---

## Summary table

| Category | P13 | P14A | P14B | P15 | Verdict |
|---|---|---|---|---|---|
| Metal/IOGPU warn/err | 0 | 0 | 0 | 0 | none |
| Jetsam/memory-pressure | 0 | 0 | 0 | 0 | none (only wired-limit INFO) |
| Runner crash | 0 | 0 | 0 | **1** (BatchPoolingCache reshape) | P15 only |
| Runner restart | 0 | 0 | 0 | **1** (auto-restart 23:23:53) | P15 only |

**V1 implication:** The only within-boot process-state perturbation in the P13–P15 window is the **P15 runner crash + restart at 23:23:46–53**, which the campaign doc already flagged as the reason P15's 30.06 was measured on a restarted runner. There is **no Metal/IOGPU error and no memory-pressure/Jetsam event** in any of the four boots to explain the decode variance. The P15 restart is the sole V1-relevant event, and it is already documented.

---

## Appendix — raw evidence files
- `raw/metal_iogpu_aug31.txt` — Metal/IOGPU grep counts (all 0) + note on Sep 1 text hits
- `raw/memory_jetsam_aug31.txt` — memory/jetsam grep counts (all 0) + wired-limit samples
- `raw/p15_crash_node1.txt`, `raw/p15_crash_node2.txt` — P15 crash + restart verbatim

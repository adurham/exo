# Task C — Metal hardware counters via xctrace (bounded attempt) — **SUCCEEDED**

## Result: xctrace CLI WORKS headless on studio2 (no failure to report).

### Exact commands (studio2, against the live bench process pid 23410)
```
cd /tmp/p20_pmp_A && env -u PYTHONPATH <venv>/bin/python p20_pmp_taskD_load.py 60 &   # target pid 23410
xctrace record --template 'Metal System Trace' --time-limit 30s --attach 23410 --output /tmp/x_metal.trace
```
Exact stdout:
```
Starting recording with the Metal System Trace template. Attaching to: python (23410). Time limit: 30.0 s
Ctrl-C to stop the recording
Reached specified time limit, ending recording...
Recording completed. Saving output file...
Output file saved as: x_metal.trace
REC_RC=0
```
Trace: 30.88 s, target `python pid=23410 termination-reason=exit(0)`, device M4 Max
(recommended working set 117.19 GiB), instruments-version 27.0 (27A266a).

`xctrace export --input /tmp/x_metal.trace --toc` → rc=0.  Per-table export rc=0.
(The single 347 MB `.trace` bundle was exported then removed on the node; the
extracted XML tables below are retained.)

### What the trace contains (extracted, raw/C/)
- `metal-gpu-intervals`: 9400 rows. Category sums over the 30.6 s span: dominant
  **"Command Buffer 0:Compute" = 19616 ms** of nested compute intervals → GPU
  compute-busy ≈ 64 % of the window; top-level `GPU Execution` Active 36.17 ms.
- `metal-application-command-buffer-submissions`: **1681** command buffers submitted.
- `metal-command-buffer-completed`: **2377** completed.
- `metal-application-intervals`: 2521 (label prefixes: Command Buffer 1681,
  Compute Command 1, other 839).
- `metal-gpu-execution-points`: 18800.
- `device-gpu-info`: M4 Max / Apple / max-working-set 125829120000 (117.19 GiB) / headless=No.

### Caveat (the real limitation, not a failure)
The default Metal System Trace counter set is `(null)` and **Shader Timeline:
Disabled**.  Consequently:
- `gpu-counter-info` lists only **one** hardware counter ("RT Unit Active");
  `gpu-counter-value` is 11.9 M samples of that one always-~0 counter (2.4 GB —
  not retained).
- `gpu-shader-profiler-sample` / `gpu-aps-stream` / `gpu-performance-device-state-intervals`
  are **empty**.

So xctrace headless gives **command-buffer-level GPU intervals** (submit/complete
counts + nested compute busy time) but **NOT** per-kernel/shader or per-op-class
GPU-time attribution.  For Task-B-style per-op-class attribution the Shader
Timeline must be enabled in an explicit counter set (not achievable with the
plain `--template` CLI).  Not pursued further (bounded attempt, ~15 min used).

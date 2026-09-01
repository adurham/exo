# Passive Telemetry Kit — exo cluster research campaign

Instrument for discriminating **hypothesis (A) boot-level GPU/Metal state** vs
**hypothesis (B) within-boot process state** (MLX allocator pool growth /
fragmentation and runner restarts) during the NEXT cluster launch.

The two highest-value fields are the actual discriminators:
- **`mlx.cache_bytes`** — the MLX allocator pool size. A step-change in decode
  coincident with an allocator-pool growth step supports (B).
- **`runner.processes[].pid` / `.lstart`** — restart detection. A changed
  PID/lstart between checkpoints means the runner restarted.

Decision rule the kit supports: if decode moves **>= 1.5 tok/s within a boot**
coincident with an allocator-pool growth step OR a runner restart → (B)
supported. If decode stays flat within the boot while the boot-level number
differs from prior boots that also had flat telemetry → (A) gains first real
support.

---

## Files

| File | Purpose |
|---|---|
| `collect_telemetry.py` | The kit. One JSON object per invocation, appended to a JSONL file. |
| `_sanity_broken_sampler.py` | Local-only test demonstrating a broken sampler does not abort the run. Not part of the cluster kit. |

---

## How to invoke at each checkpoint

Run with the **exo venv python** (so `mlx` is importable). On each node, at
each checkpoint:

```bash
# T0 — before any load (right after cluster launch, before warmup)
/Users/adam.durham/repos/exo/.venv/bin/python \
  /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/telemetry/collect_telemetry.py \
  T0 --out /Users/adam.durham/telemetry_$(hostname).jsonl

# after warmup
... collect_telemetry.py warmup --out /Users/adam.durham/telemetry_$(hostname).jsonl

# after each benchmark rep
... collect_telemetry.py rep1 --out /Users/adam.durham/telemetry_$(hostname).jsonl
... collect_telemetry.py rep2 --out /Users/adam.durham/telemetry_$(hostname).jsonl
```

Use a **separate output file per node** (the `$(hostname)` suffix does this) so
the two nodes' checkpoints don't interleave in one file. Each invocation
appends one line; a whole run is a single parseable JSONL artifact.

### CLI args

| Arg | Default | Meaning |
|---|---|---|
| `LABEL` (positional, required) | — | Checkpoint label, e.g. `T0`, `warmup`, `rep1`. Stored verbatim in the record. |
| `--out FILE` | `telemetry_<hostname>.jsonl` in cwd | JSONL output file (append mode). |
| `--runner-pattern RE` | built-in set (see below) | Regex for identifying the exo runner process(es). |
| `--powermetrics-interval-ms N` | `200` | powermetrics sample interval in ms. |
| `--powermetrics-samples N` | `2` | Number of powermetrics samples. |

The default runner pattern is
`(-m exo|exo -v|batch_generator|exo\.worker|exo\.main)`. It can be overridden
via `--runner-pattern` or the `EXO_TELEMETRY_RUNNER_PATTERN` env var if the
runner's invocation changes.

---

## Output schema

One JSON object per line. Top-level keys:

| Key | Type | Meaning |
|---|---|---|
| `timestamp` | str (ISO-8601, local tz) | When the checkpoint was taken. |
| `hostname` | str | Node hostname. |
| `label` | str | Caller-supplied checkpoint label. |
| `mlx` | object | MLX GPU memory (sampler a). |
| `powermetrics` | object | GPU clocks / residency / power (sampler b). |
| `memory_pressure` | object | Free/available memory + pressure (sampler c). |
| `wired_limit` | object | `iogpu.wired_limit_mb` sysctl (sampler d). |
| `runner` | object | Runner process identity (sampler e). |
| `elapsed_seconds` | float | Wall time of the whole checkpoint. |

Every sampler object has an `error` field: `null` on success, or a string
describing the failure. **A non-null `error` means that sampler's data is
absent — do not treat missing fields as zero.** Raw command text is retained
under a `raw` key where applicable for auditability.

### `mlx` (a)

| Key | Type | Meaning |
|---|---|---|
| `api` | str | Which MLX API surface was detected: `mx.* (current top-level)` or `mx.metal.* (deprecated)`. |
| `active_bytes` | int | MLX Metal **active** memory (the trustworthy per-process number; excludes the reclaimable pool). |
| `cache_bytes` | int | MLX **allocator cache** size (the reclaimable Metal pool — the ratchet discriminator). |
| `peak_bytes` | int | MLX peak memory. |
| `error` | str\|null | `null` on success. |

### `powermetrics` (b)

| Key | Type | Meaning |
|---|---|---|
| `samples` | list[object] | One object per sample block. |
| `n_samples` | int | Number of parsed sample blocks. |
| `raw` | str | Full raw powermetrics output (auditability). |
| `error` | str\|null | `null` on success. |

Each sample object:

| Key | Type | Meaning |
|---|---|---|
| `gpu_hw_active_residency_pct` | float | GPU HW active residency (power-state metric, not real occupancy — see note below). |
| `gpu_hw_active_freq_mhz` | float | GPU HW active frequency (clock). |
| `gpu_idle_residency_pct` | float | GPU idle residency. |
| `gpu_power_mw` | float | GPU power in milliwatts. |

**Interpretation caveat (from the 2026-08-22 investigation):** `powermetrics`
"active residency" is a power-STATE metric gated by a millisecond hysteresis
threshold — it reads 100% active for bursty low-average-load work that never
power-gates, even when real occupancy is ~30%. **The power draw (Watts) is the
tiebreaker** when residency and power disagree. Treat reduced clock as a
downstream symptom of low-average-load, not an independent root cause.

### `memory_pressure` (c)

| Key | Type | Meaning |
|---|---|---|
| `free_pct` | float | System-wide memory free percentage. |
| `pages_free` | int | Free pages. |
| `pages_active` | int | Active pages. |
| `pages_inactive` | int | Inactive pages. |
| `pages_wired` | int | Wired-down pages. |
| `pages_purgeable` | int | Purgeable pages. |
| `pages_compressor` | int | Pages used by compressor. |
| `page_size_bytes` | int | Page size (16384 on Apple Silicon). |
| `raw` | str | Full raw output. |
| `error` | str\|null | `null` on success. |

Real resident memory ≈ `(pages_wired + pages_active) * page_size_bytes`. The
`free_pct` from `memory_pressure` is the trustworthy system-level number (the
exo dashboard's "used" is the naive `total - free` and over-reads).

### `wired_limit` (d)

| Key | Type | Meaning |
|---|---|---|
| `wired_limit_mb` | int | `iogpu.wired_limit_mb` value. On the cluster this is set to 115000 by `start_cluster.sh`; a value of `0` means the limit is unset (default). |
| `raw` | str | Full raw output. |
| `error` | str\|null | `null` on success. |

### `runner` (e)

| Key | Type | Meaning |
|---|---|---|
| `processes` | list[object] | **All** matching runner processes (not just the first). |
| `count` | int | Number of matches. |
| `pattern` | str | The regex used. |
| `raw` | str | Full raw `ps` output. |
| `error` | str\|null | `null` on success. |

Each process object:

| Key | Type | Meaning |
|---|---|---|
| `pid` | int | Process ID. |
| `lstart` | str | Process start time (`Tue Sep  1 10:00:00 2026`). |
| `comm` | str | Command name (first token). |
| `args` | str | Full command line. |

**Restart detection:** compare `pid`/`lstart` across checkpoints. A changed
PID or lstart between checkpoints means the runner restarted. The pattern
matches the runner's distinctive runtime marker (`-m exo` / `exo -v` /
`batch_generator` / `exo.worker` / `exo.main`) so it identifies the runner
even if the exact process name varies, and records ALL matching processes
(including multiprocessing spawn children).

---

## Design guarantees

- **READ-ONLY and non-perturbing.** It samples; it never changes cluster
  state, never restarts anything, never sets sysctls. The powermetrics window
  is `interval_ms * samples` = 200ms × 2 = ~400ms by default, so a checkpoint
  stays fast.
- **Graceful degradation.** Every sampler catches its own failures and records
  an explicit error string in that field, then continues. A single failing
  sampler never aborts the checkpoint. This is critical — the kit runs
  unattended during an expensive cluster run.
- **No third-party dependencies** beyond stdlib + mlx. If mlx is not
  importable, the `mlx` field records an error and everything else still
  works.
- **MLX API version detection.** The MLX memory API moved across versions
  (`mx.metal.get_active_memory/get_cache_memory/get_peak_memory` → newer
  `mx.get_active_memory/get_cache_memory/get_peak_memory`). The kit detects
  both at runtime and prefers the current top-level form. It does not hardcode
  a guess.

---

## UNTESTED ON CLUSTER

This kit was **written and locally sanity-checked on `adams-macbook-pro-m4`
(a MacBook Pro, NOT a cluster node)**. It has **NOT been executed on the
cluster nodes** and **NOT been ssh'd to the nodes**. The following is an honest
account of what was and was not verified.

### Locally verified (on the MacBook Pro)

| Sampler | Locally verified? | How |
|---|---|---|
| (a) MLX memory | **YES** | Ran against the repo venv's mlx 0.32.0.dev. Confirmed both API surfaces exist; the top-level `mx.*` form is current and `mx.metal.*` is deprecated (emits a deprecation warning). The kit correctly detects and uses `mx.* (current top-level)`. Values were 0 (idle, no model loaded) — the **parsing/API-detection** is verified, but **real non-zero values** were not observed locally. |
| (b) powermetrics | **PARTIAL** | Parser validated against sample text in the exact field format used by the validated `bench/section100_gpu_ground_truth.py` parser (same regexes: `GPU HW active residency`, `GPU HW active frequency`, `GPU idle residency`, `GPU Power`), plus the documented real cluster values (1578 MHz / 55-57W and 819-1122 MHz / 4.6-7.1W). **NOT run against live powermetrics output** — `sudo -n powermetrics` requires a password on this laptop (NOPASSWD is only granted on the NODES), so the live command path was not exercised. The password-gated failure path WAS exercised (see below). |
| (c) memory_pressure | **YES** | Ran the real command locally; parser validated against real captured output (free_pct, pages_free, pages_active, etc. all parsed correctly). |
| (d) wired_limit sysctl | **YES** | Ran the real command locally; parser validated against real output. Value was `0` (unset on this laptop) — the **parsing** is verified, but the **cluster value (115000)** was not observed. |
| (e) runner process | **YES** | Ran the real `ps` command locally; parser validated against real captured output. Correctly identified 0 runner processes (no exo runner on this laptop) and correctly rejected a decoy. Also validated against a synthetic exo runner line (`.venv/bin/python -m exo -v` + a `batch_generator` spawn child) — both were identified, the decoy was skipped. |

### Graceful-degradation path (verified)

- **Password-gated powermetrics:** exercised locally — `sudo -n powermetrics`
  fails with "a password is required", and the kit records an explicit error
  in the `powermetrics` field while the rest of the checkpoint emits real
  values. This is exactly the failure mode the kit must survive on the nodes
  if the NOPASSWD rule is ever missing.
- **Deliberately-broken sampler:** `_sanity_broken_sampler.py` points the
  memory_pressure sampler at a nonexistent binary; the checkpoint still emits
  with an error field for that key and real values elsewhere. Verified.

### NOT verified (requires the cluster)

- **Live powermetrics output** on the actual M4 Max nodes (real field values,
  real sample-block structure, real timing).
- **Real non-zero MLX memory values** (active/cache/peak under a loaded model).
- **The cluster's `iogpu.wired_limit_mb` value (115000)**.
- **The actual runner process signature** on the nodes (the default pattern
  is based on the documented `.venv/bin/python -m exo -v` launch; if the
  runner's invocation differs, `--runner-pattern` / `EXO_TELEMETRY_RUNNER_PATTERN`
  must be adjusted).
- **End-to-end on the nodes** (both nodes writing separate JSONL files, the
  full checkpoint sequence T0 → warmup → repN).

### Known caveats to check on first cluster run

1. **Runner pattern false-positive risk.** The default pattern deliberately
   excludes a bare `repos/exo/.venv/bin/python` path match because a shell
   wrapper that merely invokes that python would match (observed locally) and
   its lstart changes on every new shell, corrupting restart detection. If the
   runner's actual command line does not contain `-m exo` / `exo -v` /
   `batch_generator` / `exo.worker` / `exo.main`, the runner will not be
   detected — verify the pattern against a real `ps` on the node before
   trusting the `runner` field.
2. **powermetrics field names** are from the validated `section100_gpu_ground_truth.py`
   parser and the 2026-08-22 investigation. If a future macOS/powermetrics
   version renames fields, the parser will return `n_samples: 0` and an error —
   the raw text is retained for manual inspection.
3. **`memory_pressure` free_pct** is the trustworthy system-level number; the
   dashboard's "used" over-reads. Real resident = wired + active.

---

## Local sanity-check evidence

- `python3 -m py_compile collect_telemetry.py` → **exit 0**.
- Full kit run locally (T0, warmup, rep1) → valid JSONL, each line parsed back
  with `json.loads`.
- Parsers for memory_pressure, sysctl, and ps validated against **real local
  command output**. The powermetrics parser was validated against **sample
  text in the documented field format** (not live output — see above).
- `_sanity_broken_sampler.py` demonstrates a broken sampler does not abort the
  run.

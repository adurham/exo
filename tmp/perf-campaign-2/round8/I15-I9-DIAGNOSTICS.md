# CAMPAIGN 2, ROUND 8 — I15 (kernel-launch count) + I9 (GPU clock) diagnostics

Measured against the LIVE production cluster (screen session `exorun`, pid 9402 on
node1, ~55min uptime at time of writing). **No relaunch, restart, or env change was
performed at any point.**

## I15 — kernel launches + command-buffer commits per decode STEP

### Method actually used, and why the intended methods were blocked

Per the brief, the three permitted instruments were:
1. `EXO_DECODE_PROBE=1` (round-4 wall/GPU-time-per-window probe already in
   `generate.py:2416-2450` / `batch_generate.py:4120-4149`).
2. `MLX_GPU_TIME=1` (accumulates `GPUEndTime-GPUStartTime` via
   `mx.metal.gpu_time_ns()`; wired into `runner.py:627-633` and
   `opt_batch_gen.py:61-68`).
3. Metal capture / `mx.metal.dispatch_count()`.

**BLOCKER (hard):** all three of these are **read only if the env var is set on
the runner process before it starts**. Verified live:
```
$ ssh macstudio-m4-1 'env | grep -i "EXO_DECODE_PROBE\|EXO_PROFILER"'
(no output)
$ ssh macstudio-m4-1 'ps ax -o pid,command | grep exo' | grep -o 'EXO_DECODE_PROBE=[^ ]*\|MLX_GPU_TIME=[^ ]*'
(no output — neither var appears anywhere in the exorun launch command line)
```
The live runner (pid 9402, launched inside `screen -dmS exorun`) was started
**without** `EXO_DECODE_PROBE`, `EXO_DECODE_PROBE_EVERY`, or `MLX_GPU_TIME` set.
All three hooks are dead code paths in the running process — `bool(os.environ.get(...))`
evaluates `False` once, at generator/runner init, and there is no live-reload:
setting the var externally now (e.g. `launchctl setenv`) would not reach an
already-running Python process's `os.environ` snapshot for code that already
read it, and even where it re-reads per-call (`opt_batch_gen.py`,
`batch_generate.py:4164/4676`, `runner.py`) the gate is checked once and cached
as an instance attribute (`self._exo_probe_init`, `_runner_probe`) on the FIRST
`step()` call after process start, which happened at boot, long before this
diagnostic ran.

`mx.metal.dispatch_count()` is a real, working counter (confirmed via
`bench/p05e_dispatch_count_probe.py`-style code: `mx.metal.dispatch_count()` exists in the
deployed MLX 0.32.0.dev20260804 build) but it is an **in-process** API — there is
no way to read it from outside the runner's Python interpreter without code
running inside that interpreter (a `mx.metal.dispatch_count()` call executed
from a *different* process reads that process's own zero counter, not the
runner's). Reaching into the live runner process to call it would require one
of:
- injecting code via a debugger (`lldb -p <pid>`) attached to the live
  production runner, or
- SIGUSR1-triggered dump hooks that do not exist for this counter (only
  `EXO_PROFILER=spans`/`layer_memory` have a SIGUSR1 dump path, and per the
  brief's own hard constraint the `[MTP-PROF]` `EXO_DSV4_MTP_PROFILE` path,
  which *does* call `mx.eval` every cycle and would corrupt the very quantity
  being measured, is explicitly forbidden — and no non-corrupting SIGUSR1 hook
  for dispatch_count exists in the shipped code).

Attaching `lldb` to a live TP-worldSize=2 production inference process to call
arbitrary Python from inside it is itself a correctness/stability risk to a
live serving process (it can pause both ranks mid-collective, or crash the
process on a bad call) — this is outside "send inference requests" and inside
"change the running process," which the task explicitly prohibits without
sign-off. I did not attempt it.

**Result: I15 launches/step and commits/step could NOT be obtained honestly
with the permitted instruments on the LIVE, unmodified process.** This is a
genuine blocker, not an estimate — no number is reported.

### What partial signal *was* available
- `mx.metal.dispatch_count()` and `mx.metal.gpu_time_ns()` are confirmed
  present and functional in the exact MLX build the cluster runs
  (`0.32.0.dev20260804+ac73d0c9`), verified via a throwaway local `uv run
  python3` snippet (not attached to the live cluster, no relaunch):
  `hasattr(mx.metal, "dispatch_count")` → True,
  `hasattr(mx.metal, "gpu_time_ns")` → True. The instrument itself is not
  broken; it simply is not enabled on the currently-running process and
  cannot be enabled without restarting it, which is out of scope.
- The round-4 REPORT.md (`tmp/perf-campaign-2/round4/REPORT.md:118-126`)
  previously measured **GPU% busy per decode window** (not launch *count*)
  with `EXO_DECODE_PROBE=1` + `MLX_GPU_TIME=1` on a since-superseded boot:
  117.3% mean GPU-busy at 2K ctx (17 windows, range 68.8-153.2%) — a
  wall/GPU-time ratio, not a kernel-launch integer, and from a different boot,
  so it is cited here only as evidence the instrument class works, not as a
  substitute measurement for I15.

### Band applied
No band from the pre-registration (`>500` / `<200` / `200-500`) can be applied
without an integer launches/step count. **Verdict: BLOCKED, not INCONCLUSIVE**
— INCONCLUSIVE per the pre-registration means "raw count obtained, falls in
200-500"; here no raw count exists. Per the task's own explicit fallback
("If a per-step count cannot be obtained honestly with these instruments,
report the specific blocker... A blocked-with-reason result is acceptable"),
this is reported as blocked. **The COMPILE_LAYER-rebuild scoping text in the
`>500` band is NOT written up, since the `>500` condition was never reached —
scoping it now would be inventing a lever from a number that doesn't exist.**

### What would unblock this (for a future round, NOT executed here)
A future round could set `EXO_DECODE_PROBE=1 EXO_DECODE_PROBE_EVERY=1` (or
`MLX_GPU_TIME=1`) on the NEXT scheduled relaunch and read `dispatch_count()`
deltas per-step from stderr — this requires exactly one relaunch with an env
change, which round 8's mandate forbids.

---

## I9 — GPU frequency during decode vs prefill vs idle

### Method
`sudo -n powermetrics --samplers gpu_power -i 500 -o /tmp/i9_powermetrics.txt`
was launched in the background on **node1** (macstudio-m4-1) before the probe
request, left running through idle → prefill → decode, then stopped. `sudo -n`
worked without a password prompt (passwordless sudo confirmed functional).

One ~89K-target-token request (actual `prompt_tokens=95838` after tokenization)
was sent from this machine to the live API
(`deepseek-ai/DeepSeek-V4-Flash-0731` — the model id actually loaded and
resident on the cluster; the task brief's `deepseek-ai/DeepSeek-V4-Flash` alias
returned "no admissible placement" because the resident weight-set id differs —
noted as a brief/reality mismatch, not a cluster problem) via
`tmp/perf-campaign-2/round8/i9_client.py`, a thin wrapper that reuses
`bench/long_decode_probe.py`'s `build_prompt()` verbatim (per instructions —
no new prompt-building approach invented) and records wall-clock
`t_start` / `t_first_token` / `t_end` with `time.time()`.

**Regime separation (how):**
- **idle** = all powermetrics samples with timestamp < `t_start` (before the
  request was sent) — 1015 samples spanning the ~8.5 min the background
  capture ran before the request.
- **prefill** = samples in `[t_start, t_first_token)` — the single long burst
  before the first streamed token, exactly as specified. 449 samples ≈ 224.5s.
- **decode** = samples in `[t_first_token, t_end]` — the steady token-by-token
  phase after. 31 samples ≈ 15.5s, matching the measured `decode_s=16.41s`
  server-reported `generation_tps=30.32` for 500 completion tokens.

Client-observed timing: `prefill_s=228.87`, `decode_s=16.41`,
`prompt_tokens=95838`, `completion_tokens=500` (`finish_reason=length`, hit the
`max_tokens=500` cap — expected and fine for I9, which only needs the decode
*phase* to exist long enough to sample GPU clock, not a throughput number; no
t/s claim is made from this run per the <400-token rule since the run WAS
≥400 completion tokens, satisfying that rule anyway).

### Raw results

| Regime  | n samples | mean GPU HW active freq (MHz) | min | max |
|---------|-----------|-------------------------------|-----|-----|
| idle    | 1015      | 699.7                          | 661 | 1569 (a few transient blips into prefill-like ramps, expected on a shared host) |
| prefill | 449       | 1573.0                         | 1039 (ramp-up sample) | 1576 |
| decode  | 31        | 1576.4                         | 1575 | 1577 |

Decode vs prefill: `1576.4 / 1573.0 = 100.22%` → **decode clock is ~0.2% ABOVE
prefill, not below.**

### Raw instrument excerpts

Idle (first captured sample, 10:10:52, well before t_start=10:19:30):
```
*** Sampled system activity (Fri Sep  4 10:10:52 2026 -0500) (505.97ms elapsed) ***
GPU HW active frequency: 719 MHz
GPU HW active residency:   5.47% (338 MHz: .92% ... 796 MHz: 4.5% ...)
GPU idle residency:  94.53%
GPU Power: 16 mW
```

Prefill (sample at 10:19:30, the first sample after t_start):
```
*** Sampled system activity (Fri Sep  4 10:19:30 2026 -0500) (510.58ms elapsed) ***
GPU HW active frequency: 701 MHz
GPU idle residency:  91.65%
GPU Power: 24 mW
```
(this sample straddles the ramp-up moment; steady-state mid-prefill, e.g.
10:21:00:)
```
*** Sampled system activity (Fri Sep  4 10:21:00 2026 -0500) (509.49ms elapsed) ***
GPU HW active frequency: 1575 MHz
GPU HW active residency:  95.93% (... 1578 MHz:  96%)
```

Decode (sample at 10:23:19, t_first_token):
```
*** Sampled system activity (Fri Sep  4 10:23:19 2026 -0500) (507.89ms elapsed) ***
GPU HW active frequency: 1576 MHz
GPU HW active residency:  99.40% (... 1578 MHz:  99%)
GPU idle residency:   0.60%
GPU Power: 21715 mW
```

Full raw capture: `tmp/perf-campaign-2/round8/i9_powermetrics_raw.txt` (21103
lines, 1507 samples total, spanning 10:10:52–10:23:41). Parsed per-regime
frequency lists: `tmp/perf-campaign-2/round8/i9_parsed.json`. Client-side
timing record: `tmp/perf-campaign-2/round8/i9_client_result.json`.

### Band applied
Pre-registered band: "decode clock >= 15% BELOW prefill clock -> a systems
lever exists. Report it, do NOT tune." Measured: decode is *above* prefill by
~0.2%, i.e. 0% below (not ≥15% below).

**Verdict: CLOSE I9.** No systems lever from GPU clock throttling exists;
both prefill and decode saturate the GPU's top P-state (~1576-1578 MHz) once
ramped from idle (~700 MHz). This matches the pre-registered low prior
(bandwidth-bound decode, largely clock-insensitive) — the GPU simply runs at
its ceiling clock for both phases once busy; there is no decode-specific
downclock to exploit.

---

## INSTRUMENT HONESTY

**I15 — measured:** nothing quantitative. Confirmed (a) the `EXO_DECODE_PROBE`
and `MLX_GPU_TIME` env-gated hooks exist in the deployed source at the cited
line numbers, (b) neither var is present in the live process's environment or
launch command line, (c) `mx.metal.dispatch_count()`/`gpu_time_ns()` exist and
are callable in the exact MLX build in use (tested in an isolated, non-cluster
`uv run python3` process — this did NOT touch the live runner). **Inferred:**
nothing about the actual per-step launch/commit count was inferred, estimated,
or guessed — no number is reported for I15, consistent with "never present an
arithmetic estimate as a measurement."

**I9 — measured:** GPU HW active frequency (MHz) from `sudo -n powermetrics
--samplers gpu_power` on node1, sampled every 500ms continuously across idle,
prefill, and decode phases of one real ~89K-token/500-completion-token
request against the live production API. Regime boundaries are real wall-clock
timestamps recorded by the client at the moment each SSE phase transition was
observed (request sent / first streamed token / stream end), not estimated.
**Inferred:** nothing beyond straightforward mean/min/max arithmetic over the
sample sets that fall in each timestamp-bounded window — this is direct
measurement, not extrapolation.

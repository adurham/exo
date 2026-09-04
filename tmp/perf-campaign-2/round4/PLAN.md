# CAMPAIGN 2, ROUND 4 — pre-registered experiment plan (written BEFORE any relaunch)

Date: 2026-09-03. Bands are the task file's own, applied verbatim. This file is
the pre-registration: no post-hoc bars. (Post-consult refinements are marked
[CONSULT] — methodology hardening only, bands untouched.)

## Q1 — armed ratio (live gate diagnostic)
- Method: relaunch with `EXO_DSV4_FENCE_GATE_DIAG=1` (already pass-through in
  start_cluster.sh:1770). Import-time on the runner -> one relaunch.
- Workload: real 89K-depth needle request (bench/quality_probe_dsv4.py
  build_prompt at ~89K target) + decode_probe runs. Read `[fence-gate-diag]`
  lines from BOTH nodes' ~/exo.log. [CONSULT] Report both layer-level and
  forward-level ratios; "forward armed" = zero fallback lines across all
  layers of that forward. Gate strictly on decode-window timestamps (prefill
  and transitions emit lines by design). Denominator from decode-forward
  counters, never inferred from tokens (MTP: >1 forwards per token).
  Confirm the diag does not itself force a sync (it prints shapes/env only).
- **Pre-registered band:** armed >=95% of forwards -> Q1 CLEAN, go to Q2.
  <95% -> regression of the 08-22 fix under MTP-on; root-cause, fix at the
  gate fail-closed registration-based, validate with the 08-22 three-check
  battery (needle exact-match + two more from the validation doc).

## Q2 — falsification probes (the number round 2 never got)
- **Probe A (GPU-identity all_sum):** site-specific env-gated probe
  (mlx-lm submodule commit 7f14654, `EXO_DSV4_ALLSUM_IDENTITY_PROBE=1`,
  loud one-time log, NEVER in start_cluster defaults) replacing ONLY the MoE
  per-layer all_sum. The pre-existing 2026-05-13 file-based NOP
  (`/tmp/dsv4_nop_targets`) is REJECTED as the A/B vehicle: it patches
  GLOBAL mx.distributed.all_sum — it would also NOP the DSpark agree gate
  (utils_mlx.py:459) and has_work coord collectives, changing the workload
  itself (DSpark would detach every cycle), so its A/B is invalid.
  [CONSULT] Identity output shifts MTP/DSpark acceptance rates, so tok/s is
  confounded; the primary metric is the PER-FORWARD all_sum-site bracket
  (EXO_DSV4_ALLSUM_PROBE per-layer ms) + per-window decode wall/GPU% —
  NOT tok/s. Report tok/s as secondary context only.
- **Probe B (command-buffer GPUStart/GPUEndTime):** already exists in the
  installed stack — `MLX_GPU_TIME=1` + `EXO_DECODE_PROBE=1` yield per-window
  GPU%-busy from Metal's own GPUStartTime/GPUEndTime. Baseline leg runs on
  relaunch-1; identity leg on relaunch-2. GPU-idle share during decode
  windows = direct falsification of "the drain is compute".
- **Pre-registered band (Probe A):** removes >=40% of verify share ->
  the handoff mechanism (not compute) owns the stall, and Q3 is funded.
  <15% -> the stall is genuine local compute; decode thread CLOSED at model
  level. 15-40% -> report.

## Q3 — fence cadence (only if mechanism owns it)
- **KILLING FACT (pre-registration):** `EXO_DSV4_FENCE_EVERY_N_LAYERS` is a
  DEAD KNOB. `_fence_every_n` assigned at deepseek_v4.py:2958, ZERO read
  sites tree-wide (verified grep across submodule+src+installed; the only
  reader was OPT-7, reverted in `19a07b3` for -23% prefill). With the async
  fence armed, NO per-layer blocking eval runs at c=1 decode at all, so
  there is no cadence to modulate. [CONSULT] Report as "not executable in
  this stack", not "impossible". Rebuilding OPT-7 to run the ABAB would
  re-introduce the documented -23% prefill regression on a mechanism the
  async fence already supersedes — refused.
- Honest substitute (UNREGISTERED/exploratory, labeled as such): FENCE_ASYNC
  on/off cross-boot ABAB prices the actual remaining lever. relaunch-1 is
  the ON arm; if budget permits ONE more boot, FENCE_ASYNC=0 gives A/B n=1.

## Constraints applied
- All relaunches authorized by task; leave cluster HEALTHY on shipped config.
- mx.eval before every bracket close; within-boot ratios for attribution;
  ABAB + ranges cross-boot; never a single-boot delta as evidence.
- NO pushes (local commits only). Probe A impossible-to-leave-on: env-gated
  + loud log + never in start_cluster.sh defaults (deployed via explicit
  per-run env only).
- Env vars reach the runner via EXO_ENV pass-through lines (existing:
  FENCE_GATE_DIAG:1770, MLX_GPU_TIME:1639, EXO_DECODE_PROBE:1640-41,
  ALLSUM_PROBE:1743; NEW: one ALLSUM_IDENTITY_PROBE pass-through line
  added for relaunch-2 only — never a default).
#!/usr/bin/env bash
# =============================================================================
# ROUND 3 — SDPA two-length per-call timing RUNBOOK (PREP-only artifact)
# Owner: exo prefill-tuning campaign
# Git HEAD this targets: 17d427b01 (see findings/b-sdpa-2length-prep.md)
#
# THIS SCRIPT IS A COOKBOOK. It is NOT run by this subagent. It ships as a
# deliverable so an APPROVED future execution can copy-paste it verbatim.
#
# What it runs:
#   * exactly TWO single-context-length arms: 12K and 64K
#   * SDPA per-call timing via the EXO_DSV4_SDPA_CALL_PROFILE=1 gate added by
#     artifacts/sdpa_2length_timing.patch (delivered as a PATCH — apply it to
#     the tree BEFORE this run; it is NOT applied by default).
#   * The BATCHED prefill path ONLY (prefill_batched, generate.py:1269, called
#     from batch_generate.py:3068). We reject the run if the log does NOT show
#     "Starting batched prefill:".
#   * NO SIGUSR1 (a mistimed SIGUSR1 crashed a rank in a prior session). We let
#     each request finish and read the [SDPA-CALL] auto-emissions from ~/exo.log.
#
# THE GATE MUST REACH THE RUNNER. EXO_DSV4_SDPA_CALL_PROFILE is a NEW env var;
# start_cluster.sh must allow-list it (see "ENV ALLOW-LIST" below + patch) or it
# will be stripped from the runner's environment and the probe silently no-ops.
#
# =============================================================================

set -euo pipefail

# --- Required context --------------------------------------------------------
EXO_ROOT="${EXO_ROOT:-$HOME/repos/exo}"
LOG_PATH="${LOG_PATH:-$HOME/exo.log}"        # runner stderr lands here
API="${API:-http://192.168.86.201:52415}"    # API node (from start_cluster/ab_probe)
PROBE="${PROBE:-$EXO_ROOT/bench/ab_probe_tier1.py}"

# --- Shared env for BOTH arms (standing cluster baseline) --------------------
# Keep STEP_SIZE at the standing default in BOTH arms: the per-call ratio at a
# given length is LOCAL-vs-SPARSE (2048-vs-1024 per-rank rows within one prefill),
# so we do NOT vary STEP_SIZE and do NOT build a sweep. Rationale:
#   Every 2048-token chunk fires, in the SAME `attn.sdpa` span (deepseek_v4.py:4865):
#     * 21 SparseCompressedAttention layers  -> banded rows L_q = 1024 (seq-split)
#     *  2 LocalAttention layers             -> full rows  L_q = 2048 (NO seq-split)
#   At identical depth, per-rank KV/pool are the same. So the per-call ratio
#   R(L) = mean(local_ms) / mean(sparse_ms) is the same 2048-vs-1024 row doubling
#   the round-2 4.06x came from, measured WITHOUT any step-size sweep. At 64K the
#   pooled KV is ~5x deeper, so a FIXED per-call overhead becomes a smaller
#   fraction of each call -> R drops toward ~2-3 if it's overhead, stays ~4 if it
#   is a real multiplicative constant.
COMMON_ENV=(
  "EXO_DSV4_SDPA_CALL_PROFILE=1"            # THE probe toggle (this round)
  "EXO_PREFILL_STEP_SIZE=2048"              # standing default (start_cluster.sh:88)
  "EXO_DSV4_SEQ_SPLIT=1"                    # standing default (start_cluster.sh:124)
  "EXO_DSV4_SEQSPLIT_BALANCED=1"            # standing default (start_cluster.sh:113)
  "MLX_JACCL_DATA_RECV_POOL=0"              # standing baseline
  "DSV4_SHARDING=Tensor"                    # TP for both prefill & decode (standing)
  "EXO_PROFILER=spans"                      # optional: span attribution of the same run
  "EXO_PROFILER_LEVEL=1"
)

# === GOTCHA #1 — profiler sync MUST be paired with a raised watchdog =========
# If you enable EXO_PROFILER_SYNC_SPANS=1 you MUST ALSO export
#   EXO_RUNNER_HANG_TIMEOUT_SECONDS=600
# (default 45s watchdog in supervisor.py SIGKILLs the runner if the gap between
# progress-callback events exceeds it; sync-mode serializes spans so the gap can
# blow past 45s). We recommend NOT enabling sync here — our probe does its own
# explicit mx.eval(out), so sync-spans is redundant and only slows the run.
# If you insist on sync spans, set BOTH of:
#   export EXO_PROFILER_SYNC_SPANS=1
#   export EXO_RUNNER_HANG_TIMEOUT_SECONDS=600
: "${EXO_RUNNER_HANG_TIMEOUT_SECONDS:=600}"   # safest default; harmless either way

# -----------------------------------------------------------------------------
# ARM 1 — 12K context
# -----------------------------------------------------------------------------
run_arm_12k() {
  echo "[RUNBOOK] ARM 12K — target_tokens=12000"
  env "${COMMON_ENV[@]}" \
      EXO_RUNNER_HANG_TIMEOUT_SECONDS="$EXO_RUNNER_HANG_TIMEOUT_SECONDS" \
      python3 "$PROBE" 12000 --tag sdpa2len-12k
  echo "[RUNBOOK] ARM 12K complete. Grep the log:"
  echo "    grep '\[SDPA-CALL\]' \"$LOG_PATH\""
  verify_batched_path
}

# -----------------------------------------------------------------------------
# ARM 2 — 64K context
# -----------------------------------------------------------------------------
run_arm_64k() {
  echo "[RUNBOOK] ARM 64K — target_tokens=64000"
  env "${COMMON_ENV[@]}" \
      EXO_RUNNER_HANG_TIMEOUT_SECONDS="$EXO_RUNNER_HANG_TIMEOUT_SECONDS" \
      python3 "$PROBE" 64000 --tag sdpa2len-64k
  echo "[RUNBOOK] ARM 64K complete. Grep the log:"
  echo "    grep '\[SDPA-CALL\]' \"$LOG_PATH\""
  verify_batched_path
}

# --- GOTCHA #2 — only accept the BATCHED prefill path -------------------------
# prefill_batched prints "Starting batched prefill:" (generate.py:1400). The
# eager stream_generate fallback does NOT — if we don't see it, the numbers were
# produced by the wrong path and are invalid (per round-2 discipline).
verify_batched_path() {
  if grep -q "Starting batched prefill:" "$LOG_PATH"; then
    echo "[RUNBOOK] OK: batched prefill path confirmed."
  else
    echo "[RUNBOOK] ERROR: 'Starting batched prefill:' NOT found in $LOG_PATH." \
         "The run likely fell back to eager stream_generate. Numbers INVALID." >&2
    return 1
  fi
}

# =============================================================================
# USAGE:
#   1) Apply the instrumentation patch (APPROVED execution only):
#        cd "$EXO_ROOT"
#        git apply tmp/prefill-round3-20260902/artifacts/sdpa_2length_timing.patch
#        git add mlx-lm/mlx_lm/models/deepseek_v4.py && git commit -m "round3 sdpa probe" && git push origin
#        # then re-pull/reinstall the mlx-lm submodule on BOTH cluster nodes and relaunch
#        # (exo-source-development: a git reset is not enough if uv installs from a remote git source)
#
#   2) ENV ALLOW-LIST (start_cluster.sh) — add this line to the DSv4 block so the
#      NEW var reaches the runner; otherwise it is stripped and the probe no-ops:
#          [ -n "${EXO_DSV4_SDPA_CALL_PROFILE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_SDPA_CALL_PROFILE=$EXO_DSV4_SDPA_CALL_PROFILE"
#
#   3) Run both arms (this script, or just the two functions).
#
#   4) Collect: grep '\[SDPA-CALL\]' ~/exo.log > /tmp/sdpa_2len.log
#      (pull from BOTH nodes; the probe runs on both ranks — dedupe by L/tag in
#      the analyzer, rank-0 and rank-1 see the same L_q shape under TP).
#
#   5) Analyze:
#        python3 tmp/prefill-round3-20260902/artifacts/sdpa_2length_analyze.py \
#          --log /tmp/sdpa_2len.log \
#          --prefill-wall-sec 30   # per-arm wall clock (reductio input)
# =============================================================================
"${@:-run_arm_12k run_arm_64k}"

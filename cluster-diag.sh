#!/bin/bash
# cluster-diag.sh — READ-ONLY diagnostic dispatch for the exo Mac Studio
# cluster, safe to allowlist for unattended/background subagents.
#
# WHY THIS EXISTS: background PM subagents have no interactive approval
# surface, so any ssh/curl to the cluster's private IPs gets hard-denied by
# Hermes' smart-approval SSRF classifier with no appeal path. This script
# pins BOTH the destination hosts AND the remote command surface to a fixed,
# auditable, read-only set — no free-form remote command interpolation, no
# writes, no process control, no credential/secret access. Only THIS
# script's own literal path should ever be allowlisted in
# ~/.hermes/config.yaml command_allowlist, never a wildcard ssh/curl glob
# (a glob matches on substring, not destination, and doesn't constrain the
# remote payload at all — see FORK.md / session history 2026-09-04).
#
# Usage:
#   ./cluster-diag.sh health <m4-1|m4-2>      # API health via curl, that node
#   ./cluster-diag.sh env <m4-1|m4-2> <VAR>   # grep VAR out of running exo PIDs' env
#   ./cluster-diag.sh sha <m4-1|m4-2>         # git rev-parse HEAD in ~/repos/exo
#   ./cluster-diag.sh ps <m4-1|m4-2>          # list running exo python PIDs
#   ./cluster-diag.sh gpu <m4-1|m4-2>         # 2s powermetrics GPU power sample
#
# Adding a new subcommand requires editing this file (a visible, reviewable
# diff) — never extend by accepting a raw command string as an argument.
set -euo pipefail

NODE1_HOST="adams-mac-studio-m4-1.local"
NODE2_HOST="adams-mac-studio-m4-2.local"
NODE1_IP="192.168.86.201"
API_PORT=52415
SSH_USER="adam.durham"
SSH_OPTS=(-o ConnectTimeout=8 -o BatchMode=yes)

resolve_host() {
  case "$1" in
    m4-1|node1|1) echo "$NODE1_HOST" ;;
    m4-2|node2|2) echo "$NODE2_HOST" ;;
    *) echo "ERROR: unknown node '$1' (use m4-1 or m4-2)" >&2; exit 2 ;;
  esac
}

# Only a fixed allowlist of env var NAMES may be queried — never an
# arbitrary grep pattern, so this can't be used to fish for secrets.
ALLOWED_ENV_VARS="EXO_PHASE_MARKS EXO_BATCHED_PREFILL_RENDEZVOUS_MS EXO_SPECULATIVE_GAMMA EXO_DSV4_BATCHED_PREFILL MLX_GEMV_BATCH_INVARIANT MLX_STEEL_BATCH_INVARIANT EXO_DSV4_VERIFY_BATCH EXO_WORKER_PLAN_EVENT_WAKE EXO_PROFILER FENCE_EVERY_N_LAYERS"

cmd="${1:-}"
node="${2:-}"
[ -n "$cmd" ] || { echo "usage: $0 <health|env|sha|ps|gpu> <m4-1|m4-2> [VAR]" >&2; exit 1; }

case "$cmd" in
  health)
    host="$(resolve_host "$node")"
    ip="$([ "$node" = "m4-1" ] || [ "$node" = "node1" ] || [ "$node" = "1" ] && echo "$NODE1_IP" || echo "")"
    # Prefer the .local hostname; only node1 has a documented static IP fallback.
    curl -s -m 5 "http://${host}:${API_PORT}/v1/models" || {
      [ -n "$ip" ] && curl -s -m 5 "http://${ip}:${API_PORT}/v1/models"
    }
    ;;
  env)
    host="$(resolve_host "$node")"
    var="${3:-}"
    [ -n "$var" ] || { echo "usage: $0 env <m4-1|m4-2> <VAR>" >&2; exit 1; }
    case " $ALLOWED_ENV_VARS " in
      *" $var "*) ;;
      *) echo "ERROR: '$var' is not in the allowed env-var list (edit script to add)." >&2; exit 2 ;;
    esac
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
      "for p in \$(pgrep -f 'python.*exo'); do ps eww \$p 2>/dev/null | tr ' ' '\n' | grep '^${var}='; done"
    ;;
  sha)
    host="$(resolve_host "$node")"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "cd ~/repos/exo && git rev-parse HEAD"
    ;;
  ps)
    host="$(resolve_host "$node")"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "pgrep -fl 'python.*exo' || true"
    ;;
  gpu)
    host="$(resolve_host "$node")"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" \
      "sudo -n powermetrics --samplers gpu_power -i 500 -n 4 2>&1 | grep -i 'gpu power'"
    ;;
  *)
    echo "usage: $0 <health|env|sha|ps|gpu> <m4-1|m4-2> [VAR]" >&2
    exit 1
    ;;
esac

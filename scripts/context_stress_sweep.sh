#!/bin/bash
# Multi-level context stress test: prefill + decode measurements, back to
# back, watching for OOM/stall/crash. Run from the controller machine
# against the cluster's public API. Pipeline mode is single-request-only,
# so iterations are strictly sequential (bench/mtp_longctx_probe.py
# already enforces this per-level).
#
# Usage: bash scripts/context_stress_sweep.sh
# (adjust BASE_URL/MODEL/LEVELS below for a different cluster/model)
#
# Requires: bench/mtp_longctx_probe.py present, and SSH access to both
# cluster nodes for health checks (adjust the ssh hostnames below if
# your node names differ).
#
# History: originally run ad hoc 2026-07-22 to validate the Pipeline +
# PP DSpark speculative-decode config (2K-250K tokens, 11 requests,
# zero stalls/crashes/OOM, 100% needle-recall, 27-33 tok/s decode --
# see refs/pp-dspark-required-flags-and-poolingcache-fix-2026-07-20.md
# in the exo-cluster-development skill). Re-run 2026-07-23 against a
# fresh ./start_cluster.sh launch after DSV4_SHARDING defaulted to
# Pipeline (commit 03f7cab6) to confirm a plain launch reproduces the
# same validated config -- identical clean result (see that commit's
# session for the full comparison table). Checked into the repo here
# so it's a reusable regression check rather than a one-off /tmp script.
set -uo pipefail
cd "$(dirname "$0")/.."

BASE_URL="http://192.168.86.201:52415"
MODEL="mlx-community/DeepSeek-V4-Flash"
NODE1="macstudio-m4-1"
NODE2="macstudio-m4-2"
LOG=/tmp/stress_sweep_results.log
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

check_health() {
    local label="$1"
    log "--- health check: $label ---"
    curl -s -m 10 "$BASE_URL/state" > /tmp/stress_state_check.json 2>&1
    python3 -c "
import json
try:
    d = json.load(open('/tmp/stress_state_check.json'))
    runners = d.get('runners', {})
    for rid, r in runners.items():
        print('  runner', rid, r)
    insts = d.get('instances', {})
    print('  instance count:', len(insts))
except Exception as e:
    print('  STATE CHECK FAILED:', e)
" | tee -a "$LOG"
    ssh "$NODE1" "top -l 1 -n 0 | grep -E 'PhysMem'" 2>&1 | sed 's/^/  node1 /' | tee -a "$LOG"
    ssh "$NODE2" "top -l 1 -n 0 | grep -E 'PhysMem'" 2>&1 | sed 's/^/  node2 /' | tee -a "$LOG"
}

check_log_signatures() {
    # NOTE: this grep pattern includes "OUTLIER" which also matches
    # exo's own pre-existing per-cycle stall-detection log line (that's
    # intentional -- it's the signal you actually want here). It does
    # NOT filter out benign noise like model_cards validation Tracebacks
    # at startup -- inspect matches manually before treating a nonzero
    # count as a real problem; compare against the baseline count taken
    # before the sweep starts, not against zero.
    local label="$1"
    local n1_out n2_out
    n1_out=$(ssh "$NODE1" "grep -icE 'OutOfMemory|out of memory|MemoryError|Metal.*allocat.*fail|SIGABRT|SIGKILL|Fatal|Traceback|RunnerFailed|OUTLIER' ~/exo.log" 2>&1)
    n2_out=$(ssh "$NODE2" "grep -icE 'OutOfMemory|out of memory|MemoryError|Metal.*allocat.*fail|SIGABRT|SIGKILL|Fatal|Traceback|RunnerFailed|OUTLIER' ~/exo.log" 2>&1)
    log "  [$label] cumulative log signature counts -- node1: $n1_out  node2: $n2_out"
}

log "=== STRESS SWEEP START ==="
check_health "baseline"
check_log_signatures "baseline"

# target_tokens, iters, max_tokens -- tune per available session time.
# Prefill cost scales roughly linearly with target_tokens; 250K tokens
# took ~550-575s TTFT on a 2-node Mac Studio M4 DSv4-Flash-8bit cluster
# (2026-07-22 and 2026-07-23 measurements, consistent both times). Push
# higher levels only with time budget to match -- this is not a hard
# technical ceiling, just a session-length tradeoff from the original run.
LEVELS=(
    "2000 3 150"
    "20000 3 150"
    "75000 2 150"
    "150000 2 150"
    "250000 1 150"
)

for entry in "${LEVELS[@]}"; do
    read -r TOKENS ITERS MAXTOK <<< "$entry"
    log "=== LEVEL: target_tokens=$TOKENS iters=$ITERS max_tokens=$MAXTOK ==="
    .venv/bin/python bench/mtp_longctx_probe.py \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --target-tokens "$TOKENS" \
        --chars-per-token 5.52 \
        --iters "$ITERS" \
        --max-tokens "$MAXTOK" \
        --label "L${TOKENS}" \
        --seed $((7749 + TOKENS)) \
        2>&1 | tee -a "$LOG"
    PROBE_EXIT=${PIPESTATUS[0]}
    log "  probe exit code: $PROBE_EXIT"
    check_health "after L${TOKENS}"
    check_log_signatures "after L${TOKENS}"
    sleep 20
done

log "=== STRESS SWEEP COMPLETE ==="

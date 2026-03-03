#!/bin/bash
# Stress Test: Incrementally test context lengths from 10K to 100K tokens.
#
# Usage: ./scripts/stress_test_context.sh [API_HOST] [MODEL_ID]
#
# Defaults:
#   API_HOST = 192.168.86.201:52415
#   MODEL_ID = mlx-community/MiniMax-M2.5-5bit
#
# For each context size:
#   1. Sends a prefill prompt of the target length
#   2. Generates 50 decode tokens
#   3. Logs TTFT, decode throughput, memory stats
#   4. Stops on crash and preserves logs

set -euo pipefail

API_HOST="${1:-192.168.86.201:52415}"
MODEL_ID="${2:-mlx-community/MiniMax-M2.5-5bit}"
DECODE_TOKENS=50
START_CTX=10000
END_CTX=100000
STEP_CTX=10000
LOG_DIR="/tmp/stress_test_$(date +%Y%m%d_%H%M%S)"
NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")

mkdir -p "$LOG_DIR"

echo "=== Context Length Stress Test ==="
echo "API: http://$API_HOST"
echo "Model: $MODEL_ID"
echo "Range: ${START_CTX} to ${END_CTX} (step ${STEP_CTX})"
echo "Decode tokens per run: $DECODE_TOKENS"
echo "Logs: $LOG_DIR"
echo "=================================="

# Generate a repeating prompt of approximately N tokens.
# Uses "Hello world " repeated (each repetition is ~2-3 tokens; we overshoot slightly).
generate_prompt() {
    local target_tokens=$1
    # ~2.5 tokens per "Hello world " repetition
    local repeats=$(( target_tokens * 10 / 25 ))
    local prompt=""
    for ((i=0; i<repeats; i++)); do
        prompt+="Hello world "
    done
    echo "$prompt"
}

# Snapshot remote logs before each run
snapshot_logs() {
    local label=$1
    for node in "${NODES[@]}"; do
        ssh "$node" "wc -l /tmp/exo.log 2>/dev/null" > "$LOG_DIR/${label}_${node}_linecount.txt" 2>/dev/null || true
    done
}

# Collect remote logs after each run (only new lines since snapshot)
collect_logs() {
    local label=$1
    for node in "${NODES[@]}"; do
        local prev_lines=$(cat "$LOG_DIR/${label}_${node}_linecount.txt" 2>/dev/null | awk '{print $1}')
        prev_lines=${prev_lines:-0}
        ssh "$node" "tail -n +$((prev_lines + 1)) /tmp/exo.log" > "$LOG_DIR/${label}_${node}.log" 2>/dev/null || true
    done
}

# Extract memory stats from node logs
extract_memory_stats() {
    local label=$1
    echo "  Memory stats:"
    for node in "${NODES[@]}"; do
        local log_file="$LOG_DIR/${label}_${node}.log"
        if [ -f "$log_file" ]; then
            local last_decode_log=$(grep '\[DECODE ' "$log_file" | tail -1)
            if [ -n "$last_decode_log" ]; then
                echo "    $node: $last_decode_log"
            fi
        fi
    done
}

echo ""
for ctx in $(seq $START_CTX $STEP_CTX $END_CTX); do
    label="ctx_${ctx}"
    echo "--- Testing context length: $ctx tokens ---"

    # Snapshot log positions
    snapshot_logs "$label"

    # Generate the prompt
    echo "  Generating prompt (~$ctx tokens)..."
    PROMPT=$(generate_prompt "$ctx")
    PROMPT_LEN=${#PROMPT}
    echo "  Prompt length: $PROMPT_LEN chars"

    # Build the request JSON
    REQUEST=$(jq -n \
        --arg model "$MODEL_ID" \
        --arg prompt "$PROMPT" \
        --argjson max_tokens "$DECODE_TOKENS" \
        '{model: $model, prompt: $prompt, max_tokens: $max_tokens, stream: false}')

    # Send the request and measure time
    echo "  Sending request..."
    START_TIME=$(date +%s%N)

    HTTP_CODE=$(curl -s -o "$LOG_DIR/${label}_response.json" -w "%{http_code}" \
        --max-time 600 \
        -X POST "http://$API_HOST/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$REQUEST" 2>"$LOG_DIR/${label}_curl_err.txt") || true

    END_TIME=$(date +%s%N)
    ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

    # Collect logs from all nodes
    collect_logs "$label"

    # Check result
    if [ "$HTTP_CODE" != "200" ]; then
        echo "  FAILED! HTTP $HTTP_CODE after ${ELAPSED_MS}ms"
        echo "  Response: $(cat "$LOG_DIR/${label}_response.json" 2>/dev/null | head -5)"
        echo "  Curl error: $(cat "$LOG_DIR/${label}_curl_err.txt" 2>/dev/null)"
        echo ""
        echo "  Collecting crash logs..."
        for node in "${NODES[@]}"; do
            echo "  --- $node (last 30 lines) ---"
            ssh "$node" "tail -30 /tmp/exo.log" 2>/dev/null || echo "  (unreachable)"
        done
        echo ""
        echo "STOPPED at context length $ctx. Logs preserved in $LOG_DIR"
        exit 1
    fi

    # Parse response for usage stats
    PROMPT_TOKENS=$(jq -r '.usage.prompt_tokens // "?"' "$LOG_DIR/${label}_response.json")
    COMPLETION_TOKENS=$(jq -r '.usage.completion_tokens // "?"' "$LOG_DIR/${label}_response.json")
    TOTAL_TIME_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)

    # Calculate decode throughput (approximate: total time includes prefill)
    # For better accuracy, look for TTFT in the response or logs
    TTFT=$(jq -r '.usage.prompt_tokens_details.time_to_first_token_ms // "?"' "$LOG_DIR/${label}_response.json" 2>/dev/null || echo "?")
    if [ "$TTFT" != "?" ] && [ "$TTFT" != "null" ]; then
        DECODE_TIME_MS=$(( ELAPSED_MS - ${TTFT%.*} ))
        if [ "$DECODE_TIME_MS" -gt 0 ] && [ "$COMPLETION_TOKENS" != "?" ]; then
            DECODE_TPS=$(echo "scale=1; $COMPLETION_TOKENS * 1000 / $DECODE_TIME_MS" | bc)
            echo "  OK: ${PROMPT_TOKENS} prompt + ${COMPLETION_TOKENS} completion in ${TOTAL_TIME_S}s"
            echo "  TTFT: ${TTFT}ms | Decode: ${DECODE_TPS} tok/s"
        else
            echo "  OK: ${PROMPT_TOKENS} prompt + ${COMPLETION_TOKENS} completion in ${TOTAL_TIME_S}s"
        fi
    else
        echo "  OK: ${PROMPT_TOKENS} prompt + ${COMPLETION_TOKENS} completion in ${TOTAL_TIME_S}s"
    fi

    extract_memory_stats "$label"
    echo ""

    # Brief pause between runs to let memory settle
    sleep 2
done

echo "=== Stress test complete ==="
echo "All context lengths tested successfully!"
echo "Logs: $LOG_DIR"

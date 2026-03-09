#!/bin/bash
# Benchmark with full hardware tracing via macmon on all inference nodes.
#
# Usage: ./scripts/benchmark_with_trace.sh [API_HOST] [MODEL_ID] [LABEL]
#
# Runs macmon on each Studio during the benchmark, then collects:
#   - macmon JSON (GPU freq, usage, power, temp, memory) at 250ms intervals
#   - exo.log from each node
#   - benchmark results from the API
#
# Output: tmp/trace_<LABEL>/

set -euo pipefail

API_HOST="${1:-192.168.86.201:52415}"
MODEL_ID="${2:-mlx-community/MiniMax-M2.5-5bit}"
LABEL="${3:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="tmp/trace_${LABEL}"
MACMON_INTERVAL=250  # ms between samples

NODES=("macstudio-m4-1" "macstudio-m4-2")
CONTEXTS=(1000 10000 30000 50000 70000 100000)
DECODE_TOKENS=32

mkdir -p "$OUT_DIR"

echo "=== Benchmark with Hardware Trace ==="
echo "API:      http://$API_HOST"
echo "Model:    $MODEL_ID"
echo "Label:    $LABEL"
echo "Output:   $OUT_DIR/"
echo "Interval: ${MACMON_INTERVAL}ms"
echo ""

# --- Helper: generate prompt of approximate token count ---
generate_prompt() {
    local target_tokens=$1
    # ~1 token per 4 chars for English text; overshoot slightly
    local chars=$((target_tokens * 4))
    local base="The quick brown fox jumps over the lazy dog. "
    local prompt=""
    while [ ${#prompt} -lt $chars ]; do
        prompt="${prompt}${base}"
    done
    echo "$prompt"
}

# --- Start macmon on all nodes ---
start_macmon() {
    local ctx_label=$1
    for node in "${NODES[@]}"; do
        echo "  Starting macmon on $node..."
        ssh "$node" "macmon pipe -i $MACMON_INTERVAL --soc-info > /tmp/macmon_trace.jsonl 2>&1 &" &
    done
    sleep 1  # let macmon start
}

# --- Stop macmon and collect data ---
stop_macmon() {
    local ctx_label=$1
    for node in "${NODES[@]}"; do
        # Kill macmon
        ssh "$node" "pkill -f 'macmon pipe' 2>/dev/null || true"
        # Pull trace
        scp "$node:/tmp/macmon_trace.jsonl" "$OUT_DIR/${node}_macmon_${ctx_label}.jsonl" 2>/dev/null || true
        ssh "$node" "rm -f /tmp/macmon_trace.jsonl" 2>/dev/null || true
    done
}

# --- Run one benchmark request ---
run_benchmark() {
    local ctx=$1
    local ctx_label=$2

    echo ""
    echo "=== Context: ~${ctx} tokens ==="

    # Generate prompt
    local prompt
    prompt=$(generate_prompt "$ctx")

    # Start hardware tracing
    start_macmon "$ctx_label"

    # Fire the request
    local start_time
    start_time=$(date +%s.%N)

    local response
    response=$(curl -sf --max-time 600 "http://$API_HOST/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg model "$MODEL_ID" \
            --arg prompt "$prompt" \
            --argjson max_tokens "$DECODE_TOKENS" \
            '{
                model: $model,
                messages: [{role: "user", content: ("Repeat the word hello exactly 32 times: " + $prompt)}],
                max_completion_tokens: $max_tokens,
                temperature: 0
            }'
        )" 2>&1)

    local end_time
    end_time=$(date +%s.%N)

    # Stop hardware tracing
    stop_macmon "$ctx_label"

    # Parse response
    local usage
    usage=$(echo "$response" | jq -r '.usage // empty' 2>/dev/null)
    if [ -n "$usage" ]; then
        local prompt_tokens completion_tokens ttft total_time
        prompt_tokens=$(echo "$usage" | jq -r '.prompt_tokens')
        completion_tokens=$(echo "$usage" | jq -r '.completion_tokens')
        total_time=$(echo "$end_time - $start_time" | bc)
        echo "  Prompt: ${prompt_tokens} tokens, Completion: ${completion_tokens} tokens"
        echo "  Total time: ${total_time}s"
    else
        echo "  WARNING: No usage data in response"
        echo "  Response: $(echo "$response" | head -c 500)"
    fi

    # Save raw response
    echo "$response" > "$OUT_DIR/response_${ctx_label}.json"

    sleep 2  # cooldown between requests
}

# --- Main benchmark loop ---
echo ""
echo "Starting benchmark..."

for ctx in "${CONTEXTS[@]}"; do
    ctx_label="${ctx}"
    run_benchmark "$ctx" "$ctx_label"
done

# --- Collect logs from nodes ---
echo ""
echo "=== Collecting node logs ==="
for node in "${NODES[@]}" "macbook-m4"; do
    echo -n "  $node... "
    if scp "$node:~/exo.log" "$OUT_DIR/${node}.log" 2>/dev/null; then
        SIZE=$(du -h "$OUT_DIR/${node}.log" | awk '{print $1}')
        echo "$SIZE"
    else
        echo "FAILED"
    fi
done

# --- Collect git commits ---
echo ""
echo "Git commits:" | tee "$OUT_DIR/commits.txt"
for node in "${NODES[@]}" "macbook-m4"; do
    COMMIT=$(ssh "$node" "cd ~/repos/exo && git rev-parse --short HEAD" 2>/dev/null || echo "unreachable")
    echo "  $node: $COMMIT" | tee -a "$OUT_DIR/commits.txt"
done

echo ""
echo "=== Trace complete ==="
echo "Output: $OUT_DIR/"
echo ""
echo "Analyze with:"
echo "  python3 scripts/analyze_trace.py $OUT_DIR"

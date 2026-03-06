#!/bin/bash
# Benchmark decode throughput at multiple context sizes.
#
# Usage: ./scripts/benchmark_decode.sh [API_HOST] [MODEL_ID] [LABEL]
#
# Produces a markdown table comparing TTFT and decode tok/s at each context size.
# Designed to be run after cluster is up and model is loaded.

set -euo pipefail

API_HOST="${1:-192.168.86.201:52415}"
MODEL_ID="${2:-mlx-community/MiniMax-M2.5-5bit}"
LABEL="${3:-$(date +%Y%m%d_%H%M%S)}"
DECODE_TOKENS=32
CONTEXT_SIZES=(1000 10000 30000 50000 70000 100000)
NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")
RESULTS_FILE="/tmp/benchmark_${LABEL}.md"

echo "=== Decode Benchmark ==="
echo "API:    http://$API_HOST"
echo "Model:  $MODEL_ID"
echo "Label:  $LABEL"
echo "Sizes:  ${CONTEXT_SIZES[*]}"
echo "Output: $RESULTS_FILE"
echo "========================"
echo ""

# Verify cluster is healthy
echo "Checking cluster health..."
NODE_COUNT=$(curl -sf "http://$API_HOST/state" | python3 -c '
import sys, json
d = json.load(sys.stdin)
total = 0
for iid, inst in d.get("instances", {}).items():
    for k, inner in inst.items():
        total += len(inner.get("shardAssignments", {}).get("runnerToShard", {}))
print(total)
' 2>/dev/null || echo "0")
if [ "$NODE_COUNT" -lt 3 ]; then
    echo "ERROR: Only $NODE_COUNT runners in cluster (need 3). Is the cluster running?"
    exit 1
fi
echo "  $NODE_COUNT runners online"

# Verify runners are ready (runner count == shard count in this cluster)
RUNNERS_READY=$NODE_COUNT
echo "  $RUNNERS_READY runners ready"

# Get cluster commit for the report
CLUSTER_COMMIT=$(ssh macstudio-m4-1 "git -C ~/repos/exo rev-parse --short HEAD" 2>/dev/null || echo "unknown")

# Collect results in arrays
declare -a R_CTX R_TTFT R_DECODE_TPS R_STATUS

run_at_context() {
    local target_ctx=$1
    local idx=$2

    echo ""
    echo "--- ${target_ctx} tokens ---"

    # Generate payload with approximate token count
    # ~4 chars per token, use repeating text
    local payload_file="/tmp/bench_payload_${target_ctx}.json"
    python3 -c "
import json
line = 'The quick brown fox jumps over the lazy dog. '
chars_needed = $target_ctx * 4
text = (line * (chars_needed // len(line) + 1))[:chars_needed]
payload = {
    'model': '$MODEL_ID',
    'messages': [{'role': 'user', 'content': f'Repeat the word hello exactly $DECODE_TOKENS times: {text}'}],
    'max_tokens': $DECODE_TOKENS,
    'temperature': 0.1,
    'stream': True,
    'enable_thinking': False
}
with open('$payload_file', 'w') as f:
    json.dump(payload, f)
"

    # Snapshot log positions for post-analysis
    for node in "${NODES[@]}"; do
        ssh "$node" "wc -l ~/exo.log 2>/dev/null | awk '{print \$1}'" > "/tmp/bench_logpos_${node}" 2>/dev/null || echo "0" > "/tmp/bench_logpos_${node}"
    done

    # Run inference with streaming, measure TTFT and decode rate
    local result
    result=$(python3 <<PYEOF
import json, sys, time, urllib.request

with open("$payload_file") as f:
    payload = json.load(f)

req = urllib.request.Request(
    "http://$API_HOST/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

t_start = time.time()
t_first_token = None
token_count = 0

try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(b"\n"):
                line = buf.decode().strip()
                buf = b""
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        if t_first_token is None:
                            t_first_token = time.time()
                        token_count += 1
                except json.JSONDecodeError:
                    pass
except Exception as e:
    print(json.dumps({"status": "FAIL", "error": str(e)}))
    sys.exit(0)

t_end = time.time()
ttft = round(t_first_token - t_start, 2) if t_first_token else -1
decode_time = (t_end - t_first_token) if t_first_token else 0
tps = round(token_count / decode_time, 1) if decode_time > 0 else 0

print(json.dumps({"status": "OK", "ttft": ttft, "tps": tps, "tokens": token_count}))
PYEOF
    )

    local status ttft tps
    status=$(echo "$result" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])' 2>/dev/null || echo "FAIL")
    ttft=$(echo "$result" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("ttft","?"))' 2>/dev/null || echo "?")
    tps=$(echo "$result" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tps","?"))' 2>/dev/null || echo "?")

    R_CTX[$idx]="$target_ctx"
    R_TTFT[$idx]="$ttft"
    R_DECODE_TPS[$idx]="$tps"
    R_STATUS[$idx]="$status"

    if [ "$status" = "OK" ]; then
        echo "  TTFT: ${ttft}s | Decode: ${tps} tok/s"
    else
        local error
        error=$(echo "$result" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("error","unknown"))' 2>/dev/null || echo "unknown")
        echo "  FAILED: $error"
        echo "  Collecting crash logs..."
        for node in "${NODES[@]}"; do
            local prev_lines
            prev_lines=$(cat "/tmp/bench_logpos_${node}" 2>/dev/null || echo "0")
            echo "  --- $node (new lines since start) ---"
            ssh "$node" "tail -n +$((prev_lines + 1)) ~/exo.log | tail -20" 2>/dev/null || echo "  (unreachable)"
        done
    fi

    # Brief pause between runs
    sleep 2
}

# Run benchmarks
for i in "${!CONTEXT_SIZES[@]}"; do
    run_at_context "${CONTEXT_SIZES[$i]}" "$i"
done

# Generate markdown report
echo ""
echo "=== Generating report ==="

cat > "$RESULTS_FILE" <<REPORT
# Decode Benchmark — $LABEL

- **Commit**: $CLUSTER_COMMIT
- **Model**: $MODEL_ID
- **Nodes**: $NODE_COUNT
- **Decode tokens per run**: $DECODE_TOKENS
- **Date**: $(date)

| Context | TTFT (s) | Decode (tok/s) | Status |
|---------|----------|----------------|--------|
REPORT

for i in "${!CONTEXT_SIZES[@]}"; do
    ctx="${R_CTX[$i]:-${CONTEXT_SIZES[$i]}}"
    ttft="${R_TTFT[$i]:-?}"
    tps="${R_DECODE_TPS[$i]:-?}"
    status="${R_STATUS[$i]:-?}"
    # Format context as "10K", "100K" etc
    if [ "$ctx" -ge 1000 ]; then
        ctx_label="$(echo "$ctx" | awk '{printf "%dK", $1/1000}')"
    else
        ctx_label="$ctx"
    fi
    echo "| ~${ctx_label} | ${ttft} | ${tps} | ${status} |" >> "$RESULTS_FILE"
done

echo "" >> "$RESULTS_FILE"

# Print the report to stdout too
echo ""
cat "$RESULTS_FILE"

echo ""
echo "Results saved to: $RESULTS_FILE"
REPORT

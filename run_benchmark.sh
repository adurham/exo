#!/bin/bash

# A/B Benchmark Script — Creates an instance, runs a prompt, captures stats
#
# Usage:
#   ./run_benchmark.sh [--side A|B] [--label "description"]
#
# This script:
#   1. Creates a Pipeline instance with MiniMax-M2.5-5bit, MLX RDMA, all 3 nodes
#   2. Waits for the model to finish downloading/loading
#   3. Sends a complex prompt via the OpenAI-compatible /v1/chat/completions endpoint
#   4. Streams the response, measuring TTFT and TPS
#   5. Saves a results summary to /tmp/bench_results_<side>_<timestamp>.txt

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
API_HOST="192.168.86.201"
API_PORT="52415"
API_BASE="http://${API_HOST}:${API_PORT}"
MODEL_ID="mlx-community/MiniMax-M2.5-5bit"
SHARDING="Pipeline"
INSTANCE_META="MlxJaccl"
MIN_NODES=3

SIDE="B"
LABEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --side) SIDE="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="/tmp/bench_results_${SIDE}_${TIMESTAMP}.txt"

# ── Complex test prompt ───────────────────────────────────────────────
PROMPT='You are a senior distributed systems architect. I need you to design a fault-tolerant, horizontally scalable real-time bidding (RTB) system that can handle 1 million bid requests per second with a 99th percentile latency of under 10ms. The system must support: (1) real-time feature lookups from a distributed feature store, (2) ML model inference for bid price prediction using an ensemble of gradient-boosted trees and a neural network, (3) budget pacing with second-level granularity across 10,000 concurrent campaigns, (4) frequency capping using probabilistic data structures, and (5) a feedback loop for online learning from win/loss notifications. Please provide a detailed architecture with specific technology choices, data flow diagrams described in text, capacity planning calculations, failure mode analysis, and a phased rollout plan. Be extremely thorough and specific.'

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  Exo A/B Benchmark — Side $SIDE                                      ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Model:    $MODEL_ID"
echo "  Sharding: $SHARDING"
echo "  Backend:  $INSTANCE_META"
echo "  Nodes:    $MIN_NODES"
echo "  Label:    ${LABEL:-"(none)"}"
echo "  Results:  $RESULTS_FILE"
echo ""

# ── Step 1: Verify cluster is healthy ─────────────────────────────────
echo "[1/5] Verifying cluster health..."
CLUSTER_STATE=$(curl -sf "${API_BASE}/state" 2>&1) || { echo "ERROR: API not responding at ${API_BASE}"; exit 1; }
NODE_COUNT=$(echo "$CLUSTER_STATE" | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["topology"]["nodes"]))' 2>/dev/null)
if [ "$NODE_COUNT" -lt "$MIN_NODES" ]; then
    echo "ERROR: Only $NODE_COUNT node(s) in cluster, need $MIN_NODES"
    exit 1
fi
echo "  Cluster healthy: $NODE_COUNT nodes ✓"

# ── Step 2: Create instance ───────────────────────────────────────────
echo ""
echo "[2/5] Creating Pipeline instance..."
CREATE_RESPONSE=$(curl -sf -X POST "${API_BASE}/place_instance" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"$MODEL_ID\",
        \"sharding\": \"$SHARDING\",
        \"instance_meta\": \"$INSTANCE_META\",
        \"min_nodes\": $MIN_NODES
    }" 2>&1)

echo "  Response: $CREATE_RESPONSE"
COMMAND_ID=$(echo "$CREATE_RESPONSE" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("commandId",""))' 2>/dev/null || echo "")
echo "  Command ID: ${COMMAND_ID:-"(unknown)"}"

# ── Step 3: Wait for model to be ready ────────────────────────────────
echo ""
echo -n "[3/5] Waiting for model to load..."
MODEL_READY=false
LOAD_START=$(python3 -c 'import time; print(time.time())')
for i in $(seq 1 120); do
    STATE=$(curl -sf "${API_BASE}/state" 2>/dev/null || echo "{}")
    # Check if there's an active instance with runners loaded
    INSTANCE_COUNT=$(echo "$STATE" | python3 -c '
import sys, json
d = json.load(sys.stdin)
instances = d.get("instances", {})
# Look for any instance that has runners
ready = 0
for iid, inst in instances.items():
    runners = d.get("runners", {})
    # Check if runners are active for this instance
    for rid, runner in runners.items():
        if runner.get("instanceId") == iid:
            status = runner.get("status", "")
            if isinstance(status, dict) and "Running" in status:
                ready += 1
print(ready)
' 2>/dev/null || echo "0")

    if [ "$INSTANCE_COUNT" -ge "$MIN_NODES" ]; then
        LOAD_END=$(python3 -c 'import time; print(time.time())')
        LOAD_TIME=$(python3 -c "print(f'{$LOAD_END - $LOAD_START:.1f}')")
        echo " READY! ($LOAD_TIME seconds to load)"
        MODEL_READY=true
        break
    fi
    echo -n "."
    sleep 3
done

if [ "$MODEL_READY" = false ]; then
    echo ""
    echo "WARNING: Model may not be fully loaded yet. Proceeding anyway..."
    LOAD_TIME="timeout"
fi

# Give it a moment to stabilize
sleep 3

# ── Step 4: Run inference with streaming ──────────────────────────────
echo ""
echo "[4/5] Running inference..."
echo "  Prompt: ${PROMPT:0:80}..."
echo ""

# Use Python for precise timing of streaming response
INFERENCE_RESULTS=$(python3 << 'PYEOF'
import json, sys, time, urllib.request

API_BASE = "http://192.168.86.201:52415"
MODEL_ID = "mlx-community/MiniMax-M2.5-5bit"

PROMPT = """You are a senior distributed systems architect. I need you to design a fault-tolerant, horizontally scalable real-time bidding (RTB) system that can handle 1 million bid requests per second with a 99th percentile latency of under 10ms. The system must support: (1) real-time feature lookups from a distributed feature store, (2) ML model inference for bid price prediction using an ensemble of gradient-boosted trees and a neural network, (3) budget pacing with second-level granularity across 10,000 concurrent campaigns, (4) frequency capping using probabilistic data structures, and (5) a feedback loop for online learning from win/loss notifications. Please provide a detailed architecture with specific technology choices, data flow diagrams described in text, capacity planning calculations, failure mode analysis, and a phased rollout plan. Be extremely thorough and specific."""

payload = json.dumps({
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": PROMPT}],
    "stream": True,
    "max_tokens": 4096,
    "temperature": 0.7
}).encode()

req = urllib.request.Request(
    f"{API_BASE}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)

t_start = time.time()
t_first_token = None
token_count = 0
full_response = []
prompt_tokens = 0
completion_tokens = 0

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        buffer = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buffer += chunk
            if buffer.endswith(b"\n"):
                line = buffer.decode("utf-8").strip()
                buffer = b""
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if t_first_token is None:
                            t_first_token = time.time()
                        token_count += 1
                        full_response.append(content)
                        # Print tokens as they arrive
                        print(content, end="", flush=True, file=sys.stderr)

                    # Capture usage from the final chunk
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
                except json.JSONDecodeError:
                    pass
except Exception as e:
    print(f"\nERROR during inference: {e}", file=sys.stderr)

t_end = time.time()
print("", file=sys.stderr)  # newline after streaming

ttft = (t_first_token - t_start) if t_first_token else -1
total_time = t_end - t_start
decode_time = (t_end - t_first_token) if t_first_token else 0
tps = (token_count / decode_time) if decode_time > 0 else 0
response_len = len("".join(full_response))

# Output structured results to stdout
results = {
    "ttft_seconds": round(ttft, 2),
    "total_seconds": round(total_time, 2),
    "decode_seconds": round(decode_time, 2),
    "token_count": token_count,
    "tps": round(tps, 2),
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "response_chars": response_len,
}
print(json.dumps(results))
PYEOF
)

echo ""
echo "  Raw results: $INFERENCE_RESULTS"

# ── Step 5: Save results ──────────────────────────────────────────────
echo ""
echo "[5/5] Saving results..."

TTFT=$(echo "$INFERENCE_RESULTS" | python3 -c 'import sys,json; print(json.load(sys.stdin)["ttft_seconds"])' 2>/dev/null || echo "N/A")
TPS=$(echo "$INFERENCE_RESULTS" | python3 -c 'import sys,json; print(json.load(sys.stdin)["tps"])' 2>/dev/null || echo "N/A")
TOTAL=$(echo "$INFERENCE_RESULTS" | python3 -c 'import sys,json; print(json.load(sys.stdin)["total_seconds"])' 2>/dev/null || echo "N/A")
TOKENS=$(echo "$INFERENCE_RESULTS" | python3 -c 'import sys,json; print(json.load(sys.stdin)["token_count"])' 2>/dev/null || echo "N/A")
PROMPT_T=$(echo "$INFERENCE_RESULTS" | python3 -c 'import sys,json; print(json.load(sys.stdin)["prompt_tokens"])' 2>/dev/null || echo "N/A")
COMP_T=$(echo "$INFERENCE_RESULTS" | python3 -c 'import sys,json; print(json.load(sys.stdin)["completion_tokens"])' 2>/dev/null || echo "N/A")

cat > "$RESULTS_FILE" << RESULTS
══════════════════════════════════════════════════════
  Exo A/B Benchmark Results — Side $SIDE
  $(date)
══════════════════════════════════════════════════════

  Label:              ${LABEL:-"(none)"}
  Model:              $MODEL_ID
  Sharding:           $SHARDING
  Backend:            $INSTANCE_META
  Nodes:              $NODE_COUNT
  Load Time:          ${LOAD_TIME:-"N/A"} seconds

  ── Performance ──
  TTFT:               ${TTFT}s
  Tokens/sec (TPS):   ${TPS}
  Total Time:         ${TOTAL}s
  Tokens Generated:   ${TOKENS}
  Prompt Tokens:      ${PROMPT_T}
  Completion Tokens:  ${COMP_T}

  ── Raw JSON ──
  $INFERENCE_RESULTS

══════════════════════════════════════════════════════
RESULTS

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  Benchmark Complete — Side $SIDE                                      ║"
echo "╠═══════════════════════════════════════════════════════════════════════╣"
echo "║  TTFT:    ${TTFT}s"
echo "║  TPS:     ${TPS} tokens/sec"
echo "║  Total:   ${TOTAL}s (${TOKENS} tokens)"
echo "║  Results: $RESULTS_FILE"
echo "╚═══════════════════════════════════════════════════════════════════════╝"

#!/bin/bash
# Benchmark speculative decoding acceptance rate across temperatures
# Usage: ./scripts/bench_speculative.sh [api_url]

set -euo pipefail

API="${1:-http://192.168.86.201:52415}"
MODEL="mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit"
MAX_TOKENS=200

PROMPTS=(
    "List the first 20 elements of the periodic table with their atomic numbers."
    "Write a Python function that implements binary search on a sorted list."
    "Explain how TCP three-way handshake works step by step."
    "What are the differences between compiled and interpreted programming languages?"
    "Describe the water cycle in detail, including all major stages."
)

echo "=== Speculative Decoding Benchmark ==="
echo "API: $API"
echo "Model: $MODEL"
echo "Max tokens: $MAX_TOKENS"
echo "Prompts: ${#PROMPTS[@]}"
echo ""

# Clear Studio logs before benchmark
ssh macstudio-m4-1 'echo "" > ~/exo.log' 2>/dev/null || true

for temp in 0.0 0.3 0.5 0.7 1.0; do
    echo "--- Temperature: $temp ---"
    total_accepted=0
    total_generated=0
    total_exchanges=0

    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"

        result=$(curl -s --max-time 120 "$API/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$(python3 -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': '''$prompt'''}],
    'max_tokens': $MAX_TOKENS,
    'temperature': $temp,
    'stream': False,
}))" 2>/dev/null)")

        if [ -z "$result" ] || [ "$result" = "null" ]; then
            echo "  Prompt $((i+1)): FAILED (no response)"
            continue
        fi

        gen=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
        accepted=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens_details']['accepted_prediction_tokens'])" 2>/dev/null || echo "0")

        if [ "$gen" -gt 0 ]; then
            rate=$(python3 -c "print(f'{$accepted / $gen * 100:.1f}' if $gen > 0 else '0')")
            echo "  Prompt $((i+1)): gen=$gen accepted=$accepted rate=${rate}%"
            total_accepted=$((total_accepted + accepted))
            total_generated=$((total_generated + gen))
        else
            echo "  Prompt $((i+1)): FAILED (0 tokens)"
        fi

        # Brief pause between requests
        sleep 1
    done

    if [ "$total_generated" -gt 0 ]; then
        overall_rate=$(python3 -c "print(f'{$total_accepted / $total_generated * 100:.1f}')")
        echo "  TOTAL: temp=$temp gen=$total_generated accepted=$total_accepted rate=${overall_rate}%"
    fi
    echo ""
done

# Pull speculative stats from Studio 1 logs
echo "=== Runner-side speculative stats (from stderr) ==="
ssh macstudio-m4-1 'python3 -c "
for line in open(\"/Users/adam.durham/exo.log\"):
    if \"[speculative]\" in line and len(line) < 500:
        print(line.rstrip()[:300])
"' 2>/dev/null | tail -30

echo ""
echo "Done."

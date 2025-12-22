#!/bin/bash

# Test script for EXO generation with timeout and throughput measurement

API_URL="http://100.93.253.67:52415"
MODEL="mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit"
TIMEOUT=120  # 2 minute timeout
MAX_TOKENS=100

echo "Testing EXO generation..."
echo "API: $API_URL"
echo "Model: $MODEL"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Check if instance exists
echo "Checking for running instance..."
INSTANCE_CHECK=$(curl -s "${API_URL}/state" | python3 -c "import sys, json; data=json.load(sys.stdin); instances=data.get('instances', {}); print('FOUND' if instances else 'NOT_FOUND')" 2>/dev/null)

if [ "$INSTANCE_CHECK" != "FOUND" ]; then
    echo "ERROR: No instance found. Please launch an instance via the dashboard first."
    exit 1
fi

echo "Instance found. Starting generation test..."
echo ""

# Run the test with timeout
START_TIME=$(date +%s.%N)
TOKEN_COUNT=0
CONTENT=""
FIRST_TOKEN_TIME=""

curl --max-time $TIMEOUT -X POST "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Tell me a bedtime story about a brave little robot who discovers a magical forest.\"}],
    \"stream\": true,
    \"max_tokens\": ${MAX_TOKENS}
  }" 2>/dev/null | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        # Extract JSON from "data: {...}"
        json_data="${line#data: }"
        
        # Check for [DONE]
        if [[ "$json_data" == "[DONE]" ]]; then
            break
        fi
        
        # Parse JSON and extract content
        delta=$(echo "$json_data" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('choices', [{}])[0].get('delta', {}).get('content', ''))" 2>/dev/null)
        
        if [ -n "$delta" ]; then
            if [ -z "$FIRST_TOKEN_TIME" ]; then
                FIRST_TOKEN_TIME=$(date +%s.%N)
                FIRST_TOKEN_DELAY=$(echo "$FIRST_TOKEN_TIME - $START_TIME" | bc)
                echo "First token received after ${FIRST_TOKEN_DELAY}s"
            fi
            
            TOKEN_COUNT=$((TOKEN_COUNT + 1))
            CONTENT="${CONTENT}${delta}"
            echo -n "$delta"
        fi
    fi
done

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo ""
echo "=== RESULTS ==="
echo "Total tokens: $TOKEN_COUNT"
echo "Total time: ${TOTAL_TIME}s"
if [ -n "$FIRST_TOKEN_TIME" ]; then
    GENERATION_TIME=$(echo "$END_TIME - $FIRST_TOKEN_TIME" | bc)
    if (( $(echo "$GENERATION_TIME > 0" | bc -l) )); then
        TOKENS_PER_SEC=$(echo "scale=2; $TOKEN_COUNT / $GENERATION_TIME" | bc)
        echo "Generation time: ${GENERATION_TIME}s"
        echo "Tokens/second: $TOKENS_PER_SEC"
        
        # Check if meets requirement
        if (( $(echo "$TOKENS_PER_SEC >= 5" | bc -l) )); then
            echo "✅ PASS: Meets requirement of >=5 tokens/second"
        else
            echo "❌ FAIL: Below requirement of 5 tokens/second"
        fi
    fi
fi
echo ""
echo "=== GENERATED CONTENT ==="
echo "$CONTENT"
echo ""


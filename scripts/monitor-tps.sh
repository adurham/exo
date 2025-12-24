#!/bin/bash

# TPS Monitoring Script for Static Cluster
# Monitors tokens/second throughput and swap usage
# CRITICAL: Must achieve ≥5 tokens/second and zero swap usage

MASTER_API_URL="${MASTER_API_URL:-http://100.67.156.10:52415}"
MODEL_ID="${MODEL_ID:-mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit}"
NODES=("adams-macbook-pro-m1" "adams-mac-studio-m4" "adams-macbook-pro-m4" "adams-work-macbook-pro-m4")

MIN_TPS=5.0
CHECK_INTERVAL=5  # seconds between checks
MAX_SAMPLES=100   # number of samples to keep for averaging

# Function to check swap usage on a node
check_swap() {
    local node=$1
    ssh "$node" "vm_stat | grep 'Swap' | awk '{print \$3}' | sed 's/\\.//'" 2>/dev/null || echo "0"
}

# Function to verify zero swap on all nodes
check_all_swap() {
    local swap_detected=false
    for node in "${NODES[@]}"; do
        local swap_used=$(check_swap "$node")
        if [ "$swap_used" != "0" ]; then
            echo "❌ SWAP DETECTED on $node: $swap_used pages"
            swap_detected=true
        fi
    done
    if [ "$swap_detected" = true ]; then
        return 1
    fi
    return 0
}

# Function to send a test request and measure TPS
measure_tps() {
    local prompt="${1:-Hello, how are you?}"
    local max_tokens="${2:-50}"
    
    local start_time=$(date +%s.%N)
    
    # Send request to API
    local response=$(curl -s -X POST "$MASTER_API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_ID\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": $max_tokens,
            \"stream\": false
        }" 2>/dev/null)
    
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    
    # Extract token count from response
    local token_count=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Try to get usage.completion_tokens
    usage = data.get('usage', {})
    tokens = usage.get('completion_tokens', 0)
    print(tokens)
except:
    print(0)
" 2>/dev/null || echo "0")
    
    # Calculate TPS
    local tps=0
    if [ "$(echo "$elapsed > 0" | bc)" -eq 1 ] && [ "$token_count" -gt 0 ]; then
        tps=$(echo "scale=2; $token_count / $elapsed" | bc)
    fi
    
    echo "$tps:$token_count:$elapsed"
}

# Function to monitor continuously
monitor_continuous() {
    local samples=()
    local total_tps=0
    local sample_count=0
    local swap_failures=0
    local tps_failures=0
    
    echo "========================================="
    echo "TPS Monitoring Started"
    echo "========================================="
    echo "Master API: $MASTER_API_URL"
    echo "Model: $MODEL_ID"
    echo "Minimum TPS Required: $MIN_TPS tokens/second"
    echo "Check Interval: $CHECK_INTERVAL seconds"
    echo "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        # Check swap usage
        if ! check_all_swap; then
            swap_failures=$((swap_failures + 1))
            echo "⚠️  WARNING: Swap usage detected (failure #$swap_failures)"
        else
            echo "✅ Swap: OK"
        fi
        
        # Measure TPS
        local result=$(measure_tps "Generate a short response." 30)
        local tps=$(echo "$result" | cut -d: -f1)
        local tokens=$(echo "$result" | cut -d: -f2)
        local elapsed=$(echo "$result" | cut -d: -f3)
        
        if [ -z "$tps" ] || [ "$tps" = "0" ]; then
            echo "❌ ERROR: Failed to measure TPS"
            sleep $CHECK_INTERVAL
            continue
        fi
        
        # Add to samples
        samples+=("$tps")
        if [ ${#samples[@]} -gt $MAX_SAMPLES ]; then
            samples=("${samples[@]:1}")  # Remove oldest sample
        fi
        
        # Calculate average TPS
        local sum=0
        for sample in "${samples[@]}"; do
            sum=$(echo "$sum + $sample" | bc)
        done
        local avg_tps=$(echo "scale=2; $sum / ${#samples[@]}" | bc)
        
        # Check if TPS meets requirement
        local tps_ok=false
        if [ "$(echo "$tps >= $MIN_TPS" | bc)" -eq 1 ]; then
            tps_ok=true
        else
            tps_failures=$((tps_failures + 1))
        fi
        
        # Display status
        local status_icon="✅"
        if [ "$tps_ok" = false ]; then
            status_icon="❌"
        fi
        
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] $status_icon TPS: $tps tokens/s (avg: $avg_tps tokens/s over ${#samples[@]} samples)"
        echo "           Tokens: $tokens, Time: ${elapsed}s"
        
        if [ "$tps_ok" = false ]; then
            echo "           ⚠️  WARNING: TPS below minimum requirement of $MIN_TPS tokens/s (failure #$tps_failures)"
        fi
        
        # Summary every 10 checks
        if [ $((sample_count % 10)) -eq 0 ] && [ $sample_count -gt 0 ]; then
            echo ""
            echo "--- Summary (last 10 checks) ---"
            echo "  TPS Failures: $tps_failures"
            echo "  Swap Failures: $swap_failures"
            echo "  Average TPS: $avg_tps tokens/s"
            echo ""
        fi
        
        sample_count=$((sample_count + 1))
        sleep $CHECK_INTERVAL
    done
}

# Main execution
if ! command -v bc &> /dev/null; then
    echo "❌ ERROR: 'bc' command not found. Please install it: brew install bc"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: 'python3' command not found"
    exit 1
fi

# Check if Master API is accessible
if ! curl -s --max-time 5 "$MASTER_API_URL/state" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot reach Master API at $MASTER_API_URL"
    echo "Please ensure the Master is running and accessible"
    exit 1
fi

# Start monitoring
monitor_continuous


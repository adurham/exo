#!/bin/bash
set -e

# Add SSH options for Tailscale IPs
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# Static cluster configuration - using Tailscale IPs
MASTER_NODE="100.67.156.10"
WORKER_NODES=("100.93.253.67" "100.80.147.125" "100.82.48.77")
WORKER_HOSTNAMES=("adams-mac-studio-m4" "adams-macbook-pro-m4" "adams-work-macbook-pro-m4")
WORKER_NODE_IDS=("static-worker-0-adams-mac-studio-m4" "static-worker-1-adams-macbook-pro-m4" "static-worker-2-adams-work-macbook-pro-m4")
MASTER_API_URL="http://100.67.156.10:52415"
MODEL_PATH="\$HOME/.exo/models/mlx-community--Qwen3-235B-A22B-Instruct-2507-4bit"

# Function to prefix output with node name
prefix_output() {
    local node=$1
    while IFS= read -r line; do
        echo "[$node] $line"
    done
}

# Function to check swap usage on a node (macOS)
check_swap() {
    local node=$1
    # On macOS, use sysctl vm.swapusage - it shows "used = XM" format
    # Extract just the numeric value (handles "0.00M" format)
    ssh $SSH_OPTS "$node" "sysctl vm.swapusage 2>/dev/null | awk -F'used = ' '{print \$2}' | awk '{print \$1}' | sed 's/M//' | sed 's/G/000/' | head -1" 2>/dev/null || echo "0.00"
}

# Function to verify zero swap usage
verify_zero_swap() {
    local node=$1
    local swap_used_str=$(check_swap "$node")
    # Convert to numeric value using bc (handles "0.00" format)
    local swap_used=$(echo "$swap_used_str" | bc 2>/dev/null || echo "0")
    # Allow up to 1MB swap (essentially zero, macOS may report tiny amounts)
    local swap_threshold=1.0
    # Check if swap_used is greater than threshold (using bc for floating point comparison)
    if command -v bc &> /dev/null; then
        if [ "$(echo "$swap_used <= $swap_threshold" | bc 2>/dev/null)" = "1" ]; then
            echo "✅ Swap usage verified: ${swap_used_str}MB on $node (below ${swap_threshold}MB threshold)"
            return 0
        fi
    else
        # Fallback: check if it's "0" or empty or very small
        if [ -z "$swap_used_str" ] || [ "$swap_used_str" = "0" ] || [ "$swap_used_str" = "0.00" ]; then
            echo "✅ Swap usage verified: zero on $node"
            return 0
        fi
    fi
    echo "❌ ERROR: Swap usage detected on $node: ${swap_used_str}MB (threshold: ${swap_threshold}MB)"
    return 1
}

# Function to verify model exists
verify_model() {
    local node=$1
    # Use HOME variable expansion in the SSH command
    if ssh $SSH_OPTS "$node" "test -d \$HOME/.exo/models/mlx-community--Qwen3-235B-A22B-Instruct-2507-4bit" 2>/dev/null; then
        echo "✅ Model found at \$HOME/.exo/models/mlx-community--Qwen3-235B-A22B-Instruct-2507-4bit on $node"
        return 0
    else
        echo "❌ ERROR: Model not found at \$HOME/.exo/models/mlx-community--Qwen3-235B-A22B-Instruct-2507-4bit on $node"
        return 1
    fi
}

# Function to check if Master API is responding
check_master_api() {
    local response=$(curl -s --max-time 5 "$MASTER_API_URL/state" 2>/dev/null || echo "")
    if [ -z "$response" ]; then
        return 1
    fi
    return 0
}

# Function to check if all workers are registered
check_workers_registered() {
    local expected_workers=3
    local response=$(curl -s --max-time 5 "$MASTER_API_URL/state" 2>/dev/null || echo "")
    if [ -z "$response" ]; then
        return 1
    fi
    # Count worker nodes in topology (excluding master)
    local worker_count=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    nodes = data.get('topology', {}).get('nodes', [])
    # Workers are nodes with 'static-worker' in node_id
    workers = [n for n in nodes if 'static-worker' in str(n.get('nodeId', ''))]
    print(len(workers))
except:
    print(0)
" 2>/dev/null || echo "0")
    if [ "$worker_count" -ge "$expected_workers" ]; then
        return 0
    fi
    return 1
}

echo "========================================="
echo "Deploying Static 4-Node MLX RDMA Cluster"
echo "========================================="
echo ""
echo "NOTE: Make sure to run scripts/build-and-distribute.sh first"
echo "      to pre-build and distribute Rust components to all nodes"
echo ""

# Ask user if they want to skip build/distribution (for faster re-deployments)
SKIP_BUILD="${SKIP_BUILD:-no}"
if [ "$SKIP_BUILD" != "yes" ]; then
    read -p "Have you run build-and-distribute.sh? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Please run: ./scripts/build-and-distribute.sh"
        exit 1
    fi
fi
echo ""


# Step 1: Verify model exists on all worker nodes
echo "Step 1: Verifying model location on worker nodes..."
model_ok=true
for node in "${WORKER_NODES[@]}"; do
    if ! verify_model "$node"; then
        model_ok=false
    fi
done
if [ "$model_ok" = false ]; then
    echo "❌ ERROR: Model verification failed. Please ensure model exists on all worker nodes."
    exit 1
fi
echo ""

# Step 2: Check swap usage on all nodes (must be zero)
echo "Step 2: Checking swap usage on all nodes..."
swap_ok=true
for node in "$MASTER_NODE" "${WORKER_NODES[@]}"; do
    if ! verify_zero_swap "$node"; then
        swap_ok=false
    fi
done
if [ "$swap_ok" = false ]; then
    echo "❌ ERROR: Swap usage detected. This will kill performance. Please free up memory."
    exit 1
fi
echo ""

# Step 3: Kill any existing exo processes
echo "Step 3: Stopping any existing exo processes..."
for node in "$MASTER_NODE" "${WORKER_NODES[@]}"; do
    echo "[$node] Stopping existing processes..."
    ssh $SSH_OPTS "$node" "pkill -9 -f 'exo.*master_app' 2>/dev/null || true; pkill -9 -f 'exo.*worker_app' 2>/dev/null || true; pkill -9 -f 'uv run.*master_app' 2>/dev/null || true; pkill -9 -f 'uv run.*worker_app' 2>/dev/null || true" 2>&1 | prefix_output "$node" || true
done
sleep 2
echo ""

# Step 4: Start Master
echo "Step 4: Starting Master on $MASTER_NODE..."
# Use explicit uv path (macOS Homebrew location)
ssh $SSH_OPTS "$MASTER_NODE" "cd ~/repos/exo && export PATH=\"/opt/homebrew/bin:\$HOME/.local/bin:\$PATH\" && nohup bash -c 'uv run python -m exo.master_app' > ~/.exo/master.log 2>&1 &" 2>&1 | prefix_output "$MASTER_NODE" || true
sleep 3

# Wait for Master to be ready
echo "Waiting for Master API to be ready..."
MAX_WAIT=30
ELAPSED=0
INTERVAL=1
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if check_master_api; then
        echo "✅ Master API is ready"
        break
    fi
    echo "Waiting for Master API... ($ELAPSED/$MAX_WAIT seconds)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if ! check_master_api; then
    echo "❌ ERROR: Master API did not become ready within $MAX_WAIT seconds"
    echo "Check logs: ssh $MASTER_NODE 'tail -n 50 ~/.exo/master.log'"
    exit 1
fi
echo ""

# Step 5: Start Workers
echo "Step 5: Starting Workers on all worker nodes..."
for i in "${!WORKER_NODES[@]}"; do
    node="${WORKER_NODES[$i]}"
    node_id="${WORKER_NODE_IDS[$i]}"
    echo "[$node] Starting Worker with node_id=$node_id..."
    ssh $SSH_OPTS "$node" "cd ~/repos/exo && export PATH=\"/opt/homebrew/bin:\$HOME/.local/bin:\$PATH\" && nohup bash -c 'uv run python -m exo.worker_app --node-id $node_id' > ~/.exo/worker.log 2>&1 &" 2>&1 | prefix_output "$node" || true
done
sleep 5
echo ""

# Step 6: Verify all workers are registered
echo "Step 6: Verifying all workers are registered with Master..."
MAX_WAIT=60
ELAPSED=0
INTERVAL=2
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if check_workers_registered; then
        echo "✅ All workers are registered!"
        break
    fi
    echo "Waiting for workers to register... ($ELAPSED/$MAX_WAIT seconds)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if ! check_workers_registered; then
    echo "❌ ERROR: Not all workers registered within $MAX_WAIT seconds"
    echo "Checking current status..."
    curl -s "$MASTER_API_URL/state" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    nodes = data.get('topology', {}).get('nodes', [])
    print(f'Registered nodes: {len(nodes)}')
    for n in nodes:
        node_id = str(n.get('nodeId', 'unknown'))
        print(f'  - {node_id[:50]}...')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null || echo "Could not check node status"
    exit 1
fi
echo ""

# Step 7: Final swap check
echo "Step 7: Final swap usage check..."
swap_ok=true
for node in "$MASTER_NODE" "${WORKER_NODES[@]}"; do
    if ! verify_zero_swap "$node"; then
        swap_ok=false
    fi
done
if [ "$swap_ok" = false ]; then
    echo "❌ WARNING: Swap usage detected after deployment. Performance may be degraded."
    # Don't exit, just warn
fi
echo ""

echo "========================================="
echo "✅ Deployment Complete!"
echo "========================================="
echo "Master API: $MASTER_API_URL"
echo "Workers: ${WORKER_NODES[*]}"
echo ""
echo "To check logs:"
echo "  Master: ssh $MASTER_NODE 'tail -f ~/.exo/master.log'"
echo "  Worker: ssh <worker-node> 'tail -f ~/.exo/worker.log'"
echo ""
echo "To check status:"
echo "  curl $MASTER_API_URL/state"
echo ""


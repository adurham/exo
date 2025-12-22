#!/bin/bash
set -e

sleep 10

NODES=("macstudio-m4" "macbook-m4" "work-macbook-m4")

# API endpoint (assuming macstudio-m4 is the master)
API_URL="http://100.93.253.67:52415/state"

# Function to prefix output with node name
prefix_output() {
    local node=$1
    while IFS= read -r line; do
        echo "[$node] $line"
    done
}

# Function to check if all nodes are connected
check_nodes_connected() {
    local expected_nodes=3
    local response=$(curl -s --max-time 5 "$API_URL" 2>/dev/null || echo "")
    if [ -z "$response" ]; then
        return 1
    fi
    local node_count=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(len(d.get('topology', {}).get('nodes', [])))" 2>/dev/null || echo "0")
    if [ "$node_count" -ge "$expected_nodes" ]; then
        return 0
    fi
    return 1
}

# Deploy to all nodes in parallel, prefixing each line with node name
for node in "${NODES[@]}"; do
    echo "[$node] Deploying..."
    ssh "$node" "cd ~/repos/exo/ && git fetch && git reset --hard origin/new_main && bash deploy.sh" 2>&1 | prefix_output "$node" &
done

# Wait for all background jobs to complete
wait

echo ""
echo "Deployment to all nodes complete!"
echo ""
echo "Waiting for all 3 nodes to connect (up to 60 seconds)..."

# Wait up to 60 seconds for all nodes to connect
MAX_WAIT=60
ELAPSED=0
INTERVAL=2

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if check_nodes_connected; then
        echo "✅ All 3 nodes are connected!"
        exit 0
    fi
    echo "Waiting for nodes to connect... ($ELAPSED/$MAX_WAIT seconds)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "❌ ERROR: Not all 3 nodes connected within $MAX_WAIT seconds"
echo "Checking current node status..."
curl -s "$API_URL" | python3 -c "import sys, json; d=json.load(sys.stdin); nodes = d.get('topology', {}).get('nodes', []); print(f'Connected nodes: {len(nodes)}'); [print(f'  - {n.get(\"nodeId\", \"unknown\")[:30]}...') for n in nodes]" 2>/dev/null || echo "Could not check node status"
exit 1


#!/bin/bash
set -e

# Test networking between Master and Workers
# Step 1: Basic connectivity test

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
MASTER_NODE="100.67.156.10"
WORKER_NODES=("100.93.253.67" "100.80.147.125" "100.82.48.77")
MASTER_API_URL="http://100.67.156.10:52415"

echo "========================================="
echo "Step 1: Testing Basic Networking"
echo "========================================="
echo ""

# Test Master API
echo "Testing Master API..."
if curl -s --max-time 5 "$MASTER_API_URL/state" > /dev/null 2>&1; then
    echo "✅ Master API is reachable"
else
    echo "❌ Master API is not reachable"
    exit 1
fi

# Test SSH connectivity
echo ""
echo "Testing SSH connectivity to all nodes..."
all_ok=true
for node in "$MASTER_NODE" "${WORKER_NODES[@]}"; do
    if ssh $SSH_OPTS "$node" "echo 'OK'" > /dev/null 2>&1; then
        echo "✅ SSH to $node works"
    else
        echo "❌ SSH to $node failed"
        all_ok=false
    fi
done

if [ "$all_ok" = false ]; then
    echo "❌ Some SSH connections failed"
    exit 1
fi

# Test HTTP from workers to master
echo ""
echo "Testing HTTP connectivity from workers to master..."
all_ok=true
for node in "${WORKER_NODES[@]}"; do
    if ssh $SSH_OPTS "$node" "curl -s --max-time 5 $MASTER_API_URL/state > /dev/null 2>&1" 2>&1; then
        echo "✅ Worker $node can reach Master API"
    else
        echo "❌ Worker $node cannot reach Master API"
        all_ok=false
    fi
done

if [ "$all_ok" = false ]; then
    echo "❌ Some workers cannot reach Master API"
    exit 1
fi

echo ""
echo "✅ All basic networking tests passed!"


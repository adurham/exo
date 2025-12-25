#!/bin/bash
set -e

# Test RDMA connectivity between worker nodes over Thunderbolt
# Uses ibverbs tools to check RDMA device connectivity

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
WORKER_NODES=("100.93.253.67" "100.80.147.125" "100.82.48.77")
WORKER_NODE_IDS=("static-worker-0-adams-mac-studio-m4" "static-worker-1-adams-macbook-pro-m4" "static-worker-2-adams-work-macbook-pro-m4")

# Thunderbolt IP mappings
# Rank 0 -> Rank 1: 192.168.202.1 -> 192.168.202.2
# Rank 0 -> Rank 2: 192.168.203.1 -> 192.168.203.2
# Rank 1 -> Rank 2: 192.168.205.1 -> 192.168.205.2

echo "========================================="
echo "Step 2: Testing RDMA Connectivity"
echo "========================================="
echo ""

# Test 1: Check if ibverbs/rdma tools are available
echo "Checking for RDMA tools on worker nodes..."
all_have_ibverbs=true
for node in "${WORKER_NODES[@]}"; do
    if ssh $SSH_OPTS "$node" "which ibv_devices > /dev/null 2>&1" 2>/dev/null; then
        echo "✅ $node has ibverbs tools"
    else
        echo "❌ $node missing ibverbs tools (ibv_devices not found)"
        all_have_ibverbs=false
    fi
done

if [ "$all_have_ibverbs" = false ]; then
    echo "⚠️  Warning: Some nodes missing ibverbs tools. RDMA may not be available."
    echo "   Continuing with basic connectivity tests..."
fi

# Test 2: List available RDMA devices
echo ""
echo "Listing RDMA devices on each worker..."
for i in "${!WORKER_NODES[@]}"; do
    node="${WORKER_NODES[$i]}"
    node_id="${WORKER_NODE_IDS[$i]}"
    echo "[$node] RDMA devices:"
    ssh $SSH_OPTS "$node" "ibv_devices 2>/dev/null || echo '  (ibv_devices not available)'" 2>&1 | sed "s/^/  /" || true
done

# Test 3: Check Thunderbolt interface connectivity via ping
echo ""
echo "Testing Thunderbolt interface connectivity..."
echo "Rank 0 -> Rank 1 (192.168.202.1 -> 192.168.202.2):"
if ssh $SSH_OPTS "${WORKER_NODES[0]}" "ping -c 2 -W 1 192.168.202.2 > /dev/null 2>&1" 2>&1; then
    echo "✅ Rank 0 can ping Rank 1 via Thunderbolt"
else
    echo "❌ Rank 0 cannot ping Rank 1 via Thunderbolt"
fi

echo "Rank 0 -> Rank 2 (192.168.203.1 -> 192.168.203.2):"
if ssh $SSH_OPTS "${WORKER_NODES[0]}" "ping -c 2 -W 1 192.168.203.2 > /dev/null 2>&1" 2>&1; then
    echo "✅ Rank 0 can ping Rank 2 via Thunderbolt"
else
    echo "❌ Rank 0 cannot ping Rank 2 via Thunderbolt"
fi

echo "Rank 1 -> Rank 2 (192.168.205.1 -> 192.168.205.2):"
if ssh $SSH_OPTS "${WORKER_NODES[1]}" "ping -c 2 -W 1 192.168.205.2 > /dev/null 2>&1" 2>&1; then
    echo "✅ Rank 1 can ping Rank 2 via Thunderbolt"
else
    echo "❌ Rank 1 cannot ping Rank 2 via Thunderbolt"
fi

# Test 4: Check if Thunderbolt interfaces are configured
echo ""
echo "Checking Thunderbolt interface configuration..."
for i in "${!WORKER_NODES[@]}"; do
    node="${WORKER_NODES[$i]}"
    rank=$i
    echo "[Rank $rank] Thunderbolt interfaces:"
    ssh $SSH_OPTS "$node" "ifconfig | grep -E 'inet.*192\.168\.(202|203|205)' || echo '  (No Thunderbolt IPs found)'" 2>&1 | sed "s/^/  /" || true
done

echo ""
echo "✅ RDMA connectivity tests completed!"
echo ""
echo "Note: Full RDMA functionality will be tested when workers start and attempt MLX distributed initialization."


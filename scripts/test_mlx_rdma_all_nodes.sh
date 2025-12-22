#!/bin/bash
# Test MLX RDMA on all nodes simultaneously
# This script tests if MLX can initialize RDMA on each node

set -e

# Configuration
RANK_0_IP="192.168.202.1"  # Mac Studio Thunderbolt IP
COORDINATOR_PORT=52414
WORLD_SIZE=3

# Node configurations (rank, hostname)
declare -a NODES=(
    "0:macstudio-m4"
    "1:macbook-m4"
    "2:work-macbook-m4"
)

echo "=========================================="
echo "Testing MLX RDMA on all nodes"
echo "=========================================="
echo "Coordinator IP: ${RANK_0_IP}:${COORDINATOR_PORT}"
echo "World Size: ${WORLD_SIZE}"
echo ""

# Test each node
for node_config in "${NODES[@]}"; do
    IFS=':' read -r rank hostname <<< "$node_config"
    echo "----------------------------------------"
    echo "Testing rank ${rank} on ${hostname}"
    echo "----------------------------------------"
    
    ssh "${hostname}" "cd /Users/adam.durham/repos/exo && \
        uv run python scripts/test_mlx_rdma.py \
            --rank ${rank} \
            --world-size ${WORLD_SIZE} \
            --coordinator-ip ${RANK_0_IP} \
            --coordinator-port ${COORDINATOR_PORT} \
            --devices-file ./test_hosts_{rank}.json" 2>&1 | \
        sed "s/^/[${hostname}] /" || {
        echo "[${hostname}] ✗ Test failed or node unreachable"
    }
    
    echo ""
done

echo "=========================================="
echo "All tests complete"
echo "=========================================="


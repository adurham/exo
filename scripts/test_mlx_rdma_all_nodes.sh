#!/bin/bash
# Test MLX RDMA on all nodes simultaneously
# This script tests if MLX can initialize RDMA on each node
# IMPORTANT: All nodes must initialize at roughly the same time for RDMA to work

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
echo "Testing MLX RDMA on all nodes (simultaneously)"
echo "=========================================="
echo "Coordinator IP: ${RANK_0_IP}:${COORDINATOR_PORT}"
echo "World Size: ${WORLD_SIZE}"
echo ""
echo "NOTE: All nodes will initialize at the same time"
echo "      This is required for RDMA to work"
echo ""

# Start all tests in background
declare -a PIDS=()
for node_config in "${NODES[@]}"; do
    IFS=':' read -r rank hostname <<< "$node_config"
    echo "Starting test for rank ${rank} on ${hostname}..."
    
    (
        ssh "${hostname}" "cd /Users/adam.durham/repos/exo && \
            .venv/bin/python scripts/test_mlx_rdma.py \
                --rank ${rank} \
                --world-size ${WORLD_SIZE} \
                --coordinator-ip ${RANK_0_IP} \
                --coordinator-port ${COORDINATOR_PORT} \
                --devices-file ./test_hosts_{rank}.json" 2>&1 | \
            sed "s/^/[${hostname}] /"
    ) &
    PIDS+=($!)
done

echo ""
echo "Waiting for all tests to complete..."
echo ""

# Wait for all background jobs
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    IFS=':' read -r rank hostname <<< "${NODES[$i]}"
    if wait $pid; then
        echo "[${hostname}] ✓ Test completed successfully"
    else
        echo "[${hostname}] ✗ Test failed"
        FAILED=1
    fi
done

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "✗ Some tests failed"
fi
echo "=========================================="

exit $FAILED


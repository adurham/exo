#!/usr/bin/env bash
# Run the RDMA fence stress test on both Mac Studios.
# Usage: ./tests/run_rdma_fence_stress.sh
#
# Prerequisites:
#   - SSH access to macstudio-m4-1 and macstudio-m4-2
#   - Both Studios have origin/main checked out and built
#   - The .venv on each Studio has mlx installed from the local submodule

set -euo pipefail

STUDIO1="macstudio-m4-1"
STUDIO2="macstudio-m4-2"
REPO_DIR="\$HOME/repos/exo"
COORDINATOR_PORT=59989

echo "=== RDMA Fence Stress Test ==="
echo "Pulling latest on both Studios..."
ssh "$STUDIO1" "cd $REPO_DIR && git pull --recurse-submodules && uv pip install -e ./mlx --no-build-isolation" &
ssh "$STUDIO2" "cd $REPO_DIR && git pull --recurse-submodules && uv pip install -e ./mlx --no-build-isolation" &
wait
echo "Build complete on both Studios."
echo ""

# Get Studio 1's IP for the coordinator
STUDIO1_IP=$(ssh "$STUDIO1" "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "192.168.86.201")

echo "Starting test (coordinator=$STUDIO1_IP:$COORDINATOR_PORT)..."
echo "---"

# Launch both ranks in parallel, stream output with rank prefix
ssh "$STUDIO1" "cd $REPO_DIR && RANK=0 COORDINATOR=0.0.0.0:$COORDINATOR_PORT .venv/bin/python tests/test_rdma_fence_stress.py" 2>&1 | sed 's/^/[studio1] /' &
PID1=$!

# Small delay so rank 0 starts the coordinator first
sleep 1

ssh "$STUDIO2" "cd $REPO_DIR && RANK=1 COORDINATOR=$STUDIO1_IP:$COORDINATOR_PORT .venv/bin/python tests/test_rdma_fence_stress.py" 2>&1 | sed 's/^/[studio2] /' &
PID2=$!

# Wait for both
FAIL=0
wait $PID1 || FAIL=1
wait $PID2 || FAIL=1

echo ""
if [ $FAIL -eq 0 ]; then
    echo "=== ALL PASSED ==="
else
    echo "=== SOME TESTS FAILED ==="
    exit 1
fi

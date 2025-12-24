#!/bin/bash
set -e

# Build and distribute script for static cluster
# Pre-builds Rust components locally and distributes to nodes

MASTER_NODE="100.67.156.10"
WORKER_NODES=("100.93.253.67" "100.80.147.125" "100.82.48.77")
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
REPO_PATH="~/repos/exo"

echo "========================================="
echo "Building Rust Components Locally"
echo "========================================="

# Build the project locally
echo "Building exo with uv..."
cd "$(dirname "$0")/.."
uv build 2>&1 | tail -50

# Find the built wheel
WHEEL_PATH=$(find . -name "exo-*.whl" -type f | head -1)
if [ -z "$WHEEL_PATH" ]; then
    echo "❌ ERROR: Could not find built wheel"
    exit 1
fi

echo "✅ Built wheel: $WHEEL_PATH"

# Create a temporary directory for distribution
DIST_DIR="/tmp/exo-build-$$"
mkdir -p "$DIST_DIR"
cp "$WHEEL_PATH" "$DIST_DIR/"

echo ""
echo "========================================="
echo "Distributing to Nodes"
echo "========================================="

# Function to distribute to a node
distribute_to_node() {
    local node=$1
    local node_type=$2
    
    echo "[$node] Distributing wheel to $node_type node..."
    
    # Create directory on remote node
    ssh $SSH_OPTS "$node" "mkdir -p $REPO_PATH/dist" 2>&1
    
    # Copy wheel to node
    scp $SSH_OPTS "$DIST_DIR"/*.whl "$node:$REPO_PATH/dist/" 2>&1
    
    # Install the wheel on the remote node
    ssh $SSH_OPTS "$node" "cd $REPO_PATH && export PATH=\"/opt/homebrew/bin:\$HOME/.local/bin:\$PATH\" && uv pip install --force-reinstall --no-deps dist/*.whl" 2>&1 | sed "s/^/[$node] /"
    
    echo "✅ [$node] Distribution complete"
}

# Distribute to master
distribute_to_node "$MASTER_NODE" "master"

# Distribute to workers
for node in "${WORKER_NODES[@]}"; do
    distribute_to_node "$node" "worker"
done

# Cleanup
rm -rf "$DIST_DIR"

echo ""
echo "========================================="
echo "✅ Build and Distribution Complete!"
echo "========================================="


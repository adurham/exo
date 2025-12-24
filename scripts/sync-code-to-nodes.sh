#!/bin/bash
set -e

# Sync code to nodes script - copies source files to all nodes
MASTER_NODE="100.67.156.10"
WORKER_NODES=("100.93.253.67" "100.80.147.125" "100.82.48.77")
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
REPO_PATH="~/repos/exo"

# Function to sync code to a node
sync_to_node() {
    local node=$1
    echo "[$node] Syncing code..."
    
    # Create repo directory if it doesn't exist
    ssh $SSH_OPTS "$node" "mkdir -p $REPO_PATH/src" 2>&1
    
    # Copy source files
    scp $SSH_OPTS -r src/ "$node:$REPO_PATH/" 2>&1
    
    # Copy pyproject.toml and other config files
    scp $SSH_OPTS pyproject.toml "$node:$REPO_PATH/" 2>&1 || true
    scp $SSH_OPTS README.md "$node:$REPO_PATH/" 2>&1 || true
    
    echo "✅ [$node] Code synced"
}

echo "========================================="
echo "Syncing Code to All Nodes"
echo "========================================="

# Sync to master
sync_to_node "$MASTER_NODE"

# Sync to workers
for node in "${WORKER_NODES[@]}"; do
    sync_to_node "$node"
done

echo ""
echo "✅ Code sync complete!"


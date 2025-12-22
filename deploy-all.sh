#!/bin/bash
set -e

NODES=("macstudio-m4" "macbook-m4" "work-macbook-m4")

for node in "${NODES[@]}"; do
    echo "========================================="
    echo "Deploying to $node..."
    echo "========================================="
    ssh "$node" "cd ~/repos/exo/ && git pull && bash deploy.sh" || {
        echo "ERROR: Deployment to $node failed!"
        exit 1
    }
    echo ""
done

echo "========================================="
echo "Deployment to all nodes complete!"
echo "========================================="


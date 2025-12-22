#!/bin/bash
set -e

NODES=("macstudio-m4" "macbook-m4" "work-macbook-m4")

# Deploy to all nodes in parallel
for node in "${NODES[@]}"; do
    echo "========================================="
    echo "Deploying to $node..."
    echo "========================================="
    ssh "$node" "cd ~/repos/exo/ && git pull && bash deploy.sh" &
done

# Wait for all background jobs to complete
wait

echo ""
echo "========================================="
echo "Deployment to all nodes complete!"
echo "========================================="


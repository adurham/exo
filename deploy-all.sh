#!/bin/bash
set -e

NODES=("macstudio-m4" "macbook-m4" "work-macbook-m4")

# Function to prefix output with node name
prefix_output() {
    local node=$1
    while IFS= read -r line; do
        echo "[$node] $line"
    done
}

# Deploy to all nodes in parallel, prefixing each line with node name
for node in "${NODES[@]}"; do
    echo "========================================="
    echo "Deploying to $node..."
    echo "========================================="
    ssh "$node" "cd ~/repos/exo/ && git pull && bash deploy.sh" 2>&1 | prefix_output "$node" &
done

# Wait for all background jobs to complete
wait

echo ""
echo "========================================="
echo "Deployment to all nodes complete!"
echo "========================================="


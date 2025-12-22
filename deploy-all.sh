#!/bin/bash
set -e

sleep 10

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
    echo "[$node] Deploying..."
    ssh "$node" "cd ~/repos/exo/ && git fetch && git reset --hard origin/new_main && bash deploy.sh" 2>&1 | prefix_output "$node" &
done

# Wait for all background jobs to complete
wait

echo ""
echo "Deployment to all nodes complete!"


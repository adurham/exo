#!/bin/bash
set -e

NODES=("macstudio-m4" "macbook-m4" "work-macbook-m4")
LOG_DIR="$HOME/.exo/deploy-logs"
mkdir -p "$LOG_DIR"

# Deploy to all nodes in parallel, each logging to its own file
for node in "${NODES[@]}"; do
    log_file="$LOG_DIR/deploy-${node}-$(date +%Y%m%d-%H%M%S).log"
    echo "========================================="
    echo "Deploying to $node (log: $log_file)..."
    echo "========================================="
    ssh "$node" "cd ~/repos/exo/ && git pull && bash deploy.sh" > "$log_file" 2>&1 &
done

# Wait for all background jobs to complete
wait

echo ""
echo "========================================="
echo "Deployment to all nodes complete!"
echo "Logs saved to: $LOG_DIR"
echo "========================================="


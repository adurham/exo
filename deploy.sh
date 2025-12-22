#!/bin/bash
set -e

# Source uv environment if available
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
fi

# Add common paths
export PATH="$HOME/.local/bin:$PATH"

cd ~/repos/exo/

# Kill any existing exo processes
echo "Stopping any existing exo processes..."
pkill -f "uv run exo" || pkill -f "exo" || echo "No existing exo processes found"
sleep 2

# Reset to clean state and pull latest
echo "Resetting to clean state..."
git reset --hard HEAD
git clean -fd
git fetch
git reset --hard origin/new_main

# Build dashboard if npm is available, otherwise skip
if command -v npm &> /dev/null; then
    echo "Building dashboard..."
    cd dashboard
    npm install
    npm run build
    cd ..
else
    echo "Warning: npm not found, skipping dashboard build (using existing build if available)"
fi

# Run exo in background (logging to ~/.exo/exo.log per node)
echo "Starting exo in background..."
nohup uv run exo > /dev/null 2>&1 &
echo "Exo started with PID: $!"
echo "Logs are being written to ~/.exo/exo.log on this node"


#!/bin/bash

# Standardized Exo Cluster Startup Script
# Usage: ./start_cluster.sh
# Detects the current host and sets up the appropriate environment for the 2-node M4 cluster.

# Define Node Constants
M4_1_IP="192.168.86.201"
M4_1_PEER_ID="12D3KooWDGQKAJUYpqTHzBhVpGzYxQagWRwFqJPzkEYzHxt3SSUg"
M4_2_IP="192.168.86.202"
M4_2_PEER_ID="12D3KooWQDzFqvjsgFRfheeV7uvtVUP1gruphpgoVELP9pkHBses"

# Get current IP (simple check, assuming en0 or similar primary interface)
CURRENT_IP=$(ipconfig getifaddr en0)

if [ "$CURRENT_IP" == "$M4_1_IP" ]; then
    echo "Detected M4-1 ($M4_1_IP)"
    # Peer with M4-2
    export EXO_DISCOVERY_PEERS="/ip4/$M4_2_IP/tcp/52415/p2p/$M4_2_PEER_ID"
elif [ "$CURRENT_IP" == "$M4_2_IP" ]; then
    echo "Detected M4-2 ($M4_2_IP)"
    # Peer with M4-1
    export EXO_DISCOVERY_PEERS="/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID"
else
    echo "Unknown host IP: $CURRENT_IP. Defaulting to standalone mode or check network settings."
    # Optional: Exit or continue without peers
fi

# Check for macmon (required for UI stats)
MACMON_PATH="$HOME/.cargo/bin/macmon"
if [ ! -f "$MACMON_PATH" ]; then
    echo "WARNING: macmon binary not found at $MACMON_PATH"
    echo "UI system stats will be missing. Please install macmon."
else
    echo "macmon detected at $MACMON_PATH"
fi

echo "Starting Exo with EXO_DISCOVERY_PEERS=$EXO_DISCOVERY_PEERS"

# Launch Exo in background slightly differently if needed, or just foreground
# Using nohup pattern from manual steps for persistence if running in terminal
# nohup python3 -m exo.main > /tmp/exo.log 2>&1 &
# But for a script, usually we want to see output or manage it. 
# Let's run it directly so it can be Ctrl-C'd or managed by the user designated runner.
# If the user wants background, they can run the script with &

# Build Dashboard (as requested to refresh UI assets)
if [ -d "dashboard" ]; then
    echo "Building dashboard..."
    
    # Try to load NVM if present
    export NVM_DIR="$HOME/.nvm"
    if [ -s "$NVM_DIR/nvm.sh" ]; then
        echo "Loading NVM..."
        source "$NVM_DIR/nvm.sh"
    fi

    cd dashboard
    # Check if npm is available
    if command -v npm &> /dev/null; then
        npm install && npm run build
    else
        echo "WARNING: npm not found. Skipping dashboard build."
    fi
    cd ..
else
    echo "WARNING: dashboard directory not found. Skipping build."
fi

# Activate venv if it exists in the repo
SOURCE_DIR=$(dirname "$0")
if [ -d "$SOURCE_DIR/.venv" ]; then
    source "$SOURCE_DIR/.venv/bin/activate"
fi

# Run Exo from source to pick up local changes
# If uv is available, use it to run python directly
# FORCE local resources to be used
export EXO_RESOURCES_DIR="$PWD/resources"
echo "Using resources from: $EXO_RESOURCES_DIR"

if command -v uv &> /dev/null; then
  echo "Starting Exo (via uv run python -m exo.main)..."
  uv run python3 -m exo.main
else
  echo "Starting Exo (via python3 -m exo.main)..."
  python3 -m exo.main
fi

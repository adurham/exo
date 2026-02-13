#!/bin/bash

# Standardized Exo Cluster Startup Script
# Usage: ./start_cluster.sh
# Detects the current host and sets up the appropriate environment for the 2-node M4 cluster.

# Define Node Constants
M4_1_IP="192.168.86.201"
M4_1_PEER_ID="12D3KooWDGQKAJUYpqTHzBhVpGzYxQagWRwFqJPzkEYzHxt3SSUg"
M4_2_IP="192.168.86.202"
M4_2_PEER_ID="12D3KooWQDzFqvjsgFRfheeV7uvtVUP1gruphpgoVELP9pkHBses"

# Get current IPs (check all interfaces to correctly identify the node)
CURRENT_IPS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}')

IS_M4_1=false
IS_M4_2=false

for IP in $CURRENT_IPS; do
    if [ "$IP" == "$M4_1_IP" ]; then
        IS_M4_1=true
        break
    fi
    if [ "$IP" == "$M4_2_IP" ]; then
        IS_M4_2=true
        break
    fi
done

if [ "$IS_M4_1" = true ]; then
    echo "Detected M4-1 ($M4_1_IP)"
    # Peer with M4-2
    export EXO_DISCOVERY_PEERS="/ip4/$M4_2_IP/tcp/52415/p2p/$M4_2_PEER_ID"
elif [ "$IS_M4_2" = true ]; then
    echo "Detected M4-2 ($M4_2_IP)"
    # Peer with M4-1
    export EXO_DISCOVERY_PEERS="/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID"
else
    echo "Unknown host IPs: $CURRENT_IPS. Assuming remote controller."
    echo "Attempting to start cluster on known nodes via SSH..."
    echo "-----------------------------------------------------"

    # Define nodes to start (using SSH config aliases)
    NODES=("macstudio-m4-1" "macstudio-m4-2")

    # 1. Kill existing processes, git pull, and force reinstall bindings (rebuilds Rust bindings)
    for NODE in "${NODES[@]}"; do
        echo "Preparing $NODE..."
        # Use zsh -l -c to ensure environment (PATH, etc.) is loaded
        # Clean target directory to force fresh build and reinstall bindings
        ssh "$NODE" "zsh -l -c 'pkill -f \"exo.main\" || true; cd ~/repos/exo && git pull && rm -rf rust/exo_pyo3_bindings/target && uv pip install --force-reinstall ./rust/exo_pyo3_bindings && uv pip install -e .'"
    done

    # 2. Start Exo on each node
    for NODE in "${NODES[@]}"; do
        echo "Starting Exo on $NODE..."
        # Use nohup to keep running after disconnection
        # Executing the script via zsh -l to ensure it picks up the correct IP logic and env
        ssh "$NODE" "zsh -l -c 'nohup ~/repos/exo/start_cluster.sh > /tmp/exo.log 2>&1 &'"
        if [ $? -eq 0 ]; then
            echo "Successfully triggered start on $NODE."
        else
            echo "Failed to trigger start on $NODE."
        fi
    done

    echo "Cluster start commands issued. Waiting for cluster to stabilize..."
    
    # 3. Health Check / Topology Verification
    # We will poll the topology via the first node (assuming it's up) until we see both nodes.
    MAX_RETRIES=30
    RETRY_DELAY=2
    
    # Get the LAN IP for the first node from SSH config (or use known IP for check)
    CHECK_URL="http://192.168.86.201:52415/topology" 

    for ((i=1; i<=MAX_RETRIES; i++)); do
        # echo "Checking cluster topology ($i/$MAX_RETRIES)..."
        
        RESPONSE=$(curl -s "$CHECK_URL")
        
        if [ -n "$RESPONSE" ]; then
            # Check if both M4_1_PEER_ID and M4_2_PEER_ID are present in the response
            if echo "$RESPONSE" | grep -q "$M4_1_PEER_ID" && echo "$RESPONSE" | grep -q "$M4_2_PEER_ID"; then
                echo "Cluster is HEALTHY! Both nodes detected."
                exit 0
            fi
        fi
        
        printf "."
        sleep $RETRY_DELAY
    done
    
    echo ""
    echo "TIMEOUT: Cluster did not stabilize within the expected time."
    echo "Fetching logs from macstudio-m4-1 for debugging:"
    echo "------------------------------------------------"
    ssh macstudio-m4-1 "tail -n 20 /tmp/exo.log"
    echo "------------------------------------------------"
    exit 1
fi

# Check for macmon (required for UI stats)
MACMON_PATH="$HOME/.cargo/bin/macmon"
if [ ! -f "$MACMON_PATH" ]; then
    echo "WARNING: macmon binary not found at $MACMON_PATH"
else
    echo "macmon detected at $MACMON_PATH"
fi

echo "Starting Exo with EXO_DISCOVERY_PEERS=$EXO_DISCOVERY_PEERS"

# Build Dashboard (as requested to refresh UI assets)
if [ -d "dashboard" ]; then
    echo "Checking dashboard..."
    # Skipping heavy build step on remote startup to speed up/reduce logs unless forced
    # Assuming dashboard is already built or static assets are fine.
    # Uncomment if build is strictly required on every run:
    # Try to load NVM if present
    # export NVM_DIR="$HOME/.nvm"
    # if [ -s "$NVM_DIR/nvm.sh" ]; then
    #     echo "Loading NVM..."
    #     source "$NVM_DIR/nvm.sh"
    # fi

    # cd dashboard
    # # Check if npm is available
    # if command -v npm &> /dev/null; then
    #     npm install && npm run build
    # else
    #     echo "WARNING: npm not found. Skipping dashboard build."
    # fi
    # cd ..
else
    echo "WARNING: dashboard directory not found."
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

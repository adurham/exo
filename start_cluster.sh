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

    # Thunderbolt Connectivity Check
    echo "Verifying Thunderbolt Connectivity..."
    TB_M4_1="192.168.200.1"
    TB_M4_2="192.168.200.2"

    # Check M4-1 IP
    CURRENT_M4_1_TB=$(ssh macstudio-m4-1 "ifconfig en2 2>/dev/null | grep 'inet ' | awk '{print \$2}'")
    if [ "$CURRENT_M4_1_TB" != "$TB_M4_1" ]; then
        echo "ERROR: macstudio-m4-1 Thunderbolt (en2) IP is '$CURRENT_M4_1_TB', expected '$TB_M4_1'."
        exit 1
    fi

    # Check M4-2 IP
    CURRENT_M4_2_TB=$(ssh macstudio-m4-2 "ifconfig en2 2>/dev/null | grep 'inet ' | awk '{print \$2}'")
    if [ "$CURRENT_M4_2_TB" != "$TB_M4_2" ]; then
        echo "WARNING: macstudio-m4-2 Thunderbolt (en2) IP is '$CURRENT_M4_2_TB', expected '$TB_M4_2'."
        echo "Attempting to auto-fix..."
        ssh macstudio-m4-2 "sudo networksetup -setmanual 'EXO Thunderbolt 1' $TB_M4_2 255.255.255.0"
        sleep 2
        CURRENT_M4_2_TB=$(ssh macstudio-m4-2 "ifconfig en2 2>/dev/null | grep 'inet ' | awk '{print \$2}'")
        if [ "$CURRENT_M4_2_TB" != "$TB_M4_2" ]; then
             echo "ERROR: Failed to set IP. Still '$CURRENT_M4_2_TB'. Please fix manually."
             exit 1
        fi
        echo "Auto-fix successful. IP set to $TB_M4_2."
    fi

    # Check Ping
    if ! ssh macstudio-m4-1 "ping -c 1 -W 1 $TB_M4_2" &> /dev/null; then
        echo "ERROR: macstudio-m4-1 cannot ping macstudio-m4-2 over Thunderbolt ($TB_M4_2)."
        exit 1
    fi
    echo "Thunderbolt Link Verified ($TB_M4_1 <-> $TB_M4_2)."


    # 1. Kill existing processes, git pull, and force reinstall bindings (rebuilds Rust bindings)
    # 1. Cleanup, Update, and Build
    for NODE in "${NODES[@]}"; do
        echo "Preparing $NODE..."
        # Aggressive cleanup: kill by port and name, and remove screen sessions
        echo "Setting Metal memory limit on $NODE..."
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=115000"
        ssh "$NODE" "lsof -ti:52415,52416 | xargs kill -9 2>/dev/null || true"
        ssh "$NODE" "pkill -9 -f 'exo.main' || true"
        ssh "$NODE" "screen -wipe || true"

        # Update and Build
        # Use zsh -l -c to ensure environment (PATH, etc.) is loaded
        ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && uv pip install --force-reinstall ./rust/exo_pyo3_bindings && uv pip install -e .'" || { echo "Failed to update/build on $NODE"; exit 1; }
    done

    # 3. Start Exo on each node
    for NODE in "${NODES[@]}"; do
        echo "Starting Exo on $NODE..."
        if [ "$NODE" == "macstudio-m4-1" ]; then
             # M4-1 connects to M4-2 via Thunderbolt IP (192.168.200.2)
             ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && EXO_DISCOVERY_PEERS=/ip4/192.168.200.2/tcp/52415/p2p/$M4_2_PEER_ID PYTHONUNBUFFERED=1 RUST_BACKTRACE=1 uv run python -m exo.main > /tmp/exo.log 2>&1'"
        else
             # M4-2 connects to M4-1 via Thunderbolt IP (192.168.200.1)
             ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && EXO_DISCOVERY_PEERS=/ip4/192.168.200.1/tcp/52415/p2p/$M4_1_PEER_ID PYTHONUNBUFFERED=1 RUST_BACKTRACE=1 uv run python -m exo.main > /tmp/exo.log 2>&1'"
        fi
        if [ $? -eq 0 ]; then
            echo "Successfully triggered start on $NODE."
        else
            echo "Failed to trigger start on $NODE."
        fi
    done

    # 3. Health Check / Topology Verification
    echo -n "Cluster start commands issued. Waiting for cluster to stabilize..."
    for i in {1..90}; do
        response=$(curl -s "http://$M4_1_IP:52415/state")
        if [ -n "$response" ]; then
            node_count=$(echo "$response" | jq '.topology.nodes | length' 2>/dev/null)
        else
            node_count=0
        fi

        # Default to 0 if jq failed or returned null
        if [ -z "$node_count" ] || [ "$node_count" == "null" ]; then
            node_count=0
        fi

        if [ "$node_count" -ge 2 ]; then
            echo "Cluster is HEALTHY! Node count: $node_count"
            exit 0
        fi
        echo -n "."
        sleep 2
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

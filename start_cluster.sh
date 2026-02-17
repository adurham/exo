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
    # Get local commit hash to enforce consistency
    LOCAL_COMMIT=$(git rev-parse --short HEAD)
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "Local commit: $LOCAL_COMMIT on branch $CURRENT_BRANCH"

    # 1. Cleanup, Update, and Build
    for NODE in "${NODES[@]}"; do
        echo "Preparing $NODE..."
        # Aggressive cleanup: kill by port and name, and remove screen sessions
        echo "Setting Metal memory limit on $NODE..."
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=115000"
        
        echo "Killing existing Exo processes on $NODE..."
        # Loop until processes are gone
        for i in {1..5}; do
            ssh "$NODE" "lsof -ti:52415,52416 | xargs kill -9 2>/dev/null || true"
            ssh "$NODE" "pkill -9 -f 'exo.main' || true"
            ssh "$NODE" "pkill -9 -f 'python.*exo' || true"
            
            # Check if still running
            if ssh "$NODE" "pgrep -f 'exo.main'" > /dev/null; then
                echo "  Processes still running, retrying kill..."
                sleep 1
            else
                break
            fi
        done
        
        ssh "$NODE" "screen -wipe || true"

        # Update and Build
        # Use zsh -l -c to ensure environment (PATH, etc.) is loaded
        # FORCE update to origin/main to avoid "Already up to date" issues on stale branches
        # Ensure Xcode is selected and initialized for Metal tools
        echo "Ensuring Xcode developer directory is set on $NODE..."
        ssh "$NODE" "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer || echo 'Failed to set xcode-select, proceeding anyway...'"
        ssh "$NODE" "sudo xcodebuild -runFirstLaunch || echo 'xcodebuild -runFirstLaunch failed or already done'"
        ssh "$NODE" "sudo xcodebuild -downloadComponent MetalToolchain || echo 'Metal Toolchain download failed or already installed'"
        
        # Update and Build Logic
        echo "Checking MLX submodule status on $NODE..."
        OLD_MLX=$(ssh "$NODE" "cd ~/repos/exo && git submodule status mlx" | awk '{print $1}' | sed 's/[+-]//g')
        
        # Pull latest changes
        ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && git fetch origin && git reset --hard && git checkout $CURRENT_BRANCH && git reset --hard origin/$CURRENT_BRANCH && git submodule sync && git submodule update --init --recursive'" || { echo "Failed to update repo on $NODE"; exit 1; }
        
        NEW_MLX=$(ssh "$NODE" "cd ~/repos/exo && git submodule status mlx" | awk '{print $1}' | sed 's/[+-]//g')
        
        BUILD_CMD="uv sync"
        if [ "$OLD_MLX" != "$NEW_MLX" ] || [ "$FORCE_REBUILD" == "1" ]; then
            REASON="MLX submodule changed ($OLD_MLX -> $NEW_MLX)"
            if [ "$FORCE_REBUILD" == "1" ]; then REASON="FORCE_REBUILD=1 set"; fi
            echo "Forcing clean rebuild on $NODE (Reason: $REASON)..."
            # Remove venv and build artifacts to force fresh compilation
            BUILD_CMD="rm -rf .venv mlx/build && uv cache clean && uv sync"
        fi

        # Ensure 'metal' is in PATH and Run Build
        echo "Running build on $NODE..."
        ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/repos/exo && $BUILD_CMD'" || { echo "Failed to build on $NODE"; exit 1; }

        # Verify Remote Commit
        REMOTE_COMMIT=$(ssh "$NODE" "cd ~/repos/exo && git rev-parse --short HEAD")
        if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
            echo "CRITICAL ERROR: Node $NODE is on commit $REMOTE_COMMIT, but local is $LOCAL_COMMIT."
            echo "The cluster is out of sync. Please fix git issues on $NODE and try again."
            exit 1
        fi
        echo "Node $NODE verified on commit $REMOTE_COMMIT."
    done

    # 3. Start Exo on each node
    for NODE in "${NODES[@]}"; do
        echo "Starting Exo on $NODE..."
        if [ "$NODE" == "macstudio-m4-1" ]; then
             # M4-1 connects to M4-2 via Thunderbolt IP (192.168.200.2)
             ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && EXO_FAST_SYNCH=${EXO_FAST_SYNCH:-} MLX_JACCL_RING=${MLX_JACCL_RING:-} IBV_FORK_SAFE=${IBV_FORK_SAFE:-1} EXO_MLX_WIRED_LIMIT_RATIO=${EXO_MLX_WIRED_LIMIT_RATIO:-} EXO_DISCOVERY_PEERS=/ip4/192.168.200.2/tcp/52415/p2p/$M4_2_PEER_ID PYTHONUNBUFFERED=1 RUST_BACKTRACE=1 uv run python -m exo.main > /tmp/exo.log 2>&1'"
        else
             # M4-2 connects to M4-1 via Thunderbolt IP (192.168.200.1)
             ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && EXO_FAST_SYNCH=${EXO_FAST_SYNCH:-} MLX_JACCL_RING=${MLX_JACCL_RING:-} IBV_FORK_SAFE=${IBV_FORK_SAFE:-1} EXO_MLX_WIRED_LIMIT_RATIO=${EXO_MLX_WIRED_LIMIT_RATIO:-} EXO_DISCOVERY_PEERS=/ip4/192.168.200.1/tcp/52415/p2p/$M4_1_PEER_ID PYTHONUNBUFFERED=1 RUST_BACKTRACE=1 uv run python -m exo.main > /tmp/exo.log 2>&1'"
        fi
        if [ $? -eq 0 ]; then
            echo "Successfully triggered start on $NODE."
        else
            echo "Failed to trigger start on $NODE."
        fi
    done


    # 3. Start Local Resources (Dashboard & Exo)
    # Build Dashboard (as requested to refresh UI assets)
    if [ -d "dashboard" ]; then
        echo "Checking dashboard..."
        # Try to load NVM if present
        export NVM_DIR="$HOME/.nvm"
        if [ -s "$NVM_DIR/nvm.sh" ]; then
            # Supress NVM output
            source "$NVM_DIR/nvm.sh" > /dev/null 2>&1
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
        echo "WARNING: dashboard directory not found."
    fi

    # Activate venv if it exists in the repo
    SOURCE_DIR=$(dirname "$0")
    if [ -d "$SOURCE_DIR/.venv" ]; then
        source "$SOURCE_DIR/.venv/bin/activate"
    fi

    # Run Exo from source to pick up local changes
    # FORCE local resources to be used
    export EXO_RESOURCES_DIR="$PWD/resources"
    echo "Using resources from: $EXO_RESOURCES_DIR"

    if command -v uv &> /dev/null; then
      echo "Starting Exo in background (screen session 'exo_local')..."
      # Kill existing session if present to avoid duplicates
      screen -X -S exo_local quit > /dev/null 2>&1 || true
      screen -dmS exo_local uv run exo
    else
      echo "Starting Exo in background (screen session 'exo_local')..."
      screen -X -S exo_local quit > /dev/null 2>&1 || true
      screen -dmS exo_local python3 -m exo.main
    fi

    # 4. Health Check / Topology Verification
    echo -n "Cluster start commands issued. Waiting for cluster to stabilize..."
    CLUSTER_READY=false
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
            CLUSTER_READY=true
            break
        fi
        echo -n "."
        sleep 2
    done
    
    if [ "$CLUSTER_READY" = false ]; then
        echo ""
        echo "TIMEOUT: Cluster did not stabilize within the expected time."
        echo "Fetching logs from macstudio-m4-1 for debugging:"
        echo "------------------------------------------------"
        ssh macstudio-m4-1 "tail -n 20 /tmp/exo.log"
        echo "------------------------------------------------"
        exit 1
    fi
fi

# Check for macmon (required for UI stats)
MACMON_PATH="$HOME/.cargo/bin/macmon"
if [ ! -f "$MACMON_PATH" ]; then
    echo "WARNING: macmon binary not found at $MACMON_PATH"
else
    echo "macmon detected at $MACMON_PATH"
fi

export IBV_FORK_SAFE=${IBV_FORK_SAFE:-1}

echo "Starting Exo with EXO_DISCOVERY_PEERS=$EXO_DISCOVERY_PEERS"

#!/bin/bash

# Standardized Exo Cluster Startup Script
# Usage: ./start_cluster.sh
# Detects the current host and sets up the appropriate environment for the 2-node M4 cluster.

export EXO_FAST_SYNCH=off
export EXO_LIBP2P_NAMESPACE=MAC_STUDIO_CLUSTER

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

    # 1. Cleanup, Update, and Build
    for NODE in "${NODES[@]}"; do
        echo "Preparing $NODE..."
        echo "Setting Metal memory limit on $NODE..."
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=115000"
        
        echo "Killing existing Exo processes on $NODE..."
        for i in {1..5}; do
            ssh "$NODE" "lsof -ti:52415,52416 | xargs kill -9 2>/dev/null || true"
            ssh "$NODE" "pkill -9 -f 'exo.main' || true"
            ssh "$NODE" "pkill -9 -f 'python.*exo' || true"
            
            if ssh "$NODE" "pgrep -f 'exo.main'" > /dev/null; then
                sleep 1
            else
                break
            fi
        done
        
        ssh "$NODE" "screen -wipe || true"

        echo "Ensuring Xcode developer directory on $NODE..."
        ssh "$NODE" "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer || true"
        
        # Update and Build Logic
        TARGET_BRANCH="main"
        ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && git fetch origin && git reset --hard && git checkout $TARGET_BRANCH && git reset --hard origin/$TARGET_BRANCH && git submodule sync && git submodule update --init --recursive'" || { echo "Failed to update repo on $NODE"; exit 1; }
        
        echo "Running build on $NODE..."
        ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/repos/exo && uv sync'" || { echo "Failed to build on $NODE"; exit 1; }
    done

    # 2. Inter-Node Git Sync Check (M4-1 vs M4-2)
    echo "Verifying commit consistency between nodes..."
    COMMIT_M4_1=$(ssh macstudio-m4-1 "cd ~/repos/exo && git rev-parse --short HEAD")
    COMMIT_M4_2=$(ssh macstudio-m4-2 "cd ~/repos/exo && git rev-parse --short HEAD")

    if [ "$COMMIT_M4_1" != "$COMMIT_M4_2" ]; then
        echo "CRITICAL ERROR: Cluster out of sync!"
        echo "macstudio-m4-1: $COMMIT_M4_1"
        echo "macstudio-m4-2: $COMMIT_M4_2"
        exit 1
    fi
    echo "Nodes synchronized on commit $COMMIT_M4_1."

    # 3. Start Exo on each node
    for NODE in "${NODES[@]}"; do
        echo "Starting Exo on $NODE..."
        if [ "$NODE" == "macstudio-m4-1" ]; then
             # M4-1 connects to M4-2 via Thunderbolt IP (192.168.200.2)
             ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && EXO_KV_BITS=false EXO_BATCH_COMPLETION_SIZE=8 EXO_FAST_SYNCH=off EXO_MLX_WIRED_LIMIT_RATIO=0.87 EXO_DISCOVERY_PEERS=/ip4/192.168.200.2/tcp/52415/p2p/$M4_2_PEER_ID PYTHONUNBUFFERED=1 uv run python -m exo.main > /tmp/exo.log 2>&1'"
        else
             # M4-2 connects to M4-1 via Thunderbolt IP (192.168.200.1)
             ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && EXO_KV_BITS=false EXO_BATCH_COMPLETION_SIZE=8 EXO_FAST_SYNCH=off EXO_MLX_WIRED_LIMIT_RATIO=0.87 EXO_DISCOVERY_PEERS=/ip4/192.168.200.1/tcp/52415/p2p/$M4_1_PEER_ID PYTHONUNBUFFERED=1 uv run python -m exo.main > /tmp/exo.log 2>&1'"
        fi
    done

    # 4. Health Check / Topology Verification
    echo -n "Waiting for cluster to stabilize..."
    CLUSTER_READY=false
    for i in {1..90}; do
        response=$(curl -s "http://$M4_1_IP:52415/state")
        node_count=$(echo "$response" | jq '.topology.nodes | length' 2>/dev/null)

        # Handle null or empty node_count to prevent integer expression errors
        if [ -z "$node_count" ] || [ "$node_count" == "null" ]; then
            node_count=0
        fi

        if [ "$node_count" -ge 2 ]; then
            echo " HEALTHY! (Nodes: $node_count)"
            CLUSTER_READY=true
            break
        fi
        echo -n "."
        sleep 2
    done
    
    if [ "$CLUSTER_READY" = false ]; then
        echo ""
        echo "TIMEOUT: Cluster did not stabilize."
        echo "Fetching logs from macstudio-m4-1:"
        ssh macstudio-m4-1 "tail -n 20 /tmp/exo.log"
        exit 1
    fi
fi

# Final environment export
export IBV_FORK_SAFE=${IBV_FORK_SAFE:-1}

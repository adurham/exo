#!/bin/bash

# Standardized Exo Cluster Startup Script
# Usage: ./start_cluster.sh
# Detects the current host and sets up the appropriate environment for the 3-node M4 cluster.

export EXO_FAST_SYNCH=on
export EXO_EVAL_DEBUG=1
export EXO_TP_DEBUG=1
export EXO_MAX_ACTIVE_TASKS=30
export EXO_LIBP2P_NAMESPACE=MAC_STUDIO_CLUSTER
export IBV_FORK_SAFE=1

# Exo Runtime Variables
export EXO_KV_BITS=false
export EXO_BATCH_COMPLETION_SIZE=8
export EXO_MLX_WIRED_LIMIT_RATIO=0.87
export PYTHONUNBUFFERED=1

# Define Node Constants
M4_1_IP="192.168.86.201"
M4_1_PEER_ID="12D3KooWDGQKAJUYpqTHzBhVpGzYxQagWRwFqJPzkEYzHxt3SSUg"
M4_2_IP="192.168.86.202"
M4_2_PEER_ID="12D3KooWQDzFqvjsgFRfheeV7uvtVUP1gruphpgoVELP9pkHBses"
MBP_IP="192.168.86.203"
MBP_PEER_ID="12D3KooWGtRYJcQpFLQBc3AFbES1A3BrFy55GyNLMNLNm64bHv16"

# Get current IPs (check all interfaces to correctly identify the node)
CURRENT_IPS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}')

IS_M4_1=false
IS_M4_2=false
IS_MBP=false

for IP in $CURRENT_IPS; do
    if [ "$IP" == "$M4_1_IP" ]; then
        IS_M4_1=true
        break
    fi
    if [ "$IP" == "$M4_2_IP" ]; then
        IS_M4_2=true
        break
    fi
    if [ "$IP" == "$MBP_IP" ]; then
        IS_MBP=true
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
elif [ "$IS_MBP" = true ]; then
    echo "Detected MacBook Pro ($MBP_IP)"
    # Peer with M4-1
    export EXO_DISCOVERY_PEERS="/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID"
else
    echo "Unknown host IPs: $CURRENT_IPS. Running as remote controller."
fi

# Full cluster setup — always runs regardless of which machine launches the script
echo "Starting cluster setup..."
echo "-----------------------------------------------------"

# Define nodes to start (using SSH config aliases)
NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")

    # Thunderbolt Connectivity Check
    echo "Discovering active Thunderbolt IPs..."

get_node_tb_ips() {
    local node=$1
    # 1. Ask the node for its Thunderbolt device names (e.g., en1, en2)
    local devices=$(ssh "$node" "networksetup -listallhardwareports" | awk '/Hardware Port: Thunderbolt/{getline; print $2}')
    
    # 2. Iterate through them locally, asking the node about each one individually
    for dev in $devices; do
        if ssh "$node" "ifconfig $dev" 2>/dev/null | grep -q "status: active"; then
            ssh "$node" "ifconfig $dev" | awk '/inet / && !/127\.0\.0\.1/{print $2}'
        fi
    done
}

find_shared_ip() {
    local target_ips=$1
    local peer_ips=$2
    for tip in $target_ips; do
        local t_subnet=$(echo "$tip" | awk -F. '{print $1"."$2"."$3}')
        for pip in $peer_ips; do
            local p_subnet=$(echo "$pip" | awk -F. '{print $1"."$2"."$3}')
            if [ "$t_subnet" == "$p_subnet" ]; then
                echo "$tip"
                return 0
            fi
        done
    done
    return 1
}

echo "Fetching active Thunderbolt IPs from all nodes..."
TB_M4_1_IPS=$(get_node_tb_ips "macstudio-m4-1")
TB_M4_2_IPS=$(get_node_tb_ips "macstudio-m4-2")
TB_MBP_IPS=$(get_node_tb_ips "macbook-m4")

# Match IPs by their shared broadcast domains
M4_1_TO_M4_2=$(find_shared_ip "$TB_M4_1_IPS" "$TB_M4_2_IPS")
M4_1_TO_MBP=$(find_shared_ip "$TB_M4_1_IPS" "$TB_MBP_IPS")

M4_2_TO_M4_1=$(find_shared_ip "$TB_M4_2_IPS" "$TB_M4_1_IPS")
M4_2_TO_MBP=$(find_shared_ip "$TB_M4_2_IPS" "$TB_MBP_IPS")

MBP_TO_M4_1=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_1_IPS")
MBP_TO_M4_2=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_2_IPS")

echo "macstudio-m4-1 routes: -> M4-2 ($M4_1_TO_M4_2), -> MBP ($M4_1_TO_MBP)"
echo "macstudio-m4-2 routes: -> M4-1 ($M4_2_TO_M4_1), -> MBP ($M4_2_TO_MBP)"
echo "macbook-m4   routes: -> M4-1 ($MBP_TO_M4_1), -> M4-2 ($MBP_TO_M4_2)"

# Verify all 6 connection points were successfully discovered
if [ -z "$M4_1_TO_M4_2" ] || [ -z "$M4_1_TO_MBP" ] || [ -z "$M4_2_TO_M4_1" ] || [ -z "$M4_2_TO_MBP" ] || [ -z "$MBP_TO_M4_1" ] || [ -z "$MBP_TO_M4_2" ]; then
    echo "CRITICAL ERROR: Could not map a full 3-way Thunderbolt mesh subnet topology!"
    exit 1
fi

# Validate each node has 2 active Thunderbolt interfaces (catches loose cables)
echo "Verifying direct Thunderbolt links..."
for node in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    active_count=$(echo "$(get_node_tb_ips "$node")" | grep -c '.')
    if [ "$active_count" -lt 2 ]; then
        echo "CRITICAL ERROR: $node has only $active_count active Thunderbolt interface(s) — expected 2."
        echo "Check physical Thunderbolt cable connections!"
        exit 1
    fi
    echo "  $node: $active_count active TB interfaces ✓"
done

# Direct-link pings — clear any stale cross-subnet routes from previous runs first,
# then ping. Without routes, pings can only succeed over direct physical links — no relay.
echo "Testing direct-link connectivity (clearing stale routes first)..."
for node in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    ssh "$node" "for r in \$(netstat -rn | awk '/192\.168\.(200|201|202)\./{print \$1}' | sort -u); do sudo route delete -net \$r 2>/dev/null; done" &> /dev/null
done

# M4-1 ↔ M4-2 (direct link)
if ! ssh macstudio-m4-1 "ping -c 1 -W 1 $M4_2_TO_M4_1" &> /dev/null; then echo "ERROR: macstudio-m4-1 cannot directly reach M4-2 ($M4_2_TO_M4_1). Check cable!"; exit 1; fi
if ! ssh macstudio-m4-2 "ping -c 1 -W 1 $M4_1_TO_M4_2" &> /dev/null; then echo "ERROR: macstudio-m4-2 cannot directly reach M4-1 ($M4_1_TO_M4_2). Check cable!"; exit 1; fi

# M4-1 ↔ MBP (direct link)
if ! ssh macstudio-m4-1 "ping -c 1 -W 1 $MBP_TO_M4_1" &> /dev/null; then echo "ERROR: macstudio-m4-1 cannot directly reach MBP ($MBP_TO_M4_1). Check cable!"; exit 1; fi
if ! ssh macbook-m4 "ping -c 1 -W 1 $M4_1_TO_MBP" &> /dev/null; then echo "ERROR: macbook-m4 cannot directly reach M4-1 ($M4_1_TO_MBP). Check cable!"; exit 1; fi

# M4-2 ↔ MBP (direct link)
if ! ssh macstudio-m4-2 "ping -c 1 -W 1 $MBP_TO_M4_2" &> /dev/null; then echo "ERROR: macstudio-m4-2 cannot directly reach MBP ($MBP_TO_M4_2). Check cable!"; exit 1; fi
if ! ssh macbook-m4 "ping -c 1 -W 1 $M4_2_TO_MBP" &> /dev/null; then echo "ERROR: macbook-m4 cannot directly reach M4-2 ($M4_2_TO_MBP). Check cable!"; exit 1; fi

echo "All 6 direct Thunderbolt links verified ✓"

# RoCEv2 (RDMA) Protection Domain Allocation Check
# A degraded Thunderbolt cable will pass `ping` (using USB-C fallback Ethernet), but fail to allocate
# an RDMA Protection Domain, causing `jaccl` to instantly crash when Exo starts.
echo "Verifying RoCEv2 (RDMA) support over Thunderbolt..."
for NODE in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    echo -n "  Testing RDMA allocation on $NODE... "
    # We use `timeout 2` because a successful PD allocation will hang waiting for a coordinator.
    # We run it within the uv environment to ensure mlx is available.
    # If the Thunderbolt cable is degraded, allocating the Protection Domain crashes immediately.
    RDMA_CHECK=$(ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && timeout 2 uv run python -c \"import mlx.core as mx; mx.distributed.init(strict=False, backend='\\''jaccl'\\'')\" 2>&1'" || true)
    
    if echo "$RDMA_CHECK" | grep -q "Couldn't allocate protection domain"; then
        echo "FAIL ❌"
        echo "CRITICAL ERROR: Failed to allocate RDMA Protection Domain on $NODE!"
        echo "One of your Thunderbolt cables has fallen back to standard USB-C Ethernet."
        echo "Please re-seat the cables on $NODE."
        exit 1
    else
        echo "OK ✓"
    fi
done

# Enable IP forwarding and add cross-subnet routes
# Each node has 2 direct links, but needs a route for the 3rd subnet it's not on.
echo "Enabling IP forwarding and configuring cross-subnet routes..."

SUBNET_M4_1_M4_2=$(echo "$M4_1_TO_M4_2" | awk -F. '{print $1"."$2"."$3".0/24"}')
SUBNET_M4_1_MBP=$(echo "$M4_1_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')
SUBNET_M4_2_MBP=$(echo "$M4_2_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')

for NODE in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    ssh "$NODE" "sudo sysctl -w net.inet.ip.forwarding=1" &> /dev/null
done

# M4-1 is NOT on the M4-2<->MBP subnet → route via M4-2
ssh macstudio-m4-1 "sudo route delete -net $SUBNET_M4_2_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_2_MBP $M4_2_TO_M4_1" &> /dev/null
# M4-2 is NOT on the M4-1<->MBP subnet → route via M4-1
ssh macstudio-m4-2 "sudo route delete -net $SUBNET_M4_1_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_1_MBP $M4_1_TO_M4_2" &> /dev/null
# MBP is NOT on the M4-1<->M4-2 subnet → route via M4-1
ssh macbook-m4 "sudo route delete -net $SUBNET_M4_1_M4_2 2>/dev/null; sudo route add -net $SUBNET_M4_1_M4_2 $M4_1_TO_MBP" &> /dev/null

echo "Cross-subnet routes configured."

# 1. Cleanup, Update, and Build
for NODE in "${NODES[@]}"; do
    echo "Preparing $NODE..."
    echo "Setting Metal memory limit on $NODE..."
    if [[ "$NODE" == *"macbook"* ]]; then
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=31000"
    else
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=115000"
    fi
    
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
    
    echo "Ensuring build dependencies on $NODE..."
    ssh "$NODE" "/opt/homebrew/bin/brew install cmake 2>/dev/null || true"

    # Check if MLX needs rebuilding:
    # 1. Compare installed version hash to submodule HEAD (detects submodule update)
    # 2. Check if any C++ source file is newer than the compiled .so (detects stale cmake cache)
    INSTALLED_MLX=$(ssh "$NODE" "cd ~/repos/exo && .venv/bin/python -c \"import mlx.core; v=mlx.core.__version__; print(v.split('+')[-1] if '+' in v else 'none')\" 2>/dev/null || echo none")
    SUBMODULE_MLX=$(ssh "$NODE" "cd ~/repos/exo/mlx && git rev-parse --short HEAD")
    STALE_SO=$(ssh "$NODE" "find ~/repos/exo/mlx/mlx -name '*.cpp' -newer ~/repos/exo/mlx/python/mlx/core.cpython-*-darwin.so 2>/dev/null | head -1")

    if [ "$INSTALLED_MLX" != "$SUBMODULE_MLX" ] || [ -n "$STALE_SO" ]; then
        if [ -n "$STALE_SO" ]; then
            echo "MLX C++ sources newer than compiled .so on $NODE, forcing clean rebuild..."
        else
            echo "MLX submodule changed on $NODE ($INSTALLED_MLX → $SUBMODULE_MLX), rebuilding..."
        fi
        # Clean cmake build cache to ensure ALL source files are recompiled
        ssh "$NODE" "rm -rf ~/repos/exo/mlx/build"
        ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=/opt/homebrew/bin:\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/repos/exo && uv sync --reinstall-package mlx'" || { echo "Failed to build on $NODE"; exit 1; }
    else
        echo "MLX unchanged on $NODE ($INSTALLED_MLX), skipping C++ rebuild."
        ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=/opt/homebrew/bin:\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/repos/exo && uv sync'" || { echo "Failed to build on $NODE"; exit 1; }
    fi




    echo "Building dashboard on $NODE..."
    ssh "$NODE" "zsh -l -c 'source ~/.zshrc; cd ~/repos/exo/dashboard && npm install && npm run build'" || { echo "Failed to build dashboard on $NODE"; exit 1; }
done

# 2. Inter-Node Git Sync Check (M4-1 vs M4-2 vs MBP)
echo "Verifying commit consistency between nodes..."
COMMIT_M4_1=$(ssh macstudio-m4-1 "cd ~/repos/exo && git rev-parse --short HEAD")
COMMIT_M4_2=$(ssh macstudio-m4-2 "cd ~/repos/exo && git rev-parse --short HEAD")
COMMIT_MBP=$(ssh macbook-m4 "cd ~/repos/exo && git rev-parse --short HEAD")

if [ "$COMMIT_M4_1" != "$COMMIT_M4_2" ] || [ "$COMMIT_M4_1" != "$COMMIT_MBP" ]; then
    echo "CRITICAL ERROR: Cluster out of sync!"
    echo "macstudio-m4-1: $COMMIT_M4_1"
    echo "macstudio-m4-2: $COMMIT_M4_2"
    echo "macbook-m4: $COMMIT_MBP"
    exit 1
fi
echo "Nodes synchronized on commit $COMMIT_M4_1."

# 3. Start Exo on each node
for NODE in "${NODES[@]}"; do
    echo "Starting Exo on $NODE..."
    
    # Build the dynamic environment string based on the current exports
    # KV cache size (same for all nodes — effective window is limited by smallest)
    KV_SIZE="${EXO_MAX_KV_SIZE:-}"
    KV_KEEP="${EXO_KEEP_KV_SIZE:-}"
    EXO_ENV="EXO_KV_BITS=${EXO_KV_BITS:-false} EXO_MAX_KV_SIZE=$KV_SIZE EXO_KEEP_KV_SIZE=$KV_KEEP EXO_BATCH_COMPLETION_SIZE=${EXO_BATCH_COMPLETION_SIZE:-8} EXO_MLX_WIRED_LIMIT_RATIO=${EXO_MLX_WIRED_LIMIT_RATIO:-0.87} PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}"
    if [ -n "$EXO_FAST_SYNCH" ]; then
        EXO_ENV="$EXO_ENV EXO_FAST_SYNCH=$EXO_FAST_SYNCH"
    fi
    if [ -n "$EXO_TP_DEBUG" ]; then
        EXO_ENV="$EXO_ENV EXO_TP_DEBUG=$EXO_TP_DEBUG"
    fi
    if [ -n "$EXO_EVAL_DEBUG" ]; then
        EXO_ENV="$EXO_ENV EXO_EVAL_DEBUG=$EXO_EVAL_DEBUG"
    fi
    if [ -n "$EXO_MAX_ACTIVE_TASKS" ]; then
        EXO_ENV="$EXO_ENV EXO_MAX_ACTIVE_TASKS=$EXO_MAX_ACTIVE_TASKS"
    fi
    if [ -n "$EXO_PREFILL_STEP_SIZE" ]; then
        EXO_ENV="$EXO_ENV EXO_PREFILL_STEP_SIZE=$EXO_PREFILL_STEP_SIZE"
    fi
    
    if [ "$NODE" == "macstudio-m4-1" ]; then
         ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_2_TO_M4_1/tcp/52415/p2p/$M4_2_PEER_ID uv run python -m exo.main > /tmp/exo.log 2>&1'"
    elif [ "$NODE" == "macstudio-m4-2" ]; then
         ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_1_TO_M4_2/tcp/52415/p2p/$M4_1_PEER_ID uv run python -m exo.main > /tmp/exo.log 2>&1'"
    else
         ssh "$NODE" "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $EXO_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_1_TO_MBP/tcp/52415/p2p/$M4_1_PEER_ID uv run python -m exo.main > /tmp/exo.log 2>&1'"
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

    if [ "$node_count" -ge 3 ]; then
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


# Final environment export
export IBV_FORK_SAFE=${IBV_FORK_SAFE:-1}

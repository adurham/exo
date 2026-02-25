#!/bin/bash

# A/B Testing Script — "B" Side (Pure Upstream in ~/repos/exo-upstream)
# Run from your dev MacBook — SSHes into all 3 cluster nodes.
#
# Clones upstream exo into ~/repos/exo-upstream (if not already present),
# fetches latest upstream/main, rebuilds everything (Rust bindings, dashboard),
# and starts the cluster with STOCK upstream settings.
#
# MLX is pulled from upstream's pyproject.toml (git source, NOT your local submodule).
# mlx-lm comes from PyPI. No local editable overrides.
#
# Logs go to /tmp/exo_B.log (does NOT conflict with /tmp/exo.log from start_cluster.sh).

set -euo pipefail

UPSTREAM_URL="https://github.com/exo-explore/exo.git"
UPSTREAM_DIR="repos/exo-upstream"  # relative to $HOME
NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")
LOG_FILE="/tmp/exo_B.log"

# Peer IDs (same hardware → same libp2p keys regardless of which exo build runs)
M4_1_IP="192.168.86.201"
M4_1_PEER_ID="12D3KooWDGQKAJUYpqTHzBhVpGzYxQagWRwFqJPzkEYzHxt3SSUg"
M4_2_IP="192.168.86.202"
M4_2_PEER_ID="12D3KooWQDzFqvjsgFRfheeV7uvtVUP1gruphpgoVELP9pkHBses"
MBP_IP="192.168.86.203"
MBP_PEER_ID="12D3KooWGtRYJcQpFLQBc3AFbES1A3BrFy55GyNLMNLNm64bHv16"

echo "============================================"
echo "  A/B Testing — Side B (Pure Upstream)"
echo "  Repo: ~/$UPSTREAM_DIR"
echo "============================================"

# ======================================================================
# STEP 1: Kill existing exo processes on all nodes
# ======================================================================
echo ""
echo "[1/7] Killing existing exo processes..."
for NODE in "${NODES[@]}"; do
    (
        for i in {1..5}; do
            ssh "$NODE" "lsof -ti:52415,52416 | xargs kill -9 2>/dev/null || true"
            ssh "$NODE" "pkill -9 -f 'exo.main' || true"
            ssh "$NODE" "pkill -9 -f 'python.*exo' || true"

            if ssh "$NODE" "pgrep -f 'exo.main'" > /dev/null 2>&1; then
                sleep 1
            else
                break
            fi
        done
        ssh "$NODE" "screen -wipe 2>/dev/null || true"
    ) &
done
wait
echo "    Done"

# ======================================================================
# STEP 2: Clone or update upstream repo on all nodes
# ======================================================================
echo ""
echo "[2/7] Cloning / updating upstream repo on all nodes..."
for NODE in "${NODES[@]}"; do
    echo "  → $NODE..."
    ssh "$NODE" "
        if [ ! -d ~/$UPSTREAM_DIR/.git ]; then
            echo '    Cloning upstream into ~/$UPSTREAM_DIR...'
            git clone $UPSTREAM_URL ~/$UPSTREAM_DIR
        fi
        cd ~/$UPSTREAM_DIR && \
        git fetch origin && \
        git reset --hard && \
        git checkout main 2>/dev/null || git checkout -b main origin/main && \
        git reset --hard origin/main
    "
    EXO_SHA=$(ssh "$NODE" "cd ~/$UPSTREAM_DIR && git rev-parse --short HEAD")
    echo "    $NODE: exo=$EXO_SHA ✓"
done

# ======================================================================
# STEP 3: Thunderbolt mesh discovery & routing
# ======================================================================
echo ""
echo "[3/7] Discovering Thunderbolt mesh topology..."

get_node_tb_ips() {
    local node=$1
    local devices=$(ssh "$node" "networksetup -listallhardwareports" | awk '/Hardware Port: Thunderbolt/{getline; print $2}')
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

echo "  Fetching active Thunderbolt IPs from all nodes..."
TB_M4_1_IPS=$(get_node_tb_ips "macstudio-m4-1")
TB_M4_2_IPS=$(get_node_tb_ips "macstudio-m4-2")
TB_MBP_IPS=$(get_node_tb_ips "macbook-m4")

M4_1_TO_M4_2=$(find_shared_ip "$TB_M4_1_IPS" "$TB_M4_2_IPS")
M4_1_TO_MBP=$(find_shared_ip "$TB_M4_1_IPS" "$TB_MBP_IPS")
M4_2_TO_M4_1=$(find_shared_ip "$TB_M4_2_IPS" "$TB_M4_1_IPS")
M4_2_TO_MBP=$(find_shared_ip "$TB_M4_2_IPS" "$TB_MBP_IPS")
MBP_TO_M4_1=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_1_IPS")
MBP_TO_M4_2=$(find_shared_ip "$TB_MBP_IPS" "$TB_M4_2_IPS")

echo "  macstudio-m4-1 routes: -> M4-2 ($M4_1_TO_M4_2), -> MBP ($M4_1_TO_MBP)"
echo "  macstudio-m4-2 routes: -> M4-1 ($M4_2_TO_M4_1), -> MBP ($M4_2_TO_MBP)"
echo "  macbook-m4     routes: -> M4-1 ($MBP_TO_M4_1), -> M4-2 ($MBP_TO_M4_2)"

if [ -z "$M4_1_TO_M4_2" ] || [ -z "$M4_1_TO_MBP" ] || [ -z "$M4_2_TO_M4_1" ] || [ -z "$M4_2_TO_MBP" ] || [ -z "$MBP_TO_M4_1" ] || [ -z "$MBP_TO_M4_2" ]; then
    echo "CRITICAL ERROR: Could not map a full 3-way Thunderbolt mesh subnet topology!"
    exit 1
fi

# Validate each node has 2 active Thunderbolt interfaces
echo "  Verifying direct Thunderbolt links..."
for node in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    active_count=$(echo "$(get_node_tb_ips "$node")" | grep -c '.')
    if [ "$active_count" -lt 2 ]; then
        echo "CRITICAL ERROR: $node has only $active_count active Thunderbolt interface(s) — expected 2."
        echo "Check physical Thunderbolt cable connections!"
        exit 1
    fi
    echo "    $node: $active_count active TB interfaces ✓"
done

# Clear stale routes, then test direct-link connectivity
echo "  Testing direct-link connectivity..."
for node in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    ssh "$node" "for r in \$(netstat -rn | awk '/192\.168\.(200|201|202)\./{print \$1}' | sort -u); do sudo route delete -net \$r 2>/dev/null; done" &> /dev/null
done

if ! ssh macstudio-m4-1 "ping -c 1 -W 1 $M4_2_TO_M4_1" &> /dev/null; then echo "ERROR: M4-1 cannot reach M4-2. Check cable!"; exit 1; fi
if ! ssh macstudio-m4-2 "ping -c 1 -W 1 $M4_1_TO_M4_2" &> /dev/null; then echo "ERROR: M4-2 cannot reach M4-1. Check cable!"; exit 1; fi
if ! ssh macstudio-m4-1 "ping -c 1 -W 1 $MBP_TO_M4_1" &> /dev/null; then echo "ERROR: M4-1 cannot reach MBP. Check cable!"; exit 1; fi
if ! ssh macbook-m4 "ping -c 1 -W 1 $M4_1_TO_MBP" &> /dev/null; then echo "ERROR: MBP cannot reach M4-1. Check cable!"; exit 1; fi
if ! ssh macstudio-m4-2 "ping -c 1 -W 1 $MBP_TO_M4_2" &> /dev/null; then echo "ERROR: M4-2 cannot reach MBP. Check cable!"; exit 1; fi
if ! ssh macbook-m4 "ping -c 1 -W 1 $M4_2_TO_MBP" &> /dev/null; then echo "ERROR: MBP cannot reach M4-2. Check cable!"; exit 1; fi
echo "  All 6 direct Thunderbolt links verified ✓"

# RDMA Protection Domain check
echo "  Verifying RoCEv2 (RDMA) support over Thunderbolt..."
for NODE in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    echo -n "    Testing RDMA allocation on $NODE... "
    # Use the upstream repo's uv environment for this check
    RDMA_CHECK=$(ssh "$NODE" "zsh -l -c 'cd ~/$UPSTREAM_DIR && timeout 2 uv run python -c \"import mlx.core as mx; mx.distributed.init(strict=False, backend='\\''jaccl'\\''')\" 2>&1'" || true)
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
echo "  Configuring cross-subnet routes..."
SUBNET_M4_1_M4_2=$(echo "$M4_1_TO_M4_2" | awk -F. '{print $1"."$2"."$3".0/24"}')
SUBNET_M4_1_MBP=$(echo "$M4_1_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')
SUBNET_M4_2_MBP=$(echo "$M4_2_TO_MBP" | awk -F. '{print $1"."$2"."$3".0/24"}')

for NODE in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
    ssh "$NODE" "sudo sysctl -w net.inet.ip.forwarding=1" &> /dev/null
done

ssh macstudio-m4-1 "sudo route delete -net $SUBNET_M4_2_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_2_MBP $M4_2_TO_M4_1" &> /dev/null
ssh macstudio-m4-2 "sudo route delete -net $SUBNET_M4_1_MBP 2>/dev/null; sudo route add -net $SUBNET_M4_1_MBP $M4_1_TO_M4_2" &> /dev/null
ssh macbook-m4 "sudo route delete -net $SUBNET_M4_1_M4_2 2>/dev/null; sudo route add -net $SUBNET_M4_1_M4_2 $M4_1_TO_MBP" &> /dev/null
echo "  Cross-subnet routes configured ✓"

# ======================================================================
# STEP 4: Set Metal wired memory limits
# ======================================================================
echo ""
echo "[4/7] Setting Metal memory limits..."
for NODE in "${NODES[@]}"; do
    if [[ "$NODE" == *"macbook"* ]]; then
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=31000"
    else
        ssh "$NODE" "sudo sysctl iogpu.wired_limit_mb=115000"
    fi
done

# ======================================================================
# STEP 5: Build (uv sync + Rust bindings + dashboard)
# ======================================================================
echo ""
echo "[5/7] Building upstream exo on all nodes..."
for NODE in "${NODES[@]}"; do
    echo "  → $NODE..."

    echo "    Ensuring Xcode developer directory..."
    ssh "$NODE" "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer || true"

    echo "    Ensuring cmake..."
    ssh "$NODE" "/opt/homebrew/bin/brew install cmake 2>/dev/null || true"

    echo "    Running uv sync (upstream deps from PyPI/git)..."
    ssh "$NODE" "export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer && export PATH=/opt/homebrew/bin:\$(dirname \$(xcrun -f metal)):\$PATH && zsh -l -c 'cd ~/$UPSTREAM_DIR && uv sync'" || { echo "Failed to build on $NODE"; exit 1; }

    echo "    Building dashboard..."
    ssh "$NODE" "zsh -l -c 'source ~/.zshrc; cd ~/$UPSTREAM_DIR/dashboard && npm install && npm run build'" || { echo "Failed to build dashboard on $NODE"; exit 1; }

    echo "    $NODE done ✓"
done

# Deploy MiniMax-M2.5-5bit model card (not in upstream's repo but model is
# already downloaded at ~/.exo/models/ from the fork's previous usage).
echo ""
echo "  Deploying MiniMax-M2.5-5bit model card to all nodes..."
CARD_TMP=$(mktemp /tmp/minimax-5bit-card.XXXXXX)
cat > "$CARD_TMP" << 'CARDEOF'
model_id = "mlx-community/MiniMax-M2.5-5bit"
n_layers = 62
hidden_size = 3072
supports_tensor = true
tasks = ["TextGeneration"]
family = "minimax"
quantization = "5bit"
base_model = "MiniMax M2.5"
capabilities = ["text", "thinking"]

[storage_size]
in_bytes = 157262670267
CARDEOF
for NODE in "${NODES[@]}"; do
    scp -q "$CARD_TMP" "$NODE:~/repos/exo-upstream/resources/inference_model_cards/mlx-community--MiniMax-M2.5-5bit.toml"
done
rm -f "$CARD_TMP"
echo "  Model card deployed ✓"

# ======================================================================
# STEP 6: Verify commit consistency
# ======================================================================
echo ""
echo "[6/7] Verifying commit consistency between nodes..."
COMMIT_M4_1=$(ssh macstudio-m4-1 "cd ~/$UPSTREAM_DIR && git rev-parse --short HEAD")
COMMIT_M4_2=$(ssh macstudio-m4-2 "cd ~/$UPSTREAM_DIR && git rev-parse --short HEAD")
COMMIT_MBP=$(ssh macbook-m4 "cd ~/$UPSTREAM_DIR && git rev-parse --short HEAD")

if [ "$COMMIT_M4_1" != "$COMMIT_M4_2" ] || [ "$COMMIT_M4_1" != "$COMMIT_MBP" ]; then
    echo "CRITICAL ERROR: Cluster out of sync!"
    echo "  macstudio-m4-1: $COMMIT_M4_1"
    echo "  macstudio-m4-2: $COMMIT_M4_2"
    echo "  macbook-m4:     $COMMIT_MBP"
    exit 1
fi
echo "  All nodes on upstream commit $COMMIT_M4_1 ✓"

# ======================================================================
# STEP 7: Start exo on each node with STOCK upstream settings
# ======================================================================
echo ""
echo "[7/7] Starting cluster with STOCK upstream settings..."

# Generate a per-node launcher script locally, scp it, then run via nohup.
# This avoids: (1) multi-layer shell quoting issues, (2) screen pty disconnect
# from stdout redirect, (3) upstream __main__.py silently exiting with -m flag.
launch_node() {
    local node=$1
    local peers_env=$2
    local local_tmp
    local_tmp=$(mktemp /tmp/exo_B_launch.XXXXXX)

    cat > "$local_tmp" << LAUNCHEOF
#!/bin/zsh
source ~/.zshrc 2>/dev/null || true
export PATH="/opt/homebrew/bin:\$PATH"

export EXO_LIBP2P_NAMESPACE=MAC_STUDIO_CLUSTER
export IBV_FORK_SAFE=1
export PYTHONUNBUFFERED=1
export EXO_DISCOVERY_PEERS=${peers_env}

cd ~/repos/exo-upstream
exec uv run python -c 'from exo.main import main; main()'
LAUNCHEOF
    chmod +x "$local_tmp"
    scp -q "$local_tmp" "${node}:/tmp/exo_B_launch.sh"
    rm -f "$local_tmp"

    # Launch detached with nohup; redirect stdout+stderr to log
    ssh "$node" "nohup /tmp/exo_B_launch.sh > $LOG_FILE 2>&1 &"
    echo "  Started $node"
}

launch_node "macstudio-m4-1" "/ip4/$M4_2_TO_M4_1/tcp/52415/p2p/$M4_2_PEER_ID"
launch_node "macstudio-m4-2" "/ip4/$M4_1_TO_M4_2/tcp/52415/p2p/$M4_1_PEER_ID"
launch_node "macbook-m4"     "/ip4/$M4_1_TO_MBP/tcp/52415/p2p/$M4_1_PEER_ID"

# ======================================================================
# Health Check — wait for 3 nodes
# ======================================================================
echo ""
echo -n "Waiting for cluster to stabilize..."
CLUSTER_READY=false
for i in {1..90}; do
    response=$(curl -s "http://$M4_1_IP:52415/state")
    node_count=$(echo "$response" | jq '.topology.nodes | length' 2>/dev/null)

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
    ssh macstudio-m4-1 "tail -n 20 $LOG_FILE"
    exit 1
fi

echo ""
echo "============================================"
echo "  B-Side cluster is UP (pure upstream)"
echo "  Upstream commit: $COMMIT_M4_1"
echo "  Repo: ~/$UPSTREAM_DIR"
echo "  Logs: ssh <node> 'tail -f $LOG_FILE'"
echo ""
echo "  Create a Pipeline (MLX RDMA) instance"
echo "  with all 3 nodes and run your test prompt."
echo ""
echo "  To switch back to A-Side (your fork):"
echo "    ./start_cluster.sh"
echo "============================================"

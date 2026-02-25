#!/bin/bash

# A/B Testing Script - "A" Side (Pure Upstream)
# Run from your dev MacBook — SSHes into all 3 cluster nodes.
#
# Switches exo to upstream/main on all nodes, rebuilds the dashboard,
# and starts the cluster with STOCK settings. uv handles all dependencies
# (mlx, mlx-lm) from upstream's pyproject.toml automatically.
#
# To restore optimized build after testing:
#   for N in macstudio-m4-1 macstudio-m4-2 macbook-m4; do
#     ssh $N 'cd ~/repos/exo && git checkout main && cd mlx && git checkout main && cd ../mlx-lm && git checkout main'
#   done
#   ./start_cluster.sh

set -e

NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")

echo "============================================"
echo "  A/B Testing - Side A (Pure Upstream)"
echo "============================================"

# ---- Step 1: Kill any running exo processes ----
echo ""
echo "[1/4] Killing existing exo processes..."
for NODE in "${NODES[@]}"; do
    ssh "$NODE" "pkill -9 -f 'exo.main' 2>/dev/null || true" &
done
wait
sleep 2
echo "    Done"

# ---- Step 2: Checkout upstream/main on all nodes ----
echo ""
echo "[2/4] Switching exo to upstream/main on all nodes..."
for NODE in "${NODES[@]}"; do
    echo "  → $NODE..."
    ssh "$NODE" "cd ~/repos/exo && \
        (git remote get-url upstream 2>/dev/null || git remote add upstream https://github.com/exo-explore/exo.git) && \
        git fetch upstream && \
        git checkout --force --detach upstream/main 2>&1 | tail -1"
    EXO_SHA=$(ssh "$NODE" "cd ~/repos/exo && git rev-parse --short HEAD")
    echo "    $NODE: exo=$EXO_SHA ✓"
done

# ---- Step 3: Rebuild dashboard on all nodes ----
echo ""
echo "[3/4] Rebuilding dashboard on all nodes..."
for NODE in "${NODES[@]}"; do
    echo "  → $NODE..."
    ssh "$NODE" "zsh -l -c 'source ~/.zshrc; cd ~/repos/exo/dashboard && npm install --silent && npm run build'" 2>&1 | tail -2
    echo "    $NODE done"
done

# ---- Step 4: Rebuild Rust bindings and sync deps ----
echo ""
echo "[4/5] Rebuilding exo + Rust bindings on all nodes..."
echo "  This uninstalls stale bindings from the fork and rebuilds from upstream."
for NODE in "${NODES[@]}"; do
    echo "  → $NODE..."
    ssh "$NODE" "zsh -l -c 'cd ~/repos/exo && uv pip uninstall exo_pyo3_bindings exo 2>&1 | tail -3 && uv sync --reinstall-package exo_pyo3_bindings --reinstall-package exo 2>&1 | tail -3'"
    echo "    $NODE done"
done

# ---- Step 5: Start exo on all nodes with STOCK settings ----
echo ""
echo "[5/5] Starting cluster with STOCK settings..."
echo ""
echo "  Pure upstream exo — uv handles all deps (mlx, mlx-lm)."
echo "  No EXO_FAST_SYNCH, no EXO_MAX_ACTIVE_TASKS, no adaptive throttle."
echo ""

# Peer IDs & IPs
M4_1_IP="192.168.86.201"
M4_2_IP="192.168.86.202"
M4_1_PEER_ID="12D3KooWDGQKAJUYpqTHzBhVpGzYxQagWRwFqJPzkEYzHxt3SSUg"
M4_2_PEER_ID="12D3KooWQDzFqvjsgFRfheeV7uvtVUP1gruphpgoVELP9pkHBses"

# Minimal stock env — only what upstream needs
STOCK_ENV="EXO_LIBP2P_NAMESPACE=MAC_STUDIO_CLUSTER IBV_FORK_SAFE=1 PYTHONUNBUFFERED=1"

ssh macstudio-m4-1 "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $STOCK_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_2_IP/tcp/52415/p2p/$M4_2_PEER_ID uv run python -m exo.main > /tmp/exo.log 2>&1'"
echo "  Started macstudio-m4-1"

ssh macstudio-m4-2 "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $STOCK_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID uv run python -m exo.main > /tmp/exo.log 2>&1'"
echo "  Started macstudio-m4-2"

ssh macbook-m4 "screen -dmS exorun zsh -l -c 'cd ~/repos/exo && $STOCK_ENV EXO_DISCOVERY_PEERS=/ip4/$M4_1_IP/tcp/52415/p2p/$M4_1_PEER_ID uv run python -m exo.main > /tmp/exo.log 2>&1'"
echo "  Started macbook-m4"

echo ""
echo "============================================"
echo "  A-Side cluster is UP (pure upstream)"
echo "  Logs: ssh <node> 'tail -f /tmp/exo.log'"
echo ""
echo "  Create a Pipeline (MLX RDMA) instance"
echo "  with all 3 nodes and run your test prompt."
echo ""
echo "  To restore optimized build after testing:"
echo "    for N in macstudio-m4-1 macstudio-m4-2 macbook-m4; do"
echo '      ssh $N "cd ~/repos/exo && git checkout main && cd mlx && git checkout main && cd ../mlx-lm && git checkout main"'
echo "    done"
echo "    ./start_cluster.sh"
echo "============================================"

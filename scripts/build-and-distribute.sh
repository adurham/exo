#!/bin/bash
set -e

# Build and distribute script for static cluster
# Pre-builds Rust components locally and distributes to nodes

MASTER_NODE="100.67.156.10"
WORKER_NODES=("100.93.253.67" "100.80.147.125" "100.82.48.77")
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
REPO_PATH="~/repos/exo"

echo "========================================="
echo "Building Rust Components Locally"
echo "========================================="

# Change to repo root
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

# Build the project locally with uv sync (this builds Rust components)
echo "Building exo with uv sync..."
uv sync 2>&1 | tail -100

# Find the built Rust extension module
PYO3_LIB=$(find .venv -name "*exo_pyo3*.so" -o -name "*exo_pyo3*.dylib" 2>/dev/null | head -1)
if [ -z "$PYO3_LIB" ]; then
    echo "❌ ERROR: Could not find built Rust extension module"
    echo "Trying to locate in different paths..."
    find . -name "*exo_pyo3*" -type f 2>/dev/null | head -20
    exit 1
fi

echo "✅ Found Rust extension: $PYO3_LIB"

# Get the site-packages directory
SITE_PACKAGES=$(dirname "$PYO3_LIB")
echo "✅ Site-packages: $SITE_PACKAGES"

# Create a temporary directory for distribution
DIST_DIR="/tmp/exo-build-$$"
mkdir -p "$DIST_DIR/exo_pyo3_bindings"

# Copy the Rust extension and any related files
cp "$PYO3_LIB" "$DIST_DIR/exo_pyo3_bindings/"
# Copy any .pyi stub files if they exist
find "$SITE_PACKAGES/exo_pyo3_bindings" -name "*.pyi" -exec cp {} "$DIST_DIR/exo_pyo3_bindings/" \; 2>/dev/null || true
# Copy __init__.py if it exists
find "$SITE_PACKAGES/exo_pyo3_bindings" -name "__init__.py" -exec cp {} "$DIST_DIR/exo_pyo3_bindings/" \; 2>/dev/null || true

echo ""
echo "========================================="
echo "Distributing to Nodes"
echo "========================================="

# Function to distribute to a node
distribute_to_node() {
    local node=$1
    local node_type=$2
    
    echo "[$node] Distributing Rust bindings to $node_type node..."
    
    # Find the site-packages directory on remote node
    local remote_site_packages=$(ssh $SSH_OPTS "$node" "cd $REPO_PATH && export PATH=\"/opt/homebrew/bin:\$HOME/.local/bin:\$PATH\" && uv pip list --format json 2>/dev/null | python3 -c 'import sys, json; pkgs=json.load(sys.stdin); print([p[\"location\"] for p in pkgs if p[\"name\"]==\"exo\"][0] if any(p[\"name\"]==\"exo\" for p in pkgs) else \"\")' 2>/dev/null || echo ''" 2>&1)
    
    if [ -z "$remote_site_packages" ]; then
        # Try to find it in the uv environment
        remote_site_packages=$(ssh $SSH_OPTS "$node" "cd $REPO_PATH && export PATH=\"/opt/homebrew/bin:\$HOME/.local/bin:\$PATH\" && python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))' 2>/dev/null || echo '.venv/lib/python3.13/site-packages'" 2>&1)
    fi
    
    # Create exo_pyo3_bindings directory on remote node
    ssh $SSH_OPTS "$node" "mkdir -p $REPO_PATH/.venv/lib/python3.13/site-packages/exo_pyo3_bindings" 2>&1
    
    # Copy the Rust extension and files to remote node
    scp $SSH_OPTS -r "$DIST_DIR/exo_pyo3_bindings/"* "$node:$REPO_PATH/.venv/lib/python3.13/site-packages/exo_pyo3_bindings/" 2>&1
    
    echo "✅ [$node] Rust bindings distributed to .venv/lib/python3.13/site-packages/exo_pyo3_bindings/"
}

# Distribute to master
distribute_to_node "$MASTER_NODE" "master"

# Distribute to workers
for node in "${WORKER_NODES[@]}"; do
    distribute_to_node "$node" "worker"
done

# Cleanup
rm -rf "$DIST_DIR"

echo ""
echo "========================================="
echo "✅ Build and Distribution Complete!"
echo "========================================="


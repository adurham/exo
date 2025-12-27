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

# Check for rustup (needed for nightly toolchain)
if ! command -v rustup &> /dev/null; then
    echo "❌ ERROR: rustup not found. Please install rustup:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "   rustup default nightly"
    exit 1
fi

# Ensure we're using Rust nightly
echo "Setting Rust toolchain to nightly..."
rustup default nightly 2>&1 | grep -v "info:" || rustup toolchain install nightly 2>&1 | grep -v "info:" || true
rustup default nightly 2>&1 | grep -v "info:" || true

# Generate protobuf files
echo "Generating protobuf files..."
bash scripts/generate-proto.sh 2>&1 | tail -20
if [ ! -f "src/exo/generated/cluster_pb2.py" ]; then
    echo "❌ ERROR: Failed to generate protobuf files"
    exit 1
fi
echo "✅ Protobuf files generated"

# Build the Rust components directly with cargo
echo "Building Rust bindings with cargo..."
cd rust/exo_pyo3_bindings
cargo +nightly build --release 2>&1 | tail -50
cd ../..

# Build with uv sync to ensure it's properly installed in .venv
echo "Building with uv sync to install in .venv..."
export PATH="$HOME/.cargo/bin:$PATH"
uv sync 2>&1 | tail -50

# Find the installed Rust extension module in .venv
PYO3_LIB=$(find .venv/lib -name "*exo_pyo3*.so" -o -name "*exo_pyo3*.dylib" 2>/dev/null | head -1)
if [ -z "$PYO3_LIB" ]; then
    echo "❌ ERROR: Could not find built Rust extension module in .venv"
    echo "Trying to locate in target directory..."
    PYO3_LIB=$(find target/release -name "libexo_pyo3_bindings.dylib" 2>/dev/null | head -1)
    if [ -z "$PYO3_LIB" ]; then
        echo "Could not find Rust extension module anywhere"
        find . -name "*exo_pyo3*" -type f 2>/dev/null | head -20
        exit 1
    fi
    echo "⚠️  Found in target directory, but should be in .venv - will copy manually"
fi

echo "✅ Found Rust extension: $PYO3_LIB"

# Get the site-packages directory
SITE_PACKAGES=$(dirname "$PYO3_LIB")
echo "✅ Site-packages: $SITE_PACKAGES"

# Create a temporary directory for distribution - copy entire exo_pyo3_bindings directory
DIST_DIR="/tmp/exo-build-$$"
mkdir -p "$DIST_DIR"

# Copy the entire exo_pyo3_bindings directory structure
cp -r "$SITE_PACKAGES/exo_pyo3_bindings" "$DIST_DIR/" 2>/dev/null || {
    echo "⚠️  Could not copy entire directory, copying files individually..."
    mkdir -p "$DIST_DIR/exo_pyo3_bindings"
    cp "$PYO3_LIB" "$DIST_DIR/exo_pyo3_bindings/"
    # Ensure the .so file has the correct name that Python expects
    # Python expects exo_pyo3_bindings.cpython-313-darwin.so or similar
    if [[ "$PYO3_LIB" != *"exo_pyo3_bindings.cpython"* ]]; then
        # Rename to match Python's expected naming
        cp "$PYO3_LIB" "$DIST_DIR/exo_pyo3_bindings/exo_pyo3_bindings.cpython-313-darwin.so"
    fi
    # Copy any .pyi stub files if they exist
    find "$SITE_PACKAGES/exo_pyo3_bindings" -name "*.pyi" -exec cp {} "$DIST_DIR/exo_pyo3_bindings/" \; 2>/dev/null || true
    # Copy __init__.py if it exists
    find "$SITE_PACKAGES/exo_pyo3_bindings" -name "__init__.py" -exec cp {} "$DIST_DIR/exo_pyo3_bindings/" \; 2>/dev/null || true
}

echo ""
echo "========================================="
echo "Distributing to Nodes"
echo "========================================="

# Function to distribute to a node
distribute_to_node() {
    local node=$1
    local node_type=$2
    
    echo "[$node] Distributing Rust bindings and protobuf files to $node_type node..."
    
    # Find the site-packages directory on remote node
    local remote_site_packages=$(ssh $SSH_OPTS "$node" "cd $REPO_PATH && export PATH=\"/opt/homebrew/bin:\$HOME/.local/bin:\$PATH\" && uv pip list --format json 2>/dev/null | python3 -c 'import sys, json; pkgs=json.load(sys.stdin); print([p[\"location\"] for p in pkgs if p[\"name\"]==\"exo\"][0] if any(p[\"name\"]==\"exo\" for p in pkgs) else \"\")' 2>/dev/null || echo ''" 2>&1)
    
    if [ -z "$remote_site_packages" ]; then
        # Try to find it in the uv environment
        remote_site_packages=$(ssh $SSH_OPTS "$node" "cd $REPO_PATH && export PATH=\"/opt/homebrew/bin:\$HOME/.local/bin:\$PATH\" && python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))' 2>/dev/null || echo '.venv/lib/python3.13/site-packages'" 2>&1)
    fi
    
    # Create exo_pyo3_bindings directory on remote node
    ssh $SSH_OPTS "$node" "mkdir -p $REPO_PATH/.venv/lib/python3.13/site-packages/exo_pyo3_bindings" 2>&1
    
    # Copy the Rust extension and files to remote node
    # Ensure the directory exists
    ssh $SSH_OPTS "$node" "mkdir -p $REPO_PATH/.venv/lib/python3.13/site-packages/exo_pyo3_bindings" 2>&1
    # Copy all files from the distribution directory
    scp $SSH_OPTS -r "$DIST_DIR/exo_pyo3_bindings/"* "$node:$REPO_PATH/.venv/lib/python3.13/site-packages/exo_pyo3_bindings/" 2>&1
    
    echo "✅ [$node] Rust bindings distributed to .venv/lib/python3.13/site-packages/exo_pyo3_bindings/"
    
    # Distribute protobuf files
    echo "[$node] Distributing protobuf files..."
    ssh $SSH_OPTS "$node" "mkdir -p $REPO_PATH/src/exo/generated" 2>&1
    scp $SSH_OPTS "$REPO_ROOT/src/exo/generated/"* "$node:$REPO_PATH/src/exo/generated/" 2>&1
    echo "✅ [$node] Protobuf files distributed to src/exo/generated/"
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


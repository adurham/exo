#!/bin/bash
set -e

# Source uv environment if available
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
fi

# Add common paths (ensure uv is available)
export PATH="$HOME/.local/bin:$PATH"

# Verify uv is available, if not try to find it
UV_BIN=""
if ! command -v uv &> /dev/null; then
    # Try common locations
    if [ -f "$HOME/.local/bin/uv" ]; then
        UV_BIN="$HOME/.local/bin/uv"
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        UV_BIN="$HOME/.cargo/bin/uv"
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    if [ -z "$UV_BIN" ] && ! command -v uv &> /dev/null; then
        echo "ERROR: uv not found in PATH. Please install uv or ensure it's in PATH."
        exit 1
    fi
else
    UV_BIN=$(command -v uv)
fi

cd ~/repos/exo/

# Kill any existing exo processes and all Python processes related to exo
echo "Stopping any existing exo processes..."
# Kill by process name patterns
pkill -9 -f "uv run exo" 2>/dev/null || true
pkill -9 -f "exo" 2>/dev/null || true
pkill -9 -f "python.*exo" 2>/dev/null || true
pkill -9 -f "python.*spawn_main" 2>/dev/null || true
pkill -9 -f "python.*multiprocessing" 2>/dev/null || true
# Kill by PID if we can find them
for pid in $(pgrep -f "exo|spawn_main" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null || true
done
# Kill any Python processes in the exo directory
for pid in $(ps aux | grep -E "python.*exo|python.*\.venv.*exo" | grep -v grep | awk '{print $2}'); do
    kill -9 "$pid" 2>/dev/null || true
done
echo "Waiting for processes to terminate..."
sleep 3
# Verify processes are killed
if pgrep -f "exo|spawn_main" > /dev/null 2>&1; then
    echo "WARNING: Some exo processes are still running, force killing..."
    pkill -9 -f "exo|spawn_main" 2>/dev/null || true
    sleep 2
    # Final check - if still running, try to kill by working directory
    for pid in $(lsof +D ~/repos/exo 2>/dev/null | grep -E "python|exo" | awk '{print $2}' | sort -u); do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
fi

# Purge memory caches to free up RAM (macOS specific)
echo "Purging memory caches..."
# Use sudo purge to force memory reclamation
if command -v purge &> /dev/null; then
    sudo purge 2>&1 || echo "Warning: sudo purge failed (may require password or sudo access)"
fi

# Force memory pressure to reclaim inactive memory by allocating and freeing
# This works even without purge permissions
echo "Forcing memory reclamation..."
# Use the exo venv python if available, otherwise system python
PYTHON_BIN="python3"
if [ -f "$HOME/repos/exo/.venv/bin/python3" ]; then
    PYTHON_BIN="$HOME/repos/exo/.venv/bin/python3"
fi

$PYTHON_BIN << 'EOF'
import ctypes
import gc

# Allocate multiple chunks to create memory pressure and force inactive page reclamation
bufs = []
try:
    # Try to allocate several large chunks (5GB total) to create pressure
    for i in range(5):
        try:
            size = 1024 * 1024 * 1024  # 1GB per chunk
            buf = ctypes.create_string_buffer(size)
            # Touch memory to ensure allocation
            buf[0] = b'x'
            buf[size-1] = b'y'
            bufs.append(buf)
        except MemoryError:
            # If we can't allocate more, that's fine - we've created pressure
            break
    
    # Immediately free all buffers to reclaim memory
    del bufs
    gc.collect()
    print(f"Memory reclamation successful (allocated {len(bufs) if 'bufs' in locals() else 0} chunks)")
except MemoryError:
    print("Could not allocate memory for reclamation (system may be low on memory)")
except Exception as e:
    print(f"Memory reclamation failed: {e}")
EOF

sleep 2
# Verify memory was freed
echo "Checking memory after cleanup..."
vm_stat | head -n 5 || echo "Could not check memory stats"

# Reset to clean state and pull latest
echo "Resetting to clean state..."
git reset --hard HEAD
git clean -fd
git fetch
git reset --hard origin/new_main

# Build dashboard if npm is available, otherwise skip
if command -v npm &> /dev/null; then
    echo "Building dashboard..."
    cd dashboard
    npm install
    npm run build
    cd ..
else
    echo "Warning: npm not found, skipping dashboard build (using existing build if available)"
fi

# Run exo in background with sudo (logging to ~/.exo/exo.log per node)
echo "Starting exo in background with sudo..."
# Preserve PATH and HOME when running with sudo, and run from the correct directory
if [ -n "$UV_BIN" ]; then
    nohup sudo -E env PATH="$PATH" HOME="$HOME" bash -c "cd $HOME/repos/exo && $UV_BIN run exo" > /dev/null 2>&1 &
else
    nohup sudo -E env PATH="$PATH" HOME="$HOME" bash -c "cd $HOME/repos/exo && uv run exo" > /dev/null 2>&1 &
fi
echo "Exo started with PID: $!"
echo "Logs are being written to ~/.exo/exo.log on this node"


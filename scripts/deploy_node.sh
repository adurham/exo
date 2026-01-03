#!/bin/bash

# --- CONFIGURATION ---
# IMPORTANT: Run 'hostname' in your Mac terminal to verify this exact name
HEAD_HOSTNAME="macstudio-m4" 
# Verify this path matches where you cloned the repo on your Macs
REPO_DIR="$HOME/repos/exo" 

# --- 1. SET ENVIRONMENT ---
# SSH sessions are "non-interactive" and don't load your .zshrc by default.
# We force-load the path so 'uv', 'python', and 'git' work.
export PATH=$PATH:/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin

echo "📍 Deploying on $(hostname)..."

# --- 2. UPDATE CODE ---
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" || exit
    echo "⬇️ Pulling latest code..."
    # Safety: Discard local changes so the pull always succeeds
    git reset --hard 
    git pull origin main
else
    echo "❌ Error: Directory $REPO_DIR not found on $(hostname)!"
    exit 1
fi

# --- 3. UPDATE DEPENDENCIES ---
echo "📦 Syncing dependencies..."
# If uv asks for confirmation, this flag skips it
uv sync --frozen 

# --- 4. RESTART EXO ---
echo "🛑 Killing existing Exo processes..."
# '|| true' prevents the script from crashing if no exo process is running
pkill -f "exo" || true 
sleep 2

echo "🚀 Starting Exo..."

# Logic to decide if this is the Head node or a Worker
# Note: 'nohup' keeps the process alive after SSH disconnects
if [[ "$(hostname)" == "$HEAD_HOSTNAME" ]]; then
    echo "   -> Starting as HEAD node"
    nohup exo start > exo.log 2>&1 &
else
    echo "   -> Starting as WORKER node"
    nohup exo start --worker > exo.log 2>&1 &
fi

echo "✅ Deployment complete for $(hostname)."
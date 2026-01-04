#!/bin/bash

# --- CONFIGURATION ---
HEAD_HOSTNAME="Adams-Mac-Studio-M4"
# FIX: Updated to the correct path you found
REPO_DIR="$HOME/repos/exo" 
BRANCH="local_mac_cluster"

# --- 1. SET ENVIRONMENT ---
export PATH=$PATH:/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin

echo "📍 Deploying on $(hostname)..."

# --- 2. UPDATE CODE ---
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" || exit
    
    # Point to your fork to ensure we find the branch
    git remote set-url origin https://github.com/adurham/exo.git || true
    
    echo "⬇️ Pulling branch: $BRANCH..."
    git fetch origin
    git reset --hard "origin/$BRANCH"
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "❌ Error: Directory $REPO_DIR not found on $(hostname)!"
    exit 1
fi

# --- 3. UPDATE DEPENDENCIES ---
echo "📦 Syncing dependencies..."
uv sync --frozen 

# --- 4. RESTART EXO ---
echo "🛑 Killing existing Exo processes..."
pkill -f "exo" || true 
sleep 2

echo "🚀 Starting Exo..."

# Build dashboard if missing
if [ ! -d "dashboard/build" ]; then
    echo "📦 Building Dashboard..."
    cd dashboard && npm install && npm run build && cd ..
fi

# --- 5. CONFIGURE MEMORY LIMITS ---
# Detect host type and set wired memory limit to prevent freezing
# Studio: 75% | MacBook: 60%
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"Studio"* ]]; then
    export EXO_WIRED_LIMIT_PCT=0.75
    echo "🧠 Configured Wired Memory Limit: 75% (Studio Profile)"
else
    export EXO_WIRED_LIMIT_PCT=0.60
    echo "🧠 Configured Wired Memory Limit: 60% (Laptop Profile)"
fi

# FIX: Use 'uv run' so it finds the binary inside the virtualenv
nohup uv run exo > exo.log 2>&1 &

echo "✅ Deployment complete for $(hostname)."
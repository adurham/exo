#!/bin/bash

# --- CONFIGURATION ---
REPO_DIR="$HOME/repos/exo" 
BRANCH="local_mac_cluster"

# --- 1. SET ENVIRONMENT ---
# Force path to ensure 'uv' and 'python' are found
export PATH=$PATH:/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin

echo "📍 Deploying on $(hostname)..."

# --- 2. UPDATE CODE ---
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" || exit
    
    # --- CRITICAL FIX: POINT TO YOUR FORK ---
    # Your logs showed the Macs are pulling from 'exo-explore/exo'.
    # This command forces them to pull from YOUR fork so they find the branch.
    git remote set-url origin https://github.com/adurham/exo.git || true
    
    echo "⬇️ Pulling branch: $BRANCH..."
    git fetch origin
    
    # Force the local state to match your remote branch exactly
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

# Unified start command - relies on your code/config to determine Head vs Worker
nohup exo start > exo.log 2>&1 &

echo "✅ Deployment complete for $(hostname)."
#!/bin/bash
set -e

# Target Hosts
HOSTS=(
    "macstudio-m4"
    "macbook-m4"
    "work-macbook-m4"
)

# Target Directory on Remote Hosts
REMOTE_DIR="~/repos/exo/"

# Exclude list to avoid syncing unnecessary files
EXCLUDES=(
    "--exclude=.git"
    "--exclude=.env"
    "--exclude=.venv"
    "--exclude=__pycache__"
    "--exclude=.pytest_cache"
    "--exclude=.gemini"
    "--exclude=uv.lock"
    "--exclude=target"
    "--exclude=.idea"
    "--exclude=.vscode"
    "--exclude=.DS_Store"
)

echo "Deploying to worker nodes..."

for HOST in "${HOSTS[@]}"; do
    echo "========================================"
    echo "Syncing to ${HOST}..."
    echo "========================================"
    
    rsync -avz --delete "${EXCLUDES[@]}" ./ "${HOST}:${REMOTE_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully synced to ${HOST}"
    else
        echo "Failed to sync to ${HOST}"
        exit 1
    fi
done

echo "Configuring System Limits (requires sudo)..."
for HOST in "${HOSTS[@]}"; do
    echo "========================================"
    echo "Configuring ${HOST}..."
    
    # Calculate tiered limit: 85% for >64GB (Studio), 80% for others (MacBooks)
    # The Studio needs more headroom because it acts as the Hub (Network Buffers) + File Cache for the massive shard
    ssh -t "${HOST}" "\
        RAM_BYTES=\$(sysctl -n hw.memsize); \
        RAM_GB=\$(( \$RAM_BYTES / 1024 / 1024 / 1024 )); \
        if [ \"\$RAM_GB\" -gt 64 ]; then PERCENT=85; else PERCENT=80; fi; \
        LIMIT_MB=\$(( \$RAM_BYTES / 1024 / 1024 * \$PERCENT / 100 )); \
        echo \"[Setup] Node RAM: \${RAM_GB} GB. Using \${PERCENT}% Limit. Setting GPU Wired Limit to \${LIMIT_MB} MB\"; \
        sudo sysctl iogpu.wired_limit_mb=\$LIMIT_MB || echo \"[Warning] Failed to set GPU limit. Ensure you have sudo privileges.\"; \
        echo \"[Setup] Enabling IP Forwarding for Cluster Routing...\"; \
        sudo sysctl -w net.inet.ip.forwarding=1 || echo \"[Warning] Failed to enable IP forwarding.\""
done

echo "Starting services on all nodes in parallel..."

# Command to run on remote hosts
# We use screen to keep it running in the background and detached
# We source .zprofile and .zshrc to ensure env vars (path to node, uv, brew, etc) are loaded
REMOTE_CMD="source ~/.zprofile; source ~/.zshrc; cd ~/repos/exo/dashboard && npm install && npm run build && cd .. && uv sync --prerelease=allow && uv run exo"

for HOST in "${HOSTS[@]}"; do
    echo "Triggering service restart on ${HOST}..."
    # Start screen session detached (-dm) named 'exo' (-S exo)
    # Use zsh explicitly    # Wrap the worker command to capture exit code and keep screen open for debugging
    # We must start in ~/repos/exo because that's where the uv project matches
    # We MUST build the dashboard because exo initializes API even on workers (sometimes)
    CMD="source ~/.zprofile; source ~/.zshrc; \
         cd ~/repos/exo/dashboard && npm install && npm run build && cd .. && \
         uv run exo -vv > /tmp/exo.log 2>&1; \
         EXIT_CODE=\$?; \
         echo \"Process exited with code \$EXIT_CODE\" >> /tmp/exo_exit_code.txt; \
         echo \"Process exited with code \$EXIT_CODE. Sleeping for debugging...\"; \
         sleep 86400"

    # Aggressively kill existing instances to prevent zombies/conflicts
    ssh "${HOST}" "pkill -9 -f 'uv run exo' || true; pkill -9 -f 'python.*exo' || true; screen -S exo -X quit || true; screen -wipe || true; screen -dmS exo /bin/zsh -c '${CMD}'" &
done

wait

echo "========================================"
echo "Deployment and Restart Triggered!"
echo "========================================"

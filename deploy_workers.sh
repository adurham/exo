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
    "--exclude=node_modules"
)

echo "Deploying to worker nodes..."

for HOST in "${HOSTS[@]}"; do
    echo "========================================"
    echo "Syncing to ${HOST}..."
    echo "========================================"
    
    rsync -az --delete "${EXCLUDES[@]}" ./ "${HOST}:${REMOTE_DIR}"
    
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
        if [ \"\$RAM_GB\" -gt 64 ]; then PERCENT=95; else PERCENT=80; fi; \
        LIMIT_MB=\$(( \$RAM_BYTES / 1024 / 1024 * \$PERCENT / 100 )); \
        echo \"[Setup] Node RAM: \${RAM_GB} GB. Using \${PERCENT}% Limit. Setting GPU Wired Limit to \${LIMIT_MB} MB\"; \
        sudo sysctl iogpu.wired_limit_mb=\$LIMIT_MB || echo \"[Warning] Failed to set GPU limit. Ensure you have sudo privileges.\"; \
        echo \"[Setup] Enabling IP Forwarding for Cluster Routing...\"; \
        sudo sysctl -w net.inet.ip.forwarding=1 || echo \"[Warning] Failed to enable IP forwarding.\""
done

# Thunderbolt/RDMA Connectivity Pre-checks
echo "Checking Thunderbolt/RDMA connectivity..."
echo "========================================"

# Define the Thunderbolt IP pairs to test (source host, target host, target TB IP)
# macstudio-m4 <-> macbook-m4 via 192.168.201.x
# macstudio-m4 <-> work-macbook-m4 via 192.168.202.x
# macbook-m4 <-> work-macbook-m4 via 192.168.205.x
TB_TESTS=(
    "macstudio-m4|macbook-m4|192.168.201.2"
    "macbook-m4|macstudio-m4|192.168.201.1"
    "macstudio-m4|work-macbook-m4|192.168.202.2"
    "work-macbook-m4|macstudio-m4|192.168.202.1"
    "macbook-m4|work-macbook-m4|192.168.205.2"
    "work-macbook-m4|macbook-m4|192.168.205.1"
)

TB_CHECK_PASSED=true
for TEST in "${TB_TESTS[@]}"; do
    IFS='|' read -r SRC_HOST DEST_HOST DEST_IP <<< "$TEST"
    echo "[TB Check] $SRC_HOST -> $DEST_HOST ($DEST_IP)"
    if ssh "$SRC_HOST" "ping -c 1 -W 2 $DEST_IP > /dev/null 2>&1"; then
        echo "  ✓ OK"
    else
        echo "  ✗ FAILED - Thunderbolt connectivity issue!"
        TB_CHECK_PASSED=false
    fi
done

if [ "$TB_CHECK_PASSED" = false ]; then
    echo ""
    echo "⚠️  WARNING: Some Thunderbolt connectivity tests failed."
    echo "   RDMA/jaccl may not work correctly."
    echo "   Check Thunderbolt cables and network settings."
    echo ""
fi

# Check that RDMA over Thunderbolt kext is loaded (macOS 26.2+)
echo ""
echo "Checking RDMA over Thunderbolt status..."
for HOST in "${HOSTS[@]}"; do
    echo "[RDMA Check] $HOST"
    # Check if AppleThunderboltRDMA kext is loaded
    RDMA_KEXT=$(ssh "$HOST" "kextstat 2>/dev/null | grep -c 'AppleThunderboltRDMA' || echo '0'")
    if [ "$RDMA_KEXT" -gt 0 ]; then
        echo "  ✓ AppleThunderboltRDMA kext loaded"
    else
        echo "  ✗ AppleThunderboltRDMA kext NOT loaded"
        echo "    Enable RDMA: boot to Recovery, run 'rdma_ctl enable', reboot"
    fi
done

echo ""
echo "Starting services on all nodes in parallel..."

for HOST in "${HOSTS[@]}"; do
    # Start screen session detached (-dm) named 'exo' (-S exo)
    # We use zsh explicitly    # Wrap the worker command to capture exit code and keep screen open for debugging
    # We must start in ~/repos/exo because that's where the uv project matches
    # We MUST build the dashboard because exo initializes API even on workers (sometimes)
    
    # We use a heredoc-like strategy or just a carefully quoted string to avoid quoting hell.
    # We will write the command to a variable.
    # Note: We use single quotes for the inner command to prevent local expansion, but we need to escape single quotes inside it if any.
    REMOTE_CMD="source ~/.zprofile; source ~/.zshrc; cd ~/repos/exo; cd dashboard && npm install && npm run build && cd .. && uv run exo -vv > /tmp/exo.log 2>&1"

    # Aggressively kill existing instances to prevent zombies/conflicts
    # We start screen with /bin/zsh -c 'CMD'
    # We need to escape the CMD for the ssh shell, and then for the screen command.
    # The safest way is to let screen spawn zsh and pass the command.
    # But screen -dmS exo /bin/zsh -c "..." is tricky with nested quotes.
    # Alternative: Write the command to a temp file ON THE REMOTE and run it, then delete it. This is 'putting it in the deploy script' logic.
    
    echo "Writing startup script to remote..."
    ssh "${HOST}" "cat > /tmp/exo_startup.sh" <<EOF
#!/bin/zsh
source ~/.zprofile
source ~/.zshrc
cd ~/repos/exo/dashboard && npm install && npm run build && cd .. && uv run exo -vv > /tmp/exo.log 2>&1
EOF
    
    ssh "${HOST}" "chmod +x /tmp/exo_startup.sh; pkill -9 -f 'uv run exo' || true; pkill -9 -f 'python.*exo' || true; screen -S exo -X quit || true; screen -wipe || true; screen -dmS exo /tmp/exo_startup.sh" &
done

wait

echo "========================================"
echo "Deployment and Restart Triggered!"
echo "========================================"

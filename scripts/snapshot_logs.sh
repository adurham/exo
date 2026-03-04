#!/bin/bash
# Snapshot cluster logs from all 3 nodes before a restart or deploy.
#
# Usage: ./scripts/snapshot_logs.sh [LABEL]
#
# Pulls /tmp/exo.log from each node into timestamped local files.
# Run this BEFORE restarting the cluster to preserve crash forensics.

set -euo pipefail

LABEL="${1:-$(date +%Y%m%d_%H%M%S)}"
NODES=("macstudio-m4-1" "macstudio-m4-2" "macbook-m4")
OUT_DIR="tmp/logs_${LABEL}"

mkdir -p "$OUT_DIR"

echo "=== Snapshotting cluster logs ==="
echo "Label:  $LABEL"
echo "Output: $OUT_DIR/"
echo ""

for NODE in "${NODES[@]}"; do
    echo -n "  $NODE... "
    OUTFILE="$OUT_DIR/${NODE}.log"

    if scp "$NODE:/tmp/exo.log" "$OUTFILE" 2>/dev/null; then
        LINES=$(wc -l < "$OUTFILE" | tr -d ' ')
        SIZE=$(du -h "$OUTFILE" | awk '{print $1}')
        echo "$LINES lines ($SIZE)"

        # Extract key diagnostics
        CRASHES=$(grep -c "SIGABRT\|Traceback\|CRITICAL\|panic\|assertion" "$OUTFILE" 2>/dev/null || echo "0")
        DEADLOCKS=$(grep -c "deadlock\|DEADLOCK\|timed out waiting" "$OUTFILE" 2>/dev/null || echo "0")
        if [ "$CRASHES" -gt 0 ] || [ "$DEADLOCKS" -gt 0 ]; then
            echo "    !! $CRASHES crash indicators, $DEADLOCKS deadlock indicators"
        fi
    else
        echo "FAILED (node unreachable or no log file)"
    fi
done

# Also capture cluster state if API is up
echo ""
echo -n "  Cluster state... "
if curl -sf "http://192.168.86.201:52415/state" > "$OUT_DIR/cluster_state.json" 2>/dev/null; then
    NODE_COUNT=$(python3 -c 'import json; d=json.load(open("'"$OUT_DIR"'/cluster_state.json")); print(len(d.get("topology",{}).get("nodes",{})))' 2>/dev/null || echo "?")
    echo "saved ($NODE_COUNT nodes)"
else
    echo "API unreachable (skipped)"
    rm -f "$OUT_DIR/cluster_state.json"
fi

# Capture git commit from each node
echo ""
echo "  Git commits:"
for NODE in "${NODES[@]}"; do
    COMMIT=$(ssh "$NODE" "cd ~/repos/exo && git rev-parse --short HEAD" 2>/dev/null || echo "unreachable")
    echo "    $NODE: $COMMIT"
done

echo ""
echo "Logs saved to $OUT_DIR/"
echo "To inspect: less $OUT_DIR/<node>.log"

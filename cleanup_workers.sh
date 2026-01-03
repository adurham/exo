#!/bin/bash
HOSTS=("macstudio-m4" "macbook-m4" "work-macbook-m4")

echo "Cleaning up worker nodes..."

for HOST in "${HOSTS[@]}"; do
    echo "========================================"
    echo "Cleaning $HOST..."
    ssh "$HOST" "pkill -9 -f 'exo'; pkill -9 -f 'python3'; pkill -9 -f 'mlx'; rm -f /tmp/exo_startup.sh" 2>/dev/null
    
    # Reset GPU state (User authorized)
    echo "Resetting GPU state (killing WindowServer) on $HOST..."
    ssh "$HOST" "sudo pkill -HUP WindowServer" 2>/dev/null

    # Attempt to reset Networking (Thunderbolt bridge often gets stuck)
    echo "Resetting Network state on $HOST..."
    ssh "$HOST" "sudo killall -HUP mDNSResponder; sudo ifconfig bridge0 down && sudo ifconfig bridge0 up" 2>/dev/null

    echo "Checking for lingering processes on $HOST..."
    ssh "$HOST" "ps -A -o pid,stat,comm | grep -E 'exo|python|mlx' | grep -v grep || echo 'No lingering processes.'"
done

echo "========================================"
echo "Cleanup complete."

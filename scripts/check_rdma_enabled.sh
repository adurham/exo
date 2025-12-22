#!/bin/bash
# Check if RDMA is enabled on macOS

echo "=========================================="
echo "Checking RDMA Enablement Status"
echo "=========================================="
echo ""

# Check sysctl settings
echo "=== Sysctl RDMA Settings ==="
sysctl -a 2>/dev/null | grep -i rdma | head -10

echo ""
echo "=== NVRAM RDMA Settings ==="
nvram -p 2>/dev/null | grep -i rdma || echo "No RDMA settings in NVRAM"

echo ""
echo "=== RDMA Device Status ==="
if command -v ibv_devinfo &> /dev/null; then
    for device in $(ibv_devices 2>/dev/null | grep rdma_ | awk '{print $1}'); do
        echo "Device: $device"
        ibv_devinfo -d "$device" 2>/dev/null | grep -A 5 "port:" | head -8
        echo ""
    done
else
    echo "ibv_devinfo not found"
fi

echo ""
echo "=== Notes ==="
echo "If RDMA is not enabled:"
echo "1. Boot into Recovery Mode (hold Cmd+R during startup)"
echo "2. Open Terminal from Utilities menu"
echo "3. Run the command to enable RDMA (exact command TBD)"
echo "4. Reboot the system"
echo ""
echo "RDMA must be enabled on ALL nodes for MLX to use it."


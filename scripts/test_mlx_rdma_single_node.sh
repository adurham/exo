#!/bin/bash
# Test MLX RDMA on a single node (no coordination required)
# This just checks if MLX can detect RDMA devices and if is_available() returns True

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hostname>"
    echo "Example: $0 macstudio-m4"
    exit 1
fi

HOSTNAME=$1

echo "=========================================="
echo "Testing MLX RDMA on ${HOSTNAME} (single node)"
echo "=========================================="
echo ""

ssh "${HOSTNAME}" "cd /Users/adam.durham/repos/exo && \
    uv run python -c \"
import mlx.core as mx
import subprocess
import sys

print('Checking RDMA devices...')
try:
    result = subprocess.run(['ibv_devices'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        devices = []
        for line in result.stdout.split('\\n'):
            line = line.strip()
            if line and not line.startswith('device') and not line.startswith('------'):
                parts = line.split()
                if parts and parts[0].startswith('rdma_'):
                    devices.append(parts[0])
        print(f'Available RDMA devices: {sorted(devices)}')
    else:
        print(f'WARNING: ibv_devices returned code {result.returncode}')
        print(f'Output: {result.stderr}')
except FileNotFoundError:
    print('ERROR: ibv_devices command not found')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)

print('')
print('Checking mx.distributed.is_available()...')
is_available = mx.distributed.is_available()
print(f'mx.distributed.is_available() = {is_available}')

if is_available:
    print('')
    print('✓ MLX thinks distributed backend is available')
    print('  This is a good sign - MLX can detect RDMA')
else:
    print('')
    print('✗ MLX thinks distributed backend is NOT available')
    print('  This means MLX cannot detect any distributed backend')
    print('  Possible causes:')
    print('    - RDMA drivers not properly installed')
    print('    - MLX on macOS doesn\\'t support RDMA')
    sys.exit(1)
\"" 2>&1 | sed "s/^/[${HOSTNAME}] /"

echo ""
echo "=========================================="
echo "Test complete"
echo "=========================================="


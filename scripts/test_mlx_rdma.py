#!/usr/bin/env python3
"""Test MLX RDMA initialization on a single node.

This script tests if MLX can initialize RDMA with the same configuration
that the main application uses. It can be run on each node to verify
RDMA is working.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:
    print("ERROR: MLX not installed. Install with: pip install mlx")
    sys.exit(1)


def get_available_rdma_devices() -> list[str]:
    """Get list of available RDMA devices on this system."""
    try:
        result = subprocess.run(
            ["ibv_devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            print(f"WARNING: ibv_devices returned code {result.returncode}")
            return []
        
        devices = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line or line.startswith("device") or line.startswith("------"):
                continue
            parts = line.split()
            if parts and parts[0].startswith("rdma_"):
                devices.append(parts[0])
        return sorted(devices)
    except FileNotFoundError:
        print("WARNING: ibv_devices command not found")
        return []
    except Exception as e:
        print(f"WARNING: Error getting RDMA devices: {e}")
        return []


def create_test_devices_file(rank: int, world_size: int, devices_file: Path) -> list[list[str | None]]:
    """Create a test devices matrix.
    
    For testing, we'll create a simple matrix where each node uses
    its first available RDMA device to connect to others.
    """
    available_devices = get_available_rdma_devices()
    
    if not available_devices:
        print(f"ERROR: No RDMA devices found on this system")
        return []
    
    # Create a simple matrix: each node uses its first device for all connections
    # This is a simplified version for testing
    matrix: list[list[str | None]] = []
    for i in range(world_size):
        row: list[str | None] = []
        for j in range(world_size):
            if i == j:
                row.append(None)
            elif i == rank:
                # This node uses its first available device
                row.append(available_devices[0] if available_devices else None)
            else:
                # Other nodes - we don't know their devices, use first available as placeholder
                row.append(available_devices[0] if available_devices else None)
        matrix.append(row)
    
    # Write the matrix to file
    with open(devices_file, "w") as f:
        json.dump(matrix, f)
    
    return matrix


def test_mlx_rdma_init(
    rank: int,
    world_size: int,
    coordinator_ip: str,
    coordinator_port: int,
    devices_file: Path,
) -> bool:
    """Test MLX RDMA initialization with given parameters."""
    
    print(f"\n{'='*60}")
    print(f"Testing MLX RDMA Initialization")
    print(f"{'='*60}")
    print(f"Rank: {rank}")
    print(f"World Size: {world_size}")
    print(f"Coordinator: {coordinator_ip}:{coordinator_port}")
    print(f"Devices File: {devices_file}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check RDMA devices
    available_devices = get_available_rdma_devices()
    print(f"\nAvailable RDMA Devices: {available_devices}")
    
    if not available_devices:
        print("ERROR: No RDMA devices found. Cannot test RDMA.")
        return False
    
    # Create devices file
    matrix = create_test_devices_file(rank, world_size, devices_file)
    if not matrix:
        print("ERROR: Failed to create devices file")
        return False
    
    print(f"\nDevices Matrix:")
    for i, row in enumerate(matrix):
        print(f"  Rank {i}: {row}")
    
    # Read back the file to verify
    with open(devices_file, "r") as f:
        file_content = f.read()
    print(f"\nDevices File Content: {file_content}")
    
    # Set environment variables (same as in the main code)
    devices_file_str = str(devices_file.relative_to(Path.cwd())) if devices_file.is_relative_to(Path.cwd()) else str(devices_file)
    os.environ["MLX_IBV_DEVICES"] = devices_file_str
    os.environ["MLX_RANK"] = str(rank)
    os.environ["MLX_IBV_COORDINATOR"] = f"{coordinator_ip}:{coordinator_port}"
    os.environ["MLX_WORLD_SIZE"] = str(world_size)
    
    # Clear MLX_HOSTFILE if it exists
    if "MLX_HOSTFILE" in os.environ:
        del os.environ["MLX_HOSTFILE"]
    
    print(f"\nEnvironment Variables:")
    print(f"  MLX_IBV_DEVICES={os.environ.get('MLX_IBV_DEVICES')}")
    print(f"  MLX_RANK={os.environ.get('MLX_RANK')}")
    print(f"  MLX_IBV_COORDINATOR={os.environ.get('MLX_IBV_COORDINATOR')}")
    print(f"  MLX_WORLD_SIZE={os.environ.get('MLX_WORLD_SIZE')}")
    print(f"  MLX_HOSTFILE={os.environ.get('MLX_HOSTFILE', 'NOT SET')}")
    
    # Check if distributed is available
    print(f"\nChecking mx.distributed.is_available()...")
    is_available = mx.distributed.is_available()
    print(f"  mx.distributed.is_available() = {is_available}")
    
    if not is_available:
        print("WARNING: mx.distributed.is_available() returned False")
        print("  This means MLX doesn't think any distributed backend is available")
        print("  This could indicate:")
        print("    - RDMA drivers not properly installed")
        print("    - Network connectivity issues")
        print("    - MLX on macOS doesn't support RDMA")
    
    # Try to initialize
    print(f"\nAttempting mx.distributed.init(backend='any', strict=False)...")
    try:
        # Use strict=False to see what happens (won't fail immediately)
        group = mx.distributed.init(backend="any", strict=False)
        
        actual_rank = group.rank()
        actual_size = group.size()
        
        print(f"\n✓ MLX distributed.init succeeded!")
        print(f"  Group rank: {actual_rank} (expected: {rank})")
        print(f"  Group size: {actual_size} (expected: {world_size})")
        
        if actual_size == 1:
            print(f"\n⚠ WARNING: Got singleton group (size=1)")
            print(f"  This means MLX fell back to non-distributed mode")
            print(f"  RDMA is NOT being used")
            return False
        
        if actual_rank != rank:
            print(f"\n⚠ WARNING: Rank mismatch!")
            print(f"  Expected rank {rank} but got {actual_rank}")
            return False
        
        if actual_size != world_size:
            print(f"\n⚠ WARNING: Size mismatch!")
            print(f"  Expected size {world_size} but got {actual_size}")
            return False
        
        print(f"\n✓✓✓ SUCCESS: MLX RDMA initialized correctly!")
        print(f"  All ranks and sizes match expectations")
        return True
        
    except RuntimeError as e:
        print(f"\n✗ MLX distributed.init failed!")
        print(f"  Error: {e}")
        print(f"\n  This indicates MLX cannot initialize any distributed backend")
        print(f"  Possible causes:")
        print(f"    - RDMA not properly configured")
        print(f"    - Coordinator not reachable")
        print(f"    - All nodes not initializing simultaneously")
        print(f"    - MLX on macOS doesn't support RDMA via MLX_IBV_DEVICES")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error during MLX distributed.init:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MLX RDMA initialization")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this node (0, 1, 2, ...)")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of nodes")
    parser.add_argument("--coordinator-ip", type=str, required=True, help="IP address of rank 0 coordinator")
    parser.add_argument("--coordinator-port", type=int, default=52414, help="Port of rank 0 coordinator")
    parser.add_argument("--devices-file", type=str, default="./test_hosts_{rank}.json", help="Path to devices file (use {rank} for rank number)")
    
    args = parser.parse_args()
    
    # Expand {rank} in devices_file path
    devices_file = Path(args.devices_file.format(rank=args.rank))
    
    success = test_mlx_rdma_init(
        rank=args.rank,
        world_size=args.world_size,
        coordinator_ip=args.coordinator_ip,
        coordinator_port=args.coordinator_port,
        devices_file=devices_file,
    )
    
    if success:
        print(f"\n{'='*60}")
        print(f"✓ TEST PASSED: MLX RDMA initialization works on this node")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED: MLX RDMA initialization does not work on this node")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Test MLX Ring backend initialization on a single node.

This script tests if MLX can initialize distributed communication
using the ring backend with a hostfile.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:
    print("ERROR: MLX not installed. Install with: pip install mlx")
    sys.exit(1)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def test_mlx_ring_init(
    rank: int,
    world_size: int,
    hosts: list[str],
    timeout_seconds: int = 10,
) -> bool:
    """Test MLX Ring backend initialization with given parameters."""
    
    print(f"\n{'='*60}")
    print(f"Testing MLX Ring Backend Initialization")
    print(f"{'='*60}")
    print(f"Rank: {rank}")
    print(f"World Size: {world_size}")
    print(f"Hosts: {hosts}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Create hostfile
    hostfile = Path(f"./test_hosts_ring_{rank}.json")
    with open(hostfile, "w") as f:
        json.dump(hosts, f)
    
    print(f"\nHostfile: {hostfile}")
    with open(hostfile, "r") as f:
        file_content = f.read()
    print(f"Hostfile Content: {file_content}")
    
    # Set environment variables
    os.environ["MLX_HOSTFILE"] = str(hostfile)
    os.environ["MLX_RANK"] = str(rank)
    os.environ["MLX_RING_VERBOSE"] = "1"
    
    # Clear RDMA-specific vars if they exist
    for var in ["MLX_IBV_DEVICES", "MLX_IBV_COORDINATOR", "MLX_WORLD_SIZE"]:
        if var in os.environ:
            del os.environ[var]
    
    print(f"\nEnvironment Variables:")
    print(f"  MLX_HOSTFILE={os.environ.get('MLX_HOSTFILE')}")
    print(f"  MLX_RANK={os.environ.get('MLX_RANK')}")
    print(f"  MLX_RING_VERBOSE={os.environ.get('MLX_RING_VERBOSE')}")
    
    # Check if distributed is available
    print(f"\nChecking mx.distributed.is_available()...")
    is_available = mx.distributed.is_available()
    print(f"  mx.distributed.is_available() = {is_available}")
    
    if not is_available:
        print("WARNING: mx.distributed.is_available() returned False")
        return False
    
    # Try to initialize with timeout
    print(f"\nAttempting mx.distributed.init(backend='ring', strict=False)...")
    print(f"  (This may take a few seconds if other nodes are also initializing)")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        group = mx.distributed.init(backend="ring", strict=False)
        signal.alarm(0)  # Cancel timeout
        
        actual_rank = group.rank()
        actual_size = group.size()
        
        print(f"\n✓ MLX distributed.init succeeded!")
        print(f"  Group rank: {actual_rank} (expected: {rank})")
        print(f"  Group size: {actual_size} (expected: {world_size})")
        
        if actual_size == 1:
            print(f"\n⚠ WARNING: Got singleton group (size=1)")
            print(f"  This means MLX fell back to non-distributed mode")
            print(f"  Ring backend is NOT working")
            return False
        
        if actual_rank != rank:
            print(f"\n⚠ WARNING: Rank mismatch!")
            print(f"  Expected rank {rank} but got {actual_rank}")
            return False
        
        if actual_size != world_size:
            print(f"\n⚠ WARNING: Size mismatch!")
            print(f"  Expected size {world_size} but got {actual_size}")
            return False
        
        print(f"\n✓✓✓ SUCCESS: MLX Ring backend initialized correctly!")
        print(f"  All ranks and sizes match expectations")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print(f"\n✗ MLX distributed.init timed out after {timeout_seconds} seconds")
        print(f"  This usually means:")
        print(f"    - Other nodes are not initializing simultaneously")
        print(f"    - Network connectivity issues")
        print(f"    - Ports are not accessible")
        return False
    except RuntimeError as e:
        signal.alarm(0)
        print(f"\n✗ MLX distributed.init failed!")
        print(f"  Error: {e}")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"\n✗ Unexpected error during MLX distributed.init:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MLX Ring backend initialization")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this node (0, 1, 2, ...)")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of nodes")
    parser.add_argument("--hosts", type=str, required=True, help="Comma-separated list of host:port addresses")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    # Parse hosts
    hosts = [h.strip() for h in args.hosts.split(",")]
    
    if len(hosts) != args.world_size:
        print(f"ERROR: Number of hosts ({len(hosts)}) doesn't match world_size ({args.world_size})")
        sys.exit(1)
    
    success = test_mlx_ring_init(
        rank=args.rank,
        world_size=args.world_size,
        hosts=hosts,
        timeout_seconds=args.timeout,
    )
    
    if success:
        print(f"\n{'='*60}")
        print(f"✓ TEST PASSED: MLX Ring backend works on this node")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED: MLX Ring backend does not work on this node")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()


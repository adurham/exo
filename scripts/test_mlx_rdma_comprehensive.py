#!/usr/bin/env python3
"""Comprehensive MLX RDMA test that tries multiple approaches to get RDMA working.

This script systematically tests different configurations to find what works.
"""

import json
import os
import signal
import subprocess
import sys
import time
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
    except Exception:
        return []


def check_rdma_port_status(device: str) -> dict:
    """Check RDMA port status for a device."""
    try:
        result = subprocess.run(
            ["ibv_devinfo", "-d", device],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"status": "unknown", "error": f"ibv_devinfo failed: {result.returncode}"}
        
        status = {"device": device, "ports": []}
        current_port = None
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("port:"):
                if current_port:
                    status["ports"].append(current_port)
                current_port = {"number": line.split(":")[1].strip()}
            elif current_port and "state:" in line:
                current_port["state"] = line.split(":")[1].strip()
            elif current_port and "link_layer:" in line:
                current_port["link_layer"] = line.split(":")[1].strip()
        
        if current_port:
            status["ports"].append(current_port)
        return status
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_devices_matrix_for_node(
    rank: int,
    world_size: int,
    node_devices: dict[int, list[str]],
) -> list[list[str | None]]:
    """Create devices matrix using actual devices from each node.
    
    Args:
        rank: Current node's rank
        world_size: Total number of nodes
        node_devices: Dict mapping rank -> list of available devices on that node
    """
    matrix: list[list[str | None]] = [
        [None for _ in range(world_size)] for _ in range(world_size)
    ]
    
    # For each row (source node), use its first available device for all connections
    for i in range(world_size):
        if i in node_devices and node_devices[i]:
            device = node_devices[i][0]  # Use first available device
            for j in range(world_size):
                if i != j:
                    matrix[i][j] = device
    
    return matrix


def test_mlx_rdma(
    rank: int,
    world_size: int,
    coordinator_ip: str,
    coordinator_port: int,
    devices_file: Path,
    timeout_seconds: int = 30,
) -> bool:
    """Test MLX RDMA initialization with comprehensive diagnostics."""
    
    print(f"\n{'='*70}")
    print(f"Comprehensive MLX RDMA Test - Rank {rank}")
    print(f"{'='*70}")
    print(f"World Size: {world_size}")
    print(f"Coordinator: {coordinator_ip}:{coordinator_port}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Step 1: Check RDMA devices
    print(f"\n[Step 1] Checking RDMA devices...")
    available_devices = get_available_rdma_devices()
    print(f"  Available devices: {available_devices}")
    
    if not available_devices:
        print("  ✗ No RDMA devices found!")
        return False
    
    # Step 2: Check port status
    print(f"\n[Step 2] Checking RDMA port status...")
    active_ports = []
    for device in available_devices:
        status = check_rdma_port_status(device)
        print(f"  {device}: {status}")
        if "ports" in status:
            for port in status["ports"]:
                state = port.get("state", "").upper()
                if "PORT_ACTIVE" in state:
                    active_ports.append(device)
                    break
    
    if not active_ports:
        print("  ✗ No active RDMA ports found!")
        return False
    
    print(f"  ✓ Active ports: {active_ports}")
    
    # Step 3: Create devices file
    print(f"\n[Step 3] Creating devices matrix...")
    # CRITICAL: Only use ACTIVE ports for the matrix
    # MLX will fail if we try to use a device with PORT_DOWN
    if not active_ports:
        print("  ✗ No active ports available - cannot create matrix!")
        return False
    
    # CRITICAL: MLX needs the FULL matrix with ALL rows filled in
    # Each node must know what devices ALL other nodes are using
    # For testing, we need to know what devices each node uses
    # Hardcode known devices for testing (from actual system configuration)
    matrix = [[None for _ in range(world_size)] for _ in range(world_size)]
    
    # Known device mappings based on actual network topology:
    # Rank 0 (macstudio-m4): 
    #   - en2 (192.168.201.1) connects to rank 1 (MacBook M4 Max)
    #   - en3 (192.168.202.1) connects to rank 2 (MacBook M4 Pro)
    # Rank 1 (macbook-m4):
    #   - en1 (192.168.201.2) connects to rank 0 (Studio)
    #   - en2 (192.168.203.1) connects to rank 2 (MacBook M4 Pro)
    # Rank 2 (work-macbook-m4):
    #   - en1 (192.168.202.2) connects to rank 0 (Studio)
    #   - en2 (192.168.203.2) connects to rank 1 (MacBook M4 Max)
    device_matrix = {
        (0, 1): "rdma_en2",  # Studio en2 -> MacBook M4 Max
        (0, 2): "rdma_en3",  # Studio en3 -> MacBook M4 Pro
        (1, 0): "rdma_en1",  # MacBook M4 Max en1 -> Studio
        (1, 2): "rdma_en2",  # MacBook M4 Max en2 -> MacBook M4 Pro
        (2, 0): "rdma_en1",  # MacBook M4 Pro en1 -> Studio
        (2, 1): "rdma_en2",  # MacBook M4 Pro en2 -> MacBook M4 Max
    }
    
    # Fill in matrix: matrix[i][j] = device on node i that connects to node j
    for i in range(world_size):
        for j in range(world_size):
            if i != j:
                # Use the known device mapping, or fall back to first active port
                device = device_matrix.get((i, j), active_ports[0] if active_ports else None)
                matrix[i][j] = device
    
    print(f"  Matrix for rank {rank}:")
    for i, row in enumerate(matrix):
        print(f"    Row {i}: {row}")
    
    with open(devices_file, "w") as f:
        json.dump(matrix, f)
    
    print(f"  ✓ Devices file created: {devices_file}")
    
    # Step 4: Set environment variables
    print(f"\n[Step 4] Setting environment variables...")
    
    # Clear any existing MLX vars
    for key in list(os.environ.keys()):
        if key.startswith("MLX_"):
            del os.environ[key]
    
    # Set RDMA-specific vars
    devices_file_str = str(devices_file.relative_to(Path.cwd())) if devices_file.is_relative_to(Path.cwd()) else str(devices_file)
    os.environ["MLX_IBV_DEVICES"] = devices_file_str
    os.environ["MLX_RANK"] = str(rank)
    # CRITICAL: MLX uses MLX_JACCL_COORDINATOR, not MLX_IBV_COORDINATOR
    os.environ["MLX_JACCL_COORDINATOR"] = f"{coordinator_ip}:{coordinator_port}"
    os.environ["MLX_WORLD_SIZE"] = str(world_size)
    
    # Ensure MLX_HOSTFILE is NOT set
    if "MLX_HOSTFILE" in os.environ:
        del os.environ["MLX_HOSTFILE"]
    
    print(f"  MLX_IBV_DEVICES={os.environ.get('MLX_IBV_DEVICES')}")
    print(f"  MLX_RANK={os.environ.get('MLX_RANK')}")
    print(f"  MLX_JACCL_COORDINATOR={os.environ.get('MLX_JACCL_COORDINATOR')}")
    print(f"  MLX_WORLD_SIZE={os.environ.get('MLX_WORLD_SIZE')}")
    print(f"  MLX_HOSTFILE={os.environ.get('MLX_HOSTFILE', 'NOT SET')}")
    
    # Step 5: Check MLX availability
    print(f"\n[Step 5] Checking MLX distributed availability...")
    is_available = mx.distributed.is_available()
    print(f"  mx.distributed.is_available() = {is_available}")
    
    if not is_available:
        print("  ✗ MLX doesn't think distributed is available!")
        return False
    
    # Step 6: Try initialization with timeout
    print(f"\n[Step 6] Attempting MLX distributed initialization...")
    print(f"  This will wait up to {timeout_seconds} seconds for other nodes...")
    print(f"  All {world_size} nodes must initialize simultaneously!")
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Initialization timed out after {timeout_seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Try with strict=False first to see what happens
        print(f"  Trying backend='any', strict=False...")
        group = mx.distributed.init(backend="any", strict=False)
        signal.alarm(0)
        
        actual_rank = group.rank()
        actual_size = group.size()
        
        print(f"\n  Result:")
        print(f"    Group rank: {actual_rank} (expected: {rank})")
        print(f"    Group size: {actual_size} (expected: {world_size})")
        
        if actual_size == 1:
            print(f"\n  ✗ Got singleton group (size=1)")
            print(f"    MLX fell back to non-distributed mode")
            print(f"    RDMA is NOT being used")
            
            # Try with strict=True to see the actual error
            print(f"\n  Trying backend='any', strict=True to see error...")
            try:
                group2 = mx.distributed.init(backend="any", strict=True)
                print(f"    Unexpected: strict=True succeeded! size={group2.size()}")
            except RuntimeError as e:
                print(f"    Error with strict=True: {e}")
            
            return False
        
        if actual_rank != rank:
            print(f"\n  ⚠ Rank mismatch: expected {rank}, got {actual_rank}")
            return False
        
        if actual_size != world_size:
            print(f"\n  ⚠ Size mismatch: expected {world_size}, got {actual_size}")
            return False
        
        print(f"\n  ✓✓✓ SUCCESS! MLX RDMA initialized correctly!")
        print(f"    All ranks and sizes match expectations")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print(f"\n  ✗ Initialization timed out")
        print(f"    This usually means other nodes aren't initializing")
        return False
    except RuntimeError as e:
        signal.alarm(0)
        print(f"\n  ✗ MLX distributed.init failed!")
        print(f"    Error: {e}")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"\n  ✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive MLX RDMA test")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this node (0, 1, 2, ...)")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of nodes")
    parser.add_argument("--coordinator-ip", type=str, required=True, help="IP address of rank 0 coordinator")
    parser.add_argument("--coordinator-port", type=int, default=52414, help="Port of rank 0 coordinator")
    parser.add_argument("--devices-file", type=str, default="./test_rdma_{rank}.json", help="Path to devices file")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    devices_file = Path(args.devices_file.format(rank=args.rank))
    
    success = test_mlx_rdma(
        rank=args.rank,
        world_size=args.world_size,
        coordinator_ip=args.coordinator_ip,
        coordinator_port=args.coordinator_port,
        devices_file=devices_file,
        timeout_seconds=args.timeout,
    )
    
    if success:
        print(f"\n{'='*70}")
        print(f"✓ TEST PASSED: MLX RDMA works on rank {args.rank}")
        print(f"{'='*70}")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print(f"✗ TEST FAILED: MLX RDMA does not work on rank {args.rank}")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    main()


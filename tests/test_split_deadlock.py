#!/usr/bin/env python3
"""
Test that group.split() with a singleton sub-group does not deadlock.

Reproduces the bug where a singleton node (new_size=1) returned early from
split() before participating in the port all-gather, causing the remaining
nodes to block forever.

This spawns 3 local processes connected via TCP ring, then each calls
group.split() with colors [0, 0, 1] to form a 2-node sub-group + 1 singleton.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

# Timeout for the entire test (seconds)
TEST_TIMEOUT = 30

# The worker script that each process runs
WORKER_SCRIPT = r'''
import os
import sys
import mlx.core as mx

def main():
    rank = int(os.environ.get("MLX_RANK", "-1"))
    print(f"[rank {rank}] Initializing ring...", flush=True)
    
    world = mx.distributed.init(strict=True, backend="ring")
    print(f"[rank {rank}] Ring initialized: rank={world.rank()}, size={world.size()}", flush=True)
    
    # Verify basic communication works
    x = mx.array([float(world.rank())])
    y = mx.distributed.all_sum(x, group=world)
    mx.eval(y)
    expected = sum(range(world.size()))
    assert y.item() == expected, f"all_sum failed: got {y.item()}, expected {expected}"
    print(f"[rank {rank}] all_sum OK: {y.item()}", flush=True)
    
    # Now test split() with colors that create a singleton
    # Ranks 0,1 get color=0 (2-node sub-group)
    # Rank 2 gets color=1 (singleton)
    color = 0 if world.rank() < 2 else 1
    print(f"[rank {rank}] Calling split(color={color})...", flush=True)
    
    sub = world.split(color)
    
    print(f"[rank {rank}] split() returned! sub.rank={sub.rank()}, sub.size={sub.size()}", flush=True)
    
    # For the 2-node sub-group, verify communication on the sub-ring
    if color == 0:
        x = mx.array([float(sub.rank()) + 10.0])
        y = mx.distributed.all_sum(x, group=sub)
        mx.eval(y)
        expected = 10.0 + 11.0  # rank 0 sends 10, rank 1 sends 11
        assert y.item() == expected, f"sub all_sum failed: got {y.item()}, expected {expected}"
        print(f"[rank {rank}] Sub-ring all_sum OK: {y.item()}", flush=True)
    else:
        print(f"[rank {rank}] Singleton sub-group, no sub-ring communication needed", flush=True)
    
    # Test that the parent ring still works after split()
    # (This may or may not work depending on implementation)
    print(f"[rank {rank}] SUCCESS - split() completed without deadlock", flush=True)

if __name__ == "__main__":
    main()
'''


def find_free_ports(n: int) -> list[int]:
    """Find n free TCP ports."""
    import socket
    ports = []
    socks = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


def main():
    n_nodes = 3
    
    # Find free ports for the ring
    ports = find_free_ports(n_nodes)
    
    # Build MLX_HOSTFILE: [[addr1], [addr2], [addr3]]
    hostfile = [[f"127.0.0.1:{p}"] for p in ports]
    hostfile_json = json.dumps(hostfile)
    
    print(f"Ring topology: {hostfile_json}")
    print(f"Ports: {ports}")
    print(f"Timeout: {TEST_TIMEOUT}s")
    print()
    
    # Write the worker script to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(WORKER_SCRIPT)
        worker_path = f.name
    
    # Write hostfile JSON to a temp file (C++ reads MLX_HOSTFILE as a file path)
    hostfile_fd, hostfile_path = tempfile.mkstemp(suffix='.json', prefix='mlx_hostfile_')
    with os.fdopen(hostfile_fd, 'w') as hf:
        hf.write(hostfile_json)
    
    try:
        # Determine python executable
        python = sys.executable
        
        # Spawn 3 processes
        procs = []
        for rank in range(n_nodes):
            env = os.environ.copy()
            env["MLX_HOSTFILE"] = hostfile_path
            env["MLX_RANK"] = str(rank)
            env["MLX_RING_VERBOSE"] = "1"
            
            proc = subprocess.Popen(
                [python, worker_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            procs.append(proc)
            print(f"Spawned rank {rank} (pid={proc.pid})")
        
        print()
        
        # Wait for all processes with timeout
        start = time.monotonic()
        all_done = False
        outputs = [""] * n_nodes
        
        while time.monotonic() - start < TEST_TIMEOUT:
            all_done = True
            for i, proc in enumerate(procs):
                if proc.poll() is None:
                    all_done = False
                else:
                    # Read remaining output
                    if outputs[i] == "":
                        out = proc.stdout.read()
                        if out:
                            outputs[i] = out
            
            if all_done:
                break
            time.sleep(0.5)
        
        # Print outputs
        for i in range(n_nodes):
            if outputs[i] == "" and procs[i].stdout:
                outputs[i] = procs[i].stdout.read() or ""
            print(f"=== Rank {i} output ===")
            print(outputs[i])
        
        if not all_done:
            print(f"\n❌ DEADLOCK DETECTED — processes did not complete within {TEST_TIMEOUT}s")
            print("Killing remaining processes...")
            for i, proc in enumerate(procs):
                if proc.poll() is None:
                    print(f"  Killing rank {i} (pid={proc.pid})")
                    proc.kill()
                    proc.wait()
            sys.exit(1)
        
        # Check exit codes
        failed = False
        for i, proc in enumerate(procs):
            if proc.returncode != 0:
                print(f"❌ Rank {i} exited with code {proc.returncode}")
                failed = True
        
        if failed:
            print("\n❌ TEST FAILED — some processes exited with non-zero status")
            sys.exit(1)
        
        # Check for SUCCESS messages
        all_success = all("SUCCESS" in outputs[i] for i in range(n_nodes))
        if all_success:
            print("\n✅ TEST PASSED — split() with singleton completed without deadlock")
        else:
            print("\n❌ TEST FAILED — not all processes reported SUCCESS")
            sys.exit(1)
    
    finally:
        os.unlink(worker_path)
        os.unlink(hostfile_path)
        # Clean up any remaining processes
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()

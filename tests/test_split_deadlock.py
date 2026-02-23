#!/usr/bin/env python3
"""
Test that group.split() with a singleton sub-group does not deadlock.

Reproduces two bugs:
1. Singleton node returning early from split() before participating in the
   port all-gather, causing the remaining nodes to block forever.
2. Sub-ring constructor rebinding the same port after split() closes the
   listener, causing SO_REUSEPORT to route connections to the wrong socket.

This spawns 3 local processes connected via TCP ring using DIFFERENT loopback
IPs (127.0.0.1/2/3) to simulate cross-machine connections, mirroring how the
real cluster uses different Thunderbolt IPs per node with 0.0.0.0 for self.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

# Timeout for the entire test (seconds)
TEST_TIMEOUT = 30

# Maximum time (seconds) for sub-ring creation — catches port-rebind hangs
SUB_RING_TIMEOUT = 10

# IP used for peer addresses in the hostfile. Each rank sees 0.0.0.0 for self
# and this IP for all other peers — mirroring the real cluster pattern where
# each node sees its own address as 0.0.0.0 and peers via Thunderbolt IPs.
PEER_IP = "127.0.0.1"

# The worker script that each process runs.
# NOTE: The hostfile uses 0.0.0.0 for self and the peer's actual IP for others,
# exactly like the real cluster does with Thunderbolt IPs.
WORKER_SCRIPT = r'''
import os
import sys
import time
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
    
    t0 = time.monotonic()
    sub = world.split(color)
    split_time = time.monotonic() - t0
    
    print(f"[rank {rank}] split() returned in {split_time:.2f}s! sub.rank={sub.rank()}, sub.size={sub.size()}", flush=True)
    
    # Timing assertion: sub-ring creation should be fast (< SUB_RING_TIMEOUT)
    sub_ring_timeout = float(os.environ.get("SUB_RING_TIMEOUT", "10"))
    if split_time > sub_ring_timeout:
        print(f"[rank {rank}] TIMING FAIL - split() took {split_time:.2f}s (max {sub_ring_timeout}s)", flush=True)
        sys.exit(2)
    
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
    
    print(f"[rank {rank}] SUCCESS - split() completed without deadlock", flush=True)

if __name__ == "__main__":
    main()
'''


def find_free_ports(n: int) -> list[int]:
    """Find n free TCP ports by binding to 0.0.0.0."""
    import socket
    ports: list[int] = []
    socks: list[socket.socket] = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


def main():
    n_nodes = 3
    
    # Find free ports, each bound to a different loopback IP
    ports = find_free_ports(n_nodes)
    
    # Build MLX_HOSTFILE for each rank, using 0.0.0.0 for self (like real cluster).
    # Real cluster hostfile per rank: [self=0.0.0.0:PORT, peer1=IP:PORT, peer2=IP:PORT]
    hostfiles: list[str] = []
    for rank in range(n_nodes):
        entries = []
        for i in range(n_nodes):
            if i == rank:
                entries.append([f"0.0.0.0:{ports[i]}"])
            else:
                entries.append([f"{PEER_IP}:{ports[i]}"])
        hostfiles.append(json.dumps(entries))
    
    print(f"Peer IP: {PEER_IP}")
    print(f"Ports: {ports}")
    for r in range(n_nodes):
        print(f"  Rank {r} hostfile: {hostfiles[r]}")
    print(f"Timeout: {TEST_TIMEOUT}s, Sub-ring max: {SUB_RING_TIMEOUT}s")
    print()
    
    # Write the worker script to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(WORKER_SCRIPT)
        worker_path = f.name
    
    # Write per-rank hostfile JSON to temp files
    hostfile_paths: list[str] = []
    for rank in range(n_nodes):
        fd, path = tempfile.mkstemp(suffix='.json', prefix=f'mlx_hostfile_r{rank}_')
        with os.fdopen(fd, 'w') as hf:
            hf.write(hostfiles[rank])
        hostfile_paths.append(path)
    
    procs: list[subprocess.Popen[str]] = []
    try:
        python = sys.executable
        
        for rank in range(n_nodes):
            env = os.environ.copy()
            env["MLX_HOSTFILE"] = hostfile_paths[rank]
            env["MLX_RANK"] = str(rank)
            env["MLX_RING_VERBOSE"] = "1"
            env["SUB_RING_TIMEOUT"] = str(SUB_RING_TIMEOUT)
            
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
        outputs: list[str] = [""] * n_nodes
        
        while time.monotonic() - start < TEST_TIMEOUT:
            all_done = True
            for i, proc in enumerate(procs):
                if proc.poll() is None:
                    all_done = False
                else:
                    if outputs[i] == "" and proc.stdout:
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
            rc = proc.returncode
            if rc == 2:
                print(f"❌ Rank {i} TIMING FAIL — sub-ring creation too slow")
                failed = True
            elif rc != 0:
                print(f"❌ Rank {i} exited with code {rc}")
                failed = True
        
        if failed:
            print("\n❌ TEST FAILED — some processes exited with non-zero status")
            sys.exit(1)
        
        all_success = all("SUCCESS" in outputs[i] for i in range(n_nodes))
        if all_success:
            print("\n✅ TEST PASSED — split() with singleton completed without deadlock")
        else:
            print("\n❌ TEST FAILED — not all processes reported SUCCESS")
            sys.exit(1)
    
    finally:
        os.unlink(worker_path)
        for path in hostfile_paths:
            os.unlink(path)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()

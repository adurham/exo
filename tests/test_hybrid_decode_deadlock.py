#!/usr/bin/env python3
"""
Test that hybrid TP+PP decode token sync doesn't deadlock.

Reproduces the bug where mx.eval(sampled, *_pending) evaluates the parent-ring
all_sum before the pipeline sends finish, deadlocking because the PP-tail node
can't participate in the all_sum until it receives pipeline data.

Pattern under test (3-node hybrid TP+PP decode step):
  Rank 0 (TP master): sub-ring all_sum + pipeline send to rank 2 + parent ring all_sum
  Rank 1 (TP peer):   sub-ring all_sum + parent ring all_sum
  Rank 2 (PP tail):   pipeline recv from rank 0 + parent ring all_sum

The fix: evaluate model forward + pipeline sends FIRST, then do token sync.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

TEST_TIMEOUT = 30
PEER_IP = "127.0.0.1"

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
    
    # Split: ranks 0,1 = TP group (color 0), rank 2 = singleton (color 1)
    color = 0 if world.rank() < 2 else 1
    sub = world.split(color)
    print(f"[rank {rank}] split() done: sub.rank={sub.rank()}, sub.size={sub.size()}", flush=True)
    
    is_tp_node = (color == 0)
    is_pp_tail = (world.rank() == 2)
    
    # Simulate the hybrid TP+PP decode step:
    # 1. TP nodes do a sub-ring all_sum (simulating TP layer)
    # 2. TP master (rank 0) sends pipeline data to PP tail (rank 2) via parent ring
    # 3. PP tail receives pipeline data via parent ring
    # 4. ALL nodes do token sync all_sum on parent ring
    
    # Step 1: TP sub-ring all_sum (simulating TP layer forward pass)
    if is_tp_node:
        tp_data = mx.array([float(sub.rank()) + 1.0])
        tp_result = mx.distributed.all_sum(tp_data, group=sub)
        print(f"[rank {rank}] TP sub-ring all_sum created (lazy)", flush=True)
    
    # Step 2+3: Pipeline send/recv via parent ring
    # We simulate this with all_gather: rank 0 sends hidden states, rank 2 receives
    # (In real code, this uses custom send/recv ops on the parent ring)
    pipeline_data = mx.array([100.0]) if world.rank() == 0 else mx.array([0.0])
    pipeline_result = mx.distributed.all_sum(pipeline_data, group=world)
    print(f"[rank {rank}] Pipeline all_sum created (lazy)", flush=True)
    
    # Step 4: Token sync all_sum on parent ring
    # PP tail contributes the "correct" token, TP nodes contribute zeros
    if is_pp_tail:
        # PP tail's token depends on pipeline_result (received data)
        token = pipeline_result * 0.0 + 42.0  # "correct" token, depends on pipeline
    else:
        # TP nodes contribute zero — but use zeros_like which has NO data dependency
        # THIS IS THE BUG: zeros_like doesn't create a graph edge to tp_result
        dummy_token = mx.array([0.0])
        token = mx.zeros_like(dummy_token)
    
    token_sync = mx.distributed.all_sum(token, group=world)
    print(f"[rank {rank}] Token sync all_sum created (lazy)", flush=True)
    
    # THE CRITICAL PART: eval ordering
    # Bug pattern: mx.eval(token_sync, pipeline_result, tp_result) — all in one eval
    # Fix pattern: mx.eval(pipeline_result, tp_result) THEN mx.eval(token_sync)
    
    fix_mode = os.environ.get("FIX_DECODE_DEADLOCK", "0") == "1"
    
    t0 = time.monotonic()
    if fix_mode:
        # FIXED: two-phase eval
        print(f"[rank {rank}] Phase 1: eval model + pipeline...", flush=True)
        if is_tp_node:
            mx.eval(tp_result, pipeline_result)
        else:
            mx.eval(pipeline_result)
        print(f"[rank {rank}] Phase 2: eval token sync...", flush=True)
        mx.eval(token_sync)
    else:
        # BUGGY: single eval — can deadlock
        print(f"[rank {rank}] Single eval (model + pipeline + token sync)...", flush=True)
        if is_tp_node:
            mx.eval(token_sync, pipeline_result, tp_result)
        else:
            mx.eval(token_sync, pipeline_result)
    
    elapsed = time.monotonic() - t0
    print(f"[rank {rank}] eval completed in {elapsed:.2f}s, token={token_sync.item()}", flush=True)
    
    if elapsed > 5.0:
        print(f"[rank {rank}] TIMING FAIL - eval took {elapsed:.2f}s", flush=True)
        sys.exit(2)
    
    assert token_sync.item() == 42.0, f"Expected 42.0, got {token_sync.item()}"
    print(f"[rank {rank}] SUCCESS", flush=True)

if __name__ == "__main__":
    main()
'''


def find_free_ports(n: int) -> list[int]:
    import socket
    ports, socks = [], []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


def run_test(fix_mode: bool) -> tuple[bool, list[str]]:
    """Run the 3-node hybrid decode test. Returns (passed, outputs)."""
    n_nodes = 3
    ports = find_free_ports(n_nodes)
    
    hostfiles: list[str] = []
    for rank in range(n_nodes):
        entries = []
        for i in range(n_nodes):
            if i == rank:
                entries.append([f"0.0.0.0:{ports[i]}"])
            else:
                entries.append([f"{PEER_IP}:{ports[i]}"])
        hostfiles.append(json.dumps(entries))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(WORKER_SCRIPT)
        worker_path = f.name
    
    hostfile_paths: list[str] = []
    for rank in range(n_nodes):
        fd, path = tempfile.mkstemp(suffix='.json', prefix=f'mlx_hf_r{rank}_')
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
            env["FIX_DECODE_DEADLOCK"] = "1" if fix_mode else "0"
            
            proc = subprocess.Popen(
                [python, worker_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            procs.append(proc)
        
        start = time.monotonic()
        outputs: list[str] = [""] * n_nodes
        
        while time.monotonic() - start < TEST_TIMEOUT:
            all_done = all(p.poll() is not None for p in procs)
            for i, proc in enumerate(procs):
                if proc.poll() is not None and outputs[i] == "" and proc.stdout:
                    outputs[i] = proc.stdout.read() or ""
            if all_done:
                break
            time.sleep(0.5)
        
        for i in range(n_nodes):
            if outputs[i] == "" and procs[i].stdout:
                outputs[i] = procs[i].stdout.read() or ""
        
        all_done = all(p.poll() is not None for p in procs)
        if not all_done:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            return False, outputs
        
        all_success = all("SUCCESS" in outputs[i] for i in range(n_nodes))
        all_zero = all(procs[i].returncode == 0 for i in range(n_nodes))
        return all_success and all_zero, outputs
        
    finally:
        os.unlink(worker_path)
        for path in hostfile_paths:
            os.unlink(path)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def main():
    # Test 1: The buggy pattern (single eval) — may deadlock on localhost too
    print("=" * 60)
    print("TEST 1: Single eval (buggy pattern)")
    print("=" * 60)
    passed, outputs = run_test(fix_mode=False)
    for i, out in enumerate(outputs):
        print(f"--- Rank {i} ---")
        print(out)
    if passed:
        print("⚠️  Single eval passed (deadlock may only occur cross-machine)")
    else:
        print("✅ Single eval deadlocked as expected")
    print()
    
    # Test 2: The fixed pattern (two-phase eval) — should always pass
    print("=" * 60)
    print("TEST 2: Two-phase eval (fixed pattern)")
    print("=" * 60)
    passed, outputs = run_test(fix_mode=True)
    for i, out in enumerate(outputs):
        print(f"--- Rank {i} ---")
        print(out)
    if passed:
        print("\n✅ Two-phase eval PASSED — decode token sync works")
    else:
        print("\n❌ Two-phase eval FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

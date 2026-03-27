#!/usr/bin/env python3
"""Reproduces the PP pipeline post-loop deadlock.

The pipeline prefill is offset: R0 sends chunk 0 in iter 0, R1 receives in
iter 1. After the main loop, R0 enters the post-loop first and sends.
R1 is still draining the last main-loop chunk. R0's post-loop send lands
in R1's pre-posted recv buffers from the MAIN loop recv_like. When R1
enters the post-loop and creates a NEW recv_like, it pre-posts NEW recv
buffers — but R0's data already went to the OLD buffers. R1 polls the
new buffers and deadlocks.

Run on two Studios:
  Studio 1: MLX_JACCL_NUM_BUFFERS=4 MLX_JACCL_FRAME_SIZE=8192 RANK=0 COORDINATOR=0.0.0.0:59989 .venv/bin/python3 tests/test_pipeline_postloop_deadlock.py
  Studio 2: MLX_JACCL_NUM_BUFFERS=4 MLX_JACCL_FRAME_SIZE=8192 RANK=1 COORDINATOR=<ip>:59989 .venv/bin/python3 tests/test_pipeline_postloop_deadlock.py
"""
import json
import os
import sys
import tempfile
import time

import mlx.core as mx

RANK = int(os.environ.get("RANK", "0"))
COORDINATOR = os.environ.get("COORDINATOR", "0.0.0.0:59989")
OTHER = 1 - RANK
HIDDEN = 5120
DTYPE = mx.bfloat16

def init():
    devices = [[None, "rdma_en5"], ["rdma_en5", None]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(devices, f)
        os.environ["MLX_IBV_DEVICES"] = f.name
    os.environ["MLX_RANK"] = str(RANK)
    os.environ["MLX_JACCL_COORDINATOR"] = COORDINATOR
    os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    group = mx.distributed.init(backend="jaccl", strict=True)
    print(f"[R{RANK}] init OK", flush=True)
    return group


def simulate_pipeline_prefill(group, n_chunks=2):
    """Simulates the offset pipeline prefill pattern from generate.py."""
    rank = RANK
    world_size = 2
    n_leading = rank                    # R0=0, R1=1
    n_trailing = world_size - 1 - rank  # R0=1, R1=0
    n_total = n_leading + n_chunks + n_trailing

    print(f"[R{rank}] pipeline: {n_chunks} real + {n_leading} leading + {n_trailing} trailing = {n_total} iters", flush=True)

    # Main loop (offset pipeline)
    pending_sends: list = []
    real_idx = 0
    for i in range(n_total):
        is_leading = i < n_leading
        is_trailing = i >= n_leading + n_chunks

        if is_leading or is_trailing:
            # Dummy iteration — no model, no send/recv
            print(f"[R{rank}] iter {i}: dummy", flush=True)
            continue

        # Real iteration: GPU compute producing 256×5120 bf16 hidden state (2.5MB)
        # This matches the real model's output size for a 256-token chunk
        x = mx.random.normal((256, HIDDEN)).astype(DTYPE)
        # Simulate 30 layers of GPU compute
        for _layer in range(5):
            x = x @ mx.random.normal((HIDDEN, HIDDEN)).astype(DTYPE)
        mx.eval(x)  # eval GPU output BEFORE send (production pattern)

        if rank == 0:
            # Queue send then flush (production pattern: queue_sends=True)
            pending_sends.append(mx.distributed.send(x, OTHER, group=group))
        else:
            # R1 receives hidden state
            received = mx.distributed.recv_like(x, OTHER, group=group)
            mx.eval(received)
            print(f"[R{rank}] iter {i}: recv chunk {real_idx}", flush=True)

        # Flush queued sends (like flush_prefill_sends)
        for s in pending_sends:
            mx.async_eval(s)

        # Simulate cache eval (like mx.eval(*cache_states) in production)
        cache = mx.random.normal((256, 2048)).astype(DTYPE)
        mx.eval(cache)

        # Drain async sends
        for s in pending_sends:
            mx.eval(s)
        if rank == 0 and pending_sends:
            print(f"[R{rank}] iter {i}: send+drain chunk {real_idx}", flush=True)
        pending_sends.clear()

        real_idx += 1

    print(f"[R{rank}] main loop done", flush=True)

    # Post-loop: 2 iterations of send/recv (like generate.py lines 239-245)
    for post_i in range(2):
        x = mx.random.normal((1, HIDDEN)).astype(DTYPE)
        mx.eval(x)

        if rank == 0:
            sent = mx.distributed.send(x, OTHER, group=group)
            mx.async_eval(sent)
            print(f"[R{rank}] post-loop {post_i}: async_send", flush=True)
        else:
            received = mx.distributed.recv_like(x, OTHER, group=group)
            print(f"[R{rank}] post-loop {post_i}: recv_like created (pre-posted)", flush=True)
            mx.eval(received)
            print(f"[R{rank}] post-loop {post_i}: recv done", flush=True)

        if rank == 0:
            mx.eval(sent)
            print(f"[R{rank}] post-loop {post_i}: drain done", flush=True)

    print(f"[R{rank}] post-loop done", flush=True)

    # Barrier (like the post-prefill barrier)
    token = mx.array([rank], dtype=mx.int32)
    mx.eval(token)
    g = mx.distributed.all_gather(token.reshape(1), group=group)
    mx.eval(g)
    print(f"[R{rank}] barrier done", flush=True)


def main():
    group = init()

    # First "request" — should work
    print(f"\n[R{RANK}] === REQUEST 1 ===", flush=True)
    simulate_pipeline_prefill(group, n_chunks=4)

    # Simulate decode: 2000 tokens with GPU compute + all_gather per token
    # This matches the real decode path: model forward (GPU) → all_gather token
    print(f"\n[R{RANK}] === DECODE (2000 tokens) ===", flush=True)
    for i in range(2000):
        # GPU compute: model forward (single token through layers)
        x = mx.random.normal((1, HIDDEN)).astype(DTYPE)
        for _layer in range(3):  # simulate a few GPU layers
            x = x @ mx.random.normal((HIDDEN, HIDDEN)).astype(DTYPE)
        mx.eval(x)

        # Hidden state exchange via all_gather (like _pp_step)
        gathered = mx.distributed.all_gather(x.reshape(1, -1), group=group)
        mx.eval(gathered)

        # Token sync via all_gather
        token = mx.array([i % 100], dtype=mx.int32)
        mx.eval(token)
        g = mx.distributed.all_gather(token.reshape(1), group=group)
        mx.eval(g)

        if i % 500 == 0:
            print(f"[R{RANK}] decode {i}/2000", flush=True)
    print(f"[R{RANK}] decode done", flush=True)

    # Second "request" — this is where the cluster deadlocks
    print(f"\n[R{RANK}] === REQUEST 2 ===", flush=True)
    simulate_pipeline_prefill(group, n_chunks=2)

    print(f"\n[R{RANK}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

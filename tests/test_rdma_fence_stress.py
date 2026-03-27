#!/usr/bin/env python3
"""
RDMA fence stress test — exercises the drain_fences() code path under
realistic fence pressure (hundreds of GPU ops + RDMA).

Run on two Studios connected via Thunderbolt 5 RDMA:
  Studio 1:  RANK=0 COORDINATOR=0.0.0.0:59989 .venv/bin/python tests/test_rdma_fence_stress.py
  Studio 2:  RANK=1 COORDINATOR=<studio1-ip>:59989 .venv/bin/python tests/test_rdma_fence_stress.py

Tests:
  1. Separated eval (production path): GPU eval -> send eval -> all_gather eval
  2. Combined eval (deadlock path): GPU + send + all_gather in one eval
  3. High fence pressure: 200 matmuls + send/recv in one eval
  4. Rapid fire: 50 iterations of compute+send to stress fence queue
  5. Prefill simulation: large tensor send with concurrent GPU work (async)
"""

import json
import os
import sys
import tempfile
import time
import traceback

import mlx.core as mx

RANK = int(os.environ.get("RANK", os.environ.get("MLX_RANK", "0")))
COORDINATOR = os.environ.get("COORDINATOR", "0.0.0.0:59989")
OTHER_RANK = 1 - RANK
TIMEOUT = 30  # seconds per test

# These must be set in the SHELL before python starts (inline const read at import).
# Setting them in Python is too late.
if os.environ.get("MLX_JACCL_NUM_BUFFERS") is None:
    print("FATAL: MLX_JACCL_NUM_BUFFERS not set in shell environment.", flush=True)
    print("Usage: MLX_JACCL_NUM_BUFFERS=4 MLX_JACCL_FRAME_SIZE=8192 RANK=0 COORDINATOR=... python3 tests/test_rdma_fence_stress.py", flush=True)
    sys.exit(1)


def init_group() -> mx.distributed.Group:
    # JACCL requires: MLX_IBV_DEVICES (JSON connectivity), MLX_RANK, MLX_JACCL_COORDINATOR
    # NOTE: MLX_JACCL_NUM_BUFFERS and MLX_JACCL_FRAME_SIZE are inline const globals
    # read at library load (import time). They MUST be set in the shell environment
    # BEFORE python starts, not here. See run_rdma_fence_stress.sh.
    devices = [[None, "rdma_en5"], ["rdma_en5", None]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(devices, f)
        devfile = f.name
    os.environ["MLX_IBV_DEVICES"] = devfile
    os.environ["MLX_RANK"] = str(RANK)
    os.environ["MLX_JACCL_COORDINATOR"] = COORDINATOR
    os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    group = mx.distributed.init(backend="jaccl", strict=True)
    assert group.rank() == RANK
    assert group.size() == 2
    print(f"[R{RANK}] JACCL initialized: rank={group.rank()}, size={group.size()}", flush=True)
    return group


def timed(name: str, fn, timeout: float = TIMEOUT):
    """Run fn directly (no threads — daemon threads corrupt RDMA state).
    Uses SIGALRM for deadlock detection."""
    print(f"[R{RANK}] {name} ... ", end="", flush=True)
    t0 = time.perf_counter()
    import signal
    class TimeoutError(Exception):
        pass
    def handler(signum, frame):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout))
    try:
        fn()
        signal.alarm(0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"OK ({elapsed_ms:.1f}ms)", flush=True)
        return True, elapsed_ms
    except TimeoutError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"TIMEOUT ({timeout}s) - DEADLOCK", flush=True)
        return False, elapsed_ms
    except Exception as e:
        signal.alarm(0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"ERROR: {e} ({elapsed_ms:.1f}ms)", flush=True)
        traceback.print_exception(type(e), e, e.__traceback__)
        return False, elapsed_ms
    finally:
        signal.signal(signal.SIGALRM, old)


def make_gpu_chain(x: mx.array, n_ops: int) -> mx.array:
    """Chain of matmuls to generate many fence tasks."""
    for _ in range(n_ops):
        x = x @ mx.eye(x.shape[-1])
    return x


def test_separated_eval(group: mx.distributed.Group) -> None:
    """Production path: eval GPU, then eval send, then eval all_gather.
    This is what auto_parallel.py does — should always work."""
    x = mx.random.normal((1, 128))
    # Phase 1: GPU compute
    y = make_gpu_chain(x, 50)
    mx.eval(y)
    # Phase 2: send/recv
    if RANK == 0:
        sent = mx.distributed.send(y, OTHER_RANK, group=group)
        mx.eval(sent)
    else:
        received = mx.distributed.recv(y.shape, y.dtype, OTHER_RANK, group=group)
        mx.eval(received)
        y = received
    # Phase 3: all_gather
    gathered = mx.distributed.all_gather(y, group=group)
    mx.eval(gathered)
    assert gathered.shape[0] == 2, f"Expected shape[0]=2, got {gathered.shape}"


def test_combined_eval(group: mx.distributed.Group) -> None:
    """Deadlock-prone path: GPU + send in one eval graph.
    This is the scenario drain_fences() must handle."""
    x = mx.random.normal((1, 128))
    y = make_gpu_chain(x, 50)
    # Don't eval y separately — build send/recv into the same graph
    if RANK == 0:
        sent = mx.distributed.send(y, OTHER_RANK, group=group)
        mx.eval(sent)
    else:
        received = mx.distributed.recv((1, 128), mx.float32, OTHER_RANK, group=group)
        mx.eval(received)


def test_high_fence_pressure(group: mx.distributed.Group) -> None:
    """200 matmuls + send/recv in one eval — mimics real model fence count.
    The old drain_fences() crashed with bus errors here."""
    x = mx.random.normal((1, 256))
    y = make_gpu_chain(x, 200)  # 200 ops = ~200 fence pairs
    if RANK == 0:
        sent = mx.distributed.send(y, OTHER_RANK, group=group)
        mx.eval(sent)
    else:
        received = mx.distributed.recv((1, 256), mx.float32, OTHER_RANK, group=group)
        mx.eval(received)
    # Verify data
    gathered = mx.distributed.all_gather(
        y if RANK == 0 else received, group=group
    )
    mx.eval(gathered)
    assert gathered.shape == (2, 256), f"Unexpected shape: {gathered.shape}"


def test_rapid_fire(group: mx.distributed.Group) -> None:
    """50 iterations of compute+send — stresses fence queue throughput."""
    for i in range(50):
        x = mx.random.normal((1, 64))
        y = make_gpu_chain(x, 10)
        if RANK == 0:
            sent = mx.distributed.send(y, OTHER_RANK, group=group)
            mx.eval(sent)
        else:
            received = mx.distributed.recv((1, 64), mx.float32, OTHER_RANK, group=group)
            mx.eval(received)


def test_prefill_simulation(group: mx.distributed.Group) -> None:
    """Simulates the prefill deadlock: async_eval(send) + eval(cache).
    This is the exact race condition from flush_prefill_sends()."""
    for chunk in range(5):
        # Simulate model forward (GPU compute)
        x = mx.random.normal((512, 256))  # larger tensor like prefill
        y = make_gpu_chain(x, 30)
        mx.eval(y)  # eval compute first

        # Simulate flush_prefill_sends: async send
        if RANK == 0:
            sent = mx.distributed.send(y, OTHER_RANK, group=group)
            mx.async_eval(sent)
        else:
            received = mx.distributed.recv(y.shape, y.dtype, OTHER_RANK, group=group)
            mx.async_eval(received)

        # Simulate concurrent cache quantization (generates fence tasks
        # on the same CPU stream while the send poll loop is running)
        cache = mx.random.normal((512, 128))
        quantized = make_gpu_chain(cache, 5)
        mx.eval(quantized)

        # Drain the async send — must complete before next iteration
        if RANK == 0:
            mx.eval(sent)
        else:
            mx.eval(received)

        # Drain the async send
        if RANK == 0:
            mx.eval(sent)
        else:
            mx.eval(received)


def test_all_gather_repeated(group: mx.distributed.Group) -> None:
    """Repeated all_gather — matches the decode token-sync pattern."""
    x = mx.ones((1, 128))
    mx.eval(x)
    for i in range(20):
        gathered = mx.distributed.all_gather(x, group=group)
        mx.eval(gathered)
        assert gathered.shape[0] == 2
        x = gathered[:1]  # use first rank's data for next iteration


def test_production_decode_pattern(group: mx.distributed.Group) -> None:
    """Exact buffer sizes from the PP decode path:
    1. Hidden state all_gather: (1, H) bf16 where H=5120 (Qwen3.5-397B)
    2. K>1 header all_gather: (1, K+1) int32 where K=3 → 16 bytes per rank
    3. Token sync all_gather: (1,) int32 → 4 bytes per rank
    4. K>1 result all_gather: (2,) int32 → 8 bytes per rank
    These are the exact shapes from generate.py _pp_step."""
    for step in range(50):
        # 1. Hidden state exchange — (1, 5120) bf16 = 10240 bytes
        hidden = mx.random.normal((1, 5120)).astype(mx.bfloat16)
        mx.eval(hidden)
        gathered_hidden = mx.distributed.all_gather(
            hidden.reshape(1, -1), group=group
        )
        mx.eval(gathered_hidden)
        assert gathered_hidden.shape == (2, 5120), f"hidden shape: {gathered_hidden.shape}"

        # 2. K>1 header — (1, 4) int32 = 16 bytes
        header = mx.zeros((1, 4), dtype=mx.int32)
        mx.eval(header)
        gathered_header = mx.distributed.all_gather(header, group=group)
        mx.eval(gathered_header)
        assert gathered_header.shape == (2, 4), f"header shape: {gathered_header.shape}"

        # 3. Token sync — (1,) int32 = 4 bytes
        token = mx.array([42], dtype=mx.int32)
        mx.eval(token)
        gathered_token = mx.distributed.all_gather(token.reshape(1), group=group)
        mx.eval(gathered_token)
        assert gathered_token.shape == (2,), f"token shape: {gathered_token.shape}"

        # 4. K>1 result — (2,) int32 = 8 bytes
        result = mx.array([0, 42], dtype=mx.int32)
        mx.eval(result)
        gathered_result = mx.distributed.all_gather(result.reshape(2), group=group)
        mx.eval(gathered_result)


def test_tiny_buffer_all_gather(group: mx.distributed.Group) -> None:
    """Stress test with very small buffers — the sizes most likely to hit
    page boundary issues. 100 iterations each."""
    for nbytes_label, arr in [
        ("4B int32", mx.array([1], dtype=mx.int32)),
        ("8B 2xint32", mx.array([1, 2], dtype=mx.int32)),
        ("16B 4xint32", mx.zeros(4, dtype=mx.int32)),
        ("32B 8xint32", mx.zeros(8, dtype=mx.int32)),
        ("64B 16xfloat32", mx.zeros(16, dtype=mx.float32)),
    ]:
        mx.eval(arr)
        for i in range(100):
            gathered = mx.distributed.all_gather(arr, group=group)
            mx.eval(gathered)


def test_drain_cq(group: mx.distributed.Group) -> None:
    """Drain any stale CQ entries by doing a barrier all_gather."""
    token = mx.array([RANK], dtype=mx.int32)
    mx.eval(token)
    g = mx.distributed.all_gather(token.reshape(1), group=group)
    mx.eval(g)


def test_prefill_pipeline_send_recv(group: mx.distributed.Group) -> None:
    """Simulates the actual prefill pipeline: multiple chunks of send/recv
    with GPU compute between them, matching PipelineFirstLayer/PipelineLastLayer.
    5 chunks of 256 tokens × 5120 hidden dim bf16 = ~2.5MB per chunk."""
    for chunk in range(5):
        # GPU compute (30 layers worth of matmuls)
        x = mx.random.normal((256, 5120)).astype(mx.bfloat16)
        y = make_gpu_chain(x, 10)
        mx.eval(y)  # eval GPU first (production pattern)

        # Send/recv (production pattern: eval output, then eval send separately)
        if RANK == 0:
            sent = mx.distributed.send(y, OTHER_RANK, group=group)
            mx.eval(sent)
        else:
            received = mx.distributed.recv((256, 5120), mx.bfloat16, OTHER_RANK, group=group)
            mx.eval(received)

        # Then all_gather for token sync
        token = mx.array([chunk], dtype=mx.int32)
        mx.eval(token)
        gathered = mx.distributed.all_gather(token.reshape(1), group=group)
        mx.eval(gathered)


def main():
    group = init_group()

    tests = [
        ("1. Separated eval (production path)", lambda: test_separated_eval(group)),
        ("2. Combined eval (deadlock path)", lambda: test_combined_eval(group)),
        ("3. High fence pressure (200 ops)", lambda: test_high_fence_pressure(group)),
        ("4. Rapid fire (50 iterations)", lambda: test_rapid_fire(group)),
        ("5. Prefill simulation (async+sync)", lambda: test_prefill_simulation(group)),
        ("6. Production decode pattern (50 steps)", lambda: test_production_decode_pattern(group)),
        ("7. Tiny buffer all_gather (500x)", lambda: test_tiny_buffer_all_gather(group)),
        ("8. CQ drain barrier", lambda: test_drain_cq(group)),
        ("9. Prefill pipeline 2.6MB send/recv (5 chunks)", lambda: test_prefill_pipeline_send_recv(group)),
    ]

    results = []
    for name, fn in tests:
        ok, ms = timed(name, fn)
        results.append((name, ok, ms))
        if not ok:
            print(f"[R{RANK}] STOPPING — test failed", flush=True)
            break

    print(f"\n[R{RANK}] === RESULTS ===", flush=True)
    all_passed = True
    for name, ok, ms in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({ms:.0f}ms)", flush=True)
        if not ok:
            all_passed = False

    if all_passed:
        print(f"\n[R{RANK}] ALL TESTS PASSED", flush=True)
    else:
        print(f"\n[R{RANK}] SOME TESTS FAILED", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

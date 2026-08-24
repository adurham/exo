#!/usr/bin/env python3
"""prefill_cliff_mechanism_local.py — local M4 Max microbench that empirically
tests the June-2026 ~340K prefill cliff mechanism hypothesis.

Reproduces the era shape: mx.argsort/-partition on (1, L=128, P) bf16, where
P = pooled length. Sweeps P (~40K..120K) at 4K resolution to look for a step
discontinuity, and A/B's MLX_MAX_MB_PER_BUFFER=50 (June default for M4 Max
'g16s') vs =200 (production default since 2026-06-24). Each config runs in a
FRESH child process so the env is picked up at MLX import time.

Two workloads:
  (a) "single_argsort": pure per-op wall time
  (b) "chain_21x":       21 sequential argsorts with a small matmul between
                         them, submitted WITHOUT intermediate eval — this is
                         where command-buffer accounting/flush behavior can
                         cause the "bimodal stall" pattern documented for
                         B=2 in docs/prefill-throughput-breakthrough-2026-06-24.md.

Reads: prints one JSON line per (workload, op, P, MAX_MB) row for post-hoc
analysis. Run: `.venv/bin/python bench/prefill_cliff_mechanism_local.py`
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---- worker child: runs when called with the WORKER env sentinel -------------

WORKER_SENTINEL = "PREFILL_CLIFF_MECH_WORKER"


def _child_main() -> None:
    # env is already set by parent before this python invocation
    import mlx.core as mx  # noqa: WPS433 — deferred import is the point

    workload = os.environ["CLIFF_WORKLOAD"]
    op = os.environ["CLIFF_OP"]  # argsort | argpartition
    P = int(os.environ["CLIFF_P"])
    L = int(os.environ.get("CLIFF_L", "128"))
    K = int(os.environ.get("CLIFF_K", "512"))
    reps = int(os.environ.get("CLIFF_REPS", "8"))
    warmup = int(os.environ.get("CLIFF_WARMUP", "3"))
    chain = int(os.environ.get("CLIFF_CHAIN", "21"))
    max_mb = os.environ.get("MLX_MAX_MB_PER_BUFFER", "<default>")

    # Fresh input each call to avoid cache reuse of a canonical buffer
    def make_scores(seed: int):
        mx.random.seed(seed)
        return mx.random.uniform(shape=(1, L, P), dtype=mx.bfloat16)

    # Small "layer work" matmul between argsorts in chain mode (~64x64 GEMM,
    # trivial compute — its point is to be a *distinct op with a small buffer*
    # so the argsort temporaries are the dominant per-op byte contributor).
    filler_a = mx.random.uniform(shape=(64, 64), dtype=mx.bfloat16)
    filler_b = mx.random.uniform(shape=(64, 64), dtype=mx.bfloat16)
    mx.eval(filler_a, filler_b)

    def do_op(scores):
        if op == "argsort":
            return mx.argsort(-scores, axis=-1)[..., :K]
        elif op == "argpartition":
            return mx.argpartition(-scores, kth=K - 1, axis=-1)[..., :K]
        else:
            raise ValueError(f"unknown op {op!r}")

    def run_single(seed: int) -> float:
        scores = make_scores(seed)
        mx.eval(scores)
        t0 = time.perf_counter()
        out = do_op(scores)
        mx.eval(out)
        return time.perf_counter() - t0

    def run_chain(seed: int) -> float:
        # 21 argsorts submitted WITHOUT intermediate eval, then one eval —
        # mimics the ~21 ratio-4 indexer layers in a single DSv4 forward chunk.
        scores = make_scores(seed)
        mx.eval(scores)
        t0 = time.perf_counter()
        acc = None
        for _ in range(chain):
            out = do_op(scores)
            # A little layer-like work so the chain is not one giant fused
            # graph of identical ops (schedulers may coalesce).
            fill = filler_a @ filler_b
            acc = out if acc is None else out + acc[..., : out.shape[-1]]
            # Cast the tiny filler into the accumulator dependency chain
            # so its bytes count toward the same command buffer.
            acc = acc + mx.sum(fill).astype(mx.uint32)
        mx.eval(acc)
        return time.perf_counter() - t0

    runner = run_single if workload == "single_argsort" else run_chain

    # Warmup
    for w in range(warmup):
        runner(seed=1000 + w)

    # Timed reps
    times: list[float] = []
    peaks: list[int] = []
    for r in range(reps):
        # Reset the peak-memory counter around each rep so we measure this
        # rep's incremental peak, not a monotone envelope.
        if hasattr(mx, "reset_peak_memory"):
            try:
                mx.reset_peak_memory()
            except Exception:
                pass
        t = runner(seed=r)
        times.append(t)
        try:
            peaks.append(int(mx.get_peak_memory()))
        except Exception:
            peaks.append(-1)

    result = {
        "workload": workload,
        "op": op,
        "P": P,
        "L": L,
        "K": K,
        "chain": chain if workload != "single_argsort" else 1,
        "MLX_MAX_MB_PER_BUFFER": max_mb,
        "MLX_MAX_OPS_PER_BUFFER": os.environ.get(
            "MLX_MAX_OPS_PER_BUFFER", "<default>"
        ),
        "reps": reps,
        "warmup": warmup,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.fmean(times),
        "stdev_s": statistics.pstdev(times),
        "all_s": times,
        "peak_bytes": peaks,
    }
    # Emit ONE line of JSON so parent parses trivially.
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


# ---- parent: sweeps ---------------------------------------------------------


def _sweep() -> None:
    here = Path(__file__).resolve()

    # Coarse P sweep, then a refinement pass around any interesting jump.
    P_values = list(range(40_000, 120_001, 8_000))  # coarse: 40..120K, 8K step

    ops = ["argsort", "argpartition"]
    workloads = ["single_argsort", "chain_21x"]
    max_mb_configs = ["50", "200"]  # June default (M4 Max s-arch) vs prod

    reps = int(os.environ.get("REPS", "8"))
    warmup = int(os.environ.get("WARMUP", "3"))
    L = int(os.environ.get("L", "128"))
    K = int(os.environ.get("K", "512"))
    chain = int(os.environ.get("CHAIN", "21"))

    out_path = Path(os.environ.get(
        "OUT_PATH",
        "/Users/adam.durham/repos/exo/bench/prefill_cliff_mechanism_results.jsonl",
    ))

    results: list[dict] = []
    total = len(workloads) * len(ops) * len(P_values) * len(max_mb_configs)
    done = 0
    t_wall = time.perf_counter()

    with out_path.open("w") as f:
        for workload in workloads:
            for op in ops:
                for max_mb in max_mb_configs:
                    for P in P_values:
                        done += 1
                        env = os.environ.copy()
                        env["MLX_MAX_MB_PER_BUFFER"] = max_mb
                        env["MLX_MAX_OPS_PER_BUFFER"] = max_mb  # parity
                        env["CLIFF_WORKLOAD"] = workload
                        env["CLIFF_OP"] = op
                        env["CLIFF_P"] = str(P)
                        env["CLIFF_L"] = str(L)
                        env["CLIFF_K"] = str(K)
                        env["CLIFF_REPS"] = str(reps)
                        env["CLIFF_WARMUP"] = str(warmup)
                        env["CLIFF_CHAIN"] = str(chain)
                        env[WORKER_SENTINEL] = "1"

                        proc = subprocess.run(
                            [sys.executable, str(here)],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=300,
                        )
                        if proc.returncode != 0:
                            row = {
                                "workload": workload, "op": op, "P": P,
                                "MLX_MAX_MB_PER_BUFFER": max_mb,
                                "error": proc.stderr[-500:],
                            }
                        else:
                            # Parse the LAST line (workers may emit warnings).
                            line = proc.stdout.strip().splitlines()[-1]
                            row = json.loads(line)
                        results.append(row)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        elapsed = time.perf_counter() - t_wall
                        print(
                            f"[{done:3d}/{total}] wl={workload:>14s} op={op:>12s} "
                            f"P={P:>6d} MB={max_mb} "
                            f"median={row.get('median_s', float('nan')):.4f}s "
                            f"min={row.get('min_s', float('nan')):.4f}s "
                            f"max={row.get('max_s', float('nan')):.4f}s  "
                            f"(elapsed {elapsed:.1f}s)"
                        )

    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    if os.environ.get(WORKER_SENTINEL) == "1":
        _child_main()
    else:
        _sweep()

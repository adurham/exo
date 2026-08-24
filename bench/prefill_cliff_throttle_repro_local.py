#!/usr/bin/env python3
"""prefill_cliff_throttle_repro_local.py

Attempts to reproduce a bimodal multi-second stall regime driven by the
fork's eval-driver throttle in mlx/transforms.cpp:285-299:

    if (n_active_tasks() > MAX_ACTIVE_TASKS ||
        (get_active_memory() > get_memory_limit() && n_active_tasks() > 0)) {
      gpu::finalize(...); scheduler::wait_for_one();
      while (get_active_memory() > get_memory_limit() && n_active_tasks() > 0)
          scheduler::wait_for_one();
    }

We fake the "active memory near the limit" condition on this local M4 Max by
setting `mx.set_memory_limit(...)` to a value slightly ABOVE current usage,
then holding a ballast tensor so `get_active_memory()` sits at ~95-105% of
the limit. Each chunk builds a lazy async chain (21 x matmul + argsort at
DSv4-era shape) and evals it. If the memory-limit branch is the discrete
trigger, arm B (over-limit) should show massive/bimodal per-chunk stalls
while arm A (under-limit) runs smoothly.

Arms:
  A: active ~50% of limit
  B: active ~105% of limit (crosses branch)
  C: over-limit + MLX_MAX_MB_PER_BUFFER=200 (fewer commits → fewer throttle
     re-entries per chain, even if the memory branch is armed)
  D: over-limit + per-op mx.eval (mimics EXO_PROFILER_SYNC_SPANS=1 —
     n_active_tasks stays near 0-1 → EITHER branch is disabled)

Run as fresh child processes so MLX_MAX_MB_PER_BUFFER and EXO_MAX_ACTIVE_TASKS
are picked up at mlx import time.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "prefill_cliff_throttle_repro_local_results.jsonl"


def child_run(cfg: dict) -> dict:
    """Run one measurement as a fresh child process. Returns parsed dict."""
    env = os.environ.copy()
    for k, v in cfg["env"].items():
        env[k] = str(v)
    # Force child mode so we import mlx here.
    env["REPRO_CHILD"] = "1"
    env["REPRO_CFG"] = json.dumps(cfg)
    cmd = [sys.executable, str(__file__), "--child"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=cfg.get("child_timeout", 300))
    if proc.returncode != 0:
        return {"cfg": cfg, "ok": False,
                "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-2000:]}
    # Last line = JSON result.
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    try:
        result = json.loads(lines[-1])
    except Exception as e:
        return {"cfg": cfg, "ok": False,
                "parse_error": str(e), "stdout": proc.stdout[-2000:]}
    return {"cfg": cfg, "ok": True, "result": result}


def child_main():
    cfg = json.loads(os.environ["REPRO_CFG"])
    import mlx.core as mx

    # Config
    P = int(cfg["P"])                # pooled length (DSv4 era: 85000)
    L = int(cfg["L"])                # prefill chunk seq length (era: 128)
    n_layers = int(cfg["n_layers"])  # per chunk (era: ~21 ratio-4 indexer layers)
    n_chunks = int(cfg["n_chunks"])  # chunks to time
    sync_per_op = bool(cfg.get("sync_per_op", False))
    limit_gb = float(cfg["limit_gb"])          # memory_limit to install
    ballast_gb = float(cfg["ballast_gb"])      # ballast to hold
    head_dim = int(cfg.get("head_dim", 128))   # index_head_dim in DSv4 = 128
    n_heads = int(cfg.get("n_heads", 64))      # index_n_heads = 64

    # Snapshot the default limit before we clobber it.
    default_limit = mx.set_memory_limit(int(limit_gb * (1 << 30)))
    dev_info = mx.device_info()

    # Ballast: single big buffer, keep alive. Size ~ballast_gb.
    ballast = None
    if ballast_gb > 0:
        # Use bf16 to save allocation time; still counts in active_memory bytes.
        n_elems = int(ballast_gb * (1 << 30) / 2)
        # Reshape into a 2D array — mx.zeros with a huge 1-D shape uses 8-byte
        # elems worth of shape metadata otherwise fine.
        side = int(n_elems**0.5)
        # Round side to multiple of 128 for tidy alignment.
        side = (side // 128) * 128
        ballast = mx.zeros((side, side), dtype=mx.bfloat16)
        mx.eval(ballast)

    # Snapshot memory state at start.
    pre_active = mx.get_active_memory()
    pre_cache = mx.get_cache_memory()
    installed_limit = mx.set_memory_limit(int(limit_gb * (1 << 30)))
    # (re-set to make sure it's still the value we want after any allocator dance)

    # Fake "layer compute" op: matmul on a tiny shape to keep tasks flowing
    # without dominating time. Then argsort on the era shape.
    # (B=1, n_heads=64, L=128, head_dim=128) x (B=1, n_heads=64, head_dim=128, P)
    # → score (1, 64, 128, P) → collapse heads → (1, 128, P) → argsort.
    # We approximate: build scores directly, argsort.
    def one_chunk():
        # Build score (1, L, P) bf16 chained across n_layers (so a single
        # mx.eval at the end actually forces ALL layer argsorts). We can't
        # make argsort's output feed the next argsort's score cleanly, so
        # instead we accumulate all `idx` arrays into a batch list and pass
        # them jointly to mx.eval — that WILL evaluate every element.
        arrs = []
        for _ in range(n_layers):
            scores = mx.random.uniform(shape=(1, L, P), dtype=mx.bfloat16)
            scale = mx.array([[[1.0]]], dtype=mx.bfloat16)
            scores = scores * scale
            idx = mx.argsort(-scores, axis=-1)[..., :512]
            arrs.append(idx)
            if sync_per_op:
                mx.eval(idx)
        if sync_per_op:
            return arrs[-1]
        # Force evaluation of ALL argsorts, not just the last one.
        mx.eval(*arrs)
        return arrs[-1]

    # Warmup
    for _ in range(2):
        _ = one_chunk()

    # Timed chunks
    per_chunk_s = []
    per_chunk_active_end = []
    for i in range(n_chunks):
        t0 = time.perf_counter()
        _ = one_chunk()
        # Ensure fully drained for wall-time measurement of THIS chunk.
        mx.synchronize()
        t1 = time.perf_counter()
        per_chunk_s.append(t1 - t0)
        per_chunk_active_end.append(mx.get_active_memory())

    result = {
        "P": P, "L": L, "n_layers": n_layers, "n_chunks": n_chunks,
        "sync_per_op": sync_per_op, "limit_gb": limit_gb, "ballast_gb": ballast_gb,
        "pre_active_gb": pre_active / (1 << 30),
        "pre_cache_gb": pre_cache / (1 << 30),
        "installed_limit_gb": installed_limit / (1 << 30),
        "default_limit_gb": default_limit / (1 << 30),
        "device_max_rec_gb": dev_info["max_recommended_working_set_size"] / (1 << 30),
        "device_memsize_gb": dev_info["memory_size"] / (1 << 30),
        "per_chunk_s": per_chunk_s,
        "per_chunk_active_end_gb": [a / (1 << 30) for a in per_chunk_active_end],
        "median_chunk_s": statistics.median(per_chunk_s),
        "min_chunk_s": min(per_chunk_s),
        "max_chunk_s": max(per_chunk_s),
        "mean_chunk_s": statistics.mean(per_chunk_s),
        "stdev_chunk_s": statistics.stdev(per_chunk_s) if len(per_chunk_s) > 1 else 0.0,
        "env_MLX_MAX_MB_PER_BUFFER": os.environ.get("MLX_MAX_MB_PER_BUFFER"),
        "env_EXO_MAX_ACTIVE_TASKS": os.environ.get("EXO_MAX_ACTIVE_TASKS"),
        "mlx_ver": mx.__version__,
    }
    del ballast
    print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--P", type=int, default=85000, help="pooled length (era 85000)")
    ap.add_argument("--L", type=int, default=128, help="chunk seq length")
    ap.add_argument("--layers", type=int, default=21, help="argsort ops per chunk")
    ap.add_argument("--chunks", type=int, default=25)
    ap.add_argument("--limit-gb", type=float, default=None,
                    help="memory_limit to install (default: pick automatically)")
    args = ap.parse_args()

    if args.child:
        child_main()
        return

    # Discover machine.
    import mlx.core as as_mx  # only to get device info in the parent
    info = as_mx.device_info()
    memsize_gb = info["memory_size"] / (1 << 30)
    max_rec_gb = info["max_recommended_working_set_size"] / (1 << 30)
    default_block_limit_gb = min(1.5 * max_rec_gb, 0.95 * memsize_gb)
    print(f"# Machine: {info['device_name']} memsize={memsize_gb:.2f} GB "
          f"max_rec={max_rec_gb:.2f} GB default_limit={default_block_limit_gb:.2f} GB")

    # Pick a low limit and a ballast to cross it in arm B.
    # Ballast ~4 GB. In arm A: limit=8 GB (ballast 4 → 50% of limit).
    # In arms B/C/D: limit=3.5 GB (ballast 4 → over the limit).
    common_env = {"EXO_MAX_ACTIVE_TASKS": "5"}

    cfgs = [
        # Arm A: active well below limit → no throttle branch triggers.
        {
            "arm": "A_under_limit",
            "P": args.P, "L": args.L, "n_layers": args.layers,
            "n_chunks": args.chunks, "sync_per_op": False,
            "limit_gb": 8.0, "ballast_gb": 4.0,
            "env": {**common_env, "MLX_MAX_MB_PER_BUFFER": "50"},
        },
        # Arm B: active OVER the limit → memory branch of throttle fires.
        {
            "arm": "B_over_limit_MB50",
            "P": args.P, "L": args.L, "n_layers": args.layers,
            "n_chunks": args.chunks, "sync_per_op": False,
            "limit_gb": 3.5, "ballast_gb": 4.0,
            "env": {**common_env, "MLX_MAX_MB_PER_BUFFER": "50"},
        },
        # Arm C: over limit but MB=200 → fewer commits per chain, so even if
        # the memory branch fires it re-enters the throttle less often.
        {
            "arm": "C_over_limit_MB200",
            "P": args.P, "L": args.L, "n_layers": args.layers,
            "n_chunks": args.chunks, "sync_per_op": False,
            "limit_gb": 3.5, "ballast_gb": 4.0,
            "env": {**common_env, "MLX_MAX_MB_PER_BUFFER": "200"},
        },
        # Arm D: over limit but sync per op → n_active_tasks stays at 0-1 →
        # neither branch of the throttle fires (memory branch also requires
        # n_active_tasks > 0).
        {
            "arm": "D_over_limit_sync",
            "P": args.P, "L": args.L, "n_layers": args.layers,
            "n_chunks": args.chunks, "sync_per_op": True,
            "limit_gb": 3.5, "ballast_gb": 4.0,
            "env": {**common_env, "MLX_MAX_MB_PER_BUFFER": "50"},
        },
    ]

    with open(RESULTS, "a") as f:
        f.write(f"\n# run at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"cwd={os.getcwd()} pid={os.getpid()}\n")
        for cfg in cfgs:
            print(f"## running arm={cfg['arm']}")
            t0 = time.time()
            r = child_run(cfg)
            t1 = time.time()
            r["arm"] = cfg["arm"]
            r["wall_s"] = t1 - t0
            f.write(json.dumps(r) + "\n")
            f.flush()
            if r["ok"]:
                res = r["result"]
                print(f"  arm={cfg['arm']} med={res['median_chunk_s']*1000:.1f}ms "
                      f"min={res['min_chunk_s']*1000:.1f}ms "
                      f"max={res['max_chunk_s']*1000:.1f}ms "
                      f"stdev={res['stdev_chunk_s']*1000:.1f}ms "
                      f"active_end~{res['per_chunk_active_end_gb'][-1]:.2f}GB "
                      f"limit={res['installed_limit_gb']:.2f}GB")
            else:
                print(f"  arm={cfg['arm']} FAILED")
                print(f"  stderr:\n{r.get('stderr','')[-1000:]}")


if __name__ == "__main__":
    main()

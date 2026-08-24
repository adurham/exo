#!/usr/bin/env python3
"""
prefill_cliff_gclimit_repro_local.py

DECISIVE local reproduction: does the MLX allocator gc_limit cache-release
threshold (allocator.cpp:66,149-151) produce a sharp per-chunk knee when
`mem_required = active + cache + size` crosses `gc_limit`?

Design:
- One fresh child process per arm (so mx.set_memory_limit takes effect
  before allocator activity).
- ENV: MLX_MAX_MB_PER_BUFFER=50 (era), EXO_MAX_ACTIVE_TASKS=5 (era).
- Ballast bf16 tensor of `ballast_gb` sits live for the whole run.
- Per "chunk" (one pseudo-forward): 21 sequential pseudo-layers each
  = matmul chain producing an indexer-like score transient (S_t bytes),
    argsort of a (1, L=128, P) bf16, and one dependent matmul.
    All lazy; one mx.eval per chunk (unless sync_per_layer arm).
- Sweep `ballast_gb` around the crossing so watermark walks across gc_limit.
- Record per-chunk wall + active/cache/peak memory.
- Record max/median ratio (bimodality proxy).

Arms:
  A   ballast comfortably below gc_limit (control)
  B1-B4 fine ballast steps walking watermark across gc_limit
  C   worst B + MLX_MAX_MB_PER_BUFFER=200
  D   worst B + mx.eval per-layer (sync)
  E   worst B + tiled transients (S_t/10 per piece)
  F   worst B + tiny wired-limit (residency pressure), only if A-E give null

Machine safety: total (ballast + transients + slack) capped ≤ 80% of
physical RAM. On a 36 GB M4 Max, physical=38.65 GB, cap ≈ 30 GB.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import statistics
from pathlib import Path


HERE = Path(__file__).parent
RESULTS = HERE / "prefill_cliff_gclimit_repro_local_results.jsonl"


def child_arm(arm_json: str) -> None:
    """Run inside the fresh child process."""
    arm = json.loads(arm_json)
    import mlx.core as mx

    # --- setup ---
    mem_limit_gb = arm["mem_limit_gb"]
    ballast_gb = arm["ballast_gb"]
    P = arm["P"]
    L = arm["L"]
    n_layers = arm["n_layers"]
    n_chunks = arm["n_chunks"]
    warmup = arm["warmup"]
    sync_per_layer = arm["sync_per_layer"]
    tile_transient = arm.get("tile_transient", 1)
    matmul_dim = arm["matmul_dim"]  # controls S_t size
    label = arm["label"]

    # Lower gc_limit by lowering memory_limit (allocator.cpp:89-96 semantics)
    prior_lim = mx.set_memory_limit(int(mem_limit_gb * 1024 * 1024 * 1024))
    # Optional wired-limit squeeze for residency-pressure arm
    if arm.get("wired_limit_gb") is not None:
        try:
            mx.set_wired_limit(int(arm["wired_limit_gb"] * 1024 * 1024 * 1024))
        except Exception as e:
            print(f"[child] set_wired_limit failed: {e}", file=sys.stderr)

    # Ballast: bf16 array; hold reference so it stays live
    ballast_elems = int(ballast_gb * 1024 * 1024 * 1024) // 2
    # break into shape (N, 1024) for reasonable allocation
    ballast_rows = max(1, ballast_elems // 1024)
    ballast = mx.zeros((ballast_rows, 1024), dtype=mx.bfloat16)
    mx.eval(ballast)

    active_after_ballast = mx.get_active_memory()
    cache_after_ballast = mx.get_cache_memory()
    print(
        f"[child {label}] limit={mem_limit_gb}GB ballast={ballast_gb}GB "
        f"active={active_after_ballast/1e9:.2f}GB cache={cache_after_ballast/1e9:.2f}GB",
        file=sys.stderr,
    )

    # --- workload ---
    def one_layer(seed: int, S_t_bytes: int):
        """Build a pseudo-indexer layer that produces S_t bytes of transient scratch."""
        # matmul chain builds a large intermediate then reduces
        # Shape: (matmul_dim, matmul_dim) x (matmul_dim, matmul_dim) → matmul_dim^2 elements bf16
        A = mx.random.normal((matmul_dim, matmul_dim), dtype=mx.bfloat16)
        B = mx.random.normal((matmul_dim, matmul_dim), dtype=mx.bfloat16)
        C = A @ B
        # generate score-like tensor of shape (1, L, P) bf16
        scores = mx.random.normal((1, L, P), dtype=mx.bfloat16)
        # argsort scores → uint32 same shape (indexer output)
        idx = mx.argsort(-scores, axis=-1)[..., :64]
        # dependent op consuming both
        C_sum = C.sum()
        idx_sum = idx.sum()
        return C_sum + idx_sum.astype(mx.bfloat16)

    S_t_bytes = matmul_dim * matmul_dim * 2

    def one_chunk(chunk_idx: int):
        arrs = []
        if tile_transient == 1:
            for li in range(n_layers):
                arrs.append(one_layer(chunk_idx * 1000 + li, S_t_bytes))
            if sync_per_layer:
                for a in arrs:
                    mx.eval(a)
            else:
                mx.eval(*arrs)
        else:
            # tiled: each layer produces `tile_transient` smaller pieces
            for li in range(n_layers):
                for ti in range(tile_transient):
                    arrs.append(
                        one_layer(chunk_idx * 10000 + li * 100 + ti, S_t_bytes // tile_transient)
                    )
                if sync_per_layer:
                    for a in arrs[-tile_transient:]:
                        mx.eval(a)
            if not sync_per_layer:
                mx.eval(*arrs)

    # warmup
    for i in range(warmup):
        one_chunk(-100 - i)

    # timed
    per_chunk = []
    for i in range(n_chunks):
        t0 = time.perf_counter()
        one_chunk(i)
        t1 = time.perf_counter()
        per_chunk.append(
            {
                "chunk": i,
                "wall_ms": (t1 - t0) * 1000.0,
                "active_gb": mx.get_active_memory() / 1e9,
                "cache_gb": mx.get_cache_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
        )

    out = {
        "label": label,
        "arm": arm,
        "prior_limit_bytes": prior_lim,
        "active_after_ballast_gb": active_after_ballast / 1e9,
        "cache_after_ballast_gb": cache_after_ballast / 1e9,
        "per_chunk": per_chunk,
    }
    print("RESULT_JSON:" + json.dumps(out))


def run_arm(arm: dict) -> dict:
    """Spawn a fresh child, wait for RESULT_JSON, return parsed result."""
    env = os.environ.copy()
    env["MLX_MAX_MB_PER_BUFFER"] = str(arm.get("mlx_max_mb", 50))
    env["MLX_MAX_OPS_PER_BUFFER"] = str(arm.get("mlx_max_ops", 200))
    env["EXO_MAX_ACTIVE_TASKS"] = str(arm.get("exo_max_active_tasks", 5))
    env["PYTHONWARNINGS"] = "ignore"

    cmd = [
        sys.executable,
        __file__,
        "--child",
        json.dumps(arm),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=1200
    )
    if proc.returncode != 0:
        print(f"[parent {arm['label']}] child failed rc={proc.returncode}", file=sys.stderr)
        print(f"  stderr: {proc.stderr[-2000:]}", file=sys.stderr)
        return {"label": arm["label"], "arm": arm, "error": proc.stderr[-2000:]}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return {"label": arm["label"], "arm": arm, "error": "no RESULT_JSON"}


def summarize(res: dict) -> dict:
    if "error" in res:
        return {"label": res["label"], "error": res["error"]}
    walls = [c["wall_ms"] for c in res["per_chunk"]]
    caches = [c["cache_gb"] for c in res["per_chunk"]]
    peaks = [c["peak_gb"] for c in res["per_chunk"]]
    med = statistics.median(walls)
    mn = min(walls)
    mx_ = max(walls)
    return {
        "label": res["label"],
        "n": len(walls),
        "median_ms": round(med, 2),
        "min_ms": round(mn, 2),
        "max_ms": round(mx_, 2),
        "stdev_ms": round(statistics.stdev(walls) if len(walls) >= 2 else 0.0, 2),
        "max_over_median": round(mx_ / med, 3),
        "bimodal": (mx_ / med) > 1.20,
        "active_after_ballast_gb": round(res["active_after_ballast_gb"], 2),
        "cache_first_ms": round(caches[0], 2),
        "cache_last_ms": round(caches[-1], 2),
        "peak_last_gb": round(peaks[-1], 2),
    }


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child_arm(sys.argv[2])
        return

    # Machine-safety: measure physical RAM
    import ctypes
    import platform

    if platform.system() == "Darwin":
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        # sysctl hw.memsize
        import subprocess as sp
        physical_bytes = int(sp.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    else:
        physical_bytes = 32 * 1024**3
    physical_gb = physical_bytes / (1024**3)
    max_workload_gb = physical_gb * 0.80
    print(f"[parent] physical RAM = {physical_gb:.2f} GB, workload cap = {max_workload_gb:.2f} GB")

    # Base workload: 21 layers, L=128, P=85000 (era shape). matmul_dim controls per-layer
    # transient. matmul_dim=4096 → 32 MB / matmul intermediate per layer (bf16 4096x4096).
    # 21 layers * ~50 MB per epoch peak scratch.
    #
    # Choose mem_limit=6 GB so gc_limit=6 GB. Then ballast walks watermark:
    #   - A: ballast=3 GB → watermark ~ 3+scratch (well under 6)
    #   - B1: ballast=4.5 GB
    #   - B2: ballast=5.0 GB
    #   - B3: ballast=5.5 GB
    #   - B4: ballast=5.8 GB
    # Total including ~2 GB peak scratch and cache is ~7.5 GB max — well under 30 GB cap.

    base = {
        "mem_limit_gb": 6.0,
        "P": 85000,
        "L": 128,
        "n_layers": 21,
        "n_chunks": 18,
        "warmup": 2,
        "sync_per_layer": False,
        "tile_transient": 1,
        "matmul_dim": 4096,
        "mlx_max_mb": 50,
    }

    arms = []
    for ballast_gb, label in [
        (3.0, "A_control"),
        (4.5, "B1_below_gc"),
        (5.0, "B2_near_gc"),
        (5.3, "B3_at_gc"),
        (5.6, "B4_across_gc"),
    ]:
        a = dict(base)
        a["ballast_gb"] = ballast_gb
        a["label"] = label
        arms.append(a)

    # C: MB=200 at the worst ballast (B4)
    c = dict(base)
    c.update({"ballast_gb": 5.6, "mlx_max_mb": 200, "label": "C_mb200_across_gc"})
    arms.append(c)

    # D: sync per layer at worst ballast
    d = dict(base)
    d.update({"ballast_gb": 5.6, "sync_per_layer": True, "label": "D_sync_across_gc"})
    arms.append(d)

    # E: tile transients (era tiled-P analog) at worst ballast
    e = dict(base)
    e.update({"ballast_gb": 5.6, "tile_transient": 10, "label": "E_tiled_across_gc"})
    arms.append(e)

    RESULTS.write_text("")

    summaries = []
    for arm in arms:
        # safety cap
        approx = arm["ballast_gb"] + 3.0
        if approx > max_workload_gb:
            print(f"[parent] SKIP {arm['label']}: est {approx:.1f} GB > cap")
            continue
        print(f"[parent] running {arm['label']} (ballast={arm['ballast_gb']} GB, "
              f"mb={arm['mlx_max_mb']}, sync={arm['sync_per_layer']}, "
              f"tile={arm['tile_transient']})")
        t0 = time.perf_counter()
        res = run_arm(arm)
        t1 = time.perf_counter()
        with RESULTS.open("a") as f:
            f.write(json.dumps(res) + "\n")
        s = summarize(res)
        s["wall_s"] = round(t1 - t0, 1)
        summaries.append(s)
        print(f"  → {s}")

    print("\n=== SUMMARY ===")
    hdr = f"{'label':<25} {'n':>3} {'med_ms':>8} {'min_ms':>8} {'max_ms':>8} {'stdev':>7} {'max/med':>8} {'bimodal':>8} {'active_ball':>11}"
    print(hdr)
    for s in summaries:
        if "error" in s:
            print(f"{s['label']:<25} ERROR: {s['error'][:80]}")
            continue
        print(
            f"{s['label']:<25} {s['n']:>3} {s['median_ms']:>8} {s['min_ms']:>8} "
            f"{s['max_ms']:>8} {s['stdev_ms']:>7} {s['max_over_median']:>8} "
            f"{'YES' if s['bimodal'] else 'no':>8} {s['active_after_ballast_gb']:>11}"
        )


if __name__ == "__main__":
    main()

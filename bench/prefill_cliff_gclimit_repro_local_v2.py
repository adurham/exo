#!/usr/bin/env python3
"""
prefill_cliff_gclimit_repro_local_v2.py — round-3 pt2

Follow-up: previous run confirmed cache_gb collapses in lockstep with
headroom (direct evidence gc_limit release is firing), but wall-time
effect is smooth (~15%), not bimodal. This variant pushes harder:
  - Lower gc_limit relative to per-chunk transient (make GC a hard wall)
  - Larger per-layer transients (score tensor P scaled up)
  - More chunks to catch bimodal outliers
  - Add residency-set-pressure arm (F): lower wired_limit
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
RESULTS = HERE / "prefill_cliff_gclimit_repro_local_v2_results.jsonl"


def child_arm(arm_json: str) -> None:
    arm = json.loads(arm_json)
    import mlx.core as mx

    mem_limit_gb = arm["mem_limit_gb"]
    ballast_gb = arm["ballast_gb"]
    P = arm["P"]
    L = arm["L"]
    n_layers = arm["n_layers"]
    n_chunks = arm["n_chunks"]
    warmup = arm["warmup"]
    sync_per_layer = arm["sync_per_layer"]
    matmul_dim = arm["matmul_dim"]
    label = arm["label"]

    prior_lim = mx.set_memory_limit(int(mem_limit_gb * 1024 * 1024 * 1024))
    if arm.get("wired_limit_gb") is not None:
        try:
            mx.set_wired_limit(int(arm["wired_limit_gb"] * 1024 * 1024 * 1024))
        except Exception as e:
            print(f"[child] set_wired_limit failed: {e}", file=sys.stderr)

    ballast_elems = int(ballast_gb * 1024 * 1024 * 1024) // 2
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

    def one_layer(_seed: int):
        A = mx.random.normal((matmul_dim, matmul_dim), dtype=mx.bfloat16)
        B = mx.random.normal((matmul_dim, matmul_dim), dtype=mx.bfloat16)
        C = A @ B
        scores = mx.random.normal((1, L, P), dtype=mx.bfloat16)
        idx = mx.argsort(-scores, axis=-1)[..., :64]
        return C.sum() + idx.sum().astype(mx.bfloat16)

    def one_chunk(chunk_idx: int):
        arrs = []
        for li in range(n_layers):
            arrs.append(one_layer(chunk_idx * 1000 + li))
        if sync_per_layer:
            for a in arrs:
                mx.eval(a)
        else:
            mx.eval(*arrs)

    for i in range(warmup):
        one_chunk(-100 - i)

    per_chunk = []
    for i in range(n_chunks):
        t0 = time.perf_counter()
        one_chunk(i)
        t1 = time.perf_counter()
        per_chunk.append({
            "chunk": i,
            "wall_ms": (t1 - t0) * 1000.0,
            "active_gb": mx.get_active_memory() / 1e9,
            "cache_gb": mx.get_cache_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
        })

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
    env = os.environ.copy()
    env["MLX_MAX_MB_PER_BUFFER"] = str(arm.get("mlx_max_mb", 50))
    env["MLX_MAX_OPS_PER_BUFFER"] = str(arm.get("mlx_max_ops", 200))
    env["EXO_MAX_ACTIVE_TASKS"] = str(arm.get("exo_max_active_tasks", 5))
    env["PYTHONWARNINGS"] = "ignore"

    cmd = [sys.executable, __file__, "--child", json.dumps(arm)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    if proc.returncode != 0:
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
        "cache_min_gb": round(min(caches), 3),
        "cache_max_gb": round(max(caches), 3),
        "active_after_ballast_gb": round(res["active_after_ballast_gb"], 2),
    }


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child_arm(sys.argv[2])
        return

    # HARDER regime: bigger transient per layer, tighter gc_limit
    # matmul_dim=6144 → 6144x6144x2 = 72 MB per matmul intermediate
    # P=100000 → score tensor (1,128,100000) bf16 = 25.6 MB per layer
    # 21 layers → ~2 GB transient scratch per chunk under MB=50 stacking
    base = {
        "mem_limit_gb": 5.0,
        "P": 100000,
        "L": 128,
        "n_layers": 21,
        "n_chunks": 30,
        "warmup": 3,
        "sync_per_layer": False,
        "matmul_dim": 6144,
        "mlx_max_mb": 50,
    }

    arms = []
    # Fine ballast sweep across gc_limit=5 GB
    for ballast_gb, label in [
        (2.0, "V2_A_control"),
        (3.0, "V2_B1_low"),
        (3.5, "V2_B2_below"),
        (4.0, "V2_B3_near"),
        (4.3, "V2_B4_at"),
        (4.5, "V2_B5_across"),
    ]:
        a = dict(base)
        a["ballast_gb"] = ballast_gb
        a["label"] = label
        arms.append(a)

    # C: MB=200 at worst ballast
    c = dict(base)
    c.update({"ballast_gb": 4.5, "mlx_max_mb": 200, "label": "V2_C_mb200_worst"})
    arms.append(c)

    # D: sync per layer at worst ballast
    d = dict(base)
    d.update({"ballast_gb": 4.5, "sync_per_layer": True, "label": "V2_D_sync_worst"})
    arms.append(d)

    # F: residency pressure — small wired_limit
    f = dict(base)
    f.update({"ballast_gb": 4.5, "wired_limit_gb": 3.0, "label": "V2_F_wired3_worst"})
    arms.append(f)

    RESULTS.write_text("")

    summaries = []
    for arm in arms:
        print(f"[parent] running {arm['label']}")
        t0 = time.perf_counter()
        res = run_arm(arm)
        t1 = time.perf_counter()
        with RESULTS.open("a") as f_out:
            f_out.write(json.dumps(res) + "\n")
        s = summarize(res)
        s["wall_s"] = round(t1 - t0, 1)
        summaries.append(s)
        print(f"  → {s}")

    print("\n=== V2 SUMMARY ===")
    print(f"{'label':<25} {'med':>7} {'min':>7} {'max':>7} {'sd':>6} {'m/med':>6} {'cache_min':>10} {'cache_max':>10} {'active_ball':>11}")
    for s in summaries:
        if "error" in s:
            print(f"{s['label']:<25} ERROR")
            continue
        print(
            f"{s['label']:<25} {s['median_ms']:>7} {s['min_ms']:>7} {s['max_ms']:>7} "
            f"{s['stdev_ms']:>6} {s['max_over_median']:>6} "
            f"{s['cache_min_gb']:>10} {s['cache_max_gb']:>10} {s['active_after_ballast_gb']:>11}"
        )


if __name__ == "__main__":
    main()

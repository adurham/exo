# P14d: Part B small-M ceiling, tile-ALIGNED only, PROPERLY CONVERGED
# chained wall-clock (n_iters scaled so total wall time per measurement is
# always >=15ms, empirically confirmed sufficient for convergence at this
# GPU/dispatch overhead scale -- see P14c convergence diagnostic: M=32 needed
# n_chain>=256 i.e. ~256*45us=11.5ms; using a floor of 15ms of wall time
# guarantees convergence at every M tested here without hand-tuning n_iters
# per M).
import os
for v in ("MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"
import json
import time
from pathlib import Path
import mlx.core as mx

HIDDEN = 4096
INTER = 1024
GROUP_SIZE = 32
BITS = 4
QUANT_MODE = "mxfp4"
PEAK_TFLOPS_MEASURED = 15.21e12
OUT = Path("/Users/adam.durham/repos/exo/tmp/p14-20260914")
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def wall_clock_chained_timed_floor(fn, pool_size, min_wall_ms=15.0, warmup=20, max_iters=4000):
    for i in range(warmup):
        mx.eval(fn(i % pool_size))
    mx.synchronize()
    # Calibration pass: find n_iters that gets us to at least min_wall_ms
    n_iters = 16
    while True:
        outs = []
        t0 = time.perf_counter()
        for i in range(n_iters):
            outs.append(fn(i % pool_size))
        mx.eval(*outs)
        mx.synchronize()
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000
        if wall_ms >= min_wall_ms or n_iters >= max_iters:
            return (t1 - t0) / n_iters, n_iters
        n_iters = min(n_iters * 2, max_iters)


def make_mxfp4_weight(out_dim, in_dim, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((out_dim, in_dim)) * 0.02).astype(mx.float32)
    wq, scales = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode=QUANT_MODE)
    mx.eval(wq, scales)
    return wq, scales


def make_bf16_weight(out_dim, in_dim, seed):
    mx.random.seed(seed + 999)
    w = (mx.random.normal((out_dim, in_dim)) * 0.02).astype(mx.bfloat16)
    mx.eval(w)
    return w


def bench_mxfp4(M, out_dim, in_dim, wq, scales, pool_n=32):
    pool = [(mx.random.normal((M, in_dim)) * 0.1).astype(mx.bfloat16) for _ in range(pool_n)]
    mx.eval(*pool)

    def fn(i):
        return mx.quantized_matmul(pool[i], wq, scales, transpose=True,
                                    group_size=GROUP_SIZE, bits=BITS, mode=QUANT_MODE)

    s_per_call, n_used = wall_clock_chained_timed_floor(fn, pool_n)
    flops = 2 * M * in_dim * out_dim
    tflops = flops / s_per_call / 1e12
    return {"M": M, "s_per_call": s_per_call, "n_iters_used": n_used, "dtype": "mxfp4",
            "achieved_tflops": tflops, "pct_of_measured_peak": tflops / (PEAK_TFLOPS_MEASURED / 1e12) * 100}


def bench_bf16(M, out_dim, in_dim, w, pool_n=32):
    pool = [(mx.random.normal((M, in_dim)) * 0.1).astype(mx.bfloat16) for _ in range(pool_n)]
    mx.eval(*pool)
    wt = w.T
    mx.eval(wt)

    def fn(i):
        return mx.matmul(pool[i], wt)

    s_per_call, n_used = wall_clock_chained_timed_floor(fn, pool_n)
    flops = 2 * M * in_dim * out_dim
    tflops = flops / s_per_call / 1e12
    return {"M": M, "s_per_call": s_per_call, "n_iters_used": n_used, "dtype": "bf16",
            "achieved_tflops": tflops, "pct_of_measured_peak": tflops / (PEAK_TFLOPS_MEASURED / 1e12) * 100}


SHAPES = {
    "fused_gate_up": {"out_dim": 2 * INTER, "in_dim": HIDDEN},
    "down_proj": {"out_dim": HIDDEN, "in_dim": INTER},
}


def main():
    log("=== P14d: Part B small-M ceiling, tile-ALIGNED, CONVERGED (>=15ms floor per point) ===")
    mx.random.seed(20260914)
    m_values = [32, 64, 96, 128, 160, 224, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384]
    rows = []
    for shape_name, shape in SHAPES.items():
        out_dim, in_dim = shape["out_dim"], shape["in_dim"]
        wq, scales = make_mxfp4_weight(out_dim, in_dim, seed=hash(shape_name) % 10000)
        w_bf16 = make_bf16_weight(out_dim, in_dim, seed=hash(shape_name) % 10000)
        for M in m_values:
            r_mxfp4 = bench_mxfp4(M, out_dim, in_dim, wq, scales)
            r_bf16 = bench_bf16(M, out_dim, in_dim, w_bf16)
            row = {"shape": shape_name, "M": M, "mxfp4": r_mxfp4, "bf16": r_bf16}
            rows.append(row)
            log(f"  [{shape_name}] M={M:6d}: mxfp4 {r_mxfp4['achieved_tflops']:.2f}TF "
                f"({r_mxfp4['pct_of_measured_peak']:.1f}%peak, n={r_mxfp4['n_iters_used']}) | "
                f"bf16 {r_bf16['achieved_tflops']:.2f}TF ({r_bf16['pct_of_measured_peak']:.1f}%peak, n={r_bf16['n_iters_used']}) | "
                f"quant-tax={r_mxfp4['achieved_tflops']/r_bf16['achieved_tflops']*100:.1f}% of bf16")
    out_path = OUT / f"p14d_converged_smallm_{int(time.time())}.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    log(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

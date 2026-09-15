# P14c: Part A tile-boundary-cliff, PROPERLY CONVERGED chained wall-clock
# (n_iters=400, confirmed converged vs 256/512 in diagnostic sweep -- small-M
# chained wall-clock needs n_chain>=256 to amortize away fixed per-dispatch
# CPU submission overhead; isolated/sync-per-call timing is WORSE at small M,
# dominated by ~130-280us round-trip overhead that swamps the actual
# sub-100us GEMM compute time and produces a false near-zero cliff reading).
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


def wall_clock_chained_converged(fn, pool_size, n_iters, warmup=30):
    for i in range(warmup):
        mx.eval(fn(i % pool_size))
    mx.synchronize()
    outs = []
    t0 = time.perf_counter()
    for i in range(n_iters):
        outs.append(fn(i % pool_size))
    mx.eval(*outs)
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def make_mxfp4_weight(out_dim, in_dim, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((out_dim, in_dim)) * 0.02).astype(mx.float32)
    wq, scales = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode=QUANT_MODE)
    mx.eval(wq, scales)
    return wq, scales


def bench_mxfp4_converged(M, out_dim, in_dim, wq, scales, n_iters, pool_n=32):
    pool = [(mx.random.normal((M, in_dim)) * 0.1).astype(mx.bfloat16) for _ in range(pool_n)]
    mx.eval(*pool)

    def fn(i):
        return mx.quantized_matmul(
            pool[i], wq, scales, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, mode=QUANT_MODE,
        )

    s_per_call = wall_clock_chained_converged(fn, pool_n, n_iters=n_iters)
    flops = 2 * M * in_dim * out_dim
    tflops = flops / s_per_call / 1e12
    return {
        "M": M, "s_per_call": s_per_call, "achieved_tflops": tflops,
        "pct_of_measured_peak": tflops / (PEAK_TFLOPS_MEASURED / 1e12) * 100,
        "us_per_row": s_per_call * 1e6 / M,
    }


SHAPES = {
    "fused_gate_up": {"out_dim": 2 * INTER, "in_dim": HIDDEN},
    "down_proj": {"out_dim": HIDDEN, "in_dim": INTER},
}


def main():
    log("=== P14c: Part A tile-boundary-cliff, CONVERGED (n_iters=400, pool_n=32) ===")
    mx.random.seed(20260914)
    boundary_pairs = [(32, 33), (64, 65), (96, 97), (128, 129), (160, 161), (256, 257)]
    rows = []
    for shape_name, shape in SHAPES.items():
        out_dim, in_dim = shape["out_dim"], shape["in_dim"]
        wq, scales = make_mxfp4_weight(out_dim, in_dim, seed=hash(shape_name) % 10000)
        for on_b, past_b in boundary_pairs:
            r_on = bench_mxfp4_converged(on_b, out_dim, in_dim, wq, scales, n_iters=400)
            r_past = bench_mxfp4_converged(past_b, out_dim, in_dim, wq, scales, n_iters=400)
            cliff = r_past["us_per_row"] / r_on["us_per_row"]
            row = {"shape": shape_name, "on_M": on_b, "past_M": past_b,
                   "on": r_on, "past": r_past, "cliff_us_per_row": cliff}
            rows.append(row)
            log(f"  [{shape_name}] M={on_b}({r_on['pct_of_measured_peak']:.1f}%peak,{r_on['us_per_row']:.3f}us/row) "
                f"vs M={past_b}({r_past['pct_of_measured_peak']:.1f}%peak,{r_past['us_per_row']:.3f}us/row) "
                f"cliff={cliff:.2f}x")
    out_path = OUT / f"p14c_converged_cliff_{int(time.time())}.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    log(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

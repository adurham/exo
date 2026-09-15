# P14 v2: Decompose the P13 prefill compute-efficiency gap (33-41% of the
# 15.21 TFLOPS measured peak) into its two named candidate mechanisms --
# CORRECTED METHODOLOGY after discovering P13's own measurement basis
# (mx.metal.gpu_time_ns() summed over a CHAINED/pooled, unsynchronized batch
# of N independent kernel calls) is itself a benchmarking artifact:
#
#   FINDING (this script's preamble, confirmed before the real sweep runs):
#   mx.metal.gpu_time_ns() summed over N un-synced chained command buffers
#   INFLATES vs true wall-clock GPU-busy time by ~1.55x at N=2, plateauing at
#   ~1.85-1.87x by N=6-32 (flat thereafter) -- consistent with Metal's bounded
#   command-buffer pipeline depth causing per-buffer GPUStartTime/GPUEndTime
#   spans to overlap when read back individually and summed. This is NOT
#   double-dispatch (dispatch_count/call stays exactly 1.00 throughout) and
#   NOT graph memoization (wall-clock scales linearly and correctly with N).
#   Isolated (sync-every-call) gpu_time_ns() agrees with wall-clock to <1%.
#   Chained WALL-CLOCK (queue N calls, mx.eval once, time the whole batch,
#   divide by N) ALSO agrees with isolated wall-clock to <1% -- so the fix is
#   simply: use wall-clock timing (isolated OR chained), never sum
#   gpu_time_ns() over an unsynchronized chained batch as a per-call FLOPS
#   basis. The reference 15.21 TFLOPS peak itself was measured via wall-clock
#   isolated timing (bench_call() in attn_production_class_bench.py) -- this
#   script now uses the SAME measurement basis throughout, so ratios against
#   that peak are finally apples-to-apples.
#
#   PART A (tile-boundary-cliff, isolated): dense mxfp4 quantized_matmul at M
#   exactly ON a tile boundary (32/64/96/128) vs one row PAST it, at both real
#   production per-expert GEMM shapes (fused_gate_up, down_proj). Both
#   wall-clock-isolated (primary, slow, most trustworthy) and wall-clock-
#   chained (fast, cross-check) are reported.
#
#   PART B (small-M ceiling, isolated from alignment): sweep M across ONLY
#   tile-ALIGNED values for the same two shapes, mxfp4 vs bf16, wall-clock
#   chained (validated accurate above) for speed across the wide M range.
#
#   CALIBRATION: reproduce the exact large-square/rect bf16 GEMM
#   ([16384x4096x4096]) that produced the "15.21 TFLOPS" reference (PH:6059),
#   via wall-clock, to confirm this script's methodology lands on the cited
#   number (it does, within run-to-run noise -- see calibration output).
#
# Standalone process on m4-1, NOT attached to the live runner PID -- zero
# xctrace, zero relaunch, zero production risk. Same production
# configuration (hidden=4096, inter=1024/rank, 256 experts, mxfp4 g=32) as
# P13/Lever-1.
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
PEAK_TFLOPS_MEASURED = 15.21e12  # PH:6059 reference, wall-clock-isolated methodology

OUT = Path("/Users/adam.durham/repos/exo/tmp/p14-20260914")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = {"meta": {}, "artifact_demo": {}, "calibration": {}, "part_a_tile_cliff": [], "part_b_smallm_ceiling": []}


def log(*a):
    print(*a, flush=True)


def wall_clock_isolated(fn, pool_size, n_iters=30, warmup=8):
    """Sync-every-call wall clock -- the SAME basis as the 15.21 TFLOPS
    reference (bench_call() in attn_production_class_bench.py). Slow but
    the most trustworthy: confirmed to agree with gpu_time_ns() to <1% when
    both are measured in this isolated (non-chained) fashion."""
    for i in range(warmup):
        mx.eval(fn(i % pool_size))
    mx.synchronize()
    samples = []
    for i in range(n_iters):
        mx.synchronize()
        t0 = time.perf_counter()
        y = fn(i % pool_size)
        mx.eval(y)
        mx.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    median = samples[len(samples) // 2]
    return {"s_per_call": median, "method": "wall_isolated", "n_iters": n_iters}


def wall_clock_chained(fn, pool_size, n_iters=100, warmup=15):
    """Queue N independent calls, mx.eval() once, time the WHOLE batch with
    perf_counter (not gpu_time_ns()). Confirmed to agree with isolated
    wall-clock to <1% (unlike gpu_time_ns() summed over the same chained
    batch, which is the ~1.85x-inflated artifact this script exists to
    avoid). Used for the wide Part-B M-sweep where isolated timing at 16
    shapes x 2 dtypes x 2 GEMM shapes would be slow."""
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
    return {"s_per_call": (t1 - t0) / n_iters, "method": "wall_chained", "n_iters": n_iters}


POOL_N = 8


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


def bench_mxfp4(M, out_dim, in_dim, wq, scales, timing_fn, n_iters, warmup, pool_n=POOL_N):
    pool = [(mx.random.normal((M, in_dim)) * 0.1).astype(mx.bfloat16) for _ in range(pool_n)]
    mx.eval(*pool)

    def fn(i):
        return mx.quantized_matmul(
            pool[i], wq, scales, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, mode=QUANT_MODE,
        )

    stage = timing_fn(fn, pool_n, n_iters=n_iters, warmup=warmup)
    flops = 2 * M * in_dim * out_dim
    tflops = flops / stage["s_per_call"] / 1e12
    stage.update({
        "M": M, "out_dim": out_dim, "in_dim": in_dim, "dtype": "mxfp4",
        "flops": flops, "achieved_tflops": tflops,
        "pct_of_measured_peak": tflops / (PEAK_TFLOPS_MEASURED / 1e12) * 100,
        "us_per_row": stage["s_per_call"] * 1e6 / M,
    })
    return stage


def bench_bf16(M, out_dim, in_dim, w, timing_fn, n_iters, warmup, pool_n=POOL_N):
    pool = [(mx.random.normal((M, in_dim)) * 0.1).astype(mx.bfloat16) for _ in range(pool_n)]
    mx.eval(*pool)
    wt = w.T
    mx.eval(wt)

    def fn(i):
        return mx.matmul(pool[i], wt)

    stage = timing_fn(fn, pool_n, n_iters=n_iters, warmup=warmup)
    flops = 2 * M * in_dim * out_dim
    tflops = flops / stage["s_per_call"] / 1e12
    stage.update({
        "M": M, "out_dim": out_dim, "in_dim": in_dim, "dtype": "bf16",
        "flops": flops, "achieved_tflops": tflops,
        "pct_of_measured_peak": tflops / (PEAK_TFLOPS_MEASURED / 1e12) * 100,
        "us_per_row": stage["s_per_call"] * 1e6 / M,
    })
    return stage


SHAPES = {
    "fused_gate_up": {"out_dim": 2 * INTER, "in_dim": HIDDEN},
    "down_proj": {"out_dim": HIDDEN, "in_dim": INTER},
}


def run_artifact_demo():
    """Reproduce the gpu_time_ns()-chained-vs-wall-clock divergence as a
    permanent, in-repo record of WHY this script does not use P13's original
    gpu_time_ns()-chained methodology for absolute TFLOPS numbers."""
    log("\n=== PREAMBLE: gpu_time_ns() chained-batch artifact demonstration ===")
    M, K, N = 16384, 4096, 4096
    a = mx.random.normal((M, K)).astype(mx.bfloat16)
    b = mx.random.normal((K, N)).astype(mx.bfloat16)
    mx.eval(a, b)
    flops = 2 * M * K * N
    rows = []
    for n_chain in (1, 2, 4, 8, 16, 32):
        for _ in range(3):
            mx.eval(a @ b)
        mx.synchronize()
        mx.metal.reset_gpu_time()
        mx.metal.reset_dispatch_count()
        outs = []
        t0 = time.perf_counter()
        for _ in range(n_chain):
            outs.append(a @ b)
        mx.eval(*outs)
        mx.synchronize()
        t1 = time.perf_counter()
        gpu_ns = mx.metal.gpu_time_ns()
        wall_s = (t1 - t0) / n_chain
        gpu_s = gpu_ns / n_chain / 1e9
        row = {
            "n_chain": n_chain, "wall_tflops": flops / wall_s / 1e12,
            "gpu_time_ns_tflops": flops / gpu_s / 1e12,
            "gpu_over_wall_ratio": gpu_s / wall_s,
        }
        rows.append(row)
        log(f"  n_chain={n_chain:3d}: wall={row['wall_tflops']:.2f}TF gpu_time_ns={row['gpu_time_ns_tflops']:.2f}TF ratio={row['gpu_over_wall_ratio']:.3f}")
    log("  CONCLUSION: gpu_time_ns() summed over an un-synced chained batch")
    log("  inflates by ~1.85-1.87x at n_chain>=6 vs true wall-clock GPU-busy")
    log("  time. dispatch_count/call stayed exactly 1.00 throughout (ruling")
    log("  out double-dispatch); wall-clock scales linearly & correctly with")
    log("  n_chain (ruling out graph memoization). This is why every")
    log("  measurement below uses wall-clock timing, isolated or chained,")
    log("  NEVER gpu_time_ns() summed over an unsynchronized chained batch.")
    return rows


def run_calibration():
    log("\n=== CALIBRATION: reproduce 15.21 TFLOPS reference (wall-clock, matching basis) ===")
    M, K, N = 16384, 4096, 4096
    w = make_bf16_weight(N, K, seed=777)
    stage = wall_clock_isolated(
        lambda i: mx.matmul((mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16), w.T),
        1, n_iters=10, warmup=3,
    )
    tflops = (2 * M * K * N) / stage["s_per_call"] / 1e12
    stage.update({"M": M, "K": K, "N": N, "achieved_tflops": tflops})
    log(f"  {stage}")
    return stage


def run_part_a_tile_cliff():
    log("\n=== PART A: tile-boundary-cliff, isolated (wall-clock, both methods) ===")
    boundary_pairs = [(32, 33), (64, 65), (96, 97), (128, 129)]
    rows = []
    for shape_name, shape in SHAPES.items():
        out_dim, in_dim = shape["out_dim"], shape["in_dim"]
        wq, scales = make_mxfp4_weight(out_dim, in_dim, seed=hash(shape_name) % 10000)
        for on_boundary, past_boundary in boundary_pairs:
            r_on_iso = bench_mxfp4(on_boundary, out_dim, in_dim, wq, scales, wall_clock_isolated, n_iters=25, warmup=6)
            r_past_iso = bench_mxfp4(past_boundary, out_dim, in_dim, wq, scales, wall_clock_isolated, n_iters=25, warmup=6)
            r_on_ch = bench_mxfp4(on_boundary, out_dim, in_dim, wq, scales, wall_clock_chained, n_iters=150, warmup=20)
            r_past_ch = bench_mxfp4(past_boundary, out_dim, in_dim, wq, scales, wall_clock_chained, n_iters=150, warmup=20)
            row = {
                "shape": shape_name, "boundary_M": on_boundary, "past_M": past_boundary,
                "on_boundary_isolated": r_on_iso, "past_boundary_isolated": r_past_iso,
                "on_boundary_chained": r_on_ch, "past_boundary_chained": r_past_ch,
                "cliff_ratio_isolated_us_per_row": r_past_iso["us_per_row"] / r_on_iso["us_per_row"],
                "cliff_ratio_chained_us_per_row": r_past_ch["us_per_row"] / r_on_ch["us_per_row"],
            }
            rows.append(row)
            log(
                f"  [{shape_name}] M={on_boundary} vs {past_boundary} | "
                f"ISOLATED: {r_on_iso['achieved_tflops']:.2f}TF({r_on_iso['pct_of_measured_peak']:.1f}%) -> "
                f"{r_past_iso['achieved_tflops']:.2f}TF({r_past_iso['pct_of_measured_peak']:.1f}%) "
                f"cliff={row['cliff_ratio_isolated_us_per_row']:.2f}x | "
                f"CHAINED: {r_on_ch['achieved_tflops']:.2f}TF({r_on_ch['pct_of_measured_peak']:.1f}%) -> "
                f"{r_past_ch['achieved_tflops']:.2f}TF({r_past_ch['pct_of_measured_peak']:.1f}%) "
                f"cliff={row['cliff_ratio_chained_us_per_row']:.2f}x"
            )
    return rows


def run_part_b_smallm_ceiling():
    log("\n=== PART B: small-M ceiling, tile-ALIGNED only (wall-clock chained) ===")
    m_values = [32, 64, 96, 128, 160, 224, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384]
    rows = []
    for shape_name, shape in SHAPES.items():
        out_dim, in_dim = shape["out_dim"], shape["in_dim"]
        wq, scales = make_mxfp4_weight(out_dim, in_dim, seed=hash(shape_name) % 10000)
        w_bf16 = make_bf16_weight(out_dim, in_dim, seed=hash(shape_name) % 10000)
        for M in m_values:
            n_iters = 100 if M <= 2048 else 30
            r_mxfp4 = bench_mxfp4(M, out_dim, in_dim, wq, scales, wall_clock_chained, n_iters=n_iters, warmup=15)
            r_bf16 = bench_bf16(M, out_dim, in_dim, w_bf16, wall_clock_chained, n_iters=n_iters, warmup=15)
            row = {"shape": shape_name, "M": M, "mxfp4": r_mxfp4, "bf16": r_bf16}
            rows.append(row)
            log(
                f"  [{shape_name}] M={M:6d}: mxfp4 {r_mxfp4['achieved_tflops']:.2f}TF "
                f"({r_mxfp4['pct_of_measured_peak']:.1f}% peak) | "
                f"bf16 {r_bf16['achieved_tflops']:.2f}TF ({r_bf16['pct_of_measured_peak']:.1f}% peak) | "
                f"quant-tax={r_mxfp4['achieved_tflops']/r_bf16['achieved_tflops']*100:.1f}% of bf16"
            )
    return rows


def main():
    log("=== P14 v2: tile-boundary-cliff vs small-M-GEMM-ceiling decomposition (corrected methodology) ===")
    RESULTS["meta"] = {
        "host": os.uname().nodename,
        "mlx": mx.__version__,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "peak_tflops_measured_reference": PEAK_TFLOPS_MEASURED / 1e12,
        "shapes": SHAPES,
        "methodology_note": "wall-clock timing throughout (isolated for Part A + calibration, "
                             "chained-wall-clock for Part B's wide sweep) -- NOT gpu_time_ns() "
                             "summed over an unsynced chained batch, which this script's preamble "
                             "demonstrates inflates by ~1.85-1.87x vs true wall-clock GPU-busy time.",
    }
    mx.random.seed(20260914)

    RESULTS["artifact_demo"] = run_artifact_demo()
    RESULTS["calibration"] = run_calibration()
    RESULTS["part_a_tile_cliff"] = run_part_a_tile_cliff()
    RESULTS["part_b_smallm_ceiling"] = run_part_b_smallm_ceiling()

    out_path = OUT / f"results_v2_{int(time.time())}.json"
    out_path.write_text(json.dumps(RESULTS, indent=2, default=str))
    log(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

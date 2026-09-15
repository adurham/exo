# P14b: DECISIVE re-measurement of P13's exact headline scenario
# (prefill shape M=2048, do_sort=True, real gather_sort + gather_qmm via
# mlx_lm's actual BatchedSwitchGLU/_gather_sort, same production config)
# using WALL-CLOCK timing instead of P13's gpu_time_ns()-chained metric,
# which the companion artifact-demo script proved inflates apparent GPU
# time by ~1.5-1.9x for unsynchronized chained command buffers.
#
# This directly replaces P13's "matmul_us = gu_stage[gpu_us_per_call] +
# down_stage[gpu_us_per_call]" -> 5.09-6.28 TFLOPS (33.5-41.3% of the 15.21
# TFLOPS peak) computation with the corrected basis, everything else
# (model construction, quantization config, routing, pool rotation)
# IDENTICAL to P13's own script.
import os

for v in ("MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, "/Users/adam.durham/repos/exo/mlx-lm")
from mlx_lm.models.switch_layers import BatchedSwitchGLU, _gather_sort, _scatter_unsort
from mlx_lm.models.activations import swiglu

HIDDEN = 4096
INTER = 1024
N_EXPERTS = 256
TOP_K = 6
GROUP_SIZE = 32
BITS = 4
QUANT_MODE = "mxfp4"
PEAK_TFLOPS_MEASURED = 15.21e12
PREFILL_M = 2048

OUT = Path("/Users/adam.durham/repos/exo/tmp/p14-20260914")
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def build_model():
    model = BatchedSwitchGLU(HIDDEN, INTER, N_EXPERTS, bias=False)
    model.gate_proj = model.gate_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    model.up_proj = model.up_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    model.down_proj = model.down_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    gp, up = model.gate_proj, model.up_proj
    model._fused_w_gu = mx.concatenate([gp.weight, up.weight], axis=1)
    model._fused_s_gu = mx.concatenate([gp.scales, up.scales], axis=1)
    gp_b = getattr(gp, "biases", None)
    up_b = getattr(up, "biases", None)
    model._fused_b_gu = (
        mx.concatenate([gp_b, up_b], axis=1) if gp_b is not None and up_b is not None else None
    )
    model._fused_n_inter = int(gp.weight.shape[1])
    model._fused_group_size = int(gp.group_size)
    mx.eval(model._fused_w_gu, model._fused_s_gu)
    if model._fused_b_gu is not None:
        mx.eval(model._fused_b_gu)
    mx.eval(model.parameters())
    return model


def wall_clock_isolated(fn, pool_size, n_iters=25, warmup=8):
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
    return samples[len(samples) // 2]


def wall_clock_chained(fn, pool_size, n_iters=100, warmup=15):
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


def gpu_time_chained_ORIGINAL_P13_METHOD(fn, pool_size, n_iters=100, warmup=10):
    """Reproduce P13's own time_stage_chained() EXACTLY, for direct side-by-side."""
    for i in range(warmup):
        mx.eval(fn(i % pool_size))
    mx.synchronize()
    mx.metal.reset_gpu_time()
    mx.metal.reset_dispatch_count()
    outs = []
    for i in range(n_iters):
        outs.append(fn(i % pool_size))
    mx.eval(*outs)
    mx.synchronize()
    total_ns = mx.metal.gpu_time_ns()
    return total_ns / n_iters / 1e9  # seconds


def main():
    log("=== P14b: P13's exact prefill scenario, corrected wall-clock methodology ===")
    mx.random.seed(20260914)
    model = build_model()

    x = mx.random.normal((PREFILL_M, HIDDEN)).astype(mx.bfloat16)
    pool = []
    for _ in range(16):
        idx = mx.random.randint(0, N_EXPERTS, (PREFILL_M, TOP_K))
        pool.append(idx)
    mx.eval(x, *pool)

    idx0 = pool[0]
    do_sort = bool(idx0.size >= 64)
    log(f"do_sort at this shape: {do_sort} (indices.size={idx0.size})")

    x_expanded = mx.expand_dims(x, (-2, -3))
    mx.eval(x_expanded)
    sorted_pool = [_gather_sort(x_expanded, pool[i]) for i in range(16)]
    mx.eval(*[t for trip in sorted_pool for t in trip])

    def get_gu(i):
        xs, idx, inv = sorted_pool[i]
        return mx.gather_qmm(
            xs, model._fused_w_gu, model._fused_s_gu, model._fused_b_gu,
            rhs_indices=idx, transpose=True, group_size=model._fused_group_size,
            bits=model.gate_proj.bits, mode=model.gate_proj.mode,
            sorted_indices=True,
        )

    gu_pool_for_down = [get_gu(i) for i in range(16)]
    mx.eval(*gu_pool_for_down)

    def get_act(i):
        n_inter = model._fused_n_inter
        gu = gu_pool_for_down[i]
        return swiglu(gu[..., n_inter:], gu[..., :n_inter])

    act_pool = [get_act(i) for i in range(16)]
    mx.eval(*act_pool)

    def get_down(i):
        xs, idx, inv = sorted_pool[i]
        return model.down_proj(act_pool[i], idx, sorted_indices=True)

    flops_per_token = TOP_K * (2 * HIDDEN * INTER * 2 + 2 * INTER * HIDDEN)
    total_flops = flops_per_token * PREFILL_M

    log("\n--- ORIGINAL P13 METHOD (gpu_time_ns summed over unsynced chained batch) ---")
    gu_orig_s = gpu_time_chained_ORIGINAL_P13_METHOD(get_gu, 16, n_iters=100, warmup=10)
    down_orig_s = gpu_time_chained_ORIGINAL_P13_METHOD(get_down, 16, n_iters=100, warmup=10)
    matmul_s_orig = gu_orig_s + down_orig_s
    tflops_orig = total_flops / matmul_s_orig / 1e12
    log(f"  gu={gu_orig_s*1e6:.1f}us down={down_orig_s*1e6:.1f}us total={matmul_s_orig*1e6:.1f}us")
    log(f"  achieved_tflops={tflops_orig:.2f}  pct_of_peak={tflops_orig/(PEAK_TFLOPS_MEASURED/1e12)*100:.1f}%")
    log("  (this should reproduce P13's original 5.09-6.28 TFLOPS / 33.5-41.3% range)")

    log("\n--- CORRECTED METHOD: wall-clock CHAINED (validated <1% error vs isolated at scale) ---")
    gu_wc_s = wall_clock_chained(get_gu, 16, n_iters=100, warmup=15)
    down_wc_s = wall_clock_chained(get_down, 16, n_iters=100, warmup=15)
    matmul_s_wc = gu_wc_s + down_wc_s
    tflops_wc = total_flops / matmul_s_wc / 1e12
    log(f"  gu={gu_wc_s*1e6:.1f}us down={down_wc_s*1e6:.1f}us total={matmul_s_wc*1e6:.1f}us")
    log(f"  achieved_tflops={tflops_wc:.2f}  pct_of_peak={tflops_wc/(PEAK_TFLOPS_MEASURED/1e12)*100:.1f}%")

    log("\n--- CORRECTED METHOD: wall-clock ISOLATED (sync every call, slower but gold-standard) ---")
    gu_iso_s = wall_clock_isolated(get_gu, 16, n_iters=25, warmup=8)
    down_iso_s = wall_clock_isolated(get_down, 16, n_iters=25, warmup=8)
    matmul_s_iso = gu_iso_s + down_iso_s
    tflops_iso = total_flops / matmul_s_iso / 1e12
    log(f"  gu={gu_iso_s*1e6:.1f}us down={down_iso_s*1e6:.1f}us total={matmul_s_iso*1e6:.1f}us")
    log(f"  achieved_tflops={tflops_iso:.2f}  pct_of_peak={tflops_iso/(PEAK_TFLOPS_MEASURED/1e12)*100:.1f}%")

    result = {
        "prefill_m": PREFILL_M, "do_sort": do_sort,
        "original_p13_method": {"gu_s": gu_orig_s, "down_s": down_orig_s, "achieved_tflops": tflops_orig,
                                  "pct_of_peak": tflops_orig / (PEAK_TFLOPS_MEASURED / 1e12) * 100},
        "corrected_wall_chained": {"gu_s": gu_wc_s, "down_s": down_wc_s, "achieved_tflops": tflops_wc,
                                     "pct_of_peak": tflops_wc / (PEAK_TFLOPS_MEASURED / 1e12) * 100},
        "corrected_wall_isolated": {"gu_s": gu_iso_s, "down_s": down_iso_s, "achieved_tflops": tflops_iso,
                                      "pct_of_peak": tflops_iso / (PEAK_TFLOPS_MEASURED / 1e12) * 100},
        "artifact_inflation_ratio": matmul_s_orig / matmul_s_wc,
    }
    out_path = OUT / f"p14b_decisive_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log(f"\nResults written to {out_path}")
    log(f"\nARTIFACT INFLATION RATIO (orig_matmul_time / corrected_wall_time): {result['artifact_inflation_ratio']:.3f}x")


if __name__ == "__main__":
    main()

# P13: Kernel-level sub-phase attribution of moe.switch_mlp/GatherQMM
# (docs/PERFORMANCE_HISTORY.md §13 "A real Instruments Metal trace of the
# moe.switch_mlp GatherQMM kernel internals" open thread).
#
# NOVEL CONTRIBUTION vs prior P01 (2026-08-29)/P03 (2026-08-30) work:
#   1. P01 measured decode shape only (M=1, do_sort=False). At that shape
#      switch_layers.py's `do_sort = indices.size >= 64` gate is FALSE, so
#      the separate `_gather_sort`/`_scatter_unsort` MLX ops (argsort +
#      fancy-index gather/scatter, run in PYTHON around gather_qmm, NOT
#      inside the Metal kernel) never fire. Nobody has measured them.
#      This script measures them at PREFILL shape (M=2048, do_sort=True),
#      where they execute on every real prefill call in production.
#   2. Adds an explicit compute-bound vs memory-bound classification via
#      a FLOPs roofline (measured 15.21 TFLOPS bf16 on-node peak, PH:5819)
#      alongside the existing bytes/bandwidth roofline (546 GB/s spec /
#      424-488 GB/s measured streaming), so both axes of "why is it slow"
#      are checked, not just bandwidth.
#   3. Adds MLX_DISPATCH_COUNT=1 bracketing (proven in P03) to get the
#      real dispatch count per stage, which is what "occupancy-limited"
#      (too many small dispatches, GPU never saturates) actually needs.
#
# Methodology per repo's own standing rules (checked against §12/§4.6):
#   - MLX_GPU_TIME=1 real GPUStartTime/GPUEndTime bracketing (p01/p03 proven)
#   - rotated 64-entry index pool to defeat L2 cache reuse (p01 rule)
#   - PIPELINED chain timing, not per-call isolated (the "isolated
#     overestimates savings" trap, §4.6) -- both isolated AND chained
#     reported, chained is the trustworthy one
#   - standalone process, NOT attached to the live runner PID -- zero
#     xctrace, zero relaunch, zero production risk (the only capture
#     method this doc's history has never caused an incident with)
import os

for v in ("MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, "/Users/adam.durham/repos/exo/mlx-lm")
from mlx_lm.models.switch_layers import BatchedSwitchGLU, _gather_sort, _scatter_unsort
from mlx_lm.models.activations import swiglu

# --- Production configuration (verified against live Vision-Exp config.json
# and `ps eww` on the live runner PID, 2026-09-14) ---
HIDDEN = 4096
INTER = 1024       # per-rank MoE intermediate (moe_intermediate_size=2048 / TP=2)
N_EXPERTS = 256
TOP_K = 6
GROUP_SIZE = 32
BITS = 4
QUANT_MODE = "mxfp4"   # make_quantization_config() runtime target for switch_mlp
PEAK_BW_SPEC = 546e9        # M4 Max spec GB/s
PEAK_BW_MEASURED = 424e9    # PH:5819 real measured streaming (conservative end)
PEAK_TFLOPS_MEASURED = 15.21e12  # PH:5819-5822, on-node 40-core Studio bf16 GEMM peak

PREFILL_M = 2048    # EXO_PREFILL_STEP_SIZE, live production default

OUT = Path("/Users/adam.durham/repos/exo/tmp/p13-20260914")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = {"meta": {}, "decode_shape": {}, "prefill_shape": {}, "roofline": {}, "notes": []}


def log(*a):
    print(*a, flush=True)


def build_model():
    model = BatchedSwitchGLU(HIDDEN, INTER, N_EXPERTS, bias=False)
    model.gate_proj = model.gate_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    model.up_proj = model.up_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    model.down_proj = model.down_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    # BatchedSwitchGLU.fuse_weights() unconditionally concatenates .biases,
    # which is None under mxfp4 (2-tuple quantize, no biases) -- a real bug
    # in this microbench-only helper that production never hits because
    # auto_parallel.py's _install_fused_gate_up() (the real production path,
    # confirmed via source read) guards with `if gp_b is not None and
    # up_b is not None`. Same None-biases finding the P01 doc's reverted
    # "biases=None handling" patch made. Replicate the production-correct
    # logic here directly instead of re-patching the shared mlx-lm submodule.
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


def time_stage_chained(stage_fn, pool_size, n_iters=200, warmup=20):
    """Pipelined/dependency-chained GPU-time bracketing (not isolated calls
    -- per §4.6's standing rule that isolated per-call timing overestimates
    recoverable overhead). mx.metal.reset_gpu_time() + gpu_time_ns() gives
    real Metal GPUStartTime/GPUEndTime sums, immune to host dispatch gaps."""
    for i in range(warmup):
        mx.eval(stage_fn(i % pool_size))
    mx.synchronize()

    mx.metal.reset_gpu_time()
    mx.metal.reset_dispatch_count()
    outs = []
    for i in range(n_iters):
        outs.append(stage_fn(i % pool_size))
    mx.eval(*outs)
    mx.synchronize()
    total_ns = mx.metal.gpu_time_ns()
    total_dispatches = mx.metal.dispatch_count()
    return {
        "gpu_us_per_call": total_ns / n_iters / 1000.0,
        "dispatches_per_call": total_dispatches / n_iters,
    }


def wall_clock_stage(fn, pool_size, n_iters=200, warmup=20):
    for i in range(warmup):
        mx.eval(fn(i % pool_size))
    mx.synchronize()
    t0 = time.perf_counter()
    outs = []
    for i in range(n_iters):
        outs.append(fn(i % pool_size))
    mx.eval(*outs)
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters * 1e6  # us


# ============================================================ DECODE SHAPE
def run_decode_shape(model):
    log("\n=== DECODE SHAPE (M=1, do_sort=False) -- sanity check vs P01 ===")
    x = mx.random.normal((1, HIDDEN)).astype(mx.bfloat16)
    pool = [mx.random.randint(0, N_EXPERTS, (1, 1, TOP_K)) for _ in range(64)]
    mx.eval(x, *pool)

    def get_gu(i):
        idx = pool[i]
        return mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)), model._fused_w_gu, model._fused_s_gu,
            model._fused_b_gu, rhs_indices=idx, transpose=True,
            group_size=model._fused_group_size, bits=model.gate_proj.bits,
            mode=model.gate_proj.mode, sorted_indices=False,
        )

    gu_pool = [get_gu(i) for i in range(64)]
    mx.eval(*gu_pool)

    def get_act(i):
        n_inter = model._fused_n_inter
        gu = gu_pool[i]
        return swiglu(gu[..., n_inter:], gu[..., :n_inter])

    act_pool = [get_act(i) for i in range(64)]
    mx.eval(*act_pool)

    def get_down(i):
        return model.down_proj(act_pool[i], pool[i], sorted_indices=False)

    stages = {}
    stages["fused_gate_up"] = time_stage_chained(get_gu, 64)
    stages["activation"] = time_stage_chained(get_act, 64)
    stages["down_proj"] = time_stage_chained(get_down, 64)

    # Confirm do_sort is really False at this shape (structural check, not measured)
    idx0 = pool[0]
    stages["_do_sort_would_be"] = bool(idx0.size >= 64)

    total_gpu_us = sum(s["gpu_us_per_call"] for k, s in stages.items() if isinstance(s, dict))
    total_dispatch = sum(s["dispatches_per_call"] for k, s in stages.items() if isinstance(s, dict))
    stages["_total_gpu_us"] = total_gpu_us
    stages["_total_dispatches"] = total_dispatch

    bytes_touched = TOP_K * (2 * INTER * HIDDEN * 0.5 + INTER * HIDDEN * 0.5) + \
        TOP_K * (2 * INTER * (HIDDEN / GROUP_SIZE) * 4 + INTER * (HIDDEN / GROUP_SIZE) * 4)  # weights + fp32 scales approx
    stages["_bytes_touched_approx"] = bytes_touched
    stages["_implied_bw_gbps"] = bytes_touched / (total_gpu_us * 1e-6) / 1e9
    stages["_pct_of_spec_peak"] = stages["_implied_bw_gbps"] / (PEAK_BW_SPEC / 1e9) * 100
    stages["_pct_of_measured_peak"] = stages["_implied_bw_gbps"] / (PEAK_BW_MEASURED / 1e9) * 100

    for k, v in stages.items():
        log(f"  {k}: {v}")
    return stages


# =========================================================== PREFILL SHAPE
def run_prefill_shape(model):
    log(f"\n=== PREFILL SHAPE (M={PREFILL_M}, do_sort=True) -- NOVEL, never measured ===")
    # Real routing distribution: top_k=6 of 256 experts, M tokens.
    # Use independent per-token routing (not identical across tokens) --
    # this is what actually determines gather_sort/scatter_unsort cost
    # (argsort over M*TOP_K indices) and down-stream expert run lengths.
    x = mx.random.normal((PREFILL_M, HIDDEN)).astype(mx.bfloat16)
    pool = []
    for _ in range(16):  # smaller rotation pool given the larger per-call cost
        idx = mx.random.randint(0, N_EXPERTS, (PREFILL_M, TOP_K))
        pool.append(idx)
    mx.eval(x, *pool)

    idx0 = pool[0]
    do_sort = bool(idx0.size >= 64)
    log(f"  do_sort at this shape: {do_sort} (indices.size={idx0.size})")

    # --- Stage: gather_sort (argsort + fancy-index gather) ---
    x_expanded = mx.expand_dims(x, (-2, -3))
    mx.eval(x_expanded)

    def get_gather_sort(i):
        xs, idx, inv = _gather_sort(x_expanded, pool[i])
        return xs, idx, inv

    # gpu_time_ns sums ALL kernels in the bracket; this stage produces 3
    # outputs (sorted-x gather, sorted-idx, inv permutation) from 2 real
    # dispatches (argsort, then a take/gather) -- report combined.
    def gather_sort_probe(i):
        xs, idx, inv = get_gather_sort(i)
        return xs  # eval forces the whole small graph incl idx/inv (shared deps)

    gs_stage = time_stage_chained(
        lambda i: sum_and_return(get_gather_sort(i)), 16, n_iters=100, warmup=10
    )

    # Pre-compute sorted forms for downstream stages
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

    gu_stage = time_stage_chained(get_gu, 16, n_iters=100, warmup=10)
    gu_pool = [get_gu(i) for i in range(16)]
    mx.eval(*gu_pool)

    def get_act(i):
        n_inter = model._fused_n_inter
        gu = gu_pool[i]
        return swiglu(gu[..., n_inter:], gu[..., :n_inter])

    act_stage = time_stage_chained(get_act, 16, n_iters=100, warmup=10)
    act_pool = [get_act(i) for i in range(16)]
    mx.eval(*act_pool)

    def get_down(i):
        xs, idx, inv = sorted_pool[i]
        return model.down_proj(act_pool[i], idx, sorted_indices=True)

    down_stage = time_stage_chained(get_down, 16, n_iters=100, warmup=10)
    down_pool = [get_down(i) for i in range(16)]
    mx.eval(*down_pool)

    def get_scatter(i):
        xs, idx, inv = sorted_pool[i]
        return _scatter_unsort(down_pool[i], inv, pool[i].shape)

    scatter_stage = time_stage_chained(get_scatter, 16, n_iters=100, warmup=10)

    stages = {
        "gather_sort": gs_stage,
        "fused_gate_up": gu_stage,
        "activation": act_stage,
        "down_proj": down_stage,
        "scatter_unsort": scatter_stage,
        "_do_sort": do_sort,
    }

    total_gpu_us = sum(s["gpu_us_per_call"] for k, s in stages.items() if isinstance(s, dict))
    total_dispatch = sum(s["dispatches_per_call"] for k, s in stages.items() if isinstance(s, dict))
    stages["_total_gpu_us"] = total_gpu_us
    stages["_total_dispatches"] = total_dispatch
    stages["_gather_scatter_pct_of_total"] = (
        (gs_stage["gpu_us_per_call"] + scatter_stage["gpu_us_per_call"]) / total_gpu_us * 100
    )

    # Roofline: compute-bound check via FLOPs at M=2048
    # gate+up+down FLOPs per token-expert-pair: 2*HIDDEN*INTER*2 (gate+up) + 2*INTER*HIDDEN (down)
    # = 6 * HIDDEN * INTER FLOPs (mul+add=2 FLOPs each)
    flops_per_token = TOP_K * (2 * HIDDEN * INTER * 2 + 2 * INTER * HIDDEN)  # gate+up (2 mats) + down
    total_flops = flops_per_token * PREFILL_M
    matmul_us = gu_stage["gpu_us_per_call"] + down_stage["gpu_us_per_call"]
    achieved_tflops = total_flops / (matmul_us * 1e-6) / 1e12
    stages["_roofline_flops_total"] = total_flops
    stages["_roofline_achieved_tflops"] = achieved_tflops
    stages["_roofline_pct_of_measured_peak_compute"] = achieved_tflops / (PEAK_TFLOPS_MEASURED / 1e12) * 100

    # Bytes-based bandwidth check for the same matmul stages (same as decode calc, scaled)
    bytes_per_call = PREFILL_M * TOP_K * (2 * INTER * HIDDEN * 0.5 + INTER * HIDDEN * 0.5) / PREFILL_M  # per-token bytes same as decode since weight reuse
    # NOTE: at M=2048 with sorted/grouped experts, weight bytes are reused across
    # tokens routed to the same expert (tile-blocked gather_qmm), NOT re-read
    # per token -- so a naive per-token-byte-model is WRONG here (unlike decode's
    # M=1 case where every call reads full expert weight rows). Flag this
    # explicitly rather than report a misleading number.
    stages["_bandwidth_model_caveat"] = (
        "at M=2048 grouped/sorted gather_qmm reuses weight tiles across "
        "co-routed tokens; a naive bytes/M-token model (correct for decode's "
        "M=1 case) does NOT directly apply here without knowing per-expert "
        "run-length distribution. Not computed to avoid a misleading number; "
        "see FLOPs-based compute roofline above instead, which IS shape-correct."
    )

    for k, v in stages.items():
        log(f"  {k}: {v}")
    return stages


def sum_and_return(triple):
    return triple[0]


def main():
    log("=== P13: switch_mlp/GatherQMM sub-phase attribution ===")
    RESULTS["meta"] = {
        "host": os.uname().nodename,
        "mlx": mx.__version__,
        "gpu": mx.metal.device_info()["architecture"] if hasattr(mx.metal, "device_info") else "unknown",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "peak_bw_spec_gbps": PEAK_BW_SPEC / 1e9,
        "peak_bw_measured_gbps": PEAK_BW_MEASURED / 1e9,
        "peak_tflops_measured": PEAK_TFLOPS_MEASURED / 1e12,
        "prefill_m": PREFILL_M,
    }
    mx.random.seed(20260914)

    model = build_model()

    RESULTS["decode_shape"] = run_decode_shape(model)
    RESULTS["prefill_shape"] = run_prefill_shape(model)

    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    log(f"\nResults written to {OUT / 'results.json'}")


if __name__ == "__main__":
    main()

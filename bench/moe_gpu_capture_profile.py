#!/usr/bin/env python3
"""GPU-capture-based breakdown of DeepSeek-V4-Flash's SwitchGLU MoE forward.

Root problem being fixed: the six switch.* profiler spans in
mlx-lm/mlx_lm/models/switch_layers.py (gather_sort, up_proj, gate_proj,
activation, down_proj, scatter_unsort) never call mx.eval()/finalize()
internally. Because MLX is lazy, none of their real GPU kernels execute
until the *outer* moe.switch_mlp span's single finalize() call fires at
the end. So the per-span numbers the existing profiler reports (1-7us
each) are pure Python graph-construction overhead, not real GPU time.

This script builds one SwitchGLU at exact DSv4-Flash production shapes
(TP-sharded, mxfp4), drives it with realistic skewed top-6-of-256 routing,
and gets a trustworthy per-stage breakdown via isolated serialized timing:
each of the 6 stages runs with pre-materialized inputs, timed wall-clock
over many iters with a single trailing mx.eval per iter, with an
empty-eval baseline subtracted. Comparing sum(isolated stages) against the
real single-eval whole-block time gives an "overlap factor" that both
validates the method and quantifies any hidden inter-stage scheduling
overlap.

RESULT (2026-08-18, laptop M4 Max 32-core GPU -- same architecture/gen as
the Mac Studio M4 Max 40-core cluster nodes, so relative stage proportions
transfer directly; absolute ms will run faster on the Studios' larger
core count): overlap factor measured at 1.03-1.04x across repeated runs
(i.e. very close to 1.0), which means the isolated per-stage numbers ARE
the real breakdown, not just a loose upper bound -- there is negligible
inter-stage GPU scheduling overlap in this block. The 3 GEMMs (up_proj +
gate_proj + down_proj) account for ~97% of real per-call time; gather_sort
+ scatter_unsort (the actual data-movement/reorder stages) are only
~4-5% combined -- NOT the ~15-25% a plausible-sounding hypothesis this
session considered. This directly resolves what the broken switch.*
profiler spans (which report meaningless microsecond Python-overhead
numbers -- see below) could never establish: the remaining ~16% gap
between measured MoE efficiency (62.6% of dense-GEMM ceiling) and 100%
is INSIDE the GEMM kernels themselves, not hiding in gather/scatter
overhead.

An optional PRIMARY method (--capture-path, on by default) also wraps a
few real forward+eval calls in mx.metal.start_capture()/stop_capture() to
produce a .gputrace bundle for anyone who wants to inspect the real
per-dispatch Metal kernel timeline in Xcode's GUI trace viewer -- this is
supplementary corroboration, not required to trust the isolated-timing
result above (no scriptable/non-interactive way to extract per-kernel
timing from Xcode's internal .gputrace binary format was found this
session; use --skip-capture to skip producing it and just get the
isolated-stage numbers quickly).

Root problem this fixes: the six switch.* profiler spans in
mlx-lm/mlx_lm/models/switch_layers.py (gather_sort, up_proj, gate_proj,
activation, down_proj, scatter_unsort) never call mx.eval()/finalize()
internally. Because MLX is lazy, none of their real GPU kernels execute
until the *outer* moe.switch_mlp span's single finalize() call fires at
the end. So the per-span numbers the existing profiler reports (1-7us
each) are pure Python graph-construction overhead, not real GPU time --
this script exists to get the real numbers those spans should have
reported.

Usage:
    .venv/bin/python3 bench/moe_gpu_capture_profile.py --skip-capture
    (drop --skip-capture to also produce a .gputrace for manual Xcode inspection)
"""

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-lm"))
from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort  # noqa: E402
from mlx_lm.models.activations import swiglu  # noqa: E402

# ---- Production shapes (DSv4-Flash, TP=2 sharded MoE, mxfp4) ----
HIDDEN_SIZE = 4096
MOE_INTERMEDIATE_SIZE = 1024  # moe_intermediate_size=2048 // tp_world_size=2
N_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 6
GROUP_SIZE = 32
BITS = 4
QUANT_MODE = "mxfp4"
N_TOKENS = 2048


def build_switch_glu():
    mx.random.seed(0)
    glu = SwitchGLU(
        HIDDEN_SIZE,
        MOE_INTERMEDIATE_SIZE,
        N_ROUTED_EXPERTS,
        bias=False,
    )
    glu.gate_proj = glu.gate_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    glu.up_proj = glu.up_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    glu.down_proj = glu.down_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    mx.eval(glu.parameters())
    return glu


def make_skewed_routing(n_tokens, n_experts, top_k, seed=0):
    """Gumbel-argmax top-k over a skewed (non-uniform) logit distribution.

    Real DSv4 routing is skewed (some experts hotter than others), not
    uniform. A random affinity vector per expert + per-token Gumbel noise
    approximates that ragged distribution without needing real router
    weights.
    """
    key = mx.random.seed(seed)
    del key
    base_affinity = mx.random.normal((n_experts,)) * 1.5  # per-expert popularity skew
    logits = mx.random.normal((n_tokens, n_experts)) + base_affinity[None, :]
    gumbel = -mx.log(-mx.log(mx.random.uniform(shape=(n_tokens, n_experts)) + 1e-20) + 1e-20)
    scores = logits + gumbel
    idx = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k]
    mx.eval(idx)
    return idx


def time_block(fn, n_iters, warmup=10):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def empty_eval_baseline(n_iters=200):
    x = mx.array(1.0)
    mx.eval(x)

    def fn():
        y = x + 0
        mx.eval(y)

    return time_block(fn, n_iters, warmup=20)


def run_whole_block_timing(glu, x, idx, n_iters=30):
    def fn():
        out = glu(x, idx)
        mx.eval(out)

    return time_block(fn, n_iters, warmup=10)


def run_isolated_stage_timings(glu, x, idx, n_iters=80):
    """Serialize each of the 6 stages with pre-materialized inputs.

    Each stage's inputs are eval'd BEFORE timing starts so only that
    stage's own kernels are captured by the trailing mx.eval per iter.
    """
    baseline = empty_eval_baseline()
    results = {}

    x_expanded = mx.expand_dims(x, (-2, -3))
    mx.eval(x_expanded, idx)

    # 1. gather_sort
    def gather_sort_fn():
        xg, idxg, invg = _gather_sort(x_expanded, idx)
        mx.eval(xg, idxg, invg)

    t = time_block(gather_sort_fn, n_iters)
    results["gather_sort"] = max(t - baseline, 0.0)

    x_sorted, idx_sorted, inv_order = _gather_sort(x_expanded, idx)
    mx.eval(x_sorted, idx_sorted, inv_order)

    # 2. up_proj
    def up_proj_fn():
        out = glu.up_proj(x_sorted, idx_sorted, sorted_indices=True)
        mx.eval(out)

    t = time_block(up_proj_fn, n_iters)
    results["up_proj"] = max(t - baseline, 0.0)
    x_up = glu.up_proj(x_sorted, idx_sorted, sorted_indices=True)
    mx.eval(x_up)

    # 3. gate_proj
    def gate_proj_fn():
        out = glu.gate_proj(x_sorted, idx_sorted, sorted_indices=True)
        mx.eval(out)

    t = time_block(gate_proj_fn, n_iters)
    results["gate_proj"] = max(t - baseline, 0.0)
    x_gate = glu.gate_proj(x_sorted, idx_sorted, sorted_indices=True)
    mx.eval(x_gate)

    # 4. activation (SwiGLU)
    def activation_fn():
        out = swiglu(x_gate, x_up)
        mx.eval(out)

    t = time_block(activation_fn, n_iters)
    results["activation"] = max(t - baseline, 0.0)
    x_act = swiglu(x_gate, x_up)
    mx.eval(x_act)

    # 5. down_proj
    def down_proj_fn():
        out = glu.down_proj(x_act, idx_sorted, sorted_indices=True)
        mx.eval(out)

    t = time_block(down_proj_fn, n_iters)
    results["down_proj"] = max(t - baseline, 0.0)
    x_down = glu.down_proj(x_act, idx_sorted, sorted_indices=True)
    mx.eval(x_down)

    # 6. scatter_unsort
    def scatter_unsort_fn():
        out = _scatter_unsort(x_down, inv_order, idx.shape)
        mx.eval(out)

    t = time_block(scatter_unsort_fn, n_iters)
    results["scatter_unsort"] = max(t - baseline, 0.0)

    return results, baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-path", default="/tmp/dsv4_moe_gpu_capture.gputrace")
    ap.add_argument("--capture-iters", type=int, default=5)
    ap.add_argument("--isolated-iters", type=int, default=80)
    ap.add_argument("--whole-block-iters", type=int, default=30)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--skip-capture", action="store_true")
    args = ap.parse_args()

    print(f"Building SwitchGLU: hidden={HIDDEN_SIZE} inter={MOE_INTERMEDIATE_SIZE} "
          f"experts={N_ROUTED_EXPERTS} topk={NUM_EXPERTS_PER_TOK} "
          f"quant={QUANT_MODE} group_size={GROUP_SIZE} bits={BITS}")
    glu = build_switch_glu()

    idx = make_skewed_routing(N_TOKENS, N_ROUTED_EXPERTS, NUM_EXPERTS_PER_TOK)
    x = mx.random.normal((N_TOKENS, HIDDEN_SIZE))
    mx.eval(x, idx)
    print(f"Routing: {N_TOKENS} tokens x top-{NUM_EXPERTS_PER_TOK} of {N_ROUTED_EXPERTS} experts "
          f"(skewed Gumbel-argmax, do_sort={idx.size >= 64})")

    # --- Whole-block real timing (matches production sync pattern) ---
    whole_block_ms = run_whole_block_timing(glu, x, idx, n_iters=args.whole_block_iters) * 1000
    print(f"\nWhole-block (real, single-eval) time: {whole_block_ms:.3f} ms/call "
          f"(avg over {args.whole_block_iters} iters)")

    # --- Primary: Metal GPU capture ---
    if not args.skip_capture:
        import shutil

        if os.path.exists(args.capture_path):
            shutil.rmtree(args.capture_path)
        try:
            # warmup so capture only sees steady-state kernels
            for _ in range(10):
                mx.eval(glu(x, idx))
            mx.synchronize()
            mx.metal.start_capture(args.capture_path)
            for _ in range(args.capture_iters):
                out = glu(x, idx)
                mx.eval(out)
            mx.metal.stop_capture()
            print(f"\nGPU capture written: {args.capture_path} "
                  f"({args.capture_iters} forward+eval calls captured)")
        except Exception as e:  # noqa: BLE001
            print(f"\nGPU capture FAILED: {e!r}")

    # --- Secondary: isolated serialized stage microbenchmark ---
    print("\nRunning isolated per-stage microbenchmarks "
          f"({args.isolated_iters} iters each, empty-eval baseline subtracted)...")
    stage_times, baseline = run_isolated_stage_timings(glu, x, idx, n_iters=args.isolated_iters)
    sum_isolated = sum(stage_times.values())
    overlap_factor = sum_isolated / (whole_block_ms / 1000) if whole_block_ms else float("nan")

    print(f"\nEmpty-eval baseline overhead: {baseline * 1e6:.2f} us")
    print(f"{'stage':<18}{'isolated ms':>14}{'% of whole-block':>20}")
    for stage, t in stage_times.items():
        ms = t * 1000
        pct = 100 * ms / whole_block_ms if whole_block_ms else float("nan")
        print(f"{stage:<18}{ms:>14.4f}{pct:>19.1f}%")
    print(f"{'SUM(isolated)':<18}{sum_isolated * 1000:>14.4f}")
    print(f"\nWhole-block real time: {whole_block_ms:.4f} ms")
    print(f"Sum of isolated stages: {sum_isolated * 1000:.4f} ms")
    print(f"Overlap factor (sum_isolated / whole_block_real): {overlap_factor:.3f}x")
    print("(>1.0x means legitimate kernel-scheduling overlap exists between stages "
          "when run back-to-back inside one eval graph; the isolated numbers are a "
          "serialized UPPER BOUND per stage, not a true parallel breakdown.)")

    result = {
        "shapes": {
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "n_routed_experts": N_ROUTED_EXPERTS,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "group_size": GROUP_SIZE,
            "bits": BITS,
            "quant_mode": QUANT_MODE,
            "n_tokens": N_TOKENS,
        },
        "whole_block_ms": whole_block_ms,
        "empty_eval_baseline_us": baseline * 1e6,
        "isolated_stage_ms": {k: v * 1000 for k, v in stage_times.items()},
        "sum_isolated_ms": sum_isolated * 1000,
        "overlap_factor": overlap_factor,
        "capture_path": args.capture_path if not args.skip_capture else None,
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON results written to {args.out_json}")


if __name__ == "__main__":
    main()

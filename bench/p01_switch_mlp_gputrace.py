import os
import time
import argparse
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

# Ensure we can import from mlx-lm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-lm"))
from mlx_lm.models.switch_layers import BatchedSwitchGLU
from mlx_lm.models.activations import swiglu

# --- Production Configuration ---
HIDDEN = 4096
INTER = 1024  # per-rank intermediate (2048 // 2)
N_EXPERTS = 256
TOP_K = 6
GROUP_SIZE = 32
BITS = 4
QUANT_MODE = "mxfp4"
PEAK_BW = 546e9  # 546 GB/s
BYTES_TOUCHED = 47.186 * 1024 * 1024  # 47.186 MB

def build_model():
    # Reproduce exactly the DSv4-Flash sharded MoE layer
    model = BatchedSwitchGLU(HIDDEN, INTER, N_EXPERTS, bias=False)
    
    # Quantize all projections
    model.gate_proj = model.gate_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    model.up_proj = model.up_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    model.down_proj = model.down_proj.to_quantized(GROUP_SIZE, BITS, mode=QUANT_MODE)
    
    # Fuse gate and up
    model.fuse_weights()
    mx.eval(model.parameters())
    return model

def get_data():
    # B=1, L=1 decode shape
    x = mx.random.normal((1, HIDDEN)).astype(mx.bfloat16)
    # Rotate indices to defeat cache reuse (64-entry pool)
    indices_pool = [mx.random.randint(0, N_EXPERTS, (1, 1, TOP_K)) for _ in range(64)]
    mx.eval(x, *indices_pool)
    return x, indices_pool

def run_independent(model, x, pool, n_iters=300):
    """Independent calls: maximum overlap, minimal host-sync."""
    outs = []
    for i in range(n_iters):
        idx = pool[i % 64]
        outs.append(model(x, idx))
    
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(*outs)
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters

def run_chained(model, x, pool, n_iters=300):
    """Dependency-chained: reproduces production per-token serialization."""
    # We create a chain where each token depends on the previous result to prevent
    # the compiler from merging or overlapping the calls too aggressively.
    # For a simple bench, we can just use the output of i to influence i+1.
    
    # Warmup
    for i in range(10):
        mx.eval(model(x, pool[i % 64]))
    mx.synchronize()
    
    t0 = time.perf_counter()
    
    # To create a real chain without changing weights:
    # we can use a simple dependency on the previous result's sum
    curr = x
    nodes = []
    for i in range(n_iters):
        idx = pool[i % 64]
        out = model(curr, idx)
        nodes.append(out)
        # Use the sum of the previous output to perturb the input slightly
        # this creates a hard data dependency without causing shape explosion
        curr = x + mx.mean(out) * 0.0 
        
    mx.eval(*nodes)
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters

def run_capture(model, x, pool, path, n_calls=5):
    print(f"Starting capture to {path}...")
    # Warmup
    for i in range(10):
        mx.eval(model(x, pool[i % 64]))
    mx.synchronize()
    
    mx.metal.start_capture(path)
    for i in range(n_calls):
        idx = pool[i % 64]
        out = model(x, idx)
        mx.eval(out)
    mx.metal.stop_capture()
    print("Capture complete.")

def run_attribution(model, x, pool, n_iters=50):
    """Bracketed GPU timing for per-stage attribution with warmup and averaging."""
    if not hasattr(mx.metal, "gpu_time_ns"):
        print("mx.metal.gpu_time_ns not available. Ensure MLX_GPU_TIME=1 is set.")
        return None

    # Pre-calculate rotated pools to ensure DRAM-real measurements for all stages
    print("Pre-calculating rotated pools for DRAM-real attribution...")
    gu_pool = []
    x_act_pool = []
    
    # We need a fixed function to generate the inputs for the pools
    def get_gu(idx):
        return mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)),
            model._fused_w_gu,
            model._fused_s_gu,
            model._fused_b_gu,
            rhs_indices=idx,
            transpose=True,
            group_size=model._fused_group_size,
            bits=model.gate_proj.bits,
            mode=model.gate_proj.mode,
            sorted_indices=False,
        )

    def get_act(gu):
        n_inter = model._fused_n_inter
        x_gate = gu[..., :n_inter]
        x_up = gu[..., n_inter:]
        return swiglu(x_up, x_gate)

    for i in range(64):
        idx = pool[i]
        gu = get_gu(idx)
        act = get_act(gu)
        gu_pool.append(gu)
        x_act_pool.append(act)
    
    mx.eval(*gu_pool, *x_act_pool)
    mx.synchronize()

    def time_stage(name, stage_fn):
        # Warmup
        for i in range(10):
            res = stage_fn(i % 64)
            mx.eval(res)
        mx.synchronize()
        
        mx.metal.reset_gpu_time()
        for i in range(n_iters):
            res = stage_fn(i % 64)
            mx.eval(res)
        mx.synchronize()
        total_ns = mx.metal.gpu_time_ns()
        return total_ns / n_iters / 1000.0  # average us

    # Stage 1: Fused Gate+Up
    def gate_up_fn(i):
        return get_gu(pool[i])
    
    t_gu = time_stage("fused_gate_up", gate_up_fn)
    
    # Stage 2: Activation
    def act_fn(i):
        return get_act(gu_pool[i])
    
    t_act = time_stage("activation", act_fn)
    
    # Stage 3: Down
    def down_fn(i):
        return model.down_proj(x_act_pool[i], pool[i], sorted_indices=False)
    
    t_down = time_stage("down_proj", down_fn)
    
    return {
        "fused_gate_up": t_gu,
        "activation": t_act,
        "down_proj": t_down,
        "total": t_gu + t_act + t_down
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bench", "capture", "attribute", "all"], default="all")
    parser.add_argument("--capture-path", type=str, required=True)
    args = parser.parse_args()
    
    model = build_model()
    x, pool = get_data()
    
    if args.mode in ["bench", "all"]:
        print("\n--- Timing ---")
        t_ind = run_independent(model, x, pool)
        print(f"Independent: {t_ind*1e6:.2f} us | BW: {(BYTES_TOUCHED/t_ind)/1e9:.2f} GB/s | { (BYTES_TOUCHED/t_ind)/PEAK_BW*100:.2f}% peak")
        
        t_chain = run_chained(model, x, pool)
        print(f"Chained:     {t_chain*1e6:.2f} us | BW: {(BYTES_TOUCHED/t_chain)/1e9:.2f} GB/s | { (BYTES_TOUCHED/t_chain)/PEAK_BW*100:.2f}% peak")
        
    if args.mode in ["capture", "all"]:
        run_capture(model, x, pool, args.capture_path)
        
    if args.mode in ["attribute", "all"]:
        print("\n--- Per-Stage Attribution (via gpu_time_ns) ---")
        attr = run_attribution(model, x, pool)
        if attr:
            print(json.dumps(attr, indent=2))

if __name__ == "__main__":
    main()

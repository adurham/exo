# P01b — Multi-layer pipelining loss measurement for DeepSeek-V4-Flash.
# Tests the hypothesis that inter-layer dependency chains in production
# decode lead to on-GPU busy growth that is invisible to single-layer
# census microbenches.

# Target: m4-2 (rank 0)

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# ── production env config (start_cluster.sh defaults) ──
_PROD_ENV = {
    "EXO_DSV4_INDEX_TOPK": "512",
    "EXO_KV_CACHE_BITS": "0",
    "EXO_COMPUTE_DTYPE": "bf16",
    "EXO_DSV4_SPARSE_SDPA_TILE": "128",
    "EXO_DSV4_SEQ_SPLIT": "1",
    "EXO_DSV4_EXACT_TOPK": "1",
    "EXO_DSV4_TOPK_FUSED": "0",
    "EXO_DSV4_SPARSE_FUSED_SDPA": "0",
    "EXO_DSV4_ATTN_ALLSUM": "0",
    "EXO_DSV4_SINGLE_GATHER": "1",
    "EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES": "8388608",
}
for _k, _v in _PROD_ENV.items():
    os.environ[_k] = _v
for _k in ("EXO_DSV4_INDEXER_PBLOCK", "EXO_DSV4_QA_KV_FUSED", "EXO_DSV4_FP32_ACT",
           "EXO_DSV4_MTP", "EXO_PROFILER", "EXO_DSV4_SECTION_TIME"):
    os.environ.pop(_k, None)

import mlx.core as mx
import mlx.nn as nn

_REPO = Path(__file__).resolve().parent.parent
_MLXLM = _REPO / "mlx-lm"
if not _MLXLM.is_dir():
    _MLXLM = Path.home() / "repos" / "exo" / "mlx-lm"
sys.path.insert(0, str(_MLXLM))

from mlx_lm.models import deepseek_v4 as dv4
from mlx_lm.models.cache import (
    CacheList, PoolingCache, RotatingKVCache,
)

# ── production config ──
CFG = dict(
    model_type="deepseek_v4",
    vocab_size=129280,
    hidden_size=4096,
    intermediate_size=18432,
    moe_intermediate_size=2048,
    num_hidden_layers=43,
    num_attention_heads=64,
    num_key_value_heads=1,
    n_shared_experts=1,
    n_routed_experts=256,
    num_experts_per_tok=6,
    head_dim=512,
    index_head_dim=128,
    index_n_heads=64,
    index_topk=512,
    o_groups=8,
    o_lora_rank=1024,
    q_lora_rank=1024,
    qk_rope_head_dim=64,
    sliding_window=128,
    max_position_embeddings=1048576,
    rms_norm_eps=1e-6,
    rope_theta=10000,
    compress_rope_theta=160000,
    rope_scaling=dict(beta_fast=32, beta_slow=1, factor=16,
                      original_max_position_embeddings=65536, type="yarn"),
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    topk_method="noaux_tc",
    norm_topk_prob=True,
    attention_bias=False,
    compress_ratios=[0,0] + [4,128]*20 + [4],
    num_nextn_predict_layers=1,
    hidden_act="silu",
    swiglu_limit=10.0,
    tie_word_embeddings=False,
)

DTYPE = mx.bfloat16
HEAD_DIM = 512
INDEX_DIM = 128
SW = 128

def _quant_predicate(path, module):
    if not hasattr(module, "to_quantized"):
        return False
    full = "model.layers.0.attn." + path if path else "model.layers.0.attn"
    if ".attn.w" in full or ".attn.indexer.wq" in full:
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    return {"group_size": 64, "bits": 8, "mode": "affine"}

def build_attn(ratio):
    args = dv4.ModelArgs.from_dict(CFG)
    # Use a representative layer index for the ratio
    idx = 0 if ratio == 0 else (2 if ratio == 4 else 3)
    attn = dv4.v4_attention_factory(args, idx)
    attn.set_dtype(DTYPE)
    nn.quantize(attn, class_predicate=_quant_predicate)
    mx.eval(attn.parameters())
    return attn

def _fill_rotating(rc: RotatingKVCache, L: int, B=1, D=HEAD_DIM):
    rc.keys = mx.random.normal((B, 1, rc.max_size, D)).astype(DTYPE)
    rc.values = mx.zeros((B, 1, rc.max_size, 0), dtype=DTYPE)
    rc.offset = L
    rc._idx = rc.max_size
    mx.eval(rc.keys, rc.values)

def _fill_pool(pc: PoolingCache, L: int, dim: int, B=1):
    P = L // pc.ratio
    alloc = max(pc.step, ((P + 1 + pc.step - 1) // pc.step) * pc.step)
    pc._pool_storage = mx.random.normal((B, alloc, dim)).astype(DTYPE)
    pc._pool_offset = P
    pc._pending_offset_bump = 0
    rem = L % pc.ratio
    out_dim = dim * (2 if pc.ratio == 4 else 1)
    if rem:
        pc.buf_kv = mx.random.normal((B, pc.ratio, out_dim)).astype(DTYPE)
        pc.buf_gate = mx.random.normal((B, pc.ratio, out_dim)).astype(DTYPE)
        pc.remainder = rem
    if pc.ratio == 4:
        half = out_dim // 2
        pc._overlap_kv_carry = mx.random.normal((B, 1, pc.ratio, half)).astype(DTYPE)
        pc._overlap_gate_carry = mx.random.normal((B, 1, pc.ratio, half)).astype(DTYPE)
    mx.eval(pc._pool_storage)
    if pc.buf_kv is not None: mx.eval(pc.buf_kv, pc.buf_gate)
    if pc._overlap_kv_carry is not None: mx.eval(pc._overlap_kv_carry, pc._overlap_gate_carry)
    return P

def make_cache(ratio, L):
    rc = RotatingKVCache(max_size=SW)
    _fill_rotating(rc, L)
    if ratio == 0: return rc, dict(P_comp=0, P_idx=0)
    comp = PoolingCache(ratio)
    P_comp = _fill_pool(comp, L, HEAD_DIM)
    if ratio == 4:
        idx = PoolingCache(ratio)
        P_idx = _fill_pool(idx, L, INDEX_DIM)
        return CacheList(rc, comp, idx), dict(P_comp=P_comp, P_idx=P_idx)
    return CacheList(rc, comp), dict(P_comp=P_comp, P_idx=0)

def get_telemetry():
    return {
        "active_mem": mx.get_active_memory(),
        "cache_mem": mx.get_cache_memory(),
        "peak_mem": mx.get_peak_memory(),
    }

def time_decode_steps(layers, caches, steps, warmup, x_init, mode="async"):
    # layers is a list of attention modules
    # caches is a list of corresponding caches
    B, Lq, H = 1, 1, CFG["hidden_size"]
    x = x_init
    
    # Warmup
    for _ in range(warmup):
        curr = x
        for i in range(len(layers)):
            curr = layers[i](curr, mask=None, cache=caches[i])
        mx.eval(curr)
    mx.synchronize()

    samples = []
    for _ in range(steps):
        mx.synchronize()
        t0 = time.perf_counter()
        
        curr = x
        for i in range(len(layers)):
            curr = layers[i](curr, mask=None, cache=caches[i])
            if mode == "async":
                mx.async_eval(curr)
        
        mx.eval(curr)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="500,100026,352599")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="p01b_results.json")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    
    # We use a representative chain: 4 layers [4, 128, 4, 128]
    chain_ratios = [4, 128, 4, 128]
    B, Lq, H = 1, 1, CFG["hidden_size"]
    x_init = mx.random.normal((B, Lq, H)).astype(DTYPE)
    mx.eval(x_init)

    results = {"multi": {}, "single": {}}

    for L in depths:
        print(f"\\n=== Depth L = {L:,} ===")
        
        # 1. Multi-layer measurement
        multi_depth_samples = []
        for r in range(args.repeats):
            # Fresh build for each repeat to avoid caching artifacts
            layers = [build_attn(ratio) for ratio in chain_ratios]
            caches = [make_cache(ratio, L)[0] for ratio in chain_ratios]
            
            mem_before = get_telemetry()
            samples = time_decode_steps(layers, caches, args.steps, args.warmup, x_init)
            mem_after = get_telemetry()
            
            multi_depth_samples.append(statistics.median(samples))
            mx.clear_cache()
            
        results["multi"][L] = {
            "median": statistics.median(multi_depth_samples),
            "spread": statistics.stdev(multi_depth_samples) if len(multi_depth_samples) > 1 else 0,
            "mem_before": get_telemetry(),
            "mem_after": get_telemetry(),
        }

        # 2. Single-layer baseline
        # We measure one layer of each ratio, and sum them scaled by the chain count
        # Chain: 2x r4, 2x r128
        single_ratios = {4: 2, 128: 2}
        total_single_samples = []
        
        # We perform the single-layer measurement as: sum(count * median_of_single_layer)
        # and repeat the whole block 3 times.
        for r in range(args.repeats):
            step_sum = 0
            for ratio, count in single_ratios.items():
                layer = build_attn(ratio)
                cache, _ = make_cache(ratio, L)
                
                # 256 steps of single layer
                samples = []
                for _ in range(args.steps):
                    mx.synchronize()
                    t0 = time.perf_counter()
                    out = layer(x_init, mask=None, cache=cache)
                    mx.async_eval(out)
                    mx.synchronize()
                    samples.append((time.perf_counter() - t0) * 1e3)
                
                step_sum += count * statistics.median(samples)
                mx.clear_cache()
            
            total_single_samples.append(step_sum)

        results["single"][L] = {
            "median": statistics.median(total_single_samples),
            "spread": statistics.stdev(total_single_samples) if len(total_single_samples) > 1 else 0,
        }
        
        print(f"  Multi:  {results['multi'][L]['median']:.4f} ms (±{results['multi'][L]['spread']:.4f})")
        print(f"  Single: {results['single'][L]['median']:.4f} ms (±{results['single'][L]['spread']:.4f})")
        print(f"  Loss:   {results['multi'][L]['median'] - results['single'][L]['median']:.4f} ms")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\nResults written to {args.out}")

if __name__ == "__main__":
    main()

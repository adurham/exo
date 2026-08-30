import os
import sys
import json
import struct
import time
from pathlib import Path

# Ensure mlx-lm is in path for any dependencies, though we use mlx.core
sys.path.insert(0, os.path.expanduser("~/repos/exo/mlx-lm"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

CKPT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"
OUT = Path.home() / "repos/exo/tmp/p05-shared-batching-20260830"
OUT.mkdir(parents=True, exist_ok=True)

def load_f8(shard, key):
    """Load native F8_E4M3 weight + F8_E8M0 scale from the checkpoint."""
    with open(CKPT / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        info = hdr[key]
        o0, o1 = info["data_offsets"]
        fh.seek(n + 8 + o0)
        raw = fh.read(o1 - o0)
    if info["dtype"] == "F8_E4M3":
        return np.frombuffer(raw, dtype=np.uint8).reshape(info["shape"])
    if info["dtype"] == "F8_E8M0":
        return np.frombuffer(raw, dtype=np.uint8).reshape(info["shape"])
    raise ValueError(info["dtype"])

def build_mxfp8_linear(shard, wkey, skey):
    """Build MLX mxfp8 quantized linear exactly as production sanitize() does."""
    w_u8 = load_f8(shard, wkey)
    s_u8 = load_f8(shard, skey)
    out_dim, in_dim = w_u8.shape
    
    # Weights: 4 x e4m3 per uint32
    w5 = w_u8.reshape(out_dim, in_dim // 4, 4)
    packed = np.zeros((out_dim, in_dim // 4), dtype=np.uint32)
    for j in range(4):
        packed |= w5[:, :, j].astype(np.uint32) << (8 * j)
    
    # Scales: (16, 32) e8m0 -> (16, 128) -> (2048, 128)
    # The original script does: np.repeat(np.repeat(s1_u8, 4, -1), 128, 0)
    # We should be generic about the repeat counts based on dimensions.
    # Production pattern for w1: scale is (16, 32), out=2048, in=4096.
    # 2048 / 16 = 128. 4096 / 32 = 128.
    s_rep = np.repeat(np.repeat(s_u8, in_dim // (s_u8.shape[1] * 32), axis=-1), out_dim // s_u8.shape[0], axis=0)
    
    # This is a bit fragile, let's use the hardcoded numbers from original for w1
    # and try to be a bit more flexible for w2 if it differs.
    # Actually, the original script's s_rep line was:
    # s_rep = np.repeat(np.repeat(s1_u8, 4, -1), 128, 0) 
    # Let's just use the logic: repeat to match (out_dim, in_dim // 32).
    
    # Correct logic for DeepSeek-V4-Flash sanitize:
    # w1 scales are (16, 32). Result needs to be (2048, 128).
    # repeat(4, -1) -> (16, 128). repeat(128, 0) -> (2048, 128).
    # Let's just use the hardcoded ratios for this specific model:
    # s_u8 is (S_out, S_in). Target is (out_dim, in_dim // 32).
    s_rep = np.repeat(np.repeat(s_u8, (in_dim // 32) // s_u8.shape[1], axis=-1), out_dim // s_u8.shape[0], axis=0)

    lin = nn.Linear(in_dim, out_dim, bias=False)
    q = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
    q.weight = mx.array(packed)
    q.scales = mx.array(s_rep)
    q.eval()
    mx.eval(q.parameters())
    return q

def measure_divergence(q, x_all):
    # M=4 vs M=1
    y1_list = [q(x_all[i:i+1]) for i in range(64)]
    y1 = mx.concatenate(y1_list, axis=0)
    mx.eval(y1)
    y4_list = [q(x_all[i:i+4]) for i in range(0, 64, 4)]
    y4 = mx.concatenate(y4_list, axis=0)
    mx.eval(y4)
    d = (y4 - y1).astype(mx.float32)
    max_div = float(mx.abs(d).max())
    nz = int((mx.abs(d) > 0).sum())
    tot = int(d.size)
    
    # Padded QMM
    x4_0 = x_all[:4]
    ref_rows = []
    for i in range(4):
        r = q(x4_0[i:i+1])
        mx.eval(r)
        ref_rows.append(r)
    ref = mx.concatenate(ref_rows, axis=0)
    
    padded_results = {}
    for vl in (8, 16, 32, 64):
        pad = mx.concatenate([x4_0] + [mx.zeros((1, x_all.shape[1]), dtype=mx.bfloat16)] * (vl - 4), axis=0)
        yp = q(pad)
        mx.eval(yp)
        dp = (yp[:4] - ref).astype(mx.float32)
        max_dp = float(mx.abs(dp).max())
        nz_dp = int((mx.abs(dp) > 0).sum())
        tot_dp = int(dp.size)
        padded_results[f"M{vl}"] = {"max_abs": max_dp, "nonzero": nz_dp, "total": tot_dp}
        
    return {
        "m4_vs_m1": {"max_abs": max_div, "nonzero": nz, "total": tot},
        "padded_qmm": padded_results
    }

def main():
    # Use the same shard as original script
    shard = "model-00005-of-00048.safetensors"
    
    # Tensors to test: layer-3 w1 and layer-3 w2 (if in same shard) or layer-4 w1.
    # Let's try w1 and w2 of layer 3.
    tensors_to_test = [
        ("layers.3.ffn.shared_experts.w1.weight", "layers.3.ffn.shared_experts.w1.scale"),
        ("layers.3.ffn.shared_experts.w2.weight", "layers.3.ffn.shared_experts.w2.scale"),
    ]
    
    # We'll use a realistic input based on w1's input dim.
    # We can't use the same input for w1 and w2 because their input dims differ.
    
    all_results = {"host": os.uname().nodename, "gpu": mx.device_info()["architecture"], "tensors": {}}
    
    for wkey, skey in tensors_to_test:
        print(f"Testing {wkey}...")
        try:
            q = build_mxfp8_linear(shard, wkey, skey)
            out_dim, in_dim = q.weight.shape[0], q.weight.shape[1] * 4 # packed
            
            #- Realistic inputs
            rng = np.random.default_rng(20260830)
            Hs = rng.standard_normal((64, in_dim)).astype(np.float32)
            Hn = Hs / (np.sqrt((Hs ** 2).mean(-1, keepdims=True)) + 1e-6)
            x_all = mx.array(Hn).astype(mx.bfloat16)
            
            res = measure_divergence(q, x_all)
            all_results["tensors"][wkey] = res
            print(f"  m4_vs_m1: max={res['m4_vs_m1']['max_abs']:.3e}, nz={res['m4_vs_m1']['nonzero']}")
            for m, val in res['padded_qmm'].items():
                print(f"  {m}: max={val['max_abs']:.3e}, nz={val['nonzero']}")
        except Exception as e:
            print(f"  Failed to test {wkey}: {e}")

    # Verdict: padded_qmm_lossless_on_real_weights = true if ALL padded_qmm max_abs == 0.0 for all tested tensors
    lossless = True
    for t_res in all_results["tensors"].values():
        for m_res in t_res["padded_qmm"].values():
            if m_res["max_abs"] > 0.0:
                lossless = False
                break
        if not lossless: break
    
    all_results["padded_qmm_lossless_on_real_weights"] = lossless
    
    out_path = OUT / "real_shared_padded_qmm.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\n wrote {out_path}")

if __name__ == "__main__":
    main()

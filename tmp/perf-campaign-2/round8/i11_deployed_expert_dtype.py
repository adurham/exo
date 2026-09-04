#!/usr/bin/env python3
"""Verify what precision the DEPLOYED routed experts actually load at.

Loads ONLY two small tensors from the resident checkpoint (no model
build, no cluster interaction) and reports the dtype MLX sees, which is
what mlx-lm's `_is_mxfp_override` gate keys on
(mlx-lm/mlx_lm/utils.py: keeps the mxfp override only when the on-disk
scales are uint8).
"""
import glob
import json

import mlx.core as mx

D = "/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"
wm = json.load(open(D + "/model.safetensors.index.json"))["weight_map"]

for key in [
    "layers.3.ffn.experts.0.w1.weight",
    "layers.3.ffn.experts.0.w1.scale",
    "layers.3.ffn.experts.0.w2.weight",
    "layers.3.ffn.experts.0.w2.scale",
]:
    shard = wm[key]
    arrs = mx.load(f"{D}/{shard}")
    a = arrs[key]
    print(f"{key}: mlx dtype={a.dtype} shape={tuple(a.shape)} nbytes={a.nbytes:,}")
    del arrs, a
    mx.clear_cache()

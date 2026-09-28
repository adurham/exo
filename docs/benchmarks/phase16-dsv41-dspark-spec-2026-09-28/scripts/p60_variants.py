#!/usr/bin/env python3
"""p59 -- isolate the live acceptance collapse (single node).

(1) incremental ctx append (as the live loop does) vs bulk append, world=1,
    anchors from short context upward;
(2) draft MoE TP split: rank0 + rank1 partials (identity all_sum) vs full.
"""
import json, os, sys
import numpy as np
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.hyper_connections import hc_pre
from mlx_lm.models.deepseek_v41.layers import RMSNorm
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_linear

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
REF = HOME + "/p30-records-clamp"
ck = Exl3Checkpoint(MODEL)
args = ModelArgs.from_dict(ck.config)
ids = json.load(open(HOME + "/p30_prompt_ids.json"))[:531]
n = len(ids)
emb = nn.Embedding(args.vocab_size, args.dim)
emb.load_weights([("weight", mx.array(ck.np("embed.weight")).astype(mx.bfloat16))])
head_lin = eb.Exl3Proj(load_dense_linear(ck, "head"))
nrm = RMSNorm(args.dim, args.norm_eps)
nrm.load_weights([("weight", mx.array(ck.np("norm.weight")).astype(mx.float32))])
h39 = mx.load(f"{REF}/h_39.npy"); pm39 = mx.load(f"{REF}/pm_39.npy")
body_arg = np.array(mx.argmax(head_lin(nrm(hc_pre(h39, pm39)).astype(mx.float16))[0], axis=-1))
del h39, pm39
T = mx.concatenate([mx.load(f"{REF}/h_{L-1:02d}.npy").mean(axis=2)
                    for L in args.dspark_target_layer_ids], axis=-1)
mx.eval(T)
head = eb.build_mtp(ck, args)


def d1_rate(anchors, incremental):
    hits = []
    if incremental:
        dsc = head.make_cache(1)
        filled = 0
    for p in anchors:
        if incremental:
            while filled < p:
                head.append_ctx(T[:, filled:filled + 1], dsc)
                filled += 1
        else:
            dsc = head.make_cache(1)
            head.append_ctx(T[:, :p], dsc)
        d, _ = head.draft(mx.array([ids[p]]), emb, head_lin, dsc, width=4)
        hits.append(int(np.array(d)[0, 0] == body_arg[p]))
    return np.mean(hits) * 100



from mlx_lm.models.deepseek_v41 import mtp as MT, fakequant as FQ
anch = list(range(8, 60, 2)) + list(range(140, 200, 3))
T0 = T
print(f"baseline (input taps fp32, fq on)   {d1_rate(anch, True):5.1f}%", flush=True)
T = T0.astype(mx.bfloat16)
print(f"taps bf16                           {d1_rate(anch, True):5.1f}%", flush=True)
T = T0
MT.fake_quant_fp8_ue8m0 = lambda x, b=32: x
print(f"no draft-KV fake-quant              {d1_rate(anch, True):5.1f}%", flush=True)
T = mx.concatenate([mx.load(f"{REF}/h_{L:02d}.npy").mean(axis=2) for L in args.dspark_target_layer_ids], axis=-1)
print(f"output taps + no fake-quant (=live) {d1_rate(anch, True):5.1f}%", flush=True)

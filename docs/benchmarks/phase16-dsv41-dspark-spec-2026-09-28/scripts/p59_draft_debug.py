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


short = list(range(8, 60, 2))
longa = list(range(140, 260, 3))
print(f"bulk        short-ctx d1==body {d1_rate(short, False):5.1f}%   long-ctx {d1_rate(longa, False):5.1f}%", flush=True)
print(f"incremental short-ctx d1==body {d1_rate(short, True):5.1f}%   long-ctx {d1_rate(longa, True):5.1f}%", flush=True)


class FakeGroup:
    def rank(self): return 0
    def size(self): return 2


_real = mx.distributed.all_sum
mx.distributed.all_sum = lambda x, group=None, **k: x if isinstance(group, FakeGroup) else _real(x, group=group, **k)
fg = FakeGroup()
r0 = eb.build_mtp(ck, args, rank=0, world=2, group=fg)
r1 = eb.build_mtp(ck, args, rank=1, world=2, group=fg)
x = (mx.random.normal((1, 5, args.dim)) * 0.5).astype(mx.bfloat16)
for s in range(3):
    yf = head.stages[s].ffn(x).astype(mx.float32)
    y0 = r0.stages[s].ffn(x).astype(mx.float32)
    y1 = r1.stages[s].ffn(x).astype(mx.float32)
    shared = head.stages[s].ffn.shared_experts(x.reshape(-1, args.dim)).astype(mx.float32).reshape(yf.shape)
    ysum = y0 + y1 - shared                   # each rank added the replicated shared expert once
    c = ((ysum * yf).sum() / (mx.linalg.norm(ysum) * mx.linalg.norm(yf))).item()
    print(f"draft stage {s} MoE TP split cos={c:.7f}", flush=True)

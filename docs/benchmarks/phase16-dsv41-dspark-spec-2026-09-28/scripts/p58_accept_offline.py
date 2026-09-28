#!/usr/bin/env python3
"""p58 -- offline draft-acceptance check from the saved p30 40-layer trace.

Uses the stored per-layer stream (p30-records-clamp/h_XX, pm_XX) for the
531-token prompt. The body argmax at every position comes from h_39 -> head.
For anchors p, the draft window is filled with taps for positions < p, the
draft runs from anchor token p, and its tokens are compared with
  (a) the body's own argmax at p (the verify target for d1),
  (b) the real next prompt tokens (teacher-forced prefix).
Variants: TAP=input (layer input, reference) vs TAP=output (old port).
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
body_logits = head_lin(nrm(hc_pre(h39, pm39)).astype(mx.float16))
body_arg = np.array(mx.argmax(body_logits[0], axis=-1))
del h39, pm39
tap_ids = list(args.dspark_target_layer_ids)
print("tap layers", tap_ids, flush=True)

head = eb.build_mtp(ck, args)
print(f"mtp built active={mx.get_active_memory()/1e9:.1f}GB", flush=True)


def taps(kind):
    idx = [L - 1 for L in tap_ids] if kind == "input" else tap_ids
    return mx.concatenate([mx.load(f"{REF}/h_{L:02d}.npy").mean(axis=2) for L in idx], axis=-1)


anchors = list(range(140, n - 6, 3))
for kind in ("input", "output"):
    T = taps(kind)
    mx.eval(T)
    a_body, a_real = [], []
    for p in anchors:
        dsc = head.make_cache(1)
        head.append_ctx(T[:, :p], dsc)          # ctx positions 0..p-1 (ring keeps last 128)
        d, _ = head.draft(mx.array([ids[p]]), emb, head_lin, dsc, width=5)
        d = np.array(d)[0]
        a_body.append(int(d[0] == body_arg[p]))
        k = 0
        while k < 5 and p + 1 + k < n and d[k] == ids[p + 1 + k]:
            k += 1
        a_real.append(k)
    body_vs_real = np.mean([body_arg[p] == ids[p + 1] for p in anchors])
    print(f"TAP={kind:6s} anchors={len(anchors)}  d1==body_argmax {np.mean(a_body)*100:5.1f}%  "
          f"prefix vs real text mean {np.mean(a_real):.2f} (d1 hit {np.mean([x > 0 for x in a_real])*100:.1f}%)  "
          f"[body top-1 vs real text {body_vs_real*100:.1f}%]", flush=True)

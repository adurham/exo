#!/usr/bin/env python3
"""p62 -- where the 13 ms draft round goes (single node, rank-0 half-width experts)."""
import os, sys, time
import numpy as np
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.deepseek_v41 import exl3_build as eb, mtp as MT, moe as MO
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_linear

ck = Exl3Checkpoint(HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
args = ModelArgs.from_dict(ck.config)
head = eb.build_mtp(ck, args, rank=0, world=2, group=None)
emb = nn.Embedding(args.vocab_size, args.dim)
emb.load_weights([("weight", mx.array(ck.np("embed.weight")).astype(mx.bfloat16))])
from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.exl3.loader import load_dense_layer
lm = eb.Exl3Proj(load_dense_linear(ck, "head"))
dsc = head.make_cache(1)
head.append_ctx((mx.random.normal((1, 100, 3 * args.dim)) * 0.1).astype(mx.bfloat16), dsc)
W = int(os.environ.get("W", "3"))


def t(label):
    for _ in range(3):
        d, _ = head.draft(mx.array([1000]), emb, lm, dsc, width=W); mx.eval(d)
    ts = []
    for _ in range(15):
        s = time.perf_counter(); d, _ = head.draft(mx.array([1000]), emb, lm, dsc, width=W); mx.eval(d)
        ts.append(time.perf_counter() - s)
    print(f"{label:28s} {np.median(ts)*1e3:6.2f} ms", flush=True)


t("full draft")
_orig_proj = type(lm).__call__
def _proj(self, x):
    if self is lm:
        return mx.broadcast_to(x[..., :1] * 0, x.shape[:-1] + (self._lin.out_features,)).astype(x.dtype)
    return _orig_proj(self, x)
type(lm).__call__ = _proj
t("no lm head")
type(lm).__call__ = _orig_proj
_mh = head.markov_head
head.markov_head = lambda e: mx.broadcast_to(e[..., :1] * 0, e.shape[:-1] + (args.vocab_size,))
t("no markov head")
head.markov_head = _mh
_ffn = MT.DraftMoE.__call__
MT.DraftMoE.__call__ = lambda self, x: x * 0
t("no draft MoE")
MT.DraftMoE.__call__ = _ffn
_ex = eb.Exl3Experts.__call__
eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(x[:, None, :] * 0, idx.shape + (x.shape[-1],))
t("no routed experts")
eb.Exl3Experts.__call__ = _ex
_at = MT.DraftAttention.draft_block
MT.DraftAttention.draft_block = lambda self, x, c: x
t("no draft attention")
MT.DraftAttention.draft_block = _at

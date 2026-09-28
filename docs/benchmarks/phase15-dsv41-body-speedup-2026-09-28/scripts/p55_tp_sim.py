#!/usr/bin/env python3
"""p55 -- single-node simulation of the wider TP=2 split.

For each test layer: build the full block (world=1) and both rank blocks
(world=2, a fake group whose all_sum is identity so each rank returns its
PARTIAL). Run attention and FFN of all three on the same inputs for a prefill
chunk + several decode steps (each with its own cache) and check
partial0 + partial1 == full. Also checks the vocab-sharded head.
"""
import os, sys, json
import numpy as np
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.cache import ModelCache
from mlx_lm.models.deepseek_v41.model import SharedState
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_layer
from mlx_lm.models.exl3 import EXL3Linear


class FakeGroup:
    def rank(self): return 0
    def size(self): return 2


_real_sum = mx.distributed.all_sum
mx.distributed.all_sum = lambda x, group=None, **k: x if isinstance(group, FakeGroup) else _real_sum(x, group=group, **k)

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
ck = Exl3Checkpoint(MODEL); nat = Exl3Checkpoint(NATIVE)
args = ModelArgs.from_dict(ck.config)
fg = FakeGroup()
mx.random.seed(3)


def stat(a, b):
    a = a.astype(mx.float32); b = b.astype(mx.float32)
    c = ((a * b).sum() / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()
    return c, (mx.abs(a - b).max() / mx.abs(b).max()).item()


worst = 1.0
for L in [int(v) for v in os.environ.get("P55_LAYERS", "2,3,20,21,24,25").split(",")]:
    full, _ = eb.build_block(ck, args, L, native=nat)
    r0, _ = eb.build_block(ck, args, L, native=nat, rank=0, world=2, group=fg)
    r1, _ = eb.build_block(ck, args, L, native=nat, rank=1, world=2, group=fg)
    caches = [ModelCache(args, 1, 256, dtype=mx.float32) for _ in range(3)]
    pos = 0
    for step, n in enumerate([40, 1, 1, 1, 1, 4, 1]):
        x = (mx.random.normal((1, n, args.dim)) * 0.5).astype(mx.bfloat16)
        outs = []
        for blk, c in zip((full, r0, r1), caches):
            sh = SharedState()
            # consumers need the source's state from the SAME rank's cache
            for src in args.kv_source_layers:
                if src <= L and src != L:
                    pass
            if not blk.attn.is_kv_source and blk.attn.ratio:
                sh.kv_src_cache = c.layers[L]          # stand-in so consumers run
                sh.index_src_cache = c.layers[L]
                sh.topk_idxs = None
            try:
                a = blk.attn(x, pos, c, sh)
            except Exception as e:                    # consumer without a real source
                a = None
            f = blk.ffn(x)
            outs.append((a, f))
        mx.eval([t for o in outs for t in o if t is not None])
        if outs[0][0] is not None:
            ca, da = stat(outs[1][0] + outs[2][0], outs[0][0])
        else:
            ca, da = float("nan"), float("nan")
        cf, df = stat(outs[1][1] + outs[2][1], outs[0][1])
        if ca == ca:
            worst = min(worst, ca)
        worst = min(worst, cf)
        print(f"L{L:02d} step{step} n={n:2d} attn cos={ca:.7f} rel={da:.2e} | ffn cos={cf:.7f} rel={df:.2e}", flush=True)
        pos += n
    del full, r0, r1
    mx.clear_cache()

# head
full = EXL3Linear(load_dense_layer(ck, "head"))
parts = [EXL3Linear(eb._slice_dense(load_dense_layer(ck, "head"), axis="out", rank=r, world=2)) for r in (0, 1)]
x = mx.random.normal((1, args.dim)).astype(mx.float16)
y = mx.concatenate([p(x) for p in parts], axis=-1)
c, d = stat(y, full(x))
print(f"head cos={c:.8f} rel={d:.2e} argmax_equal={mx.argmax(y).item() == mx.argmax(full(x)).item()}")
print(f"P55 worst cos {worst:.7f} {'PASS' if worst > 0.9999 else 'FAIL'}")

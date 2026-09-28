#!/usr/bin/env python3
"""p46 -- deepseek_v41 model-file gate against the p30 clamped trace.

Runs the NEW mlx_lm.models.deepseek_v41 package layer-major (build one block
via exl3_build.build_block, run, free) over the same 531-token prompt p30 used,
and compares the hc stream after every layer to ~/p30-records-clamp/h_XX.npy.
p30 used reconstructed-bf16 dense weights in nn.Linear; this path runs the
EXL3Linear kernels, so exact equality is not expected -- the plan's bar is
cosine >= 0.999 per layer. Then: head via EXL3Linear -> teacher-forced NLL,
compared to p35 (mean 1.003, top-1 78.3%).
Env: P46_LAYERS (default 0-39), P46_WORLD (1), P46_RANK (0).
"""
import json, os, sys, time
import numpy as np
HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState
from mlx_lm.models.deepseek_v41.cache import ModelCache
from mlx_lm.models.deepseek_v41.engram import EngramHasher
from mlx_lm.models.deepseek_v41.hyper_connections import make_identity_pre_mix, hc_pre
from mlx_lm.models.deepseek_v41.layers import RMSNorm
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_dense_linear

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
REF = HOME + "/p30-records-clamp"
ck = Exl3Checkpoint(MODEL); nat = Exl3Checkpoint(NATIVE)
args = ModelArgs.from_dict(ck.config)
ISOLATE = os.environ.get("P46_ISOLATE", "0") == "1"
world = int(os.environ.get("P46_WORLD", "1")); rank = int(os.environ.get("P46_RANK", "0"))
spec = os.environ.get("P46_LAYERS", "0-39")
a, b = spec.split("-"); layers = list(range(int(a), int(b) + 1))
ids = json.load(open(HOME + "/p30_prompt_ids.json"))[:531]
n = len(ids)
emb = mx.array(ck.np("embed.weight"))              # fp32, same as p30
h = mx.broadcast_to(emb[mx.array([ids])][:, :, None, :], (1, n, args.hc_mult, args.dim))
del emb
cache = ModelCache(args, 1, n + 64, dtype=mx.float32)
pre_mix = make_identity_pre_mix(1, n, args.hc_mult)
shared = SharedState()
hasher = EngramHasher(args, json.load(open(HOME + "/repos/ref/deepseek-v41-mlx/engram_token_map.json")))
hashes = mx.array(hasher(np.array([ids], dtype=np.int64), 0, cache.engram_ids))

def cos(x, y):
    x = x.astype(mx.float32).reshape(-1); y = y.astype(mx.float32).reshape(-1)
    return float((x * y).sum() / (mx.sqrt((x * x).sum()) * mx.sqrt((y * y).sum())))

worst = 1.0; t_all = time.time()
print(f"[p46] isolate={ISOLATE} layers={layers[0]}-{layers[-1]} world={world} rank={rank} n={n}", flush=True)
for L in layers:
    t0 = time.time()
    blk, rep = eb.build_block(ck, args, L, native=nat, rank=rank, world=world)
    t1 = time.time()
    if ISOLATE and L > 0:
        h = mx.load(f"{REF}/h_{L-1:02d}.npy"); pre_mix = mx.load(f"{REF}/pm_{L-1:02d}.npy")
    if blk.engram is not None:
        h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index])
    h, pre_mix = blk(h, pre_mix, 0, cache, shared)
    mx.eval(h, pre_mix)
    ref = mx.load(f"{REF}/h_{L:02d}.npy"); refpm = mx.load(f"{REF}/pm_{L:02d}.npy")
    c = cos(h, ref); cp = cos(pre_mix, refpm)
    rel = float(mx.max(mx.abs(h.astype(mx.float32) - ref.astype(mx.float32)))) / float(mx.max(mx.abs(ref.astype(mx.float32))))
    worst = min(worst, c)
    rms = float(mx.sqrt(mx.mean(mx.square(h.astype(mx.float32)))))
    print(f"[p46] L{L:02d} build {t1-t0:4.1f}s fwd {time.time()-t1:5.2f}s cos_h={c:.6f} cos_pm={cp:.6f} "
          f"maxrel={rel:.3e} rms={rms:.3f} plain={rep['plain']} dense={rep['dense_groups']} "
          f"opt_absent={len(rep['optional_absent'])} peak={mx.get_peak_memory()/1e9:.1f}GB", flush=True)
    del blk
    mx.clear_cache()
print(f"[p46] stack done {time.time()-t_all:.1f}s worst_cos={worst:.6f} "
      f"GATE_0999={'PASS' if worst >= 0.999 else 'FAIL'}", flush=True)

if layers[-1] == args.n_layers - 1:
    nrm = RMSNorm(args.dim, args.norm_eps)
    nrm.load_weights([("weight", mx.array(ck.np("norm.weight")).astype(mx.float32))])
    x = nrm(hc_pre(h, pre_mix))
    head = eb.Exl3Proj(load_dense_linear(ck, "head"))
    logits = head(x.astype(mx.float16)).astype(mx.float32)[0]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(ids[1:])
    nll = -mx.take_along_axis(lp[:-1], tgt[:, None], axis=-1)[:, 0]
    top1 = mx.argmax(logits[:-1], axis=-1) == tgt
    mx.eval(nll, top1)
    nl = np.array(nll)
    print(f"[p46] NLL mean={nl.mean():.4f} median={np.median(nl):.4f} top1={float(top1.astype(mx.float32).mean())*100:.1f}% "
          f"(p35 ref: mean 1.003, median 0.029, top1 78.3%)", flush=True)
print("P46_DONE", flush=True)

#!/usr/bin/env python3
"""p66 -- what scales with context (single node, <=8 GB).

Layers 2 (ratio-2 kv+index source) and 20 (ratio-1 kv+index+candidate
source), rank-0 half-width experts. Prefill to context C with the prefill
driver, then time: one more prefill chunk (512 and 128 rows) and 16 decode
steps. Repeat with one component stubbed. Prints peak memory every run.
Env: P66_CTX (default 2048,8192,16384), P66_STUB (none|idx|sattn|comp|experts|lin).
"""
import json, os, sys, time
import numpy as np
HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P66_PKG", HOME + "/dsv41-test"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb, attention as A, indexer as IX, compressor as CP
from mlx_lm.models.deepseek_v41 import prefill as PF

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("P66_LAYERS", "2,20").split(",")]
CTXS = [int(x) for x in os.environ.get("P66_CTX", "2048,8192,16384").split(",")]
STUB = os.environ.get("P66_STUB", "none")

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
if STUB == "idx":
    _orig = IX.Indexer.__call__
    def _cheap(self, x, qr, sp, off, c, s, ik, sh):
        nb = ik.shape[1]; k = min(self.index_topk, nb); n = x.shape[1]
        return mx.broadcast_to(mx.arange(k, dtype=mx.int32)[None, None] + off, (x.shape[0], n, k)) + (x[..., :1] * 0).astype(mx.int32)
    IX.Indexer.__call__ = _cheap
elif STUB == "sattn":
    A.sparse_attn = lambda q, kv, sink, idx, sc, *a, **k: q
elif STUB == "experts":
    eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(x[:, None, :] * 0, idx.shape + (x.shape[-1],))
print(f"[p66] layers={LAYERS} stub={STUB} active={mx.get_active_memory()/1e9:.1f}GB", flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))


def t_chunk(cache, rows, reps=3):
    ts = []
    for _ in range(reps):
        from mlx_lm.models.deepseek_v41 import spec as SP
        pos = cache.offset
        sn = SP.snap(cache, pos)
        s = time.perf_counter()
        out = model(mx.array([base[:rows]]), cache, last_logit_only=True, argmax=True)
        mx.eval(out)
        ts.append(time.perf_counter() - s)
        SP.rollback(cache, sn, pos, SP.stashes(cache))
    return float(np.median(ts)) * 1e3


for C in CTXS:
    ids = (base * (C // len(base) + 1))[:C]
    cache = model.make_cache(1, max_seq_len=C + 2048)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    s = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True)
    mx.eval(am)
    tp = time.perf_counter() - s
    c512 = t_chunk(cache, 512)
    c128 = t_chunk(cache, 128)
    c4 = t_chunk(cache, 4)
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    ds = []
    for i in range(20):
        s = time.perf_counter()
        nxt = model(nxt[None], cache, last_logit_only=True, argmax=True).reshape(-1)[-1:].astype(mx.int32)
        mx.eval(nxt)
        ds.append(time.perf_counter() - s)
    print(f"[p66] ctx={C:6d} prefill {tp:6.1f}s ({C/tp:6.1f} tok/s) | +512 rows {c512:7.1f} ms | +128 {c128:6.1f} ms | "
          f"R4 {c4:6.1f} ms | decode {np.median(ds[4:])*1e3:6.1f} ms (first {ds[0]*1e3:.0f}) | peak {mx.get_peak_memory()/1e9:.2f}GB", flush=True)
    del cache; mx.clear_cache()
print("P66_DONE", flush=True)

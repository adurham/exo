#!/usr/bin/env python3
"""p48 -- single-node decode micro-harness for the deepseek_v41 body.

Builds a layer SUBSET (rank-0 half-width experts, no all_sum) covering every
layer kind: 0,1 (window, engram@1), 2,3 (ratio-2 source+indexer / consumer),
20,21 (ratio-1 source+candidates / consumer), 24,25 (index source w/o keys /
consumer). Unfenced decode timing; P48_STUB ablates one component to get its
TRUE cost; P48_CHECK compares logits step-by-step vs a saved reference.
Env: P48_STUB=none|hc|shared|experts|attn|gate|head, P48_STEPS, P48_SAVE, P48_CHECK
"""
import json, os, sys, time
import numpy as np
HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P48_PKG", HOME + "/dsv41-test"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import model as M, moe as MO, attention as A

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("P48_LAYERS", "0,1,2,3,20,21,24,25").split(",")]
STUB = os.environ.get("P48_STUB", "none")
STEPS = int(os.environ.get("P48_STEPS", "40"))

CMP = set(filter(None, os.environ.get("P48_COMPILE", "").split(",")))
if CMP:
    from mlx_lm.models.deepseek_v41 import hyper_connections as HC, layers as LY, fakequant as FQ, indexer as IX
    C = mx.compile
    if "mix" in CMP:
        _mix = C(HC.hc_mixes)
        M.hc_mixes = lambda x, fn, sc, base, hc, it, ne, he: _mix(x, fn, sc, base, hc, it, ne, he)
    if "prepost" in CMP:
        M.hc_pre = C(HC.hc_pre); M.hc_post = C(HC.hc_post)
    if "rope" in CMP:
        rt = C(LY.rope_tail); A.rope_tail = rt; IX.rope_tail = rt
    if "fq" in CMP:
        for mod in (A, IX):
            for nm in ("fake_quant_fp8_ue8m0", "fake_quant_fp4_e4m3", "fake_quant_fp4_ue8m0"):
                if hasattr(mod, nm):
                    setattr(mod, nm, C(getattr(FQ, nm)))
    if "rms" in CMP:
        _rms = C(lambda x, w, eps: (w * (x.astype(mx.float32) * mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + eps))).astype(x.dtype))
        LY.RMSNorm.__call__ = lambda self, x: _rms(x, self.weight, self.eps)
    print(f"[p48] compile {sorted(CMP)}", flush=True)

ASYNC = int(os.environ.get("P48_ASYNC", "0"))
if ASYNC:
    _ob = M.Block.__call__
    def _ab(self, x, pre_mix, start_pos, cache, shared, _k=[0]):
        h, pm = _ob(self, x, pre_mix, start_pos, cache, shared)
        _k[0] += 1
        if _k[0] % ASYNC == 0:
            mx.async_eval(h, pm)
        return h, pm
    M.Block.__call__ = _ab
    print(f"[p48] async_eval every {ASYNC} blocks", flush=True)

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[p48] built layers={LAYERS} active={mx.get_active_memory()/1e9:.1f}GB stub={STUB}", flush=True)

if STUB == "hc":          # keep graph dependencies, drop the hc math
    M.hc_mixes = lambda x, *a: (mx.full(x.shape[:2] + (4,), 0.25), mx.full(x.shape[:2] + (4,), 0.5),
                                mx.full(x.shape[:2] + (4, 4), 0.25))
    M.hc_post = lambda h, r, p, c: r + h[:, :, None, :]
    M.hc_pre = lambda x, pm: x[:, :, 0]
elif STUB == "shared":
    MO.SharedExpert.__call__ = lambda self, x: x * 0
elif STUB == "experts":
    eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(x[:, None, :] * 0, idx.shape + (x.shape[-1],))
elif STUB == "attn":
    A.Attention.__call__ = lambda self, x, sp, c, s: x
elif STUB == "gate":
    MO.Gate.__call__ = lambda self, x: (x[:, :6] * 0 + 0.25, mx.broadcast_to(mx.arange(6), x.shape[:-1] + (6,)))
elif STUB == "head":
    model.head = lambda h: mx.broadcast_to(h[..., :1], h.shape[:-1] + (129280,))
elif STUB == "lin":
    from mlx_lm.models.exl3 import exl3_linear as EL
    EL.EXL3Linear.__call__ = lambda self, x: mx.broadcast_to(x[..., :1].astype(mx.float16) * 0, x.shape[:-1] + (self.out_features,))
elif STUB == "rope":
    from mlx_lm.models.deepseek_v41 import indexer as IX
    A.rope_tail = lambda x, rd, c, s, inverse=False: x; IX.rope_tail = A.rope_tail
elif STUB == "fq":
    from mlx_lm.models.deepseek_v41 import indexer as IX
    A.fake_quant_fp8_ue8m0 = lambda x, b=32: x; A.fake_quant_fp4_e4m3 = lambda x, b=16: x; IX.fake_quant_fp4_ue8m0 = lambda x, b=32: x
elif STUB == "sattn":
    A.sparse_attn = lambda q, kv, sink, idx, sc, chunk=256: q
elif STUB == "idx":
    from mlx_lm.models.deepseek_v41 import indexer as IX
    IX.Indexer.__call__ = lambda self, x, qr, sp, off, c, s, ik, sh: mx.zeros((x.shape[0], x.shape[1], min(self.index_topk, ik.shape[1])), mx.int32) + x[..., :1].astype(mx.int32) * 0
elif STUB == "rms":
    from mlx_lm.models.deepseek_v41 import layers as LY
    LY.RMSNorm.__call__ = lambda self, x: x
elif STUB == "engram":
    from mlx_lm.models.deepseek_v41 import engram as EN
    EN.Engram.__call__ = lambda self, x, hid: x
allids = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = allids[:64]
feed = allids[64:64 + STEPS]          # teacher-forced: identical inputs in every arm
cache = model.make_cache(1, max_seq_len=len(ids) + STEPS + 16)
logits = model(mx.array([ids]), cache, last_logit_only=True)
mx.eval(logits)
steps, builds, out, disp = [], [], [], []
for i in range(STEPS):
    s = time.perf_counter()
    logits = model(mx.array([[feed[i]]]), cache, last_logit_only=True)
    tok = mx.argmax(logits[:, -1], axis=-1)
    b = time.perf_counter()
    d0 = mx.metal.dispatch_count() if hasattr(mx.metal, "dispatch_count") else 0
    mx.eval(tok, logits)
    disp.append((mx.metal.dispatch_count() if hasattr(mx.metal, "dispatch_count") else 0) - d0)
    steps.append(time.perf_counter() - s); builds.append(b - s)
    out.append(logits[0, -1].astype(mx.float32))
if os.environ.get("P48_PROF") == "1":
    import cProfile, pstats, io
    pr = cProfile.Profile(); pr.enable()
    for i in range(10):
        lg = model(mx.array([[feed[i % len(feed)]]]), cache, last_logit_only=True); mx.eval(lg)
    pr.disable(); sio = io.StringIO()
    pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(28); print(sio.getvalue()[:6000])
if os.environ.get("P48_CENSUS") == "1":
    import collections
    from mlx_lm.models.deepseek_v41 import layers as LY, fakequant as FQ, indexer as IX, compressor as CP, sparse_attention as SA, engram as EN
    from mlx_lm.models.exl3 import exl3_linear as EL
    cnt = collections.Counter(); calls = collections.Counter()
    dc = mx.metal.dispatch_count
    def wrap(name, fn):
        def w(*a, **k):
            mx.eval([t for t in list(a) + list(k.values()) if isinstance(t, mx.array)])
            d0 = dc(); r = fn(*a, **k)
            outs = r if isinstance(r, (tuple, list)) else [r]
            mx.eval([t for t in outs if isinstance(t, mx.array)])
            cnt[name] += dc() - d0; calls[name] += 1
            return r
        return w
    EL.EXL3Linear.__call__ = wrap("EXL3Linear", EL.EXL3Linear.__call__)
    eb.Exl3Proj.__call__ = wrap("Exl3Proj(single)", eb.Exl3Proj.__call__)
    eb.Exl3Member.__call__ = wrap("Exl3Member(fused grp)", eb.Exl3Member.__call__)
    eb.Exl3GroupedStack.__call__ = wrap("wo_a stack", eb.Exl3GroupedStack.__call__)
    for nm in ("fake_quant_fp8_ue8m0", "fake_quant_fp4_e4m3"):
        setattr(A, nm, wrap(nm, getattr(FQ, nm)))
    IX.fake_quant_fp4_ue8m0 = wrap("fake_quant_fp4_ue8m0(idx)", FQ.fake_quant_fp4_ue8m0)
    A.rope_tail = wrap("rope_tail(attn)", LY.rope_tail); IX.rope_tail = wrap("rope_tail(idx)", LY.rope_tail)
    A.sparse_attn = wrap("sparse_attn", SA.sparse_attn)
    LY.RMSNorm.__call__ = wrap("RMSNorm", LY.RMSNorm.__call__)
    IX.Indexer.__call__ = wrap("Indexer(total)", IX.Indexer.__call__)
    IX.Indexer.publish_keys = wrap("Indexer.publish_keys(total)", IX.Indexer.publish_keys)
    CP.Compressor.__call__ = wrap("Compressor(total)", CP.Compressor.__call__)
    A.Attention.__call__ = wrap("Attention(total)", A.Attention.__call__)
    MO.Gate.__call__ = wrap("Gate", MO.Gate.__call__)
    MO.SharedExpert.__call__ = wrap("SharedExpert(total)", MO.SharedExpert.__call__)
    eb.Exl3Experts.__call__ = wrap("Exl3Experts", eb.Exl3Experts.__call__)
    MO.MoE.__call__ = wrap("MoE(total)", MO.MoE.__call__)
    EN.Engram.__call__ = wrap("Engram", EN.Engram.__call__)
    from mlx_lm.models.deepseek_v41 import hc_fused as HF
    M.mixes_and_collapse = wrap("hc mixes_and_collapse", HF.mixes_and_collapse)
    M.hc_expand = wrap("hc_expand", HF.hc_expand)
    NS = 5
    for i in range(NS):
        lg = model(mx.array([[feed[i]]]), cache, last_logit_only=True); mx.eval(lg)
    nl = len(LAYERS)
    print(f"[p48] DISPATCH CENSUS per layer ({nl} layers, {NS} steps):", flush=True)
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"   {k:32s} {v/NS/nl:7.1f} disp/layer  ({calls[k]/NS/nl:.1f} calls/layer, {v/max(calls[k],1):.1f}/call)", flush=True)
bt = np.array(builds[5:])
print(f"[p48] graph-build (python) median {np.median(bt)*1e3:.2f} ms/step  eval {np.median(np.array(steps[5:])-bt)*1e3:.2f} ms  dispatches/step {int(np.median(disp[5:]))}", flush=True)
st = np.array(steps[5:])
print(f"[p48] decode median {np.median(st)*1e3:.2f} ms/step (p10 {np.percentile(st,10)*1e3:.2f} "
      f"p90 {np.percentile(st,90)*1e3:.2f}) over {len(st)} steps, {len(LAYERS)} layers "
      f"=> {np.median(st)*1e3/len(LAYERS):.3f} ms/layer", flush=True)
if os.environ.get("P48_SAVE"):
    mx.save(os.environ["P48_SAVE"], mx.stack(out))
if os.environ.get("P48_CHECK"):
    ref = mx.load(os.environ["P48_CHECK"]); cur = mx.stack(out)
    am = (mx.argmax(ref, -1) == mx.argmax(cur, -1)).astype(mx.float32).mean().item()
    d = mx.abs(ref - cur).max().item(); c = ((ref*cur).sum(-1)/(mx.linalg.norm(ref,axis=-1)*mx.linalg.norm(cur,axis=-1))).min().item()
    print(f"[p48] CHECK argmax_agree={am*100:.1f}% max|d|={d:.4g} min_cos={c:.7f} exact={bool(mx.array_equal(ref,cur).item())}", flush=True)

# P07: first per-kernel GPU capture of the PREFILL-shape non-GEMM remainder ops
# (moe.post_combine / shared_experts / moe.gate / rmsnorm / rope / masks /
# kv_cache / indexer score-GEMM + pmask + topk). P03-mirroring harness
# (decode L=1..4) reshaped to the production prefill chunk: L=2048 total with
# EXO_DSV4_SEQ_SPLIT=1 TP=2 -> L_band=1024 query rows per rank.
# p01/P03 proven recipe: standalone process on macstudio-m4-1 beside the live
# runner, MLX_GPU_TIME=1 bracketing (fresh graph per timed call, mx.eval per
# call, median of >=2 passes), MLX_DISPATCH_COUNT=1, METAL_CAPTURE_ENABLED=1.
# Real checkpoint weights where cheap + synthetic same-dtype/shape rotation
# banks for L2 defeat. Denominator rule (pre-registered): arithmetic intensity
# < 10 FLOP/byte -> bandwidth ceiling (424 GB/s real streaming / 546 spec);
# >= 10 -> FLOPS ceiling (11.66 TFLOPS dense per node).
import os

for v in ("METAL_CAPTURE_ENABLED", "MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"
for v in ("MLX_GEMV_BATCH_INVARIANT", "MLX_STEEL_BATCH_INVARIANT",
          "EXO_DSV4_HC_COLLAPSE_KERNEL", "EXO_DSV4_HC_EXPAND_KERNEL"):
    assert os.environ.get(v) == "1", f"{v}=1 required (production parity)"
os.environ.setdefault("EXO_DSV4_INDEX_TOPK", "512")
os.environ.setdefault("EXO_DSV4_TAIL_PMASK", "1")
os.environ.setdefault("EXO_DSV4_EXACT_TOPK", "1")

import json
import math
import re
import struct
import sys
import time
import random
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, "/Users/adam.durham/repos/exo/mlx-lm")
from mlx_lm.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4RoPE, _gate_route, _hash_gate_route, _moe_post_combine,
    _limited_swiglu, _indexer_score, Indexer, ModelArgs,
)
from mlx_lm.models.cache import PoolingCache, RotatingKVCache  # noqa: E402

OUT = Path("/Users/adam.durham/repos/exo/tmp/p07-20260830")
CAP = OUT / "captures"
CAP.mkdir(parents=True, exist_ok=True)
RESULTS = {"meta": {}, "l2_sanity": {}, "ops": {}, "errors": []}
H, V, HC_MULT, N_EXPERTS, TOP_K = 4096, 129280, 4, 256, 6
INTER_RANK = 1024          # per-rank shared-expert intermediate (2048 // 2)
NL, HASH_LAYERS = 43, 3
L_BAND = 1024              # per-rank prefill query rows (2048 / TP=2, SPLIT=1)
CTX = 220000               # reference span-profile context depth
P = 55000                  # compressed pool length = ceil(220000 / 4)
IDX_HEADS, IDX_DIM, IDX_TOPK = 64, 128, 512
LOCAL_WINDOW = 128         # sliding_window for local-sparse layers
SPEC_GBPS, REAL_GBPS, TFLOPS = 546.0, 424.0, 11.66e12

def save():
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))

def log(*a):
    print(*a, flush=True)

log("=== P07 prefill-remainder capture ===")
RESULTS["meta"] = {
    "host": os.uname().nodename, "mlx": mx.__version__,
    "gpu": mx.device_info()["architecture"],
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "spec_gbps": SPEC_GBPS, "real_streaming_gbps": REAL_GBPS,
    "flops_ceiling": TFLOPS,
    "model": "deepseek-ai--DeepSeek-V4-Flash-0731 (TP rank prefill shapes)",
    "L_band": L_BAND, "L_chunk": 2048, "ctx": CTX, "P_pool": P,
}

# ---------------------------------------------------------------- weight bank
CKPT = "/Users/adam.durham/.exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"

def scan_headers():
    m, hl = {}, {}
    for f in sorted(Path(CKPT).glob("model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hl[str(f)] = n + 8
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                m[k] = (str(f), v["dtype"], tuple(v["shape"]), tuple(v["data_offsets"]))
    return m, hl

t0 = time.perf_counter()
HDRS, HDR_LEN = scan_headers()
log(f"header scan: {len(HDRS)} keys in {time.perf_counter()-t0:.1f}s")

def load_mlx(key):
    f, dt, sh, off = HDRS[key]
    if dt == "BF16":
        return load_bf16_range(key)
    from safetensors import safe_open
    with safe_open(f, framework="numpy") as fh:
        arr = fh.get_tensor(key)
    t = mx.array(arr)
    if sh and t.size == int(np.prod(sh)):
        return t.reshape(sh)
    return t

def load_mxfp8_qlinear(prefix, out_feats, in_feats):
    """Build an nn.QuantizedLinear for a checkpoint mxfp8 layer.

    VERIFIED live (2026-08-31, vs the actually-loaded production module):
      - weight: ckpt F8_E4M3 (out, in) bytes -> little-endian <u4 view
        (8192, 256), byte-identical to the loader's model.weight.
      - scales: ckpt F8_E8M0 (64 heads, 8 in-groups) -> np.repeat(128, axis=0)
        then np.repeat(4, axis=1) -> uint8 (8192, 32), byte-identical to
        model.scales (each 32-block out-feature row repeats 4x over the
        128-wide head-dim blocks it governs)."""
    fw, dw, shw, offw = HDRS[prefix + ".weight"]
    assert dw == "F8_E4M3", (prefix, dw, shw)
    fs, ds, shs, offs = HDRS[prefix + ".scale"]
    assert ds == "F8_E8M0", (prefix, ds, shs)
    with open(fw, "rb") as fh:
        fh.seek(HDR_LEN[fw] + offw[0])
        wraw = np.frombuffer(fh.read(offw[1] - offw[0]), dtype=np.uint8)
    with open(fs, "rb") as fh:
        fh.seek(HDR_LEN[fs] + offs[0])
        sraw = np.frombuffer(fh.read(offs[1] - offs[0]), dtype=np.uint8)
    n_heads, n_groups = shs  # (64, 8)
    per_head = out_feats // n_heads
    w_u32 = np.frombuffer(wraw.tobytes(), dtype="<u4").reshape(out_feats, in_feats // 4)
    s_u8 = np.repeat(sraw.reshape(n_heads, n_groups), per_head, axis=0)
    s_u8 = np.repeat(s_u8, 32 // n_groups, axis=1)  # (8192, 32)
    m = nn.QuantizedLinear(in_feats, out_feats, bias=False,
                           group_size=32, bits=8, mode="mxfp8")
    m.weight = mx.array(w_u32)
    m.scales = mx.array(s_u8)
    mx.eval(m.weight, m.scales)
    return m

def load_bf16_range(key):
    f, dt, sh, off = HDRS[key]
    assert dt == "BF16", (key, dt)
    with open(f, "rb") as fh:
        fh.seek(HDR_LEN[f] + off[0])
        raw = fh.read(off[1] - off[0])
    return mx.array(np.frombuffer(raw, dtype=np.uint16)).view(mx.bfloat16).reshape(sh)

rng = random.Random(20260831)
mx.random.seed(20260831)

# --- real set-0 weights + runtime assertion battery ---
gate0_w = load_mlx("layers.3.ffn.gate.weight")
gate0_b = load_mlx("layers.3.ffn.gate.bias")
norm0_w = load_mlx("layers.3.attn_norm.weight")
mx.eval(gate0_w, gate0_b, norm0_w)

got = {
    "gate.weight": (str(gate0_w.dtype), tuple(gate0_w.shape), gate0_w.nbytes),
    "gate.bias": (str(gate0_b.dtype), tuple(gate0_b.shape), gate0_b.nbytes),
    "norm.weight": (str(norm0_w.dtype), tuple(norm0_w.shape), norm0_w.nbytes),
}
exp = {
    "gate.weight": ("bfloat16", (N_EXPERTS, H), N_EXPERTS * H * 2),
    "gate.bias": ("float32", (N_EXPERTS,), N_EXPERTS * 4),
    "norm.weight": ("bfloat16", (H,), H * 2),
}
for k, e in exp.items():
    g = got[k]
    ok = e[0] in g[0] and g[1] == e[1] and g[2] == e[2]
    log(f"ASSERT {k}: {'OK' if ok else 'MISMATCH'} {g}")
    if not ok:
        RESULTS["errors"].append(f"assert {k}: {g} != {e}")
assert not RESULTS["errors"], RESULTS["errors"]

# --- quantized shared-experts MLP (production mxfp8 path, per-rank INTER_RANK) ---
def mk_shared():
    class M(nn.Module):
        pass
    m = M()
    m.gate_proj = nn.Linear(H, INTER_RANK, bias=False).to_quantized(group_size=32, bits=8, mode="mxfp8")
    m.up_proj = nn.Linear(H, INTER_RANK, bias=False).to_quantized(group_size=32, bits=8, mode="mxfp8")
    m.down_proj = nn.Linear(INTER_RANK, H, bias=False).to_quantized(group_size=32, bits=8, mode="mxfp8")
    def call(x):
        return m.down_proj(_limited_swiglu(m.gate_proj(x), m.up_proj(x), 10.0))
    return m, call

# rotation banks: set 0 real, rest synthetic same dtype/shape
N_ROT, N_ROT_SHARED = 64, 16
gate_ws = [gate0_w] + [(mx.random.normal((N_EXPERTS, H)) * 0.02).astype(mx.bfloat16)
                       for _ in range(N_ROT - 1)]
gate_bs = [gate0_b] + [mx.zeros((N_EXPERTS,), dtype=mx.float32) for _ in range(N_ROT - 1)]
norm_ws = [norm0_w] + [mx.random.normal((H,)).astype(mx.bfloat16) for _ in range(N_ROT - 1)]
shared_mods, shared_fns = [], []
for _ in range(N_ROT_SHARED):
    m, call = mk_shared()
    shared_mods.append(m)
    shared_fns.append(call)
perm = list(range(N_ROT)); rng.shuffle(perm)
perm_s = list(range(N_ROT_SHARED)); rng.shuffle(perm_s)
mx.eval(*gate_ws, *gate_bs, *norm_ws,
        *[t for m in shared_mods for t in (m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight,
                                           m.gate_proj.scales, m.up_proj.scales, m.down_proj.scales)])
log(f"weight bank ready; shared weight bytes/rank = {shared_mods[0].gate_proj.weight.nbytes * 3}")

SHARED_W_BYTES = (shared_mods[0].gate_proj.weight.nbytes
                  + shared_mods[0].up_proj.weight.nbytes
                  + shared_mods[0].down_proj.weight.nbytes)
SHARED_SCALES_BYTES = (shared_mods[0].gate_proj.scales.nbytes
                       + shared_mods[0].up_proj.scales.nbytes
                       + shared_mods[0].down_proj.scales.nbytes)

rope = DeepseekV4RoPE(64, 160000.0, {"type": "yarn", "factor": 16,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 32, "beta_slow": 1})
mx.eval(rope.parameters())
norms = [nn.RMSNorm(H, eps=1e-6) for w in norm_ws]
for m, w in zip(norms, norm_ws):
    m.weight = w
mx.eval(*[m.weight for m in norms])
log("modules ready")

# ---------------------------------------------------------------- bench harness
def bench(name, mk_call, n, warmup, bytes_modeled=None, flops_modeled=None,
          L=None, extra=None, passes=3):
    """p01/P03 time_stage pattern per timed call. Denomination rule:
    intensity = flops/bytes; < 10 FLOP/byte -> bandwidth ceiling (424/546
    GB/s); >= 10 -> FLOPS ceiling (11.66 TFLOPS)."""
    try:
        for j in range(warmup):
            out = mk_call(j % 512)
            mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
        mx.synchronize()
        recs = []
        for _p in range(passes):
            t0 = time.perf_counter()
            mx.metal.reset_gpu_time()
            d0 = mx.metal.dispatch_count()
            for i in range(n):
                out = mk_call(i)
                mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
            mx.synchronize()
            recs.append({"wall_us": (time.perf_counter() - t0) / n * 1e6,
                         "gpu_us": mx.metal.gpu_time_ns() / n / 1e3,
                         "dispatches": (mx.metal.dispatch_count() - d0) / n})
        srt = {k: sorted(x[k] for x in recs) for k in recs[0]}
        r = {k: srt[k][len(srt[k]) // 2] for k in srt}
        r["bytes"] = bytes_modeled
        r["flops"] = flops_modeled
        r["intensity_flop_per_byte"] = (flops_modeled / bytes_modeled
                                        if (bytes_modeled and flops_modeled) else None)
        r["ceiling"] = None
        if bytes_modeled and flops_modeled:
            inten = flops_modeled / bytes_modeled
            if inten < 10:
                r["ceiling"] = "bandwidth"
                r["gbps"] = bytes_modeled / (r["gpu_us"] * 1e-6) / 1e9 if r["gpu_us"] > 0 else 0.0
                r["pct_spec"] = 100 * r["gbps"] / SPEC_GBPS
                r["pct_real"] = 100 * r["gbps"] / REAL_GBPS
            else:
                r["ceiling"] = "flops"
                r["tflops"] = flops_modeled / (r["gpu_us"] * 1e-6) / 1e12 if r["gpu_us"] > 0 else 0.0
                r["pct_ceiling"] = 100 * r["tflops"] * 1e12 / TFLOPS
        elif bytes_modeled:
            r["ceiling"] = "bandwidth(bytes-only model)"
            r["gbps"] = bytes_modeled / (r["gpu_us"] * 1e-6) / 1e9 if r["gpu_us"] > 0 else 0.0
            r["pct_spec"] = 100 * r["gbps"] / SPEC_GBPS
            r["pct_real"] = 100 * r["gbps"] / REAL_GBPS
        r["passes"] = [{k: x[k] for k in ("wall_us", "gpu_us", "dispatches")} for x in recs]
        if L is not None:
            r["L"] = L
        if extra:
            r.update(extra)
        RESULTS["ops"].setdefault(name, {})[str(L) if L is not None else "-"] = r
        if r.get("ceiling") == "flops":
            log(f"{name:<30} L={r.get('L','-')} gpu={r['gpu_us']:9.2f}us "
                f"wall={r['wall_us']:9.2f}us disp={r['dispatches']:7.1f} "
                f"{r['tflops']:7.2f}TF {r['pct_ceiling']:5.1f}%ceil [flops]")
        elif r.get("gbps") is not None:
            log(f"{name:<30} L={r.get('L','-')} gpu={r['gpu_us']:9.2f}us "
                f"wall={r['wall_us']:9.2f}us disp={r['dispatches']:7.1f} "
                f"{r['gbps']:7.1f}GB/s {r['pct_spec']:5.1f}%spec {r['pct_real']:5.1f}%real [bw]")
        else:
            log(f"{name:<30} L={r.get('L','-')} gpu={r['gpu_us']:9.2f}us "
                f"wall={r['wall_us']:9.2f}us disp={r['dispatches']:7.1f}")
        save()
        return r
    except Exception as e:
        RESULTS["errors"].append(f"{name} L={L}: {type(e).__name__}: {e}")
        log(f"ERROR {name} L={L}: {e}")
        save()
        return None

def xs_bank(L, n=8, seed=1):
    mx.random.seed(seed)
    return [mx.random.normal((1, L, H)).astype(mx.bfloat16) for _ in range(n)]

# ================================================================ L2 sanity
def l2_sanity():
    out = {}
    xs = xs_bank(L_BAND, 8, 11)
    for label, nsets in ((f"gate_2_L{L_BAND}", 2), (f"gate_{N_ROT}_L{L_BAND}", N_ROT)):
        mk = lambda i, ns=nsets, xs=xs: _gate_route(
            xs[i % 8], gate_ws[perm[i % N_ROT] % ns], gate_bs[perm[i % N_ROT] % ns],
            TOP_K, 1.5, True, "sqrtsoftplus")
        r = bench(f"l2sanity.{label}", mk, 100, 8,
                  N_EXPERTS * H * 2 + L_BAND * H * 2 + L_BAND * 9 * TOP_K, None,
                  L_BAND)
        if r:
            out[label] = round(r["gbps"], 1)
    for label, nsets in ((f"shared_2_L{L_BAND}", 2), (f"shared_{N_ROT_SHARED}_L{L_BAND}", N_ROT_SHARED)):
        mk = lambda i, ns=nsets, xs=xs: shared_fns[perm_s[i % N_ROT_SHARED] % ns](xs[i % 8])
        r = bench(f"l2sanity.{label}", mk, 60, 8,
                  SHARED_W_BYTES + SHARED_SCALES_BYTES + L_BAND * (H * 2 * 5 + INTER_RANK * 2 * 3), None,
                  L_BAND)
        if r:
            out[label] = round(r["gbps"], 1)
    RESULTS["l2_sanity"] = out
    log("L2 SANITY:", json.dumps(out))
    save()

l2_sanity()

# ================================================================ op 1: moe.post_combine
def moe_post_combine():
    per = TOP_K * H * 2 * L_BAND + TOP_K * 4 * L_BAND + 3 * H * 2 * L_BAND
    ys = [mx.random.normal((1, L_BAND, TOP_K, H)).astype(mx.bfloat16) for _ in range(8)]
    sc = [mx.random.normal((1, L_BAND, TOP_K)).astype(mx.float32) for _ in range(8)]
    sh = [mx.random.normal((1, L_BAND, H)).astype(mx.bfloat16) for _ in range(8)]
    mx.eval(*ys, *sc, *sh)
    mkH = lambda i: _moe_post_combine(ys[i % 8], sc[i % 8], sh[i % 8])
    r = bench("moe.post_combine.hot", mkH, 200, 12, per, 3.0 * L_BAND * (2.0 + TOP_K),
              L_BAND, extra={"note": "L2-hot primary (production-adjacent: y/shared fresh-written)"})
    # DRAM variant: rotate > L2 (M4 Max L2 = 16 MB; per-call traffic 75.5 MB -> 4 sets)
    nsets = max(8, min(64, int(192e6 // per)))
    mx.random.seed(600)
    ysR = [mx.random.normal((1, L_BAND, TOP_K, H)).astype(mx.bfloat16) for _ in range(nsets)]
    scR = [mx.random.normal((1, L_BAND, TOP_K)).astype(mx.float32) for _ in range(nsets)]
    shR = [mx.random.normal((1, L_BAND, H)).astype(mx.bfloat16) for _ in range(nsets)]
    mx.eval(*ysR, *scR, *shR)
    mkR = lambda i: _moe_post_combine(ysR[i % nsets], scR[i % nsets], shR[i % nsets])
    bench("moe.post_combine.dram", mkR, nsets, 6, per, 3.0 * L_BAND * (2.0 + TOP_K), L_BAND,
          extra={"note": f"rotated {nsets} sets ({nsets * per / 1e6:.0f} MB) — DRAM-real"})

# --- shared_experts MLP forward (the other piece of the post_combine span) ---
def shared_experts():
    xs = xs_bank(L_BAND, 8, 300)
    act = L_BAND * (H * 2 * 4 + INTER_RANK * 2 * 3 + H * 2)
    B = SHARED_W_BYTES + SHARED_SCALES_BYTES + act
    F = 3 * 2.0 * L_BAND * INTER_RANK * H
    mk = lambda i: shared_fns[perm_s[i % N_ROT_SHARED]](xs[i % 8])
    bench("moe.shared_experts", mk, 60, 8, B, F, L_BAND,
          extra={"note": "cold-weight DRAM-real (rotated 16 sets)"})
    # also report weight-only streaming GB/s for readability
    per = SHARED_W_BYTES + SHARED_SCALES_BYTES + act
    RESULTS["ops"]["moe.shared_experts"].setdefault(str(L_BAND), {})["weight_bytes"] = SHARED_W_BYTES

# ================================================================ op 3: moe.gate
def moe_gate():
    G_FLOPS = 2.0 * L_BAND * N_EXPERTS * H
    G_BYTES = N_EXPERTS * H * 2 + L_BAND * H * 2 + L_BAND * N_EXPERTS * 8 + N_EXPERTS * 4
    xs = xs_bank(L_BAND, 8, 100)
    mk = lambda i: _gate_route(xs[i % 8], gate_ws[perm[i % N_ROT]],
                               gate_bs[perm[i % N_ROT]], TOP_K, 1.5, True, "sqrtsoftplus")
    bench("moe.gate.routed", mk, 100, 8, G_BYTES, G_FLOPS, L_BAND,
          extra={"note": "weight-rotated 64 sets; per-rank L_band rows"})
    ids = mx.random.randint(0, V, (1, L_BAND), dtype=mx.int32)
    mx.eval(ids)
    # hash gate needs tid2eid; load the real table (int32 129280x6)
    tid2eid = load_mlx("layers.0.ffn.gate.tid2eid").astype(mx.int32)
    mx.eval(tid2eid)
    mkh = lambda i: _hash_gate_route(ids, xs[i % 8], gate_ws[perm[i % N_ROT]],
                                     tid2eid, 1.5, True, "sqrtsoftplus")
    bench("moe.gate.hash", mkh, 100, 8, G_BYTES + L_BAND * 24, G_FLOPS, L_BAND,
          extra={"note": "weight-rotated; layers 0-2 only (3 of 43)"})

# ================================================================ op 4: tail spans
def tails():
    # RMSNorm at L_band
    per = L_BAND * H * 2 * 2 + H * 2
    xs = xs_band_bank = xs_bank(L_BAND, 8, 700)
    mk = lambda i: norms[perm[i % N_ROT] % N_ROT](xs_band_bank[i % 8])
    bench("tail.rmsnorm", mk, 200, 12, per, 2.0 * L_BAND * H, L_BAND,
          extra={"note": "L2-hot primary (production: x fresh from hc_expand); 87 calls/chunk"})
    # DeepseekV4RoPE on q (1, 64 heads, L_band, 512) and on kv
    for nm, shp in (("tail.rope.q_L512", (1, 64, L_BAND, 512)),
                    ("tail.rope.q_idx", (1, IDX_HEADS, L_BAND, IDX_DIM)),
                    ("tail.rope.kv", (1, 1, L_BAND, 512))):
        per = 2 * int(np.prod(shp)) * 2
        mx.random.seed(1000)
        bufs = [mx.random.normal(shp).astype(mx.bfloat16) for _ in range(8)]
        mx.eval(*bufs)
        mk = lambda i, bufs=bufs: rope(bufs[i % 8], offset=218880 + 137 * i)
        bench(nm, mk, 200, 12, per, None, L_BAND)
    # attn.mask (windowed causal bool) — real create_causal_mask shape
    S = LOCAL_WINDOW
    linds = mx.arange(218880, 218880 + L_BAND)[:, None]
    rinds = mx.arange(218880 + L_BAND)[None]
    mk = lambda i: (linds >= rinds) & (linds < rinds + S)
    band_bytes = L_BAND * (S + 1)
    bench("tail.attn_mask.windowed", mk, 100, 8, band_bytes, 3.0 * band_bytes, L_BAND,
          extra={"note": "windowed-causal band (1,1,L,128+1) bool; local-sparse layers"})
    # full (no local) causal mask for compressed layers
    mkf = lambda i: (linds >= rinds)
    full_bytes = L_BAND * (218880 + L_BAND)
    bench("tail.attn_mask.full", mkf, 60, 6, full_bytes, 2.0 * full_bytes, L_BAND,
          extra={"note": "full causal bool (1,1,L,N) for compressed layers"})
    # kv_cache write (local RotatingKVCache, window 128). Steady-state
    # production shape: cache already full (offset > max_size), so a
    # 1024-row update takes the _update_concat path (temporal reorder +
    # trim + concat). Prime the cache with 128 slots, then rotate through
    # 1024-row (1,1,L,512) bf16 writes.
    cache = RotatingKVCache(max_size=128)
    kv0 = mx.random.normal((1, 1, 128, 512)).astype(mx.bfloat16)
    v0 = mx.zeros((1, 1, 128, 0))
    k_full, _ = cache.update_and_fetch(kv0, v0)  # fills the ring
    mx.eval(k_full)
    cache.offset = 218880  # deep in steady state (rotation active)
    kvs = [mx.random.normal((1, 1, L_BAND, 512)).astype(mx.bfloat16) for _ in range(8)]
    vs = [mx.zeros((1, 1, L_BAND, 0)) for _ in range(8)]
    mx.eval(*kvs, *vs)
    for j in range(2):
        kk, _ = cache.update_and_fetch(kvs[j], vs[j])
        mx.eval(kk)
    mx.synchronize()
    def kv_write(i):
        kk, _ = cache.update_and_fetch(kvs[i % 8], vs[i % 8])
        return kk
    bench("tail.kv_cache.write", kv_write, 40, 6, 2 * L_BAND * 512 * 2, None, L_BAND,
          extra={"note": "steady-state windowed update: _update_concat (temporal reorder "
                         "+ trim + append) of (1,1,1024,512) into 128-slot ring"})

# ================================================================ op 2: attn.indexer
def indexer_ops(use_layer=4):
    # weight-bank: wq_b is MXFP8-quantized in the checkpoint (F8_E4M3 blocks +
    # E8M0 scales, group32/bits8/mode mxfp8 per make_quantization_config);
    # weights_proj is plain BF16 (64, 4096).
    wq_b = load_mxfp8_qlinear(f"layers.{use_layer}.attn.indexer.wq_b", 8192, 1024)
    wproj = load_mlx(f"layers.{use_layer}.attn.indexer.weights_proj.weight")
    mx.eval(wq_b.weight, wq_b.scales, wproj)
    log(f"indexer weights: wq_b {type(wq_b).__name__} {wq_b.weight.shape} "
        f"mode={wq_b.mode}, wproj {wproj.shape} {wproj.dtype}")

    # pooled bank: (1, P, 128) bf16 rotated > L2
    nsets = max(4, min(8, int(192e6 // (P * IDX_DIM * 2))))
    pooled_bank = [(mx.random.normal((1, P, IDX_DIM)) * 0.05).astype(mx.bfloat16)
                   for _ in range(nsets)]

    # 1) The FOLDED SCORE GEMM alone (OPT-6 folded single (L,128)@(128,P))
    #    q_weighted is precomputed: measure JUST the matmul + output write.
    S_FLOPS = 2.0 * L_BAND * IDX_DIM * P
    S_BYTES = L_BAND * IDX_DIM * 2 + P * IDX_DIM * 2 + L_BAND * P * 2
    def mk_score(i):
        qw = (mx.random.normal((1, L_BAND, IDX_DIM)) * 0.05).astype(mx.bfloat16)
        return _indexer_score_tile_like(qw, pooled_bank[i % nsets])
    def _indexer_score_tile_like(qw, pooled):
        # bit-identical inner GEMM of _indexer_score (post-OPT-6 fold)
        return qw @ pooled.swapaxes(-1, -2)
    bench("attn.indexer.score_gemm.folded", mk_score, 30, 4, S_BYTES, S_FLOPS, L_BAND,
          passes=3,
          extra={"note": "THE never-measured GEMM: single (1024,128)@(128,55000) bf16 "
                         "(OPT-6 folded, docs/indexer-prefill-decomposition-2026-08-24.md:254)"})

    # 2) FULL _indexer_score (sigmoid weights + fold + GEMM)
    xs = xs_bank(L_BAND, 4, 500)
    q4 = mx.random.normal((1, IDX_HEADS, L_BAND, IDX_DIM)).astype(mx.bfloat16)
    mx.eval(q4)
    def mk_full(i):
        w = (mx.sigmoid(xs[i % 4] @ wproj.T) * (IDX_DIM ** -0.5 * IDX_HEADS ** -0.5))
        q_blhd = q4.transpose(0, 2, 1, 3)
        q_weighted = (w[..., None] * q_blhd).sum(axis=2)
        return q_weighted @ pooled_bank[i % nsets].swapaxes(-1, -2)
    bench("attn.indexer.score_full", mk_full, 20, 3,
          S_BYTES + L_BAND * H * 2 + IDX_HEADS * L_BAND * 4 + IDX_DIM * L_BAND * 2, S_FLOPS,
          L_BAND, passes=3,
          extra={"note": "weights_proj sigmoid + H-fold reduce + GEMM (the real span "
                         "minus pmask/topk); pooled rotated"})

    # 3) pmask apply (tail-optimized production path; P=55000 fixed)
    pm = mx.random.randint(0, 2, (L_BAND, P)).astype(mx.bool_)
    scores0 = mx.random.normal((1, L_BAND, P)).astype(mx.bfloat16)
    mx.eval(pm, scores0)
    neg = mx.finfo(mx.bfloat16).min
    vis_min, vis_max = min((218880 + 1) // 4, P), min((218880 + L_BAND) // 4 + 1, P)
    def mk_pmask(i):
        parts = [scores0[..., :vis_min]]
        if vis_max > vis_min:
            parts.append(mx.where(pm[None, :, vis_min:vis_max],
                                  scores0[..., vis_min:vis_max], neg))
        if P > vis_max:
            parts.append(mx.full((1, L_BAND, P - vis_max), neg, dtype=mx.bfloat16))
        return mx.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
    band = vis_max - vis_min
    bench("attn.indexer.pmask.apply", mk_pmask, 100, 8, L_BAND * P * 2, 2.0 * L_BAND * band, L_BAND,
          extra={"note": "EXO_DSV4_TAIL_PMASK=1 production form: only band cols are row-dependent; "
                         f"band={band} cols, tail {P - vis_max} cols const-fill"})

    # 4) topk: production prefill path = argpartition(-scores, 512)
    def mk_topk(i):
        return mx.argpartition(-scores0, kth=IDX_TOPK - 1, axis=-1)[..., :IDX_TOPK]
    bench("attn.indexer.topk.argpartition", mk_topk, 30, 4, L_BAND * P * 2, 2.0 * L_BAND * P * math.log2(P), L_BAND,
          passes=3,
          extra={"note": "prefill path (L=1024>16): argpartition(-scores, kth=511)[..., :512]"})


def indexer_score_real_module(use_layer=4):
    # REAL Indexer q-side path minus compressor. wq_b is MXFP8 (same quantized
    # matmul the production module runs); weights_proj BF16. q projection +
    # rope + score GEMM + argpartition topk, end to end.
    wq_b = load_mxfp8_qlinear(f"layers.{use_layer}.attn.indexer.wq_b", 8192, 1024)
    wproj = load_mlx(f"layers.{use_layer}.attn.indexer.weights_proj.weight")
    mx.eval(wq_b.weight, wq_b.scales, wproj)
    q_res = mx.random.normal((1, L_BAND, 1024)).astype(mx.bfloat16)
    xs = xs_bank(L_BAND, 4, 800)
    w = (mx.sigmoid(xs[0] @ wproj.T) * (IDX_DIM ** -0.5 * IDX_HEADS ** -0.5))
    q4 = mx.random.normal((1, IDX_HEADS, L_BAND, IDX_DIM)).astype(mx.bfloat16)
    mx.eval(q_res, q4, w)
    pooled = (mx.random.normal((1, P, IDX_DIM)) * 0.05).astype(mx.bfloat16)
    mx.eval(pooled)

    def mk_real(i):
        q = wq_b(q_res)  # mxfp8 quantized matmul, (1, L_BAND, 8192)
        q = q.reshape(1, L_BAND, IDX_HEADS, IDX_DIM).transpose(0, 2, 1, 3)
        q = rope(q, offset=218880 + (i % 4))
        wq = (w[..., None] * q.transpose(0, 2, 1, 3)).sum(axis=2)
        scores = wq @ pooled.swapaxes(-1, -2)
        return mx.argpartition(-scores, kth=IDX_TOPK - 1, axis=-1)[..., :IDX_TOPK]
    F = 2.0 * L_BAND * IDX_DIM * P
    B = (wq_b.weight.nbytes + wq_b.scales.nbytes + wproj.nbytes + P * IDX_DIM * 2
         + L_BAND * 1024 * 2 + L_BAND * P * 2)
    bench("attn.indexer.combined_q_rope_score_topk", mk_real, 20, 3, B, F, L_BAND,
          passes=3,
          extra={"note": "wq_b mxfp8 proj + rope + sigmoid-fold + score GEMM + argpartition topk "
                         "in one call (the whole prefill indexer q-side minus compressor/pool)"})


moe_post_combine()
shared_experts()
moe_gate()
tails()
indexer_ops(use_layer=4)
indexer_score_real_module(use_layer=4)

save()
log("=== isolated ops complete ===")

# ================================================================ per-chunk rollup
def G(name, field="gpu_us"):
    e = RESULTS["ops"].get(name, {}).get(str(L_BAND))
    return e.get(field) if e else None

roll = {}
try:
    counts = {
        "moe.post_combine (43)": (43, G("moe.post_combine.dram") or G("moe.post_combine.hot")),
        "moe.shared_experts (43)": (43, G("moe.shared_experts")),
        "moe.gate.routed (40)": (40, G("moe.gate.routed")),
        "moe.gate.hash (3)": (3, G("moe.gate.hash")),
        "tail.rmsnorm (87)": (87, G("tail.rmsnorm")),
        "tail.rope (43q+43kv+21idx)": (43, G("tail.rope.q_L512")),
        "tail.kv_cache.write (22 local)": (22, G("tail.kv_cache.write")),
        "attn.indexer (21 sparse)": (21, G("attn.indexer.combined_q_rope_score_topk")),
        "attn.indexer.score_gemm (21)": (21, G("attn.indexer.score_gemm.folded")),
        "attn.indexer.topk (21)": (21, G("attn.indexer.topk.argpartition")),
        "attn.indexer.pmask (21)": (21, G("attn.indexer.pmask.apply")),
    }
    for name, (k, us) in counts.items():
        if us:
            roll[name + "_us_per_chunk"] = round(k * us, 1)
        else:
            roll[name + "_us_per_chunk"] = "UNDETERMINED"
    roll["_NOTE"] = ("per-chunk = per-call x calls(43-layer, L_band=1024, ctx=220K); "
                     "attn.indexer uses score+rope+topk combined call (excl. compressor)")
except Exception as e:
    RESULTS["errors"].append(f"rollup: {e}")
    log(f"ROLLUP ERROR: {e}")
RESULTS["rollup"] = roll
log("ROLLUP:", json.dumps(roll, indent=1))
log("=== P07 CAPTURE COMPLETE ===")
save()
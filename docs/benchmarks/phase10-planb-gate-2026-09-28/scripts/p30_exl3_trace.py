#!/usr/bin/env python3
"""p30_exl3_trace.py v2 -- full 40-layer routing trace on the real EXL3 checkpoint.

Layer-major: each layer is built from EXL3 tensors (trellis -> kernels; attention
via reconstruct_public_mlx), the 531-token prompt runs through it, gate picks are
recorded, the layer is freed. One checkpoint pass; every gate sees the REAL
model's hidden states. Prefill position i is causally valid, so this is a valid
decode-routing trace.

v2 changes: raw os.pread checkpoint reader (BF16-capable -- numpy cannot hold
bf16); lazy engram row reader (never materializes the 98 GB/101 GB tables);
EXL3SwitchGLU for experts (trellis resident, prefill kernel streams experts).

Modes: --selftest | --full
Env: TRACE_TOKENS, TRACE_LAYERS, TRACE_OUT, TRACE_ENGRAM, TRACE_FORCE
"""
from __future__ import annotations

import gc
import json
import os
import struct
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

HOME = os.path.expanduser("~")
MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
PORT = HOME + "/repos/ref/deepseek-v41-mlx"

sys.path.insert(0, PORT)
sys.path.insert(0, HOME + "/repos/ref/PonyExl3")
os.environ.setdefault("EXL3_MM_MAX_ROWS", "100000")  # chunked prefill needs the segmented path at 531 rows

from ponyexl3.ref.layer import EXL3Layer  # noqa: E402
from ponyexl3.ref.codebook import codebook_mode_from_flags  # noqa: E402
from ponyexl3.mlx.reconstruct import reconstruct_public_mlx  # noqa: E402
from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU  # noqa: E402
from ponyexl3.mlx.exl3_moe import _MOE_ACTIVATIONS  # noqa: E402

from deepseek_v41_mlx.config import ModelArgs  # noqa: E402
from deepseek_v41_mlx.model import Block, SharedState  # noqa: E402
from deepseek_v41_mlx.cache import ModelCache  # noqa: E402
from deepseek_v41_mlx.hyper_connections import make_identity_pre_mix  # noqa: E402
import deepseek_v41_mlx.moe as moe_mod  # noqa: E402
import deepseek_v41_mlx.engram as engram_mod  # noqa: E402

# --- cheap zero-init expert buffers (SwitchLinear defaults to fp32 zeros:
# 54 GB/projection transient at V4.1 dims before load_weights replaces them) ---
import mlx_lm.models.switch_layers as _sl  # noqa: E402


def _sl_init(self, input_dims, output_dims, num_experts, bias=True, **_kw):
    nn.Module.__init__(self)
    self.weight = mx.zeros((num_experts, output_dims, input_dims), dtype=mx.bfloat16)
    if bias:
        self.bias = mx.zeros((num_experts, output_dims), dtype=mx.bfloat16)


_sl.SwitchLinear.__init__ = _sl_init


# ================= raw checkpoint reader (BF16-capable) =====================
_NP = {"F16": np.float16, "F32": np.float32, "I16": np.int16, "I32": np.int32,
       "U8": np.uint8, "I8": np.int8, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8,
       "I64": np.int64, "U16": np.uint16, "U32": np.uint32}


class ST:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.hdr = json.loads(f.read(n))
        self.base = 8 + n
        self.fd = None

    def _o(self, name):
        return self.hdr[name]["data_offsets"]

    def np_arr(self, name):
        o = self._o(name)
        dt = self.hdr[name]["dtype"]
        shp = self.hdr[name]["shape"]
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        buf = os.pread(self.fd, o[1] - o[0], self.base + o[0])
        if dt == "BF16":
            u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shp)
        return np.frombuffer(buf, _NP[dt]).reshape(shp)

    def rows(self, name, row_idx):
        """Gather specific rows of a 2-D tensor via one pread per row (or a span)."""
        o = self._o(name)
        shp = self.hdr[name]["shape"]
        w = shp[1] * np.dtype(_NP[self.hdr[name]["dtype"]]).itemsize
        buf = bytearray(len(row_idx) * w)
        mv = memoryview(buf)
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        for i, r in enumerate(row_idx):
            mv[i * w:(i + 1) * w] = os.pread(self.fd, w, self.base + o[0] + r * w)
        dt = self.hdr[name]["dtype"]
        if dt == "BF16":
            u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(len(row_idx), shp[1])
        return np.frombuffer(buf, _NP[dt]).reshape(len(row_idx), shp[1])


IDX = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
NIDX = json.load(open(os.path.join(NATIVE, "model.safetensors.index.json")))["weight_map"]
CFG = json.load(open(os.path.join(MODEL, "config.json")))
ARGS = ModelArgs.from_dict(CFG)
# V4.1 routed experts use the CLAMPED SwiGLU (swiglu_limit=10.0). The
# library gained "silu_clamp" as an additive mode (patch_exl3_clamp.py);
# set TRACE_ACT=silu to reproduce the pre-patch unclamped behavior.
ENV_ACT = os.environ.get("TRACE_ACT", "silu_clamp")
_st: dict = {}


def shard(name):
    p = os.path.join(MODEL, IDX[name])
    if p not in _st:
        if len(_st) > 10:
            k = next(iter(_st))
            if _st[k].fd is not None:
                os.close(_st[k].fd)
            _st.pop(k)
        _st[p] = ST(p)
    return _st[p]


def np_t(name):
    return shard(name).np_arr(name)


def mx_t(name, dtype=None):
    a = np_t(name)
    x = mx.array(a)
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    return x


def exl3_w(prefix):
    """Reconstruct one EXL3 group -> mx.array fp16, shape [in, out]."""
    t = np_t(prefix + ".trellis")
    suh = np_t(prefix + ".suh") if (prefix + ".suh") in IDX else None
    svh = np_t(prefix + ".svh") if (prefix + ".svh") in IDX else None
    k = int(t.shape[2] * 16 // 256)
    lay = EXL3Layer(key=prefix, in_features=len(suh), out_features=len(svh),
                    k=k, trellis=t, suh=suh, svh=svh, mul1=True)
    w = reconstruct_public_mlx(lay)
    del lay, t, suh, svh
    return w.astype(mx.bfloat16)


# ================= lazy engram ==============================================
class LazyEngramTable(nn.Module):
    """Row-on-demand reader for one engram table (never materializes it)."""

    def __init__(self, layer_id):
        super().__init__()
        wname = "layers.%d.engram.embed.weight" % layer_id
        sname = "layers.%d.engram.embed.scale" % layer_id
        self._wname, self._sname = wname, sname
        self._wst = ST(os.path.join(NATIVE, NIDX[wname]))
        self._sst = ST(os.path.join(NATIVE, NIDX[sname]))
        shp = self._wst.hdr[wname]["shape"]
        self.num_embeddings, self.dim = shp

    def __call__(self, indices):
        idx = np.array(indices, dtype=np.int64).reshape(-1)
        uniq = np.unique(idx)
        w = self._wst.rows(self._wname, uniq)          # uint8 [u, 256]
        s = self._sst.rows(self._sname, uniq)          # uint8 [u, 8]
        wf = mx.from_fp8(mx.array(w), mx.float32)
        sf = mx.power(mx.array(2.0), (s.astype(np.int32) - 127).astype(np.float32))
        rows = (wf.reshape(len(uniq), -1, 32) * sf[..., None]).reshape(len(uniq), self.dim)
        inv = mx.array(np.searchsorted(uniq, idx).astype(np.int32))
        out = rows[inv]
        return out.reshape(tuple(np.array(indices).shape) + (self.dim,))


# Monkeypatch EngramEmbedding so Block construction never allocates the table.
def _stub_init(self, num_embeddings, dim, block=32):
    nn.Module.__init__(self)
    self.block = block
    self.weight = mx.zeros((0, 0), dtype=mx.uint8)
    self.scale = mx.zeros((0, 0), dtype=mx.uint8)


engram_mod.EngramEmbedding.__init__ = _stub_init


# ================= layer builder ============================================
class Exl3Experts(nn.Module):
    """Adapter: EXL3SwitchGLU with the port SwitchGLU call convention."""

    def __init__(self, sg):
        super().__init__()
        self._sg = sg

    def __call__(self, x, indices):
        squeeze = (x.ndim == 2)
        if squeeze:
            x = x[None]
        if indices.ndim == 2:
            indices = indices[None]
        y = self._sg(x, indices)
        if y.ndim == 2:                      # flattened fallback
            y = y.reshape(x.shape[0], x.shape[1], indices.shape[-1], -1)
        if squeeze:
            y = y[0]
        return y


def build_exl3_experts(L):
    if ENV_ACT not in _MOE_ACTIVATIONS:
        raise SystemExit("activation %r not in library frozenset %s -- run patch_exl3_clamp.py first" % (ENV_ACT, sorted(_MOE_ACTIVATIONS)))
    p = "layers.%d.ffn.experts." % L
    t0 = time.time()
    gu, up, dn, gs_, gv, ds, dv = [], [], [], [], [], [], []
    for e in range(ARGS.n_routed_experts):
        pre = p + "%d." % e
        gu.append(np_t(pre + "w1.trellis"))
        up.append(np_t(pre + "w3.trellis"))
        dn.append(np_t(pre + "w2.trellis"))
        gs_.append(np.stack([np_t(pre + "w1.suh"), np_t(pre + "w3.suh")]))
        gv.append(np.concatenate([np_t(pre + "w1.svh"), np_t(pre + "w3.svh")]))
        ds.append(np_t(pre + "w2.suh"))
        dv.append(np_t(pre + "w2.svh"))
    t1 = time.time()
    k = int(gu[0].shape[2] * 16 // 256)
    sg = EXL3SwitchGLU(
        gu_trellis=mx.concatenate([mx.array(a) for a in gu + up], axis=1).view(mx.uint16),
        gu_suh=mx.stack([mx.array(a) for a in gs_]).astype(mx.float16),
        gu_svh=mx.stack([mx.array(a) for a in gv]).astype(mx.float16),
        dn_trellis=mx.concatenate([mx.array(a) for a in dn], axis=1).view(mx.uint16),
        dn_suh=mx.stack([mx.array(a) for a in ds]).astype(mx.float16),
        dn_svh=mx.stack([mx.array(a) for a in dv]).astype(mx.float16),
        k=k, cb=codebook_mode_from_flags(mcg=False, mul1=True), activation=ENV_ACT)
    mx.eval(sg._gu_trellis, sg._gu_suh, sg._gu_svh,
            sg._dn_trellis, sg._dn_suh, sg._dn_svh)
    nbytes = sum(int(np.prod(a.shape)) * a.itemsize for a in
                 (np.array(0),))  # placeholder; real size below
    print("    [ex] E=%d k=%d build %.1fs v2ok=%s" % (
        ARGS.n_routed_experts, k, time.time() - t1, sg._v2_ok()), flush=True)
    return Exl3Experts(sg)


def load_layer_weights(L):
    p = "layers.%d." % L
    w = {}
    w["attn.wq_a.weight"] = exl3_w(p + "attn.wq_a").T
    w["attn.q_norm.weight"] = mx_t(p + "attn.q_norm.weight", mx.float32)
    w["attn.wq_b.weight"] = exl3_w(p + "attn.wq_b").T
    w["attn.wkv.weight"] = exl3_w(p + "attn.wkv").T
    w["attn.kv_norm.weight"] = mx_t(p + "attn.kv_norm.weight", mx.float32)
    slices = [exl3_w(p + "attn.wo_a.slice.%d" % g) for g in range(ARGS.o_groups)]
    wo = mx.stack(slices)
    w["attn.wo_a.weight"] = mx.transpose(wo, (0, 2, 1)).reshape(
        ARGS.o_groups * ARGS.o_lora_rank, -1)
    del slices, wo
    w["attn.wo_b.weight"] = exl3_w(p + "attn.wo_b").T
    w["attn.attn_sink"] = mx_t(p + "attn.attn_sink", mx.float32)

    if L in ARGS.kv_source_layers:
        w["attn.compressor.norm.weight"] = mx_t(p + "attn.compressor.norm.weight", mx.float32)
        w["attn.compressor.wkv.weight"] = exl3_w(p + "attn.compressor.wkv").T
        if ARGS.compress_ratio(L) > 1:
            w["attn.compressor.wgate.weight"] = exl3_w(p + "attn.compressor.wgate").T
    if L in ARGS.index_source_layers:
        w["attn.indexer.wq_b.weight"] = exl3_w(p + "attn.indexer.wq_b").T
        w["attn.indexer.weights_proj.weight"] = mx_t(p + "attn.indexer.weights_proj.weight")
        if L in ARGS.kv_source_layers:
            w["attn.indexer.wk.weight"] = exl3_w(p + "attn.indexer.wk").T
            w["attn.indexer.k_norm.weight"] = mx_t(p + "attn.indexer.k_norm.weight", mx.float32)

    if L in ARGS.engram_layer_ids:
        w["engram.wkv.weight"] = exl3_w(p + "engram.wkv").T
        w["engram.q_weight"] = mx_t(p + "engram.q_weight", mx.float32)
        w["engram.k_weight"] = mx_t(p + "engram.k_weight", mx.float32)

    w["ffn.gate.weight"] = mx_t(p + "ffn.gate.weight")
    w["ffn.gate.bias"] = mx_t(p + "ffn.gate.bias", mx.float32)
    if (p + "ffn.gate.bias_vl") in IDX:
        w["ffn.gate.bias_vl"] = mx_t(p + "ffn.gate.bias_vl", mx.float32)
    for wn in ("w1", "w2", "w3"):
        w["ffn.shared_experts.%s.weight" % wn] = exl3_w(p + "ffn.shared_experts.%s" % wn).T

    w["attn_norm.weight"] = mx_t(p + "attn_norm.weight", mx.float32)
    w["ffn_norm.weight"] = mx_t(p + "ffn_norm.weight", mx.float32)
    for nm in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
               "hc_attn_scale", "hc_ffn_scale"):
        w[nm] = mx_t(p + nm, mx.float32)
    return w


def build_layer(L):
    t0 = time.time()
    layer = Block(L, ARGS)
    if L in ARGS.engram_layer_ids:
        layer.engram.embed = LazyEngramTable(L)
    layer.ffn.experts = build_exl3_experts(L)
    t1 = time.time()
    lw = load_layer_weights(L)
    expected = {k for k, _ in tree_flatten(layer.parameters())}
    extra = set(lw) - expected
    missing = expected - set(lw)
    if extra or missing:
        print("  [w] L%d extra=%s missing=%s" % (L, sorted(extra)[:8], sorted(missing)[:8]),
              flush=True)
    layer.load_weights(list(lw.items()), strict=False)
    layer.ffn.gate._trace_layer_id = L
    mx.eval([v for _, v in tree_flatten(layer.parameters())])
    print("    [b] L%d total build %.1fs (experts %.1fs)" % (L, time.time() - t0, t1 - t0),
          flush=True)
    del lw
    return layer


# ================= gate trace ==============================================
RECORDS: list = []


def traced_call(self, x):
    weights, indices = _orig_gate_call(self, x)
    try:
        arr = indices if isinstance(indices, mx.array) else mx.array(indices)
        lid = getattr(self, "_trace_layer_id", -1)
        for row in arr.reshape(-1, arr.shape[-1]).tolist():
            RECORDS.append([lid, [int(i) for i in row]])
    except Exception as exc:  # noqa: BLE001
        print("  trace error:", exc, flush=True)
    return weights, indices


_orig_gate_call = moe_mod.Gate.__call__
moe_mod.Gate.__call__ = traced_call


# ================= run modes ===============================================
def prompt_ids(n):
    ids = json.load(open(os.path.join(HOME, "p30_prompt_ids.json")))
    return ids[:min(n, len(ids))]


def initial_stream(ids):
    ew = mx_t("embed.weight")
    h = ew[mx.array([ids])]
    h = mx.broadcast_to(h[:, :, None, :], (1, len(ids), ARGS.hc_mult, ARGS.dim))
    mx.eval(h)
    return h


def selftest():
    print("[t] ARGS: layers=%d engram=%s kv_src=%s idx_src=%s cand=%s" % (
        ARGS.n_layers, ARGS.engram_layer_ids, ARGS.kv_source_layers,
        ARGS.index_source_layers, ARGS.candidate_source_layer), flush=True)
    g = np_t("layers.0.ffn.gate.weight")
    print("[t] gate.weight", g.shape, g.dtype, "sample", g[0, :3], flush=True)
    t0 = time.time()
    w = exl3_w("layers.0.attn.wq_a")
    print("[t] wq_a reconstruct %.1fs -> %s %s" % (time.time() - t0, w.shape, w.dtype), flush=True)
    le = LazyEngramTable(1)
    ind = np.array([[[0, 5, 9], [1234567, 42, 999]]])
    r = le(ind)
    mx.eval(r)
    print("[t] engram rows -> %s %s norm=%.4f" % (
        r.shape, r.dtype, float(mx.sqrt(mx.mean(mx.square(r.astype(mx.float32)))))), flush=True)
    ids = prompt_ids(64)
    t0 = time.time()
    h = initial_stream(ids)
    print("[t] embed %.1fs %s" % (time.time() - t0, h.shape), flush=True)
    t0 = time.time()
    layer = build_layer(0)
    print("[t] layer0 built %.1fs active=%.1fGB peak=%.1fGB" % (
        time.time() - t0, mx.get_active_memory() / 1e9, mx.get_peak_memory() / 1e9), flush=True)
    cache = ModelCache(ARGS, 1, len(ids) + 8, dtype=mx.float32)
    pre_mix = make_identity_pre_mix(1, len(ids), ARGS.hc_mult)
    shared = SharedState()
    t0 = time.time()
    h, pre_mix = layer(h, pre_mix, 0, cache, shared)
    mx.eval(h)
    hn = float(mx.sqrt(mx.mean(mx.square(h.astype(mx.float32)))))
    print("[t] fwd %.2fs h=%s rms=%.4f finite=%s records=%d" % (
        time.time() - t0, h.shape, hn, bool(mx.all(mx.isfinite(h))), len(RECORDS)), flush=True)
    print("[t] first records:", RECORDS[:2], flush=True)


def run_full():
    n = int(os.environ.get("TRACE_TOKENS", "768"))
    spec = os.environ.get("TRACE_LAYERS", "0-39")
    layers = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            layers += list(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    force = os.environ.get("TRACE_FORCE", "0") == "1"
    rdir = os.path.expanduser(os.environ.get("TRACE_RDIR", "~/p30-records"))
    os.makedirs(rdir, exist_ok=True)
    ids = prompt_ids(n)
    print("[full] prompt=%d layers=%s" % (len(ids), layers[:3] + ["..."] if len(layers) > 3 else layers),
          flush=True)

    t0 = time.time()
    h = initial_stream(ids)
    print("[full] embed %.1fs" % (time.time() - t0), flush=True)
    cache = ModelCache(ARGS, 1, len(ids) + 64, dtype=mx.float32)
    pre_mix = make_identity_pre_mix(1, len(ids), ARGS.hc_mult)
    shared = SharedState()

    hashes = None
    tm_path = os.path.join(PORT, "engram_token_map.json")
    if os.path.exists(tm_path):
        hasher = engram_mod.EngramHasher(ARGS, json.load(open(tm_path)))
        hashes = mx.array(hasher(np.array([ids], dtype=np.int64), 0, cache.engram_ids))
        print("[full] engram hashes %s" % (hashes.shape,), flush=True)
        cache.engram_ids = None            # release the id buffer (hashes are computed)
    else:
        print("[full] WARNING: no engram token map -- engram layers get no hashes", flush=True)

    t_start = time.time()
    for L in layers:
        out_f = os.path.join(rdir, "L%02d.json" % L)
        h_f = os.path.join(rdir, "h_%02d.npy" % L)
        pm_f = os.path.join(rdir, "pm_%02d.npy" % L)
        if (os.path.exists(out_f) and os.path.exists(h_f)
                and os.path.exists(pm_f) and not force):
            h = mx.load(h_f)
            pre_mix = mx.load(pm_f)
            mx.eval(h, pre_mix)
            prev = json.load(open(out_f))
            print("[full] L%-2d cached (%d rows) -- skip, state restored" % (L, len(prev)), flush=True)
            continue
        t0 = time.time()
        layer = build_layer(L)
        if layer.engram is not None and hashes is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
        t1 = time.time()
        RECORDS.clear()
        h, pre_mix = layer(h, pre_mix, 0, cache, shared)
        mx.eval(h)
        t2 = time.time()
        rows = list(RECORDS)
        json.dump(rows, open(out_f, "w"))
        mx.save(h_f, h)
        mx.save(pm_f, pre_mix)
        hn = float(mx.sqrt(mx.mean(mx.square(h.astype(mx.float32)))))
        print("[full] L%-2d build %5.1fs fwd %6.2fs rms=%.3f rec=%d active=%.1fGB peak=%.1fGB" % (
            L, t1 - t0, t2 - t1, hn, len(rows), mx.get_active_memory() / 1e9,
            mx.get_peak_memory() / 1e9), flush=True)
        del layer
        gc.collect()
        mx.clear_cache()
    print("[full] pass %.1fs" % (time.time() - t_start), flush=True)

    all_recs = []
    for L in layers:
        f = os.path.join(rdir, "L%02d.json" % L)
        if os.path.exists(f):
            all_recs += json.load(open(f))
    out = os.path.expanduser(os.environ.get("TRACE_OUT", "~/p30_exl3_trace.json"))
    gate_meta = {str(L): {"n_experts": ARGS.n_routed_experts,
                          "topk": ARGS.n_activated_experts}
                 for L in sorted(set(r[0] for r in all_recs))}
    payload = {"n_steps": len(ids), "prompt_tokens": len(ids),
               "gate_meta": gate_meta, "records": all_recs}
    json.dump(payload, open(out, "w"))
    print("[full] wrote %s (%.2f MB, %d records)" % (
        out, os.path.getsize(out) / 1e6, len(all_recs)), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--full"
    print("=== p30v2 %s %s ===" % (mode, time.strftime("%F %T")), flush=True)
    if mode == "--selftest":
        selftest()
    else:
        run_full()
    print("=== p30v2 done %s ===" % time.strftime("%F %T"), flush=True)

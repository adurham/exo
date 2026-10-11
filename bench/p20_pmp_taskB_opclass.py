#!/usr/bin/env python3
"""P20 prefill measurement pack -- Task B: per-op-class GPU-time attribution.

ONE real decoder-layer slice, replayed AS PRODUCTION COMPOSES IT, at a chosen
prefill context.  Two measurements:

  (a) UNBRACKETED whole-layer GPU-busy + wall   (ground truth: production
      graph/compile boundaries intact, ONE lazy graph drained at the end).
  (b) BRACKETED per-op-class GPU-busy via MLX_GPU_TIME=1: every production
      module whose output is an op-class boundary is wrapped so that its OWN
      result is mx.eval'd (eval-fenced -- the phase-22 methodology), and the
      gpu_time delta since the previous fence is charged to it.  A running
      counter + nesting means each GPU-busy ns is charged exactly once; an
      outer wrapper gets only its non-inner residual (its glue).

Σ(bracketed) vs (a) is the bracketing distortion (lost overlap, broken
mx.compile fusion, extra command-buffer boundaries).  >~10% => per-class shares
unreliable at that resolution.

TP all_sum: NOT attributable via gpu_time (MLX's filter DROPS the JACCL
RDMA/CPU-only collective buffers, timestamps<=0).  `collective.all_sum` is
stubbed to identity (so NO RDMA touches the live link) and its cost is bounded a
second way (analytic payload/BW) plus the by-difference read against the live
unfenced per-chunk wall.

OFFLINE, single node (studio2), idle-guarded.  MTL_DISABLE_TIMEOUT=1.

Modes:  real (EXL3 checkpoint) | smoke (tiny random-weight config; plumbing only)
Run:    env -u PYTHONPATH <venv>/bin/python bench/p20_pmp_taskB_opclass.py ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics as stats
import sys
import time

# ---------------------------------------------------------------------------
# env MUST be set before `import mlx.core`
os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")   # match production
os.environ.setdefault("MLX_GPU_TIME", "1")          # before import mlx
os.environ.setdefault("EXL3_MM_SEG", "v19d")        # production EXL3 default
# match the LIVE production dense config (from the running exo process):
#   DSV41_DENSE=affine6, DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64
#   MLX_MAX_OPS_PER_BUFFER=200, MTL_COMMAND_BUFFER_TIMEOUT=0
if os.environ.get("P20_PMP_MODE", "real") == "real":
    os.environ.setdefault("DSV41_DENSE", "affine6")
    os.environ.setdefault("DSV41_DENSE_POLICY",
                          "layers.*.ffn.shared_experts.*=q8g64")
    os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")
os.environ.setdefault("MTL_COMMAND_BUFFER_TIMEOUT", "0")
# smoke mode only: the tiny config cannot satisfy the production hierarchical
# indexer's block-size assert; disable HIER there.  Never touched in real mode.
if os.environ.get("P20_PMP_MODE", "real") == "smoke":
    os.environ.setdefault("DSV41_INDEXER_HIER", "0")
_ENV = {k: os.environ.get(k) for k in
        ("MLX_GPU_TIME", "MTL_DISABLE_TIMEOUT", "EXL3_MM_SEG", "DSV41_INDEXER_HIER",
         "DSV41_DENSE", "DSV41_DENSE_POLICY", "MLX_MAX_OPS_PER_BUFFER",
         "MTL_COMMAND_BUFFER_TIMEOUT")}

_repo = os.environ.get("P20_PMP_MLXLN")             # smoke/local: point at repo mlx-lm
if _repo:
    sys.path.insert(0, _repo)

import numpy as np                                   # noqa: E402
import mlx.core as mx                                # noqa: E402
import mlx.nn as nn                                  # noqa: E402

HAVE_GPU_TIME = hasattr(mx.metal, "gpu_time_ns") and hasattr(mx.metal, "reset_gpu_time")

# ---------------------------------------------------------------------------
# constants / paths (overridable by env)
HOME = os.path.expanduser("~")
MODEL_DIR = os.environ.get(
    "P20_PMP_MODEL",
    HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
NATIVE_DIR = os.environ.get(
    "P20_PMP_NATIVE",
    HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram")
TRACE = os.environ.get("P20_PMP_TRACE", "")          # optional routing trace
USE_TRACE_ROUTING = os.environ.get("P20_PMP_TRACE_ROUTING", "1") == "1"

# existing TB5 RDMA numbers (do NOT bench the live link):
#   raw TB5 link ~80 Gbps ~= 10 GB/s per direction
#   (docs/pp-prefill-tp-decode-phase-swap-design-2026-08-16.md:222)
RDMA_GB_S_RAW = float(os.environ.get("P20_PMP_RDMA_GBS", "10.0"))
RDMA_GB_S_EFF = float(os.environ.get("P20_PMP_RDMA_GBS_EFF", "5.0"))
RDMA_LAT_US = float(os.environ.get("P20_PMP_RDMA_LAT_US", "20.0"))

# production live unfenced per-chunk wall at 16K (phase22 README, ~8380 ms/chunk)
LIVE_UNFENCED_MS_PER_CHUNK_16K = float(os.environ.get("P20_PMP_LIVE_CHUNK_MS", "8380"))

CHUNK = 2048


def _drain(out):
    if isinstance(out, (tuple, list)):
        mx.eval(*out)
    else:
        mx.eval(out)


# ===========================================================================
# collective stub -- record call count + payload; NO RDMA on the live link
# ===========================================================================
ALLSUM = {"n": 0, "bytes": 0, "shapes": []}


def _stub_all_sum(x, group=None):
    ALLSUM["n"] += 1
    ALLSUM["bytes"] += int(x.size * x.dtype.size)
    if len(ALLSUM["shapes"]) < 64:
        ALLSUM["shapes"].append(list(x.shape))
    return x


def install_allsum_stub():
    from mlx_lm.models.deepseek_v41 import collective as _coll
    _coll.all_sum = _stub_all_sum


# ===========================================================================
# eval-fenced attribution: running-counter recorder + production span context
# ===========================================================================
class Recorder:
    def __init__(self):
        self.active = False
        self.rec = {}
        self._last = 0

    def begin(self):
        mx.synchronize()
        if HAVE_GPU_TIME:
            mx.metal.reset_gpu_time()
        self._last = mx.metal.gpu_time_ns() if HAVE_GPU_TIME else 0
        self.rec = {}
        self.active = True

    def note(self, cls, result):
        _drain(result)
        now = mx.metal.gpu_time_ns() if HAVE_GPU_TIME else 0
        d = now - self._last
        self._last = now
        self.rec[cls] = self.rec.get(cls, 0) + d

    def final(self, cls="(unattributed residual)"):
        mx.synchronize()
        now = mx.metal.gpu_time_ns() if HAVE_GPU_TIME else 0
        d = now - self._last
        self._last = now
        if d:
            self.rec[cls] = self.rec.get(cls, 0) + d

    def end(self):
        self.active = False


REC = Recorder()
SPAN_CTX: list[str] = []            # innermost production span name


class CtxSpanHook:
    """ProfilerHook that only tracks the innermost production span name (no eval,
    no sync) so module wrappers can classify by call site."""
    @contextlib.contextmanager
    def span(self, name):
        SPAN_CTX.append(name)
        try:
            yield
        finally:
            SPAN_CTX.pop()

    def finalize(self, x):
        return x

    def on_layer_start(self, layer_idx, kind):
        return

    def on_layer_end(self, layer_idx, kind):
        return


def _inner_span():
    return SPAN_CTX[-1] if SPAN_CTX else ""


# ---------------------------------------------------------------------------
# classification: (module kind, innermost span) -> op class
# ---------------------------------------------------------------------------
def classify(kind, trellis=None, tn_base=None, out_e=None):
    inner = _inner_span()
    if kind in ("exl3linear", "affineproj"):
        if inner.startswith("attn.compressor"):
            return "CSA/HCA compression"
        if inner.startswith("attn.indexer"):
            return "indexer"
        if inner.startswith("moe.shared_experts"):
            return "dense proj (shared expert)"
        if inner.startswith("attn.proj_qkv"):
            return "dense proj (attn qkv+rope)"
        if inner.startswith("attn.o_proj"):
            return "dense proj (attn o)"
        return "dense proj (other)"
    if kind == "switchglu":
        return "MoE experts total"
    if kind == "mmseg":
        if trellis is not None and id(trellis) in DN_TRELLIS_IDS:
            return "MoE expert (down)"
        return "MoE expert (gate)" if tn_base == 0 else "MoE expert (up)"
    return {
        "sparse_attn": "sdpa (sparse/compressed)",
        "indexer": "indexer",
        "compressor": "CSA/HCA compression",
        "norm": "norms (RMSNorm)",
        "gate": "MoE routing gate (score+topk)",
        "engram": "engram (LazyEngramTable)",
        "hc": "hyper-connections (HC)",
    }.get(kind, kind)


DN_TRELLIS_IDS: set[int] = set()


def _wrap(fn, kind):
    def inner(*a, **k):
        res = fn(*a, **k)
        if REC.active:
            if kind == "mmseg":
                # inner_mm_seg_mlx(xh, trellis, k, cb, tab, nbr, n_rows=, tn_base=, ...)
                tr = a[1] if len(a) > 1 else k.get("trellis")
                tbase = k.get("tn_base", None)
                oute = k.get("out_e", None)
                REC.note(classify(kind, trellis=tr, tn_base=tbase, out_e=oute), res)
            else:
                REC.note(classify(kind), res)
        return res
    inner.__name__ = getattr(fn, "__name__", kind)
    return inner


def install_module_wrappers():
    """Wrap every production module/function that is an op-class boundary."""
    from mlx_lm.models.exl3 import exl3_moe as _em
    from mlx_lm.models.exl3 import exl3_linear as _el
    from mlx_lm.models.exl3 import gemv_metal as _gm
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx_lm.models.deepseek_v41 import sparse_attention as _sa
    from mlx_lm.models.deepseek_v41 import attention as _attn
    from mlx_lm.models.deepseek_v41 import indexer as _idx
    from mlx_lm.models.deepseek_v41 import compressor as _cmp
    from mlx_lm.models.deepseek_v41 import layers as _layers
    from mlx_lm.models.deepseek_v41 import moe as _moe
    from mlx_lm.models.deepseek_v41 import hc_fused as _hc
    from mlx_lm.models.deepseek_v41 import model as _modelmod
    from mlx_lm.models.deepseek_v41 import exl3_build as _build

    # low-level EXL3 GEMM launch (gate/up/down split) + the stacked SwitchGLU
    _gm.inner_mm_seg_mlx = _wrap(_gm.inner_mm_seg_mlx, "mmseg")
    if hasattr(_em, "inner_mm_seg_mlx"):
        _em.inner_mm_seg_mlx = _gm.inner_mm_seg_mlx
    _em.EXL3SwitchGLU.__call__ = _wrap(_em.EXL3SwitchGLU.__call__, "switchglu")
    SwitchGLU.__call__ = _wrap(SwitchGLU.__call__, "switchglu")
    # dense EXL3 linear (all attention qkv/o, shared expert, compressor, indexer)
    _el.EXL3Linear.__call__ = _wrap(_el.EXL3Linear.__call__, "exl3linear")
    # production dense = affine6 (AffineProj) for the non-shared-expert groups
    _build.AffineProj.__call__ = _wrap(_build.AffineProj.__call__, "affineproj")
    # sdpa / indexer / compressor / norms / gate / engram / hc
    _sa.sparse_attn = _wrap(_sa.sparse_attn, "sparse_attn")
    if hasattr(_attn, "sparse_attn"):
        _attn.sparse_attn = _sa.sparse_attn
    _idx.Indexer.__call__ = _wrap(_idx.Indexer.__call__, "indexer")
    _cmp.Compressor.__call__ = _wrap(_cmp.Compressor.__call__, "compressor")
    _layers.RMSNorm.__call__ = _wrap(_layers.RMSNorm.__call__, "norm")
    _moe.Gate.__call__ = _wrap(_moe.Gate.__call__, "gate")
    _build.LazyEngramTable.__call__ = _wrap(_build.LazyEngramTable.__call__, "engram")
    _hc.mixes_and_collapse = _wrap(_hc.mixes_and_collapse, "hc")
    _hc.hc_expand = _wrap(_hc.hc_expand, "hc")
    _modelmod.mixes_and_collapse = _hc.mixes_and_collapse
    _modelmod.hc_expand = _hc.hc_expand


# ===========================================================================
# dummy TP group (non-None => production takes the SHARDED branches; all_sum
# is stubbed above, so no RDMA occurs)
# ===========================================================================
class _DummyGroup:
    pass


# ===========================================================================
# builders
# ===========================================================================
def build_real_block(layer_id, rank=0, world=2):
    from mlx_lm.models.exl3.loader import Exl3Checkpoint
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    from mlx_lm.models.deepseek_v41.exl3_build import build_block

    ck = Exl3Checkpoint(MODEL_DIR)
    native = Exl3Checkpoint(NATIVE_DIR) if os.path.isdir(NATIVE_DIR) else ck
    args = ModelArgs.from_dict(ck.config)
    blk, rep = build_block(ck, args, layer_id, native=native,
                           rank=rank, world=world, group=_DummyGroup())
    sg = getattr(blk.ffn.experts, "_sg", None)
    if sg is not None and getattr(sg, "_dn_trellis", None) is not None:
        DN_TRELLIS_IDS.add(id(sg._dn_trellis))
    return blk, args, rep


def _mini_args(layer_id):
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    return ModelArgs(
        vocab_size=512, dim=256, n_layers=3, moe_inter_dim=64,
        n_heads=8, head_dim=64, rope_head_dim=16, q_lora_rank=64,
        o_lora_rank=64, o_groups=4, window_size=32,
        n_routed_experts=8, n_activated_experts=2, n_shared_experts=1,
        index_n_heads=4, index_head_dim=32, index_topk=8,
        hc_mult=4, hc_sinkhorn_iters=20,
        compress_ratios=(0, 2, 1), kv_source_layers=(1, 2),
        index_source_layers=(1, 2), candidate_source_layer=1,
        candidate_topk_blocks=4, candidate_block_size=8,
        max_seq_len=8192, engram_layer_ids=(), engram_num_embeddings=(),
    )


def build_smoke_block(layer_id):
    from mlx_lm.models.deepseek_v41.model import Block
    args = _mini_args(layer_id)
    blk = Block(layer_id, args)
    mx.eval(blk.parameters())
    return blk, args, {"smoke": True}


# ===========================================================================
# routing trace -> per-layer top-k expert ids
# ===========================================================================
_TRACE_CACHE = {}


def load_trace(path, layer_id, n_tokens, seed=20261011):
    if not path or not os.path.exists(path):
        return None
    if path not in _TRACE_CACHE:
        d = json.load(open(path))
        per = {}
        for rec in d["records"]:
            per.setdefault(int(rec[0]), []).append(rec[1])
        _TRACE_CACHE[path] = per
    rows = _TRACE_CACHE[path].get(layer_id)
    if not rows:
        return None
    rng = np.random.RandomState(seed)
    sel = rng.randint(0, len(rows), size=n_tokens)
    return np.stack([rows[i] for i in sel]).astype(np.int32)   # [n_tokens, topk]


class TraceGate(nn.Module):
    """Gate that keeps production SCORING but selects experts from the trace."""
    def __init__(self, base, idx_np):
        super().__init__()
        self._base = base
        self._idx = mx.array(idx_np) if idx_np is not None else None

    def __call__(self, x, image_mask=None):
        b = self._base
        if self._idx is None:
            return b(x, image_mask)
        sc = (x.astype(mx.float32) @ b.weight.astype(mx.float32).T) / b.gate_temp
        if b.score_func == "softmax":
            sc = mx.softmax(sc, axis=-1)
        elif b.score_func == "sigmoid":
            sc = mx.sigmoid(sc)
        else:
            sc = mx.sqrt(nn.softplus(sc))
        idx = self._idx
        w = mx.take_along_axis(sc, idx, axis=-1)
        if b.norm_topk_prob and b.topk > 1:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        w = w * b.route_scale
        if REC.active:
            REC.note("MoE routing gate (score+topk)", (w, idx))
        return w, idx


# ===========================================================================
# inputs + cache + shared state for ONE layer at a given start_pos
# ===========================================================================
def build_state(args, layer_id, start_pos, n, kind="real"):
    from mlx_lm.models.deepseek_v41.cache import ModelCache
    from mlx_lm.models.deepseek_v41.model import SharedState
    from mlx_lm.models.deepseek_v41.hyper_connections import make_identity_pre_mix

    end_pos = start_pos + n
    cap = int(end_pos) + CHUNK
    cache = ModelCache(args, bsz=1, max_seq_len=cap + CHUNK, initial_capacity=cap)
    shared = SharedState()

    ratio = args.compress_ratio(layer_id)
    is_kv_src = layer_id in args.kv_source_layers
    is_idx_src = layer_id in args.index_source_layers

    if not is_kv_src and ratio:
        src = args.kv_source_for(layer_id)
        shared.kv_src_cache = cache.layers[src]
        if is_idx_src:
            owners = [s for s in args.kv_source_layers if s <= layer_id]
            shared.index_src_cache = cache.layers[max(owners)]
        else:
            compress_len = end_pos // ratio
            offset = min(start_pos, args.window_size) + n
            rng = np.random.RandomState(7 + layer_id)
            cidx = offset + rng.randint(0, max(1, compress_len), size=(1, n, args.index_topk))
            shared.topk_idxs = mx.array(cidx.astype(np.int32))
    elif is_kv_src and is_idx_src:
        shared.index_src_cache = cache.layers[layer_id]   # owner reads its own keys

    hc = args.hc_mult
    x = mx.random.normal((1, n, hc, args.dim)).astype(mx.bfloat16)
    pre_mix = make_identity_pre_mix(1, n, hc)
    mx.eval(x, pre_mix)
    return dict(cache=cache, shared=shared, x=x, pre_mix=pre_mix,
                start_pos=start_pos, end_pos=end_pos, ratio=ratio,
                is_kv_src=is_kv_src, is_idx_src=is_idx_src)


def make_engram_inputs(blk, layer_id, args, n, kind="real"):
    if blk.engram is None:
        return None
    n_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
    hi = blk.engram.layer_hash_index
    n_emb = args.engram_num_embeddings[hi] if args.engram_num_embeddings else 4096
    rng = np.random.RandomState(11 + layer_id)
    ids = rng.randint(0, int(n_emb), size=(1, n, n_cols)).astype(np.int64)
    return mx.array(ids)


# ===========================================================================
# the layer replay (engram -> block)
# ===========================================================================
def run_layer(blk, state, eng_hash):
    x = state["x"]
    if eng_hash is not None:
        x = blk.engram(x, eng_hash, None)
    return blk(x, state["pre_mix"], state["start_pos"], state["cache"], state["shared"])


# ===========================================================================
# measurements
# ===========================================================================
def measure_unbracketed(blk, state, eng_hash, reps, label):
    from mlx_lm.profiler import unregister
    unregister()                                   # production lazy graph, unfenced
    REC.active = False
    for _ in range(2):
        _drain(run_layer(blk, state, eng_hash)); mx.synchronize()
    walls, gpus = [], []
    for _ in range(reps):
        mx.synchronize()
        if HAVE_GPU_TIME:
            mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        out = run_layer(blk, state, eng_hash)
        _drain(out)
        mx.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        gpus.append((mx.metal.gpu_time_ns() if HAVE_GPU_TIME else 0) / 1e6)
    return dict(label=label, mode="unbracketed",
                wall_ms=stats.median(walls), wall_min=min(walls),
                gpu_ms=stats.median(gpus), gpu_min=min(gpus),
                wall_all=walls, gpu_all=gpus)


def measure_bracketed(blk, state, eng_hash, reps, label):
    from mlx_lm.profiler import register, unregister
    register(CtxSpanHook())        # track innermost production span for classify()
    try:
        _drain(run_layer(blk, state, eng_hash)); mx.synchronize()
        per_rep = []
        walls = []
        for _ in range(reps):
            REC.begin()
            t0 = time.perf_counter()
            out = run_layer(blk, state, eng_hash)
            _drain(out)
            REC.final()
            walls.append((time.perf_counter() - t0) * 1e3)
            REC.end()
            per_rep.append(dict(REC.rec))
        names = set()
        for r in per_rep:
            names |= set(r)
        med = {nm: stats.median([r.get(nm, 0) for r in per_rep]) / 1e6 for nm in names}
        return dict(label=label, mode="bracketed", wall_ms=stats.median(walls),
                    gpu_total_ms=sum(med.values()), per_class_ms=med, reps=reps)
    finally:
        unregister()


def aggregate(bracketed):
    per = bracketed["per_class_ms"]
    total = bracketed["gpu_total_ms"] or 1e-9
    rows = [dict(cls=nm, ms=ms, pct=100.0 * ms / total)
            for nm, ms in sorted(per.items(), key=lambda kv: -kv[1])]
    return dict(total_ms=total, rows=rows)


def all_sum_bound(n_layers_total=40):
    """Bound the TP all_sum cost two ways (analytic payload/BW)."""
    payload_b = CHUNK * 5120 * 2                       # [tokens,5120] bf16
    per_layer_bytes = 2 * payload_b                    # attn all_sum + moe all_sum
    per_chunk_bytes = per_layer_bytes * n_layers_total
    calls_per_chunk = 2 * n_layers_total
    t_raw = per_chunk_bytes / (RDMA_GB_S_RAW * 1e9) * 1e3
    t_eff = per_chunk_bytes / (RDMA_GB_S_EFF * 1e9) * 1e3
    lat = calls_per_chunk * RDMA_LAT_US / 1e3
    return dict(per_chunk_MB=per_chunk_bytes / 1e6, calls_per_chunk=calls_per_chunk,
                lower_ms_raw=t_raw + lat, lower_ms_eff=t_eff + lat,
                raw_gbs=RDMA_GB_S_RAW, eff_gbs=RDMA_GB_S_EFF, lat_us=RDMA_LAT_US)


# ===========================================================================
# main
# ===========================================================================
def run_one(layer_id, ctx, reps, mode, trace_path, n_layers_total=40):
    if mode == "real":
        blk, args, rep = build_real_block(layer_id)
    else:
        blk, args, rep = build_smoke_block(layer_id)
    start_pos = ctx - CHUNK
    state = build_state(args, layer_id, start_pos, CHUNK, kind=mode)
    eng_hash = make_engram_inputs(blk, layer_id, args, CHUNK, kind=mode)

    if USE_TRACE_ROUTING and trace_path and hasattr(blk.ffn, "gate"):
        idx = load_trace(trace_path, layer_id, CHUNK)
        if idx is not None:
            blk.ffn.gate = TraceGate(blk.ffn.gate, idx)
            rep = dict(rep); rep["trace_routing"] = True

    ALLSUM["n"] = 0; ALLSUM["bytes"] = 0; ALLSUM["shapes"] = []

    # per-forward all_sum census (one instrumented pass)
    _drain(run_layer(blk, state, eng_hash)); mx.synchronize()
    allsum_per_forward = ALLSUM["n"]
    allsum_bytes_per_forward = ALLSUM["bytes"]
    allsum_shapes = list(ALLSUM["shapes"])

    unb = measure_unbracketed(blk, state, eng_hash, reps=reps + 2, label=f"L{layer_id}@{ctx}")
    brk = measure_bracketed(blk, state, eng_hash, reps=reps, label=f"L{layer_id}@{ctx}")
    agg = aggregate(brk)

    # closure is against the unbracketed WALL (production ground truth); the
    # unbracketed GPU-busy-SUM can EXCEED wall when unfenced buffers overlap.
    gap_vs_wall = ((agg["total_ms"] - unb["wall_ms"]) / unb["wall_ms"] * 100.0
                   if unb["wall_ms"] else float("nan"))
    gap_vs_busy = ((agg["total_ms"] - unb["gpu_ms"]) / unb["gpu_ms"] * 100.0
                   if unb["gpu_ms"] > 0 else float("nan"))
    busy_minus_wall = unb["gpu_ms"] - unb["wall_ms"]

    return dict(
        layer=layer_id, ctx=ctx, chunk=CHUNK, start_pos=start_pos, mode=mode,
        ratio=state["ratio"], is_kv_src=state["is_kv_src"], is_idx_src=state["is_idx_src"],
        build_report=rep,
        unbracketed=unb, bracketed_agg=agg, bracketed_total_ms=agg["total_ms"],
        sigma_bracketed_vs_whole_gap_pct=gap_vs_wall,
        gap_vs_wall_pct=gap_vs_wall, gap_vs_gpu_busy_pct=gap_vs_busy,
        gpu_vs_wall=dict(gpu_busy_ms=unb["gpu_ms"], wall_ms=unb["wall_ms"],
                         busy_minus_wall_ms=busy_minus_wall,
                         busy_over_wall=unb["gpu_ms"] / unb["wall_ms"] if unb["wall_ms"] else 0),
        allsum_per_forward=allsum_per_forward,
        allsum_bytes_per_forward=allsum_bytes_per_forward,
        allsum_shapes=allsum_shapes,
        allsum_bound=all_sum_bound(n_layers_total),
        live_unfenced_ms_per_chunk=(
            LIVE_UNFENCED_MS_PER_CHUNK_16K if ctx <= 16384 else None),
        env=_ENV, have_gpu_time=HAVE_GPU_TIME,
        mlx=mx.__version__, host=os.uname().nodename,
    )


def print_table(res):
    print("=" * 82)
    print(f"Task B op-class share table  layer={res['layer']} ctx={res['ctx']} "
          f"chunk={res['chunk']} ratio={res['ratio']} kv_src={res['is_kv_src']} "
          f"idx_src={res['is_idx_src']}")
    print("=" * 82)
    print(f"UNBRACKETED  GPU-busy-SUM {res['unbracketed']['gpu_ms']:9.2f} ms   "
          f"WALL {res['unbracketed']['wall_ms']:9.2f} ms   "
          f"busy-wall {res['gpu_vs_wall']['busy_minus_wall_ms']:+9.2f} ms "
          f"(busy/wall {res['gpu_vs_wall']['busy_over_wall']*100:.1f}%)")
    print(f"BRACKETED    S(per-class) {res['bracketed_total_ms']:9.2f} ms   "
          f"gap vs WALL {res['gap_vs_wall_pct']:+.1f}%   "
          f"gap vs busy {res['gap_vs_gpu_busy_pct']:+.1f}%")
    print("-" * 82)
    print(f"{'op-class':<40s} {'ms':>9s} {'share':>7s}")
    for r in res["bracketed_agg"]["rows"]:
        print(f"{r['cls']:<40s} {r['ms']:9.3f} {r['pct']:6.2f}%")
    print("-" * 82)
    print("all_sum stubbed: %d calls/forward, %.1f MB/forward (gpu_time DROPS RDMA buffers)"
          % (res["allsum_per_forward"], res["allsum_bytes_per_forward"] / 1e6))
    ab = res["allsum_bound"]
    print("all_sum bound (analytic, 40 layers/chunk): payload %.0f MB/chunk, "
          "calls %d -> %.0f ms @ %.0fGB/s_raw, %.0f ms @ %.0fGB/s_eff"
          % (ab["per_chunk_MB"], ab["calls_per_chunk"], ab["lower_ms_raw"],
             ab["raw_gbs"], ab["lower_ms_eff"], ab["eff_gbs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=int(os.environ.get("P20_PMP_LAYER", "3")))
    ap.add_argument("--layers", type=str, default=os.environ.get("P20_PMP_LAYERS", ""))
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("P20_PMP_CTX", "100352")))
    ap.add_argument("--reps", type=int, default=int(os.environ.get("P20_PMP_REPS", "5")))
    ap.add_argument("--mode", default=os.environ.get("P20_PMP_MODE", "real"))
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--out", default=os.environ.get("P20_PMP_OUT", ""))
    a = ap.parse_args()

    install_allsum_stub()
    install_module_wrappers()

    layers = [int(x) for x in a.layers.split(",") if x.strip()] or [a.layer]
    print(f"[taskB] mode={a.mode} mlx={mx.__version__} gpu_time={HAVE_GPU_TIME} "
          f"layers={layers} ctx={a.ctx} reps={a.reps} trace={a.trace or '-'}")
    results = []
    for L in layers:
        res = run_one(L, a.ctx, a.reps, a.mode, a.trace)
        print_table(res)
        results.append(res)
        mx.clear_cache()

    blob = dict(env=_ENV, mode=a.mode, results=results,
                mlx=mx.__version__, host=os.uname().nodename,
                task="B: per-op-class GPU-time attribution",
                note="all_sum stubbed (no RDMA on live link); gpu_time DROPS "
                     "JACCL RDMA/CPU-only buffers; bounded analytically + by difference")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(blob, fh, indent=1, default=float)
        print(f"[taskB] wrote {a.out}")
    return blob


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""p20_q1d_native_experts.py -- Round 1 / GATE 1: OFFLINE native-quant EXPERT microbench.

Question GATE 1 answers: can a NATIVE MLX quant (mx.quantize + mx.gather_qmm) drop-in
replace the EXL3 2.9bpw trellis fused expert kernel and save >=30% WALL on the
x40-layer expert stack, correctness clean, mem-fit OK?

Method (BENCH-ONLY, single node studio2, no engine touch, no writes under ~/repos/exo):
  * EXL3 baseline: load_experts(ckpt, 20, n_experts=384, rank=0, world=2,
    activation="silu_clamp") -> per-rank H=1152, D=5120, E=384, k=2 trellis.
    Timed at the MODULE boundary via mod(x[1,R,D], idx[1,R,6]).
  * TRUE fp16 weights: reconstruct_public_mlx(load_dense_layer(
    "layers.20.ffn.experts.{e}.w{1,2,3}")) -> [in,out]; slice intermediate to rank-0
    (cols 0:1152 for w1/w3, rows 0:1152 for w2); transpose to native [out,in]
    layout; Wgu=[2H,D]=[gate|up], Wdn=[D,H].
  * MEMORY-LEAN build: the node is production-live (only ~22 GiB free); weights are
    built ARM-BY-ARM by reconstructing one expert at a time, quantizing it
    (mx.quantize groups along the last dim, so per-expert == whole-tensor) and
    slice-assigning into the pre-allocated quantized buffers (mx in-place setitem).
  * Native forward follows the mlx_lm switch_layers BatchedSwitchGLU reference shape
    convention: 'sorted' (_gather_sort + gather_qmm(sorted_indices=True) + scatter-
    unsort, geometry x[N,1,in]/rhs[N]) and 'unsorted' (gather_qmm(sorted_indices=False),
    geometry x[R,1,1,in]/rhs[R,kk]).  Both fuse gate+up into one gather_qmm; an unfused
    (2-dispatch) variant is also run.  The faster LEGITIMATE variant is used for the gate.
  * Routing: REAL DSv4.1 gate on correlated draft rows.  Unique-expert sweep forces
    exactly {6,12,18,24} distinct experts.
  * A2 cache-busting: every timed rep re-draws a fresh hidden state (fresh routing);
    warm (back-to-back) AND flushed (>=1.5 GiB unrelated scratch read between reps,
    OUTSIDE the timed region) modes.  THE GATE NUMBER IS THE FLUSHED ONE.
  * Timing: median per-call WALL (perf_counter around eval+synchronize) AND per-op GPU
    ms (mx.metal.reset_gpu_time -> build -> eval -> synchronize -> gpu_time_ns).
  * Correctness: (A6a) reconstructed fp16 TRUE-W path vs EXL3 fused kernel on a 24-expert
    subset must be NEAR-EXACT; (A6b) each native arm vs EXL3 (cosine + rel L2).  A
    fast-but-wrong arm is DISQUALIFIED.

Env: MLX_GPU_TIME=1 must be set BEFORE `import mlx` (asserted).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time

os.environ.setdefault("MLX_GPU_TIME", "1")
os.environ.setdefault("MLX_DISPATCH_COUNT", "1")
os.environ.setdefault("EXL3_MOE_V2", "1")          # read at exl3_moe import
os.environ.setdefault("EXL3_MOE_MM", "1")
os.environ.setdefault("EXL3_MOE_FUSED", "1")
os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")  # match the production runner

assert os.environ.get("MLX_GPU_TIME") == "1", "run with MLX_GPU_TIME=1 set before import mlx"

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.expanduser("~/repos/exo"), "mlx-lm"))
from mlx_lm.models.exl3.loader import (  # noqa: E402
    Exl3Checkpoint, load_experts, load_dense_layer,
)
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx  # noqa: E402

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = 20
N_MOE = 40
TOPK = 6
E_EXP = 384
WORLD, RANK = 2, 0
D = 5120
H_FULL = 2304
H = H_FULL // WORLD
CORR = 0.03
CLAMP = float(os.environ.get("EXL3_MOE_CLAMP", "10.0") or 10.0)
ROUND_MS = 86.03
BASE_TPS = 34.299
FLUSH_BYTES = int(1.5 * (1 << 30))
N_REF = 24                                        # experts reconstructed for the A6a check


@mx.compile
def _act(g: mx.array, u: mx.array) -> mx.array:
    g = mx.minimum(g, CLAMP)
    u = mx.clip(u, -CLAMP, CLAMP)
    return nn.silu(g) * u


def make_rows(R, seed, corr=CORR):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(D).astype(np.float32)
    if R == 1:
        return base[None, :].copy()
    rows = base[None, :] + rng.standard_normal((R, D)).astype(np.float32) * corr
    rows[0] = base
    return rows


def route_idx(gw, gb, rows, topk, temp=1.0):
    s = rows @ gw.T / temp
    s = np.sqrt(np.logaddexp(s, 0.0))
    return np.argpartition(-(s + gb[None, :]), topk - 1, axis=-1)[:, :topk].astype(np.int32)


def force_idx(R, kk, n_unique, seed):
    rng = np.random.default_rng(seed + 4242)
    experts = rng.choice(E_EXP, n_unique, replace=False)
    flat = np.array([experts[s % n_unique] for s in range(R * kk)], dtype=np.int32)
    return flat.reshape(R, kk)


def reconstruct_expert(ckpt, e):
    """TRUE fp16 [2H,D]=[gate|up] and [D,H] for expert e (rank-0 intermediate slice)."""
    pre = f"layers.{LAYER}.ffn.experts.{e}"
    w1 = reconstruct_public_mlx(load_dense_layer(ckpt, pre + ".w1"))
    w3 = reconstruct_public_mlx(load_dense_layer(ckpt, pre + ".w3"))
    w2 = reconstruct_public_mlx(load_dense_layer(ckpt, pre + ".w2"))
    gu = mx.concatenate([w1[:, :H].T, w3[:, :H].T], axis=0)     # [2H,D]
    dn = w2[:H, :].T                                            # [D,H]
    return gu, dn


# ------------------------------------------------------------------ natives ---
def native_forward(x2d, idx, arm, variant="sorted", gu_fused=True):
    """Native drop-in at the module boundary: x[R,D] + idx[R,kk] -> per-slot y[R,kk,D]."""
    Rk, kk = idx.shape
    N = Rk * kk
    if variant == "sorted":
        flat = mx.array(idx.reshape(-1).astype(np.int32))
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        rs = flat[order]
        xs = x2d.reshape(Rk, 1, 1, D).flatten(0, -3)[order // kk]       # [N,1,D]
        if gu_fused:
            gu = mx.gather_qmm(xs, arm["gu_q"], arm["gu_s"], arm["gu_b"], rhs_indices=rs,
                               transpose=True, group_size=arm["gu_gs"], bits=arm["gu_bits"],
                               sorted_indices=True)
        else:
            g = mx.gather_qmm(xs, arm["gate_q"], arm["gate_s"], arm["gate_b"], rhs_indices=rs,
                              transpose=True, group_size=arm["gu_gs"], bits=arm["gu_bits"],
                              sorted_indices=True)
            u = mx.gather_qmm(xs, arm["up_q"], arm["up_s"], arm["up_b"], rhs_indices=rs,
                              transpose=True, group_size=arm["gu_gs"], bits=arm["gu_bits"],
                              sorted_indices=True)
            gu = mx.concatenate([g, u], axis=-1)                       # [N,1,2H]
        h = _act(gu[..., :H], gu[..., H:])
        y = mx.gather_qmm(h, arm["dn_q"], arm["dn_s"], arm["dn_b"], rhs_indices=rs,
                          transpose=True, group_size=arm["dn_gs"], bits=arm["dn_bits"],
                          sorted_indices=True)
        return y[inv].reshape(Rk, kk, D)
    i2 = mx.array(idx.astype(np.int32))
    if gu_fused:
        gu = mx.gather_qmm(x2d.reshape(Rk, 1, 1, D), arm["gu_q"], arm["gu_s"], arm["gu_b"],
                           rhs_indices=i2, transpose=True, group_size=arm["gu_gs"],
                           bits=arm["gu_bits"], sorted_indices=False).reshape(Rk, kk, 2 * H)
    else:
        g = mx.gather_qmm(x2d.reshape(Rk, 1, 1, D), arm["gate_q"], arm["gate_s"], arm["gate_b"],
                          rhs_indices=i2, transpose=True, group_size=arm["gu_gs"],
                          bits=arm["gu_bits"], sorted_indices=False)
        u = mx.gather_qmm(x2d.reshape(Rk, 1, 1, D), arm["up_q"], arm["up_s"], arm["up_b"],
                          rhs_indices=i2, transpose=True, group_size=arm["gu_gs"],
                          bits=arm["gu_bits"], sorted_indices=False)
        gu = mx.concatenate([g, u], axis=-1).reshape(Rk, kk, 2 * H)
    h = _act(gu[..., :H], gu[..., H:]).reshape(N, 1, 1, H)
    rr = mx.array(idx.reshape(-1).astype(np.int32)).reshape(N, 1)
    y = mx.gather_qmm(h, arm["dn_q"], arm["dn_s"], arm["dn_b"], rhs_indices=rr,
                      transpose=True, group_size=arm["dn_gs"], bits=arm["dn_bits"],
                      sorted_indices=False).reshape(N, D)
    return y.reshape(Rk, kk, D)


def true_forward(x2d, idx, Wgu, Wdn):
    Rk, kk = idx.shape
    N = Rk * kk
    rows = mx.array(idx.reshape(-1).astype(np.int32))
    r = mx.arange(N) // kk
    xr = x2d[r]
    gu = (xr[:, None, :] @ mx.take(Wgu, rows, axis=0).transpose(0, 2, 1)).reshape(N, 2 * H)
    h = _act(gu[:, :H], gu[:, H:])
    y = (h[:, None, :] @ mx.take(Wdn, rows, axis=0).transpose(0, 2, 1)).reshape(N, D)
    return y.reshape(Rk, kk, D)


# ------------------------------------------------------- streamed arm build ---
def build_arm_streamed(tag, ckpt, gu_bg, dn_bg, split_gu=False):
    """Memory-lean arm build: reconstruct+quantize expert-by-expert, in-place assign."""
    gb_, gs_ = gu_bg
    db_, ds_ = dn_bg
    gu0, dn0 = reconstruct_expert(ckpt, 0)
    parts = [("gate", gu0[:H]), ("up", gu0[H:])] if split_gu else [("gu", gu0)]
    store = {}
    for name, W in parts:
        q, s, b = mx.quantize(W, group_size=gs_, bits=gb_)
        store[name] = (mx.zeros((E_EXP,) + q.shape, q.dtype),
                       mx.zeros((E_EXP,) + s.shape, s.dtype),
                       mx.zeros((E_EXP,) + b.shape, b.dtype))
        store[name][0][0] = q; store[name][1][0] = s; store[name][2][0] = b
    q, s, b = mx.quantize(dn0, group_size=ds_, bits=db_)
    store["dn"] = (mx.zeros((E_EXP,) + q.shape, q.dtype),
                   mx.zeros((E_EXP,) + s.shape, s.dtype),
                   mx.zeros((E_EXP,) + b.shape, b.dtype))
    store["dn"][0][0] = q; store["dn"][1][0] = s; store["dn"][2][0] = b
    mx.eval(*[t for v in store.values() for t in v])
    del gu0, dn0
    for e in range(1, E_EXP):
        gu, dn = reconstruct_expert(ckpt, e)
        for name, W in (([("gate", gu[:H]), ("up", gu[H:])] if split_gu else [("gu", gu)])):
            q, s, b = mx.quantize(W, group_size=gs_, bits=gb_)
            v = store[name]; v[0][e] = q; v[1][e] = s; v[2][e] = b
        q, s, b = mx.quantize(dn, group_size=ds_, bits=db_)
        v = store["dn"]; v[0][e] = q; v[1][e] = s; v[2][e] = b
        del gu, dn
        if e % 96 == 0:
            mx.eval(*[t for vv in store.values() for t in vv])
    mx.eval(*[t for vv in store.values() for t in vv])
    arm = {"tag": tag, "gu_bits": gb_, "gu_gs": gs_, "dn_bits": db_, "dn_gs": ds_,
           "split_gu": split_gu}
    if split_gu:
        arm["gate_q"], arm["gate_s"], arm["gate_b"] = store["gate"]
        arm["up_q"], arm["up_s"], arm["up_b"] = store["up"]
        arm["gu_bytes"] = sum(t.nbytes for t in store["gate"] + store["up"])
    else:
        arm["gu_q"], arm["gu_s"], arm["gu_b"] = store["gu"]
        arm["gu_bytes"] = sum(t.nbytes for t in store["gu"])
    arm["dn_q"], arm["dn_s"], arm["dn_b"] = store["dn"]
    arm["dn_bytes"] = sum(t.nbytes for t in store["dn"])
    arm["total_bytes"] = arm["gu_bytes"] + arm["dn_bytes"]
    return arm


def free_arm(arm):
    arm.clear()
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


# ------------------------------------------------------------------ timing ----
def bracket_wall_gpu(fn, reps, warmup, flush_buf=None):
    def _eval(o):
        mx.eval(*o) if isinstance(o, (list, tuple)) else mx.eval(o)
    for _ in range(warmup):
        _eval(fn()); mx.synchronize()
    walls, gpus = [], []
    for _ in range(reps):
        if flush_buf is not None:
            mx.eval(mx.sum(flush_buf)); mx.synchronize()
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        _eval(fn()); mx.synchronize()
        t1 = time.perf_counter()
        walls.append((t1 - t0) * 1e3)
        gpus.append(mx.metal.gpu_time_ns() / 1e6)
    if len(walls) > 2:
        walls, gpus = walls[1:], gpus[1:]
    return walls, gpus


def med(xs):
    return float(statistics.median(xs))


def summarize(walls, gpus):
    return {"wall_ms": round(med(walls), 4), "gpu_ms": round(med(gpus), 4),
            "wall_min": round(min(walls), 4), "wall_max": round(max(walls), 4),
            "wall_spread": round(max(walls) - min(walls), 4), "n": len(walls)}


def run_modes(fn_factory, reps, warmup, flush_buf):
    out = {}
    for mode in ("warm", "flushed"):
        walls, gpus = bracket_wall_gpu(fn_factory(), reps, warmup,
                                       flush_buf if mode == "flushed" else None)
        out[mode] = summarize(walls, gpus)
    return out


def cos(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def metrics(a, b):
    """cosine, relative-L2 and max-abs error in float64 (fp32 overflows on these outputs)."""
    an = np.asarray(a).reshape(-1).astype(np.float64)
    bn = np.asarray(b).reshape(-1).astype(np.float64)
    return (cos(an, bn), float(np.linalg.norm(an - bn) / (np.linalg.norm(bn) + 1e-30)),
            float(np.abs(an - bn).max()))


def corr_vs_exl3(fwd, mod, gw, gb, R, kk, seeds):
    cs, rel, ma = [], [], []
    for sd in seeds:
        rows = make_rows(R, sd)
        idx = route_idx(gw, gb, rows, kk)
        a = fwd(mx.array(rows.astype(np.float16)), idx)
        b = mod(mx.array(rows.reshape(1, R, D).astype(np.float16)),
                mx.array(idx.reshape(1, R, kk))).reshape(R, kk, D)
        mx.eval(a, b)
        c, r, m = metrics(np.asarray(a), np.asarray(b))
        cs.append(c); rel.append(r); ma.append(m)
    return {"cosine_mean": float(np.mean(cs)), "cosine_min": float(np.min(cs)),
            "rel_l2_mean": float(np.mean(rel)), "rel_l2_max": float(np.max(rel)),
            "max_abs_err": float(np.max(ma)), "n_seeds": len(seeds)}


# ------------------------------------------------------------------ callers ---
class NativeCaller:
    def __init__(self, gw, gb, R, kk, seed0, arm, unique=None, variant="sorted", gu_fused=True):
        self.gw, self.gb, self.R, self.kk, self.seed0 = gw, gb, R, kk, seed0
        self.arm, self.unique, self.variant, self.gu_fused = arm, unique, variant, gu_fused
        self.i = 0

    def __call__(self):
        self.i += 1
        rows = make_rows(self.R, self.seed0 + 100 * self.i)
        idx = (route_idx(self.gw, self.gb, rows, self.kk) if self.unique is None
               else force_idx(self.R, self.kk, self.unique, self.seed0 + 100 * self.i))
        return native_forward(mx.array(rows.astype(np.float16)), idx, self.arm,
                              self.variant, self.gu_fused)


class Exl3Caller:
    def __init__(self, mod, gw, gb, R, kk, seed0, unique=None):
        self.mod, self.gw, self.gb, self.R, self.kk, self.seed0 = mod, gw, gb, R, kk, seed0
        self.unique = unique
        self.i = 0

    def __call__(self):
        self.i += 1
        rows = make_rows(self.R, self.seed0 + 100 * self.i)
        idx = (route_idx(self.gw, self.gb, rows, self.kk) if self.unique is None
               else force_idx(self.R, self.kk, self.unique, self.seed0 + 100 * self.i))
        x = mx.array(rows.reshape(1, self.R, D).astype(np.float16))
        return self.mod(x, mx.array(idx.reshape(1, self.R, self.kk))).reshape(self.R, self.kk, D)


class NativePrefillCaller:
    def __init__(self, gw, gb, S, kk, seed0, arm, variant):
        self.gw, self.gb, self.S, self.kk, self.seed0 = gw, gb, S, kk, seed0
        self.arm, self.variant = arm, variant
        self.i = 0

    def __call__(self):
        self.i += 1
        rows = make_rows(self.S, self.seed0 + 100 * self.i)
        idx = route_idx(self.gw, self.gb, rows, self.kk)
        return native_forward(mx.array(rows.reshape(self.S, D).astype(np.float16)), idx,
                              self.arm, self.variant, True)


class Exl3PrefillCaller:
    def __init__(self, mod, gw, gb, S, kk, seed0):
        self.mod, self.gw, self.gb, self.S, self.kk, self.seed0 = mod, gw, gb, S, kk, seed0
        self.i = 0

    def __call__(self):
        self.i += 1
        rows = make_rows(self.S, self.seed0 + 100 * self.i)
        idx = route_idx(self.gw, self.gb, rows, self.kk)
        x = mx.array(rows.reshape(1, self.S, D).astype(np.float16))
        return self.mod(x, mx.array(idx.reshape(1, self.S, self.kk)))


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-decode", type=int, default=15)
    ap.add_argument("--reps-prefill", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sweep", default="6,12,18,24")
    ap.add_argument("--out", default=os.environ.get("P20_OUT", "/tmp/q1d_native_experts.json"))
    ap.add_argument("--no-prefill", action="store_true")
    args = ap.parse_args()

    t_start = time.perf_counter()
    ckpt = Exl3Checkpoint(MODEL)
    gw = ckpt.np(f"layers.{LAYER}.ffn.gate.weight").astype(np.float32)
    gb = ckpt.np(f"layers.{LAYER}.ffn.gate.bias").astype(np.float32)
    t0 = time.perf_counter()
    mod = load_experts(ckpt, LAYER, n_experts=E_EXP, rank=RANK, world=WORLD,
                       activation="silu_clamp")
    t_load = time.perf_counter() - t0
    assert (mod.input_dims, mod.hidden_dims, mod.num_experts) == (D, H, E_EXP)
    print(f"[load] E={E_EXP} D={D} H={H} k={int(mod._k)} gu_tiles={mod._gu_tiles} "
          f"dn_tiles={mod._dn_tiles} in {t_load:.1f}s", flush=True)

    meta = {
        "bench": "p20_q1d_native_experts", "node": socket.gethostname(), "mlx": mx.__version__,
        "layer": LAYER, "n_moe_layers": N_MOE, "E": E_EXP, "D": D, "H_per_rank": H,
        "H_full": H_FULL, "topk": TOPK, "rank": RANK, "world": WORLD, "k_trellis": int(mod._k),
        "reps_decode": args.reps_decode, "reps_prefill": args.reps_prefill, "warmup": args.warmup,
        "n_seeds": args.seeds, "sweep": args.sweep, "route_scale": 1.5, "gate_temp": 1.0,
        "corr": CORR, "clamp": CLAMP, "flush_bytes": FLUSH_BYTES, "load_s": round(t_load, 1),
        "round_ms_ref": ROUND_MS, "base_tps_ref": BASE_TPS, "n_ref_experts": N_REF,
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "env": {k: os.environ.get(k) for k in
                ("EXL3_MOE_V2", "EXL3_MOE_FUSED", "EXL3_MOE_MM", "EXL3_MOE_CLAMP",
                 "MLX_GPU_TIME", "MLX_DISPATCH_COUNT", "MTL_DISABLE_TIMEOUT")},
    }
    result = {"meta": meta, "correctness": {}, "exl3_baseline": {}, "arms": {}}

    flush_buf = mx.zeros((FLUSH_BYTES // 2,), dtype=mx.float16)
    mx.eval(flush_buf)
    seeds_dec = [1000 + s for s in range(args.seeds)]
    R, KK = 4, TOPK
    sweep_pts = [int(x) for x in args.sweep.split(",")]

    # ---- A6a on a small expert subset (validates reconstruction + orientation) ----
    print(f"[A6a] reconstruct {N_REF} experts + TRUE-fp16 vs EXL3 fused kernel", flush=True)
    t0 = time.perf_counter()
    gu_list, dn_list = [], []
    for e in range(N_REF):
        gu, dn = reconstruct_expert(ckpt, e)
        gu_list.append(gu); dn_list.append(dn)
    Wgu_sub = mx.stack(gu_list); Wdn_sub = mx.stack(dn_list)
    mx.eval(Wgu_sub, Wdn_sub)
    del gu_list, dn_list
    print(f"[A6a] recon {N_REF} experts in {time.perf_counter()-t0:.1f}s", flush=True)
    a6a_cos, a6a_rel, a6a_ma = [], [], []
    for sd in seeds_dec:
        rows = make_rows(R, sd)
        idx = (np.arange(R * KK).reshape(R, KK) % N_REF).astype(np.int32)
        a = true_forward(mx.array(rows.astype(np.float16)), idx, Wgu_sub, Wdn_sub)
        b = mod(mx.array(rows.reshape(1, R, D).astype(np.float16)),
                mx.array(idx.reshape(1, R, KK))).reshape(R, KK, D)
        mx.eval(a, b)
        c, r, m = metrics(np.asarray(a), np.asarray(b))
        a6a_cos.append(c); a6a_rel.append(r); a6a_ma.append(m)
    a6a = {"cosine_mean": float(np.mean(a6a_cos)), "cosine_min": float(np.min(a6a_cos)),
           "rel_l2_mean": float(np.mean(a6a_rel)), "rel_l2_max": float(np.max(a6a_rel)),
           "max_abs_err": float(np.max(a6a_ma)), "n_seeds": len(seeds_dec),
           "n_ref_experts": N_REF,
           "note": "forced indices in [0,N_REF); validates reconstruct/transpose/sign/activation"}
    near_exact = a6a["cosine_min"] > 0.999 and a6a["rel_l2_max"] < 0.05
    a6a["near_exact"] = bool(near_exact)
    result["correctness"]["a6a_true_fp16_vs_exl3"] = a6a
    print(f"[A6a] cosine_min={a6a['cosine_min']:.6f} rel_l2_max={a6a['rel_l2_max']:.2e} "
          f"max_abs={a6a['max_abs_err']:.3e} near_exact={near_exact}", flush=True)
    if not near_exact:
        print("!!!! A6a NOT NEAR-EXACT -> reconstruction/transpose BUG; results DISQUALIFIED !!!!",
              flush=True)
    del Wgu_sub, Wdn_sub
    free_arm({})

    # ---- EXL3 baseline (same session) --------------------------------------
    ex = {}
    for name, uniq in (("R4_natural", None), ("R4_u24", 24)):
        ex[name] = run_modes(lambda u=uniq: Exl3Caller(mod, gw, gb, R, KK, 7000, u),
                             args.reps_decode, args.warmup, flush_buf)
    ex["R1"] = run_modes(lambda: Exl3Caller(mod, gw, gb, 1, KK, 7000, None),
                         args.reps_decode, args.warmup, flush_buf)
    result["exl3_baseline"] = ex
    for k in ("R4_natural", "R4_u24", "R1"):
        print(f"[exl3 {k}] warm wall={ex[k]['warm']['wall_ms']:.4f} gpu={ex[k]['warm']['gpu_ms']:.4f}"
              f" | flushed wall={ex[k]['flushed']['wall_ms']:.4f} gpu={ex[k]['flushed']['gpu_ms']:.4f}",
              flush=True)

    # ---- anchor arm ---------------------------------------------------------
    print("[arm] building q4g64 (fused gu, streamed)", flush=True)
    t0 = time.perf_counter()
    anchor = build_arm_streamed("q4g64", ckpt, (4, 64), (4, 64), split_gu=False)
    print(f"[arm] q4g64 built in {time.perf_counter()-t0:.0f}s total {anchor['total_bytes']/1e9:.2f} GB "
          f"(gu {anchor['gu_bytes']/1e6:.0f} MB, dn {anchor['dn_bytes']/1e6:.0f} MB)", flush=True)
    var_corr, var_time = {}, {}
    for var in ("sorted", "unsorted"):
        var_corr[var] = corr_vs_exl3(
            lambda x, i, v=var: native_forward(x, i, anchor, v, True), mod, gw, gb, R, KK, seeds_dec)
        print(f"[A6b q4g64 {var}] cosine_mean={var_corr[var]['cosine_mean']:.6f} "
              f"rel_l2_mean={var_corr[var]['rel_l2_mean']:.2e}", flush=True)
        var_time[var] = run_modes(
            lambda v=var: NativeCaller(gw, gb, R, KK, 7000, anchor, None, v, True),
            args.reps_decode, args.warmup, flush_buf)
        print(f"[q4g64 {var} R4] warm wall={var_time[var]['warm']['wall_ms']:.4f} "
              f"flushed wall={var_time[var]['flushed']['wall_ms']:.4f}", flush=True)
    valid = [v for v in ("sorted", "unsorted") if var_corr[v]["cosine_min"] > 0.99] or ["sorted"]
    VARIANT = min(valid, key=lambda v: var_time[v]["flushed"]["wall_ms"])
    result["correctness"]["a6b_variant_probe"] = {
        **{f"{v}_cosine_mean": var_corr[v]["cosine_mean"] for v in var_corr},
        **{f"{v}_rel_l2_mean": var_corr[v]["rel_l2_mean"] for v in var_corr},
        **{f"{v}_R4_flushed_wall_ms": var_time[v]["flushed"]["wall_ms"] for v in var_time},
        "chosen_variant": VARIANT,
    }
    print(f"[variant] chosen = {VARIANT}", flush=True)
    anchor_rec = {"spec": {"tag": "q4g64", "gu_bits_group": [4, 64], "dn_bits_group": [4, 64],
                           "gu_fused": True, "variant": VARIANT},
                  "bytes": {"gu_per_expert": anchor["gu_bytes"] // E_EXP,
                            "dn_per_expert": anchor["dn_bytes"] // E_EXP,
                            "total": anchor["total_bytes"]},
                  "variants": {v: {"correctness": var_corr[v], "R4_natural": var_time[v]}
                               for v in var_corr}}

    def full_arm(arm, rec, variant):
        rec["R4_natural"] = run_modes(
            lambda: NativeCaller(gw, gb, R, KK, 7000, arm, None, variant, True),
            args.reps_decode, args.warmup, flush_buf)
        rec["R1"] = run_modes(
            lambda: NativeCaller(gw, gb, 1, KK, 7000, arm, None, variant, True),
            args.reps_decode, args.warmup, flush_buf)
        rec["sweep"] = {}
        for u in sweep_pts:
            cell = run_modes(
                lambda uu=u: NativeCaller(gw, gb, R, KK, 7000, arm, uu, variant, True),
                args.reps_decode, args.warmup, flush_buf)
            ub = (arm["gu_bytes"] + arm["dn_bytes"]) // E_EXP * u
            for mode in ("warm", "flushed"):
                cell[mode]["bytes_moved"] = ub
                cell[mode]["eff_gbs"] = round(ub / (cell[mode]["wall_ms"] * 1e-3) / 1e9, 2)
            rec["sweep"][f"u{u}"] = cell
        rec["correctness_vs_exl3"] = corr_vs_exl3(
            lambda x, i: native_forward(x, i, arm, variant, True), mod, gw, gb, R, KK, seeds_dec)
        return rec

    full_arm(anchor, anchor_rec, VARIANT)
    result["arms"]["q4g64"] = anchor_rec
    _print_arm("q4g64", anchor_rec)

    exl3_24_fl = ex["R4_u24"]["flushed"]["wall_ms"]
    exl3_nat_fl = ex["R4_natural"]["flushed"]["wall_ms"]
    nat24_fl = anchor_rec["sweep"]["u24"]["flushed"]["wall_ms"]
    save24 = 1.0 - nat24_fl / exl3_24_fl
    save_nat = 1.0 - nat24_fl / exl3_nat_fl
    result["gate1"] = {
        "exl3_R4_natural_flushed_wall_ms": exl3_nat_fl,
        "exl3_R4_u24_flushed_wall_ms": exl3_24_fl,
        "native_q4g64_u24_flushed_wall_ms": nat24_fl,
        "pct_saved_vs_exl3_u24": round(save24 * 100, 2),
        "pct_saved_vs_exl3_natural": round(save_nat * 100, 2),
        "gate_pct_saved": round(save24 * 100, 2),
        "gate_pass": bool(save24 >= 0.30), "threshold_pct": 30.0,
        "headline_condition": "native q4g64 @ 24-unique FLUSHED vs EXL3 @ 24-unique FLUSHED",
    }
    print(f"[GATE1] native(24u,fl)={nat24_fl:.4f} vs exl3(24u,fl)={exl3_24_fl:.4f} -> "
          f"saved {save24*100:.1f}% pass={result['gate1']['gate_pass']} | "
          f"vs exl3 natural(fl)={exl3_nat_fl:.4f} saved {save_nat*100:.1f}%", flush=True)

    # ---- unfused-gu comparison (same variant) ------------------------------
    print("[arm] building q4g64 unfused-gu (streamed)", flush=True)
    anchor_unf = build_arm_streamed("q4g64_unfusedgu", ckpt, (4, 64), (4, 64), split_gu=True)
    rec_unf = {"spec": {"tag": "q4g64_unfusedgu", "gu_bits_group": [4, 64],
                        "dn_bits_group": [4, 64], "gu_fused": False, "variant": VARIANT},
               "bytes": anchor_rec["bytes"]}
    rec_unf["R4_natural"] = run_modes(
        lambda: NativeCaller(gw, gb, R, KK, 7000, anchor_unf, None, VARIANT, False),
        args.reps_decode, args.warmup, flush_buf)
    rec_unf["correctness_vs_exl3"] = corr_vs_exl3(
        lambda x, i: native_forward(x, i, anchor_unf, VARIANT, False), mod, gw, gb, R, KK, seeds_dec)
    result["arms"]["q4g64_unfusedgu"] = rec_unf
    print(f"[q4g64_unfusedgu] warm wall={rec_unf['R4_natural']['warm']['wall_ms']:.4f} "
          f"flushed wall={rec_unf['R4_natural']['flushed']['wall_ms']:.4f}", flush=True)
    free_arm(anchor_unf)

    # ---- prefill S=2048 no-regress check (anchor still resident) -----------
    if not args.no_prefill:
        S = 2048
        print(f"[prefill] S={S} (EXL3 _prefill vs native q4g64)", flush=True)
        pf = {"exl3": run_modes(lambda: Exl3PrefillCaller(mod, gw, gb, S, KK, 9000),
                                args.reps_prefill, args.warmup, flush_buf),
              "native_q4g64": run_modes(
                  lambda: NativePrefillCaller(gw, gb, S, KK, 9000, anchor, VARIANT),
                  args.reps_prefill, args.warmup, flush_buf)}
        pf["no_regress_warm"] = bool(pf["native_q4g64"]["warm"]["wall_ms"]
                                     <= pf["exl3"]["warm"]["wall_ms"] * 1.05)
        result["prefill_S2048_check"] = pf
        for k in ("exl3", "native_q4g64"):
            print(f"[prefill {k}] warm wall={pf[k]['warm']['wall_ms']:.3f} gpu={pf[k]['warm']['gpu_ms']:.3f}"
                  f" | flushed wall={pf[k]['flushed']['wall_ms']:.3f}", flush=True)
    free_arm(anchor)

    # ---- further arms only if anchor passes --------------------------------
    arm_specs = [("q5g64", (5, 64), (5, 64), ""), ("q4gu_q6dn", (4, 64), (6, 64), ""),
                 ("q6g64", (6, 64), (6, 64), ""), ("q4g32", (4, 32), (4, 32), "conditional")]
    if result["gate1"]["gate_pass"]:
        for (tag, gu, dn, flag) in arm_specs:
            if flag == "conditional":
                q5 = result["arms"].get("q5g64", {}).get("sweep", {}).get("u24", {}).get("flushed", {})
                if q5.get("wall_ms", 1e9) <= nat24_fl:
                    print(f"[arm] skip {tag} (q5g64 already <= anchor)", flush=True)
                    continue
            print(f"[arm] building {tag} (streamed)", flush=True)
            t0 = time.perf_counter()
            a = build_arm_streamed(tag, ckpt, gu, dn, split_gu=False)
            print(f"[arm] {tag} built in {time.perf_counter()-t0:.0f}s total {a['total_bytes']/1e9:.2f} GB",
                  flush=True)
            rec = {"spec": {"tag": tag, "gu_bits_group": list(gu), "dn_bits_group": list(dn),
                            "gu_fused": True, "variant": VARIANT},
                   "bytes": {"gu_per_expert": a["gu_bytes"] // E_EXP,
                             "dn_per_expert": a["dn_bytes"] // E_EXP, "total": a["total_bytes"]}}
            full_arm(a, rec, VARIANT)
            result["arms"][tag] = rec
            _print_arm(tag, rec)
            free_arm(a)
    else:
        print("[arm] anchor q4g64 FAILED the 30% gate -> STOP extra arms", flush=True)

    # ---- implied round arithmetic ------------------------------------------
    saved_x40 = (exl3_24_fl - nat24_fl) * N_MOE
    new_round = ROUND_MS - saved_x40
    result["implied"] = {
        "exl3_x40_wall_ms": round(exl3_24_fl * N_MOE, 3),
        "native_x40_wall_ms": round(nat24_fl * N_MOE, 3),
        "saved_x40_ms": round(saved_x40, 3), "implied_round_ms": round(new_round, 3),
        "net_tps_a3pct": round(0.97 * BASE_TPS * ROUND_MS / new_round, 3),
        "net_tps_a1pct": round(0.99 * BASE_TPS * ROUND_MS / new_round, 3),
        "formula": "net_tps = (1-a) * BASE_TPS * 86.03/(86.03 - saved_x40)",
        "gate_kernel_ratio_gpu": round(
            anchor_rec["sweep"]["u24"]["flushed"]["gpu_ms"] / ex["R4_u24"]["flushed"]["gpu_ms"], 3),
    }
    result["meta"]["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"[done] wrote {args.out} in {result['meta']['elapsed_s']}s", flush=True)


def _print_arm(tag, rec):
    for mode in ("warm", "flushed"):
        g = rec["R4_natural"][mode]
        su = rec["sweep"]["u24"][mode]
        print(f"[{tag} R4 {mode:7s}] wall={g['wall_ms']:.4f} gpu={g['gpu_ms']:.4f} "
              f"| u24 {mode} wall={su['wall_ms']:.4f} gbs={su.get('eff_gbs')}", flush=True)
    c = rec.get("correctness_vs_exl3")
    if c:
        print(f"[{tag} A6b] cosine_mean={c['cosine_mean']:.6f} rel_l2_mean={c['rel_l2_mean']:.2e}",
              flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""p20_q1c_opclass.py -- Q1c OFFLINE per-op-class GPU-time split of the DSv4.1
MoE EXPERT block (BENCH-ONLY, studio2).  Answers ONE question: at production
decode (draft R=1 / verify R=4) and prefill (S=2048) shapes, is the expert block
dominated by the expert GEMM KERNEL (ALU/issue) or by non-kernel overhead
(gather / scatter / routing)?

Mechanism (from q5/proto_gpu_time.py + proto_granularity.py):
  * MLX_GPU_TIME=1 must be set BEFORE `import mlx.core`.
  * per stage: mx.metal.reset_gpu_time(); build stage; mx.eval(out);
    mx.synchronize(); read mx.metal.gpu_time_ns().  One eval of a fused graph
    commits buffers whose GPUStart/GPUEnd spans SUM into the counter, so
    eval-ing each sub-stage separately brackets each sub-stage's GPU ms.
  * mx.metal.dispatch_count() cross-checked per stage.

Two arms (env read at exl3_moe import time, so each arm is its OWN process):
  * fused   (EXL3_MOE_FUSED=1, production): R=1/R=4 -> _decode_fused2 (3 ops);
            S=2048 -> _prefill (segmented).  Fused kernels hide act/gather
            internally -> decomposed by calling the internal primitives.
  * unfused (EXL3_MOE_FUSED=0): R=1 -> _decode (mapped GEMV, fully split);
            R=4 -> _prefill (the fused R<=8 decode branch is disabled);
            S=2048 -> _prefill.

Router (deepseek_v41/moe.py Gate): sqrt(softplus(x@W.T/temp)); bias-only reorder
of argpartition top-k; weights from UNBIASED scores; /(sum+1e-20); *route_scale.

Nothing under ~/repos/exo on the node is modified.  No HTTP, no engine touch.
"""
from __future__ import annotations
import os, sys, json, time, socket, statistics, subprocess, argparse

# --------------------------------------------------------------------------
# DRIVER: when P20_ARM is unset, this process launches itself twice (one per
# arm, since EXL3_MOE_FUSED is read at import time) and merges the two JSONs.
# --------------------------------------------------------------------------
def _driver(out_path, reps_decode, reps_prefill, warmup, seeds, prefill512):
    here = os.path.abspath(__file__)
    parts = {}
    for arm in ("fused", "unfused"):
        tmp = f"/tmp/q1c_{arm}.json"
        env = dict(os.environ)
        env["P20_ARM"] = arm
        env["MLX_GPU_TIME"] = "1"
        env["MLX_DISPATCH_COUNT"] = "1"
        env["P20_OUT"] = tmp
        cmd = [sys.executable, here,
               "--reps-decode", str(reps_decode),
               "--reps-prefill", str(reps_prefill),
               "--warmup", str(warmup), "--seeds", str(seeds)]
        if prefill512:
            cmd.append("--prefill512")
        print(f"[driver] === arm={arm} -> {tmp} ===", flush=True)
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"[driver] arm={arm} FAILED rc={r.returncode}", flush=True)
            sys.exit(r.returncode)
        with open(tmp) as f:
            parts[arm] = json.load(f)
    merged = _merge(parts)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print("[driver] wrote", out_path, flush=True)


def _merge(parts):
    fused = parts["fused"]
    unfused = parts["unfused"]
    meta = fused["meta"]
    meta["arms"] = ["fused", "unfused"]
    meta["unfused_note"] = (
        "EXL3_MOE_FUSED=0: R=1 -> _decode (mapped GEMV, fully split); R=4 -> "
        "_prefill (the R<=8 fused decode branch is disabled, so verify falls "
        "to the prefill sort/table machinery); S=2048 -> _prefill."
    )
    sh = fused["shapes"]; su = unfused["shapes"]
    # ---- bucket view at the production verify shape (fused arm) -----------
    v = sh["R4_topk6"]
    st = v["stages"]
    gu = st.get("gateup_kernel", 0.0)
    dn = st.get("down_kernel(act_inline)", 0.0)
    kernel = gu + dn
    router = fused["router"]["R4"]["gpu_ms"]
    gather_scatter = (st.get("prep_gather_hadamard", 0.0)
                      + v.get("unattributed_ms", 0.0))
    total = v["whole_gpu_ms"]
    N_MOE = 40
    verdict = {
        "verify_shape": "R=4 topk6 (gamma+1, gamma=3)",
        "fused_kernel_ms_gateup_plus_down": round(kernel, 4),
        "fused_prep_gather_hadamard_ms": round(st.get("prep_gather_hadamard", 0.0), 4),
        "fused_unattributed_ms": round(v.get("unattributed_ms", 0.0), 4),
        "router_ms": round(router, 4),
        "whole_module_ms": round(total, 4),
        "kernel_x40_ms": round(kernel * N_MOE, 3),
        "whole_x40_ms": round(total * N_MOE, 3),
        "kernel_lt_25ms_x40": bool(kernel * N_MOE < 25.0),
        "verdict": ("KERNEL-DOMINATED" if kernel >= 0.5 * total else
                    "NON-KERNEL-DOMINATED (gather/scatter/routing >= kernel)"),
        "kernel_share_of_whole": round(kernel / total, 3) if total else None,
    }
    # unfused-arm attribution of the non-kernel share (verify falls to _prefill)
    uv = su.get("R4_topk6", {})
    verdict["unfused_arm_R4_path"] = uv.get("path", "?")
    return {
        "meta": meta,
        "fused": fused,
        "unfused": unfused,
        "totals_x40": _totals(fused, unfused),
        "verdict": verdict,
    }


def _totals(fused, unfused):
    N = 40
    out = {"n_moe_layers": N}
    for arm, part in (("fused", fused), ("unfused", unfused)):
        d = {}
        for shape, cell in part["shapes"].items():
            if "whole_gpu_ms" in cell:
                d[shape] = {"whole_x40_ms": round(cell["whole_gpu_ms"] * N, 3)}
        r = part.get("router", {})
        for shape, cell in r.items():
            d.setdefault(shape, {})["router_x40_ms"] = round(cell["gpu_ms"] * N, 3)
        out[arm] = d
    return out


if os.environ.get("P20_ARM") is None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-decode", type=int, default=15)
    ap.add_argument("--reps-prefill", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prefill512", action="store_true")
    ap.add_argument("--out", default=os.environ.get("P20_OUT", "/tmp/pricing/q1c/q1c_opclass.json"))
    a = ap.parse_args()
    _driver(a.out, a.reps_decode, a.reps_prefill, a.warmup, a.seeds, a.prefill512)
    sys.exit(0)

# --------------------------------------------------------------------------
# CHILD: one arm.  Set exl3 env BEFORE importing the tree.
# --------------------------------------------------------------------------
ARM = os.environ["P20_ARM"]
os.environ.setdefault("EXL3_MOE_V2", "1")
os.environ.setdefault("EXL3_MOE_MM", "1")
os.environ["EXL3_MOE_FUSED"] = "1" if ARM == "fused" else "0"

assert os.environ.get("MLX_GPU_TIME") == "1", "run with MLX_GPU_TIME=1"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

REPO = os.path.expanduser("~/repos/exo")
MLXLM = os.path.join(REPO, "mlx-lm")
sys.path.insert(0, MLXLM)
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts  # noqa: E402
import mlx_lm.models.exl3.exl3_moe as EM  # noqa: E402

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = 20
N_MOE = 40
D_EXP, E_EXP, TOPK = 5120, 384, 6
CORR = 0.03


def make_rows(R, D, seed):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(D).astype(np.float32)
    if R == 1:
        return base[None, :].copy()
    rows = base[None, :] + rng.standard_normal((R, D)).astype(np.float32) * CORR
    rows[0] = base
    return rows


def route_idx(gw, gb, rows, topk, temp=1.0):
    """Real DSv4.1 gate: sqrt(softplus(x@W.T/temp)); bias reorders top-k."""
    s = rows @ gw.T / temp
    s = np.sqrt(np.logaddexp(s, 0.0))
    biased = s + gb[None, :]
    idx = np.argpartition(-biased, topk - 1, axis=-1)[:, :topk]
    return idx.astype(np.int32)


def bracket(fn, reps, warmup):
    """GPU ms per stage: reset -> build+eval -> synchronize -> read."""
    for _ in range(warmup):
        o = fn()
        mx.eval(*o) if isinstance(o, (list, tuple)) else mx.eval(o)
        mx.synchronize()
    s = []
    for _ in range(reps):
        mx.metal.reset_gpu_time()
        o = fn()
        mx.eval(*o) if isinstance(o, (list, tuple)) else mx.eval(o)
        mx.synchronize()
        s.append(mx.metal.gpu_time_ns() / 1e6)
    return s


def med(xs):
    return float(statistics.median(xs))


# ---------------- router (deepseek_v41/moe.py Gate) ------------------------
def bench_router(gw, gb, xf32, topk, route_scale, temp, reps, warmup):
    gwm = mx.array(gw); gbm = mx.array(gb)
    R = xf32.shape[0]
    def build():
        s = (xf32 @ gwm.T) / temp
        s = mx.sqrt(nn.softplus(s))
        biased = s + gbm
        inds = mx.argpartition(-biased, topk - 1, axis=-1)[..., :topk]
        w = mx.take_along_axis(s, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        w = w * route_scale
        return [w, inds]
    return bracket(build, reps, warmup)


# ---------------- fused decode (R=1 or R=4) --------------------------------
def bench_fused_decode(mod, x2d, indices, reps, warmup):
    kA, kB = mod._kernels2()
    R, kk = int(indices.shape[0]), int(indices.shape[1])
    E_sel = R * kk
    D, H = mod.input_dims, mod.hidden_dims
    sel = indices.reshape(-1)
    sel_u = sel.astype(mx.uint32)
    suh_sel = mod._gu_suh[sel].reshape(E_sel * 2, D)
    x_rep = mx.broadcast_to(x2d[:, None, :], (R, kk * 2, D)).reshape(E_sel * 2, D)
    dims = mx.array([D // 16, mod._gu_tiles,
                     int(mod._gu_trellis.shape[1]), mod.num_experts], dtype=mx.uint32)
    trellis_u32 = mod._gu_trellis.reshape(-1).view(mx.uint32)
    perm = EM._fwd_perm_u32()
    GT = EM._GEM_THREADS
    A2 = EM._A2_TILES
    gridd = (E_sel * (2 * mod._gu_tiles // A2) * GT, 1, 1)
    dims_b = mx.array([H // 16, mod._dn_tiles,
                       int(mod._dn_trellis.shape[1]), mod.num_experts], dtype=mx.uint32)
    dn_trellis_u32 = mod._dn_trellis.reshape(-1).view(mx.uint32)

    def b_prep():
        return EM._rows_prep()(x_rep, suh_sel)
    xh = b_prep(); mx.eval(xh)

    def b_gateup():
        return kA(inputs=[xh.reshape(-1), trellis_u32, perm, sel_u, dims],
                  template=[("T", mx.float16)], grid=gridd,
                  threadgroup=(GT, 1, 1),
                  output_shapes=[(E_sel * 2 * H,)], output_dtypes=[mx.float16])[0]
    ygu = b_gateup(); mx.eval(ygu)

    def b_down():
        return kB(inputs=[ygu, dn_trellis_u32, perm, sel_u,
                          mod._gu_svh.reshape(-1), mod._dn_suh.reshape(-1),
                          mod._dn_svh.reshape(-1), dims_b],
                  template=[("T", mx.float16)],
                  grid=(E_sel * (mod._dn_tiles // 8) * GT, 1, 1),
                  threadgroup=(GT, 1, 1),
                  output_shapes=[(E_sel * D,)], output_dtypes=[mx.float16])[0]

    st = {}
    st["prep_gather_hadamard"] = med(bracket(b_prep, reps, warmup))
    st["gateup_kernel"] = med(bracket(b_gateup, reps, warmup))
    st["down_kernel(act_inline)"] = med(bracket(b_down, reps, warmup))
    return st


# ---------------- unfused decode (R=1 -> _decode, mapped GEMV) -------------
def bench_unfused_decode(mod, x2d, sel, reps, warmup):
    E_sel = int(sel.shape[0])
    gt = mod._gu_tiles
    D, H = mod.input_dims, mod.hidden_dims
    ar = mx.arange(gt, dtype=mx.uint32)
    sel_u = sel[:, None].astype(mx.uint32)
    tile_map = mx.concatenate(
        [sel_u * gt + ar, (mod.num_experts + sel_u) * gt + ar], axis=1).reshape(-1)
    ar_gu = mx.arange(2 * gt, dtype=mx.uint32)
    proj = (ar_gu >= gt).astype(mx.uint32)
    tile_sub = (mx.arange(E_sel, dtype=mx.uint32)[:, None] * 2 + proj).reshape(-1)
    suh_sel = mod._gu_suh[sel].reshape(E_sel * 2, D)
    xb = mx.broadcast_to(x2d, (E_sel * 2, D))

    def b_gu_prep():
        return EM._rows_prep()(xb, suh_sel)
    xh = b_gu_prep(); mx.eval(xh)

    def b_gu_kernel():
        return EM._mapped_gemv(xh, mod._gu_trellis, mod._k, mod._cb,
                               tile_map, tile_sub).reshape(E_sel, 2 * H)
    y = b_gu_kernel(); mx.eval(y)

    def b_gu_finish():
        return EM._rows_finish()(y.astype(mx.float16), mod._gu_svh[sel])
    yf = b_gu_finish(); mx.eval(yf)

    def b_act():
        g, u = mx.split(yf, 2, axis=-1)
        return EM._moe_gate_activation(g, u, mod._activation)
    h = b_act(); mx.eval(h)

    ar_dn = mx.arange(mod._dn_tiles, dtype=mx.uint32)
    tile_map_d = (sel[:, None].astype(mx.uint32) * mod._dn_tiles + ar_dn).reshape(-1)
    tile_sub_d = mx.repeat(mx.arange(E_sel, dtype=mx.uint32), mod._dn_tiles)

    def b_dn_prep():
        return EM._rows_prep()(h, mod._dn_suh[sel])
    xhd = b_dn_prep(); mx.eval(xhd)

    def b_dn_kernel():
        return EM._mapped_gemv(xhd, mod._dn_trellis, mod._k, mod._cb,
                               tile_map_d, tile_sub_d).reshape(E_sel, D)
    yd = b_dn_kernel(); mx.eval(yd)

    def b_dn_finish():
        return EM._rows_finish()(yd.astype(mx.float16), mod._dn_svh[sel])

    st = {}
    st["gu_prep_gather_hadamard"] = med(bracket(b_gu_prep, reps, warmup))
    st["gu_kernel(mapped_gemv)"] = med(bracket(b_gu_kernel, reps, warmup))
    st["gu_finish_hadamard"] = med(bracket(b_gu_finish, reps, warmup))
    st["act"] = med(bracket(b_act, reps, warmup))
    st["dn_prep_hadamard"] = med(bracket(b_dn_prep, reps, warmup))
    st["dn_kernel(mapped_gemv)"] = med(bracket(b_dn_kernel, reps, warmup))
    st["dn_finish_hadamard"] = med(bracket(b_dn_finish, reps, warmup))
    return st


# ---------------- prefill (segmented) --------------------------------------
def bench_prefill(mod, x, indices, reps, warmup):
    E, D, H = mod.num_experts, mod.input_dims, mod.hidden_dims
    B, S, kk = indices.shape
    N = B * S * kk
    flat = indices.reshape(-1)
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    sidx = flat[order]
    idx1 = sidx.reshape(N, 1).astype(mx.uint32)
    tok = (mx.arange(N, dtype=mx.uint32) // kk)[order]
    tok_x = mx.concatenate([tok, mx.zeros((EM._SEG_BM,), tok.dtype)])
    sidx_x = mx.concatenate([sidx, mx.zeros((EM._SEG_BM,), sidx.dtype)])
    xf = x.reshape(B * S, D)
    nb_max = N // EM._SEG_BM + E + 1
    gt = mod._gu_tiles
    gu_trellis = mod._gu_trellis
    dn_trellis = mod._dn_trellis
    gu_svh = mod._gu_svh
    dn_suh = mod._dn_suh
    dn_svh = mod._dn_svh

    def b_sort():
        o = mx.argsort(flat); iv = mx.argsort(o); si = flat[o]
        tk = (mx.arange(N, dtype=mx.uint32) // kk)[o]
        return [si, iv, tk]
    mx.eval(*b_sort())

    x_pairs = xf[tok_x]                       # gather (gu_prep)

    def b_gu_prep():
        xp = xf[tok_x]
        xg = EM._rows_prep()(xp, mod._gu_suh[sidx_x, 0])
        xu = EM._rows_prep()(xp, mod._gu_suh[sidx_x, 1])
        return [xg, xu]
    xg, xu = b_gu_prep(); mx.eval(xg, xu)

    def b_segtable():
        return list(EM._seg_table_fn(E, nb_max, EM._SEG_BM)(sidx))
    tab, nbr = b_segtable(); mx.eval(tab, nbr)

    def b_gu_kernel():
        g = EM.inner_mm_seg_mlx(xg, gu_trellis, mod._k, mod._cb, tab, nbr,
                                n_rows=N, tn_base=0, tiles_per_e=gt, out_e=H)
        u = EM.inner_mm_seg_mlx(xu, gu_trellis, mod._k, mod._cb, tab, nbr,
                                n_rows=N, tn_base=E * gt, tiles_per_e=gt, out_e=H)
        return [g, u]
    g, u = b_gu_kernel(); mx.eval(g, u)

    def b_gu_finish():
        gf = EM._rows_finish()(g.reshape(N, H).astype(mx.float16), gu_svh[sidx, :H])
        uf = EM._rows_finish()(u.reshape(N, H).astype(mx.float16), gu_svh[sidx, H:])
        return [gf, uf]
    gf, uf = b_gu_finish(); mx.eval(gf, uf)

    def b_act():
        return EM._moe_gate_activation(gf, uf, mod._activation)
    h = b_act(); mx.eval(h)

    def b_dn_prep():
        h_pad = mx.concatenate([h, mx.zeros((EM._SEG_BM, H), dtype=h.dtype)])
        return [EM._rows_prep()(h_pad, dn_suh[sidx_x])]
    xd = b_dn_prep()[0]; mx.eval(xd)

    def b_dn_kernel():
        return EM.inner_mm_seg_mlx(xd, dn_trellis, mod._k, mod._cb, tab, nbr,
                                   n_rows=N, tn_base=0, tiles_per_e=mod._dn_tiles,
                                   out_e=D)
    y = b_dn_kernel(); mx.eval(y)

    def b_dn_finish():
        return EM._rows_finish()(y.reshape(N, D).astype(mx.float16), dn_svh[sidx])

    def b_scatter():
        yf = EM._rows_finish()(y.reshape(N, D).astype(mx.float16), dn_svh[sidx])
        return [yf[inv]]
    mx.eval(*b_scatter())

    st = {}
    st["sort_gather_argsort"] = med(bracket(b_sort, max(3, reps // 3), warmup))
    st["gu_prep_gather_hadamard"] = med(bracket(b_gu_prep, reps, warmup))
    st["segtable"] = med(bracket(b_segtable, reps, warmup))
    st["gu_kernel(seg_mm)"] = med(bracket(b_gu_kernel, reps, warmup))
    st["gu_finish_hadamard"] = med(bracket(b_gu_finish, reps, warmup))
    st["act"] = med(bracket(b_act, reps, warmup))
    st["dn_prep_hadamard"] = med(bracket(b_dn_prep, reps, warmup))
    st["dn_kernel(seg_mm)"] = med(bracket(b_dn_kernel, reps, warmup))
    st["dn_finish_hadamard"] = med(bracket(b_dn_finish, reps, warmup))
    st["scatter(combine_in_block)"] = med(bracket(b_scatter, reps, warmup))
    return st


def sum_stages(st):
    return float(sum(st.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-decode", type=int, default=15)
    ap.add_argument("--reps-prefill", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prefill512", action="store_true")
    ap.add_argument("--out", default=os.environ.get("P20_OUT", "/tmp/q1c_opclass.json"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    ckpt = Exl3Checkpoint(MODEL)
    gw = ckpt.np(f"layers.{LAYER}.ffn.gate.weight").astype(np.float32)
    gb = ckpt.np(f"layers.{LAYER}.ffn.gate.bias").astype(np.float32)
    try:
        rs = ckpt.np(f"layers.{LAYER}.ffn.gate.bias_vl").astype(np.float32)
    except Exception:
        rs = None
    mod = load_experts(ckpt, LAYER, n_experts=E_EXP, rank=0, world=2,
                       activation="silu_clamp")
    D, H, E = mod.input_dims, mod.hidden_dims, mod.num_experts
    tload = time.perf_counter() - t0
    ROUTE_SCALE = 1.5   # routed_scaling_factor
    TEMP = 1.0
    print(f"[load] arm={ARM} E={E} D={D} H={H} k={int(mod._k)} "
          f"gu_tiles={mod._gu_tiles} dn_tiles={mod._dn_tiles} in {tload:.1f}s", flush=True)

    meta = {
        "bench": "p20_q1c_opclass",
        "arm": ARM,
        "node": socket.gethostname(),
        "mlx": mx.__version__,
        "layer": LAYER, "n_moe_layers": N_MOE,
        "E": E, "D": D, "H_per_rank": H, "topk": TOPK, "rank": 0, "world": 2,
        "k_trellis": int(mod._k),
        "env": {k: os.environ.get(k) for k in
                ("EXL3_MOE_V2", "EXL3_MOE_FUSED", "EXL3_MOE_MM",
                 "EXL3_MOE_A2_TILES", "EXL3_MOE_CLAMP",
                 "MLX_GPU_TIME", "MLX_DISPATCH_COUNT")},
        "reps_decode": args.reps_decode, "reps_prefill": args.reps_prefill,
        "warmup": args.warmup, "n_seeds": args.seeds,
        "gate_dtype": "fp32", "route_scale": ROUTE_SCALE, "gate_temp": TEMP,
        "load_s": round(tload, 1),
    }

    result = {"meta": meta, "shapes": {}, "router": {}}

    shapes = [("R1_topk6", 1), ("R4_topk6", 4)]
    seeds = [1000 + s for s in range(args.seeds)]

    # ---------------- decode draft R=1 / verify R=4 ------------------------
    for name, R in shapes:
        stage_acc = {}
        whole_acc, uniq_acc, mdist = [], [], None
        for sd in seeds:
            rows = make_rows(R, D, sd)
            idxnp = route_idx(gw, gb, rows, TOPK, TEMP)
            xm = mx.array(rows.reshape(R, D).astype(np.float16))
            im = mx.array(idxnp)
            # whole module call (production __call__)
            x3 = xm.reshape(1, R, D); i3 = im.reshape(1, R, TOPK)
            def whole():
                return mod(x3, i3)
            whole_acc.append(med(bracket(whole, args.reps_decode, args.warmup)))
            if ARM == "fused":
                st = bench_fused_decode(mod, xm, im, args.reps_decode, args.warmup)
            else:
                if R == 1:
                    st = bench_unfused_decode(mod, xm, im.reshape(-1),
                                              args.reps_decode, args.warmup)
                else:
                    st = bench_prefill(mod, x3, i3,
                                       max(3, args.reps_prefill // 2), args.warmup)
            for k, v in st.items():
                stage_acc.setdefault(k, []).append(v)
            uniq_acc.append(int(len(np.unique(idxnp))))
            if R == 4 and sd == seeds[0]:
                bc = np.bincount(idxnp.reshape(-1), minlength=E).astype(int)
                nz = bc[bc > 0]
                mdist = {"n_slots": int(idxnp.size), "n_experts_hit": int((bc > 0).sum()),
                         "max_rows_per_expert": int(bc.max()),
                         "mean_rows_per_hit_expert": round(float(nz.mean()), 3),
                         "hist_max": sorted([int(x) for x in nz], reverse=True)}
        stages = {k: round(med(v), 4) for k, v in stage_acc.items()}
        whole = med(whole_acc)
        ssum = sum_stages(stages)
        cell = {
            "R": R, "kk": TOPK,
            "path": ("_decode_fused2" if ARM == "fused"
                     else ("_decode" if R == 1 else "_prefill")),
            "whole_gpu_ms": round(whole, 4),
            "sum_stages_ms": round(ssum, 4),
            "unattributed_ms": round(whole - ssum, 4),
            "unique_experts_mean": round(float(np.mean(uniq_acc)), 2),
            "stages": stages,
            "stage_per_seed": {k: [round(x, 4) for x in v] for k, v in stage_acc.items()},
            "whole_per_seed": [round(x, 4) for x in whole_acc],
        }
        if mdist is not None:
            cell["verify_m_distribution"] = mdist
        result["shapes"][name] = cell
        print(f"[{name}] path={cell['path']} whole={whole:.4f}ms "
              f"sum={ssum:.4f} unatt={cell['unattributed_ms']:+.4f} "
              f"uniqE={cell['unique_experts_mean']}", flush=True)
        for k, v in stages.items():
            print(f"    {k:32s} {v:.4f} ms", flush=True)

    # ---------------- prefill S=2048 ---------------------------------------
    for S in ([2048, 512] if args.prefill512 else [2048]):
        name = f"S{S}_topk6"
        stage_acc = {}
        whole_acc, uniq_acc = [], []
        for sd in seeds:
            rows = make_rows(S, D, sd)
            idxnp = route_idx(gw, gb, rows, TOPK, TEMP)
            xm = mx.array(rows.reshape(1, S, D).astype(np.float16))
            im = mx.array(idxnp.reshape(1, S, TOPK))
            def whole():
                return mod(xm, im)
            whole_acc.append(med(bracket(whole, args.reps_prefill, args.warmup)))
            st = bench_prefill(mod, xm, im, args.reps_prefill, args.warmup)
            for k, v in st.items():
                stage_acc.setdefault(k, []).append(v)
            uniq_acc.append(int(len(np.unique(idxnp))))
        stages = {k: round(med(v), 4) for k, v in stage_acc.items()}
        whole = med(whole_acc)
        ssum = sum_stages(stages)
        result["shapes"][name] = {
            "S": S, "kk": TOPK, "path": "_prefill",
            "whole_gpu_ms": round(whole, 4),
            "sum_stages_ms": round(ssum, 4),
            "unattributed_ms": round(whole - ssum, 4),
            "unique_experts_mean": round(float(np.mean(uniq_acc)), 2),
            "stages": stages,
            "stage_per_seed": {k: [round(x, 4) for x in v] for k, v in stage_acc.items()},
            "whole_per_seed": [round(x, 4) for x in whole_acc],
        }
        print(f"[{name}] whole={whole:.4f}ms sum={ssum:.4f} "
              f"unatt={whole-ssum:+.4f} uniqE={result['shapes'][name]['unique_experts_mean']}",
              flush=True)
        for k, v in stages.items():
            print(f"    {k:32s} {v:.4f} ms", flush=True)

    # ---------------- router (all shapes) ----------------------------------
    for name, R in [("R1", 1), ("R4", 4), ("S2048", 2048)]:
        acc = []
        for sd in seeds:
            rows = make_rows(R, D, sd)
            xf = mx.array(rows.astype(np.float32))
            s = bench_router(gw, gb, xf, TOPK, ROUTE_SCALE, TEMP,
                             args.reps_prefill if R == 2048 else args.reps_decode,
                             args.warmup)
            acc.append(med(s))
        result["router"][name] = {
            "R": R, "gpu_ms": round(med(acc), 4),
            "per_seed": [round(x, 4) for x in acc],
        }
        print(f"[router {name}] {result['router'][name]['gpu_ms']:.4f} ms", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{ARM}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

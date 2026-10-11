#!/usr/bin/env python3
"""p20_pmp_taskA_prefill.py -- P20 MEASURE PACK / TASK A (DECISIVE).

Native affine experts (mx.quantize + mx.gather_qmm) measured +32.6% WALL vs the
EXL3 2.9bpw trellis fused kernel at the VERIFY shape (R=4, top-6). At PREFILL
shapes (chunks 2048/4096, real per-expert M-profile) the gap is UNMEASURED.

VERDICT required: does native beat EXL3 by >=20% on the expert BLOCK at prefill
shapes?  >=20% holds -> a fit plan follows; <=5% -> the lever is dead.

REUSE (do not rebuild): imports bench/p20_q1d_native_experts.py for the EXL3
loader, the streamed native requant build, the fused gate+up sorted gather_qmm
native forward, cosine metrics, and the A6b numerics helper.  The routing shape
comes from the REAL production trace exactly as bench/p20_moe_autopsy.py::
real_chunk_idx did (S tokens drawn WITH replacement from the 531 per-layer rows
-> (S, topk) idx; per-expert row histogram is then real).

Geometry (production, per-rank): E=384 experts, H=1152 (TP-split), D=5120,
world=2 rank=0, top-6.  gu N=2H K=D ; dn N=D K=H.

ARMS (routing glue IDENTICAL across arms: BOTH the EXL3 trellis path and the
native 'sorted' path do argsort -> gather_qmm(sorted_indices=True) -> scatter-
unsort; the native 'unsorted' variant is not used here):
  1. EXL3 production kernel (EXL3_MOE_V2/MM/FUSED=1, silu_clamp) @ layer 20 (k=2)
     AND layer 0 (k=3).
  2. native q4g64 (bits=4 group=64), fused gate+up sorted gather_qmm.
  3. native q3g128 (bits=3 group=128) -- the bpw-fair arm (3.25 vs 2.9 bpw).
  4. uniform/synthetic fill reference (equal rows/expert).

Metrics per arm: ms/layer WALL + GPU (MLX_GPU_TIME=1), TFLOP/s, % of the
in-session-calibrated dense bf16 roofline (4096x8192x8192 mx.matmul), x40-layer
extrapolation, median + sigma over interleaved rounds (alternating order),
delta%.  Numerics: cosine + rel-L2 of each native arm vs EXL3 (A6b method).

Node: studio2 only, offline, idle-guarded, read-only everywhere else.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import statistics
import sys
import time
from collections import defaultdict

os.environ.setdefault("MLX_GPU_TIME", "1")
os.environ.setdefault("MLX_DISPATCH_COUNT", "1")
os.environ.setdefault("EXL3_MOE_V2", "1")
os.environ.setdefault("EXL3_MOE_MM", "1")
os.environ.setdefault("EXL3_MOE_FUSED", "1")
os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")
assert os.environ.get("MLX_GPU_TIME") == "1", "run with MLX_GPU_TIME=1 before import mlx"

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# q1d sets the same env defaults, inserts ~/repos/exo/mlx-lm and imports mlx/exl3.
import p20_q1d_native_experts as q1d  # noqa: E402

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402

E, D, H, KK = q1d.E_EXP, q1d.D, q1d.H, q1d.TOPK
N_MOE = q1d.N_MOE
WORLD, RANK = q1d.WORLD, q1d.RANK
TRACE = os.environ.get(
    "P20_TRACE",
    "/tmp/p20_pmp_A/raw/trace/p30_exl3_trace_clamp.json")
LAYERS = [int(x) for x in os.environ.get("P20_LAYERS", "20,0").split(",") if x != ""]
CHUNKS = [int(x) for x in os.environ.get("P20_CHUNKS", "2048,4096").split(",") if x != ""]
ROUNDS = int(os.environ.get("P20_ROUNDS", "5"))
WARMUP = int(os.environ.get("P20_WARMUP", "2"))
CACHE_LIMIT_GB = float(os.environ.get("P20_CACHE_GB", "5"))


def log(*a):
    print("[A]", *a, flush=True)


def set_cache_limit():
    for fn in ("set_cache_limit",):
        f = getattr(mx, fn, None) or getattr(getattr(mx, "metal", None), fn, None)
        if f is not None:
            try:
                f(int(CACHE_LIMIT_GB * (1 << 30)))
                log(f"cache limit set to {CACHE_LIMIT_GB} GB")
                return
            except Exception as exc:  # noqa: BLE001
                log("cache limit set failed:", exc)


def clear_cache():
    try:
        mx.clear_cache()
    except Exception:
        pass


# ------------------------------------------------------------------ routing ---
def load_trace(path):
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    d = json.loads(raw)
    per = defaultdict(list)
    for rec in d["records"]:
        per[int(rec[0])].append(np.asarray(rec[1], np.int32))
    return per, sha, d.get("n_steps"), d.get("prompt_tokens")


def real_idx(per, layer, S, seed):
    """Autopsy real_chunk_idx: S draws WITH replacement from the 531 layer rows."""
    rows = per[layer]
    rng = np.random.RandomState(seed)
    sel = rng.randint(0, len(rows), size=S)
    ids = np.concatenate([rows[i] for i in sel]).astype(np.int32)  # [S*KK]
    return ids.reshape(S, KK), len(rows)


def uniform_idx(S):
    npairs = S * KK
    base, rem = divmod(npairs, E)
    ids = np.empty(npairs, np.int32)
    off = 0
    for e in range(E):
        c = base + (1 if e < rem else 0)
        ids[off:off + c] = e
        off += c
    np.random.RandomState(7).shuffle(ids)
    return ids.reshape(S, KK)


def idx_stats(idx):
    flat = idx.reshape(-1)
    c = np.bincount(flat, minlength=E)
    hit = c[c > 0]
    return {"pairs": int(flat.size), "experts_hit": int(hit.size),
            "M_mean_all": round(float(flat.size / E), 3),
            "M_mean_hit": round(float(hit.mean()), 3),
            "M_median_hit": float(np.median(hit)), "M_max": int(c.max()),
            "M_p95": float(np.percentile(c, 95))}


# ------------------------------------------------------------------ timing ----
def time_interleaved(arms, rounds, warmup):
    """arms: list of (name, fn). Alternating order per round; one eval per round."""
    for _, fn in arms:
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
    res = {name: {"wall": [], "gpu": []} for name, _ in arms}
    for r in range(rounds):
        seq = arms if r % 2 == 0 else list(reversed(arms))
        for name, fn in seq:
            mx.metal.reset_gpu_time()
            t0 = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            t1 = time.perf_counter()
            res[name]["wall"].append((t1 - t0) * 1e3)
            res[name]["gpu"].append(mx.metal.gpu_time_ns() / 1e6)
    return res


def summarize(raw):
    out = {}
    for name, v in raw.items():
        w, g = v["wall"], v["gpu"]
        out[name] = {
            "wall_ms": round(statistics.median(w), 4),
            "gpu_ms": round(statistics.median(g), 4),
            "wall_sigma": round(statistics.pstdev(w), 4) if len(w) > 1 else 0.0,
            "gpu_sigma": round(statistics.pstdev(g), 4) if len(g) > 1 else 0.0,
            "wall_min": round(min(w), 4), "wall_max": round(max(w), 4),
            "gpu_min": round(min(g), 4), "gpu_max": round(max(g), 4),
            "n": len(w),
        }
    return out


# ------------------------------------------------------------------ roofline --
def calibrate_dense(T=4096, M=8192, N=8192, reps=9):
    a = mx.random.normal((T, M)).astype(mx.bfloat16)
    b = mx.random.normal((M, N)).astype(mx.bfloat16)
    mx.eval(a, b)
    ts, gs = [], []
    for _ in range(reps):
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        c = mx.matmul(a, b)
        mx.eval(c)
        mx.synchronize()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
        gs.append(mx.metal.gpu_time_ns() / 1e6)
    flops = 2 * T * M * N
    tw = statistics.median(ts)
    tg = statistics.median(gs) * 1e-3
    del a, b, c
    clear_cache()
    return {"shape": [T, M, N], "flops": flops,
            "wall_ms": round(tw * 1e3, 4), "gpu_ms": round(tg * 1e3, 4),
            "tf_wall": round(flops / tw / 1e12, 4),
            "tf_gpu": round(flops / tg / 1e12, 4),
            "reps": reps, "wall_sigma_ms": round(statistics.pstdev(ts) * 1e3, 4)}


def block_flops(S):
    return 6 * S * KK * H * D


def enrich(sumry, S, dense_gpu_tf, dense_wall_tf):
    fl = block_flops(S)
    for name, c in sumry.items():
        c["flops"] = fl
        c["tf_gpu"] = round(fl / (c["gpu_ms"] * 1e-3) / 1e12, 4)
        c["tf_wall"] = round(fl / (c["wall_ms"] * 1e-3) / 1e12, 4)
        c["pct_roofline_gpu"] = round(100.0 * c["tf_gpu"] / dense_gpu_tf, 2)
        c["pct_roofline_wall"] = round(100.0 * c["tf_wall"] / dense_wall_tf, 2)
        c["x40_wall_ms"] = round(c["wall_ms"] * N_MOE, 3)
        c["x40_gpu_ms"] = round(c["gpu_ms"] * N_MOE, 3)
    return fl


def main():
    t_start = time.perf_counter()
    set_cache_limit()

    per, trace_sha, n_steps, prompt_tokens = load_trace(TRACE)
    log(f"trace {TRACE} sha256={trace_sha[:16]}... n_steps={n_steps} layers={len(per)}")

    ckpt = q1d.Exl3Checkpoint(q1d.MODEL)
    log("loaded checkpoint index")

    result = {
        "meta": {
            "bench": "p20_pmp_taskA_prefill", "node": socket.gethostname(),
            "mlx": mx.__version__, "E": E, "D": D, "H_per_rank": H, "H_full": q1d.H_FULL,
            "topk": KK, "world": WORLD, "rank": RANK, "n_moe_layers": N_MOE,
            "layers": LAYERS, "chunks": CHUNKS, "rounds": ROUNDS, "warmup": WARMUP,
            "trace_path": TRACE, "trace_sha256": trace_sha,
            "trace_n_steps": n_steps, "trace_prompt_tokens": prompt_tokens,
            "clamp": q1d.CLAMP, "cache_limit_gb": CACHE_LIMIT_GB,
            "routing": "REAL trace, S draws WITH replacement from the 531 layer rows "
                       "(autopsy real_chunk_idx); uniform fill = equal rows/expert",
            "glue": "BOTH arms: argsort -> gather_qmm(sorted_indices=True) -> scatter-unsort "
                    "(native 'sorted' variant; EXL3 internal segmented path uses the same glue class)",
            "loadavg": [round(x, 2) for x in os.getloadavg()],
            "env": {k: os.environ.get(k) for k in
                    ("EXL3_MOE_V2", "EXL3_MOE_MM", "EXL3_MOE_FUSED", "EXL3_MOE_CLAMP",
                     "MLX_GPU_TIME", "MLX_DISPATCH_COUNT", "MTL_DISABLE_TIMEOUT")},
        },
        "dense_roofline": {}, "arms": {}, "numerics": {}, "verdict": {},
    }

    # ---- dense bf16 roofline (in-session calibrated) ----
    roofline = calibrate_dense()
    result["dense_roofline"] = roofline
    dense_gpu_tf = roofline["tf_gpu"]
    dense_wall_tf = roofline["tf_wall"]
    log(f"dense bf16 roofline 4096x8192x8192: {dense_gpu_tf:.3f} TF (gpu) "
        f"{dense_wall_tf:.3f} TF (wall)  gpu_ms={roofline['gpu_ms']}")

    # ---- build native arms ONCE (layer-20 weights; timing is shape-driven) ----
    flush = None  # prefill is compute-bound; hot interleaved timing
    log("building native q4g64 (streamed)")
    t0 = time.perf_counter()
    arm_q4 = q1d.build_arm_streamed("q4g64", ckpt, (4, 64), (4, 64), split_gu=False)
    log(f"q4g64 built {time.perf_counter()-t0:.0f}s total {arm_q4['total_bytes']/1e9:.2f} GB")
    clear_cache()
    log("building native q3g128 (streamed)")
    t0 = time.perf_counter()
    arm_q3 = q1d.build_arm_streamed("q3g128", ckpt, (3, 128), (3, 128), split_gu=False)
    log(f"q3g128 built {time.perf_counter()-t0:.0f}s total {arm_q3['total_bytes']/1e9:.2f} GB")
    clear_cache()
    result["arms_bytes"] = {
        "q4g64_total": arm_q4["total_bytes"], "q3g128_total": arm_q3["total_bytes"],
        "q4g64_gu_per_expert": arm_q4["gu_bytes"] // E,
        "q4g64_dn_per_expert": arm_q4["dn_bytes"] // E,
    }

    VARIANT = os.environ.get("P20_NATIVE_VARIANT", "sorted")

    # ---- per-layer EXL3 module + interleaved timing ----
    def run_layer(layer, mod):
        gw = ckpt.np(f"layers.{layer}.ffn.gate.weight").astype(np.float32)
        gb = ckpt.np(f"layers.{layer}.ffn.gate.bias").astype(np.float32)
        lkey = f"L{layer}"
        result["arms"][lkey] = {"k_trellis": int(mod._k), "gu_tiles": int(mod._gu_tiles),
                                "dn_tiles": int(mod._dn_tiles), "shapes": {}}
        for S in CHUNKS:
            sseed = 4000 + layer * 100 + S
            idx_real, n_rows = real_idx(per, layer, S, sseed)
            idx_unif = uniform_idx(S)
            st_real, st_unif = idx_stats(idx_real), idx_stats(idx_unif)
            log(f"{lkey} S={S} real idx {st_real}")
            x_nat = mx.array(np.random.RandomState(sseed).standard_normal((S, D)).astype(np.float16))
            x_ex = x_nat.reshape(1, S, D)
            m_real = mx.array(idx_real.reshape(1, S, KK))
            m_unif = mx.array(idx_unif.reshape(1, S, KK))
            mx.eval(x_nat, x_ex, m_real, m_unif)

            arms = [
                (f"exl3_{lkey}_real", lambda mm=m_real, xx=x_ex: mod(xx, mm)),
                ("native_q4g64_real", lambda mm=idx_real, xx=x_nat: q1d.native_forward(xx, mm, arm_q4, VARIANT, True)),
                ("native_q3g128_real", lambda mm=idx_real, xx=x_nat: q1d.native_forward(xx, mm, arm_q3, VARIANT, True)),
                (f"exl3_{lkey}_uniform", lambda mm=m_unif, xx=x_ex: mod(xx, mm)),
                ("native_q4g64_uniform", lambda mm=idx_unif, xx=x_nat: q1d.native_forward(xx, mm, arm_q4, VARIANT, True)),
                ("native_q3g128_uniform", lambda mm=idx_unif, xx=x_nat: q1d.native_forward(xx, mm, arm_q3, VARIANT, True)),
            ]
            raw = time_interleaved(arms, ROUNDS, WARMUP)
            sumry = summarize(raw)
            enrich(sumry, S, dense_gpu_tf, dense_wall_tf)
            result["arms"][lkey]["shapes"][f"S{S}"] = {
                "idx_real": st_real, "idx_uniform": st_unif,
                "trace_rows_for_layer": n_rows, "routing_seed": sseed, "cells": sumry}
            for name, c in sumry.items():
                log(f"{lkey} S={S} {name:24s} wall={c['wall_ms']:8.3f} "
                    f"gpu={c['gpu_ms']:8.3f} tf_gpu={c['tf_gpu']:6.2f} "
                    f"roof={c['pct_roofline_gpu']:5.1f}%")
            del x_nat, x_ex, m_real, m_unif
            clear_cache()

        # numerics (A6b) at R=4 on layer 20 only -- needs layer-20 native weights
        if layer == 20:
            seeds = [1000, 1001, 1002]
            for tag, arm in (("q4g64", arm_q4), ("q3g128", arm_q3)):
                n = q1d.corr_vs_exl3(
                    lambda x, i, a=arm: q1d.native_forward(x, i, a, VARIANT, True),
                    mod, gw, gb, 4, KK, seeds)
                result["numerics"][f"{tag}_vs_exl3_R4"] = n
                log(f"numerics {tag} vs EXL3: cos={n['cosine_mean']:.6f} "
                    f"rel_l2={n['rel_l2_mean']:.3e}")

    log("loading EXL3 layer 20 (k=2)")
    mod20 = q1d.load_experts(ckpt, 20, n_experts=E, rank=RANK, world=WORLD,
                             activation="silu_clamp")
    assert (mod20.input_dims, mod20.hidden_dims, mod20.num_experts) == (D, H, E)
    run_layer(20, mod20)
    if 0 in LAYERS:
        mod20.clear() if hasattr(mod20, "clear") else None
        del mod20
        clear_cache()
        log("loading EXL3 layer 0 (k=3)")
        mod0 = q1d.load_experts(ckpt, 0, n_experts=E, rank=RANK, world=WORLD,
                                activation="silu_clamp")
        run_layer(0, mod0)

    # ---- verdict: native vs EXL3 on the real-routing block at prefill shapes ----
    vd = {}
    for layer in LAYERS:
        lkey = f"L{layer}"
        for S in CHUNKS:
            cells = result["arms"][lkey]["shapes"][f"S{S}"]["cells"]
            ex_g = cells[f"exl3_{lkey}_real"]["gpu_ms"]
            ex_w = cells[f"exl3_{lkey}_real"]["wall_ms"]
            for tag in ("native_q4g64_real", "native_q3g128_real"):
                ng, nw = cells[tag]["gpu_ms"], cells[tag]["wall_ms"]
                vd[f"{lkey}_S{S}_{tag}"] = {
                    "exl3_gpu_ms": ex_g, "native_gpu_ms": ng,
                    "delta_gpu_pct": round(100.0 * (ex_g - ng) / ex_g, 2),
                    "exl3_wall_ms": ex_w, "native_wall_ms": nw,
                    "delta_wall_pct": round(100.0 * (ex_w - nw) / ex_w, 2),
                }
    result["verdict"] = vd
    # headline: best native vs EXL3 on real routing, per shape
    head = {}
    for layer in LAYERS:
        lkey = f"L{layer}"
        for S in CHUNKS:
            cells = result["arms"][lkey]["shapes"][f"S{S}"]["cells"]
            ex_g = cells[f"exl3_{lkey}_real"]["gpu_ms"]
            ex_w = cells[f"exl3_{lkey}_real"]["wall_ms"]
            best = min((cells[t]["gpu_ms"], t) for t in ("native_q4g64_real", "native_q3g128_real"))
            head[f"{lkey}_S{S}"] = {
                "exl3_gpu_ms": ex_g, "best_native": best[1], "best_native_gpu_ms": best[0],
                "delta_gpu_pct": round(100.0 * (ex_g - best[0]) / ex_g, 2),
                "exl3_wall_ms": ex_w,
                "best_native_wall_ms": cells[best[1]]["wall_ms"],
                "delta_wall_pct": round(100.0 * (ex_w - cells[best[1]]["wall_ms"]) / ex_w, 2),
            }
    result["headline"] = head
    result["meta"]["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    out = os.environ.get("P20_OUT", "/tmp/p20_pmp_A/taskA.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    log(f"wrote {out} in {result['meta']['elapsed_s']}s")
    log("HEADLINE " + json.dumps(head))


if __name__ == "__main__":
    main()

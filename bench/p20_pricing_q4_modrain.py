#!/usr/bin/env python3
"""p20_pricing_q4_modrain.py -- Q4 MoE expert-drain differential (BENCH-ONLY, studio2).

Framing (adopted from pre-dispatch review): an isolated single-node, single-layer
topk3-vs-topk6 bench measures the experts' STANDALONE COMPUTE cost, NOT drain/
overlap/exposure into a collective (a 1-node bench cannot show cross-rank overlap).
So we do a TOTALS comparison:
    standalone_expert_ms(R=4, topk6) * n_moe_layers   vs   the ~37 ms UNATTRIBUTED
in verify_block (~40% of ~92.4 ms).

Routing: cost tracks the number of UNIQUE experts touched (bandwidth-driven), not
k*R.  Real spec-decode drafts are CORRELATED -> derive indices from the REAL layer-20
gate applied to a base hidden state perturbed per draft row (consecutive rows share
hot experts).  >=3 seeds; report spread.  A uniform-random arm is included for contrast.

Real weights: layer-20 EXL3SwitchGLU, n_experts=384, rank=0 world=2 (production TP=2
per-rank intermediate slice), activation silu_clamp.  Decode-class A2/B2 path for R<=8.

No cluster mutation: no POST, no start/stop/kill, no writes to ~/repos/exo.
"""
from __future__ import annotations
import json, os, sys, time, statistics, argparse

os.environ.setdefault("EXL3_MOE_V2", "1")     # decode-class A2/B2 path (production default)
os.environ.setdefault("EXL3_MOE_FUSED", "1")
os.environ.setdefault("EXL3_MOE_MM", "1")

import numpy as np
import mlx.core as mx

REPO = os.path.expanduser("~/repos/exo")
MLXLM = os.path.join(REPO, "mlx-lm")
sys.path.insert(0, MLXLM)
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts  # noqa: E402

MODEL = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
LAYER = 20
N_MOE_LAYERS = 40          # verified: every layer (0..39) has ffn.gate + ffn.experts


def load_gate(ckpt, layer):
    w = ckpt.np(f"layers.{layer}.ffn.gate.weight").astype(np.float32)  # (E, D)
    b = ckpt.np(f"layers.{layer}.ffn.gate.bias").astype(np.float32)    # (E,)
    return w, b


CORR = 0.03  # real spec-decode draft rows differ by ~one token -> highly correlated


def make_rows(R, D, seed, corr=None):
    """R correlated draft rows: base hidden state + small per-row perturbation.

    Small perturbation => consecutive rows route to ~the same top-k => they share
    hot experts (the bandwidth-relevant quantity), as real drafts do.
    """
    corr = CORR if corr is None else corr
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(D).astype(np.float32)
    if R == 1:
        return base[None, :].copy()
    rows = base[None, :] + rng.standard_normal((R, D)).astype(np.float32) * corr
    rows[0] = base
    return rows


def route_topk(gate_w, gate_b, rows, kk, mode="correlated", hot=None, seed=0):
    """Return (R,kk) int32 expert indices.

    mode 'correlated'/'uniform': real-gate topk on rows (correlated by construction)
    or uniform-random.  'hot': real-gate scores restricted to the first `hot` experts
    (biases to a hot subset) -- stresses the unique-expert (bandwidth) hypothesis.
    """
    R = rows.shape[0]
    if mode == "uniform":
        rng = np.random.default_rng(seed + 777)
        return rng.integers(0, gate_w.shape[0], size=(R, kk)).astype(np.int32)
    logits = rows @ gate_w.T + gate_b                      # (R,E)
    scores = 1.0 / (1.0 + np.exp(-logits))                 # sigmoid (noaux_tc-ish)
    if mode == "hot" and hot is not None:
        mask = np.zeros_like(scores)
        mask[:, :hot] = scores[:, :hot]
        scores = mask
    idx = np.argsort(-scores, axis=-1)[:, :kk]
    return idx.astype(np.int32)


def timed_call(module, rows, idx, reps, warmup, sync):
    """Per-call wall (ms) with optional mx.eval (forced sync at block boundary)."""
    x = mx.array(rows.reshape(1, rows.shape[0], -1))
    ia = mx.array(idx.reshape(1, idx.shape[0], idx.shape[1]))
    for _ in range(warmup):
        y = module(x, ia)
        mx.eval(y)
    times = []
    last = None
    for _ in range(reps):
        t0 = time.perf_counter()
        y = module(x, ia)
        if sync:
            mx.eval(y)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
        last = y
    mx.eval(last)                      # drain whatever is pending (async arm)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-experts", type=int, default=384)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prefill", action="store_true", help="also run S=512 prefill-class")
    ap.add_argument("--out", default="/tmp/pricing/q4_modrain.json")
    args = ap.parse_args()

    t_load0 = time.perf_counter()
    ckpt = Exl3Checkpoint(MODEL)
    gate_w, gate_b = load_gate(ckpt, LAYER)
    module = load_experts(ckpt, LAYER, n_experts=args.n_experts,
                          rank=args.rank, world=args.world, activation="silu_clamp")
    D = module.input_dims
    H = module.hidden_dims
    E = module.num_experts
    t_load = (time.perf_counter() - t_load0)
    print(f"[load] E={E} D={D} H={H} (world={args.world} rank={args.rank}) "
          f"in {t_load:.1f}s; gate {gate_w.shape}", flush=True)

    result = {
        "meta": {
            "bench": "p20_pricing_q4_modrain",
            "layer": LAYER, "n_experts": E, "rank": args.rank, "world": args.world,
            "D": D, "H": H, "k_trellis": int(module._k),
            "n_moe_layers": N_MOE_LAYERS, "reps": args.reps, "warmup": args.warmup,
            "n_seeds": args.seeds,
            "env": {k: os.environ.get(k) for k in
                    ("EXL3_MOE_V2", "EXL3_MOE_FUSED", "EXL3_MOE_MM", "EXL3_MOE_A2_TILES")},
            "load_s": round(t_load, 1),
            "decode_class": "R<=8 -> A2/B2 (_decode_fused2)",
        },
        "runs": {},
    }

    # ---- main matrix: R in {1,4} x kk in {3,6} x {correlated, uniform} x sync/async
    for R in (1, 4):
        for kk in (3, 6):
            for mode in ("correlated", "uniform"):
                cells_sync = []
                cells_async = []
                uniq = []
                for s in range(args.seeds):
                    rows = make_rows(R, D, seed=1000 + s)
                    idx = route_topk(gate_w, gate_b, rows, kk, mode=mode, seed=s)
                    uniq.append(int(len(np.unique(idx))))
                    tS = timed_call(module, rows, idx, args.reps, args.warmup, True)
                    tA = timed_call(module, rows, idx, args.reps, args.warmup, False)
                    cells_sync.append(statistics.median(tS))
                    cells_async.append(statistics.median(tA))
                key = f"R{R}_k{kk}_{mode}"
                result["runs"][key] = {
                    "R": R, "kk": kk, "mode": mode,
                    "unique_experts_mean": round(float(np.mean(uniq)), 2),
                    "unique_experts_per_seed": uniq,
                    "sync_ms_median_per_seed": [round(v, 4) for v in cells_sync],
                    "sync_ms_median": round(statistics.median(cells_sync), 4),
                    "sync_ms_spread": round(max(cells_sync) - min(cells_sync), 4),
                    "dispatch_only_ms_median": round(statistics.median(cells_async), 4),
                    "exposed_drain_bound_ms": round(statistics.median(cells_sync) -
                                          statistics.median(cells_async), 4),
                    "sync_ms_min": round(min(cells_sync), 4),
                    "sync_ms_max": round(max(cells_sync), 4),
                }
                print(f"[{key}] uniqueE~{np.mean(uniq):.1f} "
                      f"sync={statistics.median(cells_sync):.4f}ms "
                      f"dispatch={statistics.median(cells_async):.4f}ms "
                      f"exposed_bound={result['runs'][key]['exposed_drain_bound_ms']:.4f}ms", flush=True)

    # ---- hot-subset arm (R=4, kk=6): bias routing to a hot subset of size `hot`
    for hot in (24, 12):
        rows = make_rows(4, D, seed=1000)
        idx = route_topk(gate_w, gate_b, rows, 6, mode="hot", hot=hot)
        tS = timed_call(module, rows, idx, args.reps, args.warmup, True)
        key = f"R4_k6_hot{hot}"
        result["runs"][key] = {
            "R": 4, "kk": 6, "mode": f"hot{hot}",
            "unique_experts_mean": int(len(np.unique(idx))),
            "sync_ms_median": round(statistics.median(tS), 4),
            "sync_ms_min": round(min(tS), 4), "sync_ms_max": round(max(tS), 4),
        }
        print(f"[{key}] uniqueE={result['runs'][key]['unique_experts_mean']} "
              f"sync={statistics.median(tS):.4f}ms", flush=True)

    # ---- optional prefill-class arm S=512 (R>8 -> _prefill)
    if args.prefill:
        for kk in (6,):
            S = 512
            rows = make_rows(S, D, seed=2000)
            idx = route_topk(gate_w, gate_b, rows, kk, mode="correlated")
            tS = timed_call(module, rows, idx, max(3, args.reps // 3), args.warmup, True)
            key = f"S{S}_k{kk}_correlated"
            result["runs"][key] = {
                "R": S, "kk": kk, "mode": "correlated", "class": "prefill",
                "unique_experts_mean": int(len(np.unique(idx))),
                "sync_ms_median": round(statistics.median(tS), 4),
            }
            print(f"[{key}] uniqueE={result['runs'][key]['unique_experts_mean']} "
                  f"sync={statistics.median(tS):.4f}ms", flush=True)

    # ---- totals comparison
    prod = result["runs"]["R4_k6_correlated"]["sync_ms_median"]      # production shape
    red = result["runs"]["R4_k3_correlated"]["sync_ms_median"]
    prod_lo = result["runs"]["R4_k6_correlated"]["sync_ms_min"]
    prod_hi = result["runs"]["R4_k6_correlated"]["sync_ms_max"]
    tot = prod * N_MOE_LAYERS
    UNATTR = 37.0
    result["totals"] = {
        "standalone_expert_ms_R4_topk6": prod,
        "standalone_total_ms": round(tot, 3),
        "standalone_total_ms_range": [round(prod_lo * N_MOE_LAYERS, 3),
                                       round(prod_hi * N_MOE_LAYERS, 3)],
        "n_moe_layers": N_MOE_LAYERS,
        "unattributed_ms_reference": UNATTR,
        "totals_ratio_vs_37ms": round(tot / UNATTR, 3),
        "totals_ratio_range": [round(prod_lo * N_MOE_LAYERS / UNATTR, 3),
                                round(prod_hi * N_MOE_LAYERS / UNATTR, 3)],
        "halving_effect": {
            "R4_topk6_ms": prod, "R4_topk3_ms": red,
            "delta_ms": round(prod - red, 4),
            "ratio_topk6_over_topk3": round(prod / red, 3) if red else None,
        },
        "R1_topk6_ms": result["runs"]["R1_k6_correlated"]["sync_ms_median"],
        "R1_topk3_ms": result["runs"]["R1_k3_correlated"]["sync_ms_median"],
        "verify_block_ms_reference": 92.4,
    }
    print(f"[TOTALS] standalone R4/topk6={prod:.4f}ms x{N_MOE_LAYERS} = {tot:.2f}ms "
          f"vs {UNATTR}ms unattributed -> ratio {tot/UNATTR:.2f}x "
          f"(range {result['totals']['totals_ratio_range']})", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

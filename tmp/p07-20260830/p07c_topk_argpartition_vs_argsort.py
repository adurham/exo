# P07c: decisive isolated A/B — mx.argpartition vs mx.argsort for the
# DeepSeek-V4-Flash indexer top-k at REAL production prefill shapes.
#   scores (1, L_band=1024, P) bf16, k=512, P sweep over compressed-pool sizes
#   (P = ceil(ctx/4)): 5000/12500/25000/55000/125000 (~20K..500K ctx).
# Production expressions (deepseek_v4.py:4055,4059):
#   A: mx.argpartition(-scores, kth=k-1, axis=-1)[..., :k]
#   B: mx.argsort(-scores, axis=-1)[..., :k]
# p01/P03/P07-proven standalone recipe: run BESIDE the live runner on
# macstudio-m4-1 (no relaunch, no git ops), MLX_GPU_TIME=1 GPU-bracketed
# timing, fresh graph per timed call, mx.eval() barrier per call, rotation
# banks of DISTINCT inputs, median of >=3 passes.
import os

for v in ("MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"
for v in ("MLX_GEMV_BATCH_INVARIANT", "MLX_STEEL_BATCH_INVARIANT"):
    assert os.environ.get(v) == "1", f"{v}=1 required (production kernel parity)"
assert int(os.environ.get("MLX_MAX_OPS_PER_BUFFER", "0")) == 200
assert int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "0")) == 200

import json
import time
from pathlib import Path

import mlx.core as mx

OUT = os.path.dirname(os.path.abspath(__file__))
RESULTS = {"meta": {}, "sweep": [], "correctness": [], "l2_sanity": {}, "errors": []}


def save():
    with open(os.path.join(OUT, "results_topk_ab.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)


def log(*a):
    print(*a, flush=True)


log("=== P07c topk argpartition vs argsort A/B ===")
RESULTS["meta"] = {
    "host": os.uname().nodename,
    "mlx": mx.__version__,
    "gpu": mx.device_info().get("architecture"),
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "L_band": 1024,
    "k": 512,
    "dtype": "bfloat16",
    "expr_A_argpartition": "mx.argpartition(-scores, kth=k-1, axis=-1)[..., :k]",
    "expr_B_argsort": "mx.argsort(-scores, axis=-1)[..., :k]",
}
save()

L_BAND, K = 1024, 512
P_LIST = [5000, 12500, 25000, 55000, 125000]
L2_BYTES = 16 * 1024 * 1024  # M4 Max L2

mx.random.seed(20260830)


def make_bank(P, target_bytes=512 * 1024 * 1024):
    """Rotate DISTINCT score tensors so we do not measure an L2-resident
    best case. Bank working set >= L2 for every P here."""
    per = L_BAND * P * 2
    nsets = max(2, min(16, int(target_bytes // per)))
    bank = [mx.random.normal((1, L_BAND, P)).astype(mx.bfloat16) for _ in range(nsets)]
    mx.eval(*bank)
    return bank, per * nsets


def timed(fn, n, warmup, bank_bytes, passes=5):
    """p07 pattern: fresh graph each call, mx.eval barrier, GPU-time bracket."""
    for j in range(warmup):
        mx.eval(fn(j))
    mx.synchronize()
    recs = []
    for _p in range(passes):
        d0 = mx.metal.dispatch_count()
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        for i in range(n):
            mx.eval(fn(i))
        mx.synchronize()
        recs.append({
            "wall_us": (time.perf_counter() - t0) / n * 1e6,
            "gpu_us": mx.metal.gpu_time_ns() / n / 1e3,
            "dispatches": (mx.metal.dispatch_count() - d0) / n,
        })
    med = {k: sorted(x[k] for x in recs)[len(recs) // 2] for k in recs[0]}
    med["gbps_rw"] = bank_bytes / (med["gpu_us"] * 1e-6) / 1e9 if med["gpu_us"] > 0 else 0.0
    med["passes"] = recs
    return med


def bench_path(name, P, bank, which):
    if which == "argpartition":
        fn = lambda i: mx.argpartition(-bank[i % len(bank)], kth=K - 1, axis=-1)[..., :K]
    else:
        fn = lambda i: mx.argsort(-bank[i % len(bank)], axis=-1)[..., :K]
    n = max(8, 2 * len(bank))
    r = timed(fn, n, 4, bank[0].nbytes)
    log(f"P={P:>7} {name:<12} gpu={r['gpu_us']:9.1f}us wall={r['wall_us']:9.1f}us "
        f"disp={r['dispatches']:5.1f} pass-range=[{min(x['gpu_us'] for x in r['passes']):.1f}"
        f"..{max(x['gpu_us'] for x in r['passes']):.1f}]")
    return r


def correctness(P, bank):
    """Set-equality of selected indices + multiset-equality of selected
    scores (ties may order differently; comment asserts multiset equal)."""
    out = {"P": P}
    for tag in ("random", "forced_ties"):
        if tag == "forced_ties":
            mx.random.seed(P)  # coarse quantization -> many exact bf16 ties
            s = mx.round(bank[1] * 4.0) / 4.0
            mx.eval(s)
        else:
            s = bank[0]
        a = mx.argpartition(-s, kth=K - 1, axis=-1)[..., :K]  # (1,1024,512) i32
        b = mx.argsort(-s, axis=-1)[..., :K]
        mx.eval(a, b)
        an, bn = a.tolist(), b.tolist()
        sa, sbv = s.tolist(), s.tolist()
        idx_eq = score_multi_eq = True
        n_tie_rows = 0
        for row in range(L_BAND):
            ra, rb = set(an[0][row]), set(bn[0][row])
            if ra != rb:
                idx_eq = False
                # borderline tie? scores at min selected are equal across sets
                va = sorted(sa[0][row][i] for i in ra)
                vb = sorted(sbv[0][row][i] for i in rb)
                if va != vb:
                    score_multi_eq = False
            # does a tie straddle the k-boundary (k-th and k+1-th score equal)?
            srow = sa[0][row]
            top = sorted(srow, reverse=True)
            if top[K - 1] == top[K]:
                n_tie_rows += 1
        out[tag] = {"index_sets_equal": idx_eq,
                    "selected_score_multisets_equal": score_multi_eq,
                    "rows_with_boundary_tie": n_tie_rows}
        log(f"P={P:>7} correctness {tag}: idx_eq={idx_eq} score_multiset_eq={score_multi_eq} "
            f"boundary_tie_rows={n_tie_rows}")
    RESULTS["correctness"].append(out)
    return out


for P in P_LIST:
    try:
        bank, bank_bytes = make_bank(P)
        log(f"P={P}: bank nsets={len(bank)} working_set={bank_bytes/1e6:.0f}MB "
            f"(L2={L2_BYTES/1e6:.0f}MB) per-tensor={bank[0].nbytes/1e6:.1f}MB")
        corr = correctness(P, bank)
        ra = bench_path("argpartition", P, bank, "argpartition")
        rb = bench_path("argsort", P, bank, "argsort")
        del bank
        mx.metal.clear_cache()
        RESULTS["sweep"].append({
            "P": P, "approx_ctx": P * 4,
            "argpartition_gpu_us": ra["gpu_us"], "argpartition_wall_us": ra["wall_us"],
            "argpartition_dispatches": ra["dispatches"], "argpartition_passes": ra["passes"],
            "argsort_gpu_us": rb["gpu_us"], "argsort_wall_us": rb["wall_us"],
            "argsort_dispatches": rb["dispatches"], "argsort_passes": rb["passes"],
            "ratio_argpart_over_argsort": ra["gpu_us"] / rb["gpu_us"],
            "correctness": corr,
        })
        save()
    except Exception as e:
        RESULTS["errors"].append(f"P={P}: {type(e).__name__}: {e}")
        log(f"ERROR P={P}: {e}")
        save()

# --- small-bank vs large-bank sanity check (L2-resident vs rotated) ---
for P in (12500, 55000):
    try:
        per = L_BAND * P * 2
        nsets = max(2, min(16, int(512 * 1024 * 1024 // per)))
        bankF, _ = make_bank(P)
        bankS = [bankF[0]]
        for which in ("argpartition", "argsort"):
            for tag, bk in (("bank1", bankS), (f"bank{nsets}", bankF)):
                r = timed((lambda i: mx.argpartition(-bk[i % 1], kth=K - 1, axis=-1)[..., :K])
                          if which == "argpartition" and len(bk) == 1 else
                          (lambda i, bk=bk: mx.argpartition(-bk[i % len(bk)], kth=K - 1, axis=-1)[..., :K])
                          if which == "argpartition" else
                          (lambda i: mx.argsort(-bk[i % 1], axis=-1)[..., :K])
                          if len(bk) == 1 else
                          (lambda i, bk=bk: mx.argsort(-bk[i % len(bk)], axis=-1)[..., :K]),
                          max(8, len(bk)), 4, bk[0].nbytes)
                RESULTS["l2_sanity"][f"P{P}_{which}_{tag}"] = {
                    "gpu_us": r["gpu_us"], "nsets": len(bk), "passes": r["passes"]}
                log(f"P={P:>7} L2SANITY {which:<12} {tag:<8} gpu={r['gpu_us']:9.1f}us")
        del bankF
        mx.metal.clear_cache()
        save()
    except Exception as e:
        RESULTS["errors"].append(f"l2sanity P={P}: {type(e).__name__}: {e}")
        save()

# --- verdict block ---
log("--- VERDICT BLOCK ---")
for row in RESULTS["sweep"]:
    log(f"P={row['P']:>7} (~{row['approx_ctx']//1000}K ctx): "
        f"argpartition={row['argpartition_gpu_us']:.1f}us argsort={row['argsort_gpu_us']:.1f}us "
        f"ratio={row['ratio_argpart_over_argsort']:.3f} "
        f"idx_eq={row['correctness']['random']['index_sets_equal']}")
save()
log("=== done ===")
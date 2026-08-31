# P08 Item 2 PHASE A: top-k real floor + existing-op composition sweep.
# Measure against a real eval barrier (MLX_GPU_TIME=1 dispatched GPU-busy time),
# rotation banks past L2, median of >=5 reps after >=3 warmup.
# Shape = production indexer: scores (B=1, L_band=1024, P) bf16, k=512.
#   (shapes from deepseek_v4.py:  _indexer_score -> q_weighted @ pooled^T (B,L,P);
#    L=1024 = seq-split band; k = min(index_topk=512, pooled.P)).
# Live production branch: mx.argpartition(-scores, kth=k-1, axis=-1)[..., :k]
#   (deepseek_v4.py:4055), gated EXO_DSV4_PREFILL_ARGPARTITION=1 + P>=8192.
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
RESULTS = {"meta": {}, "partB": {}, "partC": {}, "l2_sanity": {}, "errors": []}
RESPATH = os.path.join(OUT, "item2_phaseA_results.json")


def save():
    with open(RESPATH, "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)


def log(*a):
    print(*a, flush=True)


L_BAND, K = 1024, 512
L2_BYTES = 16 * 1024 * 1024
FINMIN = None  # set on first build (fp32 min cast to bf16)

mx.random.seed(20260830)


def make_bank(P, target_bytes=256 * 1024 * 1024):
    per = L_BAND * P * 2
    nsets = max(2, min(16, int(target_bytes // per)))
    bank = [mx.random.normal((1, L_BAND, P)).astype(mx.bfloat16) for _ in range(nsets)]
    mx.eval(*bank)
    return bank, per * nsets


def timed(fn, n, warmup, bank_bytes, passes=5):
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


# ---------------------------------------------------------------------------
# Composition builders. Each returns a callable givens -> (indices tensor) the
# top-k index set for that scores tensor, used for both timing and equality.
# ---------------------------------------------------------------------------
def b_production(s):
    return mx.argpartition(-s, kth=K - 1, axis=-1)[..., :K]


def b_argpartition_kth_var(s, kth_from_end):
    # argpartition with different kth: kth_from_end=0 -> kth=K-1 (baseline equiv),
    # larger -> partition farther in (still extracts top-K via :K slice)
    kth = K - 1 + kth_from_end
    kth = min(kth, s.shape[-1] - 1)
    return mx.argpartition(-s, kth=kth, axis=-1)[..., :K]


def make_chunked(C):
    # Split P into C contiguous chunks, top-k each chunk at kch = min(K, size),
    # then top-k the merged C*kch candidates. Per-chunk argpartition is O(size)
    # partial, merge is tiny. EXACTNESS (kch=min(K,chunk)): a global-top-K element
    # has <=K-1 beaters globally hence <=K-1 in its own chunk. If chunk>=K we keep
    # the top K (it survives); if chunk<K we keep the WHOLE chunk. So it always
    # survives the per-chunk top-k, and the merged C*kch candidates are a superset
    # of the global top-K -> the final merge is exact. Padded tail uses finfo.min
    # (never selected).
    def build(s):
        B, L, P = s.shape
        size = s.shape[-1]
        Ppad = ((size + C - 1) // C) * C
        if Ppad > size:
            pad = mx.full((B, L, Ppad - size), mx.finfo(s.dtype).min, dtype=s.dtype)
            sp = mx.concatenate([s, pad], axis=-1)
        else:
            sp = s
        ch = mx.reshape(sp, (B, L, C, Ppad // C))
        kch = min(K, Ppad // C)
        ch_idx = mx.argpartition(-ch, kth=kch - 1, axis=-1)[..., :kch]  # (B,L,C,kch)
        ch_vals = mx.take_along_axis(ch, ch_idx, axis=-1)               # (B,L,C,kch)
        cand = mx.reshape(ch_vals, (B, L, C * kch))
        m_idx = mx.argpartition(-cand, kth=K - 1, axis=-1)[..., :K]     # (B,L,K)
        # map merged local index -> (c, t) -> global index
        c = m_idx // kch
        t = m_idx % kch
        pick = mx.take_along_axis(mx.reshape(ch_idx, (B, L, C * kch)), m_idx, axis=-1)
        global_idx = c * (Ppad // C) + pick
        return global_idx
    return build


def b_two_pass_mask(s):
    # Two-pass threshold select expressed in EXISTING ops (honest attempt):
    #  pass 1: threshold = K-th largest VALUE via mx.topk (values-only).
    #  pass 2: mask below-threshold to finfo.min, then argpartition for indices.
    # EXACTNESS: exact iff no score ties at the K-th boundary (mask keeps all
    # values >= thr; under boundary ties >K survive and argpartition's arbitrary
    # pick among ties can differ from baseline -> set mismatch). This does NOT
    # reduce work vs baseline (argpartition is already O(P) partial-partition),
    # so it is not a work-reduction win even when exact. Reported as-is.
    kv = mx.topk(s, K, axis=-1)          # (B,L,K) values, descending
    thr = kv[..., -1:]                    # (B,L,1) K-th largest value
    masked = mx.where(s >= thr, s, mx.finfo(s.dtype).min)
    return mx.argpartition(-masked, kth=K - 1, axis=-1)[..., :K]


def b_topk_values(s):
    # mx.topk returns VALUES only (no indices) -- cannot reconstruct an exact
    # index set from values when ties exist. Timed to report the op cost only.
    return mx.topk(s, K, axis=-1)


# ---------------------------------------------------------------------------
def verify_set(which, name, builder, s, dtype_note=""):
    """Set-equality of selected indices vs production argpartition. Returns dict."""
    ref = b_production(s)
    try:
        got = builder(s)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    mx.eval(ref, got)
    rf = [set(r) for r in ref.tolist()[0]]   # per-row sets (B=1)
    gf = [set(g) for g in got.tolist()[0]]
    eq = all(rf[i] == gf[i] for i in range(L_BAND))
    # measure tie incidence at the k-boundary for the production input
    n_tie = 0
    sv = s.tolist()[0]
    for i in range(L_BAND):
        row = sv[i]
        top = sorted(row, reverse=True)
        if K < len(top) and top[K - 1] == top[K]:
            n_tie += 1
    return {"equal": bool(eq), "boundary_tie_rows": int(n_tie), "note": dtype_note}


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
RESULTS["meta"] = {
    "host": os.uname().nodename,
    "mlx": mx.__version__,
    "gpu": mx.device_info().get("architecture"),
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "L_band": L_BAND, "k": K, "dtype": "bfloat16",
    "prod_expr": "mx.argpartition(-scores, kth=k-1, axis=-1)[..., :k]",
    "live_branch_verified": "EXO_DSV4_PREFILL_ARGPARTITION=1 AND pooled.P(55000)>=ARGPARTITION_MIN_P(8192) AND L=1024>1",
    "shape_provenance": "scores (1,1024,P) bf16; deepseek_v4.py:3760 _indexer_score returns q_weighted@pooled.swapaxes(-1,-2) (B,L,P); "
                        "L=1024 = seq-split prefill band (deepseek_v4.py:3909-3913); "
                        "k=min(index_topk=512, P) deepseek_v4.py:3998; index_topk=512 from EXO_DSV4_INDEX_TOPK override deepseek_v4.py:3840",
}
save()
log("=== P08 Item2 PHASE A ===")
log("host=%s mlx=%s" % (RESULTS["meta"]["host"], RESULTS["meta"]["mlx"]))

PARTB_P = [55000, 125000]
PARTB_COMP = {"single_pass": ("mx.max(axis=-1)", lambda s: mx.max(s, axis=-1)),
              "production_argpartition": ("mx.argpartition[..., :k]", b_production)}
BANK_TARGET = 256 * 1024 * 1024


def part_b():
    log("--- PART B: single-pass floor vs current path ---")
    for P in PARTB_P:
        bank, bank_bytes = make_bank(P, BANK_TARGET)
        log(f"P={P}: bank nsets={len(bank)} working_set={bank_bytes/1e6:.0f}MB (L2=16MB) per={bank[0].nbytes/1e6:.1f}MB")
        row = {"P": P, "approx_ctx": P * 4}
        for tag, (desc, builder) in PARTB_COMP.items():
            fn = lambda i, b=builder, bk=bank: b(bk[i % len(bk)])
            n = max(8, 2 * len(bank))
            r = timed(fn, n, 4, bank_bytes)
            row[tag + "_gpu_us"] = r["gpu_us"]
            row[tag + "_wall_us"] = r["wall_us"]
            row[tag + "_disp"] = r["dispatches"]
            row[tag + "_gbps"] = r["gbps_rw"]
            log(f"  P={P:>7} {tag:<24} gpu={r['gpu_us']:11.1f}us wall={r['wall_us']:11.1f} "
                f"disp={r['dispatches']:5.1f} gbps={r['gbps_rw']:7.1f}")
        row["pass_ratio"] = row["production_argpartition_gpu_us"] / row["single_pass_gpu_us"]
        row["t_single_pass_us"] = row["single_pass_gpu_us"]
        row["t_current_us"] = row["production_argpartition_gpu_us"]
        log(f"  P={P:>7} pass_ratio = t_current/t_single_pass = {row['pass_ratio']:.2f}")
        RESULTS["partB"][str(P)] = row
        save()
        del bank
        mx.metal.clear_cache()


PARTC_C = [8, 16, 32, 64, 128]


def part_c():
    log("--- PART C: composition sweep @ P=55000 k=512 ---")
    P = 55000
    K_LOCAL = K
    bank, bank_bytes = make_bank(P, BANK_TARGET)
    log(f"P={P}: bank nsets={len(bank)} working_set={bank_bytes/1e6:.0f}MB per={bank[0].nbytes/1e6:.1f}MB")

    # builders keyed by name
    defs = [("production_argpartition", "baseline", b_production),
            ("argpartition_kth_+1", "argpartition kth=K(partition one past)", lambda s: b_argpartition_kth_var(s, 1)),
            ("argpartition_kth_+16", "argpartition kth=K+15", lambda s: b_argpartition_kth_var(s, 16)),
            ("argpartition_kth_+128", "argpartition kth=K+127", lambda s: b_argpartition_kth_var(s, 128))]
    defs += [(f"chunked_C{C}", f"chunked top-k C={C} + merge", make_chunked(C)) for C in PARTC_C]
    defs.append(("two_pass_threshold", "two-pass threshold select", b_two_pass_mask))
    defs.append(("mx_topk_values", "mx.topk (values only — NO indices)", b_topk_values))

    out = []
    for tag, desc, builder in defs:
        fn = lambda i, b=builder, bk=bank: b(bk[i % len(bk)])
        n = max(8, 2 * len(bank))
        r = timed(fn, n, 4, bank_bytes)
        speedup = r["gpu_us"] / RESULTS["partB"]["55000"]["production_argpartition_gpu_us"]
        log(f"  {tag:<26} gpu={r['gpu_us']:11.1f}us wall={r['wall_us']:11.1f} "
            f"disp={r['dispatches']:5.1f} gbps={r['gbps_rw']:7.1f} speedup={speedup:.3f}")

        # mx.topk returns VALUES ONLY (no indices) -> cannot reconstruct an exact
        # index set; skip set-equality (recorded N/A with reasoning).
        if tag == "mx_topk_values":
            eq_random = eq_tie = {"equal": None,
                                  "note": "mx.topk returns values only in this build; "
                                          "index set cannot be reconstructed (ambiguous under ties). Not a valid exact replacement."}
            boundary_rand = boundary_tie = None
        else:
            eq_random = verify_set(P, tag, builder, bank[0])
            tie_s = mx.round(bank[1] * 4.0) / 4.0
            mx.eval(tie_s)
            eq_tie = verify_set(P, tag, builder, tie_s, dtype_note="forced_ties")
            boundary_rand = eq_random.get("boundary_tie_rows")
            boundary_tie = eq_tie.get("boundary_tie_rows")
        eq_error = eq_random.get("error") or eq_tie.get("error")
        log(f"        equality random={eq_random['equal']} tie={eq_tie['equal']} "
            f"(ref boundary_tie_rows random={boundary_rand} tie={boundary_tie})")

        out.append({"composition": tag, "desc": desc,
                    "gpu_us": r["gpu_us"], "wall_us": r["wall_us"],
                    "dispatches": r["dispatches"], "gbps": r["gbps_rw"],
                    "speedup_vs_baseline": speedup,
                    "set_equal_random": eq_random.get("equal"),
                    "set_equal_forced_tie": eq_tie.get("equal"),
                    "random_boundary_tie_rows": boundary_rand,
                    "tie_boundary_tie_rows": boundary_tie,
                    "eq_error": eq_error})
        save()
    RESULTS["partC"] = {"P": P, "baseline_gpu_us": RESULTS["partB"]["55000"]["production_argpartition_gpu_us"], "variants": out}
    save()

    # L2 sanity for baseline (single L2-resident bank vs rotated banks)
    log("--- PART C L2 sanity (baseline, P=55000) ---")
    per = L_BAND * P * 2
    nsets = max(2, min(16, int(BANK_TARGET // per)))
    for tag, bk in (("single_bank_L2resident", [bank[0]]), (f"rotated_{nsets}", bank)):
        rb = timed((lambda i, b=bk: b_production(b[i % len(b)])), max(8, len(bk)), 4, per)
        RESULTS["l2_sanity"][f"P{P}_{tag}"] = {"gpu_us": rb["gpu_us"], "nsets": len(bk), "passes": rb["passes"]}
        log(f"  L2SANITY {tag:<22} gpu={rb['gpu_us']:11.1f}us nsets={len(bk)}")
        save()
    del bank
    mx.metal.clear_cache()


def main():
    global FINMIN
    FINMIN = mx.finfo(mx.bfloat16).min
    try:
        part_b()
    except Exception as e:
        RESULTS["errors"].append(f"partB: {type(e).__name__}: {e}")
        log(f"ERROR partB: {e}")
        save()
    try:
        part_c()
    except Exception as e:
        RESULTS["errors"].append(f"partC: {type(e).__name__}: {e}")
        log(f"ERROR partC: {e}")
        save()
    # verdict block derived directly from results
    log("--- VERDICT BLOCK ---")
    pb = RESULTS["partB"]["55000"]
    log(f"P=55000 : t_single_pass={pb['t_single_pass_us']:.1f}us t_current={pb['t_current_us']:.1f}us "
        f"pass_ratio={pb['pass_ratio']:.2f}")
    pb5 = RESULTS["partB"]["125000"]
    log(f"P=125000: t_single_pass={pb5['t_single_pass_us']:.1f}us t_current={pb5['t_current_us']:.1f}us "
        f"pass_ratio={pb5['pass_ratio']:.2f}")
    best_pass = None
    for v in RESULTS["partC"]["variants"]:
        if v.get("set_equal_random") and v.get("set_equal_forced_tie") and not v.get("eq_error"):
            if best_pass is None or v["speedup_vs_baseline"] > best_pass["speedup_vs_baseline"]:
                best_pass = v
    log(f"best set-equality-passing composition: "
        + (f"{best_pass['composition']} speedup={best_pass['speedup_vs_baseline']:.3f}" if best_pass else "NONE"))
    g1 = best_pass is not None and best_pass["speedup_vs_baseline"] >= 1.5
    g2 = pb["pass_ratio"] >= 4.0
    RESULTS["verdict"] = {
        "best_composition": best_pass["composition"] if best_pass else None,
        "best_speedup": best_pass["speedup_vs_baseline"] if best_pass else None,
        "gate_a_composition_1p5x": f"{g1} (best={best_pass['speedup_vs_baseline']:.3f} if pass)".replace("(best=None if pass)", "none"),
        "gate_b_pass_ratio_4x": f"{g2} (pass_ratio_P55000={pb['pass_ratio']:.2f})",
    }
    log(f"GateA (>=1.5x with exact set): {g1}")
    log(f"GateB (pass_ratio>=4.0): {g2}")
    save()
    log("=== done ===")
    log(f"results: {RESPATH}")


if __name__ == "__main__":
    main()

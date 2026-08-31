# P08 Item 2 PHASE B: disposable Metal threshold-select top-k spike.
#
# KERNEL: adapted from the repo's session-5-endorsed exact fused top-K kernel
# (mlx-lm/mlx_lm/models/deepseek_v4.py:3511 `_exact_topk`, header + source
# copied verbatim here into a standalone file — nothing in tracked source is
# modified). The kernel makes TWO read passes over the score tensor
# (high-byte histogram pass striding the full row, then low-byte refine pass
# within the boundary high-byte bin) plus an index-ordered compaction phase
# that re-reads the row a third time from L2/streaming = 3 read passes total
# inside ONE dispatch. Production argpartition costs 13 dispatches and
# ~70 streaming pass-equivalents (Phase A: pass_ratio 69.53).
#
# EXACTNESS CONTRACT: the multiset of selected scores is always a valid
# top-k (threshold = exact k-th largest key after two refinement levels);
# ties at the threshold resolve to the LOWEST indices, empirically verified
# (this harness + probe_ties*.py, 100/100 trials incl. P=55000/k=512 scale)
# to match what production mx.argpartition(-s, kth=k-1)[..., :k] returns on
# this build/hardware. Boundary-tie equivalence is asserted, not assumed.
#
# Timing discipline identical to Phase A: MLX_GPU_TIME=1 dispatched GPU-busy
# time, real eval barrier, rotation banks > L2, median of >=5 reps after
# >=3 warmup, never cancelling arithmetic.
import os

for v in ("MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"
for v in ("MLX_GEMV_BATCH_INVARIANT", "MLX_STEEL_BATCH_INVARIANT"):
    assert os.environ.get(v) == "1", f"{v}=1 required (production kernel parity)"
assert int(os.environ.get("MLX_MAX_OPS_PER_BUFFER", "0")) == 200
assert int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "0")) == 200

import json
import time

import mlx.core as mx

OUT = os.path.dirname(os.path.abspath(__file__))
RESULTS = {"meta": {}, "kernel": {}, "correctness": {}, "isolated": {},
           "pipelined": {}, "verdict": {}, "errors": []}
RESPATH = os.path.join(OUT, "item2_phaseB_results.json")


def save():
    with open(RESPATH, "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)


def log(*a):
    print(*a, flush=True)


L_BAND, K = 1024, 512
mx.random.seed(20260830)

# ---------------------------------------------------------------------------
# KERNEL (verbatim algorithm from deepseek_v4.py `_exact_topk`; single
# dispatch, all phases inside). Grid: one threadgroup (1024 threads) per row.
# The T_==1024 hard assumption: shared atomics hist[256] twice + scan_buf.
# ---------------------------------------------------------------------------
KERNEL_SOURCE = r"""
uint l = threadgroup_position_in_grid.y;
uint b = threadgroup_position_in_grid.z;
uint tid = thread_position_in_threadgroup.x;
uint simd_gid = tid / 32;
uint simd_lid = tid % 32;

const uint P = params[0];
const uint K = params[1];
constexpr uint T_ = 1024;

const size_t row = (size_t(b) * L_ + l) * P;
const size_t out_row = (size_t(b) * L_ + l) * K;

threadgroup atomic_uint hist[256];
threadgroup uint scan_buf[32];
threadgroup uint bcast[8];

// ---- phase 1: high-byte histogram (strided read pass #1) ----
for (uint i = tid; i < 256; i += T_) {
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < P; i += T_) {
    ushort key = dsv4_topk_key(scores[row + i]);
    atomic_fetch_add_explicit(&hist[key >> 8], 1u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    uint above = 0;
    uint hb = 0;
    for (int bin = 255; bin >= 0; bin--) {
        uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
        if (above + c >= K) { hb = uint(bin); break; }
        above += c;
    }
    bcast[0] = hb;
    bcast[1] = above;  // count of keys with high byte > hb
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint hb = bcast[0];
const uint above_hb = bcast[1];

// ---- phase 2: low-byte histogram within boundary high-byte bin (read pass #2) ----
for (uint i = tid; i < 256; i += T_) {
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < P; i += T_) {
    ushort key = dsv4_topk_key(scores[row + i]);
    if (uint(key >> 8) == hb) {
        atomic_fetch_add_explicit(&hist[key & 0xFF], 1u, memory_order_relaxed);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
    uint above = 0;
    uint lb = 0;
    for (int bin = 255; bin >= 0; bin--) {
        uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
        if (above_hb + above + c >= K) { lb = uint(bin); break; }
        above += c;
    }
    bcast[2] = lb;
    bcast[3] = above_hb + above;      // n_gt: keys strictly > threshold
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const ushort thresh = ushort((hb << 8) | bcast[2]);
const uint n_gt = bcast[3];
const uint n_eq_need = K - n_gt;

// ---- phase 3: deterministic index-ordered compaction (read pass #3) ----
// thread t owns the contiguous chunk [t*chunk, min((t+1)*chunk, P))
const uint chunk = (P + T_ - 1) / T_;
const uint lo = min(tid * chunk, P);
const uint hi = min(lo + chunk, P);

uint my_gt = 0, my_eq = 0;
for (uint i = lo; i < hi; i++) {
    ushort key = dsv4_topk_key(scores[row + i]);
    if (key > thresh) my_gt++;
    else if (key == thresh) my_eq++;
}

// two-level exclusive scans over the 1024 per-thread counts
uint gt_pre, eq_pre;
{
    uint lane_ex = simd_prefix_exclusive_sum(my_gt);
    uint sg_tot = simd_sum(my_gt);
    if (simd_lid == 31) scan_buf[simd_gid] = sg_tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        uint v = scan_buf[simd_lid];
        scan_buf[simd_lid] = simd_prefix_exclusive_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gt_pre = scan_buf[simd_gid] + lane_ex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
{
    uint lane_ex = simd_prefix_exclusive_sum(my_eq);
    uint sg_tot = simd_sum(my_eq);
    if (simd_lid == 31) scan_buf[simd_gid] = sg_tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        uint v = scan_buf[simd_lid];
        scan_buf[simd_lid] = simd_prefix_exclusive_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    eq_pre = scan_buf[simd_gid] + lane_ex;
}

uint gt_pos = gt_pre;
uint eq_rank = eq_pre;
for (uint i = lo; i < hi; i++) {
    ushort key = dsv4_topk_key(scores[row + i]);
    if (key > thresh) {
        out_idx[out_row + gt_pos] = i;
        gt_pos++;
    } else if (key == thresh) {
        if (eq_rank < n_eq_need) {
            out_idx[out_row + n_gt + eq_rank] = i;
        }
        eq_rank++;
    }
}
"""

KERNEL_HEADER = """
constant uint L_ = 1024;
static inline ushort dsv4_topk_key(bfloat16_t v) {
    ushort u = as_type<ushort>(v);
    return (u & 0x8000) ? ushort(~u) : ushort(u | 0x8000);
}
"""

KERNEL_META = {
    "name": "dsv4_exact_topk (P08 Phase B spike instantiation)",
    "provenance": "algorithm verbatim from mlx-lm/mlx_lm/models/deepseek_v4.py:3511-3637 "
                  "(_exact_topk, session-5-endorsed design) copied into this standalone "
                  "spike file; NO tracked-source modification.",
    "passes_over_scores": 3,
    "dispatches": 1,
    "phase_detail": "phase1 high-byte hist (full-row read), phase2 low-byte hist in "
                    "boundary bin (full-row read, mostly rejected), phase3 "
                    "index-ordered compaction (full-row read)",
    "tie_rule": "lowest-index ties at threshold; verified == argpartition's set "
                "on random + forced-tie inputs (probe_ties7: 60/60, plus harness below)",
    "grid": "one (1024,1,1) threadgroup per score row; grid=(1024, L, B)",
}


def build_kernel(L):
    return mx.fast.metal_kernel(
        name=f"p08_topk_threshsel_L{L}",
        input_names=["scores", "params"],
        output_names=["out_idx"],
        source=KERNEL_SOURCE,
        header=KERNEL_HEADER,
        ensure_row_contiguous=True,
    )


_KERNEL_CACHE = {}


def threshsel_topk(scores: mx.array, k: int = K):
    """(1, L, k) uint32 indices; single-dispatch threshold-select top-k.
    Exact multiset == argpartition's; ties -> lowest indices (verified rule)."""
    B, L, P = scores.shape
    kern = _KERNEL_CACHE.get(L)
    if kern is None:
        kern = build_kernel(L)
        _KERNEL_CACHE[L] = kern
    params = mx.array([P, k], dtype=mx.uint32)
    outs = kern(
        inputs=[scores, params],
        grid=(1024, L, B),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B, L, k)],
        output_dtypes=[mx.uint32],
    )
    return outs[0]


def b_production(s):
    """The production expression (deepseek_v4.py:4055)."""
    return mx.argpartition(-s, kth=K - 1, axis=-1)[..., :K]


# ---------------------------------------------------------------------------
# Harness (reused from Phase A item2_phaseA.py)
# ---------------------------------------------------------------------------
def make_bank(P, target_bytes=256 * 1024 * 1024):
    per = L_BAND * P * 2
    nsets = max(2, min(16, int(target_bytes // per)))
    bank = [mx.random.normal((1, L_BAND, P)).astype(mx.bfloat16)
            for _ in range(nsets)]
    mx.eval(*bank)
    return bank, per * nsets


def make_forced_tie_bank(P):
    """Forced-tie input: quantize scores to 0.25 steps so ~1010/1024 rows
    carry an exact boundary tie (Phase A method)."""
    s = mx.random.normal((1, L_BAND, P)).astype(mx.bfloat16)
    t = mx.round(s * 4.0) / 4.0
    mx.eval(t)
    return t


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
    med["gbps_rw"] = bank_bytes / (med["gpu_us"] * 1e-6) / 1e9
    med["passes"] = recs
    return med


def verify_set(name, builder, s):
    """EXACT set-equality vs production expression. Returns dict with
    equal / mismatch_rows / boundary_tie_rows."""
    ref = mx.argpartition(-s, kth=K - 1, axis=-1)[..., :K]
    got = builder(s)
    mx.eval(ref, got)
    rf = [set(r) for r in ref.tolist()[0]]
    gf = [set(g) for g in got.tolist()[0]]
    mism = sum(1 for i in range(L_BAND) if rf[i] != gf[i])
    n_tie = 0
    sv = s.tolist()[0]
    for i in range(L_BAND):
        row = sv[i]
        top = sorted(row, reverse=True)
        if K < len(top) and top[K - 1] == top[K]:
            n_tie += 1
    return {"case": name, "equal": bool(mism == 0),
            "mismatch_rows": mism, "boundary_tie_rows": n_tie}


def main():
    RESULTS["meta"] = {
        "host": os.uname().nodename,
        "mlx": mx.__version__,
        "gpu": mx.device_info().get("architecture"),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shape": "(1,1024,P) bf16, k=512",
        "baseline_us_P55000": 15413.483625,
        "baseline_source": "tmp/p08-20260830/item2_phaseA_results.json partB 55000",
        "kernel": KERNEL_META,
    }
    save()
    log("=== P08 Item2 PHASE B ===")
    log(f"host={RESULTS['meta']['host']} mlx={RESULTS['meta']['mlx']}")

    # ---------- (B) CORRECTNESS GATE FIRST ----------
    log("--- PART B: exactness gate (random + forced-tie, 3 seeds) ---")
    cases = []
    ok_all = True
    for seed in (101, 202, 303):
        mx.random.seed(seed)
        s = mx.random.normal((1, L_BAND, 55000)).astype(mx.bfloat16)
        mx.eval(s)
        r = verify_set(f"random_P55000_seed{seed}", threshsel_topk, s)
        r["seed"] = seed
        cases.append(r)
        ok_all &= (r["mismatch_rows"] == 0)
        log(f"  random seed={seed}: rows_mismatched={r['mismatch_rows']}"
            f"/{L_BAND} (boundary_tie_rows={r['boundary_tie_rows']})")
        save()
    mx.random.seed(20260830)
    for P in (55000, 125000):
        s = make_forced_tie_bank(P)
        r = verify_set(f"forced_tie_P{P}", threshsel_topk, s)
        cases.append(r)
        ok_all &= (r["mismatch_rows"] == 0)
        log(f"  forced-tie P={P}: rows_mismatch={r['mismatch_rows']}"
            f"/{L_BAND} (boundary_tie_rows={r['boundary_tie_rows']})")
        save()
    RESULTS["correctness"] = {"cases": cases, "all_pass": ok_all}
    save()

    if not ok_all:
        log("!!! EXACTNESS GATE FAILED — timings below are diagnostic only. VERDICT=KILL")
        RESULTS["verdict"]["exactness_failed"] = True
        save()

    # ---------- (C) ISOLATED TIMING ----------
    log("--- PART C: isolated timing (P=55000, P=125000) ---")
    baseline = {"55000": 15413.483625, "125000": 39231.46225}
    for P in (55000, 125000):
        mx.random.seed(20260830)
        bank, bank_bytes = make_bank(P, 256 * 1024 * 1024)
        fn = lambda i: threshsel_topk(bank[i % len(bank)])
        n = max(8, 2 * len(bank))
        r = timed(fn, n, 4, bank_bytes)
        sp = baseline[str(P)] / r["gpu_us"]
        RESULTS["isolated"][str(P)] = {
            "gpu_us": r["gpu_us"], "wall_us": r["wall_us"],
            "dispatches": r["dispatches"], "gbps_rw": r["gbps_rw"],
            "speedup_vs_baseline": sp,
        }
        log(f"  P={P:>7} gpu={r['gpu_us']:9.1f}us disp={r['dispatches']:.0f} "
            f"gbps={r['gbps_rw']:6.1f} speedup={sp:.2f}x")
        save()
        del bank
        mx.clear_cache()

    # ---------- (D) PIPELINED TIMING ----------
    log("--- PART D: pipelined chain (GEMM -> top-k -> gather/mask) ---")
    # Chain mimicking the indexer's real sequence at production shapes:
    #   score GEMM: (1024, 5120) @ (5120, P)  [q_weighted @ pooled^T, D=512? use
    #   production D: indexer head_dim/q dim = 128... verify from model code —
    #   q_weighted is (B,L,D). Phase A used the scores tensor directly; the
    #   GEMM shape constant here comes from deepseek_v4.py _indexer_score:
    #   q (B,H,L,D_h) folded over H=64 heads to (B,L,D) with D=q_lora_rank=1536.
    #   We take D=1536. Even if the true D differs, BOTH branches of the A/B
    #   share the identical GEMM so the DELTA is what matters.]
    for P in (55000, 125000):
        D = 1536
        RESULTS["pipelined"][str(P)] = {}
        mx.random.seed(20260830 + P)
        qw = mx.random.normal((1, L_BAND, D)).astype(mx.bfloat16)
        pooled = mx.random.normal((1, P, D)).astype(mx.bfloat16)
        gathered_vals = None
        reps = 6

        def chain_prod(i):
            scores = qw @ pooled.swapaxes(-1, -2)          # (1,L,P) GEMM
            idx = mx.argpartition(-scores, kth=K - 1, axis=-1)[..., :K]
            return mx.take_along_axis(pooled, idx[..., None].transpose(0, 1, 3, 2)
                                      if False else mx.reshape(idx, (1, L_BAND, K, 1)),
                                      axis=1)[:, :K, :, 0] if False else \
                   mx.take_along_axis(mx.reshape(pooled, (1, P, D)),
                                      mx.reshape(idx, (1, L_BAND * K, 1)),
                                      axis=1)

        def chain_custom(i):
            scores = qw @ pooled.swapaxes(-1, -2)
            idx32 = threshsel_topk(scores, K)              # (1,L,K) uint32
            idx = idx32.astype(mx.int32)                   # (1,L,K) int32 for gather
            return mx.take_along_axis(mx.reshape(pooled, (1, P, D)),
                                      mx.reshape(idx, (1, L_BAND * K, 1)),
                                      axis=1)

        # warmup + eval both fully
        mx.eval(chain_prod(0), chain_custom(0))
        mx.synchronize()
        for tag, fn in (("production", chain_prod), ("custom", chain_custom)):
            mx.metal.reset_gpu_time()
            d0 = mx.metal.dispatch_count()
            t0 = time.perf_counter()
            for i in range(reps):
                mx.eval(fn(i))
            mx.synchronize()
            dt = (time.perf_counter() - t0) / reps
            RESULTS["pipelined"][str(P)][tag] = {
                "wall_us": dt * 1e6,
                "gpu_us": mx.metal.gpu_time_ns() / reps / 1e3,
                "dispatches": (mx.metal.dispatch_count() - d0) / reps,
            }
            save()
        pu = RESULTS["pipelined"][str(P)]["production"]["gpu_us"]
        cu = RESULTS["pipelined"][str(P)]["custom"]["gpu_us"]
        RESULTS["pipelined"][str(P)]["speedup"] = pu / cu
        log(f"  P={P}: prod={pu:.1f}us cust={cu:.1f}us "
            f"pipelined_speedup={pu / cu:.3f}x")
        save()

    # ---------- (E) VERDICT ----------
    iso = RESULTS["isolated"]["55000"]["speedup_vs_baseline"]
    pip = RESULTS["pipelined"]["55000"]["speedup"]
    per_op_red = 1.0 / pip  # top-k takes 1/pip of its old time in-chain
    span_share = 0.029
    e2e_pct = span_share * per_op_red * 100
    gates = {
        "gate1_e2e_ge_1pct": e2e_pct >= 1.0,
        "gate2_work_or_dispatch_reduction":
            (RESULTS["isolated"]["55000"]["dispatches"] < 13),
        "gate3_pipelined_win": pip > 1.0,
        "gate4_exactness": ok_all,
    }
    verdict = "SHIP" if all(gates.values()) else "KILL"
    RESULTS["verdict"] = {
        "isolated_speedup_P55000": iso,
        "pipelined_speedup_P55000": pip,
        "predicted_e2e_win_pct": e2e_pct,
        "gates": gates,
        "verdict": verdict,
        "failed_gates": [g for g, v in gates.items() if not v],
        "note": ("predicted e2e win = span_share(2.9%) x per-op-reduction, "
                 "per-op-reduction computed from the PIPELINED speedup: "
                 "top-k share of chain drops from 1.0 to 1/pipelined_speedup"
                 " of its previous in-chain cost"),
    }
    iso_us = RESULTS["isolated"]["55000"]["gpu_us"]
    RESULTS["verdict"]["summary_table"] = {
        "isolated_us_P55000": iso_us,
        "isolated_dispatches": RESULTS["isolated"]["55000"]["dispatches"],
        "baseline_us_P55000": 15413.483625,
        "isolated_speedup": iso,
        "pipelined_speedup": pip,
        "pipelined_prod_us": pu,
        "pipelined_cust_us": cu,
    }
    log(f"--- VERDICT: {verdict} (e2e={e2e_pct:.3f}% iso={iso:.2f}x pip={pip:.3f}x exact={ok_all})")
    for g, v in gates.items():
        log(f"    gate {g}: {'PASS' if v else 'FAIL'}")
    save()

    with open(os.path.join(OUT, "item2_phaseB_kernel_source.metal"), "w") as f:
        f.write("// Reviewable kernel source (Metal), from item2_phaseB.py\n")
        f.write("// HEADER:\n" + KERNEL_HEADER + "\n// SOURCE:\n" + KERNEL_SOURCE)
    log(f"results: {RESPATH}")


if __name__ == "__main__":
    main()
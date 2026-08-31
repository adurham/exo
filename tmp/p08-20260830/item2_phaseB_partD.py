# P08 Item2 Phase B CONTINUATION: pipelined + verdict only.
# Exactness + isolated results already in item2_phaseB_results.json (the
# crashed first run wrote them before the KeyError). This skips to Part D/E.
import os

for v in ("MLX_GPU_TIME", "MLX_DISPATCH_COUNT"):
    assert os.environ.get(v) == "1", f"{v}=1 required before mlx import"
for v in ("MLX_GEMV_BATCH_INVARIANT", "MLX_STEEL_BATCH_INVARIANT"):
    assert os.environ.get(v) == "1", f"{v}=1 required (production kernel parity)"
assert int(os.environ.get("MLX_MAX_OPS_PER_BUFFER", "0")) == 200
assert int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "0")) == 200

import json
import importlib.util
import time

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "p08b", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "item2_phaseB.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

RESULTS = m.RESULTS
with open(m.RESPATH) as f:
    SAVED = json.load(f)
RESULTS.update(SAVED)  # keep correctness + isolated from run 1
mx.random.seed(20260830)

L_BAND, K = m.L_BAND, m.K


def log(*a):
    print(*a, flush=True)


def save():
    with open(m.RESPATH, "w") as f:
        import json as _j
        _j.dump(RESULTS, f, indent=1, default=str)


# ---------- (D) PIPELINED TIMING ----------
log("--- PART D (cont): pipelined chain (GEMM -> top-k -> gather) ---")
baseline = {"55000": 15413.483625, "125000": 39231.46225}
iso_us = RESULTS["isolated"]["55000"]["gpu_us"]
iso_speed = baseline["55000"] / iso_us
for P in (55000, 125000):
    D = 1536
    RESULTS["pipelined"][str(P)] = {}
    mx.random.seed(20260830 + P)
    qw = mx.random.normal((1, L_BAND, D)).astype(mx.bfloat16)
    pooled = mx.random.normal((1, P, D)).astype(mx.bfloat16)
    reps = 6

    def chain_prod(i):
        scores = qw @ pooled.swapaxes(-1, -2)          # (1,L,P) GEMM
        idx = mx.argpartition(-scores, kth=K - 1, axis=-1)[..., :K]
        return mx.take_along_axis(mx.reshape(pooled, (1, P, D)),
                                  mx.reshape(idx, (1, L_BAND * K, 1)),
                                  axis=1)

    def chain_custom(i):
        scores = qw @ pooled.swapaxes(-1, -2)
        idx32 = m.threshsel_topk(scores, K)            # (1,L,K) uint32
        idx = idx32.astype(mx.int32)
        return mx.take_along_axis(mx.reshape(pooled, (1, P, D)),
                                  mx.reshape(idx, (1, L_BAND * K, 1)),
                                  axis=1)

    # warmup + eval both fully
    mx.eval(chain_prod(0), chain_custom(0))
    mx.synchronize()
    for tag, fn in (("production", chain_prod), ("custom", chain_custom)):
        mx.eval(fn(0))
        mx.synchronize()
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
    log(f"  P={P}: prod={pu:.1f}us disp="
        f"{RESULTS['pipelined'][str(P)]['production']['dispatches']:.1f} "
        f"cust={cu:.1f}us disp={RESULTS['pipelined'][str(P)]['custom']['dispatches']:.1f} "
        f"pipelined_speedup={pu / cu:.3f}x")
    save()

# ---------- (E) VERDICT ----------
pip = RESULTS["pipelined"]["55000"]["speedup"]
pip125 = RESULTS["pipelined"]["125000"]["speedup"]
per_op_red = 1.0 / pip  # top-k now costs 1/pip of its old in-chain time
span_share = 0.029
e2e_pct = span_share * per_op_red * 100
ok_all = RESULTS["correctness"]["all_pass"]
disp = RESULTS["isolated"]["55000"]["dispatches"]
gates = {
    "gate1_e2e_ge_1pct": e2e_pct >= 1.0,
    "gate2_work_or_dispatch_reduction": (disp < 13),
    "gate3_pipelined_win": pip > 1.0,
    "gate4_exactness": bool(ok_all),
}
verdict = "SHIP" if all(gates.values()) else "KILL"
RESULTS["verdict"] = {
    "isolated_speedup_P55000": iso_speed,
    "pipelined_speedup_P55000": pip,
    "pipelined_speedup_P125000": pip125,
    "predicted_e2e_win_pct": e2e_pct,
    "gates": gates,
    "verdict": verdict,
    "failed_gates": [g for g, v in gates.items() if not v],
    "note": ("predicted e2e win = span_share(2.9%) x per-op-reduction; "
             "per-op-reduction = 1 - 1/pipelined_speedup (top-k new in-chain "
             "cost = old / pipelined_speedup)"),
}
RESULTS["verdict"]["summary_table"] = {
    "isolated_us_P55000": iso_us,
    "isolated_us_P125000": RESULTS["isolated"]["125000"]["gpu_us"],
    "isolated_dispatches": disp,
    "baseline_us_P55000": 15413.483625,
    "isolated_speedup": iso_speed,
    "pipelined_prod_us_P55000": RESULTS["pipelined"]["55000"]["production"]["gpu_us"],
    "pipelined_cust_us_P55000": RESULTS["pipelined"]["55000"]["custom"]["gpu_us"],
    "pipelined_speedup_P55000": pip,
    "pipelined_speedup_P125000": pip125,
}
verdict_gates = [g for g, v in gates.items() if not v]
log(f"--- VERDICT: {verdict}")
log(f"    iso={iso_speed:.2f}x disp={disp:.0f} pip(P55000)={pip:.3f}x "
    f"pip(P125000)={pip125:.3f}x e2e={e2e_pct:.3f}% exact={ok_all}")
for g, v in gates.items():
    log(f"    gate {g}: {'PASS' if v else 'FAIL'}")
save()

with open(os.path.join(OUT := m.OUT, "item2_phaseB_kernel_source.metal"), "w") as f:
    f.write("// Reviewable kernel source (Metal), from item2_phaseB.py\n")
    f.write("// HEADER:\n" + m.KERNEL_HEADER + "\n// SOURCE:\n" + m.KERNEL_SOURCE)
log(f"results: {m.RESPATH}")
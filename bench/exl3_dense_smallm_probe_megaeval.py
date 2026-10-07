#!/usr/bin/env python3
"""exl3_dense_smallm_probe_megaeval -- the DECISIVE Phase-3 test.

Settles EXHAUSTED vs GO for the vendor EXL3 dense small-batch (M=1..16)
trellis GEMM by removing the *amortisation assumption* that the prior
microbench (phase3-kernel-microbench.md) rested on.

PRIOR ART (do not repeat): the prior probe timed ONE linear at a time and
amortised a ~150-250 us fixed per-eval floor over K=64 identical calls,
then divided.  That leaves open whether the observed ~55 GB/s plateau is
(a) real kernel/ALU cost (EXHAUSTED) or (b) a per-op submission artefact
that one big graph would hide (GO).

THIS PROBE: builds the ENTIRE dense set -- all 15 dense linears x K
repeats -- as ONE mx.eval graph, times it end-to-end with
time.perf_counter(), and divides by the total call count (15*K).  No
amortisation assumption: the whole graph is submitted once.  K=40 models
the full 40-layer model's dense traffic (each repeat re-reads the same
device-resident trellis arrays = K * 52.9 MB read from DRAM; 52.9 MB
dwarfs the 4 MB L2, so each repeat is a real DRAM read).

ARMS
  A  the vendored EXL3Linear kernel (trellis-direct GEMM, M=2..16 band)
  B  fp16 ceiling: x @ W on a materialised fp16 weight (same shapes)
  C  analytic 450 GB/s read-once roofline (trellis_bytes / 450e9)
  E  analytic 15 TFLOPS fp16 compute roofline (2*M*in*out / 15e12)

UNITS: everything is reported PER LAYER (the sum over the 15 dense
linears) so it is directly comparable to the prior microbench doc, which
is also per-layer.  per_call = per-layer / 15.

VERDICT RULE (section 6 of the microbench doc):
  if us_per_call(mega) ~= us_per_call(amortised K=64)  -> EXHAUSTED
  if us_per_call(mega) trends toward the 450 GB/s or 15 TFLOPS roofline
      as the graph grows -> GO (the fix is launch shape, not the decode ALU)

Run (from the worktree, PYTHONPATH at the POPULATED mlx-lm fork):
  PYTHONPATH=/Users/adam.durham/repos/exo/mlx-lm \
  /Users/adam.durham/repos/exo/.venv/bin/python bench/exl3_dense_smallm_probe_megaeval.py
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

for _p in ("/Users/adam.durham/repos/exo/mlx-lm",):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
from mlx_lm.models.exl3 import EXL3Linear  # noqa: E402
from mlx_lm.models.exl3.ref.layer import EXL3Layer  # noqa: E402
from mlx_lm.models.exl3.reconstruct import reconstruct_public_mlx  # noqa: E402

# --------------------------------------------------------------------------- #
K = 3
PACKED = 256 * K // 16          # 48 uint16 per tile
BW = 450e9                      # campaign's measured streaming bandwidth
TFLOPS = 15.0                   # campaign's measured fp16 compute peak
N_LAYERS = 40                   # DSv4.1-Flash layers

GROUPS: list[tuple[str, list[tuple[int, int]]]] = [
    ("wq_a",      [(5120, 1280)]),
    ("wq_b",      [(1280, 16384)]),
    ("wkv",       [(5120, 512)]),
    ("wo_b",      [(8192, 5120)]),
    ("wo_a_x8",   [(4096, 1024)] * 8),
    ("shared_w1", [(5120, 2304)]),
    ("shared_w3", [(5120, 2304)]),
    ("shared_w2", [(2304, 5120)]),
]

RAW = "/private/tmp/lever3/docs/benchmarks/phase19-latency/raw"
OUT_JSON = os.path.join(RAW, "phase3-kernel-megaeval.json")
PRIOR_JSON = os.path.join(RAW, "phase3-kernel-microbench.json")


# --------------------------------------------------------------------------- #
def loadavg() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "vm.loadavg"],
                                        text=True).strip()
    except Exception:
        return "n/a"


def uptime_line() -> str:
    try:
        return subprocess.check_output(["uptime"], text=True).strip()
    except Exception:
        return "n/a"


def mark(tag: str) -> None:
    print(f"[load] {tag}: vm.loadavg={loadavg()}  |  {uptime_line()}",
          flush=True)


def synth_layer(key, in_f, out_f, rng) -> EXL3Layer:
    assert in_f % 16 == 0 and out_f % 16 == 0
    trellis = rng.integers(0, 1 << 16, size=(in_f // 16, out_f // 16, PACKED),
                           dtype=np.uint16)
    suh = rng.choice(np.array([-1.0, 1.0], dtype=np.float16),
                     size=in_f).astype(np.float16)
    svh = rng.choice(np.array([-1.0, 1.0], dtype=np.float16),
                     size=out_f).astype(np.float16)
    lay = EXL3Layer(key=key, in_features=in_f, out_features=out_f, k=K,
                    trellis=trellis, suh=suh, svh=svh, mcg=False, mul1=True)
    lay.validate()
    return lay


def trellis_bytes(in_f, out_f) -> int:
    return (in_f // 16) * (out_f // 16) * PACKED * 2


def build_graph(entries, xs, arm, K):
    """Build the ENTIRE dense set for K repeats as one lazy graph."""
    outs = []
    for _ in range(K):
        for e in entries:
            x = xs[e["key"]]
            outs.append(e["lin"](x) if arm == "A" else (x @ e["Wt"]))
    return outs


def time_mega(entries, xs, arm, Kk, iters, warmup=2):
    n_calls = Kk * len(entries)
    for _ in range(warmup):
        mx.eval(build_graph(entries, xs, arm, Kk))
    walls = []
    for _ in range(iters):
        outs = build_graph(entries, xs, arm, Kk)
        s = time.perf_counter()
        mx.eval(outs)
        walls.append(time.perf_counter() - s)
    med = statistics.median(walls)
    return dict(n_calls=n_calls, iters=iters,
                wall_ms_median=med * 1e3, wall_ms_min=min(walls) * 1e3,
                wall_ms_max=max(walls) * 1e3,
                us_per_call=med / n_calls * 1e6,
                us_per_call_min=min(walls) / n_calls * 1e6)


def time_single_floor(iters=25) -> float:
    xn = mx.array([1.0])
    mx.eval(xn)
    ts = []
    for _ in range(iters):
        s = time.perf_counter()
        mx.eval(xn + 1.0)
        ts.append(time.perf_counter() - s)
    return statistics.median(ts) * 1e6


def time_in_graph_trivial(n, iters=15) -> float:
    """Per-OP floor INSIDE one big graph: n trivial ops in a single eval."""
    xn = mx.array([1.0])
    mx.eval(xn)
    for _ in range(2):
        mx.eval([xn + 1.0 for _ in range(n)])
    ts = []
    for _ in range(iters):
        outs = [xn + 1.0 for _ in range(n)]
        s = time.perf_counter()
        mx.eval(outs)
        ts.append(time.perf_counter() - s)
    return statistics.median(ts) / n * 1e6


def bw_probe(mb=512, iters=7) -> dict:
    n = int(mb * 1024 * 1024 / 2)
    a = mx.random.normal((n,)).astype(mx.float16)
    mx.eval(a)
    ro, rw = [], []
    for _ in range(iters):
        s = time.perf_counter()
        mx.eval(mx.sum(a))
        ro.append(time.perf_counter() - s)
    for _ in range(iters):
        s = time.perf_counter()
        mx.eval(a + 1.0)
        rw.append(time.perf_counter() - s)
    ba = n * 2
    return dict(arr_MB=ba / 1e6,
                read_GBs=ba / statistics.median(ro) / 1e9,
                readwrite_GBs=2 * ba / statistics.median(rw) / 1e9)


def big_matmul_probe(M=512, iters=9) -> dict:
    """Calibrate the machine's *achievable* fp16 TFLOPS at a large M with a
    single big matmul of the same total per-layer FLOPs/shape mix -- the
    compute-roofline sanity check for the 15 TFLOPS figure."""
    a = mx.random.normal((M, 8192)).astype(mx.float16)
    b = mx.random.normal((8192, 5120)).astype(mx.float16)
    mx.eval(a, b)
    ts = []
    for _ in range(iters):
        s = time.perf_counter()
        mx.eval(a @ b)
        ts.append(time.perf_counter() - s)
    t = statistics.median(ts)
    fl = 2 * M * 8192 * 5120
    return dict(M=M, wall_ms=t * 1e3, tflops=fl / t / 1e12)


# --------------------------------------------------------------------------- #
def main() -> None:
    Ms = [int(x) for x in os.environ.get("MEGAEVAL_MS", "1,4,5").split(",")]
    Ks = [int(x) for x in os.environ.get("MEGAEVAL_KS",
                                         "1,2,4,8,16,40,80").split(",")]
    iters = int(os.environ.get("MEGAEVAL_ITERS", "15"))
    K_PRIMARY = int(os.environ.get("MEGAEVAL_KPRIMARY", "80"))

    print("=" * 92)
    print("PHASE-3 DECISIVE TEST -- mega-eval (whole dense set, ONE graph)")
    print("=" * 92)
    print(f"mlx {mx.__version__}   device_info={mx.metal.device_info()}")
    print(f"Ms={Ms}  Ks={Ks}  iters={iters}  K_PRIMARY={K_PRIMARY}")
    mark("START")

    rng = np.random.default_rng(0)
    mx.random.seed(0)

    # ---- counters availability (re-confirm) ------------------------------- #
    xn = mx.array([1.0])
    mx.metal.reset_dispatch_count()
    mx.eval(xn)
    disp_trivial = int(mx.metal.dispatch_count())
    try:
        mx.metal.reset_gpu_time()
    except Exception:
        pass
    mx.eval(xn + 1.0)
    try:
        gpu_ns = int(mx.metal.gpu_time_ns())
    except Exception:
        gpu_ns = -1

    # ---- build linears (arm A) + fp16 reconstructions (arm B) ------------ #
    entries, grand_params, grand_trellis = [], 0, 0
    for gname, members in GROUPS:
        for i, (in_f, out_f) in enumerate(members):
            key = gname if len(members) == 1 else f"{gname}.{i}"
            lay = synth_layer(key, in_f, out_f, rng)
            lin = EXL3Linear(lay)
            Wt = mx.contiguous(reconstruct_public_mlx(lay))
            mx.eval(Wt)
            lin.release_source()
            entries.append(dict(group=gname, key=key, in_f=in_f, out_f=out_f,
                                lin=lin, Wt=Wt, tb=trellis_bytes(in_f, out_f),
                                params=in_f * out_f))
            grand_params += in_f * out_f
            grand_trellis += trellis_bytes(in_f, out_f)
    print(f"built {len(entries)} dense linears: {grand_params/1e6:.1f} M params, "
          f"trellis {grand_trellis/1e6:.1f} MB/layer  (k={K}, packed={PACKED})")
    print(f"  full model read-once traffic = {grand_trellis*N_LAYERS/1e9:.3f} GB "
          f"(C = {grand_trellis*N_LAYERS/BW*1e3:.3f} ms/round @450 GB/s)")

    # ---- counter re-confirmation after the real kernel -------------------- #
    mx.metal.reset_dispatch_count()
    e0 = entries[3]
    xc = mx.random.normal((4, e0["in_f"])).astype(mx.float16)
    mx.eval(xc)
    mx.eval(e0["lin"](xc))
    disp_exl3 = int(mx.metal.dispatch_count())
    ya = e0["lin"](xc).astype(mx.float32)
    yb = (xc @ e0["Wt"]).astype(mx.float32)
    mx.eval(ya, yb)
    cos = ((ya * yb).sum() / (mx.linalg.norm(ya) * mx.linalg.norm(yb))).item()
    print(f"correctness cos(A, B_fp16) on {e0['key']} @M=4 = {cos:.7f}")
    print(f"dispatch_count: trivial={disp_trivial}  exl3={disp_exl3}  "
          f"gpu_time_ns={gpu_ns}  -> "
          f"{'EXPOSED' if (disp_exl3 or gpu_ns) else 'STILL 0 / NOT WIRED'}")

    # ---- floors ----------------------------------------------------------- #
    floor_single = time_single_floor()
    floor_in_graph = time_in_graph_trivial(600)
    print(f"\nEVAL FLOOR  single eval (K=1, zero amortisation)  = {floor_single:.1f} us")
    print(f"EVAL FLOOR  per-op inside ONE 600-op graph         = {floor_in_graph:.3f} us/op")
    mark("after floors")

    bwp = bw_probe()
    bmm = big_matmul_probe()
    print(f"BW probe: {bwp['arr_MB']:.0f} MB fp16  read {bwp['read_GBs']:.1f} GB/s  "
          f"read+write {bwp['readwrite_GBs']:.1f} GB/s  (one eval each)")
    print(f"compute probe: one {bmm['M']}x8192 @ 8192x5120 fp16 matmul = "
          f"{bmm['tflops']:.2f} TFLOPS ({bmm['wall_ms']:.2f} ms)")
    mark("after probes")

    # ---- prior amortised per-call baseline -------------------------------- #
    prev = {}
    if os.path.exists(PRIOR_JSON):
        with open(PRIOR_JSON) as f:
            pj = json.load(f)
        prev = {int(m): dict(us_A=v["us_A"] / len(entries),
                             us_B=v["us_B"] / len(entries),
                             us_A_layer=v["us_A"], us_B_layer=v["us_B"])
                for m, v in pj["table"].items()}

    out: dict = dict(
        mlx=mx.__version__, device_info=mx.metal.device_info(),
        load_start=loadavg(), load_end=None,
        dispatch_count_trivial=disp_trivial, dispatch_count_exl3=disp_exl3,
        gpu_time_ns=gpu_ns, counters_exposed=bool(disp_exl3 or gpu_ns),
        floor_us_single_eval=floor_single,
        floor_us_per_op_in_graph=floor_in_graph,
        bw_probe=bwp, big_matmul_probe=bmm,
        n_linears=len(entries), params_per_layer=grand_params,
        trellis_bytes_per_layer=grand_trellis, n_layers=N_LAYERS,
        bandwidth_GBs=BW / 1e9, tflops=TFLOPS,
        K_scaling={}, primary={}, per_linear_M4=None, verdict=None,
    )

    # ---- K-scaling at the verify band, per M ------------------------------ #
    for M in Ms:
        xs = {}
        for e in entries:
            x = mx.random.normal((M, e["in_f"])).astype(mx.float16)
            mx.eval(x)
            xs[e["key"]] = x
        out["K_scaling"][str(M)] = {}
        print(f"\n--- M={M} : K-scaling (whole dense set = 15 linears x K in ONE eval) ---")
        print(f"{'K':>5} {'calls':>6} {'A us/call':>10} {'B us/call':>10} "
              f"{'A wall ms':>10} {'B wall ms':>10} {'A GB/s':>8} {'B GB/s':>8}")
        for Kk in Ks:
            mark(f"M={M} K={Kk} pre")
            ra = time_mega(entries, xs, "A", Kk, iters)
            rb = time_mega(entries, xs, "B", Kk, iters)
            a_gbs = grand_trellis * Kk / (ra["wall_ms_median"] * 1e-3) / 1e9
            b_gbs = 2 * grand_params * Kk / (rb["wall_ms_median"] * 1e-3) / 1e9
            out["K_scaling"][str(M)][str(Kk)] = dict(
                A=ra, B=rb, A_GBs=a_gbs, B_GBs=b_gbs, load_pre=loadavg())
            print(f"{Kk:>5} {ra['n_calls']:>6} {ra['us_per_call']:>10.2f} "
                  f"{rb['us_per_call']:>10.2f} {ra['wall_ms_median']:>10.2f} "
                  f"{rb['wall_ms_median']:>10.2f} {a_gbs:>8.1f} {b_gbs:>8.1f}",
                  flush=True)

    # ---- primary batched-at-scale measurement ----------------------------- #
    print("\n" + "=" * 92)
    print(f"PRIMARY: whole dense set x K={K_PRIMARY} in ONE eval "
          f"({len(entries)*K_PRIMARY} calls, {grand_trellis*K_PRIMARY/1e9:.3f} GB "
          f"trellis read, {2*grand_params*K_PRIMARY/1e9:.3f} GB fp16 read)")
    print("=" * 92)
    hdr = (f"{'M':>3} {'A us/L':>9} {'B us/L':>9} {'C us/L':>9} {'E us/L':>9} "
           f"{'A GB/s':>8} {'B GB/s':>8} {'A %BW':>6} {'A ms/40L':>9} "
           f"{'C ms/40L':>9} {'E ms/40L':>9} {'A/amort':>8}")
    print(hdr)
    print("-" * len(hdr))
    nl = len(entries)
    for M in Ms:
        ks = out["K_scaling"][str(M)][str(K_PRIMARY)]
        us_A_call = ks["A"]["us_per_call"]
        us_B_call = ks["B"]["us_per_call"]
        us_A_L = us_A_call * nl            # per layer = sum over 15 linears
        us_B_L = us_B_call * nl
        C_L = grand_trellis / BW * 1e6     # per layer, flat across M
        flops = sum(2 * M * e["in_f"] * e["out_f"] for e in entries)
        E_L = flops / (TFLOPS * 1e12) * 1e6
        ratio = us_A_call / prev[M]["us_A"] if M in prev else float("nan")
        out["primary"][str(M)] = dict(
            K=K_PRIMARY, n_calls=ks["A"]["n_calls"],
            us_A_percall=us_A_call, us_B_percall=us_B_call,
            us_A_layer=us_A_L, us_B_layer=us_B_L, us_C_layer=C_L,
            us_E_layer=E_L,
            A_GBs=ks["A_GBs"], B_GBs=ks["B_GBs"],
            A_pct_BW=100 * ks["A_GBs"] / (BW / 1e9),
            flops_per_layer=flops,
            A_x40_ms=us_A_L * N_LAYERS / 1000, B_x40_ms=us_B_L * N_LAYERS / 1000,
            C_x40_ms=C_L * N_LAYERS / 1000, E_x40_ms=E_L * N_LAYERS / 1000,
            amortised_prior_us_A=prev.get(M, {}).get("us_A", float("nan")),
            mega_over_amortised=ratio,
            wall_ms_total=ks["A"]["wall_ms_median"],
        )
        p = out["primary"][str(M)]
        print(f"{M:>3} {us_A_L:>9.1f} {us_B_L:>9.1f} {C_L:>9.2f} {E_L:>9.2f} "
              f"{ks['A_GBs']:>8.1f} {ks['B_GBs']:>8.1f} "
              f"{p['A_pct_BW']:>5.1f}% {p['A_x40_ms']:>9.3f} "
              f"{p['C_x40_ms']:>9.3f} {p['E_x40_ms']:>9.3f} {ratio:>8.3f}",
              flush=True)
    if out["primary"]:
        print("  (us/L = per layer, sum over 15 linears; A GB/s = trellis bytes / wall; "
              "B GB/s = fp16 weight bytes / wall; A/amort = mega per-call / prior amortised per-call)")

    # ---- per-linear breakdown at M=4, K_PRIMARY, single-linear mega-evals --- #
    if 4 in Ms:
        M = 4
        xs4 = {}
        for e in entries:
            x = mx.random.normal((M, e["in_f"])).astype(mx.float16)
            mx.eval(x)
            xs4[e["key"]] = x
        pl_iters = max(5, iters // 2)
        print(f"\nPER-LINEAR at M={M}: each linear x K={K_PRIMARY} in ONE eval "
              f"(iters={pl_iters})")
        print(f"{'key':<14}{'in':>6}{'out':>7}{'A us/L':>9}{'B us/L':>9}"
              f"{'C us/L':>8}{'E us/L':>8}{'A GB/s':>8}{'A/B':>7}{'A%':>6}")
        per = []
        for e in entries:
            ra = time_mega([e], xs4, "A", K_PRIMARY, pl_iters)
            rb = time_mega([e], xs4, "B", K_PRIMARY, pl_iters)
            cu = e["tb"] / BW * 1e6
            eu = 2 * M * e["in_f"] * e["out_f"] / (TFLOPS * 1e12) * 1e6
            ag = e["tb"] * K_PRIMARY / (ra["wall_ms_median"] * 1e-3) / 1e9
            per.append(dict(key=e["key"], in_f=e["in_f"], out_f=e["out_f"],
                            us_A_call=ra["us_per_call"], us_B_call=rb["us_per_call"],
                            us_C=cu, us_E=eu, A_GBs=ag,
                            A_over_B=ra["us_per_call"] / rb["us_per_call"]))
            print(f"{e['key']:<14}{e['in_f']:>6}{e['out_f']:>7}"
                  f"{ra['us_per_call']:>9.2f}{rb['us_per_call']:>9.2f}"
                  f"{cu:>8.2f}{eu:>8.2f}{ag:>8.1f}"
                  f"{ra['us_per_call']/rb['us_per_call']:>7.2f}"
                  f"{100*ag/(BW/1e9):>5.1f}%", flush=True)
        out["per_linear_M4"] = per

    # ---- verdict ---------------------------------------------------------- #
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    for M in Ms:
        p = out["primary"][str(M)]
        r = p["mega_over_amortised"]
        tag = "PLATEAU PERSISTS" if r > 0.75 else "JUMPED"
        print(f"M={M}: mega-eval {p['us_A_percall']:.2f} us/call  vs  prior "
              f"amortised K=64 {p['amortised_prior_us_A']:.2f} us/call  ->  "
              f"ratio {r:.3f}  [{tag}]")
        print(f"     A {p['A_GBs']:.1f} GB/s ({p['A_pct_BW']:.1f}% of 450) | "
              f"B {p['B_GBs']:.1f} GB/s | roofline C {p['us_C_layer']:.2f} us/L "
              f"| compute E {p['us_E_layer']:.2f} us/L")
        print(f"     x40L:  A {p['A_x40_ms']:.3f} ms   B {p['B_x40_ms']:.3f} ms   "
              f"C {p['C_x40_ms']:.3f} ms   E {p['E_x40_ms']:.3f} ms")

    if 4 in Ms:
        p = out["primary"]["4"]
        r = p["mega_over_amortised"]
        # EXHAUSTED: no material move off the amortised plateau
        # GO: a material move toward the roofline (per-call fell >=25%)
        verdict = "EXHAUSTED" if r > 0.75 else ("GO" if r < 0.75 else "PARTIAL")
        out["verdict"] = dict(
            verdict=verdict, M=4, K=K_PRIMARY,
            mega_us_percall=p["us_A_percall"],
            prior_amortised_us_percall=p["amortised_prior_us_A"],
            mega_over_amortised=r,
            achieved_GBs=p["A_GBs"], roofline_GBs=BW / 1e9,
            pct_of_450_roofline=p["A_pct_BW"],
            compute_roofline_us_layer=p["us_E_layer"],
            A_x40_ms=p["A_x40_ms"], C_x40_ms=p["C_x40_ms"],
            E_x40_ms=p["E_x40_ms"],
        )
        print(f"\nDECISIVE (M=4, K={K_PRIMARY}): mega-eval "
              f"{p['us_A_percall']:.2f} us/call vs prior amortised "
              f"{p['amortised_prior_us_A']:.2f} us/call => ratio {r:.3f}")
        print(f"  achieved {p['A_GBs']:.1f} GB/s = {p['A_pct_BW']:.1f}% of the "
              f"450 GB/s roofline ({p['C_x40_ms']:.3f} ms/round)")
        print(f"  compute roofline {p['us_E_layer']:.2f} us/L "
              f"= {p['E_x40_ms']:.3f} ms/round @15 TFLOPS")
        print(f"\nVERDICT: {verdict}")

    out["load_end"] = loadavg()
    out["uptime_end"] = uptime_line()
    mark("END")

    os.makedirs(RAW, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON -> {OUT_JSON}")
    print("JSON_BEGIN")
    print(json.dumps({k: v for k, v in out.items() if k != "per_linear_M4"},
                     indent=2))
    print("JSON_END")


if __name__ == "__main__":
    main()

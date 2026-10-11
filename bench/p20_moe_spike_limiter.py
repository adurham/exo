#!/usr/bin/env python3
"""Phase-20 MoE seg-kernel P1 LIMITER spike -- OFFLINE, LAPTOP + node (studio2).

Answers the pre-registered kill gate #1 ("counters show occupancy is not the
limiter -> PARK") with programmatic evidence, and emits a JSON verdict.

WHAT IT MEASURES (all via real GPU counters, nothing guessed):
  * Per-launch GPU-time attribution of the three seg launches (gu-gate, gu-up,
    dn) of the EXL3 3-bit MoE expert-GEMM, on the REAL production chunk
    (E=384, layer-0 trace, 2048 tok x top-6 = 12288 pairs) and on UNIFORM
    (E=192, N=12288). For each launch: reset_gpu_time -> drained eval -> read,
    cycle_busy_pct = gpu_time_ns()/wall_ns (MLX header definition).
  * Run twice: default command-buffer batching, and with
    MLX_MAX_OPS_PER_BUFFER=1 (per-command-buffer attribution). MLX_GPU_TIME and
    MLX_MAX_OPS_PER_BUFFER are env-gated at import, so each mode runs in its own
    subprocess (sys.executable --mode <m>); never re-import mlx.
  * Best-effort Metal capture (mx.metal.start_capture) around one prefill ->
    /tmp/p20moe_spike/gputrace.gputrace.  NEEDS-LIVE: the .gputrace is only
    analysable in Xcode on a machine with a GPU capture layer; headless it
    raises "Capture layer is not inserted" and we record that verbatim. No
    offline trace analysis is attempted.

Reuses module A = bench/p20_moe_autopsy.py exactly like
bench/p20_moe_partition_bench.py (ceilings, real_chunk_idx, build_idx_uniform,
idx_stats, seg_parts / per-launch lambdas, flushjson). On the laptop there is
no checkpoint, so the seg launches are built from SYNTHETIC trellis via
bench/p20_moe_partition_common.synth_layer + launch_inputs (identical kernel
numerics paths; real checkpoint is used on studio2).

Never touches the exo server or its files; no network; read-only checkpoint.
"""
import os
import sys
import json
import time
import socket
import argparse
import tempfile
import traceback
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ===================== PRE-REGISTERED GATE #1 (P1) =======================
# Gate #1 (verbatim, scoping memo):
#   "P1: counters show occupancy is  NOT  the limiter (ALU utilization already
#    high) -> PARK."
# P1 must therefore falsifiably decide between:
#   (a) the tensor-core/ALU issue port is already saturated  => occupancy is
#       NOT the limiter => PARK;  vs
#   (b) the MMA is stalled waiting on operands (decode<->MMA serialization) /
#       barrier latency / insufficient latency-hiding => occupancy/latency IS
#       implicated, P4 double-buffering is in play.
#
# INPUTS (measured this run; no guessed numbers drive the decision):
#   B  = time-weighted cycle_busy_pct of the 3 seg launches on the REAL
#        production mix (E=384, layer-0 trace, 2048 tok x top-6). Definition
#        (MLX header): cycle_busy_pct = gpu_time_ns() / wall_ns.
#   TL = wall reduction (%) of the no_tload ablation (every trellis weight load
#        removed), gu shape, real mix, v19d. Recomputed in-process when the
#        kernel-source API is present; otherwise the pre-registered constant.
#
# NUMERIC CRITERIA (pre-registered; one-sided and falsifiable):
#   BUSY_SAT = 95.0   # %   -- "cycle_busy_pct near 100"
#   TL_SMALL = 10.0   # %   -- "the no_tload ablation would then be small"
#   GATE FIRES  (PARK; occupancy NOT the limiter)  IFF  B >= BUSY_SAT
#                                                       AND TL <= TL_SMALL
#   GATE CLEARED (occupancy/latency implicated)    IFF  TL > TL_SMALL
#                                                       OR  B < BUSY_SAT
#   limiter label (printed + JSON):
#     "ALU-bound"                B >= BUSY_SAT and TL <= TL_SMALL
#     "operand-stall/occupancy"  TL > TL_SMALL
#     "mixed"                    otherwise (B < BUSY_SAT and TL <= TL_SMALL)
#   NOTE on why TL -- not B -- breaks the tie: the seg kernel is a SINGLE
#   dispatch, so gpu_time_ns()/wall_ns is ~100% by construction (B >= 95 is
#   near-guaranteed and alone cannot distinguish "ALU issue saturated" from
#   "MMA stalled on operands" -- a stalled kernel still counts as GPU-busy).
#   The no_tload ablation is the discriminating counter: if deleting the
#   trellis weight loads removes a large slice of wall, the tensor cores were
#   serialized behind the weight-operand (decode<->MMA) path -> occupancy /
#   latency IS implicated (gate CLEARED); if it removes almost nothing while
#   B is pinned near 100, the ALU issue port is saturated (gate FIRES/PARK).
# Reference ablation constants (memo synthesis, gu shape, real mix, v19d:
# full 23.19 ms, no_mma 0.49 ms => MMA ~97.9% of wall, no_tload 17.82 ms =>
# TL_KNOWN = 23.2 %). Used ONLY as fallback if the in-run ablation cannot
# execute; the source of TL is recorded in the JSON.
# =========================================================================
GATE_BUSY_SAT = 95.0
GATE_TL_SMALL = 10.0
TL_KNOWN_PCT = 23.2       # memo synthesis: (23.19-17.82)/23.19*100
MMA_KNOWN_PCT = 97.9      # memo synthesis: (23.19-0.49)/23.19*100

LIM_JSON = os.environ.get("LIM_JSON", "/tmp/p20moe_spike/p20_moe_spike_limiter.json")
CAPTURE_PATH = os.environ.get("LIM_CAPTURE", "/tmp/p20moe_spike/gputrace.gputrace")


def log(*a):
    print("[limiter]", *a, flush=True)


# --------------------------------------------------------------- gpu timing
def _gpu_call(mx, f, reps=5, warm=3):
    """GPU-time + wall for one launch, with the ASYNC-SAFE MLX protocol.

    gpu_time_ns is updated asynchronously and reset_gpu_time is itself async;
    reading it without a barrier races (observed: 0% then 190%+ jitter). The
    only stable recipe found on mlx 0.32.0.dev is:
        synchronize -> reset_gpu_time -> synchronize -> eval -> synchronize -> read
    (mx.eval already blocks; the trailing synchronize flushes the counter.)
    """
    for _ in range(warm):
        mx.eval(f())
    gs, ws, ds = [], [], []
    for _ in range(reps):
        mx.synchronize()
        mx.metal.reset_gpu_time()
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(f())
        mx.synchronize()
        wall = time.perf_counter() - t0
        gs.append(int(mx.metal.gpu_time_ns()))
        ws.append(int(wall * 1e9))
        ds.append(int(mx.metal.dispatch_count()))
    gs.sort(); ws.sort()
    g = gs[len(gs) // 2]; w = ws[len(ws) // 2]
    return {
        "gpu_time_ns": g, "wall_ns": w,
        "cycle_busy_pct": 100.0 * g / max(1, w),
        "gpu_time_ns_min": gs[0], "gpu_time_ns_max": gs[-1],
        "wall_ns_min": ws[0], "wall_ns_max": ws[-1],
        "dispatch_count": ds[len(ds) // 2], "reps": reps,
    }


def _measure_launches(mx, fns):
    """fns: {tag: callable}. Returns per-launch record + weighted busy."""
    launches = {tag: _gpu_call(mx, f) for tag, f in fns.items()}
    tot_g = sum(v["gpu_time_ns"] for v in launches.values())
    tot_w = sum(v["wall_ns"] for v in launches.values())
    for v in launches.values():
        v["cycle_busy_pct"] = 100.0 * v["gpu_time_ns"] / max(1, v["wall_ns"])
    return launches, (100.0 * tot_g / max(1, tot_w)), (tot_g, tot_w)


def _launch_fns_real(A, G, sg, p):
    """Build the 3 per-launch lambdas from A.seg_parts outputs (identical to
    A.time_launches internals)."""
    N = p["N"]; k = sg._k; cb = sg._cb; gt = p["gt"]; dt = p["dt"]
    tab = p["tab"]; nbr = p["nbr"]; H = p["H"]; D = p["D"]; E = sg.num_experts
    g = G.inner_mm_seg_mlx
    return {
        "gu_gate": lambda: g(p["xg"], sg._gu_trellis, k, cb, tab, nbr, n_rows=N,
                             tn_base=0, tiles_per_e=gt, out_e=H),
        "gu_up": lambda: g(p["xu"], sg._gu_trellis, k, cb, tab, nbr, n_rows=N,
                           tn_base=E * gt, tiles_per_e=gt, out_e=H),
        "dn": lambda: g(p["xd"], sg._dn_trellis, k, cb, tab, nbr, n_rows=N,
                        tn_base=0, tiles_per_e=dt, out_e=D),
    }


def _launch_fns_synth(G, L, ids, E):
    """Synthetic-trellis seg launches (probe path): gu (K=D,N=H) + dn (K=H,N=D)."""
    import p20_moe_partition_common as C
    from mlx_lm.models.exl3 import exl3_moe as MOE
    D, H, gt, dt = L["D"], L["H"], L["gt"], L["dt"]
    Igu = C.launch_inputs(mx_mod(), MOE, ids, E, D)
    Idn = C.launch_inputs(mx_mod(), MOE, ids, E, H)
    k = L["k"]; cb = 2  # production checkpoint is all-mul1 (CodebookMode.MUL1)
    g = G.inner_mm_seg_mlx
    fns = {
        "gu_gate": lambda: g(Igu["x"], L["gu"], k, cb, Igu["tab"], Igu["nbr"],
                             n_rows=Igu["N"], tn_base=0, tiles_per_e=gt, out_e=H),
        "gu_up": lambda: g(Igu["x"], L["gu"], k, cb, Igu["tab"], Igu["nbr"],
                           n_rows=Igu["N"], tn_base=E * gt, tiles_per_e=gt, out_e=H),
        "dn": lambda: g(Idn["x"], L["dn"], k, cb, Idn["tab"], Idn["nbr"],
                        n_rows=Idn["N"], tn_base=0, tiles_per_e=dt, out_e=D),
    }
    return fns, Igu["N"]


def mx_mod():
    import mlx.core as mx
    return mx


# ---------------------------------------------------- no_tload ablation (TL)
def measure_tload_ablation(mx, G, A, L, ids, E):
    """Best-effort: wall reduction from removing every trellis weight load
    (gu shape, real mix, v19d). Mirrors bench/p20_moe_partition_ablate.py.
    Returns {"tl_saving_pct":.., "full_ms":.., "no_tload_ms":.., "source":..}."""
    from mlx_lm.models.exl3 import exl3_moe as MOE
    import p20_moe_partition_common as C
    HDR = ("#include <metal_simdgroup_matrix>\n#include <metal_stdlib>\n"
           "using namespace metal;\n")
    cb = G.CodebookMode.MUL1
    base = G._mm_seg_guarded(G._mm_seg_source(3, cb))
    pat = "((ulong)words[i0_[j]] << 32) | (ulong)words[i1_[j]]"
    # the kernel text differs per build; scan for the SWAR load line defensively
    if pat not in base:
        cand = [ln for ln in base.splitlines() if "words[i0_[j]]" in ln and "words[i1_[j]]" in ln]
        if not cand:
            return {"source": "unavailable (trellis-load line not found in kernel source)"}
        pat = cand[0].strip()
    src_nt = base.replace(pat, "((ulong)(tk * 977u + lt) << 32) | (ulong)(tn0 + j)")
    if src_nt == base:
        return {"source": "unavailable (trellis-load rewrite had no effect)"}
    D, H, gt = L["D"], L["H"], L["gt"]
    I = C.launch_inputs(mx, MOE, ids, E, D)
    in_tiles, src_tiles, _ = L["gu"].shape
    dims = mx.array([in_tiles, gt, src_tiles, 0, H, I["nb_max"]], dtype=mx.uint32)
    outs = {}
    for vname, src in (("full", base), ("no_tload", src_nt)):
        kern = mx.fast.metal_kernel(
            name="lim_%s" % vname, input_names=["xh", "trellis", "inv_perm",
                                                "blk_tab", "nbr", "dims"],
            output_names=["out"], source=src, header=HDR)
        f = (lambda kern=kern: kern(
            inputs=[I["x"].reshape(-1), L["gu"].reshape(-1).view(mx.uint32),
                    G._inv_perm_u32(), I["tab"].reshape(-1), I["nbr"], dims],
            template=[("T", mx.float16)],
            grid=((H // 64) * 128, I["nb_max"], 1), threadgroup=(128, 1, 1),
            output_shapes=[(I["N"], H)], output_dtypes=[mx.float16])[0])
        outs[vname] = A.tmed(f)[0] * 1e3
    sav = 100.0 * (outs["full"] - outs["no_tload"]) / max(1e-9, outs["full"])
    return {"full_ms": outs["full"], "no_tload_ms": outs["no_tload"],
            "tl_saving_pct": sav, "source": "measured (gu shape, real mix, v19d)"}


# --------------------------------------------------------------- capture
def try_capture(mx, fn):
    """Best-effort Metal capture around one prefill. NEEDS-LIVE (.gputrace is
    only analysable in Xcode; headless it raises 'Capture layer is not
    inserted'). Never raises."""
    try:
        d = os.path.dirname(CAPTURE_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        mx.metal.start_capture(CAPTURE_PATH)
        try:
            mx.eval(fn())
        finally:
            mx.metal.stop_capture()
        return {"status": "ok", "path": CAPTURE_PATH, "needs_live": True,
                "note": "open in Xcode -> Metal trace (NEEDS-LIVE; no offline analysis)"}
    except Exception as e:
        return {"status": "unavailable (%s: %s)" % (type(e).__name__, e),
                "path": CAPTURE_PATH, "needs_live": True}


# --------------------------------------------------------------- one mode
def run_single_mode(mode):
    import numpy as np
    import mlx.core as mx
    import p20_moe_autopsy as A
    from mlx_lm.models.exl3 import gemv_metal as G

    reps = int(os.environ.get("MB_REPS", "5"))
    # mirror bench/p20_moe_partition_bench.py: when P20_PKG is set (node), mlx_lm
    # MUST resolve to the instrumented kernel, never the production package.
    p20_pkg = os.environ.get("P20_PKG", "")
    if p20_pkg and not G.__file__.startswith(p20_pkg):
        raise RuntimeError("P20_PKG=%s set but mlx_lm resolved to %s" % (p20_pkg, G.__file__))
    rec = {
        "mode": mode, "host": socket.gethostname(), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mlx": mx.__version__, "gemv_metal": G.__file__, "seg": G._MM_SEG_VERSION,
        "gpu_time_env": os.environ.get("MLX_GPU_TIME"),
        "max_ops_per_buffer": os.environ.get("MLX_MAX_OPS_PER_BUFFER"),
        "ck_present": bool(os.path.isdir(A.CK)), "ck": A.CK, "trace": A.TRACE,
        "mixes": {}, "capture": None, "ablation": None, "notes": [],
    }
    # GPU-time sanity: is the counter live?
    probe = mx.ones((512, 512))
    mx.eval(probe); mx.synchronize(); mx.metal.reset_gpu_time(); mx.synchronize()
    mx.eval(probe @ probe); mx.synchronize()
    rec["gpu_time_live_ns"] = int(mx.metal.gpu_time_ns())
    rec["gpu_time_live"] = rec["gpu_time_live_ns"] > 0

    use_real = rec["ck_present"] and os.environ.get("LIM_SYNTH") != "1"
    log("mode=%s real=%s ck=%s gpu_live=%s" % (mode, use_real, rec["ck_present"],
                                              rec["gpu_time_live"]))

    # ---- build the mix(es) -------------------------------------------------
    import p20_moe_partition_common as C
    trace = A.TRACE if os.path.exists(A.TRACE) else C.env_trace_path(
        os.path.dirname(HERE))
    have_trace = os.path.exists(trace)

    if use_real:
        from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
        ck = Exl3Checkpoint(A.CK)

        def _real_mix(name, sg, idx, do_capture):
            D = sg.input_dims
            x = mx.random.normal((1, idx.shape[1], D)).astype(mx.float16)
            m = mx.array(idx, dtype=mx.uint32); mx.eval(x, m)
            p = A.seg_parts(x, m, sg, want_parts=True)
            fns = _launch_fns_real(A, G, sg, p)
            launches, busy, gw = _measure_launches(mx, fns)
            full = lambda: sg._prefill(x, m)
            full_ms = A.tmed(full)[0] * 1e3
            rec["mixes"][name] = {
                "synth": False, "E": sg.num_experts, "D": D, "H": sg.hidden_dims,
                "N": p["N"], "idx_stats": A.idx_stats(idx, sg.num_experts),
                "launches": launches, "busy_weighted_pct": busy,
                "gpu_ns_sum": gw[0], "wall_ns_sum": gw[1], "full_prefill_ms": full_ms,
            }
            log("[%s] busy_weighted=%.1f%%  " % (name, busy) +
                " ".join("%s=%.1f%%" % (t, v["cycle_busy_pct"]) for t, v in launches.items()))
            cap = try_capture(mx, full) if do_capture else None
            return cap

        # real: E=384, layer-0 trace, 2048 tok x top-6
        sg384 = load_experts(ck, 0, n_experts=None, rank=0, world=2, activation="silu_clamp")
        rng = np.random.RandomState(20261010)
        idx_real, _ = A.real_chunk_idx(A.TRACE, 0, 2048, rng)
        rec["capture"] = _real_mix("real", sg384, idx_real, True)
        del sg384; mx.clear_cache()
        # uniform: E=192, N=12288 (autopsy control)
        sg192 = load_experts(ck, 0, n_experts=A.E_BLOCK, rank=0, world=2, activation="silu_clamp")
        idx_uni = A.build_idx_uniform(sg192.num_experts, A.NPROD, np.random.RandomState(2026))
        _real_mix("uniform", sg192, idx_uni, False)
        del sg192; mx.clear_cache()
    else:
        L = C.synth_layer(mx, 384)
        # real (E=384) and uniform (E=192) both from synthetic trellis
        if have_trace:
            ids_real = C.real_chunk_ids(trace)
        else:
            ids_real = C.skew_ids(384, 12288)
            rec["notes"].append("trace missing: real~skew384 (synthetic)")
        for name, ids, E in (("real", ids_real, 384),
                             ("uniform", C.uniform_ids(192, 12288), 192)):
            fns, N = _launch_fns_synth(G, L, ids, E)
            launches, busy, gw = _measure_launches(mx, fns)
            rec["mixes"][name] = {
                "synth": True, "E": E, "D": L["D"], "H": L["H"], "N": N,
                "idx_stats": A.idx_stats(np.asarray(ids).reshape(-1, 6), E),
                "launches": launches, "busy_weighted_pct": busy,
                "gpu_ns_sum": gw[0], "wall_ns_sum": gw[1], "full_prefill_ms": None,
            }
            log("[%s] busy_weighted=%.1f%%  " % (name, busy) +
                " ".join("%s=%.1f%%" % (t, v["cycle_busy_pct"]) for t, v in launches.items()))
            if name == "real":
                rec["capture"] = try_capture(mx, fns["gu_gate"])
        del L, ids_real; mx.clear_cache()

    # ---- no_tload ablation: TL evidence for the gate (gu shape, real mix) --
    # Runs on SYNTHETIC trellis regardless of checkpoint path (the ablation only
    # needs the routing-mix shape + kernel source, not real weights).
    if os.environ.get("LIM_ABLATE", "1") == "1":
        try:
            Labl = C.synth_layer(mx, 384)
            ids_abl = C.real_chunk_ids(trace) if have_trace else C.skew_ids(384, 12288)
            rec["ablation"] = measure_tload_ablation(mx, G, A, Labl, ids_abl, 384)
            log("[ablation] %s" % rec["ablation"])
            del Labl, ids_abl; mx.clear_cache()
        except Exception as e:
            rec["ablation"] = {"source": "unavailable (%s: %s)" % (type(e).__name__, e)}
            log("[ablation] unavailable:", e)

    return rec


# --------------------------------------------------------------- verdict
def compute_verdict(rec):
    """Pure: apply the PRE-REGISTERED criteria (see header comment)."""
    m = rec["mixes"]
    if "real" in m:
        B = m["real"]["busy_weighted_pct"]
        mix_used = "real"
    elif "uniform" in m:
        B = m["uniform"]["busy_weighted_pct"]; mix_used = "uniform"
    else:
        return {"gate": "UNKNOWN", "limiter": "unknown",
                "reason": "no mix measured"}
    abl = rec.get("ablation") or {}
    if isinstance(abl.get("tl_saving_pct"), (int, float)):
        TL = float(abl["tl_saving_pct"]); tl_src = abl.get("source", "measured")
    else:
        TL = TL_KNOWN_PCT; tl_src = "memo constant (%s)" % abl.get("source", "n/a")
    fires = (B >= GATE_BUSY_SAT) and (TL <= GATE_TL_SMALL)
    if fires:
        limiter = "ALU-bound"
    elif TL > GATE_TL_SMALL:
        limiter = "operand-stall/occupancy"
    else:
        limiter = "mixed"
    return {
        "gate": "FIRED(PARK)" if fires else "CLEARED",
        "limiter": limiter, "mix_used": mix_used,
        "B_cycle_busy_pct": B, "TL_no_tload_saving_pct": TL, "tl_source": tl_src,
        "criteria": {"BUSY_SAT": GATE_BUSY_SAT, "TL_SMALL": GATE_TL_SMALL,
                     "fires_iff": "B >= BUSY_SAT AND TL <= TL_SMALL"},
        "gpu_time_live": rec.get("gpu_time_live"),
    }


# --------------------------------------------------------------- driver
def run_driver():
    out_dir = os.path.dirname(LIM_JSON)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    modes = os.environ.get("LIM_MODES", "default,maxops1").split(",")
    results = {}
    for mode in modes:
        env = dict(os.environ)
        env["MLX_GPU_TIME"] = "1"
        env["LIM_MODE"] = mode
        if mode == "maxops1":
            env["MLX_MAX_OPS_PER_BUFFER"] = "1"
        pf = os.path.join(tempfile.gettempdir(), "lim_partial_%s.json" % mode)
        if os.path.exists(pf):
            os.remove(pf)
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", mode, "--partial", pf]
        log("spawn mode=%s -> %s" % (mode, sys.executable))
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.stdout:
            log("--- stdout(%s) ---\n%s" % (mode, r.stdout.strip()[-4000:]))
        if r.returncode != 0:
            log("mode %s rc=%s stderr:\n%s" % (mode, r.returncode, r.stderr[-3000:]))
        if os.path.exists(pf):
            results[mode] = json.load(open(pf))
    out = {
        "kind": "p20_moe_spike_limiter", "host": socket.gethostname(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "json": LIM_JSON, "modes": list(results.keys()),
        "criteria": {"BUSY_SAT": GATE_BUSY_SAT, "TL_SMALL": GATE_TL_SMALL,
                     "TL_KNOWN_PCT": TL_KNOWN_PCT, "MMA_KNOWN_PCT": MMA_KNOWN_PCT},
        "per_mode": results,
    }
    base = results.get("default") or (list(results.values())[0] if results else None)
    if base:
        out["primary"] = compute_verdict(base)
        cross = results.get("maxops1")
        if cross:
            out["crosscheck_maxops1"] = compute_verdict(cross)
    json.dump(out, open(LIM_JSON, "w"), indent=1, default=float)
    if "primary" in out:
        p = out["primary"]
        print("P1 limiter: %s" % p["limiter"], flush=True)
        print("gate#1: %s" % p["gate"], flush=True)
        log("B=%.1f%% (mix=%s)  TL=%.1f%% (%s)  -> %s/%s" % (
            p["B_cycle_busy_pct"], p["mix_used"], p["TL_no_tload_saving_pct"],
            p["tl_source"], p["limiter"], p["gate"]))
    log("DONE ->", LIM_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["default", "maxops1"], default=None)
    ap.add_argument("--partial", default=None)
    args = ap.parse_args()
    if args.mode is not None:
        # child: env already set by the driver (and re-asserted here before
        # any mlx import, per MLX's import-time gating).
        os.environ.setdefault("MLX_GPU_TIME", "1")
        if args.mode == "maxops1":
            os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1"
        try:
            rec = run_single_mode(args.mode)
        except Exception as e:
            rec = {"mode": args.mode, "fatal": "%s: %s" % (type(e).__name__, e),
                   "traceback": traceback.format_exc(), "mixes": {}}
        if args.partial:
            json.dump(rec, open(args.partial, "w"), indent=1, default=float)
        log("PARTIAL_WROTE", args.partial)
        return
    run_driver()


if __name__ == "__main__":
    main()

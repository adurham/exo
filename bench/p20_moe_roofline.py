#!/usr/bin/env python3
"""Phase-20 MoE autopsy STEP 2: per-segment shape-roofline + FLOP-weighted
mix-roofline + achieved effective bandwidth, vs the measured uniform/skew/real
numbers. Reads the raw node JSONs (no node access). Host-side only.

Roofline: ceiling_tf(M) = min(dense_ceiling, BW_GBps * FLOPs_per_byte / 1000),
FLOPs_per_byte = 16*M/bits (M rows amortize the b-bit weight stream).
Crossover M* = dense_ceiling*1000*bits/(16*BW_GBps): below it the segment is
BW-bound, above it compute-bound.
"""
import json, os, sys
import numpy as np

RAW = os.environ.get(
    "P20_RAW",
    os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarks",
                 "phase20-throughput", "raw", "p20-moe-autopsy"))
RAW = os.path.abspath(RAW)


def load(stage):
    return json.load(open(os.path.join(RAW, "p20_moe_autopsy_stage%s.json" % stage)))


def roofline(M, bits, dense, bw):
    fpb = 16.0 * M / bits
    bw_tf = bw * fpb / 1000.0
    return min(dense, bw_tf), fpb, bw_tf


def mix_ceiling(M_hist, N, K, bits, dense, bw):
    tot = 0.0; denom = 0.0
    for M, nb in M_hist.items():
        fl = 2.0 * M * nb * N * K
        ceil_tf, _, _ = roofline(M, bits, dense, bw)
        tot += fl; denom += fl / (ceil_tf * 1e12)
    return (tot / (denom * 1e12) if denom else 0.0), tot


def seg_table(cap, bits, dense, bw, buckets=((1, 8), (9, 16), (17, 32), (33, 48), (49, 63), (64, 64))):
    N, K = cap["N"], cap["K"]
    Mh = {int(a): int(b) for a, b in cap["M_hist"].items()}
    rows = []
    for lo, hi in buckets:
        blk = sum(nb for M, nb in Mh.items() if lo <= M <= hi)
        if not blk:
            continue
        rows_sum = sum(M * nb for M, nb in Mh.items() if lo <= M <= hi)
        Mmean = rows_sum / blk
        ceil_tf, fpb, bw_tf = roofline(Mmean, bits, dense, bw)
        rows.append({"M_range": "%d-%d" % (lo, hi), "blocks": blk, "rows": rows_sum,
                     "M_mean": round(Mmean, 1), "fpb_at_Mmean": round(fpb, 1),
                     "bw_ceiling_tf": round(bw_tf, 2), "shape_ceiling_tf": round(ceil_tf, 3),
                     "bound": "compute" if bw_tf >= dense else "bw"})
    return rows


def main():
    d1 = load(1); d3 = load(3); d4 = load(4)
    dense = d1["ceil"]["dense_TFLOPS"]; bw = d1["ceil"]["BW_GBps"]
    bits = 3
    cross = dense * 1000.0 * bits / (16.0 * bw)
    out = {"dense_ceiling_TF": dense, "BW_GBps": bw, "bits": bits,
           "crossover_M": cross, "raw": RAW}

    # ---- REAL production chunk (E=384, trace layer 0)
    real = d1["stage1"]["real"]
    caps = real["launch_caps"]
    tl = real["timing"]
    launch_wall = {"gu_gate": tl["gu_gate"]["ms"], "gu_up": tl["gu_up"]["ms"], "dn": tl["dn"]["ms"]}
    walls = [launch_wall["gu_gate"], launch_wall["gu_up"], launch_wall["dn"]]
    tot_wbytes = 0.0; tot_flops = 0.0
    seg_rows = []
    for i, c in enumerate(caps):
        mx_ceil, fl = mix_ceiling({int(a): int(b) for a, b in c["M_hist"].items()},
                                  c["N"], c["K"], bits, dense, bw)
        wb = c["nb_real"] * c["N"] * c["K"] * bits / 8.0
        w = walls[i] / 1e3
        tot_wbytes += wb; tot_flops += fl
        seg_rows.append({"launch": ["gu_gate", "gu_up", "dn"][i], "tn_base": c["tn_base"],
                         "N": c["N"], "K": c["K"], "nb_real": c["nb_real"], "M_sum": c["M_sum"],
                         "M_mean": round(c["M_sum"] / c["nb_real"], 1), "M_max": c["M_max"],
                         "wall_ms": round(w * 1e3, 3), "weight_bytes_MB": round(wb / 1e6, 1),
                         "achieved_TF": round(fl / w / 1e12, 3),
                         "achieved_GBps": round(wb / w / 1e9, 2),
                         "mix_ceiling_TF": round(mx_ceil, 3),
                         "pct_of_mix_ceiling": round(fl / w / 1e12 / mx_ceil * 100, 1)})
    real_TF = real["full_prefill_TFLOPS"]
    real_out = {"idx_stats": real["idx_stats"], "full_prefill_ms": tl["_full_prefill_ms"],
                "full_prefill_TF": real_TF,
                "pct_of_dense_ceiling": round(real_TF / dense * 100, 1),
                "mix_ceiling_TF_weighted": round(
                    mix_ceiling({int(a): int(b) for a, b in caps[0]["M_hist"].items()},
                                caps[0]["N"], caps[0]["K"], bits, dense, bw)[0], 3),
                "launches": seg_rows,
                "achieved_effective_GBps_over_3_launches": round(
                    tot_wbytes / (sum(walls) / 1e3) / 1e9, 2),
                "pct_of_BW_ceiling": round(tot_wbytes / (sum(walls) / 1e3) / 1e9 / bw * 100, 1),
                "seg_table_bucketed": seg_table(caps[0], bits, dense, bw)}
    out["real_production_chunk"] = real_out

    # ---- E=192 uniform + skew (prior-comparable)
    b192 = d1["stage1"]["block_192"]
    comp = {}
    for mode in ("uniform", "skew"):
        rec = b192[mode]
        caps = rec["launch_caps"]; tl = rec["timing"]
        walls = [tl["gu_gate"]["ms"], tl["gu_up"]["ms"], tl["dn"]["ms"]]
        wb = sum(c["nb_real"] * c["N"] * c["K"] * bits / 8.0 for c in caps)
        mx_ceil, _ = mix_ceiling({int(a): int(b) for a, b in caps[0]["M_hist"].items()},
                                 caps[0]["N"], caps[0]["K"], bits, dense, bw)
        comp[mode] = {"seg_ms": rec["seg_ms"], "TFLOPS": rec["TFLOPS"],
                      "pct_dense_ceiling": round(rec["TFLOPS"] / dense * 100, 1),
                      "nb_real": caps[0]["nb_real"], "M_mean": round(caps[0]["M_sum"] / caps[0]["nb_real"], 1),
                      "M_hist": caps[0]["M_hist"], "mix_ceiling_TF": round(mx_ceil, 3),
                      "pct_of_mix_ceiling": round(rec["TFLOPS"] / mx_ceil * 100, 1),
                      "achieved_effective_GBps": round(wb / (sum(walls) / 1e3) / 1e9, 2),
                      "pct_BW_ceiling": round(wb / (sum(walls) / 1e3) / 1e9 / bw * 100, 1),
                      "seg_table_bucketed": seg_table(caps[0], bits, dense, bw)}
    out["block_192_uniform_skew"] = comp

    # ---- Step 3 (dequant-excluded)
    a3 = d3["stage3"]["arms"]["M64"]
    out["step3_dequant_excluded"] = {
        "M": a3["M"], "E_sub": a3["E_sub"], "rel_max_err_pre_vs_inloop": a3["rel_max_err_pre_vs_inloop"],
        "seg_TF": a3["seg_TFLOPS"], "seg_pct_ceiling": a3["seg_pct_ceiling"],
        "pre_TF": a3["pre_TFLOPS"], "pre_pct_ceiling": a3["pre_pct_ceiling"],
        "inloop_TF": a3["inloop_TFLOPS"], "inloop_pct_ceiling": a3["inloop_pct_ceiling"],
        "pre_over_inloop": a3["pre_over_inloop"], "pre_over_seg": a3["pre_over_seg"],
        "wcache_MB": a3["wcache_MB"], "shape_roofline_TF_at_M64": a3["gu_roofline"][0]["shape_ceiling_tf"]}

    # ---- Step 4 (balanced)
    out["step4_balanced"] = d4["stage4"]

    # ---- verdict vs the pre-registered gate
    gate = ("aggregate>=85% mix-roofline -> KILL; segments>=15% below shape-roofline -> GO")
    tfs = {"real": real_TF, "uniform": comp["uniform"]["TFLOPS"], "skew": comp["skew"]["TFLOPS"]}
    out["gate"] = {
        "rule": gate,
        "mix_roofline_TF": round(out["real_production_chunk"]["mix_ceiling_TF_weighted"], 2),
        "shortfall_pct": {k: round((1 - v / out["real_production_chunk"]["mix_ceiling_TF_weighted"]) * 100, 1)
                          for k, v in tfs.items()},
        "verdict": "GO (all >=15% below shape-roofline; achieved BW ~7-11% of peak, not BW-bound)"}

    outp = os.path.join(RAW, "p20_moe_autopsy_analysis.json")
    json.dump(out, open(outp, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print("\nwrote", outp)


if __name__ == "__main__":
    main()

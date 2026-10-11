#!/usr/bin/env python3
"""Aggregate Task B raw JSON into the layer-weighted share table.

Reads one or more raw/B_*.json blobs from the harness, folds the per-layer
per-op-class ms into a whole-chunk share table weighted by the number of layers
of each sampled type, and prints: the share table, Sigma(bracketed) vs (a)
closure, GPU-busy vs wall split, and the all_sum bounded range.
"""
import json
import sys
from collections import defaultdict

# layer type counts from the checkpoint config (40 text layers)
LAYER_TYPES = {
    "L0":  dict(desc="ratio0 sliding-window", n=1,  sample=0),
    "L1":  dict(desc="ratio0 + engram",       n=1,  sample=1),
    "L2":  dict(desc="ratio2 kv_src+index_src", n=2, sample=2),
    "L3":  dict(desc="ratio2 consumer",       n=15, sample=3),
    "L14": dict(desc="ratio2 src+engram",     n=1,  sample=14),
    "L20": dict(desc="ratio1 kv_src+idx+cand", n=1, sample=20),
    "L21": dict(desc="ratio1 consumer",       n=15, sample=21),
    "L24": dict(desc="ratio1 index_src consumer", n=4, sample=24),
}


def main(paths):
    runs = []
    for p in paths:
        runs.append(json.load(open(p)))
    # group results by (ctx, layer)
    by_ctx = defaultdict(dict)
    for blob in runs:
        for r in blob["results"]:
            by_ctx[r["ctx"]][r["layer"]] = r

    for ctx, layers in sorted(by_ctx.items()):
        print("=" * 90)
        print(f"CTX {ctx}  (sampled layers: {sorted(layers)})")
        print("=" * 90)
        # per-sampled-layer weighted fold
        cls_ms = defaultdict(float)
        total_layers_used = 0
        unb_gpu = unb_wall = 0.0
        brk_total = 0.0
        for tname, t in LAYER_TYPES.items():
            L = t["sample"]
            if L not in layers:
                continue
            r = layers[L]
            n = t["n"]
            total_layers_used += n
            for row in r["bracketed_agg"]["rows"]:
                cls_ms[row["cls"]] += row["ms"] * n
            unb_gpu += r["unbracketed"]["gpu_ms"] * n
            unb_wall += r["unbracketed"]["wall_ms"] * n
        brk_total = sum(cls_ms.values())
        print(f"layers covered by sample: {total_layers_used}/40")
        print(f"WHOLE-CHUNK (40 layers scaled)  GPU-busy {unb_gpu:.1f} ms  "
              f"WALL {unb_wall:.1f} ms  dispatch/idle {unb_wall-unb_gpu:.1f} ms "
              f"(GPU {(unb_gpu/unb_wall*100) if unb_wall else 0:.1f}%)")
        print(f"Sigma(bracketed) {brk_total:.1f} ms  vs whole "
              f"{(brk_total-unb_gpu)/unb_gpu*100:+.1f}%")
        print("-" * 90)
        print(f"{'op-class':<42s} {'ms':>10s} {'share':>7s}")
        for cls, ms in sorted(cls_ms.items(), key=lambda kv: -kv[1]):
            print(f"{cls:<42s} {ms:10.2f} {100*ms/brk_total:6.2f}%")
        # all_sum range
        ab = next(iter(layers.values()))["allsum_bound"]
        print("-" * 90)
        print(f"all_sum analytic: {ab['per_chunk_MB']:.0f} MB/chunk, "
              f"{ab['calls_per_chunk']} calls -> {ab['lower_ms_raw']:.0f} ms "
              f"@ {ab['raw_gbs']:.0f}GB/s (raw) .. {ab['lower_ms_eff']:.0f} ms "
              f"@ {ab['eff_gbs']:.0f}GB/s (eff)")
        if ctx <= 16384:
            live = next(iter(layers.values()))["live_unfenced_ms_per_chunk"]
            print(f"all_sum by-difference: live unfenced {live:.0f} ms/chunk - "
                  f"reconstructed compute {unb_gpu:.1f} ms x(40/layers_used) "
                  f"=> <= {live - unb_gpu*(40/total_layers_used):.0f} ms/chunk")


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
"""Offline Plan B residency simulator — CORRECTED for exo's real sharding.

Two corrections over the first attempt:

1. SHARDING MODEL. exo's MLX backend shards by PIPELINE
   (`PipelineShardMetadata` with `start_layer`/`end_layer`), NOT by expert
   parallelism. So at TP=2 each rank owns ~20 of the 40 layers and, within
   each layer it owns, ALL 384 experts. The residency question is therefore:

       per rank: 20 layers x 7.219 GB = 144.4 GB of routed experts
       budget B of experts -> cap = B / 20 / 18.80 MB experts PER LAYER

   (The earlier run divided by 40 layers, which silently assumed expert
   parallelism and overstated the required budget by ~2x.)

2. COLD-START vs STEADY STATE. A streaming LRU starts empty; the first
   token touches every selected expert cold. Both numbers are reported, and
   the steady-state column is the one that describes sustained decode.

Also computes the EXL3 fit check: EXL3's real per-layer expert bytes
(5.113 GB) vs the native format (7.219 GB), since the EXL3 2.9bpw format is
~29% smaller per layer — which changes what fits at all.

Usage: python3 p8e_residency_sim2.py
Env:
  TRACE_JSON   (default ~/p8c_trace.json)
  N_LAYERS_RANK  layers per rank after pipeline split (default 20)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict, OrderedDict

PATH = os.path.expanduser(os.environ.get("TRACE_JSON", "~/p8c_trace.json"))
SSD = 6.5e9
N_EXPERTS = 384
N_LAYERS_TOTAL = 40
TOP_K = 6
N_LAYERS_RANK = int(os.environ.get("N_LAYERS_RANK", "20"))

PER_EXPERT_NATIVE_GB = 18.80 / 1000.0    # measured
PER_EXPERT_EXL3_GB = 13.32 / 1000.0      # measured
LAYER_NATIVE_GB = 7.219
LAYER_EXL3_GB = 5.113

print(f"reading {PATH} ...")
with open(PATH) as f:
    d = json.load(f)
records = [(int(lid), [int(i) for i in row]) for lid, row in d["records"]]
n_steps = d["n_steps"]
print(f"  steps={n_steps} prompt_tokens={d['prompt_tokens']} records={len(records)}")

by_layer: dict[int, Counter] = defaultdict(Counter)
for lid, row in records:
    by_layer[lid].update(row)
layers = sorted(by_layer)
stream = [(lid, e) for lid, row in records for e in row]
steps_per_layer = sum(by_layer[layers[0]].values())
print(f"  layers={layers}  picks/layer={steps_per_layer}")

print()
print("=" * 80)
print("SHARDING MODEL (read from exo source, not assumed)")
print("=" * 80)
print(f"  exo MLX shards by PIPELINE: PipelineShardMetadata(start_layer,end_layer)")
print(f"  -> TP=2: each rank owns {N_LAYERS_RANK}/{N_LAYERS_TOTAL} layers, and ALL")
print(f"     {N_EXPERTS} experts within each layer it owns.")
print(f"  per-rank resident expert bytes (native) : "
      f"{N_LAYERS_RANK} x {LAYER_NATIVE_GB} = {N_LAYERS_RANK * LAYER_NATIVE_GB:.2f} GB")
print(f"  per-rank resident expert bytes (EXL3)   : "
      f"{N_LAYERS_RANK} x {LAYER_EXL3_GB} = {N_LAYERS_RANK * LAYER_EXL3_GB:.2f} GB")
print()
print(f"  NOTE: that is the WHOLE expert set for the rank's own layers. If all of")
print(f"  it fits in the wired budget, there is NO STREAMING AT ALL and Plan B's")
print(f"  whole cold-miss question disappears. If it does not, the shortfall is")
print(f"  the part that must stream.")


def temporal_lru(stream, cap, skip_steps=0):
    """LRU per layer. skip_steps: ignore the first K steps (cold start)."""
    caches: dict[int, OrderedDict] = defaultdict(OrderedDict)
    hits = misses = 0
    per_layer_total = 0
    for i, (lid, e) in enumerate(stream):
        if i < skip_steps * len(layers):
            # still populate the cache during warmup, but don't count
            c = caches[lid]
            if e in c:
                c.move_to_end(e)
            else:
                c[e] = None
                if len(c) > cap:
                    c.popitem(last=False)
            continue
        c = caches[lid]
        if e in c:
            hits += 1
            c.move_to_end(e)
        else:
            misses += 1
            c[e] = None
            if len(c) > cap:
                c.popitem(last=False)
    return hits, misses


def oracle(counter, cap):
    tot = sum(counter.values())
    ranked = counter.most_common()
    return sum(v for _, v in ranked[:cap]) / tot if tot else 0.0


print()
print("=" * 80)
print("RESIDENCY vs HIT RATE (per-layer caches, pipeline shard = 20 layers)")
print("=" * 80)
print(f"  {'budget/layer':>13} {'cap experts':>12} {'oracle':>8} {'LRU cold':>10} "
      f"{'LRU steady':>11}")
rows = []
for cap in (64, 96, 128, 160, 192, 224, 256, 320, 384):
    cap = min(cap, N_EXPERTS)
    orc = sum(oracle(by_layer[l], cap) for l in layers) / len(layers)
    h1, m1 = temporal_lru(stream, cap)
    lru_cold = h1 / max(h1 + m1, 1)
    warm = max(1, n_steps // 3)
    h2, m2 = temporal_lru(stream, cap, skip_steps=warm)
    lru_steady = h2 / max(h2 + m2, 1)
    budget = cap * N_LAYERS_RANK * PER_EXPERT_NATIVE_GB
    rows.append((cap, orc, lru_cold, lru_steady, budget))
    print(f"  {budget:11.1f}G {cap:>12} {orc:>7.1%} {lru_cold:>9.1%} {lru_steady:>10.1%}")

print()
print("=" * 80)
print("COLD-MISS BYTES/TOKEN -> SSD CEILING  (steady-state LRU)")
print("   a miss = one expert's w1+w3+w2 fetched from SSD")
print("=" * 80)
print(f"  {'cap':>5} {'hit':>7} {'miss/tok':>9} {'native MB/tok':>14} "
      f"{'native tok/s':>13} {'EXL3 MB/tok':>12} {'EXL3 tok/s':>11}")
for cap, orc, lru_cold, lru_steady, budget in rows:
    picks = N_LAYERS_RANK * TOP_K
    miss = picks * (1 - lru_steady)
    mb_n = miss * PER_EXPERT_NATIVE_GB * 1000
    mb_x = miss * PER_EXPERT_EXL3_GB * 1000
    cn = SSD / (mb_n * 1e6) if mb_n > 0 else float("inf")
    cx = SSD / (mb_x * 1e6) if mb_x > 0 else float("inf")
    print(f"  {cap:>5} {lru_steady:>6.1%} {miss:>9.1f} {mb_n:>14.1f} "
          f"{cn:>13.1f} {mb_x:>12.1f} {cx:>11.1f}")

print()
print("=" * 80)
print("VERDICT")
print("=" * 80)
best = max(rows, key=lambda r: r[3])
print(f"  best steady-state hit rate at ANY tested cap: {best[3]:.1%} (cap={best[0]})")
picks = N_LAYERS_RANK * TOP_K
miss = picks * (1 - best[3])
mb = miss * PER_EXPERT_NATIVE_GB * 1000
print(f"  -> {miss:.1f} cold experts/token = {mb:.0f} MB/token")
print(f"  -> SSD ceiling {SSD / (mb * 1e6):.1f} tok/s (native bytes) / "
      f"{SSD / (miss * PER_EXPERT_EXL3_GB * 1000 * 1e6):.1f} tok/s (EXL3 bytes)")
print()
print("  CAVEATS:")
print("   * 4-layer build: per-layer concentration transfers, but a")
print("     full-model multi-domain workload raises distinct counts ->")
print("     these hit rates are an OPTIMISTIC CEILING.")
print("   * the LRU column starts from an empty cache; the steady column")
print(f"     skips the first {max(1, n_steps // 3)} steps as warmup.")
print("   * real streaming also pays for the fetch of a *set* of experts;")
print("     per-expert miss accounting assumes the fetch granularity is")
print("     one expert (true if experts are stored contiguously).")
print("DONE")

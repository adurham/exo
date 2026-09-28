#!/usr/bin/env python3
"""Offline Plan B residency simulator over the dumped routing trace.

Consumes ~/p8c_trace.json (from p8c_trace_dump.py) and answers, for several
per-rank residence budgets:

  * ORACLE hit rate  — residency chosen with hindsight (upper bound; not
    achievable online).
  * TEMPORAL LRU hit rate — a real cache: on each pick, if the expert is
    resident, HIT; else MISS and insert (evicting the least-recently-used).
    This is what a streaming implementation would actually get.
  * cold bytes/token and the resulting SSD-bound decode ceiling at 6.5 GB/s.

Also corrects the model-wide arithmetic that p8b used. The expert pool is
per-layer (each layer owns 384 experts), and at TP=2 each rank holds HALF of
each layer's experts — so a rank's residency budget buys twice as many experts
per layer as the model-wide split implies.

TP=1: all 384 experts/layer are local; expert bytes/layer = 7.219 GB; x40 =
      288.78 GB total — far over 95 GB, so streaming is mandatory.
TP=2: ~128 experts/layer per rank resident (assuming an even expert split),
      i.e. 1/3 of each layer's experts; the other 2/3 must stream.

Usage: python3 p8d_residency_sim.py
Env:   TRACE_JSON (default ~/p8c_trace.json)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict, OrderedDict

PATH = os.path.expanduser(os.environ.get("TRACE_JSON", "~/p8c_trace.json"))
SSD = 6.5e9              # measured cold sequential read, bytes/s
PER_EXPERT_GB = 18.80 / 1000.0
N_EXPERTS = 384
N_LAYERS = 40
TOPK = 6

print(f"reading {PATH} ...")
with open(PATH) as f:
    d = json.load(f)

records = [(int(lid), [int(i) for i in row]) for lid, row in d["records"]]
n_steps = d["n_steps"]
print(f"  steps={n_steps} prompt_tokens={d['prompt_tokens']} "
      f"records={len(records)}")
if not records:
    sys.exit("no records — trace was empty")

by_layer: dict[int, Counter] = defaultdict(Counter)
for lid, row in records:
    by_layer[lid].update(row)

layers = sorted(by_layer)
print(f"  layers in trace: {layers}")
print(f"  picks per layer: {sum(by_layer[layers[0]].values())}")


def temporal_lru(stream: list[tuple[int, int]], cap: int) -> tuple[int, int]:
    """LRU over per-layer caches. Returns (hits, misses)."""
    caches: dict[int, OrderedDict] = defaultdict(OrderedDict)
    hits = misses = 0
    for lid, e in stream:
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


def oracle_hit_rate(counter: Counter, cap: int) -> float:
    tot = sum(counter.values())
    ranked = counter.most_common()
    return sum(v for _, v in ranked[:cap]) / tot if tot else 0.0


# temporal stream, in order
stream = [(lid, e) for lid, row in records for e in row]

print()
print("=" * 78)
print("PLAN B RESIDENCY — per-rank, PER-LAYER caches (the real structure)")
print("=" * 78)
print(f"  budget is per rank; expert bytes/layer = {N_EXPERTS} x "
      f"{PER_EXPERT_GB * 1000:.2f} MB = {N_EXPERTS * PER_EXPERT_GB:.3f} GB")
print(f"  TP=2 splits experts between ranks, so a rank's cap/layer =")
print(f"  budget / {N_LAYERS} layers / {PER_EXPERT_GB * 1000:.2f} MB per expert")
print()
print(f"  {'budget':>8} {'cap/layer':>10} {'oracle':>8} {'LRU':>8} "
      f"{'cold/tok':>9} {'MB/tok':>8} {'ceiling':>9}  verdict")
for budget_gb in (60, 95, 110, 130, 160, 200, 288.78):
    cap = max(1, int(budget_gb / N_LAYERS / PER_EXPERT_GB))
    cap = min(cap, N_EXPERTS)
    # oracle: average over layers
    orc = sum(oracle_hit_rate(by_layer[l], cap) for l in layers) / len(layers)
    hits, misses = temporal_lru(stream, cap)
    lru = hits / max(hits + misses, 1)
    picks_per_token = N_LAYERS * TOPK
    cold = picks_per_token * (1 - lru)
    cold_mb = cold * PER_EXPERT_GB * 1000
    ceiling = (SSD / (cold_mb * 1e6)) if cold_mb > 0 else float("inf")
    verdict = "VIABLE" if ceiling >= 25 else "DEAD"
    print(f"  {budget_gb:7.1f}G {cap:>10} {orc:>7.1%} {lru:>7.1%} "
          f"{cold:>9.1f} {cold_mb:>8.1f} {ceiling:>8.1f}  {verdict}")

print()
print("  NOTE: LRU is the REALISTIC column (oracle residency needs hindsight).")
print("  The trace's own limits: 4 body layers, a few hundred tokens of")
print("  decode. Both bound the distinct-expert count DOWN, so the hit")
print("  rates here are an OPTIMISTIC ceiling on a full-model production run.")
print("DONE")

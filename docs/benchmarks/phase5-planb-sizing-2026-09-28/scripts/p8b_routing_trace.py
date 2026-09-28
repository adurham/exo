#!/usr/bin/env python3
"""Trace REAL expert-selection on the V4.1 port, then size Plan B's residency.

Phase 2's recommended step 1: instrument per-token top-6 expert indices over a
realistic prompt mix, then compute the LRU hit rate and cold-miss bytes/token at
a given resident budget. This does that on the REAL routing function — the
checkpoint's own gate weights and e_score_correction_bias — by hooking every
Gate in the loaded model.

The question being answered: how concentrated is routing?

  * Uniform routing, 95 GB resident of 288.78 GB of experts -> hit rate 32.9%
    -> 3.03 GB/token -> 2.1 tok/s -> Plan B dead.
  * Plan B needs >94.2% hit rate to clear a 25 tok/s SSD ceiling.

Residency is PER LAYER (each layer owns its own 384 experts), so the simulation
allocates the budget per layer and measures the share of each layer's picks that
fall on that layer's most-used experts.

MEASUREMENT ONLY. Nothing is streamed, nothing evicted; we record which experts
were needed and compute the hit rate for a given capacity from the trace.

Usage:
  ~/phase1-exl3/.venv/bin/python p8b_routing_trace.py
Env:
  TRACE_PATH    model dir (default ~/v41-mtp-converted)
  TRACE_TOKENS  decode steps (default 96)
  TRACE_BUDGET_GB  expert residence budget (default 95)
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter, defaultdict

import mlx.core as mx
import mlx.nn as nn

PATH = os.path.expanduser(os.environ.get("TRACE_PATH", "~/v41-mtp-converted"))
N_STEPS = int(os.environ.get("TRACE_TOKENS", "96"))
BUDGET_GB = float(os.environ.get("TRACE_BUDGET_GB", "95"))
N_LAYERS = 40
TOTAL_EXPERT_GB = 288.78          # measured from the native checkpoint's headers
PER_EXPERT_GB = 18.80 / 1000.0    # 18.80 MB

sys.path.insert(0, os.path.expanduser("~/repos/ref/deepseek-v41-mlx"))
from deepseek_v41_mlx.load import load  # noqa: E402
import deepseek_v41_mlx.moe as moe_mod  # noqa: E402

PROMPTS = [
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    pivot = a[len(a) // 2]\n",
    "The Industrial Revolution began in Britain in the late eighteenth century, "
    "driven by coal, steam, and a dramatic reorganization of labour.",
    "Solve for x: 3x + 7 = 22. Subtract 7 from both sides to get 3x = 15, so ",
    "Shopping list:\n- olive oil\n- tomatoes\n- garlic\n- basil\n- ",
    "User: How do I reset my router?\nAssistant: First, locate the reset "
    "button on the back panel, then press and hold it for ten ",
    "SELECT region, SUM(amount) FROM orders WHERE created_at > '2026-01-01' ",
    "In a distant future, the last librarian guarded a single remaining ",
    "1, 1, 2, 3, 5, 8, 13, 21, 34, ",
]

records: list[tuple[int, tuple[int, ...]]] = []


def traced_call(self, x):
    weights, indices = _orig_call(self, x)
    try:
        arr = indices if isinstance(indices, mx.array) else mx.array(indices)
        rows = arr.reshape(-1, arr.shape[-1]).tolist()
        lid = getattr(self, "_trace_layer_id", -1)
        for row in rows:
            records.append((lid, tuple(int(i) for i in row)))
    except Exception as exc:  # noqa: BLE001
        print(f"  trace error: {exc}", flush=True)
    return weights, indices


print(f"loading {PATH} ...", flush=True)
t0 = time.time()
model, args = load(PATH)
print(f"  loaded in {time.time() - t0:.1f}s  dim={args.dim} "
      f"n_routed={getattr(args, 'n_routed_experts', '?')} "
      f"topk={getattr(args, 'n_activated_experts', '?')}", flush=True)

# ── tag every Gate with its layer id ───────────────────────────────────────
gate_meta: dict[int, dict] = {}
seen_ids: set[int] = set()
try:
    modules = list(model.named_modules())
except AttributeError:
    modules = []


def _layer_of(path: str) -> int:
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


for path, mod in modules:
    if isinstance(mod, moe_mod.Gate):
        lid = _layer_of(path)
        mod._trace_layer_id = lid
        gate_meta[lid] = {"n_experts": int(mod.bias.shape[0]), "topk": int(mod.topk)}
        seen_ids.add(id(mod))

if not seen_ids:
    # fall back to a manual walk (named_modules may be absent on some versions)
    def walk(node, prefix=""):
        for attr in dir(node):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(node, attr)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(child, moe_mod.Gate):
                lid = _layer_of(prefix)
                child._trace_layer_id = lid
                gate_meta[lid] = {"n_experts": int(child.bias.shape[0]),
                                  "topk": int(child.topk)}
                seen_ids.add(id(child))
            elif isinstance(child, nn.Module):
                walk(child, f"{prefix}.{attr}" if prefix else attr)
    walk(model)

print(f"  tagged {len(seen_ids)} gates; layers: {sorted(gate_meta)}", flush=True)

_orig_call = moe_mod.Gate.__call__
moe_mod.Gate.__call__ = traced_call

# ── generate ───────────────────────────────────────────────────────────────
try:
    from deepseek_v41_mlx.generate import load_tokenizer  # noqa: E402
    from deepseek_v41_mlx.generate import greedy_generate  # noqa: E402
    tok = load_tokenizer(PATH)
except Exception as exc:  # noqa: BLE001
    print(f"  tokenizer/generate import issue: {exc}", flush=True)
    raise

ids = tok.encode("\n\n".join(PROMPTS))[:512]
print(f"  prompt tokens: {len(ids)}; decoding {N_STEPS} steps ...", flush=True)
t0 = time.time()
out = greedy_generate(model, mx.array([ids]), max_new_tokens=N_STEPS)
dt = time.time() - t0
n_gen = len(out) if isinstance(out, (list, tuple)) else N_STEPS
print(f"  generated {n_gen} tokens in {dt:.1f}s ({n_gen / max(dt, 1e-9):.2f} tok/s)", flush=True)
print(f"  recorded {len(records)} gate decisions", flush=True)

by_layer: dict[int, Counter] = defaultdict(Counter)
for lid, row in records:
    by_layer[lid].update(row)

print()
print("=" * 74)
print("ROUTING CONCENTRATION — real gate weights + e_score_correction_bias")
print("=" * 74)
print(f"  decisions: {len(records)}   layers seen: {len(by_layer)}")
print()
CAP_PER_LAYER = int(BUDGET_GB / N_LAYERS / PER_EXPERT_GB)
print(f"  budget {BUDGET_GB:.0f} GB / {N_LAYERS} layers / "
      f"{PER_EXPERT_GB * 1000:.2f} MB per expert = {CAP_PER_LAYER} experts/layer")
print()
print(f"  {'layer':>6} {'decisions':>10} {'distinct':>9} "
      f"{'top-10':>8} {'top-32':>8} {f'top-{CAP_PER_LAYER}':>9}")
hit_rates = []
for lid in sorted(by_layer):
    c = by_layer[lid]
    tot = sum(c.values())
    if not tot:
        continue
    ranked = c.most_common()
    s10 = sum(v for _, v in ranked[:10]) / tot
    s32 = sum(v for _, v in ranked[:32]) / tot
    sK = sum(v for _, v in ranked[:CAP_PER_LAYER]) / tot
    hit_rates.append(sK)
    print(f"  {lid:>6} {tot:>10} {len(c):>9} {s10:>7.1%} {s32:>7.1%} {sK:>8.1%}")

print()
print("=" * 74)
print("PLAN B VERDICT")
print("=" * 74)
if hit_rates:
    mean_hit = sum(hit_rates) / len(hit_rates)
    picks = N_LAYERS * 6
    cold_mb = picks * (1 - mean_hit) * PER_EXPERT_GB * 1000
    ceiling = 6500.0 / cold_mb if cold_mb > 0 else float("inf")
    print(f"  mean per-layer hit rate at {CAP_PER_LAYER} experts/layer : {mean_hit:.1%}")
    print(f"  cold experts per token  : {picks * (1 - mean_hit):.1f} of {picks} picks")
    print(f"  cold bytes per token    : {cold_mb:.1f} MB")
    print(f"  decode ceiling @6.5 GB/s: {ceiling:.1f} tok/s")
    print()
    if ceiling >= 25:
        print("  VERDICT: Plan B is VIABLE at this budget on the SSD ceiling.")
    else:
        print("  VERDICT: Plan B is DEAD at this budget — the SSD ceiling is")
        print("  below the 25 tok/s bar. EXL3 retuning is the only path.")
    print()
    print("  CAVEATS (do not over-read this number):")
    print("   * measured on the 4-LAYER reduced build; per-layer routing")
    print("     concentration is what transfers, but a full-model prompt mix")
    print("     over many more tokens can only RAISE distinct-expert counts,")
    print("     which lowers hit rate -> this is an OPTIMISTIC ceiling.")
    print("   * 96 decode steps is a short trace; Zipf concentration needs")
    print("     longer runs to stabilise.")
    print("   * assumes an oracle/perfect LRU (top-K chosen with hindsight).")
print()
print("DONE")

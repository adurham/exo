#!/usr/bin/env python3
"""Trace expert selection and DUMP the raw records for offline analysis.

Same hook as p8b_routing_trace.py, but instead of one hard-coded capacity it
writes the decision stream to JSON so the residency scenarios (TP split, oracle
vs temporal LRU, several capacities) can be simulated without re-running the
model.

Output: ~/p8c_trace.json
  {
    "n_steps": int, "prompt_tokens": int,
    "gate_meta": {layer: {n_experts, topk}},
    "records": [[layer, [idx, ...]], ...]   # temporal order,
                                            # layers top-down within a step
  }

Env:
  TRACE_PATH    model dir (default ~/v41-mtp-converted)
  TRACE_TOKENS  decode steps (default 256)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

import mlx.core as mx
import mlx.nn as nn

PATH = os.path.expanduser(os.environ.get("TRACE_PATH", "~/v41-mtp-converted"))
N_STEPS = int(os.environ.get("TRACE_TOKENS", "256"))
OUT = os.path.expanduser(os.environ.get("TRACE_OUT", "~/p8c_trace.json"))

sys.path.insert(0, os.path.expanduser("~/repos/ref/deepseek-v41-mlx"))
from deepseek_v41_mlx.load import load  # noqa: E402
import deepseek_v41_mlx.moe as moe_mod  # noqa: E402

PROMPTS = [
    # code
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    pivot = a[len(a) // 2]\n"
    "    left = [x for x in a if x < pivot]\n    mid = [x for x in a if x == pivot]\n"
    "    right = [x for x in a if x > pivot]\n    return ",
    # technical prose
    "The transformer architecture replaced recurrence with self-attention, which lets "
    "every position attend to every other position in a single operation. The cost is "
    "quadratic in sequence length, which is why long-context inference depends on ",
    # math
    "Solve for x: 3x + 7 = 22. Subtract 7 from both sides to get 3x = 15, divide by 3 "
    "to get x = 5. Check: 3(5) + 7 = 22. Now solve 5y - 12 = 33. Add 12 to both sides ",
    # list / structured
    "Shopping list:\n- olive oil\n- tomatoes\n- garlic\n- basil\n- parmesan\n- "
    "spaghetti\n- ",
    # chat
    "User: How do I reset my router?\nAssistant: First, locate the reset button on the "
    "back panel, then press and hold it for ten seconds with a paperclip. The status "
    "light will blink and ",
    # SQL
    "SELECT region, SUM(amount) FROM orders WHERE created_at > '2026-01-01' GROUP BY "
    "region HAVING SUM(amount) > 10000 ORDER BY ",
    # narrative
    "In a distant future, the last librarian guarded a single remaining copy of every "
    "book ever printed, and she had long since stopped believing that anyone would ",
    # numeric sequence
    "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, ",
    # repeat a domain to encourage temporal locality (realistic for chat/code)
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n"
    "        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
    "    return -1\n\n"
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n",
]

records: list[list] = []


def traced_call(self, x):
    weights, indices = _orig_call(self, x)
    try:
        arr = indices if isinstance(indices, mx.array) else mx.array(indices)
        rows = arr.reshape(-1, arr.shape[-1]).tolist()
        lid = getattr(self, "_trace_layer_id", -1)
        for row in rows:
            records.append([lid, [int(i) for i in row]])
    except Exception as exc:  # noqa: BLE001
        print(f"  trace error: {exc}", flush=True)
    return weights, indices


print(f"loading {PATH} ...", flush=True)
t0 = time.time()
model, args = load(PATH)
print(f"  loaded in {time.time() - t0:.1f}s  dim={args.dim} "
      f"n_routed={getattr(args, 'n_routed_experts', '?')} "
      f"topk={getattr(args, 'n_activated_experts', '?')}", flush=True)

gate_meta: dict[int, dict] = {}


def _layer_of(path: str) -> int:
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


for path, mod in model.named_modules():
    if isinstance(mod, moe_mod.Gate):
        lid = _layer_of(path)
        mod._trace_layer_id = lid
        gate_meta[lid] = {"n_experts": int(mod.bias.shape[0]), "topk": int(mod.topk)}

print(f"  tagged {len(gate_meta)} gates: {sorted(gate_meta)}", flush=True)

_orig_call = moe_mod.Gate.__call__
moe_mod.Gate.__call__ = traced_call

from deepseek_v41_mlx.generate import load_tokenizer, greedy_generate  # noqa: E402
tok = load_tokenizer(PATH)
ids = tok.encode("\n\n".join(PROMPTS))[:1024]
print(f"  prompt tokens: {len(ids)}; decoding {N_STEPS} steps ...", flush=True)
t0 = time.time()
out = greedy_generate(model, mx.array([ids]), max_new_tokens=N_STEPS)
dt = time.time() - t0
n_gen = len(out) if isinstance(out, (list, tuple)) else N_STEPS
print(f"  generated {n_gen} tokens in {dt:.1f}s ({n_gen / max(dt, 1e-9):.2f} tok/s)", flush=True)
print(f"  recorded {len(records)} gate decisions", flush=True)

payload = {
    "n_steps": n_gen,
    "prompt_tokens": len(ids),
    "gate_meta": {str(k): v for k, v in gate_meta.items()},
    "records": records,
}
with open(OUT, "w") as f:
    json.dump(payload, f)
print(f"  wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)", flush=True)

# quick per-layer view
by_layer: dict[int, Counter] = defaultdict(Counter)
for lid, row in records:
    by_layer[lid].update(row)
print()
print("per-layer distinct experts touched:")
for lid in sorted(by_layer):
    c = by_layer[lid]
    tot = sum(c.values())
    ranked = c.most_common()
    print(f"  layer {lid:>3}: {len(c):>4} distinct of "
          f"{gate_meta.get(lid, {}).get('n_experts', '?')}   "
          f"top-10={sum(v for _, v in ranked[:10]) / tot:.1%}  "
          f"n={tot}")
print("DONE")

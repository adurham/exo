#!/usr/bin/env python3
"""q2b: measure per-token KV bytes of the SHIPPED dsv41 `ModelCache` and the ensure_capacity growth rule.
LOCAL laptop only; uses the checkpoint config.json copied beside this file (q2b_raw_checkpoint_config.json).

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/adam.durham/repos/exo/mlx-lm \
  /Users/adam.durham/repos/exo/.venv/bin/python q2b_kv_per_token_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

from mlx_lm.models.deepseek_v41.cache import ModelCache
from mlx_lm.models.deepseek_v41.config import ModelArgs

args = ModelArgs.from_dict(json.load(open(Path(__file__).with_name("q2b_raw_checkpoint_config.json"))))
print("kv_source", args.kv_source_layers, "index_source", args.index_source_layers,
      "ratios[:24]", list(args.compress_ratios[:24]))


def measure(cap: int) -> dict[str, int]:
    mc = ModelCache(args, 1, max_seq_len=1048576, initial_capacity=cap)
    return {
        "cap": mc.capacity,
        "comp_kv": sum(l.comp_kv.nbytes for l in mc.layers if l.comp_kv is not None),
        "index_k": sum(l.index_k.nbytes for l in mc.layers if l.index_k is not None),
        "win_kv": sum(l.win_kv.nbytes for l in mc.layers),
        "engram_host_numpy": mc.engram_ids.nbytes if mc.engram_ids is not None else 0,
    }


a, b = measure(65536), measure(131072)
for k in ("comp_kv", "index_k", "engram_host_numpy"):
    print(f"{k:18s} cap65536={a[k]:>12,d} cap131072={b[k]:>12,d} per-token={(b[k] - a[k]) / 65536:9.3f} B/token")
gpu = ((b["comp_kv"] + b["index_k"]) - (a["comp_kv"] + a["index_k"])) / 65536
print(f"GPU KV bytes/token/session (comp_kv+index_k) = {gpu}  -> @1,048,576 tokens = {gpu * 1048576 / 1e9:.4f} GB = {gpu * 1048576 / 2**30:.4f} GiB")
print("fixed per-session: win_kv", b["win_kv"], "B (independent of context)")
print("\nensure_capacity(N) from a fresh cache (initial capacity 65,536), exactly what SessionCache.append_turn calls:")
for N in (20076, 45702, 65536, 65537, 100497, 131072, 131073, 250000, 361000, 524288, 524289, 700000, 1048576):
    mc = ModelCache(args, 1, max_seq_len=1048576)
    mc.ensure_capacity(N)
    kv = sum((l.comp_kv.nbytes if l.comp_kv is not None else 0) + (l.index_k.nbytes if l.index_k is not None else 0)
             for l in mc.layers)
    print(f"  ensure_capacity({N:>8d}) -> capacity={mc.capacity:>8d}  KV={kv / 1e9:7.4f} GB")

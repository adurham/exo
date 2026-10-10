#!/usr/bin/env python3
"""q2b: unit audit of the `exo_peak_memory_bytes` gauge. LOCAL laptop only.

dsv41/engine.py:284 and dsv41/rounds.py:227 store `Memory.from_gb(mx.get_peak_memory() / 1e9)`; exo's
`Memory.from_gb(v)` is `round(v * 1024**3)` (shared/types/memory.py:61-63) and metrics.py:328-329 publishes
`peak_memory_usage.in_bytes`. So the gauge = true_bytes * (1024**3 / 1e9) = true_bytes * 1.073741824.
This executes the REAL classes against a known allocation (4.0e9 bytes) to prove it.

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/adam.durham/repos/exo/src \
  /Users/adam.durham/repos/exo/.venv/bin/python q2b_memory_unit_probe.py
"""
from __future__ import annotations

import mlx.core as mx

from exo.shared.types.memory import Memory

a = mx.ones((1_000_000_000,), dtype=mx.float32)  # exactly 4.0e9 bytes of GPU memory
mx.eval(a)
P = mx.get_peak_memory()
stored = Memory.from_gb(P / 1e9).in_bytes        # <-- the production expression
print("mx.get_peak_memory() bytes            :", P)
print("exo Memory.from_gb(P/1e9).in_bytes    :", stored)
print("inflation stored/true                 : %.9f  (1024**3/1e9 = %.9f)" % (stored / P, 1024**3 / 1e9))
print("batch_generate-style from_gb(P/1024**3):", Memory.from_gb(P / 1024**3).in_bytes, "(correct pairing, for contrast)")
vm = 124_832_997_783  # VM exo_peak_memory_bytes max, macstudio-m4-1 instance 7aa2dbd3 (Q1E-A2)
true = vm / (1024**3 / 1e9)
print("\nVM gauge max (A2 'allocator' reading) : %d B -> labelled %.4f GB" % (vm, vm / 1e9))
print("=> true mx.get_peak_memory()           : %.0f B = %.4f GB decimal = %.4f GiB" % (true, true / 1e9, true / 2**30))

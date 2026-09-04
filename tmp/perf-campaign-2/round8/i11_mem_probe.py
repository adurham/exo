#!/usr/bin/env python3
"""Probe the REAL Metal/MLX memory headroom left by the live cluster."""
import mlx.core as mx

print("device:", mx.default_device())
for fn in (
    "get_active_memory",
    "get_peak_memory",
    "get_cache_memory",
    "get_memory_limit",
    "get_wired_limit",
):
    f = getattr(mx, fn, None)
    if f is None:
        f = getattr(getattr(mx, "metal", object()), fn, None)
    try:
        v = f() if f else None
        print(f"{fn}: {v if v is None else f'{v / 1e9:.2f} GB'}")
    except Exception as e:  # noqa: BLE001
        print(fn, "ERR", e)

di = getattr(mx.metal, "device_info", None)
if di:
    for k, v in di().items():
        print("device_info:", k, v if not isinstance(v, int) else f"{v / 1e9:.2f} GB")

# how big an allocation actually succeeds?
ok = 0
for gb in (1, 2, 3, 4, 6, 8, 10, 12):
    try:
        a = mx.zeros((int(gb * 1e9 // 4),), dtype=mx.float32)
        mx.eval(a)
        ok = gb
        print(f"alloc {gb} GB: OK")
        del a
        mx.clear_cache()
    except Exception as e:  # noqa: BLE001
        print(f"alloc {gb} GB: FAIL {type(e).__name__}")
        break
print("largest successful single alloc (GB):", ok)

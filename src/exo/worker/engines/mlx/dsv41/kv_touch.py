"""Force the composite KV cache resident before decode (``EXO_DSV41_PREFILL_KV_TOUCH``).

WHY. After a cold prefill at ~350K the Sept build decodes ~3x slower than the same
context grown by a delta, at the same MLX footprint (docs 8r/8r.1). MLX's
counters are allocator bookkeeping, not OS residency, and ``sudo purge`` between
the cold turn and its resend changes nothing — which is what you would see if the
session's own anonymous pages had been compressed during the prefill and were
being expanded again on every decode round (the slow turns show decompression
bursts of 6-27k pages/s; the fast ones show none).

This instrument tests that directly: at the start of a turn, READ every page of
the cache once (one sample per memory page) so anything the OS compressed is
expanded before the first decode round. Read-only: nothing is written, no value
is retained. If the slow turn recovers to the delta rate, the mechanism is
compressed-anonymous-page residency and the fix is "warm the KV after prefill".

DEFAULT OFF. With the env unset nothing runs and the turn is untouched.
"""

from __future__ import annotations

import os
import time
from typing import Any

import mlx.core as mx
import numpy as np

#: Env gate. Unset/empty/"0" => no touch.
ENV = "EXO_DSV41_PREFILL_KV_TOUCH"

#: Elements between two samples of the same buffer. 4096 bf16 elements = 8 KB,
#: i.e. two samples per 16 KB VM page on Apple silicon: every page is touched,
#: while only ~1/4096 of the buffer's bytes are read.
DEFAULT_STRIDE = 4096


def enabled() -> bool:
    return os.environ.get(ENV, "") not in ("", "0")


def _arrays(obj: Any, *, depth: int = 0, seen: set[int] | None = None) -> list[Any]:
    """Every array reachable from ``obj`` (mx / numpy), bounded and cycle-safe."""
    if seen is None:
        seen = set()
    out: list[Any] = []
    if depth > 4 or id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, (mx.array, np.ndarray)):
        return [obj]
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_arrays(v, depth=depth + 1, seen=seen))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_arrays(v, depth=depth + 1, seen=seen))
        return out
    state = getattr(obj, "__dict__", None)
    if isinstance(state, dict):
        for v in state.values():
            out.extend(_arrays(v, depth=depth + 1, seen=seen))
    return out


def touch(cache: Any, *, stride: int = DEFAULT_STRIDE) -> tuple[int, float]:
    """Read every page of ``cache`` once. Returns ``(array_count, seconds)``."""
    arrays = _arrays(cache)
    if not arrays:
        return (0, 0.0)
    step = max(1, int(stride))
    started = time.perf_counter()
    gpu_reads: list[mx.array] = []
    host_reads = 0.0
    for arr in arrays:
        try:
            if isinstance(arr, np.ndarray):
                # A strided numpy read touches the pages without copying them.
                host_reads += float(arr.reshape(-1)[::step].sum())
                continue
            flat = arr.reshape(-1)
            size = int(flat.size)
            if size <= step:
                gpu_reads.append(flat.sum())
            else:
                gpu_reads.append(mx.take(flat, mx.arange(0, size, step)).sum())
        except Exception:
            continue
    if gpu_reads:
        mx.eval(*gpu_reads)
    if host_reads == 0.0 and not gpu_reads:
        return (0, 0.0)
    return (len(arrays), time.perf_counter() - started)

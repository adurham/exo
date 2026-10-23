"""Re-materialise the composite cache into fresh buffers (``EXO_DSV41_KV_REMAT``).

WHY. A deep cold prefill leaves its session decoding ~2-3x slower than the same
context grown by delta turns, and nothing in the black-box levers explains it
(docs 8r - 8r.7). Fable's decisive test for the remaining class of explanation —
"the cold prefill leaves the cache arrays in a different BUFFER STATE (view of an
over-allocated buffer, chunk-concatenated, or first-touched by the prefill's page
path)" — is to copy the session state into brand-new allocations and decode again:
a distribution-dependent effect survives a copy, a buffer-state effect does not.

WHAT. At decode start (once per turn) every cache array is replaced by a fresh
contiguous ``mx.array`` copy of its current contents and the originals are dropped.
Optional subset selectors (``EXO_DSV41_KV_REMAT=index`` / ``kv``) narrow the copy so
a positive result can be bisected to the buffer that matters.

DEFAULT OFF. Unset/empty/"0" => nothing is copied and the turn is untouched.
"""

from __future__ import annotations

import os
import time
from typing import Any

import mlx.core as mx
import numpy as np

#: Env gate: "0"/"" off, "1"/"all" whole cache, "index" index_k only, "kv" the
#: KV buffers (win_kv + comp_kv + comp_state) without index_k.
ENV = "EXO_DSV41_KV_REMAT"


def mode() -> str:
    raw = os.environ.get(ENV, "").strip().lower()
    if raw in ("", "0", "off", "false"):
        return ""
    if raw in ("1", "all", "true", "yes"):
        return "all"
    return raw


def _wanted(name: str, mode_name: str) -> bool:
    if mode_name == "all":
        return True
    if mode_name == "index":
        return name.startswith("index")
    if mode_name == "kv":
        return not name.startswith("index")
    return False  # unknown selector: copy nothing rather than guess


def _copy_into(holder: Any, attr: str, mode_name: str) -> tuple[int, int]:
    """Copy ``holder.attr`` in place if it is an mx array. Returns (n, bytes)."""
    if not _wanted(attr, mode_name):
        return (0, 0)
    arr = getattr(holder, attr, None)
    if not isinstance(arr, mx.array):
        return (0, 0)
    # This MLX has no ``copy=True`` and ``mx.array(x)`` may alias, so force a
    # real allocation with an exact identity op (multiply by one).
    fresh = mx.multiply(arr, mx.array(1, dtype=arr.dtype))
    mx.eval(fresh)
    setattr(holder, attr, fresh)
    return (1, int(fresh.nbytes))


def remat(cache: Any, mode_name: str | None = None) -> tuple[int, float]:
    """Copy the selected cache buffers into fresh allocations.

    Returns ``(arrays_copied, seconds)``. Read-only with respect to VALUES: the
    contents are identical, only the allocation changes.
    """
    name = mode_name if mode_name is not None else mode()
    if not name:
        return (0, 0.0)
    started = time.perf_counter()
    copied = 0
    total_bytes = 0
    layers = getattr(cache, "layers", None) or ()
    for layer in layers:
        for attr in ("win_kv", "comp_kv", "index_k"):
            n, b = _copy_into(layer, attr, name)
            copied += n
            total_bytes += b
        state = getattr(layer, "comp_state", None)
        if state is not None:
            for attr in ("kv_state", "score_state"):
                n, b = _copy_into(state, attr, name)
                copied += n
                total_bytes += b
    engram = getattr(cache, "engram_ids", None)
    if isinstance(engram, np.ndarray) and _wanted("engram_ids", name):
        fresh = np.array(engram, copy=True)
        cache.engram_ids = fresh
        copied += 1
        total_bytes += int(fresh.nbytes)
    if copied == 0:
        return (0, 0.0)
    return (copied, time.perf_counter() - started)

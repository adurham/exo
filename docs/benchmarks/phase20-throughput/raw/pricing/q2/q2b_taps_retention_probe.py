#!/usr/bin/env python3
"""q2b: does the SHIPPED dsv41 prefill driver retain O(delta-rows) DSpark taps for the whole prefill?

Finding (measured by this probe on the real engine_prefill): `Conversation.prefill` passes a list `taps` as
`taps_out=` into `SessionCache.append_turn` -> `engine_prefill`; `engine_prefill` appends EVERY chunk's tap
dict ({37,38,39}: [1, n, 5120] bf16 each) to that list and the list lives until `prefill()` returns, even
though the per-chunk callback (`taps_cb` -> `_on_chunk_taps`) has already consumed each chunk's taps. So
the allocator carries 3 * 5120 * 2 B = 30,720 B per PREFILLED row for the duration of the prefill
(result: sum(nbytes) == rows*30,720 exactly and mx.get_active_memory() at the end equals it; q2b_taps_retention_probe.out.full.txt).

This probe drives the REAL `engine_prefill` (imported read-only from the node-equivalent checkout) with a
stub model whose only job is to return (handle, taps) with the real tap shapes/dtype and to advance
`cache.offset`. It reports mx.get_peak_memory() growth with `taps_out=<list>` (what Conversation.prefill does)
versus `taps_out=None` + return_taps=True (no accumulation). LOCAL laptop only; never touches a cluster node.

Run (the CHILD-COMMON incantation; mlx-lm from the shared checkout, exo src from the node-equivalent checkout):
  EXO_DASHBOARD_DIR=/Users/adam.durham/repos/exo/dashboard/build PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/Users/adam.durham/repos/exo/src:/Users/adam.durham/repos/exo/mlx-lm \
  /Users/adam.durham/repos/exo/.venv/bin/python q2b_taps_retention_probe.py
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from exo.worker.engines.mlx.dsv41.session import engine_prefill

HIDDEN = 5120
TAP_LAYERS = (37, 38, 39)          # dspark_target_layer_ids of the shipped checkpoint
BYTES_PER_ROW = len(TAP_LAYERS) * HIDDEN * 2  # bf16 -> 30,720 B / row


class StubModel:
    """Stands in for deepseek_v41 Model.__call__: returns (handle, taps) with real tap shapes/dtype."""

    def __init__(self) -> None:
        self._fence_every = 0
        self._fence_hook = None

    def __call__(self, piece, cache, last_logit_only=True, return_taps=False, argmax=False):
        n = int(piece.shape[1])
        cache.offset = int(cache.offset) + n
        handle = mx.zeros((1, 1), dtype=mx.float32)
        if return_taps:
            taps = {i: mx.ones((1, n, HIDDEN), dtype=mx.bfloat16) for i in TAP_LAYERS}
            return handle, taps
        return handle


def run(rows: int, chunk: int, retain: bool) -> dict[str, float]:
    mx.clear_cache()
    mx.reset_peak_memory()
    base_active = mx.get_active_memory()
    cache = SimpleNamespace(offset=0)
    ids = np.arange(rows, dtype=np.int64)
    sink: list = []
    consumed = {"chunks": 0}

    def taps_cb(_t) -> None:            # the real callback consumes each chunk's taps into the draft window
        consumed["chunks"] += 1

    engine_prefill(
        StubModel(), ids, cache,
        chunk=chunk,
        return_taps=not retain,          # retain=True mirrors Conversation.prefill (taps_out=list)
        taps_out=sink if retain else None,
        taps_cb=taps_cb,
    )
    peak = mx.get_peak_memory()
    active_end = mx.get_active_memory()   # `sink` still referenced here => retained bytes visible
    sink_nbytes = sum(a.nbytes for d in sink for a in d.values())
    out = {
        "rows": rows, "chunks_consumed": consumed["chunks"], "retain": retain,
        "peak_minus_base_B": peak - base_active,
        "active_end_minus_base_B": active_end - base_active,
        "sink_len": len(sink),
        "sink_nbytes_B": sink_nbytes,
        "expected_retained_B": BYTES_PER_ROW * rows if retain else None,
    }
    del sink
    mx.clear_cache()
    return out


def main() -> int:
    chunk = 2048
    res = []
    for rows in (20480, 40960, 81920):
        for retain in (False, True):
            res.append(run(rows, chunk, retain))
    print(f"tap bytes/row = {BYTES_PER_ROW} (3 layers x {HIDDEN} x 2 B bf16)   chunk={chunk}")
    print(f"{'rows':>7s} {'retain':>6s} {'sink':>5s} | {'sum(nbytes) GB':>14s} {'B/row':>9s} | {'peak-base GB':>12s} {'active_end-base GB':>18s}")
    for r in res:
        nb = r["sink_nbytes_B"]
        print(f"{r['rows']:7d} {str(r['retain']):>6s} {r['sink_len']:5d} | {nb / 1e9:14.4f} {nb / r['rows']:9.1f} | "
              f"{r['peak_minus_base_B'] / 1e9:12.3f} {r['active_end_minus_base_B'] / 1e9:18.3f}")
    print("\nper-row retained slope (retain=True minus retain=False, peak domain):")
    by = {(r["rows"], r["retain"]): r for r in res}
    for rows in (20480, 40960, 81920):
        d = by[(rows, True)]["peak_minus_base_B"] - by[(rows, False)]["peak_minus_base_B"]
        print(f"  rows={rows:6d}  delta_peak={d / 1e9:7.3f} GB  = {d / rows:9.1f} B/row")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for the cache re-materialisation probe (EXO_DSV41_KV_REMAT).

The probe must be off by default, must not change a single value, must replace the
allocation (that is the point), and must honour the subset selectors so a positive
result can be bisected.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import kv_remat


class _Layer:
    def __init__(self, rows: int) -> None:
        self.win_kv = mx.full((rows, 4), 1.0, dtype=mx.bfloat16)
        self.comp_kv = mx.full((rows, 2), 2.0, dtype=mx.bfloat16)
        self.index_k = mx.full((rows, 2), 3.0, dtype=mx.bfloat16)
        self.comp_state = SimpleNamespace(
            kv_state=mx.full((8,), 4.0, dtype=mx.bfloat16), score_state=None
        )


def _cache(rows: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[_Layer(rows) for _ in range(2)],
        engram_ids=np.zeros((1, rows), dtype=np.int64),
    )


def test_the_probe_is_off_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("", "0", "off"):
        monkeypatch.setenv(kv_remat.ENV, value)
        assert kv_remat.mode() == ""
        assert kv_remat.remat(_cache()) == (0, 0.0)


def test_all_copies_every_buffer_and_replaces_the_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(kv_remat.ENV, "all")
    cache = _cache()
    before_ids = [id(cache.layers[0].win_kv), id(cache.layers[0].index_k)]
    before_vals = [
        float(mx.sum(cache.layers[0].win_kv.astype(mx.float32))),
        float(mx.sum(cache.layers[0].index_k.astype(mx.float32))),
    ]
    n, secs = kv_remat.remat(cache)
    # 2 layers x (win_kv, comp_kv, index_k, kv_state) + engram history
    assert n == 9
    assert secs >= 0.0
    after_ids = [id(cache.layers[0].win_kv), id(cache.layers[0].index_k)]
    assert before_ids != after_ids, "the buffers must be NEW allocations"
    after_vals = [
        float(mx.sum(cache.layers[0].win_kv.astype(mx.float32))),
        float(mx.sum(cache.layers[0].index_k.astype(mx.float32))),
    ]
    assert before_vals == after_vals, "values must be identical"


def test_the_index_selector_touches_only_index_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(kv_remat.ENV, "index")
    cache = _cache()
    win_before = id(cache.layers[1].win_kv)
    index_before = id(cache.layers[1].index_k)
    n, _secs = kv_remat.remat(cache)
    assert n == 2, "one index_k per layer, nothing else"
    assert id(cache.layers[1].win_kv) == win_before
    assert id(cache.layers[1].index_k) != index_before


def test_the_kv_selector_skips_index_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kv_remat.ENV, "kv")
    cache = _cache()
    index_before = id(cache.layers[0].index_k)
    n, _secs = kv_remat.remat(cache)
    # 2 layers x (win_kv, comp_kv, kv_state) + engram
    assert n == 7
    assert id(cache.layers[0].index_k) == index_before


def test_an_unknown_selector_copies_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(kv_remat.ENV, "banana")
    assert kv_remat.remat(_cache()) == (0, 0.0)

"""Tests for the pre-decode cache warm (EXO_DSV41_PREFILL_KV_TOUCH).

The warm must be OFF by default, must not modify a single byte, and must find the
arrays wherever the composite cache keeps them (per-layer buffers, a nested
compressor state, a model-level engram history).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.dsv41 import kv_touch


class _Layer:
    def __init__(self, rows: int) -> None:
        self.win_kv = mx.zeros((rows, 8), dtype=mx.bfloat16)
        self.comp_kv = mx.zeros((rows, 4), dtype=mx.bfloat16)
        self.index_k = None


def _fake_cache(rows: int = 4096) -> SimpleNamespace:
    layers = [_Layer(rows) for _ in range(3)]
    layers[0].comp_state = SimpleNamespace(kv_state=mx.zeros((64,), dtype=mx.bfloat16))
    return SimpleNamespace(layers=layers, engram_ids=np.zeros((1, rows), dtype=np.int64))


def test_the_warm_is_off_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(kv_touch.ENV, raising=False)
    assert kv_touch.enabled() is False
    monkeypatch.setenv(kv_touch.ENV, "1")
    assert kv_touch.enabled() is True


def test_it_finds_every_buffer_in_the_composite_cache() -> None:
    cache = _fake_cache()
    arrays = kv_touch._arrays(cache)
    # 3 layers x (win_kv + comp_kv) + 1 nested comp_state + 1 engram history
    assert len(arrays) == 8


def test_it_reads_without_modifying_and_reports_its_cost() -> None:
    cache = _fake_cache()
    arrays = [a for a in kv_touch._arrays(cache) if isinstance(a, mx.array)]
    before = [float(mx.sum(a.astype(mx.float32))) for a in arrays]
    n, secs = kv_touch.touch(cache)
    assert n == 8
    assert secs >= 0.0
    after = [float(mx.sum(a.astype(mx.float32))) for a in arrays]
    assert before == after, "the warm must be read-only"


def test_a_stride_below_one_page_still_samples_the_whole_buffer() -> None:
    """The stride is in ELEMENTS: 4096 bf16 elements = 8 KB < a 16 KB page."""
    assert kv_touch.DEFAULT_STRIDE * 2 <= 16384  # bytes of a page
    big = mx.zeros((100_000,), dtype=mx.bfloat16)
    n, _secs = kv_touch.touch(SimpleNamespace(buf=big))
    assert n == 1


def test_an_empty_cache_is_a_no_op() -> None:
    assert kv_touch.touch(SimpleNamespace()) == (0, 0.0)

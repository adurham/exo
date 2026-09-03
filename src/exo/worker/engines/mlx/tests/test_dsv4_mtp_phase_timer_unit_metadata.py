# pyright: reportPrivateUsage=false
"""Regression guard for _PhaseTimer per-series unit metadata.

ROOT CAUSE (commit bbb0e93418, 2026-09-01): _PhaseTimer.dump() used to
stamp a hardcoded ``ms`` suffix on every series it formatted. The
integer counter ``rb_pool_restores`` therefore rendered as
``mean=18.91ms``, which was misread as a genuine 25%-of-cycle latency
hotspot -- a phantom that cost real analysis time before it was traced
back to source. The fix added per-series unit metadata
(``_ProfUnit = Literal["ms", "count"]``) so ``record()`` and ``dump()``
track and render the correct suffix per series.

These tests call the REAL ``_PhaseTimer.dump()`` (via caplog, since it
logs rather than returns a string) and assert on its actual formatted
output. They do not reimplement the formatting logic -- if the ``ms``
suffix were hardcoded again, both tests would fail against the real
output.
"""

from __future__ import annotations

import logging

import pytest

from exo.worker.engines.mlx.speculative.dsv4_mtp import _PhaseTimer


def _dump_and_capture(
    caplog: pytest.LogCaptureFixture, timer: _PhaseTimer, batch_size: int
) -> str:
    timer.end_cycle(batch_size)
    with caplog.at_level(logging.WARNING):
        timer.dump()
    return caplog.text


def test_count_series_renders_without_ms_suffix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """rb_pool_restores (unit='count') must not be stamped with 'ms'.

    This is the exact historical trap: before the fix, this counter
    series rendered as ``mean=18.91ms`` and was misread as a real
    latency hotspot.
    """
    timer = _PhaseTimer()
    timer.record("rb_pool_restores", 3.0, unit="count")
    text = _dump_and_capture(caplog, timer, batch_size=1)

    lines = [line for line in text.splitlines() if "rb_pool_restores" in line]
    assert len(lines) == 1, text
    assert "ms" not in lines[0], lines[0]
    assert "mean=  3.00" in lines[0], lines[0]


def test_ms_series_renders_with_ms_suffix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A genuine duration series (unit='ms') must keep its 'ms' suffix."""
    timer = _PhaseTimer()
    timer.record("draft", 12.34, unit="ms")
    text = _dump_and_capture(caplog, timer, batch_size=1)

    lines = [line for line in text.splitlines() if "draft" in line]
    assert len(lines) == 1, text
    assert "ms" in lines[0], lines[0]
    assert "mean= 12.34ms" in lines[0], lines[0]

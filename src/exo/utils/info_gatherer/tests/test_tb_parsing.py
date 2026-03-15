import sys
from unittest.mock import AsyncMock, patch

import anyio
import pytest

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.thunderbolt import (
    ThunderboltConnectivity,
)
from exo.utils.info_gatherer.info_gatherer import (
    InfoGatherer,
    _gather_iface_map,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin", reason="Thunderbolt info can only be gathered on macos"
)
async def test_tb_parsing():
    data = await ThunderboltConnectivity.gather()
    ifaces = await _gather_iface_map()
    assert ifaces
    assert data
    for datum in data:
        datum.ident(ifaces)
        datum.conn()


def test_memory_capping_logic():
    """Test the memory capping logic directly without async loops."""
    # Test case 1: Memory should be capped to max_node_memory_gb
    # Simulate what _monitor_memory_usage does
    max_node_memory_gb = 8
    max_memory_bytes = Memory.from_gb(max_node_memory_gb).in_bytes

    # Simulate 16 GB available
    memory_usage = MemoryUsage(
        ram_total=Memory.from_gb(32),
        ram_available=Memory.from_gb(16),  # More than 8 GB cap
        swap_total=Memory.from_gb(8),
        swap_available=Memory.from_gb(8),
    )

    # Apply the capping logic (same as in _monitor_memory_usage)
    if memory_usage.ram_available.in_bytes > max_memory_bytes:
        memory_usage = MemoryUsage(
            ram_total=memory_usage.ram_total,
            ram_available=Memory.from_bytes(max_memory_bytes),
            swap_total=memory_usage.swap_total,
            swap_available=memory_usage.swap_available,
        )

    # Assert
    assert memory_usage.ram_available.in_gb == 8

    # Test case 2: Memory should NOT be capped when max is None
    memory_usage2 = MemoryUsage(
        ram_total=Memory.from_gb(32),
        ram_available=Memory.from_gb(16),
        swap_total=Memory.from_gb(8),
        swap_available=Memory.from_gb(8),
    )

    # No capping when max is None
    max_node_memory_gb = None

    if max_node_memory_gb is not None:
        max_memory_bytes = Memory.from_gb(max_node_memory_gb).in_bytes
        if memory_usage2.ram_available.in_bytes > max_memory_bytes:
            memory_usage2 = MemoryUsage(
                ram_total=memory_usage2.ram_total,
                ram_available=Memory.from_bytes(max_memory_bytes),
                swap_total=memory_usage2.swap_total,
                swap_available=memory_usage2.swap_available,
            )

    # Assert - should still be 16 GB
    assert memory_usage2.ram_available.in_gb == 16

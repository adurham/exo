"""
Tests for memory_management.py utilities.
"""
import pytest

from exo.utils.memory_management import (
    check_swap_usage,
    flush_ram_on_shutdown,
    flush_ram_on_startup,
    get_memory_stats,
    log_memory_stats,
    verify_zero_swap,
)


def test_flush_ram_on_startup():
    """Test that flush_ram_on_startup executes without errors."""
    # Should not raise any exceptions
    flush_ram_on_startup()


def test_flush_ram_on_shutdown():
    """Test that flush_ram_on_shutdown executes without errors."""
    # Should not raise any exceptions
    flush_ram_on_shutdown()


def test_check_swap_usage():
    """Test that check_swap_usage returns valid values."""
    swap_used, swap_total = check_swap_usage()
    
    # Should return non-negative integers
    assert isinstance(swap_used, int)
    assert isinstance(swap_total, int)
    assert swap_used >= 0
    assert swap_total >= 0
    assert swap_used <= swap_total or swap_total == 0


def test_verify_zero_swap_when_zero():
    """Test that verify_zero_swap passes when swap is zero."""
    # This test may fail if swap is actually being used on the test machine
    # In that case, we'll skip the assertion
    swap_used, _ = check_swap_usage()
    
    if swap_used == 0:
        # Should not raise exception
        result = verify_zero_swap()
        assert result is True
    else:
        # If swap is used, verify_zero_swap should raise ValueError
        with pytest.raises(ValueError, match="CRITICAL: Swap usage detected"):
            verify_zero_swap()


def test_verify_zero_swap_when_used():
    """Test that verify_zero_swap raises ValueError when swap is used."""
    swap_used, _ = check_swap_usage()
    
    if swap_used > 0:
        # Should raise ValueError
        with pytest.raises(ValueError, match="CRITICAL: Swap usage detected"):
            verify_zero_swap()
    else:
        # If swap is zero, it should pass
        result = verify_zero_swap()
        assert result is True


def test_get_memory_stats():
    """Test that get_memory_stats returns valid dictionary."""
    stats = get_memory_stats()
    
    assert isinstance(stats, dict)
    
    # Check required keys exist
    required_keys = [
        "ram_total_bytes",
        "ram_available_bytes",
        "ram_used_bytes",
        "ram_percent_used",
        "swap_total_bytes",
        "swap_used_bytes",
        "swap_free_bytes",
        "swap_percent_used",
    ]
    
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    
    # Check values are reasonable
    assert stats["ram_total_bytes"] > 0
    assert stats["ram_available_bytes"] >= 0
    assert stats["ram_used_bytes"] >= 0
    assert 0 <= stats["ram_percent_used"] <= 100
    assert stats["swap_total_bytes"] >= 0
    assert stats["swap_used_bytes"] >= 0
    assert stats["swap_free_bytes"] >= 0
    assert 0 <= stats["swap_percent_used"] <= 100 or stats["swap_total_bytes"] == 0
    
    # Check that used + free = total for swap (if total > 0)
    if stats["swap_total_bytes"] > 0:
        assert (
            stats["swap_used_bytes"] + stats["swap_free_bytes"]
            == stats["swap_total_bytes"]
        )


def test_log_memory_stats():
    """Test that log_memory_stats executes without errors."""
    # Should not raise any exceptions
    log_memory_stats()


def test_memory_stats_integration():
    """Test that memory stats are consistent across calls."""
    stats1 = get_memory_stats()
    stats2 = get_memory_stats()
    
    # Values should be the same or similar (within small tolerance for timing)
    assert stats1["ram_total_bytes"] == stats2["ram_total_bytes"]
    assert stats1["swap_total_bytes"] == stats2["swap_total_bytes"]


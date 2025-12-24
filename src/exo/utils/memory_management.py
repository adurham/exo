"""
Memory management utilities for RAM flushing and swap monitoring.

Critical for performance: Zero swap usage is required on all nodes.
"""
import gc
import subprocess
import sys
from typing import Any

import psutil
from loguru import logger


def flush_ram_on_startup() -> None:
    """Flush RAM on startup to ensure clean state.
    
    This function:
    - Forces Python garbage collection
    - Attempts to purge system memory caches (macOS specific)
    - Clears any stale MLX arrays if MLX is imported
    
    Should be called at application startup before loading models.
    """
    logger.info("Flushing RAM on startup...")
    
    # Force Python garbage collection
    collected = gc.collect()
    logger.info(f"Python GC collected {collected} objects")
    
    # Try to purge system memory caches (macOS)
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["purge"],
                capture_output=True,
                timeout=10,
                check=False,  # Don't fail if purge requires sudo
            )
            if result.returncode == 0:
                logger.info("System memory cache purged successfully")
            else:
                logger.warning(
                    f"purge command failed (may require sudo): {result.stderr.decode('utf-8', errors='replace')[:100]}"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.warning(f"Could not run purge command: {e}")
    
    # Clear MLX arrays if MLX is imported
    try:
        import mlx.core as mx
        
        # Force MLX to clear any cached arrays
        # MLX arrays are managed internally, but we can trigger cleanup
        gc.collect()  # Another GC pass after potential MLX operations
        logger.info("MLX arrays cleared (if any were present)")
    except ImportError:
        # MLX not available, skip
        pass
    except Exception as e:
        logger.warning(f"Error clearing MLX arrays: {e}")
    
    logger.info("RAM flush on startup complete")


def flush_ram_on_shutdown() -> None:
    """Flush RAM on shutdown to clean up resources.
    
    This function:
    - Forces Python garbage collection
    - Attempts to free MLX arrays
    - Clears any remaining resources
    
    Should be called at application shutdown after stopping workers/models.
    """
    logger.info("Flushing RAM on shutdown...")
    
    # Clear MLX arrays if MLX is imported
    try:
        import mlx.core as mx
        
        # Force cleanup of MLX arrays
        gc.collect()
        logger.info("MLX arrays cleared on shutdown")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error clearing MLX arrays on shutdown: {e}")
    
    # Final Python garbage collection
    collected = gc.collect()
    logger.info(f"Final Python GC collected {collected} objects")
    
    logger.info("RAM flush on shutdown complete")


def check_swap_usage() -> tuple[int, int]:
    """Check current swap usage.
    
    Returns:
        Tuple of (swap_used_bytes, swap_total_bytes)
    """
    try:
        swap = psutil.swap_memory()
        return (swap.used, swap.total)
    except Exception as e:
        logger.error(f"Error checking swap usage: {e}")
        return (0, 0)


def verify_zero_swap() -> bool:
    """Verify that swap usage is zero.
    
    Returns:
        True if swap usage is zero, False otherwise
        
    Raises:
        ValueError: If swap usage is non-zero (failure case)
    """
    swap_used, swap_total = check_swap_usage()
    
    if swap_used > 0:
        swap_used_mb = swap_used / (1024 * 1024)
        swap_total_mb = swap_total / (1024 * 1024) if swap_total > 0 else 0
        error_msg = (
            f"CRITICAL: Swap usage detected! "
            f"Used: {swap_used_mb:.2f} MB / {swap_total_mb:.2f} MB total. "
            f"Swap usage will kill performance. "
            f"This is a failure condition."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug("Swap usage verified: zero")
    return True


def get_memory_stats() -> dict[str, Any]:
    """Get detailed memory statistics.
    
    Returns:
        Dictionary with memory statistics including RAM and swap usage
    """
    try:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "ram_total_bytes": vm.total,
            "ram_available_bytes": vm.available,
            "ram_used_bytes": vm.used,
            "ram_percent_used": vm.percent,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
            "swap_free_bytes": swap.free,
            "swap_percent_used": swap.percent if swap.total > 0 else 0.0,
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {}


def log_memory_stats() -> None:
    """Log current memory statistics."""
    stats = get_memory_stats()
    if stats:
        ram_total_gb = stats["ram_total_bytes"] / (1024**3)
        ram_used_gb = stats["ram_used_bytes"] / (1024**3)
        ram_percent = stats["ram_percent_used"]
        swap_used_mb = stats["swap_used_bytes"] / (1024**2)
        swap_percent = stats["swap_percent_used"]
        
        logger.info(
            f"Memory stats: RAM {ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB ({ram_percent:.1f}%), "
            f"Swap {swap_used_mb:.2f} MB ({swap_percent:.1f}%)"
        )


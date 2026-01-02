import asyncio
import os
import platform
from typing import Any, Callable, Coroutine

import anyio
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)

from .macmon import (
    MacMonError,
    Metrics,
)
from .macmon import (
    get_metrics_async as macmon_get_metrics_async,
)
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
)


async def get_metrics_async() -> Metrics | None:
    """Return detailed Metrics on macOS or a minimal fallback elsewhere."""

    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()


def get_memory_profile() -> MemoryPerformanceProfile:
    """Construct a MemoryPerformanceProfile using psutil"""
    override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
    override_memory: int | None = (
        Memory.from_mb(int(override_memory_env)).in_bytes
        if override_memory_env
        else None
    )

    profile = MemoryPerformanceProfile.from_psutil(override_memory=override_memory)

    return profile


async def start_polling_memory_metrics(
    callback: Callable[[MemoryPerformanceProfile], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 0.5,
) -> None:
    """Continuously poll and emit memory-only metrics at a faster cadence.

    Parameters
    - callback: coroutine called with a fresh MemoryPerformanceProfile each tick
    - poll_interval_s: interval between polls
    """
    while True:
        try:
            mem = get_memory_profile()
            await callback(mem)
        except MacMonError as e:
            logger.opt(exception=e).error("Memory Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
):
    poll_interval_s = 1.0
    while True:
        try:
            metrics = await get_metrics_async()
            if metrics is None:
                logger.error("get_metrics_async returned None, stopping profile loop")
                return

            network_interfaces = await get_network_interfaces()
            # these awaits could be joined but realistically they should be cached
            model_id, chip_id = await get_model_and_chip()
            friendly_name = await get_friendly_name()

            # do the memory profile last to get a fresh reading to not conflict with the other memory profiling loop
            memory_profile = get_memory_profile()

            await callback(
                NodePerformanceProfile(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=friendly_name,
                    network_interfaces=network_interfaces,
                    memory=memory_profile,
                    system=SystemPerformanceProfile(
                        gpu_usage=metrics.gpu_usage[1],
                        temp=metrics.temp.gpu_temp_avg,
                        sys_power=metrics.sys_power,
                        pcpu_usage=metrics.pcpu_usage[1],
                        ecpu_usage=metrics.ecpu_usage[1],
                        ane_power=metrics.ane_power,
                    ),
                )
            )

            await callback(
                NodePerformanceProfile(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=friendly_name,
                    network_interfaces=network_interfaces,
                    memory=memory_profile,
                    system=SystemPerformanceProfile(
                        gpu_usage=metrics.gpu_usage[1],
                        temp=metrics.temp.gpu_temp_avg,
                        sys_power=metrics.sys_power,
                        pcpu_usage=metrics.pcpu_usage[1],
                        ecpu_usage=metrics.ecpu_usage[1],
                        ane_power=metrics.ane_power,
                    ),
                )
            )

        except asyncio.TimeoutError:
            logger.warning(
                "[resource_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except MacMonError as e:
            logger.opt(exception=e).error("Resource Monitor encountered error")
            return
        except Exception as e:
            logger.opt(exception=e).error(f"Resource Monitor encountered unexpected error: {e}")
            await anyio.sleep(poll_interval_s)
        finally:
            await anyio.sleep(poll_interval_s)


class ThrashingMonitor:
    def __init__(self, pageout_threshold_mb_per_sec: int = 50):
        self.pageout_threshold_mb = pageout_threshold_mb_per_sec
        self.page_size = 4096  # Standard on macOS (CHECK: usually 4KB or 16KB on M-series)
        self.last_pageouts = None
        self.last_check_time = 0

    async def start(self):
        if platform.system() != "Darwin":
            return
            
        logger.info("Starting ThrashingMonitor...")
        self.page_size = self._get_page_size()
        
        while True:
            try:
                await self._check_vm_stat()
            except Exception as e:
                logger.error(f"ThrashingMonitor error: {e}")
            finally:
                await anyio.sleep(2.0)

    def _get_page_size(self) -> int:
        try:
            # sysctl -n hw.pagesize
            import subprocess
            output = subprocess.check_output(["sysctl", "-n", "hw.pagesize"]).decode().strip()
            return int(output)
        except Exception:
            return 16384 # Safe default for Apple Silicon (16KB)

    async def _check_vm_stat(self):
        import time
        from anyio import run_process
        
        # Parse vm_stat
        # "Pageouts:       12345."
        result = await run_process(["vm_stat"])
        output = result.stdout.decode()
        
        current_pageouts = 0
        for line in output.split("\n"):
            if "Pageouts" in line:
                # Format: "Pageouts:    12345."
                parts = line.split(":")
                if len(parts) > 1:
                    current_pageouts = int(parts[1].replace(".", "").strip())
                    break
        
        now = time.time()
        
        if self.last_pageouts is not None:
            delta_pages = current_pageouts - self.last_pageouts
            time_delta = now - self.last_check_time
            
            if time_delta > 0:
                bytes_per_sec = (delta_pages * self.page_size) / time_delta
                mb_per_sec = bytes_per_sec / (1024 * 1024)
                
                if mb_per_sec > 10: # Log warnings for moderate pressure
                     logger.warning(f"High memory pressure detected: {mb_per_sec:.2f} MB/s pageouts")

                if mb_per_sec > self.pageout_threshold_mb:
                    logger.critical(f"THRASHING DETECTED: {mb_per_sec:.2f} MB/s pageouts. Initiating emergency shutdown.")
                    # Force kill self to release resources immediately
                    os.kill(os.getpid(), 9)

        self.last_pageouts = current_pageouts
        self.last_check_time = now


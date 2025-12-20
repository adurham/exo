"""Resource monitoring abstractions for collecting system metrics.

This module provides abstractions for collecting and monitoring system
performance and memory metrics from nodes.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Callable

from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    SystemPerformanceProfile,
)


class ResourceCollector(ABC):
    """Abstract base class for resource collectors.

    Implementations collect either system performance or memory metrics.
    """

    @abstractmethod
    async def collect(self) -> SystemPerformanceProfile | MemoryPerformanceProfile:
        """Collect performance or memory metrics.

        Returns:
            Either SystemPerformanceProfile or MemoryPerformanceProfile.
        """
        ...


class SystemResourceCollector(ResourceCollector):
    """Collector for system performance metrics.

    Collects CPU, GPU, temperature, and power metrics.
    """

    async def collect(self) -> SystemPerformanceProfile:
        """Collect system performance profile.

        Returns:
            SystemPerformanceProfile with current metrics.
        """
        ...


class MemoryResourceCollector(ResourceCollector):
    """Collector for memory metrics.

    Collects RAM and swap usage information.
    """

    async def collect(self) -> MemoryPerformanceProfile:
        """Collect memory performance profile.

        Returns:
            MemoryPerformanceProfile with current memory usage.
        """
        ...


class ResourceMonitor:
    """Monitors system resources using multiple collectors.

    Collects metrics from multiple collectors concurrently and passes
    them to registered effect handlers (callbacks).

    Attributes:
        data_collectors: List of collectors to gather metrics from.
        effect_handlers: Set of callbacks to invoke with collected profiles.
    """

    data_collectors: list[ResourceCollector]
    effect_handlers: set[
        Callable[[SystemPerformanceProfile | MemoryPerformanceProfile], None]
    ]

    async def _collect(
        self,
    ) -> list[SystemPerformanceProfile | MemoryPerformanceProfile]:
        """Collect metrics from all collectors concurrently.

        Returns:
            List of profiles collected from all collectors.
        """
        tasks: list[
            Coroutine[None, None, SystemPerformanceProfile | MemoryPerformanceProfile]
        ] = [collector.collect() for collector in self.data_collectors]
        return await asyncio.gather(*tasks)

    async def collect(self) -> None:
        """Collect metrics and invoke all effect handlers.

        Gathers metrics from all collectors, then calls each effect handler
        with each collected profile.
        """
        profiles = await self._collect()
        for profile in profiles:
            for effect_handler in self.effect_handlers:
                effect_handler(profile)

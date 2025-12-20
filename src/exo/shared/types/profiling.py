"""Performance profiling types for nodes and connections.

This module defines types for tracking performance metrics of nodes
(CPU, GPU, memory, network) and network connections (throughput, latency).
"""

from typing import Self

import psutil

from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class MemoryPerformanceProfile(CamelCaseModel):
    """Memory usage profile for a node.

    Attributes:
        ram_total: Total RAM capacity.
        ram_available: Available RAM (unused).
        swap_total: Total swap space capacity.
        swap_available: Available swap space.
    """

    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        """Create profile from byte values.

        Args:
            ram_total: Total RAM in bytes.
            ram_available: Available RAM in bytes.
            swap_total: Total swap in bytes.
            swap_available: Available swap in bytes.

        Returns:
            New MemoryPerformanceProfile instance.
        """
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        """Create profile from psutil system information.

        Args:
            override_memory: Optional override for available RAM (for testing).

        Returns:
            New MemoryPerformanceProfile from system metrics.
        """
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )


class SystemPerformanceProfile(CamelCaseModel):
    """System performance metrics for a node.

    Attributes:
        gpu_usage: GPU utilization percentage (0.0-1.0).
        temp: Temperature in Celsius.
        sys_power: System power consumption in watts.
        pcpu_usage: Performance CPU utilization percentage (0.0-1.0).
        ecpu_usage: Efficiency CPU utilization percentage (0.0-1.0).
        ane_power: Apple Neural Engine power consumption in watts.
    """

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0


class NetworkInterfaceInfo(CamelCaseModel):
    """Information about a network interface.

    Attributes:
        name: Interface name (e.g., "en0", "eth0").
        ip_address: IP address assigned to this interface.
    """

    name: str
    ip_address: str


class NodePerformanceProfile(CamelCaseModel):
    """Complete performance profile for a node.

    Contains information about hardware, memory, system metrics, and
    network interfaces. Used for placement decisions and monitoring.

    Attributes:
        model_id: Hardware model identifier.
        chip_id: Chip identifier (e.g., "M1", "M2").
        friendly_name: Human-readable device name.
        memory: Memory usage profile.
        network_interfaces: List of network interfaces and their IPs.
        system: System performance metrics.
    """

    model_id: str
    chip_id: str
    friendly_name: str
    memory: MemoryPerformanceProfile
    network_interfaces: list[NetworkInterfaceInfo] = []
    system: SystemPerformanceProfile


class ConnectionProfile(CamelCaseModel):
    """Network connection performance metrics.

    Attributes:
        throughput: Connection throughput in bits per second.
        latency: Round-trip latency in milliseconds.
        jitter: Latency jitter (variance) in milliseconds.
    """

    throughput: float
    latency: float
    jitter: float

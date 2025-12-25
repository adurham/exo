"""
Static configuration for 4-node MLX RDMA cluster (1 master + 3 workers).

This module provides hardcoded topology and network configuration for the static setup.
"""
import socket
from dataclasses import dataclass
from typing import Any

from loguru import logger

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NetworkInterfaceInfo,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import Connection, NodeInfo


@dataclass(frozen=True)
class StaticWorkerNodeConfig:
    """Configuration for a static worker node."""

    node_id: NodeId
    hostname: str
    tailscale_ip: str
    rank: int
    chip_id: str
    ram_total_bytes: int
    thunderbolt_ips: dict[NodeId, str]  # peer node_id -> this node's IP for that connection


@dataclass(frozen=True)
class StaticMasterConfig:
    """Configuration for the static master node."""

    node_id: NodeId
    hostname: str
    tailscale_ip: str
    port: int = 52415


@dataclass(frozen=True)
class StaticConfig:
    """Static configuration for the 4-node cluster."""

    master: StaticMasterConfig
    workers: tuple[StaticWorkerNodeConfig, StaticWorkerNodeConfig, StaticWorkerNodeConfig]

    @property
    def worker_rank_map(self) -> dict[int, StaticWorkerNodeConfig]:
        """Map rank to worker config."""
        return {worker.rank: worker for worker in self.workers}

    @property
    def worker_node_id_map(self) -> dict[NodeId, StaticWorkerNodeConfig]:
        """Map node_id to worker config."""
        return {worker.node_id: worker for worker in self.workers}


# Hardcoded static configuration
_STATIC_MASTER_NODE_ID = NodeId("static-master-adams-macbook-pro-m1")
_STATIC_WORKER_0_NODE_ID = NodeId("static-worker-0-adams-mac-studio-m4")
_STATIC_WORKER_1_NODE_ID = NodeId("static-worker-1-adams-macbook-pro-m4")
_STATIC_WORKER_2_NODE_ID = NodeId("static-worker-2-adams-work-macbook-pro-m4")

# Thunderbolt IP mappings
# Rank 0 → Rank 1: 192.168.202.1 → 192.168.202.2
# Rank 0 → Rank 2: 192.168.203.1 → 192.168.203.2
# Rank 1 → Rank 2: 192.168.205.1 → 192.168.205.2

_STATIC_CONFIG = StaticConfig(
    master=StaticMasterConfig(
        node_id=_STATIC_MASTER_NODE_ID,
        hostname="adams-macbook-pro-m1",
        tailscale_ip="100.67.156.10",
        port=52415,
    ),
    workers=(
        StaticWorkerNodeConfig(
            node_id=_STATIC_WORKER_0_NODE_ID,
            hostname="adams-mac-studio-m4",
            tailscale_ip="100.93.253.67",
            rank=0,
            chip_id="Apple M4",
            ram_total_bytes=192 * 1024 * 1024 * 1024,  # 192GB - placeholder, should be fetched at runtime
            thunderbolt_ips={
                _STATIC_WORKER_1_NODE_ID: "192.168.202.1",  # Rank 0 → Rank 1
                _STATIC_WORKER_2_NODE_ID: "192.168.203.1",  # Rank 0 → Rank 2
            },
        ),
        StaticWorkerNodeConfig(
            node_id=_STATIC_WORKER_1_NODE_ID,
            hostname="adams-macbook-pro-m4",
            tailscale_ip="100.80.147.125",
            rank=1,
            chip_id="Apple M4",
            ram_total_bytes=128 * 1024 * 1024 * 1024,  # 128GB - placeholder, should be fetched at runtime
            thunderbolt_ips={
                _STATIC_WORKER_0_NODE_ID: "192.168.202.2",  # Rank 1 ← Rank 0 (reverse direction)
                _STATIC_WORKER_2_NODE_ID: "192.168.205.1",  # Rank 1 → Rank 2
            },
        ),
        StaticWorkerNodeConfig(
            node_id=_STATIC_WORKER_2_NODE_ID,
            hostname="adams-work-macbook-pro-m4",
            tailscale_ip="100.82.48.77",
            rank=2,
            chip_id="Apple M4",
            ram_total_bytes=128 * 1024 * 1024 * 1024,  # 128GB - placeholder, should be fetched at runtime
            thunderbolt_ips={
                _STATIC_WORKER_0_NODE_ID: "192.168.203.2",  # Rank 2 ← Rank 0 (reverse direction)
                _STATIC_WORKER_1_NODE_ID: "192.168.205.2",  # Rank 2 ← Rank 1 (reverse direction)
            },
        ),
    ),
)


def get_static_config() -> StaticConfig:
    """Get the static configuration."""
    return _STATIC_CONFIG


def get_master_url() -> str:
    """Get the master URL for worker communication."""
    config = get_static_config()
    return f"http://{config.master.tailscale_ip}:{config.master.port}"


def get_worker_config_by_hostname(hostname: str) -> StaticWorkerNodeConfig | None:
    """Get worker config by hostname."""
    config = get_static_config()
    for worker in config.workers:
        if worker.hostname == hostname:
            return worker
    return None


def get_worker_config_by_rank(rank: int) -> StaticWorkerNodeConfig | None:
    """Get worker config by MLX rank."""
    config = get_static_config()
    return config.worker_rank_map.get(rank)


def get_worker_config_by_node_id(node_id: NodeId) -> StaticWorkerNodeConfig | None:
    """Get worker config by node_id."""
    config = get_static_config()
    return config.worker_node_id_map.get(node_id)


def get_thunderbolt_ip_for_peer(
    from_node_id: NodeId, to_node_id: NodeId
) -> str | None:
    """Get the Thunderbolt IP address on from_node_id for connection to to_node_id."""
    worker_config = get_worker_config_by_node_id(from_node_id)
    if worker_config is None:
        return None
    return worker_config.thunderbolt_ips.get(to_node_id)


def create_static_topology(
    *,
    node_profiles: dict[NodeId, NodePerformanceProfile] | None = None,
) -> Topology:
    """Create a Topology from static configuration.
    
    Args:
        node_profiles: Optional dict of node_id -> NodePerformanceProfile.
            If provided, uses these profiles. Otherwise creates placeholder profiles.
    
    Returns:
        Topology with 3 worker nodes and connections between them for MLX RDMA.
    """
    config = get_static_config()
    topology = Topology()

    # Add worker nodes to topology
    for worker in config.workers:
        if node_profiles and worker.node_id in node_profiles:
            profile = node_profiles[worker.node_id]
        else:
            # Create placeholder profile - should be replaced with actual profile at runtime
            profile = NodePerformanceProfile(
                model_id="",  # Will be set when model is loaded
                chip_id=worker.chip_id,
                friendly_name=worker.hostname,
                memory=MemoryPerformanceProfile.from_bytes(
                    ram_total=worker.ram_total_bytes,
                    ram_available=worker.ram_total_bytes,  # Placeholder
                    swap_total=0,
                    swap_available=0,
                ),
                network_interfaces=[],
                system=SystemPerformanceProfile(),
            )
        node_info = NodeInfo(node_id=worker.node_id, node_profile=profile)
        topology.add_node(node_info)

    # Add connections for MLX RDMA (pipeline: Rank 0 → Rank 1 → Rank 2 → Rank 0)
    # We need bidirectional connections for MLX RDMA
    worker_0 = config.workers[0]
    worker_1 = config.workers[1]
    worker_2 = config.workers[2]

    # Rank 0 → Rank 1 (192.168.202.1 → 192.168.202.2)
    ip_0_to_1 = get_thunderbolt_ip_for_peer(worker_0.node_id, worker_1.node_id)
    if ip_0_to_1:
        connection_0_to_1 = Connection(
            local_node_id=worker_0.node_id,
            send_back_node_id=worker_1.node_id,
            send_back_multiaddr=Multiaddr(address=f"/ip4/{ip_0_to_1}/tcp/5000"),
            connection_profile=None,
        )
        topology.add_connection(connection_0_to_1)

    # Rank 1 → Rank 0 (reverse direction for RDMA)
    ip_1_to_0 = get_thunderbolt_ip_for_peer(worker_1.node_id, worker_0.node_id)
    if ip_1_to_0:
        connection_1_to_0 = Connection(
            local_node_id=worker_1.node_id,
            send_back_node_id=worker_0.node_id,
            send_back_multiaddr=Multiaddr(address=f"/ip4/{ip_1_to_0}/tcp/5000"),
            connection_profile=None,
        )
        topology.add_connection(connection_1_to_0)

    # Rank 1 → Rank 2 (192.168.205.1 → 192.168.205.2)
    ip_1_to_2 = get_thunderbolt_ip_for_peer(worker_1.node_id, worker_2.node_id)
    if ip_1_to_2:
        connection_1_to_2 = Connection(
            local_node_id=worker_1.node_id,
            send_back_node_id=worker_2.node_id,
            send_back_multiaddr=Multiaddr(address=f"/ip4/{ip_1_to_2}/tcp/5000"),
            connection_profile=None,
        )
        topology.add_connection(connection_1_to_2)

    # Rank 2 → Rank 1 (reverse direction for RDMA)
    ip_2_to_1 = get_thunderbolt_ip_for_peer(worker_2.node_id, worker_1.node_id)
    if ip_2_to_1:
        connection_2_to_1 = Connection(
            local_node_id=worker_2.node_id,
            send_back_node_id=worker_1.node_id,
            send_back_multiaddr=Multiaddr(address=f"/ip4/{ip_2_to_1}/tcp/5000"),
            connection_profile=None,
        )
        topology.add_connection(connection_2_to_1)

    # Rank 2 → Rank 0 (for pipeline completion, 192.168.203.2 → 192.168.203.1)
    ip_2_to_0 = get_thunderbolt_ip_for_peer(worker_2.node_id, worker_0.node_id)
    if ip_2_to_0:
        connection_2_to_0 = Connection(
            local_node_id=worker_2.node_id,
            send_back_node_id=worker_0.node_id,
            send_back_multiaddr=Multiaddr(address=f"/ip4/{ip_2_to_0}/tcp/5000"),
            connection_profile=None,
        )
        topology.add_connection(connection_2_to_0)

    # Rank 0 → Rank 2 (reverse direction for RDMA)
    ip_0_to_2 = get_thunderbolt_ip_for_peer(worker_0.node_id, worker_2.node_id)
    if ip_0_to_2:
        connection_0_to_2 = Connection(
            local_node_id=worker_0.node_id,
            send_back_node_id=worker_2.node_id,
            send_back_multiaddr=Multiaddr(address=f"/ip4/{ip_0_to_2}/tcp/5000"),
            connection_profile=None,
        )
        topology.add_connection(connection_0_to_2)

    logger.info(
        f"Created static topology with {len(list(topology.list_nodes()))} worker nodes"
    )
    return topology


def get_current_hostname() -> str:
    """Get the current hostname."""
    return socket.gethostname()


def is_master_node() -> bool:
    """Check if current node is the master node."""
    hostname = get_current_hostname()
    config = get_static_config()
    # Case-insensitive comparison since macOS hostnames can differ in case
    return hostname.lower() == config.master.hostname.lower()


def is_worker_node() -> bool:
    """Check if current node is a worker node."""
    hostname = get_current_hostname()
    config = get_static_config()
    return any(worker.hostname == hostname for worker in config.workers)


def get_current_worker_config() -> StaticWorkerNodeConfig | None:
    """Get the worker config for the current node, or None if not a worker."""
    hostname = get_current_hostname()
    return get_worker_config_by_hostname(hostname)


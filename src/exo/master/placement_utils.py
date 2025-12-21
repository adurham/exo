import subprocess
import sys
from collections.abc import Generator
from typing import TypeGuard, cast

from loguru import logger
from pydantic import BaseModel

from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelMetadata
from exo.shared.types.profiling import NodePerformanceProfile
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


class NodeWithProfile(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile


def narrow_all_nodes(nodes: list[NodeInfo]) -> TypeGuard[list[NodeWithProfile]]:
    return all(node.node_profile is not None for node in nodes)


def filter_cycles_by_memory(
    cycles: list[list[NodeInfo]], required_memory: Memory
) -> list[list[NodeInfo]]:
    filtered_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        total_mem = sum(
            (node.node_profile.memory.ram_available for node in cycle), start=Memory()
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cast(list[NodeInfo], cycle))
    return filtered_cycles


def get_smallest_cycles(cycles: list[list[NodeInfo]]) -> list[list[NodeInfo]]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def get_shard_assignments_for_pipeline_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeWithProfile],
):
    cycle_memory = sum(
        (node.node_profile.memory.ram_available for node in selected_cycle),
        start=Memory(),
    )
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    layers_assigned = 0
    for i, node in enumerate(selected_cycle):
        if i == len(selected_cycle) - 1:
            node_layers = total_layers - layers_assigned
        else:
            node_layers = round(
                total_layers
                * (
                    node.node_profile.memory.ram_available.in_bytes
                    / cycle_memory.in_bytes
                )
            )
            node_layers = max(1, node_layers)

        runner_id = RunnerId()

        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=world_size,
            start_layer=layers_assigned,
            end_layer=layers_assigned + node_layers,
            n_layers=total_layers,
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id
        layers_assigned += node_layers

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments_for_tensor_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeWithProfile],
):
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node in enumerate(selected_cycle):
        shard = TensorShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeInfo],
    sharding: Sharding,
) -> ShardAssignments:
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
            )


def get_hosts_from_subgraph(cycle_digraph: Topology) -> list[Host]:
    cycles = cycle_digraph.get_cycles()
    expected_length = len(list(cycle_digraph.list_nodes()))
    cycles = [cycle for cycle in cycles if len(cycle) == expected_length]
    if not cycles:
        if expected_length > 1:
            logger.warning(
                f"No cycles of length {expected_length} found even though chosen subgraph contained {expected_length} nodes"
            )
        return []

    get_thunderbolt = False
    if cycle_digraph.is_thunderbolt_cycle(cycles[0]):
        get_thunderbolt = True

    logger.info(f"Using thunderbolt cycle: {get_thunderbolt}")

    cycle = cycles[0]
    hosts: list[Host] = []
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i + 1) % len(cycle)]

        for connection in cycle_digraph.list_connections():
            if (
                connection.local_node_id == current_node.node_id
                and connection.send_back_node_id == next_node.node_id
            ):
                if get_thunderbolt and not connection.is_thunderbolt():
                    continue
                assert connection.send_back_multiaddr is not None
                host = Host(
                    ip=connection.send_back_multiaddr.ip_address,
                    port=connection.send_back_multiaddr.port,
                )
                hosts.append(host)
                break

    return hosts


def get_mlx_ibv_devices_matrix(
    selected_cycle: list[NodeInfo],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    
    For MLX RDMA, only connections over Thunderbolt interfaces (en2-en7) are used.
    Thunderbolt interfaces are identified by their interface names, not by IP address ranges.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    # MLX RDMA requires Thunderbolt connections - filter for connections where
    # the IP is on a Thunderbolt interface (en2-en7) of the target node
    get_thunderbolt = True
    logger.info("MLX RDMA filtering for connections over Thunderbolt interfaces (en2-en7)")

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            # Find the IP J uses to talk to I, filtering for Thunderbolt if needed
            connection_ips = list(_find_connection_ip(node_j, node_i, cycle_digraph, get_thunderbolt))
            interface_found = False
            for connection_ip in connection_ips:
                # This is a local IP on I, which is attached to an interface: find that interface
                if interface_name := _find_interface_name_for_ip(connection_ip, node_i):
                    matrix[i][j] = interface_name
                    logger.info(
                        f"Interface name for {connection_ip} on {node_i.node_id}: {interface_name}"
                    )
                    interface_found = True
                    break
            
            if not interface_found:
                available_ips = []
                if node_i.node_profile:
                    thunderbolt_interfaces = _get_thunderbolt_interfaces_for_node(node_i)
                    for iface in node_i.node_profile.network_interfaces:
                        if iface.name in thunderbolt_interfaces:
                            available_ips.append(f"{iface.name}:{iface.ip_address}")
                
                connection_type = "Thunderbolt" if get_thunderbolt else "any"
                logger.error(
                    f"Failed to find interface name between {node_i.node_id} and {node_j.node_id}. "
                    f"Searched {connection_type} connection IPs: {connection_ips}. "
                    f"Available RDMA interfaces on {node_i.node_id}: {available_ips or 'none'}"
                )
                raise ValueError(
                    f"Current ibv backend requires all-to-all rdma connections over Thunderbolt. "
                    f"Could not match {connection_type} connection IPs {connection_ips} to any RDMA interface on node {node_i.node_id}"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeInfo,
    node_j: NodeInfo,
    cycle_digraph: Topology,
    thunderbolt_only: bool = False,
) -> Generator[str]:
    """Find all IP addresses that connect node i to node j.
    
    If thunderbolt_only is True, only returns IPs from connections where the IP
    is on a Thunderbolt interface (en2-en7) of node_j.
    """
    for connection in cycle_digraph.list_connections():
        if (
            connection.local_node_id == node_i.node_id
            and connection.send_back_node_id == node_j.node_id
        ):
            connection_ip = connection.send_back_multiaddr.ip_address
            if thunderbolt_only:
                # Check if the connection IP is on a Thunderbolt interface of node_j
                if not _is_ip_on_thunderbolt_interface(connection_ip, node_j):
                    continue
            yield connection_ip


def _get_thunderbolt_interfaces_for_node(node_info: NodeInfo) -> set[str]:
    """
    Get the set of Thunderbolt interface names for a node.
    
    This function attempts to detect Thunderbolt interfaces using system queries.
    For local nodes, it queries the system directly. For remote nodes, it can only
    use the interface names from the profile (which should ideally include type info).
    
    TODO: NetworkInterfaceInfo should include interface type information so we can
    properly detect Thunderbolt interfaces on remote nodes without guessing.
    
    Returns a set of interface names (e.g., {'en2', 'en3'}) that are Thunderbolt interfaces.
    """
    if node_info.node_profile is None:
        return set()
    
    # Try system-based detection first (works for local interfaces)
    # Note: This only works if we're running on the same machine as the node
    # For remote nodes, we need interface type in the profile
    thunderbolt_interfaces = _get_thunderbolt_interfaces_system()
    
    # Filter to only interfaces that exist in this node's profile
    profile_interface_names = {iface.name for iface in node_info.node_profile.network_interfaces}
    thunderbolt_interfaces = thunderbolt_interfaces & profile_interface_names
    
    if thunderbolt_interfaces:
        logger.debug(f"Detected Thunderbolt interfaces via system query: {thunderbolt_interfaces}")
        return thunderbolt_interfaces
    
    # If system detection didn't find any (e.g., remote node or detection failed),
    # we cannot reliably determine Thunderbolt interfaces without type info in profile
    logger.warning(
        f"Could not detect Thunderbolt interfaces for node {node_info.node_id}. "
        f"System query found no matches in node profile. "
        f"This may cause MLX RDMA placement to fail if Thunderbolt interfaces are required."
    )
    return set()


def _get_thunderbolt_interfaces_system() -> set[str]:
    """
    Detects Thunderbolt network interfaces using networksetup on macOS.
    
    Returns a set of interface names (e.g., {'en2', 'en3'}) that are Thunderbolt interfaces.
    """
    if sys.platform != "darwin":
        return set()
    
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    
    output = result.stdout
    thunderbolt_interfaces: set[str] = set()
    
    lines = output.split("\n")
    current_hw_port: str | None = None
    current_device: str | None = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Hardware Port:"):
            current_hw_port = line.split(":", 1)[1].strip()
            current_device = None
        elif line.startswith("Device:"):
            current_device = line.split(":", 1)[1].strip()
            # Check if this hardware port is a Thunderbolt interface
            if current_hw_port and current_device:
                if current_hw_port.startswith("Thunderbolt") or current_hw_port == "Thunderbolt Bridge":
                    thunderbolt_interfaces.add(current_device)
    
    return thunderbolt_interfaces


def _is_ip_on_thunderbolt_interface(
    ip_address: str,
    node_info: NodeInfo,
) -> bool:
    """Check if an IP address is assigned to a Thunderbolt interface on the given node."""
    if node_info.node_profile is None:
        return False
    
    # Get Thunderbolt interfaces for this node
    thunderbolt_interfaces = _get_thunderbolt_interfaces_for_node(node_info)
    
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in thunderbolt_interfaces:
            continue
        if interface.ip_address == ip_address:
            return True
    
    return False


def _find_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    if node_info.node_profile is None:
        return None

    logger.info(f"Searching {node_info.node_id} for ip {ip_address}:")
    
    # Get Thunderbolt interfaces for this node
    thunderbolt_interfaces = _get_thunderbolt_interfaces_for_node(node_info)
    
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in thunderbolt_interfaces:
            continue
        logger.info(f" | {interface.name}: {interface.ip_address}")
        if interface.ip_address != ip_address:
            continue

        logger.info("Found")
        return f"rdma_{interface.name}"

    return None


def get_mlx_ibv_coordinators(
    selected_cycle: list[NodeInfo],
    coordinator_port: int,
    cycle_digraph: Topology,
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX IBV (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    
    For MLX RDMA, only connections over Thunderbolt interfaces (en2-en7) are used.
    Thunderbolt interfaces are identified by their interface names, not by IP address ranges.
    """
    rank_0_node = selected_cycle[0]
    logger.info(f"Selecting coordinator from rank 0 node: {rank_0_node.node_id}")
    
    # MLX RDMA requires Thunderbolt connections - filter for connections where
    # the IP is on a Thunderbolt interface (en2-en7) of the target node
    get_thunderbolt = True

    def get_ip_for_node(n: NodeInfo) -> str:
        if n.node_id == rank_0_node.node_id:
            return "0.0.0.0"

        for ip in _find_connection_ip(n, rank_0_node, cycle_digraph, get_thunderbolt):
            return ip

        connection_type = "Thunderbolt" if get_thunderbolt else "any"
        logger.warning(
            f"Failed to find directly connected {connection_type} ip between {n.node_id} and {rank_0_node.node_id}"
        )
        raise ValueError("Current ibv backend requires all-to-all rdma connections over Thunderbolt")

    return {
        n.node_id: f"{get_ip_for_node(n)}:{coordinator_port}" for n in selected_cycle
    }

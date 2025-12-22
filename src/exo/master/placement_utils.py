import subprocess
import sys
import ipaddress
import os
from collections.abc import Generator
from dataclasses import dataclass
from math import floor
from typing import TypeGuard, cast

from loguru import logger
from pydantic import BaseModel

from exo.shared.constants import LB_MEMBW_GBPS
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelMetadata
from exo.shared.types.profiling import NodePerformanceProfile
from exo.shared.types.topology import NodeInfo
from exo.shared.types.profiling import NetworkInterfaceInfo
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)

_THUNDERBOLT_INTERFACE_NAME_GUESS: frozenset[str] = frozenset(
    {
        "en2",
        "en3",
        "en4",
        "en5",
        "en6",
        "en7",
    }
)

_MLX_RDMA_ALLOWED_INTERFACES_ENV_VAR: str = "EXO_MLX_RDMA_INTERFACES"


class NodeWithProfile(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile


def narrow_all_nodes(nodes: list[NodeInfo]) -> TypeGuard[list[NodeWithProfile]]:
    return all(node.node_profile is not None for node in nodes)


def _normalize_chip_id(chip_id: str) -> str:
    return " ".join(chip_id.replace("Apple", "").strip().split())


def estimated_memory_bandwidth_gbps(*, chip_id: str) -> float:
    """Return an estimated SoC memory bandwidth in GB/s.

    We do not currently measure memory bandwidth directly. Instead, we use a
    conservative chip-string heuristic derived from Apple Silicon specs.

    Unknown chips fall back to the lower bound used elsewhere in the codebase.
    """

    normalized = _normalize_chip_id(chip_id).lower()

    # Keep this table intentionally small and conservative: it only needs to be
    # good enough to order common Apple Silicon parts for sharding decisions.
    # Values are approximate peak unified-memory bandwidths (GB/s).
    chip_to_gbps: dict[str, float] = {
        "m1": 68.0,
        "m1 pro": 200.0,
        "m1 max": 400.0,
        "m1 ultra": 800.0,
        "m2": 100.0,
        "m2 pro": 200.0,
        "m2 max": 400.0,
        "m2 ultra": 800.0,
        "m3": 100.0,
        "m3 pro": 150.0,
        "m3 max": 300.0,
        "m3 ultra": 800.0,
        # Default placeholders for newer chips; kept conservative.
        "m4": 120.0,
        "m4 pro": 250.0,
        "m4 max": 400.0,
    }

    # Match longest keys first to avoid "m1" catching "m1 max".
    for key in sorted(chip_to_gbps.keys(), key=len, reverse=True):
        if key in normalized:
            return chip_to_gbps[key]

    return float(LB_MEMBW_GBPS)


def calculate_usable_memory_with_buffer(
    *, available_bytes: int, total_bytes: int | None = None
) -> int:
    """Calculate usable memory with a buffer that ensures minimum absolute free space.
    
    Uses both a percentage cap (90%) and a minimum absolute free space requirement (12GB)
    to ensure larger nodes have proportionally more free space while still maintaining
    reasonable headroom on smaller nodes.

    Args:
        available_bytes: Currently available memory in bytes
        total_bytes: Total memory in bytes (optional, used for percentage cap)

    Returns:
        Usable memory in bytes after applying buffer constraints
    """
    # Minimum absolute free space: 12GB for system processes, KV cache, etc.
    min_free_bytes = 12 * (1024**3)
    
    # Percentage cap: don't use more than 90% of total memory
    max_usage_percentage = 0.9
    
    # Calculate usable memory with absolute minimum free space constraint
    usable_with_min_free = max(0, available_bytes - min_free_bytes)
    
    # Calculate usable memory with percentage cap (if total_bytes is provided)
    if total_bytes is not None:
        max_usable_by_percentage = int(total_bytes * max_usage_percentage)
        # For percentage cap, we need to consider how much is already used
        used_bytes = total_bytes - available_bytes
        usable_with_percentage = max(0, max_usable_by_percentage - used_bytes)
        
        # Take the minimum of both constraints to satisfy both
        return min(usable_with_min_free, usable_with_percentage)
    else:
        # If no total_bytes, just apply minimum free space constraint
        return usable_with_min_free


def rotate_cycle_to_best_rank_0_node(cycle: list[NodeInfo]) -> list[NodeInfo]:
    """Rotate a directed cycle so the strongest node becomes rank 0.

    In pipeline sharding, `selected_cycle[0]` becomes `device_rank=0` and receives the
    first layer range (i.e. `start_layer=0`). Additionally, for MLX RDMA the rank-0
    node is used as the coordinator root. Topology cycle enumeration can start at any
    node, so we rotate deterministically to prefer the best machine first.

    Ranking heuristic:
    - Prefer higher estimated unified-memory bandwidth (chip class proxy for speed)
    - Then prefer higher total RAM (machine capacity proxy)
    - Then prefer higher currently-available RAM (tie-breaker)
    """

    if len(cycle) <= 1:
        return list(cycle)

    if not narrow_all_nodes(cycle):
        return list(cycle)

    typed_cycle = cast(list[NodeWithProfile], cycle)

    def priority_key(node: NodeWithProfile) -> tuple[float, int, int, str]:
        membw = estimated_memory_bandwidth_gbps(chip_id=node.node_profile.chip_id)
        ram_total = node.node_profile.memory.ram_total.in_bytes
        ram_available = node.node_profile.memory.ram_available.in_bytes
        return (membw, ram_total, ram_available, str(node.node_id))

    best_node_id = max(typed_cycle, key=priority_key).node_id
    best_index = next(
        i for i, node in enumerate(typed_cycle) if node.node_id == best_node_id
    )
    return list(cycle[best_index:]) + list(cycle[:best_index])


@dataclass(frozen=True, slots=True)
class _PipelineNodeCapacity:
    node_id: NodeId
    rank_index: int
    max_layers_by_memory: int
    membw_gbps: float
    ram_total_bytes: int


def _pipeline_node_capacities(
    *,
    model_meta: ModelMetadata,
    selected_cycle: list[NodeWithProfile],
) -> list[_PipelineNodeCapacity]:
    total_layers = model_meta.n_layers
    if total_layers <= 0:
        raise ValueError("Model must have at least 1 layer")

    bytes_per_layer = model_meta.storage_size.in_bytes / total_layers
    if bytes_per_layer <= 0:
        # Extremely defensive: avoid division-by-zero / negative caps.
        bytes_per_layer = 1.0

    capacities: list[_PipelineNodeCapacity] = []
    for i, node in enumerate(selected_cycle):
        available_bytes = node.node_profile.memory.ram_available.in_bytes
        total_bytes = node.node_profile.memory.ram_total.in_bytes
        
        # Calculate usable memory with buffer that ensures minimum absolute free space
        # while capping at 90% of total memory
        available_bytes_with_buffer = calculate_usable_memory_with_buffer(
            available_bytes=available_bytes,
            total_bytes=total_bytes,
        )
        
        max_layers_by_available = int(floor(available_bytes_with_buffer / bytes_per_layer))
        # Cap at total_layers (can't assign more layers than the model has)
        # But for greedy allocation, we want to know actual capacity, so we'll use min(total_layers, capacity)
        max_layers = max(0, min(total_layers, max_layers_by_available))

        membw = estimated_memory_bandwidth_gbps(chip_id=node.node_profile.chip_id)
        ram_total = node.node_profile.memory.ram_total.in_bytes

        capacities.append(
            _PipelineNodeCapacity(
                node_id=node.node_id,
                rank_index=i,
                max_layers_by_memory=max_layers,
                membw_gbps=membw,
                ram_total_bytes=ram_total,
            )
        )

    return capacities


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
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    if total_layers < world_size:
        raise ValueError(
            f"Pipeline sharding requires n_layers >= world_size, got {total_layers=} {world_size=}"
        )

    capacities = _pipeline_node_capacities(model_meta=model_meta, selected_cycle=selected_cycle)
    
    node_id_to_node = {node.node_id: node for node in selected_cycle}
    
    ranked_capacities = sorted(
        capacities,
        key=lambda c: (c.membw_gbps, c.ram_total_bytes, str(c.node_id)),
        reverse=True,
    )
    
    logger.info(
        f"Pipeline placement ranking: {[(c.node_id, c.membw_gbps, c.ram_total_bytes, c.max_layers_by_memory) for c in ranked_capacities]}"
    )
    logger.info(
        f"Model: {model_meta.model_id}, total_layers={total_layers}, "
        f"storage_size={model_meta.storage_size.in_bytes / (1024**3):.2f} GB"
    )
    
    sorted_cycle = [
        node_id_to_node[c.node_id]
        for c in ranked_capacities
        if c.node_id in node_id_to_node
    ]
    
    logger.info(
        f"Sorted cycle order: {[node.node_id for node in sorted_cycle]}"
    )
    
    if sum(c.max_layers_by_memory for c in capacities) < total_layers:
        # This should be rare given we pre-filter cycles by total memory, but
        # storage size vs runtime memory footprint is approximate.
        logger.warning(
            "Insufficient per-node memory caps for pipeline sharding; using greedy allocation"
        )
        # Even in fallback, use greedy allocation: fastest node first
        desired_layers = [0 for _ in range(world_size)]
        remaining = total_layers
        
        node_id_to_index = {node.node_id: i for i, node in enumerate(sorted_cycle)}
        
        # Greedily fill nodes in order of speed (fastest first)
        # Use max_layers_by_memory as capacity, even if it's less than total_layers
        for c in ranked_capacities:
            if remaining <= 0:
                break
            if c.max_layers_by_memory <= 0:
                continue
            
            node_index = node_id_to_index[c.node_id]
            headroom = c.max_layers_by_memory - desired_layers[node_index]
            if headroom <= 0:
                continue
            
            # Take as many layers as this node can hold, up to remaining
            take = min(remaining, headroom)
            desired_layers[node_index] += take
            remaining -= take
        
        # If we still have layers remaining, distribute to fastest nodes first
        if remaining > 0:
            for c in ranked_capacities:
                if remaining <= 0:
                    break
                node_index = node_id_to_index[c.node_id]
                # Give at least 1 layer to fastest nodes first
                take = min(remaining, 1)
                desired_layers[node_index] += take
                remaining -= take
    else:
        # Greedy allocation: fill fastest node first, then 2nd fastest, then 3rd, etc.
        # - Fill each node to capacity before moving to the next
        # - Cap each node at its available memory limit
        # - All nodes are included in the instance (even with 0 layers) for KV cache handling
        desired_layers = [0 for _ in range(world_size)]
        remaining = total_layers

        node_id_to_index = {node.node_id: i for i, node in enumerate(sorted_cycle)}
        
        # CRITICAL: If the fastest node can hold the entire model, give it ALL layers
        # This ensures small models are concentrated on the fastest node
        # Check by comparing model storage size directly to available RAM (more reliable than layer count)
        fastest_capacity = ranked_capacities[0]
        fastest_node = node_id_to_node[fastest_capacity.node_id]
        fastest_available_ram = fastest_node.node_profile.memory.ram_available.in_bytes
        fastest_total_ram = fastest_node.node_profile.memory.ram_total.in_bytes
        model_storage_bytes = model_meta.storage_size.in_bytes
        
        logger.info(
            f"Greedy check: fastest node {fastest_capacity.node_id} has "
            f"{fastest_available_ram / (1024**3):.2f} GB available RAM, "
            f"{fastest_total_ram / (1024**3):.2f} GB total RAM, "
            f"model needs {model_storage_bytes / (1024**3):.2f} GB"
        )
        
        # Calculate maximum usable memory based on total capacity (assuming node is empty)
        # This ensures we check if the node CAN hold the model even if some RAM is currently used
        max_usable_ram = calculate_usable_memory_with_buffer(
            available_bytes=fastest_total_ram,
            total_bytes=fastest_total_ram,
        )
        
        # Also use greedy allocation for small models (< 10GB) on large nodes (> 64GB)
        is_small_model = model_storage_bytes < 10 * (1024**3)  # < 10GB
        is_large_node = fastest_total_ram > 64 * (1024**3)  # > 64GB
        
        if max_usable_ram >= model_storage_bytes or (is_small_model and is_large_node):
            # Fastest node can hold entire model (with buffer) - give it all layers
            # Use the actual available memory with buffer for layer calculation
            fastest_index = node_id_to_index[fastest_capacity.node_id]
            desired_layers = [0 for _ in range(world_size)]  # Reset to all zeros
            
            # Calculate max layers based on available memory with buffer
            available_with_buffer = calculate_usable_memory_with_buffer(
                available_bytes=fastest_available_ram,
                total_bytes=fastest_total_ram,
            )
            bytes_per_layer = model_storage_bytes / total_layers if total_layers > 0 else 0
            max_layers_with_buffer = int((available_with_buffer / bytes_per_layer)) if bytes_per_layer > 0 else total_layers
            layers_to_assign = min(total_layers, max_layers_with_buffer)
            
            desired_layers[fastest_index] = layers_to_assign
            remaining = total_layers - layers_to_assign
            
            logger.info(
                f"✓ GREEDY ALLOCATION: Fastest node {fastest_capacity.node_id} can hold entire model "
                f"({model_storage_bytes / (1024**3):.2f} GB in {fastest_total_ram / (1024**3):.2f} GB total RAM, "
                f"max usable: {max_usable_ram / (1024**3):.2f} GB, "
                f"available with buffer: {available_with_buffer / (1024**3):.2f} GB) - "
                f"assigning {layers_to_assign} layers to fastest node, "
                f"remaining {remaining} layers to other nodes"
            )
            logger.info(
                f"✓ Layer assignment: {[(sorted_cycle[i].node_id, desired_layers[i]) for i in range(world_size)]}"
            )
            
            # Distribute remaining layers to other nodes if fastest node couldn't hold all layers
            if remaining > 0:
                logger.info(
                    f"Distributing {remaining} remaining layers to other nodes"
                )
                for c in ranked_capacities:
                    if remaining <= 0:
                        break
                    node_index = node_id_to_index[c.node_id]
                    # Skip the fastest node (already assigned)
                    if node_index == fastest_index:
                        continue
                    if c.max_layers_by_memory <= 0:
                        continue
                    
                    headroom = c.max_layers_by_memory - desired_layers[node_index]
                    if headroom <= 0:
                        continue
                    
                    take = min(remaining, headroom)
                    desired_layers[node_index] += take
                    remaining -= take
                    
                    logger.info(
                        f"Remaining allocation: node {c.node_id} (membw={c.membw_gbps:.1f} GB/s, "
                        f"ram={c.ram_total_bytes / (1024**3):.1f} GB, capacity={c.max_layers_by_memory}) "
                        f"gets {take} layers (total: {desired_layers[node_index]}, remaining: {remaining})"
                    )
        else:
            # Fastest node cannot hold all layers - distribute greedily
            # Greedily fill nodes in order of speed (fastest first)
            # Fastest node gets ALL layers it can hold (with 10% buffer) before any other node gets any
            for c in ranked_capacities:
                if remaining <= 0:
                    break
                if c.max_layers_by_memory <= 0:
                    continue
                
                node_index = node_id_to_index[c.node_id]
                # max_layers_by_memory already has 10% buffer applied in _pipeline_node_capacities
                headroom = c.max_layers_by_memory - desired_layers[node_index]
                if headroom <= 0:
                    continue
                
                # Take as many layers as this node can hold, up to remaining
                # Fastest node (first in ranked_capacities) gets ALL remaining layers if it can hold them
                take = min(remaining, headroom)
                desired_layers[node_index] += take
                remaining -= take
                
                logger.info(
                    f"Greedy allocation: node {c.node_id} (membw={c.membw_gbps:.1f} GB/s, "
                    f"ram={c.ram_total_bytes / (1024**3):.1f} GB, capacity={c.max_layers_by_memory}) "
                    f"gets {take} layers (total: {desired_layers[node_index]}, remaining: {remaining})"
                )

            # If we still have layers remaining after filling nodes sequentially,
            # only distribute to slower nodes if faster nodes are at capacity
            if remaining > 0:
                for c in ranked_capacities:
                    if remaining <= 0:
                        break
                    node_index = node_id_to_index[c.node_id]
                    headroom = c.max_layers_by_memory - desired_layers[node_index]
                    if headroom <= 0:
                        continue
                    
                    # Only give layers to this node if we have remaining layers
                    # (faster nodes should already be at capacity from the first pass)
                    take = min(remaining, headroom)
                    desired_layers[node_index] += take
                    remaining -= take

        assert remaining == 0, "Allocation should exhaust all layers when caps permit"
        
        logger.info(
            f"Final layer allocation: {[(sorted_cycle[i].node_id, desired_layers[i]) for i in range(world_size)]}"
        )

    layers_assigned = 0
    for i, node in enumerate(sorted_cycle):
        node_layers = desired_layers[i]

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

    assert layers_assigned == total_layers, (
        "Pipeline sharding must assign all layers exactly once"
    )

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

    # MLX RDMA requires Thunderbolt connections - restrict candidate interfaces to TB.
    logger.info("MLX RDMA selecting RDMA interfaces over Thunderbolt (IPv4 only)")

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            # Prefer deterministic profile-based matching:
            # find a Thunderbolt IPv4 interface on node_i whose subnet contains an IPv4
            # address on node_j (supports /30 and other non-/24 masks).
            # This works for point-to-point links where both nodes have IPs in the same subnet.
            interface_name = _find_mlx_rdma_interface_for_peer(node_i, node_j)
            if interface_name is not None:
                # CRITICAL: Validate that the interface is actually Thunderbolt
                # This is a safety check - _find_mlx_rdma_interface_for_peer should only return Thunderbolt
                if not _is_thunderbolt_interface_name(interface_name, node_i):
                    raise ValueError(
                        f"Interface '{interface_name}' returned by _find_mlx_rdma_interface_for_peer "
                        f"is not a Thunderbolt interface according to node profile. "
                        f"Only Thunderbolt interfaces are allowed for MLX RDMA."
                    )
                matrix[i][j] = interface_name
                logger.info(
                    f"Matched interface {interface_name} on {node_i.node_id} for connection to {node_j.node_id} "
                    f"via subnet matching"
                )
                continue

            # Fallback: match via topology "send_back_multiaddr" addresses if we cannot
            # compute subnets from profiles (e.g., missing netmask).
            # We need the IP on node_i that node_j uses to connect to node_i.
            # _find_connection_ip(node_i, node_j) finds connections FROM node_i TO node_j,
            # which gives us the IP on node_j. But we need the IP on node_i.
            # So we look for connections FROM node_j TO node_i, which gives us the IP on node_i.
            connection_ips = list(
                _find_connection_ip(node_j, node_i, cycle_digraph, thunderbolt_only=False)
            )
            logger.debug(
                f"Looking for connection IPs on {node_i.node_id} for connection to {node_j.node_id}: "
                f"found {connection_ips}"
            )
            connection_ips = [
                ip
                for ip in connection_ips
                if _is_ip_on_thunderbolt_interface(ip, node_i)
            ]
            for connection_ip in connection_ips:
                if interface_name := _find_interface_name_for_ip(connection_ip, node_i):
                    # CRITICAL: Validate that the interface is actually Thunderbolt
                    # _find_interface_name_for_ip should only return Thunderbolt, but verify
                    if not _is_thunderbolt_interface_name(interface_name, node_i):
                        logger.error(
                            f"Interface '{interface_name}' found for IP {connection_ip} on {node_i.node_id} "
                            f"is not a Thunderbolt interface according to node profile. Skipping."
                        )
                        continue
                    matrix[i][j] = interface_name
                    logger.info(
                        f"Interface name for {connection_ip} on {node_i.node_id}: {interface_name}"
                    )
                    break
            
            if matrix[i][j] is None:
                available_ips = []
                if node_i.node_profile:
                    thunderbolt_interfaces = _get_thunderbolt_interfaces_for_node(node_i)
                    for iface in node_i.node_profile.network_interfaces:
                        if iface.name in thunderbolt_interfaces:
                            if not _is_ipv4_address(iface.ip_address):
                                continue
                            available_ips.append(
                                f"{_to_rdma_interface_name(iface.name, node_i)}:{iface.ip_address}"
                            )
                
                # Log detailed debugging info
                logger.error(
                    f"Failed to find interface name between {node_i.node_id} and {node_j.node_id}. "
                    f"Searched Thunderbolt connection IPs: {connection_ips}. "
                    f"Available RDMA interfaces on {node_i.node_id}: {available_ips or 'none'}"
                )
                
                # Try to find why _find_mlx_rdma_interface_for_peer failed
                local_ifaces = _get_mlx_rdma_thunderbolt_interfaces_for_node(node_i)
                peer_ifaces = _get_mlx_rdma_thunderbolt_interfaces_for_node(node_j)
                logger.error(
                    f"Debug: node_i ({node_i.node_id}) has {len(local_ifaces)} Thunderbolt interfaces, "
                    f"node_j ({node_j.node_id}) has {len(peer_ifaces)} Thunderbolt interfaces"
                )
                if local_ifaces:
                    for iface in local_ifaces:
                        network = _ipv4_interface_network(iface)
                        logger.error(
                            f"  node_i interface {iface.name}: IP={iface.ip_address}, "
                            f"netmask={iface.netmask}, network={network}"
                        )
                if peer_ifaces:
                    for iface in peer_ifaces:
                        network = _ipv4_interface_network(iface)
                        logger.error(
                            f"  node_j interface {iface.name}: IP={iface.ip_address}, "
                            f"netmask={iface.netmask}, network={network}"
                        )
                
                raise ValueError(
                    f"Current ibv backend requires all-to-all rdma connections over Thunderbolt. "
                    f"Could not match Thunderbolt connection IPs {connection_ips} to any RDMA interface on node {node_i.node_id}. "
                    f"Available interfaces: {available_ips}"
                )

    # CRITICAL: Final validation - ensure ALL interfaces in the matrix are Thunderbolt
    # This is a safety net to catch any non-Thunderbolt interfaces that might have slipped through
    for i, row in enumerate(matrix):
        for j, interface_name in enumerate(row):
            if interface_name is not None:
                node_i = selected_cycle[i]
                if not _is_thunderbolt_interface_name(interface_name, node_i):
                    raise ValueError(
                        f"Matrix[{i}][{j}] contains non-Thunderbolt interface '{interface_name}'. "
                        f"Only Thunderbolt interfaces are allowed for MLX RDMA. "
                        f"This is a critical error - MLX RDMA traffic must only use Thunderbolt interfaces. "
                        f"Node: {node_i.node_id}"
                    )

    logger.info("MLX RDMA devices matrix validation passed - all interfaces are Thunderbolt")
    return matrix


def _find_connection_ip(
    node_i: NodeInfo,
    node_j: NodeInfo,
    cycle_digraph: Topology,
    thunderbolt_only: bool = False,
) -> Generator[str]:
    """Find all IP addresses that connect node i to node j.

    The connection IP is the IP address that node_j should use to connect to node_i,
    so it will be on node_i's interfaces.
    
    If thunderbolt_only is True, only returns IPs from connections where the IP
    is on a Thunderbolt interface of node_i (the target node).
    """
    all_connections = list(cycle_digraph.list_connections())
    logger.debug(
        f"_find_connection_ip: Looking for connection from {node_i.node_id} to {node_j.node_id}. "
        f"Total connections in topology: {len(all_connections)}"
    )
    for connection in all_connections:
        if (
            connection.local_node_id == node_i.node_id
            and connection.send_back_node_id == node_j.node_id
        ):
            connection_ip = connection.send_back_multiaddr.ip_address
            logger.debug(
                f"Found connection from {node_i.node_id} to {node_j.node_id}: "
                f"IP={connection_ip}, thunderbolt_only={thunderbolt_only}"
            )
            if not _is_ipv4_address(connection_ip):
                logger.debug(f"Skipping non-IPv4 address: {connection_ip}")
                continue
            if thunderbolt_only:
                # Check if the connection IP is on a Thunderbolt interface of node_i
                # (the target node where the IP should be located)
                is_tb = _is_ip_on_thunderbolt_interface(connection_ip, node_i)
                logger.debug(
                    f"Connection IP {connection_ip} is on Thunderbolt interface: {is_tb}"
                )
                if not is_tb:
                    continue
            yield connection_ip


def _is_ipv4_address(ip_address: str) -> bool:
    normalized_ip = ip_address.split("%", 1)[0].strip()
    try:
        return ipaddress.ip_address(normalized_ip).version == 4
    except ValueError:
        return False


def _is_thunderbolt_interface_name(interface_name: str, node_info: NodeInfo | None = None) -> bool:
    """Check if an interface name is a Thunderbolt interface.
    
    This function first checks the node profile's is_thunderbolt field if available,
    which is the most reliable method. Falls back to name-based heuristics if profile
    data is not available.
    
    Args:
        interface_name: The interface name to check (e.g., "en1", "rdma_en2")
        node_info: Optional node info to check the profile's is_thunderbolt field
    
    Returns:
        True if the interface is Thunderbolt, False otherwise
    """
    # Remove rdma_ prefix if present for checking
    base_name = interface_name.removeprefix("rdma_")
    
    # First, try to use the profile's is_thunderbolt field if available
    # This is the most reliable method as it uses actual system detection
    if node_info and node_info.node_profile:
        for interface in node_info.node_profile.network_interfaces:
            if interface.name == base_name:
                if interface.is_thunderbolt is True:
                    return True
                elif interface.is_thunderbolt is False:
                    return False
                # If is_thunderbolt is None, fall through to heuristic check
    
    # Fallback: Use name-based heuristic (en2-en7)
    # Note: Some Macs have Thunderbolt on en1, so this is not perfect
    # but it's better than nothing when profile data is unavailable
    return base_name in _THUNDERBOLT_INTERFACE_NAME_GUESS


def _to_rdma_interface_name(interface_name: str, node_info: NodeInfo | None = None) -> str:
    """Convert an interface name to RDMA device name format.
    
    CRITICAL: Only Thunderbolt interfaces are allowed for MLX RDMA.
    This function will raise ValueError if a non-Thunderbolt interface is provided.
    
    Args:
        interface_name: The interface name to convert (e.g., "en1", "en2")
        node_info: Optional node info to validate against profile's is_thunderbolt field
    """
    # Validate that this is a Thunderbolt interface
    if not _is_thunderbolt_interface_name(interface_name, node_info):
        raise ValueError(
            f"Interface '{interface_name}' is not a Thunderbolt interface. "
            f"Only Thunderbolt interfaces can be used for MLX RDMA. "
            f"Non-Thunderbolt interfaces (e.g., en0=Ethernet, en1=Wi-Fi on some systems) are not allowed."
        )
    
    # RDMA devices on macOS are named with "rdma_" prefix (e.g., "rdma_en1", "rdma_en2")
    # as shown by `ibv_devices` command. Add the prefix if not present.
    return interface_name if interface_name.startswith("rdma_") else f"rdma_{interface_name}"


def _get_mlx_rdma_allowed_interface_names() -> frozenset[str] | None:
    configured = os.getenv(_MLX_RDMA_ALLOWED_INTERFACES_ENV_VAR)
    if configured is None or configured.strip() == "":
        return None
    names = {name.strip() for name in configured.split(",") if name.strip() != ""}
    return frozenset(names) if names else None


def _get_mlx_rdma_thunderbolt_interfaces_for_node(node_info: NodeInfo) -> list[NetworkInterfaceInfo]:
    if node_info.node_profile is None:
        return []

    thunderbolt_names = _get_thunderbolt_interfaces_for_node(node_info)
    allowed = _get_mlx_rdma_allowed_interface_names()
    if allowed is not None:
        thunderbolt_names = thunderbolt_names & set(allowed)

    all_candidates: list[NetworkInterfaceInfo] = []
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in thunderbolt_names:
            continue
        if not _is_ipv4_address(interface.ip_address):
            continue
        all_candidates.append(interface)

    # For MLX RDMA, we need ALL Thunderbolt interfaces, not just "preferred" ones.
    # Each node may have multiple Thunderbolt interfaces for different connections.
    # The MTU/up filtering was too aggressive and excluded valid interfaces.
    # Return all candidates, sorted for consistency.
    all_candidates.sort(key=lambda iface: (iface.name, iface.ip_address))
    logger.debug(
        f"Node {node_info.node_id} has {len(all_candidates)} Thunderbolt interfaces: "
        f"{[(iface.name, iface.ip_address) for iface in all_candidates]}"
    )
    return all_candidates


def _ipv4_interface_network(interface: NetworkInterfaceInfo) -> ipaddress.IPv4Network | None:
    if not _is_ipv4_address(interface.ip_address):
        return None
    if interface.netmask is None:
        return None
    
    # Convert netmask to CIDR notation if it's in IP address format
    try:
        if _is_ipv4_address(interface.netmask):
            # Netmask is in IP format (e.g., "255.255.255.252"), convert to CIDR
            netmask_ip = ipaddress.ip_address(interface.netmask)
            # Count leading ones in the netmask to get CIDR prefix length
            netmask_int = int(netmask_ip)
            prefix_len = bin(netmask_int).count("1")
            cidr = f"/{prefix_len}"
        else:
            # Assume it's already in CIDR format (e.g., "/30")
            cidr = interface.netmask if interface.netmask.startswith("/") else f"/{interface.netmask}"
        
        return ipaddress.ip_network(
            f"{interface.ip_address}{cidr}",
            strict=False,
        )
    except (ValueError, AttributeError):
        return None


def _find_mlx_rdma_interface_for_peer(
    node_local: NodeInfo,
    node_peer: NodeInfo,
) -> str | None:
    """Find the Thunderbolt interface on node_local that connects to node_peer.
    
    This works by finding interfaces on both nodes that are in the same subnet.
    For point-to-point links (e.g., /30 subnets), both nodes will have IPs in the same subnet.
    """
    local_ifaces = _get_mlx_rdma_thunderbolt_interfaces_for_node(node_local)
    peer_ifaces = _get_mlx_rdma_thunderbolt_interfaces_for_node(node_peer)
    if not local_ifaces or not peer_ifaces:
        logger.debug(
            f"Subnet matching: {node_local.node_id} has {len(local_ifaces)} interfaces, "
            f"{node_peer.node_id} has {len(peer_ifaces)} interfaces"
        )
        return None

    peer_ipv4_addresses = [
        ipaddress.ip_address(peer_iface.ip_address)
        for peer_iface in peer_ifaces
        if _is_ipv4_address(peer_iface.ip_address)
    ]

    # Try matching: find a local interface whose subnet contains a peer IP
    for local_iface in local_ifaces:
        local_network = _ipv4_interface_network(local_iface)
        if local_network is None:
            logger.debug(
                f"Could not calculate network for {node_local.node_id} interface "
                f"{local_iface.name} (IP={local_iface.ip_address}, netmask={local_iface.netmask})"
            )
            continue
        for peer_ip in peer_ipv4_addresses:
            if peer_ip in local_network:
                logger.info(
                    f"Matched {node_local.node_id} interface {local_iface.name} "
                    f"(network {local_network}) to peer {node_peer.node_id} IP {peer_ip}"
                )
                return _to_rdma_interface_name(local_iface.name, node_local)

    # Also try reverse: find a peer interface whose subnet contains a local IP
    # This handles cases where the netmask might be missing on one side
    local_ipv4_addresses = [
        ipaddress.ip_address(local_iface.ip_address)
        for local_iface in local_ifaces
        if _is_ipv4_address(local_iface.ip_address)
    ]
    for peer_iface in peer_ifaces:
        peer_network = _ipv4_interface_network(peer_iface)
        if peer_network is None:
            continue
        for local_ip in local_ipv4_addresses:
            if local_ip in peer_network:
                # Found a match via reverse subnet check - use the local interface
                # that has an IP in the peer's subnet
                for local_iface in local_ifaces:
                    if ipaddress.ip_address(local_iface.ip_address) == local_ip:
                        logger.info(
                            f"Matched {node_local.node_id} interface {local_iface.name} "
                            f"(IP {local_ip} in peer network {peer_network}) to peer {node_peer.node_id}"
                        )
                        return _to_rdma_interface_name(local_iface.name, node_local)

    logger.debug(
        f"Subnet matching failed: {node_local.node_id} has {len(local_ifaces)} interfaces, "
        f"{node_peer.node_id} has {len(peer_ifaces)} interfaces, "
        f"but no peer IPs matched any local network (and vice versa)"
    )
    return None

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

    thunderbolt_from_profile = {
        interface.name
        for interface in node_info.node_profile.network_interfaces
        if interface.is_thunderbolt is True
    }
    if thunderbolt_from_profile:
        return thunderbolt_from_profile
    
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

    heuristic_thunderbolt_interfaces = {
        interface_name
        for interface_name in profile_interface_names
        if interface_name in _THUNDERBOLT_INTERFACE_NAME_GUESS
    }
    if heuristic_thunderbolt_interfaces:
        logger.debug(
            "Falling back to name-based Thunderbolt interface detection: "
            f"{heuristic_thunderbolt_interfaces}"
        )
        return heuristic_thunderbolt_interfaces
    
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
    if not _is_ipv4_address(ip_address):
        return False
    if node_info.node_profile is None:
        return False
    
    # Get Thunderbolt interfaces for this node
    thunderbolt_interfaces = _get_thunderbolt_interfaces_for_node(node_info)
    
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in thunderbolt_interfaces:
            continue
        if not _is_ipv4_address(interface.ip_address):
            continue
        if interface.ip_address == ip_address:
            return True
    
    return False


def _find_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    if not _is_ipv4_address(ip_address):
        return None
    if node_info.node_profile is None:
        return None

    logger.info(f"Searching {node_info.node_id} for ip {ip_address}:")
    
    # Get Thunderbolt interfaces for this node
    thunderbolt_interfaces = _get_thunderbolt_interfaces_for_node(node_info)
    
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in thunderbolt_interfaces:
            continue
        logger.info(f" | {interface.name}: {interface.ip_address}")
        if not _is_ipv4_address(interface.ip_address):
            continue
        if interface.ip_address != ip_address:
            continue

        logger.info("Found")
        return _to_rdma_interface_name(interface.name, node_info)

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

    def get_ip_for_node(n: NodeInfo) -> str:
        if n.node_id == rank_0_node.node_id:
            # For rank 0, use 0.0.0.0 to listen on all interfaces
            # This allows non-rank-0 nodes on different subnets to connect
            # Each non-rank-0 node will connect to the specific IP on their subnet
            logger.info("Rank 0 coordinator using 0.0.0.0 to listen on all interfaces")
            return "0.0.0.0"

        # Prefer subnet-based selection: choose the rank-0 IPv4 on a Thunderbolt interface
        # whose subnet contains one of this node's Thunderbolt IPv4s.
        rank_0_ifaces = _get_mlx_rdma_thunderbolt_interfaces_for_node(rank_0_node)
        node_ifaces = _get_mlx_rdma_thunderbolt_interfaces_for_node(n)

        node_ipv4_addresses = [
            ipaddress.ip_address(iface.ip_address)
            for iface in node_ifaces
            if _is_ipv4_address(iface.ip_address)
        ]
        for rank_0_iface in rank_0_ifaces:
            rank_0_network = _ipv4_interface_network(rank_0_iface)
            if rank_0_network is None:
                continue
            if any(node_ip in rank_0_network for node_ip in node_ipv4_addresses):
                return rank_0_iface.ip_address

        # Fallback to topology matching.
        connection_ips = list(
            _find_connection_ip(n, rank_0_node, cycle_digraph, thunderbolt_only=False)
        )
        connection_ips = [
            ip
            for ip in connection_ips
            if _is_ip_on_thunderbolt_interface(ip, rank_0_node)
        ]
        if connection_ips:
            return connection_ips[0]

        logger.warning(
            f"Failed to find directly connected Thunderbolt ip between {n.node_id} and {rank_0_node.node_id}"
        )
        raise ValueError(
            "Current ibv backend requires all-to-all rdma connections over Thunderbolt"
        )

    return {
        n.node_id: f"{get_ip_for_node(n)}:{coordinator_port}" for n in selected_cycle
    }

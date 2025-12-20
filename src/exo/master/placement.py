"""Instance placement algorithms for model distribution.

This module provides functions for placing model instances across nodes
in the cluster, taking into account topology, memory constraints, and
sharding strategies.
"""

import os
import random
from collections.abc import Mapping
from copy import deepcopy
from typing import Sequence

from loguru import logger

from exo.master.placement_utils import (
    filter_cycles_by_memory,
    get_hosts_from_subgraph,
    get_mlx_ibv_coordinators,
    get_mlx_ibv_devices_matrix,
    get_shard_assignments,
    get_smallest_cycles,
)
from exo.networking.manual_topology import infer_role_from_ips
from exo.shared.topology import Topology
from exo.shared.types.commands import (
    CreateInstance,
    DeleteInstance,
    PlaceInstance,
)
from exo.shared.types.common import Host
from exo.shared.types.events import Event, InstanceCreated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)


def _role_for_node_info(node: NodeInfo) -> str | None:
    if node.node_profile is None:
        return None
    ips = {iface.ip_address for iface in node.node_profile.network_interfaces}
    return infer_role_from_ips(ips)


def random_ephemeral_port() -> int:
    """Generate a random ephemeral port number.

    Generates a port in the ephemeral range (49152-65535), avoiding
    the default EXO API port (52415).

    Returns:
        Random port number in the ephemeral range, excluding 52415.
    """
    port = random.randint(49153, 65535)
    return port - 1 if port <= 52415 else 52414


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    """Add an instance to the current placements without placement calculation.

    Simply adds the provided instance to the current instances mapping.
    Used when instance configuration is explicitly provided.

    Args:
        command: CreateInstance command with the instance to add.
        topology: Current cluster topology (unused but kept for interface consistency).
        current_instances: Current instance placements.

    Returns:
        New instance mapping with the instance added.

    Note:
        TODO: Add validation against topology to ensure instance nodes exist.
    """
    return {**current_instances, command.instance.instance_id: command.instance}


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    """Calculate optimal placement for a model instance.

    Selects the best cycle of nodes for placing the model based on:
    1. Minimum node count requirement
    2. Available memory on nodes
    3. Cycle size (prefers smaller cycles)
    4. Thunderbolt connectivity (prefers TB cycles)
    5. Leaf nodes (prefers cycles with leaf nodes)
    6. Available RAM (prefers cycles with more RAM)

    Creates the instance configuration with shard assignments and network
    configuration (ports, hosts, IBV coordinators) based on the selected cycle.

    Args:
        command: PlaceInstance command with model metadata and requirements.
        topology: Current cluster topology.
        current_instances: Current instance placements.

    Returns:
        New instance mapping with the placed instance added.

    Raises:
        ValueError: If no suitable cycle is found with sufficient memory.
    """
    all_nodes = list(topology.list_nodes())

    logger.info("finding cycles:")
    cycles = topology.get_cycles()
    singleton_cycles = [[node] for node in all_nodes]
    candidate_cycles = list(
        filter(lambda it: len(it) >= command.min_nodes, cycles + singleton_cycles)
    )
    cycles_with_sufficient_memory = filter_cycles_by_memory(
        candidate_cycles, command.model_meta.storage_size
    )
    if not cycles_with_sufficient_memory:
        raise ValueError("No cycles found with sufficient memory")

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    smallest_tb_cycles = [
        cycle
        for cycle in smallest_cycles
        if topology.get_subgraph_from_nodes(cycle).is_thunderbolt_cycle(cycle)
    ]

    if smallest_tb_cycles != []:
        smallest_cycles = smallest_tb_cycles

    cycles_with_leaf_nodes: list[list[NodeInfo]] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node.node_id) for node in cycle)
    ]

    selected_cycle = max(
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles,
        key=lambda cycle: sum(
            (
                node.node_profile.memory.ram_available
                for node in cycle
                if node.node_profile is not None
            ),
            start=Memory(),
        ),
    )

    shard_assignments = get_shard_assignments(
        command.model_meta, selected_cycle, command.sharding
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle)

    if os.getenv("EXO_USE_MANUAL_PIPELINE", "1") == "1":
        roles = {_role_for_node_info(n) for n in selected_cycle}
        if roles != {"A", "B", "C"}:
            raise ValueError("Manual pipeline requires nodes A, B, and C only")

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    if len(selected_cycle) == 1:
        logger.warning(
            "You have likely selected ibv for a single node instance; falling back to MlxRing"
        )

        command.instance_meta = InstanceMeta.MlxRing

    # TODO: Single node instances
    match command.instance_meta:
        case InstanceMeta.MlxJaccl:
            manual_hybrid = os.getenv("EXO_USE_MANUAL_PIPELINE", "1") == "1"
            ibv_cycle = (
                [n for n in selected_cycle if _role_for_node_info(n) in {"A", "B", "C"}]
                if manual_hybrid
                else selected_cycle
            )
            if manual_hybrid and len(ibv_cycle) != 3:
                raise ValueError("Manual hybrid topology requires A/B/C present for RDMA")

            mlx_ibv_devices = get_mlx_ibv_devices_matrix(
                ibv_cycle,
                cycle_digraph,
            )
            mlx_ibv_coordinators = get_mlx_ibv_coordinators(
                ibv_cycle,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                ibv_devices=mlx_ibv_devices,
                ibv_coordinators=mlx_ibv_coordinators,
            )
        case InstanceMeta.MlxRing:
            hosts: list[Host] = get_hosts_from_subgraph(cycle_digraph)
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts=[
                    Host(
                        ip=host.ip,
                        port=random_ephemeral_port(),
                    )
                    for host in hosts
                ],
            )

    return target_instances


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    """Remove an instance from current placements.

    Args:
        command: DeleteInstance command with instance ID to remove.
        current_instances: Current instance placements.

    Returns:
        New instance mapping with the instance removed.

    Raises:
        ValueError: If the instance ID is not found in current placements.
    """
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
) -> Sequence[Event]:
    """Generate events for transitioning from current to target instance state.

    Compares current and target instance mappings and generates events for:
    - Instances that need to be created (in target but not current)
    - Instances that need to be deleted (in current but not target)

    Args:
        current_instances: Current instance placements.
        target_instances: Target instance placements.

    Returns:
        Sequence of events (InstanceCreated and InstanceDeleted) representing
        the transition from current to target state.
    """
    events: list[Event] = []

    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    for instance_id in current_instances:
        if instance_id not in target_instances:
            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events

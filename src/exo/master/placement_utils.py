import os
from collections.abc import Generator, Mapping

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeNetworkInfo
from exo.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    HybridShardMetadata,
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
) -> list[Cycle]:
    filtered_cycles: list[Cycle] = []
    for cycle in cycles:
        if not all(node in node_memory for node in cycle):
            continue

        total_mem = sum(
            (node_memory[node_id].ram_available for node_id in cycle.node_ids),
            start=Memory(),
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
) -> list[int]:
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )

    # Largest remainder: floor each, then distribute remainder by fractional part
    raw = [f * total_layers for f in memory_fractions]
    result = [int(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
    for i in range(total_layers - sum(result)):
        result[by_remainder[i]] += 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
) -> Memory:
    total_memory = sum(
        (node_memory[node_id].ram_available for node_id in node_ids),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")
    return total_memory


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
) -> list[int]:
    layer_allocations = allocate_layers_proportionally(
        total_layers=model_card.n_layers,
        memory_fractions=[
            node_memory[node_id].ram_available / total_memory for node_id in node_ids
        ],
    )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        available_memory = node_memory[node_id].ram_available
        if required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )

    return layer_allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(model_card, cycle, node_memory)
    else:
        return _get_shard_assignments_for_pure_pipeline(model_card, cycle, node_memory)


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(pipeline_node_ids, node_memory)
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids, node_memory, pipeline_memory, model_card
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory)

    layer_allocations = _allocate_and_validate_layers(
        cycle.node_ids, node_memory, total_memory, model_card
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    draft_node_id: NodeId | None = None,
    draft_model_id: str | None = None,
):
    total_layers = model_card.n_layers
    # Draft node doesn't count toward TP world_size
    tp_nodes = [n for n in cycle if n != draft_node_id]
    world_size = len(tp_nodes)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    # Assign TP shards to non-draft nodes
    for i, node_id in enumerate(tp_nodes):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )
        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    # Assign draft shard to draft node
    if draft_node_id is not None and draft_model_id is not None:
        from exo.shared.types.worker.shards import DraftShardMetadata
        draft_shard = DraftShardMetadata(
            model_card=model_card,
            device_rank=world_size,  # rank after TP nodes
            world_size=world_size + 1,  # total group size including draft
            start_layer=0,
            end_layer=0,
            n_layers=total_layers,
            draft_model_id=draft_model_id,
        )
        runner_id = RunnerId()
        runner_to_shard[runner_id] = draft_shard
        node_to_runner[draft_node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments_for_hybrid_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    """Create shard assignments for hybrid tensor + pipeline parallel execution.

    The top N-1 nodes by available RAM form a TP sub-group (sharing the same
    layers via all-reduce).  The remaining node becomes the PP tail with its
    own disjoint layer range.
    """
    _validate_cycle(cycle)
    world_size = len(cycle)
    if world_size < 3:
        raise ValueError(
            "Hybrid TP+PP requires at least 3 nodes (2 TP + 1 PP)"
        )

    total_memory = _compute_total_memory(cycle.node_ids, node_memory)

    # Sort nodes by RAM descending — top N-1 form TP group, last is PP tail
    sorted_nodes = sorted(
        cycle.node_ids,
        key=lambda nid: node_memory[nid].ram_available,
        reverse=True,
    )
    tp_node_ids = sorted_nodes[:-1]
    pp_node_ids = sorted_nodes[-1:]

    tp_size = len(tp_node_ids)
    pp_size = 2  # always 2 stages: TP group = stage 0, PP tail = stage 1

    # Compute collective memory for each group
    tp_memory = sum(
        (node_memory[nid].ram_available for nid in tp_node_ids),
        start=Memory(),
    )
    pp_memory = sum(
        (node_memory[nid].ram_available for nid in pp_node_ids),
        start=Memory(),
    )

    # Allocate layers proportionally between the two pipeline stages
    stage_layer_counts = allocate_layers_proportionally(
        total_layers=model_card.n_layers,
        memory_fractions=[
            tp_memory / total_memory,
            pp_memory / total_memory,
        ],
    )
    tp_layers = stage_layer_counts[0]
    pp_layers = stage_layer_counts[1]

    # EXO_PP_LAYER_OFFSET: shift layers away from (negative) or toward (positive)
    # the PP tail node. E.g. -2 moves 2 layers from PP to TP.
    pp_offset = int(os.environ.get("EXO_PP_LAYER_OFFSET", "0"))
    if pp_offset != 0:
        new_pp = pp_layers + pp_offset
        if new_pp < 1:
            raise ValueError(
                f"EXO_PP_LAYER_OFFSET={pp_offset} would leave PP tail with "
                f"{new_pp} layers (need at least 1)"
            )
        new_tp = model_card.n_layers - new_pp
        if new_tp < 1:
            raise ValueError(
                f"EXO_PP_LAYER_OFFSET={pp_offset} would leave TP group with "
                f"{new_tp} layers (need at least 1)"
            )
        logger.info(
            f"EXO_PP_LAYER_OFFSET={pp_offset}: adjusting split from "
            f"{tp_layers}/{pp_layers} to {new_tp}/{new_pp}"
        )
        tp_layers = new_tp
        pp_layers = new_pp

    # TP group: layers [0, tp_layers), PP tail: layers [tp_layers, total)
    tp_start, tp_end = 0, tp_layers
    pp_start, pp_end = tp_layers, tp_layers + pp_layers

    # Validate memory for PP tail (TP nodes share weights so per-node cost is lower)
    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    pp_required = (total_storage * pp_layers) // total_layers
    pp_available = node_memory[pp_node_ids[0]].ram_available
    if pp_required > pp_available:
        raise ValueError(
            f"PP tail node has insufficient memory: "
            f"requires {pp_required.in_gb:.2f} GB for {pp_layers} layers, "
            f"but only has {pp_available.in_gb:.2f} GB available"
        )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    # Assign device_rank: TP nodes first (rank 0..tp_size-1), PP tail last
    # TP-master is device_rank=0
    tp_master_rank = 0
    pp_tail_rank = tp_size  # device_rank for the PP tail node

    for tp_rank, node_id in enumerate(tp_node_ids):
        device_rank = tp_rank
        shard = HybridShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=tp_start,
            end_layer=tp_end,
            n_layers=model_card.n_layers,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_rank=0,
            pp_size=pp_size,
            # Only TP-master sends to PP tail
            pipeline_send_to=pp_tail_rank if tp_rank == 0 else None,
            pipeline_recv_from=None,
        )
        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    # Check if the PP tail should be a draft provider
    draft_model_id = os.environ.get("EXO_DRAFT_MODEL", "") or None

    for i, node_id in enumerate(pp_node_ids):
        device_rank = pp_tail_rank + i

        if draft_model_id:
            # Draft mode: PP tail loads draft model for speculative decoding.
            # IMPORTANT: TP nodes must NOT have pipeline_send_to/recv_from set,
            # because that would wrap their layers with HybridPipelineLastLayer
            # which calls all_gather on the full group — the draft node won't
            # participate in those collectives → deadlock.
            # Draft token exchange is handled separately by RDMADraftClient/
            # draft_provider_loop using point-to-point send/recv.
            shard = HybridShardMetadata(
                model_card=model_card,
                device_rank=device_rank,
                world_size=world_size,
                start_layer=0,
                end_layer=0,
                n_layers=model_card.n_layers,
                tp_size=tp_size,
                tp_rank=-1,
                pp_rank=1,
                pp_size=pp_size,
                pipeline_send_to=tp_master_rank,
                pipeline_recv_from=tp_master_rank,
                draft_model_id=draft_model_id,
            )
            # TP nodes get all layers, NO pipeline send/recv (pure TP)
            for rid, existing_shard in runner_to_shard.items():
                if isinstance(existing_shard, HybridShardMetadata) and existing_shard.tp_rank >= 0:
                    runner_to_shard[rid] = HybridShardMetadata(
                        model_card=model_card,
                        device_rank=existing_shard.device_rank,
                        world_size=world_size,
                        start_layer=0,
                        end_layer=model_card.n_layers,
                        n_layers=model_card.n_layers,
                        tp_size=tp_size,
                        tp_rank=existing_shard.tp_rank,
                        pp_rank=0,
                        pp_size=pp_size,
                        pipeline_send_to=None,  # NO pipeline wrapping
                        pipeline_recv_from=None,  # draft exchange is separate
                        draft_rank=pp_tail_rank,  # tells TP master where draft node is
                    )
        else:
            shard = HybridShardMetadata(
                model_card=model_card,
                device_rank=device_rank,
                world_size=world_size,
                start_layer=pp_start,
                end_layer=pp_end,
                n_layers=model_card.n_layers,
                tp_size=tp_size,
                tp_rank=-1,
                pp_rank=1,
                pp_size=pp_size,
                pipeline_send_to=None,
                pipeline_recv_from=tp_master_rank,
            )
        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )
        case Sharding.Hybrid:
            return get_shard_assignments_for_hybrid_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection.sink_multiaddr.ip_address


def _find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Priority: ethernet > wifi > unknown > thunderbolt
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }

    # Ring should prioritise fastest connection. As a best-effort, we prioritise TB.
    # TODO: Profile and get actual connection speeds.
    if ring:
        priority = {
            "thunderbolt": 0,
            "maybe_ethernet": 1,
            "ethernet": 2,
            "wifi": 3,
            "unknown": 4,
        }

    # RDMA prefers ethernet coordinator
    else:
        priority = {
            "ethernet": 0,
            "wifi": 1,
            "unknown": 2,
            "maybe_ethernet": 3,
            "thunderbolt": 4,
        }
    return min(ips, key=lambda ip: priority.get(ip_to_type.get(ip, "unknown"), 2))


def get_mlx_ring_hosts_by_node(
    nodes_by_rank: list[NodeId],
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    ``nodes_by_rank`` defines the ring order (index == device_rank == MLX_RANK).
    The ring backend binds on ``nodes[rank]`` and connects to
    ``nodes[(rank+1) % size]``, so the list must be ordered by device_rank.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(nodes_by_rank)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(nodes_by_rank):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(nodes_by_rank):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = _find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = _find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }

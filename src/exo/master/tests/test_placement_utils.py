import pytest

from exo.master.placement_utils import (
    allocate_layers_proportionally,
    filter_cycles_by_memory,
    get_mlx_jaccl_coordinators,
    get_shard_assignments,
    get_shard_assignments_for_hybrid_parallel,
    get_shard_assignments_for_pipeline_parallel,
    get_smallest_cycles,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    HybridShardMetadata,
    PipelineShardMetadata,
    Sharding,
)


def test_filter_cycles_by_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(1000 * 1024)
    node2_mem = create_node_memory(1000 * 1024)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]
    assert len(cycles) == 1
    assert len(cycles[0]) == 2

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, node_memory, Memory.from_bytes(1))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 2
    assert set(n for n in filtered_cycles[0]) == {node1_id, node2_id}


def test_filter_cycles_by_insufficient_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(1000 * 1024)
    node2_mem = create_node_memory(1000 * 1024)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    # act
    filtered_cycles = filter_cycles_by_memory(
        topology.get_cycles(), node_memory, Memory.from_kb(2001)
    )

    # assert
    assert len(filtered_cycles) == 0


def test_filter_multiple_cycles_by_memory():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    node_a_mem = create_node_memory(500 * 1024)
    node_b_mem = create_node_memory(500 * 1024)
    node_c_mem = create_node_memory(1000 * 1024)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = topology.get_cycles()

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, node_memory, Memory.from_kb(1500))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 3
    assert set(n for n in filtered_cycles[0]) == {
        node_a_id,
        node_b_id,
        node_c_id,
    }


def test_get_smallest_cycles():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]  # ignore singletons

    # act
    smallest_cycles = get_smallest_cycles(cycles)

    # assert
    assert len(smallest_cycles) == 1
    assert len(smallest_cycles[0]) == 2
    assert set(n for n in smallest_cycles[0]) == {node_a_id, node_b_id}


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 518, 1024), 12, (2, 3, 7)),
        # Edge case: one node has ~90% of memory - should not over-allocate.
        # Each node must have enough memory for at least 1 layer (50 KB = 1000/20).
        ((900, 50, 50), 20, (18, 1, 1)),
    ],
)
def test_get_shard_assignments(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # create connections (A -> B -> C -> A forms a 3-cycle, plus B -> A also exists)
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(4)
    )

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    node_a_mem = create_node_memory(available_memory[0] * 1024)
    node_b_mem = create_node_memory(available_memory[1] * 1024)
    node_c_mem = create_node_memory(available_memory[2] * 1024)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=total_layers,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )

    cycles = topology.get_cycles()

    # pick the 3-node cycle deterministically (cycle ordering can vary)
    selected_cycle = next(cycle for cycle in cycles if len(cycle) == 3)

    # act
    shard_assignments, _ = get_shard_assignments(
        model_card, selected_cycle, Sharding.Pipeline, node_memory=node_memory
    )

    # assert
    runner_id_a = shard_assignments.node_to_runner[node_a_id]
    runner_id_b = shard_assignments.node_to_runner[node_b_id]
    runner_id_c = shard_assignments.node_to_runner[node_c_id]

    assert (
        shard_assignments.runner_to_shard[runner_id_a].end_layer
        - shard_assignments.runner_to_shard[runner_id_a].start_layer
        == expected_layers[0]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_b].end_layer
        - shard_assignments.runner_to_shard[runner_id_b].start_layer
        == expected_layers[1]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_c].end_layer
        - shard_assignments.runner_to_shard[runner_id_c].start_layer
        == expected_layers[2]
    )


def test_get_mlx_jaccl_coordinators():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(5)
    )
    conn_a_c = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(6)
    )

    network_a = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.5"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.2"),
        ]
    )
    network_b = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.1"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.4"),
        ]
    )
    network_c = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.3"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.6"),
        ]
    )
    node_network = {
        node_a_id: network_a,
        node_b_id: network_b,
        node_c_id: network_c,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_a)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_a_c)

    # act
    coordinators = get_mlx_jaccl_coordinators(
        node_a_id,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    # assert
    assert len(coordinators) == 3
    assert node_a_id in coordinators
    assert node_b_id in coordinators
    assert node_c_id in coordinators

    # All coordinators should have IP:PORT format
    for node_id, coordinator in coordinators.items():
        assert ":" in coordinator, (
            f"Coordinator for {node_id} should have ':' separator"
        )

    # Verify port is correct
    for node_id, coordinator in coordinators.items():
        assert coordinator.endswith(":5000"), (
            f"Coordinator for {node_id} should use port 5000"
        )

    # Rank 0 (node_a) treats this as the listen socket so should listen on all IPs
    assert coordinators[node_a_id].startswith("0.0.0.0:"), (
        "Rank 0 node should use 0.0.0.0 as coordinator listen address"
    )

    # Non-rank-0 nodes should use the specific IP from their connection to rank 0
    # node_b uses the IP from conn_b_a (node_b -> node_a)
    assert isinstance(conn_b_a.edge, SocketConnection)
    assert (
        coordinators[node_b_id] == f"{conn_b_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_b should use the IP from conn_b_a"

    # node_c uses the IP from conn_c_a (node_c -> node_a)
    assert isinstance(conn_c_a.edge, SocketConnection)
    assert coordinators[node_c_id] == (
        f"{conn_c_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_c should use the IP from conn_c_a"


class TestAllocateLayersProportionally:
    def test_empty_node_list_raises(self):
        with pytest.raises(ValueError, match="empty node list"):
            allocate_layers_proportionally(total_layers=10, memory_fractions=[])

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=0, memory_fractions=[0.5, 0.5])

    def test_negative_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=-1, memory_fractions=[0.5, 0.5])

    def test_fewer_layers_than_nodes_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(
                total_layers=2, memory_fractions=[0.33, 0.33, 0.34]
            )

    def test_equal_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.25, 0.25]
        )
        assert result == [3, 3, 3, 3]
        assert sum(result) == 12

    def test_proportional_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.50]
        )
        assert result == [3, 3, 6]
        assert sum(result) == 12

    def test_extreme_imbalance_ensures_minimum(self):
        result = allocate_layers_proportionally(
            total_layers=20, memory_fractions=[0.975, 0.0125, 0.0125]
        )
        assert all(layers >= 1 for layers in result)
        assert sum(result) == 20
        # Small nodes get minimum 1 layer
        assert result == [18, 1, 1]

    def test_single_node_gets_all_layers(self):
        result = allocate_layers_proportionally(total_layers=10, memory_fractions=[1.0])
        assert result == [10]

    def test_minimum_viable_allocation(self):
        result = allocate_layers_proportionally(
            total_layers=3, memory_fractions=[0.33, 0.33, 0.34]
        )
        assert result == [1, 1, 1]
        assert sum(result) == 3


def test_get_shard_assignments_insufficient_memory_raises():
    """Test that ValueError is raised when a node has insufficient memory for its layers."""
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    topology = Topology()

    # Node C has only 10 KB but would need 50 KB for 1 layer (1000 KB / 20 layers)
    node_a_mem = create_node_memory(900 * 1024)
    node_b_mem = create_node_memory(50 * 1024)
    node_c_mem = create_node_memory(10 * 1024)  # Insufficient memory

    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_b_a)

    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=20,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]

    with pytest.raises(ValueError, match="insufficient memory"):
        get_shard_assignments(
            model_card, selected_cycle, Sharding.Pipeline, node_memory
        )[0]  # Only need shard_assignments, not the optional cycle


class TestCfgParallelPlacement:
    def _create_ring_topology(self, node_ids: list[NodeId]) -> Topology:
        topology = Topology()
        for node_id in node_ids:
            topology.add_node(node_id)

        for i, node_id in enumerate(node_ids):
            next_node = node_ids[(i + 1) % len(node_ids)]
            conn = Connection(
                source=node_id,
                sink=next_node,
                edge=create_socket_connection(i + 1),
            )
            topology.add_connection(conn)

        return topology

    def test_two_nodes_cfg_model_uses_cfg_parallel(self):
        """Two nodes with CFG model should use CFG parallel (no pipeline)."""
        node_a = NodeId()
        node_b = NodeId()

        topology = self._create_ring_topology([node_a, node_b])
        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        cycle = cycles[0]

        node_memory = {
            node_a: create_node_memory(1000 * 1024),
            node_b: create_node_memory(1000 * 1024),
        }

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 2

        # CFG models should get CfgShardMetadata
        for shard in shards:
            assert isinstance(shard, CfgShardMetadata)
            # Both nodes should have all layers (no pipeline split)
            assert shard.start_layer == 0
            assert shard.end_layer == 60
            assert shard.cfg_world_size == 2
            # Each node is the only stage in its pipeline group
            assert shard.pipeline_world_size == 1
            assert shard.pipeline_rank == 0

        cfg_ranks = sorted(
            s.cfg_rank for s in shards if isinstance(s, CfgShardMetadata)
        )
        assert cfg_ranks == [0, 1]

    def test_four_nodes_cfg_model_uses_hybrid(self):
        """Four nodes with CFG model should use 2 CFG groups x 2 pipeline stages."""
        nodes = [NodeId() for _ in range(4)]

        topology = self._create_ring_topology(nodes)
        cycles = [c for c in topology.get_cycles() if len(c) == 4]
        cycle = cycles[0]

        node_memory = {n: create_node_memory(1000 * 1024) for n in nodes}

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 4

        # CFG models should get CfgShardMetadata
        for shard in shards:
            assert isinstance(shard, CfgShardMetadata)
            assert shard.cfg_world_size == 2
            assert shard.pipeline_world_size == 2
            assert shard.pipeline_rank in [0, 1]

        # Check we have 2 nodes in each CFG group
        cfg_0_shards = [
            s for s in shards if isinstance(s, CfgShardMetadata) and s.cfg_rank == 0
        ]
        cfg_1_shards = [
            s for s in shards if isinstance(s, CfgShardMetadata) and s.cfg_rank == 1
        ]
        assert len(cfg_0_shards) == 2
        assert len(cfg_1_shards) == 2

        # Both CFG groups should have the same layer assignments
        cfg_0_layers = [(s.start_layer, s.end_layer) for s in cfg_0_shards]
        cfg_1_layers = [(s.start_layer, s.end_layer) for s in cfg_1_shards]
        assert sorted(cfg_0_layers) == sorted(cfg_1_layers)

    def test_three_nodes_cfg_model_uses_sequential_cfg(self):
        """Three nodes (odd) with CFG model should use sequential CFG (PipelineShardMetadata)."""
        nodes = [NodeId() for _ in range(3)]

        topology = self._create_ring_topology(nodes)
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {n: create_node_memory(1000 * 1024) for n in nodes}

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 3

        # Odd node count with CFG model falls back to PipelineShardMetadata (sequential CFG)
        for shard in shards:
            assert isinstance(shard, PipelineShardMetadata)

    def test_two_nodes_non_cfg_model_uses_pipeline(self):
        """Two nodes with non-CFG model should use pure pipeline (PipelineShardMetadata)."""
        node_a = NodeId()
        node_b = NodeId()

        topology = self._create_ring_topology([node_a, node_b])
        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        cycle = cycles[0]

        node_memory = {
            node_a: create_node_memory(1000 * 1024),
            node_b: create_node_memory(1000 * 1024),
        }

        model_card = ModelCard(
            model_id=ModelId("flux-test"),
            n_layers=57,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=False,  # Non-CFG model
            tasks=[ModelTask.TextToImage],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 2

        # Non-CFG models should get PipelineShardMetadata
        for shard in shards:
            assert isinstance(shard, PipelineShardMetadata)

        # Should have actual layer sharding (pipeline)
        layer_ranges = sorted(
            (s.start_layer, s.end_layer)
            for s in shards
            if isinstance(s, PipelineShardMetadata)
        )
        # First shard starts at 0, last shard ends at 57
        assert layer_ranges[0][0] == 0
        assert layer_ranges[-1][1] == 57


class TestHybridParallelPlacement:
    """Tests for hybrid tensor + pipeline parallel placement."""

    def _create_ring_topology(self, node_ids: list[NodeId]) -> Topology:
        topology = Topology()
        for node_id in node_ids:
            topology.add_node(node_id)
        for i, node_id in enumerate(node_ids):
            next_node = node_ids[(i + 1) % len(node_ids)]
            conn = Connection(
                source=node_id,
                sink=next_node,
                edge=create_socket_connection(i + 1),
            )
            topology.add_connection(conn)
        return topology

    def _make_model_card(self, n_layers: int = 80) -> ModelCard:
        return ModelCard(
            model_id=ModelId("test-hybrid-model"),
            n_layers=n_layers,
            storage_size=Memory.from_kb(1000),
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        )

    def test_3_node_hybrid_assigns_tp_group_by_memory(self):
        """Top-2 memory nodes form TP group, smallest becomes PP tail."""
        studio1 = NodeId()
        studio2 = NodeId()
        macbook = NodeId()

        topology = self._create_ring_topology([studio1, studio2, macbook])
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {
            studio1: create_node_memory(128_000 * 1024),  # 128 GB
            studio2: create_node_memory(128_000 * 1024),  # 128 GB
            macbook: create_node_memory(64_000 * 1024),   # 64 GB
        }

        assignments, reordered_cycle = get_shard_assignments_for_hybrid_parallel(
            self._make_model_card(80), cycle, node_memory
        )

        shards = {
            nid: assignments.runner_to_shard[rid]
            for nid, rid in assignments.node_to_runner.items()
        }

        # All shards must be HybridShardMetadata
        for shard in shards.values():
            assert isinstance(shard, HybridShardMetadata)

        # Studios should be in the TP group (tp_rank >= 0)
        s1_shard = shards[studio1]
        s2_shard = shards[studio2]
        mb_shard = shards[macbook]
        assert isinstance(s1_shard, HybridShardMetadata)
        assert isinstance(s2_shard, HybridShardMetadata)
        assert isinstance(mb_shard, HybridShardMetadata)

        assert s1_shard.tp_size == 2
        assert s2_shard.tp_size == 2
        assert s1_shard.tp_rank >= 0
        assert s2_shard.tp_rank >= 0
        assert {s1_shard.tp_rank, s2_shard.tp_rank} == {0, 1}

        # MacBook is the PP tail (not in TP group)
        assert mb_shard.tp_size == 0
        assert mb_shard.tp_rank == -1

        # TP nodes must have lower ranks than PP nodes
        assert s1_shard.device_rank < mb_shard.device_rank
        assert s2_shard.device_rank < mb_shard.device_rank

        # Reordered cycle should have TP nodes first, PP tail last
        assert reordered_cycle.node_ids[:2] == [studio1, studio2]
        assert reordered_cycle.node_ids[2] == macbook

    def test_3_node_hybrid_layer_ranges_disjoint(self):
        """TP group and PP tail should have non-overlapping, covering layer ranges."""
        studio1 = NodeId()
        studio2 = NodeId()
        macbook = NodeId()

        topology = self._create_ring_topology([studio1, studio2, macbook])
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {
            studio1: create_node_memory(128_000 * 1024),
            studio2: create_node_memory(128_000 * 1024),
            macbook: create_node_memory(64_000 * 1024),
        }

        assignments, _ = get_shard_assignments_for_hybrid_parallel(
            self._make_model_card(80), cycle, node_memory
        )

        shards = {
            nid: assignments.runner_to_shard[rid]
            for nid, rid in assignments.node_to_runner.items()
        }

        s1_shard = shards[studio1]
        s2_shard = shards[studio2]
        mb_shard = shards[macbook]
        assert isinstance(s1_shard, HybridShardMetadata)
        assert isinstance(s2_shard, HybridShardMetadata)
        assert isinstance(mb_shard, HybridShardMetadata)

        # Studios share exact same layer range (TP group)
        assert s1_shard.start_layer == s2_shard.start_layer
        assert s1_shard.end_layer == s2_shard.end_layer
        assert s1_shard.start_layer == 0  # TP group always starts at 0

        # PP tail layers start where TP group ends
        assert mb_shard.start_layer == s1_shard.end_layer
        assert mb_shard.end_layer == 80  # Total layers

    def test_3_node_hybrid_layers_proportional_to_memory(self):
        """TP group (2×128GB) should get ~80% of layers, PP tail (64GB) ~20%."""
        studio1 = NodeId()
        studio2 = NodeId()
        macbook = NodeId()

        topology = self._create_ring_topology([studio1, studio2, macbook])
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {
            studio1: create_node_memory(128_000 * 1024),
            studio2: create_node_memory(128_000 * 1024),
            macbook: create_node_memory(64_000 * 1024),
        }

        assignments, _ = get_shard_assignments_for_hybrid_parallel(
            self._make_model_card(80), cycle, node_memory
        )

        shards = {
            nid: assignments.runner_to_shard[rid]
            for nid, rid in assignments.node_to_runner.items()
        }

        s1_shard = shards[studio1]
        mb_shard = shards[macbook]
        assert isinstance(s1_shard, HybridShardMetadata)
        assert isinstance(mb_shard, HybridShardMetadata)

        tp_layers = s1_shard.end_layer - s1_shard.start_layer
        pp_layers = mb_shard.end_layer - mb_shard.start_layer

        # Total should be 80
        assert tp_layers + pp_layers == 80

        # TP group has 256GB combined, PP tail has 64GB
        # Proportion: 256/320 = 0.8, so TP should get ~64 layers
        assert tp_layers >= 60  # At least ~75%
        assert pp_layers >= 1   # At least 1 layer for tail

    def test_3_node_hybrid_pipeline_communication(self):
        """TP-master sends to PP tail, PP tail receives from TP-master."""
        studio1 = NodeId()
        studio2 = NodeId()
        macbook = NodeId()

        topology = self._create_ring_topology([studio1, studio2, macbook])
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {
            studio1: create_node_memory(128_000 * 1024),
            studio2: create_node_memory(128_000 * 1024),
            macbook: create_node_memory(64_000 * 1024),
        }

        assignments, _ = get_shard_assignments_for_hybrid_parallel(
            self._make_model_card(80), cycle, node_memory
        )

        shards = {
            nid: assignments.runner_to_shard[rid]
            for nid, rid in assignments.node_to_runner.items()
        }

        s1_shard = shards[studio1]
        s2_shard = shards[studio2]
        mb_shard = shards[macbook]
        assert isinstance(s1_shard, HybridShardMetadata)
        assert isinstance(s2_shard, HybridShardMetadata)
        assert isinstance(mb_shard, HybridShardMetadata)

        # Find the TP-master (tp_rank == 0)
        tp_master = s1_shard if s1_shard.tp_rank == 0 else s2_shard
        tp_non_master = s2_shard if tp_master is s1_shard else s1_shard

        # TP-master should send to PP tail
        assert tp_master.pipeline_send_to == mb_shard.device_rank
        assert tp_master.pipeline_recv_from is None

        # TP non-master should NOT send
        assert tp_non_master.pipeline_send_to is None
        assert tp_non_master.pipeline_recv_from is None

        # PP tail should receive from TP-master
        assert mb_shard.pipeline_recv_from == tp_master.device_rank
        assert mb_shard.pipeline_send_to is None

    def test_3_node_hybrid_pp_ranks(self):
        """TP nodes have pp_rank=0, PP tail has pp_rank=1."""
        studio1 = NodeId()
        studio2 = NodeId()
        macbook = NodeId()

        topology = self._create_ring_topology([studio1, studio2, macbook])
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {
            studio1: create_node_memory(128_000 * 1024),
            studio2: create_node_memory(128_000 * 1024),
            macbook: create_node_memory(64_000 * 1024),
        }

        assignments, _ = get_shard_assignments_for_hybrid_parallel(
            self._make_model_card(80), cycle, node_memory
        )

        shards = {
            nid: assignments.runner_to_shard[rid]
            for nid, rid in assignments.node_to_runner.items()
        }

        for nid in [studio1, studio2]:
            shard = shards[nid]
            assert isinstance(shard, HybridShardMetadata)
            assert shard.pp_rank == 0
            assert shard.pp_size == 2

        mb_shard = shards[macbook]
        assert isinstance(mb_shard, HybridShardMetadata)
        assert mb_shard.pp_rank == 1
        assert mb_shard.pp_size == 2

    def test_2_node_raises(self):
        """Hybrid requires at least 3 nodes."""
        a = NodeId()
        b = NodeId()

        topology = self._create_ring_topology([a, b])
        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        cycle = cycles[0]

        node_memory = {
            a: create_node_memory(128_000 * 1024),
            b: create_node_memory(128_000 * 1024),
        }

        with pytest.raises(ValueError, match="at least 3 nodes"):
            get_shard_assignments_for_hybrid_parallel(
                self._make_model_card(80), cycle, node_memory
            )

    def test_dispatch_via_get_shard_assignments(self):
        """Sharding.Hybrid dispatches to the hybrid placement function."""
        studio1 = NodeId()
        studio2 = NodeId()
        macbook = NodeId()

        topology = self._create_ring_topology([studio1, studio2, macbook])
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {
            studio1: create_node_memory(128_000 * 1024),
            studio2: create_node_memory(128_000 * 1024),
            macbook: create_node_memory(64_000 * 1024),
        }

        assignments, reordered_cycle = get_shard_assignments(
            self._make_model_card(80), cycle, Sharding.Hybrid, node_memory
        )

        # Should produce HybridShardMetadata
        for shard in assignments.runner_to_shard.values():
            assert isinstance(shard, HybridShardMetadata)

        # Hybrid mode should return a reordered cycle
        assert reordered_cycle is not None

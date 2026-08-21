import pytest

from exo.master.placement_utils import (
    allocate_layers_proportionally,
    filter_cycles_by_memory,
    find_ip_prioritised,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_shard_assignments,
    get_shard_assignments_for_pipeline_parallel,
    get_smallest_cycles,
    is_link_local_ipv4,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.topology import (
    Connection,
    RDMAConnection,
    SocketConnection,
)
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
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
        backends=[Backend.MlxMetal],
    )

    cycles = topology.get_cycles()

    # pick the 3-node cycle deterministically (cycle ordering can vary)
    selected_cycle = next(cycle for cycle in cycles if len(cycle) == 3)

    # act
    shard_assignments = get_shard_assignments(
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
        backends=[Backend.MlxMetal],
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]

    with pytest.raises(ValueError, match="insufficient memory"):
        get_shard_assignments(
            model_card, selected_cycle, Sharding.Pipeline, node_memory
        )


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
            backends=[Backend.MlxMetal],
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
            backends=[Backend.MlxMetal],
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
            backends=[Backend.MlxMetal],
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
            backends=[Backend.MlxMetal],
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


def _rdma_conn(source: NodeId, sink: NodeId, source_iface: str, sink_iface: str):
    """A single directed RDMA edge, i.e. one end's view of one cable."""
    return Connection(
        source=source,
        sink=sink,
        edge=RDMAConnection(source_rdma_iface=source_iface, sink_rdma_iface=sink_iface),
    )


def test_jaccl_matrix_picks_the_same_cable_from_both_ends():
    """Design doc Section 116: with TWO cables between a pair of nodes, both
    directions must select the SAME physical link.

    The two ends do NOT share a device name (measured on the real cluster:
    node1 rdma_en3 <-> node2 rdma_en4), so selection cannot rely on names
    matching. It must key on something direction-independent.

    NEGATIVE CONTROL: the edges are deliberately inserted in OPPOSITE orders
    for the two directions. The old implementation took the first RDMA edge it
    found, so it picked cable A from node0's side and cable B from node1's --
    pairing QPs across two different wires, which presents as a transport that
    never connects while every port still reads PORT_ACTIVE. This test fails
    against that implementation and passes with deterministic selection.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    # Cable A: node0 rdma_en3 <-> node1 rdma_en4
    # Cable B: node0 rdma_en4 <-> node1 rdma_en3
    topology = Topology()
    # node0 -> node1 : cable A first
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en4", "rdma_en3"))
    # node1 -> node0 : cable B first (opposite insertion order on purpose)
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))

    matrix = get_mlx_jaccl_devices_matrix([node0, node1], topology)

    # Each node names its OWN local iface for reaching the peer.
    node0_iface = matrix[0][1]
    node1_iface = matrix[1][0]
    assert node0_iface is not None
    assert node1_iface is not None

    # The chosen pair must be the two ends of ONE cable. Cable A is
    # (en3, en4) and cable B is (en4, en3); if the two ends disagreed we would
    # get (en3, en3) or (en4, en4) -- both of which are NOT a real cable.
    assert {node0_iface, node1_iface} == {"rdma_en3", "rdma_en4"}, (
        f"ends disagreed on which cable to use: node0={node0_iface} node1={node1_iface}"
    )


def test_jaccl_matrix_still_works_with_a_single_link():
    """Regression guard: the one-cable case must be unchanged."""
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))

    matrix = get_mlx_jaccl_devices_matrix([node0, node1], topology)
    assert matrix[0][1] == "rdma_en3"
    assert matrix[1][0] == "rdma_en4"
    assert matrix[0][0] is None
    assert matrix[1][1] is None


# ---------------------------------------------------------------------------
# Dual-cable topology split: jaccl's TCP coordinator must resolve onto a
# DIFFERENT physical cable than the one _select_rdma_cable reserved for RDMA.
# ---------------------------------------------------------------------------


def _dual_cable_topology(node0: NodeId, node1: NodeId) -> Topology:
    """Two Thunderbolt cables between the same pair of nodes.

    Cable A: node0 rdma_en3 <-> node1 rdma_en4
    Cable B: node0 rdma_en4 <-> node1 rdma_en3
    Both directions are present for both cables, as the real profiler emits.
    """
    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en4", "rdma_en3"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en3", "rdma_en4"))
    return topology


def _tb_network(**iface_ips: str) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name=name, ip_address=ip, interface_type="thunderbolt")
            for name, ip in iface_ips.items()
        ]
    )


def test_jaccl_coordinator_avoids_the_rdma_reserved_cable():
    """The whole point of the split: with two cables, the TCP coordinator
    must NOT land on the interface RDMA claimed.

    NEGATIVE CONTROL: before this wiring, ``_find_connection_ip`` took the
    first RDMA edge it saw and yielded that interface's IP unconditionally --
    which is the same cable ``get_mlx_jaccl_devices_matrix`` selects, so
    coordinator TCP traffic shared the RDMA wire. This test pins the
    coordinator IP to the OTHER interface, so that behaviour fails here.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")
    topology = _dual_cable_topology(node0, node1)

    node_network = {
        node0: _tb_network(rdma_en3="10.0.3.10", rdma_en4="10.0.4.10"),
        node1: _tb_network(rdma_en3="10.0.4.11", rdma_en4="10.0.3.11"),
    }

    # node0 is rank 0 / the coordinator.
    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    assert coordinators[node0] == "0.0.0.0:5000"

    # Which cable did RDMA take? matrix[1][0] is node1's local iface for
    # reaching node0; the coordinator-side end of that same cable is what
    # must be excluded.
    matrix = get_mlx_jaccl_devices_matrix([node0, node1], topology)
    rdma_iface_on_coordinator = matrix[0][1]
    assert rdma_iface_on_coordinator is not None
    rdma_ip_on_coordinator = {
        iface.name: iface.ip_address for iface in node_network[node0].interfaces
    }[rdma_iface_on_coordinator]

    coordinator_ip = coordinators[node1].rsplit(":", 1)[0]
    assert coordinator_ip != rdma_ip_on_coordinator, (
        "jaccl TCP coordinator resolved onto the SAME cable reserved for "
        f"RDMA ({rdma_iface_on_coordinator} @ {rdma_ip_on_coordinator}); the "
        "dual-cable split is not wired through"
    )
    # And it must be a real address on the other cable, not junk.
    assert coordinator_ip in {"10.0.3.10", "10.0.4.10"}
    assert coordinators[node1].endswith(":5000")


def test_jaccl_coordinator_prefers_free_cable_over_lan():
    """CABLE-BEATS-LAN. With two cables, the free (non-RDMA) Thunderbolt
    cable must win over the home LAN for the TCP side channel.

    This inverts the pre-fix expectation: the ``ring=False`` table ranked
    ethernet first, so the coordinator landed on 192.168.86.202 even though
    a dedicated, idle, switch-free cable existed. NEGATIVE CONTROL: drop
    ``prefer_thunderbolt=True`` from ``get_mlx_jaccl_coordinators`` and this
    test returns the LAN address and fails.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")
    topology = _dual_cable_topology(node0, node1)
    topology.add_connection(
        Connection(
            source=node1,
            sink=node0,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.86.202/tcp/52415")
            ),
        )
    )

    node_network = {
        node0: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.86.202", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="rdma_en3",
                    ip_address="10.0.3.10",
                    interface_type="thunderbolt",
                ),
                NetworkInterfaceInfo(
                    name="rdma_en4",
                    ip_address="10.0.4.10",
                    interface_type="thunderbolt",
                ),
            ]
        ),
        node1: _tb_network(rdma_en3="10.0.4.11", rdma_en4="10.0.3.11"),
    }

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    matrix = get_mlx_jaccl_devices_matrix([node0, node1], topology)
    rdma_iface_on_coordinator = matrix[0][1]
    iface_to_ip = {
        iface.name: iface.ip_address for iface in node_network[node0].interfaces
    }
    chosen = coordinators[node1].rsplit(":", 1)[0]

    assert chosen != "192.168.86.202", (
        "jaccl TCP coordinator took the shared home LAN while a dedicated "
        "non-RDMA Thunderbolt cable was available"
    )
    assert chosen != iface_to_ip[str(rdma_iface_on_coordinator)]
    assert chosen in {"10.0.3.10", "10.0.4.10"}


def test_jaccl_coordinator_prefers_maybe_ethernet_cable_over_lan():
    """macOS types Thunderbolt enX devices as ``maybe_ethernet`` (any enX
    other than en0/en1 gets re-tagged by
    ``_get_interface_types_from_networksetup``). The real cluster therefore
    presents its TB cables as ``maybe_ethernet``, NOT ``thunderbolt`` -- so
    the preference must beat ``ethernet`` for that type too, or the fix is
    inert on the actual hardware.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")
    topology = _dual_cable_topology(node0, node1)
    topology.add_connection(
        Connection(
            source=node1,
            sink=node0,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.86.202/tcp/52415")
            ),
        )
    )

    node_network = {
        node0: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.86.202", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="rdma_en3",
                    ip_address="10.0.3.10",
                    interface_type="maybe_ethernet",
                ),
                NetworkInterfaceInfo(
                    name="rdma_en4",
                    ip_address="10.0.4.10",
                    interface_type="maybe_ethernet",
                ),
            ]
        ),
        node1: _tb_network(rdma_en3="10.0.4.11", rdma_en4="10.0.3.11"),
    }

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )
    assert coordinators[node1].rsplit(":", 1)[0] in {"10.0.3.10", "10.0.4.10"}


def test_jaccl_coordinator_single_cable_prefers_lan_over_sharing_rdma():
    """One cable + a home LAN: the LAN is correct here. Preferring
    Thunderbolt must NOT re-select the RDMA-reserved wire -- the exclusion
    outranks the preference.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))
    topology.add_connection(
        Connection(
            source=node1,
            sink=node0,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.86.202/tcp/52415")
            ),
        )
    )

    node_network = {
        node0: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.86.202", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="rdma_en3",
                    ip_address="10.0.3.10",
                    interface_type="maybe_ethernet",
                ),
            ]
        ),
        node1: _tb_network(rdma_en4="10.0.3.11"),
    }

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )
    assert coordinators[node1] == "192.168.86.202:5000"


def test_jaccl_coordinator_excludes_rdma_cable_reached_via_socket_edge():
    """The RDMA cable often ALSO carries an IP bridge, so it appears as a
    SocketConnection as well as an RDMAConnection. Excluding only the RDMA
    edge would let the reserved wire back in through its socket edge -- and
    with Thunderbolt now PREFERRED, it would win. The exclusion is therefore
    resolved to IPs and applied to socket edges too.

    NEGATIVE CONTROL: restore ``_find_connection_ip`` to yield every
    SocketConnection unconditionally and this returns 10.0.3.10.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = _dual_cable_topology(node0, node1)
    # Both cables also carry an IP bridge (as macOS TB bridging does).
    for ip in ("10.0.3.10", "10.0.4.10"):
        topology.add_connection(
            Connection(
                source=node1,
                sink=node0,
                edge=SocketConnection(
                    sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415")
                ),
            )
        )

    node_network = {
        node0: _tb_network(rdma_en3="10.0.3.10", rdma_en4="10.0.4.10"),
        node1: _tb_network(rdma_en3="10.0.4.11", rdma_en4="10.0.3.11"),
    }

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    matrix = get_mlx_jaccl_devices_matrix([node0, node1], topology)
    iface_to_ip = {
        iface.name: iface.ip_address for iface in node_network[node0].interfaces
    }
    rdma_ip = iface_to_ip[str(matrix[0][1])]
    assert coordinators[node1].rsplit(":", 1)[0] != rdma_ip


def test_jaccl_coordinator_single_cable_shares_it_rather_than_failing():
    """Regression guard for single-cable hardware (the current cluster's
    normal state): excluding the RDMA cable leaves nothing, so we must fall
    back to sharing it -- NOT raise, and NOT return None.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))

    node_network = {
        node0: _tb_network(rdma_en3="10.0.3.10"),
        node1: _tb_network(rdma_en4="10.0.3.11"),
    }

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )
    assert coordinators[node0] == "0.0.0.0:5000"
    # Only one cable exists; sharing it is correct here.
    assert coordinators[node1] == "10.0.3.10:5000"


def test_jaccl_coordinator_and_rdma_matrix_agree_on_opposite_cables():
    """End-to-end invariant across BOTH producers: for every non-coordinator
    node, the cable jaccl uses for RDMA and the cable it uses for the TCP
    coordinator must be different physical links.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")
    topology = _dual_cable_topology(node0, node1)
    node_network = {
        node0: _tb_network(rdma_en3="10.0.3.10", rdma_en4="10.0.4.10"),
        node1: _tb_network(rdma_en3="10.0.4.11", rdma_en4="10.0.3.11"),
    }

    matrix = get_mlx_jaccl_devices_matrix([node0, node1], topology)
    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    ip_to_iface = {
        iface.ip_address: iface.name for iface in node_network[node0].interfaces
    }
    coordinator_iface = ip_to_iface[coordinators[node1].rsplit(":", 1)[0]]
    assert matrix[0][1] != coordinator_iface


# ---------------------------------------------------------------------------
# Link-local (APIPA, 169.254.0.0/16) demotion.
#
# Measured on the live cluster: a Thunderbolt bridge that comes up physically
# but never negotiates a peer address gets a macOS self-assigned 169.254.x.x
# address. It profiles as a real, fast link, so the Thunderbolt-first table
# picks it -- and jaccl's TCP coordinator then hangs, because nothing on the
# far side answers on that subnet. Reachability must outrank speed.
# ---------------------------------------------------------------------------


def test_link_local_loses_to_routable_lan_even_though_cable_is_faster():
    """The core fix. A link-local Thunderbolt cable must lose to a routable
    home-LAN address, inverting the normal prefer_thunderbolt ordering.

    NEGATIVE CONTROL: drop the ``is_link_local_ipv4`` term from the sort key
    in ``find_ip_prioritised`` and this returns 169.254.x.x (the exact live
    failure) instead of the LAN address.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))
    topology.add_connection(
        Connection(
            source=node1,
            sink=node0,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.86.202/tcp/52415")
            ),
        )
    )

    node_network = {
        node0: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.86.202", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="rdma_en3",
                    ip_address="169.254.212.14",
                    interface_type="maybe_ethernet",
                ),
            ]
        ),
        node1: _tb_network(rdma_en4="169.254.99.7"),
    }

    chosen = find_ip_prioritised(
        node1,
        node0,
        topology,
        node_network,
        ring=False,
        prefer_thunderbolt=True,
    )
    assert chosen == "192.168.86.202", (
        f"link-local address {chosen} beat a routable LAN address; the TCP "
        "coordinator would hang"
    )


def test_link_local_demotion_applies_to_the_ring_table_too():
    """The ring/prefer_thunderbolt table is the one that surfaced the bug, but
    reachability is not a jaccl-specific concern -- ``ring=True`` must demote
    link-local too, or MLX ring hosts inherit the same hang.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))
    topology.add_connection(
        Connection(
            source=node1,
            sink=node0,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.86.202/tcp/52415")
            ),
        )
    )
    node_network = {
        node0: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.86.202", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="rdma_en3",
                    ip_address="169.254.212.14",
                    interface_type="thunderbolt",
                ),
            ]
        ),
        node1: _tb_network(rdma_en4="169.254.99.7"),
    }

    assert (
        find_ip_prioritised(node1, node0, topology, node_network, ring=True)
        == "192.168.86.202"
    )


def test_link_local_still_selected_when_it_is_the_only_candidate():
    """Regression guard: demotion is a RANKING change, not a filter. A cluster
    wired only over an APIPA-addressed bridge must still get an address (and
    still work if that bridge happens to be functional), not a None that
    escalates into ``ValueError``.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))

    node_network = {
        node0: _tb_network(rdma_en3="169.254.212.14"),
        node1: _tb_network(rdma_en4="169.254.99.7"),
    }

    chosen = find_ip_prioritised(
        node1,
        node0,
        topology,
        node_network,
        ring=False,
        prefer_thunderbolt=True,
    )
    assert chosen == "169.254.212.14"


def test_routable_thunderbolt_still_beats_lan_after_the_fix():
    """Regression guard for the PREVIOUS fix (CABLE-BEATS-LAN): demoting
    link-local must not disturb the ordering when every candidate is
    routable. A real 10.x Thunderbolt cable still wins over the home LAN.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")

    topology = Topology()
    topology.add_connection(_rdma_conn(node0, node1, "rdma_en3", "rdma_en4"))
    topology.add_connection(_rdma_conn(node1, node0, "rdma_en4", "rdma_en3"))
    topology.add_connection(
        Connection(
            source=node1,
            sink=node0,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.86.202/tcp/52415")
            ),
        )
    )
    node_network = {
        node0: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.86.202", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="rdma_en3",
                    ip_address="10.0.3.10",
                    interface_type="maybe_ethernet",
                ),
            ]
        ),
        node1: _tb_network(rdma_en4="10.0.3.11"),
    }

    assert (
        find_ip_prioritised(
            node1,
            node0,
            topology,
            node_network,
            ring=False,
            prefer_thunderbolt=True,
        )
        == "10.0.3.10"
    )


def test_jaccl_coordinator_end_to_end_skips_link_local_cable():
    """The real cluster shape: dual cables where the non-RDMA one is APIPA.
    The coordinator must land on the routable LAN rather than the dead cable.
    """
    node0 = NodeId("00000000-0000-0000-0000-000000000000")
    node1 = NodeId("11111111-1111-1111-1111-111111111111")
    topology = _dual_cable_topology(node0, node1)
    topology.add_connection(
        Connection(
            source=node1,
            sink=node0,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.86.202/tcp/52415")
            ),
        )
    )

    node_network = {
        node0: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="192.168.86.202", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="rdma_en3",
                    ip_address="169.254.212.14",
                    interface_type="maybe_ethernet",
                ),
                NetworkInterfaceInfo(
                    name="rdma_en4",
                    ip_address="169.254.44.9",
                    interface_type="maybe_ethernet",
                ),
            ]
        ),
        node1: _tb_network(rdma_en3="169.254.44.11", rdma_en4="169.254.212.11"),
    }

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node0,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )
    assert coordinators[node0] == "0.0.0.0:5000"
    assert coordinators[node1] == "192.168.86.202:5000"


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("169.254.0.1", True),
        ("169.254.255.254", True),
        ("169.254.212.14", True),
        ("169.253.1.1", False),
        ("169.255.1.1", False),
        ("10.0.3.10", False),
        ("192.168.86.202", False),
        ("127.0.0.1", False),
        ("fe80::1", False),  # IPv6 link-local is out of scope for this demotion
        ("not-an-ip", False),
        ("", False),
    ],
)
def testis_link_local_ipv4_boundaries(address: str, expected: bool):
    """Exact /16 boundaries, and no crash on junk input."""
    assert is_link_local_ipv4(address) is expected

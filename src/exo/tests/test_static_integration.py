"""
Integration tests for static 4-node cluster setup.
"""
import pytest

from exo.master.placement import place_instance
from exo.master.placement_utils import get_mlx_ibv_devices_matrix
from exo.shared.static_config import (
    create_static_topology,
    get_static_config,
    get_thunderbolt_ip_for_peer,
)
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


def test_static_topology_creation_and_validation():
    """Test that static topology can be created and validated."""
    topology = create_static_topology()
    
    # Should have 3 worker nodes
    nodes = list(topology.list_nodes())
    assert len(nodes) == 3
    
    # Should have connections
    connections = list(topology.list_connections())
    assert len(connections) > 0
    
    # All nodes should be connected (forming a cycle for pipeline)
    # At minimum, should have connections Rank 0→1, Rank 1→2, Rank 2→0
    config = get_static_config()
    worker_0_id = config.workers[0].node_id
    worker_1_id = config.workers[1].node_id
    worker_2_id = config.workers[2].node_id
    
    # Check for forward pipeline connections
    has_0_to_1 = any(
        c.local_node_id == worker_0_id and c.send_back_node_id == worker_1_id
        for c in connections
    )
    has_1_to_2 = any(
        c.local_node_id == worker_1_id and c.send_back_node_id == worker_2_id
        for c in connections
    )
    
    assert has_0_to_1, "Should have connection Rank 0 → Rank 1"
    assert has_1_to_2, "Should have connection Rank 1 → Rank 2"


def test_placement_with_static_topology():
    """Test that placement works with static topology."""
    from exo.shared.types.worker.instances import InstanceId
    
    # Create static topology with realistic profiles
    config = get_static_config()
    node_profiles = {}
    
    for worker in config.workers:
        node_profiles[worker.node_id] = NodePerformanceProfile(
            model_id="test-model",
            chip_id=worker.chip_id,
            friendly_name=worker.hostname,
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=worker.ram_total_bytes,
                ram_available=int(0.85 * worker.ram_total_bytes),  # 85% available
                swap_total=0,
                swap_available=0,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(),
        )
    
    topology = create_static_topology(node_profiles=node_profiles)
    
    # Create a model
    model_meta = ModelMetadata(
        model_id=ModelId("test-model"),
        pretty_name="Test Model",
        n_layers=50,
        storage_size=Memory.from_bytes(30 * 1024 * 1024 * 1024),  # 30GB
    )
    
    # Place instance
    command = PlaceInstance(
        command_id=CommandId(),
        model_meta=model_meta,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxJaccl,
        min_nodes=3,  # Use all 3 workers
    )
    
    placements = place_instance(
        command=command,
        topology=topology,
        current_instances={},
    )
    
    # Should create exactly one instance
    assert len(placements) == 1
    
    instance_id, instance = next(iter(placements.items()))
    assert isinstance(instance_id, InstanceId)
    
    # Instance should have shard assignments for all 3 nodes
    node_ids = set(instance.shard_assignments.node_to_runner.keys())
    assert len(node_ids) == 3
    
    # All worker node IDs should be in the instance
    expected_node_ids = {worker.node_id for worker in config.workers}
    assert node_ids == expected_node_ids


def test_static_thunderbolt_ip_mapping():
    """Test that static Thunderbolt IP mappings are correct."""
    config = get_static_config()
    worker_0 = config.workers[0]
    worker_1 = config.workers[1]
    worker_2 = config.workers[2]
    
    # Test Rank 0 → Rank 1
    ip = get_thunderbolt_ip_for_peer(worker_0.node_id, worker_1.node_id)
    assert ip == "192.168.202.1"
    
    # Test Rank 1 → Rank 0 (reverse)
    ip = get_thunderbolt_ip_for_peer(worker_1.node_id, worker_0.node_id)
    assert ip == "192.168.202.2"
    
    # Test Rank 0 → Rank 2
    ip = get_thunderbolt_ip_for_peer(worker_0.node_id, worker_2.node_id)
    assert ip == "192.168.203.1"
    
    # Test Rank 2 → Rank 0 (reverse)
    ip = get_thunderbolt_ip_for_peer(worker_2.node_id, worker_0.node_id)
    assert ip == "192.168.203.2"
    
    # Test Rank 1 → Rank 2
    ip = get_thunderbolt_ip_for_peer(worker_1.node_id, worker_2.node_id)
    assert ip == "192.168.205.1"
    
    # Test Rank 2 → Rank 1 (reverse)
    ip = get_thunderbolt_ip_for_peer(worker_2.node_id, worker_1.node_id)
    assert ip == "192.168.205.2"


def test_static_config_consistency():
    """Test that static config values are consistent."""
    config = get_static_config()
    
    # Master should have correct config
    assert config.master.hostname == "adams-macbook-pro-m1"
    assert config.master.tailscale_ip == "100.67.156.10"
    assert config.master.port == 52415
    
    # Workers should have unique ranks
    ranks = {worker.rank for worker in config.workers}
    assert ranks == {0, 1, 2}
    
    # Workers should have unique hostnames
    hostnames = {worker.hostname for worker in config.workers}
    assert len(hostnames) == 3
    
    # Workers should have unique Tailscale IPs
    tailscale_ips = {worker.tailscale_ip for worker in config.workers}
    assert len(tailscale_ips) == 3
    
    # All Thunderbolt IPs should be in correct subnets
    for worker in config.workers:
        for peer_node_id, tb_ip in worker.thunderbolt_ips.items():
            assert tb_ip.startswith("192.168."), f"Invalid Thunderbolt IP: {tb_ip}"


def test_static_topology_cycle_detection():
    """Test that static topology forms a valid cycle for pipeline."""
    topology = create_static_topology()
    
    # Get cycles from topology
    cycles = topology.get_cycles()
    
    # Should have at least one cycle that includes all 3 workers
    config = get_static_config()
    worker_ids = {worker.node_id for worker in config.workers}
    
    # Find a cycle that includes all workers
    valid_cycle = None
    for cycle in cycles:
        cycle_node_ids = {node.node_id for node in cycle}
        if cycle_node_ids == worker_ids and len(cycle) == 3:
            valid_cycle = cycle
            break
    
    assert valid_cycle is not None, "Should have a cycle with all 3 workers"
    assert len(valid_cycle) == 3


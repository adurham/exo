"""
Tests for static_config.py - static 4-node cluster configuration.
"""
import pytest

from exo.shared.static_config import (
    StaticConfig,
    create_static_topology,
    get_current_worker_config,
    get_master_url,
    get_static_config,
    get_thunderbolt_ip_for_peer,
    get_worker_config_by_hostname,
    get_worker_config_by_rank,
    get_worker_config_by_node_id,
)
from exo.shared.types.common import NodeId


def test_get_static_config():
    """Test that get_static_config returns a valid config."""
    config = get_static_config()
    
    assert config is not None
    assert config.master is not None
    assert len(config.workers) == 3
    
    # Check master config
    assert config.master.hostname == "adams-macbook-pro-m1"
    assert config.master.tailscale_ip == "100.67.156.10"
    assert config.master.port == 52415
    
    # Check workers
    assert config.workers[0].rank == 0
    assert config.workers[1].rank == 1
    assert config.workers[2].rank == 2
    
    assert config.workers[0].hostname == "adams-mac-studio-m4"
    assert config.workers[1].hostname == "adams-macbook-pro-m4"
    assert config.workers[2].hostname == "adams-work-macbook-pro-m4"


def test_get_master_url():
    """Test that get_master_url returns correct URL."""
    url = get_master_url()
    assert url == "http://100.67.156.10:52415"


def test_get_worker_config_by_hostname():
    """Test getting worker config by hostname."""
    config = get_static_config()
    
    worker = get_worker_config_by_hostname("adams-mac-studio-m4")
    assert worker is not None
    assert worker.rank == 0
    assert worker.hostname == "adams-mac-studio-m4"
    assert worker.tailscale_ip == "100.93.253.67"
    
    worker = get_worker_config_by_hostname("adams-macbook-pro-m4")
    assert worker is not None
    assert worker.rank == 1
    
    worker = get_worker_config_by_hostname("adams-work-macbook-pro-m4")
    assert worker is not None
    assert worker.rank == 2
    
    # Non-existent hostname
    worker = get_worker_config_by_hostname("nonexistent-hostname")
    assert worker is None


def test_get_worker_config_by_rank():
    """Test getting worker config by MLX rank."""
    worker = get_worker_config_by_rank(0)
    assert worker is not None
    assert worker.rank == 0
    assert worker.hostname == "adams-mac-studio-m4"
    
    worker = get_worker_config_by_rank(1)
    assert worker is not None
    assert worker.rank == 1
    assert worker.hostname == "adams-macbook-pro-m4"
    
    worker = get_worker_config_by_rank(2)
    assert worker is not None
    assert worker.rank == 2
    assert worker.hostname == "adams-work-macbook-pro-m4"
    
    # Invalid rank
    worker = get_worker_config_by_rank(99)
    assert worker is None


def test_get_worker_config_by_node_id():
    """Test getting worker config by node_id."""
    config = get_static_config()
    worker_0_id = config.workers[0].node_id
    
    worker = get_worker_config_by_node_id(worker_0_id)
    assert worker is not None
    assert worker.node_id == worker_0_id
    assert worker.rank == 0
    
    # Invalid node_id
    invalid_id = NodeId("invalid-node-id")
    worker = get_worker_config_by_node_id(invalid_id)
    assert worker is None


def test_get_thunderbolt_ip_for_peer():
    """Test getting Thunderbolt IP for peer connections."""
    config = get_static_config()
    worker_0 = config.workers[0]  # Rank 0
    worker_1 = config.workers[1]  # Rank 1
    worker_2 = config.workers[2]  # Rank 2
    
    # Rank 0 → Rank 1
    ip = get_thunderbolt_ip_for_peer(worker_0.node_id, worker_1.node_id)
    assert ip == "192.168.202.1"
    
    # Rank 1 → Rank 0 (reverse direction)
    ip = get_thunderbolt_ip_for_peer(worker_1.node_id, worker_0.node_id)
    assert ip == "192.168.202.2"
    
    # Rank 0 → Rank 2
    ip = get_thunderbolt_ip_for_peer(worker_0.node_id, worker_2.node_id)
    assert ip == "192.168.203.1"
    
    # Rank 2 → Rank 0 (reverse direction)
    ip = get_thunderbolt_ip_for_peer(worker_2.node_id, worker_0.node_id)
    assert ip == "192.168.203.2"
    
    # Rank 1 → Rank 2
    ip = get_thunderbolt_ip_for_peer(worker_1.node_id, worker_2.node_id)
    assert ip == "192.168.205.1"
    
    # Rank 2 → Rank 1 (reverse direction)
    ip = get_thunderbolt_ip_for_peer(worker_2.node_id, worker_1.node_id)
    assert ip == "192.168.205.2"
    
    # Invalid node IDs
    invalid_id = NodeId("invalid")
    ip = get_thunderbolt_ip_for_peer(worker_0.node_id, invalid_id)
    assert ip is None


def test_create_static_topology():
    """Test that create_static_topology creates correct topology."""
    topology = create_static_topology()
    
    # Should have 3 worker nodes
    nodes = list(topology.list_nodes())
    assert len(nodes) == 3
    
    # Check node IDs match static config
    config = get_static_config()
    node_ids = {node.node_id for node in nodes}
    expected_node_ids = {worker.node_id for worker in config.workers}
    assert node_ids == expected_node_ids
    
    # Check connections exist (should have bidirectional connections for RDMA)
    connections = list(topology.list_connections())
    assert len(connections) > 0
    
    # Verify we have connections between all worker pairs
    worker_0_id = config.workers[0].node_id
    worker_1_id = config.workers[1].node_id
    worker_2_id = config.workers[2].node_id
    
    # Check Rank 0 ↔ Rank 1 connection
    connections_0_1 = [
        c for c in connections
        if (c.local_node_id == worker_0_id and c.send_back_node_id == worker_1_id)
        or (c.local_node_id == worker_1_id and c.send_back_node_id == worker_0_id)
    ]
    assert len(connections_0_1) >= 1
    
    # Check Rank 1 ↔ Rank 2 connection
    connections_1_2 = [
        c for c in connections
        if (c.local_node_id == worker_1_id and c.send_back_node_id == worker_2_id)
        or (c.local_node_id == worker_2_id and c.send_back_node_id == worker_1_id)
    ]
    assert len(connections_1_2) >= 1
    
    # Check Rank 0 ↔ Rank 2 connection
    connections_0_2 = [
        c for c in connections
        if (c.local_node_id == worker_0_id and c.send_back_node_id == worker_2_id)
        or (c.local_node_id == worker_2_id and c.send_back_node_id == worker_0_id)
    ]
    assert len(connections_0_2) >= 1


def test_create_static_topology_with_profiles():
    """Test create_static_topology with custom node profiles."""
    from exo.shared.types.profiling import (
        MemoryPerformanceProfile,
        NodePerformanceProfile,
        SystemPerformanceProfile,
    )
    
    config = get_static_config()
    node_profiles = {
        config.workers[0].node_id: NodePerformanceProfile(
            model_id="test-model",
            chip_id="Apple M4",
            friendly_name="test-worker-0",
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=192 * 1024 * 1024 * 1024,
                ram_available=180 * 1024 * 1024 * 1024,
                swap_total=0,
                swap_available=0,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(),
        )
    }
    
    topology = create_static_topology(node_profiles=node_profiles)
    nodes = list(topology.list_nodes())
    
    # Should still have 3 nodes
    assert len(nodes) == 3
    
    # Check that worker 0 has the custom profile
    worker_0_node = next(
        node for node in nodes if node.node_id == config.workers[0].node_id
    )
    assert worker_0_node.node_profile is not None
    assert worker_0_node.node_profile.chip_id == "Apple M4"


def test_get_static_config_is_singleton():
    """Test that get_static_config returns the same instance (singleton)."""
    config1 = get_static_config()
    config2 = get_static_config()
    
    # Should be the same instance
    assert config1 is config2
    
    # Values should be equal
    assert config1.master.node_id == config2.master.node_id
    assert len(config1.workers) == len(config2.workers)


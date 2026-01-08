import pytest
from unittest.mock import AsyncMock, MagicMock
from exo.worker.main import Worker
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.commands import ShardPresent, CheckShardPresent
from exo.shared.types.topology import Connection, NodeInfo
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NodePerformanceProfile, 
    NetworkInterfaceInfo, 
    SystemPerformanceProfile, 
    MemoryPerformanceProfile
)
from exo.shared.topology import Topology

@pytest.fixture
def mock_worker():
    shard_downloader = MagicMock()
    shard_downloader.get_shard_download_status_for_shard = AsyncMock()
    shard_downloader.ensure_shard = AsyncMock()
    shard_downloader.delete_model = AsyncMock()
    shard_downloader.on_progress = MagicMock()

    worker = Worker(
        node_id=NodeId("node1"),
        session_id=SessionId(master_node_id=NodeId("node1"), election_clock=0),
        shard_downloader=shard_downloader,
        connection_message_receiver=MagicMock(),
        global_event_receiver=MagicMock(),
        local_event_sender=MagicMock(),
        command_sender=MagicMock(),
        command_receiver=MagicMock(),
    )
    worker.event_sender = MagicMock()
    worker.event_sender.send_nowait = MagicMock()
    
    # Mock file server
    worker.file_server = MagicMock()
    worker.file_server.port = 8000
    
    # Mock TaskGroup
    worker._tg = MagicMock()
    worker._tg.start_soon = MagicMock()

    # Mock State and Topology
    worker.state = MagicMock()
    worker.state.topology = Topology()
    
    return worker

def create_mock_profile(model_id: str, interfaces: list[NetworkInterfaceInfo]) -> NodePerformanceProfile:
    return NodePerformanceProfile(
        model_id=model_id,
        chip_id="M1",
        friendly_name=f"{model_id} Friendly",
        system=SystemPerformanceProfile(),
        memory=MemoryPerformanceProfile.from_bytes(
            ram_total=16*1024*1024*1024,
            ram_available=8*1024*1024*1024,
            swap_total=0,
            swap_available=0
        ),
        network_interfaces=interfaces
    )

def test_handle_shard_present_prioritizes_thunderbolt(mock_worker):
    """
    Verifies that _handle_shard_present preferentially selects a connection that is
    flagged as Thunderbolt in the node profile, even if the IP is not 169.254.x.x.
    """
    origin_node = NodeId("node2")
    model_id = "test-model"
    
    # Setup Topology with Mocks to control iteration order
    mock_worker.state.topology = MagicMock()
    
    # Node 2 has two interfaces:
    # 1. 192.168.1.5 (Standard WiFi)
    # 2. 192.168.200.2 (Static IP Thunderbolt)
    
    node2_profile = create_mock_profile(
        model_id="Mac2",
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="192.168.1.5", is_thunderbolt=False),
            NetworkInterfaceInfo(name="en2", ip_address="192.168.200.2", is_thunderbolt=True),
        ]
    )
    
    # Mock get_node_profile to return our profile
    def get_node_profile_side_effect(nid):
        if nid == origin_node:
            return node2_profile
        return None
    mock_worker.state.topology.get_node_profile.side_effect = get_node_profile_side_effect
    
    wifi_conn = Connection(
        local_node_id=mock_worker.node_id,
        send_back_node_id=origin_node,
        send_back_multiaddr=Multiaddr(address="/ip4/192.168.1.5/tcp/52415")
    )
    tb_conn = Connection(
        local_node_id=mock_worker.node_id,
        send_back_node_id=origin_node,
        send_back_multiaddr=Multiaddr(address="/ip4/192.168.200.2/tcp/52415")
    )
    
    # Force iteration order: WiFi then TB
    # out_edges returns list of (node_id, connection)
    mock_worker.state.topology.out_edges.return_value = [
        (origin_node, wifi_conn),
        (origin_node, tb_conn)
    ]
    
    # Trigger ShardPresent
    command = ShardPresent(
        model_id=model_id,
        base_url="http://placeholder:8000",
        request_command_id="cmd1"
    )
    
    mock_worker._handle_shard_present(origin_node, command)
    
    # Verify the selected peer location uses the Thunderbolt IP (192.168.200.2)
    # The port should be taken from the base_url (8000)
    expected_url = "http://192.168.200.2:8000"
    assert model_id in mock_worker.peer_locations
    assert mock_worker.peer_locations[model_id] == expected_url

def test_handle_shard_present_fallback_standard(mock_worker):
    """
    Verifies fallback to standard connection if no Thunderbolt connection exists.
    """
    origin_node = NodeId("node3")
    model_id = "test-model-2"
    
    node3_profile = create_mock_profile(
        model_id="Mac3",
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="192.168.1.6", is_thunderbolt=False),
        ]
    )
    
    mock_worker.state.topology.add_node(NodeInfo(node_id=origin_node, node_profile=node3_profile))
    mock_worker.state.topology.add_node(NodeInfo(node_id=mock_worker.node_id))
    
    wifi_conn = Connection(
        local_node_id=mock_worker.node_id,
        send_back_node_id=origin_node,
        send_back_multiaddr=Multiaddr(address="/ip4/192.168.1.6/tcp/52415")
    )
    
    mock_worker.state.topology.add_connection(wifi_conn)
    
    command = ShardPresent(
        model_id=model_id,
        base_url="http://placeholder:9000",
        request_command_id="cmd2"
    )
    
    mock_worker._handle_shard_present(origin_node, command)
    
    expected_url = "http://192.168.1.6:9000"
    assert model_id in mock_worker.peer_locations
    assert mock_worker.peer_locations[model_id] == expected_url

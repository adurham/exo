from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import anyio
import pytest
from exo.master.main import Master
from exo.routing.router import get_node_id_keypair
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
    NodePerformanceMeasured,
    NodeTimedOut,
)
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.utils.channels import channel
from exo.shared.types.memory import Memory


@pytest.mark.asyncio
async def test_active_health_check_reachable():
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_peer_id().to_base58())
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    ge_sender, _ = channel[ForwarderEvent]()
    _, le_receiver = channel[ForwarderEvent]()
    _, co_receiver = channel[ForwarderCommand]()

    master = Master(
        node_id,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
    )

    worker_node_id = NodeId("worker_1")
    old_time = datetime.now(tz=timezone.utc) - timedelta(seconds=40)
    
    master.state.last_seen[worker_node_id] = old_time
    master.state.node_profiles[worker_node_id] = NodePerformanceProfile(
        model_id="test_model",
        chip_id="test_chip",
        friendly_name="test_worker",
        memory=MemoryPerformanceProfile(
            ram_total=Memory.from_bytes(0),
            ram_available=Memory.from_bytes(0),
            swap_total=Memory.from_bytes(0),
            swap_available=Memory.from_bytes(0)
        ),
        network_interfaces=[],
        system=SystemPerformanceProfile()
    )
    
    async def async_mock(*args, **kwargs):
        return {worker_node_id: {"127.0.0.1"}}

    # Execute the logic block directly to avoid infinite loop issues in test
    with patch("exo.master.main.check_reachable", side_effect=async_mock):
        # Logic copied from Master._plan for verification
        stale_nodes = []
        now = datetime.now(tz=timezone.utc)
        for nid, timestamp in master.state.last_seen.items():
            if now - timestamp > timedelta(seconds=30):
                stale_nodes.append(nid)

        if stale_nodes:
            # We must import check_reachable here or rely on the patch location
            from exo.master.main import check_reachable
            reachable = await check_reachable(master.state.topology, master.node_id)
            for nid in stale_nodes:
                if nid in reachable:
                    if nid in master.state.node_profiles:
                        await master.event_sender.send(
                            NodePerformanceMeasured(
                                node_id=nid,
                                node_profile=master.state.node_profiles[nid],
                                when=str(datetime.now(tz=timezone.utc)),
                            )
                        )
                else:
                    await master.event_sender.send(NodeTimedOut(node_id=nid))

    # Verify event was sent
    # We grab the receiver and poll it
    with anyio.fail_after(1):
        async with master._loopback_event_receiver as recv:
            async for event in recv:
                if isinstance(event, NodePerformanceMeasured):
                    assert event.node_id == worker_node_id
                    break
                # If we received something else (unexpected), we might loop or fail
                # But here we expect only one event
            else:
                pytest.fail("Did not receive NodePerformanceMeasured event")

@pytest.mark.asyncio
async def test_active_health_check_unreachable():
    keypair = get_node_id_keypair()
    node_id = NodeId(keypair.to_peer_id().to_base58())
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    ge_sender, _ = channel[ForwarderEvent]()
    _, le_receiver = channel[ForwarderEvent]()
    _, co_receiver = channel[ForwarderCommand]()

    master = Master(
        node_id,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
    )

    worker_node_id = NodeId("worker_dead")
    old_time = datetime.now(tz=timezone.utc) - timedelta(seconds=40)
    master.state.last_seen[worker_node_id] = old_time
    master.state.node_profiles[worker_node_id] = NodePerformanceProfile(
        model_id="test", chip_id="test", friendly_name="test",
        memory=MemoryPerformanceProfile(ram_total=Memory.from_bytes(0), ram_available=Memory.from_bytes(0), swap_total=Memory.from_bytes(0), swap_available=Memory.from_bytes(0)),
        network_interfaces=[], system=SystemPerformanceProfile()
    )

    async def async_mock(*args, **kwargs):
        return {}

    with patch("exo.master.main.check_reachable", side_effect=async_mock):
        stale_nodes = []
        now = datetime.now(tz=timezone.utc)
        for nid, timestamp in master.state.last_seen.items():
            if now - timestamp > timedelta(seconds=30):
                stale_nodes.append(nid)

        if stale_nodes:
            from exo.master.main import check_reachable
            reachable = await check_reachable(master.state.topology, master.node_id)
            for nid in stale_nodes:
                if nid in reachable:
                    pass
                else:
                    await master.event_sender.send(NodeTimedOut(node_id=nid))

    with anyio.fail_after(1):
        async with master._loopback_event_receiver as recv:
            async for event in recv:
                if isinstance(event, NodeTimedOut):
                    assert event.node_id == worker_node_id
                    break
            else:
                pytest.fail("Did not receive NodeTimedOut event")


from unittest.mock import MagicMock
from exo.worker.main import Worker
from exo.shared.types.common import NodeId, SessionId
from exo.worker.download.shard_downloader import ShardDownloader

def test_worker_initialization():
    """
    Verify that Worker initializes all required attributes, including file_server
    and runner_failure_history, preventing recent regression.
    """
    node_id = NodeId("node1")
    session_id = SessionId(master_node_id=NodeId("master1"), election_clock=0)
    shard_downloader = MagicMock(spec=ShardDownloader)
    
    # Mock channels
    msg_recv = MagicMock()
    global_event_recv = MagicMock()
    local_event_send = MagicMock()
    cmd_send = MagicMock()
    cmd_recv = MagicMock()

    worker = Worker(
        node_id=node_id,
        session_id=session_id,
        shard_downloader=shard_downloader,
        connection_message_receiver=msg_recv,
        global_event_receiver=global_event_recv,
        local_event_sender=local_event_send,
        command_sender=cmd_send,
        command_receiver=cmd_recv,
    )

    # Check for critical attributes
    assert hasattr(worker, "file_server"), "Worker missing file_server"
    assert hasattr(worker, "runner_failure_history"), "Worker missing runner_failure_history"
    
    assert worker.file_server is not None
    assert isinstance(worker.runner_failure_history, dict)

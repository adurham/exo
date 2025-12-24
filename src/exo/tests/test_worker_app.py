"""
Tests for worker_app.py entry point.
"""
import socket
from unittest.mock import MagicMock, patch

import pytest

from exo.shared.static_config import (
    get_current_worker_config,
    get_master_url,
    get_static_config,
    get_worker_config_by_hostname,
)
from exo.shared.types.common import NodeId, SessionId
from exo.worker.main import Worker
from exo.worker.download.impl_shard_downloader import exo_shard_downloader


@patch("exo.worker_app.socket.gethostname")
def test_worker_gets_config_by_hostname(mock_gethostname):
    """Test that worker gets correct config based on hostname."""
    config = get_static_config()
    
    # Test with each worker hostname
    for worker in config.workers:
        mock_gethostname.return_value = worker.hostname
        
        worker_config = get_current_worker_config()
        assert worker_config is not None
        assert worker_config.hostname == worker.hostname
        assert worker_config.rank == worker.rank
        assert worker_config.node_id == worker.node_id


@patch("exo.worker_app.socket.gethostname")
def test_worker_fails_with_invalid_hostname(mock_gethostname):
    """Test that worker fails if hostname doesn't match any worker."""
    mock_gethostname.return_value = "invalid-hostname"
    
    worker_config = get_current_worker_config()
    assert worker_config is None


def test_worker_initializes_with_master_url():
    """Test that Worker can be initialized with master_url."""
    config = get_static_config()
    worker = config.workers[0]  # Use first worker
    
    node_id = worker.node_id
    session_id = SessionId(master_node_id=config.master.node_id, election_clock=0)
    master_url = get_master_url()
    
    # Create Worker with master_url (static setup)
    worker_instance = Worker(
        node_id=node_id,
        session_id=session_id,
        shard_downloader=exo_shard_downloader(),
        connection_message_receiver=None,
        global_event_receiver=None,
        local_event_sender=None,
        command_sender=None,
        master_url=master_url,
    )
    
    assert worker_instance.master_url == master_url
    assert worker_instance.connection_message_receiver is None
    assert worker_instance.global_event_receiver is None
    assert worker_instance.local_event_sender is None
    assert worker_instance.command_sender is None


def test_worker_config_by_hostname():
    """Test get_worker_config_by_hostname function."""
    config = get_static_config()
    
    # Test with each worker
    for worker in config.workers:
        found_worker = get_worker_config_by_hostname(worker.hostname)
        assert found_worker is not None
        assert found_worker.node_id == worker.node_id
        assert found_worker.rank == worker.rank
    
    # Test with invalid hostname
    invalid = get_worker_config_by_hostname("invalid-hostname")
    assert invalid is None


def test_master_url_format():
    """Test that master_url has correct format."""
    url = get_master_url()
    assert url.startswith("http://")
    assert ":52415" in url
    assert "100.67.156.10" in url


"""
Tests for master worker client (push-based state updates).
"""
import pytest

from exo.shared.types.state import State
from exo.shared.topology import Topology
from exo.master.worker_client import WorkerHTTPClient, WorkerClientPool


@pytest.mark.asyncio
async def test_worker_http_client_lifecycle():
    """Test WorkerHTTPClient lifecycle."""
    client = WorkerHTTPClient("http://localhost:8080")
    
    async with client:
        assert client._session is not None
    
    # Session should be closed after context exit
    assert client._session is None or client._session.closed


@pytest.mark.asyncio
async def test_worker_client_pool():
    """Test WorkerClientPool lifecycle."""
    pool = WorkerClientPool()
    
    # Add workers (using non-existent URLs for testing structure)
    await pool.add_worker("worker1", "http://localhost:8081")
    await pool.add_worker("worker2", "http://localhost:8082")
    
    assert len(pool._clients) == 2
    assert "worker1" in pool._clients
    assert "worker2" in pool._clients
    
    # Clean up
    await pool.close_all()
    assert len(pool._clients) == 0


@pytest.mark.asyncio
async def test_worker_client_pool_push_state():
    """Test that WorkerClientPool can push state (will fail to connect but tests structure)."""
    pool = WorkerClientPool()
    
    # Add a worker (won't actually connect in test)
    await pool.add_worker("worker1", "http://localhost:9999")  # Unlikely to be listening
    
    # Create test state
    test_state = State(topology=Topology())
    
    # Try to push (will fail to connect, but tests the code path)
    # This should not raise an exception, just log a warning
    await pool.push_state_to_all(test_state)
    
    await pool.close_all()


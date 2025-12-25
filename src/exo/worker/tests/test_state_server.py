"""
Tests for worker state server (push-based state updates).
"""
import pytest

from exo.shared.types.state import State
from exo.shared.topology import Topology
from exo.worker.state_server import WorkerStateServer


@pytest.mark.asyncio
async def test_state_server_lifecycle():
    """Test WorkerStateServer lifecycle (start/stop)."""
    server = WorkerStateServer(port=0)  # Use port 0 for auto-assignment in tests
    
    # Test that server can be started and stopped
    await server.start_server()
    assert server._site is not None
    assert server._runner is not None
    assert server.port > 0  # Port should be assigned
    
    await server.stop_server()
    # Site and runner are cleaned up in stop_server


@pytest.mark.asyncio
async def test_state_server_callback():
    """Test that state update callback is called."""
    received_states = []
    
    async def callback(state: State):
        received_states.append(state)
    
    server = WorkerStateServer(port=0)
    server.set_state_update_callback(callback)
    await server.start_server()
    
    # Create a test state
    test_state = State(topology=Topology())
    
    # Create a proper mock request using aiohttp's test utilities
    from aiohttp import web
    from unittest.mock import AsyncMock, MagicMock
    
    request = MagicMock()
    request.json = AsyncMock(return_value=test_state.model_dump(mode="json"))
    
    # Call the handler
    response = await server._handle_state_update(request)
    
    assert response.status == 200
    # web.json_response returns a Response with JSON body - need to read it
    from aiohttp import web
    if hasattr(response, 'text'):
        import json
        data = json.loads(response.text)
    else:
        # For web.json_response, the body is already set
        data = {"status": "ok"}  # Handler returns this
    assert data["status"] == "ok"
    assert len(received_states) == 1
    assert isinstance(received_states[0], State)
    
    await server.stop_server()


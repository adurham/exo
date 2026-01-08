import http.client
import pytest
from unittest.mock import MagicMock, patch
from exo.worker.utils.net_profile import check_reachability
from exo.shared.types.common import NodeId

@pytest.mark.asyncio
async def test_check_reachability_handles_exception_gracefully():
    target_ip = "192.168.1.100"
    expected_node_id = NodeId("node-1")
    self_node_id = NodeId("node-2")
    out = {}

    # Mock http.client.HTTPConnection to raise an exception
    with patch("http.client.HTTPConnection") as MockConnection:
        instance = MockConnection.return_value
        # Simulate an exception during request
        instance.request.side_effect = http.client.BadStatusLine("Bad Status Line")
        
        await check_reachability(target_ip, expected_node_id, self_node_id, out)
        
        # Should not raise exception and should not add to out
        assert out == {}
        MockConnection.assert_called_with(target_ip, 52415, timeout=1)

@pytest.mark.asyncio
async def test_check_reachability_success():
    target_ip = "192.168.1.100"
    expected_node_id = NodeId("node-1")
    self_node_id = NodeId("node-2")
    out = {}

    with patch("http.client.HTTPConnection") as MockConnection:
        instance = MockConnection.return_value
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'"node-1"'
        instance.getresponse.return_value = response
        
        await check_reachability(target_ip, expected_node_id, self_node_id, out)
        
        assert "node-1" in out
        assert target_ip in out["node-1"]

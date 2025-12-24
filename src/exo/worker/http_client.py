"""
HTTP client for Worker to communicate with Master in static setup.
"""
import httpx
from loguru import logger

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import Event, ForwarderEvent


class MasterHTTPClient:
    """HTTP client for sending events to Master."""
    
    def __init__(self, master_url: str, node_id: NodeId, session_id: SessionId):
        self.master_url = master_url.rstrip("/")
        self.node_id = node_id
        self.session_id = session_id
        self._client: httpx.AsyncClient | None = None
        self._event_index = 0
        
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            
    async def send_event(self, event: Event) -> None:
        """Send an event to the Master via HTTP."""
        if not self._client:
            raise RuntimeError("HTTP client not initialized. Use as async context manager.")
        
        # Wrap event in ForwarderEvent format
        forwarder_event = ForwarderEvent(
            origin=self.node_id,
            origin_idx=self._event_index,
            session=self.session_id,
            event=event,
        )
        
        try:
            response = await self._client.post(
                f"{self.master_url}/events",
                json=forwarder_event.model_dump(mode="json"),
            )
            response.raise_for_status()
            self._event_index += 1
            logger.debug(f"Sent event {event.__class__.__name__} to master: {response.status_code}")
        except httpx.HTTPError as e:
            logger.error(f"Failed to send event to master: {e}")
            raise


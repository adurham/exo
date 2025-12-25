"""
HTTP client for Worker to communicate with Master in static setup.
"""
import aiohttp
from loguru import logger

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import Event, ForwarderEvent


class MasterHTTPClient:
    """HTTP client for sending events to Master."""
    
    def __init__(self, master_url: str, node_id: NodeId, session_id: SessionId):
        self.master_url = master_url.rstrip("/")
        self.node_id = node_id
        self.session_id = session_id
        self._session: aiohttp.ClientSession | None = None
        self._event_index = 0
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30.0)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            
    async def send_event(self, event: Event) -> None:
        """Send an event to the Master via HTTP."""
        if not self._session:
            raise RuntimeError("HTTP client not initialized. Use as async context manager.")
        
        # Wrap event in ForwarderEvent format
        forwarder_event = ForwarderEvent(
            origin=self.node_id,
            origin_idx=self._event_index,
            session=self.session_id,
            event=event,
        )
        
        try:
            async with self._session.post(
                f"{self.master_url}/events",
                json=forwarder_event.model_dump(mode="json"),
            ) as response:
                response.raise_for_status()
                self._event_index += 1
                logger.debug(f"Sent event {event.__class__.__name__} to master: {response.status}")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to send event to master: {e}")
            raise
    
    async def fetch_state(self) -> dict:
        """Fetch the current state from the Master via HTTP."""
        if not self._session:
            raise RuntimeError("HTTP client not initialized. Use as async context manager.")
        
        try:
            async with self._session.get(f"{self.master_url}/state") as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch state from master: {e}")
            raise


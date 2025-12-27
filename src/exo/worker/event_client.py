"""HTTP client for workers to send events to master."""
import aiohttp
from loguru import logger

from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import Event, ForwarderEvent


class MasterEventClient:
    """HTTP client to POST events from worker to master."""
    
    def __init__(self, master_url: str, node_id: NodeId, session_id: SessionId):
        self.master_url = master_url
        self.node_id = node_id
        self.session_id = session_id
        self._session: aiohttp.ClientSession | None = None
        self._event_counter = 0
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def send_event(self, event: Event) -> None:
        """Send an event to the master via HTTP POST."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use as async context manager.")
        
        try:
            # Create ForwarderEvent with metadata
            forwarder_event = ForwarderEvent(
                origin_idx=self._event_counter,
                origin=self.node_id,
                session=self.session_id,
                event=event,
            )
            self._event_counter += 1
            
            # Serialize using the same approach as state push
            import json
            from datetime import datetime
            
            class DateTimeEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return super().default(obj)
            
            event_dict = forwarder_event.model_dump(by_alias=True)
            json_str = json.dumps(event_dict, cls=DateTimeEncoder)
            
            # POST to master's /events endpoint
            async with self._session.post(
                f"{self.master_url}/events",
                data=json_str,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as response:
                response.raise_for_status()
                logger.debug(
                    f"✓ Sent {event.__class__.__name__} to master "
                    f"(status {response.status})"
                )
        except aiohttp.ClientError as e:
            logger.warning(
                f"✗ Failed to send {event.__class__.__name__} to master: "
                f"{e.__class__.__name__}: {e}"
            )
        except Exception as e:
            logger.error(
                f"✗ Unexpected error sending event to master: "
                f"{e.__class__.__name__}: {e}",
                exc_info=True,
            )

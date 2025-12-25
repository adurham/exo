"""
HTTP client for Master to push state updates to Workers.
"""
import aiohttp
from loguru import logger

from exo.shared.types.state import State


class WorkerHTTPClient:
    """HTTP client for pushing state updates to a Worker."""
    
    def __init__(self, worker_url: str):
        self.worker_url = worker_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=5.0)  # Short timeout for push updates
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def push_state_update(self, state: State) -> None:
        """Push state update to Worker."""
        if not self._session:
            raise RuntimeError("HTTP client not initialized. Use as async context manager.")
        
        try:
            # Serialize state to dict (State uses camelCase aliases)
            state_dict = state.model_dump(mode="json")
            async with self._session.post(
                f"{self.worker_url}/state/update",
                json=state_dict,
            ) as response:
                response.raise_for_status()
                logger.debug(f"Pushed state update to worker {self.worker_url}: {response.status}")
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to push state update to worker {self.worker_url}: {e}")


class WorkerClientPool:
    """Pool of HTTP clients for all workers."""
    
    def __init__(self):
        self._clients: dict[str, WorkerHTTPClient] = {}
        self._sessions: dict[str, aiohttp.ClientSession] = {}
    
    async def add_worker(self, worker_id: str, worker_url: str) -> None:
        """Add a worker to the pool."""
        if worker_id not in self._clients:
            client = WorkerHTTPClient(worker_url)
            await client.__aenter__()
            self._clients[worker_id] = client
    
    async def push_state_to_all(self, state: State) -> None:
        """Push state update to all workers."""
        for worker_id, client in self._clients.items():
            try:
                await client.push_state_update(state)
            except Exception as e:
                logger.warning(f"Failed to push state to worker {worker_id}: {e}")
    
    async def close_all(self) -> None:
        """Close all client connections."""
        for client in self._clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        self._clients.clear()


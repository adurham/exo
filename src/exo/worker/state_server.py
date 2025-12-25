"""
HTTP server for Worker to receive state updates from Master.
"""
import aiohttp
from aiohttp import web
from loguru import logger

from exo.shared.types.state import State


class WorkerStateServer:
    """HTTP server for receiving state updates from Master."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._state_update_callback = None
    
    def set_state_update_callback(self, callback):
        """Set callback to be called when state is updated."""
        self._state_update_callback = callback
    
    async def _handle_state_update(self, request: web.Request) -> web.Response:
        """Handle state update from Master."""
        try:
            state_dict = await request.json()
            # Convert datetime strings to datetime objects
            from datetime import datetime
            if "lastSeen" in state_dict and isinstance(state_dict["lastSeen"], dict):
                for node_id, dt_str in state_dict["lastSeen"].items():
                    if isinstance(dt_str, str):
                        dt_str_clean = dt_str.replace("Z", "+00:00")
                        state_dict["lastSeen"][node_id] = datetime.fromisoformat(dt_str_clean)
            
            master_state = State.model_validate(state_dict)
            
            if self._state_update_callback:
                await self._state_update_callback(master_state)
            
            return web.json_response({"status": "ok"})
        except Exception as e:
            logger.error(f"Error handling state update: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def start_server(self) -> None:
        """Start the HTTP server."""
        self._app = web.Application()
        self._app.router.add_post("/state/update", self._handle_state_update)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        # If port is 0, find an available port
        if self.port == 0:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                self.port = s.getsockname()[1]
        
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await self._site.start()
        logger.info(f"Worker state server started on port {self.port}")
    
    async def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Worker state server stopped")


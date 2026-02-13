import os
import aiohttp
from aiohttp import web
from loguru import logger
from exo.shared.constants import EXO_MODELS_DIR

class FileServer:
    def __init__(self, port: int = 52416):
        self.port = port
        self.app = web.Application()
        self.app.router.add_get("/{tail:.*}", self.handle_request)
        self.runner = None
        self.site = None

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()
        logger.info(f"FileServer started on port {self.port} serving {EXO_MODELS_DIR}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("FileServer stopped")

    async def handle_request(self, request):
        path = request.match_info.get("tail", "")
        if not path:
             return web.Response(status=404, text="File not found")

        file_path = EXO_MODELS_DIR / path
        
        # Security: basic check to ensure we don't serve files outside models dir
        try:
            file_path.relative_to(EXO_MODELS_DIR)
        except ValueError:
             return web.Response(status=403, text="Access denied")
        
        if not file_path.exists():
            return web.Response(status=404, text="File not found")
            
        if not file_path.is_file():
            # Directory listing? For now, no.
            return web.Response(status=403, text="Access denied")

        return web.FileResponse(file_path)

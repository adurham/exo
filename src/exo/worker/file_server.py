from pathlib import Path
from typing import Optional

from aiohttp import web
from loguru import logger

from exo.shared.constants import EXO_MODELS_DIR


class FileServer:
    def __init__(self, node_id: str, port: int = 0):
        self.node_id = node_id
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.port = port
        self.host = "0.0.0.0"
        self._setup_routes()

    def _setup_routes(self):
        # HF API mimicry
        self.app.router.add_get("/node_id", self.get_node_id)
        self.app.router.add_get(
            "/api/models/{org}/{model}/tree/{revision}/{path:.*}", self.list_files
        )
        self.app.router.add_get(
            "/api/models/{org}/{model}/tree/{revision}", self.list_files
        )
        # Resolve/Download
        self.app.router.add_get(
            "/{org}/{model}/resolve/{revision}/{path:.*}", self.serve_file
        )

    async def start(self) -> int:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        # Retrieve the actual port if 0 was specified
        if self.site._server:  # pyright: ignore[reportPrivateUsage]
            # This is a bit hacky to get the port from aiohttp internal server object
            # usually sockets are at self.site._server.sockets
            try:
                socket = self.site._server.sockets[0]  # pyright: ignore[reportPrivateUsage]
                self.port = socket.getsockname()[1]
            except Exception as e:
                logger.error(f"Failed to retrieve port: {e}")

        logger.info(f"FileServer started on {self.host}:{self.port}")
        return self.port

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    def _get_model_path(self, repo_id: str) -> Path:
        # repo_id usually comes as "org/model", on disk it's "org--model"
        return EXO_MODELS_DIR / repo_id.replace("/", "--")

    async def list_files(self, request: web.Request) -> web.Response:
        repo_id = f"{request.match_info['org']}/{request.match_info['model']}"
        path_prefix = request.match_info.get("path", "")

        model_dir = self._get_model_path(repo_id)
        if not model_dir.exists():
            return web.json_response({"error": "Model not found"}, status=404)

        entries = []
        search_dir = model_dir / path_prefix

        if not search_dir.exists():
            return web.json_response([], status=200)

        # Assuming we just serve flat list of the requested dir.

        try:
            for item in search_dir.iterdir():
                rel_path = item.relative_to(model_dir)

                if item.is_dir():
                    entries.append(
                        {"type": "directory", "path": str(rel_path), "size": 0}
                    )
                else:
                    entries.append(
                        {
                            "type": "file",
                            "path": str(rel_path),
                            "size": item.stat().st_size,
                        }
                    )
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response(entries)

    async def serve_file(self, request: web.Request) -> web.StreamResponse:
        repo_id = f"{request.match_info['org']}/{request.match_info['model']}"
        path = request.match_info["path"]

        model_path = self._get_model_path(repo_id)
        file_path = model_path / path

        if not file_path.exists() or not file_path.is_file():
            return web.Response(status=404)

        # Compute simple ETag (size + mtime)
        stat = file_path.stat()
        etag = f'"{stat.st_size}-{stat.st_mtime}"'
        
        headers = {
            "ETag": etag,
            "Content-Length": str(stat.st_size),
            "Content-Type": "application/octet-stream"
        }

        if request.method == "HEAD":
            return web.Response(headers=headers)

        return web.FileResponse(file_path, headers=headers)

    async def get_node_id(self, request: web.Request) -> web.Response:
        return web.Response(text=self.node_id)

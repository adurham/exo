"""Peer-to-peer file transfer service for downloading model files from cluster nodes."""

import asyncio
import hashlib
from pathlib import Path
from collections.abc import Callable
from typing import Literal

import aiofiles
import aiofiles.os as aios
from aiohttp import web
import aiohttp
from loguru import logger
from pydantic import BaseModel, ConfigDict

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.worker.download.download_utils import calc_hash


class FileAvailability(BaseModel):
    """Information about file availability on a peer node."""

    model_config = ConfigDict(frozen=True)

    node_id: NodeId
    has_file: bool
    file_hash: str | None = None
    file_size: int | None = None


class PeerFileService:
    """Service for checking and downloading files from peer nodes."""

    def __init__(self, node_id: NodeId, topology: Topology, port: int = 8080):
        self.node_id = node_id
        self.topology = topology
        self.port = port
        self._server: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start_server(self) -> None:
        """Start the HTTP server to serve files to peers."""
        app = web.Application()
        app.router.add_get("/file/check/{model_id:.*}/{file_path:.*}", self._handle_check_file)
        app.router.add_get("/file/download/{model_id:.*}/{file_path:.*}", self._handle_download_file)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await self._site.start()
        logger.info(f"Peer file service started on port {self.port}")

    async def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("Peer file service stopped")

    async def _handle_check_file(self, request: web.Request) -> web.Response:
        """Handle file existence check request."""
        model_id = request.match_info["model_id"]
        file_path = request.match_info["file_path"]
        
        target_path = EXO_MODELS_DIR / model_id.replace("/", "--") / file_path
        
        if not await aios.path.exists(target_path):
            return web.json_response({
                "has_file": False,
                "file_hash": None,
                "file_size": None,
            })
        
        try:
            file_size = (await aios.stat(target_path)).st_size
            # Calculate hash (use sha256 for consistency with download verification)
            file_hash = await calc_hash(target_path, hash_type="sha256")
            
            return web.json_response({
                "has_file": True,
                "file_hash": file_hash,
                "file_size": file_size,
            })
        except Exception as e:
            logger.error(f"Error checking file {target_path}: {e}")
            return web.json_response({
                "has_file": False,
                "file_hash": None,
                "file_size": None,
            }, status=500)

    async def _handle_download_file(self, request: web.Request) -> web.StreamResponse:
        """Handle file download request."""
        model_id = request.match_info["model_id"]
        file_path = request.match_info["file_path"]
        
        target_path = EXO_MODELS_DIR / model_id.replace("/", "--") / file_path
        
        if not await aios.path.exists(target_path):
            return web.Response(status=404, text="File not found")
        
        response = web.StreamResponse()
        response.headers["Content-Type"] = "application/octet-stream"
        file_size = (await aios.stat(target_path)).st_size
        response.headers["Content-Length"] = str(file_size)
        await response.prepare(request)
        
        try:
            async with aiofiles.open(target_path, "rb") as f:
                while chunk := await f.read(8 * 1024 * 1024):  # 8MB chunks
                    await response.write(chunk)
            await response.write_eof()
            return response
        except Exception as e:
            logger.error(f"Error serving file {target_path}: {e}")
            return web.Response(status=500, text=f"Error serving file: {e}")

    async def check_peer_has_file(
        self, 
        model_id: str, 
        file_path: str,
        prefer_thunderbolt: bool = True
    ) -> FileAvailability | None:
        """Check if any peer node has the file, preferring Thunderbolt connections."""
        # Get all connected nodes
        peers = []
        for conn in self.topology.list_connections():
            if conn.local_node_id == self.node_id:
                peers.append((conn.send_back_node_id, conn.send_back_multiaddr, conn.is_thunderbolt()))
        
        # Sort: Thunderbolt first, then others
        peers.sort(key=lambda x: (not x[2], str(x[0])))
        
        for peer_id, multiaddr, is_thunderbolt in peers:
            try:
                # Extract IP from multiaddr
                ip = multiaddr.ip_address
                
                # Try to connect to peer's file service (assume same port)
                url = f"http://{ip}:{self.port}/file/check/{model_id}/{file_path}"
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("has_file"):
                                return FileAvailability(
                                    node_id=peer_id,
                                    has_file=True,
                                    file_hash=data.get("file_hash"),
                                    file_size=data.get("file_size"),
                                )
            except Exception as e:
                logger.debug(f"Failed to check peer {peer_id} for file {file_path}: {e}")
                continue
        
        return None

    async def download_from_peer(
        self,
        peer_id: NodeId,
        peer_multiaddr: Multiaddr,
        model_id: str,
        file_path: str,
        target_path: Path,
        expected_hash: str | None = None,
        on_progress: Callable[[int, int, bool], None] | None = None,
    ) -> Path:
        """Download a file from a peer node with hash verification."""
        # Extract IP from multiaddr
        ip = peer_multiaddr.ip_address
        
        url = f"http://{ip}:{self.port}/file/download/{model_id}/{file_path}"
        
        partial_path = target_path.with_suffix(target_path.suffix + ".partial")
        await aios.makedirs(partial_path.parent, exist_ok=True)
        
        # Check if we can resume
        resume_byte_pos = 0
        if await aios.path.exists(partial_path):
            resume_byte_pos = (await aios.stat(partial_path)).st_size
        
        headers = {}
        if resume_byte_pos > 0:
            headers["Range"] = f"bytes={resume_byte_pos}-"
        
        n_read = resume_byte_pos
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    raise FileNotFoundError(f"File not found on peer {peer_id}: {url}")
                if response.status not in [200, 206]:
                    raise Exception(f"Failed to download from peer {peer_id}: {response.status}")
                
                async with aiofiles.open(
                    partial_path, "ab" if resume_byte_pos else "wb"
                ) as f:
                    async for chunk in response.content.iter_chunked(8 * 1024 * 1024):
                        n_read += len(chunk)
                        await f.write(chunk)
                        if on_progress:
                            content_length = int(response.headers.get("Content-Length", 0))
                            if response.status == 206:  # Partial content
                                # For range requests, we need to add resume_byte_pos to content_length
                                total_size = content_length + resume_byte_pos
                            else:
                                total_size = content_length
                            on_progress(n_read, total_size, False)
        
        # Verify hash if provided
        if expected_hash:
            actual_hash = await calc_hash(partial_path, hash_type="sha256")
            if actual_hash != expected_hash:
                await aios.remove(partial_path)
                raise Exception(
                    f"Hash mismatch for {file_path}: expected {expected_hash}, got {actual_hash}"
                )
        
        # Move to final location
        await aios.rename(partial_path, target_path)
        if on_progress:
            file_size = (await aios.stat(target_path)).st_size
            on_progress(file_size, file_size, True)
        
        return target_path


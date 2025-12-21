"""Peer-to-peer file transfer service for downloading model files from cluster nodes."""

import asyncio
import socket
from pathlib import Path
from collections.abc import Callable

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
    port: int = 8080  # Port where the peer file service is running
    ip_address: str | None = None  # IP address of the peer (discovered during check)


def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise OSError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")


class PeerFileService:
    """Service for checking and downloading files from peer nodes."""

    def __init__(self, node_id: NodeId, topology: Topology, port: int | None = None):
        self.node_id = node_id
        self.topology = topology
        self._port = port
        self.port: int = 0  # Will be set when server starts
        self._server: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start_server(self) -> None:
        """Start the HTTP server to serve files to peers."""
        # Find an available port
        start_port = self._port if self._port is not None else 8080
        self.port = find_available_port(start_port)
        
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
        # Get all connected nodes - check both directions
        peers = []
        seen_peers = set()
        for conn in self.topology.list_connections():
            peer_id = None
            peer_multiaddr = None
            is_tb = False
            
            # Connection from us to peer
            if conn.local_node_id == self.node_id:
                peer_id = conn.send_back_node_id
                peer_multiaddr = conn.send_back_multiaddr
                is_tb = conn.is_thunderbolt()
            # Connection from peer to us (reverse direction)
            elif conn.send_back_node_id == self.node_id:
                peer_id = conn.local_node_id
                # For reverse connections, try to find the forward connection via out_edges
                for out_peer_id, out_conn in self.topology.out_edges(self.node_id):
                    if out_peer_id == conn.local_node_id:
                        peer_multiaddr = out_conn.send_back_multiaddr
                        is_tb = out_conn.is_thunderbolt()
                        break
            
            if peer_id and peer_id not in seen_peers and peer_multiaddr:
                peers.append((peer_id, peer_multiaddr, is_tb))
                seen_peers.add(peer_id)
        
        # Sort: Thunderbolt first, then others
        peers.sort(key=lambda x: (not x[2], str(x[0])))
        
        for peer_id, multiaddr, is_thunderbolt in peers:
            try:
                # Extract IP from multiaddr
                ip = multiaddr.ip_address
                
                # Try common ports (8080, 8081, 8082, etc.) since each node may use a different port
                # Start with our port, then try nearby ports
                ports_to_try = [self.port] + [p for p in range(8080, 8090) if p != self.port]
                
                for port in ports_to_try:
                    url = f"http://{ip}:{port}/file/check/{model_id}/{file_path}"
                    
                    try:
                        # Increased timeout to 5 seconds for file checks
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                            async with session.get(url) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    if data.get("has_file"):
                                        logger.info(
                                            f"Found {file_path} on peer {peer_id} at {ip}:{port} "
                                            f"(Thunderbolt: {is_thunderbolt})"
                                        )
                                        return FileAvailability(
                                            node_id=peer_id,
                                            has_file=True,
                                            file_hash=data.get("file_hash"),
                                            file_size=data.get("file_size"),
                                            port=port,
                                            ip_address=ip,
                                        )
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        # Try next port
                        logger.debug(f"Failed to check {url}: {e}")
                        continue
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
        peer_port: int | None = None,
    ) -> Path:
        """Download a file from a peer node with hash verification."""
        # Extract IP from multiaddr
        ip = peer_multiaddr.ip_address
        
        # Use provided port or try common ports
        port = peer_port if peer_port is not None else self.port
        url = f"http://{ip}:{port}/file/download/{model_id}/{file_path}"
        
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


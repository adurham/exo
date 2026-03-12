"""Lightweight HTTP file server for peer-to-peer model transfer.

Serves model files from EXO_MODELS_DIR so that peer nodes can download
model weights over the local network (e.g. Thunderbolt) instead of from
HuggingFace.

Listens on EXO_FILE_SERVER_PORT (default 52416).
"""

import asyncio

from aiohttp import web
from loguru import logger

from exo.shared.constants import EXO_FILE_SERVER_PORT, EXO_MODELS_DIR


async def _handle_model_file(request: web.Request) -> web.StreamResponse:
    """Serve a model file: GET /{org}/{model}/{file_path}

    The model_id is always two segments (e.g. mlx-community/MiniMax-M2.5-6bit).
    On disk it's normalized with "--" (e.g. mlx-community--MiniMax-M2.5-6bit).
    Supports Range headers for resumable downloads.
    """
    # Parse the full path: /{org}/{model}/{file_path...}
    full_match = request.match_info["path"]
    parts = full_match.split("/", 2)  # Split into at most 3 parts
    if len(parts) < 3:
        raise web.HTTPNotFound(text=f"Invalid path: {full_match}")

    model_id = f"{parts[0]}/{parts[1]}"  # e.g. "mlx-community/MiniMax-M2.5-6bit"
    file_path = parts[2]  # e.g. "model-00014-of-00038.safetensors"

    # Normalize model_id the same way download_utils does
    normalized = model_id.replace("/", "--")
    full_path = EXO_MODELS_DIR / normalized / file_path

    # Security: ensure we don't escape the models directory
    try:
        full_path = full_path.resolve()
        if not full_path.is_relative_to(EXO_MODELS_DIR.resolve()):
            raise web.HTTPForbidden(text="Path traversal not allowed")
    except (ValueError, OSError):
        raise web.HTTPForbidden(text="Invalid path")

    if not full_path.is_file():
        raise web.HTTPNotFound(text=f"File not found: {model_id}/{file_path}")

    file_size = full_path.stat().st_size

    # Parse Range header for resume support
    range_header = request.headers.get("Range")
    start = 0
    if range_header and range_header.startswith("bytes="):
        range_spec = range_header[6:]
        parts = range_spec.split("-")
        if parts[0]:
            start = int(parts[0])

    if start >= file_size:
        raise web.HTTPRequestRangeNotSatisfiable(
            headers={"Content-Range": f"bytes */{file_size}"}
        )

    remaining = file_size - start
    status = 206 if start > 0 else 200

    response = web.StreamResponse(
        status=status,
        headers={
            "Content-Length": str(remaining),
            "Content-Type": "application/octet-stream",
            "Accept-Ranges": "bytes",
        },
    )

    if start > 0:
        response.headers["Content-Range"] = f"bytes {start}-{file_size - 1}/{file_size}"

    await response.prepare(request)

    chunk_size = 8 * 1024 * 1024  # 8MB chunks, matching download_utils
    with open(full_path, "rb") as f:
        if start > 0:
            f.seek(start)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            await response.write(chunk)

    await response.write_eof()
    return response


async def run_file_server() -> None:
    """Start the model file server on EXO_FILE_SERVER_PORT."""
    app = web.Application()
    app.router.add_get("/{path:.*}", _handle_model_file)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", EXO_FILE_SERVER_PORT)
    try:
        await site.start()
        logger.info(f"Model file server listening on 0.0.0.0:{EXO_FILE_SERVER_PORT}")
        # Keep running until cancelled
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()

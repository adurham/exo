"""
Stub module for routing functionality (removed for static setup).

This module provides minimal stubs for functions that may still be needed by tests.
"""
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock
from loguru import logger
from pydantic import BaseModel

from exo.shared.constants import EXO_NODE_ID_KEYPAIR

if TYPE_CHECKING:
    from exo_pyo3_bindings import Keypair


# Stub for ConnectionMessage (used in Worker but not in static setup)
class ConnectionMessageType(str, Enum):
    """Stub enum for connection message types."""
    Connected = "connected"
    Disconnected = "disconnected"


class ConnectionMessage(BaseModel):
    """Stub class for connection messages (not used in static setup)."""
    connection_type: ConnectionMessageType
    node_id: str = ""
    remote_ipv4: str = ""
    remote_tcp_port: int = 0


def get_node_id_keypair(
    path: str | bytes | Path | None = None,
) -> "Keypair":
    """
    Stub function for get_node_id_keypair - generates a keypair for backward compatibility.
    
    This is a minimal implementation for backward compatibility with tests.
    In the static setup, node IDs come from static_config, not from keypairs.
    """
    if path is None:
        path = EXO_NODE_ID_KEYPAIR
    
    from exo_pyo3_bindings import Keypair
    
    def lock_path(p: str | bytes | Path) -> Path:
        return Path(str(p) + ".lock")
    
    # Operate with cross-process lock to avoid race conditions
    with FileLock(lock_path(path)):
        path_obj = Path(path)
        if path_obj.exists() and path_obj.stat().st_size > 0:
            # Read existing keypair
            with open(path, "rb") as f:
                protobuf_encoded = f.read()
                try:
                    return Keypair.from_protobuf_encoding(protobuf_encoded)
                except ValueError as e:
                    logger.warning(f"Encountered error when trying to get keypair: {e}")
        
        # Create new keypair and persist
        with open(path, "wb") as f:
            keypair = Keypair.generate_ed25519()
            f.write(keypair.to_protobuf_encoding())
            return keypair


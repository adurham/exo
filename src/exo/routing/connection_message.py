"""Connection message types for libp2p connection events.

This module defines types for serializing and deserializing connection
update messages from the libp2p networking layer.
"""

from enum import Enum

from exo_pyo3_bindings import ConnectionUpdate, ConnectionUpdateType

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import CamelCaseModel


class ConnectionMessageType(Enum):
    """Type of connection event.

    Values:
        Connected: New connection established.
        Disconnected: Connection closed.
    """

    Connected = 0
    Disconnected = 1

    @staticmethod
    def from_update_type(update_type: ConnectionUpdateType) -> "ConnectionMessageType":
        """Convert libp2p ConnectionUpdateType to ConnectionMessageType.

        Args:
            update_type: Libp2p connection update type.

        Returns:
            Corresponding ConnectionMessageType.
        """
        match update_type:
            case ConnectionUpdateType.Connected:
                return ConnectionMessageType.Connected
            case ConnectionUpdateType.Disconnected:
                return ConnectionMessageType.Disconnected


class ConnectionMessage(CamelCaseModel):
    """Message describing a connection event.

    Serializable representation of a libp2p connection update for
    distribution through the event system.

    Attributes:
        node_id: Node ID of the peer connected/disconnected.
        connection_type: Type of connection event (Connected/Disconnected).
        remote_ipv4: IPv4 address of the remote peer.
        remote_tcp_port: TCP port of the remote peer.
    """

    node_id: NodeId
    connection_type: ConnectionMessageType
    remote_ipv4: str
    remote_tcp_port: int

    @classmethod
    def from_update(cls, update: ConnectionUpdate) -> "ConnectionMessage":
        """Create ConnectionMessage from libp2p ConnectionUpdate.

        Args:
            update: Libp2p connection update.

        Returns:
            ConnectionMessage instance.
        """
        return cls(
            node_id=NodeId(update.peer_id.to_base58()),
            connection_type=ConnectionMessageType.from_update_type(update.update_type),
            remote_ipv4=update.remote_ipv4,
            remote_tcp_port=update.remote_tcp_port,
        )

"""Topology type definitions for cluster network graph.

This module defines types representing nodes and connections in the
cluster topology graph.
"""

from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import ConnectionProfile, NodePerformanceProfile
from exo.utils.pydantic_ext import CamelCaseModel


class NodeInfo(CamelCaseModel):
    """Information about a node in the topology.

    Attributes:
        node_id: Unique identifier for this node.
        node_profile: Performance profile (optional, may be None if not yet measured).
    """

    node_id: NodeId
    node_profile: NodePerformanceProfile | None = None


class Connection(CamelCaseModel):
    """A network connection between two nodes.

    Represents a directed edge in the topology graph, specifying how
    one node can reach another.

    Attributes:
        local_node_id: Node ID of the source node (this node).
        send_back_node_id: Node ID of the destination node.
        send_back_multiaddr: Multiaddr for reaching the destination node.
        connection_profile: Network performance metrics (optional).
    """

    local_node_id: NodeId
    send_back_node_id: NodeId
    send_back_multiaddr: Multiaddr
    connection_profile: ConnectionProfile | None = None

    def __hash__(self) -> int:
        """Hash connection for use in sets/dicts.

        Returns:
            Hash based on node IDs and multiaddr.
        """
        return hash(
            (
                self.local_node_id,
                self.send_back_node_id,
                self.send_back_multiaddr.address,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Compare two connections for equality.

        Args:
            other: Object to compare with.

        Returns:
            True if connections have same nodes and multiaddr.

        Raises:
            ValueError: If other is not a Connection instance.
        """
        if not isinstance(other, Connection):
            raise ValueError("Cannot compare Connection with non-Connection")
        return (
            self.local_node_id == other.local_node_id
            and self.send_back_node_id == other.send_back_node_id
            and self.send_back_multiaddr == other.send_back_multiaddr
        )

    def is_thunderbolt(self) -> bool:
        """Check if this connection uses Thunderbolt.

        Thunderbolt connections use link-local addresses (169.254.x.x).

        Returns:
            True if connection appears to be Thunderbolt.
        """
        return str(self.send_back_multiaddr.ipv4_address).startswith("169.254")

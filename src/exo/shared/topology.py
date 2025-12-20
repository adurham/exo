"""Topology management for cluster network graph.

This module provides graph-based topology tracking for the EXO cluster,
managing nodes, connections, and their properties. Uses rustworkx for
efficient graph operations.
"""

import contextlib
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from exo.shared.types.common import NodeId
from exo.shared.types.profiling import ConnectionProfile, NodePerformanceProfile
from exo.shared.types.topology import Connection, NodeInfo


class TopologySnapshot(BaseModel):
    """Immutable snapshot of topology state for serialization.

    Used to serialize/deserialize Topology instances, particularly for
    State serialization where Topology is embedded.

    Attributes:
        nodes: List of all nodes in the topology.
        connections: List of all connections between nodes.
    """

    nodes: list[NodeInfo]
    connections: list[Connection]

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Topology:
    """Graph-based representation of cluster network topology.

    Maintains a directed graph of nodes (devices) and connections (network links)
    between them. Supports queries for neighbors, cycles, subgraphs, and
    connection properties like Thunderbolt detection.

    Uses rustworkx internally for graph operations. Provides methods for
    adding/removing nodes and connections, querying relationships, and
    extracting subgraphs for placement algorithms.

    Attributes:
        _graph: Internal rustworkx directed graph.
        _node_id_to_rx_id_map: Mapping from NodeId to rustworkx node index.
        _rx_id_to_node_id_map: Reverse mapping from rustworkx index to NodeId.
        _edge_id_to_rx_id_map: Mapping from Connection to rustworkx edge index.
    """

    def __init__(self) -> None:
        """Initialize an empty topology graph."""
        self._graph: rx.PyDiGraph[NodeInfo, Connection] = rx.PyDiGraph()
        self._node_id_to_rx_id_map: dict[NodeId, int] = dict()
        self._rx_id_to_node_id_map: dict[int, NodeId] = dict()
        self._edge_id_to_rx_id_map: dict[Connection, int] = dict()

    def to_snapshot(self) -> TopologySnapshot:
        """Create an immutable snapshot of the current topology.

        Returns:
            TopologySnapshot containing all nodes and connections.
        """
        return TopologySnapshot(
            nodes=list(self.list_nodes()),
            connections=list(self.list_connections()),
        )

    @classmethod
    def from_snapshot(cls, snapshot: TopologySnapshot) -> "Topology":
        """Reconstruct a Topology from a snapshot.

        Args:
            snapshot: TopologySnapshot to reconstruct from.

        Returns:
            New Topology instance with nodes and connections from snapshot.
        """
        topology = cls()

        for node in snapshot.nodes:
            with contextlib.suppress(ValueError):
                topology.add_node(node)

        for connection in snapshot.connections:
            topology.add_connection(connection)

        return topology

    def add_node(self, node: NodeInfo) -> None:
        """Add a node to the topology.

        If the node already exists, does nothing.

        Args:
            node: NodeInfo to add.
        """
        if node.node_id in self._node_id_to_rx_id_map:
            return
        rx_id = self._graph.add_node(node)
        self._node_id_to_rx_id_map[node.node_id] = rx_id
        self._rx_id_to_node_id_map[rx_id] = node.node_id

    def node_is_leaf(self, node_id: NodeId) -> bool:
        """Check if a node is a leaf (has exactly one neighbor).

        Args:
            node_id: Node ID to check.

        Returns:
            True if node exists and has exactly one neighbor, False otherwise.
        """
        return (
            node_id in self._node_id_to_rx_id_map
            and len(self._graph.neighbors(self._node_id_to_rx_id_map[node_id])) == 1
        )

    def neighbours(self, node_id: NodeId) -> list[NodeId]:
        """Get all neighbor node IDs.

        Args:
            node_id: Node ID to get neighbors for.

        Returns:
            List of neighbor node IDs. Empty list if node doesn't exist.
        """
        return [
            self._rx_id_to_node_id_map[rx_id]
            for rx_id in self._graph.neighbors(self._node_id_to_rx_id_map[node_id])
        ]

    def out_edges(self, node_id: NodeId) -> list[tuple[NodeId, Connection]]:
        """Get all outgoing edges from a node.

        Args:
            node_id: Node ID to get outgoing edges for.

        Returns:
            List of (target_node_id, connection) tuples. Empty list if node
            doesn't exist or has no outgoing edges.
        """
        if node_id not in self._node_id_to_rx_id_map:
            return []
        return [
            (self._rx_id_to_node_id_map[nid], conn)
            for _, nid, conn in self._graph.out_edges(
                self._node_id_to_rx_id_map[node_id]
            )
        ]

    def contains_node(self, node_id: NodeId) -> bool:
        """Check if a node exists in the topology.

        Args:
            node_id: Node ID to check.

        Returns:
            True if node exists, False otherwise.
        """
        return node_id in self._node_id_to_rx_id_map

    def contains_connection(self, connection: Connection) -> bool:
        """Check if a connection exists in the topology.

        Args:
            connection: Connection to check.

        Returns:
            True if connection exists, False otherwise.
        """
        return connection in self._edge_id_to_rx_id_map

    def add_connection(
        self,
        connection: Connection,
    ) -> None:
        """Add a connection between nodes.

        Automatically adds nodes if they don't exist. If the connection
        already exists, does nothing.

        Args:
            connection: Connection to add between local_node_id and
                send_back_node_id.
        """
        if connection.local_node_id not in self._node_id_to_rx_id_map:
            self.add_node(NodeInfo(node_id=connection.local_node_id))
        if connection.send_back_node_id not in self._node_id_to_rx_id_map:
            self.add_node(NodeInfo(node_id=connection.send_back_node_id))

        if connection in self._edge_id_to_rx_id_map:
            return

        src_id = self._node_id_to_rx_id_map[connection.local_node_id]
        sink_id = self._node_id_to_rx_id_map[connection.send_back_node_id]

        rx_id = self._graph.add_edge(src_id, sink_id, connection)
        self._edge_id_to_rx_id_map[connection] = rx_id

    def list_nodes(self) -> Iterable[NodeInfo]:
        """Get an iterable of all nodes in the topology.

        Returns:
            Iterable of NodeInfo instances.
        """
        return (self._graph[i] for i in self._graph.node_indices())

    def list_connections(self) -> Iterable[Connection]:
        """Get an iterable of all connections in the topology.

        Returns:
            Iterable of Connection instances.
        """
        return (connection for _, _, connection in self._graph.weighted_edge_list())

    def get_node_profile(self, node_id: NodeId) -> NodePerformanceProfile | None:
        """Get the performance profile for a node.

        Args:
            node_id: Node ID to get profile for.

        Returns:
            NodePerformanceProfile if node exists, None otherwise.
        """
        try:
            rx_idx = self._node_id_to_rx_id_map[node_id]
            return self._graph.get_node_data(rx_idx).node_profile
        except KeyError:
            return None

    def update_node_profile(
        self, node_id: NodeId, node_profile: NodePerformanceProfile
    ) -> None:
        """Update the performance profile for a node.

        Args:
            node_id: Node ID to update.
            node_profile: New performance profile.
        """
        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph[rx_idx].node_profile = node_profile

    def update_connection_profile(self, connection: Connection) -> None:
        """Update a connection's profile data.

        Args:
            connection: Connection to update (must exist in topology).
        """
        rx_idx = self._edge_id_to_rx_id_map[connection]
        self._graph.update_edge_by_index(rx_idx, connection)

    def get_connection_profile(
        self, connection: Connection
    ) -> ConnectionProfile | None:
        """Get the connection profile for a connection.

        Args:
            connection: Connection to get profile for.

        Returns:
            ConnectionProfile if connection exists, None otherwise.
        """
        try:
            rx_idx = self._edge_id_to_rx_id_map[connection]
            return self._graph.get_edge_data_by_index(rx_idx).connection_profile
        except KeyError:
            return None

    def remove_node(self, node_id: NodeId) -> None:
        """Remove a node and all its connections.

        Removes the node and all connections involving it. If the node
        doesn't exist, does nothing.

        Args:
            node_id: Node ID to remove.
        """
        if node_id not in self._node_id_to_rx_id_map:
            return

        for connection in self.list_connections():
            if (
                connection.local_node_id == node_id
                or connection.send_back_node_id == node_id
            ):
                self.remove_connection(connection)

        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph.remove_node(rx_idx)

        del self._node_id_to_rx_id_map[node_id]
        del self._rx_id_to_node_id_map[rx_idx]

    def remove_connection(self, connection: Connection) -> None:
        """Remove a connection from the topology.

        If the connection doesn't exist, does nothing.

        Args:
            connection: Connection to remove.
        """
        if connection not in self._edge_id_to_rx_id_map:
            return
        rx_idx = self._edge_id_to_rx_id_map[connection]
        self._graph.remove_edge_from_index(rx_idx)
        del self._edge_id_to_rx_id_map[connection]

    def get_cycles(self) -> list[list[NodeInfo]]:
        """Find all simple cycles in the topology.

        Returns:
            List of cycles, where each cycle is a list of NodeInfo instances
            forming a cycle.
        """
        cycle_idxs = rx.simple_cycles(self._graph)
        cycles: list[list[NodeInfo]] = []
        for cycle_idx in cycle_idxs:
            cycle = [self._graph[idx] for idx in cycle_idx]
            cycles.append(cycle)

        return cycles

    def get_cycles_tb(self) -> list[list[NodeInfo]]:
        """Find all cycles using only Thunderbolt connections.

        Creates a subgraph with only Thunderbolt edges and finds cycles in it.

        Returns:
            List of cycles using only Thunderbolt connections.
        """
        tb_edges = [
            (u, v, conn)
            for u, v, conn in self._graph.weighted_edge_list()
            if conn.is_thunderbolt()
        ]

        tb_graph: rx.PyDiGraph[NodeInfo, Connection] = rx.PyDiGraph()
        tb_graph.add_nodes_from(self._graph.nodes())

        for u, v, conn in tb_edges:
            tb_graph.add_edge(u, v, conn)

        cycle_idxs = rx.simple_cycles(tb_graph)
        cycles: list[list[NodeInfo]] = []
        for cycle_idx in cycle_idxs:
            cycle = [tb_graph[idx] for idx in cycle_idx]
            cycles.append(cycle)

        return cycles

    def get_subgraph_from_nodes(self, nodes: list[NodeInfo]) -> "Topology":
        """Extract a subgraph containing only the specified nodes.

        Creates a new Topology containing only the given nodes and connections
        between them.

        Args:
            nodes: List of nodes to include in the subgraph.

        Returns:
            New Topology instance containing only the specified nodes and
            their interconnections.
        """
        node_idxs = [node.node_id for node in nodes]
        rx_idxs = [self._node_id_to_rx_id_map[idx] for idx in node_idxs]
        topology = Topology()
        for rx_idx in rx_idxs:
            topology.add_node(self._graph[rx_idx])
        for connection in self.list_connections():
            if (
                connection.local_node_id in node_idxs
                and connection.send_back_node_id in node_idxs
            ):
                topology.add_connection(connection)
        return topology

    def is_thunderbolt_cycle(self, cycle: list[NodeInfo]) -> bool:
        """Check if a cycle uses only Thunderbolt connections.

        Verifies that all edges between nodes in the cycle are Thunderbolt.

        Args:
            cycle: List of NodeInfo instances forming a cycle.

        Returns:
            True if all connections in the cycle are Thunderbolt, False otherwise.
        """
        node_idxs = [node.node_id for node in cycle]
        rx_idxs = [self._node_id_to_rx_id_map[idx] for idx in node_idxs]
        for rid in rx_idxs:
            for neighbor_rid in self._graph.neighbors(rid):
                if neighbor_rid not in rx_idxs:
                    continue
                has_tb = False
                for edge in self._graph.get_all_edge_data(rid, neighbor_rid):
                    if edge.is_thunderbolt():
                        has_tb = True
                        break
                if not has_tb:
                    return False
        return True

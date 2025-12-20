"""Network reachability checking utilities.

This module provides functions for checking network connectivity between nodes
in the topology.
"""

import socket

from anyio import create_task_group, to_thread

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId


async def check_reachability(
    target_ip: str, target_node_id: NodeId, out: dict[NodeId, set[str]]
) -> None:
    """Check if a target IP is reachable and record it.

    Attempts to connect to target_ip on port 52415 (API port). If successful,
    adds the IP to the out dictionary for the target_node_id.

    Args:
        target_ip: IP address to check.
        target_node_id: Node ID of the target.
        out: Dictionary to update with reachable IPs.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # 1 second timeout
    try:
        result = await to_thread.run_sync(sock.connect_ex, (target_ip, 52415))
    except socket.gaierror:
        # seems to throw on ipv6 loopback. oh well
        # logger.warning(f"invalid {target_ip=}")
        return
    finally:
        sock.close()

    if result == 0:
        if target_node_id not in out:
            out[target_node_id] = set()
        out[target_node_id].add(target_ip)


async def check_reachable(topology: Topology) -> dict[NodeId, set[str]]:
    """Check reachability for all nodes in the topology.

    Checks connectivity from all network interfaces of all nodes in parallel.

    Args:
        topology: Topology containing nodes to check.

    Returns:
        Dictionary mapping node IDs to sets of reachable IP addresses.
    """
    reachable: dict[NodeId, set[str]] = {}
    async with create_task_group() as tg:
        for node in topology.list_nodes():
            if not node.node_profile:
                continue
            for iface in node.node_profile.network_interfaces:
                tg.start_soon(
                    check_reachability, iface.ip_address, node.node_id, reachable
                )

    return reachable

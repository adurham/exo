from __future__ import annotations

"""
Static RDMA-only topology helpers.

Roles:
- A: Mac Studio (RDMA)
- B: MacBook Pro M4 Max (RDMA)
- C: MacBook Pro M4 Pro (RDMA)
"""

from dataclasses import dataclass
from typing import Callable, Literal

from exo.shared.types.common import Host, NodeId

NodeRole = Literal["A", "B", "C"]

# Static point-to-point IPs (Thunderbolt)
STATIC_IPS: dict[NodeRole, dict[NodeRole, str]] = {
    "A": {"B": "192.168.201.1", "C": "192.168.202.1"},
    "B": {"A": "192.168.201.2", "C": "192.168.203.1"},
    "C": {"A": "192.168.202.2", "B": "192.168.203.2"},
}

PREFERRED_SUBNET_PREFIXES: tuple[str, ...] = (
    "192.168.201.",
    "192.168.202.",
    "192.168.203.",
)

MANUAL_PEERS: dict[NodeRole, tuple[NodeRole, ...]] = {
    "A": ("B", "C"),
    "B": ("A", "C"),
    "C": ("A", "B"),
}

PIPELINE_ORDER: tuple[NodeRole, ...] = ("C", "B", "A")
PIPELINE_LAYER_FRACTIONS: dict[NodeRole, float] = {
    "C": 0.2,
    "B": 0.3,
    "A": 0.5,
}


@dataclass(frozen=True)
class RoleDetection:
    role: NodeRole
    has_rdma: bool


def infer_role_from_ips(local_ips: set[str]) -> NodeRole | None:
    for role, peers in STATIC_IPS.items():
        if any(ip in local_ips for ip in peers.values()):
            return role
    return None


def should_disable_autodiscovery(ip: str) -> bool:
    return any(ip.startswith(prefix) for prefix in PREFERRED_SUBNET_PREFIXES)


def peers_for_role(role: NodeRole) -> tuple[Host, ...]:
    peers: list[Host] = []
    for peer in MANUAL_PEERS[role]:
        peer_ip = STATIC_IPS[role].get(peer)
        if peer_ip is None:
            continue
        peers.append(Host(ip=peer_ip, port=0))
    return tuple(peers)


def route_via_bridge(local: NodeRole, target: NodeRole) -> NodeRole | None:
    return None


def apply_layer_range(role: NodeRole, total_layers: int) -> tuple[int, int]:
    c_end = round(total_layers * PIPELINE_LAYER_FRACTIONS["C"])
    b_end = c_end + round(total_layers * PIPELINE_LAYER_FRACTIONS["B"])
    match role:
        case "C":
            return 0, min(c_end, total_layers)
        case "B":
            return min(c_end, total_layers), min(b_end, total_layers)
        case "A":
            return min(b_end, total_layers), total_layers
    return 0, 0


SendFn = Callable[[object, NodeId], None]
RecvFn = Callable[[NodeId], object]


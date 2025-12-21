"""Main entry point for EXO cluster node.

This module provides the Node class that coordinates all components of an EXO node,
including routing, worker operations, master election, and API services.
"""

import argparse
import ipaddress
import multiprocessing as mp
import os
import signal
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, Self

import anyio
from anyio.abc import TaskGroup
from loguru import logger
from pydantic import PositiveInt

import exo.routing.topics as topics
from exo.master.api import API  # TODO: should API be in master?
from exo.master.main import Master
from exo.routing.router import Router, get_node_id_keypair
from exo.shared.constants import EXO_LOG
from exo.shared.election import Election, ElectionResult
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import Receiver, channel
from exo.utils.pydantic_ext import CamelCaseModel
from exo.worker.download.impl_shard_downloader import exo_shard_downloader
from exo.worker.main import Worker

PEER_LISTEN_PORT = 5678
BIND_HOST = "0.0.0.0"

@dataclass
class Node:
    """Coordinates all components of an EXO cluster node.

    A Node represents a single device in the EXO cluster. It manages:
    - Router: Handles message routing and topic subscriptions
    - Worker: Executes tasks (model loading, inference, etc.)
    - Election: Participates in master election (every node participates to ensure
      a master exists even if no master candidates are present)
    - Master: Optional master component that manages cluster state (every node
      starts with a master, but only the elected one is active)
    - API: Optional API server for external interactions

    The Node coordinates these components and handles master election transitions,
    ensuring workers and API are recreated with the correct session when leadership
    changes.

    Attributes:
        router: Message router handling topic-based pub/sub.
        worker: Worker instance that executes tasks assigned to this node.
        election: Election service participating in master election.
        election_result_receiver: Channel receiver for election results.
        master: Master instance if this node is/was master, None otherwise.
        api: API server instance if enabled, None otherwise.
        node_id: Unique identifier for this node.
        _tg: Internal task group for managing concurrent operations.
    """

    router: Router
    worker: Worker
    election: Election
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None

    node_id: NodeId
    listen_port: PositiveInt
    host: str
    seeds: tuple[str, ...]
    _tg: TaskGroup = field(init=False, default_factory=anyio.create_task_group)

    @classmethod
    async def create(cls, args: "Args") -> "Self":
        """Create and initialize a new Node instance.

        This factory method sets up all node components:
        - Generates node identity from keypair
        - Creates initial session (node starts as its own master)
        - Initializes router with required topics
        - Creates worker, master, election, and optionally API components
        - Connects all components via topic-based channels

        Args:
            args: Configuration arguments for node startup.

        Returns:
            Fully initialized Node instance ready to run.
        """
        keypair = get_node_id_keypair()
        node_id = NodeId(keypair.to_peer_id().to_base58())
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        router = Router.create(keypair)
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        if args.seeds:
            logger.info(f"Connecting seeds for hostname {socket.gethostname()}: {args.seeds}")
            await router.connect_seeds(args.seeds)
        else:
            logger.info(f"No seeds configured for hostname {socket.gethostname()}; relying on mDNS")

        logger.info(f"Starting node {node_id}")
        if args.spawn_api:
            api = API(
                node_id,
                session_id,
                port=args.api_port,
                global_event_receiver=router.receiver(topics.GLOBAL_EVENTS),
                command_sender=router.sender(topics.COMMANDS),
                election_receiver=router.receiver(topics.ELECTION_MESSAGES),
            )
        else:
            api = None

        worker = Worker(
            node_id,
            session_id,
            exo_shard_downloader(),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            global_event_receiver=router.receiver(topics.GLOBAL_EVENTS),
            local_event_sender=router.sender(topics.LOCAL_EVENTS),
            command_sender=router.sender(topics.COMMANDS),
        )
        master = Master(
            node_id,
            session_id,
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
        )

        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            seniority=1_000_000 if args.force_master else 0,
            election_message_sender=router.sender(topics.ELECTION_MESSAGES),
            election_message_receiver=router.receiver(topics.ELECTION_MESSAGES),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            command_receiver=router.receiver(topics.COMMANDS),
            election_result_sender=er_send,
        )

        return cls(
            router,
            worker,
            election,
            er_recv,
            master,
            api,
            node_id,
            listen_port=PEER_LISTEN_PORT,
            host=BIND_HOST,
            seeds=tuple(args.seeds or ()),
        )

    async def run(self) -> None:
        """Run the node's main event loop.

        Starts all node components concurrently in a task group:
        - Router for message routing
        - Worker for task execution
        - Election service for master election
        - Master (if present) for state management
        - API (if present) for external interface
        - Election result handler for leadership transitions

        Sets up SIGINT handler for graceful shutdown. The method blocks until
        shutdown is requested.

        Note:
            This method should be called after Node.create(). It runs indefinitely
            until shutdown() is called or SIGINT is received.
        """
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.worker.run)
            tg.start_soon(self.election.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)

    def shutdown(self) -> None:
        """Initiate graceful shutdown of the node.

        Cancels the task group, which will stop all running components. If called
        a second time (e.g., during shutdown cleanup), forces immediate exit.

        Note:
            This method is safe to call from signal handlers and will trigger
            cleanup of all node components.
        """
        if self._tg.cancel_scope.cancel_called:
            import sys

            sys.exit(1)
        self._tg.cancel_scope.cancel()

    async def _elect_loop(self) -> None:
        """Handle election results and manage master transitions.

        Listens for election results and coordinates component state based on
        leadership changes. When a new master is elected:

        1. If this node becomes master: promote self to master (if not already)
        2. If another node becomes master: demote self (shutdown local master)
        3. On new master election (any node): recreate worker with new session
        4. On new master election: reset API with new session
        5. On non-new master election: unpause API

        This ensures all components operate with consistent session state
        matching the current cluster leadership.

        Note:
            Worker and API are recreated (not just reset) when a new master
            is elected to ensure clean state with the new session ID.
        """
        with self.election_result_receiver as results:
            async for result in results:

                if (
                    result.session_id.master_node_id == self.node_id
                    and self.master is not None
                ):
                    logger.info("Node elected Master")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
                    logger.info("Node elected Master - promoting self")
                    self.master = Master(
                        self.node_id,
                        result.session_id,
                        global_event_sender=self.router.sender(topics.GLOBAL_EVENTS),
                        local_event_receiver=self.router.receiver(topics.LOCAL_EVENTS),
                        command_receiver=self.router.receiver(topics.COMMANDS),
                    )
                    self._tg.start_soon(self.master.run)
                elif (
                    result.session_id.master_node_id != self.node_id
                    and self.master is not None
                ):
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master - demoting self"
                    )
                    await self.master.shutdown()
                    self.master = None
                else:
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master"
                    )
                if result.is_new_master:
                    await anyio.sleep(0)
                    if self.worker:
                        self.worker.shutdown()
                        # TODO: add profiling etc to resource monitor
                        self.worker = Worker(
                            self.node_id,
                            result.session_id,
                            exo_shard_downloader(),
                            connection_message_receiver=self.router.receiver(
                                topics.CONNECTION_MESSAGES
                            ),
                            global_event_receiver=self.router.receiver(
                                topics.GLOBAL_EVENTS
                            ),
                            local_event_sender=self.router.sender(topics.LOCAL_EVENTS),
                            command_sender=self.router.sender(topics.COMMANDS),
                        )
                        self._tg.start_soon(self.worker.run)
                    if self.api:
                        self.api.reset(result.session_id, result.won_clock)
                else:
                    if self.api:
                        self.api.unpause(result.won_clock)


def main() -> None:
    """Main entry point for EXO cluster node.

    Parses command-line arguments, sets up logging, creates and runs a Node
    instance. Handles the full node lifecycle from startup to shutdown.

    Process:
        1. Parse command-line arguments
        2. Configure multiprocessing start method
        3. Initialize logging system
        4. Create Node instance
        5. Run node until shutdown
        6. Cleanup logging

    Note:
        This function uses anyio.run() to bridge async Node operations with
        the synchronous entry point. The node runs until SIGINT or explicit
        shutdown.
    """
    args = apply_hostname_overrides(Args.parse())

    mp.set_start_method("spawn")
    logger_setup(EXO_LOG, args.verbosity)
    logger.info("Starting EXO")

    node = anyio.run(Node.create, args)
    anyio.run(node.run)
    logger.info("EXO Shutdown complete")
    logger_cleanup()


class Args(CamelCaseModel):
    """Command-line arguments for EXO node configuration.

    Attributes:
        verbosity: Logging verbosity level. Positive values increase verbosity,
            negative decrease (quiet mode). Default is 0 (normal).
        force_master: If True, set election seniority to very high value (1,000,000)
            to force this node to win elections and become master. Default is False.
        spawn_api: If True, start the API server. Default is False (API disabled).
        api_port: Port number for the API server. Must be a positive integer.
            Default is 52415.
        tb_only: If True, restrict network to Thunderbolt connections only.
            Default is False.
        use_rdma: Whether to require RDMA; enforced per-host overrides.
        host: Bind address for networking; enforced per-host overrides. Defaults to 0.0.0.0.
        discovery_port: Port used for peer discovery; enforced per-host overrides. Defaults to 5678.
        seeds: Optional list of manual seed peers; extended per-host overrides.
    """

    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False
    use_rdma: bool = True
    host: str = BIND_HOST
    discovery_port: PositiveInt = PEER_LISTEN_PORT
    seeds: list[str] | None = None

    @classmethod
    def parse(cls) -> Self:
        """Parse command-line arguments into Args instance.

        Sets up argument parser with options:
        - `-q, --quiet`: Decrease verbosity (sets verbosity to -1)
        - `-v, --verbose`: Increase verbosity (can be repeated: -vv, -vvv)
        - `-m, --force-master`: Force this node to become master
        - `--no-api`: Disable API server (overrides default)
        - `--api-port PORT`: Set API server port (default: 52415)

        Returns:
            Validated Args instance with parsed values.

        Raises:
            SystemExit: If argument parsing fails (argparse behavior).
        """
        parser = argparse.ArgumentParser(prog="EXO")
        default_verbosity = 0
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_const",
            const=-1,
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-m",
            "--force-master",
            action="store_true",
            dest="force_master",
        )
        parser.add_argument(
            "--no-api",
            action="store_false",
            dest="spawn_api",
        )
        parser.add_argument(
            "--api-port",
            type=int,
            dest="api_port",
            default=52415,
        )
        parser.add_argument(
            "--seed",
            action="append",
            dest="seeds",
            default=None,
            help="Seed peer as host:port or multiaddr (can be passed multiple times)",
        )

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - runtime validation


def apply_hostname_overrides(args: Args) -> Args:
    seeds = list(args.seeds or [])
    seeds.extend(_env_seeds())

    local_ips = _local_ipv4s()
    subnets = _thunderbolt_subnets()
    if subnets:
        os.environ["EXO_TB_SUBNETS"] = ",".join(str(net) for net in subnets)
        logger.info(f"Using Thunderbolt subnets for discovery: {os.environ['EXO_TB_SUBNETS']}")
    else:
        logger.warning("No Thunderbolt subnets detected; mDNS not filtered")

    seeds = _dedupe_preserve_order(seeds)

    return args.model_copy(
        update={
            "use_rdma": True,
            "host": BIND_HOST,
            "discovery_port": PEER_LISTEN_PORT,
            "seeds": seeds,
            "force_master": args.force_master,  # no IP-based forcing
        },
        deep=True,
    )


def _env_seeds() -> list[str]:
    env = os.environ.get("EXO_SEEDS", "")
    tokens = []
    for token in env.replace(",", " ").split():
        if token:
            tokens.append(token)
    return tokens


def _local_ipv4s() -> set[str]:
    ips: set[str] = set()
    try:
        iface_list = subprocess.check_output(["ifconfig", "-l"], text=True).strip()
        for iface in iface_list.split():
            try:
                ip = subprocess.check_output(
                    ["ipconfig", "getifaddr", iface], text=True
                ).strip()
                if ip:
                    ips.add(ip)
            except subprocess.CalledProcessError:
                continue
    except Exception:
        pass

    for host in (socket.gethostname(), None, "localhost"):
        try:
            for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
                ips.add(info[4][0])
        except OSError:
            continue
    return ips


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _thunderbolt_subnets() -> set[ipaddress.IPv4Network]:
    """Infer Thunderbolt subnets from interface inet/netmask; skip Wi-Fi (en0) and loopback."""
    nets: set[ipaddress.IPv4Network] = set()
    try:
        ifconfig_out = subprocess.check_output(["ifconfig"], text=True)
    except Exception:
        return nets

    blocks = ifconfig_out.split("\n\n")
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        iface = lines[0].split(":")[0]
        if iface.startswith("lo"):
            continue
        if iface == "en0":  # skip Wi-Fi
            continue
        for ln in lines:
            if not ln.startswith("inet "):
                continue
            parts = ln.split()
            if len(parts) < 4 or parts[0] != "inet":
                continue
            ip = parts[1]
            mask_hex = None
            if "netmask" in parts:
                try:
                    idx = parts.index("netmask")
                    mask_hex = parts[idx + 1]
                except Exception:
                    mask_hex = None
            if not ipaddress.ip_address(ip).is_private:
                continue
            if ip.startswith("169.254."):
                continue
            if mask_hex and mask_hex.startswith("0x"):
                try:
                    mask_int = int(mask_hex, 16)
                    mask_str = str(ipaddress.IPv4Address(mask_int))
                except Exception:
                    mask_str = "255.255.255.0"
            else:
                mask_str = "255.255.255.0"
            try:
                net = ipaddress.IPv4Network(f"{ip}/{mask_str}", strict=False)
            except Exception:
                continue
            if net.prefixlen > 30:
                continue
            nets.add(net)
    return nets


def _seeds_from_subnets(
    subnets: Iterable[ipaddress.IPv4Network], local_ips: set[str]
) -> list[str]:
    seeds: list[str] = []
    for net in subnets:
        added = 0
        for host in net.hosts():
            host_str = str(host)
            if host_str in local_ips:
                continue
            seeds.append(f"{host_str}:{PEER_LISTEN_PORT}")
            added += 1
            if added >= 32:
                break  # avoid massive dial lists on large subnets
    return seeds

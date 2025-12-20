"""Main entry point for EXO cluster node.

This module provides the Node class that coordinates all components of an EXO node,
including routing, worker operations, master election, and API services.
"""

import argparse
import multiprocessing as mp
import signal
from dataclasses import dataclass, field
from typing import Self

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

        return cls(router, worker, election, er_recv, master, api, node_id)

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
    args = Args.parse()

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
    """

    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False

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

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - We are intentionally validating here, we can't do it statically

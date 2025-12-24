"""
Standalone Master application entry point for static 4-node cluster.

This is the entry point for running the Master node. It initializes:
- Master with static topology
- API server (OpenAI-compatible endpoints)
- HTTP endpoints for worker communication
"""
import argparse
import signal
import socket
import sys

import anyio
from loguru import logger

from exo.master.api import API
from exo.master.main import Master
from exo.shared.constants import EXO_LOG
from exo.shared.election import ElectionMessage
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.static_config import (
    get_static_config,
    is_master_node,
)
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import ForwarderEvent
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.memory_management import (
    flush_ram_on_startup,
    flush_ram_on_shutdown,
    verify_zero_swap,
    log_memory_stats,
)


async def main() -> None:
    """Main entry point for Master application."""
    parser = argparse.ArgumentParser(description="Exo Master (Static Cluster)")
    parser.add_argument(
        "--api-port",
        type=int,
        default=52415,
        help="Port for API server (default: 52415)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    logger_setup(EXO_LOG, verbosity=0 if args.log_level == "INFO" else 1 if args.log_level == "DEBUG" else -1)
    logger.info("Starting Exo Master (Static Cluster Mode)")

    # Flush RAM on startup and verify zero swap
    flush_ram_on_startup()
    log_memory_stats()
    try:
        verify_zero_swap()
        logger.info("Swap usage verified: zero")
    except ValueError as e:
        logger.error(f"CRITICAL: {e}")
        # Continue anyway but log the error

    # Verify we're on the master node
    hostname = socket.gethostname()
    if not is_master_node():
        logger.warning(
            f"Current hostname ({hostname}) does not match master node hostname. "
            "Continuing anyway..."
        )

    # Get static configuration
    config = get_static_config()
    node_id = config.master.node_id
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    logger.info("Creating static topology with 3 worker nodes")

    # Create channels for Master-API communication (bypassing Router)
    command_send, command_recv = channel[ForwarderCommand]()
    event_send, event_recv = channel[ForwarderEvent]()
    election_send, election_recv = channel[ElectionMessage]()

    # Initialize Master (with static topology initialization)
    logger.info("Initializing Master")
    master = Master(
        node_id=node_id,
        session_id=session_id,
        command_receiver=command_recv,  # Receive commands from API
        local_event_receiver=None,  # No Router for static setup
        global_event_sender=event_send,  # Send events to API
        initial_topology=None,  # Will use static topology from create_static_topology()
    )

    # Initialize API
    logger.info(f"Initializing API on port {args.api_port}")
    api = API(
        node_id=node_id,
        session_id=session_id,
        port=args.api_port,
        global_event_receiver=event_recv,  # Receive events from Master
        command_sender=command_send,  # Send commands to Master
        election_receiver=election_recv,  # No elections, but API expects this channel
    )
    # Set master reference in API so it can access master's state
    api._master_ref = master

    # Print startup banner
    print_startup_banner(args.api_port)

    # Setup signal handlers for graceful shutdown
    shutdown_event = anyio.Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run Master and API in parallel
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(master.run)
            tg.start_soon(api.run)
            # Wait for shutdown signal
            async def wait_for_shutdown():
                await shutdown_event.wait()
                tg.cancel_scope.cancel()
            tg.start_soon(wait_for_shutdown)
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.exception(f"Error in service: {exc}")
    finally:
        logger.info("Shutting down Master")
        await master.shutdown()
        flush_ram_on_shutdown()
        log_memory_stats()
        logger.info("Master shutdown complete")
        logger_cleanup()


if __name__ == "__main__":
    try:
        anyio.run(main)
    except KeyboardInterrupt:
        logger.info("Master interrupted by user")
    except Exception as e:
        logger.exception(f"Master crashed: {e}")
        sys.exit(1)


"""
Standalone Worker application entry point for static 4-node cluster.

This is the entry point for running a Worker node. It initializes:
- Worker with MLX execution capabilities
- HTTP client for Master communication
- Registers with Master on startup
"""
import argparse
import signal
import socket
import sys

import anyio
from loguru import logger

from exo.shared.constants import EXO_LOG
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.static_config import (
    get_master_url,
    get_static_config,
    get_worker_config_by_node_id,
)
from exo.shared.types.common import NodeId, SessionId
from exo.worker.download.impl_shard_downloader import exo_shard_downloader
from exo.worker.main import Worker
from exo.utils.banner import print_startup_banner
from exo.utils.memory_management import (
    flush_ram_on_startup,
    flush_ram_on_shutdown,
    verify_zero_swap,
    log_memory_stats,
)


async def main() -> None:
    """Main entry point for Worker application."""
    parser = argparse.ArgumentParser(description="Exo Worker (Static Cluster)")
    parser.add_argument(
        "--master-url",
        type=str,
        default=None,
        help="Master URL (default: from static config)",
    )
    parser.add_argument(
        "--node-id",
        type=NodeId,
        required=True,
        help="NodeId of this worker (e.g., 'static-worker-0-adams-mac-studio-m4')",
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
    logger.info("Starting Exo Worker (Static Cluster Mode)")

    # Flush RAM on startup and verify zero swap
    flush_ram_on_startup()
    log_memory_stats()
    try:
        verify_zero_swap()
        logger.info("Swap usage verified: zero")
    except ValueError as e:
        logger.error(f"CRITICAL: {e}")
        # Continue anyway but log the error

    # Get worker config for this node - use node_id from args instead of hostname
    # (hostname case may differ, so we use explicit node_id)
    worker_config = get_worker_config_by_node_id(args.node_id)
    if worker_config is None:
        logger.error(f"Could not find worker config for node_id {args.node_id}")
        sys.exit(1)

    # Get Master URL
    master_url = args.master_url or get_master_url()
    logger.info(f"Master URL: {master_url}")

    # Get static configuration
    config = get_static_config()
    node_id = worker_config.node_id
    session_id = SessionId(master_node_id=config.master.node_id, election_clock=0)

    logger.info(f"Worker node_id: {node_id}, rank: {worker_config.rank}")

    # Initialize Worker (channels are None for simplified mode - will use HTTP instead)
    logger.info("Initializing Worker")
    worker = Worker(
        node_id=node_id,
        session_id=session_id,
        shard_downloader=exo_shard_downloader(),
        connection_message_receiver=None,  # No Router for static setup
        global_event_receiver=None,  # No Router for static setup - will use HTTP
        local_event_sender=None,  # No Router for static setup - will use HTTP
        command_sender=None,  # No Router for static setup - will use HTTP
        master_url=master_url,  # HTTP URL for Master communication
    )

    # Print startup banner (workers don't have a port, use 0)
    print_startup_banner(0)

    # Setup signal handlers for graceful shutdown
    shutdown_event = anyio.Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run Worker
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(worker.run)
            # Wait for shutdown signal
            async def wait_for_shutdown():
                await shutdown_event.wait()
                tg.cancel_scope.cancel()
            tg.start_soon(wait_for_shutdown)
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.exception(f"Error in worker: {exc}")
    finally:
        flush_ram_on_shutdown()
        log_memory_stats()
        logger.info("Worker shutdown complete")
        logger_cleanup()


if __name__ == "__main__":
    try:
        anyio.run(main)
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.exception(f"Worker crashed: {e}")
        sys.exit(1)


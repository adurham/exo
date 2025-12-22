import argparse
import multiprocessing as mp
import os
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
from exo.master.placement_utils import estimated_memory_bandwidth_gbps
from exo.routing.router import Router, get_node_id_keypair
from exo.shared.constants import EXO_LOG
from exo.shared.election import Election, ElectionResult
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import NodePerformanceMeasured
from exo.utils.channels import Receiver, channel
from exo.utils.pydantic_ext import CamelCaseModel
from exo.worker.download.impl_shard_downloader import exo_shard_downloader
from exo.worker.main import Worker
from exo.worker.utils.profile import get_memory_profile
from exo.worker.utils.system_info import get_model_and_chip


# I marked this as a dataclass as I want trivial constructors.
@dataclass
class Node:
    router: Router
    worker: Worker
    election: Election  # Every node participates in election, as we do want a node to become master even if it isn't a master candidate if no master candidates are present.
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None

    node_id: NodeId
    _tg: TaskGroup = field(init=False, default_factory=anyio.create_task_group)

    @classmethod
    async def create(cls, args: "Args") -> "Self":
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
        # We start every node with a master
        master = Master(
            node_id,
            session_id,
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
        )

        model_id, chip_id = await get_model_and_chip()
        memory_profile = get_memory_profile()
        membw = estimated_memory_bandwidth_gbps(chip_id=chip_id)
        ram_total = memory_profile.ram_total.in_bytes
        
        logger.info(
            f"Node hardware: {chip_id}, {membw} GB/s, {ram_total} bytes RAM"
        )
        
        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            # If someone manages to assemble 1 MILLION devices into an exo cluster then. well done. good job champ.
            seniority=1_000_000 if args.force_master else 0,
            # nb: this DOES feedback right now. i have thoughts on how to address this,
            # but ultimately it seems not worth the complexity
            election_message_sender=router.sender(topics.ELECTION_MESSAGES),
            election_message_receiver=router.receiver(topics.ELECTION_MESSAGES),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            command_receiver=router.receiver(topics.COMMANDS),
            election_result_sender=er_send,
            membw_gbps=membw,
            ram_total_bytes=ram_total,
        )

        return cls(router, worker, election, er_recv, master, api, node_id)

    async def run(self):
        # Flush any stale resources on startup
        import sys
        try:
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            sys.stderr.flush()
        except (BrokenPipeError, OSError):
            pass
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            signal.signal(signal.SIGTERM, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.worker.run)
            tg.start_soon(self.election.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)
            tg.start_soon(self._update_election_hardware)

    def shutdown(self):
        # if this is our second call to shutdown, just sys.exit
        if self._tg.cancel_scope.cancel_called:
            import sys

            sys.exit(1)
        logger.info("Node shutdown initiated, canceling all tasks")
        self._tg.cancel_scope.cancel()

    async def _elect_loop(self):
        with self.election_result_receiver as results:
            async for result in results:
                # This function continues to have a lot of very specific entangled logic
                # At least it's somewhat contained

                # I don't like this duplication, but it's manageable for now.
                # TODO: This function needs refactoring generally

                # Ok:
                # On new master:
                # - Elect master locally if necessary
                # - Shutdown and re-create the worker
                # - Shut down and re-create the API

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
                    try:
                        await self.master.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down master: {e}")
                    self.master = None
                else:
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master"
                    )
                if result.is_new_master:
                    await anyio.sleep(0)
                    if self.worker:
                        try:
                            self.worker.shutdown()
                        except Exception as e:
                            logger.warning(f"Error shutting down worker: {e}")
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

    async def _update_election_hardware(self) -> None:
        """Listen for this node's performance profile and update election hardware info."""
        with self.router.receiver(topics.GLOBAL_EVENTS) as events:
            async for event in events:
                if isinstance(event.event, NodePerformanceMeasured):
                    if event.event.node_id == self.node_id:
                        profile = event.event.node_profile
                        membw = estimated_memory_bandwidth_gbps(chip_id=profile.chip_id)
                        ram_total = profile.memory.ram_total.in_bytes
                        self.election.update_hardware_info(membw, ram_total)
                        logger.info(
                            f"Updated election hardware info: {membw} GB/s, {ram_total} bytes"
                        )


def main():
    args = Args.parse()

    mp.set_start_method("spawn")
    # Patch multiprocessing to handle BrokenPipeError when flushing stdout/stderr
    # This happens when stdout/stderr are redirected to /dev/null
    import multiprocessing.util
    original_flush = multiprocessing.util._flush_std_streams
    def patched_flush_std_streams():
        try:
            original_flush()
        except (BrokenPipeError, OSError):
            pass
    multiprocessing.util._flush_std_streams = patched_flush_std_streams
    
    # TODO: Refactor the current verbosity system
    logger_setup(EXO_LOG, args.verbosity)
    os.environ["EXO_VERBOSITY"] = str(args.verbosity)
    logger.info("Starting EXO")

    try:
        node = anyio.run(Node.create, args)
        anyio.run(node.run)
    except KeyboardInterrupt:
        logger.info("EXO interrupted by user")
    except Exception as e:
        logger.opt(exception=e).error("EXO crashed with exception")
        raise
    finally:
        logger.info("EXO Shutdown complete, flushing logs and cleaning up")
        # Flush all log handlers
        import sys
        try:
            for handler in logger._core.handlers.values():
                try:
                    handler._sink.flush()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            sys.stderr.flush()
        except (BrokenPipeError, OSError):
            pass
        logger_cleanup()


class Args(CamelCaseModel):
    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False

    @classmethod
    def parse(cls) -> Self:
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

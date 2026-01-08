from datetime import datetime, timezone
import time
from random import random

import anyio
from anyio import CancelScope, create_task_group, current_time, fail_after
from anyio.abc import TaskGroup
from loguru import logger

from exo.routing.connection_message import ConnectionMessage, ConnectionMessageType
from exo.shared.apply import apply
from exo.shared.types.commands import (
    CheckShardPresent,
    BaseCommand,
    DeleteModel,
    DeviceDownloadModel,
    ForwarderCommand,
    RequestEventLog,
    ShardPresent,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    Event,
    EventId,
    ForwarderEvent,
    IndexedEvent,
    NodeDownloadProgress,
    NodeDownloadRemoved,
    NodeMemoryMeasured,
    NodePerformanceMeasured,
    TaskCreated,
    TaskStatusUpdated,
    TopologyEdgeCreated,
    TopologyEdgeDeleted,
)
from exo.shared.types.models import ModelId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import MemoryPerformanceProfile, NodePerformanceProfile
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    CreateRunner,
    DownloadModel,
    Shutdown,
    Task,
    TaskStatus,
)
from exo.shared.types.topology import Connection
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.event_buffer import OrderedBuffer
from exo.worker.download.download_utils import (
    map_repo_download_progress_to_download_progress_data,
)
from exo.worker.download.shard_downloader import RepoDownloadProgress, ShardDownloader
from exo.worker.file_server import FileServer
from exo.worker.plan import plan
from exo.worker.runner.runner_supervisor import RunnerSupervisor
from exo.worker.utils import start_polling_memory_metrics, start_polling_node_metrics
from exo.worker.utils.net_profile import check_reachable


class Worker:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        shard_downloader: ShardDownloader,
        *,
        connection_message_receiver: Receiver[ConnectionMessage],
        global_event_receiver: Receiver[ForwarderEvent],
        local_event_sender: Sender[ForwarderEvent],
        command_sender: Sender[ForwarderCommand],
        command_receiver: Receiver[ForwarderCommand],
    ):
        self.node_id: NodeId = node_id
        self.session_id: SessionId = session_id

        self.shard_downloader: ShardDownloader = shard_downloader
        self._pending_downloads: dict[RunnerId, ShardMetadata] = {}

        self.global_event_receiver = global_event_receiver
        self.local_event_sender = local_event_sender
        self.local_event_index = 0
        self.command_sender = command_sender
        self.command_receiver = command_receiver
        self.connection_message_receiver = connection_message_receiver
        self.event_buffer = OrderedBuffer[Event]()
        self.out_for_delivery: dict[EventId, ForwarderEvent] = {}

        self.state: State = State()
        self.download_status: dict[ModelId, DownloadProgress] = {}
        self.runners: dict[RunnerId, RunnerSupervisor] = {}
        self._tg: TaskGroup | None = None

        self._nack_cancel_scope: CancelScope | None = None
        self._nack_attempts: int = 0
        self._nack_base_seconds: float = 0.5
        self._nack_cap_seconds: float = 10.0

        self.event_sender, self.event_receiver = channel[Event]()

        self.file_server = FileServer(self.node_id)
        self.peer_locations: dict[ModelId, str] = {}
        self.discovery_start_times: dict[ModelId, float] = {}
        self.runner_failure_history: dict[RunnerId, tuple[float, int]] = {}

    async def run(self):
        logger.info("Starting Worker")

        # TODO: CLEANUP HEADER
        async def resource_monitor_callback(
            node_performance_profile: NodePerformanceProfile,
        ) -> None:
            await self.event_sender.send(
                NodePerformanceMeasured(
                    node_id=self.node_id,
                    node_profile=node_performance_profile,
                    when=str(datetime.now(tz=timezone.utc)),
                ),
            )

        async def memory_monitor_callback(
            memory_profile: MemoryPerformanceProfile,
        ) -> None:
            await self.event_sender.send(
                NodeMemoryMeasured(
                    node_id=self.node_id,
                    memory=memory_profile,
                    when=str(datetime.now(tz=timezone.utc)),
                )
            )

        # END CLEANUP

        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self.plan_step)
            tg.start_soon(start_polling_node_metrics, resource_monitor_callback)

            tg.start_soon(start_polling_memory_metrics, memory_monitor_callback)
            tg.start_soon(self._emit_existing_download_progress)
            tg.start_soon(self._connection_message_event_writer)
            tg.start_soon(self._resend_out_for_delivery)
            tg.start_soon(self._event_applier)
            tg.start_soon(self._forward_events)
            tg.start_soon(self._poll_connection_updates)
            tg.start_soon(self.file_server.start)
            tg.start_soon(self._command_processor)

        # Actual shutdown code - waits for all tasks to complete before executing.
        self.local_event_sender.close()
        self.command_sender.close()
        self.command_receiver.close()
        await self.file_server.stop()
        for runner in self.runners.values():
            runner.shutdown()

    async def _event_applier(self):
        with self.global_event_receiver as events:
            async for f_event in events:
                if f_event.origin != self.session_id.master_node_id:
                    continue
                self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                event_id = f_event.event.event_id
                if event_id in self.out_for_delivery:
                    del self.out_for_delivery[event_id]

                # 2. for each event, apply it to the state
                indexed_events = self.event_buffer.drain_indexed()
                if indexed_events:
                    self._nack_attempts = 0

                if not indexed_events and (
                    self._nack_cancel_scope is None
                    or self._nack_cancel_scope.cancel_called
                ):
                    assert self._tg
                    # Request the next index.
                    self._tg.start_soon(
                        self._nack_request, self.state.last_event_applied_idx + 1
                    )
                    continue
                elif indexed_events and self._nack_cancel_scope:
                    self._nack_cancel_scope.cancel()

                for idx, event in indexed_events:
                    self.state = apply(self.state, IndexedEvent(idx=idx, event=event))

    async def plan_step(self):
        while True:
            await anyio.sleep(0.1)
            # 3. based on the updated state, we plan & execute an operation.
            task: Task | None = plan(
                self.node_id,
                self.runners,
                self.download_status,
                self.state.downloads,
                self.state.instances,
                self.state.runners,
                self.state.tasks,
                self.runner_failure_history,
            )
            if task is None:
                continue
            logger.debug(f"Worker plan: {task.__class__.__name__}")
            assert task.task_status
            await self.event_sender.send(TaskCreated(task_id=task.task_id, task=task))

            # lets not kill the worker if a runner is unresponsive
            match task:
                case CreateRunner():
                    self._create_supervisor(task)
                    await self.event_sender.send(
                        TaskStatusUpdated(
                            task_id=task.task_id, task_status=TaskStatus.Complete
                        )
                    )
                case DownloadModel(shard_metadata=shard):
                    if shard.model_meta.model_id not in self.download_status:
                        progress = DownloadPending(
                            shard_metadata=shard, node_id=self.node_id
                        )
                        self.download_status[shard.model_meta.model_id] = progress
                        await self.event_sender.send(
                            NodeDownloadProgress(download_progress=progress)
                        )
                    initial_progress = (
                        await self.shard_downloader.get_shard_download_status_for_shard(
                            shard
                        )
                    )
                    if initial_progress.status == "complete":
                        progress = DownloadCompleted(
                            shard_metadata=shard, node_id=self.node_id
                        )
                        self.download_status[shard.model_meta.model_id] = progress
                        await self.event_sender.send(
                            NodeDownloadProgress(download_progress=progress)
                        )
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id,
                                task_status=TaskStatus.Complete,
                            )
                        )
                    else:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Running
                            )
                        )
                        self._tg.start_soon(
                            self._handle_shard_download_process, task, initial_progress
                        )
                case Shutdown(runner_id=runner_id):
                    # Track failure history for backoff
                    now = time.time()
                    last_time, count = self.runner_failure_history.get(runner_id, (0, 0))
                    if now - last_time < 300:  # 5 minute window
                        count += 1
                    else:
                        count = 1
                    self.runner_failure_history[runner_id] = (now, count)

                    try:
                        with fail_after(3):
                            await self.runners.pop(runner_id).start_task(task)
                    except TimeoutError:
                        await self.event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.TimedOut
                            )
                        )
                case task:
                    await self.runners[self._task_to_runner_id(task)].start_task(task)

    def shutdown(self):
        if self._tg:
            self._tg.cancel_scope.cancel()

    def _task_to_runner_id(self, task: Task):
        instance = self.state.instances[task.instance_id]
        return instance.shard_assignments.node_to_runner[self.node_id]

    async def _connection_message_event_writer(self):
        with self.connection_message_receiver as connection_messages:
            async for msg in connection_messages:
                await self.event_sender.send(
                    self._convert_connection_message_to_event(msg)
                )

    def _convert_connection_message_to_event(self, msg: ConnectionMessage):
        match msg.connection_type:
            case ConnectionMessageType.Connected:
                return TopologyEdgeCreated(
                    edge=Connection(
                        local_node_id=self.node_id,
                        send_back_node_id=msg.node_id,
                        send_back_multiaddr=Multiaddr(
                            address=f"/ip4/{msg.remote_ipv4}/tcp/{msg.remote_tcp_port}"
                        ),
                    )
                )

            case ConnectionMessageType.Disconnected:
                return TopologyEdgeDeleted(
                    edge=Connection(
                        local_node_id=self.node_id,
                        send_back_node_id=msg.node_id,
                        send_back_multiaddr=Multiaddr(
                            address=f"/ip4/{msg.remote_ipv4}/tcp/{msg.remote_tcp_port}"
                        ),
                    )
                )

    async def _nack_request(self, since_idx: int) -> None:
        # We request all events after (and including) the missing index.
        # This function is started whenever we receive an event that is out of sequence.
        # It is cancelled as soon as we receiver an event that is in sequence.

        if since_idx < 0:
            logger.warning(f"Negative value encountered for nack request {since_idx=}")
            since_idx = 0

        with CancelScope() as scope:
            self._nack_cancel_scope = scope
            delay: float = self._nack_base_seconds * (2.0**self._nack_attempts)
            delay = min(self._nack_cap_seconds, delay)
            self._nack_attempts += 1
            try:
                await anyio.sleep(delay)
                logger.info(
                    f"Nack attempt {self._nack_attempts}: Requesting Event Log from {since_idx}"
                )
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self.node_id,
                        command=RequestEventLog(since_idx=since_idx),
                    )
                )
            finally:
                if self._nack_cancel_scope is scope:
                    self._nack_cancel_scope = None

    async def _resend_out_for_delivery(self) -> None:
        # This can also be massively tightened, we should check events are at least a certain age before resending.
        # Exponential backoff would also certainly help here.
        while True:
            await anyio.sleep(1 + random())
            for event in self.out_for_delivery.copy().values():
                await self.local_event_sender.send(event)

    ## Op Executors

    def _create_supervisor(self, task: CreateRunner) -> RunnerSupervisor:
        """Creates and stores a new AssignedRunner with initial downloading status."""
        runner = RunnerSupervisor.create(
            bound_instance=task.bound_instance,
            event_sender=self.event_sender.clone(),
        )
        self.runners[task.bound_instance.bound_runner_id] = runner
        assert self._tg
        self._tg.start_soon(runner.run)
        return runner

    async def _handle_shard_download_process(
        self,
        task: DownloadModel,
        initial_progress: RepoDownloadProgress,
    ):
        """Manages the shard download process with progress tracking."""
        model_id = task.shard_metadata.model_meta.model_id
        
        # 1. Immediately report Ongoing to avoid task-loop spam
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=task.shard_metadata,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
        )
        self.download_status[task.shard_metadata.model_meta.model_id] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        endpoint: str | None = None
        
        while True:
            endpoint = self.peer_locations.get(model_id)
            if endpoint:
                break

            # Check global state to see if anyone has it or is downloading it
            has_peer = False
            for node_downloads in self.state.downloads.values():
                for download in node_downloads:
                    if download.shard_metadata.model_meta.model_id == model_id:
                        if isinstance(download, (DownloadCompleted, DownloadOngoing)):
                            has_peer = True
                            break
                if has_peer:
                    break

            if has_peer:
                # Someone has it or is downloading it.
                # If we don't have an endpoint yet, trigger discovery to find WHO and WHERE.
                if model_id not in self.discovery_start_times:
                    logger.debug(f"Checking P2P availability for {model_id} (found in global state)")
                    self.discovery_start_times[model_id] = time.time()
                    self.command_sender.send_nowait(
                        ForwarderCommand(
                            origin=self.node_id,
                            command=CheckShardPresent(model_id=str(model_id)),
                        )
                    )
                elif time.time() - self.discovery_start_times[model_id] > 2.0:
                    # Retry discovery periodically
                    logger.debug(f"Waiting for P2P endpoint for {model_id}...")
            else:
                # No one has it in global state.
                if model_id not in self.discovery_start_times:
                    # Trigger discovery just in case global state is lagging
                    logger.debug(f"Triggering P2P discovery for {model_id}")
                    self.discovery_start_times[model_id] = time.time()
                    self.command_sender.send_nowait(
                        ForwarderCommand(
                            origin=self.node_id,
                            command=CheckShardPresent(model_id=str(model_id)),
                        )
                    )

            if (
                not endpoint
                and model_id in self.discovery_start_times
                and time.time() - self.discovery_start_times[model_id] < 1.0
            ):
                # Still waiting for discovery
                pass
            
            elif not endpoint and model_id in self.discovery_start_times:
                if self.node_id == self.session_id.master_node_id:
                    logger.debug(f"P2P discovery timed out for {model_id}, falling back to default")
                    endpoint = None
                    break
                else:
                    if time.time() - self.discovery_start_times[model_id] > 10.0:
                        logger.debug(f"Waiting for coordinator for {model_id}...")
                        # We can reset discovery start time so we occasionally re-check
                        del self.discovery_start_times[model_id]
            
            await anyio.sleep(1.0)


        last_progress_time = 0.0
        throttle_interval_secs = 1.0

        # TODO: i hate callbacks
        def download_progress_callback(
            shard: ShardMetadata, progress: RepoDownloadProgress
        ) -> None:
            nonlocal self
            nonlocal last_progress_time
            if progress.status == "complete":
                status = DownloadCompleted(shard_metadata=shard, node_id=self.node_id)
                self.download_status[shard.model_meta.model_id] = status
                # Footgun!
                self.event_sender.send_nowait(
                    NodeDownloadProgress(download_progress=status)
                )
                self.event_sender.send_nowait(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
            elif (
                progress.status == "in_progress"
                and current_time() - last_progress_time > throttle_interval_secs
            ):
                status = DownloadOngoing(
                    node_id=self.node_id,
                    shard_metadata=shard,
                    download_progress=map_repo_download_progress_to_download_progress_data(
                        progress
                    ),
                )
                self.download_status[shard.model_meta.model_id] = status
                self.event_sender.send_nowait(
                    NodeDownloadProgress(download_progress=status)
                )
                last_progress_time = current_time()

        self.shard_downloader.on_progress(download_progress_callback)
        await self.shard_downloader.ensure_shard(task.shard_metadata, False, endpoint)

    async def _forward_events(self) -> None:
        with self.event_receiver as events:
            async for event in events:
                fe = ForwarderEvent(
                    origin_idx=self.local_event_index,
                    origin=self.node_id,
                    session=self.session_id,
                    event=event,
                )
                logger.debug(
                    f"Worker published event {self.local_event_index}: {event}"
                )
                self.local_event_index += 1
                await self.local_event_sender.send(fe)
                self.out_for_delivery[event.event_id] = fe

    async def _poll_connection_updates(self):
        while True:
            # TODO: EdgeDeleted
            edges = set(self.state.topology.list_connections())
            conns = await check_reachable(self.state.topology, self.node_id)
            for nid in conns:
                for ip in conns[nid]:
                    if "127.0.0.1" in ip or "localhost" in ip:
                        logger.warning(
                            f"Loopback connection should not happen: {ip=} for {nid=}"
                        )

                    edge = Connection(
                        local_node_id=self.node_id,
                        send_back_node_id=nid,
                        # nonsense multiaddr
                        send_back_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415")
                        if "." in ip
                        # nonsense multiaddr
                        else Multiaddr(address=f"/ip6/{ip}/tcp/52415"),
                    )
                    if edge not in edges:
                        logger.debug(f"ping discovered {edge=}")
                        await self.event_sender.send(TopologyEdgeCreated(edge=edge))

            for nid, conn in self.state.topology.out_edges(self.node_id):
                if (
                    nid not in conns
                    or conn.send_back_multiaddr.ip_address not in conns.get(nid, set())
                ):
                    logger.debug(f"ping failed to discover {conn=}")
                    await self.event_sender.send(TopologyEdgeDeleted(edge=conn))

            await anyio.sleep(10)

    async def _emit_existing_download_progress(self) -> None:
        try:
            while True:
                logger.info("Fetching and emitting existing download progress...")
                async for (
                    _,
                    progress,
                ) in self.shard_downloader.get_shard_download_status():
                    if progress.status == "complete":
                        status = DownloadCompleted(
                            node_id=self.node_id, shard_metadata=progress.shard
                        )
                    elif progress.status in ["in_progress", "not_started"]:
                        if progress.downloaded_bytes_this_session.in_bytes == 0:
                            status = DownloadPending(
                                node_id=self.node_id, shard_metadata=progress.shard
                            )
                        else:
                            status = DownloadOngoing(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                download_progress=map_repo_download_progress_to_download_progress_data(
                                    progress
                                ),
                            )
                    else:
                        continue

                    self.download_status[progress.shard.model_meta.model_id] = status
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=status)
                    )
                logger.info("Done emitting existing download progress.")
                await anyio.sleep(5 * 60)  # 5 minutes
        except Exception as e:
            logger.error(f"Error emitting existing download progress: {e}")

    async def _handle_device_download_model(self, command: DeviceDownloadModel):
        logger.info(f"Received request to download model: {command.model_id}")
        try:
            from exo.worker.download.impl_shard_downloader import build_full_shard
            shard = await build_full_shard(command.model_id)
            self._tg.start_soon(self.shard_downloader.ensure_shard, shard)
        except Exception as e:
            logger.error(f"Failed to start download for {command.model_id}: {e}")

    async def _handle_delete_model(self, command: DeleteModel):
        logger.info(f"Received request to delete model: {command.model_id}")
        try:
            await self.shard_downloader.delete_model(command.model_id)
            if command.model_id in self.download_status:
                del self.download_status[command.model_id]
            await self.event_sender.send(
                NodeDownloadRemoved(node_id=self.node_id, model_id=command.model_id)
            )
        except Exception as e:
            logger.error(f"Failed to delete model {command.model_id}: {e}")

    async def _command_processor(self):
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                command = forwarder_command.command
                if isinstance(command, CheckShardPresent):
                    self._handle_check_shard_present(forwarder_command.origin, command)
                elif isinstance(command, ShardPresent):
                    self._handle_shard_present(forwarder_command.origin, command)
                elif isinstance(command, DeviceDownloadModel):
                    await self._handle_device_download_model(command)
                elif isinstance(command, DeleteModel):
                    await self._handle_delete_model(command)

    def _handle_check_shard_present(self, origin: NodeId, command: CheckShardPresent):
        if origin == self.node_id:
            return
        has_it = self.has_shard(ModelId(command.model_id))
        logger.debug(f"Received CheckShardPresent from {origin} for {command.model_id}. has_it={has_it}")
        if has_it:
            port = self.file_server.port
            if port == 0:
                logger.warning("FileServer port is 0, cannot respond to CheckShardPresent")
                return
            self.command_sender.send_nowait(
                ForwarderCommand(
                    origin=self.node_id,
                    command=ShardPresent(
                        model_id=command.model_id,
                        base_url=f"http://placeholder:{port}",
                        request_command_id=command.command_id,
                    ),
                )
            )

    def _handle_shard_present(self, origin: NodeId, command: ShardPresent):
        # find connection to origin
        best_conn = None
        for _, conn in self.state.topology.out_edges(self.node_id):
            if conn.send_back_node_id == origin:
                if self._is_connection_thunderbolt(conn):
                    best_conn = conn
                    break
                if best_conn is None:
                    best_conn = conn

        if best_conn:
            ip = best_conn.send_back_multiaddr.ip_address or "127.0.0.1"
            port = command.base_url.split(":")[-1]  # hacky parsing
            url = f"http://{ip}:{port}"
            self.peer_locations[ModelId(command.model_id)] = url
            logger.info(
                f"Discovered peer for {command.model_id} at {url} (from {origin}). TB={self._is_connection_thunderbolt(best_conn)}"
            )
            return

        logger.warning(f"Received ShardPresent from {origin} but no connection found")

    def _is_connection_thunderbolt(self, conn: Connection) -> bool:
        if conn.is_thunderbolt():
            return True
        
        # Fallback: Check node profile for interface metadata
        profile = self.state.topology.get_node_profile(conn.send_back_node_id)
        if profile:
            target_ip = conn.send_back_multiaddr.ip_address
            for iface in profile.network_interfaces:
                if iface.ip_address == target_ip and iface.is_thunderbolt:
                    return True
        return False

    def has_shard(self, model_id: ModelId) -> bool:
        status = self.download_status.get(model_id)
        return status is not None and isinstance(status, DownloadCompleted)

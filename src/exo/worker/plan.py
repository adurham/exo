# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.tasks import (
    ConnectToGroup,
    CreateRunner,
    DownloadModel,
    ImageEdits,
    ImageGeneration,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgress,
)
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerId,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.shared.constants import EXO_FILE_SERVER_PORT
from exo.shared.topology import Topology
from exo.shared.types.profiling import NodeNetworkInfo
from exo.shared.types.topology import SocketConnection
from exo.worker.runner.runner_supervisor import RunnerSupervisor


def _get_best_peer_ip(
    node_id: NodeId,
    peer_id: NodeId,
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> str | None:
    other_network = node_network.get(peer_id, NodeNetworkInfo())
    
    # Get all IPv4 addresses from the peer's network info
    ips = [
        iface.ip_address 
        for iface in other_network.interfaces 
        if iface.ip_address and ":" not in iface.ip_address
    ]
    
    if not ips:
        # Fallback to topology connections if node_network is empty
        connections = topology.get_all_connections_between(node_id, peer_id)
        ips = [
            conn.sink_multiaddr.ip_address 
            for conn in connections 
            if isinstance(conn, SocketConnection)
        ]
        
    if not ips:
        return None

    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }
    
    # Priority: Interface Type (Thunderbolt=0)
    priority = {
        "thunderbolt": 0,
        "ethernet": 1,
        "maybe_ethernet": 2,
        "wifi": 3,
        "unknown": 4,
    }
    
    def get_priority(ip: str) -> int:
        return priority.get(ip_to_type.get(ip, "unknown"), 4)

    return min(ips, key=get_priority)


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    input_chunk_buffer: Mapping[CommandId, dict[int, str]] | None = None,
    input_chunk_counts: Mapping[CommandId, int] | None = None,
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        _kill_runner(runners, all_runners, instances)
        or _create_runner(node_id, runners, instances)
        or _model_needs_download(node_id, runners, global_download_status, instances, topology, node_network)
        or _init_distributed_backend(runners, all_runners)
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners, input_chunk_buffer)
    )


def _kill_runner(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    instances: Mapping[InstanceId, Instance],
) -> Shutdown | None:
    for runner in runners.values():
        runner_id = runner.bound_instance.bound_runner_id
        if (instance_id := runner.bound_instance.instance.instance_id) not in instances:
            return Shutdown(instance_id=instance_id, runner_id=runner_id)

        for (
            global_runner_id
        ) in runner.bound_instance.instance.shard_assignments.node_to_runner.values():
            if runner_id == global_runner_id:
                continue

            if isinstance(all_runners.get(global_runner_id, None), RunnerFailed):
                return Shutdown(
                    instance_id=instance_id,
                    runner_id=runner_id,
                )


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    instances: Mapping[InstanceId, Instance],
) -> CreateRunner | None:
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            continue

        if runner_id in runners:
            continue

        shard = instance.shard(runner_id)
        assert shard is not None

        return CreateRunner(
            instance_id=instance.instance_id,
            bound_instance=BoundInstance(
                instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
            ),
        )


def _model_needs_download(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    topology: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> DownloadModel | None:
    local_downloads = global_download_status.get(node_id, [])
    download_status = {
        dp.shard_metadata.model_card.model_id: dp for dp in local_downloads
    }

    for runner in runners.values():
        shard = runner.bound_instance.bound_shard
        model_id = shard.model_card.model_id
        
        # Check if we already have the model or are downloading it
        if not isinstance(runner.status, RunnerIdle):
            continue
            
        if model_id in download_status and isinstance(
            download_status[model_id],
            (DownloadOngoing, DownloadCompleted, DownloadFailed),
        ):
             # We don't invalidate download_status randomly in case a file gets deleted on disk
             # But if it failed, we might want to retry? For now, logic stays same: if failed, we don't auto-retry immediately here to avoid loops
             continue

        # Logic for P2P
        instance = runner.bound_instance.instance
        if not instance:
             # Should not happen
             return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=shard,
            )

        # 1. Determine Leader
        participating_nodes = sorted(instance.shard_assignments.node_to_runner.keys())
        leader_node_id = participating_nodes[0]
        
        # 2. Check if anyone has it
        peer_with_model: NodeId | None = None
        for peer_id in participating_nodes:
            if peer_id == node_id:
                continue
            peer_downloads = global_download_status.get(peer_id, [])
            if any(
                isinstance(dp, DownloadCompleted) and dp.shard_metadata.model_card.model_id == model_id
                for dp in peer_downloads
            ):
                peer_with_model = peer_id
                break
        
        # 3. Construct repo_url if we found a peer
        repo_url: str | None = None
        if peer_with_model:
            best_ip = _get_best_peer_ip(node_id, peer_with_model, topology, node_network)
            if best_ip:
                 repo_url = f"http://{best_ip}:{EXO_FILE_SERVER_PORT}"
        
        # 4. Decide action
        if repo_url:
             # Download from peer
             return DownloadModel(
                instance_id=instance.instance_id,
                shard_metadata=shard,
                repo_url=repo_url
            )
        
        if node_id == leader_node_id:
             # I am leader, and no one else has it -> Download from HF
             return DownloadModel(
                instance_id=instance.instance_id,
                shard_metadata=shard,
            )
        else:
             # I am follower, and leader (or anyone) doesn't have it yet -> Wait
             continue

    return None


def _init_distributed_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
):
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        is_single_node_instance = len(shard_assignments.runner_to_shard) == 1
        if is_single_node_instance:
            continue

        runner_is_idle = isinstance(runner.status, RunnerIdle)
        all_runners_connecting = all(
            isinstance(
                all_runners.get(global_runner_id),
                (RunnerConnecting, RunnerIdle),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if not (runner_is_idle and all_runners_connecting):
            continue

        runner_id = runner.bound_instance.bound_runner_id

        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        world_size = shard.world_size

        assert device_rank < world_size
        assert device_rank >= 0

        accepting_ranks = device_rank < world_size - 1

        # Rank = n-1
        connecting_rank_ready = device_rank == world_size - 1 and all(
            isinstance(all_runners.get(global_runner_id, None), RunnerConnecting)
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if not (accepting_ranks or connecting_rank_ready):
            continue

        return ConnectToGroup(instance_id=instance.instance_id)

    return None


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        all_local_downloads_complete = all(
            nid in global_download_status
            and any(
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata.model_card.model_id == shard_assignments.model_id
                for dp in global_download_status[nid]
            )
            for nid in shard_assignments.node_to_runner
        )
        if not all_local_downloads_complete:
            continue

        is_single_node_instance = len(instance.shard_assignments.runner_to_shard) == 1
        if is_single_node_instance and isinstance(runner.status, RunnerIdle):
            return LoadModel(instance_id=instance.instance_id)

        is_runner_waiting = isinstance(runner.status, RunnerConnected)

        all_ready_for_model = all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerConnected, RunnerLoading, RunnerLoaded),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if is_runner_waiting and all_ready_for_model:
            return LoadModel(instance_id=instance.instance_id)

    return None


def _ready_to_warmup(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> StartWarmup | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments
        shard = runner.bound_instance.bound_shard
        device_rank = shard.device_rank
        runner_id = runner.bound_instance.bound_runner_id
        world_size = shard.world_size

        is_runner_loaded = isinstance(runner.status, RunnerLoaded)

        assert device_rank < world_size
        assert device_rank >= 0

        # Rank != 0
        accepting_ranks_ready = device_rank > 0 and all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerLoaded, RunnerWarmingUp),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        # Rank = 0
        connecting_rank_ready = device_rank == 0 and all(
            isinstance(all_runners.get(global_runner_id, None), RunnerWarmingUp)
            for global_runner_id in shard_assignments.runner_to_shard
            if global_runner_id != runner_id
        )

        if is_runner_loaded and (accepting_ranks_ready or connecting_rank_ready):
            return StartWarmup(instance_id=instance.instance_id)

    return None


def _pending_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
    all_runners: Mapping[RunnerId, RunnerStatus],
    input_chunk_buffer: Mapping[CommandId, dict[int, str]] | None = None,
) -> Task | None:
    for task in tasks.values():
        # for now, just forward chat completions
        # TODO(ciaran): do this better!
        if not isinstance(task, (TextGeneration, ImageGeneration, ImageEdits)):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        # For ImageEdits tasks, verify all input chunks have been received
        if isinstance(task, ImageEdits) and task.task_params.total_input_chunks > 0:
            cmd_id = task.command_id
            expected = task.task_params.total_input_chunks
            received = len((input_chunk_buffer or {}).get(cmd_id, {}))
            if received < expected:
                continue  # Wait for all chunks to arrive

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            # I have a design point here; this is a state race in disguise as the task status doesn't get updated to completed fast enough
            # however, realistically the task status should be set to completed by the LAST runner, so this is a true race
            # the actual solution is somewhat deeper than this bypass - TODO!
            if task.task_id in runner.completed:
                continue

            # TODO: Check ordering aligns with MLX distributeds expectations.

            if isinstance(runner.status, RunnerReady) and all(
                isinstance(all_runners[global_runner_id], (RunnerReady, RunnerRunning))
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            ):
                return task

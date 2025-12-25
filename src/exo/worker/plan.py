# pyright: reportUnusedImport = false

from collections.abc import Mapping, Sequence

from exo.shared.types.common import NodeId
from exo.shared.types.tasks import (
    ChatCompletion,
    CreateRunner,
    DownloadModel,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWaitingForModel,
    RunnerWarmingUp,
)
from exo.shared.types.worker.shards import ShardMetadata
from exo.worker.runner.runner_supervisor import RunnerSupervisor


def plan(
    node_id: NodeId,
    # Runners is expected to be FRESH and so should not come from state
    runners: Mapping[RunnerId, RunnerSupervisor],
    # DL_status is expected to be FRESH and so should not come from state
    download_status: Mapping[ShardMetadata, DownloadProgress],
    # gdls is not expected to be fresh
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    instances: Mapping[InstanceId, Instance],
    all_runners: Mapping[RunnerId, RunnerStatus],  # all global
    tasks: Mapping[TaskId, Task],
) -> Task | None:
    # Python short circuiting OR logic should evaluate these sequentially.
    return (
        _kill_runner(runners, all_runners, instances)
        or _create_runner(node_id, runners, instances)
        or _model_needs_download(runners, download_status)
        or _load_model(runners, all_runners, global_download_status)
        or _ready_to_warmup(runners, all_runners)
        or _pending_tasks(runners, tasks, all_runners)
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
    from loguru import logger
    
    for instance in instances.values():
        runner_id = instance.shard_assignments.node_to_runner.get(node_id, None)
        if runner_id is None:
            logger.debug(
                f"_create_runner: Instance {instance.instance_id} has no runner for node {node_id}. "
                f"node_to_runner: {instance.shard_assignments.node_to_runner}"
            )
            continue

        if runner_id in runners:
            logger.debug(
                f"_create_runner: Runner {runner_id} already exists in runners for instance {instance.instance_id}"
            )
            continue

        shard = instance.shard(runner_id)
        assert shard is not None

        logger.info(
            f"_create_runner: Creating runner {runner_id} for instance {instance.instance_id} on node {node_id}"
        )
        return CreateRunner(
            instance_id=instance.instance_id,
            bound_instance=BoundInstance(
                instance=instance, bound_runner_id=runner_id, bound_node_id=node_id
            ),
        )
    
    logger.debug(
        f"_create_runner: No runner to create. instances={list(instances.keys())}, "
        f"runners={list(runners.keys())}, node_id={node_id}"
    )
    return None


def _model_needs_download(
    runners: Mapping[RunnerId, RunnerSupervisor],
    download_status: Mapping[ShardMetadata, DownloadProgress],
) -> DownloadModel | None:
    for runner in runners.values():
        if (
            isinstance(runner.status, RunnerWaitingForModel)
            and runner.bound_instance.bound_shard not in download_status
        ):
            # We don't invalidate download_status randomly in case a file gets deleted on disk
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )


""" --- TODO!
def _init_backend(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> LoadModel | None:
    for runner in runner.values()
    pass
"""


def _load_model(
    runners: Mapping[RunnerId, RunnerSupervisor],
    all_runners: Mapping[RunnerId, RunnerStatus],
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> LoadModel | None:
    for runner in runners.values():
        instance = runner.bound_instance.instance
        shard_assignments = instance.shard_assignments

        all_downloads_complete_local = all(
            nid in global_download_status
            and any(
                isinstance(dp, DownloadCompleted)
                and dp.shard_metadata == shard_assignments.runner_to_shard[rid]
                for dp in global_download_status[nid]
            )
            for nid, rid in shard_assignments.node_to_runner.items()
        )

        runner_is_waiting = isinstance(runner.status, RunnerWaitingForModel)

        all_runners_expecting_model = all(
            isinstance(
                all_runners.get(global_runner_id),
                (RunnerWaitingForModel, RunnerLoading, RunnerLoaded),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        if (
            all_downloads_complete_local
            and runner_is_waiting
            and all_runners_expecting_model
        ):
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

        # Rank != n-1 (accepting ranks - all layers before the last)
        # These can start warmup when all runners are Loaded or WarmingUp
        accepting_ranks_ready = device_rank != world_size - 1 and all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerLoaded, RunnerWarmingUp),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        # Rank = n-1 (connecting rank - the last rank that closes the pipeline)
        # This rank should start warmup only after all other ranks are already WarmingUp
        # Exception: If this is NOT rank 0 (i.e., device_rank != 0), allow initial warmup when all are Loaded
        # This handles the case where world_size=2 and rank 1 is both the connecting rank and a non-zero rank
        connecting_rank_ready = device_rank == world_size - 1 and (
            # Case 1: All other runners are already WarmingUp (normal case)
            all(
                isinstance(all_runners.get(global_runner_id, None), RunnerWarmingUp)
                for global_runner_id in shard_assignments.runner_to_shard
                if global_runner_id != runner_id
            )
            # Case 2: Initial warmup for non-zero connecting rank - all runners are Loaded
            # Rank 0 should never use this path (it should wait for others to warmup first)
            or (device_rank != 0 and all(
                isinstance(
                    all_runners.get(global_runner_id, None),
                    RunnerLoaded,
                )
                for global_runner_id in shard_assignments.runner_to_shard
            ))
        )

        if is_runner_loaded and (accepting_ranks_ready or connecting_rank_ready):
            return StartWarmup(instance_id=instance.instance_id)

    return None


def _pending_tasks(
    runners: Mapping[RunnerId, RunnerSupervisor],
    tasks: Mapping[TaskId, Task],
    all_runners: Mapping[RunnerId, RunnerStatus],
) -> Task | None:
    for task in tasks.values():
        # for now, just forward chat completions
        if not isinstance(task, ChatCompletion):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue

        for runner in runners.values():
            if task.instance_id != runner.bound_instance.instance.instance_id:
                continue

            if isinstance(runner.status, RunnerReady) and all(
                isinstance(all_runners[global_runner_id], (RunnerReady, RunnerRunning))
                for global_runner_id in runner.bound_instance.instance.shard_assignments.runner_to_shard
            ):
                return task

"""Core planning logic for the Worker component.

This module provides the plan function, which acts as a state machine decision
maker. It evaluates the current state of runners, downloads, instances, and
tasks to determine the next Task the worker should execute. The planning logic
handles tasks such as creating runners, starting downloads, loading models,
coordinating warmup sequences, and shutting down runners when necessary.
"""

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
    """Determine if a runner should be shut down.

    Checks for runners whose instances have been deleted or whose peers
    have failed, requiring shutdown.

    Args:
        runners: Local runner supervisors.
        all_runners: Global runner statuses.
        instances: Current instance configurations.

    Returns:
        Shutdown task if a runner should be killed, None otherwise.
    """
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
    return None


def _create_runner(
    node_id: NodeId,
    runners: Mapping[RunnerId, RunnerSupervisor],
    instances: Mapping[InstanceId, Instance],
) -> CreateRunner | None:
    """Determine if a new runner should be created.

    Checks for instances assigned to this node that don't have a runner yet.

    Args:
        node_id: Node ID of this worker.
        runners: Local runner supervisors.
        instances: Instance configurations.

    Returns:
        CreateRunner task if a runner should be created, None otherwise.
    """
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
    return None


def _model_needs_download(
    runners: Mapping[RunnerId, RunnerSupervisor],
    download_status: Mapping[ShardMetadata, DownloadProgress],
) -> DownloadModel | None:
    """Determine if a model shard needs to be downloaded.

    Checks for runners waiting for models that don't have download status
    (indicating download hasn't started).

    Args:
        runners: Local runner supervisors.
        download_status: Current download progress.

    Returns:
        DownloadModel task if a download should start, None otherwise.

    Note:
        download_status is not invalidated when files are deleted on disk,
        so this only checks if download has started, not if it completed.
    """
    for runner in runners.values():
        if (
            isinstance(runner.status, RunnerWaitingForModel)
            and runner.bound_instance.bound_shard not in download_status
        ):
            return DownloadModel(
                instance_id=runner.bound_instance.instance.instance_id,
                shard_metadata=runner.bound_instance.bound_shard,
            )
    return None


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
    """Determine if a model should be loaded into a runner.

    Checks if all downloads for an instance are complete globally, the runner
    is waiting for the model, and all runners for the instance are expecting
    the model (in waiting/loading/loaded states).

    Args:
        runners: Local runner supervisors.
        all_runners: Global runner statuses.
        global_download_status: Download status from all nodes.

    Returns:
        LoadModel task if model should be loaded, None otherwise.
    """
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
    """Determine if a runner is ready to start warmup.

    For pipeline parallelism, warmup coordination depends on device rank:
    - Non-last ranks (0 to n-2): Start when all ranks are loaded or warming up
    - Last rank (n-1): Start when all other ranks are already warming up

    This ensures proper initialization order for distributed models.

    Args:
        runners: Local runner supervisors.
        all_runners: Global runner statuses.

    Returns:
        StartWarmup task if warmup should start, None otherwise.
    """
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

        accepting_ranks_ready = device_rank != world_size - 1 and all(
            isinstance(
                all_runners.get(global_runner_id, None),
                (RunnerLoaded, RunnerWarmingUp),
            )
            for global_runner_id in shard_assignments.runner_to_shard
        )

        connecting_rank_ready = device_rank == world_size - 1 and all(
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
) -> Task | None:
    """Determine if there are pending tasks ready to execute.

    Finds tasks that are pending and whose runners are ready (RunnerReady
    or RunnerRunning state).

    Args:
        runners: Local runner supervisors.
        tasks: Pending tasks.
        all_runners: Global runner statuses.

    Returns:
        A pending task if one is ready to execute, None otherwise.
    """
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

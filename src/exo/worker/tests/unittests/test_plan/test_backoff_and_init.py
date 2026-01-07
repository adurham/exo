
import time
from typing import Any

import exo.worker.plan as plan_mod
from exo.shared.types.tasks import ConnectToGroup, CreateRunner
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from exo.shared.types.worker.instances import BoundInstance, Instance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerConnecting,
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerStatus,
)
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

def test_plan_backoff_prevents_creation():
    """
    If a runner failed recently, plan() should NOT emit CreateRunner.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )

    runners: dict[Any, Any] = {}
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[Any, Any] = {}
    
    # Simulate a failure 1 second ago
    failure_history = {RUNNER_1_ID: (time.time() - 1, 1)}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        download_status={},
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        runner_failure_history=failure_history,
    )

    assert result is None

def test_plan_backoff_allows_creation_after_wait():
    """
    If a runner failed long ago, plan() SHOULD emit CreateRunner.
    """
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID},
        runner_to_shard={RUNNER_1_ID: shard},
    )

    runners: dict[Any, Any] = {}
    instances = {INSTANCE_1_ID: instance}
    all_runners: dict[Any, Any] = {}
    
    # Simulate a failure 100 seconds ago (backoff for count 1 is 5s)
    failure_history = {RUNNER_1_ID: (time.time() - 100, 1)}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        download_status={},
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        runner_failure_history=failure_history,
    )

    assert isinstance(result, CreateRunner)

def test_init_backend_waits_for_download():
    """
    If model is not fully downloaded, _init_distributed_backend should NOT return ConnectToGroup.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    # Runner is Idle, waiting to connect. Peer is connecting.
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())
    
    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerIdle(),
        RUNNER_2_ID: RunnerConnecting(),
    }
    
    download_status = {MODEL_A_ID: DownloadPending(shard_metadata=shard1, node_id=NODE_A)}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        download_status=download_status,
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )
    
    # Should NOT be ConnectToGroup
    assert not isinstance(result, ConnectToGroup)

def test_init_backend_connects_when_downloaded():
    """
    If model IS downloaded, _init_distributed_backend SHOULD return ConnectToGroup.
    """
    shard1 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=1, world_size=2)
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard1, RUNNER_2_ID: shard2},
    )
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RUNNER_1_ID, bound_node_id=NODE_A
    )
    runner = FakeRunnerSupervisor(bound_instance=bound_instance, status=RunnerIdle())
    
    runners = {RUNNER_1_ID: runner}
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        RUNNER_1_ID: RunnerIdle(),
        RUNNER_2_ID: RunnerConnecting(),
    }
    
    download_status = {MODEL_A_ID: DownloadCompleted(shard_metadata=shard1, node_id=NODE_A)}

    result = plan_mod.plan(
        node_id=NODE_A,
        runners=runners,
        download_status=download_status,
        global_download_status={NODE_A: []},
        instances=instances,
        all_runners=all_runners,
        tasks={},
    )
    
    assert isinstance(result, ConnectToGroup)

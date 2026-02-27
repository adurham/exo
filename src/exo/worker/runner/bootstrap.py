import os
import sys

# Critical setup BEFORE other imports
# Valid values: "1", "true", "on", "yes", "y"
os.environ["IBV_FORK_SAFE"] = "1"

# Flush to ensure debug output is seen
sys.stdout.flush()

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    global logger
    logger = _logger

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override == "on" or (
        fast_synch_override != "off"
        and (
            isinstance(bound_instance.instance, MlxJacclInstance)
            and len(bound_instance.instance.jaccl_devices) >= 2
        )
    ):
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    # IBV_FORK_SAFE is set at module level now


    # Add file logging for debugging
    _logger.add(f"/tmp/exo_runner_debug_{bound_instance.bound_runner_id}.log", level="DEBUG", enqueue=True)
    logger = _logger
    
    # Debug: Log environment and limits
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.debug(f"Process limits - NOFILE: soft={soft}, hard={hard}")
    try:
        soft_mem, hard_mem = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        logger.debug(f"Process limits - MEMLOCK: soft={soft_mem}, hard={hard_mem}")
    except Exception as e:
        logger.debug(f"Could not get RLIMIT_MEMLOCK: {e}")
    
    # Attempt to raise limit strictly
    try:
        new_soft = max(soft, 10240)
        new_hard = max(hard, 10240)
        # MacOS hard limit is often unlimited (inf), handled by resource module usually
        # But we must respect the physical hard limit if set
        if hard != resource.RLIM_INFINITY and new_soft > hard:
             new_soft = hard
        
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, getattr(resource, 'RLIM_INFINITY', new_hard)))
        logger.info(f"Updated RLIMIT_NOFILE to {resource.getrlimit(resource.RLIMIT_NOFILE)}")
    except Exception as e:
        logger.warning(f"Failed to raise RLIMIT_NOFILE: {e}")

    # Set MLX wired limit early
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            max_rec_size = int(mx.metal.device_info()["max_recommended_working_set_size"])
            ratio = float(os.getenv("EXO_MLX_WIRED_LIMIT_RATIO", "0.75"))
            safe_limit = int(max_rec_size * ratio)
            mx.set_wired_limit(safe_limit)
            logger.info(f"Bootstrapped MLX wired limit in entrypoint to {safe_limit / 1e9:.2f} GB ({ratio * 100:.0f}%)")
    except Exception as e:
        logger.warning(f"Failed to set MLX wired limit: {e}")



    logger.debug(f"Environment: {os.environ}")

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import main
        else:
            from exo.worker.runner.llm_inference.runner import main

        main(bound_instance, event_sender, task_receiver, cancel_receiver)

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")

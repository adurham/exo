"""Bootstrap entrypoint for runner processes.

This module provides the entrypoint function that initializes the runner
process, sets up environment variables, and delegates to the main runner
function.
"""

import os

import loguru

from exo.shared.types.events import Event
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.utils.channels import MpReceiver, MpSender

logger: "loguru.Logger"
"""Global logger instance, set by entrypoint."""


if os.getenv("EXO_TESTS") == "1":
    logger = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
) -> None:
    """Entrypoint for runner process.

    Sets up environment variables (e.g., MLX_METAL_FAST_SYNCH for Jaccl
    instances) and delegates to the main runner function.

    Args:
        bound_instance: Instance this runner will handle.
        event_sender: Multiprocessing sender for events.
        task_receiver: Multiprocessing receiver for tasks.
        _logger: Logger instance to use.
    """
    if (
        isinstance(bound_instance.instance, MlxJacclInstance)
        and len(bound_instance.instance.ibv_devices) >= 2
    ):
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    global logger
    logger = _logger

    # Import main after setting global logger - this lets us just import logger from this module
    from exo.worker.runner.runner import main

    main(bound_instance, event_sender, task_receiver)

import os
from pathlib import Path

import loguru

from exo.shared.constants import EXO_LOG
from exo.shared.logging import logger_setup
from exo.shared.types.events import Event
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.utils.channels import MpReceiver, MpSender

logger: "loguru.Logger"


if os.getenv("EXO_TESTS") == "1":
    logger = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
) -> None:
    try:
        # Patch multiprocessing to handle BrokenPipeError when flushing stdout/stderr
        # This must be done early, before any multiprocessing operations
        import multiprocessing.util
        original_flush = multiprocessing.util._flush_std_streams
        def patched_flush_std_streams():
            try:
                original_flush()
            except (BrokenPipeError, OSError):
                pass
        multiprocessing.util._flush_std_streams = patched_flush_std_streams
        
        if (
            isinstance(bound_instance.instance, MlxJacclInstance)
            and len(bound_instance.instance.ibv_devices) >= 2
        ):
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"

        global logger
        verbosity = int(os.getenv("EXO_VERBOSITY", "0"))
        logger_setup(EXO_LOG, verbosity)
        logger = loguru.logger
        logger.info("Runner bootstrap: logging configured")

        # Import main after setting global logger - this lets us just import logger from this module
        from exo.worker.runner.runner import main

        logger.info("Runner bootstrap: calling main()")
        main(bound_instance, event_sender, task_receiver)
    except Exception as e:
        # Try to log the error if logger is available
        try:
            logger.error(f"Runner bootstrap error: {e}", exc_info=True)
        except:
            # Fallback to stderr if logger isn't set up
            import sys
            import traceback
            print(f"Runner bootstrap error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        raise

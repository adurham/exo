import logging
import sys
from pathlib import Path

from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger

# Patch multiprocessing to handle BrokenPipeError when flushing stdout/stderr
# This must be done at import time, before any multiprocessing operations
# This happens when stdout/stderr are redirected to /dev/null or closed
try:
    import multiprocessing.util
    original_flush = multiprocessing.util._flush_std_streams
    def patched_flush_std_streams():
        try:
            original_flush()
        except (BrokenPipeError, OSError):
            pass
    multiprocessing.util._flush_std_streams = patched_flush_std_streams
except Exception:
    # If patching fails, continue anyway - this is a best-effort fix
    pass


class InterceptLogger(HypercornLogger):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.error_logger
        # TODO: Decide if we want to provide access logs
        # assert self.access_logger
        # self.access_logger.handlers = [_InterceptHandler()]
        self.error_logger.handlers = [_InterceptHandler()]


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=3, exception=record.exc_info).log(level, record.getMessage())


def logger_setup(log_file: Path | None, verbosity: int = 0):
    """Set up logging for this process - formatting, file handles, verbosity and output"""
    logger.remove()

    # replace all stdlib loggers with _InterceptHandlers that log to loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0)

    if verbosity == 0:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:hh:mm:ss.SSSSA} | <level>{level: <8}</level>] <level>{message}</level>",
            level="INFO",
            colorize=True,
            enqueue=True,
        )
    else:
        logger.add(
            sys.__stderr__,  # type: ignore
            format="[ {time:HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} ] <level>{message}</level>",
            level="DEBUG",
            colorize=True,
            enqueue=True,
        )
    if log_file:
        # File handler level depends on verbosity: DEBUG if verbose, INFO otherwise
        file_level = "DEBUG" if verbosity > 0 else "INFO"
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level=file_level,
            colorize=False,
            enqueue=True,
            rotation="1 week",
        )


def logger_cleanup():
    """Flush all queues before shutting down so any in-flight logs are written to disk"""
    logger.complete()


""" --- TODO: Capture MLX Log output:
import contextlib
import sys
from loguru import logger

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass

logger.remove()
logger.add(sys.__stdout__)

stream = StreamToLogger()
with contextlib.redirect_stdout(stream):
    print("Standard output is sent to added handlers.")
"""

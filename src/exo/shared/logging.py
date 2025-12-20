"""Logging configuration and utilities.

This module provides logging setup that integrates loguru with Python's standard
logging library and Hypercorn's logging system.
"""

import logging
import sys
from pathlib import Path

from hypercorn import Config
from hypercorn.logging import Logger as HypercornLogger
from loguru import logger


class InterceptLogger(HypercornLogger):
    """Logger that intercepts Hypercorn logs and routes them to loguru.

    Replaces Hypercorn's default logging handlers with loguru handlers.
    """

    def __init__(self, config: Config):
        """Initialize the intercept logger.

        Args:
            config: Hypercorn configuration.
        """
        super().__init__(config)
        assert self.error_logger
        self.error_logger.handlers = [_InterceptHandler()]


class _InterceptHandler(logging.Handler):
    """Handler that routes standard library logs to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to loguru.

        Args:
            record: Log record to emit.
        """
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=3, exception=record.exc_info).log(level, record.getMessage())


def logger_setup(log_file: Path | None, verbosity: int = 0) -> None:
    """Set up logging for this process.

    Configures loguru with appropriate formatting, file handles, verbosity,
    and output destinations. Replaces all existing loguru handlers and sets up
    interception of standard library logging.

    Args:
        log_file: Optional path to log file. If provided, logs are written
            to this file with rotation (weekly). If None, no file logging.
        verbosity: Verbosity level. 0 = INFO, >0 = DEBUG. Negative values
            may reduce verbosity further.
    """
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
        logger.add(
            log_file,
            format="[ {time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} ] {message}",
            level="INFO",
            colorize=False,
            enqueue=True,
            rotation="1 week",
        )


def logger_cleanup() -> None:
    """Flush all log queues before shutdown.

    Ensures all in-flight logs are written to disk before the process exits.
    Should be called during graceful shutdown.
    """
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

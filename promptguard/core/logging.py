"""Logging utilities for PromptGuard."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

# Package logger with NullHandler (silent by default)
_package_logger = logging.getLogger("promptguard")
_package_logger.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: The module name (typically __name__).

    Returns:
        A logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing request")
    """
    return logging.getLogger(name)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for production environments.

    Produces structured log output suitable for log aggregation
    systems like Elasticsearch, Datadog, or CloudWatch.

    Example output:
        {"timestamp": "2024-01-01 12:00:00", "level": "INFO", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add custom fields if present
        for key in ("model", "provider", "attempt", "duration_ms"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(
    level: str = "INFO",
    format_style: str = "text",
    stream: Any = None,
) -> None:
    """Configure PromptGuard logging (call once at app startup).

    By default, PromptGuard logs nothing (NullHandler). Call this function
    to enable logging output.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR").
            DEBUG shows detailed request/response info.
            INFO shows repair attempts, successful operations.
            WARNING shows validation failures, retries needed.
            ERROR shows provider errors, repair exhaustion.
        format_style: Either "text" for human-readable or "json" for
            structured logging suitable for log aggregation.
        stream: Output stream (defaults to sys.stderr).

    Example:
        # Enable debug logging to stderr
        configure_logging(level="DEBUG")

        # Enable JSON logging for production
        configure_logging(level="INFO", format_style="json")

        # Log to a file
        with open("promptguard.log", "w") as f:
            configure_logging(stream=f)
    """
    if stream is None:
        stream = sys.stderr

    logger = logging.getLogger("promptguard")

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)

    if format_style == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)  # Let handler control the level

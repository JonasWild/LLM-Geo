"""Concise console and durable file logging for LLM-GEO."""

from __future__ import annotations

import logging
import os
from pathlib import Path


LOGGER_NAME = "llm_geo"
HTTP_LOGGER_LEVELS = {
    "httpx": logging.INFO,
    "urllib3.connectionpool": logging.DEBUG,
}


class _ForwardToLlmGeo(logging.Handler):
    """Forward dependency records through the currently active LLM-GEO handlers."""

    def emit(self, record: logging.LogRecord) -> None:
        get_logger().handle(record)


def _http_logging_enabled() -> bool:
    value = os.getenv("LLM_GEO_LOG_HTTP", "true").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "LLM_GEO_LOG_HTTP must be true/false, yes/no, on/off, or 1/0"
    )


def _configure_http_logging(enabled: bool) -> None:
    """Route HTTP client metadata into LLM-GEO without logging payload bodies."""
    for name, level in HTTP_LOGGER_LEVELS.items():
        dependency_logger = logging.getLogger(name)
        for handler in dependency_logger.handlers[:]:
            if isinstance(handler, _ForwardToLlmGeo):
                dependency_logger.removeHandler(handler)
        if enabled:
            dependency_logger.setLevel(level)
            dependency_logger.addHandler(_ForwardToLlmGeo())
            dependency_logger.propagate = False
        else:
            dependency_logger.setLevel(logging.WARNING)
            dependency_logger.propagate = True


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    *,
    log_http: bool | None = None,
) -> logging.Logger:
    """Configure meaningful LLM-GEO logs without noisy dependency output."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)

    for dependency in ("openai", "rasterio", "fiona", "matplotlib"):
        logging.getLogger(dependency).setLevel(logging.WARNING)
    _configure_http_logging(
        _http_logging_enabled() if log_http is None else log_http
    )
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def close_file_logging() -> None:
    """Flush and release file handlers while keeping console logging active."""
    logger = get_logger()
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

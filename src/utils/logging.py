"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import sys

from src.config import get_settings


def setup_logging() -> None:
    """Configure logging for the application.

    Uses standard library logging with a clean format.
    Structlog can be swapped in later for JSON logging in production.
    """
    settings = get_settings()

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Quiet noisy libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.app_debug else logging.WARNING
    )

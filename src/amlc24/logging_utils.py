"""Logging setup shared by pipelines, scripts, and notebooks.

Everything inside ``src/`` logs rather than prints. Notebooks call
``setup_logging()`` once; the stream handler then makes those records visible in
the cell output, and ``add_file_handler`` additionally tees them into the run's
``log.txt`` so a finished run carries its own log.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
ROOT_LOGGER_NAME = "amlc24"


def setup_logging(level: int | str = logging.INFO, quiet_libraries: bool = True) -> logging.Logger:
    """Install a single stdout handler on the ``amlc24`` logger. Idempotent.

    Uses stdout rather than stderr so Jupyter renders log lines as normal output
    instead of red error blocks.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(h, "_amlc24_stream", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        handler.setLevel(level)
        handler._amlc24_stream = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    if quiet_libraries:
        for noisy in ("PIL", "urllib3", "matplotlib", "matplotlib.font_manager",
                      "filelock", "huggingface_hub", "datasets"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def add_file_handler(path: str | Path, level: int | str = logging.INFO) -> logging.Handler:
    """Tee ``amlc24`` log records into ``path``. Returns the handler so the
    caller can remove it when the run finishes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(p, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handler.setLevel(level)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.addHandler(handler)
    logger.info("Logging to file: %s", p)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    """Detach and close a handler added by ``add_file_handler``."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handler.flush()
    handler.close()
    if handler in logger.handlers:
        logger.removeHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``amlc24`` root.

    Modules normally use ``logging.getLogger(__name__)``; because they live in
    the ``amlc24`` package that already yields a child of the root logger. This
    helper exists for scripts outside the package.
    """
    if name == "__main__" or not name.startswith(ROOT_LOGGER_NAME):
        name = f"{ROOT_LOGGER_NAME}.{name.rsplit('.', 1)[-1]}"
    return logging.getLogger(name)


__all__ = ["setup_logging", "add_file_handler", "remove_handler", "get_logger"]

"""Structured logging to console and file, plus peak memory tracking."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil


def setup_logging(log_dir: Path, timestamp: str | None = None) -> logging.Logger:
    """Configure root/package logger writing to console and outputs/logs/run_<timestamp>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    log_file = log_dir / f"run_{timestamp}.log"
    logger = logging.getLogger("lupus_p1")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Initialized logging to %s", log_file)
    return logger


def get_current_rss_mb() -> float:
    """Return current process resident set size (RSS) in MiB."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def get_peak_memory_mb() -> float:
    """Return peak memory usage of the current process in MiB."""
    peak_mb = get_current_rss_mb()
    try:
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            ru_mb = float(rusage.ru_maxrss) / (1024.0 * 1024.0)
        else:
            ru_mb = float(rusage.ru_maxrss) / 1024.0
        peak_mb = max(peak_mb, ru_mb)
    except Exception:
        pass
    return peak_mb


def log_memory(logger: logging.Logger, context: str) -> None:
    """Log current RSS and peak RSS alongside system available memory."""
    curr_mb = get_current_rss_mb()
    peak_mb = get_peak_memory_mb()
    vm = psutil.virtual_memory()
    avail_mb = float(vm.available) / (1024.0 * 1024.0)
    logger.info(
        "[MEMORY] %s | current RSS: %.1f MB | peak RSS: %.1f MB | system available: %.1f MB",
        context,
        curr_mb,
        peak_mb,
        avail_mb,
    )

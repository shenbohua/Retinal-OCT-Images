from __future__ import annotations

"""Common utility helpers: seed, logging, timing, and JSON output."""

import hashlib
import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible numpy/python behavior."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create stream+file logger used by command handlers."""
    logger = logging.getLogger("oct_project")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA1 hash for duplicate-checking and data lineage."""
    sha1 = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def dump_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON with stable formatting for run configs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """Context manager that returns elapsed wall-clock seconds."""
    info: dict[str, float] = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - start

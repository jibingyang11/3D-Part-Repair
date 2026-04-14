"""General utility functions."""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from functools import wraps


def setup_logger(name: str, log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """Setup a logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] {elapsed:.3f}s")
        return result
    return wrapper


def save_json(data: dict, path: str):
    """Save dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_path(p: str) -> str:
    """Normalize path separators for cross-platform compatibility."""
    return str(Path(p))

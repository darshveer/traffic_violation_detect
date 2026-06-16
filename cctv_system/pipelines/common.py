"""Shared helpers: path resolution, config loading, logging and device selection.

This module intentionally has *no* heavy (torch / ultralytics) imports at module
load time so it can be imported cheaply from anywhere, including tests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Package root = the `cctv_system/` directory (parent of `pipelines/`).
PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent

_LOG_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logging once and return the project logger.

    Parameters
    ----------
    level : int
        Logging level (e.g. ``logging.INFO``).
    log_file : str, optional
        If given, also write logs to this file.

    Returns
    -------
    logging.Logger
        The ``cctv`` namespace logger.
    """
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        _LOG_CONFIGURED = True
    return logging.getLogger("cctv")


def resolve_path(path: Optional[str | os.PathLike], base: Optional[Path] = None) -> Optional[Path]:
    """Resolve ``path`` to an absolute :class:`Path`.

    Absolute paths are returned unchanged. Relative paths are resolved against
    ``base`` (default: the package root).

    Parameters
    ----------
    path : str or os.PathLike, optional
        The path to resolve. ``None`` is passed through as ``None``.
    base : Path, optional
        Base directory for relative paths. Defaults to :data:`PACKAGE_ROOT`.

    Returns
    -------
    Path or None
        The absolute path, or ``None`` if ``path`` was ``None``.
    """
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or PACKAGE_ROOT) / p


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the YAML file (absolute, or relative to the package root).

    Returns
    -------
    dict
        Parsed YAML content (empty dict if the file is empty).

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    resolved = resolve_path(path)
    assert resolved is not None
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def select_device(requested: str = "auto") -> str:
    """Pick a torch device string, falling back to CPU when CUDA is unavailable.

    Parameters
    ----------
    requested : str
        ``"auto"``, ``"cuda"``, ``"cuda:N"`` or ``"cpu"``.

    Returns
    -------
    str
        A concrete device string usable by torch / Ultralytics
        (``"cuda:0"`` / ``"cpu"`` ...).
    """
    logger = logging.getLogger("cctv")
    try:
        import torch  # local import keeps this module torch-free at import time
    except ImportError:  # pragma: no cover - torch always present at runtime
        logger.warning("torch not importable; defaulting device to 'cpu'.")
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    req = (requested or "auto").lower()

    if req == "cpu":
        return "cpu"
    if req == "auto":
        return "cuda:0" if cuda_ok else "cpu"
    # explicit cuda request
    if req.startswith("cuda"):
        if cuda_ok:
            return requested
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    logger.warning("Unknown device '%s'; falling back to CPU.", requested)
    return "cpu"

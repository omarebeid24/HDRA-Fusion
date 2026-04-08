"""
Logging and directory utilities for HDRA-Fusion.
"""

import os
import logging


def setup_logger(log_dir: str) -> logging.Logger:
    """
    Create a logger that writes to both a file and stdout.

    Parameters
    ----------
    log_dir : str
        Directory where run.log will be written.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("hdra_fusion")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def make_dirs(*dirs: str) -> None:
    """Create all provided directories, ignoring existing ones."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

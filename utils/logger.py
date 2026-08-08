"""
Logger Utility
==============

Creates a logger that writes to both the console and a log file.

Project:
    Semiconductor Image Restoration

Author:
    Yogeshwaran
"""

from pathlib import Path
import logging


def get_logger(
    name: str = "SemiconductorRestoration",
    log_dir: str | Path = "logs",
    log_file: str = "training.log",
) -> logging.Logger:
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name.

    log_dir : str | Path
        Directory to save log files.

    log_file : str
        Log filename.

    Returns
    -------
    logging.Logger
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File output
    file_handler = logging.FileHandler(
        log_dir / log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
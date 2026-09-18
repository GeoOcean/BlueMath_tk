import logging
import os
from datetime import datetime
from typing import Union

import pytz


def _coerce_level(level: Union[int, str]) -> int:
    """Return a numeric logging level from an int or level name."""
    if isinstance(level, int):
        return level
    name = str(level).upper()
    value = logging.getLevelName(name)
    if isinstance(value, int) and value != 0:
        return value
    raise ValueError(f"Unknown log level: {level!r}")


def _console_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]


def _file_handlers(logger: logging.Logger) -> list[logging.FileHandler]:
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]


def get_file_logger(
    name: str,
    logs_path: str = None,
    level: Union[int, str] = "INFO",
    console: bool = True,
    console_level: Union[int, str] = "WARNING",
) -> logging.Logger:
    """
    Creates and returns a logger that writes log messages to a file.

    Parameters
    ----------
    name : str
        The name of the logger.
    logs_path : str, optional
        The file path where the log messages will be written. Default is None.
    level : Union[int, str], optional
        The logging level for the logger and file handler. Default is "INFO".
    console : bool
        Whether to add or not console / terminal logs. Default is True.
    console_level : Union[int, str], optional
        The logging level for console / terminal logs. Default is "WARNING".

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    Safe to call more than once for the same *name*: existing handlers are
    updated (file level, console level, console on/off) instead of returning
    a stale configuration.

    Examples
    --------
    >>> from bluemath_tk.core.logging import get_file_logger
    >>> # Create a logger that writes to "app.log"
    >>> logger = get_file_logger("my_app_logger", "app.log")
    >>> # Log messages
    >>> logger.info("This is an info message.")
    >>> logger.warning("This is a warning message.")
    >>> logger.error("This is an error message.")
    >>> # The output will be saved in "app.log" with the format:
    >>> # 2023-10-22 14:55:23,456 - my_app_logger - INFO - This is an info message.
    >>> # 2023-10-22 14:55:23,457 - my_app_logger - WARNING - This is a warning message.
    >>> # 2023-10-22 14:55:23,458 - my_app_logger - ERROR - This is an error message.
    """

    file_level = _coerce_level(level)
    stream_level = _coerce_level(console_level)

    logger = logging.getLogger(name)
    logger.setLevel(file_level)
    logger.propagate = False  # Avoid duplicate logs via the root logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handlers = _file_handlers(logger)
    if file_handlers:
        for handler in file_handlers:
            handler.setLevel(file_level)
            if handler.formatter is None:
                handler.setFormatter(formatter)
    else:
        date_str = datetime.now(pytz.timezone("Europe/Madrid")).strftime("%Y-%m-%d")
        if logs_path is None:
            os.makedirs("logs", exist_ok=True)
            logs_path = os.path.join("logs", f"{name.strip()}_{date_str}.log")
        else:
            log_dir = os.path.dirname(logs_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(logs_path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    for handler in _console_handlers(logger):
        logger.removeHandler(handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(stream_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

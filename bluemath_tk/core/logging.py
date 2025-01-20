import os
from datetime import datetime
import pytz
import logging
from typing import Union


def get_file_logger(
    name: str, logs_path: str = None, level: Union[int, str] = "INFO"
) -> logging.Logger:
    """
    Creates and returns a logger that writes log messages to a file.

    Parameters
    ----------
    name : str
        The name of the logger.
    logs_path : str, optional
        The file path where the log messages will be written (default is None).
    level : Union[int, str], optional
        The logging level (default is "INFO").

    Returns
    -------
    logging.Logger
        Configured logger instance.

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

    # If a logger with the specified name already exists, return it
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    # Create a logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs

    # Get current date to append to logs_path
    date_str = datetime.now(pytz.timezone("Europe/Madrid")).strftime("%Y-%m-%d")

    # Create a file handler to write logs to the specified file
    if logs_path is None:
        os.makedirs("logs", exist_ok=True)
        logs_path = os.path.join("logs", f"{name.strip()}_{date_str}.log")
    else:
        os.makedirs(os.path.dirname(logs_path))
    file_handler = logging.FileHandler(logs_path)

    # Also ouput logs in the console
    console_handler = logging.StreamHandler()

    # Define a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

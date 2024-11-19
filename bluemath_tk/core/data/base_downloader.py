from abc import ABC
import logging
from ..logging import get_file_logger


class BlueMathDownloader(ABC):
    def __init__(self, base_path):
        self._base_path = base_path
        self._logger = get_file_logger(
            name="BlueMathDownloader",
            level=logging.INFO,
        )

    @property
    def base_path(self):
        return self._base_path

    @base_path.setter
    def base_path(self, value):
        self._base_path = value

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    def set_logger_name(self, name: str):
        """Sets the name of the logger."""
        self.logger = get_file_logger(name=name)

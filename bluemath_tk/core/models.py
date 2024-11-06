from abc import ABC, abstractmethod
from .utils import get_file_logger
import logging


class BlueMathModel(ABC):
    def __init__(self):
        self._logger = get_file_logger(name=self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    # @abstractmethod
    # def perform_action(self):
    #     """Abstract method to perform an action."""
    #     pass

    def __private_method(self):
        """Private method not accessible outside the class."""
        self.logger.info("This is a private method only used internally.")
        return "This is hidden"

    def _internal_use_only(self):
        """Protected method for internal use only."""
        self.logger.info("This is a protected method used for internal purposes.")
        return "Internal value"

    def log_and_raise_error(self, message):
        """Logs an error message and raises an exception."""
        self.logger.error(message)
        raise ValueError(message)

from abc import ABC, abstractmethod
from .logging import get_file_logger
from .data import normalize, denormalize
import logging
import pandas as pd
from typing import Tuple


class BlueMathModel(ABC):
    def __init__(self):
        self._logger = get_file_logger(name=self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    def set_logger_name(self, name: str):
        """Sets the name of the logger."""
        self.logger = get_file_logger(name=name)

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

    def normalize(
        self, data: pd.DataFrame, custom_scale_factor: dict = {}
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize data to 0-1 using min max scaler approach
        """
        self.logger.info("Normalizing data to range 0-1 using min max scaler approach")
        normalized_data, scale_factor = normalize(
            data=data, custom_scale_factor=custom_scale_factor, logger=self.logger
        )
        return normalized_data, scale_factor

    def denormalize(
        self, normalized_data: pd.DataFrame, scale_factor: dict
    ) -> pd.DataFrame:
        """
        Denormalize data using provided scale_factor
        """
        return denormalize(normalized_data=normalized_data, scale_factor=scale_factor)

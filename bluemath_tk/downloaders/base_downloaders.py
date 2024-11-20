from abc import ABC, abstractmethod
import logging
from ..core.logging import get_file_logger


class BlueMathDownloader(ABC):
    def __init__(
        self, base_path_to_download: str, debug: bool = True, check: bool = False
    ) -> None:
        if not isinstance(base_path_to_download, str):
            raise ValueError("base_path_to_download must be a string")
        self._base_path_to_download: str = base_path_to_download
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")
        self._debug: bool = debug
        if not isinstance(check, bool):
            raise ValueError("check must be a boolean")
        self._check: bool = check
        self._logger: logging.Logger = get_file_logger(
            name="BlueMathDownloader",
            level=logging.INFO,
        )
        if self.debug:
            self.logger.setLevel(logging.DEBUG)

    @property
    def base_path_to_download(self) -> str:
        return self._base_path_to_download

    @base_path_to_download.setter
    def base_path_to_download(self, value: str) -> None:
        self._base_path_to_download = value

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    @property
    def check(self) -> bool:
        return self._check

    @check.setter
    def check(self, value: bool) -> None:
        self._check = value

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value

    def set_logger_name(self, name: str) -> None:
        """Sets the name of the logger."""
        self.logger = get_file_logger(name=name)

    @abstractmethod
    def download_data(self, *args, **kwargs) -> None:
        pass

import os
import time
from abc import abstractmethod
from datetime import datetime
from typing import Any, Callable, Optional

import xarray as xr

from ..core.models import BlueMathModel
from ._download_result import DownloadResult


class BaseDownloader(BlueMathModel):
    """
    Abstract class for BlueMath downloaders.

    Attributes
    ----------
    base_path_to_download : str
        The base path to download the data.
    debug : bool, optional
        If True, the logger will be set to DEBUG level. Default is True.

    Methods
    -------
    download_data(*args, **kwargs)
        Downloads the data. This method must be implemented in the child class.

    Notes
    -----
    - This class is an abstract class and should not be instantiated.
    - The download_data method must be implemented in the child class.
    """

    def __init__(
        self,
        base_path_to_download: str,
        debug: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        show_progress: bool = True,
    ) -> None:
        """
        The constructor for BaseDownloader class.

        Parameters
        ----------
        base_path_to_download : str
            The base path to download the data.
        debug : bool, optional
            If True, the logger will be set to DEBUG level. Default is True.
        max_retries : int, optional
            Maximum number of retry attempts for failed downloads. Default is 3.
        retry_delay : float, optional
            Initial delay between retries in seconds. Default is 1.0.
        retry_backoff : float, optional
            Exponential backoff multiplier for retry delays. Default is 2.0.
        show_progress : bool, optional
            Whether to show progress bars for downloads. Default is True.

        Raises
        ------
        ValueError
            If base_path_to_download is not a string.
            If debug is not a boolean.

        Notes
        -----
        - The logger will be set to INFO level.
        - If debug is True, the logger will be set to DEBUG level.
        - Retry mechanism uses exponential backoff to avoid overwhelming APIs.
        - Use `dry_run` parameter in download methods to check without downloading.
        """

        super().__init__()
        if not isinstance(base_path_to_download, str):
            raise ValueError("base_path_to_download must be a string")
        self._base_path_to_download: str = base_path_to_download
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")
        self._debug: bool = debug
        if not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self._max_retries: int = max_retries
        if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            raise ValueError("retry_delay must be a non-negative number")
        self._retry_delay: float = float(retry_delay)
        if not isinstance(retry_backoff, (int, float)) or retry_backoff <= 0:
            raise ValueError("retry_backoff must be a positive number")
        self._retry_backoff: float = float(retry_backoff)
        if not isinstance(show_progress, bool):
            raise ValueError("show_progress must be a boolean")
        self._show_progress: bool = show_progress

    @property
    def base_path_to_download(self) -> str:
        return self._base_path_to_download

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return self._max_retries

    @property
    def retry_delay(self) -> float:
        """Initial retry delay in seconds."""
        return self._retry_delay

    @property
    def retry_backoff(self) -> float:
        """Exponential backoff multiplier."""
        return self._retry_backoff

    @property
    def show_progress(self) -> bool:
        """Whether to show progress bars."""
        return self._show_progress

    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        retry_backoff: Optional[float] = None,
        error_message: str = "Operation failed",
        **kwargs,
    ) -> Any:
        """
        Execute a function with retry logic and exponential backoff.

        This method automatically retries failed operations with exponential
        backoff, which is useful for handling transient API errors or network issues.

        Parameters
        ----------
        func : Callable
            The function to execute with retry logic.
        *args
            Positional arguments to pass to func.
        max_retries : int, optional
            Maximum number of retry attempts. If None, uses self.max_retries.
        retry_delay : float, optional
            Initial delay between retries in seconds. If None, uses self.retry_delay.
        retry_backoff : float, optional
            Exponential backoff multiplier. If None, uses self.retry_backoff.
        error_message : str, optional
            Base error message for logging. Default is "Operation failed".
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        Any
            The return value of func if successful.

        Raises
        ------
        Exception
            The last exception raised by func if all retries are exhausted.

        Examples
        --------
        >>> def download_file(url):
        ...     # Simulated download that might fail
        ...     return requests.get(url)
        >>> result = downloader.retry_with_backoff(
        ...     download_file, "https://example.com/data.nc"
        ... )
        """

        max_retries = max_retries if max_retries is not None else self.max_retries
        retry_delay = retry_delay if retry_delay is not None else self.retry_delay
        retry_backoff = (
            retry_backoff if retry_backoff is not None else self.retry_backoff
        )

        last_exception = None
        current_delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"{error_message} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= retry_backoff
                else:
                    self.logger.error(
                        f"{error_message} after {max_retries + 1} attempts: {e}"
                    )

        # If we get here, all retries failed
        raise last_exception

    def check_file_complete(
        self,
        file_path: str,
        expected_time_range: Optional[tuple] = None,
        time_coord: str = "time",
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a NetCDF file is complete and valid.

        This method verifies that a file exists, can be opened, and optionally
        checks if it contains the expected time range.

        Parameters
        ----------
        file_path : str
            Path to the file to check.
        expected_time_range : tuple, optional
            Tuple of (start_time, end_time) as strings to verify.
            Format: ("YYYY-MM-DDTHH:MM", "YYYY-MM-DDTHH:MM")
        time_coord : str, optional
            Name of the time coordinate in the NetCDF file. Default is "time".

        Returns
        -------
        tuple[bool, Optional[str]]
            (is_complete, reason)
            - is_complete: True if file is complete and valid, False otherwise.
            - reason: Explanation if file is not complete, None if complete.

        Examples
        --------
        >>> is_complete, reason = downloader.check_file_complete(
        ...     "/path/to/file.nc",
        ...     expected_time_range=("2020-01-01T00:00", "2020-01-31T23:00")
        ... )
        >>> if not is_complete:
        ...     print(f"File incomplete: {reason}")
        """

        if not os.path.exists(file_path):
            return False, "File does not exist"

        try:
            with xr.open_dataset(file_path) as ds:
                # Check if time coordinate exists
                if time_coord not in ds.coords:
                    # Try alternative time coordinate names
                    alt_time_coords = ["valid_time", "Time", "datetime"]
                    found_time = False
                    for alt_coord in alt_time_coords:
                        if alt_coord in ds.coords:
                            time_coord = alt_coord
                            found_time = True
                            break
                    if not found_time:
                        return (
                            False,
                            f"No time coordinate found (tried: {time_coord}, {alt_time_coords})",
                        )

                # Check expected time range if provided
                if expected_time_range:
                    start_time, end_time = expected_time_range
                    try:
                        time_values = ds[time_coord].values
                        if len(time_values) == 0:
                            return False, "File has no time data"

                        last_time = str(time_values[-1])

                        if end_time not in last_time:
                            return (
                                False,
                                f"File ends at {last_time} instead of {end_time}",
                            )
                    except Exception as e:
                        return False, f"Error checking time range: {e}"

                # File is complete
                return True, None

        except Exception as e:
            return False, f"Error opening file: {e}"

    def create_download_result(
        self, start_time: Optional[datetime] = None
    ) -> DownloadResult:
        """
        Create a new DownloadResult instance with timing information.

        Parameters
        ----------
        start_time : datetime, optional
            Start time for the download operation. If None, uses current time.

        Returns
        -------
        DownloadResult
            A new DownloadResult instance ready for tracking downloads.
        """

        result = DownloadResult()
        result.start_time = start_time if start_time else datetime.now()

        return result

    def finalize_download_result(
        self, result: DownloadResult, message: Optional[str] = None
    ) -> DownloadResult:
        """
        Finalize a DownloadResult with end time and summary message.

        Parameters
        ----------
        result : DownloadResult
            The result to finalize.
        message : str, optional
            Custom summary message. If None, generates a default message.

        Returns
        -------
        DownloadResult
            The finalized result with end_time and message set.
        """

        result.end_time = datetime.now()

        # Recalculate duration after setting end_time
        if result.start_time and result.end_time:
            delta = result.end_time - result.start_time
            result.duration_seconds = delta.total_seconds()

        result.success = len(result.error_files) == 0

        if message is None:
            # Generate default message
            parts = []
            if result.downloaded_files:
                parts.append(f"{len(result.downloaded_files)} downloaded")
            if result.skipped_files:
                parts.append(f"{len(result.skipped_files)} skipped")
            if result.error_files:
                parts.append(f"{len(result.error_files)} errors")
            result.message = f"Download complete: {', '.join(parts)}"
        else:
            result.message = message

        return result

    @abstractmethod
    def download_data(self, *args, **kwargs) -> None:
        pass

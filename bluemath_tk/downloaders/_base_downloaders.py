from abc import abstractmethod
from datetime import datetime
from typing import List, Optional

from ..core.models import BlueMathModel
from ._download_result import DownloadResult


class BaseDownloader(BlueMathModel):
    """
    Abstract base class for BlueMath downloaders.

    All downloaders should:
    1. Have a `download_data` method that routes to product-specific methods
    2. Have product-specific methods like `download_data_<product>`
    3. Use DownloadResult to track download status

    Attributes
    ----------
    product : str
        The product name (e.g., "SWOT", "ERA5").
    product_config : dict
        Product configuration dictionary.
    base_path_to_download : str
        Base path where downloaded files are stored.
    debug : bool
        If True, logger is set to DEBUG level.
    """

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        debug: bool = True,
    ) -> None:
        """
        Initialize the BaseDownloader.

        Parameters
        ----------
        product : str
            The product to download data from.
        base_path_to_download : str
            The base path to download the data.
        debug : bool, optional
            If True, the logger will be set to DEBUG level. Default is True.
        """

        super().__init__()
        if not isinstance(product, str):
            raise ValueError("product must be a string")
        self._product: str = product
        if not isinstance(base_path_to_download, str):
            raise ValueError("base_path_to_download must be a string")
        self._base_path_to_download: str = base_path_to_download
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")
        self._debug: bool = debug

    @property
    def product(self) -> str:
        return self._product

    @property
    def base_path_to_download(self) -> str:
        return self._base_path_to_download

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    @abstractmethod
    def product_config(self) -> dict:
        pass

    def list_datasets(self) -> List[str]:
        """
        List all available datasets for the product.

        Returns
        -------
        List[str]
            List of available dataset names.
        """

        return list(self.product_config.get("datasets", {}).keys())

    def create_download_result(
        self, start_time: Optional[datetime] = None
    ) -> DownloadResult:
        """
        Create a new DownloadResult instance.

        Parameters
        ----------
        start_time : Optional[datetime], optional
            The start time of the download operation. If None, the current time is used.

        Returns
        -------
        DownloadResult
            A new DownloadResult instance.
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
            The DownloadResult to finalize.
        message : Optional[str], optional
            The message to add to the DownloadResult.

        Returns
        -------
        DownloadResult
            The finalized DownloadResult.
        """

        result.end_time = datetime.now()

        if result.start_time and result.end_time:
            delta = result.end_time - result.start_time
            result.duration_seconds = delta.total_seconds()

        result.success = len(result.error_files) == 0

        if message is None:
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
    def download_data(self, *args, **kwargs) -> DownloadResult:
        """
        Download data for the product.

        Routes to product-specific methods like download_data_<product>().

        Parameters
        ----------
        *args
            Arguments passed to product-specific download method.
        **kwargs
            Keyword arguments (e.g., force, dry_run).

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.
        """

        pass

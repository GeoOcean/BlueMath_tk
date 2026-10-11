import json
import os
import re
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ..core.models import BlueMathModel
from ._download_result import DownloadResult
from ._transfer import PART_SUFFIX, RemoteFileNotFound, retry_call, validate_file


def load_config(*path_parts: str) -> dict:
    """
    Load a JSON configuration file.

    Parameters
    ----------
    *path_parts : str
        Parts of the path, joined with os.path.join.

    Returns
    -------
    dict
        The parsed configuration.
    """

    with open(os.path.join(*path_parts)) as f:
        return json.load(f)


def _to_snake_case(name: str) -> str:
    """Convert a product name to snake case: "OpenData" -> "open_data", "ERA5" -> "era5"."""

    return re.sub(r"(?<=[a-z0-9])(?=[A-Z][a-z])", "_", name).lower()


class BaseDownloader(BlueMathModel):
    """
    Abstract base class for BlueMath downloaders.

    All downloaders should:

    1. Declare their products in the class attribute ``products_configs``,
       loaded from ``<provider>/<PRODUCT>/<PRODUCT>_config.json``.
    2. Implement product-specific methods named ``download_data_<product>``
       (snake case), which ``download_data`` routes to, or override
       ``download_data`` when all products share the same logic.
    3. Write files through ``fetch_file``, so that every file is downloaded to a
       ``.part`` file, validated and then atomically renamed, with retries.
    4. Use DownloadResult to track download status.

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
    retries : int
        Number of attempts per file.
    retry_backoff : float
        Base waiting time in seconds between attempts (doubles each attempt).
    """

    products_configs: Dict[str, dict] = {}

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        debug: bool = True,
        retries: int = 3,
        retry_backoff: float = 30.0,
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
        retries : int, optional
            Number of attempts per file. Default is 3.
        retry_backoff : float, optional
            Base waiting time in seconds between attempts. Default is 30.

        Raises
        ------
        ValueError
            If the product is not in ``products_configs`` or an argument has a
            wrong type.
        """

        super().__init__()
        if not isinstance(product, str):
            raise ValueError("product must be a string")
        if product not in self.products_configs:
            raise ValueError(
                f"Product '{product}' not found. Available: {self.list_products()}"
            )
        self._product: str = product
        if not isinstance(base_path_to_download, str):
            raise ValueError("base_path_to_download must be a string")
        self._base_path_to_download: str = base_path_to_download
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")
        self._debug: bool = debug
        if not isinstance(retries, int) or retries < 1:
            raise ValueError("retries must be a positive integer")
        self.retries: int = retries
        self.retry_backoff: float = float(retry_backoff)

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
    def product_config(self) -> dict:
        """
        Configuration of the product, loaded from its JSON file.

        Returns
        -------
        dict
            Product configuration dictionary.
        """
        return self.products_configs[self.product]

    @classmethod
    def list_products(cls) -> List[str]:
        """
        List the products this downloader supports.

        Returns
        -------
        List[str]
            List of product names.
        """

        return list(cls.products_configs.keys())

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
            result.message = f"Download complete: {', '.join(parts) or 'nothing to do'}"
        else:
            result.message = message

        return result

    def fetch_file(
        self,
        output_file: str,
        fetch: Callable[[str], None],
        result: DownloadResult,
        force: bool = False,
        dry_run: bool = True,
        resumable: bool = False,
        validate: bool = True,
    ) -> None:
        """
        Download a single file safely and record the outcome in ``result``.

        ``fetch`` is called with a temporary ``<output_file>.part`` path and must
        write the whole file there. The ``.part`` file is then validated and
        renamed to ``output_file``, so ``output_file`` only ever exists complete.
        Failed attempts are retried ``self.retries`` times.

        Parameters
        ----------
        output_file : str
            Final path of the file.
        fetch : Callable[[str], None]
            Function that writes the file to the path it receives.
        result : DownloadResult
            Result to update.
        force : bool, optional
            Download even if ``output_file`` exists. Default is False.
        dry_run : bool, optional
            Only record what would be downloaded. Default is True.
        resumable : bool, optional
            If True, an existing ``.part`` file is kept between attempts so that
            ``fetch`` can resume it. Default is False.
        validate : bool, optional
            Check the downloaded file with validate_file. Default is True.
        """

        if not force and os.path.exists(output_file):
            result.add_skipped(output_file, "Already downloaded")
            return

        if dry_run:
            result.add_skipped(output_file, "Would download (dry run)")
            return

        part_file = output_file + PART_SUFFIX
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        def attempt() -> None:
            if not resumable and os.path.exists(part_file):
                os.remove(part_file)
            fetch(part_file)
            if validate:
                try:
                    validate_file(part_file)
                except Exception:
                    os.remove(part_file)
                    raise
            os.replace(part_file, output_file)

        try:
            self.logger.debug(f"Downloading: {output_file}")
            retry_call(
                attempt,
                retries=self.retries,
                backoff=self.retry_backoff,
                logger=self.logger,
            )
            result.add_downloaded(output_file)
            self.logger.info(f"Downloaded: {output_file}")

        except RemoteFileNotFound as e:
            self._remove_quietly(part_file)
            result.add_skipped(output_file, f"Not available on server: {e}")
            self.logger.info(f"Not available on server: {e}")

        except Exception as e:
            if not resumable:
                self._remove_quietly(part_file)
            result.add_error(output_file, e)
            self.logger.error(f"Error downloading {output_file}: {e}")

    @staticmethod
    def _remove_quietly(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def download_data(self, *args, dry_run: bool = True, **kwargs) -> DownloadResult:
        """
        Download data for the product.

        Routes to the product-specific method ``download_data_<product>``, where
        the product name is converted to snake case (e.g. "OpenData" -> "open_data").

        Parameters
        ----------
        *args
            Arguments passed to product-specific download method.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is True.
        **kwargs
            Keyword arguments passed to product-specific download method (e.g., force).

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.

        Raises
        ------
        ValueError
            If there is no download method for the product.
        """

        method_name = f"download_data_{_to_snake_case(self.product)}"
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Download for product {self.product} not supported")

        return method(*args, dry_run=dry_run, **kwargs)

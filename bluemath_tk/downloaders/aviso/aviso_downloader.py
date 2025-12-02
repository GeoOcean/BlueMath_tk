import ftplib
import json
import os
from typing import List, Optional

from .._base_downloaders import BaseDownloader
from .._download_result import DownloadResult


class AvisoDownloader(BaseDownloader):
    """
    Simple downloader for AVISO data.

    Downloads all available files from the FTP base path specified in the config.

    Attributes
    ----------
    product : str
        The product to download data from (e.g., "SWOT")
    product_config : dict
        Product configuration loaded from config files
    datasets : dict
        All available datasets for the product

    Examples
    --------
    >>> from bluemath_tk.downloaders.aviso.aviso_downloader import AvisoDownloader
    >>>
    >>> # Initialize with specific product
    >>> downloader = AvisoDownloader(
    ...     product="SWOT",
    ...     base_path_to_download="./swot_data",
    ...     username="your_username",
    ...     password="your_password"
    ... )
    >>>
    >>> # List available datasets
    >>> datasets = downloader.list_datasets()
    >>> print(datasets)
    >>>
    >>> # Download data for specific dataset and cycles
    >>> result = downloader.download_data(
    ...     dataset="swot-l3-expert",
    ...     cycles=["cycle_001"],
    ...     force=False
    ... )
    >>> print(result)
    """

    # Product configurations loaded from JSON files
    products_configs = {
        "SWOT": json.load(
            open(os.path.join(os.path.dirname(__file__), "SWOT", "SWOT_config.json"))
        )
    }

    def __init__(
        self,
        product: str,
        base_path_to_download: str,
        username: str,
        password: str,
        debug: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        show_progress: bool = True,
    ) -> None:
        """
        Initialize the AvisoDownloader.

        Parameters
        ----------
        product : str
            The product to download data from (e.g., "SWOT").
        base_path_to_download : str
            Base path where downloaded files will be stored.
        username : str
            AVISO FTP username.
        password : str
            AVISO FTP password.
        debug : bool, optional
            If True, sets logger to DEBUG level. Default is True.
        max_retries : int, optional
            Maximum number of retry attempts. Default is 3.
        retry_delay : float, optional
            Initial retry delay in seconds. Default is 1.0.
        retry_backoff : float, optional
            Exponential backoff multiplier. Default is 2.0.
        show_progress : bool, optional
            Whether to show progress bars. Default is True.

        Raises
        ------
        ValueError
            If the product configuration is not found.
        """

        super().__init__(
            base_path_to_download=base_path_to_download,
            debug=debug,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            show_progress=show_progress,
        )
        self._product = product
        self._product_config = self.products_configs.get(product)
        if self._product_config is None:
            available_products = list(self.products_configs.keys())
            raise ValueError(
                f"Product '{product}' not found. Available products: {available_products}"
            )
        self.set_logger_name(
            f"AvisoDownloader-{product}", level="DEBUG" if debug else "INFO"
        )
        # Get FTP server from config
        self._ftp_server = self.product_config.get("ftp_server")
        if self._ftp_server is None:
            raise ValueError("FTP server not found in product configuration")
        # Initialize FTP client and login (don't store password)
        self._client = ftplib.FTP(self._ftp_server)
        self._client.login(username, password)
        self.logger.info(f"---- AVISO DOWNLOADER INITIALIZED ({product}) ----")

    @property
    def product(self) -> str:
        """The product name (e.g., 'SWOT')."""
        return self._product

    @property
    def product_config(self) -> dict:
        """Product configuration dictionary loaded from config file."""
        return self._product_config

    @property
    def ftp_server(self) -> str:
        """FTP server address from product configuration."""
        return self._ftp_server

    @property
    def client(self) -> ftplib.FTP:
        """FTP client connection (initialized and logged in)."""
        return self._client

    def list_datasets(self) -> List[str]:
        """
        List all available datasets for the product.

        Returns
        -------
        List[str]
            List of available dataset names.
        """

        return list(self.product_config["datasets"].keys())

    def download_data(
        self,
        dry_run: bool = False,
        *args,
        **kwargs,
    ) -> DownloadResult:
        """
        Download data for the product.

        Routes to product-specific download methods based on the product type.

        Parameters
        ----------
        dry_run : bool, optional
            If True, only check what would be downloaded without actually downloading.
            Default is False.
        *args
            Arguments passed to product-specific download method.
        **kwargs
            Keyword arguments passed to product-specific download method.

        Returns
        -------
        DownloadResult
            Result with information about downloaded, skipped, and error files.

        Raises
        ------
        ValueError
            If the product is not supported.
        """

        if self.product == "SWOT":
            return self.download_data_swot(dry_run=dry_run, *args, **kwargs)
        else:
            raise ValueError(f"Download for product {self.product} not supported")

    def download_data_swot(
        self,
        dataset: str,
        cycles: Optional[List[str]] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Download SWOT data for a specific dataset.

        Downloads all .nc files from specified cycles. Files are saved to:
        base_path_to_download/dataset/cycle/filename.nc

        Parameters
        ----------
        dataset : str
            The dataset to download (e.g., "swot-l3-expert").
            Use list_datasets() to see available datasets.
        cycles : List[str], optional
            List of cycle folder names to download (e.g., ["cycle_001", "cycle_002"]).
            If None, uses cycles from dataset configuration. Default is None.
        force : bool, optional
            Force re-download even if file exists. Default is False.
        dry_run : bool, optional
            If True, only check what would be downloaded. Default is False.

        Returns
        -------
        DownloadResult
            Result with all downloaded files and download statistics.

        Raises
        ------
        ValueError
            If dataset is not found or no cycles are available.
        """

        # Validate dataset
        if dataset not in self.list_datasets():
            raise ValueError(
                f"Dataset '{dataset}' not found. Available datasets: {self.list_datasets()}"
            )

        dataset_config = self.product_config["datasets"][dataset]
        ftp_base_path = dataset_config["ftp_base_path"]
        result = self.create_download_result()

        try:
            # Get cycles from dataset config if not specified
            if cycles is None:
                cycles = dataset_config.get("cycles", [])
                if not cycles:
                    raise ValueError(
                        f"No cycles specified for dataset '{dataset}' and cycles parameter not provided"
                    )

            self.logger.info(f"Downloading dataset: {dataset}")
            self.logger.info(f"Cycles: {cycles}")

            all_downloaded_files = []

            # Process each cycle
            for cycle in cycles:
                self.logger.info(f"Processing cycle: {cycle}")

                # List all .nc files in this cycle
                files = self._list_all_files_in_cycle(ftp_base_path, cycle)

                if not files:
                    self.logger.warning(f"No files found in cycle {cycle}")
                    continue

                self.logger.info(f"Found {len(files)} files in cycle {cycle}")

                # Download files for this cycle
                downloaded_files = self._download_files(
                    files=files,
                    dataset=dataset,
                    ftp_base_path=ftp_base_path,
                    cycle=cycle,
                    force=force,
                    dry_run=dry_run,
                    result=result,
                )

                all_downloaded_files.extend(downloaded_files)

            result.downloaded_files = all_downloaded_files
            return self.finalize_download_result(result)

        except Exception as e:
            result.add_error("download_operation", e)
            return self.finalize_download_result(result)

    def _list_all_files_in_cycle(
        self,
        ftp_base_path: str,
        cycle: str,
    ) -> List[str]:
        """
        List all .nc files from a cycle directory on FTP server.

        Parameters
        ----------
        ftp_base_path : str
            FTP base path for the dataset.
        cycle : str
            Cycle directory name (e.g., "cycle_001").

        Returns
        -------
        List[str]
            List of .nc filenames (without path) found in the cycle directory.
        """

        files = []
        # Navigate to cycle directory
        self._client.cwd(ftp_base_path)
        self._client.cwd(cycle)
        # Get directory listing
        items = []
        self._client.retrlines("LIST", items.append)
        # Parse listing and filter for .nc files
        for item in items:
            parts = item.split()
            if len(parts) >= 9:  # Valid LIST entry has at least 9 parts
                name = " ".join(parts[8:])  # Filename might contain spaces
                if name.endswith(".nc"):
                    files.append(name)

        return files

    def _download_files(
        self,
        files: List[str],
        dataset: str,
        ftp_base_path: str,
        cycle: str,
        force: bool,
        dry_run: bool,
        result: DownloadResult,
    ) -> List[str]:
        """
        Download all files from the list.

        Files are saved to: base_path_to_download/dataset/cycle/filename.nc

        Parameters
        ----------
        files : List[str]
            List of filenames to download (without path).
        dataset : str
            Dataset name (used in local path).
        ftp_base_path : str
            FTP base path for the dataset.
        cycle : str
            Cycle directory name (used in local path).
        force : bool
            Force re-download even if file exists.
        dry_run : bool
            If True, only simulate download.
        result : DownloadResult
            Download result object to update.

        Returns
        -------
        List[str]
            List of local file paths for successfully downloaded files only.
        """

        downloaded_files = []

        for filename in files:
            # Construct local path: base_path/dataset/cycle/filename
            local_path = os.path.join(
                self.base_path_to_download, dataset, cycle, filename
            )

            # Skip if file already exists (unless force=True)
            if not force and os.path.exists(local_path):
                result.add_skipped(local_path, "Already downloaded")
                continue

            # Handle dry run
            if dry_run:
                result.add_skipped(local_path, f"Would download {filename} (dry run)")
                continue

            # Download file
            try:
                # Create directory structure if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # Download function with retry mechanism
                def _download():
                    # Navigate to cycle directory on FTP
                    self._client.cwd(ftp_base_path)
                    self._client.cwd(cycle)
                    # Download file
                    with open(local_path, "wb") as f:
                        self._client.retrbinary(f"RETR {filename}", f.write)

                self.retry_with_backoff(
                    _download, error_message=f"Failed to download {filename}"
                )

                result.add_downloaded(local_path)
                self.logger.info(f"Downloaded: {filename} -> {local_path}")
                downloaded_files.append(local_path)

            except Exception as e:
                result.add_error(local_path, e)
                self.logger.error(f"Error downloading {filename}: {e}")

        return downloaded_files
